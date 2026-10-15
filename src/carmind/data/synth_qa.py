"""teacher 모델로 청크에서 합성 QA(질문-답변) 학습 데이터를 만든다.

흐름:
    청크 → teacher(OpenAI Responses API, JSON 스키마 구조화 출력) → QA 0~4개 + 근거(evidence)
    → 근거 검증: evidence가 청크 원문에 실제로 있는지 확인 (지어낸 답 걸러내기)
    → data/interim/qa/<model>/<doc_id>.jsonl (청크 1개 = 1줄)

이미 생성한 청크는 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 진행한다.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import NamedTuple

import openai
import yaml

from carmind.budget import DailyTokenBudget
from carmind.llm import StructuredLLM

PROMPT_VERSION = "v3"  # v2: 지시어 질문·문서 언급·추가 서술 금지, 브랜드명 코드 치환 / v3: 길이 하한 제거, 마무리 문장 금지

QA_TYPES = ["fact", "procedure", "safety", "troubleshoot"]

INSTRUCTIONS = """\
당신은 자동차 취급설명서로 학습 데이터를 만드는 전문가입니다.
주어진 설명서 발췌(청크)를 읽고, 차량 소유자가 실제로 물어볼 법한 질문과 정확한 답변을 만드세요.

질문 규칙:
- 0~4개. 청크에 실질적인 정보가 없으면(법률 고지, 목차성 문장 등) 빈 배열을 반환하세요.
- 서로 다른 내용을 묻게 하세요. 같은 사실을 표현만 바꿔 묻지 마세요.
- 문서를 보지 않은 사람이 묻는 것처럼 자연스럽게 쓰세요. "이 문서에서", "위 내용에 따르면" 같은 표현 금지.
- 질문만 읽어도 무엇을 묻는지 알 수 있어야 합니다. 기능·부품 이름을 질문에 직접 쓰고,
  "이 기능", "해당 장치", "그거" 같은 지시어로 대신하지 마세요.
- 존댓말 질문과 일상 구어체 질문을 섞으세요. (예: "타이어 공기압은 얼마로 맞춰야 하나요?", "계기판에 거북이 모양 불이 켜졌는데 왜 그래?")
- '[아이콘]'처럼 그림을 봐야만 알 수 있는 내용은 질문하지 마세요.
- 유형(type): fact(사실·사양), procedure(사용 방법·절차), safety(경고·주의 사항), troubleshoot(증상 → 원인·조치)

답변 규칙:
- 청크에 있는 정보만 사용하세요. 청크에 없는 내용을 추측하거나 덧붙이지 마세요.
  특히 청크에 없는 비교·부정·해석 문장(예: "A가 아니라 B입니다", "~라는 뜻은 아닙니다")을 만들지 마세요.
- 정비사가 직접 설명하듯 답하세요. "설명서에는", "안내되어 있습니다" 같은 문서 언급 금지.
- 질문에 직접 답한 뒤, 원문에 있는 조건이나 절차를 설명하세요.
  길이는 원문 내용만큼만 쓰세요. 원문 근거가 한 문장이면 답변도 한 문장이면 충분합니다.
- 원문에 없는 결과·위험·권고·마무리 문장을 덧붙이지 마세요.
  (예: "~하면 사고가 날 수 있습니다", "~하는 것이 좋습니다", "주의해야 합니다", "한 번 더 시도해 보세요")
- 관련된 [경고]·[주의] 내용이 있으면 반드시 답변에 포함하세요.
- 특정 브랜드명, 서비스센터·정비 네트워크 이름은 "공식 서비스센터"로 바꿔 쓰세요.

근거(evidence):
- 답변의 근거가 된 청크 원문 구절을 1~3개, 원문 그대로(글자 하나 바꾸지 말고) 인용하세요. 각 구절은 한 문장 이내.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": QA_TYPES},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["type", "question", "answer", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}


@dataclass
class QAPair:
    type: str
    question: str
    answer: str
    evidence: list[str]
    grounded: bool  # 모든 evidence가 청크 원문에 실제로 있는가


@dataclass
class ChunkResult:
    chunk_id: str
    doc_id: str
    model: str
    prompt_version: str
    qa_pairs: list[QAPair] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class Replacement(NamedTuple):
    src: str
    dst: str
    loose_space: bool = False  # True면 글자 사이 공백·줄바꿈이 있어도 매칭 (PDF 줄바꿈으로 끊긴 이름)


def load_replacements(path: Path) -> list[Replacement]:
    """로컬 치환 규칙(data/anonymize.yaml)을 읽는다. 없으면 빈 목록.

    형식:  replacements:
             - from: "원문 표현"
               to: "공식 서비스센터"
               loose_space: true   # 선택. "원문\n표현", "원 문표현"처럼 끊긴 형태도 치환
    매칭은 대소문자를 구분하지 않는다 (영문 표기가 "Abc", "ABC"로 섞여 나옴).
    브랜드명이 코드·저장소에 들어가지 않도록 규칙은 git에 올라가지 않는 data/ 아래에 둔다.
    """
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Replacement(r["from"], r["to"], bool(r.get("loose_space", False))) for r in data.get("replacements", [])]


@cache
def _rule_pattern(src: str, loose_space: bool) -> re.Pattern:
    if loose_space:
        body = r"\s*".join(re.escape(ch) for ch in src if not ch.isspace())
    else:
        body = re.escape(src)
    return re.compile(body, re.IGNORECASE)


# 브랜드와 무관한 정리 규칙 (일반 명사라 코드에 둔다). 로컬 치환 규칙을 적용한 뒤 실행
GENERIC_CLEANUP = [
    (re.compile(r"당사\s*직영\s*"), ""),  # "당사 직영 공식 서비스센터" → "공식 서비스센터"
    (re.compile(r"당사(?!자)"), "제조사"),  # 제조사 1인칭 표현 ("당사가 보증하는 부품"). "당사자"는 제외
    # 각각 치환된 두 서비스망 이름이 이어 붙은 흔적: "공식 서비스센터나 공식 서비스센터", "공식 서비스센터 공식 서비스센터"
    (re.compile(r"(공식 서비스센터)(?:\s*(?:나|또는|,|및)?\s*공식 서비스센터)+"), r"\1"),
]


def anonymize(text: str, replacements: list[tuple[str, str]] | list[Replacement]) -> str:
    """프롬프트 지시만으로는 100% 지켜지지 않아서 코드로 확실히 치환한다.

    여러 번 적용해도 결과가 같아서(멱등) 생성 단계와 필터 단계에서 모두 적용한다.
    규칙을 고치면 QA를 다시 생성하지 않고 필터만 다시 돌려도 반영된다.
    """
    for rule in replacements:
        src, dst = rule[0], rule[1]
        loose_space = len(rule) > 2 and rule[2]
        text = _rule_pattern(src, loose_space).sub(lambda _m, dst=dst: dst, text)
    for pattern, dst in GENERIC_CLEANUP:
        text = pattern.sub(dst, text)
    return text


def squash(text: str) -> str:
    """비교용 정규화: 공백·줄바꿈 차이와 목록 기호를 무시한다."""
    text = re.sub(r"^- ", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", "", text)


def is_grounded(evidence: list[str], chunk_text: str) -> bool:
    """evidence가 하나 이상 있고, 전부 청크 원문에 들어 있으면 True."""
    target = squash(chunk_text)
    return bool(evidence) and all(squash(e) in target for e in evidence if e.strip())


class OpenAITeacher:
    def __init__(self, model: str, effort: str = "low", replacements: list[tuple[str, str]] | None = None):
        self.model = model
        self.llm = StructuredLLM(model, effort)
        self.replacements = replacements or []

    def generate(self, chunk: dict) -> ChunkResult:
        result = ChunkResult(chunk["chunk_id"], chunk["doc_id"], self.model, PROMPT_VERSION)
        try:
            resp = self.llm.call(INSTRUCTIONS, chunk["text"], SCHEMA, "qa", f"carmind-synth-qa-{PROMPT_VERSION}")
        except openai.APIError as e:
            result.error = f"{type(e).__name__}: {e}"
            return result

        result.input_tokens, result.output_tokens, result.cost_usd = (
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )
        for qa in resp.data["qa_pairs"]:
            qa["question"] = anonymize(qa["question"], self.replacements)
            qa["answer"] = anonymize(qa["answer"], self.replacements)
            # evidence는 원문 대조용이라 치환하지 않는다
            result.qa_pairs.append(QAPair(**qa, grounded=is_grounded(qa["evidence"], chunk["text"])))
        return result


def load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["error"] is None:  # 에러난 청크는 다시 시도
            done.add(row["chunk_id"])
    return done


def generate_all(
    chunks: list[dict],
    teacher: OpenAITeacher,
    out_path: Path,
    workers: int = 8,
    budget: DailyTokenBudget | None = None,
) -> list[ChunkResult]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"{len(chunks)} chunks: {len(done)} done, {len(todo)} to generate ({teacher.model})")
    if budget and budget.exhausted:
        print(f"  오늘 토큰 상한 도달 ({budget.used:,}/{budget.limit:,}) → 건너뜀")
        return []

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool, out_path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(teacher.generate, c) for c in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.cancelled():
                continue
            r = fut.result()
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            f.flush()  # 중간에 끊겨도 완료분은 남게
            results.append(r)
            if budget:
                budget.record(r.input_tokens + r.output_tokens)
                if budget.exhausted:
                    # 아직 시작 안 한 요청은 취소. 이미 실행 중인 요청(최대 workers개)은 끝까지 받는다
                    cancelled = sum(fu.cancel() for fu in futures if not fu.done() and not fu.running())
                    if cancelled:
                        print(f"  오늘 토큰 상한 도달 ({budget.used:,}/{budget.limit:,}) → {cancelled}개 취소, 내일 이어서")
            if i % 50 == 0 or i == len(todo):
                cost = sum(x.cost_usd for x in results)
                print(f"  {i}/{len(todo)}  cost=${cost:.4f}", flush=True)
    return results
