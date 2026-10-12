"""LLM 판정자(LLM-as-a-Judge)로 합성 QA의 답변 충실도를 검사한다.

근거 인용 검증(grounded)은 "인용한 문장이 원문에 있는가"만 확인한다.
답변에 원문에 없는 문장이 덧붙여져도 통과하므로, 답변 전체를 원문과 대조하는 판정이 따로 필요하다.

- 같은 청크의 QA를 한 번에 판정한다 (청크 원문을 한 번만 보내 토큰 절약)
- 생성 모델과 다른 계열의 모델을 쓰는 것이 좋다 (자기가 만든 답에 후한 self-preference bias)
- 출력: data/interim/qa_judged/<judge>/<doc_id>.jsonl (청크 1개 = 1줄), 중간에 끊겨도 이어서 진행
"""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import openai

from carmind.llm import StructuredLLM

JUDGE_VERSION = "v2"  # v2: 브랜드명 → "제조사" 일반화도 허용

INSTRUCTIONS = """\
당신은 자동차 취급설명서 기반 QA 데이터의 품질을 검수하는 엄격한 심사자입니다.
[원문]과 여러 개의 [QA]가 주어집니다. 각 QA를 원문만 근거로 판정하세요. 상식이나 배경지식으로 보충하지 마세요.

판정 항목:
- faithful: 답변의 모든 내용(사실, 수치, 조건, 원인·결과, 부정·비교)이 원문에서 직접 확인되면 true.
  원문에 없는 내용이 하나라도 있으면 false. 표현을 바꾸거나 요약한 것은 괜찮습니다.
  브랜드명을 "제조사"로, 서비스센터·정비 네트워크 이름을 "공식 서비스센터"로 바꾼 것은 허용합니다.
- unsupported: faithful이 false이면 원문에 없는 답변 속 문장(또는 구절)을 그대로 적으세요. true이면 빈 배열.
- answers_question: 답변이 질문에 직접 답하면 true. 질문과 어긋나거나 핵심을 빠뜨리면 false.
- standalone: 원문을 보지 않은 사람도 질문만 읽고 무엇을 묻는지 알 수 있으면 true.

모든 QA에 대해 qa_index 순서대로 판정 결과를 반환하세요.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "qa_index": {"type": "integer"},
                    "faithful": {"type": "boolean"},
                    "unsupported": {"type": "array", "items": {"type": "string"}},
                    "answers_question": {"type": "boolean"},
                    "standalone": {"type": "boolean"},
                },
                "required": ["qa_index", "faithful", "unsupported", "answers_question", "standalone"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    qa_id: str
    faithful: bool
    unsupported: list[str]
    answers_question: bool
    standalone: bool
    question: str = ""  # 판정 대상 질문 원문. 판정을 QA에 합칠 때 같은 QA인지 확인하는 데 쓴다

    @property
    def passed(self) -> bool:
        return self.faithful and self.answers_question and self.standalone


@dataclass
class JudgeResult:
    chunk_id: str
    judge: str
    judge_version: str
    verdicts: list[Verdict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


def build_input(chunk_text: str, qas: list[dict]) -> str:
    parts = [f"[원문]\n{chunk_text}\n"]
    for i, qa in enumerate(qas):
        parts.append(f"[QA {i}]\n질문: {qa['question']}\n답변: {qa['answer']}\n")
    return "\n".join(parts)


class Judge:
    def __init__(self, model: str, effort: str = "low"):
        self.model = model
        self.llm = StructuredLLM(model, effort)

    def judge_chunk(self, chunk: dict, qas: list[dict]) -> JudgeResult:
        result = JudgeResult(chunk["chunk_id"], self.model, JUDGE_VERSION)
        try:
            resp = self.llm.call(
                INSTRUCTIONS, build_input(chunk["text"], qas), SCHEMA, "judge", f"carmind-judge-{JUDGE_VERSION}"
            )
        except openai.APIError as e:
            result.error = f"{type(e).__name__}: {e}"
            return result

        result.input_tokens, result.output_tokens, result.cost_usd = (
            resp.input_tokens, resp.output_tokens, resp.cost_usd
        )
        by_index = {v["qa_index"]: v for v in resp.data["verdicts"]}
        if set(by_index) != set(range(len(qas))):
            result.error = f"verdict 개수 불일치: {sorted(by_index)} vs {len(qas)}개"
            return result
        for i, qa in enumerate(qas):
            v = by_index[i]
            result.verdicts.append(
                Verdict(
                    qa["qa_id"], v["faithful"], v["unsupported"], v["answers_question"], v["standalone"], qa["question"]
                )
            )
        return result


def load_done(out_path: Path) -> dict[str, dict]:
    if not out_path.exists():
        return {}
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    return {r["chunk_id"]: r for r in rows if r["error"] is None}


def judge_all(
    candidates: list[dict], chunks: dict[str, dict], judge: Judge, out_path: Path, workers: int = 8
) -> list[JudgeResult]:
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for qa in candidates:
        by_chunk[qa["chunk_id"]].append(qa)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    def fully_judged(cid: str) -> bool:
        """청크의 후보 QA가 전부(같은 질문으로) 판정됐을 때만 완료로 본다.
        필터를 바꿔 청크에 새 QA가 생기면 그 청크는 다시 판정한다 (새 QA가 판정 없이 빠지는 것 방지)."""
        if cid not in done:
            return False
        judged = {v["qa_id"]: v.get("question") for v in done[cid]["verdicts"]}
        return all(judged.get(qa["qa_id"]) == qa["question"] for qa in by_chunk[cid])

    todo = [cid for cid in by_chunk if not fully_judged(cid)]
    print(f"{len(by_chunk)} chunks ({len(candidates)} QA): {len(done)} done, {len(todo)} to judge ({judge.model})")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool, out_path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(judge.judge_chunk, chunks[cid], by_chunk[cid]) for cid in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            f.flush()
            results.append(r)
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  cost=${sum(x.cost_usd for x in results):.4f}", flush=True)
    return results
