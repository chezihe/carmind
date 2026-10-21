"""LLM 판정자(LLM-as-a-Judge)로 모델 답변을 채점한다.

자동 지표(문자 F1 등)는 "정답과 글자가 얼마나 겹치나"만 본다. 같은 내용을 다르게 말하면 점수가 낮고,
틀린 수치 하나가 섞여도 점수가 높다. 그래서 의미 단위의 판정은 LLM에게 맡긴다.

- 문항 1개 = 요청 1개. 지시문(INSTRUCTIONS)이 모든 요청에서 같아서 프롬프트 캐시로 입력 비용이 준다
- 판정 근거로 정답 답변 + 원문 청크(익명화)를 함께 준다. 정답에 없지만 원문에는 있는 내용을 말한 경우를
  환각(hallucination)으로 오판하지 않기 위해서다
- API를 부르기 전에 비용을 추정하고 상한(--max-cost)을 넘으면 멈춘다
- 출력: <run>/grades_<judge>.jsonl (문항 1개 = 1줄), 중간에 끊겨도 이어서 진행
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import openai

from carmind.evaluation.data import append_jsonl, read_jsonl
from carmind.llm import PRICING

GRADER_VERSION = "v1"

INSTRUCTIONS = """\
당신은 자동차 취급설명서 QA에서 모델 답변을 채점하는 엄격한 심사자입니다.
[질문], [정답], [원문], [모델 답변]이 주어집니다. [정답]과 [원문]만 근거로 판정하세요. 상식이나 배경지식으로 보충하지 마세요.

판정 항목:
- correctness: 모델 답변이 질문에 맞게 답했는지.
  correct   = [정답]의 핵심 내용(수치, 조건, 절차, 원인·조치)을 빠짐없이 맞게 말함. 표현이 달라도 됩니다.
  partial   = 핵심 일부만 맞거나, 맞는 내용과 틀린 내용이 섞여 있음.
  incorrect = 핵심이 틀렸거나, 질문과 무관하거나, 답하지 않음.
- hallucination: [정답]과 [원문] 어디에서도 확인되지 않는 사실(수치, 기능, 절차, 경고 등)을 단정적으로 말하면 true.
  일반적인 인사말·맺음말이나 "서비스센터에 문의하세요" 같은 무해한 권고는 환각으로 보지 않습니다.
- safety_omission: 질문 유형이 safety일 때만 판단합니다. [정답]에 있는 경고·주의 사항(위험, 금지 행동, 사고·부상 가능성)을
  모델 답변이 빠뜨렸으면 true. 유형이 safety가 아니면 항상 false.
- reason: 판정 이유를 한국어 한두 문장으로 짧게 적으세요.

브랜드명은 "제조사", 서비스 네트워크 이름은 "공식 서비스센터"로 바뀌어 있을 수 있습니다. 이 차이는 무시하세요.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "hallucination": {"type": "boolean"},
        "safety_omission": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correctness", "hallucination", "safety_omission", "reason"],
    "additionalProperties": False,
}

# 비용 추정용 값. tiktoken 없이 글자 수로 어림한다.
# 한국어는 GPT 계열 토크나이저에서 대략 1~1.5글자가 1토큰이라, 1글자=1토큰으로 잡으면 넉넉한(보수적) 추정이 된다.
# 영문·숫자·공백(ASCII)은 평균 4글자 ≈ 1토큰.
TOKENS_PER_NON_ASCII_CHAR = 1.0
TOKENS_PER_ASCII_CHAR = 0.25
REQUEST_OVERHEAD_TOKENS = 30  # 메시지 구분자 등 형식 토큰
# 출력 = JSON 판정(~80토큰) + 추론(reasoning) 토큰. effort=low에서도 수백 토큰을 쓸 수 있어 넉넉히 잡는다
EST_OUTPUT_TOKENS = 400


def estimate_tokens(text: str) -> int:
    n_ascii = sum(1 for ch in text if ord(ch) < 128)
    n_other = len(text) - n_ascii
    return int(n_other * TOKENS_PER_NON_ASCII_CHAR + n_ascii * TOKENS_PER_ASCII_CHAR) + 1


def build_input(item: dict) -> str:
    """item: question, type, reference, evidence(list), source(원문 청크), prediction."""
    evidence = "\n".join(f"- {e}" for e in item.get("evidence") or []) or "(없음)"
    return (
        f"[질문 유형]\n{item['type']}\n\n"
        f"[질문]\n{item['question']}\n\n"
        f"[정답]\n{item['reference']}\n\n"
        f"[정답 근거 문장]\n{evidence}\n\n"
        f"[원문]\n{item.get('source') or '(없음)'}\n\n"
        f"[모델 답변]\n{item['prediction'] or '(빈 답변)'}"
    )


@dataclass
class CostEstimate:
    n: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def estimate_cost(items: list[dict], model: str, output_tokens_per_item: int = EST_OUTPUT_TOKENS) -> CostEstimate:
    """캐시 할인 없이(최악의 경우) 계산한다. 실제로는 지시문이 캐시되어 이보다 싸다."""
    instr = estimate_tokens(INSTRUCTIONS)
    inp = sum(instr + estimate_tokens(build_input(it)) + REQUEST_OVERHEAD_TOKENS for it in items)
    out = output_tokens_per_item * len(items)
    price = PRICING[model]
    return CostEstimate(len(items), inp, out, (inp * price["input"] + out * price["output"]) / 1e6)


def prediction_hash(prediction: str) -> str:
    """답변이 바뀌었는데(예: 다시 생성) 예전 판정이 붙는 것을 막기 위한 지문."""
    return hashlib.sha1(prediction.encode("utf-8")).hexdigest()[:12]


@dataclass
class Grade:
    qa_id: str
    type: str
    judge: str
    grader_version: str
    pred_hash: str
    correctness: str | None = None
    hallucination: bool | None = None
    safety_omission: bool | None = None
    reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class Grader:
    def __init__(self, llm, judge: str):
        """llm: StructuredLLM 또는 같은 call()을 가진 객체 (테스트에서는 가짜 객체를 넣는다)."""
        self.llm = llm
        self.judge = judge

    def grade(self, item: dict) -> Grade:
        g = Grade(item["qa_id"], item["type"], self.judge, GRADER_VERSION, prediction_hash(item["prediction"]))
        try:
            resp = self.llm.call(
                INSTRUCTIONS, build_input(item), SCHEMA, "grade", f"carmind-eval-grade-{GRADER_VERSION}"
            )
        except openai.APIError as e:
            g.error = f"{type(e).__name__}: {e}"
            return g
        d = resp.data
        g.correctness, g.hallucination, g.reason = d["correctness"], d["hallucination"], d["reason"]
        # safety가 아닌 문항은 판정자가 true를 줘도 의미가 없으므로 코드에서 확실히 false로 둔다
        g.safety_omission = d["safety_omission"] if item["type"] == "safety" else False
        g.input_tokens, g.output_tokens, g.cost_usd = resp.input_tokens, resp.output_tokens, resp.cost_usd
        return g


def load_valid_grades(path: Path, items: list[dict]) -> dict[str, dict]:
    """재사용할 수 있는 기존 판정: 오류 없이 끝났고, 채점 버전과 답변 지문이 지금과 같은 것만."""
    hashes = {it["qa_id"]: prediction_hash(it["prediction"]) for it in items}
    valid = {}
    for r in read_jsonl(path):
        if r.get("error") or r.get("grader_version") != GRADER_VERSION:
            continue
        if hashes.get(r["qa_id"]) == r.get("pred_hash"):
            valid[r["qa_id"]] = r  # 같은 qa_id가 여러 줄이면 나중 줄이 이긴다
    return valid


def grade_all(items: list[dict], grader: Grader, out_path: Path, workers: int = 8) -> list[Grade]:
    """items를 병렬로 채점해 out_path에 한 줄씩 덧붙인다. 이미 판정된 문항은 호출하는 쪽에서
    load_valid_grades()로 걸러서 넘긴다. 이번 실행의 결과를 돌려준다."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(grader.grade, it) for it in items]
        results = []
        for i, fut in enumerate(as_completed(futures), 1):
            g = fut.result()
            append_jsonl(out_path, asdict(g))
            results.append(g)
            if i % 20 == 0 or i == len(items):
                print(f"  {i}/{len(items)}  cost=${sum(x.cost_usd for x in results):.4f}", flush=True)
    return results
