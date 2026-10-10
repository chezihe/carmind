"""OpenAI 구조화 출력 호출 공통 코드 (QA 생성·판정에서 함께 사용).

JSON 스키마(strict)로 출력 형식을 강제하고, usage로 비용을 계산한다.
"""

import json
from dataclasses import dataclass

import openai

# 백만 토큰당 USD (Standard tier). OpenAI 가격표 기준(2026-10 확인).
PRICING = {
    "gpt-6-astra": {"input": 10.00, "cached_input": 1.00, "output": 50.00},
    "gpt-6.1-sol": {"input": 2.00, "cached_input": 0.10, "output": 10.00},
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    # 데이터 공유 무료 한도 대상 모델 (한도 내 사용 시 실제 청구 $0, 여기 값은 한도 초과 시 기준)
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
}


@dataclass
class StructuredResponse:
    data: dict
    input_tokens: int
    output_tokens: int
    cost_usd: float


class StructuredLLM:
    def __init__(self, model: str, effort: str = "low"):
        if model not in PRICING:
            raise SystemExit(f"{model}의 가격이 PRICING에 없습니다. llm.py에 추가해 주세요.")
        self.model = model
        self.effort = effort
        self.client = openai.OpenAI()

    def call(self, instructions: str, input: str, schema: dict, name: str, cache_key: str) -> StructuredResponse:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input,
            reasoning={"effort": self.effort},
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            # 지시문(instructions)이 모든 요청에서 같으므로 캐시로 입력 비용을 줄인다
            prompt_cache_key=cache_key,
            store=False,
        )
        u = response.usage
        # cached_tokens는 input_tokens에 포함되고, reasoning 토큰은 output_tokens에 포함된다
        cached = u.input_tokens_details.cached_tokens or 0
        price = PRICING[self.model]
        cost = (
            (u.input_tokens - cached) * price["input"]
            + cached * price["cached_input"]
            + u.output_tokens * price["output"]
        ) / 1e6
        return StructuredResponse(json.loads(response.output_text), u.input_tokens, u.output_tokens, cost)
