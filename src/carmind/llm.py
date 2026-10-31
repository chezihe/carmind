"""구조화 출력 LLM 호출 공통 코드 (QA 생성·판정에서 함께 사용).

JSON 스키마(strict)로 출력 형식을 강제하고, usage로 비용을 계산한다.
모델 이름으로 제공자를 고른다: claude-로 시작하면 Claude, 나머지는 OpenAI (make_llm).
"""

import json
from dataclasses import dataclass

import anthropic
import openai

# 백만 토큰당 USD (Standard tier). OpenAI 가격표 기준(2026-10 확인).
PRICING = {
    "gpt-6-astra": {"input": 10.00, "cached_input": 1.00, "output": 50.00},
    "gpt-6.1-sol": {"input": 2.00, "cached_input": 0.10, "output": 10.00},
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    # 데이터 공유 무료 한도 대상 모델 (한도 내 사용 시 실제 청구 $0, 여기 값은 한도 초과 시 기준)
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    # Claude 표준 가격 (2026-10 확인). OpenAI와 달리 캐시 쓰기에 할증이 붙는다 (5분 TTL = 입력 단가의 1.25배).
    # claude-haiku-5-5는 프롬프트가 100K 토큰을 넘으면 5배 단가지만, 여기서는 청크 단위 요청이라 넘지 않는다
    "claude-fable-5-1": {"input": 10.00, "cached_input": 0.25, "cache_write": 12.50, "output": 50.00},
    "claude-opus-5-5": {"input": 4.00, "cached_input": 0.20, "cache_write": 5.00, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "cached_input": 0.20, "cache_write": 2.50, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "cached_input": 0.01, "cache_write": 0.125, "output": 0.50},
}

# Claude는 max_tokens가 필수다. 추론(thinking) 토큰도 여기에 포함되므로 넉넉히 잡는다 (비스트리밍 권장값)
CLAUDE_MAX_TOKENS = 16000


@dataclass
class StructuredResponse:
    data: dict
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LLMOutputError(Exception):
    """응답은 왔지만 쓸 수 없는 경우 (거절, 길이 초과로 잘림). 토큰은 이미 청구됐으므로 사용량을 함께 넘긴다."""

    def __init__(self, message: str, input_tokens: int, output_tokens: int, cost_usd: float):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd


# 호출하는 쪽에서 잡아 error로 기록하고, 다음 실행 때 그 항목만 재시도한다
LLM_ERRORS = (openai.APIError, anthropic.APIError, LLMOutputError)


def failed_usage(e: Exception) -> tuple[int, int, float]:
    """실패한 호출의 (입력 토큰, 출력 토큰, 비용). API 오류는 응답이 없으니 0."""
    if isinstance(e, LLMOutputError):
        return e.input_tokens, e.output_tokens, e.cost_usd
    return 0, 0, 0.0


class StructuredLLM:
    def __init__(self, model: str, effort: str = "low"):
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


class ClaudeLLM:
    """StructuredLLM과 같은 call()을 제공한다. 재시도는 OpenAI 쪽처럼 SDK 기본값(지수 백오프 2회)에 맡긴다."""

    def __init__(self, model: str, effort: str = "low", client: anthropic.Anthropic | None = None):
        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def call(self, instructions: str, input: str, schema: dict, name: str, cache_key: str) -> StructuredResponse:
        # name, cache_key는 OpenAI 전용. 서버 측 fallbacks는 일부러 켜지 않는다: 거절 시 다른 모델이 대신 답하면
        # 파일 이름의 모델과 실제 판정 모델이 달라지고 단가도 어긋난다. 거절은 오류로 남겨 다시 시도한다
        response = self.client.messages.create(
            model=self.model,
            max_tokens=CLAUDE_MAX_TOKENS,
            # 지시문이 모든 요청에서 같으므로 끝에 캐시 지점을 둔다. 512토큰 미만이면 오류 없이 캐시만 안 된다
            system=[{"type": "text", "text": instructions, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": input}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        u = response.usage
        # OpenAI와 달리 input_tokens는 캐시를 뺀 나머지만 센다. 장부와 비교할 수 있게 프롬프트 전체 크기로 맞춘다
        cache_write = u.cache_creation_input_tokens or 0
        cache_read = u.cache_read_input_tokens or 0
        input_tokens = u.input_tokens + cache_write + cache_read
        price = PRICING[self.model]
        cost = (
            u.input_tokens * price["input"]
            + cache_write * price["cache_write"]
            + cache_read * price["cached_input"]
            + u.output_tokens * price["output"]
        ) / 1e6
        # refusal·max_tokens이면 스키마를 지킨 JSON이라는 보장이 없으므로 본문을 읽기 전에 거른다
        if response.stop_reason != "end_turn":
            raise LLMOutputError(f"stop_reason={response.stop_reason}", input_tokens, u.output_tokens, cost)
        # 추론(thinking) 블록이 앞에 올 수 있어 text 블록을 찾는다
        text = next(b.text for b in response.content if b.type == "text")
        return StructuredResponse(json.loads(text), input_tokens, u.output_tokens, cost)


def make_llm(model: str, effort: str = "low") -> StructuredLLM | ClaudeLLM:
    if model not in PRICING:
        raise SystemExit(f"{model}의 가격이 PRICING에 없습니다. llm.py에 추가해 주세요.")
    return ClaudeLLM(model, effort) if model.startswith("claude-") else StructuredLLM(model, effort)
