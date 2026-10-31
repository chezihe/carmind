import json
from types import SimpleNamespace

import anthropic
import openai
import pytest

from carmind.budget import DailyTokenBudget
from carmind.data.judge import SCHEMA as JUDGE_SCHEMA
from carmind.data.judge import Judge
from carmind.data.synth_qa import SCHEMA as QA_SCHEMA
from carmind.data.synth_qa import Teacher
from carmind.evaluation.grade import Grader, estimate_cost, grades_filename
from carmind.llm import (
    PRICING,
    ClaudeLLM,
    LLMOutputError,
    StructuredLLM,
    failed_usage,
    make_llm,
)

MODEL = "claude-haiku-5-5"


class FakeMessages:
    """client.messages 자리에 넣는 가짜. 네트워크 없이 요청 인자를 기록하고 정해진 응답을 돌려준다."""

    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _response(data=None, stop_reason="end_turn", input_tokens=100, write=0, read=0, output_tokens=20):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if data is not None:
        content.append(SimpleNamespace(type="text", text=json.dumps(data, ensure_ascii=False)))
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        cache_creation_input_tokens=write,
        cache_read_input_tokens=read,
        output_tokens=output_tokens,
    )
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)


def _claude(response, model=MODEL):
    fake = FakeMessages(response)
    return ClaudeLLM(model, "low", client=SimpleNamespace(messages=fake)), fake


def test_make_llm_routes_by_model_name(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", lambda: "claude-client")
    monkeypatch.setattr(openai, "OpenAI", lambda: "openai-client")
    assert isinstance(make_llm("claude-sonnet-5-5"), ClaudeLLM)
    assert isinstance(make_llm("gpt-6-luna"), StructuredLLM)  # 기본 경로는 그대로 OpenAI


def test_make_llm_rejects_unpriced_model():
    with pytest.raises(SystemExit):
        make_llm("claude-unknown")


def test_script_defaults_stay_openai():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "scripts"
    for script, flag in [("synth_qa.py", "--model"), ("judge_qa.py", "--judge"), ("eval_grade.py", "--judge")]:
        assert f'"{flag}", default="gpt-6-luna"' in (root / script).read_text(encoding="utf-8")


def test_claude_request_uses_schema_effort_and_cache():
    llm, fake = _claude(_response({"qa_pairs": []}))
    llm.call("지시문", "청크", QA_SCHEMA, "qa", "key")
    k = fake.kwargs
    assert k["model"] == MODEL and k["messages"] == [{"role": "user", "content": "청크"}]
    assert k["system"][0]["text"] == "지시문" and k["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert k["output_config"]["effort"] == "low"
    assert k["output_config"]["format"] == {"type": "json_schema", "schema": QA_SCHEMA}
    # API 버전 2026-10-21부터 400이 되는 샘플링 인자는 보내지 않는다
    assert not {"temperature", "top_p", "top_k"} & set(k)


def test_claude_parses_text_block_after_thinking():
    llm, _ = _claude(_response({"qa_pairs": [{"question": "질문"}]}))
    assert llm.call("i", "x", QA_SCHEMA, "qa", "key").data == {"qa_pairs": [{"question": "질문"}]}


def test_claude_usage_totals_and_cost():
    llm, _ = _claude(_response({}, input_tokens=100, write=1000, read=4000, output_tokens=50), "claude-sonnet-5-5")
    r = llm.call("i", "x", QA_SCHEMA, "qa", "key")
    assert r.input_tokens == 5100 and r.output_tokens == 50  # 캐시분까지 합친 프롬프트 전체
    assert r.cost_usd == pytest.approx((100 * 2.00 + 1000 * 2.50 + 4000 * 0.20 + 50 * 10.00) / 1e6)


def test_claude_usage_accepts_missing_cache_fields():
    llm, _ = _claude(_response({}, write=None, read=None))
    assert llm.call("i", "x", QA_SCHEMA, "qa", "key").input_tokens == 100


def test_cache_write_is_input_price_times_1_25():
    for model, price in PRICING.items():
        if model.startswith("claude-"):
            assert price["cache_write"] == pytest.approx(price["input"] * 1.25)


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_response_raises_with_billed_usage(stop_reason):
    llm, _ = _claude(_response(None, stop_reason=stop_reason))
    with pytest.raises(LLMOutputError) as e:
        llm.call("i", "x", QA_SCHEMA, "qa", "key")
    assert failed_usage(e.value)[:2] == (100, 20) and failed_usage(e.value)[2] > 0


def test_teacher_records_refusal_as_retryable_error(monkeypatch):
    llm, _ = _claude(_response(None, stop_reason="refusal"))
    monkeypatch.setattr("carmind.data.synth_qa.make_llm", lambda *a: llm)
    r = Teacher(MODEL).generate({"chunk_id": "c", "doc_id": "d", "text": "원문"})
    assert "refusal" in r.error and r.qa_pairs == [] and r.input_tokens == 100 and r.cost_usd > 0


def test_teacher_generates_with_claude(monkeypatch):
    qa = {"type": "fact", "question": "질문", "answer": "답", "evidence": ["원문"]}
    llm, _ = _claude(_response({"qa_pairs": [qa]}))
    monkeypatch.setattr("carmind.data.synth_qa.make_llm", lambda *a: llm)
    r = Teacher(MODEL).generate({"chunk_id": "c", "doc_id": "d", "text": "원문"})
    assert r.error is None and r.model == MODEL and r.qa_pairs[0].grounded


def test_judge_works_with_claude(monkeypatch):
    v = {"qa_index": 0, "faithful": True, "unsupported": [], "answers_question": True, "standalone": True}
    llm, fake = _claude(_response({"verdicts": [v]}))
    monkeypatch.setattr("carmind.data.judge.make_llm", lambda *a: llm)
    r = Judge(MODEL).judge_chunk({"chunk_id": "c", "text": "원문"}, [{"qa_id": "q", "question": "Q", "answer": "A"}])
    assert r.error is None and r.judge == MODEL and r.verdicts[0].passed
    assert fake.kwargs["output_config"]["format"]["schema"] == JUDGE_SCHEMA


def _grade_item():
    return {
        "qa_id": "q0", "type": "fact", "question": "질문", "reference": "정답",
        "evidence": [], "source": "원문", "prediction": "답",
    }


def test_grader_records_claude_judge_and_error_usage():
    llm, _ = _claude(_response(None, stop_reason="max_tokens"))
    g = Grader(llm, MODEL).grade(_grade_item())
    assert g.judge == MODEL and "max_tokens" in g.error and g.input_tokens == 100 and g.cost_usd > 0


def test_grades_files_are_separated_by_judge():
    assert grades_filename(MODEL) != grades_filename("gpt-6-luna")
    assert MODEL in grades_filename(MODEL)


def test_estimate_prices_claude_input_at_cache_write_rate():
    est = estimate_cost([_grade_item()], MODEL, output_tokens_per_item=100)
    price = PRICING[MODEL]
    assert est.cost_usd == pytest.approx((est.input_tokens * price["cache_write"] + 100 * price["output"]) / 1e6)


def test_claude_tokens_never_count_toward_openai_free_pool(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    DailyTokenBudget(ledger, MODEL, limit=None).record(500)
    assert DailyTokenBudget(ledger, "gpt-5.4-mini", limit=100).used == 0
    assert DailyTokenBudget(ledger, MODEL, limit=100).exhausted  # Claude는 모델별로 따로 센다
