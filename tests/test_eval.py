"""평가 파이프라인 단위 테스트 (모델·API 호출 없음)."""

import json
import subprocess
from collections import Counter
from types import SimpleNamespace

import pytest

from carmind.evaluation.data import (
    append_jsonl,
    done_ids,
    git_commit,
    stratified_sample,
)
from carmind.evaluation.generate import (
    GenConfig,
    check_resume,
    count_generated,
    pending,
    prompt_messages,
    render_prompt,
)
from carmind.evaluation.grade import (
    GRADER_VERSION,
    INSTRUCTIONS,
    RUBRIC_HASH,
    SCHEMA,
    Grader,
    estimate_cost,
    estimate_tokens,
    grade_all,
    grades_filename,
    load_valid_grades,
    prediction_hash,
    resolve_safety_omission,
)
from carmind.evaluation.metrics import (
    char_ngram_f1,
    exact_match,
    has_repetition_loop,
    normalize,
)
from carmind.evaluation.report import aggregate, compare_table, same_questions
from carmind.llm import PRICING

# ---------- 자동 지표 ----------


def test_normalize_ignores_spaces_and_punct():
    assert normalize("260 kPa입니다.") == normalize("260kPa 입니다")
    assert exact_match("W5W, 5W입니다.", "w5w 5w입니다")
    assert not exact_match("5W", "W5W")


def test_char_f1_partial_credit_for_korean_endings():
    assert char_ngram_f1("타이어 공기압은 260 kPa", "타이어 공기압은 260 kPa") == 1.0
    partial = char_ngram_f1("타이어 공기압이 260 kPa입니다", "타이어 공기압은 260 kPa")
    assert 0.5 < partial < 1.0  # 조사·어미가 달라도 부분 점수
    assert char_ngram_f1("엔진오일", "와이퍼") == 0.0
    assert char_ngram_f1("", "") == 1.0 and char_ngram_f1("", "정답") == 0.0


def test_char_f1_hand_computed():
    # "abc" → {ab, bc}, "abd" → {ab, bd}: 겹침 1 → P=R=0.5 → F1=0.5
    assert char_ngram_f1("abc", "abd") == pytest.approx(0.5)


def test_repetition_loop_detection():
    loop = "엔진을 끄고 점검하십시오. " * 4
    assert has_repetition_loop(loop)
    assert has_repetition_loop("하십시오" * 10)  # 띄어쓰기 없는 반복
    normal = "시동을 끄십시오. 주차 브레이크를 거십시오. 비상등을 켜십시오."
    assert not has_repetition_loop(normal)


# ---------- 문항 추출 ----------


def _rows():
    counts = {"fact": 40, "safety": 38, "procedure": 37, "troubleshoot": 15}
    return [{"qa_id": f"d:{t}:{i:03d}", "type": t} for t, n in counts.items() for i in range(n)]


def test_stratified_sample_keeps_ratio_and_is_deterministic():
    rows = _rows()
    a = stratified_sample(rows, 13, seed=1)
    assert len(a) == 13
    assert Counter(r["type"] for r in a) == {"fact": 4, "safety": 4, "procedure": 4, "troubleshoot": 1}
    assert a == stratified_sample(list(reversed(rows)), 13, seed=1)  # 입력 순서와 무관
    assert a != stratified_sample(rows, 13, seed=2)
    assert len(stratified_sample(rows, None, seed=1)) == len(rows)


# ---------- 생성: 프롬프트·길이·이어하기 ----------


def test_prompt_matches_training_format_without_answer():
    qa = {"qa_id": "d:1:0", "question": "Q?", "answer": "A."}
    msgs = prompt_messages(qa, "rag", "원문")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "[참고 문서]\n원문\n\n[질문]\nQ?"


def test_count_generated_detects_truncation():
    assert count_generated([5, 6, 99, 0, 0], {99}, 5) == (2, False)
    assert count_generated([5, 6, 7], {99}, 3) == (3, True)


def test_generation_resume(tmp_path):
    out = tmp_path / "predictions.jsonl"
    qs = [{"qa_id": f"q{i}"} for i in range(3)]
    append_jsonl(out, {"qa_id": "q1", "prediction": "x"})
    assert [q["qa_id"] for q in pending(qs, out)] == ["q0", "q2"]


def test_resume_rejects_different_config(tmp_path):
    cfg = GenConfig(model="m", adapter=None, mode="closed")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"generation": {**cfg.identity(), "batch_size": 4}}), encoding="utf-8")
    check_resume(meta, GenConfig(model="m", adapter=None, mode="closed", batch_size=32))  # 배치 크기만 다르면 허용
    with pytest.raises(SystemExit):
        check_resume(meta, GenConfig(model="m", adapter="a", mode="closed"))


def test_done_ids_skips_errors(tmp_path):
    p = tmp_path / "g.jsonl"
    append_jsonl(p, {"qa_id": "a", "error": None})
    append_jsonl(p, {"qa_id": "b", "error": "timeout"})
    assert done_ids(p) == {"a"}


# ---------- 채점: 스키마·비용 추정·가짜 LLM·이어하기 ----------


def _item(qa_id="q0", type_="safety", prediction="답"):
    return {
        "qa_id": qa_id, "type": type_, "question": "질문", "reference": "정답",
        "evidence": ["근거"], "source": "원문", "prediction": prediction,
    }


class FakeLLM:
    def __init__(self, data):
        self.data, self.calls = data, 0

    def call(self, instructions, input, schema, name, cache_key):
        self.calls += 1
        return SimpleNamespace(data=self.data, input_tokens=100, output_tokens=20, cost_usd=0.001)


def test_schema_is_strict():
    assert SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["required"]) == set(SCHEMA["properties"])


def test_estimate_cost_uses_pricing():
    assert estimate_tokens("가나다") >= 3 and estimate_tokens("abcd" * 10) < 15
    est = estimate_cost([_item(), _item("q1")], "gpt-6-luna", output_tokens_per_item=100)
    price = PRICING["gpt-6-luna"]
    assert est.n == 2 and est.output_tokens == 200
    assert est.cost_usd == pytest.approx((est.input_tokens * price["input"] + 200 * price["output"]) / 1e6)


def _judged(omission=True, warnings=("주행 중 조작 금지",), correctness="partial"):
    return {
        "correctness": correctness, "hallucination": False, "reference_warnings": list(warnings),
        "safety_omission": omission, "reason": "일부 누락",
    }


def test_grader_forces_safety_omission_false_for_non_safety():
    data = _judged()
    assert Grader(FakeLLM(data), "j").grade(_item(type_="fact")).safety_omission is False
    g = Grader(FakeLLM(data), "j").grade(_item(type_="safety"))
    assert g.safety_omission is True and g.correctness == "partial" and g.cost_usd == 0.001
    assert g.reference_warnings == ["주행 중 조작 금지"]


def test_chunk_only_warning_is_not_omission():
    # 판정자가 [원문]에만 있는 경고를 보고 true를 줬어도, [정답]에서 뽑은 경고가 없으면 누락이 아니다
    g = Grader(FakeLLM(_judged(warnings=[])), "j").grade(_item(type_="safety"))
    assert g.safety_omission is False and g.correctness == "partial"
    assert resolve_safety_omission("safety", ["  "], True) is False  # 공백뿐인 항목도 경고 없음으로 본다
    assert resolve_safety_omission("safety", ["화상 위험"], True) is True
    assert resolve_safety_omission("safety", ["화상 위험"], False) is False  # 경고가 있어도 판정자가 false면 그대로
    assert resolve_safety_omission("fact", ["화상 위험"], True) is False


def test_rubric_mentions_reference_only_and_warnings_come_first():
    assert "[원문]에만 있는 경고" in INSTRUCTIONS
    keys = list(SCHEMA["properties"])
    assert keys.index("reference_warnings") < keys.index("safety_omission")


def test_grade_output_records_rubric_version(tmp_path):
    out = tmp_path / grades_filename("j")
    assert out.name == f"grades_j_{GRADER_VERSION}.jsonl"
    grade_all([_item()], Grader(FakeLLM(_judged()), "j"), out, workers=1)
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["grader_version"] == GRADER_VERSION and row["rubric_hash"] == RUBRIC_HASH
    assert row["reference_warnings"] == ["주행 중 조작 금지"]


def test_old_rubric_grades_are_not_reused(tmp_path):
    out = tmp_path / "grades.jsonl"
    base = {"qa_id": "q0", "error": None, "pred_hash": prediction_hash("답")}
    append_jsonl(out, {**base, "grader_version": "v1"})  # 예전 버전 (rubric_hash 없음)
    append_jsonl(out, {**base, "grader_version": GRADER_VERSION, "rubric_hash": "00000000"})  # 문구만 바뀐 경우
    assert load_valid_grades(out, [_item()]) == {}
    append_jsonl(out, {**base, "grader_version": GRADER_VERSION, "rubric_hash": RUBRIC_HASH})
    assert set(load_valid_grades(out, [_item()])) == {"q0"}


def test_grade_resume_skips_valid_and_regrades_changed(tmp_path):
    out = tmp_path / "grades.jsonl"
    llm = FakeLLM(_judged(omission=False, correctness="correct"))
    items = [_item("q0"), _item("q1")]
    grade_all(items, Grader(llm, "j"), out, workers=2)
    assert llm.calls == 2 and set(load_valid_grades(out, items)) == {"q0", "q1"}

    # q1의 답변이 바뀌면(다시 생성) 예전 판정은 버리고 다시 채점 대상이 된다
    changed = [_item("q0"), _item("q1", prediction="다른 답")]
    assert set(load_valid_grades(out, changed)) == {"q0"}
    # 오류로 끝난 판정도 재시도 대상
    append_jsonl(out, {
        "qa_id": "q0", "error": "x", "grader_version": GRADER_VERSION, "rubric_hash": RUBRIC_HASH,
        "pred_hash": prediction_hash("답"),
    })
    assert "q0" in load_valid_grades(out, items)  # 앞선 성공 기록은 그대로 유효


# ---------- 집계·비교 ----------


def _pred(qa_id, type_, pred, ref, truncated=False):
    return {"qa_id": qa_id, "type": type_, "prediction": pred, "reference": ref, "n_tokens": 10, "truncated": truncated}


def test_aggregate_scores_and_safety_denominator():
    preds = [
        _pred("a", "safety", "정답", "정답"),
        _pred("b", "safety", "오답", "정답", truncated=True),
        _pred("c", "fact", "정답", "정답"),
        _pred("d", "fact", "x", "정답"),  # 판정 없음(오류) → correctness 분모에서 빠짐
    ]
    grades = {
        "a": {"type": "safety", "correctness": "correct", "hallucination": False, "safety_omission": False},
        "b": {"type": "safety", "correctness": "partial", "hallucination": True, "safety_omission": True},
        "c": {"type": "fact", "correctness": "incorrect", "hallucination": False, "safety_omission": False},
    }
    r = aggregate(preds, grades)
    o = r["overall"]
    assert o["n"] == 4 and o["n_graded"] == 3
    assert o["score"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert o["correct"] == pytest.approx(1 / 3) and o["hallucination"] == pytest.approx(1 / 3)
    assert o["safety_omission"] == pytest.approx(0.5)  # safety 2개 중 1개
    assert o["truncation_rate"] == pytest.approx(0.25) and o["exact_match"] == pytest.approx(0.5)
    assert r["by_type"]["fact"]["safety_omission"] is None
    assert r["by_type"]["safety"]["score"] == pytest.approx(0.75)


def test_safety_omission_conditional_on_correctness():
    def g(correctness, omission):
        return {"type": "safety", "correctness": correctness, "hallucination": False, "safety_omission": omission}

    grades = {
        "c1": g("correct", False), "c2": g("correct", False), "c3": g("correct", False), "c4": g("correct", True),
        "p1": g("partial", True), "p2": g("partial", False),
        "i1": g("incorrect", True),
        "f1": {"type": "fact", "correctness": "correct", "hallucination": False, "safety_omission": False},
    }
    preds = [_pred(q, gr["type"], "x", "y") for q, gr in grades.items()]
    o = aggregate(preds, grades)["overall"]
    assert o["safety_omission"] == pytest.approx(3 / 7)  # safety 7개 중 3개 (fact는 분모에서 빠짐)
    assert o["n_safety_correct"] == 4 and o["safety_omission_correct"] == pytest.approx(1 / 4)
    assert o["n_safety_partial"] == 2 and o["safety_omission_partial"] == pytest.approx(1 / 2)
    # 해당 등급의 safety 문항이 없으면 비율은 None(표에서는 "-")
    only_fact = aggregate([preds[-1]], {"f1": grades["f1"]})["overall"]
    assert only_fact["n_safety_correct"] == 0 and only_fact["safety_omission_correct"] is None


def test_aggregate_without_grades_has_free_metrics_only():
    o = aggregate([_pred("a", "fact", "정답", "정답")], {})["overall"]
    assert o["n_graded"] == 0 and "score" not in o and o["char_f1"] == 1.0


def test_compare_table_and_same_questions():
    preds = [_pred("a", "fact", "정답", "정답")]
    r1 = {"meta": {"qa_ids": ["a"]}, **aggregate(preds, {})}
    r2 = {"meta": {"qa_ids": ["a"]}, **aggregate(preds, {})}
    table = compare_table({"run1": r1, "run2": r2})
    assert "| metric | run1 | run2 |" in table and "| fact score |" in table
    assert same_questions({"run1": r1, "run2": r2})
    assert not same_questions({"run1": r1, "run3": {"meta": {"qa_ids": ["b"]}}})


def test_git_commit_marks_untracked_as_dirty(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init")
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    git("add", ".gitignore")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "ignore")
    assert not git_commit(tmp_path).endswith("-dirty")

    # .gitignore 대상은 무시
    (tmp_path / "ignored.txt").write_text("x")
    assert not git_commit(tmp_path).endswith("-dirty")

    # 미추적 새 파일은 dirty
    (tmp_path / "new.py").write_text("x")
    assert git_commit(tmp_path).endswith("-dirty")


class TemplateStub:
    """enable_thinking을 받는 템플릿 흉내 (프롬프트 끝부분만 확인한다)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None):
        tail = "<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n"
        return "<|im_start|>assistant\n" + tail


def test_render_prompt_requires_think_off():
    tok = TemplateStub()
    assert render_prompt(tok, [], {"enable_thinking": False}).endswith("</think>\n\n")
    with pytest.raises(ValueError, match="생각 모드"):
        render_prompt(tok, [])


def test_genconfig_template_kwargs_absent_when_unset():
    cfg = GenConfig(model="m", adapter=None, mode="closed")
    assert "chat_template_kwargs" not in cfg.identity() and "chat_template_kwargs" not in cfg.meta()
    on = GenConfig(model="m", adapter=None, mode="closed", chat_template_kwargs={"enable_thinking": False})
    assert on.identity()["chat_template_kwargs"] == {"enable_thinking": False}
    assert on.meta()["chat_template_kwargs"] == {"enable_thinking": False}


def test_resume_rejects_changed_template_kwargs(tmp_path):
    meta = tmp_path / "meta.json"
    old = GenConfig(model="m", adapter=None, mode="closed")
    meta.write_text(json.dumps({"generation": old.meta()}), encoding="utf-8")
    check_resume(meta, GenConfig(model="m", adapter=None, mode="closed"))
    with pytest.raises(SystemExit):
        check_resume(meta, GenConfig(model="m", adapter=None, mode="closed", chat_template_kwargs={"enable_thinking": False}))
