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
)
from carmind.evaluation.grade import (
    GRADER_VERSION,
    SCHEMA,
    Grader,
    estimate_cost,
    estimate_tokens,
    grade_all,
    load_valid_grades,
    prediction_hash,
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


def test_grader_forces_safety_omission_false_for_non_safety():
    data = {"correctness": "partial", "hallucination": False, "safety_omission": True, "reason": "일부 누락"}
    assert Grader(FakeLLM(data), "j").grade(_item(type_="fact")).safety_omission is False
    g = Grader(FakeLLM(data), "j").grade(_item(type_="safety"))
    assert g.safety_omission is True and g.correctness == "partial" and g.cost_usd == 0.001


def test_grade_resume_skips_valid_and_regrades_changed(tmp_path):
    out = tmp_path / "grades.jsonl"
    llm = FakeLLM({"correctness": "correct", "hallucination": False, "safety_omission": False, "reason": "ok"})
    items = [_item("q0"), _item("q1")]
    grade_all(items, Grader(llm, "j"), out, workers=2)
    assert llm.calls == 2 and set(load_valid_grades(out, items)) == {"q0", "q1"}

    # q1의 답변이 바뀌면(다시 생성) 예전 판정은 버리고 다시 채점 대상이 된다
    changed = [_item("q0"), _item("q1", prediction="다른 답")]
    assert set(load_valid_grades(out, changed)) == {"q0"}
    # 오류로 끝난 판정도 재시도 대상
    append_jsonl(out, {"qa_id": "q0", "error": "x", "grader_version": GRADER_VERSION, "pred_hash": prediction_hash("답")})
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
