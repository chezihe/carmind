from carmind.data.filter import QARecord, dedup_questions, flatten, rule_violation


def rec(qa_id, question, answer="충분히 긴 답변 문장입니다. 레버를 당기십시오.", grounded=True):
    return QARecord(qa_id, "c", "d", "fact", question, answer, ["근거"], grounded, "m", "v2")


def test_rules():
    assert rule_violation(rec("1", "앞좌석 높이는 어떻게 조절하나요?"), []) is None
    assert rule_violation(rec("1", "앞좌석 높이는 어떻게 조절하나요?", grounded=False), []) == "not_grounded"
    assert rule_violation(rec("1", "이 기능은 언제 작동하나요?"), []) == "deictic_question"
    assert rule_violation(rec("1", "타이어 공기압은?", "설명서에 따르면 260 kPa로 맞추면 됩니다."), []) == "doc_reference"
    assert rule_violation(rec("1", "점검은 어디서 받나요?", "가나다센터에서 점검받으십시오."), ["가나다센터"]) == "banned_term"


def test_dedup_keeps_first_and_drops_near_duplicate():
    records = [
        rec("a", "앞좌석 등받이 각도는 어떻게 조절하나요?"),
        rec("b", "앞좌석 등받이 각도는 어떻게 조절하나요"),  # 문장부호만 다름
        rec("c", "타이어 공기압은 얼마로 맞춰야 하나요?"),
    ]
    kept, dropped = dedup_questions(records)
    assert [r.qa_id for r in kept] == ["a", "c"]
    assert dropped[0][1] == "a"


def test_flatten_skips_errors_and_is_sorted():
    results = [
        {"chunk_id": "d:2", "doc_id": "d", "model": "m", "prompt_version": "v2", "error": None,
         "qa_pairs": [{"type": "fact", "question": "q", "answer": "a", "evidence": [], "grounded": True}]},
        {"chunk_id": "d:1", "doc_id": "d", "model": "m", "prompt_version": "v2", "error": "boom", "qa_pairs": []},
    ]
    assert [r.qa_id for r in flatten(results)] == ["d:2:0"]
