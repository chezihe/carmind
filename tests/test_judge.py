from carmind.data.judge import SCHEMA, Verdict, build_input


def test_build_input_numbers_qas():
    text = build_input("원문", [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}])
    assert "[원문]\n원문" in text and "[QA 0]" in text and "[QA 1]\n질문: Q2" in text


def test_verdict_passed_requires_all():
    assert Verdict("x", True, [], True, True).passed
    assert not Verdict("x", False, ["추가 문장"], True, True).passed


def test_schema_is_strict():
    item = SCHEMA["properties"]["verdicts"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])


def test_verdict_keeps_question_for_consistency_check():
    from dataclasses import asdict

    assert asdict(Verdict("x", True, [], True, True, "타이어 공기압은?"))["question"] == "타이어 공기압은?"


def test_verdict_matches_checks_answer_and_accepts_legacy():
    from carmind.data.judge import verdict_matches

    qa = {"question": "Q", "answer": "A2"}
    assert not verdict_matches({"question": "Q", "answer": "A1"}, qa)  # 답변이 바뀌면 다시 판정
    assert verdict_matches({"question": "Q"}, qa)  # answer 필드 없는 예전 기록
    assert not verdict_matches({"question": "다른 질문"}, qa)
