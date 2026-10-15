from carmind.data.synth_qa import SCHEMA, is_grounded

CHUNK = "[장 > 절]\n[경고]\n- 고전압 부품을 임의로 분해하지 마십시오.\n타이어 공기압은 260 kPa입니다."


def test_grounded_ignores_whitespace_and_bullets():
    assert is_grounded(["고전압 부품을 임의로  분해하지 마십시오."], CHUNK)
    assert is_grounded(["타이어 공기압은\n260 kPa입니다."], CHUNK)


def test_not_grounded_when_evidence_invented_or_missing():
    assert not is_grounded(["타이어 공기압은 300 kPa입니다."], CHUNK)
    assert not is_grounded([], CHUNK)


def test_schema_is_strict():
    item = SCHEMA["properties"]["qa_pairs"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])


def test_anonymize_and_load_replacements(tmp_path):
    from carmind.data.synth_qa import anonymize, load_replacements

    p = tmp_path / "anonymize.yaml"
    p.write_text('replacements:\n  - from: "가나다센터"\n    to: "공식 서비스센터"\n', encoding="utf-8")
    rules = load_replacements(p)
    assert anonymize("가까운 가나다센터에서 점검하십시오.", rules) == "가까운 공식 서비스센터에서 점검하십시오."
    assert load_replacements(tmp_path / "missing.yaml") == []


def test_generic_cleanup_is_idempotent_and_merges_duplicates():
    from carmind.data.synth_qa import anonymize

    rules = [("가나센터", "공식 서비스센터"), ("다라망", "공식 서비스센터")]
    text = "당사 직영 가나센터나 다라망에서 점검하십시오. 품질은 당사가 보증합니다."
    once = anonymize(text, rules)
    assert once == "공식 서비스센터에서 점검하십시오. 품질은 제조사가 보증합니다."
    assert anonymize(once, rules) == once
    assert anonymize("가나센터 다라망에 방문", rules) == "공식 서비스센터에 방문"


def test_anonymize_ignores_case_and_loose_space_rules(tmp_path):
    from carmind.data.synth_qa import anonymize, load_replacements

    p = tmp_path / "anonymize.yaml"
    p.write_text(
        "replacements:\n"
        '  - from: "Abcd Genuine"\n    to: "순정"\n'
        '  - from: "가나다센터"\n    to: "공식 서비스센터"\n    loose_space: true\n'
        '  - from: "라마"\n    to: "제조사"\n',
        encoding="utf-8",
    )
    rules = load_replacements(p)
    assert anonymize("ABCD Genuine ATF 또는 abcd genuine ATF", rules) == "순정 ATF 또는 순정 ATF"
    # PDF 줄바꿈·띄어쓰기로 끊긴 이름도 loose_space 규칙은 치환
    assert anonymize("가나\n다센터나 가나다\n센터, 가 나다센터", rules) == "공식 서비스센터"
    # loose_space가 없는 규칙은 끊긴 형태를 건드리지 않는다 (짧은 단어가 엉뚱하게 걸리지 않게)
    assert anonymize("라\n마", rules) == "라\n마"
    once = anonymize("당사 직영 가나\n다센터에서 ABCD Genuine 오일로 교환", rules)
    assert once == "공식 서비스센터에서 순정 오일로 교환"
    assert anonymize(once, rules) == once


def test_generic_cleanup_keeps_dangsaja():
    from carmind.data.synth_qa import anonymize

    assert anonymize("다른 당사자의 사생활", []) == "다른 당사자의 사생활"
    assert anonymize("당사가 보증하는 부품", []) == "제조사가 보증하는 부품"
