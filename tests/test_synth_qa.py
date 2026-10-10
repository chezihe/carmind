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
