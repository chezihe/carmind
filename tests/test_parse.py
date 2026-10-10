from carmind.data.parse import (
    ICON_PLACEHOLDER,
    Line,
    assemble,
    join_wrapped,
    line_from_spans,
)

BODY = {"Body"}


def raw_line(*spans, x=31.0, y=100.0):
    """PyMuPDF get_text('dict')의 line 구조를 흉내 낸다. spans: (text, font, size)"""
    return {
        "bbox": (x, y, x + 100, y + 10),
        "spans": [{"text": t, "font": f, "size": s} for t, f, s in spans],
    }


def test_join_wrapped():
    assert join_wrapped("여러 ", "위험을") == "여러 위험을"  # 단어 경계
    assert join_wrapped("탑승", "자가") == "탑승자가"  # 단어 중간
    assert join_wrapped("합니다.", "다음") == "합니다.\n다음"  # 문장 끝


def test_label_glyph_restored():
    line = line_from_spans(raw_line(("҃Ҋ", "Label", 9.9)), BODY)
    assert (line.kind, line.text) == ("label", "[경고]")
    assert line_from_spans(raw_line(("҃", "Label", 9.9)), BODY) is None  # 쪼개진 앞 조각


def test_icon_font_replaced():
    line = line_from_spans(raw_line(("메뉴 > ", "Body", 8.5), ("Ĥ", "Icon", 8.5), (" 선택", "Body", 8.5)), BODY)
    assert ICON_PLACEHOLDER in line.text and "Ĥ" not in line.text


def test_hidden_image_id_becomes_marker():
    line = line_from_spans(raw_line(("2C_WarningLamp", "Body", 2.0)), BODY)
    assert line.kind == "image"
    assert line_from_spans(raw_line(("작은 글씨", "Body", 2.0)), BODY) is None


def test_assemble_bullets_and_paragraphs():
    lines = [
        Line(31, 25, 12.0, "녹화 해제 조건"),
        Line(31, 40, 8.5, "다음과 같습니다."),
        Line(33, 52, 8.5, "•", "bullet"),
        Line(41, 52, 8.5, "시간이 초과된 "),
        Line(41, 61, 8.5, "경우"),
        Line(33, 73, 8.5, "•", "bullet"),
        Line(41, 73, 8.5, "배터리가 부족한 경우"),
    ]
    assert assemble(lines) == "녹화 해제 조건\n다음과 같습니다.\n- 시간이 초과된 경우\n- 배터리가 부족한 경우"


def test_should_join_uses_document_vocab():
    from collections import Counter

    from carmind.data.parse import should_join

    vocab = Counter({"서비스센터": 5, "버튼을": 3, "누르고": 2, "탑승": 1})
    assert should_join("가까운 서비스센", "터에서", vocab)  # 합친 단어가 사전에 있음
    assert not should_join("시동 버튼을", "누르고 P", vocab)  # 각각 온전한 단어
    assert should_join("운전자 또는 탑승", "자가 심각한", vocab)  # 판단 불가 → 붙임
