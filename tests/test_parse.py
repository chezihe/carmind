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


def test_white_image_id_is_hidden_but_white_text_is_kept():
    white_id = raw_line(("2C_DriverAirbag", "Body", 6.0))
    white_id["spans"][0]["color"] = 0xFFFFFF
    assert line_from_spans(white_id, BODY).kind == "image"
    white_text = raw_line(("사양 적용 시", "Body", 6.0))  # 검은 배경 위 흰 글씨
    white_text["spans"][0]["color"] = 0xFFFFFF
    assert line_from_spans(white_text, BODY).kind == "text"


def test_boilerplate_key_ignores_page_numbers():
    from carmind.data.parse import boilerplate_key

    assert boilerplate_key("편의 장치239") == boilerplate_key("편의 장치 240") == "편의장치"


def test_inline_bullet_item_joins_hanging_indent():
    first = raw_line(("•", "Body", 9.0), (" 엔진이 최상의 상태로 유지되도록 차량", "Body", 9.0), x=33, y=100)
    first["spans"][0]["bbox"] = (33, 100, 36, 110)
    first["spans"][1]["bbox"] = (41, 100, 200, 110)
    line = line_from_spans(first, BODY)
    assert line.inline_bullet and line.x == 41 and not line.text.startswith("•")
    cont = line_from_spans(raw_line(("을 올바르게 운행하십시오.", "Body", 9.0), x=41, y=111), BODY)
    assert assemble([line, cont]) == "- 엔진이 최상의 상태로 유지되도록 차량을 올바르게 운행하십시오."


def test_two_column_reading_order():
    from carmind.data.parse import order_two_column

    lines = [
        Line(31, 25, 12, "전체 폭 제목", x1=400),
        Line(31, 50, 9, "왼쪽 1", x1=200),
        Line(230, 50, 9, "오른쪽 1", x1=400),
        Line(31, 61, 9, "왼쪽 2", x1=200),
        Line(230, 61, 9, "오른쪽 2", x1=400),
    ]
    assert [ln.text for ln in order_two_column(lines, mid=218)] == ["전체 폭 제목", "왼쪽 1", "왼쪽 2", "오른쪽 1", "오른쪽 2"]


def test_hyphenated_image_id_is_hidden():
    line = raw_line(("2C_BuiltInCamMenu_2-1", "Body", 2.0))
    assert line_from_spans(line, BODY).kind == "image"


def test_numbered_item_hanging_indent_joins():
    lines = [
        Line(221, 224, 9.0, "1. 모든 도어(후드 및 테일게이트 포함)를 닫으"),
        Line(230, 234, 9.0, "십시오."),
        Line(221, 247, 9.0, "2. 스마트 키의 도어 잠금 버튼을 누르십시오."),
    ]
    assert assemble(lines) == "1. 모든 도어(후드 및 테일게이트 포함)를 닫으십시오.\n2. 스마트 키의 도어 잠금 버튼을 누르십시오."


def test_inline_bullet_size_ignores_bigger_bullet_glyph():
    first = raw_line(("•", "Body", 10.7), (" 도어 잠금 버튼을 누르면 도어가 잠깁니", "Body", 9.0), x=229, y=259)
    first["spans"][0]["bbox"] = (229, 259, 234, 272)
    first["spans"][1]["bbox"] = (239, 259, 385, 272)
    cont = raw_line(("다. 이 때 비상 경고등이 깜빡입니다.", "Body", 9.0), x=239, y=272)
    out = assemble([line_from_spans(first, BODY), line_from_spans(cont, BODY)])
    assert out == "- 도어 잠금 버튼을 누르면 도어가 잠깁니다. 이 때 비상 경고등이 깜빡입니다."


def test_nested_sub_item_is_not_merged_into_parent():
    lines = [
        Line(41, 100, 9.0, "- 회전 저항(RRC): 하중에 대한 회전 저항의 비율"),
        Line(49, 111, 9.0, "- 1~4등급까지 있으며 1등급에 가까울수록 효율이 높습니다."),
    ]
    assert assemble(lines).count("\n") == 1
    # 하위 항목 기호가 상위 항목 본문과 같은 x에 있어도 합치지 않는다
    same_x = [Line(41, 100, 9.0, "- 회전 저항의 비율"), Line(41, 111, 9.0, "- 1~4등급까지 있습니다.")]
    assert assemble(same_x) == "- 회전 저항의 비율\n- 1~4등급까지 있습니다."
