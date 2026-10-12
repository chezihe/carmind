"""PDF 매뉴얼 파싱: 목차(TOC) 기준으로 섹션을 나누고 텍스트를 정제한다.

PDF는 "글자를 좌표에 찍어 놓은 그림"에 가깝다. 문단·목록·표 같은 구조 정보가 없어서
글자의 위치(x, y)와 크기를 보고 구조를 다시 추론해야 한다.

처리 순서 (페이지마다):
    1. 줄(Line) 단위로 위치·글자 크기·텍스트를 모은다
    2. 버릴 것 제거: 머리말·꼬리말(여러 페이지 상하단에 반복되는 줄), 표 영역 안의 글자(표는 따로 변환)
    3. 깨진 글자 복원: 라벨 전용 폰트('҃Ҋ' → '[경고]'), 아이콘 폰트 → '[아이콘]'
    4. 숨은 이미지 ID(아주 작거나 흰색인 글자)는 본문에서 빼고 이미지 라벨로 보관
    5. 위치 순으로 정렬(2단 문서는 왼쪽 단 → 오른쪽 단) 후, 줄을 문단·목록 항목으로 다시 조립

그 다음 문서 전체 텍스트를 목차 제목 위치로 잘라 섹션을 만든다.
출력: 섹션 하나가 레코드 하나인 JSONL
"""

import bisect
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

# 라벨 글리프 → 원래 글자. 페이지를 이미지로 렌더링해서 눈으로 확인한 값이다.
# 한 라벨이 두 span으로 쪼개져 나오기도 해서('҃' + 'Ҋ') 조각도 등록한다.
LABEL_GLYPHS = {
    "҃Ҋ": "경고",
    "Ҋ": "경고",
    "҃": None,  # 쪼개진 앞 조각은 버린다
    "઱੄": "주의",
    "ঌইفӝ": "알아두기",
}

ICON_PLACEHOLDER = "[아이콘]"
HIDDEN_TEXT_MAX_SIZE = 3.0  # pt. 이보다 작은 글자는 사람 눈에 안 보이는 숨은 텍스트
TOP_MARGIN = 45  # pt. 페이지 상단 여백 (머리말 후보 영역)
BOTTOM_MARGIN = 50  # pt. 페이지 하단 여백 (꼬리말 후보 영역)
BOILERPLATE_MIN_PAGES = 5  # 여백 영역에서 이만큼 이상 반복되면 머리말·꼬리말로 본다
PAGE_NUMBER_RE = re.compile(r"^[\dIVX]+(-\d+)?$")  # "53", "3-47", "I-6"
WHITE = 0xFFFFFF
BULLET = "•"
IMAGE_ID_RE = re.compile(r"^\w+_[\w-]+$")  # "2C_DriverAirbag", "2C_BuiltInCamMenu_2-1"
IMAGE_MARKER = "<<img:{}>>"
IMAGE_MARKER_RE = re.compile(r"<<img:([\w-]+)>>")
# 내용이 아니라 쪽 번호 안내뿐인 장(최상위 목차). 학습 데이터에서 뺀다
NON_CONTENT_CHAPTERS = {"색인", "찾아보기", "목차", "차례"}
SENTENCE_END = (".", ":", "?", "!", ")")


@dataclass
class Section:
    doc_id: str
    section_id: str  # 예: "manual_a_2025:0042"
    path: list[str]  # 목차 경로: [장, 절, 항]
    page_start: int  # 1부터 시작
    page_end: int
    text: str
    image_ids: list[str] = field(default_factory=list)


@dataclass
class Line:
    x: float
    y: float
    size: float  # 글자 크기(pt). 제목/본문 구분에 사용
    text: str
    kind: str = "text"  # text | bullet | label | table | image
    x1: float = 0.0  # 오른쪽 끝. 2단 문서에서 어느 단에 속하는지 판단에 사용
    col: int = -1  # -1: 1단 문서 또는 양쪽 단에 걸친 줄, 0: 왼쪽 단, 1: 오른쪽 단
    col_x: float = 0.0  # 속한 단의 왼쪽 끝. 단 안에서의 들여쓰기 비교용
    inline_bullet: bool = False  # 줄 첫머리에 •가 붙어 있는 목록 항목 (매뉴얼에 따라 •가 별도 객체이기도 함)


@dataclass
class DocContext:
    """문서 전체에서 한 번 계산해 모든 페이지 처리에 쓰는 정보."""

    body_fonts: set[str]
    vocab: Counter  # 줄 안쪽에 등장한 단어 빈도. 줄바꿈 위치의 띄어쓰기 판단에 사용
    boilerplate: set[str] = field(default_factory=set)  # 머리말·꼬리말 (boilerplate_key로 정규화)
    two_column: bool = False  # 2단 레이아웃 문서인가


# ---------- 1~4. 페이지에서 줄 추출 ----------


def detect_body_fonts(doc: pymupdf.Document, sample_pages: int = 50) -> set[str]:
    """한글이 실제로 쓰인 폰트 = 본문 폰트. 그 외 폰트의 글자는 아이콘으로 본다.

    폰트 이름을 하드코딩하지 않아서 다른 제조사 매뉴얼에도 그대로 쓸 수 있다.
    """
    fonts = set()
    for page in doc.pages(0, min(sample_pages, doc.page_count)):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if re.search(r"[가-힣]", span["text"]):
                        fonts.add(span["font"])
    return fonts


def boilerplate_key(text: str) -> str:
    """머리말·꼬리말 비교용 정규화: 숫자·공백을 지운다. "편의 장치239"와 "편의 장치240"이 같아진다."""
    return re.sub(r"[\d\s]+", "", text)


def in_margin(y: float, page_height: float) -> bool:
    return y < TOP_MARGIN or y > page_height - BOTTOM_MARGIN


def detect_boilerplate(doc: pymupdf.Document) -> set[str]:
    """여러 페이지의 상하단 여백에 반복해서 나오는 줄 = 머리말·꼬리말.

    위치만으로 지우면 안 된다: 어떤 매뉴얼은 섹션 제목이 페이지 맨 위에 있다.
    반복 빈도를 함께 보면 장 이름·쪽 번호처럼 매 페이지 반복되는 줄만 골라낼 수 있다.
    """
    counts: Counter = Counter()
    for page in doc:
        h = page.rect.height
        keys = set()
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if in_margin(line["bbox"][1], h):
                    key = boilerplate_key("".join(s["text"] for s in line["spans"]))
                    if key:
                        keys.add(key)
        counts.update(keys)  # 한 페이지에서 여러 번 나와도 1번으로 센다
    return {k for k, n in counts.items() if n >= BOILERPLATE_MIN_PAGES}


WORD_EDGE_PUNCT = "()[]'\"‘’“”,.·"


def build_vocab(doc: pymupdf.Document) -> Counter:
    """각 줄의 '안쪽' 단어 빈도를 센다.

    줄의 첫 단어와 마지막 단어는 줄바꿈으로 잘린 조각일 수 있어서 제외한다.
    안쪽 단어는 확실히 온전한 단어이므로, 줄바꿈 지점에서 붙여 쓸지 띄어 쓸지 판단하는 사전이 된다.
    """
    vocab: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                words = "".join(s["text"] for s in line["spans"]).split()
                vocab.update(w.strip(WORD_EDGE_PUNCT) for w in words[1:-1])
    return vocab


COLUMN_GAP = 5  # pt. 페이지 중앙선에서 이만큼 넘어가도 같은 단으로 본다


def column_of(x0: float, x1: float, mid: float) -> int:
    if x1 <= mid + COLUMN_GAP:
        return 0
    if x0 >= mid - COLUMN_GAP:
        return 1
    return -1  # 중앙선을 가로지름 = 양쪽 단에 걸친 줄


def detect_two_column(doc: pymupdf.Document, step: int = 10) -> bool:
    """페이지를 표본 조사해서 절반 넘게 2단이면 2단 문서로 본다.

    페이지마다 따로 판단하지 않는 이유: 1단 문서에도 경고등 아이콘 표처럼 격자로 배치된 페이지가 있어서
    2단으로 잘못 보일 수 있다. 문서 단위로 정하면 1단 문서의 결과는 전혀 바뀌지 않는다.
    """
    pages = range(min(30, doc.page_count // 4), max(1, doc.page_count - 30), step)
    votes = 0
    for pno in pages:
        page = doc[pno]
        mid = page.rect.width / 2
        cols = Counter()
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if "".join(s["text"] for s in line["spans"]).strip():
                    cols[column_of(line["bbox"][0], line["bbox"][2], mid)] += 1
        total = sum(cols.values())
        if cols[0] >= 5 and cols[1] >= 5 and cols[-1] <= 0.25 * total:
            votes += 1
    return votes > len(pages) / 2


def line_from_spans(raw_line: dict, body_fonts: set[str]) -> Line | None:
    x, y = raw_line["bbox"][0], raw_line["bbox"][1]
    x1 = raw_line["bbox"][2]
    size = max(s["size"] for s in raw_line["spans"])
    stripped = "".join(s["text"] for s in raw_line["spans"]).strip()

    if not stripped:
        return None
    if stripped == BULLET:
        return Line(x, y, size, BULLET, "bullet", x1)
    if stripped in LABEL_GLYPHS:
        label = LABEL_GLYPHS[stripped]
        return Line(x, y, size, f"[{label}]", "label", x1) if label else None
    # 숨은 이미지 ID: 매뉴얼에 따라 아주 작은 글자(2pt)이거나 흰색 글자(6pt)로 들어 있다.
    # 흰색 글자 전체를 숨김으로 보면 안 된다: 검은 배경 위 흰 글씨("사양 적용 시")는 정상 텍스트다
    is_white = all(s.get("color") == WHITE for s in raw_line["spans"] if s["text"].strip())
    if IMAGE_ID_RE.match(stripped) and (size < HIDDEN_TEXT_MAX_SIZE or is_white):
        return Line(x, y, size, IMAGE_MARKER.format(stripped), "image", x1)
    if size < HIDDEN_TEXT_MAX_SIZE:
        return None

    spans = raw_line["spans"]
    inline_bullet = False
    if stripped.startswith(BULLET) and len(spans) > 1 and spans[0]["text"].strip() == BULLET:
        # "• 항목 내용": •를 떼고, 들여쓰기 기준은 • 뒤 본문 시작 위치로 (둘째 줄과 맞춰 이어 붙이기 위해)
        inline_bullet = True
        spans = spans[1:]
        x = next((sp["bbox"][0] for sp in spans if sp["text"].strip()), x)
        # •는 본문보다 크게 인쇄된다(10.7pt vs 9pt). 크기 비교(제목/본문 구분)는 본문 글자 기준으로
        size = max((sp["size"] for sp in spans if sp["text"].strip()), default=size)

    parts = []
    for span in spans:
        text = span["text"]
        if text.strip() and span["font"] not in body_fonts:
            parts.append(f" {ICON_PLACEHOLDER} ")
        else:
            parts.append(text)
    return Line(x, y, size, "".join(parts).lstrip(), "text", x1, inline_bullet=inline_bullet)


def table_to_markdown(table, raw_lines: list[dict], ctx: DocContext) -> str:
    """표를 마크다운으로 변환한다.

    표 라이브러리의 기본 텍스트 추출은 폰트를 구분하지 않아 아이콘 글자가 깨진 채 들어간다.
    그래서 칸(cell) 영역만 표에서 받고, 칸 안의 글자는 본문과 같은 규칙(line_from_spans)으로 처리한다.
    """
    rows = []
    for row in table.rows:
        cells = []
        for bbox in row.cells:
            if bbox is None:  # 병합된 칸
                cells.append("")
                continue
            rect = pymupdf.Rect(bbox)
            cell = ""
            for raw_line in raw_lines:
                x0, y0, x1, y1 = raw_line["bbox"]
                if rect.contains(pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)):
                    line = line_from_spans(raw_line, ctx.body_fonts)
                    if line is not None and line.kind not in ("image", "bullet"):
                        # 줄 끝 공백이 단어 경계 정보라서 strip하지 않고 본문과 같은 규칙으로 잇는다
                        cell = join_wrapped(cell, line.text, ctx.vocab) if cell else line.text
            # 마크다운 표 안에서는 줄바꿈과 '|'를 쓸 수 없다
            cells.append(normalize(cell).replace("\n", " ").replace("|", "/"))
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    separator = "|" + "---|" * len(table.rows[0].cells)
    return "\n".join([rows[0], separator, *rows[1:]])


def is_boilerplate(line: Line, page_height: float, ctx: DocContext) -> bool:
    if not in_margin(line.y, page_height):
        return False
    return PAGE_NUMBER_RE.match(line.text.strip()) is not None or boilerplate_key(line.text) in ctx.boilerplate


def extract_lines(page: pymupdf.Page, ctx: DocContext) -> list[Line]:
    h = page.rect.height
    # 상하단 여백 안에만 있는 "표"는 장 번호 탭 같은 머리말·꼬리말 장식이다 (매 페이지 반복됨)
    tables = [
        t for t in page.find_tables().tables if not (t.bbox[3] <= TOP_MARGIN or t.bbox[1] >= h - BOTTOM_MARGIN)
    ]
    table_rects = [pymupdf.Rect(t.bbox) for t in tables]

    raw_lines = [ln for block in page.get_text("dict")["blocks"] for ln in block.get("lines", [])]
    lines = [
        Line(t.bbox[0], t.bbox[1], 0, table_to_markdown(t, raw_lines, ctx), "table", t.bbox[2]) for t in tables
    ]
    for raw_line in raw_lines:
        line = line_from_spans(raw_line, ctx.body_fonts)
        if line is None or is_boilerplate(line, page.rect.height, ctx):
            continue
        if any(rect.contains(pymupdf.Point(line.x + 1, line.y + 1)) for rect in table_rects):
            continue
        lines.append(line)

    if ctx.two_column:
        return order_two_column(lines, page.rect.width / 2)
    # 위→아래, 같은 높이면 왼→오른. 글머리 기호(•)가 페이지 끝에 따로 추출되기 때문에 필요
    lines.sort(key=lambda ln: (round(ln.y), ln.x))
    return lines


def order_two_column(lines: list[Line], mid: float) -> list[Line]:
    """2단 페이지의 읽는 순서: 양쪽 단에 걸친 줄(전체 폭 제목·표)을 경계로 구간을 나누고,
    각 구간 안에서 왼쪽 단을 위→아래로 다 읽은 뒤 오른쪽 단을 읽는다.
    """
    for ln in lines:
        ln.col = column_of(ln.x, ln.x1, mid)
    for c in (0, 1):
        members = [ln for ln in lines if ln.col == c and ln.kind == "text"]
        edge = min((ln.x for ln in members), default=0.0)
        for ln in lines:
            if ln.col == c:
                ln.col_x = edge

    by_y = lambda ln: (round(ln.y), ln.x)
    left = sorted((ln for ln in lines if ln.col == 0), key=by_y)
    right = sorted((ln for ln in lines if ln.col == 1), key=by_y)
    ordered: list[Line] = []
    for full in sorted((ln for ln in lines if ln.col == -1), key=by_y):
        ordered += [ln for ln in left if ln.y < full.y] + [ln for ln in right if ln.y < full.y]
        left = [ln for ln in left if ln.y >= full.y]
        right = [ln for ln in right if ln.y >= full.y]
        ordered.append(full)
    return ordered + left + right


# ---------- 5. 줄 → 문단 조립 ----------


def starts_new_paragraph(prev: Line, cur: Line) -> bool:
    """이전 줄과 이어지는 문장인지, 새 문단인지 판단한다."""
    if prev.kind != "text" or cur.kind != "text":
        return True
    if abs(prev.size - cur.size) > 0.3:  # 글자 크기가 다르면 제목 ↔ 본문 경계
        return True
    if abs((prev.x - prev.col_x) - (cur.x - cur.col_x)) > 3:  # 단 안에서의 들여쓰기가 다르면 다른 문단
        return True
    if prev.col == 0 and cur.col == 1 and cur.y < prev.y:
        return False  # 왼쪽 단 맨 아래에서 끊긴 문장이 오른쪽 단 맨 위로 이어짐
    return cur.y - prev.y > prev.size * 2  # 줄 간격이 크게 벌어지면 다른 문단


def join_wrapped(prev_text: str, cur_text: str, vocab: Counter | None = None) -> str:
    """PDF 자동 줄바꿈 복원.

    - 앞 줄 끝에 공백이 있으면 확실한 단어 경계 → 그대로 이어 붙임 ("여러 " + "위험")
    - 문장이 끝난 줄 뒤는 새 문장 → 줄바꿈
    - 공백이 없으면 애매하다. 단어 중간("탑승"+"자가")일 수도, 공백이 생략된 단어 경계
      ("버튼을"+"누르고")일 수도 있다 → 문서 단어 사전(vocab)으로 판단 (should_join)
    """
    if prev_text.endswith(" "):
        return prev_text + cur_text
    if prev_text.endswith(SENTENCE_END):
        return prev_text + "\n" + cur_text
    if vocab is not None and not should_join(prev_text, cur_text, vocab):
        return prev_text + " " + cur_text
    return prev_text + cur_text


def should_join(prev_text: str, cur_text: str, vocab: Counter) -> bool:
    """줄 끝 조각 a와 다음 줄 첫 조각 b를 붙여 쓸지 판단한다.

    a+b가 사전에 있으면 붙인다 ("서비스센"+"터").
    a+b는 없는데 a, b가 각각 사전에 있으면 띄운다 ("버튼을"+"누르고").
    둘 다 아니면 붙인다 (한국어 매뉴얼은 단어 중간 줄바꿈이 흔하다).
    """
    a = (prev_text.split() or [""])[-1].strip(WORD_EDGE_PUNCT)
    b = (cur_text.split() or [""])[0].strip(WORD_EDGE_PUNCT)
    if not a or not b:
        return True
    if vocab[a + b] > 0:
        return True
    return not (vocab[a] > 0 and vocab[b] > 0)


LIST_MARKER_RE = re.compile(r"^(\d+\.|-)\s")  # "1. ", "- " 로 시작하는 목록 항목
HANGING_INDENT_MAX = 12  # pt. 목록 첫 줄보다 이만큼 이내로 오른쪽에서 시작하는 줄은 같은 항목의 둘째 줄


def is_hanging_continuation(first: Line, first_text: str, cur: Line) -> bool:
    """번호 목록의 내어쓰기: "1. 모든 도어를 닫으" / "   십시오." 처럼 둘째 줄이 번호 뒤 본문 위치에서 시작한다."""
    return (
        LIST_MARKER_RE.match(first_text) is not None
        and cur.kind == "text"
        # 다음 줄도 목록 기호로 시작하면 내어쓰기가 아니라 들여쓴 하위 항목이다
        and not cur.inline_bullet
        and LIST_MARKER_RE.match(cur.text.lstrip()) is None
        and 0 < cur.x - first.x <= HANGING_INDENT_MAX
        and abs(first.size - cur.size) <= 0.3
        and 0 < cur.y - first.y <= first.size * 2
    )


def assemble(lines: list[Line], vocab: Counter | None = None) -> str:
    bullets = [ln for ln in lines if ln.kind == "bullet"]
    paragraphs: list[str] = []
    prev: Line | None = None
    para_lines = 0  # 현재 문단의 줄 수 (내어쓰기는 첫 줄 바로 다음 줄에서만 판단)

    for line in lines:
        if line.kind == "bullet":
            continue
        # 같은 단에서, 같은 높이의 바로 왼쪽에 •가 있으면 목록 항목
        is_item = line.kind == "text" and (
            line.inline_bullet
            or any(b.col == line.col and abs(b.y - line.y) <= 1.5 and 0 < line.x - b.x < 20 for b in bullets)
        )
        # 목록 기호로 시작하는 줄은 위치와 상관없이 새 항목 (하위 항목 "- "가 상위 항목 본문과 같은 x에 있기도 함)
        starts_with_marker = line.kind == "text" and LIST_MARKER_RE.match(line.text.lstrip()) is not None
        hanging = prev is not None and para_lines == 1 and is_hanging_continuation(prev, paragraphs[-1], line)
        can_join = hanging or not starts_new_paragraph(prev, line) if prev is not None else False
        if prev is not None and not is_item and not starts_with_marker and can_join:
            paragraphs[-1] = join_wrapped(paragraphs[-1], line.text, vocab)
            para_lines += 1
            # 내어쓰기 줄 다음부터는 둘째 줄 위치를 기준으로 들여쓰기를 비교한다
            prev = line
            continue
        paragraphs.append(f"- {line.text}" if is_item else line.text)
        para_lines = 1
        prev = line

    return normalize("\n".join(p.strip() for p in paragraphs))


def normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_page(page: pymupdf.Page, ctx: DocContext) -> str:
    return assemble(extract_lines(page, ctx), ctx.vocab)


# ---------- 목차 기준 섹션 분할 ----------


def split_by_toc(doc: pymupdf.Document, doc_id: str) -> list[Section]:
    """목차 제목 위치를 기준으로 문서 전체 텍스트를 섹션으로 자른다.

    1. 모든 페이지 텍스트를 이어 붙이고, 각 페이지가 시작하는 위치(offset)를 기록
    2. 목차 순서대로, 해당 페이지~다음 페이지 범위에서 제목 문자열을 찾음
       (못 찾으면 그 페이지의 시작 위치를 사용)
    3. 이번 제목 위치 ~ 다음 제목 위치 = 한 섹션
    """
    ctx = DocContext(
        body_fonts=detect_body_fonts(doc),
        vocab=build_vocab(doc),
        boilerplate=detect_boilerplate(doc),
        two_column=detect_two_column(doc),
    )
    page_offsets, parts, pos = [], [], 0
    for page in doc:
        text = extract_page(page, ctx) + "\n"
        page_offsets.append(pos)
        parts.append(text)
        pos += len(text)
    full = "".join(parts)

    def page_of(offset: int) -> int:
        return bisect.bisect_right(page_offsets, offset)  # 1부터 시작

    toc = doc.get_toc()
    starts, prev = [], 0
    for _, title, page in toc:
        page = min(max(page, 1), doc.page_count)
        page_start = page_offsets[page - 1]
        # 제목은 해당 페이지나 다음 페이지에서만 찾는다. 범위를 열어 두면 문서 끝 '색인'에 있는
        # 같은 제목까지 가 버려서, 그 뒤 섹션이 모두 비게 된다
        window_end = page_offsets[page + 1] if page + 1 < doc.page_count else len(full)
        found = full.find(title, max(page_start, prev), window_end)
        prev = found if found != -1 else max(page_start, prev)
        starts.append(prev)
    starts.append(len(full))

    sections, stack = [], []
    for i, (level, title, _) in enumerate(toc):
        stack = stack[: level - 1] + [title]
        if stack[0].strip() in NON_CONTENT_CHAPTERS:
            continue
        raw = full[starts[i] : starts[i + 1]]
        sections.append(
            Section(
                doc_id=doc_id,
                section_id=f"{doc_id}:{i:04d}",
                path=list(stack),
                page_start=page_of(starts[i]),
                page_end=page_of(max(starts[i], starts[i + 1] - 1)),
                text=normalize(IMAGE_MARKER_RE.sub("", raw)),
                image_ids=IMAGE_MARKER_RE.findall(raw),
            )
        )
    return sections


def parse_pdf(pdf_path: Path, out_path: Path) -> list[Section]:
    doc_id = pdf_path.stem
    with pymupdf.open(pdf_path) as doc:
        sections = split_by_toc(doc, doc_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(asdict(s), ensure_ascii=False) + "\n" for s in sections), encoding="utf-8"
    )
    return sections
