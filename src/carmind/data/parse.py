"""PDF 매뉴얼 파싱: 목차(TOC) 기준으로 섹션을 나누고 텍스트를 정제한다.

PDF는 "글자를 좌표에 찍어 놓은 그림"에 가깝다. 문단·목록·표 같은 구조 정보가 없어서
글자의 위치(x, y)와 크기를 보고 구조를 다시 추론해야 한다.

처리 순서 (페이지마다):
    1. 줄(Line) 단위로 위치·글자 크기·텍스트를 모은다
    2. 버릴 것 제거: 꼬리말(페이지 하단), 표 영역 안의 글자(표는 따로 변환)
    3. 깨진 글자 복원: 라벨 전용 폰트('҃Ҋ' → '[경고]'), 아이콘 폰트 → '[아이콘]'
    4. 숨은 이미지 ID(2pt 글자)는 본문에서 빼고 이미지 라벨로 보관
    5. 위치 순으로 정렬 후, 줄을 문단·목록 항목으로 다시 조립

그 다음 문서 전체 텍스트를 목차 제목 위치로 잘라 섹션을 만든다.
출력: 섹션 하나가 레코드 하나인 JSONL
"""

import bisect
import json
import re
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
FOOTER_MARGIN = 50  # pt. 페이지 하단에서 이 거리 안은 꼬리말(장 이름 + 쪽 번호)
BULLET = "•"
IMAGE_ID_RE = re.compile(r"^\w+_\w+$")
IMAGE_MARKER = "<<img:{}>>"
IMAGE_MARKER_RE = re.compile(r"<<img:(\w+)>>")
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


def line_from_spans(raw_line: dict, body_fonts: set[str]) -> Line | None:
    x, y = raw_line["bbox"][0], raw_line["bbox"][1]
    size = max(s["size"] for s in raw_line["spans"])
    stripped = "".join(s["text"] for s in raw_line["spans"]).strip()

    if not stripped:
        return None
    if stripped == BULLET:
        return Line(x, y, size, BULLET, "bullet")
    if stripped in LABEL_GLYPHS:
        label = LABEL_GLYPHS[stripped]
        return Line(x, y, size, f"[{label}]", "label") if label else None
    if size < HIDDEN_TEXT_MAX_SIZE:
        return Line(x, y, size, IMAGE_MARKER.format(stripped), "image") if IMAGE_ID_RE.match(stripped) else None

    parts = []
    for span in raw_line["spans"]:
        text = span["text"]
        if text.strip() and span["font"] not in body_fonts:
            parts.append(f" {ICON_PLACEHOLDER} ")
        else:
            parts.append(text)
    return Line(x, y, size, "".join(parts))


def table_to_markdown(table, raw_lines: list[dict], body_fonts: set[str]) -> str:
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
            texts = []
            for raw_line in raw_lines:
                x0, y0, x1, y1 = raw_line["bbox"]
                if rect.contains(pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)):
                    line = line_from_spans(raw_line, body_fonts)
                    if line is not None and line.kind != "image":
                        texts.append(line.text.strip())
            cells.append(" ".join(texts).replace("|", "/"))
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    separator = "|" + "---|" * len(table.rows[0].cells)
    return "\n".join([rows[0], separator, *rows[1:]])


def extract_lines(page: pymupdf.Page, body_fonts: set[str]) -> list[Line]:
    footer_y = page.rect.height - FOOTER_MARGIN
    tables = page.find_tables().tables
    table_rects = [pymupdf.Rect(t.bbox) for t in tables]

    raw_lines = [ln for block in page.get_text("dict")["blocks"] for ln in block.get("lines", [])]
    lines = [Line(t.bbox[0], t.bbox[1], 0, table_to_markdown(t, raw_lines, body_fonts), "table") for t in tables]
    for raw_line in raw_lines:
        line = line_from_spans(raw_line, body_fonts)
        if line is None or line.y >= footer_y:
            continue
        if any(rect.contains(pymupdf.Point(line.x + 1, line.y + 1)) for rect in table_rects):
            continue
        lines.append(line)

    # 위→아래, 같은 높이면 왼→오른. 글머리 기호(•)가 페이지 끝에 따로 추출되기 때문에 필요
    lines.sort(key=lambda ln: (round(ln.y), ln.x))
    return lines


# ---------- 5. 줄 → 문단 조립 ----------


def starts_new_paragraph(prev: Line, cur: Line) -> bool:
    """이전 줄과 이어지는 문장인지, 새 문단인지 판단한다."""
    if prev.kind != "text" or cur.kind != "text":
        return True
    if abs(prev.size - cur.size) > 0.3:  # 글자 크기가 다르면 제목 ↔ 본문 경계
        return True
    if abs(prev.x - cur.x) > 3:  # 들여쓰기가 다르면 다른 문단
        return True
    return cur.y - prev.y > prev.size * 2  # 줄 간격이 크게 벌어지면 다른 문단


def join_wrapped(prev_text: str, cur_text: str) -> str:
    """PDF 자동 줄바꿈 복원.

    - 단어 경계에서 끊기면 앞 줄 끝에 공백이 남아 있다 → 그대로 이어 붙임 ("여러 " + "위험")
    - 문장이 끝난 줄 뒤는 새 문장 → 줄바꿈
    - 둘 다 아니면 단어 중간에서 끊긴 것 → 공백 없이 이어 붙임 ("탑승" + "자가")
    """
    if prev_text.endswith(" "):
        return prev_text + cur_text
    if prev_text.endswith(SENTENCE_END):
        return prev_text + "\n" + cur_text
    return prev_text + cur_text


def assemble(lines: list[Line]) -> str:
    bullet_rows = {round(ln.y) for ln in lines if ln.kind == "bullet"}
    paragraphs: list[str] = []
    prev: Line | None = None

    for line in lines:
        if line.kind == "bullet":
            continue
        is_item = line.kind == "text" and any(abs(round(line.y) - y) <= 1 for y in bullet_rows)
        if prev is not None and not is_item and not starts_new_paragraph(prev, line):
            paragraphs[-1] = join_wrapped(paragraphs[-1], line.text)
        else:
            paragraphs.append(f"- {line.text}" if is_item else line.text)
        prev = line

    return normalize("\n".join(p.strip() for p in paragraphs))


def normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_page(page: pymupdf.Page, body_fonts: set[str]) -> str:
    return assemble(extract_lines(page, body_fonts))


# ---------- 목차 기준 섹션 분할 ----------


def split_by_toc(doc: pymupdf.Document, doc_id: str) -> list[Section]:
    """목차 제목 위치를 기준으로 문서 전체 텍스트를 섹션으로 자른다.

    1. 모든 페이지 텍스트를 이어 붙이고, 각 페이지가 시작하는 위치(offset)를 기록
    2. 목차 순서대로, 해당 페이지 이후에서 제목 문자열을 찾음
       (못 찾으면 그 페이지의 시작 위치를 사용)
    3. 이번 제목 위치 ~ 다음 제목 위치 = 한 섹션
    """
    body_fonts = detect_body_fonts(doc)
    page_offsets, parts, pos = [], [], 0
    for page in doc:
        text = extract_page(page, body_fonts) + "\n"
        page_offsets.append(pos)
        parts.append(text)
        pos += len(text)
    full = "".join(parts)

    def page_of(offset: int) -> int:
        return bisect.bisect_right(page_offsets, offset)  # 1부터 시작

    toc = doc.get_toc()
    starts, prev = [], 0
    for _, title, page in toc:
        page_start = page_offsets[min(max(page, 1), doc.page_count) - 1]
        found = full.find(title, max(page_start, prev))
        prev = found if found != -1 else max(page_start, prev)
        starts.append(prev)
    starts.append(len(full))

    sections, stack = [], []
    for i, (level, title, _) in enumerate(toc):
        stack = stack[: level - 1] + [title]
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
