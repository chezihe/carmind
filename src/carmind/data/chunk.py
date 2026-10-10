"""섹션 → 청크 재분할.

섹션 길이는 수십~수천 토큰으로 편차가 크다. QA 생성·검색·학습에 쓰기 좋게
[min_tokens, max_tokens] 범위의 청크로 다시 나눈다.

    짧은 섹션: 같은 상위 목차(부모)를 가진 다음 섹션과 합친다
    긴 섹션:   의미 단위(unit)로 쪼갠 뒤 max_tokens를 넘지 않게 다시 담는다
    unit:     기본은 한 줄. 단, 표는 통째로, 라벨([경고] 등)은 바로 다음 내용과 한 덩어리
              → 경고 라벨과 그 내용이 서로 다른 청크로 갈라지지 않게 한다

각 청크 앞에는 목차 경로를 머리말로 붙여, 조각만 봐도 무슨 내용인지 알 수 있게 한다.
"""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

TokenCounter = Callable[[str], int]
LABEL_RE = re.compile(r"^\[(경고|주의|알아두기)\]$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_ids: list[str]
    path: list[str]
    page_start: int
    page_end: int
    text: str  # 목차 경로 머리말 + 본문
    n_tokens: int
    labels: list[str] = field(default_factory=list)  # 포함된 안전 라벨 종류
    image_ids: list[str] = field(default_factory=list)
    sha256: str = ""


def body_without_title(section: dict) -> str:
    """섹션 본문 첫 줄의 제목(목차 경로 머리말과 중복)을 뺀다."""
    text = section["text"]
    title = section["path"][-1]
    text = text.removeprefix(title)
    return text.strip()


def merge_short_sections(
    sections: list[dict], count: TokenCounter, min_tokens: int, max_tokens: int
) -> list[dict]:
    """짧은 섹션을 같은 부모를 가진 바로 앞 섹션과 합친다.

    - 앞 섹션이 짧으면: 앞 섹션에 이어 붙인다 (앞이 min_tokens를 채울 때까지)
    - 지금 섹션이 짧으면: 합쳐도 max_tokens 이하일 때만 앞 섹션에 붙인다
      (장의 마지막 섹션처럼 뒤에 합칠 형제가 없는 경우를 처리)
    제목만 있는 섹션(본문 없음)은 버린다. 내용이 하위 섹션에 들어 있는 경우다.
    """
    merged: list[dict] = []
    for sec in sections:
        body = body_without_title(sec)
        if not body:
            continue
        prev = merged[-1] if merged else None
        if prev is not None and prev["parent"] == sec["path"][:-1]:
            prev_short = count(prev["body"]) < min_tokens
            cur_short = count(body) < min_tokens
            combined = f"{prev['body']}\n\n{sec['path'][-1]}\n{body}"
            if prev_short or (cur_short and count(combined) <= max_tokens):
                prev["body"] = combined
                prev["section_ids"].append(sec["section_id"])
                prev["page_end"] = sec["page_end"]
                prev["image_ids"] = prev["image_ids"] + sec["image_ids"]
                prev["path"] = sec["path"][:-1] or prev["path"]  # 합친 청크의 경로는 공통 부모까지
                continue
        merged.append(
            {**sec, "body": body, "parent": sec["path"][:-1], "section_ids": [sec["section_id"]]}
        )
    return merged


def split_units(body: str) -> list[str]:
    """본문을 자르면 안 되는 최소 단위(unit)로 나눈다."""
    units: list[str] = []
    pending_label = None
    table: list[str] = []

    def push(unit: str) -> None:
        nonlocal pending_label
        if pending_label:
            unit = f"{pending_label}\n{unit}"
            pending_label = None
        units.append(unit)

    for line in body.split("\n"):
        if line.startswith("|"):
            table.append(line)
            continue
        if table:
            push("\n".join(table))
            table = []
        if not line.strip():
            continue
        if LABEL_RE.match(line):
            if pending_label:  # 라벨이 연달아 나오면 앞 라벨은 혼자 unit
                units.append(pending_label)
            pending_label = line
            continue
        push(line)
    if table:
        push("\n".join(table))
    if pending_label:
        units.append(pending_label)
    return units


def split_oversized(unit: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """max_tokens보다 큰 unit을 나눈다.

    - 표: 행 단위로 나누고 조각마다 헤더(첫 2줄)를 반복한다. 헤더가 없으면 열의 의미를 알 수 없다
    - 문단: 문장 단위로 나눈다
    """
    lines = unit.split("\n")
    label = []
    if LABEL_RE.match(lines[0]):
        label, lines = [lines[0]], lines[1:]

    if lines and lines[0].startswith("|") and len(lines) > 2:
        header, rows = lines[:2], lines[2:]
        groups = pack_units(rows, count, max_tokens - count("\n".join(label + header)) - 1)
        return ["\n".join(label + header + [g]) for g in groups]

    sentences = re.split(r"(?<=[.!?])\s+", "\n".join(lines))
    if len(sentences) == 1:
        return [unit]  # 더 나눌 수 없으면 그대로 둔다
    return pack_units(label + sentences, count, max_tokens)


def pack_units(units: list[str], count: TokenCounter, max_tokens: int) -> list[str]:
    """unit을 순서대로 담다가 max_tokens를 넘기 직전에 새 청크를 시작한다(greedy packing)."""
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if count(unit) > max_tokens and len(unit.split("\n")) + len(unit.split(". ")) > 2:
            pieces = split_oversized(unit, count, max_tokens)
            if len(pieces) > 1:
                if current:
                    chunks.append("\n".join(current))
                    current = []
                chunks.extend(pieces[:-1])
                unit = pieces[-1]
        candidate = "\n".join([*current, unit])
        if current and count(candidate) > max_tokens:
            chunks.append("\n".join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def make_header(path: list[str]) -> str:
    return "[" + " > ".join(path) + "]"


def chunk_sections(
    sections: list[dict], count: TokenCounter, min_tokens: int = 100, max_tokens: int = 512
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in merge_short_sections(sections, count, min_tokens, max_tokens):
        header = make_header(sec["path"])
        budget = max_tokens - count(header) - 1
        for body in pack_units(split_units(sec["body"]), count, budget):
            text = f"{header}\n{body}"
            chunks.append(
                Chunk(
                    chunk_id=f"{sec['doc_id']}:{len(chunks):05d}",
                    doc_id=sec["doc_id"],
                    section_ids=sec["section_ids"],
                    path=sec["path"],
                    page_start=sec["page_start"],
                    page_end=sec["page_end"],
                    text=text,
                    n_tokens=count(text),
                    labels=sorted({m.group(1) for m in re.finditer(r"\[(경고|주의|알아두기)\]", body)}),
                    image_ids=sec["image_ids"],
                    sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                )
            )
    return chunks


def dedup_exact(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """본문이 완전히 같은 청크를 제거한다. (거의 같은 청크는 MinHash 단계에서 처리)"""
    seen: set[str] = set()
    kept = []
    for c in chunks:
        if c.sha256 in seen:
            continue
        seen.add(c.sha256)
        kept.append(c)
    return kept, len(chunks) - len(kept)


def chunk_file(
    sections_path: Path, out_path: Path, count: TokenCounter, min_tokens: int, max_tokens: int
) -> list[Chunk]:
    sections = [json.loads(line) for line in sections_path.read_text(encoding="utf-8").splitlines()]
    chunks, _ = dedup_exact(chunk_sections(sections, count, min_tokens, max_tokens))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8"
    )
    return chunks
