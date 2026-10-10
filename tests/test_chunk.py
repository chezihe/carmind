from carmind.data.chunk import chunk_sections, pack_units, split_units


def count(text: str) -> int:
    """테스트용 토큰 카운터: 글자 수"""
    return len(text)


def section(i, path, text, page=1):
    return {
        "doc_id": "doc",
        "section_id": f"doc:{i:04d}",
        "path": path,
        "page_start": page,
        "page_end": page,
        "text": text,
        "image_ids": [],
    }


def test_label_stays_with_its_content():
    units = split_units("설명\n[경고]\n- 분해하지 마십시오.\n다음")
    assert units == ["설명", "[경고]\n- 분해하지 마십시오.", "다음"]


def test_table_is_one_unit():
    units = split_units("표 앞\n| a | b |\n|---|---|\n| 1 | 2 |\n표 뒤")
    assert units[1] == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_large_table_split_repeats_header():
    table = "| 항목 | 값 |\n|---|---|\n" + "\n".join(f"| 행{i} | {i * 111} |" for i in range(10))
    pieces = pack_units([table], count, max_tokens=60)
    assert len(pieces) > 1
    assert all(p.startswith("| 항목 | 값 |\n|---|---|") for p in pieces)


def test_title_only_section_dropped_and_short_sections_merged():
    sections = [
        section(0, ["장"], "장"),  # 제목만 → 버림
        section(1, ["장", "가"], "가\n짧은 내용"),
        section(2, ["장", "나"], "나\n이것도 짧음"),
    ]
    chunks = chunk_sections(sections, count, min_tokens=20, max_tokens=200)
    assert len(chunks) == 1
    assert chunks[0].section_ids == ["doc:0001", "doc:0002"]
    assert chunks[0].text.startswith("[장]")  # 합친 청크의 경로는 공통 부모


def test_chunks_respect_max_tokens():
    body = "\n".join(f"문장 {i}입니다." for i in range(50))
    chunks = chunk_sections([section(0, ["장", "긴 섹션"], f"긴 섹션\n{body}")], count, 10, 80)
    assert len(chunks) > 1
    assert all(c.n_tokens <= 80 for c in chunks)
