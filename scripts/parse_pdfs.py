"""data/raw/*.pdf → data/interim/sections/<id>.jsonl

uv run python scripts/parse_pdfs.py
"""

import argparse
from pathlib import Path

from carmind.data.parse import parse_pdf

parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=Path, default=Path("data/raw"))
parser.add_argument("--out", type=Path, default=Path("data/interim/sections"))
args = parser.parse_args()

for pdf in sorted(args.raw.glob("*.pdf")):
    sections = parse_pdf(pdf, args.out / f"{pdf.stem}.jsonl")
    chars = sum(len(s.text) for s in sections)
    empty = sum(1 for s in sections if not s.text)
    print(f"[ok] {pdf.stem}: {len(sections)} sections, {chars:,} chars, empty={empty}")
