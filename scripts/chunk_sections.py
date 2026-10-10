"""data/interim/sections/*.jsonl → data/processed/chunks/<id>.jsonl

uv run python scripts/chunk_sections.py --env local
토큰 수는 학습에 쓸 모델(model.name)의 토크나이저로 센다.
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from carmind.config import load_config
from carmind.data.chunk import chunk_file

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="local")
parser.add_argument("--sections", type=Path, default=Path("data/interim/sections"))
parser.add_argument("--out", type=Path, default=Path("data/processed/chunks"))
args = parser.parse_args()

cfg = load_config(args.env)
tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)


def count(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


for path in sorted(args.sections.glob("*.jsonl")):
    chunks = chunk_file(path, args.out / path.name, count, cfg.data.chunk.min_tokens, cfg.data.chunk.max_tokens)
    tokens = sorted(c.n_tokens for c in chunks)
    n = len(tokens)
    print(
        f"[ok] {path.stem}: {n} chunks, tokens p10/p50/p90/max = "
        f"{tokens[n // 10]}/{tokens[n // 2]}/{tokens[9 * n // 10]}/{tokens[-1]}, "
        f"over max = {sum(t > cfg.data.chunk.max_tokens for t in tokens)}"
    )
