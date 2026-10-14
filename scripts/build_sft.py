"""최종 QA → SFT 대화 데이터

uv run python scripts/build_sft.py --mode closed   # data/processed/sft/closed/{train,val}.jsonl
uv run python scripts/build_sft.py --mode rag      # data/processed/sft/rag/{train,val}.jsonl
"""

import argparse
from pathlib import Path

from carmind.training.sft_data import build

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["closed", "rag"], required=True)
parser.add_argument("--qa", type=Path, default=Path("data/processed/qa"))
parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks"))
parser.add_argument("--out", type=Path, default=Path("data/processed/sft"))
parser.add_argument("--val-ratio", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

stats = build(args.qa, args.chunks, args.out / args.mode, args.mode, args.val_ratio, args.seed)
print(f"[{args.mode}] 문서 {stats['docs']}권 → train {stats['train']} / val {stats['val']}")
