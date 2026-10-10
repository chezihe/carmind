"""원본 PDF 수집.

uv run python scripts/download_sources.py --sources data/sources.yaml
"""

import argparse
from pathlib import Path

from carmind.data.download import download_all

parser = argparse.ArgumentParser()
parser.add_argument("--sources", type=Path, default=Path("data/sources.yaml"))
parser.add_argument("--out", type=Path, default=Path("data/raw"))
args = parser.parse_args()

download_all(args.sources, args.out)
