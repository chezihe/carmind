"""현재 환경의 설정과 장비를 출력한다.

uv run python scripts/check_env.py --env local
"""

import argparse

import torch

from carmind.config import load_config
from carmind.device import resolve_device, resolve_dtype

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="local")
args = parser.parse_args()

cfg = load_config(args.env)
device = resolve_device(cfg.runtime.device)
dtype = resolve_dtype(cfg.runtime.dtype, device)

print(f"env    : {cfg.env}")
print(f"model  : {cfg.model.name}")
print(f"samples: {cfg.data.max_samples or 'all'}")
print(f"device : {device}")
print(f"dtype  : {dtype}")

x = torch.randn(1024, 1024, device=device, dtype=dtype)
print(f"matmul : OK {tuple((x @ x).shape)}")
