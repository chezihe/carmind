"""LoRA SFT 학습.

uv run python scripts/train_sft.py                                    # configs/training/sft_smoke.yaml
uv run python scripts/train_sft.py --max-steps 5 --out outputs/dry    # 드라이런: s/step 측정
uv run python scripts/train_sft.py --config configs/training/<다른 설정>.yaml

결과: <output_dir>/adapter (LoRA 가중치), train_log.jsonl (loss 기록), summary.json
"""

import argparse
from pathlib import Path

from carmind.training.sft_train import load_sft_config, train

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/training/sft_smoke.yaml"))
parser.add_argument("--max-steps", type=int, default=None, help="설정의 max_steps 대신 사용 (드라이런용)")
parser.add_argument("--out", type=Path, default=None, help="설정의 output_dir 대신 사용")
parser.add_argument("--dtype", default=None, choices=["auto", "bfloat16", "float16", "float32"])
args = parser.parse_args()

cfg = load_sft_config(args.config)
if args.out:
    cfg.output_dir = args.out
if args.dtype:
    cfg.runtime.dtype = args.dtype

summary = train(cfg, max_steps=args.max_steps)
print(
    f"완료: {summary['max_steps']} step, {summary['train_time_sec'] / 60:.1f}분 ({summary['sec_per_step']:.2f}s/step), "
    f"val loss {summary['val_loss_start']:.4f} → {summary['val_loss_end']:.4f}"
)
print(f"저장: {cfg.output_dir}/adapter")
