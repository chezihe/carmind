"""LoRA SFT 학습.

uv run python scripts/train_sft.py                                    # configs/training/sft_smoke.yaml
uv run python scripts/train_sft.py --max-steps 5 --out outputs/dry    # 드라이런: s/step 측정
uv run python scripts/train_sft.py --config configs/training/<다른 설정>.yaml
uv run python scripts/train_sft.py --config configs/training/sft_7b_closed.yaml \
    --set model.name=Qwen/Qwen2.5-0.5B-Instruct --set train.num_epochs=null --set train.max_steps=4   # 설정 일부 덮어쓰기

결과: <output_dir>/adapter (LoRA 가중치), train_log.jsonl (loss 기록), summary.json
체크포인트(train.save_every_*)를 켜 두면 끊긴 뒤 같은 명령을 다시 실행할 때 마지막 체크포인트에서 이어서 학습한다.
"""

import argparse
import sys
from pathlib import Path

from carmind.training.sft_train import load_sft_config, train

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/training/sft_smoke.yaml"))
parser.add_argument("--max-steps", type=int, default=None, help="설정의 max_steps 대신 사용 (드라이런용)")
parser.add_argument("--out", type=Path, default=None, help="설정의 output_dir 대신 사용")
parser.add_argument("--dtype", default=None, choices=["auto", "bfloat16", "float16", "float32"])
parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="설정 덮어쓰기 (예: train.max_steps=5, data.max_train_samples=null). 여러 번 사용 가능")
args = parser.parse_args()

# 출력이 파일·tee로 갈 때도 줄 단위로 바로 내보낸다 (접속이 끊겨도 로그에 진행 상황이 남도록)
sys.stdout.reconfigure(line_buffering=True)

cfg = load_sft_config(args.config, args.overrides)
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
