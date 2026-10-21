"""평가셋 문항에 모델 답변 생성 → outputs/eval/<run_name>/predictions.jsonl

uv run python scripts/eval_generate.py --eval data/processed/eval/<doc>.jsonl --limit 60 --run base_closed
uv run python scripts/eval_generate.py --eval ... --limit 60 --mode rag --adapter outputs/sft_smoke_rag/adapter --run lora_rag
uv run python scripts/eval_generate.py --config configs/eval/runpod.yaml --eval ... --run 7b_closed   # GPU 서버

같은 --limit, --seed면 어떤 모델이든 같은 문항(type별 비율 유지)을 뽑는다 → 실행끼리 비교 가능.
끊겼다가 같은 명령을 다시 실행하면 답이 없는 문항만 이어서 생성한다.
rag 모드는 문항을 만든 원문 청크를 그대로 넣는 oracle retrieval 설정이다 (carmind.evaluation.generate 참고).
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from carmind.data.synth_qa import load_replacements
from carmind.evaluation.data import (
    git_commit,
    load_chunks,
    load_eval_set,
    stratified_sample,
)
from carmind.evaluation.generate import GenConfig, Generator, check_resume, pending

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=None, help="아래 옵션의 기본값을 담은 yaml (CLI가 우선)")
parser.add_argument("--eval", type=Path, nargs="+", help="평가셋 jsonl (여러 개 가능)")
parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks"))
parser.add_argument("--anonymize", type=Path, default=Path("data/anonymize.yaml"))
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--adapter", default=None, help="LoRA adapter 폴더 (없으면 base 모델)")
parser.add_argument("--mode", choices=["closed", "rag"], default="closed")
parser.add_argument("--limit", type=int, default=None, help="type별 비율을 유지해 뽑을 문항 수")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
parser.add_argument("--run", default=None, help="실행 이름 (기본: 모델·adapter·모드로 자동)")
parser.add_argument("--out", type=Path, default=Path("outputs/eval"))
pre, _ = parser.parse_known_args()
if pre.config:
    # yaml 키는 CLI 옵션 이름과 같게 쓴다 (max_new_tokens 등). 명령줄에 준 값이 yaml보다 우선
    parser.set_defaults(**(yaml.safe_load(pre.config.read_text(encoding="utf-8")) or {}))
args = parser.parse_args()
if not args.eval:
    parser.error("--eval이 필요합니다 (config 또는 명령줄)")

run_name = args.run or "_".join(
    [args.model.split("/")[-1], Path(args.adapter).parent.name if args.adapter else "base", args.mode]
)
run_dir = args.out / run_name
pred_path, meta_path = run_dir / "predictions.jsonl", run_dir / "meta.json"

cfg = GenConfig(
    model=args.model,
    adapter=str(args.adapter) if args.adapter else None,
    mode=args.mode,
    max_new_tokens=args.max_new_tokens,
    batch_size=args.batch_size,
    repetition_penalty=args.repetition_penalty,
    device=args.device,
    dtype=args.dtype,
)
check_resume(meta_path, cfg)

questions = stratified_sample(load_eval_set(args.eval), args.limit, args.seed)
chunks = {}
if args.mode == "rag":
    chunks = load_chunks(args.chunks, {q["doc_id"] for q in questions}, load_replacements(args.anonymize))
    missing = [q["qa_id"] for q in questions if q["chunk_id"] not in chunks]
    if missing:
        raise SystemExit(f"원문 청크가 없는 문항 {len(missing)}개 (예: {missing[:3]})")

todo = pending(questions, pred_path)
print(f"[{run_name}] {len(questions)}문항 중 {len(questions) - len(todo)}개 완료, {len(todo)}개 생성 ({args.mode})")

meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
    "run_name": run_name,
    "generation": {k: v for k, v in vars(cfg).items() if k != "extra"},
    "eval_files": [str(p) for p in args.eval],
    "limit": args.limit,
    "seed": args.seed,
    "qa_ids": [q["qa_id"] for q in questions],
    "git_commit": git_commit(),
    "sessions": [],
}
if meta["qa_ids"] != [q["qa_id"] for q in questions]:
    raise SystemExit("기존 실행과 문항 목록이 다릅니다 (--limit/--seed/--eval 확인). --run을 바꾸세요.")

if todo:
    run_dir.mkdir(parents=True, exist_ok=True)
    stats = Generator(cfg).run(todo, chunks, pred_path)
    # 이어하기로 여러 번 나눠 돌릴 수 있으므로 실행(session)마다 속도·커밋을 따로 남긴다
    meta["sessions"].append(
        {"finished_at": datetime.now(UTC).isoformat(timespec="seconds"), "git_commit": git_commit(),
         "batch_size": cfg.batch_size, **stats}
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {stats['seconds']}s, {stats['sec_per_question']}s/문항, {stats['tokens_per_sec']} tok/s → {pred_path}")
