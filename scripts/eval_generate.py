"""평가셋 문항에 모델 답변 생성 → outputs/eval/<run_name>/predictions.jsonl

uv run python scripts/eval_generate.py --eval data/processed/eval/<doc>.jsonl --limit 60 --run base_closed
uv run python scripts/eval_generate.py --eval ... --limit 60 --mode rag --adapter outputs/sft_smoke_rag/adapter --run lora_rag
uv run python scripts/eval_generate.py --config configs/eval/runpod.yaml --eval ... --run 7b_closed   # GPU 서버

같은 --limit, --seed면 어떤 모델이든 같은 문항(type별 비율 유지)을 뽑는다 → 실행끼리 비교 가능.
끊겼다가 같은 명령을 다시 실행하면 답이 없는 문항만 이어서 생성한다.
rag 모드는 문항을 만든 원문 청크를 그대로 넣는 oracle retrieval 설정이다 (carmind.evaluation.generate 참고).
rag_retrieved 모드는 실제 검색기로 찾은 상위 k개 청크를 넣는다:
uv run python scripts/eval_generate.py --eval ... --limit 60 --mode rag_retrieved --retriever hybrid --top-k 3 \
    --adapter outputs/sft_smoke_rag/adapter --run lora_rag_ret_top3
"""

import argparse
import json
import sys
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
parser.add_argument("--mode", choices=["closed", "rag", "rag_retrieved"], default="closed")
# --- rag_retrieved 전용 ---
parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
parser.add_argument("--top-k", type=int, default=1)
parser.add_argument("--scope", choices=["per-doc", "all-docs"], default="per-doc",
                    help="per-doc: 문항의 문서(차종) 안에서만 검색, all-docs: 모든 문서에서 검색")
parser.add_argument("--bm25-tokenizer", default="bigram", choices=["bigram", "word_bigram", "kiwi"])
parser.add_argument("--embed", default="bge-m3", help="dense·hybrid용 임베딩 모델 (carmind.retrieval.dense.SPECS)")
parser.add_argument("--embed-batch-size", type=int, default=8)
parser.add_argument("--context-max-tokens", type=int, default=1500,
                    help="[참고 문서] 본문 토큰 상한. 넘으면 낮은 순위 청크부터 자른다")
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
sys.stdout.reconfigure(line_buffering=True)  # tee·파일로 보낼 때도 진행 상황이 바로 보이도록
if not args.eval:
    parser.error("--eval이 필요합니다 (config 또는 명령줄)")

run_name = args.run or "_".join(
    [args.model.split("/")[-1], Path(args.adapter).parent.name if args.adapter else "base", args.mode]
)
run_dir = args.out / run_name
pred_path, meta_path = run_dir / "predictions.jsonl", run_dir / "meta.json"

retrieval = None
if args.mode == "rag_retrieved":
    retrieval = {
        "method": args.retriever, "top_k": args.top_k, "scope": args.scope,
        "bm25_tokenizer": args.bm25_tokenizer if args.retriever != "dense" else None,
        "embed": args.embed if args.retriever != "bm25" else None,
        "context_max_tokens": args.context_max_tokens,
    }
cfg = GenConfig(
    model=args.model,
    adapter=str(args.adapter) if args.adapter else None,
    mode=args.mode,
    max_new_tokens=args.max_new_tokens,
    batch_size=args.batch_size,
    repetition_penalty=args.repetition_penalty,
    device=args.device,
    dtype=args.dtype,
    retrieval=retrieval,
)
check_resume(meta_path, cfg)

questions = stratified_sample(load_eval_set(args.eval), args.limit, args.seed)
chunks = {}
if args.mode == "rag":
    chunks = load_chunks(args.chunks, {q["doc_id"] for q in questions}, load_replacements(args.anonymize))
    missing = [q["qa_id"] for q in questions if q["chunk_id"] not in chunks]
    if missing:
        raise SystemExit(f"원문 청크가 없는 문항 {len(missing)}개 (예: {missing[:3]})")


def build_contexts(todo: list[dict]) -> dict[str, dict]:
    """문항마다 검색 → 상위 k개 청크를 이어 붙인 컨텍스트. 생성 모델을 올리기 전에 끝내고 임베딩 모델을 내린다."""
    from transformers import AutoTokenizer

    from carmind.retrieval.corpus import Retriever, build_context, load_corpus
    from carmind.retrieval.dense import SPECS, DenseEncoder
    from carmind.retrieval.metrics import gold_rank

    replacements = load_replacements(args.anonymize)
    # 토큰 수는 생성 모델의 토크나이저로 센다 (Generator와 같은 선택 규칙)
    tok_src = args.adapter if args.adapter and (Path(args.adapter) / "tokenizer_config.json").exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    encoder = DenseEncoder(SPECS[args.embed], batch_size=args.embed_batch_size) if args.retriever != "bm25" else None
    groups: dict[str | None, list[dict]] = {}
    for q in todo:
        groups.setdefault(q["doc_id"] if args.scope == "per-doc" else None, []).append(q)
    contexts = {}
    for doc_id, qs in groups.items():
        corpus = load_corpus(args.chunks, [doc_id] if doc_id else None, replacements)
        retriever = Retriever(corpus, args.retriever, args.bm25_tokenizer, encoder)
        for q, ranking in zip(qs, retriever.search_many([q["question"] for q in qs]), strict=True):
            top = ranking[:args.top_k]
            text, truncated = build_context([retriever.text[c] for c in top], args.context_max_tokens, count_tokens)
            contexts[q["qa_id"]] = {
                "text": text, "retrieved": top, "gold_rank": gold_rank(ranking, q["chunk_id"]),
                "context_tokens": count_tokens(text), "context_truncated": truncated,
            }
    if encoder:
        encoder.close()
    n_hit = sum(1 for c in contexts.values() if c["gold_rank"] and c["gold_rank"] <= args.top_k)
    n_trunc = sum(c["context_truncated"] for c in contexts.values())
    print(f"검색 완료: 정답 청크 포함 {n_hit}/{len(contexts)}, 길이 상한으로 잘림 {n_trunc}개")
    return contexts


todo = pending(questions, pred_path)
print(f"[{run_name}] {len(questions)}문항 중 {len(questions) - len(todo)}개 완료, {len(todo)}개 생성 ({args.mode})")

meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
    "run_name": run_name,
    "generation": cfg.meta(),
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
    # 생성 전에 meta를 먼저 써 둔다. 첫 실행이 중간에 끊겨도 다음 실행이 check_resume으로 설정을 비교할 수 있다
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    contexts = build_contexts(todo) if args.mode == "rag_retrieved" else None
    stats = Generator(cfg).run(todo, chunks, pred_path, contexts)
    # 이어하기로 여러 번 나눠 돌릴 수 있으므로 실행(session)마다 속도·커밋을 따로 남긴다
    meta["sessions"].append(
        {"finished_at": datetime.now(UTC).isoformat(timespec="seconds"), "git_commit": git_commit(),
         "batch_size": cfg.batch_size, **stats}
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {stats['seconds']}s, {stats['sec_per_question']}s/문항, {stats['tokens_per_sec']} tok/s → {pred_path}")
