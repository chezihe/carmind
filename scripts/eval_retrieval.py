"""검색기 평가 (API 비용 없음): BM25(토크나이저별) / dense(임베딩 모델별) / hybrid(RRF)의 Recall@k·MRR.

uv run python scripts/eval_retrieval.py --eval data/processed/eval/<doc>.jsonl
uv run python scripts/eval_retrieval.py --eval ... --embed bge-m3 e5-base --tokenizers bigram kiwi

정답은 문항을 만든 원문 청크 1개다. 같은 내용이 다른 청크에도 있으면 그 청크를 찾아도 '실패'로 세므로
recall은 실제보다 낮게 나온다. 그래서 1위가 틀린 문항에 대해 그 청크가 정답 근거 문장(evidence)을 모두
포함하는지도 세어 '근거 포함 기준' 보정치를 함께 낸다.
결과: <out>/results.json, report.md, misses.jsonl (최고 방법의 1위 실패 문항 표본)
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from carmind.data.synth_qa import anonymize, is_grounded, load_replacements, squash
from carmind.evaluation.data import load_eval_set
from carmind.retrieval.bm25 import BM25
from carmind.retrieval.corpus import embed_corpus, load_corpus, top_ids
from carmind.retrieval.dense import SPECS, DenseEncoder
from carmind.retrieval.fusion import rrf
from carmind.retrieval.metrics import answer_coverage, gold_rank, mrr, recall_at_k
from carmind.retrieval.tokenize import get_tokenizer

KS = (1, 3, 5, 10)

parser = argparse.ArgumentParser()
parser.add_argument("--eval", type=Path, nargs="+", required=True)
parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks"))
parser.add_argument("--anonymize", type=Path, default=Path("data/anonymize.yaml"))
parser.add_argument("--tokenizers", nargs="+", default=["bigram", "word_bigram", "kiwi"])
parser.add_argument("--embed", nargs="*", default=["bge-m3"], choices=sorted(SPECS))
parser.add_argument("--scopes", nargs="+", default=["per-doc", "all-docs"], choices=["per-doc", "all-docs"])
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=100, help="순위 목록 길이 (MRR은 이 안에서만)")
parser.add_argument("--latency-n", type=int, default=50, help="질의 1개씩 재는 지연 시간 표본 수")
parser.add_argument("--miss-sample", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=Path("outputs/retrieval/eval"))
args = parser.parse_args()
sys.stdout.reconfigure(line_buffering=True)

replacements = load_replacements(args.anonymize)
questions = sorted(load_eval_set(args.eval), key=lambda q: q["qa_id"])
queries = [q["question"] for q in questions]
eval_docs = sorted({q["doc_id"] for q in questions})
print(f"문항 {len(questions)}개, 평가 문서 {len(eval_docs)}개")


def score(rankings: list[list[str]], idx: list[int] | None = None) -> dict:
    """엄격(정답 청크만 정답) + 관대(근거 문장을 모두 포함한 청크도 정답) 두 기준으로 잰다."""
    idx = range(len(questions)) if idx is None else idx
    ranks = [gold_rank(rankings[i], questions[i]["chunk_id"]) for i in idx]
    lenient = [lenient_rank(i, rankings[i]) for i in idx]
    return {
        **{f"R@{k}": round(recall_at_k(ranks, k), 4) for k in KS}, "MRR": round(mrr(ranks), 4),
        **{f"evR@{k}": round(recall_at_k(lenient, k), 4) for k in KS}, "evMRR": round(mrr(lenient), 4),
        "n": len(ranks),
    }


_grounded_cache: dict[tuple[int, str], bool] = {}


def lenient_rank(i: int, ranking: list[str], max_k: int = max(KS)) -> int | None:
    """정답 청크이거나 정답 근거 문장(evidence)을 모두 포함한 첫 청크의 순위. 상위 max_k개만 본다."""
    q = questions[i]
    for r, cid in enumerate(ranking[:max_k], start=1):
        if cid == q["chunk_id"]:
            return r
        key = (i, cid)
        if key not in _grounded_cache:
            _grounded_cache[key] = bool(evidence[i]) and all(e in squashed[cid] for e in evidence[i])
        if _grounded_cache[key]:
            return r
    return None


def by_type(rankings: list[list[str]]) -> dict:
    groups = defaultdict(list)
    for i, q in enumerate(questions):
        groups[q["type"]].append(i)
    return {t: score(rankings, idx) for t, idx in sorted(groups.items())}


def per_doc_rank(rank_fn) -> list[list[str]]:
    """per-doc 범위: 문항의 문서마다 따로 만든 인덱스로 검색."""
    out = [None] * len(questions)
    for doc in eval_docs:
        idx = [i for i, q in enumerate(questions) if q["doc_id"] == doc]
        for i, r in zip(idx, rank_fn(doc, [queries[i] for i in idx]), strict=True):
            out[i] = r
    return out


results: dict = {"n_questions": len(questions), "scopes": {}, "latency_ms": {}}
all_rankings: dict[tuple[str, str], list[list[str]]] = {}
corpora = {
    "per-doc": {d: load_corpus(args.chunks, [d], replacements) for d in eval_docs},
    "all-docs": {None: load_corpus(args.chunks, None, replacements)},
}
text_of = {c.chunk_id: c.text for c in corpora["all-docs"][None]}
squashed = {cid: squash(t) for cid, t in text_of.items()}
evidence = [[squash(anonymize(e, replacements)) for e in q.get("evidence", []) if e.strip()] for q in questions]
print(f"코퍼스: per-doc {sum(len(v) for v in corpora['per-doc'].values())}청크, all-docs {len(text_of)}청크")


def run_scope(scope: str, rank_fn_for_corpus):
    if scope == "per-doc":
        return per_doc_rank(lambda doc, qs: rank_fn_for_corpus(corpora["per-doc"][doc], qs))
    return rank_fn_for_corpus(corpora["all-docs"][None], queries)


# --- BM25 ---
for tok_name in args.tokenizers:
    tok = get_tokenizer(tok_name)
    for scope in args.scopes:
        t0, query_sec = time.time(), [0.0]  # 색인 시간은 빼고 검색 시간만 지연 시간으로 센다

        def bm25_fn(chunks, qs, tok=tok, query_sec=query_sec):
            index = BM25([c.text for c in chunks], tok)
            ids = [c.chunk_id for c in chunks]
            t = time.time()
            ranked = [top_ids(index.scores(q), ids, args.depth) for q in qs]
            query_sec[0] += time.time() - t
            return ranked

        all_rankings[(scope, f"bm25-{tok_name}")] = run_scope(scope, bm25_fn)
        results["latency_ms"][f"{scope}/bm25-{tok_name}"] = round(1000 * query_sec[0] / len(queries), 2)
        print(f"  bm25-{tok_name} [{scope}] {time.time() - t0:.1f}s (색인 포함)")

# --- dense ---
for embed_name in args.embed:
    enc = DenseEncoder(SPECS[embed_name], args.device, args.batch_size)
    t0 = time.time()
    q_emb = enc.encode(queries, "query")
    emb = {scope: {k: embed_corpus(enc, v) for k, v in corpora[scope].items()} for scope in args.scopes}
    print(f"  {embed_name} 임베딩 {time.time() - t0:.0f}s (캐시가 있으면 빠름)")
    for scope in args.scopes:
        t = time.time()
        if scope == "per-doc":
            ranked = [None] * len(questions)
            for doc in eval_docs:
                ids = [c.chunk_id for c in corpora["per-doc"][doc]]
                for i, q in enumerate(questions):
                    if q["doc_id"] == doc:
                        ranked[i] = top_ids(emb[scope][doc] @ q_emb[i], ids, args.depth)
        else:
            ids = [c.chunk_id for c in corpora["all-docs"][None]]
            ranked = [top_ids(emb[scope][None] @ q_emb[i], ids, args.depth) for i in range(len(questions))]
        search_ms = 1000 * (time.time() - t) / len(questions)
        all_rankings[(scope, f"dense-{embed_name}")] = ranked
        results["latency_ms"][f"{scope}/dense-{embed_name} (search only)"] = round(search_ms, 2)
    # 질의 인코딩 지연: 실제 서비스처럼 1개씩, 캐시 없이 (첫 호출은 워밍업으로 버림)
    sample = random.Random(args.seed).sample(queries, min(args.latency_n, len(queries)))
    enc.encode_uncached(sample[0], "query")
    t = time.time()
    for s in sample:
        enc.encode_uncached(s, "query")
    results["latency_ms"][f"dense-{embed_name} query encode (batch 1, {enc.device})"] = round(
        1000 * (time.time() - t) / len(sample), 1)
    enc.close()

# --- hybrid: 각 범위에서 MRR이 가장 높은 BM25 + 각 dense 모델 ---
for scope in args.scopes:
    bm25_names = [m for (s, m) in all_rankings if s == scope and m.startswith("bm25-")]
    if not bm25_names or not args.embed:
        continue
    best_bm25 = max(bm25_names, key=lambda m: score(all_rankings[(scope, m)])["MRR"])
    for embed_name in args.embed:
        t = time.time()
        b, d = all_rankings[(scope, best_bm25)], all_rankings[(scope, f"dense-{embed_name}")]
        all_rankings[(scope, f"hybrid-{best_bm25[5:]}+{embed_name}")] = [
            rrf([b[i], d[i]])[:args.depth] for i in range(len(questions))
        ]
        results["latency_ms"][f"{scope}/hybrid fusion only"] = round(1000 * (time.time() - t) / len(questions), 2)

for (scope, method), ranked in all_rankings.items():
    results["scopes"].setdefault(scope, {})[method] = {"overall": score(ranked), "by_type": by_type(ranked)}


# --- 1위 실패 분석: 1위 청크가 정답 근거를 포함하는가 ---
def miss_analysis(scope: str, method: str) -> dict:
    ranked = all_rankings[(scope, method)]
    misses = [i for i, q in enumerate(questions) if ranked[i][0] != q["chunk_id"]]
    rows = []
    for i in misses:
        q = questions[i]
        top = ranked[i][0]
        ev = [anonymize(e, replacements) for e in q.get("evidence", [])]
        rows.append({
            "qa_id": q["qa_id"], "type": q["type"], "question": q["question"], "reference": q["answer"],
            "gold": q["chunk_id"], "gold_rank": gold_rank(ranked[i], q["chunk_id"]), "top1": top,
            "top1_other_doc": top.rsplit(":", 1)[0] != q["doc_id"],
            "top1_has_all_evidence": is_grounded(ev, text_of[top]),
            "top1_answer_coverage": round(answer_coverage(q["answer"], text_of[top]), 3),
            "gold_answer_coverage": round(answer_coverage(q["answer"], text_of[q["chunk_id"]]), 3),
        })
    n_ev = sum(r["top1_has_all_evidence"] for r in rows)
    # 근거 포함 기준 R@1: 정답 청크가 1위이거나, 1위 청크가 근거 문장을 모두 포함
    return {
        "scope": scope, "method": method, "n_misses": len(rows), "misses_with_all_evidence": n_ev,
        "R@1_strict": round(1 - len(rows) / len(questions), 4),
        "R@1_evidence_lenient": round(1 - (len(rows) - n_ev) / len(questions), 4),
        "rows": rows,
    }


args.out.mkdir(parents=True, exist_ok=True)
analyses = {}
for scope in args.scopes:
    best = max(results["scopes"][scope], key=lambda m: results["scopes"][scope][m]["overall"]["MRR"])
    a = miss_analysis(scope, best)
    sample = random.Random(args.seed).sample(a["rows"], min(args.miss_sample, len(a["rows"])))
    analyses[scope] = {k: v for k, v in a.items() if k != "rows"} | {"sample": sample}
    with (args.out / f"misses_{scope}.jsonl").open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r | {"top1_text": text_of[r["top1"]], "gold_text": text_of[r["gold"]]},
                               ensure_ascii=False) + "\n")
results["miss_analysis"] = {s: {k: v for k, v in a.items() if k != "sample"} for s, a in analyses.items()}


# --- 리포트 ---
def table(rows: dict[str, dict]) -> list[str]:
    cols = [*(f"R@{k}" for k in KS), "MRR", "evR@1", "evR@3", "evR@5"]
    lines = ["| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {m} | " + " | ".join(f"{r['overall'][c]:.3f}" for c in cols) + " |" for m, r in rows.items()]
    return lines


md = [f"# 검색 평가 ({len(questions)}문항, 정답 = 원문 청크 1개)", "",
      "evR@k: 정답 청크 또는 정답 근거 문장(evidence)을 모두 포함한 청크가 상위 k개 안에 있는 비율 (중복 내용 보정)", ""]
for scope, rows in results["scopes"].items():
    md += [f"## {scope}", "", *table(rows), ""]
    best = max(rows, key=lambda m: rows[m]["overall"]["MRR"])
    md += [f"### {scope} 최고 방법 ({best}) 유형별", "", "| type | n | R@1 | R@3 | R@5 | R@10 | MRR |", "|---|---|---|---|---|---|---|"]
    md += [f"| {t} | {s['n']} | " + " | ".join(f"{s[c]:.3f}" for c in ("R@1", "R@3", "R@5", "R@10", "MRR")) + " |"
           for t, s in rows[best]["by_type"].items()]
    a = results["miss_analysis"][scope]
    md += ["", (f"1위 실패 {a['n_misses']}개 중 1위 청크가 근거 문장을 모두 포함: {a['misses_with_all_evidence']}개 "
           f"→ R@1 엄격 {a['R@1_strict']:.3f}, 근거 포함 기준 {a['R@1_evidence_lenient']:.3f}"), ""]
md += ["## 지연 시간 (ms/질의)", "", *(f"- {k}: {v}" for k, v in results["latency_ms"].items())]
(args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
(args.out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
print(f"저장: {args.out}/results.json, report.md, misses_*.jsonl")
