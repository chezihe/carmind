"""후보 QA → LLM 판정(충실도) → 최종 학습 데이터

uv run python scripts/judge_qa.py --judge gpt-6-luna --limit 20   # 파일럿 (청크 20개)
uv run python scripts/judge_qa.py --judge gpt-6-luna              # 전체
출력: data/interim/qa_judged/<judge>/<doc>.jsonl (판정), data/processed/qa/<doc>.jsonl (통과한 최종 QA)
      holdout 문서(data/splits.yaml)는 data/processed/eval/<doc>.jsonl (평가셋, 학습 데이터와 분리)
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from carmind.data.judge import Judge, judge_all, load_done, verdict_matches
from carmind.data.splits import final_qa_dir, load_holdout
from carmind.data.synth_qa import anonymize, load_replacements

parser = argparse.ArgumentParser()
parser.add_argument("--candidates", type=Path, default=Path("data/interim/qa_candidates"))
parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks"))
parser.add_argument("--judge", default="gpt-6-luna")
parser.add_argument("--effort", default="low", choices=["low", "medium", "high"])
parser.add_argument("--limit", type=int, default=None, help="무작위로 뽑을 청크 수 (파일럿용)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--out", type=Path, default=Path("data"))
parser.add_argument("--anonymize", type=Path, default=Path("data/anonymize.yaml"))
parser.add_argument("--doc", default=None, help="이 문서만 처리 (예: manual_xxx_2025)")
parser.add_argument("--splits", type=Path, default=Path("data/splits.yaml"), help="holdout 문서 목록 (평가셋으로 출력)")
args = parser.parse_args()
holdout = load_holdout(args.splits)

load_dotenv()
replacements = load_replacements(args.anonymize)
judge = Judge(args.judge, args.effort)

for path in sorted(args.candidates.glob("*.jsonl")):
    if args.doc and path.stem != args.doc:
        continue
    candidates = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    chunk_file = args.chunks / path.name
    chunks = {c["chunk_id"]: c for c in map(json.loads, chunk_file.read_text(encoding="utf-8").splitlines())}
    # QA 답변과 같은 치환을 원문에도 적용해서 judge가 치환된 표현을 원문에 없는 내용으로 보지 않게 한다
    for c in chunks.values():
        c["text"] = anonymize(c["text"], replacements)
    if args.limit:
        picked = set(random.Random(args.seed).sample(sorted({qa["chunk_id"] for qa in candidates}), args.limit))
        candidates = [qa for qa in candidates if qa["chunk_id"] in picked]

    if not candidates:
        continue
    # 판정 파일은 생성 버전별로 분리한다. 섞이면 다른 버전 QA의 판정이 같은 qa_id에 잘못 붙는다
    gen_version = f"{candidates[0]['teacher']}_{candidates[0]['prompt_version']}"
    judged_path = args.out / "interim" / "qa_judged" / args.judge / gen_version / path.name
    new = judge_all(candidates, chunks, judge, judged_path, args.workers)
    if new and all(r.error for r in new):
        print("모든 요청이 실패했습니다:", new[0].error)
        continue

    verdicts = {v["qa_id"]: v for r in load_done(judged_path).values() for v in r["verdicts"]}
    # 안전장치: 판정이 지금 후보 QA와 같은 질문에 대한 것인지 확인
    mismatched = [qa["qa_id"] for qa in candidates if qa["qa_id"] in verdicts and not verdict_matches(verdicts[qa["qa_id"]], qa)]
    if mismatched:
        raise SystemExit(f"판정과 QA가 다릅니다 ({len(mismatched)}개, 예: {mismatched[:3]}). 판정 파일을 확인하세요: {judged_path}")
    judged = [qa for qa in candidates if qa["qa_id"] in verdicts]
    fails = Counter()
    final = []
    for qa in judged:
        v = verdicts[qa["qa_id"]]
        for key in ("faithful", "answers_question", "standalone"):
            if not v[key]:
                fails[key] += 1
        if v["faithful"] and v["answers_question"] and v["standalone"]:
            final.append({**qa, "judge": args.judge})

    print(f"\n[{path.stem}] 판정 {len(judged)}개 → 통과 {len(final)} ({len(final) / max(1, len(judged)):.0%})")
    print(f"  실패 항목: {dict(fails)}")
    print(f"  이번 실행 비용 ${sum(r.cost_usd for r in new):.4f}, 토큰 입력 {sum(r.input_tokens for r in new):,} / 출력 {sum(r.output_tokens for r in new):,}")
    if not args.limit:
        # holdout 문서는 build_sft가 읽는 processed/qa가 아니라 processed/eval로 보낸다
        out = final_qa_dir(args.out / "processed", path.stem, holdout) / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final), encoding="utf-8")
        print(f"  최종 데이터 → {out}")
