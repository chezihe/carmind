"""청크 → 합성 QA 생성.

uv run python scripts/synth_qa.py --limit 20                     # 파일럿 (무작위 20개)
uv run python scripts/synth_qa.py --model gpt-6.1-sol --limit 20  # 다른 teacher와 비교
uv run python scripts/synth_qa.py                                 # 전체

API 키는 .env의 OPENAI_API_KEY를 읽는다.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv

from carmind.budget import DailyTokenBudget
from carmind.data.synth_qa import (
    PROMPT_VERSION,
    OpenAITeacher,
    generate_all,
    load_replacements,
)

parser = argparse.ArgumentParser()
parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks"))
parser.add_argument("--out", type=Path, default=Path("data/interim/qa"))
parser.add_argument("--model", default="gpt-6-luna")
parser.add_argument("--effort", default="low", choices=["low", "medium", "high"])
parser.add_argument("--limit", type=int, default=None, help="무작위로 뽑을 청크 수 (파일럿용)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--anonymize", type=Path, default=Path("data/anonymize.yaml"))
parser.add_argument("--max-daily-tokens", type=int, default=None, help="오늘(UTC) 사용 토큰 상한 (무료 한도 보호)")
parser.add_argument("--ledger", type=Path, default=Path("data/interim/usage_ledger.jsonl"))
parser.add_argument("--splits", type=Path, default=Path("data/splits.yaml"), help="holdout 문서 목록 (생성 제외)")
args = parser.parse_args()

holdout = set()
if args.splits.exists():
    holdout = set((yaml.safe_load(args.splits.read_text(encoding="utf-8")) or {}).get("holdout", []))
budget = DailyTokenBudget(args.ledger, args.model, args.max_daily_tokens)
if args.max_daily_tokens:
    print(f"오늘 사용량 {budget.used:,} / 상한 {args.max_daily_tokens:,} 토큰 ({budget.pool})")

load_dotenv()
teacher = OpenAITeacher(args.model, args.effort, load_replacements(args.anonymize))

for path in sorted(args.chunks.glob("*.jsonl")):
    if path.stem in holdout:
        print(f"[{path.stem}] holdout 문서 → 건너뜀")
        continue
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    total = len(chunks)
    if args.limit:
        chunks = random.Random(args.seed).sample(chunks, min(args.limit, total))

    results = generate_all(chunks, teacher, args.out / args.model / PROMPT_VERSION / path.name, args.workers, budget)
    ok = [r for r in results if r.error is None]
    pairs = [qa for r in ok for qa in r.qa_pairs]
    if not ok:
        print("성공한 요청이 없습니다:", results[0].error if results else "no results")
        continue

    cost = sum(r.cost_usd for r in ok)
    print(f"\n[{path.stem}] {len(ok)} ok / {len(results) - len(ok)} errors")
    print(f"  QA {len(pairs)}개 (청크당 {len(pairs) / len(ok):.2f}), 빈 청크 {sum(not r.qa_pairs for r in ok)}개")
    print(f"  유형: {dict(Counter(qa.type for qa in pairs))}")
    print(f"  근거 검증 통과: {sum(qa.grounded for qa in pairs)}/{len(pairs)}")
    print(f"  토큰: 입력 {sum(r.input_tokens for r in ok):,} / 출력 {sum(r.output_tokens for r in ok):,}")
    print(f"  비용 ${cost:.4f} → 이 문서 전체({total}청크) 예상 ${cost / len(ok) * total:.2f}")
