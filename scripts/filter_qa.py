"""합성 QA → 규칙 필터 + 중복 제거 → 후보(candidates)

uv run python scripts/filter_qa.py --teacher gpt-5.4-mini --prompt-version v2
출력: data/interim/qa_candidates/<doc>.jsonl (통과), data/interim/qa_rejected/<doc>.jsonl (탈락 + 사유)
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from carmind.data.filter import dedup_questions, flatten, rule_violation
from carmind.data.synth_qa import load_replacements

parser = argparse.ArgumentParser()
parser.add_argument("--qa", type=Path, default=Path("data/interim/qa"))
parser.add_argument("--teacher", default="gpt-5.4-mini")
parser.add_argument("--prompt-version", default="v3")
parser.add_argument("--anonymize", type=Path, default=Path("data/anonymize.yaml"))
parser.add_argument("--dedup-threshold", type=float, default=0.95)
parser.add_argument("--out", type=Path, default=Path("data/interim"))
parser.add_argument("--doc", default=None, help="이 문서만 처리 (예: manual_xxx_2025)")
args = parser.parse_args()

banned = [src for src, _ in load_replacements(args.anonymize)]  # 치환됐어야 할 표현이 남아 있으면 탈락


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


for path in sorted((args.qa / args.teacher / args.prompt_version).glob("*.jsonl")):
    if args.doc and path.stem != args.doc:
        continue
    records = flatten([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])
    rejected, passed = [], []
    for qa in records:
        reason = rule_violation(qa, banned)
        if reason:
            rejected.append({**asdict(qa), "reason": reason})
        else:
            passed.append(qa)
    kept, dups = dedup_questions(passed, args.dedup_threshold)
    rejected += [{**asdict(qa), "reason": "duplicate", "duplicate_of": of} for qa, of in dups]

    write(args.out / "qa_candidates" / path.name, [asdict(qa) for qa in kept])
    write(args.out / "qa_rejected" / path.name, rejected)
    print(f"[{path.stem}] {len(records)} QA → 통과 {len(kept)}, 탈락 {len(rejected)}")
    print(f"  탈락 사유: {dict(Counter(r['reason'] for r in rejected))}")
