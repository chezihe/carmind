"""여러 평가 실행의 report.json을 나란히 비교한다.

uv run python scripts/eval_report.py --compare outputs/eval/base_closed outputs/eval/lora_closed ...
결과는 표준 출력 + (--save를 주면) 그 파일에 마크다운으로 저장.
report.json은 eval_grade.py가 만든다.
"""

import argparse
import json
from pathlib import Path

from carmind.evaluation.report import compare_table, same_questions

parser = argparse.ArgumentParser()
parser.add_argument("--compare", type=Path, nargs="+", required=True, help="비교할 실행 폴더 2개 이상")
parser.add_argument("--save", type=Path, default=None)
args = parser.parse_args()
if len(args.compare) < 2:
    parser.error("--compare에는 실행 폴더가 2개 이상 필요합니다")

reports = {d.name: json.loads((d / "report.json").read_text(encoding="utf-8")) for d in args.compare}
lines = []
if not same_questions(reports):
    lines.append("> 주의: 실행마다 평가 문항이 다릅니다. 점수 차이에 문항 차이가 섞여 있습니다.\n")
judges = {n: (r.get("grading") or {}).get("judge") for n, r in reports.items()}
if len(set(judges.values())) > 1:
    lines.append(f"> 주의: judge가 다릅니다: {judges}\n")
for name, r in reports.items():
    g = r["meta"]["generation"]
    lines.append(f"- **{name}**: {g['model']} / adapter={g['adapter']} / mode={g['mode']}")
lines += ["", compare_table(reports)]
text = "\n".join(lines)
print(text)
if args.save:
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(text, encoding="utf-8")
