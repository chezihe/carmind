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
parser.add_argument("--report-name", default="report", help="읽을 리포트 파일 이름 (예: report_safety)")
args = parser.parse_args()
if len(args.compare) < 2:
    parser.error("--compare에는 실행 폴더가 2개 이상 필요합니다")

reports = {d.name: json.loads((d / f"{args.report_name}.json").read_text(encoding="utf-8")) for d in args.compare}
lines = []
if not same_questions(reports):
    lines.append("> 주의: 실행마다 평가 문항이 다릅니다. 점수 차이에 문항 차이가 섞여 있습니다.\n")
judges = {n: (r.get("grading") or {}).get("judge") for n, r in reports.items()}
if len(set(judges.values())) > 1:
    lines.append(f"> 주의: judge가 다릅니다: {judges}\n")
# 채점 기준이 다른 판정끼리 비교하지 않도록 버전·지시문 지문도 확인한다 (v1 리포트에는 rubric_hash가 없음)
rubrics = {n: ((r.get("grading") or {}).get("grader_version"), (r.get("grading") or {}).get("rubric_hash"))
           for n, r in reports.items()}
if len(set(rubrics.values())) > 1:
    lines.append(f"> 주의: 채점 기준(버전, 지문)이 다릅니다: {rubrics}\n")
for name, r in reports.items():
    g = r["meta"]["generation"]
    lines.append(f"- **{name}**: {g['model']} / adapter={g['adapter']} / mode={g['mode']}")
lines += ["", compare_table(reports)]
text = "\n".join(lines)
print(text)
if args.save:
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(text, encoding="utf-8")
