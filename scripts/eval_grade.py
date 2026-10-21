"""생성된 답변을 LLM judge로 채점하고 report.json / report.md를 만든다.

uv run python scripts/eval_grade.py outputs/eval/<run>                 # 비용 추정 → 채점 → 리포트
uv run python scripts/eval_grade.py outputs/eval/<run> --dry-run       # 비용 추정만 (API 호출 없음)
uv run python scripts/eval_grade.py outputs/eval/<run> --no-judge      # 자동 지표만으로 리포트

API를 부르기 전에 남은 문항의 비용을 추정해 출력하고, --max-cost를 넘으면 멈춘다.
끊겼다가 다시 실행하면 판정이 없는(또는 답변이 바뀐) 문항만 채점한다.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from carmind.data.synth_qa import anonymize, load_replacements
from carmind.evaluation.data import load_chunks, load_eval_set, read_jsonl
from carmind.evaluation.grade import (
    GRADER_VERSION,
    Grader,
    estimate_cost,
    grade_all,
    load_valid_grades,
)
from carmind.evaluation.report import aggregate, to_markdown, write_report

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--judge", default="gpt-6-luna")
parser.add_argument("--effort", default="low", choices=["low", "medium", "high"])
parser.add_argument("--max-cost", type=float, default=1.0, help="추정 비용(USD)이 이보다 크면 채점하지 않음")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks"))
parser.add_argument("--anonymize", type=Path, default=Path("data/anonymize.yaml"))
parser.add_argument("--dry-run", action="store_true", help="비용 추정만 하고 끝냄")
parser.add_argument("--no-judge", action="store_true", help="채점 없이 자동 지표로만 리포트")
args = parser.parse_args()

meta = json.loads((args.run_dir / "meta.json").read_text(encoding="utf-8"))
preds = read_jsonl(args.run_dir / "predictions.jsonl")
missing = set(meta["qa_ids"]) - {p["qa_id"] for p in preds}
if missing:
    print(f"주의: 아직 생성되지 않은 문항 {len(missing)}개 (eval_generate.py를 다시 실행하면 이어서 생성)")

grades_path = args.run_dir / f"grades_{args.judge}.jsonl"
grades: dict[str, dict] = {}
grading = None
if not args.no_judge:
    # judge에게 줄 근거(정답 근거 문장 + 원문 청크)는 평가셋·청크 파일에서 가져와 학습 때와 같은 규칙으로 익명화한다
    replacements = load_replacements(args.anonymize)
    source = {q["qa_id"]: q for q in load_eval_set([Path(p) for p in meta["eval_files"]])}
    chunks = load_chunks(args.chunks, {source[p["qa_id"]]["doc_id"] for p in preds}, replacements)
    items = [
        {
            **p,
            "evidence": [anonymize(e, replacements) for e in source[p["qa_id"]].get("evidence", [])],
            "source": chunks.get(source[p["qa_id"]]["chunk_id"]),
        }
        for p in preds
    ]
    grades = load_valid_grades(grades_path, items)
    todo = [it for it in items if it["qa_id"] not in grades]
    est = estimate_cost(todo, args.judge)
    print(
        f"[{meta['run_name']}] 채점 {len(items)}문항 중 {len(grades)}개 완료, {len(todo)}개 남음 ({args.judge})\n"
        f"  추정: 입력 {est.input_tokens:,} + 출력 {est.output_tokens:,} 토큰 ≈ ${est.cost_usd:.4f} "
        f"(캐시 할인 없는 보수적 추정, 상한 ${args.max_cost})"
    )
    if est.cost_usd > args.max_cost:
        raise SystemExit("추정 비용이 상한을 넘어 중단합니다. --limit을 줄이거나 --max-cost를 올리세요.")
    if args.dry_run:
        raise SystemExit(0)
    if todo:
        load_dotenv()
        from carmind.llm import StructuredLLM

        new = grade_all(todo, Grader(StructuredLLM(args.judge, args.effort), args.judge), grades_path, args.workers)
        errors = [g for g in new if g.error]
        if errors:
            print(f"판정 오류 {len(errors)}개 (예: {errors[0].error}). 다시 실행하면 그 문항만 재시도합니다.")
        grades = load_valid_grades(grades_path, items)

    # 비용은 지금 유효한 판정 + 실패한 시도까지 파일에 남은 전부를 합산한다 (실제로 낸 돈)
    rows = read_jsonl(grades_path)
    grading = {
        "judge": args.judge,
        "grader_version": GRADER_VERSION,
        "effort": args.effort,
        "graded": len(grades),
        "errors": len(items) - len(grades),
        "requests": len(rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "cost_usd": sum(r["cost_usd"] for r in rows),
        "estimated_cost_usd_last_run": est.cost_usd,
    }

report = {"meta": meta, "grading": grading, **aggregate(preds, grades)}
write_report(args.run_dir, report)
print(to_markdown(report))
print(f"저장: {args.run_dir}/report.json, report.md")
