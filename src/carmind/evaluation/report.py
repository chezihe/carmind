"""생성 결과 + 판정 결과 → 전체·유형별 집계(report.json, report.md)와 여러 실행 비교표.

점수(score) = correct 1점, partial 0.5점, incorrect 0점의 평균.
correct 비율만 보면 "반쯤 맞힌" 개선이 안 보이고, 세 비율을 다 보면 모델 간 순위를 매기기 어려워서
한 숫자로 요약한 값을 함께 둔다.

비율의 분모
- correctness·환각: 판정이 성공한 문항 수 (판정 오류 문항은 빼고 따로 개수를 적는다)
- safety_omission: safety 유형 문항 중 판정 성공한 수 (다른 유형에는 의미가 없음)
- 자동 지표: 생성된 모든 문항
"""

import json
from collections import defaultdict
from pathlib import Path

from carmind.evaluation.metrics import free_metrics

SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summarize(preds: list[dict], grades: dict[str, dict]) -> dict:
    """한 묶음(전체 또는 한 유형)의 지표. grades: qa_id → 판정 (없거나 오류인 문항은 빠져 있음)."""
    fm = [free_metrics(p["prediction"], p["reference"]) for p in preds]
    out = {
        "n": len(preds),
        "exact_match": _mean([float(m["exact_match"]) for m in fm]),
        "char_f1": _mean([m["char_f1"] for m in fm]),
        "mean_chars": _mean([m["chars"] for m in fm]),
        "mean_tokens": _mean([p["n_tokens"] for p in preds]),
        "truncation_rate": _mean([float(p["truncated"]) for p in preds]),
        "repetition_rate": _mean([float(m["repetition"]) for m in fm]),
    }
    graded = [grades[p["qa_id"]] for p in preds if p["qa_id"] in grades]
    out["n_graded"] = len(graded)
    if graded:
        for label in SCORE:
            out[label] = _mean([float(g["correctness"] == label) for g in graded])
        out["score"] = _mean([SCORE[g["correctness"]] for g in graded])
        out["hallucination"] = _mean([float(g["hallucination"]) for g in graded])
        safety = [g for g in graded if g["type"] == "safety"]
        out["n_safety_graded"] = len(safety)
        out["safety_omission"] = _mean([float(g["safety_omission"]) for g in safety])
    return out


def aggregate(preds: list[dict], grades: dict[str, dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        by_type[p["type"]].append(p)
    return {
        "overall": summarize(preds, grades),
        "by_type": {t: summarize(rows, grades) for t, rows in sorted(by_type.items())},
    }


# (키, 표에 쓸 이름, 형식) — report.md와 비교표가 같은 순서·형식을 쓰도록 한 곳에 둔다
COLUMNS = [
    ("n", "n", "int"),
    ("n_graded", "graded", "int"),
    ("score", "score", "f3"),
    ("correct", "correct", "pct"),
    ("partial", "partial", "pct"),
    ("incorrect", "incorrect", "pct"),
    ("hallucination", "hallucination", "pct"),
    ("safety_omission", "safety omission", "pct"),
    ("exact_match", "exact match", "pct"),
    ("char_f1", "char-bigram F1", "f3"),
    ("mean_chars", "mean chars", "f0"),
    ("mean_tokens", "mean tokens", "f0"),
    ("truncation_rate", "truncated", "pct"),
    ("repetition_rate", "repetition", "pct"),
]


def fmt(value, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "int":
        return str(value)
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "f0":
        return f"{value:.0f}"
    return f"{value:.3f}"


def to_markdown(report: dict) -> str:
    meta = report.get("meta", {})
    gen = meta.get("generation", {})
    lines = [
        f"# 평가 결과: {meta.get('run_name', '')}\n",
        f"- model: `{gen.get('model')}` / adapter: `{gen.get('adapter')}` / mode: `{gen.get('mode')}`",
        (
            f"- decoding: greedy, max_new_tokens={gen.get('max_new_tokens')}, "
            f"repetition_penalty={gen.get('repetition_penalty')}"
        ),
        f"- git: `{meta.get('git_commit')}`",
    ]
    if g := report.get("grading"):
        lines.append(
            f"- judge: `{g['judge']}` ({g['grader_version']}), 비용 ${g['cost_usd']:.4f}, "
            f"토큰 입력 {g['input_tokens']:,} / 출력 {g['output_tokens']:,}, 판정 오류 {g['errors']}개"
        )
    groups = {"overall": report["overall"], **report["by_type"]}
    lines += ["", "| metric | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for key, name, kind in COLUMNS:
        lines.append(f"| {name} | " + " | ".join(fmt(s.get(key), kind) for s in groups.values()) + " |")
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path, report: dict) -> None:
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(to_markdown(report), encoding="utf-8")


def compare_table(reports: dict[str, dict]) -> str:
    """여러 실행의 report를 나란히 놓는다. 행 = 지표, 열 = 실행.

    아래쪽에 유형별 score도 붙인다 (전체 점수가 같아도 유형별로 강약이 다를 수 있음).
    """
    names = list(reports)
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, name, kind in COLUMNS:
        lines.append(f"| {name} | " + " | ".join(fmt(reports[n]["overall"].get(key), kind) for n in names) + " |")
    types = sorted({t for r in reports.values() for t in r["by_type"]})
    for t in types:
        for key, kind in (("score", "f3"), ("char_f1", "f3")):
            cells = [fmt(reports[n]["by_type"].get(t, {}).get(key), kind) for n in names]
            lines.append(f"| {t} {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def same_questions(reports: dict[str, dict]) -> bool:
    """비교하는 실행들이 같은 문항으로 평가됐는지 (다르면 점수 차이가 문항 차이일 수 있다)."""
    sets = {n: tuple(r.get("meta", {}).get("qa_ids", [])) for n, r in reports.items()}
    return len(set(sets.values())) <= 1
