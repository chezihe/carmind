"""학습/평가 분할(hold-out) 규칙.

hold-out 문서는 평가 전용이라 학습 데이터에 섞이면 안 된다.
build_sft는 data/processed/qa/*.jsonl을 전부 읽기 때문에, hold-out 문서의 최종 QA는
처음부터 다른 폴더(data/processed/eval/)에 쓰게 해서 실수로 섞일 여지를 없앤다.
"""

from pathlib import Path

import yaml


def load_holdout(splits: Path) -> set[str]:
    """splits.yaml의 holdout 문서 id 목록. 파일이 없으면 빈 집합 (전부 학습용)"""
    if not splits.exists():
        return set()
    return set((yaml.safe_load(splits.read_text(encoding="utf-8")) or {}).get("holdout", []))


def final_qa_dir(processed: Path, doc_id: str, holdout: set[str]) -> Path:
    """최종 QA를 쓸 폴더: hold-out이면 평가셋(eval), 아니면 학습용(qa)"""
    return processed / ("eval" if doc_id in holdout else "qa")
