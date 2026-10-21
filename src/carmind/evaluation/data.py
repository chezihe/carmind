"""평가셋 로딩·부분집합 추출·이어하기(resume)용 공통 유틸.

평가셋은 학습에 쓰지 않은 holdout 문서의 QA(data/processed/eval/*.jsonl)다.
문서 이름은 코드에 적지 않고 CLI 인자(--eval)로만 받는다.
"""

import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

from carmind.data.synth_qa import anonymize


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    """한 줄씩 바로 써서, 중간에 끊겨도 그때까지의 결과가 남게 한다 (이어하기의 전제)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_eval_set(paths: list[Path]) -> list[dict]:
    rows = [r for p in paths for r in read_jsonl(p)]
    ids = [r["qa_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("평가셋에 중복된 qa_id가 있습니다.")
    return rows


def stratified_sample(rows: list[dict], n: int | None, seed: int, key: str = "type") -> list[dict]:
    """type별 비율을 유지하며 n개를 뽑는다 (seed가 같으면 항상 같은 문항).

    그냥 무작위로 60개를 뽑으면 troubleshoot(전체의 ~12%)처럼 적은 유형이 몇 개 안 들어가거나
    실행마다 비율이 달라져 유형별 점수를 비교하기 어렵다. 그래서 유형별 할당량을 먼저 정한다.
    할당량은 비율 × n의 정수부를 주고, 남는 자리는 소수부가 큰 유형부터 하나씩 준다(최대 잉여 방식).

    여러 모델을 같은 문항으로 비교해야 하므로 결과는 qa_id 순으로 정렬해 순서까지 고정한다.
    """
    if n is None or n >= len(rows):
        return sorted(rows, key=lambda r: r["qa_id"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    names = sorted(groups)
    exact = {k: len(groups[k]) * n / len(rows) for k in names}
    quota = {k: int(exact[k]) for k in names}
    leftover = n - sum(quota.values())
    for k in sorted(names, key=lambda k: (-(exact[k] - quota[k]), k))[:leftover]:
        quota[k] += 1

    rng = random.Random(seed)
    picked = []
    for k in names:  # 이름 순으로 돌아야 seed가 같을 때 rng 소비 순서도 같다
        members = sorted(groups[k], key=lambda r: r["qa_id"])
        picked.extend(rng.sample(members, quota[k]))
    return sorted(picked, key=lambda r: r["qa_id"])


def load_chunks(chunk_dir: Path, doc_ids: set[str], replacements: list) -> dict[str, str]:
    """평가 문항의 원문 청크를 읽어 학습 데이터와 똑같이 익명화한다.

    rag 학습 데이터(sft_data.build)는 청크 원문에 anonymize()를 적용했다.
    평가 때 다른 문자열을 넣으면 학습·평가 조건이 달라지므로 같은 함수·같은 규칙을 쓴다.
    judge에게 보여 주는 원문도 같은 이유로 익명화한다 (정답 답변이 이미 익명화돼 있음).
    """
    chunks = {}
    for doc_id in sorted(doc_ids):
        for c in read_jsonl(chunk_dir / f"{doc_id}.jsonl"):
            chunks[c["chunk_id"]] = anonymize(c["text"], replacements)
    return chunks


def done_ids(path: Path, key: str = "qa_id") -> set[str]:
    """이미 결과가 있는 qa_id. 오류로 끝난 행(error가 있는 행)은 다시 하도록 빼 둔다."""
    return {r[key] for r in read_jsonl(path) if not r.get("error")}


def git_commit(cwd: Path | None = None) -> str | None:
    """실행 당시 코드 버전. 커밋 안 된 수정이 있으면 '-dirty'를 붙여 재현 불가능함을 표시한다.

    미추적 새 파일도 dirty로 친다 (porcelain 기본값이 .gitignore는 알아서 거른다).
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=False
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty else "")
