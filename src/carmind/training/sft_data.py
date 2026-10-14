"""최종 QA → SFT 학습용 대화 데이터 (train / val).

두 가지 형식을 만든다. 같은 QA로 두 방식을 학습시켜 비교하기 위함이다.
    closed: [system, user(질문), assistant(답)]
            모델이 매뉴얼 내용을 외워서 답해야 한다
    rag:    [system, user(참고 문서 + 질문), assistant(답)]
            모델이 주어진 문서를 읽고 답하는 법을 배운다 (나중에 검색 결과를 넣어 주는 구조와 같음)

train/val은 청크 단위로 나눈다. 같은 원문에서 나온 비슷한 질문이 양쪽에 섞이면
val이 train을 미리 본 셈이 되어 점수가 부풀려진다(데이터 누수).
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SYSTEM_CLOSED = (
    "당신은 자동차 취급설명서 내용을 잘 아는 정비 도우미입니다. "
    "차량 사용자의 질문에 정확하고 간결하게 답하세요. "
    "관련된 경고·주의 사항이 있으면 함께 알려 주세요."
)
SYSTEM_RAG = (
    "당신은 자동차 취급설명서 내용을 잘 아는 정비 도우미입니다. "
    "[참고 문서]의 내용만 근거로 질문에 정확하고 간결하게 답하세요. "
    "관련된 경고·주의 사항이 있으면 함께 알려 주세요."
)


def to_messages(qa: dict, mode: str, chunk_text: str | None = None) -> list[dict]:
    if mode == "closed":
        return [
            {"role": "system", "content": SYSTEM_CLOSED},
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
    if mode == "rag":
        if chunk_text is None:
            raise ValueError(f"rag 형식에는 청크 원문이 필요합니다: {qa['qa_id']}")
        return [
            {"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"[참고 문서]\n{chunk_text}\n\n[질문]\n{qa['question']}"},
            {"role": "assistant", "content": qa["answer"]},
        ]
    raise ValueError(f"알 수 없는 형식: {mode}")


def split_by_chunk(qas: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """청크 단위로 train/val을 나눈다. 같은 청크의 QA는 항상 같은 쪽에 들어간다."""
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for qa in qas:
        by_chunk[qa["chunk_id"]].append(qa)
    chunk_ids = sorted(by_chunk)
    random.Random(seed).shuffle(chunk_ids)
    n_val = max(1, round(len(chunk_ids) * val_ratio))
    val_ids = set(chunk_ids[:n_val])
    train = [qa for cid in chunk_ids if cid not in val_ids for qa in by_chunk[cid]]
    val = [qa for cid in chunk_ids if cid in val_ids for qa in by_chunk[cid]]
    return train, val


def build(
    qa_dir: Path, chunk_dir: Path, out_dir: Path, mode: str, val_ratio: float = 0.05, seed: int = 42
) -> dict[str, int]:
    qas = [json.loads(line) for p in sorted(qa_dir.glob("*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]
    chunks = {}
    if mode == "rag":
        for p in sorted(chunk_dir.glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                c = json.loads(line)
                chunks[c["chunk_id"]] = c["text"]

    train, val = split_by_chunk(qas, val_ratio, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        records = [
            {"qa_id": qa["qa_id"], "type": qa["type"], "messages": to_messages(qa, mode, chunks.get(qa["chunk_id"]))}
            for qa in rows
        ]
        (out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
    return {"train": len(train), "val": len(val), "docs": len({qa["doc_id"] for qa in qas})}
