import pytest

from carmind.training.sft_data import split_by_chunk, to_messages

QA = {"qa_id": "d:1:0", "chunk_id": "d:1", "question": "타이어 공기압은?", "answer": "260 kPa입니다."}


def test_closed_and_rag_messages():
    closed = to_messages(QA, "closed")
    assert [m["role"] for m in closed] == ["system", "user", "assistant"]
    assert closed[1]["content"] == "타이어 공기압은?"
    rag = to_messages(QA, "rag", "[타이어]\n공기압 260 kPa")
    assert rag[1]["content"].startswith("[참고 문서]\n[타이어]") and rag[1]["content"].endswith("타이어 공기압은?")
    with pytest.raises(ValueError):
        to_messages(QA, "rag")


def test_split_keeps_chunk_together():
    qas = [{"qa_id": f"d:{c}:{i}", "chunk_id": f"d:{c}"} for c in range(20) for i in range(3)]
    train, val = split_by_chunk(qas, val_ratio=0.2, seed=0)
    assert len(train) + len(val) == 60
    assert not {q["chunk_id"] for q in train} & {q["chunk_id"] for q in val}
    assert split_by_chunk(qas, 0.2, 0) == (train, val)  # 같은 seed면 같은 결과
