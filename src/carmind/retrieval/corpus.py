"""검색 코퍼스(익명화된 청크)와 검색기 조립.

인덱싱하는 텍스트는 모델이 [참고 문서]로 보게 될 텍스트와 같아야 한다. 그래서 학습·평가와 같은
anonymize() 규칙을 적용한 청크를 색인한다 (원문의 브랜드명으로는 검색되지만 프롬프트에는 다른 글자가 들어가는
어긋남을 막는다. 질문도 이미 익명화돼 있다).

검색 범위(scope)
- per-doc (기본): 사용자의 차종을 안다고 보고 그 차의 매뉴얼 청크 안에서만 찾는다. 실제 서비스에 가까운 설정.
- all-docs:      모든 매뉴얼을 한꺼번에 찾는다. 차종마다 거의 같은 문단이 있어 다른 차의 청크를 고를 수 있어 더 어렵다.
범위마다 인덱스를 따로 만든다 (BM25의 IDF·평균 길이도 그 범위의 문서로 계산해야 맞다).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from carmind.data.synth_qa import anonymize
from carmind.evaluation.data import read_jsonl
from carmind.retrieval.bm25 import BM25
from carmind.retrieval.dense import DenseEncoder
from carmind.retrieval.fusion import rrf
from carmind.retrieval.tokenize import get_tokenizer

METHODS = ("bm25", "dense", "hybrid")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str  # 익명화된 텍스트


def load_corpus(chunk_dir: Path, doc_ids: list[str] | None, replacements: list) -> list[Chunk]:
    """doc_ids가 None이면 chunk_dir의 모든 문서. 순서는 문서 이름 → 파일 안 순서로 고정."""
    paths = sorted(chunk_dir.glob("*.jsonl")) if doc_ids is None else [chunk_dir / f"{d}.jsonl" for d in sorted(doc_ids)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"청크 파일이 없습니다: {missing}")
    return [Chunk(c["chunk_id"], c["doc_id"], anonymize(c["text"], replacements)) for p in paths for c in read_jsonl(p)]


def top_ids(scores: np.ndarray, ids: list[str], depth: int) -> list[str]:
    """점수 내림차순 상위 depth개의 chunk_id. 동점이면 코퍼스 순서(stable)."""
    order = np.argsort(-scores, kind="stable")[:depth]
    return [ids[i] for i in order]


def embed_corpus(encoder: DenseEncoder, chunks: list[Chunk]) -> np.ndarray:
    """문서별로 나눠 인코딩한다. 캐시 키가 문서 단위가 되어 per-doc과 all-docs가 같은 캐시를 나눠 쓴다."""
    parts, start = [], 0
    while start < len(chunks):
        end = start
        while end < len(chunks) and chunks[end].doc_id == chunks[start].doc_id:
            end += 1
        parts.append(encoder.encode([c.text for c in chunks[start:end]], "passage"))
        start = end
    return np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)


class Retriever:
    """method: bm25 | dense | hybrid(bm25 + dense를 RRF). search()는 상위 depth개의 chunk_id를 돌려준다.

    hybrid는 두 검색기의 상위 depth개씩만 합친다 (그 밖의 순위는 RRF 점수가 거의 0이라 결과에 영향이 없다).
    """

    def __init__(self, chunks: list[Chunk], method: str, tokenizer: str = "bigram",
                 encoder: DenseEncoder | None = None, rrf_k: int = 60, depth: int = 100):
        if method not in METHODS:
            raise ValueError(f"method는 {METHODS} 중 하나: {method}")
        if method != "bm25" and encoder is None:
            raise ValueError(f"{method}에는 encoder가 필요합니다")
        self.chunks, self.method, self.encoder, self.rrf_k, self.depth = chunks, method, encoder, rrf_k, depth
        self.ids = [c.chunk_id for c in chunks]
        self.text = {c.chunk_id: c.text for c in chunks}
        self.bm25 = BM25([c.text for c in chunks], get_tokenizer(tokenizer)) if method != "dense" else None
        self.emb = embed_corpus(encoder, chunks) if method != "bm25" else None

    def bm25_rank(self, query: str) -> list[str]:
        return top_ids(self.bm25.scores(query), self.ids, self.depth)

    def dense_rank(self, query_emb: np.ndarray) -> list[str]:
        return top_ids(self.emb @ query_emb, self.ids, self.depth)

    def search_many(self, queries: list[str]) -> list[list[str]]:
        q_emb = self.encoder.encode(queries, "query") if self.method != "bm25" else None
        out = []
        for i, q in enumerate(queries):
            if self.method == "bm25":
                out.append(self.bm25_rank(q))
            elif self.method == "dense":
                out.append(self.dense_rank(q_emb[i]))
            else:
                out.append(rrf([self.bm25_rank(q), self.dense_rank(q_emb[i])], self.rrf_k)[:self.depth])
        return out

    def search(self, query: str) -> list[str]:
        return self.search_many([query])[0]


def build_context(texts: list[str], max_tokens: int, count_tokens) -> tuple[str, bool]:
    """검색된 청크(순위 순)를 빈 줄로 이어 [참고 문서] 본문을 만든다. (본문, 잘렸는지)를 돌려준다.

    청크는 "[섹션 경로]" 줄로 시작하므로 빈 줄만으로도 경계가 드러난다. 학습 때는 청크 1개만 넣었으므로
    번호·구분자 같은 새 형식을 추가하지 않는다 (학습 분포와의 차이를 최소화).
    합계가 max_tokens를 넘으면 순위가 낮은 청크부터 자른다: 넘치는 청크는 들어갈 만큼만 앞부분을 남기고
    그 뒤 청크는 버린다. 1위 청크는 항상 (필요하면 잘라서라도) 들어간다.
    count_tokens: 문자열 → 토큰 수 (생성 모델의 토크나이저). 테스트에서는 글자 수 등으로 바꿔 끼운다.
    """
    sep = "\n\n"
    parts, used, truncated = [], 0, False
    for text in texts:
        cost = count_tokens(text) + (count_tokens(sep) if parts else 0)
        if used + cost <= max_tokens:
            parts.append(text)
            used += cost
            continue
        truncated = True
        budget = max_tokens - used - (count_tokens(sep) if parts else 0)
        if budget > 0:
            # 토큰 수에 맞게 글자를 이분 탐색으로 자른다 (토크나이저 종류와 무관하게 동작)
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if count_tokens(text[:mid]) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            if lo > 0:
                parts.append(text[:lo])
        break
    return sep.join(parts), truncated
