"""BM25 (Okapi BM25)를 직접 구현한다. 수식이 짧고, 각 항이 무엇을 하는지 보이게 하려는 목적이다.

질의 q의 토큰 t마다 점수를 더한다.

    score(q, d) = Σ_t  IDF(t) · tf(t,d) · (k1 + 1) / ( tf(t,d) + k1 · (1 - b + b · |d| / avgdl) )

    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

- tf(t,d): 문서 d 안에서 t의 등장 횟수. 많이 나올수록 점수가 오르지만 k1 때문에 포화(saturation)된다.
  ("경고등"이 10번 나온 문서가 1번 나온 문서보다 10배 관련 있지는 않다)
- k1 (기본 1.5): 포화 속도. 0이면 tf를 무시(있다/없다만), 클수록 tf에 선형에 가까워진다.
- |d| / avgdl: 문서 길이를 평균 길이로 나눈 값. 긴 문서는 우연히 단어를 포함할 확률이 높으므로 tf를 깎는다.
- b (기본 0.75): 길이 보정 강도. 0이면 보정 없음, 1이면 완전 보정.
- IDF: 드문 토큰일수록 크다. "차량"처럼 거의 모든 청크에 있는 토큰은 0에 가까워 변별력이 없다.
  +1을 넣은 형태(Lucene 방식)라 df가 N/2를 넘어도 IDF가 음수가 되지 않는다.

질의 토큰이 중복되면(bigram은 흔함) 그 토큰을 중복 횟수만큼 더한다. rank_bm25 등 흔한 구현과 같은 동작이다.

속도: 청크 수천 개라 역색인(token → [(문서, tf)])만 만들어 두면 질의 하나에 ms 단위로 끝난다.
"""

import math
from collections import Counter, defaultdict

import numpy as np

from carmind.retrieval.tokenize import Tokenizer


class BM25:
    def __init__(self, docs: list[str], tokenizer: Tokenizer, k1: float = 1.5, b: float = 0.75):
        self.tokenizer, self.k1, self.b = tokenizer, k1, b
        tokenized = [tokenizer(d) for d in docs]
        self.n_docs = len(docs)
        self.doc_len = np.array([len(t) for t in tokenized], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        # 역색인: 토큰마다 (문서 번호 배열, tf 배열)
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        for i, toks in enumerate(tokenized):
            for tok, tf in Counter(toks).items():
                postings[tok][i] = tf
        self.postings = {
            tok: (np.fromiter(p.keys(), dtype=np.int64), np.fromiter(p.values(), dtype=np.float64))
            for tok, p in postings.items()
        }
        # 문서 길이 보정 항은 질의와 무관하므로 미리 계산해 둔다
        self._norm = k1 * (1 - b + b * self.doc_len / max(self.avgdl, 1e-9))

    def idf(self, token: str) -> float:
        df = len(self.postings[token][0]) if token in self.postings else 0
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def scores(self, query: str) -> np.ndarray:
        """모든 문서의 BM25 점수 (길이 n_docs). 질의 토큰이 하나도 없는 문서는 0."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        for tok, qtf in Counter(self.tokenizer(query)).items():
            if tok not in self.postings:
                continue
            idx, tf = self.postings[tok]
            out[idx] += qtf * self.idf(tok) * tf * (self.k1 + 1) / (tf + self._norm[idx])
        return out
