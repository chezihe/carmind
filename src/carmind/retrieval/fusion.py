"""Reciprocal Rank Fusion (RRF, Cormack et al. 2009).

BM25 점수(0~수십)와 코사인 유사도(-1~1)는 척도가 달라 그냥 더하면 한쪽이 지배한다.
점수 정규화(min-max 등)는 질의마다 분포가 달라 불안정하다. RRF는 점수를 버리고 순위만 쓴다.

    RRF(d) = Σ_검색기  1 / (k + rank_검색기(d))        (rank는 1부터, 목록에 없으면 0을 더함)

k=60은 원 논문의 값으로, 1위와 2위의 차이를 완만하게 만들어 한 검색기의 1위가 혼자 결과를 정하지 않게 한다.
두 검색기에서 모두 상위에 있는 문서가 한쪽에서만 1위인 문서보다 앞서게 되는 것이 핵심이다.
"""

from collections import defaultdict


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """여러 순위 목록(앞이 1위)을 합쳐 하나의 순위로. 동점이면 먼저 나온 목록·순위 순으로 안정 정렬."""
    score: dict[str, float] = defaultdict(float)
    order: dict[str, int] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking, start=1):
            score[doc] += 1.0 / (k + r)
            order.setdefault(doc, len(order))
    return sorted(score, key=lambda d: (-score[d], order[d]))
