"""검색 평가 지표. 정답(gold)은 문항을 만든 원문 청크 1개다.

- Recall@k: 상위 k개 안에 정답 청크가 있으면 1, 없으면 0 → 문항 평균.
  정답이 1개뿐이라 Hit@k와 같다. RAG에서는 "top-k를 넣었을 때 정답 근거가 프롬프트에 들어가는 비율"이다.
- MRR: 정답 순위의 역수(1위=1, 2위=0.5, ...) 평균. 목록(상위 N개) 밖이면 0.
  Recall@k가 "들어갔나"라면 MRR은 "얼마나 위에 있나"를 본다.

주의: 같은 내용이 여러 청크에 중복되거나(공통 경고문 등) 이웃 청크에도 답이 있으면,
정답 청크가 아닌 청크를 찾아도 실제로는 답할 수 있다. 그래서 이 지표는 실제 검색 품질을 낮게(보수적으로) 잰다.
"""


def gold_rank(ranking: list[str], gold: str) -> int | None:
    """정답의 순위(1부터). 목록에 없으면 None."""
    try:
        return ranking.index(gold) + 1
    except ValueError:
        return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / max(1, len(ranks))


def mrr(ranks: list[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r is not None) / max(1, len(ranks))


def answer_coverage(reference: str, chunk_text: str) -> float:
    """정답 답변의 문자 bigram 중 청크에 들어 있는 비율 (공백 무시). 검색 실패 문항의 '사실상 정답' 여부를 어림한다.

    답변은 청크를 바꿔 말한 문장이라 1.0이 나오기는 어렵다. 높을수록 그 청크로도 답할 수 있을 가능성이 크다.
    """
    ref = "".join(reference.split())
    target = "".join(chunk_text.split())
    grams = {ref[i:i + 2] for i in range(len(ref) - 1)}
    return sum(1 for g in grams if g in target) / max(1, len(grams))
