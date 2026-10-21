"""API 없이 계산하는 자동 지표 (무료, 즉시 계산).

LLM judge보다 거칠지만 비용이 0이라 모든 실행에 항상 붙인다. judge 점수와 같이 보면
"judge가 후하게 준 건지, 답이 실제로 정답과 겹치는지"를 교차 확인할 수 있다.

- exact match: 정규화 후 정답과 완전히 같은 비율. 서술형 답에서는 거의 0이라 참고용
- 문자 bigram F1: 한국어는 조사·어미가 붙어 단어(띄어쓰기) 단위로 비교하면 "공기압은" ≠ "공기압이"가 된다.
  글자 두 개씩 묶어 비교하면 어간이 같을 때 부분 점수를 받는다 (SQuAD식 token F1의 한국어 대체)
- 반복률: 작은 모델이 greedy 디코딩에서 같은 구절을 무한 반복하는 현상(degeneration)을 잡는다
"""

import re
import unicodedata
from collections import Counter

_PUNCT_OR_SPACE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """비교용 정규화: 유니코드 정규화(NFKC) + 소문자 + 공백·문장부호 제거.

    "260 kPa입니다." 와 "260kPa입니다"처럼 표기만 다른 답을 같은 것으로 본다.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return _PUNCT_OR_SPACE.sub("", text)


def exact_match(prediction: str, reference: str) -> bool:
    return normalize(prediction) == normalize(reference)


def char_ngrams(text: str, n: int = 2) -> Counter:
    s = normalize(text)
    if len(s) < n:
        # 한 글자짜리 답("네")도 비교할 수 있게 글자 자체를 n-gram으로 쓴다
        return Counter([s]) if s else Counter()
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def char_ngram_f1(prediction: str, reference: str, n: int = 2) -> float:
    """문자 n-gram 다중집합(multiset)의 겹침으로 precision/recall/F1을 계산한다.

    precision = 예측 중 정답과 겹치는 비율 (군더더기가 많으면 낮아짐)
    recall    = 정답 중 예측이 다룬 비율 (핵심을 빠뜨리면 낮아짐)
    """
    pred, ref = char_ngrams(prediction, n), char_ngrams(reference, n)
    if not pred or not ref:
        return float(pred == ref)  # 둘 다 비었으면 1, 한쪽만 비었으면 0
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def has_repetition_loop(text: str, n: int = 4, min_count: int = 3, char_n: int = 12) -> bool:
    """같은 구절이 반복되는 '루프'가 있으면 True.

    두 가지로 본다.
    1) 띄어쓰기 단위 n-gram(기본 4어절)이 min_count번 이상 나옴
    2) 띄어쓰기가 없는 반복("...하십시오하십시오하십시오")에 대비해 공백을 뺀 문자열에서
       char_n 글자 조각이 min_count번 이상 나옴
    정상 답변에서도 짧은 상투구("하십시오.")는 반복되므로 n을 넉넉히 잡아 오탐을 줄인다.
    """
    words = text.split()
    word_grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    if word_grams and max(word_grams.values()) >= min_count:
        return True
    s = re.sub(r"\s+", "", text)
    char_grams = Counter(s[i:i + char_n] for i in range(len(s) - char_n + 1))
    return bool(char_grams) and max(char_grams.values()) >= min_count


def free_metrics(prediction: str, reference: str) -> dict:
    """문항 1개의 자동 지표. 집계(report.py)에서 평균을 낸다."""
    return {
        "exact_match": exact_match(prediction, reference),
        "char_f1": char_ngram_f1(prediction, reference),
        "chars": len(prediction),
        "repetition": has_repetition_loop(prediction),
    }
