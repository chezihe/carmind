"""BM25용 한국어 토큰화.

한국어는 조사·어미가 단어 뒤에 붙는 교착어라 공백으로만 자르면 "엔진오일을", "엔진오일은", "엔진오일"이
모두 다른 토큰이 되어 서로 매칭되지 않는다. 대처 방법은 크게 두 가지다.

1) 문자 n-gram: 단어 안을 2글자씩 겹쳐 자른다 ("엔진오일을" → 엔진, 진오, 오일, 일을).
   조사가 붙어도 앞부분 bigram은 그대로 겹치므로 형태소 분석기 없이도 부분 일치가 된다.
   사전·외부 의존성이 없고 신조어·부품명에도 강하지만, 토큰 수가 많고 우연한 일치(노이즈)도 늘어난다.
2) 형태소 분석: 사전 기반으로 "엔진/오일/을"처럼 나누고 조사·어미를 버린다 (kiwipiepy).
   토큰이 의미 단위라 정밀하지만, 사전에 없는 말은 잘못 쪼갤 수 있다.

어느 쪽이 나은지는 데이터마다 달라서 셋 다 만들어 두고 평가(scripts/eval_retrieval.py)로 고른다.
"""

import re
from collections.abc import Callable

# 한글·영문·숫자 덩어리만 남긴다. 구두점·기호는 검색 단서가 거의 없고 bigram 노이즈만 만든다
_WORD = re.compile(r"[0-9a-z]+|[가-힣]+")

Tokenizer = Callable[[str], list[str]]


def words(text: str) -> list[str]:
    """소문자화 후 한글 덩어리 / 영숫자 덩어리로 나눈다 ("ABS경고등" → ["abs", "경고등"])."""
    return _WORD.findall(text.lower())


def char_bigrams(text: str) -> list[str]:
    """단어마다 문자 bigram. 단어 경계를 넘는 bigram은 만들지 않는다 (띄어쓰기 오류보다 노이즈가 더 커서).

    한 글자 단어("휠", "키")는 bigram이 없으므로 그 글자 자체를 토큰으로 남긴다.
    """
    out = []
    for w in words(text):
        if len(w) == 1:
            out.append(w)
        else:
            out.extend(w[i:i + 2] for i in range(len(w) - 1))
    return out


def word_bigrams(text: str) -> list[str]:
    """공백 단어 + 문자 bigram. 단어 전체가 그대로 일치하면(예: 부품명) 점수를 더 받게 한다."""
    return [*words(text), *char_bigrams(text)]


# 내용어 품사만 남긴다: 명사(NN*), 동사·형용사 어간(VV, VA), 부사(MAG), 어근(XR), 외국어(SL), 숫자(SN), 한자(SH)
_KIWI_KEEP = ("NN", "NR", "NP", "VV", "VA", "MAG", "XR", "SL", "SN", "SH")


class KiwiTokenizer:
    """kiwipiepy 형태소 분석 → 내용어만. 사전 로딩이 무거우므로(수백 ms) 객체를 한 번 만들어 재사용한다."""

    def __init__(self):
        from kiwipiepy import Kiwi  # 선택 의존성: 이 토크나이저를 쓸 때만 불러온다

        self.kiwi = Kiwi()

    def __call__(self, text: str) -> list[str]:
        return [t.form.lower() for t in self.kiwi.tokenize(text) if t.tag.startswith(_KIWI_KEEP)]


def get_tokenizer(name: str) -> Tokenizer:
    if name == "bigram":
        return char_bigrams
    if name == "word_bigram":
        return word_bigrams
    if name == "kiwi":
        return KiwiTokenizer()
    raise ValueError(f"알 수 없는 토크나이저: {name}")
