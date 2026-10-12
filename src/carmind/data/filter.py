"""합성 QA 품질 필터링: 규칙 필터 → 중복 제거.

    1. 규칙 필터: 근거 검증 실패, 길이 이상, 지시어 질문, 문서 언급, 치환되지 않은 브랜드명
    2. 중복 제거: 질문의 글자 3-gram을 MinHash로 요약하고 LSH로 비슷한 후보만 비교
       (모든 쌍을 비교하면 O(n²)이라 데이터가 늘수록 느려진다)

LLM 판정(답변 충실도)은 비용이 들어서 별도 단계(judge.py)로 분리한다.
탈락한 QA도 사유와 함께 저장해서, 필터가 너무 엄격하거나 느슨하지 않은지 사람이 확인할 수 있게 한다.
"""

import re
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

# 질문만 읽어서는 무엇을 묻는지 알 수 없는 지시어 시작
DEICTIC_RE = re.compile(r"^(이|그|저|해당|위|아래)\s?(기능|장치|버튼|경우|것|거|부분|표시|경고등)")
# 답변에서 원문 문서를 "출처로" 언급하는 메타 표현 (정비사처럼 답해야 함).
# '설명서' 단어 자체는 막지 않는다: 취급설명서 보관 방법, 부품에 동봉된 사용설명서 안내처럼 정당한 내용이 있다
DOC_REF_RE = re.compile(r"안내되어 있|명시되어 있|나와 있습니다|설명서에 따르면|설명서에는|문서에")

MIN_QUESTION_CHARS = 8
MAX_QUESTION_CHARS = 150
MIN_ANSWER_CHARS = 5  # 프롬프트 v3에서 짧은 답을 허용했다 ("즉시 충전을 중지하십시오." = 14자)
MAX_ANSWER_CHARS = 800


@dataclass
class QARecord:
    qa_id: str
    chunk_id: str
    doc_id: str
    type: str
    question: str
    answer: str
    evidence: list[str]
    grounded: bool
    teacher: str
    prompt_version: str


def flatten(results: list[dict]) -> list[QARecord]:
    """청크 단위 생성 결과 → QA 단위 레코드. 에러난 청크는 제외."""
    records = []
    for r in sorted(results, key=lambda x: x["chunk_id"]):  # 생성 순서와 무관하게 결정적으로
        if r["error"] is not None:
            continue
        for i, qa in enumerate(r["qa_pairs"]):
            records.append(
                QARecord(
                    qa_id=f"{r['chunk_id']}:{i}",
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    teacher=r["model"],
                    prompt_version=r["prompt_version"],
                    **qa,
                )
            )
    return records


def rule_violation(qa: QARecord, banned_terms: list[str]) -> str | None:
    """첫 번째로 걸린 규칙 이름을 돌려준다. 통과하면 None."""
    q, a = qa.question.strip(), qa.answer.strip()
    if not qa.grounded:
        return "not_grounded"
    if not MIN_QUESTION_CHARS <= len(q) <= MAX_QUESTION_CHARS:
        return "question_length"
    if not MIN_ANSWER_CHARS <= len(a) <= MAX_ANSWER_CHARS:
        return "answer_length"
    if DEICTIC_RE.match(q):
        return "deictic_question"
    if DOC_REF_RE.search(a):
        return "doc_reference"
    if any(term in q or term in a for term in banned_terms):
        return "banned_term"
    return None


def shingles(text: str, n: int = 3) -> set[str]:
    """공백·문장부호를 지우고 글자 n-gram 집합으로 만든다. 한국어는 단어보다 글자 단위가 안정적이다."""
    s = re.sub(r"[\s\W_]+", "", text)
    return {s[i : i + n] for i in range(max(1, len(s) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def dedup_questions(
    records: list[QARecord], threshold: float = 0.95, num_perm: int = 128
) -> tuple[list[QARecord], list[tuple[QARecord, str]]]:
    """질문이 거의 같은 QA를 제거한다. 먼저 나온 것을 남긴다.

    LSH는 후보를 빠르게 찾는 근사 방법이라 오탐이 있다 → 후보는 실제 Jaccard로 한 번 더 확인한다.
    임계값이 0.95로 높은 이유: 글자 유사도는 의미 차이를 모른다. 0.6~0.9 구간에는
    "18인치/20인치", "충전 도어/도어"처럼 글자는 거의 같고 뜻이 다른 질문이 많았다.
    띄어쓰기만 다른 질문("에어 필터"/"에어필터")은 유사도 1.0이라 그대로 걸러진다.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept, dropped = [], []
    grams: dict[str, set[str]] = {}
    for qa in records:
        g = shingles(qa.question)
        m = MinHash(num_perm=num_perm)
        for s in g:
            m.update(s.encode("utf-8"))
        dup_of = next((k for k in lsh.query(m) if jaccard(g, grams[k]) >= threshold), None)
        if dup_of:
            dropped.append((qa, dup_of))
            continue
        lsh.insert(qa.qa_id, m)
        grams[qa.qa_id] = g
        kept.append(qa)
    return kept, dropped
