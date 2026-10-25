"""검색 패키지 단위 테스트 (모델 다운로드 없음)."""

import json
import math

import numpy as np
import pytest

from carmind.evaluation.generate import GenConfig, check_resume, prompt_messages
from carmind.retrieval.bm25 import BM25
from carmind.retrieval.corpus import (
    Chunk,
    Retriever,
    build_context,
    embed_corpus,
    top_ids,
)
from carmind.retrieval.dense import SPECS, cache_key, with_prefix
from carmind.retrieval.fusion import rrf
from carmind.retrieval.metrics import answer_coverage, gold_rank, mrr, recall_at_k
from carmind.retrieval.tokenize import char_bigrams, word_bigrams, words
from carmind.training.sft_data import to_messages


# --- 토크나이저 ---
def test_words_splits_hangul_and_alnum_and_lowercases():
    assert words("ABS경고등이 켜짐!") == ["abs", "경고등이", "켜짐"]


def test_char_bigrams_within_words_only():
    assert char_bigrams("엔진오일 휠") == ["엔진", "진오", "오일", "휠"]


def test_char_bigrams_match_despite_particle():
    # 조사가 붙어도 앞부분 bigram은 같다 → 형태소 분석 없이 부분 일치
    assert set(char_bigrams("엔진오일")) <= set(char_bigrams("엔진오일을"))


def test_word_bigrams_keeps_whole_words():
    assert word_bigrams("타이어 공기압") == ["타이어", "공기압", "타이", "이어", "공기", "기압"]


# --- BM25 ---
def test_bm25_matches_formula_by_hand():
    docs = ["a a b", "b c", "c c c d"]
    bm = BM25(docs, str.split, k1=1.5, b=0.75)
    n, avgdl = 3, (3 + 2 + 4) / 3
    idf_a = math.log((n - 1 + 0.5) / (1 + 0.5) + 1)
    tf, dl = 2, 3
    expected = idf_a * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    s = bm.scores("a")
    assert s[0] == pytest.approx(expected)
    assert s[1] == 0 and s[2] == 0


def test_bm25_rare_term_outweighs_common_term():
    bm = BM25(["x common", "y common", "z common rare"], str.split)
    assert bm.idf("rare") > bm.idf("common") > 0  # +1 형태라 흔한 토큰도 음수가 아님
    assert int(np.argmax(bm.scores("common rare"))) == 2


def test_bm25_length_normalization_prefers_shorter_doc():
    bm = BM25(["k pad pad pad pad pad", "k"], str.split)
    s = bm.scores("k")
    assert s[1] > s[0]


def test_bm25_unknown_query_is_all_zero():
    assert BM25(["a b"], str.split).scores("zzz").tolist() == [0.0]


# --- RRF ---
def test_rrf_scores_and_order():
    fused = rrf([["a", "b", "c"], ["b", "c", "a"]], k=60)
    # b: 1/62 + 1/61, a: 1/61 + 1/63, c: 1/63 + 1/62
    assert fused == ["b", "a", "c"]


def test_rrf_doc_in_both_lists_beats_single_top():
    assert rrf([["x", "both"], ["y", "both"]], k=60)[0] == "both"


def test_rrf_tie_is_stable():
    assert rrf([["a"], ["b"]]) == ["a", "b"]


# --- 지표 ---
def test_recall_and_mrr():
    ranks = [gold_rank(["g", "x"], "g"), gold_rank(["x", "y", "g"], "g"), gold_rank(["x"], "g")]
    assert ranks == [1, 3, None]
    assert recall_at_k(ranks, 1) == pytest.approx(1 / 3)
    assert recall_at_k(ranks, 3) == pytest.approx(2 / 3)
    assert mrr(ranks) == pytest.approx((1 + 1 / 3) / 3)
    assert recall_at_k([], 1) == 0 and mrr([]) == 0


def test_answer_coverage():
    assert answer_coverage("080 600", "연락처: 080600") == 1.0
    assert answer_coverage("가나", "다라") == 0.0


# --- dense (모델 없이) ---
def test_e5_prefixes_and_bge_none():
    assert with_prefix(["q"], SPECS["e5-base"], "query") == ["query: q"]
    assert with_prefix(["p"], SPECS["e5-base"], "passage") == ["passage: p"]
    assert with_prefix(["q"], SPECS["bge-m3"], "query") == ["q"]
    with pytest.raises(ValueError):
        with_prefix(["q"], SPECS["e5-base"], "doc")


def test_cache_key_changes_with_text_and_model():
    k = cache_key(SPECS["bge-m3"], ["a", "b"])
    assert k == cache_key(SPECS["bge-m3"], ["a", "b"])
    assert k != cache_key(SPECS["bge-m3"], ["ab"])  # 구분자 덕분에 이어 붙인 텍스트와 구별
    assert k != cache_key(SPECS["e5-base"], ["a", "b"])


class FakeEncoder:
    """글자별 one-hot 비슷한 임베딩. 같은 글자를 공유하면 유사도가 높다."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, kind):
        self.calls.append((kind, len(texts)))
        vocab = "abcdefgh"
        m = np.array([[t.count(ch) for ch in vocab] for t in texts], dtype=np.float32) + 1e-6
        return m / np.linalg.norm(m, axis=1, keepdims=True)


CHUNKS = [Chunk("d1:0", "d1", "aaa"), Chunk("d1:1", "d1", "bbb"), Chunk("d2:0", "d2", "ccc ddd")]


def test_embed_corpus_encodes_per_doc():
    enc = FakeEncoder()
    emb = embed_corpus(enc, CHUNKS)
    assert emb.shape == (3, 8) and enc.calls == [("passage", 2), ("passage", 1)]


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_retriever_methods_find_obvious_chunk(method):
    r = Retriever(CHUNKS, method, tokenizer="bigram", encoder=FakeEncoder())
    assert r.search("bbb")[0] == "d1:1"
    assert sorted(r.search("ccc")) == sorted(r.ids)


def test_retriever_requires_encoder_for_dense():
    with pytest.raises(ValueError):
        Retriever(CHUNKS, "dense")


def test_top_ids_stable_on_ties():
    assert top_ids(np.array([1.0, 2.0, 2.0]), ["a", "b", "c"], 2) == ["b", "c"]


# --- 컨텍스트 조립 ---
def test_build_context_joins_in_rank_order():
    assert build_context(["A", "B"], 100, len) == ("A\n\nB", False)


def test_build_context_truncates_lower_ranked_first():
    text, truncated = build_context(["aaaa", "bbbbbbbb", "cccc"], 10, len)
    # aaaa(4) + \n\n(2) + 남은 4글자만 bbbb, cccc는 버림
    assert text == "aaaa\n\nbbbb" and truncated


def test_build_context_top1_always_included_even_if_long():
    assert build_context(["x" * 20], 5, len) == ("xxxxx", True)


# --- 새 생성 모드 ---
QA = {"qa_id": "d:1:0", "chunk_id": "d:1", "question": "질문?", "answer": "답"}


def test_rag_retrieved_prompt_is_rag_format():
    ctx = "[섹션]\n청크1\n\n[섹션]\n청크2"
    msgs = prompt_messages(QA, "rag_retrieved", ctx)
    assert msgs == to_messages(QA, "rag", ctx)[:-1]
    assert msgs[-1]["content"] == f"[참고 문서]\n{ctx}\n\n[질문]\n질문?"


def test_closed_and_rag_prompts_unchanged():
    assert prompt_messages(QA, "closed", None) == to_messages(QA, "closed")[:-1]
    assert prompt_messages(QA, "rag", "원문") == to_messages(QA, "rag", "원문")[:-1]


def test_genconfig_identity_without_retrieval_matches_old_meta(tmp_path):
    cfg = GenConfig(model="m", adapter=None, mode="rag")
    assert "retrieval" not in cfg.identity() and "retrieval" not in cfg.meta()
    # retrieval 필드가 생기기 전에 만든 meta로도 이어하기가 막히지 않아야 한다
    old = {"model": "m", "adapter": None, "mode": "rag", "max_new_tokens": 256, "batch_size": 8,
           "repetition_penalty": 1.0, "device": "auto", "dtype": "auto"}
    (tmp_path / "meta.json").write_text(json.dumps({"generation": old}), encoding="utf-8")
    check_resume(tmp_path / "meta.json", cfg)


def test_genconfig_retrieval_settings_block_mixed_resume(tmp_path):
    a = GenConfig(model="m", adapter=None, mode="rag_retrieved", retrieval={"method": "bm25", "top_k": 1})
    b = GenConfig(model="m", adapter=None, mode="rag_retrieved", retrieval={"method": "bm25", "top_k": 3})
    (tmp_path / "meta.json").write_text(json.dumps({"generation": a.meta()}), encoding="utf-8")
    check_resume(tmp_path / "meta.json", a)
    with pytest.raises(SystemExit):
        check_resume(tmp_path / "meta.json", b)
