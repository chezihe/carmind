"""검색기(retriever): 질문에 맞는 매뉴얼 청크를 찾아 RAG 프롬프트의 [참고 문서]로 넣는다.

- tokenize: BM25용 한국어 토큰화 (문자 bigram / 공백+bigram / 형태소)
- bm25:     어휘 일치 기반 희소(sparse) 검색
- dense:    임베딩 모델 기반 밀집(dense) 검색 + 디스크 캐시
- fusion:   두 검색 결과를 순위로 합치는 RRF
- metrics:  Recall@k, MRR
- corpus:   익명화된 청크 코퍼스와 검색기 조립
"""
