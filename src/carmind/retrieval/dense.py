"""임베딩 모델로 질문·청크를 벡터로 바꿔 코사인 유사도로 찾는 밀집(dense) 검색.

BM25는 글자가 겹쳐야 찾는다. "시동이 안 걸려요"와 "엔진이 시동되지 않을 때"처럼 표현이 다르면 놓친다.
임베딩은 의미가 가까우면 벡터도 가깝게 학습돼 있어 이런 바꿔 말하기(paraphrase)에 강하다.
반대로 부품명·수치 같은 정확한 일치는 BM25가 낫다. 그래서 둘을 RRF로 합친 hybrid도 함께 본다.

모델 선택 (M2 16GB, 다른 MPS 작업과 메모리 공유):
- BAAI/bge-m3 (기본): XLM-RoBERTa-large 기반 568M. 한국어가 포함된 다국어 검색 벤치마크(MIRACL 등)에서
  강하고, 최대 8192토큰이라 512토큰 청크가 잘리지 않는다. fp16이면 가중치 ~1.1GB라 맥북에서도 여유가 있다.
  질의·문서에 접두어(prefix)가 필요 없다.
- intfloat/multilingual-e5-base (비교용): 278M으로 2배 이상 빠르지만 최대 512토큰이라 긴 청크 끝이 잘릴 수 있다.
  e5는 학습 때 "query: " / "passage: " 접두어를 붙였으므로 추론 때도 반드시 붙여야 성능이 나온다
  (빠뜨리면 질의와 문서가 같은 종류의 텍스트로 취급돼 검색 품질이 떨어진다).

임베딩은 outputs/retrieval/cache/에 .npy로 캐시한다(outputs/는 gitignore). 키는 모델 이름·접두어·텍스트 전체의
해시라서, 청크나 익명화 규칙이 바뀌면 자동으로 새로 계산된다.
"""

import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbedSpec:
    model: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512


SPECS = {
    "bge-m3": EmbedSpec("BAAI/bge-m3", max_seq_length=1024),
    "e5-base": EmbedSpec("intfloat/multilingual-e5-base", "query: ", "passage: ", 512),
    "e5-large": EmbedSpec("intfloat/multilingual-e5-large", "query: ", "passage: ", 512),
}


def with_prefix(texts: list[str], spec: EmbedSpec, kind: str) -> list[str]:
    if kind not in ("query", "passage"):
        raise ValueError(f"kind는 query 또는 passage: {kind}")
    prefix = spec.query_prefix if kind == "query" else spec.passage_prefix
    return [prefix + t for t in texts]


def cache_key(spec: EmbedSpec, texts: list[str]) -> str:
    h = hashlib.sha1()
    h.update(f"{spec.model}|{spec.max_seq_length}\x00".encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class DenseEncoder:
    """SentenceTransformer를 감싼다. 모델은 캐시에 없는 텍스트를 처음 인코딩할 때만 불러온다.

    batch_size를 작게(기본 8) 두는 이유: 같은 MPS를 쓰는 다른 생성 작업과 메모리를 나눠 쓰기 때문이다.
    다 쓰면 close()로 모델을 내리고 MPS 캐시를 비운다 (그 뒤에 생성 모델을 올릴 자리를 만든다).
    """

    def __init__(self, spec: EmbedSpec, device: str = "auto", batch_size: int = 8,
                 cache_dir: Path | None = Path("outputs/retrieval/cache")):
        self.spec, self.device_name, self.batch_size, self.cache_dir = spec, device, batch_size, cache_dir
        self.model = None

    def _load(self):
        import torch
        from sentence_transformers import SentenceTransformer

        from carmind.device import resolve_device

        device = resolve_device(self.device_name)
        model = SentenceTransformer(self.spec.model, device=str(device))
        if device.type != "cpu":
            model = model.half()  # 검색 순위에는 fp16 오차가 거의 영향을 주지 않고 메모리는 절반
        model.max_seq_length = self.spec.max_seq_length
        self.model, self._torch, self.device = model, torch, device

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        """L2 정규화된 float32 임베딩 (n, dim). 정규화했으므로 내적 = 코사인 유사도."""
        inputs = with_prefix(texts, self.spec, kind)
        path = self.cache_dir / f"{cache_key(self.spec, inputs)}.npy" if self.cache_dir else None
        if path and path.exists():
            return np.load(path)
        if self.model is None:
            self._load()
        emb = self.model.encode(
            inputs, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=len(inputs) > 200,
        ).astype(np.float32)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb

    def encode_uncached(self, text: str, kind: str) -> np.ndarray:
        """질의 1개를 캐시 없이 인코딩한다 (실제 서비스의 질의당 지연 시간 측정용)."""
        if self.model is None:
            self._load()
        return self.model.encode(with_prefix([text], self.spec, kind), normalize_embeddings=True,
                                 convert_to_numpy=True)[0]

    def close(self) -> None:
        if self.model is None:
            return
        device = self.device
        self.model = None
        gc.collect()
        if device.type == "mps":
            self._torch.mps.empty_cache()
        elif device.type == "cuda":
            self._torch.cuda.empty_cache()
