"""실행 장비(device)와 연산 정밀도(dtype)를 결정한다.

같은 코드가 맥북(MPS)과 GPU 서버(CUDA)에서 모두 돌도록,
장비에 의존하는 결정은 이 모듈 한 곳에서만 내린다.
"""

from typing import Literal

import torch

DeviceName = Literal["auto", "cuda", "mps", "cpu"]
DtypeName = Literal["auto", "bfloat16", "float16", "float32"]


def resolve_device(name: DeviceName = "auto") -> torch.device:
    """'auto'면 사용 가능한 장비를 cuda > mps > cpu 순으로 고른다."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda를 요청했지만 사용할 수 없습니다.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps를 요청했지만 사용할 수 없습니다.")
    return torch.device(name)


def resolve_dtype(name: DtypeName, device: torch.device) -> torch.dtype:
    """'auto'면 장비에 맞는 정밀도를 고른다.

    - cuda: bf16 지원 GPU(A100, H100, L4 등)는 bfloat16, 아니면 float16
    - mps:  float16 (Apple GPU는 bf16 연산 지원이 제한적)
    - cpu:  float32
    """
    if name != "auto":
        return getattr(torch, name)

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32
