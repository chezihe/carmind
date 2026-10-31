"""YAML 설정 로더.

기본 설정(base.yaml) 위에 환경별 설정(env/*.yaml)을 덮어써서 최종 설정을 만든다.
    base.yaml        모든 환경 공통 값
    env/local.yaml   맥북: 작은 모델, 적은 데이터
    env/runpod.yaml  GPU 서버: 실제 학습용 모델과 데이터
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from carmind.device import DeviceName, DtypeName

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class RuntimeConfig(BaseModel):
    device: DeviceName = "auto"
    dtype: DtypeName = "auto"
    seed: int = 42


class ModelConfig(BaseModel):
    name: str
    # attention 구현. None이면 transformers 기본값(보통 sdpa). flash_attention_2는 패키지를 따로 설치해야 한다
    attn_implementation: str | None = None
    # apply_chat_template에 그대로 넘길 인자. Qwen3.5처럼 템플릿이 옵션을 받는 모델에서 쓴다 (예: enable_thinking: false).
    # None이면 아무것도 넘기지 않으므로 Qwen2.5 프롬프트는 이 필드가 생기기 전과 똑같다
    chat_template_kwargs: dict[str, Any] | None = None


class ChunkConfig(BaseModel):
    min_tokens: int = 100
    max_tokens: int = 512


class DataConfig(BaseModel):
    max_samples: int | None = None  # None이면 전체 사용
    chunk: ChunkConfig = ChunkConfig()


class Config(BaseModel):
    env: str
    runtime: RuntimeConfig = RuntimeConfig()
    model: ModelConfig
    data: DataConfig = DataConfig()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """중첩 dict를 재귀적으로 합친다. 같은 키는 override 값이 이긴다."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(env: str = "local", config_dir: Path = CONFIG_DIR) -> Config:
    base = yaml.safe_load((config_dir / "base.yaml").read_text(encoding="utf-8")) or {}
    override = yaml.safe_load((config_dir / "env" / f"{env}.yaml").read_text(encoding="utf-8")) or {}
    return Config(env=env, **deep_merge(base, override))
