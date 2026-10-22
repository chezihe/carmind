#!/usr/bin/env bash
# RunPod 스크립트 공통 환경 변수. setup.sh / run_all.sh가 source 한다 (직접 실행하지 않음).
#
# RunPod은 pod를 멈추면(stop) 컨테이너 디스크가 지워지고 /workspace(볼륨)만 남는다.
# 그래서 모델 캐시·uv 캐시·uv가 받은 파이썬을 모두 /workspace 아래에 둔다.
# 맥북에서 run_all.sh를 TINY=1로 돌릴 때는 /workspace가 없으므로 기본 위치를 그대로 쓴다.

if [[ -d /workspace ]]; then
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/.cache/uv-python}"
fi

# uv 설치 위치 (setup.sh가 설치)
export PATH="$HOME/.local/bin:$PATH"

# setup.sh에서 맞춰 둔 패키지(특히 CUDA 버전별 torch)를 uv run이 lock 기준으로 되돌리지 않게 한다
export UV_NO_SYNC=1

# 파이썬 출력이 tee·파일로 갈 때도 버퍼에 쌓지 않고 바로 내보낸다 (접속이 끊겨도 로그에 진행 상황이 남도록)
export PYTHONUNBUFFERED=1

# 7B 모델 기본값
export MODEL_7B="${MODEL_7B:-Qwen/Qwen2.5-7B-Instruct}"
