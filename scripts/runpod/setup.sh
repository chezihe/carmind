#!/usr/bin/env bash
# [pod에서 실행] 학습·평가 환경 준비. 여러 번 실행해도 안전하다 (pod를 stop → start 한 뒤에도 다시 실행).
#
#   cd /workspace/carmind && bash scripts/runpod/setup.sh
#
# 하는 일
#   1. 필요한 시스템 도구(tmux, rsync) 설치
#   2. GPU 확인 (nvidia-smi) → 드라이버가 지원하는 CUDA 버전 확인
#   3. uv 설치 → uv sync (uv.lock 그대로)
#      PyPI의 리눅스 torch 휠은 CUDA 13.0용이라 드라이버가 CUDA 13 미만이면 cu126 휠로 바꿔 설치한다
#   4. 데이터 tarball 풀기 (/workspace/carmind_data.tar.gz가 있고 아직 안 풀었으면)
#   5. torch.cuda 동작 확인 → 7B 모델 미리 다운로드 (HF 캐시는 /workspace 아래)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
# shellcheck source=scripts/runpod/env.sh
source scripts/runpod/env.sh

DATA_TAR="${DATA_TAR:-/workspace/carmind_data.tar.gz}"
TORCH_VERSION="2.14.1"   # uv.lock에 고정된 버전과 같아야 한다

echo "== 1. 시스템 도구"
missing=()
for tool in tmux rsync curl; do
  command -v "$tool" >/dev/null || missing+=("$tool")
done
if ((${#missing[@]})); then
  apt-get update -qq && apt-get install -y -qq "${missing[@]}"
fi

echo "== 2. GPU 확인"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# nvidia-smi 머리말의 "CUDA Version: 12.8" = 이 드라이버가 돌릴 수 있는 최대 CUDA 버전
DRIVER_CUDA="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || echo 0.0)"
echo "드라이버가 지원하는 CUDA: ${DRIVER_CUDA}"

echo "== 3. uv + 파이썬 패키지"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
uv sync --frozen
if [[ "${DRIVER_CUDA%%.*}" -lt 13 ]]; then
  # PyPI torch(CUDA 13.0 내장)는 드라이버 580 미만에서 cuda를 못 쓴다. 같은 버전의 CUDA 12.6 빌드로 바꾼다.
  # env.sh의 UV_NO_SYNC=1 덕분에 이후 uv run이 lock 기준(CUDA 13)으로 되돌리지 않는다.
  echo "드라이버 CUDA ${DRIVER_CUDA} < 13 → torch ${TORCH_VERSION}+cu126으로 교체"
  uv pip install --python .venv/bin/python --reinstall-package torch \
    "torch==${TORCH_VERSION}" --extra-index-url https://download.pytorch.org/whl/cu126
fi

echo "== 4. 데이터"
if [[ -f data/processed/sft/closed/train.jsonl ]]; then
  echo "데이터 이미 있음 (data/processed)"
elif [[ -f "$DATA_TAR" ]]; then
  tar -xzf "$DATA_TAR" -C "$REPO_DIR"
  echo "풀기 완료: $DATA_TAR"
else
  echo "데이터가 없습니다. 맥북에서 pack_data.sh로 만든 tarball을 ${DATA_TAR}에 올리세요." >&2
  exit 1
fi
for f in data/processed/sft/closed/train.jsonl data/processed/sft/rag/train.jsonl data/anonymize.yaml; do
  [[ -f "$f" ]] || { echo "없음: $f" >&2; exit 1; }
done
ls data/processed/eval/*.jsonl >/dev/null

echo "== 5. torch.cuda 확인"
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch에서 CUDA를 쓸 수 없습니다 (드라이버·torch 빌드 확인)"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | {torch.cuda.get_device_name(0)} | "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB | bf16={torch.cuda.is_bf16_supported()}")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("행렬곱 확인:", float((x @ x).float().abs().mean()))
PY

echo "== 6. 모델 다운로드 → ${HF_HOME:-기본 HF 캐시}"
# 공개 모델이라 토큰이 없어도 받을 수 있다. HF_TOKEN을 export 해 두면 속도 제한이 느슨해진다.
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('${MODEL_7B}'))"

df -h /workspace 2>/dev/null || true
echo "준비 완료. 다음: tmux new -s carmind  →  bash scripts/runpod/run_all.sh"
