#!/usr/bin/env bash
# [pod에서 tmux 안에서 실행] 학습 → 평가 답변 생성을 순서대로 한 번에 돌린다.
#
#   tmux new -s carmind
#   cd /workspace/carmind && bash scripts/runpod/run_all.sh
#   (Ctrl-b d 로 빠져나와도 계속 돈다. 다시 붙기: tmux attach -t carmind)
#
#   TINY=1 bash scripts/runpod/run_all.sh    # 맥북 점검용: 0.5B 모델, 몇 step, 평가 몇 문항 → outputs/runpod_tiny/
#
#   RUN_PROFILE=9b bash scripts/runpod/run_all.sh          # Qwen3.5-9B: sft_9b_*, eval/qwen35_9b_{sft,base}_{closed,rag}
#   RUN_PROFILE=9b TINY=1 bash scripts/runpod/run_all.sh   # 맥북 점검용: 같은 9B 설정 + Qwen3.5-0.8B 모델
#   프로필마다 폴더가 달라서 같은 pod에서 이어 돌려도 다른 모델의 결과를 덮어쓰지 않는다.
#
# 순서 (앞 단계 결과가 뒤 단계 입력이라 이 순서를 지킨다)
#   1. closed 학습              → outputs/sft_7b_closed/adapter
#   2. closed 평가 답변 (SFT)   → outputs/eval/qwen7b_sft_closed
#   3. rag 학습                 → outputs/sft_7b_rag/adapter
#   4. rag 평가 답변 (SFT)      → outputs/eval/qwen7b_sft_rag
#   5. base 모델 평가 답변 (closed, rag) → outputs/eval/qwen7b_base_{closed,rag}  (7B 기준선)
#
# 이어하기: 중간에 끊기면 같은 명령을 다시 실행하면 된다.
#   - 학습: summary.json이 있으면 건너뛰고, 없으면 마지막 체크포인트에서 이어서 학습
#   - 평가: 이미 답이 있는 문항은 건너뛰고 남은 문항만 생성 (다 있으면 모델도 안 불러옴)
# 채점은 여기서 하지 않는다. OpenAI 키는 맥북에만 두고, fetch_results.sh로 받아서 맥북에서 채점한다.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
# shellcheck source=scripts/runpod/env.sh
source scripts/runpod/env.sh

# 평가셋 파일은 이름을 적지 않고 폴더에 있는 것을 모두 쓴다
shopt -s nullglob
EVAL_FILES=(data/processed/eval/*.jsonl)
shopt -u nullglob
((${#EVAL_FILES[@]})) || { echo "평가셋이 없습니다: data/processed/eval/*.jsonl" >&2; exit 1; }

RUN_PROFILE="${RUN_PROFILE:-7b}"
case "$RUN_PROFILE" in
  7b)
    CFG_TAG=7b; RUN_PREFIX=qwen7b; EVAL_CONFIG=configs/eval/runpod.yaml
    STEP_SUFFIX=""; TINY_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
    ;;
  9b)
    CFG_TAG=9b; RUN_PREFIX=qwen35_9b; EVAL_CONFIG=configs/eval/runpod_9b.yaml
    STEP_SUFFIX="_9b"; TINY_MODEL="Qwen/Qwen3.5-0.8B"
    ;;
  *) echo "알 수 없는 RUN_PROFILE: $RUN_PROFILE (7b | 9b)" >&2; exit 1 ;;
esac

if [[ "${TINY:-0}" == 1 ]]; then
  # 맥북 점검용: 7B 설정 파일은 그대로 두고 --set으로 모델·장비·길이만 덮어쓴다 (경로·순서·이어하기 흐름 확인용)
  OUT_ROOT="outputs/runpod_tiny"
  TRAIN_SET=(
    --set "model.name=${TINY_MODEL}" --set runtime.device=auto --set runtime.dtype=float32
    --set data.max_train_samples=48 --set data.max_val_samples=8
    --set train.num_epochs=null --set "train.max_steps=${TINY_STEPS:-4}" --set train.warmup_steps=1
    --set train.batch_size=2 --set train.grad_accum=2
    --set train.log_every=1 --set train.eval_every=2
    --set train.save_every_steps=2 --set train.save_every_minutes=null
  )
  EVAL_SET=(--model "$TINY_MODEL" --device auto --dtype float32 --limit "${TINY_LIMIT:-4}"
            --batch-size 4 --max-new-tokens 32)
else
  OUT_ROOT="outputs"
  TRAIN_SET=()
  EVAL_SET=()
fi
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/run_all${STEP_SUFFIX}.log"

note() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"
}

# 명령을 실행하며 화면과 단계별 로그 파일에 같이 남긴다. 실패하면(pipefail) 스크립트 전체가 멈춘다.
run_logged() {
  local name="$1"
  shift
  local started=$SECONDS
  note "시작: $name  (로그: $LOG_DIR/$name.log)"
  "$@" 2>&1 | tee -a "$LOG_DIR/$name.log"
  note "끝:   $name  ($(((SECONDS - started) / 60))분)"
}

train_step() {
  local name="$1" config="$2" out="$3"
  if [[ -f "$out/summary.json" && -f "$out/adapter/adapter_model.safetensors" ]]; then
    note "건너뜀: $name (이미 완료: $out/summary.json)"
    return
  fi
  run_logged "$name" uv run python scripts/train_sft.py --config "$config" --out "$out" ${TRAIN_SET[@]+"${TRAIN_SET[@]}"}
}

gen_step() {
  local run="$1" mode="$2" adapter="${3:-}"
  local args=(--config "$EVAL_CONFIG" --eval "${EVAL_FILES[@]}" --mode "$mode"
              --run "$run" --out "$OUT_ROOT/eval")
  [[ -n "$adapter" ]] && args+=(--adapter "$adapter")
  run_logged "gen_$run" uv run python scripts/eval_generate.py "${args[@]}" ${EVAL_SET[@]+"${EVAL_SET[@]}"}
}

note "===== run_all 시작 (RUN_PROFILE=$RUN_PROFILE, TINY=${TINY:-0}, 결과: $OUT_ROOT) ====="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tee -a "$MAIN_LOG"
fi

train_step "train_closed$STEP_SUFFIX" "configs/training/sft_${CFG_TAG}_closed.yaml" "$OUT_ROOT/sft_${CFG_TAG}_closed"
gen_step "${RUN_PREFIX}_sft_closed" closed "$OUT_ROOT/sft_${CFG_TAG}_closed/adapter"
train_step "train_rag$STEP_SUFFIX" "configs/training/sft_${CFG_TAG}_rag.yaml" "$OUT_ROOT/sft_${CFG_TAG}_rag"
gen_step "${RUN_PREFIX}_sft_rag" rag "$OUT_ROOT/sft_${CFG_TAG}_rag/adapter"
gen_step "${RUN_PREFIX}_base_closed" closed
gen_step "${RUN_PREFIX}_base_rag" rag

note "===== run_all 완료 ====="
echo "맥북에서: bash scripts/runpod/fetch_results.sh <IP> <포트>  →  채점 후 pod를 꼭 멈추거나(stop) 지우세요(terminate)."
