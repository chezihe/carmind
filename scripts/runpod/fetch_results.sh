#!/usr/bin/env bash
# [맥북에서 실행] pod의 outputs/(adapter, 학습 로그, 평가 답변)를 맥북 outputs/로 가져온다.
#
#   bash scripts/runpod/fetch_results.sh <IP> <포트> [ssh 키 경로]
#   FETCH_CHECKPOINTS=1 bash scripts/runpod/fetch_results.sh ...   # 중간 체크포인트까지 (수 GB)
#
# 경로 구조가 맥북과 같으므로 받은 뒤 바로 scripts/eval_grade.py outputs/eval/<run>으로 채점할 수 있다.
# 여러 번 실행해도 바뀐 파일만 받는다 (학습 도중 로그만 확인할 때도 쓸 수 있음).
set -euo pipefail

if (($# < 2)); then
  echo "사용법: $0 <IP> <포트> [ssh 키 경로]" >&2
  exit 1
fi
HOST="$1"
PORT="$2"
KEY="${3:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/carmind/outputs}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

excludes=(--exclude "*.tar.gz")
# 체크포인트(학습 재개용 optimizer 상태 포함)는 끝난 뒤에는 필요 없고 크므로 기본으로 뺀다
[[ "${FETCH_CHECKPOINTS:-0}" == 1 ]] || excludes+=(--exclude "checkpoints/")

rsync -avz --partial "${excludes[@]}" \
  -e "ssh -p ${PORT} -i ${KEY} -o StrictHostKeyChecking=accept-new" \
  "root@${HOST}:${REMOTE_DIR}/" outputs/

echo "== 받은 평가 실행"
ls -d outputs/eval/qwen* 2>/dev/null || echo "(아직 없음)"
echo "채점: uv run python scripts/eval_grade.py outputs/eval/<run> --dry-run   # 비용 확인 후 --dry-run 빼고 실행"
