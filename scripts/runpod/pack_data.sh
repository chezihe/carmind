#!/usr/bin/env bash
# [맥북에서 실행] pod에 올릴 데이터만 tarball로 묶는다.
#
#   bash scripts/runpod/pack_data.sh            # → outputs/carmind_data.tar.gz
#
# 넣는 것:   SFT 학습 데이터(closed, rag), 평가셋, 평가셋 문서의 청크(rag 평가 입력), 익명화 규칙
# 넣지 않는 것: 원본 PDF(raw), 중간 산출물(interim), QA 원본, .env(API 키) — pod에서는 쓰지 않는다
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
OUT="${1:-outputs/carmind_data.tar.gz}"

files=(
  data/processed/sft/closed/train.jsonl
  data/processed/sft/closed/val.jsonl
  data/processed/sft/rag/train.jsonl
  data/processed/sft/rag/val.jsonl
  data/anonymize.yaml
)
shopt -s nullglob
eval_files=(data/processed/eval/*.jsonl)
shopt -u nullglob
((${#eval_files[@]})) || { echo "평가셋이 없습니다: data/processed/eval/*.jsonl" >&2; exit 1; }
files+=("${eval_files[@]}")

# rag 평가는 문항을 만든 원문 청크를 넣으므로, 평가셋에 나오는 문서(doc_id)의 청크 파일만 고른다.
# 문서 이름은 여기 적지 않고 평가셋 내용에서 읽는다.
while IFS= read -r chunk; do
  files+=("$chunk")
done < <(uv run python - "${eval_files[@]}" <<'PY'
import json, sys
from pathlib import Path
doc_ids = {json.loads(line)["doc_id"] for p in sys.argv[1:] for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()}
for d in sorted(doc_ids):
    path = Path("data/processed/chunks") / f"{d}.jsonl"
    if not path.exists():
        sys.exit(f"청크 파일이 없습니다: {path}")
    print(path)
PY
)

for f in "${files[@]}"; do
  [[ -f "$f" ]] || { echo "없음: $f" >&2; exit 1; }
done

mkdir -p "$(dirname "$OUT")"
# COPYFILE_DISABLE: macOS tar가 ._로 시작하는 메타데이터 파일을 끼워 넣지 않게 한다
COPYFILE_DISABLE=1 tar -czf "$OUT" "${files[@]}"

echo "== 포함 파일"
tar -tzvf "$OUT"
echo "== 크기: $(du -h "$OUT" | cut -f1)  → $OUT"
echo "올리기: scp -P <포트> -i ~/.ssh/id_ed25519 $OUT root@<IP>:/workspace/"
