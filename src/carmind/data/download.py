"""원본 문서(PDF) 수집.

sources.yaml에 적힌 URL을 내려받아 data/raw/에 저장하고,
파일마다 sha256 해시를 manifest.jsonl에 기록한다(데이터 계보 추적용).

sources.yaml 형식:
    sources:
      - id: manual_a_2025      # 저장 파일명 (data/raw/<id>.pdf)
        url: https://...
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel


class Source(BaseModel):
    id: str
    url: str


def load_sources(path: Path) -> list[Source]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Source(**item) for item in data.get("sources", [])]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(source: Source, out_dir: Path, client: httpx.Client) -> Path:
    """한 파일을 내려받는다. 이미 있으면 건너뛴다.

    임시 파일(.part)에 먼저 쓰고 완료 후 이름을 바꿔서,
    중간에 끊겨도 깨진 파일이 정상 파일처럼 남지 않게 한다.
    """
    dest = out_dir / f"{source.id}.pdf"
    if dest.exists():
        return dest

    tmp = dest.with_suffix(".pdf.part")
    with client.stream("GET", source.url) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type:
            raise ValueError(f"{source.id}: PDF가 아닙니다 (content-type={content_type})")
        with tmp.open("wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    tmp.rename(dest)
    return dest


def download_all(sources_path: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for source in load_sources(sources_path):
            path = download(source, out_dir, client)
            record = {
                "id": source.id,
                "url": source.url,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                "recorded_at": datetime.now(UTC).isoformat(),
            }
            records.append(record)
            print(f"[ok] {source.id}  {record['bytes'] / 1e6:.1f}MB  {record['sha256'][:12]}")

    manifest = out_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return records
