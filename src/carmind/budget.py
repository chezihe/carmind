"""하루 토큰 사용량 상한 (무료 한도 초과 방지).

실행할 때마다 사용 토큰을 장부(ledger)에 날짜별로 기록하고, 같은 날 여러 번 실행해도 합산한다.
무료 한도는 여러 모델이 함께 쓰는 묶음(pool) 단위라서 모델이 아니라 pool로 집계한다.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

# 같은 무료 한도를 공유하는 모델 묶음. 여기에 없는 모델(Claude 포함)은 모델 이름이 곧 pool
FREE_POOLS = {
    "gpt-5.4-mini": "openai-free-small",
    "gpt-5.4-nano": "openai-free-small",
}


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


class DailyTokenBudget:
    def __init__(self, ledger: Path, model: str, limit: int | None):
        self.ledger = ledger
        self.pool = FREE_POOLS.get(model, model)
        self.limit = limit
        self.used = self._used_today()

    def _used_today(self) -> int:
        if not self.ledger.exists():
            return 0
        rows = [json.loads(line) for line in self.ledger.read_text(encoding="utf-8").splitlines()]
        return sum(r["tokens"] for r in rows if r["date"] == today() and r["pool"] == self.pool)

    def record(self, tokens: int) -> None:
        self.used += tokens
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"date": today(), "pool": self.pool, "tokens": tokens}) + "\n")

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.used >= self.limit
