from carmind.budget import DailyTokenBudget


def test_budget_accumulates_across_runs_and_pools(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    b1 = DailyTokenBudget(ledger, "gpt-5.4-mini", limit=100)
    b1.record(60)
    assert not b1.exhausted
    # 다른 실행·같은 무료 pool의 다른 모델도 합산된다
    b2 = DailyTokenBudget(ledger, "gpt-5.4-nano", limit=100)
    assert b2.used == 60
    b2.record(50)
    assert b2.exhausted
    # 다른 pool은 따로 센다
    assert DailyTokenBudget(ledger, "gpt-6-luna", limit=100).used == 0


def test_no_limit_never_exhausted(tmp_path):
    b = DailyTokenBudget(tmp_path / "l.jsonl", "gpt-5.4-mini", limit=None)
    b.record(10**9)
    assert not b.exhausted
