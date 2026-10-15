from pathlib import Path

from carmind.data.splits import final_qa_dir, load_holdout


def test_load_holdout(tmp_path):
    p = tmp_path / "splits.yaml"
    p.write_text("holdout:\n  - manual_a\n", encoding="utf-8")
    assert load_holdout(p) == {"manual_a"}
    assert load_holdout(tmp_path / "missing.yaml") == set()


def test_holdout_goes_to_eval_not_training_dir():
    processed = Path("data/processed")
    assert final_qa_dir(processed, "manual_a", {"manual_a"}) == processed / "eval"
    assert final_qa_dir(processed, "manual_b", {"manual_a"}) == processed / "qa"
