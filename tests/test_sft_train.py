"""체크포인트·이어하기·설정 덮어쓰기 테스트.

실제 언어모델 대신 선형층 하나짜리 작은 모델에 LoRA를 붙여, 저장 → 값 망가뜨리기 → 복원 흐름을 확인한다.
"""

import json
import random
from pathlib import Path

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model

from carmind.training.sft_train import (
    BatchStream,
    TrainConfig,
    apply_overrides,
    latest_checkpoint,
    load_checkpoint,
    load_sft_config,
    resume_identity,
    save_checkpoint,
    trim_log,
)

ROOT = Path(__file__).resolve().parents[1]


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.q_proj(x)


def make_model_and_optimizer(seed: int = 0):
    torch.manual_seed(seed)
    model = get_peft_model(Tiny(), PeftLoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    return model, optimizer


def train_one_step(model, optimizer):
    loss = model(torch.randn(4, 8)).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def lora_params(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def test_apply_overrides_parses_yaml_values():
    raw = {"train": {"max_steps": 100}, "model": {"name": "a"}}
    out = apply_overrides(raw, ["train.max_steps=5", "model.name=Qwen/X", "data.max_train_samples=null", "train.x=true"])
    assert out["train"] == {"max_steps": 5, "x": True}
    assert out["model"]["name"] == "Qwen/X"
    assert out["data"]["max_train_samples"] is None
    with pytest.raises(ValueError):
        apply_overrides({}, ["train.max_steps"])


def test_train_length_needs_exactly_one_of_steps_or_epochs():
    assert TrainConfig(num_epochs=1).max_steps is None
    with pytest.raises(ValueError):
        TrainConfig()
    with pytest.raises(ValueError):
        TrainConfig(max_steps=10, num_epochs=1)


def test_7b_configs_load_with_full_data_and_one_epoch():
    for name, max_len in (("sft_7b_closed", 512), ("sft_7b_rag", 1024)):
        cfg = load_sft_config(ROOT / f"configs/training/{name}.yaml")
        assert cfg.data.max_train_samples is None and cfg.data.max_seq_len == max_len
        assert cfg.train.num_epochs == 1 and cfg.train.batch_size * cfg.train.grad_accum == 16
        assert cfg.train.save_every_minutes


def test_batch_stream_matches_old_shuffle_order():
    """같은 seed면 기존 batches()와 같은 순서 → smoke 결과와 데이터 순서가 같다."""
    stream = BatchStream(10, 3, seed=42)
    order = list(range(10))
    random.Random(42).shuffle(order)
    got = [stream.next_indices() for _ in range(4)]
    assert got == [order[0:3], order[3:6], order[6:9], order[9:10]]
    assert stream.epoch == 1
    stream.next_indices()
    assert stream.epoch == 2


def test_batch_stream_state_roundtrip(tmp_path):
    a = BatchStream(7, 2, seed=1)
    for _ in range(5):  # epoch 경계를 넘긴 상태에서 저장
        a.next_indices()
    torch.save(a.state_dict(), tmp_path / "s.pt")
    b = BatchStream(7, 2, seed=999)
    b.load_state_dict(torch.load(tmp_path / "s.pt", weights_only=True))
    assert [a.next_indices() for _ in range(10)] == [b.next_indices() for _ in range(10)]


def test_checkpoint_save_load_restores_adapter_optimizer_and_data(tmp_path):
    model, opt = make_model_and_optimizer()
    stream = BatchStream(5, 2, seed=0)
    for _ in range(3):
        train_one_step(model, opt)
        stream.next_indices()
    saved_params = lora_params(model)
    saved_opt_step = opt.state_dict()["state"][0]["step"].item()
    path = save_checkpoint(tmp_path, 3, model, opt, stream, {"identity": {"a": 1}}, keep=2)
    assert path.name == "step_000003" and latest_checkpoint(tmp_path) == path
    expected_next = BatchStream(5, 2, seed=0)
    expected_next.load_state_dict(stream.state_dict())

    # 새 프로세스에서 다시 시작한 상황: 처음 상태의 모델·optimizer를 만들고 체크포인트를 불러온다
    model2, opt2 = make_model_and_optimizer(seed=123)
    stream2 = BatchStream(5, 2, seed=0)
    state = load_checkpoint(path, model2, opt2, stream2)
    assert state["step"] == 3 and state["identity"] == {"a": 1}
    for name, value in lora_params(model2).items():
        assert torch.equal(value, saved_params[name]), name
    assert opt2.state_dict()["state"][0]["step"].item() == saved_opt_step
    assert stream2.next_indices() == expected_next.next_indices()


def test_checkpoint_keeps_latest_n_and_ignores_partial(tmp_path):
    model, opt = make_model_and_optimizer()
    stream = BatchStream(4, 2, seed=0)
    for step in (10, 20, 30):
        save_checkpoint(tmp_path, step, model, opt, stream, {}, keep=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_000020", "step_000030"]
    # 저장 도중 끊겨 남은 폴더(.tmp, state.pt 없음)는 최신으로 치지 않는다
    (tmp_path / "step_000040.tmp").mkdir()
    (tmp_path / "step_000050").mkdir()
    assert latest_checkpoint(tmp_path).name == "step_000030"
    assert latest_checkpoint(tmp_path / "없음") is None


def test_trim_log_drops_entries_after_checkpoint(tmp_path):
    log = tmp_path / "train_log.jsonl"
    rows = [{"step": s, "split": "train", "loss": 1.0} for s in (0, 10, 20, 30)]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    kept = trim_log(log, 20)
    assert [r["step"] for r in kept] == [0, 10, 20]
    assert len(log.read_text(encoding="utf-8").splitlines()) == 3


def test_resume_identity_ignores_logging_and_save_intervals():
    base = load_sft_config(ROOT / "configs/training/sft_7b_closed.yaml")
    changed = load_sft_config(
        ROOT / "configs/training/sft_7b_closed.yaml",
        ["train.eval_every=1", "train.save_every_minutes=1", "output_dir=elsewhere"],
    )
    assert resume_identity(base) == resume_identity(changed)
    other_lr = load_sft_config(ROOT / "configs/training/sft_7b_closed.yaml", ["train.lr=1e-4"])
    assert resume_identity(base) != resume_identity(other_lr)
