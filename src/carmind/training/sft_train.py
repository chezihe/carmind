"""LoRA SFT 학습 루프.

HF Trainer 대신 PyTorch 학습 루프를 직접 쓴다. 한 step에서 일어나는 일
(forward → loss → backward → 모아서 업데이트)이 코드에 그대로 드러나서 따라가기 쉽고,
MPS/CUDA 차이는 device.py가 정해 준 device/dtype만 바꾸면 된다.

LoRA: 원래 가중치 W는 얼려 두고(학습 안 함), 옆에 작은 행렬 두 개(A: d×r, B: r×d)를 붙여
W + (alpha/r)·B·A 로 계산한다. r이 작아서 학습할 파라미터가 전체의 1% 안팎으로 줄고,
결과물(adapter)도 수십 MB라 원본 모델과 따로 저장·교체할 수 있다.
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel

from carmind.config import ModelConfig, RuntimeConfig
from carmind.training.sft_tokenize import IGNORE_INDEX, collate, encode_example

PAD_MULTIPLE = 32  # 배치 길이를 32의 배수로 맞춘다 (collate 설명 참고)


class SFTDataConfig(BaseModel):
    train: Path
    val: Path
    max_train_samples: int | None = None  # None이면 전체 사용
    max_val_samples: int | None = None
    max_seq_len: int = 512


class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str]


class TrainConfig(BaseModel):
    lr: float = 2e-4
    warmup_steps: int = 10
    max_steps: int
    batch_size: int = 4
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    log_every: int = 10
    eval_every: int = 50


class SFTConfig(BaseModel):
    runtime: RuntimeConfig = RuntimeConfig()
    model: ModelConfig
    data: SFTDataConfig
    lora: LoraConfig
    train: TrainConfig
    output_dir: Path


def load_sft_config(path: Path) -> SFTConfig:
    return SFTConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


def load_messages(path: Path, limit: int | None, seed: int) -> list[dict]:
    """jsonl을 읽고, limit이 있으면 seed로 고정된 무작위 부분집합을 뽑는다.

    앞에서부터 자르면 파일 순서상 한두 권의 문서만 들어가므로 무작위로 뽑는다.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def tokenize_rows(rows: list[dict], tokenizer, max_seq_len: int) -> list[dict]:
    encoded = [encode_example(r["messages"], tokenizer, max_seq_len) for r in rows]
    return [e for e in encoded if e is not None]


def batches(examples: list[dict], batch_size: int, pad_token_id: int, rng: random.Random | None):
    """rng를 주면 섞어서(학습용), 안 주면 순서대로(평가용) 배치를 만든다."""
    order = list(range(len(examples)))
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        yield collate([examples[j] for j in order[i:i + batch_size]], pad_token_id, PAD_MULTIPLE)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """warmup 동안 0 → lr로 선형 증가, 이후 cosine으로 0까지 감소.

    처음부터 큰 lr로 시작하면 아직 0인 LoRA B 행렬이 크게 튀어 학습이 불안정해질 수 있다.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, examples: list[dict], batch_size: int, pad_token_id: int, device: torch.device) -> float:
    """val loss를 토큰 가중 평균으로 계산한다.

    배치마다 답변 길이가 다르므로 배치 평균을 다시 평균내면 짧은 답이 과하게 반영된다.
    그래서 (배치 loss × 그 배치의 답변 토큰 수)를 모두 더해 전체 토큰 수로 나눈다.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in batches(examples, batch_size, pad_token_id, rng=None):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        # 모델 내부에서 label을 한 칸 밀어(다음 토큰 예측) 계산하므로 토큰 수도 같은 기준으로 센다
        n = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    empty_cache(device)
    return total_loss / max(1, total_tokens)


def empty_cache(device: torch.device) -> None:
    """평가처럼 모양이 다른 텐서를 많이 만든 뒤에는 캐시를 비워 메모리가 계속 쌓이지 않게 한다."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def peak_memory_gb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        # MPS는 peak 값을 따로 주지 않으므로 호출 시점의 드라이버 할당량을 쓴다 (학습 중 여러 번 재서 최댓값)
        return torch.mps.driver_allocated_memory() / 1e9
    return None


def train(cfg: SFTConfig, max_steps: int | None = None) -> dict:
    """설정대로 LoRA SFT를 돌리고 로그·adapter를 output_dir에 저장한다.

    max_steps를 주면 설정값 대신 그만큼만 돈다 (s/step을 재는 드라이런용).
    """
    # 무거운 라이브러리는 실제 학습할 때만 불러온다 (테스트·설정 로딩을 가볍게 유지)
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from carmind.device import resolve_device, resolve_dtype

    tcfg = cfg.train.model_copy(update={"max_steps": max_steps}) if max_steps else cfg.train
    random.seed(cfg.runtime.seed)
    torch.manual_seed(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)
    dtype = resolve_dtype(cfg.runtime.dtype, device)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    train_rows = load_messages(cfg.data.train, cfg.data.max_train_samples, cfg.runtime.seed)
    val_rows = load_messages(cfg.data.val, cfg.data.max_val_samples, cfg.runtime.seed)
    train_ex = tokenize_rows(train_rows, tokenizer, cfg.data.max_seq_len)
    val_ex = tokenize_rows(val_rows, tokenizer, cfg.data.max_seq_len)
    print(f"데이터: train {len(train_ex)} / val {len(val_ex)} (max_seq_len {cfg.data.max_seq_len})")

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=dtype).to(device)
    if tcfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # 원래 가중치가 얼어 있으면 입력 임베딩에 grad가 없어 checkpointing이 끊긴다. 강제로 grad를 켠다.
        model.enable_input_require_grads()

    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터: {trainable:,} / 전체 {total:,} ({trainable / total:.2%})  device={device} dtype={dtype}")

    # 얼린 가중치는 optimizer에 넣지 않는다 (AdamW 상태가 파라미터마다 2배씩 메모리를 먹기 때문)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tcfg.lr, weight_decay=0.0)

    log: list[dict] = []
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    def record(entry: dict) -> None:
        log.append(entry)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 학습 전 val loss = 기준선. 이후 값과 비교해야 "내려갔다"를 말할 수 있다.
    start = time.time()
    val0 = evaluate(model, val_ex, tcfg.batch_size, pad_id, device)
    record({"step": 0, "split": "val", "loss": val0, "elapsed": time.time() - start})
    print(f"step    0 | val loss {val0:.4f}")

    rng = random.Random(cfg.runtime.seed)
    data_iter = batches(train_ex, tcfg.batch_size, pad_id, rng)
    peak_mem = 0.0
    window: list[float] = []
    train_start = time.time()
    model.train()

    for step in range(tcfg.max_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, tcfg)

        # gradient accumulation: 작은 배치를 grad_accum번 forward/backward 해서 grad를 모은 뒤 한 번만 업데이트.
        # 메모리는 batch_size만큼만 쓰면서 실효 배치는 batch_size × grad_accum이 된다.
        step_loss = 0.0
        for _ in range(tcfg.grad_accum):
            batch = next(data_iter, None)
            if batch is None:  # 한 epoch를 다 돌면 다시 섞어서 처음부터
                data_iter = batches(train_ex, tcfg.batch_size, pad_id, rng)
                batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / tcfg.grad_accum  # 더해지는 grad의 크기를 한 배치 기준으로 맞춘다
            loss.backward()
            step_loss += loss.item()

        if not math.isfinite(step_loss):
            raise RuntimeError(f"step {step + 1}에서 loss가 {step_loss}입니다. dtype을 float32로 바꿔 보세요.")

        # grad가 갑자기 커지는 step이 있어도 한 번에 너무 멀리 가지 않도록 크기를 자른다
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        window.append(step_loss)
        done = step + 1
        if done % tcfg.log_every == 0 or done == tcfg.max_steps:
            elapsed = time.time() - train_start
            mem = peak_memory_gb(device)
            peak_mem = max(peak_mem, mem or 0.0)
            avg = sum(window) / len(window)  # 마지막 log 이후 step들의 평균 (한 step 값은 들쭉날쭉하다)
            record({"step": done, "split": "train", "loss": avg, "lr": lr_at(step, tcfg), "elapsed": elapsed})
            print(f"step {done:4d} | train loss {avg:.4f} | lr {lr_at(step, tcfg):.2e} | {elapsed / done:.2f}s/step")
            window = []
        if done % tcfg.eval_every == 0 or done == tcfg.max_steps:
            v = evaluate(model, val_ex, tcfg.batch_size, pad_id, device)
            record({"step": done, "split": "val", "loss": v, "elapsed": time.time() - train_start})
            print(f"step {done:4d} | val loss {v:.4f}")

    train_time = time.time() - train_start
    # adapter(LoRA 가중치)만 저장한다. 원본 모델은 허브에서 다시 받으면 되므로 수십 MB면 충분하다.
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")

    summary = {
        "model": cfg.model.name,
        "device": str(device),
        "dtype": str(dtype),
        "train_samples": len(train_ex),
        "val_samples": len(val_ex),
        "max_steps": tcfg.max_steps,
        "effective_batch": tcfg.batch_size * tcfg.grad_accum,
        "trainable_params": trainable,
        "total_params": total,
        "train_time_sec": train_time,
        "sec_per_step": train_time / tcfg.max_steps,
        "peak_memory_gb": peak_mem or None,
        "val_loss_start": val0,
        "val_loss_end": next(e["loss"] for e in reversed(log) if e["split"] == "val"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    return summary
