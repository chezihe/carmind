"""LoRA SFT 학습 루프.

HF Trainer 대신 PyTorch 학습 루프를 직접 쓴다. 한 step에서 일어나는 일
(forward → loss → backward → 모아서 업데이트)이 코드에 그대로 드러나서 따라가기 쉽고,
MPS/CUDA 차이는 device.py가 정해 준 device/dtype만 바꾸면 된다.

LoRA: 원래 가중치 W는 얼려 두고(학습 안 함), 옆에 작은 행렬 두 개(A: d×r, B: r×d)를 붙여
W + (alpha/r)·B·A 로 계산한다. r이 작아서 학습할 파라미터가 전체의 1% 안팎으로 줄고,
결과물(adapter)도 수십 MB라 원본 모델과 따로 저장·교체할 수 있다.

체크포인트와 이어하기: 시간 단위로 돈을 내는 GPU에서는 중간에 끊겨도(접속 끊김, 서버 회수)
처음부터 다시 돌지 않도록 일정 간격으로 output_dir/checkpoints/step_XXXXXX/에 상태를 저장한다.
같은 명령을 다시 실행하면 가장 최근 체크포인트에서 이어서 학습한다.
"""

import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, model_validator

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
    # 이름 목록이면 이름 끝이 일치하는 모듈, 문자열이면 전체 이름에 대한 정규식 (peft 규칙)
    target_modules: list[str] | str


class TrainConfig(BaseModel):
    lr: float = 2e-4
    warmup_steps: int = 10
    # 학습 길이는 둘 중 하나로 정한다. num_epochs를 주면 데이터 수에 맞춰 max_steps를 계산한다
    max_steps: int | None = None
    num_epochs: float | None = None
    batch_size: int = 4
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    log_every: int = 10
    eval_every: int = 50
    # 체크포인트 저장 간격. 둘 다 주면 먼저 닿는 쪽에서 저장한다. 둘 다 None이면 저장하지 않는다
    save_every_steps: int | None = None
    save_every_minutes: float | None = None
    keep_checkpoints: int = 2  # 저장 도중 끊겨 최신 것이 깨져도 하나 전 것으로 돌아갈 수 있게 2개 유지

    @model_validator(mode="after")
    def _check_length(self):
        if (self.max_steps is None) == (self.num_epochs is None):
            raise ValueError("train.max_steps와 train.num_epochs 중 정확히 하나만 지정하세요.")
        return self


class SFTConfig(BaseModel):
    runtime: RuntimeConfig = RuntimeConfig()
    model: ModelConfig
    data: SFTDataConfig
    lora: LoraConfig
    train: TrainConfig
    output_dir: Path


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """"train.max_steps=5" 같은 점 표기 덮어쓰기를 yaml dict에 적용한다.

    값은 yaml 문법으로 읽으므로 5 → int, null → None, true → bool이 된다.
    GPU용 설정 파일을 그대로 두고 맥북에서 작은 모델·몇 step으로 흐름만 확인할 때 쓴다.
    """
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"덮어쓰기는 key=value 형식이어야 합니다: {item}")
        *parents, leaf = key.split(".")
        node = raw
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    return raw


def load_sft_config(path: Path, overrides: list[str] | None = None) -> SFTConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SFTConfig(**apply_overrides(raw, overrides or []))


def load_messages(path: Path, limit: int | None, seed: int) -> list[dict]:
    """jsonl을 읽고, limit이 있으면 seed로 고정된 무작위 부분집합을 뽑는다.

    앞에서부터 자르면 파일 순서상 한두 권의 문서만 들어가므로 무작위로 뽑는다.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def tokenize_rows(rows: list[dict], tokenizer, max_seq_len: int, template_kwargs: dict | None = None) -> list[dict]:
    encoded = [encode_example(r["messages"], tokenizer, max_seq_len, template_kwargs) for r in rows]
    return [e for e in encoded if e is not None]


def batches(examples: list[dict], batch_size: int, pad_token_id: int, rng: random.Random | None):
    """rng를 주면 섞어서(학습용), 안 주면 순서대로(평가용) 배치를 만든다."""
    order = list(range(len(examples)))
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        yield collate([examples[j] for j in order[i:i + batch_size]], pad_token_id, PAD_MULTIPLE)


class BatchStream:
    """학습 배치의 순서를 끝없이 공급하고, 그 위치를 저장·복원할 수 있게 한다.

    이어하기를 할 때 데이터 순서가 처음부터 다시 시작되면 앞부분 샘플을 두 번 보고 뒷부분은 못 보게 된다.
    그래서 "이번 epoch의 섞인 순서 + 어디까지 썼는지 + 다음 epoch를 섞을 난수 상태"를 체크포인트에 같이 저장한다.
    섞는 방식은 batches()와 같아서, 같은 seed면 이전 버전과 같은 순서의 배치가 나온다.
    """

    def __init__(self, n_examples: int, batch_size: int, seed: int):
        self.n = n_examples
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.order: list[int] = []
        self.pos = 0
        self.epoch = 0

    def next_indices(self) -> list[int]:
        if self.pos >= len(self.order):  # 한 epoch를 다 돌면 다시 섞어서 처음부터
            self.order = list(range(self.n))
            self.rng.shuffle(self.order)
            self.pos = 0
            self.epoch += 1
        idx = self.order[self.pos:self.pos + self.batch_size]
        self.pos += len(idx)
        return idx

    def state_dict(self) -> dict:
        return {"order": list(self.order), "pos": self.pos, "epoch": self.epoch, "rng": self.rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        self.order = list(state["order"])
        self.pos = state["pos"]
        self.epoch = state["epoch"]
        # torch.save를 거치면 tuple이 list로 바뀔 수 있어 getstate()와 같은 모양으로 되돌린다
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))


# ---------------------------------------------------------------- 체크포인트

CKPT_PREFIX = "step_"
STATE_FILE = "state.pt"


def resume_identity(cfg: SFTConfig) -> dict:
    """이어하기 때 '같은 실행인지' 판단할 설정. 이게 다르면 섞인 학습이 되므로 막는다.

    로그·평가·저장 간격과 output_dir은 결과(가중치)에 영향이 없으므로 바꿔서 이어 돌릴 수 있게 뺀다.
    """
    d = cfg.model_dump(mode="json")
    for k in ("log_every", "eval_every", "save_every_steps", "save_every_minutes", "keep_checkpoints"):
        d["train"].pop(k)
    d.pop("output_dir")
    # None이면 키를 빼서, 이 필드가 생기기 전에 저장한 체크포인트도 같은 실행으로 인식한다
    if d["model"]["chat_template_kwargs"] is None:
        d["model"].pop("chat_template_kwargs")
    return d


def latest_checkpoint(ckpt_root: Path) -> Path | None:
    """완전히 저장된(state.pt가 있는) 체크포인트 중 step이 가장 큰 것. 저장 중이던 .tmp 폴더는 무시한다."""
    if not ckpt_root.exists():
        return None
    done = [p for p in ckpt_root.iterdir() if p.name.startswith(CKPT_PREFIX) and (p / STATE_FILE).exists()]
    return max(done, key=lambda p: int(p.name[len(CKPT_PREFIX):]), default=None)


def save_checkpoint(ckpt_root: Path, step: int, model, optimizer, stream: BatchStream, extra: dict, keep: int) -> Path:
    """adapter + optimizer 상태 + 데이터 위치 + 난수 상태 + step을 저장한다.

    스케줄러는 lr_at(step) 함수라 따로 상태가 없고 step만 있으면 같은 lr이 나온다.
    저장 도중 끊기면 반쯤 쓴 폴더가 남으므로, .tmp 폴더에 다 쓴 뒤 이름을 바꾼다(rename은 한 번에 일어남).
    """
    final = ckpt_root / f"{CKPT_PREFIX}{step:06d}"
    tmp = ckpt_root / f"{final.name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter")
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "stream": stream.state_dict(),
        # LoRA dropout이 쓰는 torch 난수까지 복원해야 끊기지 않았을 때와 같은 학습이 된다
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        **extra,
    }
    torch.save(state, tmp / STATE_FILE)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)

    # 오래된 체크포인트 정리 (최근 keep개만 남김)
    olds = sorted(
        (p for p in ckpt_root.iterdir() if p.name.startswith(CKPT_PREFIX) and (p / STATE_FILE).exists()),
        key=lambda p: int(p.name[len(CKPT_PREFIX):]),
    )
    for p in olds[:-keep] if keep > 0 else []:
        shutil.rmtree(p, ignore_errors=True)
    return final


def load_checkpoint(path: Path, model, optimizer, stream: BatchStream) -> dict:
    """save_checkpoint로 저장한 상태를 모델·optimizer·stream에 되돌리고 state dict를 돌려준다."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_file = path / "adapter" / "adapter_model.safetensors"
    result = set_peft_model_state_dict(model, load_file(str(adapter_file)))
    unexpected = getattr(result, "unexpected_keys", None)
    if unexpected:
        raise RuntimeError(f"체크포인트 adapter가 현재 모델과 맞지 않습니다: {unexpected[:3]}")
    state = torch.load(path / STATE_FILE, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    stream.load_state_dict(state["stream"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if state.get("mps_rng") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps_rng"])
    return state


def trim_log(log_path: Path, step: int) -> list[dict]:
    """체크포인트 이후에 기록된 로그(다시 돌릴 구간)를 지우고 남은 기록을 돌려준다."""
    if not log_path.exists():
        return []
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [r for r in rows if r["step"] <= step]
    log_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    return kept


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
    output_dir/checkpoints에 체크포인트가 있으면 가장 최근 것에서 이어서 학습한다.
    """
    # 무거운 라이브러리는 실제 학습할 때만 불러온다 (테스트·설정 로딩을 가볍게 유지)
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from carmind.device import resolve_device, resolve_dtype

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
    train_ex = tokenize_rows(train_rows, tokenizer, cfg.data.max_seq_len, cfg.model.chat_template_kwargs)
    val_ex = tokenize_rows(val_rows, tokenizer, cfg.data.max_seq_len, cfg.model.chat_template_kwargs)
    print(f"데이터: train {len(train_ex)} / val {len(val_ex)} (max_seq_len {cfg.data.max_seq_len})")

    # num_epochs로 정했으면 실제 샘플 수(너무 길어 버린 것 제외)로 optimizer step 수를 계산한다
    tcfg = cfg.train
    if max_steps:
        tcfg = tcfg.model_copy(update={"max_steps": max_steps, "num_epochs": None})
    elif tcfg.max_steps is None:
        steps = math.ceil(len(train_ex) * tcfg.num_epochs / (tcfg.batch_size * tcfg.grad_accum))
        tcfg = tcfg.model_copy(update={"max_steps": steps})
        print(f"{tcfg.num_epochs} epoch → {steps} step (실효 배치 {tcfg.batch_size * tcfg.grad_accum})")

    model_kwargs = {"dtype": dtype}
    if cfg.model.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.model.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_kwargs).to(device)
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

    # 이어하기: 체크포인트가 있으면 adapter·optimizer·데이터 위치를 되돌리고 그 다음 step부터 시작
    ckpt_root = out_dir / "checkpoints"
    identity = resume_identity(cfg)
    identity["train"]["max_steps"] = tcfg.max_steps  # --max-steps나 num_epochs로 정해진 실제 값 (lr 스케줄이 달라짐)
    stream = BatchStream(len(train_ex), tcfg.batch_size, cfg.runtime.seed)
    log_path = out_dir / "train_log.jsonl"
    start_step, prior_time, peak_mem = 0, 0.0, 0.0
    resume_from = latest_checkpoint(ckpt_root)
    if resume_from is not None:
        state = load_checkpoint(resume_from, model, optimizer, stream)
        if state["identity"] != identity:
            raise SystemExit(f"{resume_from}는 다른 설정으로 만든 체크포인트입니다. output_dir을 바꾸거나 지우세요.")
        start_step, prior_time, peak_mem = state["step"], state["train_time"], state["peak_mem"]
        log = trim_log(log_path, start_step)
        print(f"이어하기: {resume_from.name} (step {start_step}/{tcfg.max_steps}, epoch {stream.epoch})", flush=True)
    else:
        log = []
        log_path.write_text("", encoding="utf-8")

    def record(entry: dict) -> None:
        # 한 줄씩 바로 열고 닫아 쓰므로 접속이 끊겨도 그때까지의 기록이 파일에 남는다
        log.append(entry)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if start_step == 0:
        # 학습 전 val loss = 기준선. 이후 값과 비교해야 "내려갔다"를 말할 수 있다.
        start = time.time()
        val0 = evaluate(model, val_ex, tcfg.batch_size, pad_id, device)
        record({"step": 0, "split": "val", "loss": val0, "elapsed": time.time() - start})
        print(f"step    0 | val loss {val0:.4f}", flush=True)
    else:
        val0 = next(e["loss"] for e in log if e["step"] == 0 and e["split"] == "val")

    def next_batch() -> dict:
        return collate([train_ex[j] for j in stream.next_indices()], pad_id, PAD_MULTIPLE)

    window: list[float] = []
    session_start = last_save = time.time()
    model.train()

    def checkpoint(done: int) -> None:
        nonlocal last_save
        elapsed_total = prior_time + time.time() - session_start
        extra = {"identity": identity, "train_time": elapsed_total, "peak_mem": peak_mem}
        path = save_checkpoint(ckpt_root, done, model, optimizer, stream, extra, tcfg.keep_checkpoints)
        last_save = time.time()
        print(f"체크포인트 저장: {path}", flush=True)

    for step in range(start_step, tcfg.max_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, tcfg)

        # gradient accumulation: 작은 배치를 grad_accum번 forward/backward 해서 grad를 모은 뒤 한 번만 업데이트.
        # 메모리는 batch_size만큼만 쓰면서 실효 배치는 batch_size × grad_accum이 된다.
        step_loss = 0.0
        for _ in range(tcfg.grad_accum):
            batch = {k: v.to(device) for k, v in next_batch().items()}
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
            elapsed = prior_time + time.time() - session_start
            mem = peak_memory_gb(device)
            peak_mem = max(peak_mem, mem or 0.0)
            avg = sum(window) / len(window)  # 마지막 log 이후 step들의 평균 (한 step 값은 들쭉날쭉하다)
            record({"step": done, "split": "train", "loss": avg, "lr": lr_at(step, tcfg), "elapsed": elapsed})
            # 남은 시간은 이번 실행(이어하기 이후)의 속도로 추정한다
            sec_per_step = (time.time() - session_start) / (done - start_step)
            eta_min = sec_per_step * (tcfg.max_steps - done) / 60
            print(
                f"step {done:4d}/{tcfg.max_steps} | train loss {avg:.4f} | lr {lr_at(step, tcfg):.2e} | "
                f"{sec_per_step:.2f}s/step | 남은 시간 ~{eta_min:.0f}분",
                flush=True,
            )
            window = []
        if done % tcfg.eval_every == 0 or done == tcfg.max_steps:
            v = evaluate(model, val_ex, tcfg.batch_size, pad_id, device)
            record({"step": done, "split": "val", "loss": v, "elapsed": prior_time + time.time() - session_start})
            print(f"step {done:4d} | val loss {v:.4f}", flush=True)

        # 마지막 step은 아래에서 adapter를 저장하므로 체크포인트를 따로 만들지 않는다
        due_steps = tcfg.save_every_steps and done % tcfg.save_every_steps == 0
        due_time = tcfg.save_every_minutes and time.time() - last_save >= tcfg.save_every_minutes * 60
        if (due_steps or due_time) and done < tcfg.max_steps:
            checkpoint(done)

    train_time = prior_time + time.time() - session_start
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
        "resumed_from_step": start_step or None,
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
