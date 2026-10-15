"""학습 전(base) / 후(base + LoRA adapter) 답변 비교.

uv run python scripts/compare_answers.py                       # val에서 고정된 5문항
uv run python scripts/compare_answers.py --n 10 --adapter outputs/<다른 실험>/adapter
uv run python scripts/compare_answers.py --config configs/training/sft_smoke_rag.yaml \
    --closed-adapter outputs/sft_smoke/adapter                 # rag: closed 학습 모델 답도 함께

같은 질문을 두 모델에 greedy로(무작위성 없이) 생성시켜, 차이가 학습 때문인지
샘플링 운 때문인지 헷갈리지 않게 한다. 결과: <adapter 상위 폴더>/compare.{md,json}

rag 설정이면 val 프롬프트에 참고 문서가 들어 있으므로 base와 학습 모델이 같은 문서를 보고 답한다.
그래야 "문서를 주기만 해도 되는지 / 학습이 더 보태는지"를 나눠 볼 수 있다.
--closed-adapter를 주면 같은 질문을 문서 없이(closed 형식) closed 학습 모델에도 물어 세 가지를 나란히 놓는다.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from carmind.device import resolve_device, resolve_dtype
from carmind.training.sft_data import SYSTEM_CLOSED
from carmind.training.sft_train import load_sft_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/training/sft_smoke.yaml"))
parser.add_argument("--adapter", type=Path, default=None, help="기본: <output_dir>/adapter")
parser.add_argument("--closed-adapter", type=Path, default=None, help="rag 비교 때 closed 형식으로도 답할 adapter")
parser.add_argument("--n", type=int, default=5)
parser.add_argument("--max-new-tokens", type=int, default=200)
args = parser.parse_args()

cfg = load_sft_config(args.config)
adapter = args.adapter or cfg.output_dir / "adapter"
device = resolve_device(cfg.runtime.device)
dtype = resolve_dtype(cfg.runtime.dtype, device)

# 학습에 쓰지 않은 val에서 seed로 고른다 → 실행할 때마다 같은 문항
val = [json.loads(line) for line in cfg.data.val.read_text(encoding="utf-8").splitlines()]
samples = random.Random(cfg.runtime.seed).sample(val, args.n)

tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
base = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=dtype).to(device)
# base 위에 adapter를 얹는다. disable_adapter()로 잠시 끄면 같은 객체로 원래 모델 답도 얻을 수 있어
# 모델을 두 번 올리지 않아도 된다 (메모리 절약).
model = PeftModel.from_pretrained(base, adapter).to(device)
if args.closed_adapter:
    # adapter를 이름을 붙여 하나 더 얹고 set_adapter로 바꿔 끼운다 (base는 여전히 한 번만 올림)
    model.load_adapter(args.closed_adapter, adapter_name="closed")
model.eval()


@torch.no_grad()
def generate(messages: list[dict]) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    # 출력에는 프롬프트가 앞에 붙어 있으므로 새로 생성된 부분만 잘라 낸다
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def split_user(content: str) -> tuple[str | None, str]:
    """rag user 턴("[참고 문서]\n…\n\n[질문]\n…")을 (문서, 질문)으로 나눈다. closed면 (None, 질문)."""
    if content.startswith("[참고 문서]\n") and "\n\n[질문]\n" in content:
        doc, question = content.removeprefix("[참고 문서]\n").rsplit("\n\n[질문]\n", 1)
        return doc, question
    return None, content


results = []
for i, row in enumerate(samples, 1):
    prompt_messages = row["messages"][:-1]  # system + user (정답은 빼고)
    context, question = split_user(row["messages"][-2]["content"])
    model.set_adapter("default")
    with model.disable_adapter():
        before = generate(prompt_messages)
    after = generate(prompt_messages)
    result = {
        "qa_id": row["qa_id"],
        "type": row["type"],
        "question": question,
        "context": context,
        "reference": row["messages"][-1]["content"],
        "base": before,
        "tuned": after,
    }
    if args.closed_adapter:
        model.set_adapter("closed")
        result["closed_tuned"] = generate([{"role": "system", "content": SYSTEM_CLOSED}, {"role": "user", "content": question}])
    results.append(result)
    print(f"[{i}/{len(samples)}] {row['qa_id']} 완료")

out_dir = adapter.parent
(out_dir / "compare.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
md = [f"# 학습 전후 답변 비교\n\nmodel: {cfg.model.name} / adapter: {adapter} / greedy, max_new_tokens={args.max_new_tokens}\n"]
for i, r in enumerate(results, 1):
    doc = f"**참고 문서** (base·학습 모델 모두 이 문서를 보고 답함)\n\n```text\n{r['context']}\n```\n\n" if r["context"] else ""
    closed = f"\n**closed 학습 모델 (문서 없음, {args.closed_adapter})**\n\n{r['closed_tuned']}\n" if "closed_tuned" in r else ""
    md.append(
        f"## {i}. [{r['type']}] {r['question']}\n\n{doc}"
        f"**정답**\n\n{r['reference']}\n\n**학습 전 (base)**\n\n{r['base']}\n\n**학습 후 (LoRA)**\n\n{r['tuned']}\n{closed}"
    )
(out_dir / "compare.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print(f"저장: {out_dir}/compare.md, compare.json")
