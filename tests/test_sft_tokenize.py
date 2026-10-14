from pathlib import Path

import pytest

from carmind.training.sft_tokenize import IGNORE_INDEX, collate, encode_example
from carmind.training.sft_train import TrainConfig, load_sft_config, lr_at


class FakeTokenizer:
    """모델 다운로드 없이 테스트하기 위한 가짜 토크나이저.

    chat template은 Qwen과 같은 모양(<|im_start|>역할\\n내용<|im_end|>\\n)으로 흉내 내고,
    토큰화는 글자 하나 = 토큰 하나(유니코드 코드값)로 단순화했다.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


def decode(ids):
    return "".join(chr(i) for i in ids)


MESSAGES = [
    {"role": "system", "content": "정비 도우미"},
    {"role": "user", "content": "타이어 공기압은?"},
    {"role": "assistant", "content": "260 kPa입니다."},
]


def test_only_assistant_tokens_are_learned():
    ex = encode_example(MESSAGES, FakeTokenizer(), max_seq_len=1000)
    assert len(ex["input_ids"]) == len(ex["labels"])
    learned = [t for t in ex["labels"] if t != IGNORE_INDEX]
    # 답변과 끝 토큰만 loss 대상. system/user/역할 표시는 모두 가려져야 한다
    assert decode(learned) == "260 kPa입니다.<|im_end|>\n"
    n_masked = ex["labels"].index(learned[0])
    assert decode(ex["input_ids"][:n_masked]).endswith("<|im_start|>assistant\n")
    assert all(t == IGNORE_INDEX for t in ex["labels"][:n_masked])
    # 가리지 않은 부분의 label은 input_ids와 같다 (shift는 모델 내부에서 한다)
    assert ex["labels"][n_masked:] == ex["input_ids"][n_masked:]


def test_truncation():
    tok = FakeTokenizer()
    prompt_len = len(tok.apply_chat_template(MESSAGES[:-1], add_generation_prompt=True))
    ex = encode_example(MESSAGES, tok, max_seq_len=prompt_len + 3)
    assert len(ex["input_ids"]) == prompt_len + 3
    assert decode([t for t in ex["labels"] if t != IGNORE_INDEX]) == "260"
    # 프롬프트만으로 꽉 차서 답변이 하나도 안 남으면 버린다
    assert encode_example(MESSAGES, tok, max_seq_len=prompt_len) is None


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        encode_example(MESSAGES[:-1], FakeTokenizer(), 1000)

    class OddTemplate(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return ("PROMPT:" if add_generation_prompt else "FULL:") + super().apply_chat_template(messages)

    with pytest.raises(ValueError):
        encode_example(MESSAGES, OddTemplate(), 1000)


def test_collate_pads_and_masks():
    batch = [{"input_ids": [5, 6, 7], "labels": [IGNORE_INDEX, 6, 7]}, {"input_ids": [8], "labels": [8]}]
    out = collate(batch, pad_token_id=0)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert out["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert collate(batch, pad_token_id=0, pad_multiple=4)["input_ids"].shape == (2, 4)


def test_lr_schedule():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=100)
    assert lr_at(0, cfg) == pytest.approx(1e-4)
    assert lr_at(9, cfg) == pytest.approx(1e-3)  # warmup 끝에서 최대
    assert lr_at(55, cfg) == pytest.approx(5e-4)  # cosine 중간
    assert lr_at(99, cfg) < 1e-5


def test_smoke_config_loads():
    cfg = load_sft_config(Path(__file__).resolve().parents[1] / "configs/training/sft_smoke.yaml")
    assert cfg.train.max_steps > 0 and cfg.lora.target_modules
