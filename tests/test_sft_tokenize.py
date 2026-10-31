import re
from pathlib import Path

import pytest

from carmind.training.sft_tokenize import IGNORE_INDEX, collate, encode_example
from carmind.training.sft_train import (
    TrainConfig,
    load_sft_config,
    lr_at,
    resume_identity,
)


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


class ThinkTokenizer(FakeTokenizer):
    """Qwen3.5-9B 템플릿 흉내: 마지막 답변은 항상 빈 think 블록과 함께, 프롬프트는 기본이 생각 모드."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None):
        text = ""
        for m in messages:
            think = "<think>\n\n</think>\n\n" if m["role"] == "assistant" else ""
            text += f"<|im_start|>{m['role']}\n{think}{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            text += "<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n"
        return text


def test_think_off_labels_only_answer():
    ex = encode_example(MESSAGES, ThinkTokenizer(), 1000, {"enable_thinking": False})
    learned = [t for t in ex["labels"] if t != IGNORE_INDEX]
    assert decode(learned) == "260 kPa입니다.<|im_end|>\n"
    n_masked = ex["labels"].index(learned[0])
    # 빈 think 블록은 프롬프트 쪽이라 입력에는 있지만 loss에서는 가려진다
    assert decode(ex["input_ids"][:n_masked]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_think_on_is_rejected():
    # 옵션을 빠뜨리면 "\n</think>\n\n"이 label에 섞이므로 학습 전에 멈춘다
    with pytest.raises(ValueError, match="think"):
        encode_example(MESSAGES, ThinkTokenizer(), 1000)


CONFIGS = Path(__file__).resolve().parents[1] / "configs/training"


@pytest.mark.parametrize("mode", ["closed", "rag"])
def test_9b_config_matches_7b_except_model(mode):
    a = load_sft_config(CONFIGS / f"sft_7b_{mode}.yaml").model_dump()
    b = load_sft_config(CONFIGS / f"sft_9b_{mode}.yaml").model_dump()
    assert b["model"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert b["model"]["name"] == "Qwen/Qwen3.5-9B"
    assert a["data"] == b["data"] and a["train"] == b["train"] and a["runtime"] == b["runtime"]
    assert {k: v for k, v in a["lora"].items() if k != "target_modules"} == {
        k: v for k, v in b["lora"].items() if k != "target_modules"
    }


def test_9b_lora_regex_targets_text_layers_only():
    pattern = load_sft_config(CONFIGS / "sft_9b_closed.yaml").lora.target_modules
    for name in (
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.3.self_attn.q_proj",
        "model.layers.31.mlp.down_proj",
        "model.language_model.layers.3.self_attn.o_proj",  # 멀티모달 클래스로 불러도 텍스트 층은 잡힌다
    ):
        assert re.fullmatch(pattern, name), name
    for name in (
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.visual.blocks.0.attn.qkv",
        "model.layers.0.linear_attn.conv1d",
        "lm_head",
        "model.embed_tokens",
    ):
        assert not re.fullmatch(pattern, name), name


def test_7b_resume_identity_has_no_new_keys():
    # 이 키가 identity에 들어가면 이미 저장된 7B 체크포인트가 "다른 설정"으로 거부된다
    identity = resume_identity(load_sft_config(CONFIGS / "sft_7b_closed.yaml"))
    assert set(identity["model"]) == {"name", "attn_implementation"}
