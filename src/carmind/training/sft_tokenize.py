"""대화(messages) → 학습용 토큰 (input_ids, labels).

SFT에서 모델이 배워야 하는 것은 "질문을 보고 답을 쓰는 법"이다.
system/user 부분은 매번 주어지는 입력이므로 그 토큰까지 맞히도록 학습시키면
쓸데없이 질문 문장을 외우는 데 힘을 쓰게 된다. 그래서 assistant 답변 토큰만
loss에 들어가도록, 나머지 위치의 label은 -100으로 가린다
(PyTorch CrossEntropyLoss가 -100 위치는 계산에서 빼 준다).

모델이 다르면 대화를 이어 붙이는 형식(chat template)도 다르므로 직접 문자열을
만들지 않고 토크나이저의 apply_chat_template을 쓴다. 덕분에 설정에서 모델 이름만
바꿔도 같은 코드가 동작한다.
"""

import torch

IGNORE_INDEX = -100  # CrossEntropyLoss의 기본 ignore_index


def encode_example(
    messages: list[dict], tokenizer, max_seq_len: int, template_kwargs: dict | None = None
) -> dict[str, list[int]] | None:
    """마지막 assistant 답변만 loss 대상으로 표시한 input_ids / labels를 만든다.

    방법: "답변 직전까지의 프롬프트"(add_generation_prompt=True)와 "답변까지 포함한 전체"를
    각각 문자열로 렌더링한다. 프롬프트는 전체의 앞부분과 정확히 같으므로, 프롬프트 길이만큼의
    토큰을 -100으로 가리면 나머지(답변 + 끝 토큰 <|im_end|>)만 남는다.
    끝 토큰까지 학습해야 모델이 답을 마친 뒤 멈추는 법을 배운다.

    max_seq_len보다 길면 뒤쪽(답변 끝)을 자른다. 잘라서 답변 토큰이 하나도 남지 않으면
    배울 것이 없으므로 None을 돌려준다 (호출하는 쪽에서 버린다).

    template_kwargs: apply_chat_template에 넘길 옵션. Qwen3.5는 마지막 답변을 항상 빈 think 블록과 함께
    렌더링한다 ("<think>\\n\\n</think>\\n\\n답변"). 프롬프트도 enable_thinking=False로 만들어야 빈 블록까지
    프롬프트에 포함되어 가려지고, label에는 답변만 남는다. 생성할 때도 같은 프롬프트로 시작하므로 모델은
    빈 블록 뒤에 바로 답을 쓰는 법을 배운다.
    """
    if messages[-1]["role"] != "assistant":
        raise ValueError("마지막 메시지는 assistant 답변이어야 합니다.")

    kwargs = template_kwargs or {}
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **kwargs)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    if not full_text.startswith(prompt_text):
        # 일부 템플릿은 마지막 턴 여부에 따라 앞부분을 다르게 렌더링한다. 그런 모델이면 경계를 믿을 수 없다.
        raise ValueError("chat template의 프롬프트가 전체 대화의 앞부분과 일치하지 않습니다.")

    answer_text = full_text[len(prompt_text):]
    # enable_thinking을 안 넘기면 프롬프트가 "<think>\n"에서 끝나 "</think>"가 조용히 학습 대상에 섞인다
    if "</think>" in answer_text and "</think>" not in messages[-1]["content"]:
        raise ValueError("label에 think 블록이 섞였습니다. model.chat_template_kwargs에 enable_thinking: false를 주세요.")

    # 문자열을 통째로 토큰화하면 경계에서 토큰이 합쳐질 수 있으므로 두 부분을 따로 토큰화해 이어 붙인다.
    # 특수 토큰(<|im_start|> 등)은 템플릿 문자열 안에 이미 들어 있으므로 add_special_tokens=False.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:max_seq_len]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_seq_len]
    if all(x == IGNORE_INDEX for x in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def collate(batch: list[dict[str, list[int]]], pad_token_id: int, pad_multiple: int = 1) -> dict[str, torch.Tensor]:
    """길이가 다른 샘플들을 배치 안에서 가장 긴 길이에 맞춰 오른쪽 패딩한다.

    전체를 max_seq_len으로 고정 패딩하면 대부분(중앙값 ~150토큰)이 패딩이라 연산이 낭비된다.
    패딩 위치는 attention_mask=0으로 무시하고, label도 -100으로 loss에서 뺀다.

    pad_multiple: 길이를 이 값의 배수로 올린다. 배치마다 길이가 제각각이면 GPU 메모리 캐시가
    모양(shape)별로 따로 쌓여 점점 불어나는데(특히 MPS), 배수로 맞추면 모양의 가짓수가 줄어든다.
    """
    width = max(len(x["input_ids"]) for x in batch)
    width = -(-width // pad_multiple) * pad_multiple  # 올림
    input_ids, labels, attention_mask = [], [], []
    for x in batch:
        pad = width - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad)
        labels.append(x["labels"] + [IGNORE_INDEX] * pad)
        attention_mask.append([1] * len(x["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }
