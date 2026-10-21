"""평가 문항에 대한 모델 답변 생성 (base 모델 + 선택적 LoRA adapter).

모드
    closed: 질문만 준다. 모델이 매뉴얼 내용을 외워서 답해야 한다
    rag:    문항을 만든 원문 청크를 [참고 문서]로 함께 준다.
            정답이 들어 있는 문서를 그대로 넣어 주는 "oracle retrieval" 설정이라,
            검색이 완벽할 때의 상한(upper bound)을 재는 셈이다. 실제 검색기(retriever)를 붙인
            평가는 나중에 따로 하고, 그때 점수가 여기서 얼마나 떨어지는지가 검색 품질의 비용이 된다.

프롬프트는 학습 때와 같은 to_messages()로 만든다. 시스템 프롬프트나 [참고 문서] 형식이 한 글자라도
다르면 "학습한 형식으로 물었을 때"의 성능이 아니게 되기 때문이다.

디코딩은 greedy(do_sample=False)에 repetition_penalty=1.0으로 고정한다. 모델마다 generation_config의
기본값(temperature, repetition_penalty 등)이 달라서, 명시하지 않으면 모델 비교가 디코딩 설정 비교가 된다.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from carmind.evaluation.data import append_jsonl, read_jsonl
from carmind.training.sft_data import to_messages


@dataclass
class GenConfig:
    model: str
    adapter: str | None
    mode: str
    max_new_tokens: int = 256
    batch_size: int = 8
    repetition_penalty: float = 1.0
    device: str = "auto"
    dtype: str = "auto"
    # 실행마다 바뀌어도 결과에 영향이 없는 값은 아래 필드에 둔다 (meta 비교에서 제외)
    extra: dict = field(default_factory=dict)

    def identity(self) -> dict:
        """이어하기 때 '같은 실행인지' 판단할 필드. 이게 다르면 섞인 결과가 되므로 막는다.

        batch_size·device는 결과 텍스트를 거의 바꾸지 않지만(패딩 위치·연산 순서로 미세하게 달라질 수는 있음)
        메모리 사정에 따라 바꿔 이어 돌릴 수 있게 비교에서 뺀다.
        """
        d = asdict(self)
        for k in ("batch_size", "device", "extra"):
            d.pop(k)
        return d


def prompt_messages(qa: dict, mode: str, chunk_text: str | None) -> list[dict]:
    """학습 데이터와 같은 메시지에서 정답(assistant 턴)만 뺀다."""
    return to_messages(qa, mode, chunk_text)[:-1]


def check_resume(meta_path: Path, cfg: GenConfig) -> None:
    """기존 실행 폴더에 다른 설정으로 이어 쓰려 하면 중단한다."""
    if not meta_path.exists():
        return
    old = json.loads(meta_path.read_text(encoding="utf-8"))["generation"]
    new = cfg.identity()
    diff = {k: (old.get(k), v) for k, v in new.items() if old.get(k) != v}
    if diff:
        raise SystemExit(f"{meta_path.parent}는 다른 설정으로 만든 실행입니다 (기존, 새): {diff}. --run을 바꾸세요.")


def pending(questions: list[dict], out_path: Path) -> list[dict]:
    """아직 답이 없는 문항만 남긴다 (이어하기)."""
    done = {r["qa_id"] for r in read_jsonl(out_path)}
    return [q for q in questions if q["qa_id"] not in done]


def count_generated(token_ids: list[int], stop_ids: set[int], max_new_tokens: int) -> tuple[int, bool]:
    """새로 만든 토큰 열에서 (답변 토큰 수, 길이 제한으로 잘렸는지)를 구한다.

    배치 생성에서는 먼저 끝난 문장 뒤를 pad 토큰으로 채우므로, 첫 종료 토큰 위치가 답변 길이다.
    종료 토큰 없이 max_new_tokens를 다 쓰면 말이 끝나기 전에 잘린 것(truncated)이다.
    """
    for i, t in enumerate(token_ids):
        if t in stop_ids:
            return i, False
    return len(token_ids), len(token_ids) >= max_new_tokens


class Generator:
    def __init__(self, cfg: GenConfig):
        # 무거운 라이브러리는 실제 생성할 때만 불러온다 (테스트를 가볍게 유지)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from carmind.device import resolve_device, resolve_dtype

        self.cfg = cfg
        self.torch = torch
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype, self.device)
        # adapter 폴더에 학습 때 저장한 tokenizer가 있으면 그걸 쓴다 (chat template이 학습 때와 같도록)
        tok_src = cfg.adapter if cfg.adapter and (Path(cfg.adapter) / "tokenizer_config.json").exists() else cfg.model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        # 배치 생성은 왼쪽 패딩이어야 한다. 오른쪽에 pad가 있으면 모델이 pad 뒤에 이어서 생성하게 된다.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=self.dtype).to(self.device)
        if cfg.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, cfg.adapter).to(self.device)
            # LoRA 가중치를 원래 가중치에 합쳐 두면 매 토큰마다 adapter 연산을 따로 하지 않아 생성이 빨라진다
            model = model.merge_and_unload()
        model.eval()
        self.model = model

        gen_eos = model.generation_config.eos_token_id
        eos = gen_eos if isinstance(gen_eos, list) else [gen_eos]
        self.stop_ids = {t for t in [*eos, self.tokenizer.eos_token_id] if t is not None}

    def _generate_batch(self, batch_messages: list[list[dict]]) -> list[dict]:
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                # 모델 generation_config의 샘플링 값이 섞이지 않게 명시적으로 끈다
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=self.cfg.repetition_penalty,
                eos_token_id=sorted(self.stop_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:].tolist()
        results = []
        for ids in new_tokens:
            n, truncated = count_generated(ids, self.stop_ids, self.cfg.max_new_tokens)
            text = self.tokenizer.decode(ids[:n], skip_special_tokens=True).strip()
            results.append({"prediction": text, "n_tokens": n, "truncated": truncated})
        return results

    def run(self, questions: list[dict], chunks: dict[str, str], out_path: Path) -> dict:
        """questions를 배치로 생성해 한 줄씩 out_path에 덧붙인다. 속도 통계를 돌려준다."""
        mode = self.cfg.mode
        items = [(q, prompt_messages(q, mode, chunks.get(q["chunk_id"]) if mode == "rag" else None)) for q in questions]
        # 길이가 비슷한 프롬프트끼리 묶으면 패딩이 줄어 배치가 빨라진다 (결과 파일 순서는 상관없음)
        items.sort(key=lambda x: len(x[1][-1]["content"]))
        bs = self.cfg.batch_size
        start, total_tokens = time.time(), 0
        for i in range(0, len(items), bs):
            batch = items[i:i + bs]
            for (q, _), r in zip(batch, self._generate_batch([m for _, m in batch]), strict=True):
                total_tokens += r["n_tokens"]
                append_jsonl(
                    out_path,
                    {"qa_id": q["qa_id"], "type": q["type"], "question": q["question"], "reference": q["answer"], **r},
                )
            done = min(i + bs, len(items))
            elapsed = time.time() - start
            print(f"  {done}/{len(items)}  {elapsed:.0f}s  ({total_tokens / max(elapsed, 1e-9):.1f} tok/s)", flush=True)
        elapsed = time.time() - start
        return {
            "n": len(items),
            "seconds": round(elapsed, 1),
            "sec_per_question": round(elapsed / max(1, len(items)), 2),
            "generated_tokens": total_tokens,
            "tokens_per_sec": round(total_tokens / max(elapsed, 1e-9), 1),
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
