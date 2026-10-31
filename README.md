# CarMind

자동차 도메인 특화 멀티모달 LLM 및 정비/매뉴얼 Agent 프로젝트.

계기판 경고등·엔진룸 사진과 질문을 입력하면, 도메인 학습된 VLM이 이미지를 이해하고
Agent가 매뉴얼 검색·DTC 조회 등 도구를 활용해 진단과 조치 방법을 안내한다.

현재는 텍스트 매뉴얼 QA까지 진행했다. 매뉴얼로 합성 QA를 만들어 Qwen2.5를 LoRA 학습하고, 학습에 쓰지 않은 매뉴얼로 평가한다.

```
[데이터 파이프라인] → [학습(SFT/DPO)] → [자동 평가] → [서빙(vLLM/SGLang)] → [Agent]
```

## 로드맵

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | 데이터셋 구축 및 전처리 자동화 (매뉴얼 파싱, 합성 QA, 필터링, 판정) | ✅ |
| 2 | 도메인 학습 (SFT/LoRA 완료, DPO, VLM fine-tuning 예정) | 🟡 |
| 3 | 자동 평가 체계 (hold-out 평가셋, LLM-as-a-Judge, 검색 평가 완료) | 🟡 |
| 4 | 서빙 최적화 (prefix caching, 양자화, 부하 테스트, prefix-aware routing) | ⬜ |
| 5 | 도메인 Agent (LangGraph, MCP 도구, agent 평가) | ⬜ |

## 결과

학습에 쓰지 않은 매뉴얼 1권, 1,316문항으로 평가했다. `closed`는 질문만, `rag`는 질문과 정답 문서를 입력한다.

| 모델 | 입력 | score | 정답률 | 환각 |
|---|---|---|---|---|
| Qwen2.5-7B | closed | 0.280 | 3.0% | 99.9% |
| Qwen2.5-7B + LoRA | closed | 0.469 | 22.9% | 57.4% |
| Qwen2.5-7B | rag | 0.901 | 80.7% | 10.6% |
| Qwen2.5-7B + LoRA | rag | 0.949 | 90.0% | 0.4% |

- 성능 향상의 대부분은 base 모델에서 나왔고, LoRA 학습의 가장 큰 효과는 환각 감소다.
- rag 결과는 정답 문서를 직접 넣은 값이라 실제 서비스 성능의 상한이다.

검색기는 같은 평가셋에서 매뉴얼 1권 안의 정답 청크 찾기로 측정했다.

| 방법 | Recall@1 | Recall@10 | MRR |
|---|---|---|---|
| BM25 | 0.657 | 0.966 | 0.770 |
| bge-m3 | 0.627 | 0.942 | 0.741 |
| hybrid (RRF) | 0.694 | 0.973 | 0.798 |

## LLM 제공자 선택

QA 생성·판정·채점은 기본값이 OpenAI이다. 모델 이름이 `claude-`로 시작하면 Claude로 호출한다 (`.env`에 `ANTHROPIC_API_KEY` 필요).

```bash
uv run python scripts/synth_qa.py --model claude-sonnet-5-5 --limit 20
uv run python scripts/judge_qa.py --judge claude-haiku-5-5 --limit 20
uv run python scripts/eval_grade.py outputs/eval/<run> --judge claude-haiku-5-5 --dry-run
```

채점 결과는 judge 이름별 파일(`grades_<judge>_<버전>.jsonl`)에 쌓인다. 판정자가 다르면 점수 기준도 달라지므로 같은 judge의 리포트끼리만 비교한다.

## 디렉토리 구조

```
carmind/
├── configs/            # 단계별 설정 (data / env / training / eval)
├── data/               # raw → interim → processed, eval (git 미추적)
├── pipelines/data/     # 전처리 자동화 플로우 (Prefect)
├── src/carmind/
│   ├── data/           # 파싱, 합성, 필터링, 판정
│   ├── training/       # SFT 학습
│   ├── retrieval/      # BM25, dense, RRF 검색
│   ├── evaluation/     # 답변 생성, 채점, 메트릭, 리포트
│   ├── serving/        # vLLM/SGLang 런처, router
│   └── agent/          # LangGraph agent
│       └── tools/      # MCP 도구 서버
├── scripts/            # 실행 스크립트
│   └── runpod/         # GPU 서버 학습·평가
├── benchmarks/         # 서빙 벤치마크 결과
├── notebooks/          # 탐색·분석
├── reports/            # 실험 리포트, ablation 표
├── docker/
└── tests/
```
