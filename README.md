# CarMind

자동차 도메인 특화 멀티모달 LLM 및 정비/매뉴얼 Agent 프로젝트.

계기판 경고등·엔진룸 사진과 질문을 입력하면, 도메인 학습된 VLM이 이미지를 이해하고
Agent가 매뉴얼 검색·DTC 조회 등 도구를 활용해 진단과 조치 방법을 안내한다.

```
[데이터 파이프라인] → [학습(SFT/DPO)] → [자동 평가] → [서빙(vLLM/SGLang)] → [Agent]
```

## 로드맵

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | 데이터셋 구축 및 전처리 자동화 (매뉴얼 파싱, 합성 QA, augmentation, dedup) | ⬜ |
| 2 | 도메인 학습 (SFT/LoRA, DPO, VLM fine-tuning, ablation) | ⬜ |
| 3 | 자동 평가 체계 (도메인 벤치마크, LLM-as-a-Judge 검증, forgetting 측정) | ⬜ |
| 4 | 서빙 최적화 (prefix caching, 양자화, 부하 테스트, prefix-aware routing) | ⬜ |
| 5 | 도메인 Agent (LangGraph, MCP 도구, agent 평가) | ⬜ |

## 디렉토리 구조

```
carmind/
├── configs/            # 단계별 설정 (data / training / eval / serving)
├── data/               # raw → interim → processed, eval (git 미추적)
├── pipelines/data/     # 전처리 자동화 플로우 (Prefect)
├── src/carmind/
│   ├── data/           # 파싱, 합성, augmentation, 필터링
│   ├── training/       # SFT, DPO, VLM 학습
│   ├── evaluation/     # 벤치마크, judge, 메트릭
│   ├── serving/        # vLLM/SGLang 런처, router
│   └── agent/          # LangGraph agent
│       └── tools/      # MCP 도구 서버
├── scripts/            # 실행 스크립트
├── benchmarks/         # 서빙 벤치마크 결과
├── notebooks/          # 탐색·분석
├── reports/            # 실험 리포트, ablation 표
├── docker/
└── tests/
```
