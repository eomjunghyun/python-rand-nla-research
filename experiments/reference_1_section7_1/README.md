# Reference 1 Section 7.1 실험

이 폴더는 Reference 1 논문의 Section 7.1에 해당하는 Experiment 1~4를 노트북으로 재현한 실험 폴더다.

## 비교 방법

정리된 실행 스크립트는 모든 Experiment 1~4에 아래 다섯 방법을 동일하게 적용한다.

- `Non-random`
- `Random Sampling`
- `Random Projection`
- `CountSketch`
- `SIGN Bidirectional`

## 노트북 목록

- `exp1_live.ipynb`: 노드 수 `n` 변화 실험
- `exp2_live.ipynb`: `alpha_n` 변화 실험
- `exp3_live.ipynb`: community 수 `K` 변화 실험
- `exp4_live.ipynb`: `alpha_n = 2 / sqrt(n)` 조건에서의 `n` 변화 실험
- `run_all_methods_section7_1.py`: Experiment 1~4를 다섯 방법으로 한 번에 실행하는 정리 스크립트
- `run_sign_section7_1.py`: Wang et al. (2025) SIGN 방법론을 Section 7.1 Exp1~4에 추가 적용하는 실행 스크립트

각 노트북은 실험 실행, raw/summary CSV 저장, metric/runtime figure 저장, 단계별 timing breakdown plot 저장을 수행한다.

## 공통 코드

실험 생성, metric 계산, 시각화 보조 함수는 주로 `src/common.py`에 모여 있다.

## 결과 저장 위치

기본 출력 경로는 아래와 같다.

- `experiments/reference_1_section7_1/results/all_methods_5way/`
- `experiments/reference_1_section7_1/results/exp1_paper_aligned_live/`
- `experiments/reference_1_section7_1/results/exp2_section7_1_results/`
- `experiments/reference_1_section7_1/results/exp3_section7_1_results/`
- `experiments/reference_1_section7_1/results/exp4_section7_1_results/`
- `experiments/reference_1_section7_1/results/sign_section7_1_wang2025/`

정리된 5방법 보고서는 `results/all_methods_5way/section7_1_five_method_report.md`에 저장된다.

## 작성 규칙

README와 실험 설명은 기본적으로 한글로 작성한다. 함수명, 변수명, metric 이름처럼 코드와 직접 맞닿은 용어는 영어를 섞어 쓴다.
