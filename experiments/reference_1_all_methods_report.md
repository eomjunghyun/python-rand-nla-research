# Reference 1 Experiments 7.1-8.2 정리 보고서

## 2026-05-19 실행 기준

Reference 1의 Section 7.1부터 8.2까지 Python 산출물을 아래 다섯 방법 기준으로 정리했다.

- `Non-random`
- `Random Sampling`
- `Random Projection`
- `CountSketch`
- `SIGN Bidirectional`

모든 정리 실행은 반복 수 `reps=20`, seed `2026` 기준이다. Section 8.1의 Random Sampling은 논문 설정에 맞춰 `p=0.7`, `p=0.8` 두 변형을 모두 유지한다.

## Python 산출물

| Section | 실험 | 주요 결과 위치 | 상태 |
|---|---|---|---|
| 7.1 | Experiment 1-4 | `experiments/reference_1_section7_1/results/all_methods_5way/` | 5방법 raw/summary/plot/timing/report 완료 |
| 7.2 | Model 1-3 | `experiments/reference_1_section7_2/results/exp72_models123_paper_aligned_live/` | 5방법 raw/summary/plot/timing 완료 |
| 7.2 | Model 4-6 | `experiments/reference_1_section7_2/results/exp72_models456_paper_aligned_live/` | 5방법 raw/summary/plot/timing 완료 |
| 8.1 | Email rank 42 | `experiments/reference_1_section8_1/results/all_methods_5way/email_rank42/` | 5방법 accuracy/pairwise ARI 완료 |
| 8.1 | Email rank 30 | `experiments/reference_1_section8_1/results/all_methods_5way/email_rank30/` | 5방법 accuracy/pairwise ARI 완료 |
| 8.1 | Political/statisticians paper rank | `experiments/reference_1_section8_1/results/all_methods_5way/remaining_paper_rank/` | 5방법 accuracy/pairwise ARI 완료 |
| 8.1 | Political/statisticians rank 5 | `experiments/reference_1_section8_1/results/all_methods_5way/remaining_rank5/` | 5방법 accuracy/pairwise ARI 완료 |
| 8.2 | Table 4 baseline | `experiments/reference_1_section8_2/results/exp8_2_table4_paper_aligned/` | RP/CountSketch/RS/Non-random timing 완료 |
| 8.2 | Table 4 + SIGN Bidirectional | `experiments/reference_1_section8_2/results/sign_section8_2_wang2025/` | 5방법 timing/report 완료 |

## 핵심 보고서

- Section 7.1: `experiments/reference_1_section7_1/results/all_methods_5way/section7_1_five_method_report.md`
- Section 7.2 Python: `experiments/reference_1_section7_2/section7_2_python_experiment_report.md`
- Section 8.1 Python: `experiments/reference_1_section8_1/section8_1_experiment_report.md`
- Section 8.2 Python: `experiments/reference_1_section8_2/section8_2_experiment_report.md`
- Section 8.2 SIGN Bidirectional: `experiments/reference_1_section8_2/results/sign_section8_2_wang2025/sign_section8_2_report.md`

## 8.2 최종 Table 4-style 요약

`experiments/reference_1_section8_2/results/sign_section8_2_wang2025/table4_with_sign_median_time.md` 기준:

| Networks | Random projection | CountSketch | SIGN Bidirectional | Random sampling | Non-random | SIGN / RP | SIGN / Non-random |
|---|---:|---:|---:|---:|---:|---:|---:|
| DBLP | 0.618 | 0.635 | 1.253 | 3.119(0.391) | 0.398 | 2.03x | 3.15x |
| Youtube | 5.517 | 5.498 | 10.265 | 11.660(1.819) | 1.446 | 1.86x | 7.10x |
| Internet | 4.172 | 4.121 | 9.447 | 12.229(1.684) | 1.822 | 2.26x | 5.19x |

Random Sampling의 괄호 밖 값은 sampling 포함 시간이고, 괄호 안 값은 sampling 제외 eigenvector computation 시간이다.

## MATLAB 비교 상태

현재 저장소에 MATLAB 구현이 있는 섹션은 7.2와 8.1이다. 두 MATLAB 구현 모두 5방법 기준 산출물이 존재하며, Python과 같은 파일 구조를 따른다.

| Section | MATLAB 위치 | 상태 |
|---|---|---|
| 7.2 | `experiments/reference_1_section7_2_matlab/` | 5방법 raw/summary/plot/report 존재 |
| 8.1 | `experiments/reference_1_section8_1_matlab/` | 5방법 raw/summary/table/pairwise ARI/report 존재 |
| 7.1 | 없음 | Python 5방법 결과 완료, MATLAB harness는 아직 없음 |
| 8.2 | 없음 | Python 5방법 timing 결과 완료, MATLAB harness는 아직 없음 |

MATLAB과 Python은 RNG, eigensolver, QR, KMeans 구현이 다르므로 절대 수치가 완전히 일치하지 않는다. 비교할 때는 동일 실험 정의와 출력 스키마를 기준으로 method별 경향을 비교한다.

## 검증 메모

- 수정된 Python 스크립트는 `python -m py_compile`로 문법 확인했다.
- Section 7.1, 7.2, 8.1, 8.2의 주요 raw CSV에서 5방법 method label이 존재하는지 확인했다.
- Section 8.2는 SNAP `com-dblp`, `com-youtube`, `as-skitter` gzip edge list를 사용해 전체 20회 반복을 완료했다.
