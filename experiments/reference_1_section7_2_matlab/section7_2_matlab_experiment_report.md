# Reference 1 Section 7.2 MATLAB 실험 보고서

이 보고서는 `experiments/reference_1_section7_2`의 Python 실험을 MATLAB 코드로 다시 구현한 뒤, MATLAB에서 직접 실행해 얻은 결과를 정리한다. 실험은 Reference 1 논문 Section 7.2의 Model 1-6에 해당하며, 세 방법의 오차와 실행 시간을 비교한다.

## 1. 실행 요약

| 항목 | 값 |
|---|---|
| 구현 위치 | `experiments/reference_1_section7_2_matlab/` |
| MATLAB 실행 파일 | `/Applications/MATLAB_R2026a.app/bin/matlab` |
| MATLAB 버전 | R2026a Update 1 |
| 반복 횟수 | 20 |
| seed | 2026 |
| n 값 | 200, 400, 600, 800, 1000, 1200 |
| Model 1-3 출력 row 수 | raw 1080개, summary 54개 |
| Model 4-6 출력 row 수 | raw 1080개, summary 54개 |

실행 명령은 다음과 같다.

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/reference_1_section7_2_matlab'); run_all_sec72_matlab('reps',20,'seed',2026,'no_progress',true)"
```

## 2. 실험 방법

비교한 방법은 세 가지다.

| 방법 | 설명 |
|---|---|
| Non-random | 원래 adjacency matrix에서 leading eigenvectors를 직접 구한 뒤 k-means를 수행하는 기준 방법 |
| Random Projection | Gaussian random projection과 power iteration으로 spectral subspace를 근사한 뒤 k-means를 수행 |
| Random Sampling | edge를 확률 `p=0.7`로 샘플링하고 `1/p`로 rescale한 matrix에서 spectral clustering 수행 |

공통 파라미터는 `K=3`, `q=2`, `r=10`, `p=0.7`이다. Model 3과 Model 6은 rank-deficient 설정이므로 `K_prime=2`를 사용했고, 나머지는 `K_prime=3`을 사용했다.

평가 지표는 Python 실험과 같은 형식으로 맞췄다.

| 지표 | 의미 |
|---|---|
| `error_P` | 추정 행렬과 true probability matrix의 spectral norm error |
| `error_Theta` | true community membership과 추정 membership 사이의 normalized label error |
| `error_B` | block probability matrix 추정의 max absolute error |
| `time_sec` | 방법별 알고리즘 실행 시간 |

## 3. 산출물

Model 1-3 결과:

| 파일 | 설명 |
|---|---|
| `results/exp72_models123_paper_aligned_live/sec72_models123_raw_per_rep.csv` | 반복별 raw 결과 |
| `results/exp72_models123_paper_aligned_live/sec72_models123_summary_mean_std.csv` | 평균/표준편차 summary |
| `results/exp72_models123_paper_aligned_live/sec72_models123_metrics_figure5_like.png` | Figure 5 형식의 metric plot |
| `results/exp72_models123_paper_aligned_live/sec72_models123_runtime.png` | runtime plot |

Model 4-6 결과:

| 파일 | 설명 |
|---|---|
| `results/exp72_models456_paper_aligned_live/sec72_models456_raw_per_rep.csv` | 반복별 raw 결과 |
| `results/exp72_models456_paper_aligned_live/sec72_models456_summary_mean_std.csv` | 평균/표준편차 summary |
| `results/exp72_models456_paper_aligned_live/sec72_models456_metrics_figure6_like.png` | Figure 6 형식의 metric plot |
| `results/exp72_models456_paper_aligned_live/sec72_models456_runtime.png` | runtime plot |

## 4. 전체 그림

### 4.1 Model 1-3 metric

![Model 1-3 MATLAB metrics](results/exp72_models123_paper_aligned_live/sec72_models123_metrics_figure5_like.png)

### 4.2 Model 1-3 runtime

![Model 1-3 MATLAB runtime](results/exp72_models123_paper_aligned_live/sec72_models123_runtime.png)

### 4.3 Model 4-6 metric

![Model 4-6 MATLAB metrics](results/exp72_models456_paper_aligned_live/sec72_models456_metrics_figure6_like.png)

### 4.4 Model 4-6 runtime

![Model 4-6 MATLAB runtime](results/exp72_models456_paper_aligned_live/sec72_models456_runtime.png)

## 5. 대표 수치: n = 1200

아래 표는 가장 큰 크기인 `n=1200`에서의 평균 결과다. 표준편차는 CSV 파일에 함께 저장되어 있으며, 여기서는 가독성을 위해 평균만 표시한다.

### 5.1 Model 1-3

| Model | Method | error_P | error_Theta | error_B | time_sec |
|---:|---|---:|---:|---:|---:|
| 1 | Non-random | 21.598 | 0.0000 | 0.0023 | 0.0063 |
| 1 | Random Projection | 12.485 | 0.0000 | 0.0023 | 0.0054 |
| 1 | Random Sampling | 15.657 | 0.0000 | 0.0031 | 0.0165 |
| 2 | Non-random | 23.354 | 0.0003 | 0.0027 | 0.0074 |
| 2 | Random Projection | 13.383 | 0.0003 | 0.0098 | 0.0056 |
| 2 | Random Sampling | 16.570 | 0.0009 | 0.0056 | 0.0178 |
| 3 | Non-random | 27.982 | 0.0000 | 0.0029 | 0.0070 |
| 3 | Random Projection | 16.974 | 0.0000 | 0.0030 | 0.0055 |
| 3 | Random Sampling | 26.967 | 0.0036 | 0.0039 | 0.0181 |

Model 1-3에서는 `error_P` 기준으로 모든 `n`과 모든 model에서 Random Projection이 가장 낮았다. `error_Theta`는 큰 `n`에서 거의 0으로 수렴했다. 특히 `n=1200`에서는 Model 1과 Model 3에서 Non-random과 Random Projection이 모두 0을 기록했고, Model 2에서도 두 방법이 0.00025로 거의 같은 수준이었다.

`error_B`는 Non-random이 가장 낮거나 Random Projection과 거의 같은 수준이다. 즉 Random Projection은 probability matrix `P`의 spectral norm 근사에서는 확실히 유리했지만, block matrix `B` 추정에서는 Non-random의 안정성이 조금 더 좋게 나타났다.

실행 시간은 `n=1200` 기준 Random Projection이 세 모델 모두에서 가장 짧았다. Random Sampling은 edge sampling과 sampled matrix eigensolver 비용 때문에 가장 느렸다.

### 5.2 Model 4-6

| Model | Method | error_P | error_Theta | error_B | time_sec |
|---:|---|---:|---:|---:|---:|
| 4 | Non-random | 15.057 | 0.0598 | 0.4757 | 0.0086 |
| 4 | Random Projection | 11.003 | 0.0620 | 0.4762 | 0.0069 |
| 4 | Random Sampling | 12.978 | 0.1310 | 0.4775 | 0.0196 |
| 5 | Non-random | 14.305 | 0.2049 | 0.4826 | 0.0095 |
| 5 | Random Projection | 10.639 | 0.2089 | 0.4829 | 0.0075 |
| 5 | Random Sampling | 12.620 | 0.3115 | 0.4834 | 0.0208 |
| 6 | Non-random | 18.637 | 1.4196 | 0.9186 | 0.0160 |
| 6 | Random Projection | 13.655 | 1.5628 | 0.9300 | 0.0130 |
| 6 | Random Sampling | 20.872 | 1.5821 | 0.9249 | 0.0264 |

Model 4-6도 `error_P` 기준으로 모든 `n`과 모든 model에서 Random Projection이 가장 낮았다. degree-corrected 구조가 들어간 Model 4-6에서는 Model 1-3보다 `error_Theta`가 높게 남아 있다. 특히 Model 6은 rank-deficient 구조와 degree correction이 동시에 들어가 `error_Theta`가 크게 나타났다.

Model 4와 Model 5에서는 Non-random과 Random Projection의 `error_Theta`가 비슷하고, Random Sampling은 더 높다. Model 6에서는 세 방법 모두 membership recovery가 어렵지만, `error_P`만 놓고 보면 Random Projection이 여전히 가장 좋다.

실행 시간은 Model 4-6에서도 Random Projection이 가장 짧고, Random Sampling이 가장 길었다. 이 결과는 MATLAB 구현에서도 random projection의 비용 이점이 유지된다는 것을 보여준다.

## 6. 핵심 관찰

1. `error_P`에서는 Random Projection이 가장 강했다.

   MATLAB 결과에서 Model 1-6의 모든 `n` 값에 대해 Random Projection이 가장 낮은 평균 `error_P`를 기록했다. 이는 randomized subspace approximation이 원래 adjacency/probability 구조의 leading spectral information을 잘 잡는다는 해석과 맞다.

2. `error_Theta`는 모델 구조에 따라 난이도가 크게 달라졌다.

   Model 1-3에서는 큰 `n`에서 membership error가 거의 사라졌다. 반면 Model 4-6에서는 degree correction 때문에 `Theta` 복원이 더 어렵고, Model 6에서는 rank-deficient 구조까지 겹쳐 error가 크게 유지되었다.

3. `error_B`는 Non-random이 약간 더 안정적이었다.

   `B` 추정은 최종 label alignment와 block 평균에 민감하다. Random Projection이 `P`의 spectral norm error는 낮췄지만, `B` 추정에서는 Non-random과 비슷하거나 약간 큰 error를 보였다.

4. 실행 시간은 Random Projection이 가장 유리했다.

   일부 작은 `n=200`에서는 Non-random과 큰 차이가 없지만, 대부분의 설정에서 Random Projection이 가장 빠르다. Random Sampling은 sampled matrix를 만들고 다시 eigensolver를 적용해야 해서 평균 실행 시간이 가장 길었다.

## 7. Python 결과와의 관계

이 MATLAB 구현은 Python `src.common`을 호출하지 않고 같은 실험 절차를 MATLAB 코드로 다시 작성한 것이다. 출력 파일명, CSV column, plot 형식은 Python 실험과 맞췄다.

다만 MATLAB과 Python/NumPy는 다음 부분이 다르므로 수치가 완전히 같지는 않다.

- random number generator
- eigen solver 구현
- k-means 초기화와 반복 세부 구현
- 부동소수점 연산 순서

따라서 이 보고서의 수치는 Python 결과의 byte-for-byte 재현이 아니라, 같은 실험 정의를 MATLAB로 재구현했을 때의 MATLAB 실행 결과로 읽어야 한다.

## 8. 결론

MATLAB 재구현 결과에서도 Section 7.2의 큰 결론은 분명하다. Random Projection은 모든 모델과 `n` 값에서 `error_P`가 가장 낮고, 실행 시간도 가장 짧거나 매우 짧다. Non-random은 `B` 추정과 일부 `Theta` 복원에서 안정적이지만, `P` error와 runtime 측면에서는 Random Projection이 우세하다. Random Sampling은 세 방법 중 가장 불안정하고 느린 편으로 나타났다.

결론적으로, 이 MATLAB 결과는 random projection 기반 spectral clustering이 Section 7.2 synthetic model들에서 정확도와 속도 사이의 가장 좋은 균형을 제공한다는 점을 뒷받침한다.
