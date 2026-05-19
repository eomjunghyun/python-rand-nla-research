# Reference 1 Section 7.1 Five-Method 실험 보고서

## 목적

Section 7.1의 Exp1~Exp4를 동일한 실행 경로에서 다시 돌려 `Non-random`, `Random Sampling`, `Random Projection`, `CountSketch`, `SIGN Bidirectional` 다섯 방법을 모두 비교했다. 모든 raw 결과에는 `time_sec`와 세부 timing breakdown이 포함된다.

## 설정

```json
{
  "exp1": {
    "n_values": [
      200,
      400,
      600,
      800,
      1000,
      1200
    ],
    "K": 3,
    "K_prime": 3,
    "alpha_n": 0.2,
    "lam": 0.5,
    "q": 2,
    "r": 10,
    "p": 0.7,
    "reps": 20,
    "seed": 2026
  },
  "exp2": {
    "alpha_values": [
      0.05,
      0.1,
      0.15,
      0.2
    ],
    "n": 1152,
    "K": 3,
    "K_prime": 3,
    "lam": 0.5,
    "q": 2,
    "r": 10,
    "p": 0.7,
    "reps": 20,
    "seed": 2026
  },
  "exp3": {
    "K_values": [
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    "n": 1152,
    "alpha_n": 0.2,
    "lam": 0.5,
    "q": 2,
    "r": 10,
    "p": 0.7,
    "reps": 20,
    "seed": 2026
  },
  "exp4": {
    "n_values": [
      200,
      400,
      600,
      800,
      1000,
      1200
    ],
    "K": 2,
    "K_prime": 2,
    "lam": 0.5,
    "q": 2,
    "r": 10,
    "p": 0.7,
    "reps": 20,
    "seed": 2026
  }
}
```

## 방법론 커버리지

| experiment | method | present |
| --- | --- | --- |
| exp1 | Random Projection | yes |
| exp1 | Random Sampling | yes |
| exp1 | Non-random | yes |
| exp1 | CountSketch | yes |
| exp1 | SIGN Bidirectional | yes |
| exp2 | Random Projection | yes |
| exp2 | Random Sampling | yes |
| exp2 | Non-random | yes |
| exp2 | CountSketch | yes |
| exp2 | SIGN Bidirectional | yes |
| exp3 | Random Projection | yes |
| exp3 | Random Sampling | yes |
| exp3 | Non-random | yes |
| exp3 | CountSketch | yes |
| exp3 | SIGN Bidirectional | yes |
| exp4 | Random Projection | yes |
| exp4 | Random Sampling | yes |
| exp4 | Non-random | yes |
| exp4 | CountSketch | yes |
| exp4 | SIGN Bidirectional | yes |

## 마지막 grid point 요약

각 experiment에서 가장 큰 x축 값의 평균 결과다.

| endpoint | method | error_P_mean | error_Theta_mean | error_B_mean | time_mean | experiment |
| --- | --- | --- | --- | --- | --- | --- |
| 1200 | Random Projection | 14.66 | 0.00125 | 0.004109 | 0.03025 | exp1 |
| 1200 | Random Sampling | 18.44 | 0.006625 | 0.003443 | 0.07656 | exp1 |
| 1200 | Non-random | 23.25 | 0.00075 | 0.002338 | 0.4768 | exp1 |
| 1200 | CountSketch | 14.68 | 0.0015 | 0.004413 | 0.03359 | exp1 |
| 1200 | SIGN Bidirectional | 15.36 | 0.01013 | 0.007037 | 0.04043 | exp1 |
| 0.2 | Random Projection | 14.43 | 0.001302 | 0.003504 | 0.02936 | exp2 |
| 0.2 | Random Sampling | 18.24 | 0.007292 | 0.003469 | 0.07384 | exp2 |
| 0.2 | Non-random | 22.76 | 0.0002604 | 0.002047 | 0.4652 | exp2 |
| 0.2 | CountSketch | 14.48 | 0.001042 | 0.003772 | 0.03251 | exp2 |
| 0.2 | SIGN Bidirectional | 15.14 | 0.009505 | 0.00648 | 0.04016 | exp2 |
| 8 | Random Projection | 15.01 | 5.579 | 0.07546 | 0.1243 | exp3 |
| 8 | Random Sampling | 25.55 | 4.234 | 0.0389 | 0.3021 | exp3 |
| 8 | Non-random | 21.28 | 1.497 | 0.02437 | 0.5 | exp3 |
| 8 | CountSketch | 14.86 | 5.593 | 0.07584 | 0.1274 | exp3 |
| 8 | SIGN Bidirectional | 14.82 | 5.914 | 0.07798 | 0.136 | exp3 |
| 1200 | Random Projection | 9.588 | 0.09017 | 0.003764 | 0.03302 | exp4 |
| 1200 | Random Sampling | 11.99 | 0.06942 | 0.001286 | 0.084 | exp4 |
| 1200 | Non-random | 14.12 | 0.02292 | 0.0007247 | 0.4792 | exp4 |
| 1200 | CountSketch | 9.423 | 0.08358 | 0.003342 | 0.0342 | exp4 |
| 1200 | SIGN Bidirectional | 9.907 | 0.2564 | 0.006327 | 0.04545 | exp4 |

## 산출물

| experiment | raw_csv | summary_csv | timing_csv | metrics_png | runtime_png |
| --- | --- | --- | --- | --- | --- |
| exp1 | exp1/exp1_raw_per_rep.csv | exp1/exp1_summary_mean_std.csv | exp1/exp1_timing_breakdown_summary.csv | exp1/figure1_like_metrics.png | exp1/figure1_like_runtime.png |
| exp2 | exp2/exp2_raw_per_rep.csv | exp2/exp2_summary_mean_std.csv | exp2/exp2_timing_breakdown_summary.csv | exp2/figure2_like_metrics.png | exp2/figure2_like_runtime.png |
| exp3 | exp3/exp3_raw_per_rep.csv | exp3/exp3_summary_mean_std.csv | exp3/exp3_timing_breakdown_summary.csv | exp3/figure3_like_metrics.png | exp3/figure3_like_runtime.png |
| exp4 | exp4/exp4_raw_per_rep.csv | exp4/exp4_summary_mean_std.csv | exp4/exp4_timing_breakdown_summary.csv | exp4/figure4_like_metrics.png | exp4/figure4_like_runtime.png |

## 그림

### exp1

![exp1 metrics](exp1/figure1_like_metrics.png)

![exp1 runtime](exp1/figure1_like_runtime.png)

### exp2

![exp2 metrics](exp2/figure2_like_metrics.png)

![exp2 runtime](exp2/figure2_like_runtime.png)

### exp3

![exp3 metrics](exp3/figure3_like_metrics.png)

![exp3 runtime](exp3/figure3_like_runtime.png)

### exp4

![exp4 metrics](exp4/figure4_like_metrics.png)

![exp4 runtime](exp4/figure4_like_runtime.png)

