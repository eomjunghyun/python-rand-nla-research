# 균일 HSBM 실험

이 폴더는 고정 크기 `d`의 hyperedge만 생성하는 균일 HSBM(uniform HSBM) 실험을 모아둔다. 현재 생성 모델은 planted `d`-uniform HSBM을 따르도록 맞춰져 있다.

## 생성 모델

노드 집합은 `V={1,...,n}`이고, 각 노드는 하나의 community에 속한다. community label은 기본적으로 balanced하게 생성된다.

고정 hyperedge 크기 `d`에 대해 모든 후보 hyperedge `e={i_1,...,i_d}`는 다음 확률로 독립 생성된다.

```text
P(A_e = 1 | z) = p_{d,n},  z_{i_1}=...=z_{i_d} 인 경우
P(A_e = 1 | z) = q_{d,n},  그렇지 않은 경우
```

확률은 아래 sparse-regime 형태를 사용한다.

```text
p_{d,n} = a_d * rho_n / n^{d-1}
q_{d,n} = b_d * rho_n / n^{d-1}
```

여기서 `a_d > b_d > 0`, `rho_n > 0`이다.

## 파일 목록

- `EXP-20260426-004_uniform_hsbm_n_scaling_zhou_laplacian.ipynb`
  - `generate_planted_uniform_hsbm_instance`를 사용한다.
  - 현재 기본 설정은 `K=3`, `m=3`, `rho_n=4.0`이다.
  - `n=1000`부터 `10000`까지 1000 간격으로 변화시킨다.
  - Zhou normalized hypergraph Laplacian `Delta=I-Theta`를 만든 뒤, `Theta=I-Delta`의 가장 큰 고유값에 대응하는 고유벡터를 사용해 spectral clustering을 수행한다. 이는 `Delta`의 가장 작은 고유값에 대응하는 고유벡터를 사용하는 것과 같은 eigenspace를 사용한다.
  - misclassification rate, ARI, NMI, CPU time, wall-clock time, peak memory, 단계별 runtime을 기록한다.

- `EXP-20260427-001_uniform_hsbm_rho_n_sweep_zhou_theta.ipynb`
  - `rho_n`만 변화시키는 전용 실험 노트북이다.
  - 기본 설정은 `n=10000`, `K=3`, `m=3`, `a_in=36.0`, `b_out=4.0`, `reps=10`이며, 이 값들은 고정한다.
  - 기본 sweep 값은 `rho_n in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]`이다.
  - 각 `rho_n`에 대해 `p_in = a_in * rho_n / n^{m-1}`, `p_out = b_out * rho_n / n^{m-1}`를 사용한다.
  - Zhou operator `Theta`의 가장 큰 고유값에 대응하는 고유벡터를 사용해 spectral clustering을 수행한다.
  - misclassification rate, ARI, NMI, CPU time, wall-clock time, peak memory, 단계별 runtime을 기록한다.

- `EXP-20260427-002_uniform_hsbm_K_sweep_zhou_theta.ipynb`
  - `K`만 변화시키는 전용 실험 노트북이다.
  - 기본 설정은 `n=10000`, `m=3`, `a_in=36.0`, `b_out=4.0`, `rho_n=4.0`, `reps=10`, `kmeans_n_init=20`이며, 이 값들은 고정한다.
  - 기본 sweep 값은 `K in [2, 3, 4, 5, 6, 8, 10, 12]`이다.
  - `K`가 변하면 within-community 후보 hyperedge 비율과 평균 차수도 함께 변하므로, empirical degree와 expected degree를 함께 기록한다.
  - Zhou operator `Theta`의 가장 큰 고유값에 대응하는 고유벡터를 사용해 spectral clustering을 수행한다.
  - misclassification rate, ARI, NMI, CPU time, wall-clock time, peak memory, 단계별 runtime을 기록한다.

## 설계상 주의점

- `Theta=I-Delta`의 가장 큰 고유값 고유벡터를 쓰는 구현은 `Delta`의 가장 작은 고유값 고유벡터를 쓰는 구현과 같은 eigenspace를 사용한다.
- 오분류율은 예측 cluster label을 Hungarian matching으로 true label에 최적으로 정렬한 뒤 계산한다.
- `rho_n`은 평균 incidence degree를 직접 키우는 sparsity/density 파라미터이다. 현재 `K=3`, `m=3`, `a_in=36`, `b_out=4`에서는 평균 차수가 대략 `3.78 * rho_n`이다.
- `K` sweep에서 `a_in`, `b_out`, `rho_n`을 고정하면 `K` 증가에 따라 within-community 후보 비율과 평균 차수가 같이 변한다. 따라서 `K` sweep 결과는 순수한 군집 수 효과와 밀도 변화 효과가 섞여 있다.
- 하이퍼파라미터를 바꿔 같은 결과 폴더에 다시 저장하면 이전 결과 해석이 애매해진다. 중요한 설정 변경에는 새 `EXPERIMENT_ID`와 새 결과 폴더를 사용한다.

## 결과 저장 위치

각 노트북은 실행 결과를 아래 구조로 저장한다.

```text
experiments/균일 HSBM 실험/results/{EXPERIMENT_ID}_{EXPERIMENT_SLUG}/
```

현재 결과 폴더:

- `EXP-20260426-004_uniform_hsbm_n_scaling_zhou_laplacian/`
  - `n` scaling 실험의 raw/summary CSV와 plot을 저장한다.
- `EXP-20260427-001_uniform_hsbm_rho_schedule_probe/`
  - `rho_n=1`, `rho_n=log(n)`, `rho_n=2log(n)`를 빠르게 비교한 탐색용 probe 결과이다.
- `EXP-20260427-001_uniform_hsbm_rho_n_sweep_zhou_theta/`
  - `n=10000`에서 다른 하이퍼파라미터를 고정하고 `rho_n`만 변화시킨 정식 sweep 결과이다.
  - `rho_n <= 1`에서는 misclassification이 약 `0.66`으로 랜덤 수준에 가깝고, `rho_n=2`부터 회복이 시작된다.
  - `rho_n=4`에서는 평균 misclassification이 약 `0.017`, `rho_n=8`에서는 약 `0.00057`, `rho_n=16`에서는 완전 회복을 보인다.
- `EXP-20260427-002_uniform_hsbm_K_sweep_zhou_theta/`
  - `n=10000`, `rho_n=4.0`에서 다른 하이퍼파라미터를 고정하고 `K`만 변화시킨 정식 sweep 결과이다.
  - `K=2`에서는 거의 완전 회복, `K=3`에서는 평균 misclassification이 약 `0.017`로 안정적이다.
  - `K=4`에서는 평균 misclassification이 약 `0.154`로 성능이 크게 낮아지고, `K>=5`에서는 거의 회복하지 못한다.
  - 단, 이 설정에서는 `K`가 커질수록 expected degree와 within-community 후보 비율이 함께 감소하므로, 순수한 `K` 효과로만 해석하면 안 된다.

## 작성 규칙

앞으로 이 폴더의 README, 마크다운 설명, 코드 주석은 기본적으로 한글로 작성한다. 모델명, 함수명, 지표명은 필요하면 영어를 그대로 사용한다.

실험 노트북, 결과 저장 구조, 핵심 알고리즘 선택, 하이퍼파라미터 sweep 등 이 폴더의 내용에 변화가 생기면 반드시 이 README에 함께 반영한다.
