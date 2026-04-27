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
  - 기본 설정은 `K=3`, `m=3`, `rho_n=1.0`이다.
  - `n=1000`부터 `10000`까지 1000 간격으로 변화시킨다.
  - Zhou normalized hypergraph Laplacian 기반 spectral clustering을 수행한다.
  - misclassification rate, ARI, NMI, CPU time, wall-clock time, peak memory, 단계별 runtime을 기록한다.

## 결과 저장 위치

각 노트북은 실행 결과를 아래 구조로 저장한다.

```text
experiments/균일 HSBM 실험/results/{EXPERIMENT_ID}_{EXPERIMENT_SLUG}/
```

## 작성 규칙

앞으로 이 폴더의 README, 마크다운 설명, 코드 주석은 기본적으로 한글로 작성한다. 모델명, 함수명, 지표명은 필요하면 영어를 그대로 사용한다.
