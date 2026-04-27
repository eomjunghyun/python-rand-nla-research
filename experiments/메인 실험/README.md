# 메인 실험

이 폴더는 비균일 HSBM(non-uniform HSBM)을 대상으로 한 주요 실험 노트북을 모아둔다. 공통적으로 `generate_nonuniform_hsbm_instance`로 하이퍼그래프를 생성하고, Zhou normalized hypergraph Laplacian 기반 spectral clustering을 수행한다.

## 파일 목록

- `EXP-20260425-001_nonuniform_hsbm_n_scaling_zhou_laplacian.ipynb`
  - 작은 범위의 `n` 변화 실험이다.
  - `n=200, 400, 800, 1200`에 대해 misclassification rate, ARI, NMI, CPU time, wall-clock time, peak memory, 단계별 runtime을 기록한다.

- `EXP-20260426-002_nonuniform_hsbm_large_n_1000_to_10000_zhou_laplacian.ipynb`
  - 큰 범위의 `n` 변화 실험이다.
  - `n=1000`부터 `10000`까지 1000 간격으로 변화시킨다.

- `EXP-20260426-003_nonuniform_hsbm_fixed_n_k_scaling_zhou_laplacian.ipynb`
  - `n`을 고정하고 community 수 `K`를 변화시키는 실험이다.
  - 기본 설정은 `N_FIXED=5000`, `K_VALUES=[2, 3, 4, 5, 6, 8, 10]`이다.

## 결과 저장 위치

각 노트북은 실행 결과를 아래 구조로 저장한다.

```text
experiments/메인 실험/results/{EXPERIMENT_ID}_{EXPERIMENT_SLUG}/
```

저장되는 주요 파일은 raw CSV, summary CSV, 설정 JSON, summary plot, runtime breakdown plot이다.

## 작성 규칙

앞으로 이 폴더의 README, 마크다운 설명, 코드 주석은 기본적으로 한글로 작성한다. 필요한 경우 `spectral clustering`, `wall-clock time`, `runtime breakdown`처럼 실험 용어는 영어를 섞어 쓴다.
