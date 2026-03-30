## `src/randnla/common.py` 상세 설명

`common.py`는 Exp1~Exp4에서 반복되는 공통 로직을 한곳에 모은 유틸리티 모듈입니다.  
핵심 목표는 다음 3가지입니다.

1. 실험 데이터(SBM) 생성
2. 세 가지 방법(Random Projection, Random Sampling, Non-random) 실행
3. 논문 정렬 지표(Error for P, Theta, B) 계산 및 시각화

---

## 전체 동작 과정 (큰 흐름)

실험 스크립트(`experiments/.../exp*_...py`)는 보통 아래 순서로 `common.py` 함수를 호출합니다.

1. 실험 축 설정  
`n`, `alpha_n`, `K` 중 하나를 변화시키는 루프를 구성합니다.

2. 반복 샘플 생성  
각 반복에서 `generate_sbm_instance(...)`로 `(A, P, B_true, y_true, Theta_true)`를 만듭니다.

3. 방법 3개 실행  
`run_random_projection`, `run_random_sampling`, `run_non_random`를 각각 실행해 `(A_hat, y_pred)`를 얻습니다.

4. 지표 계산  
`evaluate_metrics(...)`로 `error_P`, `error_Theta`, `error_B`를 계산합니다.

5. 결과 저장  
반복별 raw 레코드를 DataFrame으로 만든 뒤 `summarize_metrics(...)`로 평균/표준편차를 집계합니다.

6. 그림 생성  
`plot_metric_panels(...)`, `plot_runtime(...)`로 논문 Figure 스타일의 그래프를 저장합니다.

---

## 상수

### `METHODS`
역할: 방법 이름의 표준 순서를 고정합니다.  
값: `["Random Projection", "Random Sampling", "Non-random"]`  
사용처: 결과 정렬, plot 루프.

### `METHOD_COLORS`
역할: 방법별 색상을 고정합니다.  
사용처: 선 그래프에서 방법마다 같은 색을 유지.

---

## 진행 상태 클래스

### `LiveProgress`
역할: 긴 실험 루프에서 진행률/ETA를 한 줄로 실시간 출력.

### `__init__(total_steps)`
입력: 전체 step 수.  
동작: 카운터, 시작 시간, 스피너, 바 폭 초기화.

### `_fmt(sec)`
입력: 초 단위 시간(float).  
출력: `MM:SS` 또는 `HH:MM:SS` 문자열.  
동작: 사람이 읽기 쉬운 시간 문자열 변환.

### `update(x_name, x_value, rep, reps, method)`
입력: 현재 실험축 이름/값(`n`, `alpha_n`, `K` 등), 반복 인덱스, 방법명.  
출력: 없음(터미널 출력).  
동작: 완료 step 증가, 처리율 계산, ETA 계산, 진행바 문자열 갱신.

### `close()`
역할: 마지막 줄바꿈 처리.

---

## SBM 생성 함수

### `make_balanced_labels(n, K, rng)`
역할: 가능한 균형 분할로 true 라벨 벡터 생성.  
입력: 노드 수 `n`, 커뮤니티 수 `K`, 난수기.  
출력: `y_true` shape `(n,)`.  
동작: 각 커뮤니티 크기를 `n//K` 기준으로 배분하고, 나머지를 앞 커뮤니티부터 1개씩 추가한 뒤 셔플.

### `build_B(alpha_n, lam, K)`
역할: homogeneous SBM의 블록확률행렬 `B` 구성.  
출력: `B` shape `(K, K)`.  
동작: 대각은 `alpha_n`, 비대각은 `alpha_n*(1-lam)`.

### `sample_adjacency_from_P(P, rng)`
역할: 확률행렬 `P`에서 무방향 인접행렬 `A` 샘플링.  
출력: `A` shape `(n, n)`, 대칭, 대각 0.  
동작: 상삼각에서 베르누이 샘플링 후 대칭 복원.

### `generate_sbm_instance(n, K, alpha_n, lam, rng)`
역할: 실험 1회분 ground truth + 관측 행렬 생성.  
출력: `(A, P, B_true, y_true, Theta_true)`  
동작: 라벨 생성 -> `Theta_true` one-hot 구성 -> `B_true` 생성 -> `P=Theta B Theta^T` -> `A` 샘플링.

---

## 선형대수 및 클러스터링

### `top_eigvecs_symmetric(M, k)`
역할: 대칭행렬 상위 `k` 고유벡터 추출.  
출력: `U` shape `(n, k)`.  
동작: `(M+M^T)/2`로 대칭 보정 후 `eigh`, 큰 고유값 기준 선택.

### `top_eigpairs_symmetric(M, k)`
역할: 상위 고유값/고유벡터를 함께 반환.  
출력: `(vals, vecs)` with `vals` 내림차순.

### `kmeans_on_rows(U, K, rng)`
역할: 임베딩 `U`의 각 row를 K-means로 군집화.  
출력: 예측 라벨 `y_pred` shape `(n,)`.

---

## 방법(Estimator) 구현

### `run_non_random(A, K, K_prime, rng)`
역할: baseline spectral clustering.  
동작: `A`의 상위 `K_prime` 고유벡터 -> row K-means.  
출력: `A_hat=A`, `labels`.

### `run_random_projection(A, K, K_prime, r, q, rng)`
역할: Randomized range finder 기반 근사.  
동작: `Y=A^(2q+1)Omega` -> `Q=qr(Y)` -> `C=Q^TAQ` -> `A_hat=QCQ^T` -> `C` 고유벡터를 원공간으로 올려 K-means.  
출력: `A_hat`, `labels`.

### `run_random_sampling(A, K, K_prime, p, rng)`
역할: 확률 샘플링으로 희소화 후 저랭크 복원.  
동작: 상삼각 엣지를 확률 `p`로 샘플하고 `1/p` 재스케일 -> `eigsh`(실패 시 dense fallback) -> rank-`K_prime` 복원.  
출력: `A_hat`, `labels`.

---

## 라벨 정렬 및 지표

### `spectral_norm_sym(M)`
역할: `||M||_2` 계산.  
동작: 대칭 고유값 절댓값의 최대값 반환.

### `align_labels_weighted_hungarian(y_true, y_pred, K)`
역할: 클래스 permutation 불일치를 Hungarian으로 정렬.  
동작: confusion matrix 생성 후 `1/n_k` 가중 점수를 최대화하는 매핑 계산.

### `theta_error_exact(Theta_true, y_true, y_pred, K)`
역할: 논문식 Error for Theta를 정확 계산.  
동작: `K!` permutation 완전탐색으로 최소 목적값 찾기.  
출력: `(err_theta, Theta_hat_best, best_perm)`  
주의: `K`가 커지면 factorial 비용 증가.

### `theta_error_weighted_hungarian(Theta_true, y_true, y_pred, K)`
역할: Hungarian 기반 근사형 Error for Theta 계산.  
장점: 빠름.  
출력: `(err_theta, Theta_hat_aligned)`.

### `estimate_B_hat(A_hat, Theta_hat)`
역할: plug-in 추정식으로 `B_hat` 계산.  
동작: `Theta^T A_hat Theta / (cluster_count outer)`.

### `evaluate_metrics(..., theta_mode="exact")`
역할: 실험에서 쓰는 세 지표를 한 번에 계산.  
출력: `(err_P, err_Theta, err_B)`  
동작:  
`err_P = ||A_hat - P||_2`  
`err_Theta = exact 또는 hungarian 방식`  
`err_B = ||B_hat - B_true||_inf`

---

## 집계/시각화

### `summarize_metrics(df_raw, group_cols)`
역할: raw 반복 결과를 그룹별 평균/표준편차로 요약.  
출력: `error_P_mean/std`, `error_Theta_mean/std`, `error_B_mean/std`, `time_mean/std`.

### `plot_metric_panels(summary, x_col, out_png)`
역할: Error for P/Theta/B 3패널 라인플롯 저장.  
입력: 요약 DF, x축 컬럼(`n`, `alpha_n`, `K`), 저장 경로.

### `plot_runtime(summary, x_col, out_png)`
역할: 방법별 평균 실행시간 라인플롯 저장.

---

## `theta_mode` 권장 사용법

1. Exp1, Exp2, Exp4처럼 `K`가 작으면 `theta_mode="exact"` 권장.  
2. `K`가 커지거나 속도가 중요하면 `theta_mode="hungarian"` 사용.  
3. 논문 재현에서 “정확한 정의”를 강조할 때는 `exact`를 기본으로 둡니다.

---

## 출력 해석 요약

1. `error_P`가 작을수록 `A_hat`이 population matrix `P`에 가깝습니다.  
2. `error_Theta`가 작을수록 커뮤니티 복원이 정확합니다.  
3. `error_B`가 작을수록 블록 확률행렬 추정이 정확합니다.  
4. `time_sec`는 방법별 계산비용 비교 지표입니다.

---

## Hypergraph SBM 유틸 (`src/hypergraph_sbm.py`)

하이퍼그래프 스펙트럴 클러스터링 연구를 위한 데이터 생성/행렬화 유틸입니다.

- 생성:
  - `generate_uniform_hsbm_instance(...)`
  - `generate_nonuniform_hsbm_instance(...)`
- 변환:
  - `hyperedges_to_incidence_csr(...)`
  - `clique_expansion_adjacency(...)`
  - `zhou_normalized_laplacian(...)`
- 통계:
  - `hypergraph_basic_stats(...)`
