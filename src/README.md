# `src/common.py` 함수 설명서

`common.py`는 이 프로젝트의 실험 노트북과 실행 스크립트가 공통으로 사용하는 유틸리티 모듈입니다. 균일 HSBM 하이퍼그래프 생성, 하이퍼그래프 incidence matrix와 정규화 라플라시안 구성, spectral clustering에 필요한 고유벡터 계산 보조 함수, k-means 및 clustering 평가 지표, 실험 결과 요약과 시각화 helper를 한곳에 모아 둡니다.

즉 각 실험 파일은 데이터 생성, 행렬 구성, 알고리즘 실행, metric 계산, 결과 저장/시각화에 필요한 반복 코드를 직접 다시 쓰지 않고 이 파일의 함수를 호출해 같은 방식으로 실험을 수행합니다. 특히 하이퍼그래프 실험에서는 `generate_uniform_hsbm_instance`, `make_uniform_hsbm_probs`, `hypergraph_laplacian`이 핵심 생성 및 행렬 구성 API이고, Section 7/8 계열 그래프 실험에서는 sparse graph loader, randomized eigensolver, timing/plotting helper들이 공통 기반으로 쓰입니다.

## 기본 타입과 진행 표시

### `LiveProgress(total_steps)`

콘솔에서 긴 반복 실험의 진행률을 한 줄로 표시하는 작은 progress helper입니다.

- `total_steps`: 전체 반복 단계 수입니다. 1보다 작은 값이 들어와도 내부에서는 최소 1로 보정합니다.
- `update(x_name, x_value, rep, reps, method)`: 현재 sweep 값, 반복 번호, 방법명을 받아 진행률, elapsed time, ETA를 출력합니다.
- `close()`: 진행 표시가 끝난 뒤 줄바꿈을 출력합니다.
- Section 7.1 실험처럼 `x` 값, repetition, method가 중첩된 반복문에서 사용합니다.

### `make_balanced_labels(n, K, rng)`

길이 `n`의 community label 벡터를 균등하게 만듭니다.

- 반환값은 `0, ..., K-1` label을 담은 `np.ndarray`입니다.
- 각 군집 크기는 `floor(n/K)` 또는 `ceil(n/K)`입니다.
- 마지막에 `rng.shuffle`을 적용하므로 label 순서가 노드 번호와 정렬되어 있지 않습니다.
- 균일 HSBM과 SBM 생성 함수의 기본 true label 생성에 사용됩니다.

## 균일 HSBM 생성

### `sample_uniform_hsbm_hyperedges_exact(labels, m, p_in, p_out, rng)`

모든 `m`-크기 후보 하이퍼엣지를 열거해서 Bernoulli sampling을 수행합니다.

- 후보 집합은 `{e subset {0,...,n-1}: |e|=m}`입니다.
- 후보 `e`의 모든 노드가 같은 label이면 확률 `p_in`으로 선택합니다.
- 두 개 이상의 label이 섞인 후보는 확률 `p_out`으로 선택합니다.
- 반환값은 정렬된 tuple hyperedge들의 list입니다.
- `C(n,m)` 전체를 훑기 때문에 작은 `n` 또는 작은 `m`에서 정확한 독립 Bernoulli 모델을 확인할 때 적합합니다.

### `sample_uniform_hsbm_hyperedges_sparse(labels, m, p_in, p_out, rng)`

큰 후보 공간을 전부 열거하지 않고 within 후보와 mixed 후보를 직접 샘플링합니다.

- 각 community 내부 후보 수 `C(n_k,m)`에 대해 `Binomial(C(n_k,m), p_in)`으로 within edge 수를 먼저 뽑습니다.
- mixed 후보 수에 대해 `Binomial(N_mixed, p_out)`으로 mixed edge 수를 뽑습니다.
- 실제 edge는 중복 없이 뽑으며, 필요하면 작은 후보 공간은 exhaustive fallback을 사용합니다.
- 독립 Bernoulli 모델의 edge count 분포를 보존하면서 큰 `n` 실험을 빠르게 돌리기 위한 sampler입니다.
- 충분한 unique edge를 모으지 못하면 `RuntimeError`를 냅니다.

### `generate_uniform_hsbm_instance(n, K, m, p_in, p_out, rng, labels=None, sampling="auto", max_enumeration=1500000)`

균일 HSBM 인스턴스를 생성하는 중심 함수입니다.

- `n`: 노드 수입니다.
- `K`: community 수입니다.
- `m`: 모든 하이퍼엣지의 크기입니다. 2 이상이어야 합니다.
- `p_in`: within-community 후보 하이퍼엣지 생성 확률입니다.
- `p_out`: mixed-community 후보 하이퍼엣지 생성 확률입니다.
- `labels`: 직접 지정하지 않으면 `make_balanced_labels`로 균등 label을 만듭니다.
- `sampling="exact"`이면 모든 후보를 열거하고, `"sparse"`이면 sparse sampler를 사용합니다.
- `sampling="auto"`이면 후보 수가 `max_enumeration` 이하일 때 exact, 그보다 크면 sparse를 선택합니다.

반환값은 `(hyperedges, y_true, Theta_true, stats)`입니다.

- `hyperedges`: `tuple[int, ...]` 형태의 hyperedge list입니다.
- `y_true`: 길이 `n`의 true label 벡터입니다.
- `Theta_true`: one-hot membership matrix입니다. shape은 `(n, K)`입니다.
- `stats`: 후보 수, within 후보 수, mixed 후보 수, sampling mode, 생성 확률 등을 담은 dict입니다.

### `generate_planted_uniform_hsbm_instance(n, K, d, a_d, b_d, rho_n, rng, labels=None, sampling="auto", max_enumeration=1500000, clip=True)`

논문 실험에서 자주 쓰는 sparse-regime 확률식을 감싼 convenience wrapper입니다.

확률식은 다음과 같습니다.

```text
p_in  = a_d * rho_n / n^(d-1)
p_out = b_d * rho_n / n^(d-1)
```

- `d`는 uniform hyperedge size입니다. 내부적으로 `generate_uniform_hsbm_instance`의 `m`으로 전달됩니다.
- `a_d`와 `b_d`는 within/mixed edge의 signal strength 상수이며 `a_d > b_d > 0`이어야 합니다.
- `rho_n`은 전체 density를 조절하는 양수 scale입니다.
- `clip=True`이면 계산된 확률을 `[0,1]`로 자릅니다. `clip=False`일 때 범위를 벗어나면 `ValueError`를 냅니다.
- 반환 구조는 `generate_uniform_hsbm_instance`와 같고, `stats`에 `d`, `a_d`, `b_d`, `rho_n`이 추가됩니다.

### `make_uniform_hsbm_probs(n, d, a_d, b_d, rho_n=1.0, clip=True)`

균일 HSBM 확률 `p_in`, `p_out`만 계산합니다.

- `p_in = a_d * rho_n / n^(d-1)`
- `p_out = b_d * rho_n / n^(d-1)`
- `a_d > b_d > 0`, `rho_n > 0`, `d >= 2`를 검증합니다.
- 반환값은 `(p_in, p_out)`입니다.
- 실험 설정을 명시적으로 기록하거나, 생성과 확률 계산을 분리하고 싶을 때 사용합니다.

## 하이퍼그래프 행렬 변환

### `hypergraph_to_star_graph(n, hyperedges, weights=None, weighting="unit")`

하이퍼그래프를 star expansion bipartite graph adjacency로 변환합니다.

- 출력 matrix shape은 `(n + |E|, n + |E|)`입니다.
- 원래 노드는 `0, ..., n-1` index를 유지합니다.
- hyperedge `e_j`는 보조 노드 `n+j`로 표현됩니다.
- 각 incidence `(v, e_j)`마다 대칭 adjacency entry를 추가합니다.
- `weighting="unit"`은 incidence weight를 `w_e`로 둡니다.
- `weighting="inverse_size"`는 `w_e / |e|`를 씁니다.
- `weighting="inverse_sqrt_size"`는 `w_e / sqrt(|e|)`를 씁니다.
- 반환값은 `(A, stats)`입니다. `A`는 CSR sparse matrix입니다.

### `hypergraph_basic_stats(n, hyperedges, labels=None)`

하이퍼그래프의 기본 통계를 계산합니다.

- `n_nodes`, `n_hyperedges`, 평균/최소/최대 hyperedge size를 반환합니다.
- hyperedge size별 histogram을 `hyperedge_size_histogram`에 담습니다.
- `labels`가 주어지면 모든 노드가 같은 community에 속하는 hyperedge 비율을 `within_community_ratio`로 계산합니다.
- 빠른 sanity check와 README/보고서용 진단값 생성에 적합합니다.

### `hyperedges_to_incidence_csr(n, hyperedges, dtype=np.float32)`

하이퍼엣지 list를 incidence matrix `H`로 변환합니다.

- `H`의 shape은 `(n, |E|)`입니다.
- `H[i, e] = 1`이면 노드 `i`가 hyperedge `e`에 속한다는 뜻입니다.
- 중복 vertex가 있는 hyperedge 또는 범위를 벗어난 vertex index는 `ValueError`를 냅니다.
- 반환값은 CSR sparse matrix입니다.

### `hypergraph_laplacian(n, hyperedges, edge_weights=None)`

정규화 하이퍼그래프 라플라시안 `L = I - Theta`를 계산합니다.

먼저 incidence matrix `H`, hyperedge weight diagonal `W`, hyperedge degree `D_e`, vertex degree `D_v`를 만듭니다.

```text
Theta = D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2)
L     = I - Theta
```

- `edge_weights`가 없으면 모든 hyperedge weight는 1입니다.
- `D_e[e,e] = |e|`입니다.
- `D_v[i,i] = sum_{e contains i} w_e`입니다.
- 고립 노드처럼 `d_v(i)=0`인 경우 `1/sqrt(d_v(i))`는 0으로 처리합니다.
- hyperedge가 하나도 없으면 identity matrix를 반환합니다.
- 반환값은 CSR sparse matrix입니다.
- spectral clustering에서는 보통 `Theta = I - L`을 다시 만들어 가장 큰 고유값의 고유벡터를 사용합니다.

## EDVW와 directed Laplacian

### `edvw_transition_matrix_from_incidence(R, hyperedge_weights=None, isolated="self_loop")`

EDVW(edge-dependent vertex weight) incidence matrix `R`에서 random walk transition matrix를 만듭니다.

- `R`의 shape은 `(|E|, |V|)`입니다.
- `R[e,v]`는 hyperedge `e` 안에서 vertex `v`가 갖는 EDVW weight입니다.
- 반환되는 `P`는 row-stochastic transition matrix입니다.
- `isolated="self_loop"`이면 isolated vertex에 self-loop를 추가합니다.
- `isolated="raise"`이면 isolated vertex가 있을 때 예외를 냅니다.
- `isolated="zero"`이면 isolated row를 0으로 둡니다.

### `edvw_transition_matrix(n, hyperedges, vertex_weights=None, hyperedge_weights=None, isolated="self_loop")`

hyperedge list에서 EDVW transition matrix를 직접 만듭니다.

- `vertex_weights`가 없으면 모든 incidence weight를 1로 둡니다.
- `vertex_weights[j]`를 주는 경우 `hyperedges[j]`와 길이가 같아야 합니다.
- 내부에서 `R`을 만든 뒤 `edvw_transition_matrix_from_incidence`를 호출합니다.

### `stationary_distribution_power(P, tol=1e-12, max_iter=10000, initial=None, return_info=False)`

row-stochastic transition matrix `P`의 stationary distribution을 power iteration으로 구합니다.

- 반복식은 `pi_{t+1} = P^T pi_t`입니다.
- 매 반복마다 합이 1이 되도록 정규화합니다.
- `return_info=True`이면 수렴 여부, 반복 횟수, 마지막 L1 차이를 함께 반환합니다.
- Chung directed Laplacian과 EDVW wrapper에서 사용됩니다.

### `chung_directed_laplacian(P, stationary=None, kind="normalized", tol=1e-12, max_iter=10000)`

Chung directed graph Laplacian을 만듭니다.

- `kind="combinatorial"`이면 `Phi - (Phi P + P^T Phi)/2`를 반환합니다.
- `kind="normalized"`이면 `I - (Phi^(1/2) P Phi^(-1/2) + Phi^(-1/2) P^T Phi^(1/2))/2`를 반환합니다.
- `stationary`가 없으면 `stationary_distribution_power`로 계산합니다.
- normalized version은 stationary distribution의 모든 entry가 양수여야 합니다.

### `chung_directed_similarity(P, stationary=None, tol=1e-12, max_iter=10000)`

Chung normalized Laplacian의 similarity matrix `T = I - L`을 반환합니다.

- RDC-Spec 계열 알고리즘에서 직접 eigenvector를 계산할 대상입니다.
- 내부 수식은 directed transition을 symmetric operator로 바꾸는 형태입니다.
- stationary distribution이 없으면 power iteration으로 계산합니다.

### `hayashi_edvw_laplacian(n, hyperedges, vertex_weights=None, hyperedge_weights=None, kind="normalized", isolated="self_loop", return_transition=False)`

Hayashi et al. EDVW random walk 기반 Laplacian wrapper입니다.

- hyperedge list와 EDVW vertex weight를 받아 transition matrix `P`를 만듭니다.
- stationary distribution을 계산합니다.
- `chung_directed_laplacian`으로 Laplacian을 반환합니다.
- `return_transition=True`이면 `(lap, P, pi)`를 반환합니다.

### `hayashi_edvw_similarity_matrix(n, hyperedges, vertex_weights=None, hyperedge_weights=None, isolated="self_loop", return_transition=False)`

EDVW random walk에서 spectral embedding에 쓰는 similarity matrix를 반환합니다.

- 내부 절차는 `hayashi_edvw_laplacian`과 같지만 최종 반환값이 `T = I - L`입니다.
- `return_transition=True`이면 `(T, P, pi)`를 반환합니다.

## 그래프 SBM과 dense randomized NLA

### `build_B(alpha_n, lam, K)`

SBM block probability matrix `B`를 만듭니다.

- diagonal은 `alpha_n`입니다.
- off-diagonal은 `alpha_n * (1-lam)`입니다.
- `lam`이 클수록 within/between probability 차이가 커집니다.

### `sample_adjacency_from_P(P, rng)`

대칭 확률 행렬 `P`에서 무방향 adjacency matrix를 샘플링합니다.

- upper triangle만 Bernoulli sampling하고 대칭으로 복사합니다.
- diagonal은 0으로 둡니다.
- 반환값은 dense `np.ndarray`입니다.

### `generate_sbm_instance(n, K, alpha_n, lam, rng)`

Section 7.1 실험용 SBM 인스턴스를 생성합니다.

- balanced label과 one-hot membership matrix를 만듭니다.
- `B_true = build_B(alpha_n, lam, K)`를 계산합니다.
- `P = Theta_true B_true Theta_true^T`를 만든 뒤 adjacency `A`를 샘플링합니다.
- 반환값은 `(A, P, B_true, y_true, Theta_true)`입니다.

### `top_eigvecs_symmetric(M, k)`

dense symmetric matrix에서 가장 큰 고유값에 대응하는 `k`개 고유벡터를 반환합니다.

- 계산 전 `M <- (M + M.T)/2`로 대칭화합니다.
- `np.linalg.eigh`를 사용하므로 full eigendecomposition입니다.
- 반환값은 shape `(n, k)`의 eigenvector matrix입니다.

### `top_eigpairs_symmetric(M, k)`

`top_eigvecs_symmetric`와 같지만 eigenvalue도 함께 반환합니다.

- 반환값은 `(vals_top, vecs_top)`입니다.
- eigenvalue는 내림차순으로 정렬됩니다.

### `normalize_rows_l2(U, eps=1e-12)`

행렬 `U`의 각 row를 L2 norm 1이 되도록 정규화합니다.

- norm이 `eps`보다 작으면 해당 row는 0으로 유지합니다.
- spectral embedding을 k-means에 넣기 전에 자주 사용합니다.

### `kmeans_on_rows(U, K, rng, normalize_rows=False)`

행렬 row를 데이터 포인트로 보고 k-means clustering을 수행합니다.

- `normalize_rows=True`이면 먼저 `normalize_rows_l2`를 적용합니다.
- `rng`에서 scikit-learn `random_state`를 생성합니다.
- `n_init=20`으로 실행합니다.
- 반환값은 예측 label 벡터입니다.

### `attach_timing_breakdown(record, algo_timing=None, instance_sec=None, metric_sec=None)`

실험 결과 record에 timing breakdown을 병합합니다.

- `instance_sec`는 데이터 생성 시간입니다.
- `metric_sec`는 metric 계산 시간입니다.
- `algo_timing`이 있으면 알고리즘 단계별 시간을 record에 추가합니다.
- 가능한 경우 `pipeline_total_sec`도 계산합니다.

### `run_non_random(A, K, K_prime, rng, normalize_rows=False, return_timing=False)`

dense adjacency `A`에 대해 full eigendecomposition 기반 spectral clustering을 수행합니다.

- `K_prime`개 eigenvector를 구합니다.
- 그 embedding row에 k-means를 적용해 `K`개 cluster를 얻습니다.
- `A_hat`은 원본 `A` copy입니다.
- `return_timing=True`이면 eigensolver와 k-means 시간을 포함합니다.

### `run_random_projection(A, K, K_prime, r, q, rng, normalize_rows=False, return_timing=False)`

dense randomized range finder 기반 spectral clustering입니다.

계산 흐름은 다음과 같습니다.

```text
Omega ~ N(0,1)^(n x (K_prime+r))
Y = A^(2q+1) Omega
Q = qr(Y)
C = Q^T A Q
U = Q * eigvecs(C)
```

- `r`은 oversampling rank입니다.
- `q`는 power iteration 횟수입니다.
- 작은 core matrix `C`에서 eigenvector를 구한 뒤 원래 공간으로 lift합니다.
- `return_timing=True`이면 projection 단계별 시간을 반환합니다.

### `run_random_sampling(A, K, K_prime, p, rng, normalize_rows=False, return_timing=False)`

dense adjacency의 upper triangle entry를 확률 `p`로 샘플링해 spectral clustering을 수행합니다.

- 선택된 entry는 `1/p`로 rescale합니다.
- 샘플링된 대칭 행렬 `A_s`에서 `eigsh`를 우선 시도합니다.
- 실패하면 dense eigendecomposition fallback을 사용합니다.
- `A_hat = U diag(vals) U^T` 형태의 low-rank reconstruction도 반환합니다.

## Metric 함수

### `spectral_norm_sym(M)`

대칭화된 matrix의 spectral norm을 계산합니다.

- 내부에서 `M <- (M + M.T)/2`를 적용합니다.
- `np.linalg.eigvalsh`로 모든 eigenvalue를 구하고 절댓값 최대를 반환합니다.

### `align_labels_weighted_hungarian(y_true, y_pred, K)`

예측 cluster label을 true label에 맞게 Hungarian matching으로 정렬합니다.

- community size 차이를 보정하기 위해 true class별 inverse size weight를 사용합니다.
- 반환값은 정렬된 예측 label입니다.

### `theta_error_exact(Theta_true, y_true, y_pred, K)`

모든 label permutation을 완전 탐색해 membership matrix error를 계산합니다.

- 작은 `K`에서 exact matching을 확인할 때 씁니다.
- 반환값은 `(best_error, best_theta_hat, best_perm)`입니다.

### `theta_error_weighted_hungarian(Theta_true, y_true, y_pred, K)`

Hungarian matching으로 membership matrix error를 근사 계산합니다.

- `theta_error_exact`보다 빠릅니다.
- 반환값은 `(error, Theta_hat)`입니다.

### `estimate_B_hat(A_hat, Theta_hat)`

추정 membership matrix를 기준으로 block probability matrix를 추정합니다.

```text
B_hat = (Theta_hat^T A_hat Theta_hat) / (cluster_size outer cluster_size)
```

- denominator가 0인 위치는 0으로 처리합니다.

### `evaluate_metrics(A_hat, y_pred, P, B_true, Theta_true, y_true, K, theta_mode="exact")`

Section 7.1 실험의 세 metric을 계산합니다.

- `err_P`: `A_hat - P`의 spectral norm입니다.
- `err_Theta`: membership recovery error입니다.
- `err_B`: block matrix 추정의 max absolute error입니다.
- `theta_mode="exact"` 또는 `"hungarian"`을 선택할 수 있습니다.

### `summarize_metrics(df, group_cols)`

실험 raw dataframe을 group별 평균과 표준편차로 요약합니다.

- 기본 metric은 `err_P`, `err_Theta`, `err_B`, `time_sec`입니다.
- 반환값은 pandas dataframe입니다.

### `extract_timing_breakdown(record)`

record에서 `_sec`로 끝나는 timing field만 추출합니다.

- timing summary를 만들기 위한 전처리 helper입니다.

### `summarize_timing_breakdown(df, group_cols)`

timing raw dataframe을 group별 평균과 표준편차로 요약합니다.

- `_sec` suffix를 가진 모든 column을 대상으로 합니다.

## 시각화 함수

### `plot_metric_panels(df_summary, x_col, title, out_png)`

`err_P`, `err_Theta`, `err_B`를 3개 panel로 그립니다.

- method별 평균선을 그리고 표준편차 band를 함께 표시합니다.
- `out_png`가 주어지면 파일로 저장합니다.

### `plot_runtime(df_summary, x_col, title, out_png)`

method별 runtime 변화를 한 그래프에 그립니다.

- `time_sec_mean`과 `time_sec_std`를 사용합니다.
- Section 7.1 Figure 스타일 runtime 비교에 사용됩니다.

## Sparse graph loader와 eigenvector helper

### `load_undirected_edgelist_csr(path, delimiter=None, comment="#", one_indexed=False, dtype=np.float32, make_unweighted=True)`

edge list 파일을 무방향 CSR adjacency matrix로 읽습니다.

- 각 줄은 `u v` 또는 `u v weight` 형식입니다.
- `one_indexed=True`이면 노드 번호를 1-based에서 0-based로 바꿉니다.
- 중복 edge는 합산 후 정리합니다.
- `make_unweighted=True`이면 모든 edge weight를 1로 둡니다.

### `upper_triangle_edges(A)`

sparse adjacency matrix의 upper triangle edge를 COO 형태로 반환합니다.

- random sampling benchmark에서 중복 없는 undirected edge 목록을 얻는 데 사용합니다.

### `eigvecs_eigsh_sparse(A, k, rng=None)`

sparse symmetric matrix에서 `eigsh`로 top-`k` eigenvector를 계산합니다.

- 가장 큰 algebraic eigenvalue 기준(`which="LA"`)입니다.
- `rng`가 있으면 초기 vector `v0`를 제공합니다.

### `eigvecs_partial_eigen_proxy_sparse(A, k, rng=None)`

부분 eigen decomposition baseline wrapper입니다.

- 현재 구현은 `eigvecs_eigsh_sparse`를 호출합니다.
- benchmark table에서 방법명을 분리하기 위한 proxy입니다.

### `eigvecs_random_projection_sparse(A, k, r, q, rng)`

sparse matrix용 Gaussian random projection eigenvector helper입니다.

- `Omega`, power iteration, QR, core matrix eigen decomposition 순서로 진행합니다.
- 반환 embedding은 원래 노드 공간의 top-`k` 근사 eigenvector입니다.

### `sample_rescaled_adjacency_from_edges(n, edges, weights, p, rng)`

edge list에서 edge를 확률 `p`로 샘플링하고 `1/p`로 rescale한 sparse adjacency를 만듭니다.

- undirected graph이므로 선택된 edge는 양방향 entry로 추가됩니다.
- random sampling spectral method의 입력 행렬을 만듭니다.

### `eigvecs_random_sampling_sparse(A, k, p, rng)`

sparse adjacency를 random sampling한 뒤 top eigenvector를 구합니다.

- `upper_triangle_edges`로 edge 후보를 얻습니다.
- `sample_rescaled_adjacency_from_edges`로 sampled matrix를 만듭니다.
- sampled matrix에 `eigsh`를 적용합니다.

### `eigvecs_random_sampling_sparse_table4(A, k, p, rng)`

Table 4 benchmark용 random sampling helper입니다.

- 기본 흐름은 `eigvecs_random_sampling_sparse`와 같습니다.
- benchmark 코드에서 method별 구현을 명확히 분리하기 위해 별도 함수로 둡니다.

### `load_large_integer_edgelist_csr(path, delimiter=None, comment="#", one_indexed=False, dtype=np.float32, make_unweighted=True)`

큰 integer edge list를 CSR adjacency로 읽습니다.

- node id를 compact integer index로 다시 매핑합니다.
- 반환값에는 adjacency와 함께 원래 node id mapping 정보가 포함됩니다.
- 대규모 real network benchmark에서 사용합니다.

## Table 4 benchmark helper

### `benchmark_table4_methods_sparse(A, K_values, methods, reps, rng, sampling_p=0.2, projection_r=10, projection_q=1)`

sparse real network에서 여러 spectral method의 runtime을 비교합니다.

- `K_values`별로 target rank를 바꿉니다.
- method별 repetition을 수행합니다.
- non-random, random projection, random sampling 계열 method를 비교할 수 있습니다.
- 반환값은 raw timing dataframe입니다.

### `summarize_table4_median_times(df)`

Table 4 raw timing 결과를 median 기준으로 요약합니다.

- method와 `K`별 median runtime을 계산합니다.
- outlier에 덜 민감한 runtime 표를 만들 때 사용합니다.

### `format_table4_markdown(summary)`

Table 4 요약 dataframe을 markdown 표 문자열로 변환합니다.

- 보고서와 README에 붙일 수 있는 형태로 formatting합니다.

### `plot_table4_median_bars(summary, out_png)`

Table 4 median runtime을 bar chart로 그립니다.

- method별 runtime 차이를 빠르게 비교하기 위한 그림입니다.

### `plot_table4_runtime_boxplots(df, out_png)`

Table 4 raw repetition 결과를 boxplot으로 그립니다.

- repetition 사이의 runtime 변동성을 확인할 수 있습니다.

### `pairwise_ari(labelings)`

여러 clustering 결과 사이의 pairwise adjusted Rand index를 계산합니다.

- `labelings`는 label vector들의 list입니다.
- 모든 쌍에 대해 ARI를 계산해 square matrix로 반환합니다.

## Section 7.1 실험 설정 클래스

### `Exp1Config`, `Exp2Config`, `Exp3Config`, `Exp4Config`

Section 7.1 네 개 실험의 dataclass 설정입니다.

- sweep 값, 반복 횟수, SBM 파라미터, randomized method 파라미터, 출력 위치를 담습니다.
- `asdict`로 쉽게 JSON 저장할 수 있도록 단순 dataclass로 유지합니다.

### `SavedExperimentOutputs`

실험 저장 결과 경로를 담는 dataclass입니다.

- raw CSV, summary CSV, timing CSV, plot PNG 경로 등을 보관합니다.
- notebook이나 script에서 저장 결과를 일관되게 다루기 위한 반환 타입입니다.

### `TimingBreakdownResult`

timing breakdown 시각화 결과를 담는 dataclass입니다.

- 입력 summary path, 출력 directory, 생성된 plot path, table dataframe을 포함합니다.

### `RuntimeCompositionResult`

runtime composition plot 결과를 담는 dataclass입니다.

- method별 runtime table과 최종 plot path를 포함합니다.

## Section 7.1 실행 함수

### `default_exp1_config()`, `default_exp2_config()`, `default_exp3_config()`, `default_exp4_config()`

각 실험의 기본 설정 dataclass를 반환합니다.

- notebook과 CLI script에서 같은 기본값을 공유하기 위한 함수입니다.

### `default_output_dir(exp_key)`

Section 7.1 결과를 저장할 기본 directory를 반환합니다.

- `experiments/reference_1_section7_1/results` 아래 실험별 폴더를 사용합니다.

### `parse_int_values(text)`, `parse_float_values(text)`

쉼표로 구분된 CLI 문자열을 int 또는 float list로 변환합니다.

- 예: `"100,200,500"` -> `[100, 200, 500]`

### `run_experiment1(config)`, `run_experiment2(config)`, `run_experiment3(config)`, `run_experiment4(config)`

Section 7.1의 네 실험을 실행합니다.

- 각 함수는 설정된 sweep 값, repetition, method 조합을 반복합니다.
- raw metric dataframe과 timing breakdown dataframe을 만듭니다.
- 내부적으로 SBM 생성, non-random/random projection/random sampling spectral clustering, metric 계산을 수행합니다.

### `summarize_experiment1(df)`, `summarize_experiment2(df)`, `summarize_experiment3(df)`, `summarize_experiment4(df)`

각 실험 raw dataframe을 논문 figure에 맞는 group column 기준으로 요약합니다.

- 평균과 표준편차를 계산합니다.
- plotting과 결과 CSV 저장에 바로 사용할 수 있습니다.

### `save_experiment_outputs(exp_key, df_raw, df_summary, outdir, detailed_timing=False, plot_basics=True)`

실험 raw/summary/timing/plot 파일을 저장합니다.

- CSV와 PNG를 정해진 이름으로 씁니다.
- 반환값은 `SavedExperimentOutputs`입니다.

### `run_and_save_experiment1(config)`, `run_and_save_experiment2(config)`, `run_and_save_experiment3(config)`, `run_and_save_experiment4(config)`

실험 실행과 저장을 한 번에 수행합니다.

- notebook에서 가장 간단히 호출할 수 있는 high-level entry point입니다.

## Timing 결과 탐색과 시각화

### `find_latest_summary(summary_filename, search_root=None)`

결과 폴더 아래에서 특정 summary CSV 이름의 최신 파일을 찾습니다.

- 수정 시간을 기준으로 가장 최근 파일을 반환합니다.
- 없으면 `None`을 반환합니다.

### `resolve_summary_path(exp_key, summary_path=None, search_root=None)`

명시된 summary path가 있으면 그대로 쓰고, 없으면 최신 summary를 찾습니다.

- 찾지 못하면 `FileNotFoundError`를 냅니다.

### `load_summary_frame(exp_key, summary_path=None, search_root=None)`

summary CSV 경로를 해석하고 pandas dataframe으로 읽습니다.

- 반환값은 `(resolved_path, df)`입니다.

### `build_timing_table(df, base_cols, metrics)`

timing summary dataframe에서 필요한 column만 골라 table을 만듭니다.

- `metric_mean`, `metric_std` column을 자동으로 찾습니다.
- base column 기준으로 정렬합니다.

### `compute_global_metric_limits(search_root=None, summary_paths=None)`

여러 실험 timing plot에서 y축 범위를 맞추기 위한 전역 min/max를 계산합니다.

- 모든 timing metric의 평균+표준편차 최대값을 훑습니다.
- method별 plot을 비교 가능하게 만들 때 사용합니다.

### `timing_method_slug(method_name)`

method 이름을 파일명에 쓰기 좋은 slug로 바꿉니다.

- 공백과 hyphen을 정리하고 lowercase를 적용합니다.

### `timing_metric_title(method_name, metric_name)`

timing plot title에 쓸 사람이 읽기 쉬운 문자열을 만듭니다.

- 내부 label mapping에 없는 metric은 원래 이름을 사용합니다.

### `render_timing_breakdown_suite(exp_key, summary_path=None, output_dir=None, global_limits=None, search_root=None, save=True, show_plots=True)`

특정 Section 7.1 실험의 timing breakdown plot 묶음을 생성합니다.

- 전체 runtime plot과 method별 단계 plot을 만듭니다.
- table dataframe도 함께 반환합니다.

### `numeric_series(df, column_name)`

dataframe column을 numeric series로 안전하게 변환합니다.

- column이 없으면 0으로 채운 series를 반환합니다.
- plot 함수에서 missing timing column을 다룰 때 사용합니다.

### `format_x_labels(values, x_col)`

plot x축 tick label을 문자열로 포맷합니다.

- `alpha_n`은 소수 둘째 자리 형식으로 표시합니다.
- 정수형 sweep 값은 정수 문자열로 표시합니다.

### `build_method_runtime_table(df, method_name, x_col)`

특정 method의 runtime 구성 table을 만듭니다.

- 전체 algorithm time과 단계별 time을 모읍니다.
- 단계별 비율 column도 추가합니다.

### `plot_total_runtime_comparison(ax, df, x_col, x_label)`

method별 전체 runtime 비교선을 주어진 matplotlib axis에 그립니다.

- `algo_total_sec_mean`이 있으면 우선 사용하고, 없으면 `time_sec_mean`을 사용합니다.

### `annotate_stack_percentages(ax, table, component_cols, x_pos, shared_ymax)`

stacked runtime bar 위에 percentage label을 표시하는 helper입니다.

- 작은 component는 bar 밖에 label을 배치하도록 설계되어 있습니다.

### `plot_method_runtime_stack(ax, table, method_name, x_col, x_label, shared_ymax)`

특정 method의 runtime component를 stacked bar로 그립니다.

- eigensolver, projection, sampling, k-means 같은 단계별 시간을 분해해 보여줍니다.

### `render_runtime_composition(exp_key, summary_path=None, output_path=None, search_root=None, save=True, show_plot=True)`

전체 runtime 비교와 method별 runtime composition을 하나의 figure로 렌더링합니다.

- 반환값은 `RuntimeCompositionResult`입니다.

### `render_all_section7_visualizations(exp_key, summary_path=None, breakdown_output_dir=None, composition_output_path=None, global_limits=None, search_root=None, save=True, show_plots=True)`

timing breakdown suite와 runtime composition figure를 한 번에 생성합니다.

- 반환값은 `(TimingBreakdownResult, RuntimeCompositionResult)`입니다.