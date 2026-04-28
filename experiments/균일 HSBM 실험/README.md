# 균일 HSBM 실험

이 폴더는 모든 하이퍼엣지의 크기가 같은 `m`-균일 하이퍼그래프 stochastic block model 실험을 담는다. 모든 노트북은 같은 생성 모델과 같은 Zhou 정규화 하이퍼그래프 라플라시안 기반 spectral clustering을 사용한다. 실험 축은 `K`, `n`, `rho_n` 세 가지이고, 각 축마다 비랜덤 `eigsh`, 가우시안 랜덤 프로젝션, 랜덤 샘플링 버전을 둔다.

- [K변화.ipynb](./K변화.ipynb): 군집 수 `K`를 변화시킨다.
- [K변화_gaussian_random_projection.ipynb](./K변화_gaussian_random_projection.ipynb): `K` 변화 실험의 가우시안 랜덤 프로젝션 버전이다.
- [K변화_random_sampling.ipynb](./K변화_random_sampling.ipynb): `K` 변화 실험의 랜덤 샘플링 버전이다.
- [n변화.ipynb](./n변화.ipynb): 노드 수 `n`을 변화시킨다.
- [n변화_gaussian_random_projection.ipynb](./n변화_gaussian_random_projection.ipynb): `n` 변화 실험의 가우시안 랜덤 프로젝션 버전이다.
- [n변화_random_sampling.ipynb](./n변화_random_sampling.ipynb): `n` 변화 실험의 랜덤 샘플링 버전이다.
- [rho_n변화.ipynb](./rho_n변화.ipynb): 밀도 파라미터 `rho_n`을 변화시킨다.
- [rho_n변화_gaussian_random_projection.ipynb](./rho_n변화_gaussian_random_projection.ipynb): `rho_n` 변화 실험의 가우시안 랜덤 프로젝션 버전이다.
- [rho_n변화_random_sampling.ipynb](./rho_n변화_random_sampling.ipynb): `rho_n` 변화 실험의 랜덤 샘플링 버전이다.

전체 결과 요약은 [결과보고서.md](./결과보고서.md)에 모았다. 랜덤화 노트북이 공통으로 사용하는 실행 코드는 [uniform_hsbm_randomized.py](./uniform_hsbm_randomized.py)에 있다.

## 전체 실험 흐름

각 반복 실험은 아래 순서로 진행된다.

1. balanced community label `z`를 만든다.
2. planted uniform HSBM 확률식으로 하이퍼엣지 집합 `E`를 생성한다.
3. 하이퍼그래프 incidence matrix `H`를 만든다.
4. Zhou normalized operator `Theta`와 Laplacian `Delta = I - Theta`를 만든다.
5. `Theta`의 가장 큰 고유값에 대응하는 `K`개 고유벡터를 계산한다.
6. 각 노드의 spectral embedding row를 L2 정규화한다.
7. 정규화된 embedding에 `K`-means를 수행해 예측 라벨 `\hat z`를 얻는다.
8. true label `z`와 `\hat z`를 비교해 misclassification rate, ARI, NMI를 계산한다.
9. 생성 시간, 라플라시안 구성 시간, 고유벡터 계산 시간, k-means 시간, 메모리 사용량을 기록한다.

수식으로는 다음 pipeline이다.

```text
z -> E -> H -> Theta = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
  -> Delta = I - Theta
  -> top-K eigenvectors of Theta
  -> row-normalization
  -> K-means
  -> metrics
```

랜덤화 버전에서는 5번 단계만 바뀐다.

- 가우시안 랜덤 프로젝션: `Theta`에 Gaussian test matrix를 곱해 낮은 차원의 부분공간 `Q`를 만들고, `Q^T Theta Q`의 고유벡터를 원래 공간으로 lift한다.
- 랜덤 샘플링: `Theta`의 sparse nonzero entry를 확률 `p=0.7`로 샘플링하고 `1/p`로 rescale한 sampled operator에 `eigsh`를 적용한다.

## 하이퍼그래프 생성 모델

### 노드와 community label

노드 집합은

```text
V = {0, 1, ..., n-1}
```

이다. 군집 수는 `K`이고, 각 노드 `i`는 하나의 community label

```text
z_i in {0, 1, ..., K-1}
```

을 가진다.

실험에서는 label을 balanced하게 만든다. 즉 community `k`의 크기를 `n_k`라고 하면

```text
sum_{k=0}^{K-1} n_k = n
```

이고, 모든 `n_k`는 가능한 한 같게 배분된다. 코드상으로는 먼저 모든 군집에 `floor(n/K)`개를 넣고, 남은 `n mod K`개를 앞쪽 군집에 하나씩 더 넣는 방식이다. 따라서

```text
n_k in {floor(n/K), ceil(n/K)}.
```

true membership matrix는

```text
Theta_true[i, k] = 1{z_i = k}
```

로 저장된다. 여기서 `Theta_true`는 모델 생성용 true label을 one-hot 형태로 표현한 것이며, 아래 Zhou operator `Theta`와는 다른 객체다.

### 후보 하이퍼엣지

모든 하이퍼엣지 크기는 고정값 `m`이다. 따라서 후보 하이퍼엣지 집합은

```text
C_m(V) = {e subset V : |e| = m}
```

이고 후보 개수는

```text
N_total = C(n, m).
```

후보 하이퍼엣지 `e = {i_1, ..., i_m}`가 하나의 community 안에 완전히 들어 있으면 within-community 후보라고 한다.

```text
e is within  <=>  z_{i_1} = z_{i_2} = ... = z_{i_m}.
```

within 후보 개수는

```text
N_within = sum_{k=0}^{K-1} C(n_k, m),
```

mixed 후보 개수는

```text
N_mixed = N_total - N_within.
```

`K변화.ipynb`에서는 `K`가 바뀌면 `n_k`가 바뀌므로 `N_within / N_total`도 같이 바뀐다. 그래서 이 노트북은 `candidate_within_fraction`을 함께 기록한다.

### 하이퍼엣지 생성 확률

각 후보 하이퍼엣지 `e`에 indicator

```text
A_e = 1{e is sampled}
```

를 둔다. planted uniform HSBM은 다음 확률을 사용한다.

```text
P(A_e = 1 | z) = p_in   if e is within,
P(A_e = 1 | z) = p_out  otherwise.
```

실험에서는 sparse-regime parameterization을 쓴다.

```text
p_in  = a_in  * rho_n / n^{m-1},
p_out = b_out * rho_n / n^{m-1}.
```

여기서

```text
a_in > b_out > 0,
rho_n > 0.
```

현재 세 노트북의 기본값은 공통적으로

```text
m = 3,
a_in = 36.0,
b_out = 4.0.
```

`clip=False`로 호출하므로 위 식이 `[0, 1]` 범위를 벗어나면 확률을 잘라내지 않고 오류가 난다. 현재 설정에서는 `p_in`, `p_out`이 모두 1보다 작다.

### exact sampler와 sparse sampler

생성 함수는 `generate_planted_uniform_hsbm_instance`이고 내부적으로 `generate_uniform_hsbm_instance`를 호출한다. sampling mode는 현재 노트북에서 모두

```text
sampling = "sparse"
```

이다.

이상적인 독립 Bernoulli 모델은 모든 후보 `e in C_m(V)`에 대해 독립적으로 `A_e ~ Bernoulli(p_in)` 또는 `Bernoulli(p_out)`를 뽑는 것이다. 후보 수 `C(n,m)`이 커지면 전부 열거하기 어려우므로 sparse sampler는 다음 방식으로 같은 분포 구조를 효율적으로 구현한다.

1. 각 community `k`에 대해 within 후보 개수

   ```text
   N_k = C(n_k, m)
   ```

   를 계산한다.

2. 그 community 안에서 뽑을 within edge 수를

   ```text
   M_k ~ Binomial(N_k, p_in)
   ```

   으로 샘플링한다.

3. community `k` 내부의 `m`개 조합 중 `M_k`개를 중복 없이 균일하게 고른다.

4. mixed 후보 전체 개수

   ```text
   N_mixed = C(n,m) - sum_k C(n_k,m)
   ```

   에 대해

   ```text
   M_mixed ~ Binomial(N_mixed, p_out)
   ```

   을 샘플링한다.

5. 모든 노드에서 `m`개를 뽑되, 한 community 안에 전부 들어간 후보는 버리고 mixed 후보만 받아들이는 rejection 방식으로 `M_mixed`개를 중복 없이 고른다.

조건부로 edge 수가 정해졌을 때는 후보 중 균일하게 고르는 방식이 독립 Bernoulli 모델의 conditional distribution과 일치한다. 따라서 sparse sampler는 큰 `n`에서 모든 후보를 직접 순회하지 않고도 within/mixed edge 수와 균일한 후보 선택을 보존한다.

### 기대 하이퍼엣지 수와 기대 평균 차수

하이퍼엣지 수의 기대값은

```text
E[|E|] = N_within * p_in + N_mixed * p_out.
```

하이퍼그래프 vertex degree를

```text
d_i^{H} = sum_{e in E} 1{i in e}
```

라고 하면 전체 incidence 수는 항상 `m |E|`이다. 따라서 평균 하이퍼그래프 차수의 기대값은

```text
E[mean degree] = m * E[|E|] / n.
```

`K변화.ipynb`는 이 값을 `expected_degree_mean`으로 저장한다. 실제 샘플에서의 평균 차수는

```text
degree_mean = (1/n) sum_i d_i^{H} = m |E| / n
```

이고, 고립 노드 비율은

```text
isolated_fraction = (1/n) sum_i 1{d_i^{H} = 0}
```

이다.

## Zhou 정규화 하이퍼그래프 라플라시안

### Incidence matrix

생성된 하이퍼엣지 집합을

```text
E = {e_1, ..., e_M}
```

이라고 하자. 여기서 `M = |E|`이다. Incidence matrix는

```text
H in R^{n x M}
```

이고 원소는

```text
H_{i,e} = 1  if i in e,
H_{i,e} = 0  otherwise.
```

모든 하이퍼엣지의 크기는 `m`이므로

```text
delta(e) = sum_i H_{i,e} = m.
```

코드에서는 일반적인 하이퍼엣지 크기도 처리할 수 있게

```text
d_e = H^T 1
```

로 계산한다.

### 하이퍼엣지 가중치

현재 실험에서는 별도 edge weight를 넘기지 않으므로 모든 하이퍼엣지 가중치는

```text
w(e) = 1.
```

따라서

```text
W = diag(w(e_1), ..., w(e_M)) = I_M.
```

### Vertex degree

Zhou operator에서 사용하는 vertex degree는 weighted incidence degree다.

```text
d_v(i) = sum_{e in E} w(e) H_{i,e}.
```

현재 `w(e)=1`이므로 이것은 하이퍼그래프 degree와 같다.

행렬로 쓰면

```text
d_v = H w.
```

그리고

```text
D_v = diag(d_v(0), ..., d_v(n-1)),
D_e = diag(delta(e_1), ..., delta(e_M)).
```

고립 노드처럼 `d_v(i)=0`인 경우 코드에서는 `1/sqrt(d_v(i))`를 0으로 둔다. 즉 해당 노드는 Zhou operator에서 연결 기여가 없고, Laplacian diagonal 쪽에 남는다.

### Zhou operator

Zhou normalized operator는

```text
Theta = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}.
```

현재 실험처럼 `w(e)=1`, `|e|=m`이면 entry-wise로

```text
Theta_{ij}
  = sum_{e: i,j in e} 1 / (m * sqrt(d_v(i) d_v(j)))
```

단, `d_v(i)=0` 또는 `d_v(j)=0`이면 해당 항은 0으로 처리된다.

이 행렬은 대칭 행렬이다. 실험 코드에서는 수치 오차를 줄이기 위해 eigensolver에 넣기 전에

```text
Theta <- (Theta + Theta^T) / 2
```

를 한 번 수행한다.

### Zhou normalized Laplacian

Zhou normalized hypergraph Laplacian은

```text
Delta = I - Theta.
```

`n변화.ipynb`에서는 함수 이름상 `zhou_normalized_laplacian`으로 `Delta`를 만든 뒤

```text
Theta = I - Delta
```

를 다시 계산한다. `K변화.ipynb`와 `rho_n변화.ipynb`도 동일하게 `Delta`를 만든 뒤 `Theta`를 사용한다. 즉 세 노트북의 실제 spectral clustering 대상은 모두 `Theta`이다.

`Theta`의 가장 큰 고유값 고유벡터를 쓰는 것은 `Delta = I - Theta`의 가장 작은 고유값 고유벡터를 쓰는 것과 같은 eigenspace를 사용한다. 왜냐하면

```text
Theta u = lambda u
```

이면

```text
Delta u = (I - Theta)u = (1 - lambda)u
```

이므로 `Theta`에서 큰 `lambda`는 `Delta`에서 작은 `1-lambda`에 대응한다.

## Spectral clustering 계산

### 고유벡터 계산

군집 수를 `K`라고 할 때, 실험은 `Theta`의 가장 큰 고유값에 대응하는 `K`개 고유벡터를 계산한다.

```text
Theta u_l = lambda_l u_l,
lambda_1 >= lambda_2 >= ...,
U = [u_1, ..., u_K] in R^{n x K}.
```

노트북 구현은 다음과 같다.

- 일반적인 경우에는 sparse ARPACK eigensolver인 `scipy.sparse.linalg.eigsh`로 가장 큰 `K`개 고유쌍만 계산한다.
- `n <= K+1`이면 `eigsh`가 요구하는 `k < n` 조건을 만족하지 못하므로 dense `numpy.linalg.eigh`를 사용한다.
- `eigsh`가 실패하는 예외 상황에서도 dense `eigh`로 fallback한다.
- 현재 세 노트북의 핵심 구현은 다음과 같다.

  ```text
  vals, vecs = spla.eigsh(
      theta,
      k=K,
      which="LA",
      tol=eigsh_tol,
      v0=rng.normal(size=n),
  )
  order = np.argsort(vals)[-K:][::-1]
  top_vals = vals[order]
  top_vecs = vecs[:, order]
  ```

여기서 `"LA"`는 largest algebraic eigenvalues를 뜻한다. 즉 `Theta`의 가장 큰 고유값에 대응하는 고유벡터를 직접 가져온다. dense fallback은 안정성 장치일 뿐이고, 큰 `n` 실험의 기본 경로는 `eigsh`다.

### Row normalization

고유벡터 행렬 `U`의 `i`번째 행을 `U_i`라고 하면, `normalize_embedding_rows=True`일 때

```text
X_i = U_i / ||U_i||_2
```

를 사용한다. `||U_i||_2`가 매우 작으면 0으로 둔다. 코드의 threshold는 `eps = 1e-12`이다.

따라서 spectral embedding matrix는

```text
X in R^{n x K}
```

이고 각 노드는 `K`차원 점 `X_i`로 표현된다.

### K-means

정규화된 row embedding에 `K`-means를 수행한다.

```text
\hat z = KMeans(X, n_clusters=K).
```

목적함수는 표준 k-means objective이다.

```text
min_{c_1,...,c_K, \hat z}
sum_{i=0}^{n-1} ||X_i - c_{\hat z_i}||_2^2.
```

현재 설정은 다음과 같다.

- `n변화.ipynb`, `rho_n변화.ipynb`: `n_init=20`
- `K변화.ipynb`: `kmeans_n_init=20`

random state는 각 반복의 RNG에서 새 정수 seed를 뽑아 사용한다.

## 측정 지표

### 오분류율

군집 라벨은 permutation ambiguity가 있다. 예를 들어 true label `0,1,2`와 예측 label `2,0,1`은 같은 partition일 수 있다. 그래서 먼저 confusion matrix를 만든다.

```text
C_{ab} = |{i : z_i = a, \hat z_i = b}|.
```

그 다음 Hungarian matching으로

```text
pi* = argmax_pi sum_{a=0}^{K-1} C_{a, pi(a)}
```

를 찾는다. 여기서 `pi`는 true cluster와 predicted cluster 사이의 일대일 대응이다.

정렬된 예측 라벨을

```text
\tilde z_i = pi^{-1}(\hat z_i)
```

처럼 true label 공간에 맞춘 뒤, 오분류율은

```text
misclassification_rate
  = (1/n) sum_{i=0}^{n-1} 1{\tilde z_i != z_i}.
```

값이 작을수록 좋고, 0이면 모든 노드를 맞춘 것이다.

### ARI

ARI는 adjusted Rand index이다. 두 clustering이 모든 노드 쌍에 대해 같은 군집/다른 군집 판단을 얼마나 비슷하게 하는지 측정하되, 무작위 clustering에서 기대되는 Rand index를 보정한다.

true clustering의 cluster 크기를 `a_i`, predicted clustering의 cluster 크기를 `b_j`, 교차표 원소를

```text
n_{ij} = |{v : z_v = i, \hat z_v = j}|
```

라고 하자. 전체 노드 수는 `n`이다. 그러면

```text
ARI =
[
  sum_{ij} C(n_{ij}, 2)
  - {sum_i C(a_i, 2) sum_j C(b_j, 2)} / C(n, 2)
]
/
[
  0.5 * {sum_i C(a_i, 2) + sum_j C(b_j, 2)}
  - {sum_i C(a_i, 2) sum_j C(b_j, 2)} / C(n, 2)
].
```

해석은 다음과 같다.

- `ARI = 1`: 두 clustering이 완전히 같다.
- `ARI ≈ 0`: 무작위 수준과 비슷하다.
- `ARI < 0`: 무작위보다 나쁠 수 있다.

코드에서는 `sklearn.metrics.adjusted_rand_score(y_true, y_pred)`를 사용한다. ARI는 label permutation에 영향을 받지 않으므로 Hungarian 정렬 전 라벨을 그대로 넣어도 된다.

### NMI

NMI는 normalized mutual information이다. true label 확률변수를 `Z`, predicted label 확률변수를 `\hat Z`라고 하면 mutual information은

```text
I(Z; \hat Z)
  = sum_{a,b} P(Z=a, \hat Z=b)
      log [ P(Z=a, \hat Z=b) / {P(Z=a) P(\hat Z=b)} ].
```

엔트로피는

```text
H(Z) = -sum_a P(Z=a) log P(Z=a),
H(\hat Z) = -sum_b P(\hat Z=b) log P(\hat Z=b).
```

sklearn의 기본 `normalized_mutual_info_score`는 arithmetic normalization을 사용하므로

```text
NMI = I(Z; \hat Z) / mean(H(Z), H(\hat Z))
    = 2 I(Z; \hat Z) / {H(Z) + H(\hat Z)}.
```

값의 범위는 보통 `[0,1]`이고, 1이면 두 clustering이 완전히 같다. NMI도 label permutation에 영향을 받지 않는다.

### 하이퍼그래프 밀도 지표

실험은 성능 지표 외에도 생성된 하이퍼그래프의 밀도를 기록한다.

하이퍼그래프 degree:

```text
d_i^H = sum_{e in E} 1{i in e}.
```

평균 degree:

```text
degree_mean = (1/n) sum_i d_i^H = m |E| / n.
```

최대 degree:

```text
degree_max = max_i d_i^H.
```

고립 노드 수와 비율:

```text
num_isolated_nodes = sum_i 1{d_i^H = 0},
isolated_fraction = num_isolated_nodes / n.
```

하이퍼엣지 수:

```text
num_hyperedges_total = |E|.
```

`K변화.ipynb`에서는 추가로 다음 이론적 기대값도 기록한다.

```text
expected_hyperedges_total = N_within p_in + N_mixed p_out,
expected_hyperedges_per_n = expected_hyperedges_total / n,
expected_degree_mean = m * expected_hyperedges_total / n,
candidate_within_fraction = N_within / N_total.
```

### 시간과 메모리 지표

각 반복은 다음 시간을 기록한다.

- `generation_wall_sec`: HSBM 하이퍼엣지 생성 시간
- `zhou_laplacian_wall_sec` 또는 `zhou_theta_build_wall_sec`: Zhou Laplacian/Theta 구성 시간
- `eigen_decomposition_wall_sec`: `scipy.sparse.linalg.eigsh`로 top-`K` 고유벡터를 계산하고 정렬하는 시간
- `embedding_normalize_wall_sec`: row normalization 시간
- `kmeans_wall_sec`: k-means 시간
- `spectral_clustering_wall_sec`: eigensolver, normalization, k-means를 포함한 spectral clustering 시간
- `metric_wall_sec`: misclassification, ARI, NMI 계산 시간
- `algorithm_total_wall_sec`: 생성, Zhou 구성, eigen, normalization, k-means 주요 단계 시간의 합
- `wall_clock_sec`: `measure_call`로 감싼 전체 반복의 wall-clock 시간
- `cpu_time_sec`: 전체 반복의 process CPU time

메모리는 다음을 기록한다.

- `peak_traced_memory_mb`: `tracemalloc`이 추적한 peak Python allocation
- `rss_before_mb`: 반복 시작 전 process resident set size
- `rss_after_mb`: 반복 종료 후 process resident set size
- `rss_delta_mb = rss_after_mb - rss_before_mb`

`rss_delta_mb`는 운영체제 메모리 allocator와 garbage collection의 영향을 받기 때문에 음수가 나올 수 있다. 따라서 절대적인 peak memory 해석에는 `peak_traced_memory_mb`와 함께 봐야 한다.

## 노트북별 하이퍼파라미터

### 공통 하이퍼파라미터

세 노트북이 공통으로 사용하는 주요 하이퍼파라미터는 다음과 같다.

| 이름 | 의미 | 기본값 |
|---|---|---:|
| `m` | 모든 하이퍼엣지의 크기 | `3` |
| `a_in` | within-community edge 확률 상수 | `36.0` |
| `b_out` | mixed edge 확률 상수 | `4.0` |
| `sampling` | 하이퍼엣지 생성 방식 | `"sparse"` |
| `max_enumeration` | `auto` 모드에서 exact 열거 허용 후보 수 | `1_500_000` |
| `normalize_embedding_rows` | spectral embedding row L2 정규화 여부 | `True` |
| `eigsh_tol` | sparse eigensolver tolerance | `1e-6` |
| `rp_oversampling` | 가우시안 랜덤 프로젝션 oversampling rank | `160` |
| `rp_power_iter` | 가우시안 랜덤 프로젝션 power iteration 파라미터 `q` | `4` |
| `random_sampling_p` | 랜덤 샘플링에서 `Theta` entry를 유지할 확률 | `0.7` |
| `reps` | 각 설정 반복 횟수 | `10` |

모델 확률은 항상

```text
p_in  = a_in  * rho_n / n^{m-1},
p_out = b_out * rho_n / n^{m-1}.
```

### n변화

[n변화.ipynb](./n변화.ipynb)는 `n`만 바꾼다.

고정값:

```text
K = 3
m = 3
a_in = 36.0
b_out = 4.0
rho_n = 4.0
reps = 10
seed = 20260426
```

sweep:

```text
n in {1000, 2000, 3000, 4000, 5000,
      6000, 7000, 8000, 9000, 10000}.
```

이때 확률은 `n`에 따라

```text
p_in(n)  = 36 * 4 / n^2 = 144 / n^2,
p_out(n) = 4  * 4 / n^2 = 16  / n^2.
```

### rho_n변화

[rho_n변화.ipynb](./rho_n변화.ipynb)는 `rho_n`만 바꾼다.

고정값:

```text
n = 5000
K = 3
m = 3
a_in = 36.0
b_out = 4.0
reps = 10
seed = 20260427
```

sweep:

```text
rho_n in {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0}.
```

이때 확률은

```text
p_in(rho_n)  = 36 * rho_n / 5000^2,
p_out(rho_n) = 4  * rho_n / 5000^2.
```

`rho_n`은 하이퍼그래프 density를 직접 키우는 파라미터다. 다른 값이 고정되어 있을 때 `rho_n`을 두 배로 키우면 `p_in`, `p_out`도 두 배가 되고, 기대 하이퍼엣지 수와 기대 평균 degree도 거의 선형으로 증가한다.

### K변화

[K변화.ipynb](./K변화.ipynb)는 `K`만 바꾼다.

고정값:

```text
n = 5000
m = 3
a_in = 36.0
b_out = 4.0
rho_n = 8.0
reps = 10
seed = 20260427
kmeans_n_init = 20
```

sweep:

```text
K in {2, 3, 4, 5, 6, 8, 10, 12}.
```

이때 확률 자체는 `K`에 의존하지 않는다.

```text
p_in  = 36 * 8 / 5000^2 = 288 / 5000^2,
p_out = 4  * 8 / 5000^2 = 32  / 5000^2.
```

하지만 `K`가 커지면 balanced community size `n_k`가 작아지고, 따라서

```text
N_within = sum_k C(n_k, 3)
```

이 감소한다. 즉 `K` sweep은 순수하게 “군집 수만 어려워지는 효과”가 아니라, within 후보 비율과 기대 평균 degree가 함께 바뀌는 실험이다. 이 때문에 `K변화.ipynb`는 `expected_degree_mean`과 `candidate_within_fraction`을 함께 저장한다.

## 결과 저장 위치

각 노트북은 실행 결과를 아래 구조로 저장한다.

```text
experiments/균일 HSBM 실험/results/{EXPERIMENT_ID}_{EXPERIMENT_SLUG}/
```

현재 결과 폴더:

- `EXP-20260426-004_uniform_hsbm_n_scaling_zhou_laplacian/`
  - `n` 변화 실험의 raw CSV, summary CSV, plot을 저장한다.
- `EXP-20260427-001_uniform_hsbm_rho_schedule_probe/`
  - `rho_n=1`, `rho_n=log(n)`, `rho_n=2log(n)`를 빠르게 비교한 탐색용 probe 결과다.
- `EXP-20260427-001_uniform_hsbm_rho_n_sweep_zhou_theta/`
  - `n=5000`에서 다른 하이퍼파라미터를 고정하고 `rho_n`만 변화시킨 정식 sweep 결과다.
- `EXP-20260427-002_uniform_hsbm_K_sweep_zhou_theta/`
  - `n=5000`, `rho_n=8.0`에서 다른 하이퍼파라미터를 고정하고 `K`만 변화시킨 정식 sweep 결과다.
- `EXP-20260428-001_uniform_hsbm_K_sweep_gaussian_random_projection/`
  - `K` 변화 실험의 가우시안 랜덤 프로젝션 결과다.
- `EXP-20260428-002_uniform_hsbm_K_sweep_random_sampling/`
  - `K` 변화 실험의 랜덤 샘플링 결과다.
- `EXP-20260428-003_uniform_hsbm_n_scaling_gaussian_random_projection/`
  - `n` 변화 실험의 가우시안 랜덤 프로젝션 결과다.
- `EXP-20260428-004_uniform_hsbm_n_scaling_random_sampling/`
  - `n` 변화 실험의 랜덤 샘플링 결과다.
- `EXP-20260428-005_uniform_hsbm_rho_n_sweep_gaussian_random_projection/`
  - `rho_n` 변화 실험의 가우시안 랜덤 프로젝션 결과다.
- `EXP-20260428-006_uniform_hsbm_rho_n_sweep_random_sampling/`
  - `rho_n` 변화 실험의 랜덤 샘플링 결과다.

## 해석상 주의점

- `Theta = I - Delta`의 가장 큰 고유값 고유벡터를 쓰는 구현은 `Delta`의 가장 작은 고유값 고유벡터를 쓰는 구현과 같은 eigenspace를 사용한다.
- 오분류율은 예측 cluster label을 Hungarian matching으로 true label에 최적으로 정렬한 뒤 계산한다.
- ARI와 NMI는 label permutation에 불변이므로 정렬하지 않은 `y_true`, `y_pred`를 그대로 넣는다.
- `rho_n`은 평균 incidence degree를 직접 키우는 sparsity/density 파라미터다.
- `K` sweep에서 `a_in`, `b_out`, `rho_n`을 고정하면 `K` 증가에 따라 within-community 후보 비율과 평균 degree도 함께 변한다. 따라서 `K` sweep 결과는 순수한 군집 수 효과와 밀도 변화 효과가 섞여 있다.
- 중요한 하이퍼파라미터를 바꿔 같은 결과 폴더에 다시 저장하면 이전 결과 해석이 어려워진다. 설정을 바꿔 새 실험을 만들 때는 새 `EXPERIMENT_ID`와 새 결과 폴더를 사용한다.

## 작성 규칙

앞으로 이 폴더의 README, 마크다운 설명, 코드 주석은 기본적으로 한글로 작성한다. 모델명, 함수명, 지표명은 필요하면 영어를 그대로 사용한다.

실험 노트북, 결과 저장 구조, 핵심 알고리즘 선택, 하이퍼파라미터 sweep 등 이 폴더의 내용에 변화가 생기면 반드시 이 README에 함께 반영한다.
