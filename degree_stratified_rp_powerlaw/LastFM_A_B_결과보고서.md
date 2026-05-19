# LastFM Asia Normalized Adjacency 실험 결과보고서

## 0. 요약

이번 실험은 LastFM Asia 그래프에서 normalized adjacency를 사용했을 때 표준 Gaussian Randomized Projection과 Degree-Stratified Randomized Projection의 label 기반 clustering 성능을 비교한 것이다.

핵심 결과는 다음과 같다.

- 데이터셋: LastFM Asia, 7,624 nodes, 27,806 edges, 18 classes
- Operator: normalized adjacency, `S = D^{-1/2} A D^{-1/2}`
- 비교 방법: Gaussian Randomized Projection vs Degree-Stratified Randomized Projection
- Sweep: `r = 0, 2, 5, 10, 20, 40`
- 반복: 각 설정 10회

가장 중요한 결과:

| r | Gaussian RP Macro F1 | Degree-Stratified RP Macro F1 | 개선폭 |
|---:|---:|---:|---:|
| 10 | 0.2239 | **0.2757** | **+0.0518** |
| 20 | 0.2326 | **0.2965** | **+0.0639** |
| 40 | 0.2654 | **0.3500** | **+0.0846** |

특히 mid-degree node group, 즉 degree 3-8 구간에서 개선폭이 가장 크게 관찰되었다.

| r | Gaussian RP Macro F1 | Degree-Stratified RP Macro F1 | 개선폭 |
|---:|---:|---:|---:|
| 10 | 0.2003 | **0.2685** | **+0.0682** |
| 20 | 0.2123 | **0.2904** | **+0.0781** |
| 40 | 0.2402 | **0.3507** | **+0.1105** |

주요 관찰은 다음과 같다.

> **Normalized adjacency 조건에서 Degree-Stratified RP는 `r=10,20,40`에서 Gaussian RP보다 높은 Macro F1, ARI, NMI를 보였다. 특히 degree 3-8의 mid-degree node group에서 개선폭이 가장 컸다.**

---

## 1. 방법론 배경

### 1.1 문제 상황

실세계 그래프는 degree 분포가 균일하지 않은 경우가 많다. LastFM Asia 같은 social network도 소수의 high-degree node와 다수의 low/mid-degree node가 함께 존재한다.

이런 power-law graph에서 spectral clustering 또는 spectral embedding을 할 때, 표준 randomized projection은 다음 한계를 가질 수 있다.

1. Random test matrix가 모든 node coordinate를 같은 방식으로 다룬다.
2. High-degree 구조가 leading spectral direction에 큰 영향을 줄 수 있다.
3. Low/mid-degree node에 있는 label signal이 sketch 과정에서 충분히 보존되지 않을 수 있다.

표준 Gaussian RP는 다음 test matrix를 사용한다.

```text
Omega ~ N(0, 1)^{n x ell}
```

여기서 모든 row, 즉 모든 node coordinate가 같은 분포에서 독립적으로 샘플링된다. 이 방식은 일반적인 low-rank approximation에는 강력하지만, degree scale별로 다른 구조가 존재하는 graph에서는 특정 degree 구간의 정보를 명시적으로 보장하지 않는다.

### 1.2 Degree-Stratified RP의 핵심 아이디어

Degree-Stratified RP는 node를 degree bucket으로 나누고, 각 bucket에 별도의 random sketch budget을 배정한다.

핵심은 다음과 같다.

```text
전체 node set V를 degree bucket B_1, ..., B_s로 분할한다.
각 bucket B_j에 sketch dimension ell_j를 배정한다.
각 bucket에서만 support를 갖는 random test matrix Omega_j를 만든다.
Y_j = S Omega_j를 계산하고, Y = [Y_1, ..., Y_s]를 사용한다.
```

이렇게 하면 전체 sketch dimension `ell`이 모든 node에 무작위로 흩어지는 대신, degree 구간별로 최소한의 representation이 보장된다.

---

## 2. 수식 정리

### 2.1 그래프와 degree

그래프의 adjacency matrix를 다음과 같이 둔다.

```text
A in R^{n x n}
```

이번 실험에서는 undirected graph를 사용하므로 `A`는 symmetric matrix이다.

Node degree vector는 다음과 같다.

```text
d = A 1
d_i = sum_j A_{ij}
```

### 2.2 Degree-tempered operator

이번 실험은 raw adjacency `A`가 아니라 normalized adjacency를 사용한다. 더 일반적으로는 다음 operator를 생각할 수 있다.

```text
S_{alpha,tau} = D_tau^{-alpha} A D_tau^{-alpha}
D_tau = D + tau I
```

여기서:

- `D`는 degree diagonal matrix이다.
- `alpha`는 degree normalization 강도이다.
- `tau`는 degree regularization 값이다.

이번 실험에서는 다음 값을 고정했다.

```text
alpha = 0.5
tau = 0
S = D^{-1/2} A D^{-1/2}
```

즉 일반적인 normalized adjacency를 사용했다.

### 2.3 Degree bucket

Degree bucket은 log-scale로 만든다.

```text
B_j = { i : 2^j <= d_i < 2^{j+1} }
```

구현에서는 다음과 같은 bucket이 생긴다.

```text
[1, 2), [2, 4), [4, 8), [8, 16), ...
```

이번 결과보고서에서는 평가를 위해 별도의 degree group도 사용했다.

| 평가 group | 조건 |
|---|---|
| `low_deg_1_2` | degree 1-2 |
| `mid_deg_3_8` | degree 3-8 |
| `high_deg_9_plus` | degree 9 이상 |

주의할 점은, 이 평가 group은 결과 해석을 위한 group이고, 알고리즘 내부 bucket은 log-degree bucket을 그대로 사용한다는 것이다.

### 2.4 Bucket mass

각 degree bucket의 mass는 다음으로 정의했다.

```text
M_j = sum_{i in B_j} d_i
```

즉 bucket 안 node들의 총 degree이다. 이 값은 해당 degree scale이 그래프에서 차지하는 총 연결량을 나타낸다.

### 2.5 Sketch dimension allocation

전체 sketch dimension은 다음과 같다.

```text
ell = k + r
```

각 bucket에 배정되는 sketch dimension은 `sqrt(M_j)`에 비례하도록 설정했다.

```text
ell_j approximately proportional to sqrt(M_j)
```

구체적으로는 모든 bucket에 최소 sketch dimension `ell_min`을 먼저 배정하고, 남은 dimension을 다음 비율로 나눈다.

```text
weight_j = sqrt(M_j)
ell_j = ell_min + floor((ell - s ell_min) * weight_j / sum_h weight_h)
```

남는 dimension은 fractional part가 큰 bucket부터 하나씩 추가한다.

이 배분 규칙은 다음 성질을 갖는다.

- High-degree bucket에 더 많은 dimension을 준다.
- 하지만 mass에 정비례시키지 않고 square root를 쓰므로 hub bucket이 전체 budget을 독점하지 못하게 한다.
- Low/mid-degree bucket에도 최소 dimension이 보장된다.

### 2.6 Degree-Stratified sketch

Bucket `B_j`에 대해, 해당 bucket에서만 support를 갖는 random matrix를 만든다.

```text
Omega_j in R^{n x ell_j}
```

조건은 다음과 같다.

```text
(Omega_j)_{i,t} = 0, if i notin B_j
(Omega_j)_{i,t} ~ N(0,1), if i in B_j
```

실제 구현에서는 dense `n x ell_j` matrix를 만들지 않고, 다음처럼 계산한다.

```text
G_j ~ N(0,1)^{|B_j| x ell_j}
Y_j = S[:, B_j] G_j
```

전체 sketch는 bucket별 sketch를 concatenate한다.

```text
Y = [Y_1, Y_2, ..., Y_s]
```

### 2.7 Power iteration

Power iteration은 spectral gap이 작거나 leading eigenspace가 불안정할 때 approximation을 안정화한다.

이번 구현에서는 symmetric operator `S`에 대해 다음 흐름을 사용한다.

```text
Y <- S Omega
for t = 1, ..., q:
    Y <- S (S Y)
    Y <- orth(Y)
```

따라서 대략적으로 다음 subspace를 잡는다.

```text
range(S^{2q+1} Omega)
```

이번 실험에서는:

```text
q = 1
```

### 2.8 Rayleigh-Ritz projection

Sketch matrix `Y`를 orthonormalize한다.

```text
Q = orth(Y)
```

이후 작은 core matrix를 만든다.

```text
B = Q^T S Q
```

`B`는 `ell x ell` matrix이므로 eigendecomposition이 싸다.

```text
B = Utilde Lambda Utilde^T
```

원래 공간의 approximate eigenvectors는 다음처럼 얻는다.

```text
U_k = Q Utilde_{1:k}
```

마지막으로 `U_k`의 row를 normalize하고 k-means를 적용한다.

---

## 3. 하이퍼파라미터 설명

| 이름 | 이번 값 | 의미 | 커지면 생기는 효과 |
|---|---:|---|---|
| `k` | 18 | 사용할 eigenvector 수이자 clustering class 수 | 더 많은 spectral direction을 쓰지만 noise와 계산량 증가 |
| `r` | 0, 2, 5, 10, 20, 40 | oversampling dimension | sketch가 안정되지만 계산량 증가 |
| `ell` | `k+r` | 전체 sketch dimension | 실제 random projection 공간 크기 |
| `q` | 1 | power iteration 횟수 | leading eigenspace 근사가 개선될 수 있으나 matrix multiply 증가 |
| `alpha` | 0.5 | degree normalization 강도 | degree hub 효과를 줄임 |
| `tau` | 0 | degree regularization | 이번 실험에서는 pure normalized adjacency 사용 |
| `ell_min` | 1 | bucket별 최소 sketch dimension | 작은 bucket이 완전히 무시되지 않음 |
| `reps` | 10 | randomized method 반복 수 | 시행별 변동 측정 |
| `kmeans_n_init` | 20 | k-means restart 횟수 | clustering 안정성 증가, 시간 증가 |
| `normalize_embedding_rows` | True | k-means 전 row normalization 여부 | spectral clustering에서 일반적으로 사용 |

이번 실험에서 가장 중요한 sweep은 `r`이다.

```text
r = 0, 2, 5, 10, 20, 40
```

`r`은 전체 sketch dimension `ell = k + r`을 결정한다. 이번 실험에서는 작은 sketch budget부터 비교하기 위해 `r=0,2,5`를 포함했고, 더 넉넉한 budget에서의 변화를 보기 위해 `r=10,20,40`도 함께 측정했다.

---

## 4. 실험 세팅

### 4.1 데이터셋

사용한 데이터셋은 SNAP LastFM Asia Social Network이다.

```text
data/lastfm_asia/lastfm_asia.zip
```

데이터 구성:

| 파일 | 내용 |
|---|---|
| `lastfm_asia_edges.csv` | LastFM 사용자 간 mutual follower edge |
| `lastfm_asia_target.csv` | 사용자 country 기반 multi-class label |

그래프 통계:

| 항목 | 값 |
|---|---:|
| Node 수 | 7,624 |
| Edge 수 | 27,806 |
| Label class 수 | 18 |
| Largest connected component | 전체 7,624 nodes |
| Degree min | 1 |
| Degree median | 4 |
| Degree mean | 7.294 |
| Degree q75 | 8 |
| Degree q90 | 16 |
| Degree q99 | 55.77 |
| Degree max | 216 |
| Degree Gini | 0.583 |
| Tail log-log CCDF R² | 0.973 |

Degree Gini와 tail R²가 높으므로 power-law 성격이 충분히 있다.

### 4.2 비교 방법

이번 실험에서는 두 방법을 비교했다.

| 방법 | 설명 |
|---|---|
| Gaussian RP | 전체 node coordinate에 대해 dense Gaussian test matrix를 사용 |
| Degree-Stratified RP | degree bucket별로 support가 제한된 Gaussian sketch를 만들고 bucket mass 기반으로 dimension 배정 |

이번 실험에서는 normalized adjacency 조건인 `alpha=0.5, tau=0`을 고정하고 `r`에 따른 변화를 측정했다.

### 4.3 평가 지표

Label 기반 clustering 지표를 사용했다.

| 지표 | 의미 |
|---|---|
| Macro F1 | Hungarian matching 후 class별 F1을 평균 |
| ARI | Adjusted Rand Index |
| NMI | Normalized Mutual Information |

전체 node뿐 아니라 degree group별로 같은 지표를 계산했다.

| Group | 조건 | Node 수 | 포함 class 수 |
|---|---|---:|---:|
| `all` | 전체 node | 7,624 | 18 |
| `low_deg_1_2` | degree 1-2 | 2,942 | 18 |
| `mid_deg_3_8` | degree 3-8 | 2,847 | 18 |
| `high_deg_9_plus` | degree 9 이상 | 1,835 | 17 |

### 4.4 실행 명령

```powershell
cd C:\Users\WWindows10\Documents\github_project\python-rand-nla-research\degree_stratified_rp_powerlaw

py .\lastfm_degree_stratum_experiment.py `
  --dataset-path ..\data\lastfm_asia\lastfm_asia.zip `
  --dataset-name lastfm-asia `
  --r-values 0,2,5,10,20,40 `
  --alpha 0.5 `
  --tau 0 `
  --q 1 `
  --reps 10 `
  --outdir results\lastfm_degree_stratum_alpha05_tau0
```

결과 폴더:

```text
degree_stratified_rp_powerlaw/results/lastfm_degree_stratum_alpha05_tau0/
```

---

## 5. 전체 Node 결과

전체 node 기준 결과는 다음과 같다.

| r | Gaussian F1 | DS-RP F1 | Gaussian ARI | DS-RP ARI | Gaussian NMI | DS-RP NMI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2133 | 0.2050 | 0.0644 | **0.0807** | 0.1475 | **0.1795** |
| 2 | **0.2229** | 0.2144 | 0.0703 | **0.0873** | 0.1535 | **0.1987** |
| 5 | 0.2175 | **0.2256** | 0.0665 | **0.0870** | 0.1492 | **0.1996** |
| 10 | 0.2239 | **0.2757** | 0.0726 | **0.1164** | 0.1592 | **0.2427** |
| 20 | 0.2326 | **0.2965** | 0.0808 | **0.1377** | 0.1689 | **0.2713** |
| 40 | 0.2654 | **0.3500** | 0.1019 | **0.1995** | 0.1967 | **0.3431** |

핵심 관찰:

- Gaussian RP는 `r`이 커져도 Macro F1이 0.21-0.27 수준에 머문다.
- DS-RP의 Macro F1은 `r=10` 이후 증가 폭이 커진다.
- **`r=40`에서 DS-RP Macro F1은 0.3500으로 Gaussian RP 0.2654보다 +0.0846 높다.**
- **`r=40`에서 DS-RP ARI는 0.1995로 Gaussian RP 0.1019보다 +0.0976 높다.**

---

## 6. Paired Difference

아래 표는 같은 반복 번호끼리 맞춘 차이이다.

```text
차이 = Degree-Stratified RP - Gaussian RP
```

| r | Macro F1 차이 | 차이 표준편차 | ARI 차이 | ARI 표준편차 | NMI 차이 |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0083 | 0.0247 | +0.0164 | 0.0182 | +0.0320 |
| 2 | -0.0085 | 0.0318 | +0.0170 | 0.0174 | +0.0452 |
| 5 | +0.0081 | 0.0209 | +0.0205 | 0.0158 | +0.0504 |
| 10 | **+0.0518** | 0.0253 | **+0.0438** | 0.0245 | **+0.0836** |
| 20 | **+0.0639** | 0.0270 | **+0.0569** | 0.0169 | **+0.1024** |
| 40 | **+0.0846** | 0.0256 | **+0.0976** | 0.0211 | **+0.1463** |

`r=10,20,40`에서는 Macro F1 개선폭이 시행별 표준편차보다 크다. 이 구간에서는 평균 개선폭이 반복 간 변동보다 크게 관찰된다.

---

## 7. Low-Degree Node 결과

Degree 1-2 node만 따로 평가했다.

| r | Gaussian F1 | DS-RP F1 | Gaussian ARI | DS-RP ARI | Gaussian NMI | DS-RP NMI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.1451** | 0.1362 | **0.0243** | 0.0203 | 0.0817 | **0.0934** |
| 2 | **0.1506** | 0.1504 | **0.0248** | 0.0245 | 0.0821 | **0.1063** |
| 5 | 0.1487 | **0.1584** | **0.0249** | 0.0247 | 0.0812 | **0.1055** |
| 10 | 0.1460 | **0.1869** | 0.0255 | **0.0350** | 0.0838 | **0.1275** |
| 20 | 0.1575 | **0.2000** | 0.0296 | **0.0435** | 0.0903 | **0.1443** |
| 40 | 0.1767 | **0.2486** | 0.0367 | **0.0742** | 0.1049 | **0.1952** |

Low-degree group은 구조 정보 자체가 적어서 절대 성능은 낮다. `r=10` 이후에는 DS-RP의 평균 Macro F1, ARI, NMI가 Gaussian RP보다 높다.

**`r=40`에서 low-degree Macro F1은 0.1767에서 0.2486으로 상승했다.**

---

## 8. Mid-Degree Node 결과

Degree 3-8 node만 따로 평가했다.

| r | Gaussian F1 | DS-RP F1 | Gaussian ARI | DS-RP ARI | Gaussian NMI | DS-RP NMI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1878 | **0.1931** | 0.0454 | **0.0679** | 0.1363 | **0.1831** |
| 2 | 0.1978 | **0.2086** | 0.0502 | **0.0747** | 0.1435 | **0.2016** |
| 5 | 0.1976 | **0.2179** | 0.0476 | **0.0751** | 0.1407 | **0.2048** |
| 10 | 0.2003 | **0.2685** | 0.0539 | **0.1041** | 0.1514 | **0.2499** |
| 20 | 0.2123 | **0.2904** | 0.0610 | **0.1271** | 0.1617 | **0.2810** |
| 40 | 0.2402 | **0.3507** | 0.0783 | **0.1973** | 0.1870 | **0.3656** |

Mid-degree group에서 개선폭이 가장 크게 관찰되었다.

- **`r=40`에서 Macro F1 개선폭은 +0.1105이다.**
- **`r=40`에서 ARI 개선폭은 +0.1191이다.**
- **`r=40`에서 NMI 개선폭은 +0.1786이다.**

이 그룹은 node 수가 2,847개이고 모든 label class가 존재한다. 세 degree group 중 개선폭이 가장 크게 관찰되었다.

---

## 9. High-Degree Node 결과

Degree 9 이상 node만 따로 평가했다.

| r | Gaussian F1 | DS-RP F1 | Gaussian ARI | DS-RP ARI | Gaussian NMI | DS-RP NMI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.3373** | 0.3042 | 0.2421 | **0.3082** | 0.3919 | **0.4318** |
| 2 | **0.3509** | 0.3141 | 0.2686 | **0.3132** | 0.4047 | **0.4565** |
| 5 | **0.3559** | 0.3331 | 0.2508 | **0.3092** | 0.3981 | **0.4554** |
| 10 | 0.3627 | **0.4042** | 0.2744 | **0.3953** | 0.4200 | **0.5327** |
| 20 | 0.3743 | **0.4321** | 0.2930 | **0.4389** | 0.4315 | **0.5736** |
| 40 | 0.4242 | **0.4819** | 0.3542 | **0.5334** | 0.4877 | **0.6548** |

High-degree group은 작은 `r`에서는 Gaussian RP의 F1이 높지만, ARI와 NMI는 처음부터 DS-RP가 더 높다. `r=10` 이후에는 F1도 DS-RP가 더 높다.

**`r=40`에서 high-degree ARI는 Gaussian RP 0.3542, DS-RP 0.5334로 측정되었다.**

---

## 10. 결과 해석

이번 실험은 operator를 `D^{-1/2} A D^{-1/2}`로 고정했다. 이 operator는 raw adjacency보다 hub의 절대 degree 효과를 줄인다. 하지만 degree heterogeneity가 완전히 사라지는 것은 아니다.

Gaussian RP는 전체 node coordinate에 대해 하나의 global random test matrix를 사용한다.

```text
Omega ~ N(0,1)^{n x ell}
```

이 방식은 전체 eigenspace 근사에는 일반적으로 강하지만, degree scale별 representation을 명시적으로 분리하지는 않는다.

반면 DS-RP는 다음 구조를 강제한다.

```text
Omega = [Omega_1, ..., Omega_s]
support(Omega_j) subset B_j
```

따라서 low, mid, high degree bucket이 각각 sketch에 반영된다. 이번 실험에서는 mid-degree group에서 두 방법의 차이가 가장 크게 관찰되었으며, 이는 degree scale별 sketch allocation이 clustering 결과와 관련될 수 있음을 시사한다.

---

## 11. 핵심 결론

이번 실험에서는 다음 결과가 관찰되었다.

> **Normalized adjacency 조건에서 Degree-Stratified RP는 `r=10` 이후 Gaussian RP보다 높은 label 기반 clustering 지표를 보였다.**

구체적으로:

- **전체 Macro F1은 `r=40`에서 Gaussian RP 0.2654, DS-RP 0.3500이다.**
- **전체 ARI는 `r=40`에서 Gaussian RP 0.1019, DS-RP 0.1995이다.**
- **Mid-degree Macro F1은 `r=40`에서 Gaussian RP 0.2402, DS-RP 0.3507이다.**
- **Mid-degree ARI는 `r=40`에서 Gaussian RP 0.0783, DS-RP 0.1973이다.**
- **High-degree NMI는 `r=40`에서 Gaussian RP 0.4877, DS-RP 0.6548이다.**

결과는 다음과 같이 정리할 수 있다.

> **LastFM Asia의 normalized adjacency 실험에서 Degree-Stratified RP는 Gaussian RP보다 높은 label 기반 clustering 지표를 보였고, 개선폭은 mid-degree node group에서 가장 컸다.**

---

## 12. 한계와 다음 실험

이번 실험에는 다음 한계가 있다.

1. LastFM Asia 한 데이터셋에서 확인한 결과이다.
2. Degree group 경계가 고정되어 있다.
3. `tau=0`인 pure normalized adjacency에 초점을 맞췄다.
4. 성능 차이가 발생한 원인을 더 직접적으로 분석하려면 bucket별 embedding energy 또는 class별 recovery 분석이 필요하다.

다음 실험 제안:

1. 반복 수를 30회로 늘려 confidence interval을 더 안정화한다.
2. Degree group을 quantile 기준으로 다시 나눠본다.
3. `tau = 0, 0.05*mean, 0.1*mean, 0.25*mean`으로 transition point를 찾는다.
4. Hybrid sketch를 추가한다.

```text
Omega = [Omega_global, Omega_degree_stratified]
```

이 후속 실험을 통해 DS-RP가 어떤 조건에서 유효한지, Gaussian RP와 어떻게 보완 관계를 갖는지 더 명확하게 설명할 수 있다.
