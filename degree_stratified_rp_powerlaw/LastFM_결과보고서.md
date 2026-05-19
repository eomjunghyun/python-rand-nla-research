# LastFM Asia 정규화 인접행렬 alpha/tau/r Sweep 결과보고서

## 1. 실험 목적

이번 실험은 LastFM Asia 실데이터에서 다음 질문을 확인하기 위해 수행했다.

- 정규화 또는 degree-tempered adjacency operator를 쓰면 label 기반 성능이 개선되는가?
- `alpha`, `tau`, `r` 값에 따라 Gaussian RP와 Degree-stratified RP의 상대 성능이 어떻게 바뀌는가?
- Degree-stratified RP가 표준 Gaussian RP보다 확실히 유리한 조건이 있는가?

이번 실험에서는 dense full general eigensolver baseline을 제외했다. LastFM Asia 전체 그래프는 7,624 nodes라서 alpha/tau/r 조합마다 dense full eigendecomposition을 반복하는 것은 비용이 크다. 대신 label 기반 지표로 Gaussian RP와 Degree-stratified RP를 직접 비교했다.

## 2. 데이터셋

데이터셋:

```text
SNAP LastFM Asia Social Network
```

다운로드 위치:

```text
data/lastfm_asia/lastfm_asia.zip
```

SNAP 공식 설명 기준으로 LastFM Asia는 Asian country의 LastFM 사용자 네트워크이며, edge는 mutual follower 관계이고 node label은 사용자의 country field에서 나온 multi-class label이다.

실험에서 읽은 파일:

| 파일 | 역할 |
|---|---|
| `lastfm_asia_edges.csv` | undirected edge list |
| `lastfm_asia_target.csv` | node label |

그래프 통계:

| 항목 | 값 |
|---|---:|
| Node 수 | 7,624 |
| Edge 수 | 27,806 |
| Label class 수 | 18 |
| 사용한 target rank `k` | 18 |
| Degree min | 1 |
| Degree median | 4 |
| Degree mean | 7.294 |
| Degree q75 | 8 |
| Degree q90 | 16 |
| Degree q99 | 55.77 |
| Degree max | 216 |
| Degree Gini | 0.583 |
| Tail alpha rough | 2.534 |
| Tail log-log CCDF R² | 0.973 |

Tail R²가 0.973으로 높고 Gini도 0.583이라, degree heterogeneity가 뚜렷하다.

## 3. Operator 설정

다음 degree-tempered operator를 사용했다.

```text
S_{alpha,tau} = D_tau^{-alpha} A D_tau^{-alpha}
D_tau = D + tau I
```

Sweep 값:

| 하이퍼파라미터 | 값 |
|---|---|
| `alpha` | `0`, `0.25`, `0.5` |
| `tau` | `0`, `mean_degree` |
| `r` | `5`, `10`, `20`, `40` |
| `q` | `1` |
| 반복 수 | `10` |
| 비교 방법 | Gaussian RP, Degree-stratified RP |

`alpha=0`은 raw adjacency와 같다. `alpha=0.5, tau=0`은 일반적인 normalized adjacency에 해당한다. `tau=mean_degree`는 degree regularization을 넣은 경우이다.

## 4. 결과 파일

결과 폴더:

```text
degree_stratified_rp_powerlaw/results/lastfm_alpha_tau_r_sweep/
```

주요 파일:

| 파일 | 내용 |
|---|---|
| `lastfm_alpha_tau_r_sweep_raw.csv` | 반복별 원자료 |
| `lastfm_alpha_tau_r_sweep_summary.csv` | alpha/tau/r/method별 평균과 표준편차 |
| `lastfm_alpha_tau_r_sweep_paired_ds_minus_gaussian.csv` | 같은 반복 번호 기준 DS-RP와 Gaussian RP의 paired difference |
| `lastfm_alpha_tau_r_sweep_bucket_allocations.csv` | Degree-stratified RP bucket allocation |
| `lastfm_alpha_tau_r_sweep_meta.json` | 설정과 그래프 진단 |
| `viz/*.png` | r sweep 시각화 |

## 5. 전체 최고 성능 조합

Macro F1 기준 상위 조합은 다음과 같다.

| 순위 | alpha | tau | r | 방법 | Macro F1 | ARI | NMI | 총 시간 sec |
|---:|---:|---|---:|---|---:|---:|---:|---:|
| 1 | 0.5 | mean | 40 | Gaussian RP | 0.3808 | 0.2317 | 0.3681 | 0.3157 |
| 2 | 0.25 | 0 | 40 | Gaussian RP | 0.3659 | 0.1981 | 0.3566 | 0.3464 |
| 3 | 0.25 | mean | 40 | Gaussian RP | 0.3652 | 0.2179 | 0.3774 | 0.2658 |
| 4 | 0.5 | mean | 40 | Degree-stratified RP | 0.3643 | 0.2205 | 0.3791 | 0.3253 |
| 5 | 0.25 | mean | 40 | Degree-stratified RP | 0.3586 | 0.1983 | 0.3618 | 0.3153 |
| 6 | 0.5 | 0 | 40 | Degree-stratified RP | 0.3576 | 0.2041 | 0.3524 | 0.3857 |

가장 좋은 단일 조합은 `alpha=0.5`, `tau=mean_degree`, `r=40`의 Gaussian RP였다.

## 6. r Sweep 요약

Macro F1 기준으로 r이 증가할수록 대부분의 설정에서 성능이 좋아졌다.

| alpha | tau | r | Gaussian F1 | DS-RP F1 | DS - Gaussian |
|---:|---|---:|---:|---:|---:|
| 0 | 0 | 5 | 0.3111 | 0.2787 | -0.0324 |
| 0 | 0 | 10 | 0.3256 | 0.3046 | -0.0210 |
| 0 | 0 | 20 | 0.3411 | 0.3213 | -0.0199 |
| 0 | 0 | 40 | 0.3521 | 0.3416 | -0.0105 |
| 0.25 | mean | 5 | 0.3083 | 0.2953 | -0.0130 |
| 0.25 | mean | 10 | 0.3373 | 0.3253 | -0.0120 |
| 0.25 | mean | 20 | 0.3530 | 0.3473 | -0.0058 |
| 0.25 | mean | 40 | 0.3652 | 0.3586 | -0.0066 |
| 0.5 | 0 | 5 | 0.2228 | 0.2240 | +0.0011 |
| 0.5 | 0 | 10 | 0.2194 | 0.2550 | +0.0356 |
| 0.5 | 0 | 20 | 0.2354 | 0.3075 | +0.0721 |
| 0.5 | 0 | 40 | 0.2651 | 0.3576 | +0.0925 |
| 0.5 | mean | 5 | 0.3077 | 0.2818 | -0.0259 |
| 0.5 | mean | 10 | 0.3302 | 0.3101 | -0.0201 |
| 0.5 | mean | 20 | 0.3421 | 0.3323 | -0.0098 |
| 0.5 | mean | 40 | 0.3808 | 0.3643 | -0.0166 |

## 7. DS-RP가 유리한 구간

DS-RP가 Gaussian RP보다 가장 크게 좋아진 구간은 `alpha=0.5`, `tau=0`이다.

| 지표 | alpha | tau | r | DS - Gaussian 평균 | 차이 표준편차 |
|---|---:|---|---:|---:|---:|
| ARI | 0.5 | 0 | 40 | +0.1011 | 0.0183 |
| ARI | 0.5 | 0 | 20 | +0.0638 | 0.0113 |
| ARI | 0.5 | 0 | 10 | +0.0431 | 0.0195 |
| Macro F1 | 0.5 | 0 | 40 | +0.0925 | 0.0318 |
| Macro F1 | 0.5 | 0 | 20 | +0.0721 | 0.0131 |
| Macro F1 | 0.5 | 0 | 10 | +0.0356 | 0.0138 |
| NMI | 0.5 | 0 | 40 | +0.1546 | 0.0192 |
| NMI | 0.5 | 0 | 20 | +0.1135 | 0.0116 |
| NMI | 0.5 | 0 | 10 | +0.0777 | 0.0190 |

이 구간에서는 개선폭이 표준편차보다 크다. 특히 `alpha=0.5, tau=0, r=20/40`에서는 DS-RP의 label 기반 성능 향상이 꽤 명확하다.

## 8. DS-RP가 불리한 구간

반대로 `tau=mean_degree`를 쓰면 대부분의 조합에서 Gaussian RP가 더 좋았다.

대표적으로 `alpha=0.5, tau=mean, r=40`에서는 다음과 같다.

| 방법 | Macro F1 | ARI | NMI |
|---|---:|---:|---:|
| Gaussian RP | 0.3808 | 0.2317 | 0.3681 |
| Degree-stratified RP | 0.3643 | 0.2205 | 0.3791 |

NMI는 DS-RP가 약간 높지만, Macro F1과 ARI는 Gaussian RP가 더 높다. 전체 최고 성능도 이 조합의 Gaussian RP였다.

## 9. 해석

이번 결과는 꽤 중요한 신호를 준다.

1. 정규화 자체는 도움이 된다.
   - raw adjacency인 `alpha=0`보다 `alpha=0.25` 또는 `alpha=0.5`에서 좋은 조합이 나왔다.
   - 특히 전체 최고 성능은 `alpha=0.5, tau=mean, r=40`이었다.

2. DS-RP는 특정 정규화 조건에서 뚜렷하게 유리하다.
   - `alpha=0.5, tau=0`에서는 Gaussian RP가 매우 약해지고, DS-RP가 큰 폭으로 개선했다.
   - 이 조건은 normalized adjacency에서 hub의 영향이 강하게 줄어든 상태이다. 이때 degree bucket별 sketch가 degree scale별 정보를 보존해 Gaussian RP보다 label 구조를 더 잘 잡은 것으로 해석할 수 있다.

3. Regularization `tau=mean`을 넣으면 Gaussian RP가 다시 강해진다.
   - `tau=mean`은 low-degree node의 불안정성을 완화하고 operator를 더 부드럽게 만든다.
   - 이 경우 global Gaussian sketch가 충분히 잘 작동해서 DS-RP의 이점이 줄어든 것으로 보인다.

4. r이 커질수록 둘 다 좋아지는 경향이 있다.
   - 특히 `r=40`에서 최고 성능이 많이 나왔다.
   - DS-RP의 이점은 `alpha=0.5, tau=0`에서 r이 커질수록 더 뚜렷해졌다.

## 10. 결론

이번 LastFM Asia sweep에서는 다음 결론을 내릴 수 있다.

- “항상 DS-RP가 Gaussian RP보다 좋다”는 결론은 아니다.
- 하지만 `alpha=0.5, tau=0`, 즉 normalized adjacency를 regularization 없이 쓰는 구간에서는 DS-RP가 Gaussian RP보다 확실히 좋았다.
- 전체 최고 성능은 `alpha=0.5, tau=mean, r=40`의 Gaussian RP였다.
- 따라서 실용적으로는 `operator 정규화/regularization`을 먼저 잘 고르는 것이 가장 중요하다.
- DS-RP는 Gaussian RP가 normalized operator에서 불안정해지는 특정 regime을 보완하는 방법으로 가능성이 있다.

## 11. 다음 실험 제안

다음 단계에서는 아래를 확인하는 것이 좋다.

1. `alpha=0.5` 고정 후 `tau`를 더 촘촘히 sweep한다.
   - `tau = 0, 0.1*mean, 0.25*mean, 0.5*mean, mean`
2. `r=20,40` 근방을 더 자세히 본다.
   - `r = 20, 30, 40, 60`
3. DS-RP의 allocation rule을 비교한다.
   - `sqrt(mass)`, uniform bucket, capped sqrt, node-count/mass 혼합
4. Hybrid sketch를 추가한다.
   - `Omega = [Omega_global, Omega_bucket]`
   - Gaussian RP의 global mixing과 DS-RP의 degree-scale 보존을 같이 쓰는 방식이다.

현재 결과만 보면 가장 유망한 후속 방향은 `alpha=0.5`에서 `tau`를 조절하면서 DS-RP와 hybrid DS-RP를 비교하는 것이다.

