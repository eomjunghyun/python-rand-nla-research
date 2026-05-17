# Reference 1 Section 8.1 MATLAB 실험 보고서

이 보고서는 기존 Section 8.1 real network accuracy experiment를 MATLAB로 다시 실행한 결과를 정리한다. 기존 방법인 Random Projection, Random Sampling, Non-random에 CountSketch와 Wang et al. (2025)의 SIGN 양방향 Nyström subspace iteration embedding을 추가했다.

## 1. 실험 설정

- 반복 횟수는 20회 기본값이며, 이번 실행의 결과 CSV에는 평균과 표준편차를 `mean(std)` 형식으로 기록했다.
- Random Projection과 CountSketch는 `q=2`, oversampling `r=10`을 사용했다.
- Random Sampling은 `p=0.7`, `p=0.8` 두 확률을 사용했다.
- SIGN Bidirectional은 첨부 논문의 SIGN 구조처럼 `A'`와 `A`를 번갈아 곱하고 QR로 양방향 subspace를 갱신한 뒤, 얻어진 left subspace에서 spectral clustering을 수행했다.
- European email과 Political blog는 ground-truth label 기준으로 평가했고, 두 statisticians 네트워크는 기존 8.1과 같이 Non-random 결과를 reference label로 둔 relative score를 사용했다.

## 2. 방법 설명

| 방법 | 의미 |
|---|---|
| Non-random | 원래 adjacency matrix에서 `eigs`로 leading eigenvectors를 직접 구하는 기준 방법 |
| Random Projection | Gaussian sketch와 power iteration으로 spectral subspace를 근사 |
| Random Sampling | edge를 확률 `p`로 남기고 `1/p`로 rescale한 sampled adjacency에서 eigenvectors 계산 |
| CountSketch | Gaussian sketch 대신 hash bucket과 sign으로 만든 sparse embedding 사용 |
| SIGN Bidirectional | Wang et al. (2025)의 generalized Nyström with subspace iteration 구조를 대칭 adjacency embedding에 적용 |

## 3. MATLAB 결과 요약

### European email network

| Method | MATLAB paper-rank F1 | MATLAB changed-rank F1 | MATLAB paper-rank NMI | MATLAB changed-rank NMI | MATLAB paper-rank ARI | MATLAB changed-rank ARI |
|---|---:|---:|---:|---:|---:|---:|
| Random Projection | 0.305(0.018) | 0.318(0.016) | 0.536(0.011) | 0.544(0.008) | 0.081(0.010) | 0.095(0.008) |
| Random Sampling (p=0.7) | 0.305(0.015) | 0.311(0.014) | 0.517(0.010) | 0.529(0.008) | 0.068(0.008) | 0.076(0.010) |
| Random Sampling (p=0.8) | 0.318(0.017) | 0.322(0.016) | 0.530(0.011) | 0.540(0.010) | 0.071(0.007) | 0.079(0.008) |
| CountSketch | 0.312(0.019) | 0.320(0.014) | 0.539(0.012) | 0.543(0.007) | 0.085(0.011) | 0.093(0.009) |
| SIGN Bidirectional | 0.307(0.018) | 0.315(0.019) | 0.534(0.010) | 0.539(0.010) | 0.082(0.010) | 0.089(0.013) |
| Non-random | 0.335(0.016) | 0.343(0.016) | 0.549(0.007) | 0.565(0.005) | 0.078(0.007) | 0.089(0.004) |

### Political blog network

| Method | MATLAB paper-rank F1 | MATLAB changed-rank F1 | MATLAB paper-rank NMI | MATLAB changed-rank NMI | MATLAB paper-rank ARI | MATLAB changed-rank ARI |
|---|---:|---:|---:|---:|---:|---:|
| Random Projection | 0.576(0.000) | 0.436(0.005) | 0.178(0.000) | 0.044(0.008) | 0.080(0.000) | 0.002(0.001) |
| Random Sampling (p=0.7) | 0.572(0.006) | 0.506(0.008) | 0.177(0.006) | 0.130(0.006) | 0.077(0.004) | 0.040(0.004) |
| Random Sampling (p=0.8) | 0.572(0.006) | 0.503(0.008) | 0.177(0.005) | 0.128(0.006) | 0.078(0.004) | 0.039(0.004) |
| CountSketch | 0.576(0.000) | 0.436(0.003) | 0.178(0.000) | 0.044(0.006) | 0.080(0.000) | 0.002(0.001) |
| SIGN Bidirectional | 0.566(0.031) | 0.520(0.040) | 0.170(0.033) | 0.135(0.039) | 0.074(0.019) | 0.048(0.023) |
| Non-random | 0.576(0.000) | 0.502(0.000) | 0.178(0.000) | 0.127(0.000) | 0.080(0.000) | 0.038(0.000) |

### Statisticians coauthor network (No true labels)

| Method | MATLAB paper-rank F1 | MATLAB changed-rank F1 | MATLAB paper-rank NMI | MATLAB changed-rank NMI | MATLAB paper-rank ARI | MATLAB changed-rank ARI |
|---|---:|---:|---:|---:|---:|---:|
| Random Projection | 0.995(0.021) | 0.847(0.211) | 0.990(0.043) | 0.741(0.309) | 0.993(0.033) | 0.735(0.331) |
| Random Sampling (p=0.7) | 0.899(0.145) | 0.758(0.260) | 0.798(0.136) | 0.684(0.204) | 0.855(0.141) | 0.725(0.231) |
| Random Sampling (p=0.8) | 0.955(0.031) | 0.837(0.223) | 0.881(0.072) | 0.761(0.215) | 0.921(0.058) | 0.787(0.241) |
| CountSketch | 0.998(0.007) | 0.876(0.173) | 0.996(0.019) | 0.776(0.275) | 0.997(0.011) | 0.779(0.296) |
| SIGN Bidirectional | 0.726(0.203) | 0.810(0.158) | 0.513(0.273) | 0.622(0.255) | 0.529(0.301) | 0.649(0.277) |

### Statisticians citation network (No true labels)

| Method | MATLAB paper-rank F1 | MATLAB changed-rank F1 | MATLAB paper-rank NMI | MATLAB changed-rank NMI | MATLAB paper-rank ARI | MATLAB changed-rank ARI |
|---|---:|---:|---:|---:|---:|---:|
| Random Projection | 0.987(0.005) | 0.856(0.141) | 0.945(0.023) | 0.665(0.303) | 0.977(0.012) | 0.711(0.292) |
| Random Sampling (p=0.7) | 0.935(0.015) | 0.884(0.100) | 0.792(0.036) | 0.692(0.196) | 0.897(0.022) | 0.791(0.203) |
| Random Sampling (p=0.8) | 0.947(0.013) | 0.840(0.122) | 0.829(0.023) | 0.613(0.238) | 0.915(0.015) | 0.700(0.247) |
| CountSketch | 0.987(0.005) | 0.910(0.124) | 0.946(0.019) | 0.783(0.264) | 0.977(0.009) | 0.823(0.259) |
| SIGN Bidirectional | 0.717(0.139) | 0.680(0.164) | 0.485(0.125) | 0.435(0.177) | 0.629(0.121) | 0.532(0.194) |

### 주요 관찰

- European email에서는 rank 42에서 rank 30으로 줄였을 때 대부분의 MATLAB 지표가 유지되거나 올라갔다. CountSketch는 Random Projection과 거의 같은 수준의 F1/NMI/ARI를 냈고, SIGN Bidirectional도 비슷한 범위에 머물렀다.
- Political blog는 기존 Python 재현과 마찬가지로 rank 2가 자연스럽고, rank 5에서는 Random Projection과 CountSketch의 ARI가 거의 사라졌다. SIGN은 rank 5에서 Random Projection/CountSketch보다 덜 무너졌지만 rank 2보다 좋지는 않았다.
- Statisticians coauthor와 citation에서는 CountSketch가 paper-rank 설정에서 Non-random reference를 매우 잘 따라갔다. 반면 SIGN Bidirectional은 이번 spectral clustering embedding 방식에서는 두 statisticians 네트워크에서 상대 점수가 낮아, low-rank approximation 품질과 clustering label 재현성이 항상 같은 방향은 아니라는 점을 보여준다.
- Rank 5 변경은 두 statisticians 네트워크에서 전반적으로 relative score를 낮췄다. 이는 기존 Python 보고서의 결론처럼, cluster count보다 큰 embedding rank가 추가 signal을 주기보다 noise 방향을 KMeans에 넣을 수 있다는 해석과 맞다.

## 4. 해석

MATLAB 결과는 Python 결과와 완전히 같은 숫자를 목표로 하지 않는다. `eigs`/ARPACK 호출, QR 부호, KMeans++ 초기화, 난수 생성기가 서로 다르기 때문이다. 따라서 해석은 절대값 하나보다 방법 간 상대적 패턴, rank 변경에 따른 변화, pairwise ARI 안정성을 중심으로 보는 것이 맞다.

CountSketch는 Gaussian projection보다 sketch 자체가 훨씬 희소하므로 큰 행렬에서는 메모리 이점이 있다. 다만 embedding dimension이 작고 QR/eigs가 뒤따르기 때문에 전체 시간은 데이터셋의 sparsity와 KMeans 반복에 따라 달라진다. SIGN Bidirectional은 한 번의 random sketch에서 시작하지만 매 iteration마다 `A'` 방향과 `A` 방향을 모두 갱신하므로 Random Projection보다 QR 단계가 더 자주 들어간다. 대신 양방향 subspace를 같이 정렬한다는 점이 Wang et al. (2025)의 핵심이다.

## 5. 산출물

- `results/section8_1_matlab_rank_comparison.csv`: 논문 rank와 변경 rank의 MATLAB 종합 비교
- `results/section8_1_matlab_rank_comparison.md`: 위 비교표의 Markdown 버전
- 각 실험별 `*_raw_per_rep.csv`, `*_summary_mean_std.csv`, `*_table2*_like.md`, `*_pairwise_ari_mean_matrix.csv`, `*_pairwise_ari_heatmap.png`
