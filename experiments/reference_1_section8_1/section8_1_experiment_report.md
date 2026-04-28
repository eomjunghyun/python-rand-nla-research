# Reference 1 Section 8.1 실험 보고서

이 보고서는 Reference 1 논문의 Section 8.1 real network accuracy experiment를 재현한 결과를 정리한다. 실험 대상은 네 개의 real network이며, 각 데이터셋마다 논문에서 사용한 target rank로 한 번, rank를 바꾼 설정으로 한 번 실행했다. 따라서 총 8개 실험이다.

보고서의 목적은 다음과 같다.

- 각 실험이 무엇을 재현하는지 설명한다.
- `target rank`, `embedding rank`, `cluster count`, `F1`, `NMI`, `ARI`, `pairwise ARI` 같은 용어를 정리한다.
- 논문 rank 결과와 rank 변경 결과를 비교한다.
- 표와 plot을 어떤 관점에서 해석해야 하는지 설명한다.
- 결과가 논문과 다르거나 rank 변경에 민감하게 반응한 이유를 해석한다.

## 1. 실험 개요

Section 8.1은 실제 네트워크 데이터에서 randomized spectral clustering 방법이 non-random spectral clustering과 비교해 어느 정도의 clustering 품질을 내는지 확인하는 실험이다.

사용한 방법은 세 종류다.

| 방법 | 의미 | 이 실험에서 보는 점 |
|---|---|---|
| Non-random spectral clustering | 원래 adjacency matrix에서 직접 leading eigenvectors를 구해 KMeans를 수행하는 기준 방법 | randomized 방법의 기준선 |
| Random Projection | random Gaussian projection과 power iteration으로 저차원 subspace를 근사한 뒤 clustering | 빠른 근사가 기준 방법과 비슷한 결과를 내는지 |
| Random Sampling | edge를 확률 `p`로 샘플링하고 `1/p`로 rescale한 sparse matrix에서 eigenvectors를 구함 | edge 일부만 사용해도 clustering이 유지되는지 |

실험은 20회 반복으로 수행했다. Random Projection, Random Sampling, KMeans 초기화에는 randomness가 있으므로 평균과 표준편차를 함께 기록한다. 표의 값 `0.320(0.013)`은 평균이 `0.320`, 표준편차가 `0.013`이라는 뜻이다.

## 2. 8개 실험 목록

| 번호 | 데이터셋 | 논문 rank 실험 | rank 변경 실험 | 평가 방식 |
|---:|---|---:|---:|---|
| 1 | European email network | rank 42 | rank 30 | ground-truth department label 기준 |
| 2 | Political blog network | rank 2 | rank 5 | ground-truth political label 기준 |
| 3 | Statisticians coauthor network | rank 3 | rank 5 | Non-random 결과를 reference label로 둔 relative score |
| 4 | Statisticians citation network | rank 3 | rank 5 | Non-random 결과를 reference label로 둔 relative score |

위 표에서 "논문 rank 실험"은 논문 Table 1/2의 target rank를 그대로 사용한 실험이다. "rank 변경 실험"은 clustering 군집 수는 그대로 두고 spectral embedding에 사용하는 eigenvector 개수만 바꾼 실험이다.

## 3. 중요한 용어

### 3.1 Target rank와 cluster count

이 실험에서 가장 헷갈리기 쉬운 부분은 `target rank`와 `cluster count`가 다르다는 점이다.

`target rank` 또는 `embedding rank`는 spectral embedding에 사용할 eigenvector 개수다. 예를 들어 rank 30이면 각 node를 30차원 좌표로 표현한다.

`cluster count`는 KMeans가 최종적으로 만들 군집 수다. 예를 들어 European email rank 30 실험에서는 embedding 차원은 30이지만, 최종 군집 수는 department label 수인 42개를 유지한다. 따라서 rank를 30으로 바꾸어도 "정답에 없는 추가 군집"이 생기는 것은 아니다. 추가 군집이 생기려면 KMeans의 `n_clusters` 자체를 바꿔야 한다.

### 3.2 Spectral clustering

Spectral clustering은 graph adjacency matrix의 leading eigenvectors를 사용해 node를 좌표 공간에 배치한 뒤, 그 좌표에 KMeans를 적용하는 방법이다. 직관적으로는 graph 연결 패턴이 비슷한 node들이 비슷한 eigenvector 좌표를 갖는다는 점을 이용한다.

### 3.3 Random Projection

Random Projection은 큰 matrix의 주요 spectral subspace를 직접 전부 계산하지 않고, random vector와 matrix multiplication으로 근사하는 방법이다. 이 실험에서는 `q=2`, `r=10`을 사용했다.

- `q`: power iteration 횟수와 관련된 파라미터다. 큰 eigenvalue 방향을 더 강조해 근사 품질을 높인다.
- `r`: oversampling 파라미터다. target rank보다 조금 더 넓은 subspace를 잡아 안정성을 높인다.

Random Projection의 핵심 질문은 "정확한 eigenvectors를 계산하지 않아도 clustering 결과가 충분히 비슷한가?"이다.

### 3.4 Random Sampling

Random Sampling은 edge를 확률 `p`로 남기고, 남은 edge weight를 `1/p`로 rescale한다. 기대값 관점에서 원래 adjacency matrix를 보존하려는 방식이다.

이 실험에서는 `p=0.7`, `p=0.8`을 사용했다. `p=0.8`은 edge를 더 많이 보존하므로 보통 `p=0.7`보다 원래 graph에 가까운 결과를 기대할 수 있다. 다만 실제 결과는 데이터셋 구조와 rank 설정에 따라 달라질 수 있다.

### 3.5 F1, NMI, ARI

세 지표 모두 clustering 결과가 reference label과 얼마나 가까운지 보는 지표다.

| 지표 | 범위/방향 | 의미 |
|---|---|---|
| F1 | 높을수록 좋음 | label alignment 후 class별 precision/recall 균형을 본다. 이 실험에서는 macro F1을 사용한다. |
| NMI | 0~1, 높을수록 좋음 | 두 label partition이 공유하는 정보량을 정규화한 값이다. label 이름 자체가 달라도 partition 구조가 비슷하면 높다. |
| ARI | 보통 -1~1, 높을수록 좋음 | node pair들이 같은 군집/다른 군집으로 묶이는 패턴이 reference와 얼마나 일치하는지 본다. random agreement를 보정한다. |

F1은 class별 label alignment가 중요하고, NMI/ARI는 partition 구조 비교에 더 가깝다. 따라서 세 지표가 항상 같은 방향으로 움직이지는 않는다.

### 3.6 Pairwise ARI와 heatmap

Pairwise ARI는 방법 A와 방법 B의 clustering 결과가 서로 얼마나 비슷한지 보는 값이다. ground-truth label과의 정확도와는 다른 질문이다.

예를 들어 Random Projection과 Non-random의 pairwise ARI가 높으면, Random Projection이 Non-random과 비슷한 partition을 만들었다는 뜻이다. 하지만 그 partition이 ground truth와 잘 맞는지는 별도의 F1/NMI/ARI 표를 봐야 한다.

Heatmap을 읽을 때는 다음을 보면 된다.

- 대각선은 자기 자신과의 비교라 항상 1이다.
- Non-random 행/열과 randomized methods 사이 값이 높으면 randomized method가 기준 spectral clustering을 잘 따라간다.
- Random Sampling `p=0.7`과 `p=0.8` 사이 값이 높으면 sampling probability를 조금 바꾸어도 결과가 안정적이다.
- 값이 낮거나 음수에 가까우면 두 방법이 전혀 다른 partition을 만들고 있다는 뜻이다.

## 4. 산출물 위치

주요 결과 파일은 다음 위치에 있다.

| 파일 | 설명 |
|---|---|
| `results/section8_1_table2_rank_comparison.md` | 논문 결과, 논문 rank 재현 결과, rank 변경 결과를 한 표에서 비교 |
| `results/section8_1_table2_rank_comparison.csv` | 위 비교표의 CSV 버전 |
| `results/exp8_1_email_eu_core_table2_like/` | European email rank 42 결과 |
| `results/exp8_1_email_eu_core_rank30_table2_like/` | European email rank 30 결과 |
| `results/exp8_1_political_blog_table2_like/` | Political blog rank 2 결과 |
| `results/exp8_1_political_blog_rank5_table2_like/` | Political blog rank 5 결과 |
| `results/exp8_1_statisticians_coauthor_table2_like/` | Statisticians coauthor rank 3 결과 |
| `results/exp8_1_statisticians_coauthor_rank5_table2_like/` | Statisticians coauthor rank 5 결과 |
| `results/exp8_1_statisticians_citation_table2_like/` | Statisticians citation rank 3 결과 |
| `results/exp8_1_statisticians_citation_rank5_table2_like/` | Statisticians citation rank 5 결과 |

## 5. 전체 결과 요약

### 5.1 큰 흐름

결과를 한 문장으로 요약하면, European email에서는 rank를 42에서 30으로 낮춰도 성능이 유지되거나 일부 개선되었지만, Political blog와 두 statisticians 네트워크에서는 rank를 5로 늘렸을 때 대체로 성능 또는 안정성이 떨어졌다.

이 결과는 "rank를 크게 잡으면 항상 좋아진다"가 아니라는 점을 보여준다. Spectral embedding 차원을 늘리면 더 많은 정보를 담을 수 있지만, 동시에 community structure와 무관한 noise eigenvector도 포함될 수 있다. 특히 cluster count가 작은 데이터셋에서 rank를 불필요하게 크게 잡으면 KMeans가 noise 방향에도 영향을 받아 결과가 나빠질 수 있다.

### 5.2 논문값과 재현값의 차이

European email에서는 논문의 F1 값보다 재현 F1이 상당히 높게 나왔다. 반면 NMI와 ARI는 논문값과 더 가까운 편이다. 이 차이는 다음 요인으로 설명할 수 있다.

- label alignment 방식 차이: F1은 label permutation alignment에 민감하다.
- directed graph를 undirected graph로 바꾸는 방식 차이.
- largest connected component 사용 여부와 node filtering 차이.
- KMeans 초기화 및 scikit-learn 버전 차이.
- eigen solver 구현 차이.

Political blog의 논문 rank 결과는 NMI/ARI가 논문과 거의 같은 수준이지만 F1은 낮다. 이 역시 class imbalance, label alignment, preprocessing 차이의 영향을 받을 수 있다.

Statisticians 네트워크는 true label이 없고 Non-random을 reference label로 쓰는 relative score다. 따라서 논문과 재현값 차이는 "진짜 정답과의 차이"가 아니라 "우리 구현의 randomized method가 우리 구현의 Non-random 기준을 얼마나 따라갔는지"의 차이다.

## 6. 데이터셋별 상세 해석

### 6.1 European email network

이 데이터셋은 email communication network이며, node label은 department label이다. Largest connected component 기준으로 node 986개, edge 16064개를 사용했다. 정답 class 수는 42개다.

실험 설정은 다음과 같다.

| 설정 | 값 |
|---|---:|
| 논문 rank | 42 |
| 변경 rank | 30 |
| cluster count | 42 |
| 반복 횟수 | 20 |
| Random Projection | `q=2`, `r=10` |
| Random Sampling | `p=0.7`, `p=0.8` |

결과는 다음과 같다.

| Methods | 논문 F1 | 논문 NMI | 논문 ARI | 재현 rank42 F1 | 재현 rank42 NMI | 재현 rank42 ARI | rank30 F1 | rank30 NMI | rank30 ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Projection | 0.165(0.007) | 0.558(0.006) | 0.100(0.009) | 0.315(0.018) | 0.548(0.010) | 0.091(0.008) | 0.320(0.013) | 0.551(0.009) | 0.100(0.012) |
| Random Sampling (p=0.7) | 0.126(0.007) | 0.417(0.010) | 0.059(0.008) | 0.271(0.014) | 0.505(0.011) | 0.079(0.008) | 0.308(0.013) | 0.531(0.008) | 0.077(0.009) |
| Random Sampling (p=0.8) | 0.131(0.005) | 0.436(0.010) | 0.064(0.006) | 0.287(0.016) | 0.526(0.009) | 0.091(0.009) | 0.327(0.014) | 0.543(0.005) | 0.083(0.007) |
| Non-random | 0.154(0.006) | 0.571(0.005) | 0.088(0.007) | 0.325(0.016) | 0.556(0.008) | 0.092(0.006) | 0.353(0.013) | 0.568(0.005) | 0.094(0.010) |

rank 30으로 바꾼 결과는 rank 42보다 F1과 NMI가 전반적으로 상승했다. ARI는 Random Projection과 Non-random에서 상승했고, Random Sampling에서는 약간 낮아졌다. 이 데이터셋에서는 42차원 embedding이 반드시 더 유리하지 않았고, 30차원으로 줄인 것이 noise를 일부 제거한 효과를 냈을 가능성이 있다.

Pairwise ARI heatmap을 보면 rank 42의 전체 평균 pairwise ARI는 약 0.752이고, rank 30의 전체 평균은 약 0.747이다. 즉 rank 30에서 ground-truth 지표 일부는 좋아졌지만, 방법들끼리 서로 내는 partition의 유사도는 약간 낮아졌다. 특히 Random Projection과 Non-random의 pairwise ARI는 rank 42에서 0.807, rank 30에서 0.766으로 낮아졌다. 이 말은 rank 30에서 Random Projection이 Non-random과 조금 더 다른 partition을 만들었지만, 그 partition이 department label 기준에서는 비슷하거나 더 좋은 점수를 얻었다는 뜻이다.

해석상 중요한 점은 "Non-random과 비슷한가"와 "정답 label과 잘 맞는가"가 완전히 같은 질문은 아니라는 것이다. Heatmap은 전자를, F1/NMI/ARI 표는 후자를 본다.

### 6.2 Political blog network

Political blog network는 blog 간 hyperlink network이며, label은 political leaning이다. 정답 class 수는 2개이고 논문 rank도 2다. 변경 실험에서는 embedding rank를 5로 늘렸지만 cluster count는 2로 유지했다.

| 설정 | 값 |
|---|---:|
| 논문 rank | 2 |
| 변경 rank | 5 |
| cluster count | 2 |
| 반복 횟수 | 20 |

결과는 다음과 같다.

| Methods | 논문 F1 | 논문 NMI | 논문 ARI | 재현 rank2 F1 | 재현 rank2 NMI | 재현 rank2 ARI | rank5 F1 | rank5 NMI | rank5 ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Projection | 0.641(0.004) | 0.178(0.004) | 0.079(0.006) | 0.576(0.000) | 0.178(0.000) | 0.080(0.000) | 0.435(0.004) | 0.042(0.006) | 0.002(0.001) |
| Random Sampling (p=0.7) | 0.642(0.003) | 0.177(0.007) | 0.077(0.007) | 0.572(0.009) | 0.178(0.007) | 0.077(0.006) | 0.501(0.008) | 0.125(0.006) | 0.037(0.004) |
| Random Sampling (p=0.8) | 0.641(0.004) | 0.177(0.008) | 0.077(0.009) | 0.572(0.004) | 0.178(0.005) | 0.077(0.003) | 0.499(0.007) | 0.125(0.005) | 0.037(0.003) |
| Non-random | 0.641(0.004) | 0.178(0.004) | 0.079(0.006) | 0.576(0.000) | 0.178(0.000) | 0.080(0.000) | 0.502(0.000) | 0.127(0.000) | 0.038(0.000) |

rank 2 결과는 논문의 NMI/ARI와 거의 일치한다. F1은 논문보다 낮지만, 방법 간 상대적 패턴은 비슷하다. Non-random과 Random Projection이 거의 같은 결과를 내고, sampling도 큰 손실 없이 비슷하게 따라간다.

rank 5에서는 모든 지표가 크게 하락했다. 특히 Random Projection의 ARI가 0.080에서 0.002로 거의 사라졌다. Pairwise ARI 관점에서도 rank 2는 방법 간 전체 평균이 약 0.946으로 매우 안정적이지만, rank 5는 약 0.448로 낮아진다. 더 구체적으로 rank 5에서 Random Projection과 Non-random의 pairwise ARI는 약 -0.011이다. 이는 Random Projection이 기준 방법과 사실상 무관한 partition을 냈다는 뜻이다.

이 데이터셋은 본질적으로 2개 정치 집단을 나누는 문제라 rank 2가 자연스럽다. rank 5는 추가 eigenvector가 유용한 community signal보다 noise나 부차적 구조를 KMeans에 넣어 성능을 떨어뜨린 것으로 해석할 수 있다.

### 6.3 Statisticians coauthor network

Statisticians coauthor network는 true label이 없다. 따라서 논문 방식에 맞춰 Non-random spectral clustering 결과를 reference로 두고 Random Projection/Random Sampling 결과가 그 reference를 얼마나 잘 재현하는지 평가한다. 이때 F1/NMI/ARI는 "진짜 정답 정확도"가 아니라 "Non-random 대비 유사도"다.

| 설정 | 값 |
|---|---:|
| 논문 rank | 3 |
| 변경 rank | 5 |
| cluster count | 3 |
| 반복 횟수 | 20 |

결과는 다음과 같다.

| Methods | 논문 F1 | 논문 NMI | 논문 ARI | 재현 rank3 F1 | 재현 rank3 NMI | 재현 rank3 ARI | rank5 F1 | rank5 NMI | rank5 ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Projection (relative) | 0.981(0.012) | 0.646(0.197) | 0.715(0.246) | 1.000(0.000) | 1.000(0.000) | 1.000(0.000) | 0.991(0.023) | 0.978(0.051) | 0.985(0.036) |
| Random Sampling (relative) (p=0.7) | 0.970(0.011) | 0.480(0.148) | 0.593(0.193) | 0.873(0.205) | 0.786(0.190) | 0.835(0.198) | 0.816(0.245) | 0.766(0.151) | 0.810(0.172) |
| Random Sampling (relative) (p=0.8) | 0.973(0.011) | 0.544(0.142) | 0.639(0.190) | 0.969(0.023) | 0.914(0.055) | 0.948(0.038) | 0.781(0.298) | 0.788(0.209) | 0.809(0.226) |

rank 3에서는 Random Projection이 Non-random과 완전히 같은 결과를 냈다. F1/NMI/ARI가 모두 1.000이다. Random Sampling도 대체로 높은 relative score를 보이며, `p=0.8`이 `p=0.7`보다 안정적으로 좋다.

rank 5에서는 Random Projection은 여전히 매우 높지만 완전 일치는 아니다. Random Sampling은 rank 3보다 떨어진다. 특히 `p=0.8`의 F1은 0.969에서 0.781로 크게 낮아졌다. Pairwise ARI 전체 평균도 rank 3의 약 0.897에서 rank 5의 약 0.819로 낮아졌다.

해석은 두 가지다. 첫째, Random Projection은 이 데이터셋에서 Non-random subspace를 상당히 안정적으로 근사한다. 둘째, Random Sampling은 rank를 늘렸을 때 sampling으로 생긴 perturbation과 추가 eigenvector noise가 결합되어 기준 partition과의 일치도가 낮아질 수 있다.

### 6.4 Statisticians citation network

Statisticians citation network도 true label이 없다. 따라서 coauthor와 마찬가지로 Non-random spectral clustering을 reference label로 사용한다.

| 설정 | 값 |
|---|---:|
| 논문 rank | 3 |
| 변경 rank | 5 |
| cluster count | 3 |
| 반복 횟수 | 20 |

결과는 다음과 같다.

| Methods | 논문 F1 | 논문 NMI | 논문 ARI | 재현 rank3 F1 | 재현 rank3 NMI | 재현 rank3 ARI | rank5 F1 | rank5 NMI | rank5 ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Projection (relative) | 0.990(0.021) | 0.881(0.166) | 0.926(0.140) | 0.984(0.005) | 0.936(0.021) | 0.973(0.011) | 0.898(0.124) | 0.752(0.267) | 0.794(0.267) |
| Random Sampling (relative) (p=0.7) | 0.981(0.019) | 0.759(0.125) | 0.863(0.120) | 0.937(0.010) | 0.797(0.024) | 0.899(0.016) | 0.854(0.116) | 0.634(0.218) | 0.732(0.232) |
| Random Sampling (relative) (p=0.8) | 0.981(0.022) | 0.770(0.163) | 0.861(0.149) | 0.950(0.011) | 0.830(0.025) | 0.917(0.014) | 0.835(0.132) | 0.602(0.261) | 0.685(0.266) |

rank 3 결과는 논문과 잘 맞는다. Random Projection은 Non-random reference와 매우 가깝고, Random Sampling도 높은 relative score를 보인다.

rank 5에서는 세 방법 모두 성능이 하락한다. Random Projection의 ARI는 0.973에서 0.794로, Random Sampling `p=0.8`의 ARI는 0.917에서 0.685로 낮아졌다. Pairwise ARI 전체 평균도 rank 3의 약 0.910에서 rank 5의 약 0.710으로 크게 낮아졌다.

이 데이터셋에서는 rank 5가 reference spectral partition을 더 불안정하게 만든다. citation network는 coauthor보다 rank 증가에 더 민감하게 반응했다. 추가 eigenvector가 citation graph의 주된 3-way 구조보다는 세부적인 방향이나 sparse/noisy pattern을 반영했을 가능성이 있다.

## 7. Rank 변경 실험의 결론

rank 변경 결과를 정리하면 다음과 같다.

| 데이터셋 | rank 변경 효과 | 해석 |
|---|---|---|
| European email: 42 -> 30 | F1/NMI 대체로 개선, ARI는 방법별 혼재 | 42차원보다 30차원이 일부 noise를 줄였을 가능성 |
| Political blog: 2 -> 5 | 크게 악화 | 2-class 문제에서 추가 eigenvectors가 clustering을 흐림 |
| Statisticians coauthor: 3 -> 5 | Random Projection은 유지, Sampling은 악화 | projection은 안정적이나 sampling은 rank 증가에 민감 |
| Statisticians citation: 3 -> 5 | 전반적으로 악화 | citation graph의 3-way spectral structure가 rank 5에서 불안정해짐 |

따라서 이번 결과는 "target rank는 클수록 좋다"가 아니라 "데이터셋의 cluster structure에 맞는 rank 선택이 중요하다"는 결론을 준다.

## 8. Plot 해석 가이드

현재 명시적으로 생성된 plot은 European email 실험의 `email_eu_pairwise_ari_heatmap.png`다. 이 plot은 방법별 clustering 결과를 서로 비교한 heatmap이다.

European email rank 42 heatmap:

- Random Projection과 Non-random의 평균 pairwise ARI는 약 0.807이다.
- Random Sampling `p=0.8`과 Non-random은 약 0.753이다.
- Random Sampling `p=0.7`과 Non-random은 약 0.720이다.
- 따라서 rank 42에서는 Random Projection이 기준 spectral clustering을 가장 잘 따라간다.

European email rank 30 heatmap:

- Random Projection과 Non-random의 평균 pairwise ARI는 약 0.766이다.
- Random Sampling `p=0.8`과 Non-random은 약 0.752이다.
- Random Sampling `p=0.7`과 Non-random은 약 0.734이다.
- rank 30에서는 방법 간 유사도가 전체적으로 비슷해지고, Random Projection과 Non-random의 유사도는 rank 42보다 낮아진다.

나머지 데이터셋은 pairwise ARI raw CSV가 있으므로 같은 방식으로 heatmap을 만들 수 있다. 이 값을 plot으로 그린다면 다음처럼 해석한다.

- Political blog rank 2는 방법 간 ARI가 높아 거의 같은 partition을 낸다.
- Political blog rank 5에서는 Random Projection이 다른 방법과 거의 맞지 않는다.
- Statisticians coauthor rank 3은 Random Projection이 Non-random과 거의 완전 일치한다.
- Statisticians citation rank 5는 전체 pairwise ARI가 크게 내려가므로 rank 증가가 방법 간 안정성을 낮춘다.

표 plot 또는 grouped bar plot을 그린다면 다음을 보면 된다.

- x축: method.
- y축: F1/NMI/ARI.
- 색상: 논문값, 논문 rank 재현값, rank 변경값.
- 같은 method에서 rank 변경 막대가 내려가면 rank 변경이 불리했다는 뜻이다.
- NMI와 ARI가 동시에 내려가면 partition 구조 자체가 나빠졌다는 해석이 강해진다.
- F1만 다르게 움직이면 label alignment나 class imbalance 영향을 의심해야 한다.

## 9. 보고 시 강조할 점

보고에서는 다음 메시지를 중심으로 설명하면 좋다.

1. Section 8.1은 randomized spectral clustering이 real network에서 기준 spectral clustering 또는 true label을 얼마나 잘 재현하는지 보는 실험이다.
2. `target rank`는 군집 수가 아니라 embedding 차원이다. rank를 바꾸어도 KMeans의 군집 수는 유지했다.
3. European email에서는 rank 30이 rank 42보다 일부 지표에서 좋아졌다. 즉 rank 축소가 noise 제거처럼 작동했을 가능성이 있다.
4. Political blog는 rank 2가 자연스러운 2-class 구조였고, rank 5는 성능을 크게 떨어뜨렸다.
5. Statisticians coauthor/citation은 true label이 없으므로 Non-random 기준 relative score다. rank 5는 대체로 기준 방법과의 일치도를 낮췄다.
6. Random Projection은 여러 데이터셋에서 Non-random을 잘 따라가는 편이다. 특히 coauthor rank 3에서는 완전 일치했다.
7. Random Sampling은 `p`가 커질수록 항상 좋아지는 것은 아니지만, 일반적으로 edge를 더 많이 보존하는 `p=0.8`이 안정적인 경우가 많다.
8. Pairwise ARI heatmap은 "정답과 맞는가"가 아니라 "방법끼리 같은 partition을 내는가"를 보여준다.

## 10. 한계와 주의사항

논문 결과와 재현 결과가 완전히 같지 않은 이유는 다음과 같다.

- 데이터 전처리 차이: directed graph를 undirected graph로 바꾸는 방식, isolated node 제거, largest connected component 선택 등이 다를 수 있다.
- eigensolver 차이: 논문에서 사용한 R/irlba 계열 구현과 현재 Python/scipy 구현이 다를 수 있다.
- KMeans randomness: 같은 embedding에서도 initialization에 따라 결과가 달라질 수 있다.
- label alignment 차이: F1은 predicted cluster label을 true label에 맞추는 방식에 민감하다.
- no true labels 데이터셋의 기준 차이: statisticians 네트워크는 true label이 없으므로 Non-random 결과가 reference다. Non-random 결과 자체가 구현에 따라 바뀌면 relative score도 함께 바뀐다.
- rank 변경 실험은 논문의 원래 실험이 아니라 sensitivity analysis다. 따라서 논문값과 직접 비교하기보다 rank 선택의 영향 분석으로 보는 것이 맞다.

## 11. 최종 결론

이번 8개 실험은 randomized spectral clustering의 재현성뿐 아니라 target rank 선택의 민감도를 함께 보여준다.

논문 rank를 사용한 경우, Political blog와 Statisticians citation은 논문 결과와 상당히 비슷한 패턴을 보였다. European email은 F1에서 논문과 차이가 크지만, NMI/ARI 관점에서는 비교적 가까운 값을 보인다. Statisticians coauthor/citation처럼 true label이 없는 경우에는 "정답 정확도"가 아니라 Non-random 기준의 상대적 재현성을 봐야 한다.

rank 변경 실험에서는 데이터셋별 차이가 뚜렷했다. European email은 rank를 30으로 낮췄을 때 일부 성능이 개선되었고, 나머지 세 데이터셋은 rank를 5로 늘리면 대체로 성능과 안정성이 낮아졌다. 따라서 Section 8.1 결과를 보고할 때는 "randomized method의 정확도"와 함께 "target rank가 결과에 미치는 영향"을 별도의 메시지로 분리해 설명하는 것이 좋다.
