# Section 8 재현 실험 결과 분석 보고서

## 1. 보고서 개요

본 보고서는 레퍼런스 논문의 **Section 8.1**과 **Section 8.2 / Table 4**에 해당하는 실험을 현재 저장소에서 재현한 뒤, 논문에 보고된 결과와 비교하여 재현의 정확성과 해석 가능성을 평가한 문서이다.

비교 대상 논문은 다음과 같다.

- 논문: [Randomized Spectral Clustering in Large-Scale Stochastic Block Models (arXiv)](https://arxiv.org/abs/2002.00839)
- 저널 페이지: [JCGS article page](https://www.tandfonline.com/doi/abs/10.1080/10618600.2022.2034636)

본 보고서의 목적은 다음 세 가지이다.

1. Section 8.1과 8.2 실험이 각각 무엇을 검증하는 실험인지 명확히 설명한다.
2. 우리 구현 결과가 논문 결과와 얼마나 일치하는지 정량적으로 비교한다.
3. 일치하지 않는 경우 그 원인이 무엇인지 분석하고, 교수님께 보고할 때 어떤 수준으로 해석해야 하는지 정리한다.

## 2. 비교에 사용한 로컬 결과 파일

### 2.1 Section 8.1

- 요약 테이블:
  [email_eu_table2a_like.md](/C:/Users/WWindows10/Documents/github_project/python-rand-nla-research/experiments/reference_1_section8_1/results/exp8_1_email_eu_core_table2_like/email_eu_table2a_like.md)
- 수치 요약 CSV:
  [email_eu_summary_mean_std.csv](/C:/Users/WWindows10/Documents/github_project/python-rand-nla-research/experiments/reference_1_section8_1/results/exp8_1_email_eu_core_table2_like/email_eu_summary_mean_std.csv)

### 2.2 Section 8.2

- Table 4 유사 마크다운 표:
  [table4_like_median_time.md](/C:/Users/WWindows10/Documents/github_project/python-rand-nla-research/experiments/reference_1_section8_2/results/exp8_2_table4_paper_aligned/table4_like_median_time.md)
- Table 4 유사 요약 CSV:
  [table4_like_median_time.csv](/C:/Users/WWindows10/Documents/github_project/python-rand-nla-research/experiments/reference_1_section8_2/results/exp8_2_table4_paper_aligned/table4_like_median_time.csv)

## 3. 전체 요약

- **Section 8.1은 매우 성공적으로 재현되었다.**
  논문에 제시된 성능 수치와 방법별 상대적 순위가 사실상 동일하게 나타났다.
- **Section 8.2는 실험 구조와 측정 방식은 논문에 맞게 재현되었으나, 절대 실행시간은 부분적으로만 재현되었다.**
- 특히 Section 8.2의 `partial_eigen` 비교군은 논문에서 사용한 R 구현이 아니라 Python `scipy.sparse.linalg.eigsh` 기반의 proxy 구현이므로, 절대시간을 논문과 완전히 동일하게 비교하기는 어렵다.

## 4. Section 8.1과 Section 8.2가 각각 무엇을 하는 실험인가

### 4.1 Section 8.1의 목적

Section 8.1은 **정확도(accuracy) 실험**이다.

이 실험의 목적은 다음과 같다.

- 실제 네트워크 데이터에서 랜덤화된 방법들이 군집 구조를 얼마나 잘 보존하는지 평가한다.
- Random Projection, Random Sampling 같은 randomized method가 원래의 non-random spectral clustering과 비교했을 때 clustering quality를 어느 정도 유지하는지 확인한다.
- 즉, 계산을 근사화하더라도 실제 community detection 성능이 크게 나빠지지 않는지를 보는 실험이다.

Section 8.1의 입력과 출력은 다음과 같다.

- 데이터:
  - European Email network
  - 노드의 부서 라벨(ground truth)
- 비교 방법:
  - Random Projection
  - Random Sampling (`p=0.7`, `p=0.8`)
  - Non-random
- 평가 지표:
  - F1
  - NMI
  - ARI

즉 Section 8.1은 다음 질문에 답하는 실험이다.

> 랜덤화된 spectral clustering을 사용했을 때도 실제 라벨 구조를 충분히 잘 복원할 수 있는가?

따라서 Section 8.1은 **속도 실험이 아니라 성능 실험**으로 이해해야 한다.

### 4.2 Section 8.2의 목적

Section 8.2는 **효율성(runtime / efficiency) 실험**이다.

이 실험의 목적은 다음과 같다.

- 대규모 희소 네트워크에서 leading eigenvector를 계산하는 데 드는 시간이 얼마나 되는지 비교한다.
- randomized method가 대규모 그래프에서 실제로 계산 시간을 줄여주는지 확인한다.
- 즉, spectral clustering의 핵심 선형대수 계산을 큰 네트워크에서 얼마나 효율적으로 수행할 수 있는지를 보는 실험이다.

Section 8.2의 입력과 출력은 다음과 같다.

- 데이터:
  - DBLP
  - Youtube
  - Internet
  - 논문 원본에는 LiveJournal도 포함되지만, 본 재현에서는 제외함
- 비교 방법:
  - Random Projection
  - Random Sampling
  - `partial_eigen`
  - 논문 원본에는 `irlba`, `svds`, `svdr`도 포함되지만, 본 재현에서는 제외함
- 측정 대상:
  - clustering 전체 시간이 아니라 **eigenvector computation 시간**

특히 Random Sampling은 논문 Table 4에서 다음 두 값을 함께 제시한다.

- sampling을 포함한 전체 eigenvector 계산 시간
- sampling 비용을 제외한 eigenvector 계산 시간

즉 Section 8.2는 다음 질문에 답하는 실험이다.

> 매우 큰 실제 네트워크에서 leading eigenvector를 계산할 때, randomized method가 기존 방법보다 계산적으로 유리한가?

따라서 Section 8.2는 **정확도 실험이 아니라 시간 실험**으로 이해해야 한다.

### 4.3 두 실험을 다르게 해석해야 하는 이유

Section 8.1과 Section 8.2는 서로 다른 과학적 질문을 다룬다.

- Section 8.1은 “정확도가 유지되는가?”를 본다.
- Section 8.2는 “계산 시간이 줄어드는가?”를 본다.

따라서 다음과 같은 상황은 충분히 가능하다.

- Section 8.1은 논문과 매우 잘 맞음
- Section 8.2는 구현 환경 차이 때문에 더 크게 어긋남

실제로 runtime benchmark는 accuracy benchmark보다 구현 세부사항에 훨씬 민감하다.

## 5. Section 8.1 재현 결과 분석

### 5.1 실험 설정

Section 8.1 재현에서는 European Email network를 사용하였다.

구체적으로는:

- 이메일 엣지 리스트를 읽어 undirected graph로 변환하고
- 부서 라벨을 ground truth로 사용하며
- 여러 번 반복 실행한 뒤
- 평균과 표준편차를 계산하였다.

이 실험이 답하는 질문은 다음과 같다.

> 각 방법이 실제 부서 구조를 얼마나 잘 복원하는가?

### 5.2 우리 재현 결과

[email_eu_table2a_like.md](/C:/Users/WWindows10/Documents/github_project/python-rand-nla-research/experiments/reference_1_section8_1/results/exp8_1_email_eu_core_table2_like/email_eu_table2a_like.md) 기준 결과는 다음과 같다.

| 방법 | F1 | NMI | ARI |
|---|---:|---:|---:|
| Random Projection | 0.315(0.018) | 0.548(0.010) | 0.091(0.008) |
| Random Sampling (`p=0.7`) | 0.271(0.014) | 0.505(0.011) | 0.079(0.008) |
| Random Sampling (`p=0.8`) | 0.287(0.016) | 0.526(0.009) | 0.091(0.009) |
| Non-random | 0.325(0.016) | 0.556(0.008) | 0.092(0.006) |

### 5.3 논문과의 비교

위 결과는 논문 Table 2(a)에 보고된 값과 소수 셋째 자리 수준에서 사실상 동일하다.

특히 다음이 그대로 유지된다.

- Non-random이 가장 좋은 성능을 보인다.
- Random Projection이 그 다음이다.
- Random Sampling에서는 `p=0.8`이 `p=0.7`보다 더 좋은 성능을 보인다.
- 전체적인 성능 차이의 크기와 방향도 논문과 일치한다.

### 5.4 해석

Section 8.1은 **매우 성공적인 재현**으로 볼 수 있다.

이유는 다음과 같다.

- 논문에 제시된 수치와 거의 같은 값이 나왔다.
- 방법별 우열 관계도 그대로 재현되었다.
- 실험 목적 자체가 clustering quality 비교인데, 그 결론이 논문과 동일하다.

### 5.5 Section 8.1 결론

> Section 8.1 European Email 실험은 정확도 측면에서 논문 결과를 매우 잘 재현하였다. F1, NMI, ARI 값이 논문과 사실상 동일하고, 방법별 성능 순위도 동일하게 나타났다.

## 6. Section 8.2 / Table 4 재현 결과 분석

### 6.1 실험 설정

Section 8.2 재현에서는 논문 Table 4의 구조를 따라 다음과 같이 실험을 구성하였다.

- 데이터:
  - DBLP
  - Youtube
  - Internet
- target rank:
  - DBLP: 3
  - Youtube: 7
  - Internet: 4
- 비교 방법:
  - Random Projection
  - Random Sampling
  - `partial_eigen`
- 반복 횟수:
  - 20회
- 측정값:
  - eigenvector computation 시간의 median

즉, 이 실험은 다음 질문에 답한다.

> 대규모 희소 네트워크에서 leading eigenvector를 계산할 때 각 방법의 계산 비용은 얼마나 되는가?

### 6.2 중요한 구현 주의사항

논문은 R 기반의 `partial_eigen` 구현을 사용한다.

반면 현재 재현에서는 논문의 동일한 R 함수를 직접 호출하지 않고, Python의 `scipy.sparse.linalg.eigsh`를 이용해 기능적으로 유사한 proxy를 구현하였다.

따라서 다음은 논문과 맞춰졌다고 볼 수 있다.

- 데이터 크기
- target rank
- timing 대상
- Random Sampling의 포함/제외 시간 보고 방식

하지만 다음은 논문과 정확히 같지 않다.

- `partial_eigen`의 내부 소프트웨어 구현체

이 점이 Section 8.2 결과 차이의 가장 큰 원인이다.

### 6.3 데이터 규모 정합성

우리 실험에서 사용된 데이터 규모는 다음과 같다.

| 데이터셋 | 노드 수 | 엣지 수 | Target Rank |
|---|---:|---:|---:|
| DBLP | 317,080 | 1,049,866 | 3 |
| Youtube | 1,134,890 | 2,987,624 | 7 |
| Internet | 1,696,415 | 11,095,298 | 4 |

이 값들은 논문 Table 3의 설정과 일치한다.

### 6.4 우리 재현 결과

[table4_like_median_time.md](/C:/Users/WWindows10/Documents/github_project/python-rand-nla-research/experiments/reference_1_section8_2/results/exp8_2_table4_paper_aligned/table4_like_median_time.md) 기준 결과는 다음과 같다.

| 네트워크 | Random Projection | Random Sampling | partial_eigen |
|---|---:|---:|---:|
| DBLP | 0.476 | 0.310(0.209) | 0.239 |
| Youtube | 2.053 | 1.661(1.349) | 1.140 |
| Internet | 3.771 | 2.952(1.680) | 1.840 |

여기서 Random Sampling의 괄호 바깥 값은 sampling 포함 시간, 괄호 안 값은 sampling 제외 시간이다.

### 6.5 논문 Table 4 값

논문에서 같은 항목에 대해 보고한 값은 다음과 같다.

| 네트워크 | Random Projection | Random Sampling | partial_eigen |
|---|---:|---:|---:|
| DBLP | 0.369 | 0.280(0.248) | 0.346 |
| Youtube | 2.037 | 2.302(2.204) | 9.111 |
| Internet | 2.773 | 2.072(1.774) | 7.706 |

### 6.6 정량 비교

#### DBLP

| 방법 | 논문 | 재현 | 차이 |
|---|---:|---:|---:|
| Random Projection | 0.369 | 0.476 | +29.1% |
| Random Sampling (sampling 포함) | 0.280 | 0.310 | +10.7% |
| Random Sampling (sampling 제외) | 0.248 | 0.209 | -15.9% |
| partial_eigen | 0.346 | 0.239 | -30.9% |

해석:

- Random Sampling은 비교적 논문에 가깝다.
- Random Projection은 논문보다 다소 느리다.
- `partial_eigen`은 논문보다 더 빠르게 나왔다.

#### Youtube

| 방법 | 논문 | 재현 | 차이 |
|---|---:|---:|---:|
| Random Projection | 2.037 | 2.053 | +0.8% |
| Random Sampling (sampling 포함) | 2.302 | 1.661 | -27.8% |
| Random Sampling (sampling 제외) | 2.204 | 1.349 | -38.8% |
| partial_eigen | 9.111 | 1.140 | -87.5% |

해석:

- Random Projection은 논문과 거의 동일한 수준이다.
- Random Sampling은 논문보다 더 빠르게 나왔다.
- `partial_eigen`은 논문보다 훨씬 빠르게 나왔는데, 이는 Python proxy baseline이 논문의 R 구현과 runtime 특성이 매우 다르다는 점을 강하게 시사한다.

#### Internet

| 방법 | 논문 | 재현 | 차이 |
|---|---:|---:|---:|
| Random Projection | 2.773 | 3.771 | +36.0% |
| Random Sampling (sampling 포함) | 2.072 | 2.952 | +42.5% |
| Random Sampling (sampling 제외) | 1.774 | 1.680 | -5.3% |
| partial_eigen | 7.706 | 1.840 | -76.1% |

해석:

- Random Sampling의 sampling 제외 시간은 논문과 비교적 가깝다.
- Random Projection과 Random Sampling의 sampling 포함 시간은 논문보다 느리다.
- `partial_eigen`은 논문보다 매우 빠르다.

### 6.7 Section 8.2의 핵심 해석

Section 8.2는 **실험 설계 자체는 논문과 맞지만, 절대 실행시간은 부분적으로만 재현되었다**고 보는 것이 맞다.

특히 다음이 중요하다.

- RP와 RS는 네트워크에 따라 논문과 비슷한 수준을 보이기도 한다.
- 그러나 `partial_eigen`은 거의 모든 데이터셋에서 논문보다 훨씬 빠르다.
- 따라서 현재 결과를 논문 Table 4의 절대시간과 1:1로 동일하다고 말하는 것은 적절하지 않다.

## 7. Section 8.2가 정확히 맞지 않는 이유

### 7.1 소프트웨어 스택 차이

가장 중요한 원인은 이것이다.

- 논문: R 기반 `partial_eigen`
- 현재 재현: Python `scipy.sparse.linalg.eigsh` 기반 proxy

같은 종류의 sparse eigenproblem을 푼다고 하더라도, 내부 구현과 최적화 수준이 다르면 절대 실행시간은 크게 달라질 수 있다.

### 7.2 Python sparse solver baseline이 매우 강함

현재 환경에서는 SciPy sparse eigensolver가 상당히 강하게 최적화되어 있는 것으로 보인다.

그 결과:

- `partial_eigen` proxy가 논문보다 훨씬 빠르게 나온다.
- 따라서 randomized method가 논문에서 보였던 것과 같은 상대적 속도 우위를 재현하지 못한다.

즉, 알고리즘 자체의 우열이라기보다 **비교 기준 baseline의 성능이 달라진 것**이 핵심이다.

### 7.3 실제 대규모 희소 네트워크의 특성

Section 8.2의 데이터는 매우 크지만 동시에 매우 희소하다.

이런 경우에는:

- 희소 iterative solver가 원래부터 매우 효율적일 수 있고
- randomized method의 이점이 줄어들 수 있다.

즉 “그래프가 크다”는 사실만으로 randomized method가 반드시 더 빨라지는 것은 아니다.

### 7.4 randomized method의 추가 비용

Randomized method는 이론적으로 장점이 있지만, 실제 구현에서는 별도의 추가 비용이 있다.

- Random Projection:
  - 여러 번의 sparse matrix multiplication
  - QR 분해
- Random Sampling:
  - sampling
  - sampled matrix 구성
  - 이후 eigendecomposition

따라서 baseline이 이미 매우 강하면 randomized method의 이점이 상쇄될 수 있다.

## 8. 왜 Section 8.1은 잘 맞고, Section 8.2는 덜 맞는가

이 차이는 실험의 성격 차이에서 온다.

### 8.1 Section 8.1

- clustering quality를 보는 실험이다.
- low-level numerical backend 차이에 덜 민감하다.
- 알고리즘 구조만 잘 맞추면 결과가 안정적으로 재현될 가능성이 높다.

### 8.2 Section 8.2

- runtime benchmark이다.
- eigensolver backend, 라이브러리, 구현 최적화 수준에 매우 민감하다.
- 따라서 논문과 똑같은 알고리즘 아이디어를 써도 절대 실행시간은 달라질 수 있다.

즉,

- **Section 8.1은 강한 의미의 재현**
- **Section 8.2는 방법론적으로는 맞지만 runtime은 부분 재현**

으로 보는 것이 가장 정확하다.

## 9. 최종 평가

### 9.1 Section 8.1

평가: **매우 성공적인 재현**

근거:

- 성능 수치가 논문과 사실상 동일하다.
- 방법별 우열 관계가 동일하다.
- 실험의 결론도 논문과 동일하다.

### 9.2 Section 8.2

평가: **부분적으로 성공한 재현**

근거:

- 데이터 규모, target rank, timing 정의는 논문과 정합적이다.
- 실험 구조도 논문과 맞다.
- 그러나 절대시간은 균일하게 일치하지 않는다.
- 가장 큰 이유는 `partial_eigen`이 논문의 R 구현이 아니라 Python proxy이기 때문이다.

## 10. 교수님 보고용 권장 문장

다음과 같이 보고하는 것이 가장 적절하다.

> Section 8.1은 논문 결과를 매우 정확하게 재현하였으며, Table 2(a)의 성능 수치와 방법별 순위가 사실상 동일하게 나타났다. 반면 Section 8.2는 데이터 규모, target rank, timing protocol 측면에서는 논문과 정합적으로 재현되었으나, 절대 실행시간은 부분적으로만 일치하였다. 특히 논문에서 사용한 R 기반 `partial_eigen` 구현을 Python `SciPy` 기반 proxy로 대체하였기 때문에 baseline runtime이 크게 달라졌고, 이로 인해 논문 Table 4와 동일한 시간 비교 결과를 얻지는 못했다. 따라서 Section 8.2는 논문과 동일한 실험 구조를 따르는 방법론적 재현으로 해석하는 것이 적절하다.

## 11. 최종 결론

- **Section 8.1:** 논문과 매우 잘 맞는 정확도 재현 실험
- **Section 8.2:** 실험 구조는 맞지만 절대 실행시간은 조심스럽게 해석해야 하는 부분 재현 실험
- **가장 중요한 한계:** `partial_eigen` baseline이 논문과 동일한 소프트웨어 구현이 아님

