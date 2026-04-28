# Reference 1 Section 8.2 실험 보고서

이 보고서는 Reference 1 논문의 Section 8.2와 Table 4를 재현한 대규모 real network 효율성 실험을 정리한다. Section 8.1이 clustering accuracy를 보는 실험이었다면, Section 8.2는 대규모 sparse graph에서 spectral clustering의 핵심 계산인 eigenvector computation이 얼마나 빠르게 수행되는지를 비교하는 runtime benchmark다.

보고서의 목적은 다음과 같다.

- Section 8.2가 어떤 실험인지 설명한다.
- Table 4에서 측정하는 시간이 정확히 무엇인지 정리한다.
- Random Projection, Random Sampling, partial_eigen이 각각 무엇을 의미하는지 설명한다.
- 재현 결과와 논문 결과를 비교한다.
- `table4_median_bar.png`, `table4_runtime_boxplots.png`, DBLP recheck plot들을 어떻게 해석해야 하는지 설명한다.
- 논문과 재현 결과가 다르게 나온 이유를 보고용으로 정리한다.

## 1. 실험 목적

Section 8.2의 핵심 질문은 다음이다.

> 큰 실제 네트워크에서 spectral clustering에 필요한 leading eigenvectors를 계산할 때, randomized method가 기존 eigen solver보다 빠른가?

여기서 중요한 점은 Section 8.2가 clustering accuracy 실험이 아니라는 것이다. 이 실험은 F1, NMI, ARI 같은 정확도 지표를 비교하지 않고, eigenvector computation 시간만 비교한다.

논문 Table 4는 네트워크별로 다음 방법들의 median runtime을 보고한다.

- Random Projection
- Random Sampling
- partial_eigen
- 논문 원본에는 `irlba`, `svds`, `svdr`도 포함되어 있지만, 현재 재현에서는 Table 4의 핵심 비교군 중 세 방법을 구현했다.

현재 저장소의 공식 재현 결과는 다음 위치에 있다.

| 파일 | 설명 |
|---|---|
| `results/exp8_2_table4_paper_aligned/table4_time_raw.csv` | 20회 반복별 raw runtime |
| `results/exp8_2_table4_paper_aligned/table4_like_median_time.csv` | dataset별 median runtime 요약 |
| `results/exp8_2_table4_paper_aligned/table4_like_median_time.md` | 논문 Table 4 형식의 markdown 표 |
| `results/exp8_2_table4_paper_aligned/table4_meta.json` | dataset 크기, rank, seed, 반복 횟수 등 메타데이터 |
| `results/exp8_2_table4_paper_aligned/viz/table4_median_bar.png` | median runtime grouped bar plot |
| `results/exp8_2_table4_paper_aligned/viz/table4_runtime_boxplots.png` | 반복별 runtime 분포 boxplot |

추가로 DBLP만 더 자세히 쪼개 본 recheck 결과가 있다.

| 파일 | 설명 |
|---|---|
| `results/dblp_live_recheck_20260331/dblp_time_raw.csv` | DBLP 반복별 total/eigen/KMeans/sampling 시간 |
| `results/dblp_live_recheck_20260331/dblp_step_time_summary.csv` | DBLP 내부 단계별 timing summary |
| `results/dblp_live_recheck_20260331/dblp_pairwise_ari_mean_matrix.csv` | DBLP 방법 간 clustering 유사도 |
| `results/dblp_live_recheck_20260331/viz/*.png` | DBLP runtime breakdown, boxplot, ARI plot |

## 2. 대상 데이터셋

Section 8.2 재현은 세 개의 대규모 실제 네트워크를 사용한다.

| 데이터셋 | 노드 수 | 엣지 수 | Target rank |
|---|---:|---:|---:|
| DBLP | 317,080 | 1,049,866 | 3 |
| Youtube | 1,134,890 | 2,987,624 | 7 |
| Internet | 1,696,415 | 11,095,298 | 4 |

`target rank`는 계산할 leading eigenvector 개수다. Section 8.2에서는 clustering 결과 자체보다 eigenvector 계산 시간을 보므로, 이 rank는 각 방법이 얼마나 큰 spectral subspace를 계산해야 하는지를 정한다. rank가 커질수록 보통 계산량이 증가한다.

현재 재현에서는 논문 원본의 LiveJournal은 포함하지 않았다. 따라서 Table 4 전체를 완전히 재현한 것이 아니라, 현재 로컬 데이터와 구현으로 가능한 DBLP, Youtube, Internet 세 데이터셋에 대한 재현이다.

## 3. 비교 방법

### 3.1 Random Projection

Random Projection은 큰 sparse adjacency matrix `A`의 leading eigenspace를 직접 계산하지 않고, random Gaussian matrix를 이용해 주요 subspace를 근사하는 방법이다.

현재 구현에서는 다음 흐름을 따른다.

1. node 수 `n`과 rank `k`에 대해 random test matrix `Omega`를 만든다.
2. `A @ Omega` 형태의 matrix multiplication을 반복한다.
3. power iteration을 통해 큰 eigenvalue 방향을 강조한다.
4. QR decomposition으로 orthonormal basis `Q`를 만든다.
5. 작은 matrix `Q.T @ A @ Q`에서 eigen decomposition을 수행한다.
6. 다시 원래 공간으로 projection하여 leading eigenvectors를 근사한다.

설정값은 다음과 같다.

| 파라미터 | 값 | 의미 |
|---|---:|---|
| `q` | 2 | power iteration 관련 파라미터 |
| `r` | 10 | oversampling 차원 |

Random Projection은 큰 matrix에서 직접 eigen decomposition을 하는 대신 matrix multiplication과 작은 matrix eigen decomposition으로 문제를 줄이는 것이 핵심이다.

여기서 주의할 점은 Random Projection 시간이 "원본 행렬 전체의 완전 고유분해 시간"이 아니라는 것이다. 예를 들어 Internet 데이터셋의 Random Projection `3.771초`는 `1,696,415 x 1,696,415` 원본 행렬의 모든 eigenvalue/eigenvector를 계산한 시간이 아니다. target rank `4` 근처의 leading eigenspace를 근사하기 위해 위의 1-6단계를 수행한 전체 시간이다. 작은 matrix `Q.T @ A @ Q`의 eigen decomposition은 이 시간 안에 포함되지만, `3.771초` 전체가 작은 matrix 분해 시간만을 뜻하지도 않는다.

### 3.2 Random Sampling

Random Sampling은 graph edge를 확률 `p`로 샘플링해 더 작은 sparse matrix를 만든 뒤, 그 matrix에서 eigenvectors를 계산하는 방법이다.

현재 설정은 다음과 같다.

| 파라미터 | 값 | 의미 |
|---|---:|---|
| `p` | 0.7 | 각 edge를 남길 확률 |

샘플링된 edge는 weight를 `1/p`로 rescale한다. 이렇게 하면 expectation 관점에서 원래 adjacency matrix의 scale을 보존한다. 예를 들어 `p=0.7`이면 edge의 약 70%만 남기고, 남은 edge에는 `1/0.7` 배의 weight를 준다.

Table 4에서는 Random Sampling 시간을 두 가지로 표시한다.

- 괄호 밖 값: sampling을 포함한 전체 시간
- 괄호 안 값: sampling을 제외한 eigenvector computation 시간

예를 들어 `0.310(0.209)`는 sampling 포함 median time이 0.310초이고, sampling 제외 median time이 0.209초라는 뜻이다.

### 3.3 partial_eigen

`partial_eigen`은 기존 partial eigen solver baseline이다. 논문에서는 R 계열 구현을 사용한다. 현재 Python 재현에서는 동일한 R 함수를 직접 호출하지 않고, `scipy.sparse.linalg.eigsh`를 이용한 proxy 구현을 사용했다.

따라서 `partial_eigen` 결과는 다음처럼 해석해야 한다.

- 알고리즘 역할: 기존 sparse eigen solver baseline.
- 현재 구현: Python/scipy proxy.
- 주의: 논문 Table 4의 `partial_eigen`과 내부 구현체가 다르므로 절대 runtime은 직접적으로 완전히 같은 의미가 아니다.

이 차이가 8.2 재현에서 가장 큰 해석상 주의점이다.

## 4. 측정 방식

공식 Table 4 재현 스크립트는 각 dataset과 method에 대해 20회 반복한다. 각 반복에서 eigenvector computation 부분만 시간을 측정하고, dataset별 median을 보고한다.

측정 대상은 다음과 같다.

| 방법 | 측정 시간 |
|---|---|
| Random Projection | random projection 기반 leading eigenvector 근사 파이프라인 전체 시간 |
| Random Sampling | sampling + sampled matrix eigenvector 계산 시간 |
| Random Sampling excl. sampling | sampled matrix eigenvector 계산 시간만 |
| partial_eigen | sparse partial eigen solver 시간 |

중요한 점은 이 공식 Table 4 재현에서는 KMeans clustering 시간, accuracy 계산 시간, ARI 계산 시간, 시각화 시간은 제외한다는 것이다. 이는 논문 Table 4가 clustering pipeline 전체 시간이 아니라 eigenvector computation 시간을 비교하기 때문이다.

따라서 Table 4의 "eigenvector computation time"은 "원본 행렬의 full eigen decomposition time"과 다르다. Section 8.2는 각 방법이 필요한 개수의 leading eigenvectors를 얼마나 빠르게 계산하거나 근사하는지를 비교한다. Random Projection은 근사 방법이고, partial_eigen은 sparse iterative solver 기반 baseline이며, 둘 다 원본 행렬의 모든 고유쌍을 계산하지 않는다.

DBLP recheck 결과에는 KMeans 시간과 내부 단계별 시간이 추가로 들어 있다. 이것은 공식 Table 4 숫자를 대체하는 것이 아니라, DBLP에서 어느 단계가 실제 병목인지 확인하기 위한 보조 분석이다.

## 5. 공식 재현 결과

공식 재현 결과는 `results/exp8_2_table4_paper_aligned/table4_like_median_time.md`에 저장되어 있다.

| Networks | Random Projection | Random Sampling | partial_eigen |
|---|---:|---:|---:|
| DBLP | 0.476 | 0.310(0.209) | 0.239 |
| Youtube | 2.053 | 1.661(1.349) | 1.140 |
| Internet | 3.771 | 2.952(1.680) | 1.840 |

각 값은 20회 반복의 median time이다. 단위는 초다. 특히 Internet의 Random Projection `3.771초`는 원본 Internet 행렬 전체를 완전 고유분해한 시간이 아니라, target rank `4` leading eigenspace를 Random Projection으로 근사하는 전체 파이프라인 시간이다.

### 5.1 DBLP

DBLP에서는 다음 순서로 빨랐다.

| 기준 | 가장 빠른 방법 | 해석 |
|---|---|---|
| sampling 포함 | partial_eigen 0.239초 | Python/scipy proxy partial eigen이 가장 빠름 |
| sampling 제외 비교 포함 | Random Sampling excl. sampling 0.209초 | 샘플링된 matrix의 eigen 계산만 보면 Random Sampling이 가장 빠름 |

DBLP에서 Random Projection은 0.476초로 partial_eigen보다 약 2.0배 느렸다. Random Sampling은 sampling 포함 0.310초로 partial_eigen보다 느리지만, sampling 제외 eigen 계산만 보면 0.209초로 partial_eigen보다 빠르다. 즉 DBLP에서는 sampling으로 matrix를 줄인 효과가 있지만, sampling 자체의 비용도 무시할 수 없다.

### 5.2 Youtube

Youtube에서는 partial_eigen proxy가 1.140초로 가장 빠르게 나왔다. Random Sampling은 1.661초, Random Projection은 2.053초다.

논문에서는 partial_eigen이 9.111초로 매우 느렸지만, 현재 재현에서는 훨씬 빠르다. 이 차이는 현재 `partial_eigen`이 논문과 같은 구현체가 아니라 `scipy.sparse.linalg.eigsh` proxy이기 때문이다.

Youtube에서 Random Sampling은 Random Projection보다 빠르다. sampling 제외 시간은 1.349초로 더 낮다. 즉 edge sampling으로 eigen 계산 자체는 줄어들지만, 현재 환경에서는 partial_eigen proxy가 더 빠르게 동작한다.

### 5.3 Internet

Internet은 세 데이터셋 중 edge 수가 가장 많다. 결과는 다음과 같다.

- Random Projection: 3.771초
- Random Sampling: 2.952초
- Random Sampling excl. sampling: 1.680초
- partial_eigen: 1.840초

sampling 포함 기준으로는 partial_eigen proxy가 가장 빠르다. 하지만 sampling 제외 eigen computation만 보면 Random Sampling이 1.680초로 partial_eigen보다 약간 빠르다. 이 데이터셋에서는 sampling 자체가 약 1.27초 정도의 비용을 만든다. 따라서 "sampling 후 eigen 계산"은 빠르지만 "sampling을 만드는 비용"까지 포함하면 이득이 줄어든다.

여기서 Random Projection의 3.771초는 Internet 원본 행렬의 완전한 eigen decomposition 시간이 아니다. Random Projection 과정에서 sparse matrix multiplication, QR decomposition, 작은 matrix `Q.T @ A @ Q` 생성, 작은 matrix eigen decomposition, 원래 공간으로의 projection을 모두 포함한 시간이다. 즉 "전체 분해시간"이라기보다 "rank 4 leading eigenvectors를 근사하기 위한 Random Projection 전체 계산 시간"으로 해석해야 한다.

## 6. 논문 Table 4와 비교

논문에서 보고한 값과 현재 재현 결과를 비교하면 다음과 같다.

### 6.1 DBLP

| 방법 | 논문 | 재현 결과 | 차이 |
|---|---:|---:|---:|
| Random Projection | 0.369 | 0.476 | +29.1% |
| Random Sampling | 0.280(0.248) | 0.310(0.209) | 포함시간 +10.7%, 제외시간 -15.9% |
| partial_eigen | 0.346 | 0.239 | -30.9% |

DBLP는 세 데이터셋 중 논문값과 가장 가까운 편이다. Random Projection은 다소 느리고, Random Sampling 포함 시간은 근접하며, sampling 제외 시간은 논문보다 빠르다. partial_eigen은 재현 결과가 더 빠르다.

### 6.2 Youtube

| 방법 | 논문 | 재현 결과 | 차이 |
|---|---:|---:|---:|
| Random Projection | 2.037 | 2.053 | +0.8% |
| Random Sampling | 2.302(2.204) | 1.661(1.349) | 포함시간 -27.8%, 제외시간 -38.8% |
| partial_eigen | 9.111 | 1.140 | -87.5% |

Youtube에서 Random Projection은 논문과 거의 동일하다. 반면 Random Sampling과 partial_eigen은 재현 결과가 훨씬 빠르다. 특히 partial_eigen 차이가 매우 크다. 이는 데이터 크기 차이보다는 구현체와 실행 환경 차이로 보는 것이 자연스럽다.

### 6.3 Internet

| 방법 | 논문 | 재현 결과 | 차이 |
|---|---:|---:|---:|
| Random Projection | 2.773 | 3.771 | +36.0% |
| Random Sampling | 2.072(1.774) | 2.952(1.680) | 포함시간 +42.5%, 제외시간 -5.3% |
| partial_eigen | 7.706 | 1.840 | -76.1% |

Internet에서는 Random Projection과 sampling 포함 Random Sampling이 논문보다 느리다. 하지만 sampling 제외 시간은 논문과 가까운 편이다. partial_eigen은 재현 결과가 논문보다 훨씬 빠르다.

### 6.4 종합 비교

| 데이터셋 | Random Projection | Random Sampling | partial_eigen | 종합 해석 |
|---|---|---|---|---|
| DBLP | 논문보다 29.1% 느림 | 포함 시간은 근접, 제외 시간은 빠름 | 재현이 더 빠름 | 부분적으로 잘 맞음 |
| Youtube | 논문과 거의 동일 | 재현이 더 빠름 | 재현이 훨씬 빠름 | baseline 구현 차이 큼 |
| Internet | 재현이 더 느림 | 제외 시간은 근접, 포함 시간은 느림 | 재현이 훨씬 빠름 | sampling 비용과 baseline 차이 큼 |

따라서 8.2는 실험 구조와 timing 정의는 논문에 맞지만, 절대 runtime은 부분적으로만 재현되었다고 보는 것이 맞다.

## 7. Plot 해석 가이드

### 7.1 `table4_median_bar.png`

경로: `results/exp8_2_table4_paper_aligned/viz/table4_median_bar.png`

이 plot은 dataset별 median runtime을 grouped bar로 보여준다. x축은 데이터셋이고, y축은 초 단위 runtime이다. 각 데이터셋 안에는 다음 네 막대가 있다.

- Random Projection
- Random Sampling
- Random Sampling excl. sampling
- partial_eigen

해석할 때는 다음을 본다.

- 같은 dataset 안에서 막대가 낮을수록 빠르다.
- Random Sampling 막대와 Random Sampling excl. sampling 막대의 차이는 sampling overhead다.
- Internet에서 두 Random Sampling 막대의 차이가 크면, edge sampling을 만드는 비용이 큰 dataset이라는 뜻이다.
- partial_eigen 막대가 논문보다 낮게 보이는 것은 Python/scipy proxy가 현재 환경에서 빠르게 동작했음을 의미하며, 논문 구현과 완전 동일 baseline으로 보면 안 된다.

이 plot만 보면 세 데이터셋 모두 partial_eigen proxy가 강하게 보인다. 하지만 이 결론은 "논문 partial_eigen보다 빠르다"가 아니라 "현재 Python proxy 기준에서는 빠르다"로 표현해야 한다.

### 7.2 `table4_runtime_boxplots.png`

경로: `results/exp8_2_table4_paper_aligned/viz/table4_runtime_boxplots.png`

이 plot은 20회 반복별 runtime 분포를 보여준다. bar plot이 median 하나만 보여준다면, boxplot은 반복 간 변동성을 보여준다.

해석할 때는 다음을 본다.

- box가 낮을수록 빠르다.
- box가 좁을수록 반복 간 runtime이 안정적이다.
- whisker나 outlier가 길면 특정 반복에서 시간이 튄 것이다.
- Random Sampling은 sampling 과정에서 random edge mask를 만들기 때문에 반복별 변동이 생길 수 있다.
- 대규모 graph에서는 OS cache, memory allocation, sparse matrix construction 차이도 반복별 runtime 변동을 만든다.

따라서 boxplot은 "어떤 방법이 빠른가"뿐 아니라 "얼마나 안정적으로 빠른가"를 확인하는 plot이다.

### 7.3 DBLP recheck plot들

DBLP recheck는 공식 Table 4보다 더 많은 내부 정보를 저장한다. 주요 plot은 다음과 같다.

| Plot | 해석 |
|---|---|
| `runtime_median_bar.png` | DBLP에서 method별 median total runtime 비교 |
| `runtime_per_rep_box.png` | DBLP 반복별 runtime 변동성 |
| `runtime_breakdown_stacked_bar.png` | method total runtime을 sampling/eigen/KMeans 등으로 분해 |
| `method_internal_steps_median.png` | method 내부 step별 median 시간 |
| `rp_power_profile_median.png` | Random Projection의 power/matmul 단계별 시간 |
| `step_summary_bar.png` | 전체 step summary를 bar 형태로 표시 |
| `ari_per_rep_box.png` | DBLP에서 method 간 clustering 유사도 변동 |

주의할 점은 DBLP recheck에는 KMeans 시간이 포함되어 있다는 것이다. 반면 공식 Table 4 재현은 eigenvector computation 시간만 본다. 따라서 DBLP recheck의 total runtime을 Table 4 공식 결과와 그대로 비교하면 안 된다.

DBLP recheck에서 중요한 관찰은 다음과 같다.

- Random Projection total median은 약 0.576초다.
- Random Sampling total median은 약 0.803초다.
- Non-random total median은 약 0.754초다.
- KMeans 시간이 Random Sampling과 Non-random에서 약 0.60초로 크다.
- eigen decomposition만 보면 Random Sampling과 Non-random은 약 0.14초, Random Projection은 약 0.23초다.

즉 DBLP 전체 clustering pipeline에서는 KMeans가 상당한 비중을 차지한다. 그러나 논문 Table 4는 KMeans를 제외한 eigenvector computation 시간을 비교하므로, 이 recheck는 "실제 pipeline 병목"을 설명하는 보조 자료로 쓰는 것이 좋다.

## 8. 결과 해석

### 8.1 Random Projection

Random Projection은 Youtube에서 논문과 거의 같은 시간이 나왔다. DBLP와 Internet에서는 논문보다 느렸다.

해석은 다음과 같다.

- Random Projection은 sparse matrix multiplication을 여러 번 수행한다.
- sparse matrix multiplication은 데이터의 sparsity pattern, 메모리 locality, BLAS/scipy 구현에 민감하다.
- Internet처럼 edge 수가 큰 graph에서는 matrix multiplication 비용이 커져 재현 결과가 논문보다 느릴 수 있다.

따라서 Random Projection은 이론적으로 큰 matrix의 spectral subspace를 빠르게 근사하는 방법이지만, 실제 runtime은 sparse matrix multiplication 구현과 하드웨어 환경에 크게 좌우된다. 이 시간은 full eigen decomposition 시간이 아니라 target rank에 해당하는 leading eigenspace 근사 시간이라는 점을 보고할 때 명확히 밝혀야 한다.

### 8.2 Random Sampling

Random Sampling은 sampling 제외 시간만 보면 DBLP와 Internet에서 논문과 비슷하거나 더 빠르다. Youtube에서는 논문보다 훨씬 빠르다.

하지만 sampling 포함 시간은 Internet에서 논문보다 느리다. 이는 sampling 자체가 큰 sparse graph에서는 무거운 작업이 될 수 있음을 보여준다.

Random Sampling 결과를 볼 때는 반드시 두 값을 분리해야 한다.

- 괄호 밖: 실제로 sampling까지 수행했을 때 사용자가 기다리는 시간.
- 괄호 안: sampling된 matrix가 이미 있다고 가정했을 때 eigen 계산 시간.

알고리즘 자체의 eigen 계산 이득을 보려면 괄호 안을 보고, end-to-end 사용 비용을 보려면 괄호 밖을 봐야 한다.

### 8.3 partial_eigen

현재 재현에서 partial_eigen은 Youtube와 Internet에서 논문보다 훨씬 빠르게 나왔다. 이 결과를 "논문이 틀렸다"거나 "partial_eigen이 randomized method보다 항상 좋다"로 해석하면 안 된다.

이유는 다음과 같다.

- 논문은 R 기반 partial eigen routine을 사용했다.
- 현재 구현은 Python/scipy `eigsh` proxy다.
- scipy/ARPACK의 sparse eigen solver 성능은 matrix 구조와 환경에 따라 매우 다르다.
- CPU, BLAS/LAPACK, scipy 버전, sparse matrix format 차이가 runtime에 크게 영향을 준다.

따라서 partial_eigen은 방법론적 baseline으로는 유효하지만, 절대시간을 논문과 1:1로 비교하는 데는 한계가 있다.

## 9. 보고 시 강조할 점

보고에서는 다음 메시지를 중심으로 설명하면 좋다.

1. Section 8.2는 accuracy가 아니라 eigenvector computation runtime 실험이다.
2. Table 4의 Random Sampling `a(b)` 표기는 `a=샘플링 포함 시간`, `b=샘플링 제외 시간`이다.
3. 현재 재현은 DBLP, Youtube, Internet 세 데이터셋을 대상으로 20회 반복 median을 보고했다.
4. Youtube의 Random Projection은 논문과 거의 같은 시간으로 재현되었다.
5. Internet의 Random Projection `3.771초`는 원본 행렬 전체 고유분해 시간이 아니라 rank `4` leading eigenspace 근사 파이프라인 시간이다.
6. DBLP의 Random Sampling은 논문과 비교적 근접하다.
7. Internet의 Random Sampling은 sampling 제외 시간은 논문과 근접하지만 sampling 포함 시간은 느리다.
8. partial_eigen은 현재 Python/scipy proxy가 논문 baseline보다 훨씬 빠르게 나와, 논문값과 절대시간 비교가 어렵다.
9. 따라서 8.2는 실험 구조와 timing 정의는 재현되었지만, 절대 runtime은 구현체와 환경 차이 때문에 부분 재현으로 보는 것이 타당하다.

## 10. 한계와 주의사항

8.2 runtime benchmark는 다음 요인에 매우 민감하다.

- CPU 종류와 core 수.
- 메모리 대역폭과 cache 상태.
- OS filesystem cache.
- scipy, numpy, BLAS/LAPACK, ARPACK 버전.
- sparse matrix 저장 형식.
- edge list 로딩과 CSR 변환 방식.
- random sampling mask 생성 방식.
- 논문이 사용한 R 구현과 현재 Python 구현의 차이.

따라서 runtime 값은 accuracy 지표처럼 독립적으로 재현되기 어렵다. 같은 알고리즘이라도 구현체가 바뀌면 결과가 크게 달라질 수 있다.

또한 현재 공식 Table 4 재현은 eigenvector computation 시간만 측정한다. 실제 clustering pipeline에서는 KMeans, label alignment, ARI/NMI 계산, 결과 저장과 시각화 시간이 추가된다. DBLP recheck가 보여주듯, 실제 pipeline에서는 KMeans 시간이 상당한 비중을 차지할 수 있다.

## 11. 최종 결론

Section 8.2 재현은 방법론적으로는 논문 Table 4의 구조를 잘 따라간다. 데이터셋, target rank, 반복 횟수, Random Projection/Random Sampling/partial_eigen 비교, Random Sampling의 포함/제외 시간 표기 방식이 모두 논문 형식과 맞춰져 있다.

그러나 절대 runtime은 완전히 일치하지 않는다. 특히 partial_eigen에서 재현 결과가 논문보다 훨씬 빠르게 나왔고, 이는 Python/scipy proxy와 논문 R 구현의 차이 때문으로 보는 것이 가장 합리적이다.

따라서 이 결과는 다음과 같이 보고하는 것이 좋다.

> Section 8.2는 실험 설계와 timing 정의는 재현되었지만, runtime benchmark 특성상 구현체와 하드웨어 환경에 민감하여 절대 시간은 부분적으로만 재현되었다. Random Projection은 Youtube에서 논문과 매우 근접했고, Random Sampling은 sampling 제외 시간을 보면 DBLP/Internet에서 논문과 가까운 편이다. 다만 partial_eigen baseline은 현재 Python proxy가 논문 구현보다 훨씬 빨라 직접적인 절대시간 비교에는 주의가 필요하다.
