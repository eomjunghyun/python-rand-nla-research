# P001 - Randomized Spectral Clustering in Large-Scale Stochastic Block Models

## 1) Bibliography

- Title: Randomized Spectral Clustering in Large-Scale Stochastic Block Models
- Authors: Hai Zhang, Xiao Guo, Xiangyu Chang
- Journal: Journal of Computational and Graphical Statistics
- Year: 2022
- Volume/Issue/Pages: 31(3), 887-906
- DOI: https://doi.org/10.1080/10618600.2022.2034636
- arXiv: https://arxiv.org/abs/2002.00839
- Local PDF: `/Users/eomjeonghyeon/Documents/RNLA/레퍼런스 논문모음/랜덤선형대수/Randomized Spectral Clustering in Large-Scale Stochastic Block Models.pdf`

## 2) Why this paper

- 대규모 네트워크에서 spectral decomposition 비용이 커지는 문제를, 랜덤화(sketching) 기반으로 줄이는 방법을 다룸.
- 우리 프로젝트의 Section 7.1 Exp1~Exp4 설정과 직접 연결됨.
- 정확도(이론 bound + 실험)와 속도(runtime)를 함께 비교할 수 있는 기준 논문.

## 3) Problem setup (핵심)

- 네트워크 커뮤니티 탐지 문제를 SBM 기반으로 정식화.
- 기존 spectral clustering은 고유분해 비용이 커서 large-scale에서 병목.
- 목표: 랜덤화 기법으로 계산량을 줄이되, 통계적 성능 저하를 최소화.

## 4) Method (핵심 아이디어)

- 비교 방법 3개:
1. Non-random spectral clustering (기준선)
2. Random Projection 기반 spectral clustering
3. Random Sampling 기반 spectral clustering

- 공통 흐름:
1. 인접행렬(또는 스케치 행렬)로부터 저차원 spectral embedding 계산
2. embedding row-wise K-means로 커뮤니티 할당
3. 성능지표로 이론/실험 비교

## 5) Theory (논문 기여 요약)

- randomized 방법 2개에 대해 다음 관점의 오차를 분석:
1. Population adjacency matrix 근사 오차
2. Misclassification (community assignment) 오차
3. Link probability matrix 추정 오차

- 결론 요약:
- 완화된 조건(mild conditions) 하에서 randomized 방법들이 original spectral clustering과 같은 수준의 이론적 bound를 달성.
- degree-corrected SBM으로도 결과를 확장.

## 6) Metrics (재현 시 사용한 핵심 지표)

- Error for P: `||A_hat - P||_2`
- Error for Theta: 라벨 permutation을 고려한 misclassification형 지표
- Error for B: `||B_hat - B||_inf`
- Runtime: 방법별 실행시간 비교

## 7) Section 7.1 재현 매핑 (우리 코드 기준)

- Exp1: `n` 변화 (기본 K=3, alpha_n=0.2, lambda=0.5)
- Exp2: `alpha_n` 변화 (기본 n=1152, K=3)
- Exp3: `K` 변화 (기본 n=1152, alpha_n=0.2)
- Exp4: high-dimensional scaling (`alpha_n = 2/sqrt(n)`, K=2)

- 공통 하이퍼파라미터(기본):
- `q=2`, `r=10`, `p=0.7`, `reps=20`, `seed=2026`

## 8) Practical takeaway

- 대규모 그래프에서 randomized sketching 기반 방법은 속도 이점이 크고,
- 적절한 조건에서는 정확도 손실 없이(또는 매우 작게) spectral clustering을 수행할 수 있다.
- 즉, "large-scale에서 계산 효율 + 통계적 타당성"을 동시에 노릴 수 있는 접근.