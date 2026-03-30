# P002 - A Stochastic Block Hypergraph model

## 1) Bibliography

- Title: A Stochastic Block Hypergraph model
- Authors: Alexis Pister, Marc Barthelemy
- Journal: Physical Review E
- Year: 2024
- Volume/Issue/Pages: 110(3), 034312
- DOI: https://doi.org/10.1103/PhysRevE.110.034312
- arXiv: https://arxiv.org/abs/2312.12536
- Local PDF: `/Users/nuyhiue/Desktop/2312.12536v2.pdf`
- Code: https://github.com/AlexisPister/HySBM

## 2) Why this paper

- 그래프 SBM을 하이퍼그래프로 단순하고 직관적으로 확장한 기준 논문.
- hyperedge formation rule을 명시적으로 조절할 수 있어, 구조 변화 메커니즘을 직접 비교하기 좋음.
- 커뮤니티 탐지/동역학/시각화 등 후속 실험의 테스트베드로 활용하기 적합.

## 3) Problem setup (핵심)

- `N`개의 node가 `K`개의 disjoint community로 나뉜다고 가정.
- planted partition 형태의 확률 구조:
1. 같은 community면 연결확률 `p`
2. 다른 community면 연결확률 `q`
3. 주로 assortative case `0 <= q <= p` 분석

- 목표: community structure를 가진 hypergraph를 생성하는 간단한 stochastic block hypergraph model 구성.
- 요구조건 3가지:
1. 모델이 단순할 것
2. `p = q`일 때 표준 random hypergraph model로 환원될 것
3. hyperedge formation process가 explicit하고 modulable할 것

## 4) Method (핵심 아이디어)

- 기본 아이디어: 각 hyperedge에서 node를 순회하며 `Prob(v -> e)`를 계산해 가입 여부를 Bernoulli로 결정.
- 즉, "node가 기존 hyperedge에 가입하는 과정"으로 전체 hypergraph를 생성.
- 비교 strategy 4개:
1. weighted: 현재 hyperedge 안 모든 node와의 community connection probability 평균 사용
2. max: 현재 hyperedge 안 node들과의 connection probability 최댓값 사용
3. min: 현재 hyperedge 안 node들과의 connection probability 최솟값 사용
4. majority: 현재 hyperedge에서 가장 많이 등장한 community 기준으로 확률 설정

- 알고리즘 흐름:
1. `E`개의 hyperedge 초기화
2. hyperedge loop 수행
3. 각 hyperedge에서 node loop 수행
4. 선택한 strategy로 node 가입 여부 결정
5. 모든 hyperedge 처리 후 종료

- 계산복잡도: 기본 `O(NE)`, `N ~ E`이면 대략 `O(N^2)`.

## 5) Theory (논문 기여 요약)

- `p = q`일 때 degree distribution, hyperedge size distribution이 각각 binomial 형태.
- `q != p`일 때도 effective parameter를 갖는 binomial 근사가 가능함을 제시.
- 제안된 effective parameter:
- `E* / E = N* / N = ((1 - 1/K) * q/p + 1/K)`
- 실험으로 위 관계를 확인.
- hyperedge composition 분석을 위해 normalized Gini coefficient `G` 도입:
1. `G`가 클수록 특정 community 지배
2. `G`가 작을수록 community가 고르게 혼합

- `q/p`가 작을수록 pure hyperedge 증가, `q/p -> 1`로 갈수록 mixed hyperedge 증가.
- structure dependence가 강한 strategy(`majority`, `weighted`)는 dominant community를 강화해 diversity를 줄이는 경향.
- structure dependence가 약한 strategy(`max`, `min`)는 더 다양한 composition을 생성.
- heterogeneity measure `Delta` 분석 결과, `max` strategy에서 중간 `q/p` 구간의 interior peak 가능성 제시.

## 6) Metrics (재현 시 사용한 핵심 지표)

- Runtime: 생성 알고리즘 실행시간
- Degree distribution: `P(k)`
- Hyperedge size distribution: `P(m)`
- Effective parameters: `N* / N`, `E* / E`
- Hyperedge composition metric: normalized Gini coefficient `G`
- Composition heterogeneity: `Delta = sqrt(mean(G^2) - mean(G)^2) / mean(G)`

## 7) Experiments 재현 매핑 (논문 구조 기준)

- Exp1: 시간복잡도 benchmark (Figure 2)
- `weighted` strategy
- `K = 4`, `N = E`, `p = 100 / N`, `q = 0.4p`
- runtime vs `N`, quadratic fit

- Exp2: degree / hyperedge size 분포 분석 (Figure 3, 4, 5)
- Figure 3, 4:
1. `weighted` strategy
2. `N = 1000`, `E = 200`, `K = 4`, `p = 0.03`
3. `q/p = 0, 0.3, 0.7, 1`
4. binomial fit, effective parameter 추정

- Figure 5:
1. `min` strategy
2. `N = 1000`, `E = 200`, `K = 4`, `p = 0.1`
3. `q/p = 0.1, 0.01`
4. bimodal hyperedge size distribution 확인

- Exp3: hyperedge composition 분석 (Figure 6 ~ Figure 12)
- Figure 6:
1. `majority` strategy
2. `N = E = 80`, `K = 4`, `p = 0.1`
3. `q = 0, 0.01, 0.05, 0.1`
4. bipartite visualization + hyperedge Gini coloring

- Figure 7: strategy별 `G` distribution 비교
- Figure 8: node traversal order(`random`, `fixed`, `community`) 영향 비교
- Figure 9: `weighted` strategy에서 `mean G` vs `q/p`, 다양한 `N/E`
- Figure 10: `q/p = 1`에서 `mean G` scaling 확인
- Figure 11: strategy별 `mean G` 비교
- Figure 12: strategy별 `Delta` vs `q/p` 비교

## 8) Practical takeaway

- 하이퍼그래프용 SBM benchmark로 쓰기 좋은, 해석 가능한 기준 모델.
- formation rule 변화에 따라 다음이 어떻게 달라지는지 한 프레임에서 비교 가능:
1. degree / size distribution
2. hyperedge purity / mixture
3. community diversity
4. hyperedge 간 이질성

- 따라서 community detection뿐 아니라 higher-order interaction 시스템에서 formation mechanism 효과를 시험하는 기본 모델로 유용.
- 다만 degree와 hyperedge size가 대체로 binomial 근사에 잘 맞는 단순 구조라,
  현실의 heavy-tailed hypergraph 설명보다는 "해석 가능한 baseline model" 성격이 강함.
