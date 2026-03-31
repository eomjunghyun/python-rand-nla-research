# Reference 2 - Experiment 3 Report

## 필수 점검 항목

1. low q/p에서 hyperedge가 pure해지는지: 평균 pure fraction (G>=0.9, q/p=0.01) = 0.840. 값이 높을수록 low q/p에서 pure hyperedge가 많음을 의미한다.
2. q/p가 1에 가까워지면 G가 작아지는지: N/E=5에서 q/p=1일 때 strategy별 mean G={'majority': 0.20566471595250865, 'max': 0.19939744343560425, 'min': 0.1977601379217557, 'weighted': 0.19160691274106387}, 평균=0.199.
3. majority, weighted가 dominant community를 더 강화하는지: N/E=5에서 q/p 전 구간 평균 mean G 비교 {'majority': 0.4876376540889044, 'max': 0.41220422248075694, 'min': 0.4270955870463446, 'weighted': 0.4453144932404462}. 값이 큰 전략일수록 더 순수한(덜 mixed) hyperedge를 만든다.
4. max 전략에서 Delta가 내부 최대값을 가지는지: overall peak = q/p 0.750, interior=True, local_peak=True, near_0.2=False.
5. 논문과 차이가 난다면 가능한 원인: seed 민감도, majority tie-breaking 규칙, Figure 6의 layout approximation(D3 대신 spring layout), histogram binning/smoothing 설정 차이.

## Figure 10 Scaling
- linear fit: y = 0.4429 x + -0.0010, R^2 = 0.9990

## 논문에서 직접 명시된 설정
- Figure 6: strategy=majority, N=E=80, K=4, p=0.1, q={0,0.01,0.05,0.1}
- Figure 7: p=0.1, N=2000, E=200, K=4, strategies={majority,max,min,weighted}, q/p sweep, 100 realizations
- Figure 8: p=0.03, q=0.4p, N=1000, E=200, K=4, orders={random,fixed,community}
- Figure 9: strategy=weighted, E=200, N/E={1,1.5,2.5,3.5,5,10}, q/p sweep, 100 realizations
- Figure 10: E=200, p=q=0.03, x=1/sqrt(N/E), y=mean G
- Figure 11: E=200, N=1000, strategies={majority,max,min,weighted}, q/p sweep
- Figure 12: p=0.03, E=200, N/E={1,1.5,2.5,3.5,5,10}, strategies={majority,max,min,weighted}
- majority tie-breaking: ties are resolved by choosing the smallest community index

## 내가 둔 구현 가정
- Community assignment uses balanced labels with random node permutation for each realization.
- Node order implementation is separated via order mode {random, fixed, community}; generation uses src.common.generate_hypergraph.
- Figure 6 layout uses networkx spring_layout as a force-directed approximation (D3 is not used in this Python script).
- P(G) uses fixed histogram bins on [0,1] without extra smoothing.
- Quick mode reduces realizations and q/p grid size for smoke testing only.

## 출력 파일
- figures/figure6_bipartite_majority.png
- figures/figure7_gini_distribution.png
- figures/figure8_order_effect.png
- figures/figure9_mean_g_weighted.png
- figures/figure10_scaling.png
- figures/figure11_strategy_comparison.png
- figures/figure12_delta.png
- results/composition_summary.json
