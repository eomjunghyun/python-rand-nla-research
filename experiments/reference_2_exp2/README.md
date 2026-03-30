# Reference 2 - Experiment 2 (Figure 3/4/5)

논문 "A Stochastic Block Hypergraph model"의 실험 2를 재현합니다.

재현 대상:
- Figure 3: degree / hyperedge size distribution + binomial 근사
- Figure 4: effective parameter (`E*/E`, `N*/N`) vs `q/p`
- Figure 5: `min` strategy에서 size distribution의 bimodal 경향 점검

공통 생성 로직은 `src/hypergraph_sbm.py`의 함수를 재사용합니다.

## Run

```bash
python3 experiments/reference_2_exp2/run_figure3_4_5.py
```

## Outputs

- `experiments/reference_2_exp2/figures/figure3_degree_and_size.png`
- `experiments/reference_2_exp2/figures/figure4_effective_parameters.png`
- `experiments/reference_2_exp2/figures/figure5_min_bimodal.png`
- `experiments/reference_2_exp2/results/figure3_summary.csv`
- `experiments/reference_2_exp2/results/figure4_fit_summary.csv`
- `experiments/reference_2_exp2/results/figure5_peak_check.json`

## Assumptions

- Figure 3/4/5의 distribution은 각 설정에서 생성한 100개 hypergraph의 샘플을 풀링해 empirical PMF를 계산함.
- degree 분포는 `Binomial(E, theta_deg)`로 근사하고, `theta_deg = mean(degree) / E` (MLE)로 추정.
- hyperedge size 분포는 알고리즘이 seed node 1개로 시작하므로 `1 + Binomial(N-1, theta_size)`로 근사하고, `theta_size = (mean(size)-1)/(N-1)`로 추정.
- effective ratio는 `q/p = 1`에서의 fitted theta를 기준으로 정규화하여 계산:
  `E*/E = theta_deg(q) / theta_deg(q/p=1)`,
  `N*/N = theta_size(q) / theta_size(q/p=1)`.
- bimodal 진단은 size PMF를 이동평균으로 smoothing한 뒤 local peak 개수를 세는 단순 규칙을 사용.
