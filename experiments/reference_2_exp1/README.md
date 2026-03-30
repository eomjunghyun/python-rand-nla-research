# Reference 2 - Experiment 1 (Figure 2 Runtime)

논문 "A Stochastic Block Hypergraph model"의 Figure 2(시간복잡도 benchmark)를 재현하는 코드입니다.

## Run

```bash
python3 experiments/reference_1_exp1/run_figure2_runtime.py
```

기본 설정:
- `strategy = weighted`
- `K = 4`
- `N = E`
- `p = 100 / N`
- `q = 0.4 * p`
- `reps = 5`

## Outputs

- `experiments/reference_1_exp1/figures/figure2_runtime.png`
- `experiments/reference_1_exp1/results/figure2_runtime.csv`
- `experiments/reference_1_exp1/results/figure2_fit.json`
