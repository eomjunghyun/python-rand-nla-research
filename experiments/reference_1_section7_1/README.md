# Section 7.1 Experiments

Notebook-based reproductions for Section 7.1 Experiments 1 through 4.

## Notebooks

- `exp1_live.ipynb`: `n` variation
- `exp2_live.ipynb`: `alpha_n` variation
- `exp3_live.ipynb`: `K` variation
- `exp4_live.ipynb`: `n` variation with `alpha_n = 2 / sqrt(n)`

Each notebook:

- runs the experiment
- saves raw and summary CSV outputs
- saves the original metric/runtime figures
- saves the 19 timing-breakdown plots
- saves the runtime-composition figure with per-step percentage labels

## Shared Code

Common experiment logic lives in `src/section7_experiments.py`.

Common visualization logic lives in `src/section7_visualizations.py`.

## Outputs

By default, outputs are written under:

- `experiments/reference_1_section7_1/results/exp1_paper_aligned_live/`
- `experiments/reference_1_section7_1/results/exp2_section7_1_results/`
- `experiments/reference_1_section7_1/results/exp3_section7_1_results/`
- `experiments/reference_1_section7_1/results/exp4_section7_1_results/`
