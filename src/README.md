# `src` Modules

This directory contains shared Python modules used by the experiment notebooks.

## Files

- `common.py`
  - shared stochastic block model generation
  - randomized / non-random spectral methods
  - metric evaluation
  - timing breakdown helpers
  - baseline metric/runtime plotting helpers

- `section7_experiments.py`
  - shared execution logic for Section 7.1 Experiments 1-4
  - default experiment configs
  - CSV saving
  - original metric/runtime figure saving

- `section7_visualizations.py`
  - the 19 timing-breakdown plots for each experiment
  - runtime-composition visualization with per-step percentages
  - shared axis-limit computation across experiments

## Section 7.1 Workflow

The notebooks in `experiments/reference_1_section7_1/` import these modules instead of duplicating experiment code.

Typical flow:

1. Build a default config from `section7_experiments.py`
2. Run the experiment and save raw/summary outputs
3. Load the timing summary CSV
4. Render timing-breakdown and runtime-composition figures through `section7_visualizations.py`
