# Section 7.2 (Models 1-6) Experiments

Reproduction scripts with live progress, timing, and paper-aligned metrics.
All clustering/evaluation pipelines follow `src/common.py`.

## Run

```bash
python experiments/reference_1_section7_2/sec72_models123_live.py --reps 20 --seed 2026
python experiments/reference_1_section7_2/sec72_models456_live.py --reps 20 --seed 2026
```

## Outputs

By default, results are written to:

- `experiments/reference_1_section7_2/results/exp72_models123_paper_aligned_live/sec72_models123_raw_per_rep.csv`
- `experiments/reference_1_section7_2/results/exp72_models123_paper_aligned_live/sec72_models123_summary_mean_std.csv`
- `experiments/reference_1_section7_2/results/exp72_models123_paper_aligned_live/sec72_models123_metrics_figure5_like.png`
- `experiments/reference_1_section7_2/results/exp72_models123_paper_aligned_live/sec72_models123_runtime.png`
- `experiments/reference_1_section7_2/results/exp72_models456_paper_aligned_live/sec72_models456_raw_per_rep.csv`
- `experiments/reference_1_section7_2/results/exp72_models456_paper_aligned_live/sec72_models456_summary_mean_std.csv`
- `experiments/reference_1_section7_2/results/exp72_models456_paper_aligned_live/sec72_models456_metrics_figure6_like.png`
- `experiments/reference_1_section7_2/results/exp72_models456_paper_aligned_live/sec72_models456_runtime.png`
