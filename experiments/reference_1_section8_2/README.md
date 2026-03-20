# Section 8.2 (DBLP) Efficiency Experiment

Large-scale sparse benchmark for Table 4 / Figure 9 style outputs.

## Expected input format

- Undirected edge list text file
- Each line: `node_u node_v`
- Comment lines start with `#` by default
- Node ids can be non-contiguous; script remaps them internally

## Run

```bash
python experiments/reference_1_section8_2/exp8_2_dblp_live.py \
  --edgelist /absolute/path/to/com-dblp.ungraph.txt \
  --target-rank 3 \
  --clusters 3 \
  --reps 20 \
  --seed 2026 \
  --q 2 \
  --r 10 \
  --p 0.7
```

## Outputs

By default, results are written to:

- `experiments/reference_1_section8_2/results/dblp_live/dblp_time_raw.csv`
- `experiments/reference_1_section8_2/results/dblp_live/dblp_table4_like_median_time.csv`
- `experiments/reference_1_section8_2/results/dblp_live/dblp_pairwise_ari_raw.csv`
- `experiments/reference_1_section8_2/results/dblp_live/dblp_pairwise_ari_mean_matrix.csv`
- `experiments/reference_1_section8_2/results/dblp_live/dblp_pairwise_ari_heatmap.png`
- `experiments/reference_1_section8_2/results/dblp_live/dblp_meta.json`
- `experiments/reference_1_section8_2/results/dblp_live/dblp_warnings.csv` (if fallback used)
