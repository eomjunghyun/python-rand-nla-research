# Section 8.2 (DBLP) Efficiency Experiment

Fresh implementation for three methods:

- Random Projection
- Random Sampling
- Non-random spectral clustering

## Input format

- Undirected edge list text file
- Each line: `node_u node_v`
- Comment lines start with `#` by default
- Node IDs may be non-contiguous (script remaps them)

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

- `dblp_time_raw.csv`
- `dblp_table4_like_median_time.csv`
- `dblp_pairwise_ari_raw.csv`
- `dblp_pairwise_ari_mean_matrix.csv`
- `dblp_pairwise_ari_heatmap.png`
- `dblp_meta.json`
