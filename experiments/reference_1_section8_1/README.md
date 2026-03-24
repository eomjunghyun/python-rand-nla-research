# Section 8.1 (European Email) Accuracy Experiment

Fresh implementation for paper-aligned accuracy evaluation:

- Random Projection (`q=2`, `r=10`)
- Random Sampling (`p=0.7`, `p=0.8`)
- Non-random spectral clustering

## Input format

- Directed edge list text file
- Each line: `node_u node_v`
- Label file format: `node_id class_id` per line
- Script converts to undirected graph and keeps largest connected component

## Run

```bash
python experiments/reference_1_section8_1/exp8_1_email_eu_core_live.py \
  --edge-path data/email-Eu-core.txt \
  --label-path data/email-Eu-core-department-labels.txt \
  --reps 20 \
  --seed 2026 \
  --q 2 \
  --r 10 \
  --p-values 0.7,0.8
```

## Outputs

- `email_eu_raw_per_rep.csv`
- `email_eu_summary_mean_std.csv`
- `email_eu_table2a_like.csv`
- `email_eu_table2a_like.md`
- `email_eu_pairwise_ari_raw.csv`
- `email_eu_pairwise_ari_mean_matrix.csv`
- `email_eu_pairwise_ari_heatmap.png`
- `email_eu_meta.json`
