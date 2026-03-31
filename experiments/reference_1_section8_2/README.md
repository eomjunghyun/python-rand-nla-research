# Section 8.2 (Table 4-style) Efficiency Experiment

Paper-aligned timing benchmark for three large-scale real networks:

- DBLP collaboration network
- Youtube social network
- Internet topology graph

Methods reproduced in this workspace:

- Random Projection
- Random Sampling
- `partial_eigen` (Python proxy using `scipy.sparse.linalg.eigsh`)

## Paper alignment

- The timing target matches Table 4 of the paper: only the eigenvector computation step is timed.
- `k-means`, ARI, and any clustering post-processing are excluded.
- For Random Sampling, both
  - time including sampling, and
  - time excluding sampling
  are reported, matching the paper's parenthesized format.

## Dataset files

Expected local files:

- `data/com-dblp.ungraph.txt`
- `data/com-youtube.ungraph.txt`
- `data/as-skitter.txt`

## Run

```bash
python experiments/reference_1_section8_2/exp8_2_live.py \
  --dblp-edgelist data/com-dblp.ungraph.txt \
  --youtube-edgelist data/com-youtube.ungraph.txt \
  --internet-edgelist data/as-skitter.txt \
  --reps 20 \
  --seed 2026 \
  --q 2 \
  --r 10 \
  --p 0.7
```

## Outputs

- `table4_time_raw.csv`
- `table4_like_median_time.csv`
- `table4_like_median_time.md`
- `table4_meta.json`
- `viz/table4_median_bar.png`
- `viz/table4_runtime_boxplots.png`

## Notes

- Target ranks are fixed from the paper's Table 3:
  - DBLP: `3`
  - Youtube: `7`
  - Internet: `4`
- Since the original paper used R implementations, `partial_eigen` here is a Python-side proxy rather than the exact same software stack.
