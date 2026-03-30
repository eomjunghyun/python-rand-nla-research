# Hypergraph SBM Generator

Hypergraph SBM (HSBM) data generation for randomized hypergraph spectral clustering studies.

## What this script supports

- Uniform HSBM (all hyperedges have the same size `m`)
- Non-uniform HSBM (mixed hyperedge sizes, e.g. `m in {2,3,4}`)
- Two probabilities:
  - `p_in`: all nodes in the hyperedge are from the same community
  - `p_out`: otherwise (mixed-community hyperedge)
- Auto mode that switches from exact Bernoulli sampling to sparse sampling for large candidate spaces

## Run

Uniform 3-uniform HSBM:

```bash
python experiments/hypergraph_sbm/generate_hsbm_live.py \
  --n 600 \
  --K 3 \
  --m 3 \
  --p-in 0.02 \
  --p-out 0.002 \
  --seed 2026 \
  --sampling auto \
  --save-incidence \
  --save-clique-adj \
  --save-laplacian
```

Non-uniform HSBM (`m=2,3,4`):

```bash
python experiments/hypergraph_sbm/generate_hsbm_live.py \
  --n 800 \
  --K 4 \
  --m-values 2,3,4 \
  --p-in-by-m 2:0.02,3:0.008,4:0.003 \
  --p-out-by-m 2:0.003,3:0.001,4:0.0004 \
  --seed 2026 \
  --sampling auto \
  --save-incidence \
  --save-clique-adj
```

## Outputs

By default (`--outdir experiments/hypergraph_sbm/results/hsbm_live`):

- `hyperedges.txt`: one hyperedge per line (`node_i node_j ...`)
- `labels.csv`: node-to-community labels (`node,label`)
- `meta.json`: generation stats and model config
- `incidence_matrix.npz` (optional)
- `clique_adjacency.npz` (optional)
- `zhou_laplacian.npz` (optional)

## Notes

- `sampling=exact` follows Bernoulli sampling over all candidate hyperedges.
- `sampling=sparse` is for larger settings and avoids full candidate enumeration.
- For very dense settings with large `n` and large `m`, generation can still be expensive.
