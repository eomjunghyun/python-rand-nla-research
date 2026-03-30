# -*- coding: utf-8 -*-

"""Generate hypergraph-SBM data for hypergraph spectral clustering research."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hypergraph_sbm import (  # noqa: E402
    clique_expansion_adjacency,
    generate_nonuniform_hsbm_instance,
    generate_uniform_hsbm_instance,
    hyperedges_to_incidence_csr,
    hypergraph_basic_stats,
    zhou_normalized_laplacian,
)


def parse_int_csv(text: str):
    vals = []
    for tok in text.split(","):
        s = tok.strip()
        if s:
            vals.append(int(s))
    return vals


def parse_prob_by_m(text: str):
    mapping = {}
    if not text:
        return mapping
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        if ":" not in s:
            raise ValueError(f"Invalid mapping token '{s}'. Use format m:p")
        m_str, p_str = s.split(":", 1)
        mapping[int(m_str.strip())] = float(p_str.strip())
    return mapping


def write_hyperedges(path: Path, hyperedges):
    with path.open("w", encoding="utf-8") as f:
        for edge in hyperedges:
            f.write(" ".join(str(int(v)) for v in edge))
            f.write("\n")


def write_labels(path: Path, y_true: np.ndarray):
    with path.open("w", encoding="utf-8") as f:
        f.write("node,label\n")
        for i, c in enumerate(y_true.tolist()):
            f.write(f"{i},{int(c)}\n")


def main():
    ap = argparse.ArgumentParser(description="Generate hypergraph SBM instances")
    ap.add_argument("--n", type=int, default=600, help="Number of nodes")
    ap.add_argument("--K", type=int, default=3, help="Number of communities")
    ap.add_argument("--m", type=int, default=3, help="Uniform hyperedge size (if --m-values not set)")
    ap.add_argument(
        "--m-values",
        type=str,
        default="",
        help="Comma-separated hyperedge sizes for non-uniform HSBM (e.g., 2,3,4)",
    )
    ap.add_argument("--p-in", type=float, default=0.02, help="Within-community hyperedge probability")
    ap.add_argument("--p-out", type=float, default=0.002, help="Cross-community hyperedge probability")
    ap.add_argument(
        "--p-in-by-m",
        type=str,
        default="",
        help="Per-size within probability map (e.g., '2:0.03,3:0.01')",
    )
    ap.add_argument(
        "--p-out-by-m",
        type=str,
        default="",
        help="Per-size cross probability map (e.g., '2:0.004,3:0.001')",
    )
    ap.add_argument(
        "--sampling",
        type=str,
        default="auto",
        choices=["auto", "exact", "sparse"],
        help="Edge generation mode",
    )
    ap.add_argument(
        "--max-enumeration",
        type=int,
        default=1500000,
        help="Auto mode threshold for switching from exact to sparse sampling",
    )
    ap.add_argument("--seed", type=int, default=2026, help="RNG seed")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/hypergraph_sbm/results/hsbm_live"),
        help="Output directory",
    )
    ap.add_argument("--save-incidence", action="store_true", help="Save incidence matrix as .npz")
    ap.add_argument("--save-clique-adj", action="store_true", help="Save clique expansion adjacency as .npz")
    ap.add_argument("--save-laplacian", action="store_true", help="Save Zhou normalized Laplacian as .npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.m_values.strip():
        m_values = parse_int_csv(args.m_values)
        pin_map = parse_prob_by_m(args.p_in_by_m)
        pout_map = parse_prob_by_m(args.p_out_by_m)
        pin = pin_map if pin_map else float(args.p_in)
        pout = pout_map if pout_map else float(args.p_out)

        hyperedges, y_true, _, model_stats = generate_nonuniform_hsbm_instance(
            n=args.n,
            K=args.K,
            m_values=m_values,
            p_in=pin,
            p_out=pout,
            rng=rng,
            sampling=args.sampling,
            max_enumeration=args.max_enumeration,
        )
    else:
        hyperedges, y_true, _, model_stats = generate_uniform_hsbm_instance(
            n=args.n,
            K=args.K,
            m=args.m,
            p_in=args.p_in,
            p_out=args.p_out,
            rng=rng,
            sampling=args.sampling,
            max_enumeration=args.max_enumeration,
        )

    stats = hypergraph_basic_stats(args.n, hyperedges, labels=y_true)
    stats["model"] = model_stats

    edges_path = outdir / "hyperedges.txt"
    labels_path = outdir / "labels.csv"
    stats_path = outdir / "meta.json"
    write_hyperedges(edges_path, hyperedges)
    write_labels(labels_path, y_true)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if args.save_incidence:
        H = hyperedges_to_incidence_csr(args.n, hyperedges)
        sp.save_npz(outdir / "incidence_matrix.npz", H)

    if args.save_clique_adj:
        A = clique_expansion_adjacency(args.n, hyperedges)
        sp.save_npz(outdir / "clique_adjacency.npz", A)

    if args.save_laplacian:
        L = zhou_normalized_laplacian(args.n, hyperedges)
        sp.save_npz(outdir / "zhou_laplacian.npz", L)

    print("HSBM generation complete")
    print(f"- nodes: {args.n}")
    print(f"- communities: {args.K}")
    print(f"- hyperedges: {len(hyperedges)}")
    print(f"- outputs: {outdir}")


if __name__ == "__main__":
    main()
