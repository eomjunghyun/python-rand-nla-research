# -*- coding: utf-8 -*-

"""LastFM A+B experiment: normalized adjacency, r sweep, degree-stratum metrics."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

from degree_stratified_rp_powerlaw import (
    align_labels_weighted_hungarian,
    degree_stratified_random_projection,
    gaussian_random_projection,
    kmeans_on_rows,
    normalize_rows_l2,
    power_law_diagnostics,
)
from lastfm_operator_r_sweep import (
    degree_tempered_operator,
    load_lastfm_asia,
    parse_csv_list,
    select_subgraph,
)


METHODS = ("Gaussian RP", "Degree-stratified RP")


@dataclass(frozen=True)
class StratumExperimentConfig:
    dataset_path: Path
    outdir: Path = Path("results/lastfm_degree_stratum_alpha05_tau0")
    dataset_name: str = "lastfm-asia"
    r_values: tuple[int, ...] = (0, 2, 5, 10, 20, 40)
    alpha: float = 0.5
    tau: str = "0"
    k: int | None = None
    q: int = 1
    reps: int = 10
    seed: int = 20260519
    ell_min: int = 1
    normalize_embedding_rows: bool = True
    kmeans_n_init: int = 20
    scale_test_matrix_by_dim: bool = False
    max_n: int = 0
    subgraph_mode: str = "none"
    no_plots: bool = False


def resolve_tau(value: str, degrees: np.ndarray):
    text = str(value).strip().lower()
    if text in {"mean", "avg"}:
        return float(np.mean(degrees))
    if text in {"median", "med"}:
        return float(np.median(degrees))
    if text in {"zero", "none"}:
        return 0.0
    return float(text)


def degree_strata(degrees: np.ndarray):
    return {
        "all": np.ones(degrees.shape[0], dtype=bool),
        "low_deg_1_2": (degrees >= 1) & (degrees <= 2),
        "mid_deg_3_8": (degrees >= 3) & (degrees <= 8),
        "high_deg_9_plus": degrees >= 9,
    }


def label_metrics_for_pred(y_true: np.ndarray, y_pred: np.ndarray, K: int):
    aligned = align_labels_weighted_hungarian(y_true, y_pred, K)
    return {
        "ARI_true": float(adjusted_rand_score(y_true, y_pred)),
        "NMI_true": float(normalized_mutual_info_score(y_true, y_pred)),
        "F1_macro_true": float(f1_score(y_true, aligned, average="macro")),
    }


def cluster_embedding(U: np.ndarray, y_true: np.ndarray, K: int, rng, cfg: StratumExperimentConfig):
    X = normalize_rows_l2(U) if cfg.normalize_embedding_rows else U
    return kmeans_on_rows(X, K, rng, n_init=cfg.kmeans_n_init)


def run_one_method(S, degrees, y_true, K, cfg: StratumExperimentConfig, r: int, rep: int, seed: int, method: str):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if method == "Gaussian RP":
        _, U, timings = gaussian_random_projection(
            S,
            K,
            r,
            cfg.q,
            rng,
            scale_by_dim=cfg.scale_test_matrix_by_dim,
        )
        bucket_rows = []
    elif method == "Degree-stratified RP":
        _, U, timings, bucket_rows = degree_stratified_random_projection(
            S,
            K,
            r,
            cfg.q,
            cfg.ell_min,
            rng,
            bucket_degrees=degrees,
            scale_by_dim=cfg.scale_test_matrix_by_dim,
        )
    else:
        raise ValueError(method)
    embedding_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = cluster_embedding(U, y_true, K, rng, cfg)
    clustering_sec = time.perf_counter() - t0

    base = {
        "method": method,
        "rep": int(rep),
        "r": int(r),
        "ell": int(K + r),
        "k": int(K),
        "q": int(cfg.q),
        "test_matrix_scaling": "by_dim" if cfg.scale_test_matrix_by_dim else "none",
        "embedding_wall_sec": float(embedding_sec),
        "clustering_wall_sec": float(clustering_sec),
        "total_method_wall_sec": float(embedding_sec + clustering_sec),
        **timings,
    }
    return base, y_pred, bucket_rows


def summarize(raw: pd.DataFrame):
    group_cols = ["group", "r", "method"]
    numeric_cols = [
        c
        for c in raw.columns
        if c not in set(group_cols + ["dataset", "method", "ds_bucket_summary"])
        and pd.api.types.is_numeric_dtype(raw[c])
    ]
    aggs: dict[str, Any] = {"runs": ("rep", "count")}
    for col in numeric_cols:
        if col == "rep":
            continue
        aggs[f"{col}_mean"] = (col, "mean")
        aggs[f"{col}_std"] = (col, "std")
    return raw.groupby(group_cols, as_index=False).agg(**aggs)


def paired_diff(raw: pd.DataFrame):
    metrics = ["ARI_true", "NMI_true", "F1_macro_true", "total_method_wall_sec"]
    rows = []
    for (group, r), block in raw.groupby(["group", "r"], sort=True):
        by_rep: dict[int, dict[str, dict[str, Any]]] = {}
        for row in block.to_dict("records"):
            by_rep.setdefault(int(row["rep"]), {})[row["method"]] = row
        for metric in metrics:
            diffs = []
            for methods in by_rep.values():
                if "Gaussian RP" in methods and "Degree-stratified RP" in methods:
                    diffs.append(
                        float(methods["Degree-stratified RP"][metric])
                        - float(methods["Gaussian RP"][metric])
                    )
            if not diffs:
                continue
            arr = np.asarray(diffs, dtype=float)
            sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            se = sd / math.sqrt(arr.size) if arr.size > 1 else 0.0
            rows.append(
                {
                    "group": group,
                    "r": int(r),
                    "metric": metric,
                    "diff_mean_ds_minus_gaussian": float(arr.mean()),
                    "diff_std": sd,
                    "diff_se": se,
                    "n_pairs": int(arr.size),
                }
            )
    return pd.DataFrame(rows)


def plot_results(summary: pd.DataFrame, paired: pd.DataFrame, outdir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    outdir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("F1_macro_true_mean", "Macro F1"),
        ("ARI_true_mean", "ARI"),
        ("NMI_true_mean", "NMI"),
    ]
    colors = {"Gaussian RP": "#4C78A8", "Degree-stratified RP": "#E45756"}
    for group in ["all", "low_deg_1_2", "mid_deg_3_8", "high_deg_9_plus"]:
        block = summary[summary["group"] == group]
        if block.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
        for ax, (col, ylabel) in zip(axes, metrics):
            for method in METHODS:
                d = block[block["method"] == method].sort_values("r")
                std_col = col.replace("_mean", "_std")
                ax.errorbar(
                    d["r"],
                    d[col],
                    yerr=d[std_col],
                    marker="o",
                    linewidth=2,
                    capsize=3,
                    label=method,
                    color=colors[method],
                )
            ax.set_title(ylabel)
            ax.set_xlabel("r")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("score")
        axes[-1].legend()
        fig.suptitle(f"LastFM normalized adjacency: {group}")
        fig.tight_layout()
        fig.savefig(outdir / f"lastfm_stratum_scores_{group}.png", dpi=200)
        plt.close(fig)

    for metric, ylabel in [("F1_macro_true", "Macro F1"), ("ARI_true", "ARI"), ("NMI_true", "NMI")]:
        d = paired[paired["metric"] == metric]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for group in ["all", "low_deg_1_2", "mid_deg_3_8", "high_deg_9_plus"]:
            g = d[d["group"] == group].sort_values("r")
            if g.empty:
                continue
            ax.plot(g["r"], g["diff_mean_ds_minus_gaussian"], marker="o", linewidth=2, label=group)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"DS-RP minus Gaussian RP: {ylabel}")
        ax.set_xlabel("r")
        ax.set_ylabel("paired difference")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"lastfm_stratum_paired_diff_{metric}.png", dpi=200)
        plt.close(fig)


def run_experiment(cfg: StratumExperimentConfig):
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    A_full, y_full, ids_full, source_meta = load_lastfm_asia(cfg.dataset_path)
    A, y_true, ids, subgraph_meta = select_subgraph(
        A_full, y_full, ids_full, cfg.max_n, cfg.subgraph_mode
    )
    K = int(cfg.k) if cfg.k is not None else int(np.unique(y_true).size)
    degrees = np.asarray(A.sum(axis=1)).ravel().astype(float)
    tau_value = resolve_tau(cfg.tau, degrees)
    S = degree_tempered_operator(A, alpha=cfg.alpha, tau=tau_value)
    strata = degree_strata(degrees)
    strata_meta = {
        name: {
            "num_nodes": int(mask.sum()),
            "num_classes_present": int(np.unique(y_true[mask]).size) if mask.sum() else 0,
            "degree_min": float(np.min(degrees[mask])) if mask.sum() else None,
            "degree_max": float(np.max(degrees[mask])) if mask.sum() else None,
        }
        for name, mask in strata.items()
    }

    print(
        f"Loaded {cfg.dataset_name}: n={A.shape[0]}, m={A.nnz // 2}, "
        f"classes={np.unique(y_true).size}, k={K}, alpha={cfg.alpha}, tau={cfg.tau}, "
        f"test_matrix_scaling={'by_dim' if cfg.scale_test_matrix_by_dim else 'none'}"
    )

    rows = []
    bucket_rows_all = []
    master_rng = np.random.default_rng(cfg.seed)
    for r in cfg.r_values:
        for rep in range(1, cfg.reps + 1):
            rep_seed = int(master_rng.integers(1, 2**31 - 1))
            print(f"r={r} rep={rep}/{cfg.reps}", flush=True)
            for method_index, method in enumerate(METHODS):
                base, y_pred, bucket_rows = run_one_method(
                    S,
                    degrees,
                    y_true,
                    K,
                    cfg,
                    r,
                    rep,
                    seed=rep_seed + 10_000 * (method_index + 1),
                    method=method,
                )
                for group, mask in strata.items():
                    metrics = label_metrics_for_pred(y_true[mask], y_pred[mask], K)
                    rows.append(
                        {
                            "dataset": cfg.dataset_name,
                            "alpha": float(cfg.alpha),
                            "tau_label": str(cfg.tau),
                            "tau_value": float(tau_value),
                            "group": group,
                            "group_n": int(mask.sum()),
                            **base,
                            **metrics,
                        }
                    )
                if method == "Degree-stratified RP":
                    for bucket in bucket_rows:
                        bucket_rows_all.append({"r": int(r), "rep": int(rep), **bucket})

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    paired = paired_diff(raw)
    buckets = pd.DataFrame(bucket_rows_all)

    raw_path = cfg.outdir / "lastfm_degree_stratum_raw.csv"
    summary_path = cfg.outdir / "lastfm_degree_stratum_summary.csv"
    paired_path = cfg.outdir / "lastfm_degree_stratum_paired_ds_minus_gaussian.csv"
    buckets_path = cfg.outdir / "lastfm_degree_stratum_bucket_allocations.csv"
    meta_path = cfg.outdir / "lastfm_degree_stratum_meta.json"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    meta = {
        "config": {
            **asdict(cfg),
            "dataset_path": str(cfg.dataset_path),
            "outdir": str(cfg.outdir),
        },
        "source": source_meta,
        "graph": {
            "full_nodes": int(A_full.shape[0]),
            "full_edges": int(A_full.nnz // 2),
            "experiment_nodes": int(A.shape[0]),
            "experiment_edges": int(A.nnz // 2),
            "num_classes": int(np.unique(y_true).size),
            "k_used": int(K),
            **subgraph_meta,
            **power_law_diagnostics(A),
        },
        "strata": strata_meta,
        "operator": "S = D^{-1/2} A D^{-1/2}",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cfg.no_plots:
        plot_results(summary, paired, cfg.outdir / "viz")

    print("Done.")
    print(f"Raw CSV     : {raw_path.resolve()}")
    print(f"Summary CSV : {summary_path.resolve()}")
    print(f"Paired CSV  : {paired_path.resolve()}")
    print(f"Buckets CSV : {buckets_path.resolve()}")
    print(f"Meta JSON   : {meta_path.resolve()}")
    return raw, summary, paired, buckets, meta


def parse_args():
    parser = argparse.ArgumentParser(description="LastFM normalized adjacency degree-stratum experiment")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/lastfm_degree_stratum_alpha05_tau0"))
    parser.add_argument("--dataset-name", type=str, default="lastfm-asia")
    parser.add_argument("--r-values", type=str, default="0,2,5,10,20,40")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=str, default="0")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--ell-min", type=int, default=1)
    parser.add_argument("--no-normalize-embedding-rows", action="store_true")
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument(
        "--scale-test-matrix-by-dim",
        action="store_true",
        help="Scale Gaussian test matrix entries by the inverse square root of their sketch dimension.",
    )
    parser.add_argument("--max-n", type=int, default=0)
    parser.add_argument(
        "--subgraph-mode",
        choices=["none", "top-degree", "degree-weighted", "uniform"],
        default="none",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = StratumExperimentConfig(
        dataset_path=args.dataset_path,
        outdir=args.outdir,
        dataset_name=args.dataset_name,
        r_values=parse_csv_list(args.r_values, int),
        alpha=args.alpha,
        tau=args.tau,
        k=args.k,
        q=args.q,
        reps=args.reps,
        seed=args.seed,
        ell_min=args.ell_min,
        normalize_embedding_rows=not args.no_normalize_embedding_rows,
        kmeans_n_init=args.kmeans_n_init,
        scale_test_matrix_by_dim=args.scale_test_matrix_by_dim,
        max_n=args.max_n,
        subgraph_mode=args.subgraph_mode,
        no_plots=args.no_plots,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
