# -*- coding: utf-8 -*-

"""Section 8.2 (DBLP) efficiency experiment on a large sparse graph."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    LiveProgress,
    eigvecs_eigsh_sparse,
    eigvecs_lobpcg_sparse,
    eigvecs_random_projection_sparse,
    eigvecs_random_sampling_sparse,
    kmeans_on_rows,
    load_undirected_edgelist_csr,
    pairwise_ari,
    singular_vecs_randomized_svd_sparse,
    singular_vecs_svds_sparse,
    upper_triangle_edges,
)


METHODS_82 = [
    "Random Projection",
    "Random Sampling",
    "irlba",
    "svds",
    "svdr",
    "partial_eigen",
]


@dataclass
class Exp82Config:
    edgelist: Path
    target_rank: int = 3
    n_clusters: int = 3
    reps: int = 20
    seed: int = 2026
    r: int = 10
    q: int = 2
    p: float = 0.7
    delimiter: str = None
    comment_prefix: str = "#"
    outdir: Path = Path("experiments/reference_1_section8_2/results/dblp_live")
    no_progress: bool = False


def _ari_mean_matrix(df_ari: pd.DataFrame):
    mat = pd.DataFrame(np.eye(len(METHODS_82)), index=METHODS_82, columns=METHODS_82)
    for _, row in df_ari.groupby(["method_i", "method_j"], as_index=False)["ari"].mean().iterrows():
        i = row["method_i"]
        j = row["method_j"]
        mat.loc[i, j] = row["ari"]
        mat.loc[j, i] = row["ari"]
    return mat


def _plot_ari_heatmap(ari_mat: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(ari_mat.values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(ari_mat.columns)))
    ax.set_yticks(range(len(ari_mat.index)))
    ax.set_xticklabels(ari_mat.columns, rotation=35, ha="right")
    ax.set_yticklabels(ari_mat.index)

    for i in range(ari_mat.shape[0]):
        for j in range(ari_mat.shape[1]):
            v = ari_mat.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v < 0.55 else "black",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ARI")
    ax.set_title("Pairwise ARI (mean across replications)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(cfg: Exp82Config):
    A, _node_to_idx = load_undirected_edgelist_csr(
        cfg.edgelist,
        delimiter=cfg.delimiter,
        comment_prefix=cfg.comment_prefix,
    )
    n_nodes = A.shape[0]
    n_edges = A.nnz // 2
    upper_rows, upper_cols = upper_triangle_edges(A)

    master_rng = np.random.default_rng(cfg.seed)
    time_rows = []
    ari_rows = []
    warn_rows = []

    total_steps = cfg.reps * len(METHODS_82)
    progress = None if cfg.no_progress else LiveProgress(total_steps)

    for rep in range(1, cfg.reps + 1):
        rep_seed = int(master_rng.integers(1, 2**31 - 1))
        rng = np.random.default_rng(rep_seed)
        label_map = {}

        t0 = perf_counter()
        _, U_rp = eigvecs_random_projection_sparse(A, cfg.target_rank, cfg.r, cfg.q, rng)
        t_rp = perf_counter() - t0
        label_map["Random Projection"] = kmeans_on_rows(U_rp, cfg.n_clusters, rng)
        time_rows.append({"rep": rep, "method": "Random Projection", "time_sec": t_rp})
        if progress is not None:
            progress.update("rep", rep, rep, cfg.reps, "Random Projection")

        U_rs, t_rs_with, t_rs_without = eigvecs_random_sampling_sparse(
            n=n_nodes,
            upper_rows=upper_rows,
            upper_cols=upper_cols,
            p=cfg.p,
            k=cfg.target_rank,
            rng=rng,
        )
        label_map["Random Sampling"] = kmeans_on_rows(U_rs, cfg.n_clusters, rng)
        time_rows.append(
            {
                "rep": rep,
                "method": "Random Sampling",
                "time_sec": t_rs_with,
                "time_sec_excl_sampling": t_rs_without,
            }
        )
        if progress is not None:
            progress.update("rep", rep, rep, cfg.reps, "Random Sampling")

        t0 = perf_counter()
        try:
            _, U_irlba = singular_vecs_svds_sparse(A, cfg.target_rank, solver="lobpcg")
        except Exception as e:
            warn_rows.append(
                {"rep": rep, "method": "irlba", "warning": f"lobpcg fallback to arpack: {e}"}
            )
            _, U_irlba = singular_vecs_svds_sparse(A, cfg.target_rank, solver="arpack")
        t_irlba = perf_counter() - t0
        label_map["irlba"] = kmeans_on_rows(U_irlba, cfg.n_clusters, rng)
        time_rows.append({"rep": rep, "method": "irlba", "time_sec": t_irlba})
        if progress is not None:
            progress.update("rep", rep, rep, cfg.reps, "irlba")

        t0 = perf_counter()
        _, U_svds = singular_vecs_svds_sparse(A, cfg.target_rank, solver="arpack")
        t_svds = perf_counter() - t0
        label_map["svds"] = kmeans_on_rows(U_svds, cfg.n_clusters, rng)
        time_rows.append({"rep": rep, "method": "svds", "time_sec": t_svds})
        if progress is not None:
            progress.update("rep", rep, rep, cfg.reps, "svds")

        t0 = perf_counter()
        _, U_svdr = singular_vecs_randomized_svd_sparse(A, cfg.target_rank, n_iter=2, rng=rng)
        t_svdr = perf_counter() - t0
        label_map["svdr"] = kmeans_on_rows(U_svdr, cfg.n_clusters, rng)
        time_rows.append({"rep": rep, "method": "svdr", "time_sec": t_svdr})
        if progress is not None:
            progress.update("rep", rep, rep, cfg.reps, "svdr")

        t0 = perf_counter()
        try:
            _, U_pe = eigvecs_lobpcg_sparse(A, cfg.target_rank, rng)
        except Exception as e:
            warn_rows.append(
                {
                    "rep": rep,
                    "method": "partial_eigen",
                    "warning": f"lobpcg fallback to eigsh: {e}",
                }
            )
            _, U_pe = eigvecs_eigsh_sparse(A, cfg.target_rank)
        t_pe = perf_counter() - t0
        label_map["partial_eigen"] = kmeans_on_rows(U_pe, cfg.n_clusters, rng)
        time_rows.append({"rep": rep, "method": "partial_eigen", "time_sec": t_pe})
        if progress is not None:
            progress.update("rep", rep, rep, cfg.reps, "partial_eigen")

        methods, mat = pairwise_ari(label_map)
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                ari_rows.append(
                    {
                        "rep": rep,
                        "method_i": methods[i],
                        "method_j": methods[j],
                        "ari": mat[i, j],
                    }
                )

    if progress is not None:
        progress.close()

    df_time_raw = pd.DataFrame(time_rows)
    df_ari_raw = pd.DataFrame(ari_rows)
    df_warn = pd.DataFrame(warn_rows)

    med_time = (
        df_time_raw.groupby("method", as_index=False)
        .agg(
            median_time_sec=("time_sec", "median"),
            median_time_excl_sampling_sec=("time_sec_excl_sampling", "median"),
        )
        .assign(method=lambda d: pd.Categorical(d["method"], categories=METHODS_82, ordered=True))
        .sort_values("method")
        .assign(method=lambda d: d["method"].astype(str))
    )
    ari_mat = _ari_mean_matrix(df_ari_raw)

    meta = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "target_rank": cfg.target_rank,
        "n_clusters": cfg.n_clusters,
        "reps": cfg.reps,
        "q": cfg.q,
        "r": cfg.r,
        "p": cfg.p,
        "edgelist": str(cfg.edgelist),
    }
    return meta, df_time_raw, med_time, df_ari_raw, ari_mat, df_warn


def main():
    parser = argparse.ArgumentParser(description="Section 8.2 DBLP efficiency experiment")
    parser.add_argument(
        "--edgelist",
        type=str,
        required=True,
        help="Path to undirected edge-list file (two columns: node_u node_v)",
    )
    parser.add_argument("--target-rank", type=int, default=3, help="target rank K")
    parser.add_argument("--clusters", type=int, default=3, help="cluster number for k-means")
    parser.add_argument("--reps", type=int, default=20, help="replications")
    parser.add_argument("--seed", type=int, default=2026, help="master random seed")
    parser.add_argument("--r", type=int, default=10, help="oversampling for random projection")
    parser.add_argument("--q", type=int, default=2, help="power iteration for random projection")
    parser.add_argument("--p", type=float, default=0.7, help="edge sampling probability")
    parser.add_argument(
        "--delimiter",
        type=str,
        default=None,
        help="Edge delimiter, default=None means whitespace split",
    )
    parser.add_argument("--comment-prefix", type=str, default="#")
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/reference_1_section8_2/results/dblp_live",
    )
    parser.add_argument("--no-progress", action="store_true")
    args, _ = parser.parse_known_args()

    cfg = Exp82Config(
        edgelist=Path(args.edgelist),
        target_rank=args.target_rank,
        n_clusters=args.clusters,
        reps=args.reps,
        seed=args.seed,
        r=args.r,
        q=args.q,
        p=args.p,
        delimiter=args.delimiter,
        comment_prefix=args.comment_prefix,
        outdir=Path(args.outdir),
        no_progress=args.no_progress,
    )

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    meta, df_time_raw, med_time, df_ari_raw, ari_mat, df_warn = run_experiment(cfg)

    time_raw_csv = cfg.outdir / "dblp_time_raw.csv"
    time_med_csv = cfg.outdir / "dblp_table4_like_median_time.csv"
    ari_raw_csv = cfg.outdir / "dblp_pairwise_ari_raw.csv"
    ari_mat_csv = cfg.outdir / "dblp_pairwise_ari_mean_matrix.csv"
    ari_png = cfg.outdir / "dblp_pairwise_ari_heatmap.png"
    meta_json = cfg.outdir / "dblp_meta.json"
    warn_csv = cfg.outdir / "dblp_warnings.csv"

    df_time_raw.to_csv(time_raw_csv, index=False)
    med_time.to_csv(time_med_csv, index=False)
    df_ari_raw.to_csv(ari_raw_csv, index=False)
    ari_mat.to_csv(ari_mat_csv, index=True)
    _plot_ari_heatmap(ari_mat, ari_png)
    pd.Series(meta).to_json(meta_json, indent=2)
    if not df_warn.empty:
        df_warn.to_csv(warn_csv, index=False)

    print("Done.")
    print(f"Nodes / edges : {meta['n_nodes']} / {meta['n_edges']}")
    print(f"Raw time CSV  : {time_raw_csv.resolve()}")
    print(f"Median table  : {time_med_csv.resolve()}")
    print(f"Raw ARI CSV   : {ari_raw_csv.resolve()}")
    print(f"Mean ARI CSV  : {ari_mat_csv.resolve()}")
    print(f"ARI heatmap   : {ari_png.resolve()}")
    print(f"Meta JSON     : {meta_json.resolve()}")
    if not df_warn.empty:
        print(f"Warnings CSV  : {warn_csv.resolve()}")


if __name__ == "__main__":
    main()
