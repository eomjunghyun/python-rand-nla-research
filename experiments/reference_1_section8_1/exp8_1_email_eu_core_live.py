# -*- coding: utf-8 -*-

"""Section 8.1 (European email) accuracy experiment.

Paper-aligned setup:
- Methods: Random Projection, Random Sampling (p=0.7, 0.8), Non-random
- Metrics: F1, NMI, ARI
- Repetitions: 20
- RP params: q=2, r=10
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    LiveProgress,
    align_labels_weighted_hungarian,
    eigvecs_eigsh_sparse,
    eigvecs_random_projection_sparse,
    eigvecs_random_sampling_sparse,
    kmeans_on_rows,
    load_undirected_edgelist_csr,
    pairwise_ari,
    upper_triangle_edges,
)


@dataclass
class Exp81Config:
    edge_path: Path
    label_path: Path
    reps: int = 20
    seed: int = 2026
    q: int = 2
    r: int = 10
    p_values: tuple = (0.7, 0.8)
    n_init: int = 20
    outdir: Path = Path("experiments/reference_1_section8_1/results/exp8_1_email_eu_core_table2_like")
    no_progress: bool = False


def parse_label_file(path: Path):
    labels = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            node = int(parts[0])
            lab = int(parts[1])
            labels[node] = lab
    return labels


def remap_to_zero_based(y: np.ndarray) -> np.ndarray:
    uniq = np.unique(y)
    mp = {int(v): i for i, v in enumerate(uniq)}
    return np.array([mp[int(v)] for v in y], dtype=int)


def largest_connected_component_indices(A):
    _, comp = connected_components(A, directed=False, return_labels=True)
    counts = np.bincount(comp)
    lcc_id = int(np.argmax(counts))
    return np.where(comp == lcc_id)[0]


def directed_graph_stats(edge_path: Path, labels_map: dict):
    edges = []
    with edge_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            edges.append((int(parts[0]), int(parts[1])))
    directed_edges = len(edges)
    max_edge_node = max(max(u, v) for u, v in edges) if edges else -1
    max_label_node = max(labels_map.keys()) if labels_map else -1
    directed_nodes = max(max_edge_node, max_label_node) + 1
    return directed_nodes, directed_edges


def load_email_eu_core_lcc(edge_path: Path, label_path: Path):
    A_und, node_to_idx = load_undirected_edgelist_csr(edge_path, delimiter=None, comment_prefix="#")
    labels_map = parse_label_file(label_path)
    directed_nodes, directed_edges = directed_graph_stats(edge_path, labels_map)

    n = A_und.shape[0]
    y_all = np.full(n, -1, dtype=int)
    for node, lab in labels_map.items():
        key = str(node)
        if key in node_to_idx:
            y_all[node_to_idx[key]] = lab

    lcc_idx = largest_connected_component_indices(A_und)
    A_lcc = A_und[lcc_idx][:, lcc_idx].tocsr()
    y_lcc = y_all[lcc_idx]
    if np.any(y_lcc < 0):
        raise ValueError("Some LCC nodes have missing labels in label file.")
    y_lcc = remap_to_zero_based(y_lcc)
    K = int(np.unique(y_lcc).size)

    stats = {
        "directed_nodes": int(directed_nodes),
        "directed_edges": int(directed_edges),
        "undirected_nodes": int(A_und.shape[0]),
        "undirected_edges": int(A_und.nnz // 2),
        "lcc_nodes": int(A_lcc.shape[0]),
        "lcc_edges": int(A_lcc.nnz // 2),
        "num_classes": int(K),
    }
    return A_lcc, y_lcc, K, stats


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, K: int):
    y_aligned = align_labels_weighted_hungarian(y_true, y_pred, K)
    f1 = float(f1_score(y_true, y_aligned, average="macro"))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    ari = float(adjusted_rand_score(y_true, y_pred))
    return f1, nmi, ari


def run_experiment(cfg: Exp81Config, A, y_true: np.ndarray, K: int):
    Kp = K
    rs_methods = [f"Random Sampling (p={p:g})" for p in cfg.p_values]
    methods = ["Random Projection"] + rs_methods + ["Non-random"]

    upper_rows, upper_cols = upper_triangle_edges(A)
    master_rng = np.random.default_rng(cfg.seed)
    rows = []
    pair_rows = []
    progress = None if cfg.no_progress else LiveProgress(cfg.reps * len(methods))

    for rep in range(1, cfg.reps + 1):
        rep_seed = int(master_rng.integers(1, 2**31 - 1))
        labels = {}

        rng_rp = np.random.default_rng(rep_seed + 11)
        t0 = perf_counter()
        _, U_rp = eigvecs_random_projection_sparse(A, k=Kp, r=cfg.r, q=cfg.q, rng=rng_rp)
        y_rp = kmeans_on_rows(U_rp, K, rng_rp)
        dt_rp = perf_counter() - t0
        f1, nmi, ari = evaluate_metrics(y_true, y_rp, K)
        rows.append(
            {
                "rep": rep,
                "method": "Random Projection",
                "F1": f1,
                "NMI": nmi,
                "ARI": ari,
                "time_rand_sec": np.nan,
                "time_post_sec": np.nan,
                "time_total_sec": dt_rp,
            }
        )
        labels["Random Projection"] = y_rp.copy()
        if progress is not None:
            progress.update("rep", rep, rep, cfg.reps, "Random Projection")

        for p in cfg.p_values:
            mname = f"Random Sampling (p={p:g})"
            rng_rs = np.random.default_rng(rep_seed + int(round(p * 1000)) + 31)
            t0 = perf_counter()
            U_rs, t_rs_with, t_rs_without = eigvecs_random_sampling_sparse(
                n=A.shape[0],
                upper_rows=upper_rows,
                upper_cols=upper_cols,
                p=p,
                k=Kp,
                rng=rng_rs,
            )
            t1 = perf_counter()
            y_rs = kmeans_on_rows(U_rs, K, rng_rs)
            t_km = perf_counter() - t1
            dt_rs = perf_counter() - t0
            t_rand = max(0.0, float(t_rs_with - t_rs_without))
            t_post = float(t_rs_without + t_km)
            f1, nmi, ari = evaluate_metrics(y_true, y_rs, K)
            rows.append(
                {
                    "rep": rep,
                    "method": mname,
                    "F1": f1,
                    "NMI": nmi,
                    "ARI": ari,
                    "time_rand_sec": t_rand,
                    "time_post_sec": t_post,
                    "time_total_sec": dt_rs,
                }
            )
            labels[mname] = y_rs.copy()
            if progress is not None:
                progress.update("rep", rep, rep, cfg.reps, mname)

        rng_nr = np.random.default_rng(rep_seed + 97)
        t0 = perf_counter()
        _, U_nr = eigvecs_eigsh_sparse(A, Kp)
        t1 = perf_counter()
        y_nr = kmeans_on_rows(U_nr, K, rng_nr)
        t_km = perf_counter() - t1
        dt_nr = perf_counter() - t0
        f1, nmi, ari = evaluate_metrics(y_true, y_nr, K)
        rows.append(
            {
                "rep": rep,
                "method": "Non-random",
                "F1": f1,
                "NMI": nmi,
                "ARI": ari,
                "time_rand_sec": 0.0,
                "time_post_sec": float((t1 - t0) + t_km),
                "time_total_sec": dt_nr,
            }
        )
        labels["Non-random"] = y_nr.copy()
        if progress is not None:
            progress.update("rep", rep, rep, cfg.reps, "Non-random")

        mlist, mat = pairwise_ari(labels)
        for i in range(len(mlist)):
            for j in range(i + 1, len(mlist)):
                pair_rows.append(
                    {
                        "rep": rep,
                        "method_i": mlist[i],
                        "method_j": mlist[j],
                        "ari": float(mat[i, j]),
                    }
                )

    if progress is not None:
        progress.close()

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def summarize(df_raw: pd.DataFrame) -> pd.DataFrame:
    return df_raw.groupby("method", as_index=False).agg(
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        NMI_mean=("NMI", "mean"),
        NMI_std=("NMI", "std"),
        ARI_mean=("ARI", "mean"),
        ARI_std=("ARI", "std"),
        time_rand_mean=("time_rand_sec", "mean"),
        time_rand_std=("time_rand_sec", "std"),
        time_post_mean=("time_post_sec", "mean"),
        time_post_std=("time_post_sec", "std"),
        time_total_mean=("time_total_sec", "mean"),
        time_total_std=("time_total_sec", "std"),
    )


def build_table2a_like(summary: pd.DataFrame, p_values: tuple) -> pd.DataFrame:
    ordered = (
        ["Random Projection"]
        + [f"Random Sampling (p={p:g})" for p in p_values]
        + ["Non-random"]
    )
    d = summary.set_index("method").loc[[m for m in ordered if m in set(summary["method"])]].reset_index()
    disp_map = {
        "Random Projection": "Random Projection",
        "Non-random": "Non-Random",
    }
    for p in p_values:
        disp_map[f"Random Sampling (p={p:g})"] = f"Random Sampling (p= {p:.1f})"

    f1_col = [f"{m:.3f}({s:.3f})" for m, s in zip(d["F1_mean"], d["F1_std"].fillna(0.0))]
    nmi_col = [f"{m:.3f}({s:.3f})" for m, s in zip(d["NMI_mean"], d["NMI_std"].fillna(0.0))]
    ari_col = [f"{m:.3f}({s:.3f})" for m, s in zip(d["ARI_mean"], d["ARI_std"].fillna(0.0))]
    methods = [disp_map.get(m, m) for m in d["method"]]
    return pd.DataFrame({"Methods": methods, "F 1": f1_col, "NMI": nmi_col, "ARI": ari_col})


def write_table2a_markdown(tbl: pd.DataFrame, out_md: Path, reps: int):
    lines = []
    lines.append("Table 2 (a): The clustering performance on the European email network.")
    lines.append("")
    lines.append("| Methods | F 1 | NMI | ARI |")
    lines.append("|---|---:|---:|---:|")
    for _, row in tbl.iterrows():
        lines.append(f"| {row['Methods']} | {row['F 1']} | {row['NMI']} | {row['ARI']} |")
    lines.append("")
    lines.append(f"Note: Values are mean(std) over {reps} replications.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ari_mean_matrix(df_pair: pd.DataFrame, methods: list):
    mat = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods, dtype=float)
    if len(df_pair) == 0:
        return mat
    d = df_pair.groupby(["method_i", "method_j"], as_index=False)["ari"].mean()
    for _, row in d.iterrows():
        i = row["method_i"]
        j = row["method_j"]
        v = float(row["ari"])
        mat.loc[i, j] = v
        mat.loc[j, i] = v
    return mat


def plot_ari_heatmap(ari_mat: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    im = ax.imshow(ari_mat.values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(ari_mat.columns)))
    ax.set_yticks(range(len(ari_mat.index)))
    ax.set_xticklabels(ari_mat.columns, rotation=30, ha="right")
    ax.set_yticklabels(ari_mat.index)
    for i in range(ari_mat.shape[0]):
        for j in range(ari_mat.shape[1]):
            v = float(ari_mat.iat[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.55 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ARI")
    ax.set_title("Pairwise ARI (mean across replications)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_p_values(p_str: str):
    vals = [float(x.strip()) for x in p_str.split(",") if x.strip()]
    vals = [v for v in vals if 0.0 < v <= 1.0]
    if not vals:
        raise ValueError("p-values must be in (0,1].")
    return tuple(vals)


def main():
    parser = argparse.ArgumentParser(description="Section 8.1 European email accuracy experiment")
    parser.add_argument("--edge-path", type=str, default="data/email-Eu-core.txt", help="directed edge list path")
    parser.add_argument("--label-path", type=str, default="data/email-Eu-core-department-labels.txt", help="node label path")
    parser.add_argument("--reps", type=int, default=20, help="replications")
    parser.add_argument("--seed", type=int, default=2026, help="master random seed")
    parser.add_argument("--q", type=int, default=2, help="power parameter for RP")
    parser.add_argument("--r", type=int, default=10, help="oversampling parameter for RP")
    parser.add_argument("--p-values", type=str, default="0.7,0.8", help="comma-separated p list for RS")
    parser.add_argument("--n-init", type=int, default=20, help="kmeans n_init")
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/reference_1_section8_1/results/exp8_1_email_eu_core_table2_like",
        help="output directory",
    )
    parser.add_argument("--no-progress", action="store_true")
    args, _ = parser.parse_known_args()

    cfg = Exp81Config(
        edge_path=Path(args.edge_path),
        label_path=Path(args.label_path),
        reps=args.reps,
        seed=args.seed,
        q=args.q,
        r=args.r,
        p_values=parse_p_values(args.p_values),
        n_init=args.n_init,
        outdir=Path(args.outdir),
        no_progress=args.no_progress,
    )

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    print("Running Section 8.1 European email experiment...")
    A, y_true, K, stats = load_email_eu_core_lcc(cfg.edge_path, cfg.label_path)
    print(f"LCC stats: nodes={stats['lcc_nodes']}, edges={stats['lcc_edges']}, K={stats['num_classes']}")

    df_raw, df_pair = run_experiment(cfg, A, y_true, K)
    df_sum = summarize(df_raw)
    table2a = build_table2a_like(df_sum, cfg.p_values)

    raw_csv = cfg.outdir / "email_eu_raw_per_rep.csv"
    sum_csv = cfg.outdir / "email_eu_summary_mean_std.csv"
    table_csv = cfg.outdir / "email_eu_table2a_like.csv"
    table_md = cfg.outdir / "email_eu_table2a_like.md"
    pair_csv = cfg.outdir / "email_eu_pairwise_ari_raw.csv"
    pair_mat_csv = cfg.outdir / "email_eu_pairwise_ari_mean_matrix.csv"
    pair_png = cfg.outdir / "email_eu_pairwise_ari_heatmap.png"
    meta_json = cfg.outdir / "email_eu_meta.json"

    df_raw.to_csv(raw_csv, index=False)
    df_sum.to_csv(sum_csv, index=False)
    table2a.to_csv(table_csv, index=False)
    write_table2a_markdown(table2a, table_md, cfg.reps)

    df_pair.to_csv(pair_csv, index=False)
    methods = (
        ["Random Projection"]
        + [f"Random Sampling (p={p:g})" for p in cfg.p_values]
        + ["Non-random"]
    )
    ari_mat = ari_mean_matrix(df_pair, methods)
    ari_mat.to_csv(pair_mat_csv, index=True)
    plot_ari_heatmap(ari_mat, pair_png)

    meta = {
        "edge_path": str(cfg.edge_path),
        "label_path": str(cfg.label_path),
        "reps": cfg.reps,
        "seed": cfg.seed,
        "q": cfg.q,
        "r": cfg.r,
        "p_values": list(cfg.p_values),
        "kmeans_n_init": cfg.n_init,
        **stats,
        "target_rank_used": K,
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Raw CSV      : {raw_csv.resolve()}")
    print(f"Summary CSV  : {sum_csv.resolve()}")
    print(f"Table2a CSV  : {table_csv.resolve()}")
    print(f"Table2a MD   : {table_md.resolve()}")
    print(f"Pairwise CSV : {pair_csv.resolve()}")
    print(f"PairMat CSV  : {pair_mat_csv.resolve()}")
    print(f"Heatmap PNG  : {pair_png.resolve()}")
    print(f"Meta JSON    : {meta_json.resolve()}")


if __name__ == "__main__":
    main()
