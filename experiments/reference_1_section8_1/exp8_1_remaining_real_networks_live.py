# -*- coding: utf-8 -*-

"""Section 8.1 accuracy experiments for the remaining real networks.

Datasets:
- political_blog: evaluated against known political labels.
- statisticians_coauthor: evaluated relative to non-random spectral clustering.
- statisticians_citation: evaluated relative to non-random spectral clustering.

This mirrors Table 2(b-d) of the reference paper.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
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
    pairwise_ari,
    upper_triangle_edges,
)


DATASET_TITLES = {
    "political_blog": "Political blog network",
    "statisticians_coauthor": "Statisticians coauthor network (No true labels)",
    "statisticians_citation": "Statisticians citation network (No true labels)",
}


@dataclass
class DatasetSpec:
    name: str
    A: sp.csr_matrix
    target_rank: int
    y_true: np.ndarray | None
    meta: dict


@dataclass
class Exp81RemainingConfig:
    data_dir: Path = Path("data/reference_1_section8_1/processed")
    datasets: tuple[str, ...] = ("political_blog", "statisticians_coauthor", "statisticians_citation")
    reps: int = 20
    seed: int = 2026
    q: int = 2
    r: int = 10
    p_values: tuple[float, ...] = (0.7, 0.8)
    embedding_rank: int | None = None
    outdir: Path = Path("experiments/reference_1_section8_1/results/exp8_1_remaining_real_networks_table2_like")
    no_progress: bool = False


def remap_to_zero_based(y: np.ndarray) -> np.ndarray:
    uniq = np.unique(y)
    mapping = {int(v): i for i, v in enumerate(uniq)}
    return np.array([mapping[int(v)] for v in y], dtype=int)


def load_dataset(data_dir: Path, name: str) -> DatasetSpec:
    A_path = data_dir / f"{name}_adjacency.npz"
    meta_path = data_dir / f"{name}_meta.json"
    label_path = data_dir / f"{name}_labels.npy"
    if not A_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset {name}. Run prepare_exp8_1_real_data.py first."
        )

    A = sp.load_npz(A_path).astype(np.float32).tocsr()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    y_true = None
    if label_path.exists():
        y_true = remap_to_zero_based(np.load(label_path))
    return DatasetSpec(
        name=name,
        A=A,
        target_rank=int(meta["target_rank"]),
        y_true=y_true,
        meta=meta,
    )


def evaluate_against_reference(y_ref: np.ndarray, y_pred: np.ndarray, K: int):
    y_aligned = align_labels_weighted_hungarian(y_ref, y_pred, K)
    f1 = float(f1_score(y_ref, y_aligned, average="macro"))
    nmi = float(normalized_mutual_info_score(y_ref, y_pred))
    ari = float(adjusted_rand_score(y_ref, y_pred))
    return f1, nmi, ari


def run_one_dataset(cfg: Exp81RemainingConfig, ds: DatasetSpec):
    A = ds.A
    K = ds.target_rank
    K_embed = int(cfg.embedding_rank) if cfg.embedding_rank is not None else K
    if K_embed < K:
        raise ValueError(f"embedding_rank must be >= cluster count {K}; got {K_embed}")
    if K_embed >= A.shape[0]:
        raise ValueError(f"embedding_rank must be < number of nodes {A.shape[0]}; got {K_embed}")
    upper_rows, upper_cols = upper_triangle_edges(A)
    methods = (
        ["Random Projection"]
        + [f"Random Sampling (p={p:g})" for p in cfg.p_values]
        + ["Non-random"]
    )

    master_rng = np.random.default_rng(cfg.seed)
    rows = []
    pair_rows = []
    progress = None if cfg.no_progress else LiveProgress(cfg.reps * len(methods))

    for rep in range(1, cfg.reps + 1):
        rep_seed = int(master_rng.integers(1, 2**31 - 1))
        labels = {}
        timings = {}

        rng_nr = np.random.default_rng(rep_seed + 97)
        t0 = perf_counter()
        _, U_nr = eigvecs_eigsh_sparse(A, K_embed)
        t1 = perf_counter()
        y_nr = kmeans_on_rows(U_nr, K, rng_nr)
        timings["Non-random"] = {
            "time_rand_sec": 0.0,
            "time_post_sec": float((t1 - t0) + (perf_counter() - t1)),
            "time_total_sec": float(perf_counter() - t0),
        }
        labels["Non-random"] = y_nr.copy()
        if progress is not None:
            progress.update(ds.name, rep, rep, cfg.reps, "Non-random")

        rng_rp = np.random.default_rng(rep_seed + 11)
        t0 = perf_counter()
        _, U_rp = eigvecs_random_projection_sparse(A, k=K_embed, r=cfg.r, q=cfg.q, rng=rng_rp)
        y_rp = kmeans_on_rows(U_rp, K, rng_rp)
        timings["Random Projection"] = {
            "time_rand_sec": np.nan,
            "time_post_sec": np.nan,
            "time_total_sec": float(perf_counter() - t0),
        }
        labels["Random Projection"] = y_rp.copy()
        if progress is not None:
            progress.update(ds.name, rep, rep, cfg.reps, "Random Projection")

        for p in cfg.p_values:
            mname = f"Random Sampling (p={p:g})"
            rng_rs = np.random.default_rng(rep_seed + int(round(p * 1000)) + 31)
            t0 = perf_counter()
            U_rs, t_rs_with, t_rs_without = eigvecs_random_sampling_sparse(
                n=A.shape[0],
                upper_rows=upper_rows,
                upper_cols=upper_cols,
                p=p,
                k=K_embed,
                rng=rng_rs,
            )
            t1 = perf_counter()
            y_rs = kmeans_on_rows(U_rs, K, rng_rs)
            t_km = perf_counter() - t1
            timings[mname] = {
                "time_rand_sec": max(0.0, float(t_rs_with - t_rs_without)),
                "time_post_sec": float(t_rs_without + t_km),
                "time_total_sec": float(perf_counter() - t0),
            }
            labels[mname] = y_rs.copy()
            if progress is not None:
                progress.update(ds.name, rep, rep, cfg.reps, mname)

        y_ref = ds.y_true if ds.y_true is not None else labels["Non-random"]
        eval_mode = "truth" if ds.y_true is not None else "relative_to_non_random"
        for method in methods:
            if ds.y_true is None and method == "Non-random":
                continue
            f1, nmi, ari = evaluate_against_reference(y_ref, labels[method], K)
            rows.append(
                {
                    "dataset": ds.name,
                    "rep": rep,
                    "method": method,
                    "eval_mode": eval_mode,
                    "F1": f1,
                    "NMI": nmi,
                    "ARI": ari,
                    **timings[method],
                }
            )

        mlist, mat = pairwise_ari(labels)
        for i in range(len(mlist)):
            for j in range(i + 1, len(mlist)):
                pair_rows.append(
                    {
                        "dataset": ds.name,
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
    return df_raw.groupby(["dataset", "method"], as_index=False).agg(
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


def method_order(p_values: tuple[float, ...], include_non_random: bool):
    ordered = ["Random Projection"] + [f"Random Sampling (p={p:g})" for p in p_values]
    if include_non_random:
        ordered.append("Non-random")
    return ordered


def display_method(method: str, p_values: tuple[float, ...]) -> str:
    if method == "Non-random":
        return "Non-Random"
    if method == "Random Projection":
        return "Random Projection"
    for p in p_values:
        if method == f"Random Sampling (p={p:g})":
            return f"Random Sampling (p= {p:.1f})"
    return method


def build_table2_like(summary: pd.DataFrame, p_values: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for dataset in summary["dataset"].drop_duplicates():
        include_non_random = dataset == "political_blog"
        dset = summary[summary["dataset"] == dataset].set_index("method")
        for method in method_order(p_values, include_non_random):
            if method not in dset.index:
                continue
            row = dset.loc[method]
            rows.append(
                {
                    "Dataset": dataset,
                    "Methods": display_method(method, p_values),
                    "F 1": f"{row['F1_mean']:.3f}({row['F1_std'] if pd.notna(row['F1_std']) else 0.0:.3f})",
                    "NMI": f"{row['NMI_mean']:.3f}({row['NMI_std'] if pd.notna(row['NMI_std']) else 0.0:.3f})",
                    "ARI": f"{row['ARI_mean']:.3f}({row['ARI_std'] if pd.notna(row['ARI_std']) else 0.0:.3f})",
                }
            )
    return pd.DataFrame(rows)


def write_table2_markdown(tbl: pd.DataFrame, out_md: Path, reps: int):
    lines = []
    lines.append("Table 2 (b-d): Section 8.1 remaining real network accuracy experiments.")
    lines.append("")
    for dataset in tbl["Dataset"].drop_duplicates():
        title = DATASET_TITLES.get(dataset, dataset)
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Methods | F 1 | NMI | ARI |")
        lines.append("|---|---:|---:|---:|")
        for _, row in tbl[tbl["Dataset"] == dataset].iterrows():
            lines.append(f"| {row['Methods']} | {row['F 1']} | {row['NMI']} | {row['ARI']} |")
        lines.append("")
    lines.append(f"Note: Values are mean(std) over {reps} replications.")
    lines.append("For the two statisticians networks, scores are relative to non-random spectral clustering.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_p_values(p_str: str):
    vals = [float(x.strip()) for x in p_str.split(",") if x.strip()]
    vals = [v for v in vals if 0.0 < v <= 1.0]
    if not vals:
        raise ValueError("p-values must be in (0,1].")
    return tuple(vals)


def parse_datasets(s: str):
    vals = tuple(x.strip() for x in s.split(",") if x.strip())
    valid = set(DATASET_TITLES)
    bad = [x for x in vals if x not in valid]
    if bad:
        raise ValueError(f"Unknown datasets {bad}; valid values are {sorted(valid)}")
    return vals


def main():
    parser = argparse.ArgumentParser(description="Section 8.1 remaining real-network accuracy experiments")
    parser.add_argument("--data-dir", type=str, default="data/reference_1_section8_1/processed")
    parser.add_argument(
        "--datasets",
        type=str,
        default="political_blog,statisticians_coauthor,statisticians_citation",
        help="comma-separated dataset names",
    )
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--p-values", type=str, default="0.7,0.8")
    parser.add_argument(
        "--embedding-rank",
        type=int,
        default=None,
        help="number of eigenvectors/spectral features; defaults to each dataset target rank",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/reference_1_section8_1/results/exp8_1_remaining_real_networks_table2_like",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cfg = Exp81RemainingConfig(
        data_dir=Path(args.data_dir),
        datasets=parse_datasets(args.datasets),
        reps=args.reps,
        seed=args.seed,
        q=args.q,
        r=args.r,
        p_values=parse_p_values(args.p_values),
        embedding_rank=args.embedding_rank,
        outdir=Path(args.outdir),
        no_progress=args.no_progress,
    )
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    raw_frames = []
    pair_frames = []
    metas = {}
    for name in cfg.datasets:
        ds = load_dataset(cfg.data_dir, name)
        metas[name] = ds.meta
        print(
            f"Running {name}: nodes={ds.meta['num_nodes']}, "
            f"edges={ds.meta['num_edges']}, target_rank={ds.target_rank}, "
            f"embedding_rank={cfg.embedding_rank or ds.target_rank}"
        )
        df_raw, df_pair = run_one_dataset(cfg, ds)
        raw_frames.append(df_raw)
        pair_frames.append(df_pair)

    df_raw_all = pd.concat(raw_frames, ignore_index=True)
    df_pair_all = pd.concat(pair_frames, ignore_index=True)
    df_summary = summarize(df_raw_all)
    table2_like = build_table2_like(df_summary, cfg.p_values)

    raw_csv = cfg.outdir / "remaining_real_networks_raw_per_rep.csv"
    summary_csv = cfg.outdir / "remaining_real_networks_summary_mean_std.csv"
    table_csv = cfg.outdir / "remaining_real_networks_table2_like.csv"
    table_md = cfg.outdir / "remaining_real_networks_table2_like.md"
    pair_csv = cfg.outdir / "remaining_real_networks_pairwise_ari_raw.csv"
    meta_json = cfg.outdir / "remaining_real_networks_meta.json"

    df_raw_all.to_csv(raw_csv, index=False)
    df_pair_all.to_csv(pair_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    table2_like.to_csv(table_csv, index=False)
    write_table2_markdown(table2_like, table_md, cfg.reps)

    meta = {
        "datasets": list(cfg.datasets),
        "reps": cfg.reps,
        "seed": cfg.seed,
        "q": cfg.q,
        "r": cfg.r,
        "p_values": list(cfg.p_values),
        "embedding_rank": cfg.embedding_rank,
        "dataset_meta": metas,
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Raw CSV     : {raw_csv.resolve()}")
    print(f"Summary CSV : {summary_csv.resolve()}")
    print(f"Table CSV   : {table_csv.resolve()}")
    print(f"Table MD    : {table_md.resolve()}")
    print(f"Pairwise CSV: {pair_csv.resolve()}")
    print(f"Meta JSON   : {meta_json.resolve()}")


if __name__ == "__main__":
    main()
