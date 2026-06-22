# -*- coding: utf-8 -*-

"""LastFM Asia alpha/tau/r sweep for degree-stratified RP."""

from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

from degree_stratified_rp_powerlaw import (
    align_labels_weighted_hungarian,
    degree_stratified_random_projection,
    gaussian_random_projection,
    kmeans_on_rows,
    normalize_rows_l2,
    plot_results,
    power_law_diagnostics,
)


METHODS = ["Gaussian RP", "Degree-stratified RP"]


@dataclass(frozen=True)
class SweepConfig:
    dataset_path: Path
    dataset_name: str = "lastfm-asia"
    outdir: Path = Path("results/lastfm_alpha_tau_r_sweep")
    alpha_values: tuple[float, ...] = (0.0, 0.25, 0.5)
    tau_values: tuple[str, ...] = ("0", "mean")
    r_values: tuple[int, ...] = (5, 10, 20, 40)
    k: int | None = None
    q: int = 1
    reps: int = 5
    seed: int = 20260519
    ell_min: int = 1
    normalize_embedding_rows: bool = True
    kmeans_n_init: int = 20
    max_n: int = 0
    subgraph_mode: str = "none"
    no_plots: bool = False


def parse_csv_list(text: str, cast):
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(cast(part))
    return tuple(values)


def find_member(zf: zipfile.ZipFile, needle: str):
    needle = needle.lower()
    for name in zf.namelist():
        if needle in Path(name).name.lower() and name.lower().endswith(".csv"):
            return name
    raise FileNotFoundError(f"Could not find CSV member containing {needle!r}")


def read_lastfm_csvs(path: Path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            edge_name = find_member(zf, "edges")
            target_name = find_member(zf, "target")
            with zf.open(edge_name) as f:
                edges = pd.read_csv(f)
            with zf.open(target_name) as f:
                target = pd.read_csv(f)
            return edges, target, {"zip_file": str(path), "edge_member": edge_name, "target_member": target_name}

    if path.is_dir():
        edge_files = sorted(path.glob("*edges*.csv"))
        target_files = sorted(path.glob("*target*.csv"))
        if not edge_files or not target_files:
            raise FileNotFoundError(
                f"Expected *edges*.csv and *target*.csv in {path}"
            )
        return (
            pd.read_csv(edge_files[0]),
            pd.read_csv(target_files[0]),
            {"edge_file": str(edge_files[0]), "target_file": str(target_files[0])},
        )

    raise FileNotFoundError(f"LastFM path must be a zip file or directory: {path}")


def first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...], fallback_index: int):
    lower_to_col = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_col:
            return lower_to_col[c.lower()]
    return df.columns[fallback_index]


def remap_labels(y: np.ndarray):
    uniq = np.unique(y)
    mapping = {int(v): i for i, v in enumerate(uniq)}
    return np.array([mapping[int(v)] for v in y], dtype=int), mapping


def load_lastfm_asia(path: Path):
    edges, target, source_meta = read_lastfm_csvs(path)
    u_col = first_existing_column(edges, ("node_1", "source", "src", "u"), 0)
    v_col = first_existing_column(edges, ("node_2", "target", "dst", "v"), 1)
    id_col = first_existing_column(target, ("id", "node", "user_id"), 0)
    label_col = first_existing_column(target, ("target", "label", "group_id", "country"), 1)

    edges = edges[[u_col, v_col]].dropna().astype(np.int64)
    target = target[[id_col, label_col]].dropna().astype(np.int64)

    target_ids = target[id_col].to_numpy(dtype=np.int64)
    edge_ids = np.unique(edges[[u_col, v_col]].to_numpy(dtype=np.int64).ravel())
    all_ids = np.unique(np.concatenate([target_ids, edge_ids]))
    id_to_idx = {int(node_id): i for i, node_id in enumerate(all_ids)}

    rows = edges[u_col].map(id_to_idx).to_numpy(dtype=np.int32)
    cols = edges[v_col].map(id_to_idx).to_numpy(dtype=np.int32)
    keep = rows != cols
    rows = rows[keep]
    cols = cols[keep]
    data = np.ones(rows.shape[0] * 2, dtype=np.float32)
    A = sp.coo_matrix(
        (data, (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(len(all_ids), len(all_ids)),
        dtype=np.float32,
    ).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()

    y = np.full(len(all_ids), -1, dtype=int)
    for node_id, label in target[[id_col, label_col]].itertuples(index=False):
        y[id_to_idx[int(node_id)]] = int(label)
    if np.any(y < 0):
        labeled = y >= 0
        A = A[labeled][:, labeled].tocsr()
        all_ids = all_ids[labeled]
        y = y[labeled]
    y, label_mapping = remap_labels(y)
    meta = {
        **source_meta,
        "edge_columns": [str(u_col), str(v_col)],
        "target_columns": [str(id_col), str(label_col)],
        "label_mapping_original_to_zero_based": label_mapping,
    }
    return A, y, all_ids, meta


def load_cora(path: Path):
    path = Path(path)

    def read_from_archive(archive_path: Path):
        if archive_path.is_file() and archive_path.suffix.lower() in {".tgz", ".gz"}:
            with tarfile.open(archive_path, "r:*") as tf:
                names = tf.getnames()
                content_name = next(n for n in names if n.endswith("cora.content"))
                cites_name = next(n for n in names if n.endswith("cora.cites"))
                with tf.extractfile(content_name) as f:
                    content = pd.read_csv(f, sep="\t", header=None)
                with tf.extractfile(cites_name) as f:
                    cites = pd.read_csv(f, sep="\t", header=None, names=["source", "target"])
            return content, cites, {
                "archive_file": str(archive_path),
                "content_member": content_name,
                "cites_member": cites_name,
            }
        if archive_path.is_dir():
            content_file = next(archive_path.rglob("cora.content"))
            cites_file = next(archive_path.rglob("cora.cites"))
            content = pd.read_csv(content_file, sep="\t", header=None)
            cites = pd.read_csv(cites_file, sep="\t", header=None, names=["source", "target"])
            return content, cites, {
                "content_file": str(content_file),
                "cites_file": str(cites_file),
            }
        raise FileNotFoundError(f"Expected Cora .tgz/.tar.gz archive or directory, got {archive_path}")

    content, cites, source_meta = read_from_archive(path)
    paper_ids = content.iloc[:, 0].to_numpy(dtype=np.int64)
    labels_raw = content.iloc[:, -1].astype(str)
    label_codes, label_names = pd.factorize(labels_raw, sort=True)

    id_to_idx = {int(node_id): i for i, node_id in enumerate(paper_ids)}
    cites = cites.dropna().astype(np.int64)
    cites = cites[cites["source"].isin(id_to_idx) & cites["target"].isin(id_to_idx)]
    rows = cites["source"].map(id_to_idx).to_numpy(dtype=np.int32)
    cols = cites["target"].map(id_to_idx).to_numpy(dtype=np.int32)

    n = int(len(paper_ids))
    data = np.ones(rows.size, dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()

    meta = {
        **source_meta,
        "dataset_format": "linqs_cora",
        "num_features": int(content.shape[1] - 2),
        "num_citation_rows": int(cites.shape[0]),
        "labels": [str(x) for x in label_names.tolist()],
        "label_counts": {
            str(label): int(count)
            for label, count in labels_raw.value_counts().sort_index().items()
        },
    }
    return A, label_codes.astype(np.int64), paper_ids, meta


def load_polblogs(path: Path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            gml_name = next(n for n in zf.namelist() if n.lower().endswith(".gml"))
            txt_name = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
            text = zf.read(gml_name).decode("latin1")
            readme = zf.read(txt_name).decode("latin1") if txt_name else ""
            source_meta = {"zip_file": str(path), "gml_member": gml_name, "readme_member": txt_name}
    elif path.is_dir():
        gml_file = next(path.rglob("*.gml"))
        text = gml_file.read_text(encoding="latin1")
        readme_file = next(path.rglob("*.txt"), None)
        readme = readme_file.read_text(encoding="latin1") if readme_file else ""
        source_meta = {"gml_file": str(gml_file), "readme_file": str(readme_file) if readme_file else None}
    else:
        raise FileNotFoundError(f"Expected PolBlogs .zip archive or directory, got {path}")

    node_blocks = re.findall(r"node\s*\[(.*?)\]", text, flags=re.S)
    edge_blocks = re.findall(r"edge\s*\[(.*?)\]", text, flags=re.S)

    node_ids = []
    labels = []
    blog_labels = []
    sources = []
    for block in node_blocks:
        node_id = int(re.search(r"\bid\s+(-?\d+)", block).group(1))
        value = int(re.search(r"\bvalue\s+(-?\d+)", block).group(1))
        label_match = re.search(r'\blabel\s+"(.*?)"', block, flags=re.S)
        source_match = re.search(r'\bsource\s+"(.*?)"', block, flags=re.S)
        node_ids.append(node_id)
        labels.append(value)
        blog_labels.append(label_match.group(1) if label_match else "")
        sources.append(source_match.group(1) if source_match else "")

    id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    rows = []
    cols = []
    dropped_edges = 0
    for block in edge_blocks:
        source = int(re.search(r"\bsource\s+(-?\d+)", block).group(1))
        target = int(re.search(r"\btarget\s+(-?\d+)", block).group(1))
        if source in id_to_idx and target in id_to_idx:
            rows.append(id_to_idx[source])
            cols.append(id_to_idx[target])
        else:
            dropped_edges += 1

    n = len(node_ids)
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()

    labels_arr = np.asarray(labels, dtype=np.int64)
    meta = {
        **source_meta,
        "dataset_format": "newman_gml_polblogs",
        "directed_source_graph": True,
        "symmetrized_for_experiment": True,
        "num_gml_nodes": int(n),
        "num_gml_directed_edges": int(len(edge_blocks)),
        "num_dropped_edges": int(dropped_edges),
        "label_meaning": {"0": "left_or_liberal", "1": "right_or_conservative"},
        "label_counts": {
            "left_or_liberal": int(np.sum(labels_arr == 0)),
            "right_or_conservative": int(np.sum(labels_arr == 1)),
        },
        "readme_excerpt": readme[:600],
    }
    return A, labels_arr, np.asarray(node_ids, dtype=np.int64), meta


def largest_connected_component(A: sp.csr_matrix):
    from scipy.sparse.csgraph import connected_components

    _, labels = connected_components(A, directed=False, return_labels=True)
    counts = np.bincount(labels)
    lcc = int(np.argmax(counts))
    idx = np.where(labels == lcc)[0]
    return A[idx][:, idx].tocsr(), idx, counts


def select_subgraph(A: sp.csr_matrix, y: np.ndarray, ids: np.ndarray, max_n: int, mode: str):
    if max_n <= 0 or max_n >= A.shape[0] or mode == "none":
        A_lcc, idx, counts = largest_connected_component(A)
        return A_lcc, y[idx], ids[idx], {
            "subgraph_mode": "largest-connected-component",
            "component_sizes_top10": [int(x) for x in sorted(counts, reverse=True)[:10]],
        }

    degrees = np.asarray(A.sum(axis=1)).ravel()
    rng = np.random.default_rng(12345)
    if mode == "top-degree":
        chosen = np.argsort(degrees)[::-1][:max_n]
    elif mode == "degree-weighted":
        weights = degrees / degrees.sum()
        chosen = rng.choice(A.shape[0], size=max_n, replace=False, p=weights)
    elif mode == "uniform":
        chosen = rng.choice(A.shape[0], size=max_n, replace=False)
    else:
        raise ValueError(f"Unknown subgraph mode: {mode}")
    chosen = np.sort(chosen)
    A_sub = A[chosen][:, chosen].tocsr()
    A_lcc, local_idx, counts = largest_connected_component(A_sub)
    idx = chosen[local_idx]
    return A_lcc, y[idx], ids[idx], {
        "subgraph_mode": mode,
        "requested_max_n": int(max_n),
        "pre_lcc_nodes": int(A_sub.shape[0]),
        "pre_lcc_edges": int(A_sub.nnz // 2),
        "component_sizes_top10": [int(x) for x in sorted(counts, reverse=True)[:10]],
    }


def resolve_tau(value: str, degrees: np.ndarray):
    text = str(value).strip().lower()
    if text in {"mean", "avg"}:
        return float(np.mean(degrees))
    if text in {"median", "med"}:
        return float(np.median(degrees))
    if text in {"zero", "none"}:
        return 0.0
    return float(text)


def degree_tempered_operator(A: sp.csr_matrix, alpha: float, tau: float):
    degrees = np.asarray(A.sum(axis=1)).ravel().astype(float)
    d_tau = degrees + float(tau)
    if np.any(d_tau <= 0.0) and float(alpha) != 0.0:
        inv = np.zeros_like(d_tau, dtype=float)
        mask = d_tau > 0.0
        inv[mask] = np.power(d_tau[mask], -float(alpha))
    else:
        inv = np.power(d_tau, -float(alpha)) if float(alpha) != 0.0 else np.ones_like(d_tau)
    S = A.astype(float).multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
    S.eliminate_zeros()
    return S


def evaluate_labels(U: np.ndarray, y_true: np.ndarray, K: int, rng, normalize: bool, n_init: int):
    X = normalize_rows_l2(U) if normalize else U
    y_pred = kmeans_on_rows(X, K, rng, n_init=n_init)
    aligned = align_labels_weighted_hungarian(y_true, y_pred, K)
    return {
        "ARI_true": float(adjusted_rand_score(y_true, y_pred)),
        "NMI_true": float(normalized_mutual_info_score(y_true, y_pred)),
        "F1_macro_true": float(f1_score(y_true, aligned, average="macro")),
    }, y_pred


def run_method(S, degrees, y_true, K, cfg: SweepConfig, r: int, rep: int, seed: int, method: str):
    import time

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if method == "Gaussian RP":
        vals, U, timings = gaussian_random_projection(S, K, r, cfg.q, rng)
        bucket_rows = []
    elif method == "Degree-stratified RP":
        vals, U, timings, bucket_rows = degree_stratified_random_projection(
            S,
            K,
            r,
            cfg.q,
            cfg.ell_min,
            rng,
            bucket_degrees=degrees,
        )
    else:
        raise ValueError(method)
    embedding_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    label_metrics, y_pred = evaluate_labels(
        U,
        y_true,
        K,
        rng,
        normalize=cfg.normalize_embedding_rows,
        n_init=cfg.kmeans_n_init,
    )
    cluster_sec = time.perf_counter() - t0
    record = {
        "method": method,
        "rep": int(rep),
        "r": int(r),
        "ell": int(K + r),
        "k": int(K),
        "q": int(cfg.q),
        "embedding_wall_sec": float(embedding_sec),
        "clustering_wall_sec": float(cluster_sec),
        "total_method_wall_sec": float(embedding_sec + cluster_sec),
        **label_metrics,
        **timings,
    }
    return record, y_pred, bucket_rows


def summarize(df: pd.DataFrame):
    group_cols = ["alpha", "tau_label", "tau_value", "r", "method"]
    numeric_cols = [
        c
        for c in df.columns
        if c not in set(group_cols + ["dataset", "tau_label", "method", "ds_bucket_summary"])
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    aggregations: dict[str, Any] = {"runs": ("rep", "count")}
    for col in numeric_cols:
        if col == "rep":
            continue
        aggregations[f"{col}_mean"] = (col, "mean")
        aggregations[f"{col}_std"] = (col, "std")
    return df.groupby(group_cols, as_index=False).agg(**aggregations)


def paired_diff_summary(df: pd.DataFrame):
    metrics = ["ARI_true", "NMI_true", "F1_macro_true", "total_method_wall_sec"]
    rows = []
    for key, group in df.groupby(["alpha", "tau_label", "tau_value", "r"], sort=True):
        by_rep = {}
        for row in group.to_dict("records"):
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
            se = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
            rows.append(
                {
                    "alpha": key[0],
                    "tau_label": key[1],
                    "tau_value": key[2],
                    "r": key[3],
                    "metric": metric,
                    "diff_mean_ds_minus_gaussian": float(arr.mean()),
                    "diff_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "diff_se": se,
                    "n_pairs": int(arr.size),
                }
            )
    return pd.DataFrame(rows)


def plot_sweep(summary: pd.DataFrame, outdir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    outdir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("ARI_true_mean", "ARI"),
        ("NMI_true_mean", "NMI"),
        ("F1_macro_true_mean", "Macro F1"),
        ("total_method_wall_sec_mean", "Total time (sec)"),
    ]
    for col, ylabel in metrics:
        if col not in summary.columns:
            continue
        for (alpha, tau_label), block in summary.groupby(["alpha", "tau_label"]):
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            for method in METHODS:
                d = block[block["method"] == method].sort_values("r")
                if d.empty:
                    continue
                ax.plot(d["r"], d[col], marker="o", linewidth=2, label=method)
                std_col = col.replace("_mean", "_std")
                if std_col in d.columns:
                    y = d[col].to_numpy(dtype=float)
                    err = d[std_col].fillna(0.0).to_numpy(dtype=float)
                    x = d["r"].to_numpy(dtype=float)
                    ax.fill_between(x, y - err, y + err, alpha=0.14)
            ax.set_xlabel("r (oversampling)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"LastFM Asia alpha={alpha}, tau={tau_label}")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            safe_tau = str(tau_label).replace(".", "p")
            safe_alpha = str(alpha).replace(".", "p")
            fig.savefig(outdir / f"{col}_alpha{safe_alpha}_tau{safe_tau}.png", dpi=180)
            plt.close(fig)


def run_sweep(cfg: SweepConfig):
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    A_full, y_full, ids_full, source_meta = load_lastfm_asia(cfg.dataset_path)
    A, y_true, ids, subgraph_meta = select_subgraph(
        A_full, y_full, ids_full, cfg.max_n, cfg.subgraph_mode
    )
    K = int(cfg.k) if cfg.k is not None else int(np.unique(y_true).size)
    degrees = np.asarray(A.sum(axis=1)).ravel().astype(float)
    graph_meta = {
        "full_nodes": int(A_full.shape[0]),
        "full_edges": int(A_full.nnz // 2),
        "experiment_nodes": int(A.shape[0]),
        "experiment_edges": int(A.nnz // 2),
        "num_classes": int(np.unique(y_true).size),
        "k_used": int(K),
        **subgraph_meta,
        **power_law_diagnostics(A),
    }

    print(
        f"Loaded {cfg.dataset_name}: n={A.shape[0]}, m={A.nnz // 2}, "
        f"classes={np.unique(y_true).size}, k={K}"
    )

    rows = []
    bucket_rows_all = []
    master_rng = np.random.default_rng(cfg.seed)
    for alpha in cfg.alpha_values:
        for tau_label in cfg.tau_values:
            tau_value = resolve_tau(tau_label, degrees)
            S = degree_tempered_operator(A, alpha=alpha, tau=tau_value)
            for r in cfg.r_values:
                for rep in range(1, cfg.reps + 1):
                    rep_seed = int(master_rng.integers(1, 2**31 - 1))
                    print(
                        f"alpha={alpha:g} tau={tau_label} r={r} rep={rep}/{cfg.reps}",
                        flush=True,
                    )
                    for i, method in enumerate(METHODS):
                        record, _, bucket_rows = run_method(
                            S,
                            degrees,
                            y_true,
                            K,
                            cfg,
                            r,
                            rep,
                            seed=rep_seed + 10_000 * (i + 1),
                            method=method,
                        )
                        record.update(
                            {
                                "dataset": cfg.dataset_name,
                                "alpha": float(alpha),
                                "tau_label": str(tau_label),
                                "tau_value": float(tau_value),
                            }
                        )
                        rows.append(record)
                        if method == "Degree-stratified RP":
                            for b in bucket_rows:
                                bucket_rows_all.append(
                                    {
                                        "alpha": float(alpha),
                                        "tau_label": str(tau_label),
                                        "tau_value": float(tau_value),
                                        "r": int(r),
                                        "rep": int(rep),
                                        **b,
                                    }
                                )

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    paired = paired_diff_summary(raw)
    buckets = pd.DataFrame(bucket_rows_all)

    raw_path = cfg.outdir / "lastfm_alpha_tau_r_sweep_raw.csv"
    summary_path = cfg.outdir / "lastfm_alpha_tau_r_sweep_summary.csv"
    paired_path = cfg.outdir / "lastfm_alpha_tau_r_sweep_paired_ds_minus_gaussian.csv"
    bucket_path = cfg.outdir / "lastfm_alpha_tau_r_sweep_bucket_allocations.csv"
    meta_path = cfg.outdir / "lastfm_alpha_tau_r_sweep_meta.json"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    buckets.to_csv(bucket_path, index=False)
    meta = {
        "config": {
            **asdict(cfg),
            "dataset_path": str(cfg.dataset_path),
            "outdir": str(cfg.outdir),
        },
        "source": source_meta,
        "graph": graph_meta,
        "operator": "S_alpha_tau = D_tau^{-alpha} A D_tau^{-alpha}",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cfg.no_plots:
        plot_sweep(summary, cfg.outdir / "viz")

    print("Done.")
    print(f"Raw CSV     : {raw_path.resolve()}")
    print(f"Summary CSV : {summary_path.resolve()}")
    print(f"Paired CSV  : {paired_path.resolve()}")
    print(f"Buckets CSV : {bucket_path.resolve()}")
    print(f"Meta JSON   : {meta_path.resolve()}")
    return raw, summary, paired, buckets, meta


def parse_args():
    parser = argparse.ArgumentParser(description="LastFM Asia alpha/tau/r sweep")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="lastfm-asia")
    parser.add_argument("--outdir", type=Path, default=Path("results/lastfm_alpha_tau_r_sweep"))
    parser.add_argument("--alpha-values", type=str, default="0,0.25,0.5")
    parser.add_argument("--tau-values", type=str, default="0,mean")
    parser.add_argument("--r-values", type=str, default="5,10,20,40")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--ell-min", type=int, default=1)
    parser.add_argument("--no-normalize-embedding-rows", action="store_true")
    parser.add_argument("--kmeans-n-init", type=int, default=20)
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
    cfg = SweepConfig(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        outdir=args.outdir,
        alpha_values=parse_csv_list(args.alpha_values, float),
        tau_values=parse_csv_list(args.tau_values, str),
        r_values=parse_csv_list(args.r_values, int),
        k=args.k,
        q=args.q,
        reps=args.reps,
        seed=args.seed,
        ell_min=args.ell_min,
        normalize_embedding_rows=not args.no_normalize_embedding_rows,
        kmeans_n_init=args.kmeans_n_init,
        max_n=args.max_n,
        subgraph_mode=args.subgraph_mode,
        no_plots=args.no_plots,
    )
    run_sweep(cfg)


if __name__ == "__main__":
    main()
