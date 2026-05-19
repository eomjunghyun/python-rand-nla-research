# -*- coding: utf-8 -*-

"""Independent experiment for degree-stratified RP on real power-law graphs."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


METHOD_ORDER = [
    "General eigensolver",
    "Gaussian RP",
    "Random sampling",
    "Degree-stratified RP",
]

METHOD_COLORS = {
    "General eigensolver": "#4C78A8",
    "Gaussian RP": "#F58518",
    "Random sampling": "#54A24B",
    "Degree-stratified RP": "#E45756",
}


@dataclass(frozen=True)
class ExperimentConfig:
    edgelist: Path
    dataset_name: str
    max_n: int = 1500
    subgraph_mode: str = "top-degree"
    k: int = 8
    r: int = 20
    q: int = 2
    reps: int = 5
    seed: int = 20260518
    sampling_p: float = 0.7
    ell_min: int = 1
    rs_eigensolver: str = "general"
    normalize_embedding_rows: bool = True
    kmeans_n_init: int = 20
    outdir: Path = Path("results/com_dblp_demo")
    delimiter: str | None = None
    comment_prefix: str = "#"
    no_plots: bool = False

    @property
    def ell(self) -> int:
        return int(self.k + self.r)


def load_large_integer_edgelist_csr(
    path: Path,
    delimiter: str | None = None,
    comment_prefix: str = "#",
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Edge list not found: {path}")

    sep = delimiter if delimiter is not None else r"\s+"
    df = pd.read_csv(
        path,
        sep=sep,
        comment=comment_prefix,
        header=None,
        usecols=[0, 1],
        names=["src", "dst"],
        dtype=np.int64,
        engine="c",
    )
    df = df[df["src"] != df["dst"]].reset_index(drop=True)

    endpoints = pd.concat([df["src"], df["dst"]], ignore_index=True)
    codes, uniques = pd.factorize(endpoints, sort=False)
    m = len(df)
    rows = codes[:m].astype(np.int32, copy=False)
    cols = codes[m:].astype(np.int32, copy=False)
    rows_sym = np.concatenate([rows, cols])
    cols_sym = np.concatenate([cols, rows])
    data = np.ones(rows_sym.shape[0], dtype=np.float32)

    A = sp.coo_matrix(
        (data, (rows_sym, cols_sym)),
        shape=(len(uniques), len(uniques)),
        dtype=np.float32,
    ).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, uniques.to_numpy()


def upper_triangle_edges(A: sp.csr_matrix):
    A_upper = sp.triu(A, k=1, format="coo")
    return A_upper.row.astype(np.int64), A_upper.col.astype(np.int64)


def sample_rescaled_adjacency_from_edges(
    n: int,
    upper_rows: np.ndarray,
    upper_cols: np.ndarray,
    p: float,
    rng: np.random.Generator,
):
    if not (0.0 < p <= 1.0):
        raise ValueError(f"Sampling probability must be in (0,1], got {p}")
    keep = rng.random(upper_rows.shape[0]) < p
    r = upper_rows[keep]
    c = upper_cols[keep]
    if r.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)

    w = np.full(r.shape[0], 1.0 / p, dtype=np.float32)
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    data = np.concatenate([w, w])
    A_s = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    A_s.sum_duplicates()
    A_s.setdiag(0.0)
    A_s.eliminate_zeros()
    return A_s


def normalize_rows_l2(U: np.ndarray, eps: float = 1e-12):
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    return np.divide(U, norms, out=np.zeros_like(U), where=norms > eps)


def kmeans_on_rows(U: np.ndarray, K: int, rng: np.random.Generator, n_init: int):
    return KMeans(
        n_clusters=K,
        n_init=n_init,
        random_state=int(rng.integers(1, 2**31 - 1)),
    ).fit_predict(U)


def align_labels_weighted_hungarian(y_true: np.ndarray, y_pred: np.ndarray, K: int):
    weights = np.zeros(K, dtype=float)
    for c in range(K):
        cnt = np.sum(y_true == c)
        weights[c] = 1.0 / max(1, cnt)
    score = np.zeros((K, K), dtype=float)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < K and 0 <= int(p) < K:
            score[int(t), int(p)] += weights[int(t)]
    row_ind, col_ind = linear_sum_assignment(-score)
    mapping = {int(c): int(r) for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(int(p), int(p)) for p in y_pred], dtype=int)


def dense_general_eigh(A: sp.csr_matrix, k: int):
    t0 = time.perf_counter()
    dense = A.toarray().astype(np.float64, copy=False)
    convert_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    vals, vecs = np.linalg.eigh(dense)
    eig_sec = time.perf_counter() - t0

    order = np.argsort(np.abs(vals))[::-1]
    top = order[:k]
    return vals[top], vecs[:, top], {
        "dense_convert_sec": float(convert_sec),
        "dense_general_eigh_sec": float(eig_sec),
        "eigen_decomposition_wall_sec": float(convert_sec + eig_sec),
    }


def sparse_eigsh_by_magnitude(A: sp.csr_matrix, k: int, rng: np.random.Generator):
    t0 = time.perf_counter()
    vals, vecs = eigsh(A, k=k, which="LM", v0=rng.normal(size=A.shape[0]))
    elapsed = time.perf_counter() - t0
    order = np.argsort(np.abs(vals))[::-1]
    return vals[order], vecs[:, order], {
        "eigsh_sec": float(elapsed),
        "eigen_decomposition_wall_sec": float(elapsed),
    }


def rayleigh_ritz_from_sketch(A: sp.csr_matrix, Y: np.ndarray, k: int):
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    Q, _ = np.linalg.qr(Y, mode="reduced")
    timings["qr_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    AQ = A @ Q
    B = Q.T @ AQ
    B = 0.5 * (B + B.T)
    timings["build_core_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(np.abs(vals))[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    timings["small_eigh_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    U = Q @ vecs[:, :k]
    timings["lift_sec"] = time.perf_counter() - t0
    return vals[:k], U, timings


def gaussian_random_projection(
    A: sp.csr_matrix,
    k: int,
    r: int,
    q: int,
    rng: np.random.Generator,
):
    n = int(A.shape[0])
    ell = int(k + r)
    timings: dict[str, Any] = {}

    t0 = time.perf_counter()
    omega = rng.standard_normal((n, ell))
    timings["rp_draw_omega_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    Y = omega
    for _ in range(2 * int(q) + 1):
        Y = A @ Y
    timings["rp_power_iter_sec"] = time.perf_counter() - t0

    vals, U, more = rayleigh_ritz_from_sketch(A, Y, k)
    timings.update({f"rp_{key}": value for key, value in more.items()})
    timings["eigen_decomposition_wall_sec"] = float(
        timings["rp_draw_omega_sec"]
        + timings["rp_power_iter_sec"]
        + timings["rp_qr_sec"]
        + timings["rp_build_core_sec"]
        + timings["rp_small_eigh_sec"]
        + timings["rp_lift_sec"]
    )
    return vals, U, timings


def random_sampling_embedding(
    A: sp.csr_matrix,
    k: int,
    p: float,
    rng: np.random.Generator,
    eigensolver: str,
):
    timings: dict[str, Any] = {"rs_sampling_probability": float(p)}
    n = int(A.shape[0])
    upper_rows, upper_cols = upper_triangle_edges(A)

    t0 = time.perf_counter()
    A_s = sample_rescaled_adjacency_from_edges(n, upper_rows, upper_cols, p, rng)
    timings["rs_sample_matrix_sec"] = time.perf_counter() - t0
    timings["rs_sampled_edges"] = int(A_s.nnz // 2)
    timings["rs_sampled_density"] = float(A_s.nnz / max(1, n * n))

    if eigensolver == "general":
        vals, U, eig_timing = dense_general_eigh(A_s, k)
        timings.update({f"rs_{key}": value for key, value in eig_timing.items()})
    elif eigensolver == "eigsh":
        vals, U, eig_timing = sparse_eigsh_by_magnitude(A_s, k, rng)
        timings.update({f"rs_{key}": value for key, value in eig_timing.items()})
    else:
        raise ValueError(f"Unknown random-sampling eigensolver: {eigensolver}")

    timings["eigen_decomposition_wall_sec"] = float(
        timings["rs_sample_matrix_sec"]
        + timings.get("rs_eigen_decomposition_wall_sec", 0.0)
    )
    return vals, U, timings


def initial_log_degree_buckets(degrees: np.ndarray):
    positive = degrees > 0
    if not np.any(positive):
        return []

    bin_ids = np.floor(np.log2(np.maximum(degrees[positive], 1))).astype(int)
    node_ids = np.where(positive)[0]
    buckets = []
    for bin_id in np.unique(bin_ids):
        idx = node_ids[bin_ids == bin_id]
        low = int(2 ** bin_id)
        high = int(2 ** (bin_id + 1))
        mass = float(np.sum(degrees[idx]))
        buckets.append({"low": low, "high": high, "idx": idx, "mass": mass})
    return buckets


def merge_buckets_to_budget(buckets: list[dict[str, Any]], max_buckets: int):
    buckets = [dict(b) for b in buckets]
    max_buckets = max(1, int(max_buckets))
    while len(buckets) > max_buckets:
        scores = [
            buckets[i]["mass"] + buckets[i + 1]["mass"]
            for i in range(len(buckets) - 1)
        ]
        merge_at = int(np.argmin(scores))
        left = buckets[merge_at]
        right = buckets[merge_at + 1]
        merged = {
            "low": int(left["low"]),
            "high": int(right["high"]),
            "idx": np.concatenate([left["idx"], right["idx"]]),
            "mass": float(left["mass"] + right["mass"]),
        }
        buckets[merge_at : merge_at + 2] = [merged]
    return buckets


def allocate_bucket_dimensions(masses: np.ndarray, ell: int, ell_min: int):
    s = int(len(masses))
    if s == 0:
        raise ValueError("No nonempty degree buckets were found.")
    if ell < s * ell_min:
        raise ValueError(
            f"Budget ell={ell} cannot give ell_min={ell_min} to {s} buckets."
        )

    dims = np.full(s, int(ell_min), dtype=int)
    remaining = int(ell - dims.sum())
    if remaining <= 0:
        return dims

    weights = np.sqrt(np.maximum(masses, 0.0))
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights)
    raw = remaining * weights / float(weights.sum())
    extra = np.floor(raw).astype(int)
    dims += extra
    leftover = int(ell - dims.sum())
    if leftover > 0:
        frac_order = np.argsort(raw - extra)[::-1]
        for i in frac_order[:leftover]:
            dims[int(i)] += 1
    return dims


def degree_stratified_random_projection(
    A: sp.csr_matrix,
    k: int,
    r: int,
    q: int,
    ell_min: int,
    rng: np.random.Generator,
    bucket_degrees: np.ndarray | None = None,
):
    ell = int(k + r)
    timings: dict[str, Any] = {}

    t0 = time.perf_counter()
    degrees = (
        np.asarray(bucket_degrees, dtype=float).ravel()
        if bucket_degrees is not None
        else np.asarray(A.sum(axis=1)).ravel()
    )
    buckets = initial_log_degree_buckets(degrees)
    max_buckets = ell // max(1, int(ell_min)) if ell_min > 0 else ell
    buckets = merge_buckets_to_budget(buckets, max_buckets=max_buckets)
    masses = np.array([b["mass"] for b in buckets], dtype=float)
    dims = allocate_bucket_dimensions(masses, ell=ell, ell_min=int(ell_min))
    timings["ds_bucket_build_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    parts = []
    for bucket, dim in zip(buckets, dims):
        idx = bucket["idx"]
        G = rng.standard_normal((idx.size, int(dim)))
        parts.append(A[:, idx] @ G)
    Y = np.hstack(parts)
    timings["ds_draw_and_initial_multiply_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(int(q)):
        Y = A @ (A @ Y)
        Y, _ = np.linalg.qr(Y, mode="reduced")
    timings["ds_power_iter_sec"] = time.perf_counter() - t0

    vals, U, more = rayleigh_ritz_from_sketch(A, Y, k)
    timings.update({f"ds_{key}": value for key, value in more.items()})
    timings["eigen_decomposition_wall_sec"] = float(
        timings["ds_bucket_build_sec"]
        + timings["ds_draw_and_initial_multiply_sec"]
        + timings["ds_power_iter_sec"]
        + timings["ds_qr_sec"]
        + timings["ds_build_core_sec"]
        + timings["ds_small_eigh_sec"]
        + timings["ds_lift_sec"]
    )

    bucket_rows = []
    for j, (bucket, dim) in enumerate(zip(buckets, dims), start=1):
        bucket_rows.append(
            {
                "bucket_id": int(j),
                "degree_low_inclusive": int(bucket["low"]),
                "degree_high_exclusive": int(bucket["high"]),
                "num_nodes": int(bucket["idx"].size),
                "mass": float(bucket["mass"]),
                "sqrt_mass": float(math.sqrt(max(0.0, bucket["mass"]))),
                "sketch_dim": int(dim),
            }
        )
    timings["ds_num_buckets"] = int(len(buckets))
    timings["ds_min_bucket_dim"] = int(np.min(dims))
    timings["ds_max_bucket_dim"] = int(np.max(dims))
    timings["ds_bucket_summary"] = ";".join(
        f"[{r['degree_low_inclusive']},{r['degree_high_exclusive']}):"
        f"n={r['num_nodes']},ell={r['sketch_dim']}"
        for r in bucket_rows
    )
    return vals, U, timings, bucket_rows


def largest_connected_component(A: sp.csr_matrix):
    _, labels = connected_components(A, directed=False, return_labels=True)
    counts = np.bincount(labels)
    lcc = int(np.argmax(counts))
    idx = np.where(labels == lcc)[0]
    return A[idx][:, idx].tocsr(), idx, counts


def select_subgraph(A: sp.csr_matrix, max_n: int, mode: str):
    n = int(A.shape[0])
    if max_n <= 0 or max_n >= n or mode == "none":
        A_lcc, idx, counts = largest_connected_component(A)
        return A_lcc, idx, {
            "subgraph_mode": "largest-connected-component",
            "component_sizes_top10": [int(x) for x in sorted(counts, reverse=True)[:10]],
        }

    degrees = np.asarray(A.sum(axis=1)).ravel()
    rng = np.random.default_rng(12345)
    if mode == "top-degree":
        chosen = np.argsort(degrees)[::-1][:max_n]
    elif mode == "degree-weighted":
        weights = degrees / degrees.sum()
        chosen = rng.choice(n, size=max_n, replace=False, p=weights)
    elif mode == "uniform":
        chosen = rng.choice(n, size=max_n, replace=False)
    else:
        raise ValueError(f"Unknown subgraph mode: {mode}")

    chosen = np.sort(chosen)
    A_sub = A[chosen][:, chosen].tocsr()
    A_lcc, local_idx, counts = largest_connected_component(A_sub)
    original_idx = chosen[local_idx]
    return A_lcc, original_idx, {
        "subgraph_mode": mode,
        "requested_max_n": int(max_n),
        "pre_lcc_nodes": int(A_sub.shape[0]),
        "pre_lcc_edges": int(A_sub.nnz // 2),
        "component_sizes_top10": [int(x) for x in sorted(counts, reverse=True)[:10]],
    }


def degree_gini(degrees: np.ndarray):
    x = np.sort(np.asarray(degrees, dtype=float))
    if x.size == 0 or float(np.sum(x)) == 0.0:
        return 0.0
    n = x.size
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x))) - ((n + 1) / n))


def power_law_diagnostics(A: sp.csr_matrix):
    degrees = np.asarray(A.sum(axis=1)).ravel()
    positive = degrees[degrees > 0]
    if positive.size == 0:
        return {}

    xmin = max(2.0, float(np.quantile(positive, 0.75)))
    tail = positive[positive >= xmin]
    if tail.size >= 2:
        alpha_hat = 1.0 + tail.size / float(np.sum(np.log(tail / xmin)))
    else:
        alpha_hat = float("nan")

    values, counts = np.unique(positive.astype(int), return_counts=True)
    ccdf = np.cumsum(counts[::-1])[::-1] / positive.size
    mask = values >= xmin
    if int(np.sum(mask)) >= 2:
        x = np.log(values[mask].astype(float))
        y = np.log(ccdf[mask].astype(float))
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        slope = float("nan")
        r2 = float("nan")

    return {
        "degree_min": float(np.min(positive)),
        "degree_q25": float(np.quantile(positive, 0.25)),
        "degree_median": float(np.median(positive)),
        "degree_mean": float(np.mean(positive)),
        "degree_q75": float(np.quantile(positive, 0.75)),
        "degree_q90": float(np.quantile(positive, 0.90)),
        "degree_q99": float(np.quantile(positive, 0.99)),
        "degree_max": float(np.max(positive)),
        "degree_gini": degree_gini(positive),
        "tail_xmin": float(xmin),
        "tail_count": int(tail.size),
        "tail_alpha_mle_rough": float(alpha_hat),
        "tail_loglog_ccdf_slope": float(slope),
        "tail_loglog_ccdf_r2": float(r2),
    }


def load_optional_labels(path: Path | None, kept_original_nodes: np.ndarray | None):
    if path is None:
        return None
    labels_map = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels_map[int(parts[0])] = int(parts[1])
    if kept_original_nodes is None:
        return None
    labels = np.array([labels_map.get(int(i), -1) for i in kept_original_nodes], dtype=int)
    if np.any(labels < 0):
        return None
    uniq = np.unique(labels)
    remap = {int(v): i for i, v in enumerate(uniq)}
    return np.array([remap[int(v)] for v in labels], dtype=int)


def spectral_quality_metrics(
    A: sp.csr_matrix,
    vals: np.ndarray,
    U: np.ndarray,
    ref_vals: np.ndarray,
    ref_U: np.ndarray,
    A_fro: float,
):
    U, _ = np.linalg.qr(np.asarray(U, dtype=float), mode="reduced")
    ref_U, _ = np.linalg.qr(np.asarray(ref_U, dtype=float), mode="reduced")
    vals = np.asarray(vals[: U.shape[1]], dtype=float)
    ref_vals = np.asarray(ref_vals[: U.shape[1]], dtype=float)

    residual = A @ U - U * vals.reshape(1, -1)
    residual_rel = float(np.linalg.norm(residual, ord="fro") / max(A_fro, 1e-12))

    denom = float(np.linalg.norm(ref_vals) + 1e-12)
    eig_rel = float(np.linalg.norm(vals - ref_vals) / denom)

    cross = U.T @ ref_U
    k = U.shape[1]
    subspace_sq = max(0.0, 2.0 * k - 2.0 * float(np.linalg.norm(cross, ord="fro") ** 2))
    subspace_distance = float(math.sqrt(subspace_sq) / math.sqrt(2.0 * k))

    rayleigh_diag = np.diag(U.T @ (A @ U))
    rayleigh_abs_sum_ratio = float(
        np.sum(np.abs(rayleigh_diag)) / max(np.sum(np.abs(ref_vals)), 1e-12)
    )
    return {
        "residual_rel_fro": residual_rel,
        "eigenvalue_rel_l2_vs_general": eig_rel,
        "subspace_distance_vs_general": subspace_distance,
        "rayleigh_abs_sum_ratio": rayleigh_abs_sum_ratio,
    }


def clustering_metrics(
    U: np.ndarray,
    k: int,
    rng: np.random.Generator,
    normalize_rows: bool,
    n_init: int,
    labels_reference: np.ndarray | None,
    labels_true: np.ndarray | None,
):
    X = normalize_rows_l2(U) if normalize_rows else U
    labels = kmeans_on_rows(X, k, rng, n_init=n_init)
    out: dict[str, Any] = {"labels": labels}
    if labels_reference is not None:
        out["ARI_vs_general_clustering"] = float(adjusted_rand_score(labels_reference, labels))
    if labels_true is not None:
        true_k = int(np.unique(labels_true).size)
        out["ARI_true"] = float(adjusted_rand_score(labels_true, labels))
        out["NMI_true"] = float(normalized_mutual_info_score(labels_true, labels))
        if true_k == k:
            aligned = align_labels_weighted_hungarian(labels_true, labels, true_k)
            out["F1_macro_true"] = float(f1_score(labels_true, aligned, average="macro"))
    return out


def make_record(
    method: str,
    rep: int,
    vals: np.ndarray,
    U: np.ndarray,
    timings: dict[str, Any],
    A: sp.csr_matrix,
    ref_vals: np.ndarray,
    ref_U: np.ndarray,
    A_fro: float,
    labels_reference: np.ndarray | None,
    labels_true: np.ndarray | None,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
):
    t0 = time.perf_counter()
    cluster = clustering_metrics(
        U=U,
        k=cfg.k,
        rng=rng,
        normalize_rows=cfg.normalize_embedding_rows,
        n_init=cfg.kmeans_n_init,
        labels_reference=labels_reference,
        labels_true=labels_true,
    )
    clustering_sec = time.perf_counter() - t0
    quality = spectral_quality_metrics(A, vals, U, ref_vals, ref_U, A_fro)
    record = {
        "dataset": cfg.dataset_name,
        "rep": int(rep),
        "method": method,
        "n": int(A.shape[0]),
        "m_edges": int(A.nnz // 2),
        "k": int(cfg.k),
        "r": int(cfg.r),
        "ell": int(cfg.ell),
        "q": int(cfg.q),
        "sampling_p": float(cfg.sampling_p),
        "rs_eigensolver": cfg.rs_eigensolver,
        "clustering_wall_sec": float(clustering_sec),
        **quality,
        **{k: v for k, v in cluster.items() if k != "labels"},
        **timings,
    }
    record["total_method_wall_sec"] = float(
        record.get("eigen_decomposition_wall_sec", 0.0) + clustering_sec
    )
    return record, cluster["labels"]


def summarize_raw(df: pd.DataFrame):
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"dataset", "method", "rs_eigensolver", "ds_bucket_summary"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    aggregations: dict[str, Any] = {"runs": ("rep", "count")}
    for col in numeric_cols:
        if col == "rep":
            continue
        aggregations[f"{col}_mean"] = (col, "mean")
        aggregations[f"{col}_std"] = (col, "std")
    summary = df.groupby("method", as_index=False).agg(**aggregations)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    return summary.sort_values("method").reset_index(drop=True)


def plot_results(df_raw: pd.DataFrame, outdir: Path):
    if plt is None:
        print("matplotlib is not installed; skipping plots.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    plot_cols = [
        ("total_method_wall_sec", "Total time (sec)", "runtime_total.png"),
        ("residual_rel_fro", "Relative residual", "quality_residual.png"),
        ("subspace_distance_vs_general", "Subspace distance vs general", "quality_subspace.png"),
        ("ARI_vs_general_clustering", "ARI vs general clustering", "clustering_ari_vs_general.png"),
    ]
    for col, ylabel, filename in plot_cols:
        if col not in df_raw.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.8))
        data = [
            df_raw.loc[df_raw["method"] == method, col].dropna().to_numpy(dtype=float)
            for method in METHOD_ORDER
        ]
        positions = np.arange(1, len(METHOD_ORDER) + 1)
        ax.boxplot(data, positions=positions, widths=0.55, showmeans=True)
        for pos, method, values in zip(positions, METHOD_ORDER, data):
            if values.size:
                x = np.full(values.size, pos, dtype=float)
                ax.scatter(x, values, s=24, alpha=0.7, color=METHOD_COLORS[method])
        ax.set_xticks(positions)
        ax.set_xticklabels(METHOD_ORDER, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def run_experiment(cfg: ExperimentConfig, label_path: Path | None = None):
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    A_full, original_nodes = load_large_integer_edgelist_csr(
        cfg.edgelist,
        delimiter=cfg.delimiter,
        comment_prefix=cfg.comment_prefix,
    )
    A, kept_local_idx, subgraph_meta = select_subgraph(A_full, cfg.max_n, cfg.subgraph_mode)
    kept_original_nodes = (
        np.asarray(original_nodes)[kept_local_idx]
        if len(original_nodes) == A_full.shape[0]
        else kept_local_idx
    )
    labels_true = load_optional_labels(label_path, kept_original_nodes)

    graph_meta = {
        "full_nodes": int(A_full.shape[0]),
        "full_edges": int(A_full.nnz // 2),
        "experiment_nodes": int(A.shape[0]),
        "experiment_edges": int(A.nnz // 2),
        **subgraph_meta,
        **power_law_diagnostics(A),
    }
    if labels_true is not None:
        graph_meta["has_true_labels"] = True
        graph_meta["num_true_classes"] = int(np.unique(labels_true).size)
    else:
        graph_meta["has_true_labels"] = False

    A_fro = float(np.linalg.norm(A.data))
    rows = []
    labels_by_method: dict[str, list[np.ndarray]] = {m: [] for m in METHOD_ORDER}
    bucket_rows_all = []

    print(
        f"Loaded {cfg.dataset_name}: full n={A_full.shape[0]}, m={A_full.nnz // 2}; "
        f"experiment n={A.shape[0]}, m={A.nnz // 2}"
    )
    print(
        "Degree diagnostics: "
        f"max={graph_meta.get('degree_max'):.0f}, "
        f"median={graph_meta.get('degree_median'):.1f}, "
        f"gini={graph_meta.get('degree_gini'):.3f}, "
        f"tail_r2={graph_meta.get('tail_loglog_ccdf_r2'):.3f}"
    )

    ref_rng = np.random.default_rng(cfg.seed + 101)
    ref_vals, ref_U, ref_timings = dense_general_eigh(A, cfg.k)
    ref_record, labels_ref = make_record(
        method="General eigensolver",
        rep=0,
        vals=ref_vals,
        U=ref_U,
        timings=ref_timings,
        A=A,
        ref_vals=ref_vals,
        ref_U=ref_U,
        A_fro=A_fro,
        labels_reference=None,
        labels_true=labels_true,
        cfg=cfg,
        rng=ref_rng,
    )
    ref_record["ARI_vs_general_clustering"] = 1.0
    rows.append(ref_record)
    labels_by_method["General eigensolver"].append(labels_ref)

    master_rng = np.random.default_rng(cfg.seed)
    for rep in range(1, cfg.reps + 1):
        rep_seed = int(master_rng.integers(1, 2**31 - 1))
        print(f"rep {rep}/{cfg.reps}")

        rng = np.random.default_rng(rep_seed + 11)
        vals, U, timings = gaussian_random_projection(A, cfg.k, cfg.r, cfg.q, rng)
        record, labels = make_record(
            "Gaussian RP", rep, vals, U, timings, A, ref_vals, ref_U, A_fro,
            labels_ref, labels_true, cfg, rng,
        )
        rows.append(record)
        labels_by_method["Gaussian RP"].append(labels)

        rng = np.random.default_rng(rep_seed + 31)
        vals, U, timings = random_sampling_embedding(
            A, cfg.k, cfg.sampling_p, rng, cfg.rs_eigensolver
        )
        record, labels = make_record(
            "Random sampling", rep, vals, U, timings, A, ref_vals, ref_U, A_fro,
            labels_ref, labels_true, cfg, rng,
        )
        rows.append(record)
        labels_by_method["Random sampling"].append(labels)

        rng = np.random.default_rng(rep_seed + 53)
        vals, U, timings, bucket_rows = degree_stratified_random_projection(
            A, cfg.k, cfg.r, cfg.q, cfg.ell_min, rng
        )
        record, labels = make_record(
            "Degree-stratified RP", rep, vals, U, timings, A, ref_vals, ref_U, A_fro,
            labels_ref, labels_true, cfg, rng,
        )
        rows.append(record)
        labels_by_method["Degree-stratified RP"].append(labels)
        for row in bucket_rows:
            bucket_rows_all.append({"rep": int(rep), **row})

    stability_rows = []
    for method, labelings in labels_by_method.items():
        if len(labelings) < 2:
            continue
        scores = []
        for i in range(len(labelings)):
            for j in range(i + 1, len(labelings)):
                scores.append(float(adjusted_rand_score(labelings[i], labelings[j])))
        stability_rows.append(
            {
                "method": method,
                "pairwise_ari_mean": float(np.mean(scores)),
                "pairwise_ari_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "num_pairs": int(len(scores)),
            }
        )

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_raw(df_raw)
    df_bucket = pd.DataFrame(bucket_rows_all)
    df_stability = pd.DataFrame(stability_rows)

    raw_path = cfg.outdir / "degree_stratified_rp_raw.csv"
    summary_path = cfg.outdir / "degree_stratified_rp_summary.csv"
    bucket_path = cfg.outdir / "degree_stratified_bucket_allocations.csv"
    stability_path = cfg.outdir / "degree_stratified_clustering_stability.csv"
    meta_path = cfg.outdir / "degree_stratified_rp_meta.json"

    df_raw.to_csv(raw_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_bucket.to_csv(bucket_path, index=False)
    df_stability.to_csv(stability_path, index=False)
    meta = {
        "config": {
            **asdict(cfg),
            "edgelist": str(cfg.edgelist),
            "outdir": str(cfg.outdir),
        },
        "graph": graph_meta,
        "metric_note": (
            "ARI_vs_general_clustering compares each approximate embedding's k-means "
            "labels with the dense general eigensolver's spectral clustering labels. "
            "It is not ground-truth classification accuracy."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if not cfg.no_plots:
        plot_results(df_raw, cfg.outdir / "viz")

    print("Done.")
    print(f"Raw CSV      : {raw_path.resolve()}")
    print(f"Summary CSV  : {summary_path.resolve()}")
    print(f"Buckets CSV  : {bucket_path.resolve()}")
    print(f"Stability CSV: {stability_path.resolve()}")
    print(f"Meta JSON    : {meta_path.resolve()}")
    return df_raw, df_summary, df_bucket, df_stability, meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Degree-stratified randomized projection on real power-law graphs"
    )
    parser.add_argument("--edgelist", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="real-powerlaw")
    parser.add_argument("--max-n", type=int, default=1500)
    parser.add_argument(
        "--subgraph-mode",
        choices=["top-degree", "degree-weighted", "uniform", "none"],
        default="top-degree",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--r", type=int, default=20)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--sampling-p", type=float, default=0.7)
    parser.add_argument("--ell-min", type=int, default=1)
    parser.add_argument("--rs-eigensolver", choices=["general", "eigsh"], default="general")
    parser.add_argument("--no-normalize-embedding-rows", action="store_true")
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--outdir", type=Path, default=Path("results/demo"))
    parser.add_argument("--delimiter", type=str, default=None)
    parser.add_argument("--comment-prefix", type=str, default="#")
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    cfg = ExperimentConfig(
        edgelist=args.edgelist,
        dataset_name=args.dataset_name,
        max_n=args.max_n,
        subgraph_mode=args.subgraph_mode,
        k=args.k,
        r=args.r,
        q=args.q,
        reps=args.reps,
        seed=args.seed,
        sampling_p=args.sampling_p,
        ell_min=args.ell_min,
        rs_eigensolver=args.rs_eigensolver,
        normalize_embedding_rows=not args.no_normalize_embedding_rows,
        kmeans_n_init=args.kmeans_n_init,
        outdir=args.outdir,
        delimiter=args.delimiter,
        comment_prefix=args.comment_prefix,
        no_plots=args.no_plots,
    )
    run_experiment(cfg, label_path=args.label_path)


if __name__ == "__main__":
    main()
