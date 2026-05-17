from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_ROOT = EXPERIMENT_DIR / "results"
REPORT_PATH = EXPERIMENT_DIR / "결과보고서.md"
LOCAL_CACHE_DIR = EXPERIMENT_DIR / ".cache"
DIAGNOSTICS_DIR = RESULTS_ROOT / "diagnostics"
SPECTRUM_DIAGNOSTIC_PATH = DIAGNOSTICS_DIR / "spectral_gap_diagnostics.csv"
RP_PARAMETER_DIAGNOSTIC_PATH = DIAGNOSTICS_DIR / "randomization_parameter_diagnostic.csv"

(LOCAL_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(LOCAL_CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR / "xdg"))
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (  # noqa: E402
    LiveProgress,
    generate_uniform_hsbm_instance,
    hypergraph_laplacian,
    make_uniform_hsbm_probs,
    normalize_rows_l2,
)


METHODS = (
    "non_random",
    "gaussian_random_projection",
    "random_sampling",
    "countsketch_random_projection",
)
METHOD_LABELS = {
    "non_random": "Non-random eigsh",
    "gaussian_random_projection": "Gaussian RP",
    "random_sampling": "Random sampling",
    "countsketch_random_projection": "CountSketch RP",
}
METHOD_ORDER = [METHOD_LABELS[m] for m in METHODS]

DENSITY_FIXED_GAP_SCHEDULE = {
    1: (16.0, 36.0, 4.0),
    2: (24.0, 36.0, 4.0),
    3: (32.0, 40.0, 8.0),
    4: (48.0, 44.0, 12.0),
    5: (64.0, 52.0, 20.0),
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    title: str
    x_col: str
    x_values: tuple[int | float, ...]
    n: int | None = None
    K: int | None = None
    rho_n: float | None = None
    m: int = 3
    center: float = 10.0
    base_gap: float = 4.0
    reps: int = 5
    seed: int = 20260507
    sampling: str = "sparse"
    max_enumeration: int = 1_500_000
    normalize_embedding_rows: bool = True
    eigsh_tol: float = 1e-6
    rp_oversampling: int = 30
    rp_power_iter: int = 1
    random_sampling_p: float = 0.3
    kmeans_n_init: int = 20

    @property
    def outdir(self) -> Path:
        return RESULTS_ROOT / self.name

    @property
    def raw_path(self) -> Path:
        return self.outdir / f"{self.name}_raw.csv"

    @property
    def summary_path(self) -> Path:
        return self.outdir / f"{self.name}_summary.csv"

    @property
    def config_path(self) -> Path:
        return self.outdir / f"{self.name}_config.json"

    @property
    def plot_path(self) -> Path:
        return self.outdir / f"{self.name}_summary.png"


def get_specs(reps: int = 5) -> dict[str, ExperimentSpec]:
    return {
        "density_background_fixed_gap": ExperimentSpec(
            name="density_background_fixed_gap",
            title="Density sweep with stronger fixed signal gap",
            x_col="density_level",
            x_values=(1, 2, 3, 4, 5),
            n=6000,
            K=6,
            reps=reps,
            seed=2026050704,
        ),
        "K_compensated_reference_signal": ExperimentSpec(
            name="K_compensated_reference_signal",
            title="K sweep with K^2 compensation and reference signal",
            x_col="K",
            x_values=(3, 4, 6, 8, 10),
            n=6000,
            reps=reps,
            seed=2026050705,
        ),
        "n_scaling_reference_signal": ExperimentSpec(
            name="n_scaling_reference_signal",
            title="n scaling at reference K=6 signal regime",
            x_col="n",
            x_values=(3000, 6000, 9000, 12000, 15000),
            K=6,
            rho_n=16.0,
            reps=reps,
            seed=2026050706,
        ),
        "rho_density_signal_control": ExperimentSpec(
            name="rho_density_signal_control",
            title="Weak-gap diagnostic: rho_n sweep with density-signal separation",
            x_col="rho_n",
            x_values=(4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
            n=6000,
            K=6,
            reps=reps,
            seed=2026050701,
        ),
        "K_compensated_rank_scaling": ExperimentSpec(
            name="K_compensated_rank_scaling",
            title="Weak-gap diagnostic: K sweep with rho_n compensated for 3-uniform within-candidate loss",
            x_col="K",
            x_values=(3, 4, 6, 8, 10),
            n=6000,
            reps=reps,
            seed=2026050702,
        ),
        "n_scaling_fixed_density_signal": ExperimentSpec(
            name="n_scaling_fixed_density_signal",
            title="Weak-gap diagnostic: n scaling at fixed density-signal regime",
            x_col="n",
            x_values=(3000, 6000, 9000, 12000, 15000),
            K=6,
            rho_n=32.0,
            reps=reps,
            seed=2026050703,
        ),
    }


def value_to_seed_component(value: int | float) -> int:
    if isinstance(value, float):
        return int(round(value * 1000))
    return int(value)


def concrete_params(spec: ExperimentSpec, x_value: int | float) -> tuple[int, int, float, float, float]:
    if spec.name == "density_background_fixed_gap":
        rho_n, a_in, b_out = DENSITY_FIXED_GAP_SCHEDULE[int(x_value)]
        return int(spec.n), int(spec.K), float(rho_n), float(a_in), float(b_out)

    if spec.name == "K_compensated_reference_signal":
        K = int(x_value)
        rho_n = 16.0 * (float(K) / 6.0) ** 2
        return int(spec.n), K, float(rho_n), 36.0, 4.0

    if spec.name == "n_scaling_reference_signal":
        return int(x_value), int(spec.K), float(spec.rho_n), 36.0, 4.0

    n = int(x_value) if spec.x_col == "n" else int(spec.n)
    K = int(x_value) if spec.x_col == "K" else int(spec.K)
    if spec.x_col == "rho_n":
        rho_n = float(x_value)
    elif spec.x_col == "K":
        rho_n = 16.0 * (float(K) / 4.0) ** 2
    else:
        rho_n = float(spec.rho_n)

    if spec.x_col == "rho_n":
        gap = spec.base_gap * math.sqrt(16.0 / rho_n)
    elif spec.x_col == "n":
        gap = spec.base_gap * math.sqrt(16.0 / float(spec.rho_n))
    else:
        gap = spec.base_gap
    a_in = spec.center + gap / 2.0
    b_out = spec.center - gap / 2.0
    if not (a_in > b_out > 0.0):
        raise ValueError(f"invalid signal constants: a_in={a_in}, b_out={b_out}")
    return n, K, rho_n, a_in, b_out


def aligned_misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    conf = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < K and 0 <= p < K:
            conf[t, p] += 1
    true_ids, pred_ids = linear_sum_assignment(-conf)
    correct = int(conf[true_ids, pred_ids].sum())
    return float(1.0 - correct / len(y_true))


def hypergraph_vertex_degree_stats(n: int, hyperedges) -> dict[str, float | int]:
    degrees = np.zeros(n, dtype=int)
    for edge in hyperedges:
        degrees[list(edge)] += 1
    return {
        "num_isolated_nodes": int(np.sum(degrees == 0)),
        "isolated_fraction": float(np.mean(degrees == 0)) if n else 0.0,
        "hypergraph_degree_mean": float(degrees.mean()) if n else 0.0,
        "hypergraph_degree_max": float(degrees.max()) if n else 0.0,
    }


def expected_uniform_hsbm_stats(labels: np.ndarray, K: int, m: int, p_in: float, p_out: float):
    n = int(labels.shape[0])
    total = math.comb(n, m)
    within = 0
    for k in range(K):
        nk = int(np.sum(labels == k))
        if nk >= m:
            within += math.comb(nk, m)
    mixed = total - within
    expected_edges = within * float(p_in) + mixed * float(p_out)
    return {
        "expected_hyperedges_total": float(expected_edges),
        "expected_hyperedges_per_n": float(expected_edges / n),
        "expected_degree_mean": float(m * expected_edges / n),
        "candidate_within_fraction": float(within / total) if total else float("nan"),
    }


def top_eigsh_embedding(theta: sp.csr_matrix, K: int, rng: np.random.Generator, eigsh_tol: float):
    n = int(theta.shape[0])
    try:
        if n <= K + 1:
            vals, vecs = np.linalg.eigh(theta.toarray())
        else:
            vals, vecs = spla.eigsh(
                theta,
                k=K,
                which="LA",
                tol=float(eigsh_tol),
                v0=rng.normal(size=n),
            )
    except Exception:
        vals, vecs = np.linalg.eigh(theta.toarray())
    order = np.argsort(vals)[-K:][::-1]
    return vals[order], vecs[:, order]


def gaussian_random_projection_embedding(
    theta: sp.csr_matrix,
    K: int,
    r: int,
    q: int,
    rng: np.random.Generator,
    eigsh_tol: float,
):
    timings: dict[str, Any] = {}
    n = int(theta.shape[0])
    ell = int(K + r)

    t0 = time.perf_counter()
    Y = rng.standard_normal(size=(n, ell))
    timings["rp_draw_omega_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(2 * int(q) + 1):
        Y = theta @ Y
    timings["rp_power_iter_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    Q, _ = np.linalg.qr(Y, mode="reduced")
    timings["rp_qr_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    B = Q.T @ (theta @ Q)
    B = 0.5 * (B + B.T)
    timings["rp_build_core_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vals, core_vecs = top_eigsh_embedding(sp.csr_matrix(B), K=K, rng=rng, eigsh_tol=eigsh_tol)
    timings["rp_small_eig_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    U = Q @ core_vecs
    timings["rp_lift_sec"] = time.perf_counter() - t0
    return vals, U, timings


def countsketch_test_matrix(n: int, ell: int, rng: np.random.Generator):
    rows = np.arange(n, dtype=np.int64)
    cols = rng.integers(0, ell, size=n, dtype=np.int64)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    omega = sp.csr_matrix((signs, (rows, cols)), shape=(n, ell), dtype=float)
    bucket_counts = np.bincount(cols, minlength=ell)
    return omega, {
        "cs_embedding_dim": int(ell),
        "cs_bucket_min_load": int(bucket_counts.min()) if ell else 0,
        "cs_bucket_max_load": int(bucket_counts.max()) if ell else 0,
        "cs_empty_buckets": int(np.sum(bucket_counts == 0)),
    }


def countsketch_random_projection_embedding(
    theta: sp.csr_matrix,
    K: int,
    r: int,
    q: int,
    rng: np.random.Generator,
    eigsh_tol: float,
):
    timings: dict[str, Any] = {}
    n = int(theta.shape[0])
    ell = int(K + r)

    t0 = time.perf_counter()
    omega, sketch_stats = countsketch_test_matrix(n=n, ell=ell, rng=rng)
    timings["cs_draw_hash_sec"] = time.perf_counter() - t0
    timings.update(sketch_stats)

    t0 = time.perf_counter()
    Y = (theta @ omega).toarray()
    timings["cs_initial_multiply_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(2 * int(q)):
        Y = theta @ Y
    timings["cs_power_iter_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    Q, _ = np.linalg.qr(Y, mode="reduced")
    timings["cs_qr_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    B = Q.T @ (theta @ Q)
    B = 0.5 * (B + B.T)
    timings["cs_build_core_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vals, core_vecs = top_eigsh_embedding(sp.csr_matrix(B), K=K, rng=rng, eigsh_tol=eigsh_tol)
    timings["cs_small_eig_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    U = Q @ core_vecs
    timings["cs_lift_sec"] = time.perf_counter() - t0
    return vals, U, timings


def sample_rescaled_symmetric_sparse_matrix(A: sp.csr_matrix, p: float, rng: np.random.Generator):
    if not (0.0 < float(p) <= 1.0):
        raise ValueError(f"random sampling probability must be in (0, 1], got {p}")
    upper = sp.triu(A, k=0, format="coo")
    keep = rng.random(upper.nnz) < float(p)
    rows = upper.row[keep]
    cols = upper.col[keep]
    data = upper.data[keep] / float(p)
    off = rows != cols
    all_rows = np.concatenate([rows, cols[off]])
    all_cols = np.concatenate([cols, rows[off]])
    all_data = np.concatenate([data, data[off]])
    sampled = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=A.shape, dtype=float).tocsr()
    sampled.sum_duplicates()
    sampled.eliminate_zeros()
    return sampled, {
        "rs_original_upper_nnz": int(upper.nnz),
        "rs_sampled_upper_nnz": int(np.sum(keep)),
        "rs_sampling_probability": float(p),
    }


def spectral_cluster_from_theta(
    theta: sp.csr_matrix,
    K: int,
    rng: np.random.Generator,
    spec: ExperimentSpec,
    method: str,
):
    theta = ((theta + theta.T) * 0.5).tocsr()
    theta.eliminate_zeros()
    timings: dict[str, Any] = {}
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    if method == "non_random":
        vals, U = top_eigsh_embedding(theta, K=K, rng=rng, eigsh_tol=spec.eigsh_tol)
    elif method == "gaussian_random_projection":
        vals, U, extra = gaussian_random_projection_embedding(
            theta,
            K=K,
            r=spec.rp_oversampling,
            q=spec.rp_power_iter,
            rng=rng,
            eigsh_tol=spec.eigsh_tol,
        )
        timings.update(extra)
    elif method == "random_sampling":
        t_sample = time.perf_counter()
        sampled_theta, sample_stats = sample_rescaled_symmetric_sparse_matrix(
            theta,
            p=spec.random_sampling_p,
            rng=rng,
        )
        timings["rs_sample_matrix_wall_sec"] = time.perf_counter() - t_sample
        timings.update(sample_stats)
        vals, U = top_eigsh_embedding(sampled_theta, K=K, rng=rng, eigsh_tol=spec.eigsh_tol)
    elif method == "countsketch_random_projection":
        vals, U, extra = countsketch_random_projection_embedding(
            theta,
            K=K,
            r=spec.rp_oversampling,
            q=spec.rp_power_iter,
            rng=rng,
            eigsh_tol=spec.eigsh_tol,
        )
        timings.update(extra)
    else:
        raise ValueError(f"unknown method: {method}")
    timings["eigen_decomposition_wall_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if spec.normalize_embedding_rows:
        U = normalize_rows_l2(U)
    timings["embedding_normalize_wall_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels = KMeans(
        n_clusters=K,
        n_init=int(spec.kmeans_n_init),
        random_state=int(rng.integers(1, 2**31 - 1)),
    ).fit_predict(U)
    timings["kmeans_wall_sec"] = time.perf_counter() - t0
    timings["spectral_clustering_wall_sec"] = time.perf_counter() - total_start
    timings["top_eigenvalue_max"] = float(np.max(vals)) if len(vals) else float("nan")
    timings["top_eigenvalue_min"] = float(np.min(vals)) if len(vals) else float("nan")
    return labels, timings


def method_seed(seed: int, method: str) -> int:
    return int(seed + (METHODS.index(method) + 1) * 10_000_000)


def run_one_instance(spec: ExperimentSpec, x_value: int | float, rep: int) -> list[dict[str, Any]]:
    n, K, rho_n, a_in, b_out = concrete_params(spec, x_value)
    seed = int(spec.seed + value_to_seed_component(x_value) * 100_000 + rep)
    rng = np.random.default_rng(seed)
    p_in, p_out = make_uniform_hsbm_probs(
        n=n,
        d=spec.m,
        a_d=a_in,
        b_d=b_out,
        rho_n=rho_n,
        clip=False,
    )

    t0 = time.perf_counter()
    hyperedges, y_true, _, gen_stats = generate_uniform_hsbm_instance(
        n=n,
        K=K,
        m=spec.m,
        p_in=p_in,
        p_out=p_out,
        rng=rng,
        sampling=spec.sampling,
        max_enumeration=spec.max_enumeration,
    )
    generation_wall_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    L = hypergraph_laplacian(n=n, hyperedges=hyperedges)
    theta = (sp.eye(n, format="csr", dtype=float) - L).tocsr()
    theta.eliminate_zeros()
    build_wall_sec = time.perf_counter() - t0

    shared = {
        "experiment": spec.name,
        spec.x_col: x_value,
        "rep": int(rep),
        "seed": int(seed),
        "n": int(n),
        "K": int(K),
        "m": int(spec.m),
        "rho_n": float(rho_n),
        "a_in": float(a_in),
        "b_out": float(b_out),
        "signal_gap": float(a_in - b_out),
        "p_in": float(p_in),
        "p_out": float(p_out),
        "num_hyperedges_total": int(len(hyperedges)),
        "theta_nnz": int(theta.nnz),
        "theta_density": float(theta.nnz / (n * n)),
        "generation_wall_sec": float(generation_wall_sec),
        "hypergraph_laplacian_build_wall_sec": float(build_wall_sec),
        "sampling_mode": gen_stats.get("sampling_mode", ""),
        **hypergraph_vertex_degree_stats(n, hyperedges),
        **expected_uniform_hsbm_stats(y_true, K, spec.m, p_in, p_out),
    }

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rng = np.random.default_rng(method_seed(seed, method))
        t0 = time.perf_counter()
        y_pred, spectral_stats = spectral_cluster_from_theta(
            theta=theta,
            K=K,
            rng=method_rng,
            spec=spec,
            method=method,
        )
        method_wall_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        mis = aligned_misclassification_rate(y_true, y_pred, K)
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        metric_wall_sec = time.perf_counter() - t0

        record = {
            **shared,
            "method": METHOD_LABELS[method],
            "method_key": method,
            "misclassification_rate": float(mis),
            "ARI": float(ari),
            "NMI": float(nmi),
            "metric_wall_sec": float(metric_wall_sec),
            **spectral_stats,
        }
        record["method_wall_sec"] = float(method_wall_sec)
        record["algorithm_total_wall_sec"] = float(
            record["generation_wall_sec"]
            + record["hypergraph_laplacian_build_wall_sec"]
            + record["eigen_decomposition_wall_sec"]
            + record["embedding_normalize_wall_sec"]
            + record["kmeans_wall_sec"]
        )
        rows.append(record)

    del L, theta, hyperedges
    gc.collect()
    return rows


def completed_run_keys(raw_path: Path, x_col: str) -> set[tuple[int, int]]:
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(raw_path, usecols=[x_col, "rep", "method"])
    complete = set()
    for (x_value, rep), group in df.groupby([x_col, "rep"], sort=False):
        if group["method"].nunique() >= len(METHODS):
            complete.add((value_to_seed_component(x_value), int(rep)))
    return complete


def append_raw_rows(raw_path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    if raw_path.exists() and raw_path.stat().st_size > 0:
        existing_columns = list(pd.read_csv(raw_path, nrows=0).columns)
        for col in existing_columns:
            if col not in df.columns:
                df[col] = np.nan
        extra_columns = [col for col in df.columns if col not in existing_columns]
        if extra_columns:
            existing = pd.read_csv(raw_path)
            for col in extra_columns:
                existing[col] = np.nan
            tmp_path = raw_path.with_suffix(".tmp.csv")
            pd.concat([existing, df[existing_columns + extra_columns]], ignore_index=True).to_csv(
                tmp_path,
                index=False,
            )
            tmp_path.replace(raw_path)
        else:
            df[existing_columns].to_csv(raw_path, mode="a", header=False, index=False)
    else:
        df.to_csv(raw_path, index=False)


def summarize_raw(df_raw: pd.DataFrame, x_col: str) -> pd.DataFrame:
    preferred = [
        "n",
        "K",
        "rho_n",
        "a_in",
        "b_out",
        "signal_gap",
        "p_in",
        "p_out",
        "num_hyperedges_total",
        "theta_nnz",
        "theta_density",
        "hypergraph_degree_mean",
        "hypergraph_degree_max",
        "expected_degree_mean",
        "candidate_within_fraction",
        "isolated_fraction",
        "misclassification_rate",
        "ARI",
        "NMI",
        "generation_wall_sec",
        "hypergraph_laplacian_build_wall_sec",
        "eigen_decomposition_wall_sec",
        "embedding_normalize_wall_sec",
        "kmeans_wall_sec",
        "spectral_clustering_wall_sec",
        "metric_wall_sec",
        "algorithm_total_wall_sec",
        "rp_power_iter_sec",
        "rp_qr_sec",
        "rp_build_core_sec",
        "rs_sample_matrix_wall_sec",
        "rs_sampled_upper_nnz",
        "cs_initial_multiply_sec",
        "cs_power_iter_sec",
        "cs_qr_sec",
        "cs_build_core_sec",
        "cs_embedding_dim",
        "top_eigenvalue_max",
        "top_eigenvalue_min",
    ]
    aggs: dict[str, tuple[str, str]] = {"reps": ("rep", "count")}
    for col in preferred:
        if col in df_raw.columns:
            label = "misclassification" if col == "misclassification_rate" else col
            aggs[f"{label}_mean"] = (col, "mean")
            aggs[f"{label}_std"] = (col, "std")
    summary = df_raw.groupby([x_col, "method"], as_index=False).agg(**aggs)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)

    if "spectral_clustering_wall_sec_mean" in summary.columns:
        base = (
            summary[summary["method"] == METHOD_LABELS["non_random"]][[x_col, "spectral_clustering_wall_sec_mean"]]
            .rename(columns={"spectral_clustering_wall_sec_mean": "non_random_spectral_sec_mean"})
        )
        summary = summary.merge(base, on=x_col, how="left")
        summary["spectral_speedup_vs_non_random"] = (
            summary["non_random_spectral_sec_mean"] / summary["spectral_clustering_wall_sec_mean"]
        )
    return summary.sort_values([x_col, "method"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, spec: ExperimentSpec) -> None:
    x = spec.x_col
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    panels = [
        ("misclassification_mean", "Misclassification"),
        ("theta_nnz_mean", "Theta nnz"),
        ("spectral_clustering_wall_sec_mean", "Spectral time (sec)"),
        ("spectral_speedup_vs_non_random", "Spectral speedup vs non-random"),
    ]
    for ax, (col, ylabel) in zip(axes.ravel(), panels):
        if col not in summary.columns:
            ax.set_visible(False)
            continue
        for method in METHOD_ORDER:
            dm = summary[summary["method"] == method].sort_values(x)
            if dm.empty:
                continue
            ax.plot(dm[x], dm[col], marker="o", linewidth=2, label=method)
        ax.set_xlabel(x)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(spec.title)
    fig.tight_layout()
    fig.savefig(spec.plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_theta_for_instance(spec: ExperimentSpec, x_value: int | float, rep: int = 1):
    n, K, rho_n, a_in, b_out = concrete_params(spec, x_value)
    seed = int(spec.seed + value_to_seed_component(x_value) * 100_000 + rep)
    rng = np.random.default_rng(seed)
    p_in, p_out = make_uniform_hsbm_probs(
        n=n,
        d=spec.m,
        a_d=a_in,
        b_d=b_out,
        rho_n=rho_n,
        clip=False,
    )
    hyperedges, y_true, _, _ = generate_uniform_hsbm_instance(
        n=n,
        K=K,
        m=spec.m,
        p_in=p_in,
        p_out=p_out,
        rng=rng,
        sampling=spec.sampling,
        max_enumeration=spec.max_enumeration,
    )
    L = hypergraph_laplacian(n=n, hyperedges=hyperedges)
    theta = (sp.eye(n, format="csr", dtype=float) - L).tocsr()
    theta.eliminate_zeros()
    meta = {
        "seed": int(seed),
        "n": int(n),
        "K": int(K),
        "rho_n": float(rho_n),
        "a_in": float(a_in),
        "b_out": float(b_out),
        "signal_gap": float(a_in - b_out),
        "num_hyperedges_total": int(len(hyperedges)),
        "theta_nnz": int(theta.nnz),
        "theta_density": float(theta.nnz / (n * n)),
    }
    return theta, y_true, meta


def run_spectral_gap_diagnostics(specs: dict[str, ExperimentSpec]) -> pd.DataFrame:
    rows = []
    diagnostic_specs = [
        "density_background_fixed_gap",
        "K_compensated_reference_signal",
    ]
    for spec_name in diagnostic_specs:
        spec = specs[spec_name]
        for x_value in spec.x_values:
            theta, _, meta = build_theta_for_instance(spec=spec, x_value=x_value, rep=1)
            K = int(meta["K"])
            k_eigs = min(theta.shape[0] - 2, K + 10)
            vals = spla.eigsh(
                theta,
                k=k_eigs,
                which="LA",
                tol=float(spec.eigsh_tol),
                v0=np.random.default_rng(int(meta["seed"]) + 9_991).normal(size=theta.shape[0]),
                return_eigenvectors=False,
            )
            vals = np.sort(vals)[::-1]
            lambda_k = float(vals[K - 1]) if len(vals) >= K else float("nan")
            lambda_kplus1 = float(vals[K]) if len(vals) > K else float("nan")
            gap = lambda_k - lambda_kplus1
            rows.append(
                {
                    "experiment": spec.name,
                    "title": spec.title,
                    "x_col": spec.x_col,
                    "x_value": x_value,
                    **meta,
                    "num_eigenvalues": int(len(vals)),
                    "lambda_1": float(vals[0]) if len(vals) else float("nan"),
                    "lambda_K": lambda_k,
                    "lambda_Kplus1": lambda_kplus1,
                    "eigengap_after_K": float(gap),
                    "relative_eigengap_after_K": float(gap / max(abs(lambda_k), 1e-12)),
                    "lambda_tail_min_recorded": float(vals[-1]) if len(vals) else float("nan"),
                }
            )
            del theta
            gc.collect()
    df = pd.DataFrame(rows)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPECTRUM_DIAGNOSTIC_PATH, index=False)
    return df


def run_randomization_parameter_diagnostic(specs: dict[str, ExperimentSpec]) -> pd.DataFrame:
    base_spec = specs["density_background_fixed_gap"]
    density_level = 5
    theta, y_true, meta = build_theta_for_instance(spec=base_spec, x_value=density_level, rep=1)
    K = int(meta["K"])
    configs = [
        ("Non-random eigsh", "non_random", "baseline", 30, 1, 0.3),
        ("Gaussian RP r=30 q=1", "gaussian_random_projection", "fast", 30, 1, 0.3),
        ("Gaussian RP r=160 q=3", "gaussian_random_projection", "wide", 160, 3, 0.3),
        ("CountSketch RP r=30 q=1", "countsketch_random_projection", "fast", 30, 1, 0.3),
        ("CountSketch RP r=160 q=3", "countsketch_random_projection", "wide", 160, 3, 0.3),
        ("Random sampling p=0.3", "random_sampling", "fast", 30, 1, 0.3),
        ("Random sampling p=0.7", "random_sampling", "less_sparse", 30, 1, 0.7),
        ("Random sampling p=0.9", "random_sampling", "near_full", 30, 1, 0.9),
        ("Random sampling p=1.0", "random_sampling", "full_control", 30, 1, 1.0),
    ]

    rows = []
    for label, method, setting, oversampling, power_iter, sampling_p in configs:
        diag_spec = ExperimentSpec(
            name="randomization_parameter_diagnostic",
            title="Randomization parameter diagnostic",
            x_col="density_level",
            x_values=(density_level,),
            n=int(meta["n"]),
            K=K,
            rho_n=float(meta["rho_n"]),
            rp_oversampling=int(oversampling),
            rp_power_iter=int(power_iter),
            random_sampling_p=float(sampling_p),
            kmeans_n_init=20,
        )
        rng = np.random.default_rng(int(meta["seed"]) + len(rows) * 1009 + 77)
        y_pred, stats = spectral_cluster_from_theta(
            theta=theta,
            K=K,
            rng=rng,
            spec=diag_spec,
            method=method,
        )
        rows.append(
            {
                "experiment": "randomization_parameter_diagnostic",
                "density_level": int(density_level),
                **meta,
                "method": label,
                "method_key": method,
                "setting": setting,
                "rp_oversampling": int(oversampling),
                "rp_power_iter": int(power_iter),
                "random_sampling_p": float(sampling_p),
                "misclassification_rate": aligned_misclassification_rate(y_true, y_pred, K),
                "ARI": adjusted_rand_score(y_true, y_pred),
                "NMI": normalized_mutual_info_score(y_true, y_pred),
                **stats,
            }
        )
    df = pd.DataFrame(rows)
    base_sec = float(
        df.loc[df["method"] == "Non-random eigsh", "spectral_clustering_wall_sec"].iloc[0]
    )
    df["spectral_speedup_vs_non_random"] = base_sec / df["spectral_clustering_wall_sec"]
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RP_PARAMETER_DIAGNOSTIC_PATH, index=False)
    del theta
    gc.collect()
    return df


def run_diagnostics(specs: dict[str, ExperimentSpec]) -> None:
    run_spectral_gap_diagnostics(specs)
    run_randomization_parameter_diagnostic(specs)


def run_spec(spec: ExperimentSpec, show_progress: bool = True) -> pd.DataFrame:
    spec.outdir.mkdir(parents=True, exist_ok=True)
    completed = completed_run_keys(spec.raw_path, spec.x_col)
    progress = LiveProgress(len(spec.x_values) * spec.reps) if show_progress else None
    for x_value in spec.x_values:
        for rep in range(1, spec.reps + 1):
            key = (value_to_seed_component(x_value), int(rep))
            if key not in completed:
                rows = run_one_instance(spec=spec, x_value=x_value, rep=rep)
                append_raw_rows(spec.raw_path, rows)
                completed.add(key)
            if progress is not None:
                progress.update(spec.x_col, x_value, rep, spec.reps, "all methods")
    if progress is not None:
        progress.close()

    df_raw = pd.read_csv(spec.raw_path)
    summary = summarize_raw(df_raw, spec.x_col)
    summary.to_csv(spec.summary_path, index=False)
    config = asdict(spec)
    config["methods"] = METHOD_ORDER
    config["design_note"] = (
        "Density and signal are separated through primary strong-signal sweeps and weak-gap "
        "diagnostics. K sweeps include rho_n compensation for the 3-uniform within-candidate loss."
    )
    spec.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_summary(summary, spec)
    return summary


def fmt(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def best_bold_rows(rows: list[list[str]], values: list[float], smaller_is_better: bool = True):
    if not values:
        return rows
    best = min(values) if smaller_is_better else max(values)
    out = []
    for row, value in zip(rows, values):
        if np.isclose(value, best):
            out.append([f"**{cell}**" for cell in row])
        else:
            out.append(row)
    return out


def load_summaries(specs: dict[str, ExperimentSpec]) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(spec.summary_path) for name, spec in specs.items()}


def write_report(specs: dict[str, ExperimentSpec]) -> Path:
    summaries = load_summaries(specs)
    lines: list[str] = []
    lines.append("# 균일 HSBM density-signal 재설계 실험 결과보고서")
    lines.append("")
    lines.append(
        "이 보고서는 기존 `rho_n`/`K` sweep에서 보였던 정확도 포화와 K 증가 시 급격한 붕괴를 피하기 위해, "
        "density, signal gap, target rank 효과를 분리해서 다시 실행한 결과입니다. 모든 결과와 CSV는 이 새 폴더 안에만 저장했습니다."
    )
    lines.append("")
    lines.append("## 실험 설계")
    lines.append("")
    lines.append("- 주요 density sweep: `n=6000`, `K=6`에서 `(rho_n, a_in, b_out)`을 `(16,36,4)`, `(24,36,4)`, `(32,40,8)`, `(48,44,12)`, `(64,52,20)`으로 바꿉니다. background density를 키우되 signal gap은 충분히 크게 유지해 정확도와 계산량을 함께 봅니다.")
    lines.append("- 주요 `K` sweep: `n=6000`, `K={3,4,6,8,10}`, `a_in=36`, `b_out=4`, `rho_n=16*(K/6)^2`. 3-uniform에서 within 후보 비율이 대략 `1/K^2`로 줄어드는 것을 보정합니다.")
    lines.append("- 주요 `n` sweep: `K=6`, `rho_n=16`, `a_in=36`, `b_out=4`, `n={3000,6000,9000,12000,15000}`. 같은 signal regime에서 scaling을 봅니다.")
    lines.append("- 진단용 weak-gap 블록도 함께 남겼습니다. `center=10` 근처에서 gap을 작게 잡으면 계산 speedup은 보이지만 non-random 자체가 random baseline에 머무는 것을 확인하기 위한 대조군입니다.")
    lines.append("- randomized method의 본 실험 설정은 계산 이점을 보기 위해 `oversampling=30`, `power_iter=1`, random sampling `p=0.3`으로 낮췄습니다. 별도 진단 표에서는 RP 폭/반복과 sampling 확률을 키워도 정확도가 회복되는지 확인했습니다.")
    lines.append("- 확률 계산은 `clip=False`로 수행해 잘못된 큰 확률이 조용히 잘리지 않게 했습니다.")
    lines.append("")

    density_summary = summaries["density_background_fixed_gap"]
    density_nr = density_summary[density_summary["method"] == METHOD_LABELS["non_random"]].sort_values("density_level")
    k_summary = summaries["K_compensated_reference_signal"]
    k_nr = k_summary[k_summary["method"] == METHOD_LABELS["non_random"]].sort_values("K")
    fast_randomized = density_summary[density_summary["method"] != METHOD_LABELS["non_random"]]
    lines.append("## 핵심 결론")
    lines.append("")
    lines.append(
        "- 새 density sweep에서는 `Theta.nnz`가 "
        f"{fmt(density_nr['theta_nnz_mean'].iloc[0], 1)}에서 {fmt(density_nr['theta_nnz_mean'].iloc[-1], 1)}까지 커졌습니다. "
        f"Non-random 오분류율은 {fmt(density_nr['misclassification_mean'].min())}~{fmt(density_nr['misclassification_mean'].max())} 범위라, "
        "`rho_n` 증가가 단순한 즉시 포화만 만들지는 않았습니다."
    )
    lines.append(
        "- 반대로 빠른 randomized 설정은 평균 오분류율이 "
        f"{fmt(fast_randomized['misclassification_mean'].mean())}로 거의 random baseline입니다. "
        "고밀도에서 speedup이 보여도 정확도를 잃은 speedup이라 실험 주장에는 쓰기 어렵습니다."
    )
    lines.append(
        "- `K^2` 보정을 넣어도 `K=8,10`에서는 Non-random 오분류율이 각각 "
        f"{fmt(k_nr.loc[k_nr['K'] == 8, 'misclassification_mean'].iloc[0])}, "
        f"{fmt(k_nr.loc[k_nr['K'] == 10, 'misclassification_mean'].iloc[0])}까지 올라갑니다. "
        "`K` 증가는 단순 density 문제가 아니라 target rank와 spectral gap 문제를 같이 키웁니다."
    )
    lines.append("")

    if SPECTRUM_DIAGNOSTIC_PATH.exists() or RP_PARAMETER_DIAGNOSTIC_PATH.exists():
        lines.append("## 스펙트럼/랜덤화 진단")
        lines.append("")
        if SPECTRUM_DIAGNOSTIC_PATH.exists():
            gap_df = pd.read_csv(SPECTRUM_DIAGNOSTIC_PATH)
            gap_rows = []
            for _, row in gap_df.iterrows():
                gap_rows.append(
                    [
                        row["experiment"],
                        row["x_col"],
                        fmt(row["x_value"]),
                        fmt(row["K"], 0),
                        fmt(row["theta_nnz"], 1),
                        fmt(row["lambda_K"], 6),
                        fmt(row["lambda_Kplus1"], 6),
                        fmt(row["relative_eigengap_after_K"], 6),
                    ]
                )
            lines.append(
                markdown_table(
                    ["block", "x", "value", "K", "Theta nnz", "lambda_K", "lambda_K+1", "relative gap"],
                    gap_rows,
                )
            )
            lines.append("")
            lines.append(
                "대표 인스턴스에서 `lambda_K`와 `lambda_{K+1}`가 매우 붙어 있습니다. "
                "이 gap이 작으면 randomized range finder가 상위 K 공간을 조금만 잘못 잡아도 clustering이 무너집니다."
            )
            lines.append("")

        if RP_PARAMETER_DIAGNOSTIC_PATH.exists():
            rp_df = pd.read_csv(RP_PARAMETER_DIAGNOSTIC_PATH)
            rp_rows = []
            for _, row in rp_df.iterrows():
                rp_rows.append(
                    [
                        row["method"],
                        row["setting"],
                        fmt(row["misclassification_rate"]),
                        fmt(row["ARI"]),
                        fmt(row["NMI"]),
                        fmt(row["spectral_clustering_wall_sec"]),
                        fmt(row["spectral_speedup_vs_non_random"]),
                    ]
                )
            lines.append(
                markdown_table(
                    ["method", "setting", "오분류율", "ARI", "NMI", "spectral초", "speedup"],
                    rp_rows,
                )
            )
            lines.append("")
            lines.append(
                "고밀도 level 5에서 RP를 `r=160,q=3`까지 키워도 정확도는 회복되지 않았고, "
                "sampling은 `p=1.0`에 가까워져야 Non-random 수준으로 돌아옵니다. 이 경우 계산 이점은 사라집니다."
            )
            lines.append("")

    overall_rows = []
    for name, spec in specs.items():
        summary = summaries[name]
        for method in METHOD_ORDER:
            dm = summary[summary["method"] == method]
            if dm.empty:
                continue
            overall_rows.append(
                [
                    spec.name,
                    method,
                    fmt(dm["misclassification_mean"].mean()),
                    fmt(dm["ARI_mean"].mean()),
                    fmt(dm["NMI_mean"].mean()),
                    fmt(dm["theta_nnz_mean"].mean(), 1),
                    fmt(dm["spectral_clustering_wall_sec_mean"].mean()),
                    fmt(dm["spectral_speedup_vs_non_random"].mean()),
                ]
            )
    lines.append("## 전체 요약")
    lines.append("")
    lines.append(
        markdown_table(
            ["block", "method", "평균 오분류율", "평균 ARI", "평균 NMI", "평균 Theta nnz", "평균 spectral초", "평균 speedup"],
            overall_rows,
        )
    )
    lines.append("")

    for name, spec in specs.items():
        summary = summaries[name]
        lines.append(f"## {spec.title}")
        lines.append("")
        for x_value, group in summary.groupby(spec.x_col, sort=True):
            lines.append(f"### {spec.x_col} = {fmt(x_value)}")
            lines.append("")
            acc_rows = []
            mis_values = []
            for _, row in group.sort_values("method").iterrows():
                acc_rows.append(
                    [
                        row["method"],
                        fmt(row["misclassification_mean"]),
                        fmt(row["ARI_mean"]),
                        fmt(row["NMI_mean"]),
                        fmt(row["spectral_clustering_wall_sec_mean"]),
                        fmt(row["spectral_speedup_vs_non_random"]),
                    ]
                )
                mis_values.append(float(row["misclassification_mean"]))
            lines.append(
                markdown_table(
                    ["방법", "오분류율", "ARI", "NMI", "spectral초", "speedup"],
                    best_bold_rows(acc_rows, mis_values, smaller_is_better=True),
                )
            )
            lines.append("")

            base = group[group["method"] == METHOD_LABELS["non_random"]].iloc[0]
            density_rows = [
                [
                    fmt(base["n_mean"], 0),
                    fmt(base["K_mean"], 0),
                    fmt(base["rho_n_mean"]),
                    fmt(base["a_in_mean"]),
                    fmt(base["b_out_mean"]),
                    fmt(base["signal_gap_mean"]),
                    fmt(base["num_hyperedges_total_mean"], 1),
                    fmt(base["hypergraph_degree_mean_mean"]),
                    fmt(base["theta_nnz_mean"], 1),
                    fmt(base["theta_density_mean"], 6),
                ]
            ]
            lines.append(
                markdown_table(
                    ["n", "K", "rho_n", "a_in", "b_out", "gap", "하이퍼엣지", "평균degree", "Theta nnz", "Theta density"],
                    density_rows,
                )
            )
            lines.append("")

        rel_plot = spec.plot_path.relative_to(EXPERIMENT_DIR).as_posix()
        lines.append("### 요약 그림")
        lines.append("")
        lines.append(f"![{spec.title}]({rel_plot})")
        lines.append("")

    lines.append("## 해석 메모")
    lines.append("")
    lines.append("- `Theta.nnz`는 하이퍼엣지 수가 아니라, 하이퍼엣지에서 함께 등장한 vertex pair의 unique support에 가깝습니다. 같은 pair가 반복되면 nnz가 아니라 weight가 커집니다.")
    lines.append("- 주요 density sweep은 signal gap을 크게 유지하면서 background density를 키운 블록입니다. weak-gap diagnostic은 speedup이 좋아 보여도 non-random 자체가 실패하는 대조군으로만 해석해야 합니다.")
    lines.append("- `K` sweep에서 `rho_n`을 `K^2`로 보정해도 큰 `K`에서는 target rank 증가와 `lambda_K-lambda_{K+1}` gap 축소가 남습니다. 따라서 K 문제는 density 하나로 해결되지 않습니다.")
    lines.append("- `speedup`은 generation/build를 제외한 spectral clustering 단계 기준입니다. randomized method의 speedup은 정확도가 Non-random과 비슷할 때만 의미가 있습니다.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT_PATH


def run_all(reps: int = 5, show_progress: bool = True) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    specs = get_specs(reps=reps)
    for spec in specs.values():
        run_spec(spec, show_progress=show_progress)
    run_diagnostics(specs)
    return write_report(specs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run redesigned uniform HSBM density-signal experiments.")
    parser.add_argument("target", choices=["rho", "K", "n", "all", "diagnostics", "report"])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    specs = get_specs(reps=args.reps)
    if args.target == "report":
        print(write_report(specs))
    elif args.target == "diagnostics":
        run_diagnostics(specs)
        print(write_report(specs))
    elif args.target == "all":
        print(run_all(reps=args.reps, show_progress=not args.no_progress))
    else:
        key = {
            "rho": "rho_density_signal_control",
            "K": "K_compensated_rank_scaling",
            "n": "n_scaling_fixed_density_signal",
        }[args.target]
        run_spec(specs[key], show_progress=not args.no_progress)
        print(write_report(specs))


if __name__ == "__main__":
    main()
