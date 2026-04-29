from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
import tracemalloc
import gc
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_ROOT = EXPERIMENT_DIR / "results"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/python-rand-nla-matplotlib-cache")

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


METHOD_LABELS = {
    "non_random": "Non-random",
    "gaussian_random_projection": "Gaussian random projection",
    "random_sampling": "Random sampling",
}

METHOD_KOREAN = {
    "Non-random": "비랜덤 eigsh",
    "Gaussian random projection": "가우시안 랜덤 프로젝션",
    "Random sampling": "랜덤 샘플링",
}


@dataclass(frozen=True)
class SweepSpec:
    sweep: str
    method: str
    experiment_id: str
    experiment_slug: str
    notebook_name: str
    title_ko: str
    x_col: str
    x_values: tuple[float | int, ...]
    n: int | None = None
    K: int | None = None
    m: int = 3
    a_in: float = 36.0
    b_out: float = 4.0
    rho_n: float | None = None
    reps: int = 10
    seed: int = 20260427
    sampling: str = "sparse"
    max_enumeration: int = 1_500_000
    normalize_embedding_rows: bool = True
    eigsh_tol: float = 1e-6
    rp_oversampling: int = 10
    rp_power_iter: int = 2
    random_sampling_p: float = 0.7
    kmeans_n_init: int = 20

    @property
    def method_label(self) -> str:
        return METHOD_LABELS[self.method]

    @property
    def outdir(self) -> Path:
        return RESULTS_ROOT / f"{self.experiment_id}_{self.experiment_slug}"

    @property
    def file_prefix(self) -> str:
        return f"{self.experiment_id}_{self.experiment_slug}"


def get_randomized_specs() -> dict[tuple[str, str], SweepSpec]:
    K_VALUES = (2, 3, 4, 5, 6, 8, 10, 12)
    N_VALUES = tuple(range(1000, 10001, 1000))
    RHO_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

    specs = [
        SweepSpec(
            sweep="K",
            method="gaussian_random_projection",
            experiment_id="EXP-20260428-001",
            experiment_slug="uniform_hsbm_K_sweep_gaussian_random_projection",
            notebook_name="K변화_gaussian_random_projection.ipynb",
            title_ko="균일 HSBM K 변화 - 가우시안 랜덤 프로젝션",
            x_col="K",
            x_values=K_VALUES,
            n=5000,
            rho_n=8.0,
            seed=20260427,
            rp_oversampling=160,
            rp_power_iter=4,
        ),
        SweepSpec(
            sweep="K",
            method="random_sampling",
            experiment_id="EXP-20260428-002",
            experiment_slug="uniform_hsbm_K_sweep_random_sampling",
            notebook_name="K변화_random_sampling.ipynb",
            title_ko="균일 HSBM K 변화 - 랜덤 샘플링",
            x_col="K",
            x_values=K_VALUES,
            n=5000,
            rho_n=8.0,
            seed=20260427,
        ),
        SweepSpec(
            sweep="n",
            method="gaussian_random_projection",
            experiment_id="EXP-20260428-003",
            experiment_slug="uniform_hsbm_n_scaling_gaussian_random_projection",
            notebook_name="n변화_gaussian_random_projection.ipynb",
            title_ko="균일 HSBM n 변화 - 가우시안 랜덤 프로젝션",
            x_col="n",
            x_values=N_VALUES,
            K=3,
            rho_n=4.0,
            seed=20260426,
            rp_oversampling=160,
            rp_power_iter=4,
        ),
        SweepSpec(
            sweep="n",
            method="random_sampling",
            experiment_id="EXP-20260428-004",
            experiment_slug="uniform_hsbm_n_scaling_random_sampling",
            notebook_name="n변화_random_sampling.ipynb",
            title_ko="균일 HSBM n 변화 - 랜덤 샘플링",
            x_col="n",
            x_values=N_VALUES,
            K=3,
            rho_n=4.0,
            seed=20260426,
        ),
        SweepSpec(
            sweep="rho_n",
            method="gaussian_random_projection",
            experiment_id="EXP-20260428-005",
            experiment_slug="uniform_hsbm_rho_n_sweep_gaussian_random_projection",
            notebook_name="rho_n변화_gaussian_random_projection.ipynb",
            title_ko="균일 HSBM rho_n 변화 - 가우시안 랜덤 프로젝션",
            x_col="rho_n",
            x_values=RHO_VALUES,
            n=5000,
            K=3,
            seed=20260427,
            rp_oversampling=160,
            rp_power_iter=4,
        ),
        SweepSpec(
            sweep="rho_n",
            method="random_sampling",
            experiment_id="EXP-20260428-006",
            experiment_slug="uniform_hsbm_rho_n_sweep_random_sampling",
            notebook_name="rho_n변화_random_sampling.ipynb",
            title_ko="균일 HSBM rho_n 변화 - 랜덤 샘플링",
            x_col="rho_n",
            x_values=RHO_VALUES,
            n=5000,
            K=3,
            seed=20260427,
        ),
    ]
    return {(spec.sweep, spec.method): spec for spec in specs}


def resolve_spec(sweep: str, method: str) -> SweepSpec:
    specs = get_randomized_specs()
    key = (sweep, method)
    if key not in specs:
        raise KeyError(f"Unknown randomized experiment spec: {key}")
    return specs[key]


def current_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024.0**2))
    except Exception:
        return float("nan")


def measure_call(fn):
    gc.collect()
    rss_before_mb = current_rss_mb()
    tracemalloc.start()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    value = fn()
    wall_clock_sec = time.perf_counter() - wall_start
    cpu_time_sec = time.process_time() - cpu_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_mb = current_rss_mb()
    return value, {
        "cpu_time_sec": float(cpu_time_sec),
        "wall_clock_sec": float(wall_clock_sec),
        "peak_traced_memory_mb": float(peak_bytes / (1024.0**2)),
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "rss_delta_mb": float(rss_after_mb - rss_before_mb)
        if np.isfinite(rss_before_mb) and np.isfinite(rss_after_mb)
        else float("nan"),
    }


def aligned_misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray, K: int):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    conf = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < K and 0 <= p < K:
            conf[t, p] += 1
    true_ids, pred_ids = linear_sum_assignment(-conf)
    pred_to_true = {int(pred): int(true) for true, pred in zip(true_ids, pred_ids)}
    for c in range(K):
        pred_to_true.setdefault(c, c)
    y_aligned = np.array([pred_to_true.get(int(c), int(c)) for c in y_pred], dtype=int)
    return float(np.mean(y_aligned != y_true)), y_aligned, conf


def hypergraph_vertex_degree_stats(n: int, hyperedges):
    degrees = np.zeros(n, dtype=int)
    for edge in hyperedges:
        degrees[list(edge)] += 1
    return {
        "num_isolated_nodes": int(np.sum(degrees == 0)),
        "isolated_fraction": float(np.mean(degrees == 0)) if n > 0 else 0.0,
        "hypergraph_degree_mean": float(degrees.mean()) if n > 0 else 0.0,
        "hypergraph_degree_max": float(degrees.max()) if n > 0 else 0.0,
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
        "candidate_within_fraction": float(within / total) if total > 0 else float("nan"),
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
):
    timings: dict[str, float] = {}
    n = int(theta.shape[0])
    ell = int(K + r)

    t0 = time.perf_counter()
    omega = rng.standard_normal(size=(n, ell))
    timings["rp_draw_omega_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    Y = omega
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
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[-K:][::-1]
    top_vals = vals[order]
    core_vecs = vecs[:, order]
    timings["rp_small_eig_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    U = Q @ core_vecs
    timings["rp_lift_sec"] = time.perf_counter() - t0
    return top_vals, U, timings


def sample_rescaled_symmetric_sparse_matrix(
    A: sp.csr_matrix,
    p: float,
    rng: np.random.Generator,
):
    if not (0.0 < float(p) <= 1.0):
        raise ValueError(f"random sampling probability must be in (0, 1], got {p}")

    upper = sp.triu(A, k=0, format="coo")
    if upper.nnz == 0:
        return sp.csr_matrix(A.shape, dtype=float), {
            "rs_original_upper_nnz": 0,
            "rs_sampled_upper_nnz": 0,
            "rs_sampling_probability": float(p),
        }

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


def randomized_sampling_embedding(
    theta: sp.csr_matrix,
    K: int,
    p: float,
    rng: np.random.Generator,
    eigsh_tol: float,
):
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    sampled_theta, sample_stats = sample_rescaled_symmetric_sparse_matrix(theta, p=p, rng=rng)
    timings["rs_sample_matrix_wall_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vals, U = top_eigsh_embedding(sampled_theta, K=K, rng=rng, eigsh_tol=eigsh_tol)
    timings["rs_eig_wall_sec"] = time.perf_counter() - t0
    timings.update(sample_stats)
    timings["rs_sampled_theta_nnz"] = int(sampled_theta.nnz)
    return vals, U, timings


def spectral_cluster_from_theta(theta: sp.csr_matrix, K: int, rng: np.random.Generator, spec: SweepSpec):
    theta = ((theta + theta.T) * 0.5).tocsr()
    theta.eliminate_zeros()
    total_start = time.perf_counter()
    timings: dict[str, Any] = {}

    t0 = time.perf_counter()
    if spec.method == "gaussian_random_projection":
        vals, U, extra = gaussian_random_projection_embedding(
            theta=theta,
            K=K,
            r=spec.rp_oversampling,
            q=spec.rp_power_iter,
            rng=rng,
        )
        timings.update(extra)
    elif spec.method == "random_sampling":
        vals, U, extra = randomized_sampling_embedding(
            theta=theta,
            K=K,
            p=spec.random_sampling_p,
            rng=rng,
            eigsh_tol=spec.eigsh_tol,
        )
        timings.update(extra)
    elif spec.method == "non_random":
        vals, U = top_eigsh_embedding(theta=theta, K=K, rng=rng, eigsh_tol=spec.eigsh_tol)
    else:
        raise ValueError(f"unknown method: {spec.method}")
    timings["eigen_decomposition_wall_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if spec.normalize_embedding_rows:
        U = normalize_rows_l2(U)
    timings["embedding_normalize_wall_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    random_state = int(rng.integers(1, 2**31 - 1))
    labels = KMeans(
        n_clusters=K,
        n_init=int(spec.kmeans_n_init),
        random_state=random_state,
    ).fit_predict(U)
    timings["kmeans_wall_sec"] = time.perf_counter() - t0
    timings["spectral_clustering_wall_sec"] = time.perf_counter() - total_start
    timings["top_eigenvalue_max"] = float(np.max(vals)) if len(vals) else float("nan")
    timings["top_eigenvalue_min"] = float(np.min(vals)) if len(vals) else float("nan")
    return labels, timings


def value_to_seed_component(value: int | float) -> int:
    if isinstance(value, float):
        return int(round(value * 1000))
    return int(value)


def concrete_params(spec: SweepSpec, x_value: int | float):
    n = int(x_value) if spec.sweep == "n" else int(spec.n)
    K = int(x_value) if spec.sweep == "K" else int(spec.K)
    rho_n = float(x_value) if spec.sweep == "rho_n" else float(spec.rho_n)
    return n, K, rho_n


def run_one_rep(spec: SweepSpec, x_value: int | float, rep: int):
    n, K, rho_n = concrete_params(spec, x_value)
    seed = int(spec.seed + value_to_seed_component(x_value) * 100_000 + rep)
    rng = np.random.default_rng(seed)

    p_in, p_out = make_uniform_hsbm_probs(
        n=n,
        d=spec.m,
        a_d=spec.a_in,
        b_d=spec.b_out,
        rho_n=rho_n,
        clip=True,
    )

    timings: dict[str, Any] = {}
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
    timings["generation_wall_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    L = hypergraph_laplacian(n=n, hyperedges=hyperedges)
    theta = (sp.eye(n, format="csr", dtype=float) - L).tocsr()
    theta.eliminate_zeros()
    timings["hypergraph_laplacian_build_wall_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred, spectral_stats = spectral_cluster_from_theta(theta=theta, K=K, rng=rng, spec=spec)
    timings["spectral_clustering_wall_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mis, _, _ = aligned_misclassification_rate(y_true, y_pred, K)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    timings["metric_wall_sec"] = time.perf_counter() - t0

    record = {
        spec.x_col: x_value,
        "method": spec.method_label,
        "rep": int(rep),
        "seed": int(seed),
        "n": int(n),
        "K": int(K),
        "m": int(spec.m),
        "rho_n": float(rho_n),
        "num_hyperedges_total": int(len(hyperedges)),
        "misclassification_rate": float(mis),
        "ARI": float(ari),
        "NMI": float(nmi),
        **timings,
        **hypergraph_vertex_degree_stats(n, hyperedges),
        **expected_uniform_hsbm_stats(y_true, K, spec.m, p_in, p_out),
        **spectral_stats,
    }
    record["algorithm_total_wall_sec"] = float(
        record["generation_wall_sec"]
        + record["hypergraph_laplacian_build_wall_sec"]
        + record["eigen_decomposition_wall_sec"]
        + record["embedding_normalize_wall_sec"]
        + record["kmeans_wall_sec"]
    )
    record["p_in"] = float(p_in)
    record["p_out"] = float(p_out)
    record["sampling_mode"] = gen_stats.get("sampling_mode", "")
    return record


def run_one_rep_measured(spec: SweepSpec, x_value: int | float, rep: int):
    record, measurement = measure_call(lambda: run_one_rep(spec=spec, x_value=x_value, rep=rep))
    record.update(measurement)
    return record


def summarize_raw(df_raw: pd.DataFrame, x_col: str) -> pd.DataFrame:
    preferred = [
        "num_hyperedges_total",
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
        "cpu_time_sec",
        "wall_clock_sec",
        "peak_traced_memory_mb",
        "rss_delta_mb",
        "rp_draw_omega_sec",
        "rp_power_iter_sec",
        "rp_qr_sec",
        "rp_build_core_sec",
        "rp_small_eig_sec",
        "rp_lift_sec",
        "rs_sample_matrix_wall_sec",
        "rs_eig_wall_sec",
        "rs_original_upper_nnz",
        "rs_sampled_upper_nnz",
        "rs_sampled_theta_nnz",
        "top_eigenvalue_max",
        "top_eigenvalue_min",
    ]
    aggregations = {"reps": ("rep", "count")}
    for col in preferred:
        if col in df_raw.columns:
            label = "misclassification" if col == "misclassification_rate" else col
            aggregations[f"{label}_mean"] = (col, "mean")
            aggregations[f"{label}_std"] = (col, "std")
    summary = df_raw.groupby([x_col, "method"], as_index=False).agg(**aggregations)
    return summary


def plot_summary(summary: pd.DataFrame, spec: SweepSpec, out_png: Path):
    x = spec.x_col
    d = summary.sort_values(x)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("misclassification_mean", "Misclassification rate"),
        ("ARI_mean", "ARI"),
        ("NMI_mean", "NMI"),
        ("algorithm_total_wall_sec_mean", "Algorithm time (sec)"),
    ]
    title_by_sweep = {
        "K": "Uniform HSBM K Sweep",
        "n": "Uniform HSBM n Scaling",
        "rho_n": "Uniform HSBM rho_n Sweep",
    }
    for ax, (col, ylabel) in zip(axes.ravel(), panels):
        if col not in d.columns:
            ax.axis("off")
            continue
        ax.plot(d[x], d[col], marker="o", linewidth=2, label=spec.method_label)
        std_col = col.replace("_mean", "_std")
        if std_col in d.columns:
            y = d[col].to_numpy(dtype=float)
            err = d[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(d[x].to_numpy(dtype=float), y - err, y + err, alpha=0.18)
        ax.set_xlabel(x)
        ax.set_ylabel(ylabel)
        ax.set_title(spec.method_label)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{title_by_sweep.get(spec.sweep, spec.sweep)} - {spec.method_label}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_spec(spec: SweepSpec, show_progress: bool = True):
    spec.outdir.mkdir(parents=True, exist_ok=True)
    total = len(spec.x_values) * spec.reps
    progress = LiveProgress(total) if show_progress else None
    rows = []

    for x_value in spec.x_values:
        for rep in range(1, spec.reps + 1):
            rows.append(run_one_rep_measured(spec=spec, x_value=x_value, rep=rep))
            if progress is not None:
                progress.update(spec.x_col, x_value, rep, spec.reps, spec.method_label)
    if progress is not None:
        progress.close()

    df_raw = pd.DataFrame(rows)
    summary = summarize_raw(df_raw, spec.x_col)

    raw_path = spec.outdir / f"{spec.file_prefix}_raw.csv"
    summary_path = spec.outdir / f"{spec.file_prefix}_summary.csv"
    config_path = spec.outdir / f"{spec.file_prefix}_config.json"
    plot_path = spec.outdir / f"{spec.file_prefix}_summary.png"

    df_raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    config = asdict(spec)
    config["method_label"] = spec.method_label
    config["outdir"] = str(spec.outdir)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_summary(summary, spec, plot_path)

    return {
        "spec": spec,
        "raw": df_raw,
        "summary": summary,
        "paths": {
            "raw": raw_path,
            "summary": summary_path,
            "config": config_path,
            "plot": plot_path,
        },
    }


def run_named_experiment(sweep: str, method: str, show_progress: bool = True):
    return run_spec(resolve_spec(sweep=sweep, method=method), show_progress=show_progress)


def notebook_source(spec: SweepSpec) -> dict[str, Any]:
    method_desc = {
        "gaussian_random_projection": (
            "가우시안 랜덤 프로젝션은 `Theta`에 Gaussian test matrix를 곱해 낮은 차원의 "
            "부분공간을 만든 뒤, 작은 core matrix의 고유벡터를 원래 공간으로 lift한다."
        ),
        "random_sampling": (
            "랜덤 샘플링은 `Theta`의 sparse nonzero entry를 확률 `p=0.7`로 뽑고 "
            "`1/p`로 rescale한 sampled operator에 `eigsh`를 적용한다."
        ),
    }[spec.method]
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {spec.title_ko}\n",
                "\n",
                f"이 노트북은 균일 HSBM의 `{spec.x_col}` 변화 실험을 `{METHOD_KOREAN[spec.method_label]}` 방식으로 실행한다.\n",
                "\n",
                f"{method_desc}\n",
                "\n",
                "하이퍼그래프 생성식은 기존 비랜덤 실험과 동일하게 `p_in = a_in * rho_n / n ** (m - 1)`, "
                "`p_out = b_out * rho_n / n ** (m - 1)`를 사용한다.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 실행 설정\n",
                "\n",
                "공통 실행 코드는 `uniform_hsbm_randomized.py`에 두었다. 이 노트북은 해당 실행기를 호출하고, "
                "raw CSV, summary CSV, 설정 JSON, 요약 그림을 `results/` 아래에 저장한다.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import sys\n",
                "import pandas as pd\n",
                "\n",
                "PROJECT_ROOT = Path.cwd()\n",
                "for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:\n",
                "    if (candidate / \"src\" / \"common.py\").exists():\n",
                "        PROJECT_ROOT = candidate\n",
                "        break\n",
                "\n",
                "EXPERIMENT_DIR = PROJECT_ROOT / \"experiments\" / \"균일 HSBM 실험\"\n",
                "if str(EXPERIMENT_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(EXPERIMENT_DIR))\n",
                "\n",
                "from uniform_hsbm_randomized import resolve_spec, run_named_experiment\n",
                "\n",
                "pd.set_option(\"display.max_columns\", 200)\n",
                "pd.set_option(\"display.width\", 220)\n",
                f"SPEC = resolve_spec(sweep={spec.sweep!r}, method={spec.method!r})\n",
                "SPEC\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 실험 실행\n",
                "\n",
                "아래 셀을 실행하면 모든 반복을 수행하고 결과 파일을 저장한다.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"outputs = run_named_experiment(sweep={spec.sweep!r}, method={spec.method!r}, show_progress=True)\n",
                "summary = outputs[\"summary\"]\n",
                "summary\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 저장된 파일\n",
                "\n",
                "실험이 끝나면 아래 경로에 raw 결과, summary 결과, 설정 파일, 요약 그림이 저장된다.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "outputs[\"paths\"]\n",
            ],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "base",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_randomized_notebooks():
    written = []
    for spec in get_randomized_specs().values():
        path = EXPERIMENT_DIR / spec.notebook_name
        path.write_text(json.dumps(notebook_source(spec), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        written.append(path)
    return written


def translate_existing_notebook_markdown():
    replacements = {
        "K변화.ipynb": {
            0: [
                "# 균일 HSBM K 변화 - hypergraph theta 비랜덤 실험\n",
                "\n",
                "이 노트북은 `n=5000`, `rho_n=8`을 고정하고 군집 수 `K`만 바꾸는 실험을 실행한다.\n",
                "\n",
                "균일 HSBM은 `p_in = a_in * rho_n / n ** (m - 1)`, "
                "`p_out = b_out * rho_n / n ** (m - 1)`를 사용한다.\n",
                "\n",
                "Spectral clustering은 normalized hypergraph operator `Theta = I - Delta`의 가장 큰 고유값에 대응하는 고유벡터를 사용한다. "
                "이는 `Delta = I - Theta`의 가장 작은 고유값 고유벡터를 쓰는 것과 같은 eigenspace를 사용한다.\n",
                "\n",
                "`a_in`, `b_out`, `m`, `n`, `rho_n`을 고정한 채 `K`를 바꾸면 within-community 후보 비율도 함께 변한다. "
                "그래서 이 노트북은 경험적 degree와 기대 degree 진단값을 함께 저장한다.\n",
            ],
            2: ["## 설정\n", "\n", "`K`만 바꾼다. `n=5000`, `rho_n=8.0`은 고정한다.\n"],
            4: ["## 보조 함수\n"],
            6: ["## K 변화 실험 실행\n"],
            8: ["## 결과 저장\n"],
            10: ["## 그림\n"],
        },
        "n변화.ipynb": {
            0: [
                "# 균일 HSBM n 변화 - hypergraph theta 비랜덤 실험\n",
                "\n",
                "이 노트북은 균일 HSBM 하이퍼그래프를 생성하고, normalized hypergraph Laplacian에서 만든 "
                "`Theta = I - Delta`로 spectral clustering을 수행한다.\n",
                "\n",
                "`n`이 커질 때 오분류율, ARI, NMI, CPU 시간, wall-clock 시간, 메모리, 알고리즘 단계별 시간을 기록한다.\n",
            ],
            2: [
                "## 설정\n",
                "\n",
                "`a_in`, `b_out`, `rho_n`은 `p_in = a_in * rho_n / n ** (m - 1)`, "
                "`p_out = b_out * rho_n / n ** (m - 1)`로 변환한다.\n",
            ],
            4: ["## 보조 함수\n"],
            6: ["## n = 1000\n"],
            8: ["## n = 2000\n"],
            10: ["## n = 3000\n"],
            12: ["## n = 4000\n"],
            14: ["## n = 5000\n"],
            16: ["## n = 6000\n"],
            18: ["## n = 7000\n"],
            20: ["## n = 8000\n"],
            22: ["## n = 9000\n"],
            24: ["## n = 10000\n"],
            26: ["## 결과 결합, 저장, 그림 생성\n"],
            30: ["## 더 큰 n을 추가하는 선택 셀\n", "\n", "sweep을 확장할 때만 아래 셀을 수정해서 사용한다.\n"],
        },
        "rho_n변화.ipynb": {
            0: [
                "# 균일 HSBM rho_n 변화 - hypergraph theta 비랜덤 실험\n",
                "\n",
                "이 노트북은 균일 HSBM의 다른 하이퍼파라미터를 고정하고 `rho_n`만 바꾸는 실험을 실행한다.\n",
                "\n",
                "각 `rho_n`에서 `p_in = a_in * rho_n / n ** (m - 1)`, "
                "`p_out = b_out * rho_n / n ** (m - 1)`를 사용한다.\n",
                "\n",
                "Spectral clustering은 normalized hypergraph operator `Theta = I - Delta`의 가장 큰 고유값에 대응하는 고유벡터를 사용한다.\n",
            ],
            2: ["## 설정\n", "\n", "`rho_n`만 바꾸고 나머지 모델 및 알고리즘 파라미터는 고정한다.\n"],
            4: ["## 보조 함수\n"],
            6: ["## rho_n 변화 실험 실행\n"],
            8: ["## 결과 저장\n"],
            10: ["## 그림\n"],
        },
    }
    for notebook_name, cell_map in replacements.items():
        path = EXPERIMENT_DIR / notebook_name
        nb = json.loads(path.read_text(encoding="utf-8"))
        for idx, source in cell_map.items():
            nb["cells"][idx]["source"] = source
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


BASE_RESULTS = [
    {
        "sweep": "n",
        "method": "Non-random",
        "x_col": "n",
        "summary": RESULTS_ROOT
        / "EXP-20260426-004_uniform_hsbm_n_scaling_zhou_laplacian"
        / "EXP-20260426-004_uniform_hsbm_n_scaling_zhou_laplacian_summary.csv",
        "plot": RESULTS_ROOT
        / "EXP-20260426-004_uniform_hsbm_n_scaling_zhou_laplacian"
        / "EXP-20260426-004_uniform_hsbm_n_scaling_zhou_laplacian_summary.png",
    },
    {
        "sweep": "K",
        "method": "Non-random",
        "x_col": "K",
        "summary": RESULTS_ROOT
        / "EXP-20260427-002_uniform_hsbm_K_sweep_zhou_theta"
        / "EXP-20260427-002_uniform_hsbm_K_sweep_zhou_theta_summary.csv",
        "plot": RESULTS_ROOT
        / "EXP-20260427-002_uniform_hsbm_K_sweep_zhou_theta"
        / "EXP-20260427-002_uniform_hsbm_K_sweep_zhou_theta_summary.png",
    },
    {
        "sweep": "rho_n",
        "method": "Non-random",
        "x_col": "rho_n",
        "summary": RESULTS_ROOT
        / "EXP-20260427-001_uniform_hsbm_rho_n_sweep_zhou_theta"
        / "EXP-20260427-001_uniform_hsbm_rho_n_sweep_zhou_theta_summary.csv",
        "plot": RESULTS_ROOT
        / "EXP-20260427-001_uniform_hsbm_rho_n_sweep_zhou_theta"
        / "EXP-20260427-001_uniform_hsbm_rho_n_sweep_zhou_theta_summary.png",
    },
]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _format_value(v: Any) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if abs(float(v)) >= 1000:
            return f"{float(v):.1f}"
        return f"{float(v):.4f}"
    return str(v)


def dataframe_to_markdown(df: pd.DataFrame, bold_rows: np.ndarray | list[bool] | None = None) -> str:
    if df.empty:
        return "_결과가 없습니다._"
    headers = list(df.columns)
    rows = [[_format_value(v) for v in row] for row in df.to_numpy()]
    if bold_rows is None:
        bold_rows = [False] * len(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row, is_bold in zip(rows, bold_rows):
        if bool(is_bold):
            row = [f"**{cell}**" if cell else cell for cell in row]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def grouped_comparison_markdown(
    part: pd.DataFrame,
    sweep: str,
    sweep_label: str,
    metric_label: str = "오분류율",
) -> str:
    sections: list[str] = []
    for x_value, group in part.groupby("x", sort=True):
        group = group.sort_values("method")
        best_rows = np.isclose(
            group["misclassification"].to_numpy(dtype=float),
            group["misclassification"].min(),
        )
        display = group[
            [
                "method",
                "misclassification",
                "ARI",
                "NMI",
                "algorithm_sec",
                "spectral_sec",
                "hyperedges",
                "degree",
            ]
        ].rename(
            columns={
                "method": "방법",
                "misclassification": metric_label,
                "algorithm_sec": "주요시간초",
                "spectral_sec": "spectral초",
                "hyperedges": "하이퍼엣지수",
                "degree": "평균degree",
            }
        )
        sections.extend(
            [
                f"### {sweep_label} = {_format_value(x_value)}\n",
                "\n",
                dataframe_to_markdown(display, bold_rows=best_rows),
                "\n\n",
            ]
        )
    return "".join(sections)


def load_report_rows():
    entries = list(BASE_RESULTS)
    for spec in get_randomized_specs().values():
        entries.append(
            {
                "sweep": spec.sweep,
                "method": spec.method_label,
                "x_col": spec.x_col,
                "summary": spec.outdir / f"{spec.file_prefix}_summary.csv",
                "plot": spec.outdir / f"{spec.file_prefix}_summary.png",
            }
        )

    rows = []
    missing = []
    for entry in entries:
        path = Path(entry["summary"])
        if not path.exists():
            missing.append(entry)
            continue
        df = pd.read_csv(path)
        x_col = entry["x_col"]
        if "method" not in df.columns:
            df["method"] = entry["method"]
        method = entry["method"]
        mis_col = _find_col(df, ["misclassification_mean", "misclassification_rate_mean"])
        ari_col = _find_col(df, ["ARI_mean", "ari_mean"])
        nmi_col = _find_col(df, ["NMI_mean", "nmi_mean"])
        runtime_col = _find_col(df, ["algorithm_total_wall_sec_mean", "wall_clock_sec_mean"])
        spectral_col = _find_col(df, ["spectral_clustering_wall_sec_mean"])
        hyper_col = _find_col(df, ["hyperedges_mean", "num_hyperedges_total_mean"])
        degree_col = _find_col(df, ["degree_mean", "hypergraph_degree_mean_mean"])
        for _, row in df.iterrows():
            hyperedges = row[hyper_col] if hyper_col else np.nan
            if degree_col:
                degree = row[degree_col]
            elif x_col == "n" and pd.notna(hyperedges):
                degree = 3.0 * float(hyperedges) / float(row[x_col])
            else:
                degree = np.nan
            rows.append(
                {
                    "sweep": entry["sweep"],
                    "x": row[x_col],
                    "method": method,
                    "misclassification": row[mis_col] if mis_col else np.nan,
                    "ARI": row[ari_col] if ari_col else np.nan,
                    "NMI": row[nmi_col] if nmi_col else np.nan,
                    "algorithm_sec": row[runtime_col] if runtime_col else np.nan,
                    "spectral_sec": row[spectral_col] if spectral_col else np.nan,
                    "hyperedges": hyperedges,
                    "degree": degree,
                    "plot": entry["plot"],
                }
            )
    return pd.DataFrame(rows), missing


def write_combined_report(path: Path | None = None):
    path = path or (EXPERIMENT_DIR / "결과보고서.md")
    df, missing = load_report_rows()

    lines = [
        "# 균일 HSBM 실험 결과보고서\n",
        "\n",
        "이 보고서는 `균일 HSBM 실험` 폴더의 비랜덤 spectral clustering, 가우시안 랜덤 프로젝션, "
        "랜덤 샘플링 실험을 한곳에 모아 정리한 것입니다. 모든 실험은 normalized hypergraph operator "
        "`Theta = I - Delta`에서 spectral embedding을 만든 뒤 k-means를 수행합니다.\n",
        "\n",
        "## 실험 구성\n",
        "\n",
        "- `n변화`: `K=3`, `rho_n=4.0`을 고정하고 `n`을 1000부터 10000까지 바꿉니다.\n",
        "- `K변화`: `n=5000`, `rho_n=8.0`을 고정하고 `K`를 바꿉니다.\n",
        "- `rho_n변화`: `n=5000`, `K=3`을 고정하고 `rho_n`을 바꿉니다.\n",
        "- `Non-random`: `Theta`에 대해 `eigsh`로 top-`K` 고유벡터를 직접 계산합니다.\n",
        "- `Gaussian random projection`: Gaussian test matrix와 power iteration으로 작은 core matrix를 만든 뒤 고유벡터를 lift합니다.\n",
        "- `Random sampling`: `Theta`의 sparse nonzero entry를 확률 `p=0.7`로 샘플링하고 `1/p`로 rescale한 뒤 `eigsh`를 적용합니다.\n",
        "- 세부 표는 같은 `n`, `K`, `rho_n` 값끼리 작은 표로 묶었습니다. 볼드 처리된 행은 해당 묶음 안에서 오분류율이 가장 낮은 결과입니다. 동률이면 여러 행을 함께 표시합니다.\n",
        "\n",
    ]

    if missing:
        lines.extend(["## 아직 없는 결과\n", "\n"])
        for item in missing:
            lines.append(f"- `{item['sweep']}` / `{item['method']}`: `{item['summary']}`\n")
        lines.append("\n")

    if df.empty:
        lines.append("아직 읽을 수 있는 summary CSV가 없습니다.\n")
        path.write_text("".join(lines), encoding="utf-8")
        return path

    overview = (
        df.groupby(["sweep", "method"], as_index=False)
        .agg(
            평균_오분류율=("misclassification", "mean"),
            평균_ARI=("ARI", "mean"),
            평균_NMI=("NMI", "mean"),
            평균_주요시간초=("algorithm_sec", "mean"),
            평균_spectral초=("spectral_sec", "mean"),
        )
        .sort_values(["sweep", "method"])
    )
    overview_best = np.isclose(
        overview["평균_오분류율"].to_numpy(dtype=float),
        overview.groupby("sweep")["평균_오분류율"].transform("min").to_numpy(dtype=float),
    )
    lines.extend(["## 전체 요약\n", "\n", dataframe_to_markdown(overview, bold_rows=overview_best), "\n\n"])

    sweep_titles = {
        "n": "n 변화 실험",
        "K": "K 변화 실험",
        "rho_n": "rho_n 변화 실험",
    }
    sweep_labels = {
        "n": "n",
        "K": "K",
        "rho_n": "rho_n",
    }
    for sweep in ["n", "K", "rho_n"]:
        part = df[df["sweep"] == sweep].copy()
        if part.empty:
            continue
        part = part.sort_values(["x", "method"])
        lines.extend(
            [
                f"## {sweep_titles[sweep]}\n",
                "\n",
                grouped_comparison_markdown(part, sweep=sweep, sweep_label=sweep_labels[sweep]),
            ]
        )

        plots = []
        for _, row in part.drop_duplicates("plot").iterrows():
            plot_path = Path(row["plot"])
            if plot_path.exists():
                rel = plot_path.relative_to(EXPERIMENT_DIR)
                plots.append(f"![{sweep_titles[sweep]} {row['method']}]({rel.as_posix()})")
        if plots:
            lines.extend(["### 그림\n", "\n", "\n\n".join(plots), "\n\n"])

    lines.extend(
        [
            "## 가우시안 랜덤 프로젝션 오분류율 해석\n",
            "\n",
            "초기 설정인 `rp_oversampling=10`, `rp_power_iter=2`에서는 가우시안 랜덤 프로젝션의 오분류율이 높았습니다. "
            "원인은 HSBM 생성 모델 자체가 아니라, 해당 설정이 `Theta`의 top-`K` 고유공간을 충분히 잘 근사하지 못한 데 있었습니다. "
            "진단용으로 `n=5000`, `K=3`, `rho_n=4.0`인 같은 인스턴스에서 비교했을 때, non-random `eigsh`의 top eigenvalues는 "
            "대략 `[1.0, 0.699, 0.698]`였지만 초기 RP 설정은 작은 core matrix에서 세 번째 고유값을 약 `0.54~0.56` 수준으로 낮게 잡았습니다. "
            "이 상태에서는 non-random eigenspace와의 principal angle도 크게 남아 row-normalization과 k-means 이후 community 방향이 섞였습니다.\n",
            "\n",
            "그래서 가우시안 랜덤 프로젝션 실험은 `rp_oversampling=160`, `rp_power_iter=4`로 올려 다시 실행했습니다. "
            "그 결과 `n` 변화 실험의 평균 오분류율은 크게 개선되어 non-random 결과에 가까워졌습니다. `rho_n` 변화 실험에서도 `rho_n >= 4`인 충분히 조밀한 구간에서는 거의 non-random에 가까운 결과가 나왔습니다. "
            "다만 `rho_n <= 1`처럼 그래프가 매우 희소한 구간은 non-random도 성능이 낮기 때문에 RP만으로 해결되는 문제가 아니며, `K` 변화 실험의 큰 `K` 구간도 within-community 후보 비율과 평균 degree가 함께 낮아져 세 방법 모두 어려운 구간입니다.\n",
            "\n",
            "정리하면, 가우시안 랜덤 프로젝션은 `r`과 `q`를 충분히 키우면 정확도는 회복됩니다. 하지만 그만큼 `Theta @ Y`, QR, core matrix 구성 비용이 증가하므로 현재 재실행 결과에서는 정확도 개선의 대가로 runtime이 늘어난 상태입니다.\n",
            "\n",
            "## 해석 메모\n",
            "\n",
            "- 오분류율은 Hungarian matching으로 예측 label을 true label에 맞춘 뒤 계산했습니다.\n",
            "- ARI와 NMI는 label permutation에 불변이므로 원 label을 그대로 사용했습니다.\n",
            "- 랜덤화 방법은 반복마다 같은 HSBM 생성 규칙을 사용하지만, spectral 단계에서 추가 난수를 사용합니다.\n",
            "- `algorithm_sec`는 생성, hypergraph Laplacian/operator 구성, spectral embedding, row normalization, k-means 주요 단계의 합입니다.\n",
            "- 큰 `n`에서는 하이퍼그래프 생성 시간이 전체 시간을 지배할 수 있으므로, spectral 단계 시간도 함께 보시는 것이 좋습니다.\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["K", "n", "rho_n"])
    parser.add_argument("--method", choices=["gaussian_random_projection", "random_sampling"])
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--write-notebooks", action="store_true")
    parser.add_argument("--translate-markdown", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)

    if args.write_notebooks:
        for path in write_randomized_notebooks():
            print(path)
    if args.translate_markdown:
        translate_existing_notebook_markdown()
        print("translated existing notebook markdown")
    if args.sweep and args.method:
        outputs = run_named_experiment(
            sweep=args.sweep,
            method=args.method,
            show_progress=not args.no_progress,
        )
        print(outputs["summary"])
        print(outputs["paths"])
    if args.write_report:
        print(write_combined_report())


if __name__ == "__main__":
    main()
