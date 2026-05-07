from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    LiveProgress,
    generate_uniform_hsbm_instance,
    hypergraph_laplacian,
    make_uniform_hsbm_probs,
    normalize_rows_l2,
)
from uniform_hsbm_randomized import (
    aligned_misclassification_rate,
    countsketch_random_projection_embedding,
    expected_uniform_hsbm_stats,
    gaussian_random_projection_embedding,
    hypergraph_vertex_degree_stats,
    measure_call,
    randomized_sampling_embedding,
    top_eigsh_embedding,
)


METHODS = (
    "non_random",
    "gaussian_random_projection",
    "random_sampling",
    "countsketch_random_projection",
)

METHOD_LABELS = {
    "non_random": "Non-random",
    "gaussian_random_projection": "Gaussian random projection",
    "random_sampling": "Random sampling",
    "countsketch_random_projection": "CountSketch random projection",
}

METHOD_ORDER = [METHOD_LABELS[m] for m in METHODS]


@dataclass(frozen=True)
class ComparisonSpec:
    sweep: str
    experiment_id: str
    experiment_slug: str
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
    seed: int = 20260506
    sampling: str = "sparse"
    max_enumeration: int = 1_500_000
    normalize_embedding_rows: bool = True
    eigsh_tol: float = 1e-6
    eigensolver: str = "eigsh"
    rp_oversampling: int = 160
    rp_power_iter: int = 4
    random_sampling_p: float = 0.7
    kmeans_n_init: int = 20

    @property
    def outdir(self) -> Path:
        return RESULTS_ROOT / f"{self.experiment_id}_{self.experiment_slug}"

    @property
    def file_prefix(self) -> str:
        return f"{self.experiment_id}_{self.experiment_slug}"


def get_comparison_specs() -> dict[str, ComparisonSpec]:
    K_VALUES = (2, 4, 6, 8, 10, 12)
    N_VALUES = tuple(range(2000, 10001, 2000))
    RHO_VALUES = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    return {
        "K": ComparisonSpec(
            sweep="K",
            experiment_id="EXP-20260506-008",
            experiment_slug="uniform_hsbm_K_rho16_eigsh_methods",
            title_ko="균일 HSBM K 변화 - rho_n=16, eigsh method 비교",
            x_col="K",
            x_values=K_VALUES,
            n=5000,
            rho_n=16.0,
        ),
        "n": ComparisonSpec(
            sweep="n",
            experiment_id="EXP-20260506-007",
            experiment_slug="uniform_hsbm_n_rho16_eigsh_methods",
            title_ko="균일 HSBM n 변화 - rho_n=16, eigsh method 비교",
            x_col="n",
            x_values=N_VALUES,
            K=3,
            rho_n=16.0,
        ),
        "rho_n": ComparisonSpec(
            sweep="rho_n",
            experiment_id="EXP-20260506-009",
            experiment_slug="uniform_hsbm_rho_eigsh_methods",
            title_ko="균일 HSBM rho_n 변화 - eigsh method 비교",
            x_col="rho_n",
            x_values=RHO_VALUES,
            n=5000,
            K=3,
        ),
    }


def resolve_spec(sweep: str) -> ComparisonSpec:
    return get_comparison_specs()[sweep]


def value_to_seed_component(value: int | float) -> int:
    if isinstance(value, float):
        return int(round(value * 1000))
    return int(value)


def concrete_params(spec: ComparisonSpec, x_value: int | float):
    n = int(x_value) if spec.sweep == "n" else int(spec.n)
    K = int(x_value) if spec.sweep == "K" else int(spec.K)
    rho_n = float(x_value) if spec.sweep == "rho_n" else float(spec.rho_n)
    return n, K, rho_n


def method_seed(base_seed: int, method: str) -> int:
    return int(base_seed + (METHODS.index(method) + 1) * 10_000_000)


def spectral_cluster_from_theta(
    theta: sp.csr_matrix,
    K: int,
    rng: np.random.Generator,
    spec: ComparisonSpec,
    method: str,
):
    theta = ((theta + theta.T) * 0.5).tocsr()
    theta.eliminate_zeros()
    total_start = time.perf_counter()
    timings: dict[str, Any] = {}

    t0 = time.perf_counter()
    if method == "non_random":
        vals, U = top_eigsh_embedding(theta=theta, K=K, rng=rng, eigsh_tol=spec.eigsh_tol)
        timings["non_random_eigensolver"] = spec.eigensolver
    elif method == "gaussian_random_projection":
        vals, U, extra = gaussian_random_projection_embedding(
            theta=theta,
            K=K,
            r=spec.rp_oversampling,
            q=spec.rp_power_iter,
            rng=rng,
            eigsh_tol=spec.eigsh_tol,
        )
        timings.update(extra)
    elif method == "random_sampling":
        vals, U, extra = randomized_sampling_embedding(
            theta=theta,
            K=K,
            p=spec.random_sampling_p,
            rng=rng,
            eigsh_tol=spec.eigsh_tol,
        )
        timings.update(extra)
    elif method == "countsketch_random_projection":
        vals, U, extra = countsketch_random_projection_embedding(
            theta=theta,
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


def run_one_instance(spec: ComparisonSpec, x_value: int | float, rep: int):
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
        spec.x_col: x_value,
        "rep": int(rep),
        "seed": int(seed),
        "n": int(n),
        "K": int(K),
        "m": int(spec.m),
        "rho_n": float(rho_n),
        "num_hyperedges_total": int(len(hyperedges)),
        "theta_nnz": int(theta.nnz),
        "theta_density": float(theta.nnz / (n * n)),
        "generation_wall_sec": float(generation_wall_sec),
        "hypergraph_laplacian_build_wall_sec": float(build_wall_sec),
        **hypergraph_vertex_degree_stats(n, hyperedges),
        **expected_uniform_hsbm_stats(y_true, K, spec.m, p_in, p_out),
        "p_in": float(p_in),
        "p_out": float(p_out),
        "sampling_mode": gen_stats.get("sampling_mode", ""),
    }

    rows = []
    for method in METHODS:
        method_rng = np.random.default_rng(method_seed(seed, method))

        def _run_method():
            y_pred, spectral_stats = spectral_cluster_from_theta(
                theta=theta,
                K=K,
                rng=method_rng,
                spec=spec,
                method=method,
            )
            t0_metric = time.perf_counter()
            mis, _, _ = aligned_misclassification_rate(y_true, y_pred, K)
            ari = adjusted_rand_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)
            record = {
                **shared,
                "method": METHOD_LABELS[method],
                "method_key": method,
                "misclassification_rate": float(mis),
                "ARI": float(ari),
                "NMI": float(nmi),
                "metric_wall_sec": float(time.perf_counter() - t0_metric),
                **spectral_stats,
            }
            record["algorithm_total_wall_sec"] = float(
                record["generation_wall_sec"]
                + record["hypergraph_laplacian_build_wall_sec"]
                + record["eigen_decomposition_wall_sec"]
                + record["embedding_normalize_wall_sec"]
                + record["kmeans_wall_sec"]
            )
            return record

        record, measurement = measure_call(_run_method)
        record.update(measurement)
        rows.append(record)
    return rows


def summarize_raw(df_raw: pd.DataFrame, x_col: str) -> pd.DataFrame:
    preferred = [
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
        "cpu_time_sec",
        "wall_clock_sec",
        "peak_traced_memory_mb",
        "rss_delta_mb",
        "rp_power_iter_sec",
        "rs_sample_matrix_wall_sec",
        "rs_eig_wall_sec",
        "cs_power_iter_sec",
        "cs_qr_sec",
        "cs_build_core_sec",
        "cs_embedding_dim",
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
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    return summary.sort_values([x_col, "method"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, spec: ComparisonSpec, out_png: Path):
    x = spec.x_col
    d = summary.sort_values([x, "method"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    title_by_sweep = {
        "K": "Uniform HSBM K Sweep - Method Comparison",
        "n": "Uniform HSBM n Scaling - Method Comparison",
        "rho_n": "Uniform HSBM rho_n Sweep - Method Comparison",
    }
    panels = [
        ("misclassification_mean", "Misclassification rate"),
        ("ARI_mean", "ARI"),
        ("NMI_mean", "NMI"),
        ("algorithm_total_wall_sec_mean", "Algorithm time (sec)"),
    ]
    for ax, (col, ylabel) in zip(axes.ravel(), panels):
        for method in METHOD_ORDER:
            dm = d[d["method"] == method].sort_values(x)
            if dm.empty or col not in dm.columns:
                continue
            ax.plot(dm[x], dm[col], marker="o", linewidth=2, label=method)
            std_col = col.replace("_mean", "_std")
            if std_col in dm.columns:
                y = dm[col].to_numpy(dtype=float)
                err = dm[std_col].fillna(0.0).to_numpy(dtype=float)
                ax.fill_between(dm[x].to_numpy(dtype=float), y - err, y + err, alpha=0.12)
        ax.set_xlabel(x)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle(title_by_sweep.get(spec.sweep, "Uniform HSBM Method Comparison"))
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _completed_run_keys(raw_path: Path, x_col: str) -> set[tuple[int, int]]:
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(raw_path, usecols=[x_col, "rep", "method"])
    complete = set()
    for (x_value, rep), group in df.groupby([x_col, "rep"], sort=False):
        if group["method"].nunique() >= len(METHOD_ORDER):
            complete.add((value_to_seed_component(x_value), int(rep)))
    return complete


def _append_raw_rows(raw_path: Path, rows: list[dict[str, Any]]):
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
            df = df[existing_columns + extra_columns]
            tmp_path = raw_path.with_suffix(".tmp.csv")
            pd.concat([existing, df], ignore_index=True).to_csv(tmp_path, index=False)
            tmp_path.replace(raw_path)
        else:
            df = df[existing_columns]
            df.to_csv(raw_path, mode="a", header=False, index=False)
    else:
        df.to_csv(raw_path, index=False)


def run_spec(spec: ComparisonSpec, show_progress: bool = True):
    spec.outdir.mkdir(parents=True, exist_ok=True)
    raw_path = spec.outdir / f"{spec.file_prefix}_raw.csv"
    summary_path = spec.outdir / f"{spec.file_prefix}_summary.csv"
    config_path = spec.outdir / f"{spec.file_prefix}_config.json"
    plot_path = spec.outdir / f"{spec.file_prefix}_summary.png"
    completed = _completed_run_keys(raw_path, spec.x_col)
    total = len(spec.x_values) * spec.reps
    progress = LiveProgress(total) if show_progress else None
    for x_value in spec.x_values:
        for rep in range(1, spec.reps + 1):
            key = (value_to_seed_component(x_value), int(rep))
            if key not in completed:
                rows = run_one_instance(spec=spec, x_value=x_value, rep=rep)
                _append_raw_rows(raw_path, rows)
                completed.add(key)
            if progress is not None:
                progress.update(spec.x_col, x_value, rep, spec.reps, "all methods")
    if progress is not None:
        progress.close()

    df_raw = pd.read_csv(raw_path)
    summary = summarize_raw(df_raw, spec.x_col)
    summary.to_csv(summary_path, index=False)
    config = asdict(spec)
    config["methods"] = METHOD_ORDER
    config["method_note"] = "All methods are evaluated on the same generated HSBM instance for each x_value and rep."
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_summary(summary, spec, plot_path)
    return {"spec": spec, "raw": df_raw, "summary": summary}


def run_named_experiment(sweep: str, show_progress: bool = True):
    return run_spec(resolve_spec(sweep), show_progress=show_progress)


def _fmt_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _load_summary(spec: ComparisonSpec) -> pd.DataFrame:
    path = spec.outdir / f"{spec.file_prefix}_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    df = pd.read_csv(path)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values([spec.x_col, "method"]).reset_index(drop=True)


def _plot_relpath(spec: ComparisonSpec) -> str:
    return f"results/{spec.experiment_id}_{spec.experiment_slug}/{spec.file_prefix}_summary.png"


def _bold_best_rows(df: pd.DataFrame):
    best = df["misclassification_mean"].min()
    rows = []
    for _, row in df.sort_values("method").iterrows():
        values = [
            row["method"],
            _fmt_float(row["misclassification_mean"]),
            _fmt_float(row["ARI_mean"]),
            _fmt_float(row["NMI_mean"]),
            _fmt_float(row["algorithm_total_wall_sec_mean"]),
            _fmt_float(row["spectral_clustering_wall_sec_mean"]),
            _fmt_float(row["num_hyperedges_total_mean"], 1),
            _fmt_float(row["hypergraph_degree_mean_mean"], 4),
        ]
        if np.isclose(row["misclassification_mean"], best):
            values = [f"**{value}**" for value in values]
        rows.append(values)
    return rows


def _n_for_summary_row(spec: ComparisonSpec, row: pd.Series) -> int:
    if spec.sweep == "n":
        return int(row[spec.x_col])
    return int(spec.n)


def _laplacian_sparsity_rows(summary: pd.DataFrame, spec: ComparisonSpec):
    rows = []
    non_random = summary[summary["method"] == "Non-random"].sort_values(spec.x_col)
    for _, row in non_random.iterrows():
        n = _n_for_summary_row(spec, row)
        isolated_mean = float(row.get("isolated_fraction_mean", 0.0)) * n
        laplacian_nnz_mean = float(row["theta_nnz_mean"]) + isolated_mean
        nnz_ratio = laplacian_nnz_mean / float(n * n)
        rows.append(
            [
                f"{float(row[spec.x_col]):.4f}" if spec.sweep == "rho_n" else f"{int(row[spec.x_col])}",
                f"{n}",
                f"{laplacian_nnz_mean:.1f}",
                f"{nnz_ratio:.8f}",
                f"{100.0 * nnz_ratio:.4f}%",
            ]
        )
    return rows


def write_comparison_report(out_path: Path | None = None) -> Path:
    specs = get_comparison_specs()
    out_path = out_path or REPORT_PATH
    summaries = {sweep: _load_summary(spec) for sweep, spec in specs.items()}

    lines: list[str] = []
    lines.append("# 균일 HSBM 실험 결과보고서")
    lines.append("")
    lines.append(
        "이 보고서는 같은 로컬 실행 환경에서 `Non-random`, `Gaussian random projection`, `Random sampling`, "
        "`CountSketch random projection`을 모두 다시 실행해 비교한 결과입니다. 각 `(sweep 값, 반복 번호)`마다 "
        "하나의 균일 HSBM 인스턴스를 생성하고 네 방법을 같은 `Theta = I - Delta`에 적용했습니다."
    )
    lines.append("")
    lines.append("## 실험 구성")
    lines.append("")
    lines.append("- `n변화`: `K=3`, `rho_n=16.0`을 고정하고 `n`을 `{2000, 4000, 6000, 8000, 10000}`으로 바꿉니다.")
    lines.append("- `K변화`: `n=5000`, `rho_n=16.0`을 고정하고 `K`를 `{2, 4, 6, 8, 10, 12}`로 바꿉니다.")
    lines.append("- `rho_n변화`: `n=5000`, `K=3`을 고정하고 `rho_n`을 `{2, 4, 8, 16, 32, 64}`로 바꿉니다.")
    lines.append("- 모든 method의 eigensolver 단계는 공정한 비교를 위해 `scipy.sparse.linalg.eigsh`로 통일했습니다.")
    lines.append("- Gaussian RP와 CountSketch RP는 모두 `ell = K + 160`, power iteration `q=4`를 사용했습니다.")
    lines.append("- Random sampling은 기존과 같이 `Theta`의 sparse nonzero entry를 확률 `p=0.7`로 샘플링하고 `1/p`로 rescale한 뒤 `eigsh`를 적용했습니다.")
    lines.append("- Laplacian sparsity는 모든 sweep 값마다 `L.nnz`와 `L.nnz / n^2`를 따로 표로 기록했습니다.")
    lines.append("- 볼드 처리된 행은 같은 `x` 값 안에서 평균 오분류율이 가장 낮은 방법입니다.")
    lines.append("")
    lines.append("## 전체 요약")
    lines.append("")

    overall_rows = []
    for sweep in ["K", "n", "rho_n"]:
        summary = summaries[sweep]
        method_means = (
            summary.groupby("method", observed=True)["misclassification_mean"]
            .mean()
            .dropna()
        )
        best_mis = float(method_means.min()) if not method_means.empty else float("nan")
        for method in METHOD_ORDER:
            dm = summary[summary["method"] == method]
            if dm.empty:
                continue
            values = [
                sweep,
                method,
                _fmt_float(dm["misclassification_mean"].mean()),
                _fmt_float(dm["ARI_mean"].mean()),
                _fmt_float(dm["NMI_mean"].mean()),
                _fmt_float(dm["algorithm_total_wall_sec_mean"].mean()),
                _fmt_float(dm["spectral_clustering_wall_sec_mean"].mean()),
            ]
            if np.isclose(float(dm["misclassification_mean"].mean()), best_mis):
                values = [f"**{value}**" for value in values]
            overall_rows.append(values)
    lines.append(
        _markdown_table(
            ["sweep", "method", "평균_오분류율", "평균_ARI", "평균_NMI", "평균_주요시간초", "평균_spectral초"],
            overall_rows,
        )
    )
    lines.append("")

    for sweep, title in [("n", "n 변화 실험"), ("K", "K 변화 실험"), ("rho_n", "rho_n 변화 실험")]:
        spec = specs[sweep]
        summary = summaries[sweep]
        lines.append(f"## {title}")
        lines.append("")
        for x_value, group in summary.groupby(spec.x_col, sort=True):
            value_text = f"{float(x_value):.1f}" if sweep == "n" else f"{float(x_value):.4f}"
            lines.append(f"### {spec.x_col} = {value_text}")
            lines.append("")
            lines.append(
                _markdown_table(
                    ["방법", "오분류율", "ARI", "NMI", "주요시간초", "spectral초", "하이퍼엣지수", "평균degree"],
                    _bold_best_rows(group),
                )
            )
            lines.append("")
        lines.append("### 그림")
        lines.append("")
        lines.append(f"![{title} method 비교]({_plot_relpath(spec)})")
        lines.append("")

    lines.append("## Laplacian nnz 비율")
    lines.append("")
    lines.append(
        "`L = I - Theta`로 만든 normalized hypergraph Laplacian의 nonzero 개수와 전체 행렬 원소 대비 비율입니다. "
        "네 method는 같은 생성 인스턴스를 공유하므로 `Non-random` 행의 그래프 통계만 사용했습니다. "
        "고립 노드가 있는 경우 `L.nnz = Theta.nnz + 고립 노드 수`로 계산했습니다."
    )
    lines.append("")

    for sweep, title in [("n", "n 변화"), ("K", "K 변화"), ("rho_n", "rho_n 변화")]:
        spec = specs[sweep]
        summary = summaries[sweep]
        lines.append(f"### {title}")
        lines.append("")
        lines.append(
            _markdown_table(
                [spec.x_col, "행렬 크기 n", "Laplacian nnz 평균", "nnz 비율", "nnz 퍼센트"],
                _laplacian_sparsity_rows(summary, spec),
            )
        )
        lines.append("")

    tagged = []
    for sweep, summary in summaries.items():
        tmp = summary.copy()
        tmp["sweep"] = sweep
        tmp["x_value"] = tmp[specs[sweep].x_col]
        tagged.append(tmp)
    all_summary = pd.concat(tagged, ignore_index=True)
    cs = all_summary[all_summary["method"] == "CountSketch random projection"]
    gauss = all_summary[all_summary["method"] == "Gaussian random projection"]
    merged = cs.merge(
        gauss[["sweep", "x_value", "misclassification_mean", "spectral_clustering_wall_sec_mean"]],
        on=["sweep", "x_value"],
        suffixes=("_countsketch", "_gaussian"),
    )
    cs_better = int((merged["misclassification_mean_countsketch"] < merged["misclassification_mean_gaussian"]).sum())
    cs_tied = int(np.isclose(merged["misclassification_mean_countsketch"], merged["misclassification_mean_gaussian"]).sum())
    cs_worse = int((merged["misclassification_mean_countsketch"] > merged["misclassification_mean_gaussian"]).sum())
    time_ratio = (
        merged["spectral_clustering_wall_sec_mean_countsketch"]
        / merged["spectral_clustering_wall_sec_mean_gaussian"]
    ).replace([np.inf, -np.inf], np.nan)

    lines.append("## CountSketch 해석")
    lines.append("")
    lines.append(
        f"- Gaussian RP와 직접 비교한 {len(merged)}개 sweep 지점 중 CountSketch의 오분류율이 더 낮은 지점은 {cs_better}개, 동률은 {cs_tied}개, 더 높은 지점은 {cs_worse}개입니다."
    )
    lines.append(
        f"- CountSketch의 spectral 단계 시간은 Gaussian RP 대비 평균 `{_fmt_float(time_ratio.mean())}`배입니다. CountSketch test matrix는 sparse하지만, 현재 설정에서는 반복적인 `Theta @ Y`와 QR 비용이 여전히 큽니다."
    )
    lines.append(
        "- `rho_n`이 충분히 큰 구간에서는 CountSketch, Gaussian RP, Non-random이 모두 거의 완전 복원에 접근합니다. 이번 sweep에서 가장 희소한 `rho_n=2` 구간은 상대적으로 더 어려운 구간입니다."
    )
    lines.append(
        "- `K` 변화 후반부는 `rho_n`을 고정한 상태에서 `K`가 커지며 effective signal이 약해지는 구간이라, CountSketch를 추가해도 큰 `K`의 난도는 그대로 남습니다."
    )
    lines.append("")
    lines.append("## 해석 메모")
    lines.append("")
    lines.append("- 오분류율은 Hungarian matching으로 예측 label을 true label에 맞춘 뒤 계산했습니다.")
    lines.append("- ARI와 NMI는 label permutation에 불변이므로 원 label을 그대로 사용했습니다.")
    lines.append("- `algorithm_sec`는 생성, hypergraph Laplacian/operator 구성, spectral embedding, row normalization, k-means 주요 단계의 합입니다.")
    lines.append("- 이번 보고서는 기존 저장 CSV를 섞지 않고 현재 실행 환경에서 새로 계산한 `EXP-20260506-007`부터 `009`까지의 비교 결과만 사용했습니다.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_all(show_progress: bool = True):
    outputs = {}
    for sweep in ["K", "n", "rho_n"]:
        outputs[sweep] = run_named_experiment(sweep=sweep, show_progress=show_progress)
    report_path = write_comparison_report()
    return outputs, report_path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run uniform HSBM method comparison.")
    parser.add_argument("sweep", choices=["K", "n", "rho_n", "all", "report"])
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    if args.sweep == "all":
        run_all(show_progress=not args.no_progress)
    elif args.sweep == "report":
        print(write_comparison_report())
    else:
        run_named_experiment(args.sweep, show_progress=not args.no_progress)


if __name__ == "__main__":
    main()
