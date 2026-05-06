from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/python-rand-nla-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/python-rand-nla-cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


COUNT_SKETCH_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = COUNT_SKETCH_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_ROOT = COUNT_SKETCH_DIR / "results"

for path in (PROJECT_ROOT, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.common import (  # noqa: E402
    LiveProgress,
    generate_uniform_hsbm_instance,
    hypergraph_laplacian,
    make_uniform_hsbm_probs,
    normalize_rows_l2,
)
from uniform_hsbm_randomized import (  # noqa: E402
    aligned_misclassification_rate,
    expected_uniform_hsbm_stats,
    hypergraph_vertex_degree_stats,
    measure_call,
)


METHOD_LABEL = "CountSketch random projection"
DENSE_RHO_MULTIPLIER = 4.0


@dataclass(frozen=True)
class CountSketchSpec:
    sweep: str
    experiment_id: str
    experiment_slug: str
    script_name: str
    title_ko: str
    x_col: str
    x_values: tuple[float | int, ...]
    n: int | None = None
    K: int | None = None
    m: int = 3
    a_in: float = 36.0
    b_out: float = 4.0
    rho_n: float | None = None
    rho_n_multiplier: float = DENSE_RHO_MULTIPLIER
    reps: int = 10
    seed: int = 20260506
    sampling: str = "sparse"
    max_enumeration: int = 1_500_000
    normalize_embedding_rows: bool = True
    countsketch_oversampling: int = 160
    countsketch_power_iter: int = 4
    kmeans_n_init: int = 20

    @property
    def method_label(self) -> str:
        return METHOD_LABEL

    @property
    def outdir(self) -> Path:
        return RESULTS_ROOT / f"{self.experiment_id}_{self.experiment_slug}"

    @property
    def file_prefix(self) -> str:
        return f"{self.experiment_id}_{self.experiment_slug}"


def get_countsketch_specs() -> dict[str, CountSketchSpec]:
    K_VALUES = (2, 3, 4, 5, 6, 8, 10, 12)
    N_VALUES = tuple(range(1000, 10001, 1000))
    BASE_RHO_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    DENSE_RHO_VALUES = tuple(float(v * DENSE_RHO_MULTIPLIER) for v in BASE_RHO_VALUES)

    specs = [
        CountSketchSpec(
            sweep="K",
            experiment_id="EXP-20260506-001",
            experiment_slug="uniform_hsbm_K_sweep_countsketch_dense",
            script_name="K변화_countsketch.py",
            title_ko="균일 HSBM K 변화 - CountSketch dense",
            x_col="K",
            x_values=K_VALUES,
            n=5000,
            rho_n=8.0 * DENSE_RHO_MULTIPLIER,
            seed=20260506,
        ),
        CountSketchSpec(
            sweep="n",
            experiment_id="EXP-20260506-002",
            experiment_slug="uniform_hsbm_n_scaling_countsketch_dense",
            script_name="n변화_countsketch.py",
            title_ko="균일 HSBM n 변화 - CountSketch dense",
            x_col="n",
            x_values=N_VALUES,
            K=3,
            rho_n=4.0 * DENSE_RHO_MULTIPLIER,
            seed=20260506,
        ),
        CountSketchSpec(
            sweep="rho_n",
            experiment_id="EXP-20260506-003",
            experiment_slug="uniform_hsbm_rho_n_sweep_countsketch_dense",
            script_name="rho_n변화_countsketch.py",
            title_ko="균일 HSBM rho_n 변화 - CountSketch dense",
            x_col="rho_n",
            x_values=DENSE_RHO_VALUES,
            n=5000,
            K=3,
            seed=20260506,
        ),
    ]
    return {spec.sweep: spec for spec in specs}


def resolve_spec(sweep: str) -> CountSketchSpec:
    specs = get_countsketch_specs()
    if sweep not in specs:
        raise KeyError(f"Unknown CountSketch experiment spec: {sweep}")
    return specs[sweep]


def value_to_seed_component(value: int | float) -> int:
    if isinstance(value, float):
        return int(round(value * 1000))
    return int(value)


def concrete_params(spec: CountSketchSpec, x_value: int | float):
    n = int(x_value) if spec.sweep == "n" else int(spec.n)
    K = int(x_value) if spec.sweep == "K" else int(spec.K)
    rho_n = float(x_value) if spec.sweep == "rho_n" else float(spec.rho_n)
    return n, K, rho_n


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
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[-K:][::-1]
    top_vals = vals[order]
    core_vecs = vecs[:, order]
    timings["cs_small_eig_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    U = Q @ core_vecs
    timings["cs_lift_sec"] = time.perf_counter() - t0
    return top_vals, U, timings


def spectral_cluster_from_theta(
    theta: sp.csr_matrix,
    K: int,
    rng: np.random.Generator,
    spec: CountSketchSpec,
):
    theta = ((theta + theta.T) * 0.5).tocsr()
    theta.eliminate_zeros()
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    vals, U, timings = countsketch_random_projection_embedding(
        theta=theta,
        K=K,
        r=spec.countsketch_oversampling,
        q=spec.countsketch_power_iter,
        rng=rng,
    )
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


def run_one_rep(spec: CountSketchSpec, x_value: int | float, rep: int):
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
        "rho_n_multiplier": float(spec.rho_n_multiplier),
        "rho_n_base_equivalent": float(rho_n / spec.rho_n_multiplier),
        "density_regime": f"rho_n_x{spec.rho_n_multiplier:g}",
        "num_hyperedges_total": int(len(hyperedges)),
        "theta_nnz": int(theta.nnz),
        "theta_density": float(theta.nnz / (n * n)),
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


def run_one_rep_measured(spec: CountSketchSpec, x_value: int | float, rep: int):
    record, measurement = measure_call(lambda: run_one_rep(spec=spec, x_value=x_value, rep=rep))
    record.update(measurement)
    return record


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
        "cs_draw_hash_sec",
        "cs_initial_multiply_sec",
        "cs_power_iter_sec",
        "cs_qr_sec",
        "cs_build_core_sec",
        "cs_small_eig_sec",
        "cs_lift_sec",
        "cs_embedding_dim",
        "cs_bucket_min_load",
        "cs_bucket_max_load",
        "cs_empty_buckets",
        "top_eigenvalue_max",
        "top_eigenvalue_min",
    ]
    aggregations = {"reps": ("rep", "count")}
    for col in preferred:
        if col in df_raw.columns:
            label = "misclassification" if col == "misclassification_rate" else col
            aggregations[f"{label}_mean"] = (col, "mean")
            aggregations[f"{label}_std"] = (col, "std")
    return df_raw.groupby([x_col, "method"], as_index=False).agg(**aggregations)


def plot_summary(summary: pd.DataFrame, spec: CountSketchSpec, out_png: Path):
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
    fig.suptitle(f"{title_by_sweep.get(spec.sweep, spec.sweep)} - CountSketch dense")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_spec(spec: CountSketchSpec, show_progress: bool = True):
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
    config["dense_note"] = (
        "The original rho_n schedule is multiplied by rho_n_multiplier to make "
        "the generated HSBM operator denser while keeping the same sweep axes."
    )
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


def run_named_experiment(sweep: str, show_progress: bool = True):
    return run_spec(resolve_spec(sweep=sweep), show_progress=show_progress)


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


def _summary_path(spec: CountSketchSpec) -> Path:
    return spec.outdir / f"{spec.file_prefix}_summary.csv"


def _plot_relpath(spec: CountSketchSpec) -> str:
    return f"results/{spec.experiment_id}_{spec.experiment_slug}/{spec.file_prefix}_summary.png"


def _load_summary(spec: CountSketchSpec) -> pd.DataFrame:
    path = _summary_path(spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return pd.read_csv(path)


def _load_existing_rho_comparison():
    base_results = EXPERIMENT_DIR / "results"
    paths = {
        "Non-random": base_results
        / "EXP-20260427-001_uniform_hsbm_rho_n_sweep_zhou_theta"
        / "EXP-20260427-001_uniform_hsbm_rho_n_sweep_zhou_theta_summary.csv",
        "Gaussian random projection": base_results
        / "EXP-20260428-005_uniform_hsbm_rho_n_sweep_gaussian_random_projection"
        / "EXP-20260428-005_uniform_hsbm_rho_n_sweep_gaussian_random_projection_summary.csv",
    }
    frames = []
    for method, path in paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "rho_n" not in df.columns or "misclassification_mean" not in df.columns:
            continue
        frames.append(df[["rho_n", "misclassification_mean"]].assign(method=method))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_countsketch_report(out_path: Path | None = None) -> Path:
    specs = get_countsketch_specs()
    out_path = out_path or (COUNT_SKETCH_DIR / "결과보고서.md")
    summaries = {sweep: _load_summary(spec) for sweep, spec in specs.items()}

    lines: list[str] = []
    lines.append("# 균일 HSBM CountSketch dense 실험 결과보고서")
    lines.append("")
    lines.append(
        "이 보고서는 `균일 HSBM 실험/countsketch` 폴더에서 실행한 CountSketch random projection 실험을 정리한 것입니다. "
        "기존 가우시안 랜덤 프로젝션과 동일한 subspace iteration 구조를 사용하되, Gaussian test matrix 대신 각 노드가 하나의 bucket과 부호만 갖는 CountSketch test matrix를 사용했습니다."
    )
    lines.append("")
    lines.append("## 실험 구성")
    lines.append("")
    lines.append("- `n변화`: `K=3`, dense `rho_n=16.0`을 고정하고 `n`을 1000부터 10000까지 바꿉니다.")
    lines.append("- `K변화`: `n=5000`, dense `rho_n=32.0`을 고정하고 `K`를 바꿉니다.")
    lines.append("- `rho_n변화`: 기존 `{0.25, 0.5, 1, 2, 4, 8, 16}` schedule에 4를 곱한 `{1, 2, 4, 8, 16, 32, 64}`를 사용합니다.")
    lines.append("- `CountSketch random projection`: `ell = K + 160`, power iteration `q=4`로 설정했습니다.")
    lines.append("- `dense`는 full matrix 저장이 아니라 `rho_n`을 4배 키운 denser HSBM regime을 뜻합니다. `n=10000`에서 모든 3-uniform 후보를 열거하는 완전 dense Bernoulli 방식은 계산량이 너무 커서 사용하지 않았습니다.")
    lines.append("- 세부 표는 같은 `n`, `K`, `rho_n` 값끼리 작은 표로 묶었습니다.")
    lines.append("")
    lines.append("## 전체 요약")
    lines.append("")

    overall_rows = []
    for sweep in ["K", "n", "rho_n"]:
        summary = summaries[sweep]
        overall_rows.append(
            [
                sweep,
                METHOD_LABEL,
                _fmt_float(summary["misclassification_mean"].mean()),
                _fmt_float(summary["ARI_mean"].mean()),
                _fmt_float(summary["NMI_mean"].mean()),
                _fmt_float(summary["algorithm_total_wall_sec_mean"].mean()),
                _fmt_float(summary["spectral_clustering_wall_sec_mean"].mean()),
                _fmt_float(summary["hypergraph_degree_mean_mean"].mean()),
            ]
        )
    lines.append(
        _markdown_table(
            ["sweep", "method", "평균_오분류율", "평균_ARI", "평균_NMI", "평균_주요시간초", "평균_spectral초", "평균degree"],
            overall_rows,
        )
    )
    lines.append("")

    section_meta = [
        ("n", "n 변화 실험"),
        ("K", "K 변화 실험"),
        ("rho_n", "rho_n 변화 실험"),
    ]
    for sweep, title in section_meta:
        spec = specs[sweep]
        summary = summaries[sweep].sort_values(spec.x_col)
        lines.append(f"## {title}")
        lines.append("")
        for _, row in summary.iterrows():
            x_value = row[spec.x_col]
            if sweep == "n":
                section_value = f"{float(x_value):.1f}"
            else:
                section_value = f"{float(x_value):.4f}"
            lines.append(f"### {spec.x_col} = {section_value}")
            lines.append("")
            rows = [
                [
                    row["method"],
                    _fmt_float(row["misclassification_mean"]),
                    _fmt_float(row["ARI_mean"]),
                    _fmt_float(row["NMI_mean"]),
                    _fmt_float(row["algorithm_total_wall_sec_mean"]),
                    _fmt_float(row["spectral_clustering_wall_sec_mean"]),
                    _fmt_float(row["num_hyperedges_total_mean"], 1),
                    _fmt_float(row["hypergraph_degree_mean_mean"], 4),
                    _fmt_float(row["theta_density_mean"], 4),
                ]
            ]
            lines.append(
                _markdown_table(
                    ["방법", "오분류율", "ARI", "NMI", "주요시간초", "spectral초", "하이퍼엣지수", "평균degree", "Theta density"],
                    rows,
                )
            )
            lines.append("")
        lines.append("### 그림")
        lines.append("")
        lines.append(f"![{title} CountSketch dense]({_plot_relpath(spec)})")
        lines.append("")

    existing_rho = _load_existing_rho_comparison()
    if not existing_rho.empty:
        current_rho = summaries["rho_n"][["rho_n", "misclassification_mean"]].assign(
            method=METHOD_LABEL
        )
        compare = pd.concat([current_rho, existing_rho], ignore_index=True)
        compare = compare[compare["rho_n"].isin([1.0, 2.0, 4.0, 8.0, 16.0])]
        if not compare.empty:
            pivot = (
                compare.pivot_table(
                    index="rho_n",
                    columns="method",
                    values="misclassification_mean",
                    aggfunc="mean",
                )
                .reset_index()
                .sort_values("rho_n")
            )
            lines.append("## 기존 결과와의 비교 메모")
            lines.append("")
            lines.append(
                "`rho_n` 변화 실험의 `rho_n in {1, 2, 4, 8, 16}` 구간은 기존 보고서의 실제 `rho_n` 값과 직접 겹치므로, 그 구간에서는 CountSketch와 기존 Gaussian/Non-random을 나란히 볼 수 있습니다."
            )
            lines.append("")
            rows = []
            for _, row in pivot.iterrows():
                rows.append(
                    [
                        _fmt_float(row["rho_n"]),
                        _fmt_float(row.get(METHOD_LABEL, float("nan"))),
                        _fmt_float(row.get("Gaussian random projection", float("nan"))),
                        _fmt_float(row.get("Non-random", float("nan"))),
                    ]
                )
            lines.append(
                _markdown_table(
                    ["rho_n", "CountSketch 오분류율", "Gaussian 오분류율", "Non-random 오분류율"],
                    rows,
                )
            )
            lines.append("")
            lines.append(
                "이 겹치는 구간에서는 CountSketch가 tuned Gaussian random projection과 매우 비슷한 패턴을 보입니다. `rho_n=2`에서는 둘 다 non-random보다 높지만, `rho_n>=4`부터는 차이가 작아지고 `rho_n>=8`에서는 거의 완전 복원에 가깝습니다."
            )
            lines.append("")

    tagged = []
    for sweep, summary in summaries.items():
        tmp = summary.copy()
        tmp["sweep"] = sweep
        tmp["x_value"] = tmp[specs[sweep].x_col]
        tagged.append(tmp)
    all_summary = pd.concat(tagged, ignore_index=True)
    best = all_summary.loc[all_summary["misclassification_mean"].idxmin()]
    worst = all_summary.loc[all_summary["misclassification_mean"].idxmax()]
    rho_summary = summaries["rho_n"].sort_values("rho_n")
    high_rho = rho_summary[rho_summary["rho_n"] >= 16.0]
    high_rho_mis = high_rho["misclassification_mean"].mean()

    lines.append("## CountSketch 결과 해석")
    lines.append("")
    lines.append(
        f"- 전체 sweep 중 가장 낮은 평균 오분류율은 `{best['sweep']}={_fmt_float(best['x_value'])}`에서 "
        f"`{_fmt_float(best['misclassification_mean'])}`이고, 가장 어려운 지점은 `{worst['sweep']}={_fmt_float(worst['x_value'])}`의 "
        f"`{_fmt_float(worst['misclassification_mean'])}`입니다."
    )
    lines.append(
        f"- `rho_n >= 16`인 조밀한 구간의 평균 오분류율은 `{_fmt_float(high_rho_mis)}`로 낮습니다. 즉 sparse regime에서 보이던 랜덤 스케치의 손실은 density를 올리면 상당히 줄어듭니다."
    )
    lines.append(
        "- 다만 `K`가 커지는 sweep에서는 `rho_n`을 4배 키워도 큰 `K` 구간이 여전히 어렵습니다. 이는 CountSketch 자체만의 문제라기보다, 기존 보고서에서 지적한 것처럼 `K` 증가와 함께 within-community 후보 비율 및 effective signal이 같이 바뀌는 효과가 남아 있기 때문입니다."
    )
    lines.append(
        "- CountSketch는 Gaussian projection보다 test matrix 생성 비용과 저장량은 작지만, 여기서는 `ell=K+160`, `q=4`를 유지했기 때문에 spectral 단계의 주된 비용은 여전히 반복적인 `Theta @ Y` 곱입니다."
    )
    lines.append("")
    lines.append("## 해석 메모")
    lines.append("")
    lines.append("- 오분류율은 Hungarian matching으로 예측 label을 true label에 맞춘 뒤 계산했습니다.")
    lines.append("- ARI와 NMI는 label permutation에 불변이므로 원 label을 그대로 사용했습니다.")
    lines.append("- `algorithm_sec`는 생성, hypergraph Laplacian/operator 구성, CountSketch spectral embedding, row normalization, k-means 주요 단계의 합입니다.")
    lines.append("- `sampling_mode=sparse`는 후보를 전부 열거하지 않는 생성 알고리즘의 이름입니다. 이번 실험의 denser 조건은 `rho_n_x4`로 기록된 밀도 배율에서 옵니다.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_all(show_progress: bool = True):
    outputs = {}
    for sweep in ["K", "n", "rho_n"]:
        outputs[sweep] = run_named_experiment(sweep=sweep, show_progress=show_progress)
    report_path = write_countsketch_report()
    return outputs, report_path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run uniform HSBM CountSketch dense experiments.")
    parser.add_argument("sweep", choices=["K", "n", "rho_n", "all", "report"])
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    if args.sweep == "all":
        run_all(show_progress=not args.no_progress)
    elif args.sweep == "report":
        path = write_countsketch_report()
        print(path)
    else:
        run_named_experiment(sweep=args.sweep, show_progress=not args.no_progress)


if __name__ == "__main__":
    main()
