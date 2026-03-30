# -*- coding: utf-8 -*-
from __future__ import annotations

"""Figure 2 runtime benchmark for "A Stochastic Block Hypergraph model"."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import LiveProgress, METHOD_COLORS  # noqa: E402


@dataclass
class Figure2RuntimeConfig:
    """Configuration for runtime benchmark."""

    n_values: list
    K: int = 4
    strategy: str = "weighted"
    reps: int = 5
    node_order: str = "random"
    seed: int = 2026
    auto_skip_large_n: bool = True
    max_seconds_per_n: float = 20.0


def build_probability_matrix(K: int, p: float, q: float) -> np.ndarray:
    """Build a planted-partition community probability matrix.

    Diagonal entries are within-community probability p, and off-diagonal
    entries are between-community probability q.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")
    if not (0.0 <= q <= p <= 1.0):
        raise ValueError(f"Expected 0 <= q <= p <= 1. Got p={p}, q={q}.")

    P = np.full((K, K), q, dtype=float)
    np.fill_diagonal(P, p)
    return P


def assign_equal_communities(N: int, K: int) -> np.ndarray:
    """Assign N nodes into K communities with as-equal-as-possible sizes."""
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}.")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")
    if K > N:
        raise ValueError(f"K must be <= N for non-empty communities. Got K={K}, N={N}.")

    sizes = np.full(K, N // K, dtype=np.int64)
    sizes[: (N % K)] += 1
    return np.repeat(np.arange(K, dtype=np.int64), sizes)


def _node_sequence(
    N: int,
    communities: np.ndarray,
    node_order: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create node traversal order for one hyperedge update."""
    if node_order == "random":
        return rng.permutation(N)
    if node_order == "fixed":
        return np.arange(N, dtype=np.int64)
    if node_order == "community":
        return np.argsort(communities, kind="stable")
    raise ValueError(f"Unsupported node_order: {node_order}")


def generate_hypergraph(
    N: int,
    E: int,
    K: int,
    p: float,
    q: float,
    strategy: str,
    node_order: str = "random",
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate a stochastic block hypergraph using the paper's join process.

    For each hyperedge:
    1) start from one random seed node,
    2) iterate over all nodes once,
    3) for each candidate node v not in edge, compute Prob(v -> e),
    4) include v via Bernoulli draw.

    Weighted strategy:
    Prob(v -> e) = average_u_in_e P[c(u), c(v)].
    """
    if strategy != "weighted":
        raise ValueError("This benchmark reproduces Figure 2 with strategy='weighted' only.")
    if N <= 0 or E <= 0:
        raise ValueError(f"N and E must be positive, got N={N}, E={E}.")

    rng = np.random.default_rng(seed)
    communities = assign_equal_communities(N, K)
    P = build_probability_matrix(K, p, q)

    hyperedges: list[np.ndarray] = []

    # Complexity note:
    # - For each hyperedge, we scan N nodes once.
    # - Prob(v->e) uses only K community counts (K is constant here), so each
    #   node check is O(1) w.r.t. N.
    # => Total O(N * E). If E = N, this becomes O(N^2).
    for _ in range(E):
        seed_node = int(rng.integers(0, N))
        in_edge = np.zeros(N, dtype=bool)
        in_edge[seed_node] = True

        members = [seed_node]
        edge_comm_counts = np.zeros(K, dtype=np.int64)
        edge_comm_counts[communities[seed_node]] = 1

        for v in _node_sequence(N, communities, node_order, rng):
            if in_edge[v]:
                continue

            c_v = int(communities[v])
            prob_join = float(np.dot(edge_comm_counts, P[:, c_v])) / float(len(members))
            prob_join = float(np.clip(prob_join, 0.0, 1.0))

            if rng.random() < prob_join:
                in_edge[v] = True
                members.append(int(v))
                edge_comm_counts[c_v] += 1

        hyperedges.append(np.asarray(members, dtype=np.int32))

    return hyperedges


def _fit_quadratic(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y ~= a*N^2 + b*N + c and report coefficients with R^2."""
    if x.size < 3:
        return {
            "status": "insufficient_points",
            "model": "a*N^2 + b*N + c",
            "num_points": int(x.size),
        }

    coeffs = np.polyfit(x, y, deg=2)
    y_hat = np.polyval(coeffs, x)
    sse = float(np.sum((y - y_hat) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0.0 else 1.0

    return {
        "status": "ok",
        "model": "a*N^2 + b*N + c",
        "a": float(coeffs[0]),
        "b": float(coeffs[1]),
        "c": float(coeffs[2]),
        "r2": float(r2),
        "num_points": int(x.size),
        "sse": sse,
    }


def run_runtime_benchmark(
    cfg: Figure2RuntimeConfig,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run runtime benchmark over N values and return raw/summary/fit results."""
    master_rng = np.random.default_rng(cfg.seed)
    records = []
    skipped_n_values = []

    total_steps = len(cfg.n_values) * cfg.reps
    progress = LiveProgress(total_steps) if show_progress else None

    for n_idx, N in enumerate(cfg.n_values):
        E = N
        p = 100.0 / float(N)
        q = 0.4 * p
        n_start = perf_counter()

        for rep in range(1, cfg.reps + 1):
            rep_seed = int(master_rng.integers(1, 2**31 - 1))

            t0 = perf_counter()
            hyperedges = generate_hypergraph(
                N=N,
                E=E,
                K=cfg.K,
                p=p,
                q=q,
                strategy=cfg.strategy,
                node_order=cfg.node_order,
                seed=rep_seed,
            )
            runtime_sec = perf_counter() - t0

            edge_sizes = np.asarray([len(e) for e in hyperedges], dtype=float)
            records.append(
                {
                    "N": N,
                    "E": E,
                    "K": cfg.K,
                    "p": p,
                    "q": q,
                    "strategy": cfg.strategy,
                    "node_order": cfg.node_order,
                    "rep": rep,
                    "runtime_sec": runtime_sec,
                    "mean_hyperedge_size": float(edge_sizes.mean()),
                }
            )

            if progress is not None:
                progress.update("N", N, rep, cfg.reps, "weighted")

        elapsed_n = perf_counter() - n_start
        if cfg.auto_skip_large_n and elapsed_n > cfg.max_seconds_per_n:
            skipped_n_values = list(cfg.n_values[n_idx + 1 :])
            break

    if progress is not None:
        progress.close()

    if not records:
        raise RuntimeError("No benchmark records were generated.")

    df_raw = pd.DataFrame(records)
    df_summary = (
        df_raw.groupby("N", as_index=False)
        .agg(
            E=("E", "first"),
            K=("K", "first"),
            p=("p", "first"),
            q=("q", "first"),
            strategy=("strategy", "first"),
            node_order=("node_order", "first"),
            reps_completed=("runtime_sec", "count"),
            runtime_mean_sec=("runtime_sec", "mean"),
            runtime_std_sec=("runtime_sec", "std"),
            mean_hyperedge_size=("mean_hyperedge_size", "mean"),
        )
        .sort_values("N")
    )
    df_summary["runtime_std_sec"] = df_summary["runtime_std_sec"].fillna(0.0)

    fit = _fit_quadratic(
        x=df_summary["N"].to_numpy(dtype=float),
        y=df_summary["runtime_mean_sec"].to_numpy(dtype=float),
    )
    fit["used_n_values"] = [int(v) for v in df_summary["N"].tolist()]
    fit["skipped_n_values"] = [int(v) for v in skipped_n_values]

    return df_raw, df_summary, fit


def plot_runtime_figure(df_summary: pd.DataFrame, fit: dict, out_png: Path) -> None:
    """Plot runtime-vs-N with error bars and optional quadratic fit."""
    color = METHOD_COLORS["Random Projection"]
    x = df_summary["N"].to_numpy(dtype=float)
    y = df_summary["runtime_mean_sec"].to_numpy(dtype=float)
    yerr = df_summary["runtime_std_sec"].to_numpy(dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        color=color,
        linewidth=2.0,
        markersize=4.5,
        capsize=3.0,
        label="Measured runtime (mean ± std)",
    )

    if fit.get("status") == "ok":
        x_fit = np.linspace(float(x.min()), float(x.max()), 200)
        y_fit = fit["a"] * x_fit**2 + fit["b"] * x_fit + fit["c"]
        ax.plot(
            x_fit,
            y_fit,
            linestyle="--",
            linewidth=2.0,
            color="#d62728",
            label=f"Quadratic fit (R^2={fit['r2']:.4f})",
        )

    ax.set_xlabel("N (with E = N)")
    ax.set_ylabel("Runtime (sec)")
    ax.set_title("Figure 2-like Runtime Benchmark (weighted strategy)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_n_values(arg: str) -> list:
    """Parse comma-separated N values."""
    vals = [v.strip() for v in arg.split(",") if v.strip()]
    if not vals:
        raise ValueError("At least one N value must be provided.")
    return [int(v) for v in vals]


def _print_setting_report(cfg: Figure2RuntimeConfig, fit: dict) -> None:
    """Print paper-stated settings and additional assumptions separately."""
    print("\n[논문에서 명시된 설정]")
    print("- strategy = weighted")
    print("- K = 4")
    print("- N = E")
    print("- p = 100 / N")
    print("- q = 0.4 * p")
    print("- community = equal-size partition")

    print("\n[내가 추가로 가정한 설정]")
    print(f"- N 후보 = {cfg.n_values}")
    print(f"- 반복 횟수(reps) = {cfg.reps}")
    print(f"- node_order = {cfg.node_order}")
    print(f"- master seed = {cfg.seed}")
    print(f"- auto skip large N = {cfg.auto_skip_large_n}")
    if cfg.auto_skip_large_n:
        print(f"- max_seconds_per_n = {cfg.max_seconds_per_n:.1f}")
    print("- hyperedge 초기화: 각 hyperedge마다 무작위 seed node 1개로 시작")
    print("- weighted 확률 계산은 edge 내 community count를 이용해 동일 수식으로 계산")
    if fit.get("status") == "ok":
        print(
            f"- quadratic fit: a={fit['a']:.6e}, b={fit['b']:.6e}, c={fit['c']:.6e}, R^2={fit['r2']:.6f}"
        )
    else:
        print("- quadratic fit: 데이터 포인트 부족으로 생략")
    if fit.get("skipped_n_values"):
        print(f"- 자동 생략된 N = {fit['skipped_n_values']}")


def main() -> None:
    """CLI entry point for Figure 2 runtime benchmark."""
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 2 runtime benchmark for HySBM generation."
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="100,200,400,800,1200,1600,2000,3000,4000",
        help="Comma-separated list of N values.",
    )
    parser.add_argument("--reps", type=int, default=5, help="Repetitions per N (>=5 recommended).")
    parser.add_argument("--seed", type=int, default=2026, help="Master seed for reproducibility.")
    parser.add_argument(
        "--node-order",
        type=str,
        default="random",
        choices=["random", "fixed", "community"],
        help="Node traversal order inside each hyperedge loop.",
    )
    parser.add_argument(
        "--max-seconds-per-n",
        type=float,
        default=20.0,
        help="Auto-skip larger N when one N takes longer than this threshold.",
    )
    parser.add_argument(
        "--disable-auto-skip",
        action="store_true",
        help="Disable auto skip for large N.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/reference_1_exp1",
        help="Output root directory containing figures/ and results/.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress output.")
    args = parser.parse_args()

    n_values = _parse_n_values(args.n_values)
    if args.reps < 5:
        raise ValueError(f"reps must be >= 5 per benchmark requirement, got {args.reps}.")

    cfg = Figure2RuntimeConfig(
        n_values=n_values,
        K=4,
        strategy="weighted",
        reps=args.reps,
        node_order=args.node_order,
        seed=args.seed,
        auto_skip_large_n=(not args.disable_auto_skip),
        max_seconds_per_n=float(args.max_seconds_per_n),
    )

    out_root = Path(args.outdir)
    fig_dir = out_root / "figures"
    res_dir = out_root / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("Running Figure 2 runtime benchmark...")
    _, df_summary, fit = run_runtime_benchmark(cfg, show_progress=(not args.no_progress))

    out_csv = res_dir / "figure2_runtime.csv"
    out_json = res_dir / "figure2_fit.json"
    out_png = fig_dir / "figure2_runtime.png"

    df_summary.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(fit, f, indent=2, ensure_ascii=False)
    plot_runtime_figure(df_summary, fit, out_png)

    print("Done.")
    print(f"Runtime CSV : {out_csv.resolve()}")
    print(f"Fit JSON    : {out_json.resolve()}")
    print(f"Figure PNG  : {out_png.resolve()}")
    _print_setting_report(cfg, fit)


if __name__ == "__main__":
    main()
