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

from src.common import LiveProgress  # noqa: E402
from src.hypergraph_sbm import generate_hypergraph, hyperedge_sizes  # noqa: E402


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

            edge_sizes = hyperedge_sizes(hyperedges).astype(float)
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


def plot_runtime_figure(
    df_summary: pd.DataFrame,
    fit: dict,
    out_png: Path,
    requested_n_values: list | None = None,
    paper_style_axes: bool = True,
) -> None:
    """Plot Figure 2 style: panel (a) runtime vs N, panel (b) time vs fitted f(N)."""
    x = df_summary["N"].to_numpy(dtype=float)
    y = df_summary["runtime_mean_sec"].to_numpy(dtype=float)
    yerr = df_summary["runtime_std_sec"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    ax0, ax1 = axes

    # (a) runtime benchmark for different N
    ax0.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        color="black",
        linewidth=1.8,
        markersize=4.2,
        capsize=3.0,
        elinewidth=1.2,
        label="Measured runtime",
    )
    ax0.set_xlabel("N")
    ax0.set_ylabel("Time (s)")
    ax0.text(0.5, -0.18, "(a)", transform=ax0.transAxes, ha="center", va="top")

    if requested_n_values:
        req_max = float(max(requested_n_values))
        used_max = float(np.max(x))
        # If larger N values were skipped, zoom to used range for readability.
        if used_max < 0.95 * req_max:
            ax0.set_xlim(0.0, used_max * 1.05)
        else:
            ax0.set_xlim(0.0, req_max)
    else:
        ax0.set_xlim(0.0, float(np.max(x)) * 1.03)

    # paper-like y-range for visual comparability
    if paper_style_axes:
        ymax = float(np.max(y + yerr)) if y.size else 1.0
        ax0.set_ylim(0.0, max(22.0, ymax * 1.08))
    else:
        ymax = float(np.max(y + yerr)) if y.size else 1.0
        ax0.set_ylim(0.0, max(1.0, ymax * 1.15))

    # (b) quadratic fit check: Time(s) vs f(N)
    if fit.get("status") == "ok":
        fN = fit["a"] * x**2 + fit["b"] * x + fit["c"]
        ax1.plot(
            fN,
            y,
            "o",
            color="black",
            markersize=4.2,
        )
        hi = float(max(np.max(fN), np.max(y))) if y.size else 1.0
        if paper_style_axes:
            hi = max(22.0, hi * 1.08)
        else:
            hi = max(1.0, hi * 1.10)
        ax1.plot([0.0, hi], [0.0, hi], linestyle="--", color="#666666", linewidth=1.5)
        ax1.set_xlim(0.0, hi)
        ax1.set_ylim(0.0, hi)
        ax1.text(
            0.03,
            0.97,
            "\n".join(
                [
                    f"R^2={fit['r2']:.4f}",
                    "f(N)=aN^2+bN+c",
                    f"a={fit['a']:.2e}, b={fit['b']:.2e}, c={fit['c']:.2e}",
                ]
            ),
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    else:
        ax1.text(
            0.5,
            0.5,
            "Not enough points\nfor quadratic fit",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )

    ax1.set_xlabel("f(N)")
    ax1.set_ylabel("Time (s)")
    ax1.text(0.5, -0.18, "(b)", transform=ax1.transAxes, ha="center", va="top")

    for ax in (ax0, ax1):
        ax.tick_params(direction="out")
        ax.grid(False)

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
        default="experiments/reference_2_exp1",
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
    plot_runtime_figure(
        df_summary=df_summary,
        fit=fit,
        out_png=out_png,
        requested_n_values=cfg.n_values,
        paper_style_axes=True,
    )

    print("Done.")
    print(f"Runtime CSV : {out_csv.resolve()}")
    print(f"Fit JSON    : {out_json.resolve()}")
    print(f"Figure PNG  : {out_png.resolve()}")
    _print_setting_report(cfg, fit)


if __name__ == "__main__":
    main()
