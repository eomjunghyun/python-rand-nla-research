# -*- coding: utf-8 -*-

"""Section 7.2 Models 4-6 reproduction using shared common utilities."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    LiveProgress,
    METHODS,
    METHOD_COLORS,
    attach_timing_breakdown,
    evaluate_metrics,
    extract_timing_breakdown,
    run_non_random,
    run_random_projection,
    run_random_sampling,
    summarize_metrics,
    summarize_timing_breakdown,
)


@dataclass
class Exp72Models456Config:
    n_values: list
    model_ids: list
    K: int = 3
    K_prime_fullrank: int = 3
    K_prime_rankdef: int = 2
    q: int = 2
    r: int = 10
    p: float = 0.7
    reps: int = 20
    seed: int = 2026


def _sizes_from_proportions(n: int, proportions: np.ndarray) -> np.ndarray:
    raw = proportions * n
    sizes = np.floor(raw).astype(int)
    remain = n - int(sizes.sum())
    if remain > 0:
        order = np.argsort(-(raw - sizes))
        for i in order[:remain]:
            sizes[i] += 1
    sizes[-1] += n - int(sizes.sum())
    return sizes


def _labels_from_sizes(sizes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = np.repeat(np.arange(len(sizes)), sizes)
    rng.shuffle(labels)
    return labels


def _sample_symmetric_adjacency(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = P.shape[0]
    tri = np.triu_indices(n, k=1)
    probs = np.clip(P[tri], 0.0, 1.0)
    edges = (rng.random(probs.shape[0]) < probs).astype(float)
    A = np.zeros((n, n), dtype=float)
    A[tri] = edges
    A += A.T
    np.fill_diagonal(A, 0.0)
    return A


def _build_B_model45(rng: np.random.Generator) -> np.ndarray:
    K = 3
    B = np.zeros((K, K), dtype=float)
    for i in range(K):
        B[i, i] = rng.uniform(0.4, 0.6)
    for i in range(K):
        for j in range(i + 1, K):
            v = rng.uniform(0.01, 0.2)
            B[i, j] = v
            B[j, i] = v
    return B


def _build_B_model6() -> np.ndarray:
    C = np.array(
        [
            [2.0 * np.sin(np.pi / 3.0), 2.0 * np.cos(np.pi / 3.0)],
            [np.sin(np.pi / 5.0), 2.0 * np.cos(np.pi / 5.0)],
            [
                (2.0 / 5.0) * np.sin(2.0 * np.pi / 5.0),
                (6.0 / 5.0) * np.cos(2.0 * np.pi / 5.0),
            ],
        ],
        dtype=float,
    )
    B = C @ C.T
    B = B / max(1.0, float(B.max()))
    return B


def _sample_theta_model4(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = np.zeros(labels.shape[0], dtype=float)
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        theta_k = rng.choice([0.2, 1.0], size=idx.size, p=[0.8, 0.2]).astype(float)
        theta_k /= max(1e-12, float(theta_k.max()))
        theta[idx] = theta_k
    return theta


def _sample_theta_model5(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = np.zeros(labels.shape[0], dtype=float)
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        theta_k = rng.choice([0.1, 0.2, 1.0], size=idx.size, p=[0.4, 0.4, 0.2]).astype(float)
        theta_k /= max(1e-12, float(theta_k.max()))
        theta[idx] = theta_k
    return theta


def generate_model456_instance(
    model_id: int,
    n: int,
    rng: np.random.Generator,
):
    if model_id not in {4, 5, 6}:
        raise ValueError("model_id must be one of {4,5,6}.")

    sizes = _sizes_from_proportions(n, np.ones(3) / 3.0)
    y_true = _labels_from_sizes(sizes, rng)

    if model_id in {4, 5}:
        B_true = _build_B_model45(rng)
    else:
        B_true = _build_B_model6()

    if model_id in {4, 6}:
        theta = _sample_theta_model4(y_true, rng)
    else:
        theta = _sample_theta_model5(y_true, rng)

    P = (theta[:, None] * theta[None, :]) * B_true[y_true][:, y_true]
    P = np.clip(P, 0.0, 1.0)
    np.fill_diagonal(P, 0.0)
    A = _sample_symmetric_adjacency(P, rng)
    return A, P, B_true, y_true


def run_experiment72_models456(
    cfg: Exp72Models456Config,
    show_progress: bool = True,
    theta_mode: str = "exact",
    detailed_timing: bool = False,
) -> pd.DataFrame:
    master_rng = np.random.default_rng(cfg.seed)
    records = []

    total_steps = len(cfg.model_ids) * len(cfg.n_values) * cfg.reps * 3
    progress = LiveProgress(total_steps) if show_progress else None

    for model_id in cfg.model_ids:
        for n in cfg.n_values:
            for rep in range(1, cfg.reps + 1):
                rep_seed = int(master_rng.integers(1, 2**31 - 1))
                rng = np.random.default_rng(rep_seed)

                t0 = perf_counter()
                A, P, B_true, y_true = generate_model456_instance(model_id=model_id, n=n, rng=rng)
                t_instance = perf_counter() - t0
                Theta_true = np.eye(cfg.K)[y_true]
                K_prime = cfg.K_prime_rankdef if model_id == 6 else cfg.K_prime_fullrank

                if detailed_timing:
                    Ahat_rp, y_rp, timing_rp = run_random_projection(
                        A,
                        cfg.K,
                        K_prime,
                        cfg.r,
                        cfg.q,
                        rng,
                        normalize_rows=True,
                        return_timing=True,
                    )
                    t_rp = timing_rp["algo_total_sec"]
                else:
                    t0 = perf_counter()
                    Ahat_rp, y_rp = run_random_projection(
                        A,
                        cfg.K,
                        K_prime,
                        cfg.r,
                        cfg.q,
                        rng,
                        normalize_rows=True,
                    )
                    t_rp = perf_counter() - t0
                    timing_rp = None

                t0 = perf_counter()
                eP_rp, eT_rp, eB_rp = evaluate_metrics(
                    Ahat_rp, y_rp, P, B_true, Theta_true, y_true, cfg.K, theta_mode=theta_mode
                )
                t_eval_rp = perf_counter() - t0
                record_rp = {
                    "model": model_id,
                    "n": n,
                    "rep": rep,
                    "method": "Random Projection",
                    "error_P": eP_rp,
                    "error_Theta": eT_rp,
                    "error_B": eB_rp,
                    "time_sec": t_rp,
                }
                if detailed_timing:
                    record_rp = attach_timing_breakdown(
                        record_rp,
                        algo_timing=timing_rp,
                        instance_sec=t_instance,
                        metric_sec=t_eval_rp,
                    )
                records.append(record_rp)
                if progress is not None:
                    progress.update("model/n", f"{model_id}/{n}", rep, cfg.reps, "Random Projection")

                if detailed_timing:
                    Ahat_rs, y_rs, timing_rs = run_random_sampling(
                        A,
                        cfg.K,
                        K_prime,
                        cfg.p,
                        rng,
                        normalize_rows=True,
                        return_timing=True,
                    )
                    t_rs = timing_rs["algo_total_sec"]
                else:
                    t0 = perf_counter()
                    Ahat_rs, y_rs = run_random_sampling(
                        A,
                        cfg.K,
                        K_prime,
                        cfg.p,
                        rng,
                        normalize_rows=True,
                    )
                    t_rs = perf_counter() - t0
                    timing_rs = None

                t0 = perf_counter()
                eP_rs, eT_rs, eB_rs = evaluate_metrics(
                    Ahat_rs, y_rs, P, B_true, Theta_true, y_true, cfg.K, theta_mode=theta_mode
                )
                t_eval_rs = perf_counter() - t0
                record_rs = {
                    "model": model_id,
                    "n": n,
                    "rep": rep,
                    "method": "Random Sampling",
                    "error_P": eP_rs,
                    "error_Theta": eT_rs,
                    "error_B": eB_rs,
                    "time_sec": t_rs,
                }
                if detailed_timing:
                    record_rs = attach_timing_breakdown(
                        record_rs,
                        algo_timing=timing_rs,
                        instance_sec=t_instance,
                        metric_sec=t_eval_rs,
                    )
                records.append(record_rs)
                if progress is not None:
                    progress.update("model/n", f"{model_id}/{n}", rep, cfg.reps, "Random Sampling")

                if detailed_timing:
                    Ahat_nr, y_nr, timing_nr = run_non_random(
                        A,
                        cfg.K,
                        K_prime,
                        rng,
                        normalize_rows=True,
                        return_timing=True,
                    )
                    t_nr = timing_nr["algo_total_sec"]
                else:
                    t0 = perf_counter()
                    Ahat_nr, y_nr = run_non_random(
                        A,
                        cfg.K,
                        K_prime,
                        rng,
                        normalize_rows=True,
                    )
                    t_nr = perf_counter() - t0
                    timing_nr = None

                t0 = perf_counter()
                eP_nr, eT_nr, eB_nr = evaluate_metrics(
                    Ahat_nr, y_nr, P, B_true, Theta_true, y_true, cfg.K, theta_mode=theta_mode
                )
                t_eval_nr = perf_counter() - t0
                record_nr = {
                    "model": model_id,
                    "n": n,
                    "rep": rep,
                    "method": "Non-random",
                    "error_P": eP_nr,
                    "error_Theta": eT_nr,
                    "error_B": eB_nr,
                    "time_sec": t_nr,
                }
                if detailed_timing:
                    record_nr = attach_timing_breakdown(
                        record_nr,
                        algo_timing=timing_nr,
                        instance_sec=t_instance,
                        metric_sec=t_eval_nr,
                    )
                records.append(record_nr)
                if progress is not None:
                    progress.update("model/n", f"{model_id}/{n}", rep, cfg.reps, "Non-random")

    if progress is not None:
        progress.close()

    return pd.DataFrame(records)


def summarize(df_raw: pd.DataFrame) -> pd.DataFrame:
    return summarize_metrics(df_raw, group_cols=["model", "n"])


def plot_models456_metrics_figure6_like(summary: pd.DataFrame, out_png: Path):
    models = [4, 5, 6]
    ycols = [
        ("error_P_mean", "Error for P"),
        ("error_Theta_mean", "Error for Theta"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.8), sharex=True)
    for i, (ycol, ylabel) in enumerate(ycols):
        for j, model_id in enumerate(models):
            ax = axes[i, j]
            for m in METHODS:
                d = summary[(summary["model"] == model_id) & (summary["method"] == m)].sort_values("n")
                ax.plot(
                    d["n"].values,
                    d[ycol].values,
                    color=METHOD_COLORS[m],
                    linewidth=2.0,
                    marker="o",
                    label=m,
                )
            if i == 0:
                ax.set_title(f"Model {model_id}")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(ycols) - 1:
                ax.set_xlabel("n")
            ax.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_models456_runtime(summary: pd.DataFrame, out_png: Path):
    models = [4, 5, 6]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    for ax, model_id in zip(axes, models):
        for m in METHODS:
            d = summary[(summary["model"] == model_id) & (summary["method"] == m)].sort_values("n")
            ax.plot(
                d["n"].values,
                d["time_mean"].values,
                color=METHOD_COLORS[m],
                linewidth=2.0,
                marker="o",
                label=m,
            )
        ax.set_title(f"Model {model_id}")
        ax.set_xlabel("n")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Runtime (sec)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_n_values(n_str: str):
    return [int(x.strip()) for x in n_str.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Section 7.2 Models 4-6")
    parser.add_argument("--reps", type=int, default=20, help="replications per n")
    parser.add_argument("--seed", type=int, default=2026, help="master random seed")
    parser.add_argument(
        "--n-values",
        type=str,
        default="200,400,600,800,1000,1200",
        help="comma-separated n values",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/reference_1_section7_2/results/exp72_models456_paper_aligned_live",
        help="output directory",
    )
    parser.add_argument("--theta-mode", choices=["exact", "hungarian"], default="exact")
    parser.add_argument(
        "--detailed-timing",
        action="store_true",
        help="record per-step timing breakdown CSVs",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip plotting")
    parser.add_argument("--no-progress", action="store_true", help="disable live progress")
    args, _ = parser.parse_known_args()

    cfg = Exp72Models456Config(
        n_values=parse_n_values(args.n_values),
        model_ids=[4, 5, 6],
        K=3,
        K_prime_fullrank=3,
        K_prime_rankdef=2,
        q=2,
        r=10,
        p=0.7,
        reps=args.reps,
        seed=args.seed,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running Section 7.2 Models 4-6...")
    df_raw = run_experiment72_models456(
        cfg,
        show_progress=(not args.no_progress),
        theta_mode=args.theta_mode,
        detailed_timing=args.detailed_timing,
    )
    df_sum = summarize(df_raw)

    raw_csv = outdir / "sec72_models456_raw_per_rep.csv"
    sum_csv = outdir / "sec72_models456_summary_mean_std.csv"
    df_raw.to_csv(raw_csv, index=False)
    df_sum.to_csv(sum_csv, index=False)

    timing_raw_csv = None
    timing_sum_csv = None
    if args.detailed_timing:
        df_timing_raw = extract_timing_breakdown(df_raw, id_cols=["model", "n", "rep", "method"])
        df_timing_sum = summarize_timing_breakdown(df_timing_raw, group_cols=["model", "n"])
        timing_raw_csv = outdir / "sec72_models456_timing_breakdown_raw.csv"
        timing_sum_csv = outdir / "sec72_models456_timing_breakdown_summary.csv"
        df_timing_raw.to_csv(timing_raw_csv, index=False)
        df_timing_sum.to_csv(timing_sum_csv, index=False)

    if not args.no_plot:
        plot_models456_metrics_figure6_like(df_sum, outdir / "sec72_models456_metrics_figure6_like.png")
        plot_models456_runtime(df_sum, outdir / "sec72_models456_runtime.png")

    print("Done.")
    print(f"Raw CSV     : {raw_csv.resolve()}")
    print(f"Summary CSV : {sum_csv.resolve()}")
    if timing_raw_csv is not None and timing_sum_csv is not None:
        print(f"Timing Raw  : {timing_raw_csv.resolve()}")
        print(f"Timing Sum  : {timing_sum_csv.resolve()}")
    if not args.no_plot:
        print(f"Metrics PNG : {(outdir / 'sec72_models456_metrics_figure6_like.png').resolve()}")
        print(f"Runtime PNG : {(outdir / 'sec72_models456_runtime.png').resolve()}")


if __name__ == "__main__":
    main()
