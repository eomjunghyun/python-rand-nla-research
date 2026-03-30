# -*- coding: utf-8 -*-

"""Section 7.1 Experiment 3 (K variation) using shared common utilities."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    LiveProgress,
    attach_timing_breakdown,
    evaluate_metrics,
    extract_timing_breakdown,
    generate_sbm_instance,
    plot_metric_panels,
    plot_runtime,
    run_non_random,
    run_random_projection,
    run_random_sampling,
    summarize_metrics,
    summarize_timing_breakdown,
)


@dataclass
class Exp3Config:
    K_values: list
    n: int = 1152
    alpha_n: float = 0.2
    lam: float = 0.5
    q: int = 2
    r: int = 10
    p: float = 0.7
    reps: int = 20
    seed: int = 2026


def run_experiment3(
    cfg: Exp3Config,
    show_progress: bool = True,
    theta_mode: str = "hungarian",
    detailed_timing: bool = False,
) -> pd.DataFrame:
    master_rng = np.random.default_rng(cfg.seed)
    records = []

    total_steps = len(cfg.K_values) * cfg.reps * 3
    progress = LiveProgress(total_steps) if show_progress else None

    for K in cfg.K_values:
        K_prime = K
        for rep in range(1, cfg.reps + 1):
            rep_seed = int(master_rng.integers(1, 2**31 - 1))
            rng = np.random.default_rng(rep_seed)

            t0 = perf_counter()
            A, P, B_true, y_true, Theta_true = generate_sbm_instance(
                n=cfg.n,
                K=K,
                alpha_n=cfg.alpha_n,
                lam=cfg.lam,
                rng=rng,
            )
            t_instance = perf_counter() - t0

            if detailed_timing:
                Ahat_rp, y_rp, timing_rp = run_random_projection(
                    A,
                    K,
                    K_prime,
                    cfg.r,
                    cfg.q,
                    rng,
                    return_timing=True,
                )
                t_rp = timing_rp["algo_total_sec"]
            else:
                t0 = perf_counter()
                Ahat_rp, y_rp = run_random_projection(A, K, K_prime, cfg.r, cfg.q, rng)
                t_rp = perf_counter() - t0
                timing_rp = None

            t0 = perf_counter()
            eP_rp, eT_rp, eB_rp = evaluate_metrics(
                Ahat_rp, y_rp, P, B_true, Theta_true, y_true, K, theta_mode=theta_mode
            )
            t_eval_rp = perf_counter() - t0
            record_rp = {
                "K": K,
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
                progress.update("K", K, rep, cfg.reps, "Random Projection")

            if detailed_timing:
                Ahat_rs, y_rs, timing_rs = run_random_sampling(
                    A,
                    K,
                    K_prime,
                    cfg.p,
                    rng,
                    return_timing=True,
                )
                t_rs = timing_rs["algo_total_sec"]
            else:
                t0 = perf_counter()
                Ahat_rs, y_rs = run_random_sampling(A, K, K_prime, cfg.p, rng)
                t_rs = perf_counter() - t0
                timing_rs = None

            t0 = perf_counter()
            eP_rs, eT_rs, eB_rs = evaluate_metrics(
                Ahat_rs, y_rs, P, B_true, Theta_true, y_true, K, theta_mode=theta_mode
            )
            t_eval_rs = perf_counter() - t0
            record_rs = {
                "K": K,
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
                progress.update("K", K, rep, cfg.reps, "Random Sampling")

            if detailed_timing:
                Ahat_nr, y_nr, timing_nr = run_non_random(
                    A,
                    K,
                    K_prime,
                    rng,
                    return_timing=True,
                )
                t_nr = timing_nr["algo_total_sec"]
            else:
                t0 = perf_counter()
                Ahat_nr, y_nr = run_non_random(A, K, K_prime, rng)
                t_nr = perf_counter() - t0
                timing_nr = None

            t0 = perf_counter()
            eP_nr, eT_nr, eB_nr = evaluate_metrics(
                Ahat_nr, y_nr, P, B_true, Theta_true, y_true, K, theta_mode=theta_mode
            )
            t_eval_nr = perf_counter() - t0
            record_nr = {
                "K": K,
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
                progress.update("K", K, rep, cfg.reps, "Non-random")

    if progress is not None:
        progress.close()

    return pd.DataFrame(records)


def summarize(df_raw: pd.DataFrame) -> pd.DataFrame:
    return summarize_metrics(df_raw, group_cols=["K"])


def parse_k_values(k_str: str):
    return [int(x.strip()) for x in k_str.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Section 7.1 Experiment 3")
    parser.add_argument("--reps", type=int, default=20, help="replications per K")
    parser.add_argument("--seed", type=int, default=2026, help="master random seed")
    parser.add_argument(
        "--K-values",
        type=str,
        default="2,3,4,5,6,7,8",
        help="comma-separated K values",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/section7_1/results/exp3_section7_1_results",
        help="output directory",
    )
    parser.add_argument("--theta-mode", choices=["exact", "hungarian"], default="hungarian")
    parser.add_argument(
        "--detailed-timing",
        action="store_true",
        help="record per-step timing breakdown CSVs",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip plotting")
    parser.add_argument("--no-progress", action="store_true", help="disable live progress")
    args, _ = parser.parse_known_args()

    cfg = Exp3Config(
        K_values=parse_k_values(args.K_values),
        n=1152,
        alpha_n=0.2,
        lam=0.5,
        q=2,
        r=10,
        p=0.7,
        reps=args.reps,
        seed=args.seed,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running Experiment 3...")
    df_raw = run_experiment3(
        cfg,
        show_progress=(not args.no_progress),
        theta_mode=args.theta_mode,
        detailed_timing=args.detailed_timing,
    )
    df_sum = summarize(df_raw)

    raw_csv = outdir / "exp3_raw_per_rep.csv"
    sum_csv = outdir / "exp3_summary_mean_std.csv"
    df_raw.to_csv(raw_csv, index=False)
    df_sum.to_csv(sum_csv, index=False)

    timing_raw_csv = None
    timing_sum_csv = None
    if args.detailed_timing:
        df_timing_raw = extract_timing_breakdown(df_raw, id_cols=["K", "rep", "method"])
        df_timing_sum = summarize_timing_breakdown(df_timing_raw, group_cols=["K"])
        timing_raw_csv = outdir / "exp3_timing_breakdown_raw.csv"
        timing_sum_csv = outdir / "exp3_timing_breakdown_summary.csv"
        df_timing_raw.to_csv(timing_raw_csv, index=False)
        df_timing_sum.to_csv(timing_sum_csv, index=False)

    if not args.no_plot:
        plot_metric_panels(df_sum, x_col="K", out_png=outdir / "figure3_like_metrics.png")
        plot_runtime(df_sum, x_col="K", out_png=outdir / "figure3_like_runtime.png")

    print("Done.")
    print(f"Raw CSV     : {raw_csv.resolve()}")
    print(f"Summary CSV : {sum_csv.resolve()}")
    if timing_raw_csv is not None and timing_sum_csv is not None:
        print(f"Timing Raw  : {timing_raw_csv.resolve()}")
        print(f"Timing Sum  : {timing_sum_csv.resolve()}")
    if not args.no_plot:
        print(f"Metrics PNG : {(outdir / 'figure3_like_metrics.png').resolve()}")
        print(f"Runtime PNG : {(outdir / 'figure3_like_runtime.png').resolve()}")


if __name__ == "__main__":
    main()
