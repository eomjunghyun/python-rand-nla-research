# -*- coding: utf-8 -*-

"""Section 7.1 Experiment 2 (alpha_n variation) using shared common utilities."""

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
    evaluate_metrics,
    generate_sbm_instance,
    plot_metric_panels,
    plot_runtime,
    run_non_random,
    run_random_projection,
    run_random_sampling,
    summarize_metrics,
)


@dataclass
class Exp2Config:
    alpha_values: list
    n: int = 1152
    K: int = 3
    K_prime: int = 3
    lam: float = 0.5
    q: int = 2
    r: int = 10
    p: float = 0.7
    reps: int = 20
    seed: int = 2026


def run_experiment2(
    cfg: Exp2Config,
    show_progress: bool = True,
    theta_mode: str = "exact",  # "exact" | "hungarian"
) -> pd.DataFrame:
    master_rng = np.random.default_rng(cfg.seed)
    records = []

    total_steps = len(cfg.alpha_values) * cfg.reps * 3
    progress = LiveProgress(total_steps) if show_progress else None

    for alpha_n in cfg.alpha_values:
        for rep in range(1, cfg.reps + 1):
            rep_seed = int(master_rng.integers(1, 2**31 - 1))
            rng = np.random.default_rng(rep_seed)

            A, P, B_true, y_true, Theta_true = generate_sbm_instance(
                n=cfg.n,
                K=cfg.K,
                alpha_n=alpha_n,
                lam=cfg.lam,
                rng=rng,
            )

            t0 = perf_counter()
            Ahat_rp, y_rp = run_random_projection(A, cfg.K, cfg.K_prime, cfg.r, cfg.q, rng)
            t_rp = perf_counter() - t0
            eP_rp, eT_rp, eB_rp = evaluate_metrics(
                Ahat_rp, y_rp, P, B_true, Theta_true, y_true, cfg.K, theta_mode=theta_mode
            )
            records.append(
                {
                    "alpha_n": alpha_n,
                    "rep": rep,
                    "method": "Random Projection",
                    "error_P": eP_rp,
                    "error_Theta": eT_rp,
                    "error_B": eB_rp,
                    "time_sec": t_rp,
                }
            )
            if progress is not None:
                progress.update("alpha_n", alpha_n, rep, cfg.reps, "Random Projection")

            t0 = perf_counter()
            Ahat_rs, y_rs = run_random_sampling(A, cfg.K, cfg.K_prime, cfg.p, rng)
            t_rs = perf_counter() - t0
            eP_rs, eT_rs, eB_rs = evaluate_metrics(
                Ahat_rs, y_rs, P, B_true, Theta_true, y_true, cfg.K, theta_mode=theta_mode
            )
            records.append(
                {
                    "alpha_n": alpha_n,
                    "rep": rep,
                    "method": "Random Sampling",
                    "error_P": eP_rs,
                    "error_Theta": eT_rs,
                    "error_B": eB_rs,
                    "time_sec": t_rs,
                }
            )
            if progress is not None:
                progress.update("alpha_n", alpha_n, rep, cfg.reps, "Random Sampling")

            t0 = perf_counter()
            Ahat_nr, y_nr = run_non_random(A, cfg.K, cfg.K_prime, rng)
            t_nr = perf_counter() - t0
            eP_nr, eT_nr, eB_nr = evaluate_metrics(
                Ahat_nr, y_nr, P, B_true, Theta_true, y_true, cfg.K, theta_mode=theta_mode
            )
            records.append(
                {
                    "alpha_n": alpha_n,
                    "rep": rep,
                    "method": "Non-random",
                    "error_P": eP_nr,
                    "error_Theta": eT_nr,
                    "error_B": eB_nr,
                    "time_sec": t_nr,
                }
            )
            if progress is not None:
                progress.update("alpha_n", alpha_n, rep, cfg.reps, "Non-random")

    if progress is not None:
        progress.close()

    return pd.DataFrame(records)


def summarize(df_raw: pd.DataFrame) -> pd.DataFrame:
    return summarize_metrics(df_raw, group_cols=["alpha_n"])


def parse_alpha_values(alpha_str: str):
    return [float(x.strip()) for x in alpha_str.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Section 7.1 Experiment 2")
    parser.add_argument("--reps", type=int, default=20, help="replications per alpha_n")
    parser.add_argument("--seed", type=int, default=2026, help="master random seed")
    parser.add_argument(
        "--alpha-values",
        type=str,
        default="0.05,0.10,0.15,0.20",
        help="comma-separated alpha_n values",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/section7_1/results/exp2_section7_1_results",
        help="output directory",
    )
    parser.add_argument("--theta-mode", choices=["exact", "hungarian"], default="exact")
    parser.add_argument("--no-plot", action="store_true", help="skip plotting")
    parser.add_argument("--no-progress", action="store_true", help="disable live progress")
    args, _ = parser.parse_known_args()

    cfg = Exp2Config(
        alpha_values=parse_alpha_values(args.alpha_values),
        n=1152,
        K=3,
        K_prime=3,
        lam=0.5,
        q=2,
        r=10,
        p=0.7,
        reps=args.reps,
        seed=args.seed,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running Experiment 2...")
    df_raw = run_experiment2(
        cfg,
        show_progress=(not args.no_progress),
        theta_mode=args.theta_mode,
    )
    df_sum = summarize(df_raw)

    raw_csv = outdir / "exp2_raw_per_rep.csv"
    sum_csv = outdir / "exp2_summary_mean_std.csv"
    df_raw.to_csv(raw_csv, index=False)
    df_sum.to_csv(sum_csv, index=False)

    if not args.no_plot:
        plot_metric_panels(df_sum, x_col="alpha_n", out_png=outdir / "figure2_like_metrics.png")
        plot_runtime(df_sum, x_col="alpha_n", out_png=outdir / "figure2_like_runtime.png")

    print("Done.")
    print(f"Raw CSV     : {raw_csv.resolve()}")
    print(f"Summary CSV : {sum_csv.resolve()}")
    if not args.no_plot:
        print(f"Metrics PNG : {(outdir / 'figure2_like_metrics.png').resolve()}")
        print(f"Runtime PNG : {(outdir / 'figure2_like_runtime.png').resolve()}")


if __name__ == "__main__":
    main()
