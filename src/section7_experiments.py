from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.common import (
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECTION7_RESULTS_ROOT = PROJECT_ROOT / "experiments" / "reference_1_section7_1" / "results"

EXPERIMENT_META = {
    "exp1": {
        "label": "Exp1 (n variation)",
        "outdir_name": "exp1_paper_aligned_live",
        "x_col": "n",
        "raw_csv": "exp1_raw_per_rep.csv",
        "summary_csv": "exp1_summary_mean_std.csv",
        "timing_raw_csv": "exp1_timing_breakdown_raw.csv",
        "timing_summary_csv": "exp1_timing_breakdown_summary.csv",
        "metrics_png": "figure1_like_metrics.png",
        "runtime_png": "figure1_like_runtime.png",
    },
    "exp2": {
        "label": "Exp2 (alpha_n variation)",
        "outdir_name": "exp2_section7_1_results",
        "x_col": "alpha_n",
        "raw_csv": "exp2_raw_per_rep.csv",
        "summary_csv": "exp2_summary_mean_std.csv",
        "timing_raw_csv": "exp2_timing_breakdown_raw.csv",
        "timing_summary_csv": "exp2_timing_breakdown_summary.csv",
        "metrics_png": "figure2_like_metrics.png",
        "runtime_png": "figure2_like_runtime.png",
    },
    "exp3": {
        "label": "Exp3 (K variation)",
        "outdir_name": "exp3_section7_1_results",
        "x_col": "K",
        "raw_csv": "exp3_raw_per_rep.csv",
        "summary_csv": "exp3_summary_mean_std.csv",
        "timing_raw_csv": "exp3_timing_breakdown_raw.csv",
        "timing_summary_csv": "exp3_timing_breakdown_summary.csv",
        "metrics_png": "figure3_like_metrics.png",
        "runtime_png": "figure3_like_runtime.png",
    },
    "exp4": {
        "label": "Exp4 (n variation, alpha_n = 2/sqrt(n))",
        "outdir_name": "exp4_section7_1_results",
        "x_col": "n",
        "raw_csv": "exp4_raw_per_rep.csv",
        "summary_csv": "exp4_summary_mean_std.csv",
        "timing_raw_csv": "exp4_timing_breakdown_raw.csv",
        "timing_summary_csv": "exp4_timing_breakdown_summary.csv",
        "metrics_png": "figure4_like_metrics.png",
        "runtime_png": "figure4_like_runtime.png",
    },
}


@dataclass
class Exp1Config:
    n_values: list[int]
    K: int = 3
    K_prime: int = 3
    alpha_n: float = 0.2
    lam: float = 0.5
    q: int = 2
    r: int = 10
    p: float = 0.7
    reps: int = 20
    seed: int = 2026


@dataclass
class Exp2Config:
    alpha_values: list[float]
    n: int = 1152
    K: int = 3
    K_prime: int = 3
    lam: float = 0.5
    q: int = 2
    r: int = 10
    p: float = 0.7
    reps: int = 20
    seed: int = 2026


@dataclass
class Exp3Config:
    K_values: list[int]
    n: int = 1152
    alpha_n: float = 0.2
    lam: float = 0.5
    q: int = 2
    r: int = 10
    p: float = 0.7
    reps: int = 20
    seed: int = 2026


@dataclass
class Exp4Config:
    n_values: list[int]
    K: int = 2
    K_prime: int = 2
    lam: float = 0.5
    q: int = 2
    r: int = 10
    p: float = 0.7
    reps: int = 20
    seed: int = 2026


@dataclass
class SavedExperimentOutputs:
    exp_key: str
    label: str
    outdir: Path
    raw_csv: Path
    summary_csv: Path
    timing_raw_csv: Path | None
    timing_summary_csv: Path | None
    metrics_png: Path | None
    runtime_png: Path | None
    raw_rows: int
    summary_rows: int

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.as_dict()])


def default_exp1_config() -> Exp1Config:
    return Exp1Config(n_values=[200, 400, 600, 800, 1000, 1200])


def default_exp2_config() -> Exp2Config:
    return Exp2Config(alpha_values=[0.05, 0.10, 0.15, 0.20])


def default_exp3_config() -> Exp3Config:
    return Exp3Config(K_values=[2, 3, 4, 5, 6, 7, 8])


def default_exp4_config() -> Exp4Config:
    return Exp4Config(n_values=[200, 400, 600, 800, 1000, 1200])


def default_output_dir(exp_key: str) -> Path:
    meta = EXPERIMENT_META[exp_key]
    return SECTION7_RESULTS_ROOT / meta["outdir_name"]


def parse_int_values(value_text: str) -> list[int]:
    return [int(item.strip()) for item in value_text.split(",") if item.strip()]


def parse_float_values(value_text: str) -> list[float]:
    return [float(item.strip()) for item in value_text.split(",") if item.strip()]


def _run_method_job(
    method_name: str,
    job,
    base_record: dict[str, object],
    P: np.ndarray,
    B_true: np.ndarray,
    Theta_true: np.ndarray,
    y_true: np.ndarray,
    K: int,
    theta_mode: str,
    detailed_timing: bool,
    instance_sec: float | None,
) -> dict[str, object]:
    if detailed_timing:
        A_hat, y_pred, algo_timing = job(return_timing=True)
        time_sec = float(algo_timing["algo_total_sec"])
    else:
        t0 = perf_counter()
        A_hat, y_pred = job(return_timing=False)
        time_sec = perf_counter() - t0
        algo_timing = None

    t0 = perf_counter()
    err_P, err_Theta, err_B = evaluate_metrics(
        A_hat,
        y_pred,
        P,
        B_true,
        Theta_true,
        y_true,
        K,
        theta_mode=theta_mode,
    )
    metric_sec = perf_counter() - t0

    record = dict(base_record)
    record.update(
        {
            "method": method_name,
            "error_P": err_P,
            "error_Theta": err_Theta,
            "error_B": err_B,
            "time_sec": time_sec,
        }
    )
    if detailed_timing:
        record = attach_timing_breakdown(
            record,
            algo_timing=algo_timing,
            instance_sec=instance_sec,
            metric_sec=metric_sec,
        )
    return record


def run_experiment1(
    cfg: Exp1Config,
    show_progress: bool = True,
    theta_mode: str = "exact",
    detailed_timing: bool = False,
) -> pd.DataFrame:
    master_rng = np.random.default_rng(cfg.seed)
    records = []
    total_steps = len(cfg.n_values) * cfg.reps * 3
    progress = LiveProgress(total_steps) if show_progress else None

    for n in cfg.n_values:
        for rep in range(1, cfg.reps + 1):
            rep_seed = int(master_rng.integers(1, 2**31 - 1))
            rng = np.random.default_rng(rep_seed)

            t0 = perf_counter()
            A, P, B_true, y_true, Theta_true = generate_sbm_instance(
                n=n,
                K=cfg.K,
                alpha_n=cfg.alpha_n,
                lam=cfg.lam,
                rng=rng,
            )
            instance_sec = perf_counter() - t0

            base_record = {"n": n, "rep": rep}
            jobs = [
                (
                    "Random Projection",
                    lambda return_timing=False: run_random_projection(
                        A, cfg.K, cfg.K_prime, cfg.r, cfg.q, rng, return_timing=return_timing
                    ),
                ),
                (
                    "Random Sampling",
                    lambda return_timing=False: run_random_sampling(
                        A, cfg.K, cfg.K_prime, cfg.p, rng, return_timing=return_timing
                    ),
                ),
                (
                    "Non-random",
                    lambda return_timing=False: run_non_random(
                        A, cfg.K, cfg.K_prime, rng, return_timing=return_timing
                    ),
                ),
            ]

            for method_name, job in jobs:
                records.append(
                    _run_method_job(
                        method_name=method_name,
                        job=job,
                        base_record=base_record,
                        P=P,
                        B_true=B_true,
                        Theta_true=Theta_true,
                        y_true=y_true,
                        K=cfg.K,
                        theta_mode=theta_mode,
                        detailed_timing=detailed_timing,
                        instance_sec=instance_sec,
                    )
                )
                if progress is not None:
                    progress.update("n", n, rep, cfg.reps, method_name)

    if progress is not None:
        progress.close()
    return pd.DataFrame(records)


def run_experiment2(
    cfg: Exp2Config,
    show_progress: bool = True,
    theta_mode: str = "exact",
    detailed_timing: bool = False,
) -> pd.DataFrame:
    master_rng = np.random.default_rng(cfg.seed)
    records = []
    total_steps = len(cfg.alpha_values) * cfg.reps * 3
    progress = LiveProgress(total_steps) if show_progress else None

    for alpha_n in cfg.alpha_values:
        for rep in range(1, cfg.reps + 1):
            rep_seed = int(master_rng.integers(1, 2**31 - 1))
            rng = np.random.default_rng(rep_seed)

            t0 = perf_counter()
            A, P, B_true, y_true, Theta_true = generate_sbm_instance(
                n=cfg.n,
                K=cfg.K,
                alpha_n=alpha_n,
                lam=cfg.lam,
                rng=rng,
            )
            instance_sec = perf_counter() - t0

            base_record = {"alpha_n": alpha_n, "rep": rep}
            jobs = [
                (
                    "Random Projection",
                    lambda return_timing=False: run_random_projection(
                        A, cfg.K, cfg.K_prime, cfg.r, cfg.q, rng, return_timing=return_timing
                    ),
                ),
                (
                    "Random Sampling",
                    lambda return_timing=False: run_random_sampling(
                        A, cfg.K, cfg.K_prime, cfg.p, rng, return_timing=return_timing
                    ),
                ),
                (
                    "Non-random",
                    lambda return_timing=False: run_non_random(
                        A, cfg.K, cfg.K_prime, rng, return_timing=return_timing
                    ),
                ),
            ]

            for method_name, job in jobs:
                records.append(
                    _run_method_job(
                        method_name=method_name,
                        job=job,
                        base_record=base_record,
                        P=P,
                        B_true=B_true,
                        Theta_true=Theta_true,
                        y_true=y_true,
                        K=cfg.K,
                        theta_mode=theta_mode,
                        detailed_timing=detailed_timing,
                        instance_sec=instance_sec,
                    )
                )
                if progress is not None:
                    progress.update("alpha_n", alpha_n, rep, cfg.reps, method_name)

    if progress is not None:
        progress.close()
    return pd.DataFrame(records)


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
            instance_sec = perf_counter() - t0

            base_record = {"K": K, "rep": rep}
            jobs = [
                (
                    "Random Projection",
                    lambda return_timing=False: run_random_projection(
                        A, K, K_prime, cfg.r, cfg.q, rng, return_timing=return_timing
                    ),
                ),
                (
                    "Random Sampling",
                    lambda return_timing=False: run_random_sampling(
                        A, K, K_prime, cfg.p, rng, return_timing=return_timing
                    ),
                ),
                (
                    "Non-random",
                    lambda return_timing=False: run_non_random(
                        A, K, K_prime, rng, return_timing=return_timing
                    ),
                ),
            ]

            for method_name, job in jobs:
                records.append(
                    _run_method_job(
                        method_name=method_name,
                        job=job,
                        base_record=base_record,
                        P=P,
                        B_true=B_true,
                        Theta_true=Theta_true,
                        y_true=y_true,
                        K=K,
                        theta_mode=theta_mode,
                        detailed_timing=detailed_timing,
                        instance_sec=instance_sec,
                    )
                )
                if progress is not None:
                    progress.update("K", K, rep, cfg.reps, method_name)

    if progress is not None:
        progress.close()
    return pd.DataFrame(records)


def run_experiment4(
    cfg: Exp4Config,
    show_progress: bool = True,
    theta_mode: str = "exact",
    detailed_timing: bool = False,
) -> pd.DataFrame:
    master_rng = np.random.default_rng(cfg.seed)
    records = []
    total_steps = len(cfg.n_values) * cfg.reps * 3
    progress = LiveProgress(total_steps) if show_progress else None

    for n in cfg.n_values:
        alpha_n = 2.0 / np.sqrt(n)
        for rep in range(1, cfg.reps + 1):
            rep_seed = int(master_rng.integers(1, 2**31 - 1))
            rng = np.random.default_rng(rep_seed)

            t0 = perf_counter()
            A, P, B_true, y_true, Theta_true = generate_sbm_instance(
                n=n,
                K=cfg.K,
                alpha_n=alpha_n,
                lam=cfg.lam,
                rng=rng,
            )
            instance_sec = perf_counter() - t0

            base_record = {"n": n, "alpha_n": alpha_n, "rep": rep}
            jobs = [
                (
                    "Random Projection",
                    lambda return_timing=False: run_random_projection(
                        A, cfg.K, cfg.K_prime, cfg.r, cfg.q, rng, return_timing=return_timing
                    ),
                ),
                (
                    "Random Sampling",
                    lambda return_timing=False: run_random_sampling(
                        A, cfg.K, cfg.K_prime, cfg.p, rng, return_timing=return_timing
                    ),
                ),
                (
                    "Non-random",
                    lambda return_timing=False: run_non_random(
                        A, cfg.K, cfg.K_prime, rng, return_timing=return_timing
                    ),
                ),
            ]

            for method_name, job in jobs:
                records.append(
                    _run_method_job(
                        method_name=method_name,
                        job=job,
                        base_record=base_record,
                        P=P,
                        B_true=B_true,
                        Theta_true=Theta_true,
                        y_true=y_true,
                        K=cfg.K,
                        theta_mode=theta_mode,
                        detailed_timing=detailed_timing,
                        instance_sec=instance_sec,
                    )
                )
                if progress is not None:
                    progress.update("n", n, rep, cfg.reps, method_name)

    if progress is not None:
        progress.close()
    return pd.DataFrame(records)


def summarize_experiment1(df_raw: pd.DataFrame) -> pd.DataFrame:
    return summarize_metrics(df_raw, group_cols=["n"])


def summarize_experiment2(df_raw: pd.DataFrame) -> pd.DataFrame:
    return summarize_metrics(df_raw, group_cols=["alpha_n"])


def summarize_experiment3(df_raw: pd.DataFrame) -> pd.DataFrame:
    return summarize_metrics(df_raw, group_cols=["K"])


def summarize_experiment4(df_raw: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_metrics(df_raw, group_cols=["n"])
    alpha_summary = df_raw.groupby(["n", "method"], as_index=False).agg(
        alpha_n_mean=("alpha_n", "mean")
    )
    return summary.merge(alpha_summary, on=["n", "method"], how="left")


def save_experiment_outputs(
    exp_key: str,
    df_raw: pd.DataFrame,
    df_summary: pd.DataFrame,
    outdir: str | Path | None = None,
    detailed_timing: bool = False,
    plot_basics: bool = True,
) -> SavedExperimentOutputs:
    meta = EXPERIMENT_META[exp_key]
    out_path = Path(outdir) if outdir is not None else default_output_dir(exp_key)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_csv = out_path / meta["raw_csv"]
    summary_csv = out_path / meta["summary_csv"]
    df_raw.to_csv(raw_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    timing_raw_csv = None
    timing_summary_csv = None
    if detailed_timing:
        if exp_key == "exp1":
            id_cols = ["n", "rep", "method"]
            group_cols = ["n"]
        elif exp_key == "exp2":
            id_cols = ["alpha_n", "rep", "method"]
            group_cols = ["alpha_n"]
        elif exp_key == "exp3":
            id_cols = ["K", "rep", "method"]
            group_cols = ["K"]
        else:
            id_cols = ["n", "alpha_n", "rep", "method"]
            group_cols = ["n", "alpha_n"]

        df_timing_raw = extract_timing_breakdown(df_raw, id_cols=id_cols)
        df_timing_summary = summarize_timing_breakdown(df_timing_raw, group_cols=group_cols)
        timing_raw_csv = out_path / meta["timing_raw_csv"]
        timing_summary_csv = out_path / meta["timing_summary_csv"]
        df_timing_raw.to_csv(timing_raw_csv, index=False)
        df_timing_summary.to_csv(timing_summary_csv, index=False)

    metrics_png = None
    runtime_png = None
    if plot_basics:
        metrics_png = out_path / meta["metrics_png"]
        runtime_png = out_path / meta["runtime_png"]
        plot_metric_panels(df_summary, x_col=meta["x_col"], out_png=metrics_png)
        plot_runtime(df_summary, x_col=meta["x_col"], out_png=runtime_png)

    return SavedExperimentOutputs(
        exp_key=exp_key,
        label=meta["label"],
        outdir=out_path,
        raw_csv=raw_csv,
        summary_csv=summary_csv,
        timing_raw_csv=timing_raw_csv,
        timing_summary_csv=timing_summary_csv,
        metrics_png=metrics_png,
        runtime_png=runtime_png,
        raw_rows=len(df_raw),
        summary_rows=len(df_summary),
    )


def run_and_save_experiment1(
    cfg: Exp1Config | None = None,
    outdir: str | Path | None = None,
    show_progress: bool = True,
    theta_mode: str = "exact",
    detailed_timing: bool = True,
    plot_basics: bool = True,
) -> SavedExperimentOutputs:
    cfg = cfg or default_exp1_config()
    df_raw = run_experiment1(
        cfg,
        show_progress=show_progress,
        theta_mode=theta_mode,
        detailed_timing=detailed_timing,
    )
    df_summary = summarize_experiment1(df_raw)
    return save_experiment_outputs(
        "exp1",
        df_raw=df_raw,
        df_summary=df_summary,
        outdir=outdir,
        detailed_timing=detailed_timing,
        plot_basics=plot_basics,
    )


def run_and_save_experiment2(
    cfg: Exp2Config | None = None,
    outdir: str | Path | None = None,
    show_progress: bool = True,
    theta_mode: str = "exact",
    detailed_timing: bool = True,
    plot_basics: bool = True,
) -> SavedExperimentOutputs:
    cfg = cfg or default_exp2_config()
    df_raw = run_experiment2(
        cfg,
        show_progress=show_progress,
        theta_mode=theta_mode,
        detailed_timing=detailed_timing,
    )
    df_summary = summarize_experiment2(df_raw)
    return save_experiment_outputs(
        "exp2",
        df_raw=df_raw,
        df_summary=df_summary,
        outdir=outdir,
        detailed_timing=detailed_timing,
        plot_basics=plot_basics,
    )


def run_and_save_experiment3(
    cfg: Exp3Config | None = None,
    outdir: str | Path | None = None,
    show_progress: bool = True,
    theta_mode: str = "hungarian",
    detailed_timing: bool = True,
    plot_basics: bool = True,
) -> SavedExperimentOutputs:
    cfg = cfg or default_exp3_config()
    df_raw = run_experiment3(
        cfg,
        show_progress=show_progress,
        theta_mode=theta_mode,
        detailed_timing=detailed_timing,
    )
    df_summary = summarize_experiment3(df_raw)
    return save_experiment_outputs(
        "exp3",
        df_raw=df_raw,
        df_summary=df_summary,
        outdir=outdir,
        detailed_timing=detailed_timing,
        plot_basics=plot_basics,
    )


def run_and_save_experiment4(
    cfg: Exp4Config | None = None,
    outdir: str | Path | None = None,
    show_progress: bool = True,
    theta_mode: str = "exact",
    detailed_timing: bool = True,
    plot_basics: bool = True,
) -> SavedExperimentOutputs:
    cfg = cfg or default_exp4_config()
    df_raw = run_experiment4(
        cfg,
        show_progress=show_progress,
        theta_mode=theta_mode,
        detailed_timing=detailed_timing,
    )
    df_summary = summarize_experiment4(df_raw)
    return save_experiment_outputs(
        "exp4",
        df_raw=df_raw,
        df_summary=df_summary,
        outdir=outdir,
        detailed_timing=detailed_timing,
        plot_basics=plot_basics,
    )
