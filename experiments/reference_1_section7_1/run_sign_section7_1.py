from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    EXPERIMENT_META,
    LiveProgress,
    attach_timing_breakdown,
    default_exp1_config,
    default_exp2_config,
    default_exp3_config,
    default_exp4_config,
    default_output_dir,
    evaluate_metrics,
    generate_sbm_instance,
    run_sign_subspace_iteration,
    summarize_metrics,
)


METHOD_ORDER = ["Non-random", "Random Projection", "SIGN", "Random Sampling"]
METHOD_COLORS = {
    "Non-random": "#2ca02c",
    "Random Projection": "#1f77b4",
    "SIGN": "#9467bd",
    "Random Sampling": "#ff7f0e",
}
OUTDIR_DEFAULT = (
    ROOT
    / "experiments"
    / "reference_1_section7_1"
    / "results"
    / "sign_section7_1_wang2025"
)


def _ordered_methods(methods: list[str]) -> list[str]:
    known = [method for method in METHOD_ORDER if method in methods]
    rest = [method for method in methods if method not in known]
    return known + rest


def _group_cols(exp_key: str) -> list[str]:
    if exp_key == "exp1":
        return ["n"]
    if exp_key == "exp2":
        return ["alpha_n"]
    if exp_key == "exp3":
        return ["K"]
    if exp_key == "exp4":
        return ["n"]
    raise ValueError(f"Unknown experiment key: {exp_key}")


def _id_cols(exp_key: str) -> list[str]:
    if exp_key == "exp1":
        return ["n", "rep", "method"]
    if exp_key == "exp2":
        return ["alpha_n", "rep", "method"]
    if exp_key == "exp3":
        return ["K", "rep", "method"]
    if exp_key == "exp4":
        return ["n", "alpha_n", "rep", "method"]
    raise ValueError(f"Unknown experiment key: {exp_key}")


def _summarize_exp(exp_key: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_metrics(df_raw, _group_cols(exp_key))
    if exp_key == "exp4" and "alpha_n" in df_raw.columns:
        alpha_summary = df_raw.groupby(["n", "method"], as_index=False).agg(
            alpha_n_mean=("alpha_n", "mean")
        )
        summary = summary.merge(alpha_summary, on=["n", "method"], how="left")
    return summary


def _run_sign_job(
    *,
    A: np.ndarray,
    P: np.ndarray,
    B_true: np.ndarray,
    Theta_true: np.ndarray,
    y_true: np.ndarray,
    K: int,
    K_prime: int,
    r: int,
    sign_k: int,
    rng: np.random.Generator,
    theta_mode: str,
    base_record: dict[str, object],
    instance_sec: float,
) -> dict[str, object]:
    A_hat, y_pred, algo_timing = run_sign_subspace_iteration(
        A,
        K,
        K_prime,
        r,
        sign_k,
        rng,
        return_timing=True,
    )
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
            "method": "SIGN",
            "error_P": err_P,
            "error_Theta": err_Theta,
            "error_B": err_B,
            "time_sec": float(algo_timing["algo_total_sec"]),
        }
    )
    return attach_timing_breakdown(
        record,
        algo_timing=algo_timing,
        instance_sec=instance_sec,
        metric_sec=metric_sec,
    )


def run_sign_experiment(exp_key: str, cfg, show_progress: bool = True) -> pd.DataFrame:
    master_rng = np.random.default_rng(cfg.seed)
    records = []

    if exp_key == "exp1":
        total_steps = len(cfg.n_values) * cfg.reps
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
                records.append(
                    _run_sign_job(
                        A=A,
                        P=P,
                        B_true=B_true,
                        Theta_true=Theta_true,
                        y_true=y_true,
                        K=cfg.K,
                        K_prime=cfg.K_prime,
                        r=cfg.r,
                        sign_k=cfg.q,
                        rng=rng,
                        theta_mode="exact",
                        base_record={"n": n, "rep": rep},
                        instance_sec=instance_sec,
                    )
                )
                if progress is not None:
                    progress.update("n", n, rep, cfg.reps, "SIGN")

    elif exp_key == "exp2":
        total_steps = len(cfg.alpha_values) * cfg.reps
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
                records.append(
                    _run_sign_job(
                        A=A,
                        P=P,
                        B_true=B_true,
                        Theta_true=Theta_true,
                        y_true=y_true,
                        K=cfg.K,
                        K_prime=cfg.K_prime,
                        r=cfg.r,
                        sign_k=cfg.q,
                        rng=rng,
                        theta_mode="exact",
                        base_record={"alpha_n": alpha_n, "rep": rep},
                        instance_sec=instance_sec,
                    )
                )
                if progress is not None:
                    progress.update("alpha_n", alpha_n, rep, cfg.reps, "SIGN")

    elif exp_key == "exp3":
        total_steps = len(cfg.K_values) * cfg.reps
        progress = LiveProgress(total_steps) if show_progress else None
        for K in cfg.K_values:
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
                records.append(
                    _run_sign_job(
                        A=A,
                        P=P,
                        B_true=B_true,
                        Theta_true=Theta_true,
                        y_true=y_true,
                        K=K,
                        K_prime=K,
                        r=cfg.r,
                        sign_k=cfg.q,
                        rng=rng,
                        theta_mode="hungarian",
                        base_record={"K": K, "rep": rep},
                        instance_sec=instance_sec,
                    )
                )
                if progress is not None:
                    progress.update("K", K, rep, cfg.reps, "SIGN")

    elif exp_key == "exp4":
        total_steps = len(cfg.n_values) * cfg.reps
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
                records.append(
                    _run_sign_job(
                        A=A,
                        P=P,
                        B_true=B_true,
                        Theta_true=Theta_true,
                        y_true=y_true,
                        K=cfg.K,
                        K_prime=cfg.K_prime,
                        r=cfg.r,
                        sign_k=cfg.q,
                        rng=rng,
                        theta_mode="exact",
                        base_record={"n": n, "alpha_n": alpha_n, "rep": rep},
                        instance_sec=instance_sec,
                    )
                )
                if progress is not None:
                    progress.update("n", n, rep, cfg.reps, "SIGN")

    else:
        raise ValueError(f"Unknown experiment key: {exp_key}")

    if progress is not None:
        progress.close()
    return pd.DataFrame(records)


def load_baseline_raw(exp_key: str) -> pd.DataFrame:
    meta = EXPERIMENT_META[exp_key]
    path = default_output_dir(exp_key) / meta["raw_csv"]
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline raw CSV: {path}")
    return pd.read_csv(path)


def plot_metric_panels_dynamic(summary: pd.DataFrame, exp_key: str, out_png: Path) -> None:
    x_col = EXPERIMENT_META[exp_key]["x_col"]
    methods = _ordered_methods(summary["method"].drop_duplicates().tolist())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ycols = [
        ("error_P_mean", "Error for P"),
        ("error_Theta_mean", "Error for Theta"),
        ("error_B_mean", "Error for B"),
    ]
    for ax, (y_col, y_label) in zip(axes, ycols):
        for method in methods:
            block = summary[summary["method"] == method].sort_values(x_col)
            ax.plot(
                block[x_col].values,
                block[y_col].values,
                color=METHOD_COLORS.get(method, "#555555"),
                linewidth=2.0,
                marker="o",
                label=method,
            )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_dynamic(summary: pd.DataFrame, exp_key: str, out_png: Path) -> None:
    x_col = EXPERIMENT_META[exp_key]["x_col"]
    methods = _ordered_methods(summary["method"].drop_duplicates().tolist())
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
    for method in methods:
        block = summary[summary["method"] == method].sort_values(x_col)
        ax.plot(
            block[x_col].values,
            block["time_mean"].values,
            color=METHOD_COLORS.get(method, "#555555"),
            linewidth=2.0,
            marker="o",
            label=method,
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel("Runtime (sec)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_existing_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[col for col in cols if col in df.columns]].copy()


def _format_float(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_format_float(v) for v in row) + " |")
    return "\n".join(lines)


def build_report(outdir: Path, artifacts: dict[str, dict[str, Path]], configs: dict[str, object]) -> Path:
    combined_raw_frames = []
    combined_summary_frames = []
    endpoint_rows = []
    delta_rows = []

    for exp_key, paths in artifacts.items():
        raw = pd.read_csv(paths["combined_raw"])
        summary = pd.read_csv(paths["combined_summary"])
        raw = raw.copy()
        summary = summary.copy()
        raw["experiment"] = exp_key
        summary["experiment"] = exp_key
        combined_raw_frames.append(raw)
        combined_summary_frames.append(summary)

        x_col = EXPERIMENT_META[exp_key]["x_col"]
        endpoint = summary[summary[x_col] == summary[x_col].max()].copy()
        endpoint_rows.append(
            endpoint[
                [
                    "experiment",
                    x_col,
                    "method",
                    "error_P_mean",
                    "error_Theta_mean",
                    "error_B_mean",
                    "time_mean",
                ]
            ].rename(columns={x_col: "endpoint"})
        )

        pivot = summary.pivot_table(
            index=x_col,
            columns="method",
            values=["error_P_mean", "error_Theta_mean", "error_B_mean", "time_mean"],
        )
        if "SIGN" in summary["method"].values and "Random Projection" in summary["method"].values:
            for metric in ["error_P_mean", "error_Theta_mean", "error_B_mean", "time_mean"]:
                diff = pivot[(metric, "SIGN")] - pivot[(metric, "Random Projection")]
                delta_rows.append(
                    {
                        "experiment": exp_key,
                        "metric": metric.replace("_mean", ""),
                        "mean_SIGN_minus_RP": float(diff.mean()),
                    }
                )

    combined_raw_all = pd.concat(combined_raw_frames, ignore_index=True)
    aggregate = (
        combined_raw_all.groupby(["experiment", "method"], as_index=False)
        .agg(
            error_P_mean=("error_P", "mean"),
            error_Theta_mean=("error_Theta", "mean"),
            error_B_mean=("error_B", "mean"),
            time_mean=("time_sec", "mean"),
        )
        .sort_values(["experiment", "method"])
    )
    aggregate["method"] = pd.Categorical(
        aggregate["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    aggregate = aggregate.sort_values(["experiment", "method"]).astype({"method": str})

    endpoint_table = pd.concat(endpoint_rows, ignore_index=True)
    endpoint_table["method"] = pd.Categorical(
        endpoint_table["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    endpoint_table = endpoint_table.sort_values(["experiment", "method"]).astype({"method": str})
    delta_table = pd.DataFrame(delta_rows)

    report_path = outdir / "sign_section7_1_report.md"
    plot_lines = []
    for exp_key, paths in artifacts.items():
        metrics_rel = paths["metrics_png"].relative_to(outdir)
        runtime_rel = paths["runtime_png"].relative_to(outdir)
        plot_lines.extend(
            [
                f"### {exp_key}",
                "",
                f"![{exp_key} metrics]({metrics_rel.as_posix()})",
                "",
                f"![{exp_key} runtime]({runtime_rel.as_posix()})",
                "",
            ]
        )

    config_payload = {
        key: asdict(value) if hasattr(value, "__dataclass_fields__") else str(value)
        for key, value in configs.items()
    }
    config_json = json.dumps(config_payload, ensure_ascii=False, indent=2)

    report = f"""# Wang 2025 SIGN 방법론의 Section 7.1 적용 보고서

## 목적

사용자가 제공한 Wang et al. (2025)의 SIGN(generalized Nystrom method with subspace iteration)을 Reference 1 Section 7.1 SBM 실험에 추가 적용했다. 기존 7.1 실험의 `Random Projection`, `Random Sampling`, `Non-random` 결과는 그대로 두고, 같은 난수 seed 흐름으로 생성되는 SBM 인스턴스에 `SIGN`을 추가 실행한 뒤 비교했다.

## 구현 메모

- 구현 함수: `src.common.run_sign_subspace_iteration`
- 실행 스크립트: `experiments/reference_1_section7_1/run_sign_section7_1.py`
- 결과 폴더: `{outdir}`
- SIGN 설정: 기존 Section 7.1의 `q=2`, `r=10`을 각각 SIGN power parameter `k=2`, oversampling `r=10`으로 사용했다.
- Section 7.1의 행렬은 대칭 SBM adjacency이므로, 논문의 비대칭 행렬용 SIGN은 여기서 `A.T`와 `A`를 번갈아 곱하는 symmetric subspace iteration으로 작동한다.
- 저랭크 근사 `A_hat`은 논문의 SIGN form으로 만든 뒤 Section 7.1 metric 계산에 맞춰 대칭화했다.
- clustering embedding은 최종 SIGN left basis `Q` 위의 작은 Rayleigh-Ritz 행렬 `Q.T @ A @ Q`에서 얻었다.

## 실험 설정

```json
{config_json}
```

## 전체 평균 요약

아래 표는 각 experiment의 모든 grid point와 반복을 평균낸 값이다.

{dataframe_to_markdown(aggregate)}

## 마지막 grid point 요약

각 experiment에서 가장 큰 x값, 즉 Exp1/Exp4는 최대 `n`, Exp2는 최대 `alpha_n`, Exp3는 최대 `K`에서의 summary다.

{dataframe_to_markdown(endpoint_table)}

## SIGN과 Random Projection의 평균 차이

음수는 SIGN이 Random Projection보다 해당 metric 또는 runtime이 작다는 뜻이다.

{dataframe_to_markdown(delta_table)}

## 그림

{chr(10).join(plot_lines)}

## 해석

Section 7.1은 원래 대칭 그래프 adjacency에 대한 spectral clustering 실험이다. 따라서 SIGN의 주된 장점인 비대칭 행렬에서 row/column space를 동시에 개선하는 효과는 완전히 드러나지 않는다. 이 실험에서 SIGN은 기존 random projection보다 한 번 더 구조화된 양방향 subspace iteration으로 볼 수 있다.

결과 해석에서는 `error_Theta`를 가장 우선해서 보면 된다. `error_P`와 `error_B`는 저랭크 reconstruction 품질의 영향을 많이 받는다. 특히 대칭 SBM에서는 spectral embedding만 충분히 좋으면 clustering은 안정적일 수 있지만, `A_hat`의 operator-norm 근사 품질은 방법별 reconstruction 방식에 더 민감하게 움직인다.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_all(outdir: Path, show_progress: bool = True) -> dict[str, dict[str, Path]]:
    outdir.mkdir(parents=True, exist_ok=True)
    configs = {
        "exp1": default_exp1_config(),
        "exp2": default_exp2_config(),
        "exp3": default_exp3_config(),
        "exp4": default_exp4_config(),
    }
    artifacts: dict[str, dict[str, Path]] = {}

    for exp_key, cfg in configs.items():
        exp_outdir = outdir / exp_key
        exp_outdir.mkdir(parents=True, exist_ok=True)

        sign_raw = run_sign_experiment(exp_key, cfg, show_progress=show_progress)
        sign_summary = _summarize_exp(exp_key, sign_raw)
        baseline_raw = load_baseline_raw(exp_key)
        combined_raw = pd.concat([baseline_raw, sign_raw], ignore_index=True, sort=False)
        combined_summary = _summarize_exp(exp_key, combined_raw)

        timing_cols = _id_cols(exp_key) + [
            col for col in sign_raw.columns if col.startswith("sign_") or col.endswith("_sec")
        ]
        sign_timing = _select_existing_cols(sign_raw, timing_cols)

        paths = {
            "sign_raw": exp_outdir / f"{exp_key}_sign_raw.csv",
            "sign_summary": exp_outdir / f"{exp_key}_sign_summary.csv",
            "sign_timing": exp_outdir / f"{exp_key}_sign_timing.csv",
            "combined_raw": exp_outdir / f"{exp_key}_combined_raw.csv",
            "combined_summary": exp_outdir / f"{exp_key}_combined_summary.csv",
            "metrics_png": exp_outdir / f"{exp_key}_sign_metrics.png",
            "runtime_png": exp_outdir / f"{exp_key}_sign_runtime.png",
        }
        sign_raw.to_csv(paths["sign_raw"], index=False)
        sign_summary.to_csv(paths["sign_summary"], index=False)
        sign_timing.to_csv(paths["sign_timing"], index=False)
        combined_raw.to_csv(paths["combined_raw"], index=False)
        combined_summary.to_csv(paths["combined_summary"], index=False)
        plot_metric_panels_dynamic(combined_summary, exp_key, paths["metrics_png"])
        plot_runtime_dynamic(combined_summary, exp_key, paths["runtime_png"])
        artifacts[exp_key] = paths

    manifest = {
        "outdir": str(outdir),
        "configs": {
            key: asdict(value) if hasattr(value, "__dataclass_fields__") else str(value)
            for key, value in configs.items()
        },
        "artifacts": {
            key: {name: str(path) for name, path in paths.items()}
            for key, paths in artifacts.items()
        },
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = build_report(outdir, artifacts, configs)
    manifest["report"] = str(report_path)
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wang 2025 SIGN on Reference 1 Section 7.1 experiments."
    )
    parser.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all(args.outdir, show_progress=not args.no_progress)
    print(f"SIGN Section 7.1 outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
