# -*- coding: utf-8 -*-

"""Run Reference 1 Section 7.1 experiments with the five-method comparison."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    EXPERIMENT_META,
    METHODS,
    default_exp1_config,
    default_exp2_config,
    default_exp3_config,
    default_exp4_config,
    run_experiment1,
    run_experiment2,
    run_experiment3,
    run_experiment4,
    save_experiment_outputs,
    summarize_experiment1,
    summarize_experiment2,
    summarize_experiment3,
    summarize_experiment4,
)


RUNNERS = {
    "exp1": (default_exp1_config, run_experiment1, summarize_experiment1, "exact"),
    "exp2": (default_exp2_config, run_experiment2, summarize_experiment2, "exact"),
    "exp3": (default_exp3_config, run_experiment3, summarize_experiment3, "hungarian"),
    "exp4": (default_exp4_config, run_experiment4, summarize_experiment4, "exact"),
}


def _format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_format_value(v) for v in row) + " |")
    return "\n".join(lines)


def build_report(outdir: Path, outputs: dict[str, object], configs: dict[str, object]) -> Path:
    coverage_rows = []
    endpoint_rows = []
    artifact_rows = []

    for exp_key, saved in outputs.items():
        summary = pd.read_csv(saved.summary_csv)
        x_col = EXPERIMENT_META[exp_key]["x_col"]
        endpoint = summary[summary[x_col] == summary[x_col].max()].copy()
        for method in METHODS:
            present = method in set(summary["method"])
            coverage_rows.append(
                {
                    "experiment": exp_key,
                    "method": method,
                    "present": "yes" if present else "no",
                }
            )
        endpoint_rows.append(
            endpoint[
                [
                    x_col,
                    "method",
                    "error_P_mean",
                    "error_Theta_mean",
                    "error_B_mean",
                    "time_mean",
                ]
            ].rename(columns={x_col: "endpoint"}).assign(experiment=exp_key)
        )
        artifact_rows.append(
            {
                "experiment": exp_key,
                "raw_csv": saved.raw_csv.relative_to(outdir).as_posix(),
                "summary_csv": saved.summary_csv.relative_to(outdir).as_posix(),
                "timing_csv": saved.timing_summary_csv.relative_to(outdir).as_posix()
                if saved.timing_summary_csv
                else "",
                "metrics_png": saved.metrics_png.relative_to(outdir).as_posix()
                if saved.metrics_png
                else "",
                "runtime_png": saved.runtime_png.relative_to(outdir).as_posix()
                if saved.runtime_png
                else "",
            }
        )

    coverage = pd.DataFrame(coverage_rows)
    endpoint = pd.concat(endpoint_rows, ignore_index=True)
    endpoint["method"] = pd.Categorical(endpoint["method"], categories=METHODS, ordered=True)
    endpoint = endpoint.sort_values(["experiment", "method"]).astype({"method": str})
    artifacts = pd.DataFrame(artifact_rows)
    config_json = json.dumps(
        {key: asdict(value) for key, value in configs.items()},
        ensure_ascii=False,
        indent=2,
    )

    plot_lines = []
    for row in artifacts.itertuples(index=False):
        plot_lines.extend(
            [
                f"### {row.experiment}",
                "",
                f"![{row.experiment} metrics]({row.metrics_png})",
                "",
                f"![{row.experiment} runtime]({row.runtime_png})",
                "",
            ]
        )

    report = f"""# Reference 1 Section 7.1 Five-Method 실험 보고서

## 목적

Section 7.1의 Exp1~Exp4를 동일한 실행 경로에서 다시 돌려 `Non-random`, `Random Sampling`, `Random Projection`, `CountSketch`, `SIGN Bidirectional` 다섯 방법을 모두 비교했다. 모든 raw 결과에는 `time_sec`와 세부 timing breakdown이 포함된다.

## 설정

```json
{config_json}
```

## 방법론 커버리지

{dataframe_to_markdown(coverage)}

## 마지막 grid point 요약

각 experiment에서 가장 큰 x축 값의 평균 결과다.

{dataframe_to_markdown(endpoint)}

## 산출물

{dataframe_to_markdown(artifacts)}

## 그림

{chr(10).join(plot_lines)}
"""
    report_path = outdir / "section7_1_five_method_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_all(args: argparse.Namespace) -> tuple[dict[str, object], Path]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    configs = {}

    for exp_key, (cfg_factory, runner, summarizer, theta_mode) in RUNNERS.items():
        cfg = cfg_factory()
        cfg = replace(cfg, reps=args.reps, seed=args.seed)
        configs[exp_key] = cfg
        exp_outdir = outdir / exp_key
        exp_outdir.mkdir(parents=True, exist_ok=True)
        print(f"Running {exp_key} with five methods...")
        raw = runner(
            cfg,
            show_progress=not args.no_progress,
            theta_mode=theta_mode,
            detailed_timing=True,
        )
        summary = summarizer(raw)
        outputs[exp_key] = save_experiment_outputs(
            exp_key,
            raw,
            summary,
            outdir=exp_outdir,
            detailed_timing=True,
            plot_basics=not args.no_plot,
        )

    manifest = {
        "methods": METHODS,
        "configs": {key: asdict(value) for key, value in configs.items()},
        "outputs": {key: value.as_dict() for key, value in outputs.items()},
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = build_report(outdir, outputs, configs)
    return outputs, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 7.1 with all five methods.")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/reference_1_section7_1/results/all_methods_5way"),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    outputs, report_path = run_all(parse_args())
    print("Done.")
    print(f"Report: {report_path.resolve()}")
    for exp_key, saved in outputs.items():
        print(f"{exp_key}: {saved.outdir.resolve()}")


if __name__ == "__main__":
    main()
