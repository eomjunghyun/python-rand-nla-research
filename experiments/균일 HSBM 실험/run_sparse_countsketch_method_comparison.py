# -*- coding: utf-8 -*-
"""Run the method comparison into a fresh sparse-CountSketch result folder."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import uniform_hsbm_method_comparison as comparison


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "results" / "EXP-20260514_sparse_cs"
REPORT_PATH = OUTPUT_ROOT / "결과보고서.md"
SWEEP_ORDER = ("n", "K", "rho_n")
RUN_ORDER = ("K", "n", "rho_n")


def get_sparse_countsketch_specs():
    base_specs = comparison.get_comparison_specs()
    replacements = {
        "n": (
            "EXP-20260514-001",
            "n_rho16_sparse_cs",
        ),
        "K": (
            "EXP-20260514-002",
            "K_rho16_sparse_cs",
        ),
        "rho_n": (
            "EXP-20260514-003",
            "rho_sparse_cs",
        ),
    }
    return {
        sweep: replace(
            base_specs[sweep],
            experiment_id=experiment_id,
            experiment_slug=experiment_slug,
        )
        for sweep, (experiment_id, experiment_slug) in replacements.items()
    }


def configure_output_root():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison.RESULTS_ROOT = OUTPUT_ROOT


def load_summary(spec):
    path = spec.outdir / f"{spec.file_prefix}_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    df = pd.read_csv(path)
    df["method"] = pd.Categorical(
        df["method"],
        categories=comparison.METHOD_ORDER,
        ordered=True,
    )
    return df.sort_values([spec.x_col, "method"]).reset_index(drop=True)


def plot_relpath(spec):
    plot_name = f"{spec.file_prefix}_summary.png"
    return f"{spec.experiment_id}_{spec.experiment_slug}/{plot_name}"


def write_report(specs) -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summaries = {sweep: load_summary(spec) for sweep, spec in specs.items()}

    lines: list[str] = []
    lines.append("# 균일 HSBM sparse CountSketch method comparison 결과보고서")
    lines.append("")
    lines.append(
        "이 보고서는 교수님 지시에 따라 CountSketch random projection을 sparse sketch matrix 저장 방식으로 실행한 새 비교 결과입니다. "
        "기존 method comparison의 실험 세팅은 바꾸지 않고, 결과만 새 폴더에 다시 생성했습니다."
    )
    lines.append("")
    lines.append("## 실행 메모")
    lines.append("")
    lines.append("- 결과 루트: `results/EXP-20260514_sparse_cs/`")
    lines.append("- CountSketch RP는 `src/common.py`의 `generate_hash_and_signs`와 `sparse_explicit_countsketch`를 사용합니다.")
    lines.append("- 즉, CountSketch 행렬 `S`를 scipy CSR sparse matrix로 저장하고 `S @ Theta`를 계산합니다.")
    lines.append("- `index_set_countsketch` 방식은 이번 HSBM method comparison 실행 경로에서 사용하지 않았습니다.")
    lines.append("- `n`, `K`, `rho_n`, 반복 횟수, seed, `ell = K + 160`, power iteration `q=4`, eigensolver `eigsh`, k-means 설정은 기존 비교 실험과 동일합니다.")
    lines.append("- Gaussian RP의 test matrix는 `N(0, 1) / sqrt(ell)`로 스케일링했습니다.")
    lines.append("- raw CSV의 `cs_sparse_explicit_sketch_sec` 컬럼으로 sparse explicit sketch 단계 시간을 확인할 수 있습니다.")
    lines.append("")
    lines.append("## 전체 요약")
    lines.append("")

    overall_rows = []
    for sweep in ("K", "n", "rho_n"):
        summary = summaries[sweep]
        method_means = summary.groupby("method", observed=True)["misclassification_mean"].mean().dropna()
        best_mis = float(method_means.min()) if not method_means.empty else float("nan")
        for method in comparison.METHOD_ORDER:
            dm = summary[summary["method"] == method]
            if dm.empty:
                continue
            values = [
                sweep,
                method,
                comparison._fmt_float(dm["misclassification_mean"].mean()),
                comparison._fmt_float(dm["ARI_mean"].mean()),
                comparison._fmt_float(dm["NMI_mean"].mean()),
                comparison._fmt_float(dm["algorithm_total_wall_sec_mean"].mean()),
                comparison._fmt_float(dm["spectral_clustering_wall_sec_mean"].mean()),
            ]
            if np.isclose(float(dm["misclassification_mean"].mean()), best_mis):
                values = [f"**{value}**" for value in values]
            overall_rows.append(values)
    lines.append(
        comparison._markdown_table(
            ["sweep", "method", "평균 오분류율", "평균 ARI", "평균 NMI", "평균 algorithm_sec", "평균 spectral_sec"],
            overall_rows,
        )
    )
    lines.append("")

    for sweep, title in [("n", "n 변화 실험"), ("K", "K 변화 실험"), ("rho_n", "rho_n 변화 실험")]:
        spec = specs[sweep]
        summary = summaries[sweep]
        lines.append(f"## {title}")
        lines.append("")
        for x_value, group in summary.groupby(spec.x_col, sort=True, observed=True):
            value_text = f"{float(x_value):.1f}" if sweep == "n" else f"{float(x_value):.4f}"
            lines.append(f"### {spec.x_col} = {value_text}")
            lines.append("")
            lines.append(
                comparison._markdown_table(
                    ["방법", "오분류율", "ARI", "NMI", "algorithm_sec", "spectral_sec", "하이퍼엣지수", "평균 degree"],
                    comparison._bold_best_rows(group),
                )
            )
            lines.append("")
        lines.append("### 그림")
        lines.append("")
        lines.append(f"![{title} method 비교]({plot_relpath(spec)})")
        lines.append("")

    lines.append("## CountSketch와 Gaussian RP 비교")
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
    lines.append(
        f"- Gaussian RP와 직접 비교한 {len(merged)}개 sweep 지점 중 CountSketch의 오분류율이 더 낮은 지점은 {cs_better}개, 동률은 {cs_tied}개, 더 높은 지점은 {cs_worse}개입니다."
    )
    lines.append(
        f"- CountSketch의 spectral 단계 시간은 Gaussian RP 대비 평균 `{comparison._fmt_float(time_ratio.mean())}`배입니다."
    )
    lines.append("")
    detail_rows = []
    for _, row in merged.sort_values(["sweep", "x_value"]).iterrows():
        detail_rows.append(
            [
                row["sweep"],
                comparison._fmt_float(row["x_value"]),
                comparison._fmt_float(row["misclassification_mean_gaussian"]),
                comparison._fmt_float(row["misclassification_mean_countsketch"]),
                comparison._fmt_float(row["spectral_clustering_wall_sec_mean_gaussian"]),
                comparison._fmt_float(row["spectral_clustering_wall_sec_mean_countsketch"]),
            ]
        )
    lines.append(
        comparison._markdown_table(
            ["sweep", "x", "Gaussian mis", "CountSketch mis", "Gaussian spectral_sec", "CountSketch spectral_sec"],
            detail_rows,
        )
    )
    lines.append("")

    lines.append("## 해석 메모")
    lines.append("")
    lines.append("- 이번 실행은 sparse explicit CountSketch 경로를 검증하기 위한 재실험입니다.")
    lines.append("- CountSketch의 `S @ Theta` 단계는 별도 컬럼으로 기록했지만, 전체 spectral 시간에는 이후 power iteration, QR, core matrix 구성, 작은 고유값 문제, lift, k-means까지 포함됩니다.")
    lines.append("- 따라서 CountSketch sparse matrix 저장 자체가 빠르더라도 전체 시간은 후속 dense 연산 비용의 영향을 함께 받습니다.")
    lines.append("- 오분류율은 Hungarian matching으로 예측 label을 true label에 맞춘 뒤 계산했고, ARI/NMI는 label permutation에 불변인 값을 사용했습니다.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT_PATH


def run_all(show_progress: bool = True):
    configure_output_root()
    specs = get_sparse_countsketch_specs()
    outputs = {}
    for sweep in RUN_ORDER:
        specs[sweep].outdir.mkdir(parents=True, exist_ok=True)
        outputs[sweep] = comparison.run_spec(specs[sweep], show_progress=show_progress)
    report_path = write_report(specs)
    return outputs, report_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run sparse-CountSketch method comparison into a fresh result folder."
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    configure_output_root()
    specs = get_sparse_countsketch_specs()
    if not args.report_only:
        for sweep in RUN_ORDER:
            specs[sweep].outdir.mkdir(parents=True, exist_ok=True)
            comparison.run_spec(specs[sweep], show_progress=not args.no_progress)
    print(write_report(specs))


if __name__ == "__main__":
    main()
