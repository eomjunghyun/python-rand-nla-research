from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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
    eigvecs_sign_sparse,
    load_large_integer_edgelist_csr,
)


METHOD_ORDER = [
    "Random Projection",
    "SIGN",
    "Random Sampling",
    "Random Sampling (excl. sampling)",
    "partial_eigen",
]
METHOD_COLORS = {
    "Random Projection": "#4C78A8",
    "SIGN": "#9467BD",
    "Random Sampling": "#F58518",
    "Random Sampling (excl. sampling)": "#E45756",
    "partial_eigen": "#54A24B",
}
METHOD_LABELS = {
    "Random Projection": "Random projection",
    "SIGN": "SIGN",
    "Random Sampling": "Random sampling",
    "Random Sampling (excl. sampling)": "Random sampling\n(excl. sampling)",
    "partial_eigen": "partial_eigen",
}


@dataclass
class NetworkSpec:
    name: str
    edgelist: Path
    target_rank: int


@dataclass
class Sign82Config:
    baseline_raw_csv: Path = Path(
        "experiments/reference_1_section8_2/results/exp8_2_table4_paper_aligned/table4_time_raw.csv"
    )
    dblp_edgelist: Path = Path("data/dblp/com-dblp.ungraph.txt")
    youtube_edgelist: Path = Path("/private/tmp/sign82_data/com-youtube.ungraph.txt.gz")
    internet_edgelist: Path = Path("/private/tmp/sign82_data/as-skitter.txt.gz")
    reps: int = 20
    seed: int = 2026
    r: int = 10
    q: int = 2
    delimiter: str | None = None
    comment_prefix: str = "#"
    outdir: Path = Path("experiments/reference_1_section8_2/results/sign_section8_2_wang2025")
    no_progress: bool = False


def build_network_specs(cfg: Sign82Config) -> list[NetworkSpec]:
    return [
        NetworkSpec("DBLP", cfg.dblp_edgelist, 3),
        NetworkSpec("Youtube", cfg.youtube_edgelist, 7),
        NetworkSpec("Internet", cfg.internet_edgelist, 4),
    ]


def benchmark_sign_sparse(cfg: Sign82Config) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    specs = build_network_specs(cfg)
    progress = None if cfg.no_progress else LiveProgress(len(specs) * cfg.reps)
    master_rng = np.random.default_rng(cfg.seed)
    rows = []
    dataset_meta = []

    for spec in specs:
        if not spec.edgelist.exists():
            dataset_meta.append(
                {
                    "dataset": spec.name,
                    "edgelist": str(spec.edgelist),
                    "target_rank": spec.target_rank,
                    "status": "missing",
                }
            )
            continue

        A, _ = load_large_integer_edgelist_csr(
            spec.edgelist,
            delimiter=cfg.delimiter,
            comment_prefix=cfg.comment_prefix,
        )
        n_nodes = int(A.shape[0])
        n_edges = int(A.nnz // 2)
        dataset_meta.append(
            {
                "dataset": spec.name,
                "edgelist": str(spec.edgelist),
                "target_rank": spec.target_rank,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "status": "ok",
            }
        )

        for rep in range(1, cfg.reps + 1):
            rep_seed = int(master_rng.integers(1, 2**31 - 1))
            rng = np.random.default_rng(rep_seed)
            t0 = perf_counter()
            _, _, timing = eigvecs_sign_sparse(
                A,
                k=spec.target_rank,
                r=cfg.r,
                power=cfg.q,
                rng=rng,
                return_timing=True,
            )
            time_sec = perf_counter() - t0
            rows.append(
                {
                    "dataset": spec.name,
                    "rep": rep,
                    "method": "SIGN",
                    "time_sec": float(time_sec),
                    "time_sec_excl_sampling": float(time_sec),
                    "time_sampling_sec": 0.0,
                    "target_rank": spec.target_rank,
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    **timing,
                }
            )
            if progress is not None:
                progress.update("dataset", spec.name, rep, cfg.reps, "SIGN")

    if progress is not None:
        progress.close()
    return pd.DataFrame(rows), dataset_meta


def summarize_with_sign(df_raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dataset, block in df_raw.groupby("dataset", sort=False):
        meta = block.iloc[0]
        get = lambda method, col="time_sec": block.loc[block["method"] == method, col]
        rp = get("Random Projection").median()
        sign = get("SIGN").median() if (block["method"] == "SIGN").any() else np.nan
        rs = get("Random Sampling").median()
        rs_excl = get("Random Sampling", "time_sec_excl_sampling").median()
        pe = get("partial_eigen").median()
        records.append(
            {
                "dataset": dataset,
                "n_nodes": int(meta["n_nodes"]),
                "n_edges": int(meta["n_edges"]),
                "target_rank": int(meta["target_rank"]),
                "random_projection_median_sec": float(rp),
                "sign_median_sec": float(sign),
                "random_sampling_median_sec": float(rs),
                "random_sampling_excl_sampling_median_sec": float(rs_excl),
                "partial_eigen_median_sec": float(pe),
                "sign_vs_random_projection": float(sign / rp) if rp > 0 else np.nan,
                "sign_vs_partial_eigen": float(sign / pe) if pe > 0 else np.nan,
                "random_sampling_display": f"{rs:.3f}({rs_excl:.3f})",
            }
        )
    return pd.DataFrame(records)


def summarize_sign_steps(df_sign_raw: pd.DataFrame) -> pd.DataFrame:
    step_cols = [
        "sign_draw_omega_sec",
        "sign_subspace_iter_sec",
        "sign_build_core_sec",
        "sign_small_eig_sec",
        "sign_lift_sec",
        "time_sec",
    ]
    agg = {}
    for col in step_cols:
        if col in df_sign_raw.columns:
            agg[f"{col}_median"] = (col, "median")
            agg[f"{col}_mean"] = (col, "mean")
    return df_sign_raw.groupby("dataset", as_index=False).agg(**agg)


def format_markdown_table(df_summary: pd.DataFrame) -> str:
    lines = [
        "Table 4-like median time (seconds) over replications, with Wang 2025 SIGN added.",
        "",
        "| Networks | Random projection | SIGN | Random sampling | partial_eigen | SIGN / RP | SIGN / partial_eigen |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in df_summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.dataset} | "
            f"{row.random_projection_median_sec:.3f} | "
            f"{row.sign_median_sec:.3f} | "
            f"{row.random_sampling_display} | "
            f"{row.partial_eigen_median_sec:.3f} | "
            f"{row.sign_vs_random_projection:.2f}x | "
            f"{row.sign_vs_partial_eigen:.2f}x |"
        )
    lines.append("")
    lines.append(
        "Note: Random Sampling values outside parentheses include sampling time; "
        "values inside parentheses exclude sampling time."
    )
    return "\n".join(lines)


def _plot_long_frame(df_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df_summary.itertuples(index=False):
        rows.extend(
            [
                {
                    "dataset": row.dataset,
                    "method_variant": "Random Projection",
                    "time_sec": row.random_projection_median_sec,
                },
                {"dataset": row.dataset, "method_variant": "SIGN", "time_sec": row.sign_median_sec},
                {
                    "dataset": row.dataset,
                    "method_variant": "Random Sampling",
                    "time_sec": row.random_sampling_median_sec,
                },
                {
                    "dataset": row.dataset,
                    "method_variant": "Random Sampling (excl. sampling)",
                    "time_sec": row.random_sampling_excl_sampling_median_sec,
                },
                {
                    "dataset": row.dataset,
                    "method_variant": "partial_eigen",
                    "time_sec": row.partial_eigen_median_sec,
                },
            ]
        )
    return pd.DataFrame(rows)


def plot_median_bars_with_sign(df_summary: pd.DataFrame, out_png: Path) -> None:
    plot_df = _plot_long_frame(df_summary)
    datasets = list(dict.fromkeys(plot_df["dataset"].tolist()))
    x = np.arange(len(datasets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    for idx, method in enumerate(METHOD_ORDER):
        block = plot_df[plot_df["method_variant"] == method]
        offsets = x + (idx - 2) * width
        bars = ax.bar(
            offsets,
            block["time_sec"].values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.6,
            label=METHOD_LABELS[method],
            hatch="//" if "excl." in method else None,
        )
        for bar in bars:
            h = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Median eigenvector runtime (sec)")
    ax.set_title("Section 8.2 Runtime with Wang 2025 SIGN")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncols=3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_boxplots_with_sign(df_raw: pd.DataFrame, out_png: Path) -> None:
    datasets = list(dict.fromkeys(df_raw["dataset"].tolist()))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        block = df_raw[df_raw["dataset"] == dataset]
        series = []
        labels = []
        variants = []
        for method in METHOD_ORDER:
            if method == "Random Sampling (excl. sampling)":
                vals = block.loc[block["method"] == "Random Sampling", "time_sec_excl_sampling"].values
            else:
                vals = block.loc[block["method"] == method, "time_sec"].values
            if vals.size == 0:
                continue
            series.append(vals)
            labels.append(METHOD_LABELS[method])
            variants.append(method)
        box = ax.boxplot(series, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, method in zip(box["boxes"], variants):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_edgecolor("black")
            if "excl." in method:
                patch.set_hatch("//")
        ax.set_title(dataset)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Per-rep eigenvector runtime (sec)")
    fig.suptitle("Section 8.2 Runtime Distribution with SIGN", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_format_number(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    cfg: Sign82Config,
    dataset_meta: list[dict[str, object]],
    df_summary: pd.DataFrame,
    df_step_summary: pd.DataFrame,
    out_paths: dict[str, Path],
) -> Path:
    report_path = cfg.outdir / "sign_section8_2_report.md"
    table_md = format_markdown_table(df_summary)
    config_json = json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str)
    dataset_meta_md = dataframe_to_markdown(pd.DataFrame(dataset_meta))
    step_cols = [
        "dataset",
        "sign_draw_omega_sec_median",
        "sign_subspace_iter_sec_median",
        "sign_build_core_sec_median",
        "sign_small_eig_sec_median",
        "sign_lift_sec_median",
        "time_sec_median",
    ]
    step_md = dataframe_to_markdown(df_step_summary[[c for c in step_cols if c in df_step_summary.columns]])
    median_rel = out_paths["median_png"].relative_to(cfg.outdir)
    box_rel = out_paths["box_png"].relative_to(cfg.outdir)

    report = f"""# Wang 2025 SIGN 방법론의 Section 8.2 적용 보고서

## 목적

Reference 1 Section 8.2는 대규모 sparse real network에서 eigenvector computation time을 비교하는 Table 4 스타일 benchmark다. 여기에 사용자가 제공한 Wang et al. (2025)의 SIGN subspace iteration 방법을 추가해 기존 `Random Projection`, `Random Sampling`, `partial_eigen` 결과와 같은 단위로 비교했다.

## 구현 메모

- sparse SIGN 구현: `src.common.eigvecs_sign_sparse`
- 실행 스크립트: `experiments/reference_1_section8_2/run_sign_section8_2.py`
- 출력 폴더: `{cfg.outdir}`
- 기존 baseline: `{cfg.baseline_raw_csv}`
- SIGN 설정: 기존 8.2와 맞춰 oversampling `r={cfg.r}`, power parameter `k=q={cfg.q}`를 사용했다.
- Section 8.2의 graph는 undirected adjacency로 읽기 때문에 `A.T`와 `A`가 같은 대칭 문제다. 따라서 여기서 SIGN은 비대칭 행렬용 장점보다는 양방향 subspace iteration의 runtime 특성을 보는 실험이다.
- timing은 Table 4와 맞춰 KMeans나 accuracy 계산 없이 eigenvector approximation pipeline만 잰다.

## 설정

```json
{config_json}
```

## 데이터셋

{dataset_meta_md}

## Median Runtime 표

{table_md}

## SIGN 내부 단계별 Median

{step_md}

## 그림

![Median runtime with SIGN]({median_rel.as_posix()})

![Runtime distribution with SIGN]({box_rel.as_posix()})

## 해석

SIGN은 Random Projection과 같은 randomized subspace family에 속하지만, 한 iteration마다 `A.T`와 `A`를 번갈아 곱고 QR을 수행한다. 대칭 sparse graph에서는 이것이 기존 Random Projection의 `A^(2q+1) Omega`와 비슷한 방향의 근사지만, QR 횟수와 matrix multiplication 횟수 구성이 다르다.

따라서 이 결과는 Wang 2025의 비대칭 행렬 low-rank approximation 장점을 직접 검증한다기보다는, 현재 8.2의 대칭 graph runtime benchmark에서 SIGN 변형이 어느 정도 비용을 갖는지 확인하는 의미가 크다. `SIGN / RP`가 1보다 작으면 SIGN이 Random Projection보다 빠르고, 1보다 크면 느리다.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_all(cfg: Sign82Config):
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    viz_dir = cfg.outdir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    df_sign_raw, dataset_meta = benchmark_sign_sparse(cfg)
    if df_sign_raw.empty:
        raise RuntimeError("No SIGN rows were produced. Check dataset paths.")

    baseline = pd.read_csv(cfg.baseline_raw_csv)
    sign_datasets = set(df_sign_raw["dataset"])
    baseline = baseline[baseline["dataset"].isin(sign_datasets)].copy()
    combined = pd.concat([baseline, df_sign_raw], ignore_index=True, sort=False)
    summary = summarize_with_sign(combined)
    step_summary = summarize_sign_steps(df_sign_raw)
    markdown = format_markdown_table(summary)

    paths = {
        "sign_raw": cfg.outdir / "sign_time_raw.csv",
        "combined_raw": cfg.outdir / "table4_with_sign_time_raw.csv",
        "summary_csv": cfg.outdir / "table4_with_sign_median_time.csv",
        "summary_md": cfg.outdir / "table4_with_sign_median_time.md",
        "step_summary": cfg.outdir / "sign_step_time_summary.csv",
        "meta_json": cfg.outdir / "sign_table4_meta.json",
        "median_png": viz_dir / "table4_with_sign_median_bar.png",
        "box_png": viz_dir / "table4_with_sign_runtime_boxplots.png",
    }

    df_sign_raw.to_csv(paths["sign_raw"], index=False)
    combined.to_csv(paths["combined_raw"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    paths["summary_md"].write_text(markdown, encoding="utf-8")
    step_summary.to_csv(paths["step_summary"], index=False)
    paths["meta_json"].write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "datasets": dataset_meta,
                "artifacts": {key: str(value) for key, value in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    plot_median_bars_with_sign(summary, paths["median_png"])
    plot_runtime_boxplots_with_sign(combined, paths["box_png"])
    report_path = build_report(cfg, dataset_meta, summary, step_summary, paths)
    return paths, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wang 2025 SIGN on Section 8.2 Table 4 benchmark.")
    parser.add_argument("--baseline-raw-csv", type=Path, default=Sign82Config.baseline_raw_csv)
    parser.add_argument("--dblp-edgelist", type=Path, default=Sign82Config.dblp_edgelist)
    parser.add_argument("--youtube-edgelist", type=Path, default=Sign82Config.youtube_edgelist)
    parser.add_argument("--internet-edgelist", type=Path, default=Sign82Config.internet_edgelist)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--delimiter", type=str, default=None)
    parser.add_argument("--comment-prefix", type=str, default="#")
    parser.add_argument("--outdir", type=Path, default=Sign82Config.outdir)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Sign82Config(
        baseline_raw_csv=args.baseline_raw_csv,
        dblp_edgelist=args.dblp_edgelist,
        youtube_edgelist=args.youtube_edgelist,
        internet_edgelist=args.internet_edgelist,
        reps=args.reps,
        seed=args.seed,
        r=args.r,
        q=args.q,
        delimiter=args.delimiter,
        comment_prefix=args.comment_prefix,
        outdir=args.outdir,
        no_progress=args.no_progress,
    )
    paths, report_path = run_all(cfg)
    print("Done.")
    print(f"Report: {report_path.resolve()}")
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
