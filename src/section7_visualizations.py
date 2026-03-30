from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.section7_experiments import EXPERIMENT_META, SECTION7_RESULTS_ROOT


plt.style.use("seaborn-v0_8-whitegrid")

METHOD_ORDER = ["Non-random", "Random Sampling", "Random Projection"]
METHOD_COLORS = {
    "Non-random": "#2ca02c",
    "Random Sampling": "#ff7f0e",
    "Random Projection": "#1f77b4",
}

METRIC_LABELS = {
    "time_sec": "Total runtime (sec)",
    "nr_eig_sec": "Spectral decomposition on A (sec)",
    "nr_kmeans_sec": "K-means on spectral embedding (sec)",
    "rs_sample_mask_sec": "Generate Bernoulli sampling mask (sec)",
    "rs_build_sampled_matrix_sec": "Construct sampled matrix A_s (sec)",
    "rs_eig_sec": "Eigen decomposition on sampled A_s (sec)",
    "rs_reconstruct_sec": "Reconstruct low-rank A_hat (sec)",
    "rs_symmetrize_sec": "Symmetrize reconstructed A_hat (sec)",
    "rs_kmeans_sec": "K-means on sampled embedding (sec)",
    "rp_draw_omega_sec": "Generate Gaussian test matrix Omega (sec)",
    "rp_power_iter_sec": "Power iterations with A (sec)",
    "rp_qr_sec": "Construct QR basis Q (sec)",
    "rp_build_core_sec": "Project A to core matrix C (sec)",
    "rp_reconstruct_sec": "Reconstruct low-rank A_hat (sec)",
    "rp_small_eig_sec": "Eigen decomposition on core matrix C (sec)",
    "rp_lift_sec": "Lift embedding back with Q (sec)",
    "rp_kmeans_sec": "K-means on projected embedding (sec)",
}

METHOD_STEP_METRICS = {
    "Non-random": [
        "nr_eig_sec",
        "nr_kmeans_sec",
    ],
    "Random Sampling": [
        "rs_sample_mask_sec",
        "rs_build_sampled_matrix_sec",
        "rs_eig_sec",
        "rs_reconstruct_sec",
        "rs_symmetrize_sec",
        "rs_kmeans_sec",
    ],
    "Random Projection": [
        "rp_draw_omega_sec",
        "rp_power_iter_sec",
        "rp_qr_sec",
        "rp_build_core_sec",
        "rp_reconstruct_sec",
        "rp_small_eig_sec",
        "rp_lift_sec",
        "rp_kmeans_sec",
    ],
}

METHOD_COMPONENTS = {
    "Non-random": [
        ("nr_eig_sec", "Spectral decomposition on original matrix A"),
        ("nr_kmeans_sec", "K-means on spectral embedding"),
    ],
    "Random Sampling": [
        ("rs_sample_mask_sec", "Generate Bernoulli sampling mask"),
        ("rs_build_sampled_matrix_sec", "Construct sampled matrix A_s"),
        ("rs_eig_sec", "Eigen decomposition on sampled A_s"),
        ("rs_reconstruct_sec", "Reconstruct low-rank A_hat"),
        ("rs_symmetrize_sec", "Symmetrize reconstructed A_hat"),
        ("rs_kmeans_sec", "K-means on sampled spectral embedding"),
    ],
    "Random Projection": [
        ("rp_draw_omega_sec", "Generate Gaussian test matrix Omega"),
        ("rp_power_iter_sec", "Power iterations with A"),
        ("rp_qr_sec", "Construct QR basis Q"),
        ("rp_build_core_sec", "Project A to core matrix C"),
        ("rp_reconstruct_sec", "Reconstruct low-rank A_hat"),
        ("rp_small_eig_sec", "Eigen decomposition on core matrix C"),
        ("rp_lift_sec", "Lift embedding back with Q"),
        ("rp_kmeans_sec", "K-means on projected embedding"),
    ],
}

METHOD_PALETTES = {
    "Non-random": ["#117733", "#CC6677", "#BBBBBB"],
    "Random Sampling": [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ],
    "Random Projection": [
        "#332288",
        "#88CCEE",
        "#44AA99",
        "#117733",
        "#999933",
        "#DDCC77",
        "#CC6677",
        "#AA4499",
        "#BBBBBB",
    ],
}

EXPERIMENT_PLOT_CONFIG = {
    "exp1": {"label": EXPERIMENT_META["exp1"]["label"], "summary_filename": "exp1_timing_breakdown_summary.csv", "display_cols": ["n"], "x_col": "n", "x_label": "n"},
    "exp2": {"label": EXPERIMENT_META["exp2"]["label"], "summary_filename": "exp2_timing_breakdown_summary.csv", "display_cols": ["alpha_n"], "x_col": "alpha_n", "x_label": "alpha_n"},
    "exp3": {"label": EXPERIMENT_META["exp3"]["label"], "summary_filename": "exp3_timing_breakdown_summary.csv", "display_cols": ["K"], "x_col": "K", "x_label": "K"},
    "exp4": {"label": EXPERIMENT_META["exp4"]["label"], "summary_filename": "exp4_timing_breakdown_summary.csv", "display_cols": ["n", "alpha_n"], "x_col": "n", "x_label": "n"},
}

ALL_TIMING_METRICS = ["time_sec"]
for method_name in METHOD_ORDER:
    ALL_TIMING_METRICS.extend(METHOD_STEP_METRICS[method_name])


@dataclass
class TimingBreakdownResult:
    exp_key: str
    summary_path: Path
    output_dir: Path
    created_paths: list[Path]
    overall_table: pd.DataFrame
    method_tables: dict[str, pd.DataFrame]


@dataclass
class RuntimeCompositionResult:
    exp_key: str
    summary_path: Path
    output_path: Path
    method_tables: dict[str, pd.DataFrame]


def find_latest_summary(summary_filename: str, search_root: Path | None = None) -> Path | None:
    search_root = search_root or SECTION7_RESULTS_ROOT
    matches = list(Path(search_root).glob(f"**/{summary_filename}"))
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime)


def resolve_summary_path(
    exp_key: str,
    summary_path: str | Path | None = None,
    search_root: Path | None = None,
) -> Path:
    if summary_path is not None:
        return Path(summary_path)
    cfg = EXPERIMENT_PLOT_CONFIG[exp_key]
    found = find_latest_summary(cfg["summary_filename"], search_root=search_root)
    if found is None:
        raise FileNotFoundError(
            f"Timing summary CSV not found for {exp_key}: {cfg['summary_filename']}"
        )
    return found


def load_summary_frame(
    exp_key: str,
    summary_path: str | Path | None = None,
    search_root: Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    resolved = resolve_summary_path(exp_key, summary_path=summary_path, search_root=search_root)
    return resolved, pd.read_csv(resolved)


def build_timing_table(df: pd.DataFrame, base_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    cols = []
    for col in base_cols:
        if col in df.columns and col not in cols:
            cols.append(col)
    for metric_name in metrics:
        for suffix in ("_mean", "_std"):
            col = f"{metric_name}{suffix}"
            if col in df.columns and col not in cols:
                cols.append(col)
    table = df.loc[:, cols].copy()
    sort_cols = [col for col in base_cols if col in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols)
    return table.reset_index(drop=True)


def compute_global_metric_limits(
    search_root: Path | None = None,
    summary_paths: dict[str, str | Path] | None = None,
) -> dict[str, tuple[float, float]]:
    loaded_frames = []
    if summary_paths:
        for exp_key, path in summary_paths.items():
            if path is None:
                continue
            loaded_frames.append((exp_key, pd.read_csv(path)))
    else:
        for exp_key in EXPERIMENT_PLOT_CONFIG:
            try:
                _, df = load_summary_frame(exp_key, search_root=search_root)
            except FileNotFoundError:
                continue
            loaded_frames.append((exp_key, df))

    limits: dict[str, tuple[float, float]] = {}
    for metric_name in ALL_TIMING_METRICS:
        max_value = 0.0
        mean_col = f"{metric_name}_mean"
        std_col = f"{metric_name}_std"
        for _, df in loaded_frames:
            if mean_col not in df.columns:
                continue
            upper = pd.to_numeric(df[mean_col], errors="coerce").fillna(0.0)
            if std_col in df.columns:
                upper = upper + pd.to_numeric(df[std_col], errors="coerce").fillna(0.0)
            if not upper.empty:
                max_value = max(max_value, float(upper.max()))
        padding = 0.05 * max_value if max_value > 0 else 1.0
        limits[metric_name] = (0.0, max_value + padding)
    return limits


def method_slug(method_name: str) -> str:
    return method_name.lower().replace("-", "").replace(" ", "_")


def metric_title(method_name: str, metric_name: str) -> str:
    return f"{method_name} - {METRIC_LABELS.get(metric_name, metric_name)}"


def _plot_single_metric(
    df: pd.DataFrame,
    method_name: str,
    x_col: str,
    x_label: str,
    metric_name: str,
    exp_label: str,
    output_path: Path | None,
    global_limits: dict[str, tuple[float, float]] | None,
    show_plot: bool,
) -> Path | None:
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"
    if mean_col not in df.columns:
        return None

    method_df = df[df["method"] == method_name].copy()
    if method_df.empty:
        return None

    method_df = method_df.sort_values(x_col)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    color = METHOD_COLORS[method_name]
    ax.plot(
        method_df[x_col],
        method_df[mean_col],
        color=color,
        linewidth=2.2,
        marker="o",
    )
    if std_col in method_df.columns:
        lower = method_df[mean_col] - method_df[std_col]
        upper = method_df[mean_col] + method_df[std_col]
        ax.fill_between(method_df[x_col], lower, upper, color=color, alpha=0.2)
    if global_limits and metric_name in global_limits:
        ax.set_ylim(*global_limits[metric_name])
    ax.set_title(f"{exp_label} | {metric_title(method_name, metric_name)}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    return output_path


def render_timing_breakdown_suite(
    exp_key: str,
    summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    global_limits: dict[str, tuple[float, float]] | None = None,
    search_root: Path | None = None,
    save: bool = True,
    show_plots: bool = True,
) -> TimingBreakdownResult:
    cfg = EXPERIMENT_PLOT_CONFIG[exp_key]
    resolved_summary_path, df = load_summary_frame(
        exp_key,
        summary_path=summary_path,
        search_root=search_root,
    )
    if global_limits is None:
        global_limits = compute_global_metric_limits(search_root=search_root)

    if output_dir is None:
        output_dir = resolved_summary_path.parent / "timing_breakdown_plots"
    output_dir = Path(output_dir)
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    overall_table = build_timing_table(df, cfg["display_cols"] + ["method"], ["time_sec"])
    method_tables = {
        method_name: build_timing_table(
            df[df["method"] == method_name].copy(),
            cfg["display_cols"],
            METHOD_STEP_METRICS[method_name],
        )
        for method_name in METHOD_ORDER
    }

    created_paths: list[Path] = []
    for method_name in METHOD_ORDER:
        out_path = output_dir / f"{method_slug(method_name)}_time_sec.png" if save else None
        created = _plot_single_metric(
            df=df,
            method_name=method_name,
            x_col=cfg["x_col"],
            x_label=cfg["x_label"],
            metric_name="time_sec",
            exp_label=cfg["label"],
            output_path=out_path,
            global_limits=global_limits,
            show_plot=show_plots,
        )
        if created is not None:
            created_paths.append(created)

    for method_name in METHOD_ORDER:
        for metric_name in METHOD_STEP_METRICS[method_name]:
            out_path = output_dir / f"{method_slug(method_name)}_{metric_name}.png" if save else None
            created = _plot_single_metric(
                df=df,
                method_name=method_name,
                x_col=cfg["x_col"],
                x_label=cfg["x_label"],
                metric_name=metric_name,
                exp_label=cfg["label"],
                output_path=out_path,
                global_limits=global_limits,
                show_plot=show_plots,
            )
            if created is not None:
                created_paths.append(created)

    return TimingBreakdownResult(
        exp_key=exp_key,
        summary_path=resolved_summary_path,
        output_dir=output_dir,
        created_paths=created_paths,
        overall_table=overall_table,
        method_tables=method_tables,
    )


def numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return pd.to_numeric(df[column_name], errors="coerce").fillna(0.0)


def format_x_labels(values: list[float], x_col: str) -> list[str]:
    labels = []
    for value in values:
        if x_col == "alpha_n":
            labels.append(f"{float(value):.2f}")
        else:
            try:
                labels.append(str(int(float(value))))
            except Exception:
                labels.append(str(value))
    return labels


def build_method_runtime_table(df: pd.DataFrame, method_name: str, x_col: str) -> pd.DataFrame | None:
    method_df = df[df["method"] == method_name].copy()
    if method_df.empty:
        return None

    method_df[x_col] = pd.to_numeric(method_df[x_col], errors="coerce")
    method_df = method_df.sort_values(x_col).reset_index(drop=True)
    total = numeric_series(method_df, "algo_total_sec_mean")
    if (total <= 0).all():
        total = numeric_series(method_df, "time_sec_mean")

    result = pd.DataFrame({x_col: method_df[x_col], "algo_total_sec_mean": total})
    component_sum = pd.Series(np.zeros(len(method_df)), index=method_df.index, dtype=float)
    for metric_name, label in METHOD_COMPONENTS[method_name]:
        values = numeric_series(method_df, f"{metric_name}_mean")
        result[label] = values
        component_sum = component_sum + values

    result["Measured overhead / remainder"] = (total - component_sum).clip(lower=0.0)
    for col in [c for c in result.columns if c not in {x_col, "algo_total_sec_mean"}]:
        pct = np.where(total > 0, 100.0 * result[col] / total, 0.0)
        result[f"{col} (%)"] = pct
    return result


def _format_percentage(pct: float) -> str:
    if pct >= 10.0:
        return f"{pct:.0f}%"
    if pct >= 1.0:
        return f"{pct:.1f}%"
    return f"{pct:.2f}%"


def _use_inside_label(height: float, pct: float, shared_ymax: float) -> bool:
    return height >= shared_ymax * 0.1 or pct >= 16.0


def plot_total_runtime_comparison(ax, df: pd.DataFrame, x_col: str, x_label: str) -> None:
    for method_name in METHOD_ORDER:
        method_df = df[df["method"] == method_name].copy()
        if method_df.empty:
            continue
        method_df[x_col] = pd.to_numeric(method_df[x_col], errors="coerce")
        method_df = method_df.sort_values(x_col)
        y = numeric_series(method_df, "algo_total_sec_mean")
        if (y <= 0).all():
            y = numeric_series(method_df, "time_sec_mean")
        ax.plot(
            method_df[x_col].to_numpy(dtype=float),
            y.to_numpy(dtype=float),
            color=METHOD_COLORS[method_name],
            linewidth=2.4,
            marker="o",
            label=method_name,
        )
    ax.set_title("Total runtime comparison")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Algorithm runtime (sec)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)


def _count_outside_labels(table: pd.DataFrame, component_cols: list[str], shared_ymax: float) -> int:
    max_count = 0
    for row_idx in range(len(table)):
        count = 0
        for col in component_cols:
            height = float(table.iloc[row_idx][col])
            pct = float(table.iloc[row_idx][f"{col} (%)"])
            if height <= 0.0 or pct <= 0.0:
                continue
            if not _use_inside_label(height, pct, shared_ymax):
                count += 1
        max_count = max(max_count, count)
    return max_count


def annotate_stack_percentages(
    ax,
    table: pd.DataFrame,
    component_cols: list[str],
    x_pos: np.ndarray,
    shared_ymax: float,
) -> None:
    outside_counts = {int(idx): 0 for idx in range(len(x_pos))}
    base_y = shared_ymax * 1.03
    y_step = shared_ymax * 0.07
    x_offset = 0.18

    for comp_idx, col in enumerate(component_cols):
        bottoms = np.zeros(len(x_pos), dtype=float)
        for prev_col in component_cols[:comp_idx]:
            bottoms = bottoms + table[prev_col].to_numpy(dtype=float)

        heights = table[col].to_numpy(dtype=float)
        pct_values = table[f"{col} (%)"].to_numpy(dtype=float)
        for bar_idx, (x_value, bottom, height, pct) in enumerate(
            zip(x_pos, bottoms, heights, pct_values)
        ):
            if height <= 0.0 or pct <= 0.0:
                continue

            label = _format_percentage(float(pct))
            if _use_inside_label(float(height), float(pct), shared_ymax):
                ax.text(
                    x_value,
                    bottom + height / 2.0,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.9,
                    },
                )
                continue

            slot = outside_counts[bar_idx]
            outside_counts[bar_idx] += 1
            side = -1 if slot % 2 == 0 else 1
            tier = slot // 2
            x_text = x_value + side * (x_offset + 0.08 * tier)
            y_text = base_y + slot * y_step
            ax.annotate(
                label,
                xy=(x_value, bottom + height),
                xytext=(x_text, y_text),
                textcoords="data",
                ha="center",
                va="bottom",
                fontsize=7,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "#999999",
                    "alpha": 0.95,
                },
                arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.8},
                clip_on=False,
            )


def plot_method_runtime_stack(
    ax,
    table: pd.DataFrame,
    method_name: str,
    x_col: str,
    x_label: str,
    shared_ymax: float,
) -> None:
    component_cols = [
        col
        for col in table.columns
        if col not in {x_col, "algo_total_sec_mean"} and not col.endswith(" (%)")
    ]
    colors = METHOD_PALETTES[method_name][: len(component_cols)]
    x_labels = format_x_labels(table[x_col].tolist(), x_col)
    x_pos = np.arange(len(x_labels))
    bottom = np.zeros(len(x_labels), dtype=float)

    for color, col in zip(colors, component_cols):
        values = table[col].to_numpy(dtype=float)
        hatch = "//" if col == "Measured overhead / remainder" else None
        edgecolor = "#666666" if col == "Measured overhead / remainder" else "white"
        ax.bar(
            x_pos,
            values,
            bottom=bottom,
            color=color,
            edgecolor=edgecolor,
            linewidth=0.8,
            hatch=hatch,
            label=col,
        )
        bottom = bottom + values

    annotate_stack_percentages(ax, table, component_cols, x_pos, shared_ymax)

    max_outside = _count_outside_labels(table, component_cols, shared_ymax)
    headroom = 1.18 + 0.12 * max_outside
    ax.set_ylim(0, shared_ymax * headroom)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Runtime (sec)")
    ax.set_xlabel(x_label)
    ax.set_title(method_name)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title=f"{method_name} steps",
    )


def render_runtime_composition(
    exp_key: str,
    summary_path: str | Path | None = None,
    output_path: str | Path | None = None,
    search_root: Path | None = None,
    save: bool = True,
    show_plot: bool = True,
) -> RuntimeCompositionResult:
    cfg = EXPERIMENT_PLOT_CONFIG[exp_key]
    resolved_summary_path, df = load_summary_frame(
        exp_key,
        summary_path=summary_path,
        search_root=search_root,
    )

    method_tables = {
        method_name: build_method_runtime_table(df, method_name, cfg["x_col"])
        for method_name in METHOD_ORDER
    }
    max_total = 0.0
    for table in method_tables.values():
        if table is None:
            continue
        max_total = max(max_total, float(table["algo_total_sec_mean"].max()))
    shared_ymax = max_total * 1.10 if max_total > 0 else 1.0

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 18),
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0, 1.0]},
    )
    plot_total_runtime_comparison(axes[0], df, cfg["x_col"], cfg["x_label"])
    for ax, method_name in zip(axes[1:], METHOD_ORDER):
        table = method_tables[method_name]
        if table is None:
            ax.set_visible(False)
            continue
        plot_method_runtime_stack(ax, table, method_name, cfg["x_col"], cfg["x_label"], shared_ymax)

    fig.suptitle(f"{cfg['label']} | Runtime comparison and composition", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.72, 0.98])

    if output_path is None:
        output_path = resolved_summary_path.parent / "runtime_composition_plots" / f"{exp_key}_runtime_composition.png"
    output_path = Path(output_path)
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)

    return RuntimeCompositionResult(
        exp_key=exp_key,
        summary_path=resolved_summary_path,
        output_path=output_path,
        method_tables=method_tables,
    )


def render_all_section7_visualizations(
    exp_key: str,
    summary_path: str | Path | None = None,
    breakdown_output_dir: str | Path | None = None,
    composition_output_path: str | Path | None = None,
    global_limits: dict[str, tuple[float, float]] | None = None,
    search_root: Path | None = None,
    save: bool = True,
    show_plots: bool = True,
) -> tuple[TimingBreakdownResult, RuntimeCompositionResult]:
    breakdown = render_timing_breakdown_suite(
        exp_key=exp_key,
        summary_path=summary_path,
        output_dir=breakdown_output_dir,
        global_limits=global_limits,
        search_root=search_root,
        save=save,
        show_plots=show_plots,
    )
    composition = render_runtime_composition(
        exp_key=exp_key,
        summary_path=summary_path,
        output_path=composition_output_path,
        search_root=search_root,
        save=save,
        show_plot=show_plots,
    )
    return breakdown, composition
