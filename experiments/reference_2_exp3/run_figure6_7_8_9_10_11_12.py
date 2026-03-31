# -*- coding: utf-8 -*-
from __future__ import annotations

"""Reproduce Experiment 3 (Figure 6-12) for "A Stochastic Block Hypergraph model".

This script studies hyperedge composition using normalized Gini statistics.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    LiveProgress,
    composition_delta,
    generate_hypergraph,
    hyperedge_community_count_matrix,
    make_balanced_labels,
    normalized_gini,
)


STRATEGIES = ("majority", "max", "min", "weighted")
NODE_ORDERS = ("random", "fixed", "community")


@dataclass
class Exp3Config:
    seed: int = 2026
    outdir: Path = Path("experiments/reference_2_exp3")
    no_progress: bool = False
    quick: bool = False

    # shared histogram setup
    gini_bins: int = 41  # [0,1] split into 40 bins

    # Figure 6
    fig6_N: int = 80
    fig6_E: int = 80
    fig6_K: int = 4
    fig6_p: float = 0.1
    fig6_q_values: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1)
    fig6_strategy: str = "majority"

    # Figure 7
    fig7_N: int = 2000
    fig7_E: int = 200
    fig7_K: int = 4
    fig7_p: float = 0.1
    fig7_q_over_p: tuple[float, ...] = (
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    )
    fig7_realizations: int = 100

    # Figure 8
    fig8_N: int = 1000
    fig8_E: int = 200
    fig8_K: int = 4
    fig8_p: float = 0.03
    fig8_q_over_p: float = 0.4
    fig8_realizations: int = 100

    # Shared for Figure 9-12
    fig9_12_E: int = 200
    fig9_12_K: int = 4
    fig9_12_p: float = 0.03
    fig9_12_n_over_e: tuple[float, ...] = (1.0, 1.5, 2.5, 3.5, 5.0, 10.0)
    fig9_12_q_over_p: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21))
    fig9_12_realizations: int = 100


# -----------------------------------------------------------------------------
# Required functions
# -----------------------------------------------------------------------------
def compute_hyperedge_composition(hyperedges, communities, K):
    """Compute per-hyperedge community counts matrix with shape (num_edges, K)."""
    return hyperedge_community_count_matrix(hyperedges, communities, K)


def compute_normalized_gini(counts_by_community):
    """Compute normalized Gini from a single per-community count vector."""
    return normalized_gini(np.asarray(counts_by_community, dtype=float))


def summarize_gini(hyperedges, communities, K):
    """Return composition matrix, G values, mean G, std G, and Delta."""
    comp = compute_hyperedge_composition(hyperedges, communities, K)
    if comp.shape[0] == 0:
        g_values = np.zeros(0, dtype=float)
    else:
        g_values = np.asarray([compute_normalized_gini(row) for row in comp], dtype=float)

    mean_g = float(np.mean(g_values)) if g_values.size > 0 else 0.0
    std_g = float(np.std(g_values)) if g_values.size > 0 else 0.0
    delta = compute_delta_from_gini(g_values)
    return {
        "composition": comp,
        "G_values": g_values,
        "mean_G": mean_g,
        "std_G": std_g,
        "delta": delta,
    }


def compute_delta_from_gini(G_values):
    """Compute relative dispersion Delta of G values."""
    return composition_delta(np.asarray(G_values, dtype=float))


def build_bipartite_graph_for_visualization(hyperedges, communities, K, g_values=None):
    """Build a bipartite graph used for Figure 6-like rendering.

    - circle nodes represent original vertices
    - square nodes represent hyperedges
    """
    comm = np.asarray(communities, dtype=np.int64)
    n = int(comm.shape[0])

    if g_values is None:
        g_summary = summarize_gini(hyperedges, comm, K)
        g_values = g_summary["G_values"]
    else:
        g_values = np.asarray(g_values, dtype=float)

    deg = np.zeros(n, dtype=np.int64)
    for edge in hyperedges:
        idx = np.asarray(edge, dtype=np.int64)
        for u in np.unique(idx):
            if 0 <= u < n:
                deg[u] += 1

    B = nx.Graph()
    for u in range(n):
        B.add_node(
            f"v{u}",
            bipartite="vertex",
            node_id=int(u),
            community=int(comm[u]),
            degree=int(deg[u]),
        )

    for e_idx, edge in enumerate(hyperedges):
        size_e = int(len(edge))
        g_e = float(g_values[e_idx]) if e_idx < g_values.shape[0] else 0.0
        e_name = f"e{e_idx}"
        B.add_node(
            e_name,
            bipartite="hyperedge",
            edge_id=int(e_idx),
            cardinality=size_e,
            gini=g_e,
        )
        for u in np.asarray(edge, dtype=np.int64):
            if 0 <= int(u) < n:
                B.add_edge(f"v{int(u)}", e_name)

    return B


def plot_gini_distribution(
    panel_curves,
    out_png,
    panel_order=None,
    line_order=None,
    legend_title="",
    suptitle="",
):
    """Plot P(G) curves for multiple panels (used by Figure 7 and Figure 8)."""
    if panel_order is None:
        panel_order = list(panel_curves.keys())

    n_panels = len(panel_order)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.8 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    if line_order is None and panel_order:
        first_panel = panel_order[0]
        line_order = list(panel_curves[first_panel].keys())
    elif line_order is None:
        line_order = []

    cmap = plt.get_cmap("viridis", max(2, len(line_order)))

    for ax_idx, panel in enumerate(panel_order):
        ax = axes[ax_idx]
        curves = panel_curves[panel]

        for line_idx, line_key in enumerate(line_order):
            if line_key not in curves:
                continue
            x, y = curves[line_key]
            ax.plot(
                x,
                y,
                linewidth=1.8,
                color=cmap(line_idx),
                label=str(line_key),
            )

        ax.set_title(panel)
        ax.set_xlabel("G")
        ax.set_ylabel("P(G)")
        ax.grid(alpha=0.25)

    for idx in range(n_panels, axes.size):
        axes[idx].axis("off")

    handles, labels = axes[0].get_legend_handles_labels() if n_panels > 0 else ([], [])
    if handles:
        fig.legend(
            handles,
            labels,
            title=legend_title,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncols=min(6, max(1, int(np.ceil(len(labels) / 2)))),
            fontsize=8,
        )

    if suptitle:
        fig.suptitle(suptitle, y=1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mean_g(
    x_values,
    series_map,
    out_png,
    xlabel,
    ylabel="mean G",
    title="",
    legend_title="",
):
    """Plot mean G curves for multiple series."""
    x = np.asarray(x_values, dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    cmap = plt.get_cmap("tab10", max(3, len(series_map)))

    for idx, (name, y_values) in enumerate(series_map.items()):
        y = np.asarray(y_values, dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.9,
            markersize=4.0,
            color=cmap(idx),
            label=str(name),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(title=legend_title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta(q_over_p_grid, delta_by_strategy, out_png, n_over_e_order):
    """Plot Delta vs q/p as strategy subplots (Figure 12 style)."""
    x = np.asarray(q_over_p_grid, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()

    cmap = plt.get_cmap("viridis", max(2, len(n_over_e_order)))

    for ax, strategy in zip(axes, STRATEGIES):
        series = delta_by_strategy[strategy]
        for idx, ratio in enumerate(n_over_e_order):
            ratio_key = _ratio_key(ratio)
            y = np.asarray(series[ratio_key], dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.2,
                linewidth=1.6,
                color=cmap(idx),
                label=str(ratio_key),
            )
        ax.set_title(strategy)
        ax.set_xlabel("q / p")
        ax.set_ylabel("Delta")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="N / E",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=min(6, len(labels)),
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ratio_key(value: float) -> str:
    return f"{float(value):g}"


def _resolve_node_order(order: str) -> str:
    order_clean = str(order).strip().lower()
    if order_clean not in NODE_ORDERS:
        raise ValueError(f"node_order must be one of {NODE_ORDERS}, got {order!r}")
    return order_clean


def _balanced_communities(n: int, K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(make_balanced_labels(n=n, K=K, rng=rng), dtype=np.int64)


def _estimate_g_distribution(g_values: np.ndarray, bins: int):
    edges = np.linspace(0.0, 1.0, bins)
    hist, bin_edges = np.histogram(np.clip(g_values, 0.0, 1.0), bins=edges)
    total = int(hist.sum())
    probs = hist.astype(float) / float(total) if total > 0 else np.zeros_like(hist, dtype=float)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, probs


def _generate_realization(
    *,
    N: int,
    E: int,
    K: int,
    p: float,
    q: float,
    strategy: str,
    node_order: str,
    seed: int,
):
    order = _resolve_node_order(node_order)

    master_rng = np.random.default_rng(seed)
    comm_seed = int(master_rng.integers(1, 2**31 - 1))
    edge_seed = int(master_rng.integers(1, 2**31 - 1))

    communities = _balanced_communities(N, K, comm_seed)
    hyperedges = generate_hypergraph(
        N=N,
        E=E,
        K=K,
        p=p,
        q=q,
        strategy=strategy,
        node_order=order,
        seed=edge_seed,
        communities=communities,
    )
    g_summary = summarize_gini(hyperedges, communities, K)
    return hyperedges, communities, g_summary


def _local_peak_summary(x: np.ndarray, y: np.ndarray):
    idx = int(np.argmax(y))
    interior = 0 < idx < (len(y) - 1)
    if interior:
        local_peak = bool(y[idx] > y[idx - 1] and y[idx] > y[idx + 1])
    else:
        local_peak = False
    x_peak = float(x[idx])
    return {
        "peak_index": int(idx),
        "peak_x": x_peak,
        "peak_y": float(y[idx]),
        "is_interior": bool(interior),
        "is_local_peak": bool(local_peak),
        "near_0_2": bool(abs(x_peak - 0.2) <= 0.12),
    }


# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------
def run_figure6(cfg: Exp3Config, out_png: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    axes = axes.ravel()

    panel_stats = []
    all_hyperedge_colors = []

    for idx, q in enumerate(cfg.fig6_q_values):
        seed = cfg.seed + 6100 + idx
        hyperedges, communities, g_summary = _generate_realization(
            N=cfg.fig6_N,
            E=cfg.fig6_E,
            K=cfg.fig6_K,
            p=cfg.fig6_p,
            q=float(q),
            strategy=cfg.fig6_strategy,
            node_order="random",
            seed=seed,
        )

        B = build_bipartite_graph_for_visualization(
            hyperedges=hyperedges,
            communities=communities,
            K=cfg.fig6_K,
            g_values=g_summary["G_values"],
        )

        # D3 force layout is unavailable in this Python-only environment.
        # We use networkx spring_layout as a force-directed approximation.
        pos = nx.spring_layout(B, seed=seed, k=0.35, iterations=220)

        ax = axes[idx]
        vertex_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "vertex"]
        edge_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "hyperedge"]

        v_comm = np.asarray([B.nodes[n]["community"] for n in vertex_nodes], dtype=int)
        v_deg = np.asarray([B.nodes[n]["degree"] for n in vertex_nodes], dtype=float)
        v_sizes = 18.0 + 12.0 * np.sqrt(np.maximum(v_deg, 1.0))

        e_g = np.asarray([B.nodes[n]["gini"] for n in edge_nodes], dtype=float)
        e_sz = np.asarray([B.nodes[n]["cardinality"] for n in edge_nodes], dtype=float)
        e_sizes = 12.0 + 14.0 * np.sqrt(np.maximum(e_sz, 1.0))

        nx.draw_networkx_edges(B, pos=pos, ax=ax, width=0.5, alpha=0.22, edge_color="#808080")
        nx.draw_networkx_nodes(
            B,
            pos=pos,
            nodelist=vertex_nodes,
            node_size=v_sizes,
            node_color=v_comm,
            cmap="tab10",
            vmin=0,
            vmax=max(1, cfg.fig6_K - 1),
            node_shape="o",
            ax=ax,
            linewidths=0.3,
            edgecolors="black",
        )
        nx.draw_networkx_nodes(
            B,
            pos=pos,
            nodelist=edge_nodes,
            node_size=e_sizes,
            node_color=e_g,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            node_shape="s",
            ax=ax,
            linewidths=0.3,
            edgecolors="black",
        )

        all_hyperedge_colors.extend(e_g.tolist())
        ax.set_axis_off()
        ax.set_title(
            f"({chr(97 + idx)}) q = {q:g} | mean G={g_summary['mean_G']:.3f}, Delta={g_summary['delta']:.3f}",
            fontsize=10,
        )

        panel_stats.append(
            {
                "q": float(q),
                "mean_G": float(g_summary["mean_G"]),
                "delta": float(g_summary["delta"]),
                "num_hyperedges": int(len(hyperedges)),
            }
        )

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array(np.asarray(all_hyperedge_colors, dtype=float))
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Normalized Gini (G)")

    fig.suptitle(
        "Figure 6-like: Bipartite hypergraph visualization (majority strategy)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "settings": {
            "strategy": cfg.fig6_strategy,
            "N": cfg.fig6_N,
            "E": cfg.fig6_E,
            "K": cfg.fig6_K,
            "p": cfg.fig6_p,
            "q_values": [float(v) for v in cfg.fig6_q_values],
        },
        "panels": panel_stats,
    }


def run_figure7(cfg: Exp3Config, out_png: Path):
    panel_curves = {strategy: {} for strategy in STRATEGIES}
    summary = {strategy: {} for strategy in STRATEGIES}

    total_steps = len(STRATEGIES) * len(cfg.fig7_q_over_p) * cfg.fig7_realizations
    progress = None if cfg.no_progress else LiveProgress(total_steps)
    master_rng = np.random.default_rng(cfg.seed + 7000)

    for strategy in STRATEGIES:
        for q_ratio in cfg.fig7_q_over_p:
            q = float(q_ratio) * cfg.fig7_p
            all_g = []

            for rep in range(1, cfg.fig7_realizations + 1):
                rep_seed = int(master_rng.integers(1, 2**31 - 1))
                _, _, g_summary = _generate_realization(
                    N=cfg.fig7_N,
                    E=cfg.fig7_E,
                    K=cfg.fig7_K,
                    p=cfg.fig7_p,
                    q=q,
                    strategy=strategy,
                    node_order="random",
                    seed=rep_seed,
                )
                all_g.append(g_summary["G_values"])

                if progress is not None:
                    progress.update("q/p", float(q_ratio), rep, cfg.fig7_realizations, strategy)

            g_values = np.concatenate(all_g) if all_g else np.zeros(0, dtype=float)
            centers, probs = _estimate_g_distribution(g_values, cfg.gini_bins)
            ratio_key = _ratio_key(q_ratio)
            panel_curves[strategy][ratio_key] = (centers, probs)
            summary[strategy][ratio_key] = {
                "mean_G": float(np.mean(g_values)) if g_values.size > 0 else 0.0,
                "delta": float(compute_delta_from_gini(g_values)),
                "pure_fraction_G_ge_0_9": float(np.mean(g_values >= 0.9)) if g_values.size > 0 else 0.0,
                "sample_size": int(g_values.size),
            }

    if progress is not None:
        progress.close()

    plot_gini_distribution(
        panel_curves=panel_curves,
        out_png=out_png,
        panel_order=list(STRATEGIES),
        line_order=[_ratio_key(v) for v in cfg.fig7_q_over_p],
        legend_title="q/p",
        suptitle="Figure 7-like: Distribution P(G) by strategy and q/p",
    )

    return {
        "settings": {
            "N": cfg.fig7_N,
            "E": cfg.fig7_E,
            "K": cfg.fig7_K,
            "p": cfg.fig7_p,
            "q_over_p": [float(v) for v in cfg.fig7_q_over_p],
            "realizations": int(cfg.fig7_realizations),
        },
        "summary": summary,
    }


def run_figure8(cfg: Exp3Config, out_png: Path):
    panel_curves = {strategy: {} for strategy in STRATEGIES}
    summary = {strategy: {} for strategy in STRATEGIES}

    q = float(cfg.fig8_q_over_p) * cfg.fig8_p
    total_steps = len(STRATEGIES) * len(NODE_ORDERS) * cfg.fig8_realizations
    progress = None if cfg.no_progress else LiveProgress(total_steps)
    master_rng = np.random.default_rng(cfg.seed + 8000)

    for strategy in STRATEGIES:
        for node_order in NODE_ORDERS:
            all_g = []
            for rep in range(1, cfg.fig8_realizations + 1):
                rep_seed = int(master_rng.integers(1, 2**31 - 1))
                _, _, g_summary = _generate_realization(
                    N=cfg.fig8_N,
                    E=cfg.fig8_E,
                    K=cfg.fig8_K,
                    p=cfg.fig8_p,
                    q=q,
                    strategy=strategy,
                    node_order=node_order,
                    seed=rep_seed,
                )
                all_g.append(g_summary["G_values"])

                if progress is not None:
                    progress.update("order", node_order, rep, cfg.fig8_realizations, strategy)

            g_values = np.concatenate(all_g) if all_g else np.zeros(0, dtype=float)
            centers, probs = _estimate_g_distribution(g_values, cfg.gini_bins)
            panel_curves[strategy][node_order] = (centers, probs)
            summary[strategy][node_order] = {
                "mean_G": float(np.mean(g_values)) if g_values.size > 0 else 0.0,
                "delta": float(compute_delta_from_gini(g_values)),
                "sample_size": int(g_values.size),
            }

    if progress is not None:
        progress.close()

    plot_gini_distribution(
        panel_curves=panel_curves,
        out_png=out_png,
        panel_order=list(STRATEGIES),
        line_order=list(NODE_ORDERS),
        legend_title="node order",
        suptitle="Figure 8-like: Node traversal order effect on P(G)",
    )

    return {
        "settings": {
            "N": cfg.fig8_N,
            "E": cfg.fig8_E,
            "K": cfg.fig8_K,
            "p": cfg.fig8_p,
            "q": float(q),
            "q_over_p": float(cfg.fig8_q_over_p),
            "realizations": int(cfg.fig8_realizations),
            "orders": list(NODE_ORDERS),
        },
        "summary": summary,
    }


def run_shared_sweep_9_to_12(cfg: Exp3Config):
    q_grid = np.asarray(cfg.fig9_12_q_over_p, dtype=float)
    n_over_e_values = list(cfg.fig9_12_n_over_e)

    stats = {
        strategy: {
            _ratio_key(ratio): {
                "mean_G_curve": [],
                "delta_curve": [],
            }
            for ratio in n_over_e_values
        }
        for strategy in STRATEGIES
    }

    total_steps = (
        len(STRATEGIES)
        * len(n_over_e_values)
        * len(q_grid)
        * cfg.fig9_12_realizations
    )
    progress = None if cfg.no_progress else LiveProgress(total_steps)
    master_rng = np.random.default_rng(cfg.seed + 9000)

    for strategy in STRATEGIES:
        for ratio in n_over_e_values:
            N = int(round(cfg.fig9_12_E * float(ratio)))
            ratio_key = _ratio_key(ratio)

            for q_ratio in q_grid:
                q = float(q_ratio) * cfg.fig9_12_p
                per_graph_mean_g = []
                per_graph_delta = []

                for rep in range(1, cfg.fig9_12_realizations + 1):
                    rep_seed = int(master_rng.integers(1, 2**31 - 1))
                    _, _, g_summary = _generate_realization(
                        N=N,
                        E=cfg.fig9_12_E,
                        K=cfg.fig9_12_K,
                        p=cfg.fig9_12_p,
                        q=q,
                        strategy=strategy,
                        node_order="random",
                        seed=rep_seed,
                    )
                    g_values = g_summary["G_values"]
                    per_graph_mean_g.append(float(np.mean(g_values)) if g_values.size > 0 else 0.0)
                    per_graph_delta.append(float(compute_delta_from_gini(g_values)))

                    if progress is not None:
                        progress.update("q/p", float(q_ratio), rep, cfg.fig9_12_realizations, strategy)

                stats[strategy][ratio_key]["mean_G_curve"].append(float(np.mean(per_graph_mean_g)))
                stats[strategy][ratio_key]["delta_curve"].append(float(np.mean(per_graph_delta)))

    if progress is not None:
        progress.close()

    return {
        "settings": {
            "E": cfg.fig9_12_E,
            "K": cfg.fig9_12_K,
            "p": cfg.fig9_12_p,
            "n_over_e": [float(v) for v in n_over_e_values],
            "q_over_p": [float(v) for v in q_grid.tolist()],
            "realizations": int(cfg.fig9_12_realizations),
            "node_order": "random",
        },
        "curves": stats,
    }


def render_figures_9_10_11_12(cfg: Exp3Config, shared_stats, fig_dir: Path):
    q_grid = np.asarray(shared_stats["settings"]["q_over_p"], dtype=float)
    n_over_e_values = [float(v) for v in shared_stats["settings"]["n_over_e"]]

    # Figure 9: weighted only, mean G vs q/p for different N/E
    weighted_series = {
        _ratio_key(r): shared_stats["curves"]["weighted"][_ratio_key(r)]["mean_G_curve"]
        for r in n_over_e_values
    }
    fig9_png = fig_dir / "figure9_mean_g_weighted.png"
    plot_mean_g(
        x_values=q_grid,
        series_map=weighted_series,
        out_png=fig9_png,
        xlabel="q / p",
        ylabel="mean G",
        title="Figure 9-like: mean G (weighted strategy)",
        legend_title="N / E",
    )

    # Figure 10: scaling at q/p = 1 for weighted strategy
    x_scaling = 1.0 / np.sqrt(np.asarray(n_over_e_values, dtype=float))
    y_scaling = np.asarray(
        [shared_stats["curves"]["weighted"][_ratio_key(r)]["mean_G_curve"][-1] for r in n_over_e_values],
        dtype=float,
    )
    coef = np.polyfit(x_scaling, y_scaling, deg=1)
    y_hat = np.polyval(coef, x_scaling)
    sse = float(np.sum((y_scaling - y_hat) ** 2))
    sst = float(np.sum((y_scaling - float(np.mean(y_scaling))) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0.0 else 1.0

    fig10_png = fig_dir / "figure10_scaling.png"
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(x_scaling, y_scaling, s=55, color="#1f77b4", edgecolors="black", linewidths=0.5)
    x_line = np.linspace(float(np.min(x_scaling)) * 0.95, float(np.max(x_scaling)) * 1.05, 200)
    y_line = np.polyval(coef, x_line)
    ax.plot(x_line, y_line, color="#d62728", linewidth=1.9)
    for x, y, ratio in zip(x_scaling, y_scaling, n_over_e_values):
        ax.text(x, y, f" N/E={ratio:g}", fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("1 / sqrt(N / E)")
    ax.set_ylabel("mean G at q/p = 1")
    ax.set_title("Figure 10-like: scaling of mean G at q/p = 1")
    ax.text(
        0.03,
        0.97,
        f"fit: y = {coef[0]:.3f}x + {coef[1]:.3f}\\nR^2 = {r2:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig10_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 11: mean G vs q/p at N/E = 5 for all strategies
    ratio_for_fig11 = 5.0
    ratio_key_fig11 = _ratio_key(ratio_for_fig11)
    strategy_series = {
        strategy: shared_stats["curves"][strategy][ratio_key_fig11]["mean_G_curve"]
        for strategy in STRATEGIES
    }
    fig11_png = fig_dir / "figure11_strategy_comparison.png"
    plot_mean_g(
        x_values=q_grid,
        series_map=strategy_series,
        out_png=fig11_png,
        xlabel="q / p",
        ylabel="mean G",
        title="Figure 11-like: strategy comparison of mean G (N/E = 5)",
        legend_title="strategy",
    )

    # Figure 12: Delta vs q/p for each strategy with N/E curves
    delta_by_strategy = {
        strategy: {
            _ratio_key(r): shared_stats["curves"][strategy][_ratio_key(r)]["delta_curve"]
            for r in n_over_e_values
        }
        for strategy in STRATEGIES
    }
    fig12_png = fig_dir / "figure12_delta.png"
    plot_delta(
        q_over_p_grid=q_grid,
        delta_by_strategy=delta_by_strategy,
        out_png=fig12_png,
        n_over_e_order=n_over_e_values,
    )

    # Peak detection for max strategy around interior q/p ~ 0.2
    max_peak_by_ratio = {}
    max_curves = shared_stats["curves"]["max"]
    for ratio in n_over_e_values:
        ratio_key = _ratio_key(ratio)
        y = np.asarray(max_curves[ratio_key]["delta_curve"], dtype=float)
        max_peak_by_ratio[ratio_key] = _local_peak_summary(q_grid, y)

    y_mean_max = np.mean(
        np.asarray([max_curves[_ratio_key(r)]["delta_curve"] for r in n_over_e_values], dtype=float),
        axis=0,
    )
    max_peak_overall = _local_peak_summary(q_grid, y_mean_max)

    return {
        "figure9": {
            "out_png": str(fig9_png),
            "weighted_mean_G": weighted_series,
        },
        "figure10": {
            "out_png": str(fig10_png),
            "x": x_scaling.tolist(),
            "y": y_scaling.tolist(),
            "linear_fit": {
                "slope": float(coef[0]),
                "intercept": float(coef[1]),
                "r2": float(r2),
            },
        },
        "figure11": {
            "out_png": str(fig11_png),
            "n_over_e": ratio_for_fig11,
            "mean_G_by_strategy": strategy_series,
        },
        "figure12": {
            "out_png": str(fig12_png),
            "delta_by_strategy": delta_by_strategy,
            "max_strategy_peak": {
                "by_n_over_e": max_peak_by_ratio,
                "overall": max_peak_overall,
            },
        },
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def _paper_settings_summary(cfg: Exp3Config):
    return [
        "Figure 6: strategy=majority, N=E=80, K=4, p=0.1, q={0,0.01,0.05,0.1}",
        "Figure 7: p=0.1, N=2000, E=200, K=4, strategies={majority,max,min,weighted}, q/p sweep, 100 realizations",
        "Figure 8: p=0.03, q=0.4p, N=1000, E=200, K=4, orders={random,fixed,community}",
        "Figure 9: strategy=weighted, E=200, N/E={1,1.5,2.5,3.5,5,10}, q/p sweep, 100 realizations",
        "Figure 10: E=200, p=q=0.03, x=1/sqrt(N/E), y=mean G",
        "Figure 11: E=200, N=1000, strategies={majority,max,min,weighted}, q/p sweep",
        "Figure 12: p=0.03, E=200, N/E={1,1.5,2.5,3.5,5,10}, strategies={majority,max,min,weighted}",
        "majority tie-breaking: ties are resolved by choosing the smallest community index",
    ]


def _implementation_assumptions(cfg: Exp3Config):
    return [
        "Community assignment uses balanced labels with random node permutation for each realization.",
        "Node order implementation is separated via order mode {random, fixed, community}; generation uses src.common.generate_hypergraph.",
        "Figure 6 layout uses networkx spring_layout as a force-directed approximation (D3 is not used in this Python script).",
        "P(G) uses fixed histogram bins on [0,1] without extra smoothing.",
        "Quick mode reduces realizations and q/p grid size for smoke testing only.",
    ]


def build_report_markdown(summary: dict, cfg: Exp3Config) -> str:
    fig7 = summary["figure7"]["summary"]
    fig8 = summary["figure8"]["summary"]
    shared = summary["shared_9_to_12"]
    fig10 = summary["figures_9_10_11_12"]["figure10"]
    max_peak = summary["figures_9_10_11_12"]["figure12"]["max_strategy_peak"]

    # point 1: low q/p purity
    low_key = _ratio_key(min(cfg.fig7_q_over_p))
    low_pure = np.mean([fig7[s][low_key]["pure_fraction_G_ge_0_9"] for s in STRATEGIES])

    # point 2: q/p ~ 1 => small G
    q_grid = np.asarray(shared["settings"]["q_over_p"], dtype=float)
    one_idx = int(np.argmin(np.abs(q_grid - 1.0)))
    ne5_key = _ratio_key(5.0)
    g_at_one = {
        s: float(shared["curves"][s][ne5_key]["mean_G_curve"][one_idx])
        for s in STRATEGIES
    }
    g_at_one_avg = float(np.mean(list(g_at_one.values())))

    # point 3: majority/weighted dominance
    mean_curve_strength = {
        s: float(np.mean(shared["curves"][s][ne5_key]["mean_G_curve"]))
        for s in STRATEGIES
    }

    # point 4: max interior peak
    max_peak_overall = max_peak["overall"]

    lines = []
    lines.append("# Reference 2 - Experiment 3 Report")
    lines.append("")
    lines.append("## 필수 점검 항목")
    lines.append("")
    lines.append(
        "1. low q/p에서 hyperedge가 pure해지는지: "
        f"평균 pure fraction (G>=0.9, q/p={low_key}) = {low_pure:.3f}. "
        "값이 높을수록 low q/p에서 pure hyperedge가 많음을 의미한다."
    )
    lines.append(
        "2. q/p가 1에 가까워지면 G가 작아지는지: "
        f"N/E=5에서 q/p=1일 때 strategy별 mean G={g_at_one}, 평균={g_at_one_avg:.3f}."
    )
    lines.append(
        "3. majority, weighted가 dominant community를 더 강화하는지: "
        "N/E=5에서 q/p 전 구간 평균 mean G 비교 "
        f"{mean_curve_strength}. 값이 큰 전략일수록 더 순수한(덜 mixed) hyperedge를 만든다."
    )
    lines.append(
        "4. max 전략에서 Delta가 내부 최대값을 가지는지: "
        f"overall peak = q/p {max_peak_overall['peak_x']:.3f}, "
        f"interior={max_peak_overall['is_interior']}, local_peak={max_peak_overall['is_local_peak']}, "
        f"near_0.2={max_peak_overall['near_0_2']}."
    )
    lines.append(
        "5. 논문과 차이가 난다면 가능한 원인: seed 민감도, majority tie-breaking 규칙, "
        "Figure 6의 layout approximation(D3 대신 spring layout), histogram binning/smoothing 설정 차이."
    )
    lines.append("")
    lines.append("## Figure 10 Scaling")
    lines.append(
        f"- linear fit: y = {fig10['linear_fit']['slope']:.4f} x + {fig10['linear_fit']['intercept']:.4f}, "
        f"R^2 = {fig10['linear_fit']['r2']:.4f}"
    )
    lines.append("")
    lines.append("## 논문에서 직접 명시된 설정")
    for item in _paper_settings_summary(cfg):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 내가 둔 구현 가정")
    for item in _implementation_assumptions(cfg):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 출력 파일")
    lines.append("- figures/figure6_bipartite_majority.png")
    lines.append("- figures/figure7_gini_distribution.png")
    lines.append("- figures/figure8_order_effect.png")
    lines.append("- figures/figure9_mean_g_weighted.png")
    lines.append("- figures/figure10_scaling.png")
    lines.append("- figures/figure11_strategy_comparison.png")
    lines.append("- figures/figure12_delta.png")
    lines.append("- results/composition_summary.json")

    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _apply_quick_mode(cfg: Exp3Config) -> Exp3Config:
    if not cfg.quick:
        return cfg

    cfg.fig6_N = 48
    cfg.fig6_E = 48

    cfg.fig7_N = 600
    cfg.fig7_E = 120
    cfg.fig7_q_over_p = (0.01, 0.3, 0.6, 1.0)
    cfg.fig7_realizations = min(cfg.fig7_realizations, 3)

    cfg.fig8_N = 400
    cfg.fig8_E = 120
    cfg.fig8_realizations = min(cfg.fig8_realizations, 3)

    cfg.fig9_12_n_over_e = (1.0, 5.0, 10.0)
    cfg.fig9_12_q_over_p = tuple(np.linspace(0.0, 1.0, 5))
    cfg.fig9_12_realizations = min(cfg.fig9_12_realizations, 3)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Reference 2 - Experiment 3 (Figure 6~12)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outdir", type=str, default="experiments/reference_2_exp3")
    parser.add_argument("--fig7-realizations", type=int, default=100)
    parser.add_argument("--fig8-realizations", type=int, default=100)
    parser.add_argument("--fig9-12-realizations", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help="fast smoke-test mode")
    parser.add_argument("--no-progress", action="store_true")
    args, _ = parser.parse_known_args()

    cfg = Exp3Config(
        seed=args.seed,
        outdir=Path(args.outdir),
        no_progress=args.no_progress,
        quick=args.quick,
        fig7_realizations=args.fig7_realizations,
        fig8_realizations=args.fig8_realizations,
        fig9_12_realizations=args.fig9_12_realizations,
    )
    cfg = _apply_quick_mode(cfg)

    fig_dir = cfg.outdir / "figures"
    res_dir = cfg.outdir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    t_all = perf_counter()
    summary = {
        "seed": int(cfg.seed),
        "quick_mode": bool(cfg.quick),
    }

    print("[Exp3] Running Figure 6 ...")
    t0 = perf_counter()
    summary["figure6"] = run_figure6(cfg, fig_dir / "figure6_bipartite_majority.png")
    print(f"[Exp3] Figure 6 done in {perf_counter() - t0:.1f}s")

    print("[Exp3] Running Figure 7 ...")
    t0 = perf_counter()
    summary["figure7"] = run_figure7(cfg, fig_dir / "figure7_gini_distribution.png")
    print(f"[Exp3] Figure 7 done in {perf_counter() - t0:.1f}s")

    print("[Exp3] Running Figure 8 ...")
    t0 = perf_counter()
    summary["figure8"] = run_figure8(cfg, fig_dir / "figure8_order_effect.png")
    print(f"[Exp3] Figure 8 done in {perf_counter() - t0:.1f}s")

    print("[Exp3] Running shared sweep for Figure 9~12 ...")
    t0 = perf_counter()
    shared = run_shared_sweep_9_to_12(cfg)
    summary["shared_9_to_12"] = shared
    summary["figures_9_10_11_12"] = render_figures_9_10_11_12(cfg, shared, fig_dir)
    print(f"[Exp3] Figure 9~12 done in {perf_counter() - t0:.1f}s")

    # diagnostics and reporting
    summary["paper_reported_settings"] = _paper_settings_summary(cfg)
    summary["implementation_assumptions"] = _implementation_assumptions(cfg)

    out_json = res_dir / "composition_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_text = build_report_markdown(summary, cfg)
    out_report = res_dir / "exp3_report.md"
    out_report.write_text(report_text, encoding="utf-8")

    elapsed = perf_counter() - t_all
    print("[Exp3] Done.")
    print(f"[Exp3] Total elapsed: {elapsed:.1f}s")
    print(f"[Exp3] Figure dir : {fig_dir.resolve()}")
    print(f"[Exp3] Summary    : {out_json.resolve()}")
    print(f"[Exp3] Report     : {out_report.resolve()}")

    print("\n[논문에서 직접 명시된 설정]")
    for item in summary["paper_reported_settings"]:
        print(f"- {item}")

    print("\n[내가 둔 구현 가정]")
    for item in summary["implementation_assumptions"]:
        print(f"- {item}")


if __name__ == "__main__":
    main()
