# -*- coding: utf-8 -*-
from __future__ import annotations

"""Reproduce Experiment 2 (Figure 3/4/5) for the HySBM paper."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import LiveProgress  # noqa: E402
from src.hypergraph_sbm import (  # noqa: E402
    generate_hypergraph,
    hyperedge_sizes,
    node_degrees_from_hyperedges,
)


@dataclass
class Exp2Config:
    """Configuration for Experiment 2."""

    num_graphs: int = 100
    seed: int = 2026
    node_order: str = "random"
    figure3_q_ratios: tuple = (0.0, 0.3, 0.7, 1.0)
    figure4_q_ratios: tuple = tuple(np.linspace(0.0, 1.0, 11))
    figure5_q_ratios: tuple = (0.1, 0.01)


def compute_node_degrees(hyperedges, N: int) -> np.ndarray:
    """Compute node degrees from generated hyperedges."""
    return node_degrees_from_hyperedges(n=N, hyperedges=hyperedges)


def compute_hyperedge_sizes(hyperedges) -> np.ndarray:
    """Compute hyperedge sizes from generated hyperedges."""
    return hyperedge_sizes(hyperedges=hyperedges)


def _safe_kl(emp: np.ndarray, fit: np.ndarray, eps: float = 1e-12) -> float:
    emp = np.asarray(emp, dtype=float)
    fit = np.asarray(fit, dtype=float)
    emp = np.clip(emp, eps, 1.0)
    fit = np.clip(fit, eps, 1.0)
    return float(np.sum(emp * np.log(emp / fit)))


def fit_effective_binomial_for_degree(
    degrees: np.ndarray,
    E: int,
    reference_theta: float | None = None,
) -> dict:
    """Fit Binomial(E, theta) to degree samples and infer effective E*."""
    deg = np.asarray(degrees, dtype=np.int64)
    if deg.size == 0:
        raise ValueError("degrees must not be empty")

    theta_hat = float(np.clip(deg.mean() / float(E), 1e-12, 1.0 - 1e-12))
    support = np.arange(0, E + 1, dtype=np.int64)

    cnt = np.bincount(np.clip(deg, 0, E), minlength=E + 1).astype(float)
    empirical_pmf = cnt / float(cnt.sum())
    fit_pmf = binom.pmf(support, n=E, p=theta_hat)

    tv = float(0.5 * np.sum(np.abs(empirical_pmf - fit_pmf)))
    kl = _safe_kl(empirical_pmf, fit_pmf)

    out = {
        "theta_hat": theta_hat,
        "support": support,
        "empirical_pmf": empirical_pmf,
        "fit_pmf": fit_pmf,
        "tv_distance": tv,
        "kl_divergence": kl,
    }
    if reference_theta is not None:
        ratio = float(theta_hat / max(reference_theta, 1e-12))
        out["effective_ratio"] = ratio
        out["effective_star"] = float(E * ratio)  # interpreted as E*
    return out


def fit_effective_binomial_for_hyperedge_size(
    sizes: np.ndarray,
    N: int,
    reference_theta: float | None = None,
) -> dict:
    """Fit 1 + Binomial(N-1, theta) to hyperedge-size samples and infer N*."""
    s = np.asarray(sizes, dtype=np.int64)
    if s.size == 0:
        raise ValueError("sizes must not be empty")

    # Hyperedges start with 1 seed node by construction, so we fit m-1.
    n_bin = N - 1
    shifted = np.clip(s - 1, 0, n_bin)
    theta_hat = float(np.clip(shifted.mean() / float(n_bin), 1e-12, 1.0 - 1e-12))

    support_size = np.arange(1, N + 1, dtype=np.int64)
    cnt_shifted = np.bincount(shifted, minlength=n_bin + 1).astype(float)
    empirical_pmf = cnt_shifted / float(cnt_shifted.sum())
    fit_pmf = binom.pmf(np.arange(0, n_bin + 1), n=n_bin, p=theta_hat)

    tv = float(0.5 * np.sum(np.abs(empirical_pmf - fit_pmf)))
    kl = _safe_kl(empirical_pmf, fit_pmf)

    out = {
        "theta_hat": theta_hat,
        "support": support_size,
        "empirical_pmf": empirical_pmf,  # on shifted support [0..N-1]
        "fit_pmf": fit_pmf,  # on shifted support [0..N-1]
        "tv_distance": tv,
        "kl_divergence": kl,
    }
    if reference_theta is not None:
        ratio = float(theta_hat / max(reference_theta, 1e-12))
        out["effective_ratio"] = ratio
        out["effective_star"] = float(N * ratio)  # interpreted as N*
    return out


def plot_degree_distribution(
    ax,
    degrees: np.ndarray,
    E: int,
    fit_info: dict,
    title: str,
) -> None:
    """Plot empirical degree PMF with fitted binomial overlay."""
    emp = fit_info["empirical_pmf"]
    fit = fit_info["fit_pmf"]
    support = np.arange(0, E + 1)

    nz = np.where(emp > 0)[0]
    left = max(0, int(nz.min()) - 2) if nz.size > 0 else 0
    right = min(E, int(nz.max()) + 2) if nz.size > 0 else E

    ax.plot(support[left : right + 1], emp[left : right + 1], "o", color="black", markersize=3.4)
    ax.plot(
        support[left : right + 1],
        fit[left : right + 1],
        color="#d62728",
        linewidth=1.8,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    ax.grid(alpha=0.25)


def plot_size_distribution(
    ax,
    sizes: np.ndarray,
    N: int,
    fit_info: dict,
    title: str,
) -> None:
    """Plot empirical hyperedge-size PMF with fitted shifted-binomial overlay."""
    emp = fit_info["empirical_pmf"]  # shifted support [0..N-1]
    fit = fit_info["fit_pmf"]  # shifted support [0..N-1]
    support = np.arange(1, N + 1)  # unshifted size support

    nz = np.where(emp > 0)[0]
    left_s = max(0, int(nz.min()) - 2) if nz.size > 0 else 0
    right_s = min(N - 1, int(nz.max()) + 2) if nz.size > 0 else N - 1

    sl = slice(left_s, right_s + 1)
    ax.plot(support[sl], emp[sl], "o", color="black", markersize=3.4)
    ax.plot(support[sl], fit[sl], color="#1f77b4", linewidth=1.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("m")
    ax.set_ylabel("P(m)")
    ax.grid(alpha=0.25)


def _collect_samples(
    *,
    N: int,
    E: int,
    K: int,
    p: float,
    q_ratio: float,
    strategy: str,
    num_graphs: int,
    node_order: str,
    master_rng: np.random.Generator,
    progress: LiveProgress | None,
    progress_tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    q = float(q_ratio) * float(p)
    degree_all = []
    size_all = []

    for rep in range(1, num_graphs + 1):
        rep_seed = int(master_rng.integers(1, 2**31 - 1))

        # If p == q, community labels do not change join probability.
        # This effectively removes planted structure and matches a random hypergraph-like case.
        edges = generate_hypergraph(
            N=N,
            E=E,
            K=K,
            p=p,
            q=q,
            strategy=strategy,
            node_order=node_order,
            seed=rep_seed,
        )

        degree_all.append(compute_node_degrees(edges, N))
        size_all.append(compute_hyperedge_sizes(edges))

        if progress is not None:
            progress.update(progress_tag, q_ratio, rep, num_graphs, strategy)

    return np.concatenate(degree_all), np.concatenate(size_all)


def _r2_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    sse = float(np.sum((y - y_hat) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    if sst <= 0.0:
        return 1.0
    return float(1.0 - sse / sst)


def _smooth_1d(y: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return y.copy()
    w = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(y_pad, w, mode="valid")


def _detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    min_rel_height: float = 0.08,
) -> tuple[int, list[int], np.ndarray]:
    ys = _smooth_1d(y, window=7)
    if ys.size < 3:
        return 0, [], ys

    thr = float(min_rel_height * np.max(ys))
    peaks = []
    for i in range(1, ys.size - 1):
        if ys[i - 1] < ys[i] and ys[i] > ys[i + 1] and ys[i] >= thr:
            peaks.append(int(x[i]))
    return len(peaks), peaks, ys


def run_experiment2(cfg: Exp2Config, outdir: Path, show_progress: bool = True) -> None:
    """Run Figure 3/4/5 workflows and save outputs."""
    fig_dir = outdir / "figures"
    res_dir = outdir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Shared constants from the paper setup for Experiment 2.
    N = 1000
    E = 200
    K = 4

    total_graphs = (
        len(cfg.figure3_q_ratios) * cfg.num_graphs
        + len(cfg.figure4_q_ratios) * cfg.num_graphs
        + len(cfg.figure5_q_ratios) * cfg.num_graphs
    )
    progress = LiveProgress(total_graphs) if show_progress else None
    master_rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    # 2-A Figure 3
    # ------------------------------------------------------------------
    p_fig3 = 30.0 / float(N)  # 0.03
    fig3_rows = []
    fig3_data = {}

    for q_ratio in cfg.figure3_q_ratios:
        deg, size = _collect_samples(
            N=N,
            E=E,
            K=K,
            p=p_fig3,
            q_ratio=float(q_ratio),
            strategy="weighted",
            num_graphs=cfg.num_graphs,
            node_order=cfg.node_order,
            master_rng=master_rng,
            progress=progress,
            progress_tag="F3 q/p",
        )
        fit_deg = fit_effective_binomial_for_degree(deg, E=E)
        fit_size = fit_effective_binomial_for_hyperedge_size(size, N=N)
        fig3_data[float(q_ratio)] = (deg, size, fit_deg, fit_size)

        fig3_rows.append(
            {
                "q_over_p": float(q_ratio),
                "q": float(q_ratio) * p_fig3,
                "strategy": "weighted",
                "num_graphs": cfg.num_graphs,
                "degree_samples": int(deg.size),
                "size_samples": int(size.size),
                "degree_theta_hat": fit_deg["theta_hat"],
                "degree_tv_distance": fit_deg["tv_distance"],
                "degree_kl_divergence": fit_deg["kl_divergence"],
                "size_theta_hat": fit_size["theta_hat"],
                "size_tv_distance": fit_size["tv_distance"],
                "size_kl_divergence": fit_size["kl_divergence"],
            }
        )

    fig3_df = pd.DataFrame(fig3_rows).sort_values("q_over_p")
    fig3_df.to_csv(res_dir / "figure3_summary.csv", index=False)

    fig3, axes = plt.subplots(2, 4, figsize=(17.5, 7.8))
    for j, q_ratio in enumerate(cfg.figure3_q_ratios):
        deg, size, fit_deg, fit_size = fig3_data[float(q_ratio)]
        plot_degree_distribution(
            ax=axes[0, j],
            degrees=deg,
            E=E,
            fit_info=fit_deg,
            title=f"Degree, q/p={q_ratio:g}",
        )
        plot_size_distribution(
            ax=axes[1, j],
            sizes=size,
            N=N,
            fit_info=fit_size,
            title=f"Size, q/p={q_ratio:g}",
        )

    fig3.suptitle("Figure 3 reproduction: empirical distributions + binomial fits", fontsize=13)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "figure3_degree_and_size.png", dpi=180, bbox_inches="tight")
    plt.close(fig3)

    # ------------------------------------------------------------------
    # 2-B Figure 4
    # ------------------------------------------------------------------
    p_fig4 = 0.03
    rows4 = []
    theta_deg_map = {}
    theta_size_map = {}

    for q_ratio in cfg.figure4_q_ratios:
        deg, size = _collect_samples(
            N=N,
            E=E,
            K=K,
            p=p_fig4,
            q_ratio=float(q_ratio),
            strategy="weighted",
            num_graphs=cfg.num_graphs,
            node_order=cfg.node_order,
            master_rng=master_rng,
            progress=progress,
            progress_tag="F4 q/p",
        )
        fit_deg = fit_effective_binomial_for_degree(deg, E=E)
        fit_size = fit_effective_binomial_for_hyperedge_size(size, N=N)
        qf = float(q_ratio)
        theta_deg_map[qf] = fit_deg["theta_hat"]
        theta_size_map[qf] = fit_size["theta_hat"]
        rows4.append(
            {
                "q_over_p": qf,
                "q": qf * p_fig4,
                "theta_degree_hat": fit_deg["theta_hat"],
                "theta_size_hat": fit_size["theta_hat"],
            }
        )

    ref_key = min(theta_deg_map.keys(), key=lambda x: abs(x - 1.0))
    ref_deg = theta_deg_map[ref_key]
    ref_size = theta_size_map[ref_key]

    for r in rows4:
        qf = float(r["q_over_p"])
        e_ratio = theta_deg_map[qf] / max(ref_deg, 1e-12)
        n_ratio = theta_size_map[qf] / max(ref_size, 1e-12)
        r["E_star"] = float(E * e_ratio)
        r["E_star_over_E"] = float(e_ratio)
        r["N_star"] = float(N * n_ratio)
        r["N_star_over_N"] = float(n_ratio)
        r["theory_ratio"] = float((1.0 - 1.0 / K) * qf + 1.0 / K)

    df4 = pd.DataFrame(rows4).sort_values("q_over_p")
    x4 = df4["q_over_p"].to_numpy(dtype=float)
    yE = df4["E_star_over_E"].to_numpy(dtype=float)
    yN = df4["N_star_over_N"].to_numpy(dtype=float)

    coefE = np.polyfit(x4, yE, 1)
    coefN = np.polyfit(x4, yN, 1)
    predE = np.polyval(coefE, x4)
    predN = np.polyval(coefN, x4)
    df4["linear_E_star_over_E"] = predE
    df4["linear_N_star_over_N"] = predN
    df4["linear_E_slope"] = float(coefE[0])
    df4["linear_E_intercept"] = float(coefE[1])
    df4["linear_N_slope"] = float(coefN[0])
    df4["linear_N_intercept"] = float(coefN[1])
    df4["linear_E_r2"] = _r2_score(yE, predE)
    df4["linear_N_r2"] = _r2_score(yN, predN)
    df4.to_csv(res_dir / "figure4_fit_summary.csv", index=False)

    fig4, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))
    theory = df4["theory_ratio"].to_numpy(dtype=float)
    ax.plot(x4, yE, "o-", color="#1f77b4", linewidth=1.8, label="E*/E (fit)")
    ax.plot(x4, yN, "s-", color="#ff7f0e", linewidth=1.8, label="N*/N (fit)")
    ax.plot(x4, predE, "--", color="#1f77b4", alpha=0.8, label=f"E*/E linear (R^2={_r2_score(yE, predE):.3f})")
    ax.plot(x4, predN, "--", color="#ff7f0e", alpha=0.8, label=f"N*/N linear (R^2={_r2_score(yN, predN):.3f})")
    ax.plot(x4, theory, "-", color="black", linewidth=2.0, label=r"Theory: ((1-1/K) q/p + 1/K)")
    ax.set_xlabel("q/p")
    ax.set_ylabel("Effective ratio")
    ax.set_title("Figure 4 reproduction: effective parameters vs q/p")
    ax.set_ylim(0.15, 1.08)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.7)
    fig4.tight_layout()
    fig4.savefig(fig_dir / "figure4_effective_parameters.png", dpi=180, bbox_inches="tight")
    plt.close(fig4)

    # ------------------------------------------------------------------
    # 2-C Figure 5
    # ------------------------------------------------------------------
    p_fig5 = 0.1
    fig5_data = {}
    peak_json = {"strategy": "min", "settings": {}, "peak_check": {}}

    for q_ratio in cfg.figure5_q_ratios:
        deg, size = _collect_samples(
            N=N,
            E=E,
            K=K,
            p=p_fig5,
            q_ratio=float(q_ratio),
            strategy="min",
            num_graphs=cfg.num_graphs,
            node_order=cfg.node_order,
            master_rng=master_rng,
            progress=progress,
            progress_tag="F5 q/p",
        )
        fit_size = fit_effective_binomial_for_hyperedge_size(size, N=N)
        emp = fit_size["empirical_pmf"]  # shifted support [0..N-1]
        support_size = np.arange(1, N + 1)
        peak_count, peak_positions, smooth = _detect_peaks(support_size, emp, min_rel_height=0.08)

        fig5_data[float(q_ratio)] = {
            "sizes": size,
            "fit": fit_size,
            "smooth": smooth,
            "peak_count": peak_count,
            "peak_positions": peak_positions,
        }
        peak_json["peak_check"][f"{q_ratio:g}"] = {
            "q_over_p": float(q_ratio),
            "q": float(q_ratio) * p_fig5,
            "peak_count": int(peak_count),
            "peak_positions": [int(v) for v in peak_positions],
            "bimodal": bool(peak_count >= 2),
        }

    # q << p with min strategy can separate pure/mixed growth:
    # one low-probability mode (mixed) and one high-probability mode (pure),
    # which may appear as bimodality in P(m).
    fig5, axes5 = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, q_ratio in zip(axes5, cfg.figure5_q_ratios):
        d = fig5_data[float(q_ratio)]
        fit = d["fit"]
        emp = fit["empirical_pmf"]
        fit_pmf = fit["fit_pmf"]
        support_size = np.arange(1, N + 1)
        smooth = d["smooth"]

        nz = np.where(emp > 0)[0]
        left = max(0, int(nz.min()) - 3) if nz.size > 0 else 0
        right = min(N - 1, int(nz.max()) + 3) if nz.size > 0 else N - 1
        sl = slice(left, right + 1)

        ax.plot(support_size[sl], emp[sl], "o", color="black", markersize=3.0, label="Empirical")
        ax.plot(support_size[sl], fit_pmf[sl], color="#1f77b4", linewidth=1.7, label="Binomial fit")
        ax.plot(support_size[sl], smooth[sl], color="#d62728", linewidth=1.4, alpha=0.9, label="Smoothed")
        for pk in d["peak_positions"]:
            ax.axvline(pk, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.65)

        ax.set_title(f"min strategy, q/p={q_ratio:g}\npeaks={d['peak_count']}", fontsize=10)
        ax.set_xlabel("m")
        ax.set_ylabel("P(m)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig5.suptitle("Figure 5 reproduction: hyperedge size distribution under min strategy", fontsize=12)
    fig5.tight_layout()
    fig5.savefig(fig_dir / "figure5_min_bimodal.png", dpi=180, bbox_inches="tight")
    plt.close(fig5)

    peak_json["settings"] = {
        "N": N,
        "E": E,
        "K": K,
        "p": p_fig5,
        "q_over_p": [float(v) for v in cfg.figure5_q_ratios],
        "num_graphs": cfg.num_graphs,
    }
    with (res_dir / "figure5_peak_check.json").open("w", encoding="utf-8") as f:
        json.dump(peak_json, f, indent=2, ensure_ascii=False)

    if progress is not None:
        progress.close()

    # ------------------------------------------------------------------
    # Summary text output
    # ------------------------------------------------------------------
    mean_tv_fig3 = float(
        0.5
        * (
            fig3_df["degree_tv_distance"].mean()
            + fig3_df["size_tv_distance"].mean()
        )
    )
    approx_ok = mean_tv_fig3 < 0.10

    r2_e = float(_r2_score(yE, predE))
    r2_n = float(_r2_score(yN, predN))
    linear_ok = (r2_e > 0.95) and (r2_n > 0.95)

    bimodal_flags = [v["bimodal"] for v in peak_json["peak_check"].values()]
    bimodal_any = any(bimodal_flags)

    print("\n[Summary]")
    print(f"1) binomial 근사 적합성: {'대체로 맞음' if approx_ok else '편차가 큼'} (mean TV={mean_tv_fig3:.4f})")
    print(f"2) E*/E, N*/N 선형성: {'거의 선형' if linear_ok else '선형성 약함'} (R^2_E={r2_e:.4f}, R^2_N={r2_n:.4f})")
    print(f"3) min 전략 bimodal 경향: {'관찰됨' if bimodal_any else '뚜렷하지 않음'}")

    print("\n[Assumptions]")
    print("- 분포는 설정별 100개 hypergraph 샘플을 풀링해 PMF를 계산.")
    print("- degree: Binomial(E, theta), size: 1+Binomial(N-1, theta)로 근사.")
    print("- effective ratio는 q/p=1에서 추정한 theta를 기준으로 정규화.")
    print("- bimodal 판정은 smoothing + local peak count(상대 높이 임계치) 규칙 사용.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Experiment 2 (Figure 3/4/5).")
    parser.add_argument("--num-graphs", type=int, default=100, help="Number of generated hypergraphs per setting.")
    parser.add_argument("--seed", type=int, default=2026, help="Master random seed.")
    parser.add_argument(
        "--node-order",
        type=str,
        default="random",
        choices=["random", "fixed", "community"],
        help="Node traversal order in hyperedge generation.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/reference_2_exp2",
        help="Output directory containing figures/ and results/.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress printing.")
    args = parser.parse_args()

    if args.num_graphs < 1:
        raise ValueError(f"--num-graphs must be >= 1, got {args.num_graphs}.")

    cfg = Exp2Config(
        num_graphs=int(args.num_graphs),
        seed=int(args.seed),
        node_order=args.node_order,
    )

    outdir = Path(args.outdir)
    print("Running Experiment 2 (Figure 3/4/5)...")
    run_experiment2(cfg=cfg, outdir=outdir, show_progress=(not args.no_progress))
    print("Done.")
    print(f"Figure 3 PNG : {(outdir / 'figures' / 'figure3_degree_and_size.png').resolve()}")
    print(f"Figure 4 PNG : {(outdir / 'figures' / 'figure4_effective_parameters.png').resolve()}")
    print(f"Figure 5 PNG : {(outdir / 'figures' / 'figure5_min_bimodal.png').resolve()}")
    print(f"Figure 3 CSV : {(outdir / 'results' / 'figure3_summary.csv').resolve()}")
    print(f"Figure 4 CSV : {(outdir / 'results' / 'figure4_fit_summary.csv').resolve()}")
    print(f"Figure 5 JSON: {(outdir / 'results' / 'figure5_peak_check.json').resolve()}")


if __name__ == "__main__":
    main()
