from dataclasses import asdict, dataclass
import itertools
import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


METHODS = ["Random Projection", "Random Sampling", "Non-random"]
METHOD_COLORS = {
    "Random Projection": "#1f77b4",
    "Random Sampling": "#ff7f0e",
    "Non-random": "#2ca02c",
}

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

plt.style.use("seaborn-v0_8-whitegrid")

TIMING_METHOD_ORDER = ["Non-random", "Random Sampling", "Random Projection"]
TIMING_METRIC_LABELS = {
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
TIMING_METHOD_STEP_METRICS = {
    "Non-random": ["nr_eig_sec", "nr_kmeans_sec"],
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
TIMING_METHOD_COMPONENTS = {
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
TIMING_METHOD_PALETTES = {
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
    "exp1": {
        "label": EXPERIMENT_META["exp1"]["label"],
        "summary_filename": "exp1_timing_breakdown_summary.csv",
        "display_cols": ["n"],
        "x_col": "n",
        "x_label": "n",
    },
    "exp2": {
        "label": EXPERIMENT_META["exp2"]["label"],
        "summary_filename": "exp2_timing_breakdown_summary.csv",
        "display_cols": ["alpha_n"],
        "x_col": "alpha_n",
        "x_label": "alpha_n",
    },
    "exp3": {
        "label": EXPERIMENT_META["exp3"]["label"],
        "summary_filename": "exp3_timing_breakdown_summary.csv",
        "display_cols": ["K"],
        "x_col": "K",
        "x_label": "K",
    },
    "exp4": {
        "label": EXPERIMENT_META["exp4"]["label"],
        "summary_filename": "exp4_timing_breakdown_summary.csv",
        "display_cols": ["n", "alpha_n"],
        "x_col": "n",
        "x_label": "n",
    },
}
ALL_TIMING_METRICS = ["time_sec"]
for _method_name in TIMING_METHOD_ORDER:
    ALL_TIMING_METRICS.extend(TIMING_METHOD_STEP_METRICS[_method_name])


class LiveProgress:
    def __init__(self, total_steps: int):
        self.total_steps = max(1, int(total_steps))
        self.done_steps = 0
        self.start_time = perf_counter()
        self.spinner = itertools.cycle(["|", "/", "-", "\\"])
        self.bar_width = 34

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def update(self, x_name: str, x_value, rep: int, reps: int, method: str):
        self.done_steps += 1
        elapsed = perf_counter() - self.start_time
        ratio = self.done_steps / self.total_steps
        rate = self.done_steps / elapsed if elapsed > 0 else 0.0
        remain = self.total_steps - self.done_steps
        eta = remain / rate if rate > 0 else 0.0

        filled = int(self.bar_width * ratio)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        spin = next(self.spinner)

        if isinstance(x_value, float):
            x_text = f"{x_name}={x_value:.3f}"
        else:
            x_text = f"{x_name}={x_value}"

        msg = (
            f"\r{spin} [{bar}] {self.done_steps:4d}/{self.total_steps:4d} "
            f"({ratio*100:5.1f}%) | {x_text} rep={rep:02d}/{reps:02d} "
            f"method={method:<17} | elapsed={self._fmt(elapsed)} eta={self._fmt(eta)}"
        )
        print(msg, end="", flush=True)

    def close(self):
        print()


def make_balanced_labels(n: int, K: int, rng: np.random.Generator) -> np.ndarray:
    sizes = np.full(K, n // K, dtype=int)
    sizes[: (n % K)] += 1
    y = np.repeat(np.arange(K), sizes)
    rng.shuffle(y)
    return y


Edge = Tuple[int, ...]

SUPPORTED_JOIN_STRATEGIES = ("weighted", "max", "min", "majority")


def build_probability_matrix(K: int, p: float, q: float) -> np.ndarray:
    """Build planted-partition probability matrix.

    Diagonal entries are p (within community), and off-diagonal entries are q.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")
    if not (0.0 <= q <= p <= 1.0):
        raise ValueError(f"Expected 0 <= q <= p <= 1. Got p={p}, q={q}.")

    P = np.full((K, K), q, dtype=float)
    np.fill_diagonal(P, p)
    return P


def assign_equal_communities(N: int, K: int) -> np.ndarray:
    """Assign nodes to K communities using an equal-size partition."""
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}.")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")
    if K > N:
        raise ValueError(f"K must be <= N for non-empty communities. Got K={K}, N={N}.")

    sizes = np.full(K, N // K, dtype=np.int64)
    sizes[: (N % K)] += 1
    return np.repeat(np.arange(K, dtype=np.int64), sizes)


def _node_sequence(
    N: int,
    communities: np.ndarray,
    node_order: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if node_order == "random":
        return rng.permutation(N)
    if node_order == "fixed":
        return np.arange(N, dtype=np.int64)
    if node_order == "community":
        return np.argsort(communities, kind="stable")
    raise ValueError(f"Unsupported node_order: {node_order}")


def _majority_community(edge_comm_counts: np.ndarray) -> int:
    """Return majority community index (ties broken by smallest index)."""
    return int(np.argmax(edge_comm_counts))


def _join_probability(
    strategy: str,
    edge_comm_counts: np.ndarray,
    c_v: int,
    P: np.ndarray,
    current_size: int,
) -> float:
    if strategy == "weighted":
        prob = float(np.dot(edge_comm_counts, P[:, c_v])) / float(current_size)
        return float(np.clip(prob, 0.0, 1.0))

    active = np.where(edge_comm_counts > 0)[0]
    if active.size == 0:
        return 0.0

    if strategy == "max":
        return float(np.clip(np.max(P[active, c_v]), 0.0, 1.0))
    if strategy == "min":
        return float(np.clip(np.min(P[active, c_v]), 0.0, 1.0))
    if strategy == "majority":
        c_major = _majority_community(edge_comm_counts)
        return float(np.clip(P[c_major, c_v], 0.0, 1.0))
    raise ValueError(f"Unsupported strategy: {strategy}")


def generate_hypergraph(
    N: int,
    E: int,
    K: int,
    p: float,
    q: float,
    strategy: str,
    node_order: str = "random",
    seed: Optional[int] = None,
    communities: Optional[np.ndarray] = None,
    return_communities: bool = False,
):
    """Generate stochastic block hypergraph via node-to-hyperedge join process.

    For each hyperedge:
    1) initialize with one random node,
    2) iterate all nodes exactly once (chosen order),
    3) skip nodes already included,
    4) compute Prob(v -> e) by selected strategy,
    5) include v by Bernoulli draw.

    Complexity:
    - The outer/inner loops scan E hyperedges and N nodes per hyperedge.
    - With fixed K, probability computation per candidate is O(1).
    - Total runtime is O(N*E); when E=N this appears as O(N^2).
    """
    if N <= 0 or E <= 0:
        raise ValueError(f"N and E must be positive, got N={N}, E={E}.")
    if strategy not in SUPPORTED_JOIN_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {SUPPORTED_JOIN_STRATEGIES}, got {strategy}."
        )

    rng = np.random.default_rng(seed)
    if communities is None:
        comm = assign_equal_communities(N, K)
    else:
        comm = np.asarray(communities, dtype=np.int64).copy()
        if comm.shape[0] != N:
            raise ValueError(
                f"communities length mismatch: expected N={N}, got {comm.shape[0]}."
            )
        if np.min(comm) < 0 or np.max(comm) >= K:
            raise ValueError("communities must contain integers in [0, K-1].")

    P = build_probability_matrix(K, p, q)
    hyperedges: List[np.ndarray] = []

    for _ in range(E):
        seed_node = int(rng.integers(0, N))
        in_edge = np.zeros(N, dtype=bool)
        in_edge[seed_node] = True

        members = [seed_node]
        edge_comm_counts = np.zeros(K, dtype=np.int64)
        edge_comm_counts[comm[seed_node]] = 1

        for v in _node_sequence(N, comm, node_order, rng):
            if in_edge[v]:
                continue
            c_v = int(comm[v])
            prob_join = _join_probability(
                strategy=strategy,
                edge_comm_counts=edge_comm_counts,
                c_v=c_v,
                P=P,
                current_size=len(members),
            )
            if rng.random() < prob_join:
                in_edge[v] = True
                members.append(int(v))
                edge_comm_counts[c_v] += 1

        hyperedges.append(np.asarray(members, dtype=np.int32))

    if return_communities:
        return hyperedges, comm
    return hyperedges


def hyperedge_sizes(hyperedges: Sequence[Edge]) -> np.ndarray:
    """Return hyperedge sizes as integer array."""
    if len(hyperedges) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray([len(e) for e in hyperedges], dtype=np.int64)


def node_degrees_from_hyperedges(n: int, hyperedges: Sequence[Edge]) -> np.ndarray:
    """Compute node degree (#incident hyperedges) for each node."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    deg = np.zeros(n, dtype=np.int64)
    for edge in hyperedges:
        for u in np.unique(np.asarray(edge, dtype=np.int64)):
            if u < 0 or u >= n:
                raise ValueError(f"Node index out of range: {u} for n={n}.")
            deg[u] += 1
    return deg


def hyperedge_community_count_matrix(
    hyperedges: Sequence[Edge],
    communities: np.ndarray,
    K: int,
) -> np.ndarray:
    """Build matrix M where M[e, k] = #nodes in edge e from community k."""
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")
    communities = np.asarray(communities, dtype=np.int64)
    out = np.zeros((len(hyperedges), K), dtype=np.int64)
    for i, edge in enumerate(hyperedges):
        idx = np.asarray(edge, dtype=np.int64)
        if idx.size == 0:
            continue
        if np.min(idx) < 0 or np.max(idx) >= communities.shape[0]:
            raise ValueError("edge node index out of bounds for provided communities.")
        out[i, :] = np.bincount(communities[idx], minlength=K)
    return out


def normalized_gini(counts: np.ndarray) -> float:
    """Compute normalized Gini coefficient in [0, 1] for nonnegative counts."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("counts must be 1D.")
    if np.any(counts < 0):
        raise ValueError("counts must be nonnegative.")
    K = counts.shape[0]
    if K <= 1:
        return 0.0
    s = float(np.sum(counts))
    if s <= 0.0:
        return 0.0
    diffs = np.abs(counts[:, None] - counts[None, :]).sum()
    g = float(diffs / (2.0 * K * s))
    g_norm = g * (K / float(K - 1))
    return float(np.clip(g_norm, 0.0, 1.0))


def hyperedge_gini_scores(
    hyperedges: Sequence[Edge],
    communities: np.ndarray,
    K: int,
) -> np.ndarray:
    """Compute normalized Gini score for every hyperedge composition."""
    comp = hyperedge_community_count_matrix(hyperedges, communities, K)
    if comp.shape[0] == 0:
        return np.zeros(0, dtype=float)
    return np.asarray([normalized_gini(row) for row in comp], dtype=float)


def composition_delta(gini_scores: np.ndarray) -> float:
    """Return heterogeneity Delta = std(G) / mean(G), with stable zero handling."""
    g = np.asarray(gini_scores, dtype=float)
    if g.size == 0:
        return 0.0
    mu = float(np.mean(g))
    if mu <= 0.0:
        return 0.0
    return float(np.std(g) / mu)


def empirical_pmf(values: np.ndarray):
    """Return support and probabilities for empirical PMF of integer-like values."""
    arr = np.asarray(values)
    if arr.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    uniq, counts = np.unique(arr.astype(np.int64), return_counts=True)
    probs = counts.astype(float) / float(arr.size)
    return uniq.astype(int), probs


def _validate_probability(p: float, name: str):
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {p}")


def _validate_hyperedge_size(m: int):
    if int(m) != m or m < 2:
        raise ValueError(f"Hyperedge size m must be an integer >= 2, got {m}")


def _count_uniform_hsbm_candidates(labels: np.ndarray, m: int):
    n = int(labels.shape[0])
    total = math.comb(n, m)
    within = 0
    for k in np.unique(labels):
        nk = int(np.sum(labels == k))
        if nk >= m:
            within += math.comb(nk, m)
    mixed = total - within
    return total, within, mixed


def _sample_unique_edges_from_pool(
    pool: np.ndarray,
    m: int,
    draws: int,
    rng: np.random.Generator,
    candidate_count: Optional[int] = None,
    accept_fn: Optional[Callable[[Edge], bool]] = None,
    max_attempt_factor: int = 80,
    exhaustive_limit: int = 250000,
):
    if draws <= 0:
        return set()
    if pool.size < m:
        return set()

    edges = set()
    attempts = 0
    max_attempts = max(1500, int(draws) * max_attempt_factor)

    while len(edges) < draws and attempts < max_attempts:
        chosen = rng.choice(pool, size=m, replace=False)
        edge = tuple(sorted(int(x) for x in chosen))
        attempts += 1
        if accept_fn is not None and not accept_fn(edge):
            continue
        edges.add(edge)

    if len(edges) >= draws:
        return edges

    if candidate_count is not None and candidate_count <= exhaustive_limit:
        all_candidates = []
        for edge in itertools.combinations(pool.tolist(), m):
            e = tuple(int(v) for v in edge)
            if accept_fn is not None and not accept_fn(e):
                continue
            all_candidates.append(e)

        if len(all_candidates) < draws:
            raise RuntimeError(
                "Not enough valid candidate hyperedges to satisfy requested draws."
            )

        order = rng.permutation(len(all_candidates))
        for idx in order:
            edges.add(all_candidates[idx])
            if len(edges) >= draws:
                break
        return edges

    raise RuntimeError(
        "Sparse sampler could not collect enough unique hyperedges. "
        "Try smaller probabilities, smaller n, or use sampling='exact'."
    )


def sample_uniform_hsbm_hyperedges_exact(
    labels: np.ndarray,
    m: int,
    p_in: float,
    p_out: float,
    rng: np.random.Generator,
):
    edges = []
    for edge in itertools.combinations(range(labels.shape[0]), m):
        labs = labels[list(edge)]
        is_within = bool(np.all(labs == labs[0]))
        p = p_in if is_within else p_out
        if rng.random() < p:
            edges.append(edge)
    return edges


def sample_uniform_hsbm_hyperedges_sparse(
    labels: np.ndarray,
    m: int,
    p_in: float,
    p_out: float,
    rng: np.random.Generator,
):
    n = int(labels.shape[0])
    _, _, mixed_total = _count_uniform_hsbm_candidates(labels, m)

    within_edges = set()
    for k in np.unique(labels):
        nodes_k = np.where(labels == k)[0]
        nk = int(nodes_k.size)
        if nk < m:
            continue
        cand_k = math.comb(nk, m)
        draws_k = int(rng.binomial(cand_k, p_in))
        sampled_k = _sample_unique_edges_from_pool(
            pool=nodes_k,
            m=m,
            draws=draws_k,
            rng=rng,
            candidate_count=cand_k,
            accept_fn=None,
        )
        within_edges.update(sampled_k)

    all_nodes = np.arange(n, dtype=int)

    def is_mixed(edge: Edge):
        labs = labels[list(edge)]
        return not bool(np.all(labs == labs[0]))

    draws_mixed = int(rng.binomial(mixed_total, p_out))
    mixed_edges = _sample_unique_edges_from_pool(
        pool=all_nodes,
        m=m,
        draws=draws_mixed,
        rng=rng,
        candidate_count=mixed_total,
        accept_fn=is_mixed,
    )

    all_edges = sorted(within_edges.union(mixed_edges))
    return all_edges


def generate_uniform_hsbm_instance(
    n: int,
    K: int,
    m: int,
    p_in: float,
    p_out: float,
    rng: np.random.Generator,
    labels: Optional[np.ndarray] = None,
    sampling: str = "auto",  # "auto" | "exact" | "sparse"
    max_enumeration: int = 1500000,
):
    _validate_hyperedge_size(m)
    _validate_probability(p_in, "p_in")
    _validate_probability(p_out, "p_out")
    if K < 1:
        raise ValueError(f"K must be >=1, got {K}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if labels is None:
        y_true = make_balanced_labels(n, K, rng)
    else:
        y_true = np.asarray(labels, dtype=int).copy()
        if y_true.shape[0] != n:
            raise ValueError(f"labels length mismatch: expected {n}, got {y_true.shape[0]}")

    if np.min(y_true) < 0 or np.max(y_true) >= K:
        raise ValueError("labels must be integer values in [0, K-1]")

    total, within_total, mixed_total = _count_uniform_hsbm_candidates(y_true, m)
    if sampling == "auto":
        sampling_mode = "exact" if total <= max_enumeration else "sparse"
    else:
        sampling_mode = sampling

    if sampling_mode == "exact":
        hyperedges = sample_uniform_hsbm_hyperedges_exact(y_true, m, p_in, p_out, rng)
    elif sampling_mode == "sparse":
        hyperedges = sample_uniform_hsbm_hyperedges_sparse(y_true, m, p_in, p_out, rng)
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    Theta_true = np.eye(K)[y_true]
    stats = {
        "n": int(n),
        "K": int(K),
        "m": int(m),
        "p_in": float(p_in),
        "p_out": float(p_out),
        "num_hyperedges": int(len(hyperedges)),
        "num_candidates_total": int(total),
        "num_candidates_within": int(within_total),
        "num_candidates_mixed": int(mixed_total),
        "sampling_mode": sampling_mode,
    }
    return hyperedges, y_true, Theta_true, stats


def _resolve_prob_for_m(
    prob: Union[float, Dict[int, float]],
    m: int,
    prob_name: str,
):
    if isinstance(prob, dict):
        if m not in prob:
            raise ValueError(f"Missing {prob_name} for m={m}")
        p = float(prob[m])
    else:
        p = float(prob)
    _validate_probability(p, f"{prob_name}[m={m}]")
    return p


def generate_nonuniform_hsbm_instance(
    n: int,
    K: int,
    m_values: Sequence[int],
    p_in: Union[float, Dict[int, float]],
    p_out: Union[float, Dict[int, float]],
    rng: np.random.Generator,
    labels: Optional[np.ndarray] = None,
    sampling: str = "auto",  # "auto" | "exact" | "sparse"
    max_enumeration: int = 1500000,
):
    m_values = sorted(set(int(m) for m in m_values))
    if not m_values:
        raise ValueError("m_values must not be empty")
    for m in m_values:
        _validate_hyperedge_size(m)

    if labels is None:
        y_true = make_balanced_labels(n, K, rng)
    else:
        y_true = np.asarray(labels, dtype=int).copy()

    per_size_stats = {}
    all_edges = []
    for m in m_values:
        pin_m = _resolve_prob_for_m(p_in, m, "p_in")
        pout_m = _resolve_prob_for_m(p_out, m, "p_out")
        edges_m, _, _, stats_m = generate_uniform_hsbm_instance(
            n=n,
            K=K,
            m=m,
            p_in=pin_m,
            p_out=pout_m,
            rng=rng,
            labels=y_true,
            sampling=sampling,
            max_enumeration=max_enumeration,
        )
        per_size_stats[str(m)] = stats_m
        all_edges.extend(edges_m)

    Theta_true = np.eye(K)[y_true]
    stats = {
        "n": int(n),
        "K": int(K),
        "m_values": [int(m) for m in m_values],
        "num_hyperedges_total": int(len(all_edges)),
        "per_size": per_size_stats,
    }
    return all_edges, y_true, Theta_true, stats


def hypergraph_basic_stats(
    n: int,
    hyperedges: Sequence[Edge],
    labels: Optional[np.ndarray] = None,
):
    sizes = np.array([len(e) for e in hyperedges], dtype=int) if hyperedges else np.array([], dtype=int)
    stats = {
        "n_nodes": int(n),
        "n_hyperedges": int(len(hyperedges)),
        "avg_hyperedge_size": float(sizes.mean()) if sizes.size > 0 else 0.0,
        "min_hyperedge_size": int(sizes.min()) if sizes.size > 0 else 0,
        "max_hyperedge_size": int(sizes.max()) if sizes.size > 0 else 0,
    }
    if sizes.size > 0:
        uniq, cnt = np.unique(sizes, return_counts=True)
        stats["hyperedge_size_histogram"] = {str(int(u)): int(c) for u, c in zip(uniq, cnt)}

    if labels is not None and len(hyperedges) > 0:
        labels = np.asarray(labels, dtype=int)
        within = 0
        for e in hyperedges:
            labs = labels[list(e)]
            if np.all(labs == labs[0]):
                within += 1
        stats["within_community_ratio"] = float(within / len(hyperedges))
    return stats


def hyperedges_to_incidence_csr(
    n: int,
    hyperedges: Sequence[Edge],
    dtype=np.float32,
):
    rows = []
    cols = []
    data = []
    for j, edge in enumerate(hyperedges):
        if len(edge) != len(set(edge)):
            raise ValueError(f"Hyperedge contains duplicate nodes: {edge}")
        for u in edge:
            if u < 0 or u >= n:
                raise ValueError(f"Node index out of range in edge {edge} for n={n}")
            rows.append(int(u))
            cols.append(int(j))
            data.append(1.0)

    H = sp.coo_matrix((data, (rows, cols)), shape=(n, len(hyperedges)), dtype=dtype).tocsr()
    H.sum_duplicates()
    return H


def _normalize_edge_weights(num_edges: int, edge_weights: Optional[Iterable[float]]):
    if edge_weights is None:
        return np.ones(num_edges, dtype=float)
    w = np.asarray(list(edge_weights), dtype=float)
    if w.shape[0] != num_edges:
        raise ValueError(
            f"edge_weights length mismatch: expected {num_edges}, got {w.shape[0]}"
        )
    if np.any(w < 0):
        raise ValueError("edge_weights must be nonnegative")
    return w


def clique_expansion_adjacency(
    n: int,
    hyperedges: Sequence[Edge],
    edge_weights: Optional[Iterable[float]] = None,
    normalize_by_size: bool = True,
    dtype=np.float32,
):
    w = _normalize_edge_weights(len(hyperedges), edge_weights)
    rows = []
    cols = []
    data = []

    for idx, edge in enumerate(hyperedges):
        m = len(edge)
        if m < 2:
            continue
        scale = float(w[idx]) / float(m - 1) if normalize_by_size else float(w[idx])
        for u, v in itertools.combinations(edge, 2):
            rows.extend((u, v))
            cols.extend((v, u))
            data.extend((scale, scale))

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=dtype).tocsr()
    A.sum_duplicates()
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def zhou_normalized_laplacian(
    n: int,
    hyperedges: Sequence[Edge],
    edge_weights: Optional[Iterable[float]] = None,
):
    if len(hyperedges) == 0:
        return sp.eye(n, format="csr", dtype=float)

    H = hyperedges_to_incidence_csr(n, hyperedges, dtype=float)
    w = _normalize_edge_weights(len(hyperedges), edge_weights)
    W = sp.diags(w, format="csr")

    d_e = np.asarray(H.sum(axis=0)).ravel()
    d_v = np.asarray(H @ w).ravel()
    d_e_inv = np.divide(1.0, d_e, out=np.zeros_like(d_e), where=(d_e > 0))
    d_v_inv_sqrt = np.divide(1.0, np.sqrt(d_v), out=np.zeros_like(d_v), where=(d_v > 0))

    D_e_inv = sp.diags(d_e_inv, format="csr")
    D_v_inv_sqrt = sp.diags(d_v_inv_sqrt, format="csr")

    theta = D_v_inv_sqrt @ H @ W @ D_e_inv @ H.T @ D_v_inv_sqrt
    lap = sp.eye(n, format="csr", dtype=float) - theta
    lap.sum_duplicates()
    lap.eliminate_zeros()
    return lap

def build_B(alpha_n: float, lam: float, K: int) -> np.ndarray:
    within = alpha_n
    between = alpha_n * (1.0 - lam)
    B = np.full((K, K), between, dtype=float)
    np.fill_diagonal(B, within)
    return B


def sample_adjacency_from_P(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = P.shape[0]
    tri = np.triu_indices(n, k=1)
    probs = P[tri]
    edges = (rng.random(probs.shape[0]) < probs).astype(float)
    A = np.zeros((n, n), dtype=float)
    A[tri] = edges
    A += A.T
    np.fill_diagonal(A, 0.0)
    return A


def generate_sbm_instance(
    n: int,
    K: int,
    alpha_n: float,
    lam: float,
    rng: np.random.Generator,
):
    y_true = make_balanced_labels(n, K, rng)
    Theta_true = np.eye(K)[y_true]
    B_true = build_B(alpha_n, lam, K)
    P = Theta_true @ B_true @ Theta_true.T
    np.fill_diagonal(P, 0.0)
    A = sample_adjacency_from_P(P, rng)
    return A, P, B_true, y_true, Theta_true


def top_eigvecs_symmetric(M: np.ndarray, k: int) -> np.ndarray:
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    idx = np.argsort(vals)[-k:]
    idx = idx[np.argsort(vals[idx])[::-1]]
    return vecs[:, idx]


def top_eigpairs_symmetric(M: np.ndarray, k: int):
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    idx = np.argsort(vals)[-k:]
    vals_top = vals[idx]
    vecs_top = vecs[:, idx]
    order = np.argsort(vals_top)[::-1]
    return vals_top[order], vecs_top[:, order]


def normalize_rows_l2(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    return np.divide(U, norms, out=np.zeros_like(U), where=(norms > eps))


def kmeans_on_rows(
    U: np.ndarray,
    K: int,
    rng: np.random.Generator,
    normalize_rows: bool = False,
) -> np.ndarray:
    if normalize_rows:
        U = normalize_rows_l2(U)
    rs = int(rng.integers(1, 2**31 - 1))
    km = KMeans(n_clusters=K, n_init=20, random_state=rs)
    return km.fit_predict(U)


def _finalize_algorithm_timing(timings: dict, total_start: float) -> dict:
    total_sec = perf_counter() - total_start
    step_sum_sec = sum(float(v) for k, v in timings.items() if k.endswith("_sec"))
    timings["algo_total_sec"] = total_sec
    timings["algo_step_sum_sec"] = step_sum_sec
    timings["algo_other_sec"] = max(0.0, total_sec - step_sum_sec)
    return timings


def attach_timing_breakdown(
    record: dict,
    algo_timing=None,
    instance_sec=None,
    metric_sec=None,
) -> dict:
    merged = dict(record)
    if instance_sec is not None:
        merged["instance_gen_sec"] = float(instance_sec)
    if metric_sec is not None:
        merged["metric_eval_sec"] = float(metric_sec)
    if algo_timing:
        merged.update(algo_timing)
        pipeline_total_sec = float(algo_timing.get("algo_total_sec", 0.0))
        if instance_sec is not None:
            pipeline_total_sec += float(instance_sec)
        if metric_sec is not None:
            pipeline_total_sec += float(metric_sec)
        merged["pipeline_total_sec"] = pipeline_total_sec
    return merged


def run_non_random(
    A: np.ndarray,
    K: int,
    K_prime: int,
    rng: np.random.Generator,
    normalize_rows: bool = False,
    return_timing: bool = False,
):
    total_start = perf_counter()
    timings = {}

    t0 = perf_counter()
    U = top_eigvecs_symmetric(A, K_prime)
    timings["nr_eig_sec"] = perf_counter() - t0

    t0 = perf_counter()
    labels = kmeans_on_rows(U, K, rng, normalize_rows=normalize_rows)
    timings["nr_kmeans_sec"] = perf_counter() - t0

    t0 = perf_counter()
    A_hat = A.copy()
    timings["nr_copy_sec"] = perf_counter() - t0

    if return_timing:
        return A_hat, labels, _finalize_algorithm_timing(timings, total_start)
    return A_hat, labels


def run_random_projection(
    A: np.ndarray,
    K: int,
    K_prime: int,
    r: int,
    q: int,
    rng: np.random.Generator,
    normalize_rows: bool = False,
    return_timing: bool = False,
):
    total_start = perf_counter()
    timings = {}
    n = A.shape[0]

    t0 = perf_counter()
    Omega = rng.standard_normal(size=(n, K_prime + r))
    timings["rp_draw_omega_sec"] = perf_counter() - t0

    t0 = perf_counter()
    Y = Omega.copy()
    for _ in range(2 * q + 1):
        Y = A @ Y
    timings["rp_power_iter_sec"] = perf_counter() - t0

    t0 = perf_counter()
    Q, _ = np.linalg.qr(Y, mode="reduced")
    timings["rp_qr_sec"] = perf_counter() - t0

    t0 = perf_counter()
    C = Q.T @ A @ Q
    timings["rp_build_core_sec"] = perf_counter() - t0

    t0 = perf_counter()
    A_hat = Q @ C @ Q.T
    timings["rp_reconstruct_sec"] = perf_counter() - t0

    t0 = perf_counter()
    Uc = top_eigvecs_symmetric(C, K_prime)
    timings["rp_small_eig_sec"] = perf_counter() - t0

    t0 = perf_counter()
    U_rp = Q @ Uc
    timings["rp_lift_sec"] = perf_counter() - t0

    t0 = perf_counter()
    labels = kmeans_on_rows(U_rp, K, rng, normalize_rows=normalize_rows)
    timings["rp_kmeans_sec"] = perf_counter() - t0

    if return_timing:
        return A_hat, labels, _finalize_algorithm_timing(timings, total_start)
    return A_hat, labels


def run_random_sampling(
    A: np.ndarray,
    K: int,
    K_prime: int,
    p: float,
    rng: np.random.Generator,
    normalize_rows: bool = False,
    return_timing: bool = False,
):
    total_start = perf_counter()
    timings = {}
    n = A.shape[0]

    t0 = perf_counter()
    tri = np.triu_indices(n, k=1)
    mask = (rng.random(tri[0].shape[0]) < p).astype(float)
    timings["rs_sample_mask_sec"] = perf_counter() - t0

    t0 = perf_counter()
    A_s = np.zeros_like(A)
    A_s[tri] = A[tri] * mask / p
    A_s += A_s.T
    np.fill_diagonal(A_s, 0.0)
    timings["rs_build_sampled_matrix_sec"] = perf_counter() - t0

    t0 = perf_counter()
    try:
        vals, vecs = eigsh(A_s, k=K_prime, which="LA")
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
    except Exception:
        vals, vecs = top_eigpairs_symmetric(A_s, K_prime)
    timings["rs_eig_sec"] = perf_counter() - t0

    t0 = perf_counter()
    A_hat = vecs @ np.diag(vals) @ vecs.T
    timings["rs_reconstruct_sec"] = perf_counter() - t0

    t0 = perf_counter()
    A_hat = 0.5 * (A_hat + A_hat.T)
    timings["rs_symmetrize_sec"] = perf_counter() - t0

    t0 = perf_counter()
    labels = kmeans_on_rows(vecs, K, rng, normalize_rows=normalize_rows)
    timings["rs_kmeans_sec"] = perf_counter() - t0

    if return_timing:
        return A_hat, labels, _finalize_algorithm_timing(timings, total_start)
    return A_hat, labels


def spectral_norm_sym(M: np.ndarray) -> float:
    M = 0.5 * (M + M.T)
    eigvals = np.linalg.eigvalsh(M)
    return float(np.max(np.abs(eigvals)))


def align_labels_weighted_hungarian(y_true: np.ndarray, y_pred: np.ndarray, K: int):
    conf = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1

    n_k = conf.sum(axis=1).astype(float)
    weight = np.where(n_k > 0, 1.0 / n_k, 0.0)[:, None]
    score = conf * weight
    cost = -score
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping_pred_to_true = {pred: true for true, pred in zip(row_ind, col_ind)}
    for c in range(K):
        mapping_pred_to_true.setdefault(c, c)

    y_aligned = np.array([mapping_pred_to_true[c] for c in y_pred], dtype=int)
    return y_aligned


def theta_error_exact(
    Theta_true: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, K: int
):
    best_val = np.inf
    best_theta_hat = None
    best_perm = None
    I = np.eye(K)

    for perm in itertools.permutations(range(K)):
        mapped = np.array([perm[c] for c in y_pred], dtype=int)
        Theta_hat = I[mapped]
        val = 0.0

        for k in range(K):
            idx = np.where(y_true == k)[0]
            n_k = len(idx)
            if n_k == 0:
                continue
            diff = Theta_hat[idx, :] - Theta_true[idx, :]
            l0 = np.count_nonzero(diff)
            val += l0 / (2.0 * n_k)

        if val < best_val:
            best_val = val
            best_theta_hat = Theta_hat
            best_perm = perm

    return float(best_val), best_theta_hat, best_perm


def theta_error_weighted_hungarian(
    Theta_true: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, K: int
):
    y_aligned = align_labels_weighted_hungarian(y_true, y_pred, K)
    Theta_hat = np.eye(K)[y_aligned]
    total = 0.0

    for k in range(K):
        idx = np.where(y_true == k)[0]
        n_k = len(idx)
        if n_k == 0:
            continue
        diff = Theta_hat[idx, :] - Theta_true[idx, :]
        l0 = np.count_nonzero(diff)
        total += l0 / (2.0 * n_k)

    return float(total), Theta_hat


def estimate_B_hat(A_hat: np.ndarray, Theta_hat: np.ndarray) -> np.ndarray:
    num = Theta_hat.T @ A_hat @ Theta_hat
    counts = Theta_hat.sum(axis=0)
    den = np.outer(counts, counts)
    B_hat = np.divide(num, den, out=np.zeros_like(num), where=(den > 0))
    return B_hat


def evaluate_metrics(
    A_hat: np.ndarray,
    y_pred: np.ndarray,
    P: np.ndarray,
    B_true: np.ndarray,
    Theta_true: np.ndarray,
    y_true: np.ndarray,
    K: int,
    theta_mode: str = "exact",  # "exact" | "hungarian"
):
    err_P = spectral_norm_sym(A_hat - P)

    if theta_mode == "exact":
        err_Theta, Theta_hat_best, _ = theta_error_exact(Theta_true, y_true, y_pred, K)
    elif theta_mode == "hungarian":
        err_Theta, Theta_hat_best = theta_error_weighted_hungarian(
            Theta_true, y_true, y_pred, K
        )
    else:
        raise ValueError(f"Unknown theta_mode: {theta_mode}")

    B_hat = estimate_B_hat(A_hat, Theta_hat_best)
    err_B = float(np.max(np.abs(B_hat - B_true)))
    return err_P, err_Theta, err_B


def summarize_metrics(df_raw: pd.DataFrame, group_cols):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    keys = list(group_cols) + ["method"]
    return df_raw.groupby(keys, as_index=False).agg(
        error_P_mean=("error_P", "mean"),
        error_P_std=("error_P", "std"),
        error_Theta_mean=("error_Theta", "mean"),
        error_Theta_std=("error_Theta", "std"),
        error_B_mean=("error_B", "mean"),
        error_B_std=("error_B", "std"),
        time_mean=("time_sec", "mean"),
        time_std=("time_sec", "std"),
    )


def extract_timing_breakdown(df_raw: pd.DataFrame, id_cols):
    keep_cols = []
    for col in list(id_cols) + [c for c in df_raw.columns if c.endswith("_sec")]:
        if col in df_raw.columns and col not in keep_cols:
            keep_cols.append(col)
    return df_raw.loc[:, keep_cols].copy()


def summarize_timing_breakdown(df_raw: pd.DataFrame, group_cols):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    keys = list(group_cols) + ["method"]
    timing_cols = [c for c in df_raw.columns if c.endswith("_sec")]
    agg = {}
    for col in timing_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")
    return df_raw.groupby(keys, as_index=False).agg(**agg)


def plot_metric_panels(summary: pd.DataFrame, x_col: str, out_png: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ycols = [
        ("error_P_mean", "Error for P"),
        ("error_Theta_mean", "Error for Theta"),
        ("error_B_mean", "Error for B"),
    ]

    for ax, (ycol, ylabel) in zip(axes, ycols):
        for m in METHODS:
            d = summary[summary["method"] == m].sort_values(x_col)
            ax.plot(
                d[x_col].values,
                d[ycol].values,
                color=METHOD_COLORS[m],
                linewidth=2.0,
                marker="o",
                label=m,
            )
        ax.set_xlabel(x_col)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(summary: pd.DataFrame, x_col: str, out_png: Path):
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
    for m in METHODS:
        d = summary[summary["method"] == m].sort_values(x_col)
        ax.plot(
            d[x_col].values,
            d["time_mean"].values,
            color=METHOD_COLORS[m],
            linewidth=2.0,
            marker="o",
            label=m,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel("Runtime (sec)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_undirected_edgelist_csr(
    path: Path,
    delimiter: str = None,
    comment_prefix: str = "#",
):
    """Load an undirected edge list into a symmetric binary CSR matrix.

    Node ids are remapped to contiguous [0, n) indices.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Edge list not found: {path}")

    rows = []
    cols = []
    node_to_idx = {}

    def idx_of(token: str) -> int:
        if token not in node_to_idx:
            node_to_idx[token] = len(node_to_idx)
        return node_to_idx[token]

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if comment_prefix and s.startswith(comment_prefix):
                continue
            parts = s.split(delimiter) if delimiter is not None else s.split()
            if len(parts) < 2:
                continue
            u = idx_of(parts[0])
            v = idx_of(parts[1])
            if u == v:
                continue
            rows.extend((u, v))
            cols.extend((v, u))

    n = len(node_to_idx)
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, node_to_idx


def upper_triangle_edges(A_csr: sp.csr_matrix):
    """Return upper-triangle edge indices (i < j)."""
    A_upper = sp.triu(A_csr, k=1, format="coo")
    return A_upper.row.astype(np.int64), A_upper.col.astype(np.int64)


def _sort_cols_by_abs_vals(vals: np.ndarray, vecs: np.ndarray):
    order = np.argsort(np.abs(vals))[::-1]
    return vals[order], vecs[:, order]


def eigvecs_eigsh_sparse(A_csr: sp.csr_matrix, k: int):
    vals, vecs = eigsh(A_csr, k=k, which="LA")
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return vals, vecs


def eigvecs_partial_eigen_proxy_sparse(A_csr: sp.csr_matrix, k: int):
    """Python proxy for irlba::partial_eigen used in paper-style timing only.

    This uses ARPACK via scipy and returns the leading eigenpairs by magnitude.
    """
    vals, vecs = eigsh(A_csr, k=k, which="LM")
    vals, vecs = _sort_cols_by_abs_vals(vals, vecs)
    return vals[:k], vecs[:, :k]


def eigvecs_random_projection_sparse(
    A_csr: sp.csr_matrix,
    k: int,
    r: int,
    q: int,
    rng: np.random.Generator,
):
    n = A_csr.shape[0]
    l = k + r
    omega = rng.standard_normal((n, l))

    Y = omega
    for _ in range(2 * q + 1):
        Y = A_csr @ Y

    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ (A_csr @ Q)
    B = 0.5 * (B + B.T)
    vals, vecs = np.linalg.eigh(B)
    vals, vecs = _sort_cols_by_abs_vals(vals, vecs)
    return vals[:k], Q @ vecs[:, :k]


def sample_rescaled_adjacency_from_edges(
    n: int,
    upper_rows: np.ndarray,
    upper_cols: np.ndarray,
    p: float,
    rng: np.random.Generator,
):
    """Sample edges with probability p and rescale kept edges by 1/p."""
    if not (0.0 < p <= 1.0):
        raise ValueError(f"Sampling probability must be in (0,1], got {p}")

    keep = rng.random(upper_rows.shape[0]) < p
    r = upper_rows[keep]
    c = upper_cols[keep]
    if r.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)

    w = np.full(r.shape[0], 1.0 / p, dtype=np.float32)
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    data = np.concatenate([w, w])

    A_s = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    A_s.sum_duplicates()
    A_s.setdiag(0.0)
    A_s.eliminate_zeros()
    return A_s


def eigvecs_random_sampling_sparse(
    n: int,
    upper_rows: np.ndarray,
    upper_cols: np.ndarray,
    p: float,
    k: int,
    rng: np.random.Generator,
):
    t0 = perf_counter()
    A_s = sample_rescaled_adjacency_from_edges(n, upper_rows, upper_cols, p, rng)
    t1 = perf_counter()
    _, vecs = eigvecs_eigsh_sparse(A_s, k=k)
    t2 = perf_counter()
    return vecs, (t2 - t0), (t2 - t1)


def eigvecs_random_sampling_sparse_table4(
    n: int,
    upper_rows: np.ndarray,
    upper_cols: np.ndarray,
    p: float,
    k: int,
    rng: np.random.Generator,
):
    """Paper-style random sampling timing for partial eigenvector computation."""
    t0 = perf_counter()
    A_s = sample_rescaled_adjacency_from_edges(n, upper_rows, upper_cols, p, rng)
    t1 = perf_counter()
    _, vecs = eigvecs_partial_eigen_proxy_sparse(A_s, k=k)
    t2 = perf_counter()
    return vecs, (t2 - t0), (t2 - t1)


def load_large_integer_edgelist_csr(
    path: Path,
    delimiter: str = None,
    comment_prefix: str = "#",
):
    """Load a large integer edge list into a symmetric binary CSR matrix.

    This path is optimized for SNAP-style datasets used in Section 8.2.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Edge list not found: {path}")

    sep = delimiter if delimiter is not None else r"\s+"
    df = pd.read_csv(
        path,
        sep=sep,
        comment=comment_prefix,
        header=None,
        usecols=[0, 1],
        names=["src", "dst"],
        dtype=np.int64,
        engine="c",
    )
    df = df[df["src"] != df["dst"]].reset_index(drop=True)

    endpoints = pd.concat([df["src"], df["dst"]], ignore_index=True)
    codes, uniques = pd.factorize(endpoints, sort=False)
    m = len(df)
    rows = codes[:m].astype(np.int32, copy=False)
    cols = codes[m:].astype(np.int32, copy=False)
    rows_sym = np.concatenate([rows, cols])
    cols_sym = np.concatenate([cols, rows])
    data = np.ones(rows_sym.shape[0], dtype=np.float32)

    A = sp.coo_matrix(
        (data, (rows_sym, cols_sym)),
        shape=(len(uniques), len(uniques)),
        dtype=np.float32,
    ).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, uniques.to_numpy()


TABLE4_PLOT_ORDER = [
    "Random Projection",
    "Random Sampling",
    "Random Sampling (excl. sampling)",
    "partial_eigen",
]

TABLE4_PLOT_COLORS = {
    "Random Projection": "#4C78A8",
    "Random Sampling": "#F58518",
    "Random Sampling (excl. sampling)": "#E45756",
    "partial_eigen": "#54A24B",
}

TABLE4_PLOT_LABELS = {
    "Random Projection": "Random projection",
    "Random Sampling": "Random sampling",
    "Random Sampling (excl. sampling)": "Random sampling\n(excl. sampling)",
    "partial_eigen": "partial_eigen",
}


def benchmark_table4_methods_sparse(
    A_csr: sp.csr_matrix,
    dataset_name: str,
    target_rank: int,
    reps: int,
    seed: int,
    r: int,
    q: int,
    p: float,
    progress=None,
):
    """Benchmark paper-style eigenvector timing for Section 8.2 Table 4."""
    n_nodes = int(A_csr.shape[0])
    n_edges = int(A_csr.nnz // 2)
    upper_rows, upper_cols = upper_triangle_edges(A_csr)
    master_rng = np.random.default_rng(seed)
    rows = []

    for rep in range(1, reps + 1):
        rep_seed = int(master_rng.integers(1, 2**31 - 1))
        rng = np.random.default_rng(rep_seed)

        t0 = perf_counter()
        eigvecs_random_projection_sparse(A_csr, target_rank, r, q, rng)
        t_rp = perf_counter() - t0
        rows.append(
            {
                "dataset": dataset_name,
                "rep": rep,
                "method": "Random Projection",
                "time_sec": t_rp,
                "time_sec_excl_sampling": t_rp,
                "time_sampling_sec": 0.0,
                "target_rank": target_rank,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
            }
        )
        if progress is not None:
            progress.update("dataset", dataset_name, rep, reps, "Random Projection")

        _, t_rs_with, t_rs_without = eigvecs_random_sampling_sparse_table4(
            n=n_nodes,
            upper_rows=upper_rows,
            upper_cols=upper_cols,
            p=p,
            k=target_rank,
            rng=rng,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "rep": rep,
                "method": "Random Sampling",
                "time_sec": float(t_rs_with),
                "time_sec_excl_sampling": float(t_rs_without),
                "time_sampling_sec": max(0.0, float(t_rs_with - t_rs_without)),
                "target_rank": target_rank,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
            }
        )
        if progress is not None:
            progress.update("dataset", dataset_name, rep, reps, "Random Sampling")

        t0 = perf_counter()
        eigvecs_partial_eigen_proxy_sparse(A_csr, target_rank)
        t_pe = perf_counter() - t0
        rows.append(
            {
                "dataset": dataset_name,
                "rep": rep,
                "method": "partial_eigen",
                "time_sec": t_pe,
                "time_sec_excl_sampling": t_pe,
                "time_sampling_sec": 0.0,
                "target_rank": target_rank,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
            }
        )
        if progress is not None:
            progress.update("dataset", dataset_name, rep, reps, "partial_eigen")

    return pd.DataFrame(rows)


def summarize_table4_median_times(df_raw: pd.DataFrame):
    """Summarize paper-style Table 4 medians into one row per dataset."""
    records = []
    for dataset, block in df_raw.groupby("dataset", sort=False):
        rp = block[block["method"] == "Random Projection"]["time_sec"].median()
        rs = block[block["method"] == "Random Sampling"]["time_sec"].median()
        rs_excl = block[block["method"] == "Random Sampling"]["time_sec_excl_sampling"].median()
        pe = block[block["method"] == "partial_eigen"]["time_sec"].median()
        meta = block.iloc[0]
        records.append(
            {
                "dataset": dataset,
                "n_nodes": int(meta["n_nodes"]),
                "n_edges": int(meta["n_edges"]),
                "target_rank": int(meta["target_rank"]),
                "random_projection_median_sec": float(rp),
                "random_sampling_median_sec": float(rs),
                "random_sampling_excl_sampling_median_sec": float(rs_excl),
                "partial_eigen_median_sec": float(pe),
                "random_sampling_display": f"{rs:.3f}({rs_excl:.3f})",
            }
        )
    return pd.DataFrame(records)


def format_table4_markdown(df_summary: pd.DataFrame):
    """Render a markdown table in the paper's Table 4 style."""
    lines = [
        "Table 4-like median time (seconds) over 20 replications.",
        "",
        "| Networks | Random projection | Random sampling | partial_eigen |",
        "|---|---:|---:|---:|",
    ]
    for row in df_summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.dataset} | "
            f"{row.random_projection_median_sec:.3f} | "
            f"{row.random_sampling_display} | "
            f"{row.partial_eigen_median_sec:.3f} |"
        )
    lines.append("")
    lines.append(
        "Note: For random sampling, the value outside parentheses includes sampling time and "
        "the value inside parentheses excludes sampling time."
    )
    return "\n".join(lines)


def _table4_plot_long_frame(df_summary: pd.DataFrame):
    rows = []
    for row in df_summary.itertuples(index=False):
        rows.extend(
            [
                {
                    "dataset": row.dataset,
                    "method_variant": "Random Projection",
                    "time_sec": row.random_projection_median_sec,
                },
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


def plot_table4_median_bars(df_summary: pd.DataFrame, out_png: Path):
    """Plot grouped median bars for the paper-style Table 4 comparison."""
    plot_df = _table4_plot_long_frame(df_summary)
    datasets = list(dict.fromkeys(plot_df["dataset"].tolist()))
    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for idx, method in enumerate(TABLE4_PLOT_ORDER):
        d = plot_df[plot_df["method_variant"] == method]
        offsets = x + (idx - 1.5) * width
        bars = ax.bar(
            offsets,
            d["time_sec"].values,
            width=width,
            color=TABLE4_PLOT_COLORS[method],
            edgecolor="black",
            linewidth=0.6,
            label=TABLE4_PLOT_LABELS[method],
            hatch="//" if "excl." in method else None,
        )
        for bar in bars:
            h = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Median eigenvector runtime (sec)")
    ax.set_title("Table 4-like Median Runtime Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_table4_runtime_boxplots(df_raw: pd.DataFrame, out_png: Path):
    """Plot per-rep runtime distributions for each dataset and method variant."""
    datasets = list(dict.fromkeys(df_raw["dataset"].tolist()))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        block = df_raw[df_raw["dataset"] == dataset]
        series = []
        for variant in TABLE4_PLOT_ORDER:
            if variant == "Random Sampling (excl. sampling)":
                vals = block.loc[block["method"] == "Random Sampling", "time_sec_excl_sampling"].values
            else:
                vals = block.loc[block["method"] == variant, "time_sec"].values
            series.append(vals)

        box = ax.boxplot(
            series,
            tick_labels=[TABLE4_PLOT_LABELS[v] for v in TABLE4_PLOT_ORDER],
            showfliers=False,
            patch_artist=True,
        )
        for patch, variant in zip(box["boxes"], TABLE4_PLOT_ORDER):
            patch.set_facecolor(TABLE4_PLOT_COLORS[variant])
            patch.set_edgecolor("black")
            if "excl." in variant:
                patch.set_hatch("//")

        ax.set_title(dataset)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Per-rep eigenvector runtime (sec)")
    fig.suptitle("Table 4-like Runtime Distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pairwise_ari(label_map: dict):
    methods = list(label_map.keys())
    m = len(methods)
    mat = np.eye(m, dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            ari = adjusted_rand_score(label_map[methods[i]], label_map[methods[j]])
            mat[i, j] = ari
            mat[j, i] = ari
    return methods, mat


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
    resolved = resolve_summary_path(
        exp_key,
        summary_path=summary_path,
        search_root=search_root,
    )
    return resolved, pd.read_csv(resolved)


def build_timing_table(
    df: pd.DataFrame,
    base_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
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


def timing_method_slug(method_name: str) -> str:
    return method_name.lower().replace("-", "").replace(" ", "_")


def timing_metric_title(method_name: str, metric_name: str) -> str:
    return f"{method_name} - {TIMING_METRIC_LABELS.get(metric_name, metric_name)}"


def _plot_single_timing_metric(
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
    ax.set_title(f"{exp_label} | {timing_metric_title(method_name, metric_name)}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(TIMING_METRIC_LABELS.get(metric_name, metric_name))
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
            TIMING_METHOD_STEP_METRICS[method_name],
        )
        for method_name in TIMING_METHOD_ORDER
    }

    created_paths: list[Path] = []
    for method_name in TIMING_METHOD_ORDER:
        out_path = output_dir / f"{timing_method_slug(method_name)}_time_sec.png" if save else None
        created = _plot_single_timing_metric(
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

    for method_name in TIMING_METHOD_ORDER:
        for metric_name in TIMING_METHOD_STEP_METRICS[method_name]:
            out_path = output_dir / f"{timing_method_slug(method_name)}_{metric_name}.png" if save else None
            created = _plot_single_timing_metric(
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


def build_method_runtime_table(
    df: pd.DataFrame,
    method_name: str,
    x_col: str,
) -> pd.DataFrame | None:
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
    for metric_name, label in TIMING_METHOD_COMPONENTS[method_name]:
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
    for method_name in TIMING_METHOD_ORDER:
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


def _count_outside_labels(
    table: pd.DataFrame,
    component_cols: list[str],
    shared_ymax: float,
) -> int:
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
    colors = TIMING_METHOD_PALETTES[method_name][: len(component_cols)]
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
        for method_name in TIMING_METHOD_ORDER
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
    for ax, method_name in zip(axes[1:], TIMING_METHOD_ORDER):
        table = method_tables[method_name]
        if table is None:
            ax.set_visible(False)
            continue
        plot_method_runtime_stack(
            ax,
            table,
            method_name,
            cfg["x_col"],
            cfg["x_label"],
            shared_ymax,
        )

    fig.suptitle(f"{cfg['label']} | Runtime comparison and composition", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.72, 0.98])

    if output_path is None:
        output_path = (
            resolved_summary_path.parent
            / "runtime_composition_plots"
            / f"{exp_key}_runtime_composition.png"
        )
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
