import itertools
import math
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

from src.common import make_balanced_labels


Edge = Tuple[int, ...]


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
