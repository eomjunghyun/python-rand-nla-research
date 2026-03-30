import itertools
from pathlib import Path
from time import perf_counter

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
    vals, vecs = eigsh(A_csr, k=k, which="LM")
    vals, vecs = _sort_cols_by_abs_vals(vals, vecs)
    return vals, vecs


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
