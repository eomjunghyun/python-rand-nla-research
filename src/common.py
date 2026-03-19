import itertools
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


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


def kmeans_on_rows(U: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    rs = int(rng.integers(1, 2**31 - 1))
    km = KMeans(n_clusters=K, n_init=20, random_state=rs)
    return km.fit_predict(U)


def run_non_random(A: np.ndarray, K: int, K_prime: int, rng: np.random.Generator):
    U = top_eigvecs_symmetric(A, K_prime)
    labels = kmeans_on_rows(U, K, rng)
    A_hat = A.copy()
    return A_hat, labels


def run_random_projection(
    A: np.ndarray, K: int, K_prime: int, r: int, q: int, rng: np.random.Generator
):
    n = A.shape[0]
    Omega = rng.standard_normal(size=(n, K_prime + r))
    Y = Omega.copy()
    for _ in range(2 * q + 1):
        Y = A @ Y
    Q, _ = np.linalg.qr(Y, mode="reduced")
    C = Q.T @ A @ Q
    A_hat = Q @ C @ Q.T
    Uc = top_eigvecs_symmetric(C, K_prime)
    U_rp = Q @ Uc
    labels = kmeans_on_rows(U_rp, K, rng)
    return A_hat, labels


def run_random_sampling(
    A: np.ndarray, K: int, K_prime: int, p: float, rng: np.random.Generator
):
    n = A.shape[0]
    tri = np.triu_indices(n, k=1)
    mask = (rng.random(tri[0].shape[0]) < p).astype(float)
    A_s = np.zeros_like(A)
    A_s[tri] = A[tri] * mask / p
    A_s += A_s.T
    np.fill_diagonal(A_s, 0.0)

    try:
        vals, vecs = eigsh(A_s, k=K_prime, which="LA")
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
    except Exception:
        vals, vecs = top_eigpairs_symmetric(A_s, K_prime)

    A_hat = vecs @ np.diag(vals) @ vecs.T
    A_hat = 0.5 * (A_hat + A_hat.T)
    labels = kmeans_on_rows(vecs, K, rng)
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
