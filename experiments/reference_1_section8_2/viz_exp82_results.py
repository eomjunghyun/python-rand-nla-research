from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def plot_runtime_median(med_csv: Path, out_png: Path):
    df = _read_csv(med_csv)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D", "#BAB0AC"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["method"], df["median_time_sec"], color=colors[: len(df)])
    ax.set_ylabel("Median runtime (sec)")
    ax.set_title("Table4-like Median Runtime")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_runtime_raw_box(raw_csv: Path, out_png: Path):
    df = _read_csv(raw_csv)
    methods = list(dict.fromkeys(df["method"].tolist()))
    data = [df.loc[df["method"] == m, "time_sec"].values for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, tick_labels=methods, showfliers=False)
    ax.set_ylabel("Runtime (sec)")
    ax.set_title("Per-rep Runtime Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ari_heatmap(mean_csv: Path, out_png: Path):
    mat = pd.read_csv(mean_csv, index_col=0)

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    im = ax.imshow(mat.values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right")
    ax.set_yticklabels(mat.index)
    ax.set_title("Mean Pairwise ARI")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = float(mat.iat[i, j])
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v < 0.55 else "black",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ARI")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ari_raw_box(raw_csv: Path, out_png: Path):
    df = _read_csv(raw_csv)
    df["pair"] = df["method_i"] + " vs " + df["method_j"]
    pairs = list(dict.fromkeys(df["pair"].tolist()))
    data = [df.loc[df["pair"] == p, "ari"].values for p in pairs]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot(data, tick_labels=pairs, showfliers=False)
    ax.set_ylabel("ARI")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-rep Pairwise ARI")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing exp8_2 CSV outputs",
    )
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    outdir = rdir / "viz"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_runtime_median(
        rdir / "dblp_table4_like_median_time.csv",
        outdir / "runtime_median_bar.png",
    )
    plot_runtime_raw_box(
        rdir / "dblp_time_raw.csv",
        outdir / "runtime_per_rep_box.png",
    )
    plot_ari_heatmap(
        rdir / "dblp_pairwise_ari_mean_matrix.csv",
        outdir / "ari_mean_heatmap.png",
    )
    plot_ari_raw_box(
        rdir / "dblp_pairwise_ari_raw.csv",
        outdir / "ari_per_rep_box.png",
    )

    print("Done.")
    print(f"Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
