# -*- coding: utf-8 -*-

"""Section 8.2 Table 4-style efficiency experiment on three real networks.

Networks:
- DBLP collaboration network
- Youtube social network
- Internet topology graph

Methods:
- Random Projection
- Random Sampling
- partial_eigen (Python proxy)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    LiveProgress,
    benchmark_table4_methods_sparse,
    format_table4_markdown,
    load_large_integer_edgelist_csr,
    plot_table4_median_bars,
    plot_table4_runtime_boxplots,
    summarize_table4_median_times,
)


@dataclass
class NetworkSpec:
    name: str
    edgelist: Path
    target_rank: int


@dataclass
class Exp82Config:
    dblp_edgelist: Path = Path("data/com-dblp.ungraph.txt")
    youtube_edgelist: Path = Path("data/com-youtube.ungraph.txt")
    internet_edgelist: Path = Path("data/as-skitter.txt")
    reps: int = 20
    seed: int = 2026
    r: int = 10
    q: int = 2
    p: float = 0.7
    delimiter: str = None
    comment_prefix: str = "#"
    outdir: Path = Path("experiments/reference_1_section8_2/results/exp8_2_table4_paper_aligned")
    no_progress: bool = False


def build_network_specs(cfg: Exp82Config):
    return [
        NetworkSpec("DBLP", cfg.dblp_edgelist, 3),
        NetworkSpec("Youtube", cfg.youtube_edgelist, 7),
        NetworkSpec("Internet", cfg.internet_edgelist, 4),
    ]


def run_experiment(cfg: Exp82Config):
    specs = build_network_specs(cfg)
    progress = None if cfg.no_progress else LiveProgress(len(specs) * cfg.reps * 3)
    raw_blocks = []
    dataset_meta = []

    for spec in specs:
        A, _ = load_large_integer_edgelist_csr(
            spec.edgelist,
            delimiter=cfg.delimiter,
            comment_prefix=cfg.comment_prefix,
        )
        df_raw = benchmark_table4_methods_sparse(
            A_csr=A,
            dataset_name=spec.name,
            target_rank=spec.target_rank,
            reps=cfg.reps,
            seed=cfg.seed,
            r=cfg.r,
            q=cfg.q,
            p=cfg.p,
            progress=progress,
        )
        raw_blocks.append(df_raw)
        dataset_meta.append(
            {
                "dataset": spec.name,
                "edgelist": str(spec.edgelist),
                "target_rank": spec.target_rank,
                "n_nodes": int(df_raw["n_nodes"].iloc[0]),
                "n_edges": int(df_raw["n_edges"].iloc[0]),
            }
        )

    if progress is not None:
        progress.close()

    df_raw_all = pd.concat(raw_blocks, ignore_index=True)
    df_summary = summarize_table4_median_times(df_raw_all)
    markdown = format_table4_markdown(df_summary)

    meta = {
        "datasets": dataset_meta,
        "reps": cfg.reps,
        "seed": cfg.seed,
        "q": cfg.q,
        "r": cfg.r,
        "p": cfg.p,
        "partial_eigen_note": "Python proxy implemented with scipy.sparse.linalg.eigsh",
    }
    return df_raw_all, df_summary, markdown, meta


def main():
    parser = argparse.ArgumentParser(description="Section 8.2 Table 4-style efficiency experiment")
    parser.add_argument("--dblp-edgelist", type=str, default="data/com-dblp.ungraph.txt")
    parser.add_argument("--youtube-edgelist", type=str, default="data/com-youtube.ungraph.txt")
    parser.add_argument("--internet-edgelist", type=str, default="data/as-skitter.txt")
    parser.add_argument("--reps", type=int, default=20, help="replications")
    parser.add_argument("--seed", type=int, default=2026, help="master random seed")
    parser.add_argument("--r", type=int, default=10, help="oversampling for random projection")
    parser.add_argument("--q", type=int, default=2, help="power iteration for random projection")
    parser.add_argument("--p", type=float, default=0.7, help="edge sampling probability")
    parser.add_argument("--delimiter", type=str, default=None, help="edge delimiter (default: whitespace)")
    parser.add_argument("--comment-prefix", type=str, default="#")
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/reference_1_section8_2/results/exp8_2_table4_paper_aligned",
    )
    parser.add_argument("--no-progress", action="store_true")
    args, _ = parser.parse_known_args()

    cfg = Exp82Config(
        dblp_edgelist=Path(args.dblp_edgelist),
        youtube_edgelist=Path(args.youtube_edgelist),
        internet_edgelist=Path(args.internet_edgelist),
        reps=args.reps,
        seed=args.seed,
        r=args.r,
        q=args.q,
        p=args.p,
        delimiter=args.delimiter,
        comment_prefix=args.comment_prefix,
        outdir=Path(args.outdir),
        no_progress=args.no_progress,
    )

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    viz_dir = cfg.outdir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    df_raw_all, df_summary, markdown, meta = run_experiment(cfg)

    raw_csv = cfg.outdir / "table4_time_raw.csv"
    summary_csv = cfg.outdir / "table4_like_median_time.csv"
    summary_md = cfg.outdir / "table4_like_median_time.md"
    meta_json = cfg.outdir / "table4_meta.json"
    median_png = viz_dir / "table4_median_bar.png"
    box_png = viz_dir / "table4_runtime_boxplots.png"

    df_raw_all.to_csv(raw_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown, encoding="utf-8")
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot_table4_median_bars(df_summary, median_png)
    plot_table4_runtime_boxplots(df_raw_all, box_png)

    print("Done.")
    print(f"Raw CSV      : {raw_csv.resolve()}")
    print(f"Summary CSV  : {summary_csv.resolve()}")
    print(f"Summary MD   : {summary_md.resolve()}")
    print(f"Median plot  : {median_png.resolve()}")
    print(f"Box plot     : {box_png.resolve()}")
    print(f"Meta JSON    : {meta_json.resolve()}")


if __name__ == "__main__":
    main()
