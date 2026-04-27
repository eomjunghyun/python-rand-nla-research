# -*- coding: utf-8 -*-

"""Prepare Section 8.1 real-network datasets.

The raw downloads used here are:
- Newman/Adamic-Glance political blogs: polblogs.gml
- Ji & Jin (2016) statisticians data: SCC2016-with-abs

Outputs are sparse adjacency matrices plus label/meta files under
data/reference_1_section8_1/processed.
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data/reference_1_section8_1/raw"
DEFAULT_OUT = ROOT / "data/reference_1_section8_1/processed"


def ensure_unzipped(zip_path: Path, marker_path: Path, extract_dir: Path):
    if marker_path.exists():
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing raw zip file: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def largest_component(A: sp.csr_matrix):
    _, labels = connected_components(A, directed=False, return_labels=True)
    counts = np.bincount(labels)
    gid = int(np.argmax(counts))
    idx = np.where(labels == gid)[0]
    return A[idx][:, idx].tocsr(), idx, counts


def write_dataset(
    outdir: Path,
    name: str,
    A: sp.csr_matrix,
    target_rank: int,
    labels: np.ndarray | None = None,
    node_names: list[str] | None = None,
    extra_meta: dict | None = None,
):
    outdir.mkdir(parents=True, exist_ok=True)
    A = A.astype(np.float32).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    sp.save_npz(outdir / f"{name}_adjacency.npz", A)

    if labels is not None:
        np.save(outdir / f"{name}_labels.npy", np.asarray(labels, dtype=int))
    if node_names is not None:
        (outdir / f"{name}_nodes.txt").write_text(
            "\n".join(str(x) for x in node_names) + "\n",
            encoding="utf-8",
        )

    meta = {
        "name": name,
        "num_nodes": int(A.shape[0]),
        "num_edges": int(A.nnz // 2),
        "target_rank": int(target_rank),
        "has_ground_truth": labels is not None,
    }
    if labels is not None:
        uniq, cnt = np.unique(labels, return_counts=True)
        meta["num_classes"] = int(len(uniq))
        meta["class_counts"] = {str(int(k)): int(v) for k, v in zip(uniq, cnt)}
    if extra_meta:
        meta.update(extra_meta)
    (outdir / f"{name}_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def parse_polblogs_gml(path: Path):
    nodes: dict[int, dict] = {}
    directed_edges: list[tuple[int, int]] = []
    current = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s == "node [":
                current = {"type": "node"}
                continue
            if s == "edge [":
                current = {"type": "edge"}
                continue
            if s == "]":
                if current is not None:
                    if current["type"] == "node":
                        nodes[int(current["id"])] = current
                    elif current["type"] == "edge":
                        directed_edges.append((int(current["source"]), int(current["target"])))
                current = None
                continue
            if current is None:
                continue
            parts = s.split(" ", 1)
            if len(parts) != 2:
                continue
            key, value = parts
            value = value.strip().strip('"')
            if key in {"id", "value"} or (current["type"] == "edge" and key in {"source", "target"}):
                value = int(value)
            current[key] = value

    return nodes, directed_edges


def prepare_political_blogs(raw_dir: Path, out_dir: Path):
    ensure_unzipped(
        raw_dir / "polblogs.zip",
        raw_dir / "polblogs/polblogs.gml",
        raw_dir / "polblogs",
    )
    gml_path = raw_dir / "polblogs/polblogs.gml"
    nodes, directed_edges = parse_polblogs_gml(gml_path)
    sorted_node_ids = sorted(nodes)
    node_to_idx = {node_id: i for i, node_id in enumerate(sorted_node_ids)}

    undirected_edges = set()
    self_loops = 0
    for u, v in directed_edges:
        if u == v:
            self_loops += 1
            continue
        a = node_to_idx[u]
        b = node_to_idx[v]
        if a > b:
            a, b = b, a
        undirected_edges.add((a, b))

    rows = []
    cols = []
    for u, v in sorted(undirected_edges):
        rows.extend((u, v))
        cols.extend((v, u))
    A = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(sorted_node_ids), len(sorted_node_ids)),
    ).tocsr()
    A_lcc, lcc_idx, comp_sizes = largest_component(A)
    node_ids_lcc = [sorted_node_ids[i] for i in lcc_idx]
    labels = np.array([int(nodes[node_id]["value"]) for node_id in node_ids_lcc], dtype=int)
    node_names = [str(nodes[node_id].get("label", node_id)) for node_id in node_ids_lcc]

    write_dataset(
        out_dir,
        "political_blog",
        A_lcc,
        target_rank=2,
        labels=labels,
        node_names=node_names,
        extra_meta={
            "source_file": str(gml_path),
            "raw_nodes": int(len(nodes)),
            "raw_directed_edges": int(len(directed_edges)),
            "raw_unique_directed_edges": int(len(set(directed_edges))),
            "raw_self_loops": int(self_loops),
            "component_sizes_top10": [int(x) for x in sorted(comp_sizes, reverse=True)[:10]],
        },
    )


def load_dense_text_matrix(path: Path, dtype=np.int8):
    return np.loadtxt(path, dtype=dtype)


def prepare_statisticians(raw_dir: Path, out_dir: Path):
    ensure_unzipped(
        raw_dir / "SCC2016-with-abs.zip",
        raw_dir / "scc2016/SCC2016-with-abs/SCC2016/Data/authorPaperBiadj.txt",
        raw_dir / "scc2016",
    )
    base = raw_dir / "scc2016/SCC2016-with-abs/SCC2016/Data"
    author_names = [
        line.strip().strip('"')
        for line in (base / "authorList.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]

    author_paper = sp.csr_matrix(load_dense_text_matrix(base / "authorPaperBiadj.txt", dtype=np.int8))
    paper_cit = sp.csr_matrix(load_dense_text_matrix(base / "paperCitAdj.txt", dtype=np.int8))

    coauthor_weighted = (author_paper @ author_paper.T).tocsr()
    coauthor = (coauthor_weighted >= 1).astype(np.float32).tocsr()
    coauthor.setdiag(0.0)
    coauthor.eliminate_zeros()
    coauthor_lcc, coauthor_idx, coauthor_sizes = largest_component(coauthor)
    write_dataset(
        out_dir,
        "statisticians_coauthor",
        coauthor_lcc,
        target_rank=3,
        node_names=[author_names[i] for i in coauthor_idx],
        extra_meta={
            "source_file": str(base / "authorPaperBiadj.txt"),
            "raw_authors": int(coauthor.shape[0]),
            "raw_edges": int(coauthor.nnz // 2),
            "component_sizes_top10": [int(x) for x in sorted(coauthor_sizes, reverse=True)[:10]],
            "ground_truth_reference": "No true labels; Section 8.1 evaluates randomized labels relative to non-random spectral clustering.",
        },
    )

    citation_weighted = (author_paper @ paper_cit @ author_paper.T).tocsr()
    citation_directed = (citation_weighted >= 1).astype(np.float32).tocsr()
    citation_directed.setdiag(0.0)
    citation_directed.eliminate_zeros()
    citation_undirected = ((citation_directed + citation_directed.T) >= 1).astype(np.float32).tocsr()
    citation_undirected.setdiag(0.0)
    citation_undirected.eliminate_zeros()
    citation_lcc, citation_idx, citation_sizes = largest_component(citation_undirected)
    write_dataset(
        out_dir,
        "statisticians_citation",
        citation_lcc,
        target_rank=3,
        node_names=[author_names[i] for i in citation_idx],
        extra_meta={
            "source_file": str(base / "paperCitAdj.txt"),
            "raw_authors": int(citation_undirected.shape[0]),
            "raw_directed_arcs": int(citation_directed.nnz),
            "raw_undirected_edges": int(citation_undirected.nnz // 2),
            "component_sizes_top10": [int(x) for x in sorted(citation_sizes, reverse=True)[:10]],
            "ground_truth_reference": "No true labels; Section 8.1 evaluates randomized labels relative to non-random spectral clustering.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare Section 8.1 real datasets")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    prepare_political_blogs(raw_dir, out_dir)
    prepare_statisticians(raw_dir, out_dir)

    for name in ("political_blog", "statisticians_coauthor", "statisticians_citation"):
        meta = json.loads((out_dir / f"{name}_meta.json").read_text(encoding="utf-8"))
        print(f"{name}: nodes={meta['num_nodes']}, edges={meta['num_edges']}, target_rank={meta['target_rank']}")


if __name__ == "__main__":
    main()
