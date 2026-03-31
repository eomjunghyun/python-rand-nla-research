"""Backward-compatible re-export for hypergraph utilities.

Core implementations now live in ``src.common``.
"""

from src.common import (
    Edge,
    SUPPORTED_JOIN_STRATEGIES,
    assign_equal_communities,
    build_probability_matrix,
    clique_expansion_adjacency,
    composition_delta,
    empirical_pmf,
    generate_hypergraph,
    generate_nonuniform_hsbm_instance,
    generate_uniform_hsbm_instance,
    hyperedge_community_count_matrix,
    hyperedge_gini_scores,
    hyperedge_sizes,
    hyperedges_to_incidence_csr,
    hypergraph_basic_stats,
    node_degrees_from_hyperedges,
    normalized_gini,
    sample_uniform_hsbm_hyperedges_exact,
    sample_uniform_hsbm_hyperedges_sparse,
    zhou_normalized_laplacian,
)

__all__ = [
    "Edge",
    "SUPPORTED_JOIN_STRATEGIES",
    "assign_equal_communities",
    "build_probability_matrix",
    "clique_expansion_adjacency",
    "composition_delta",
    "empirical_pmf",
    "generate_hypergraph",
    "generate_nonuniform_hsbm_instance",
    "generate_uniform_hsbm_instance",
    "hyperedge_community_count_matrix",
    "hyperedge_gini_scores",
    "hyperedge_sizes",
    "hyperedges_to_incidence_csr",
    "hypergraph_basic_stats",
    "node_degrees_from_hyperedges",
    "normalized_gini",
    "sample_uniform_hsbm_hyperedges_exact",
    "sample_uniform_hsbm_hyperedges_sparse",
    "zhou_normalized_laplacian",
]
