"""
Graph Theory Algorithms

This package contains implementations of various graph algorithms.

Modules:
- coloring: Graph coloring algorithms (greedy, DSatur, chromatic number)
- mst: Minimum spanning tree algorithms (Kruskal, Prim, verification)
"""

from .coloring import (
    # Verification
    verify_vertex_coloring,
    verify_edge_coloring,
    detect_coloring_conflicts,
    detect_edge_coloring_conflicts,
    
    # Coloring algorithms
    greedy_coloring,
    dsatur_coloring,
    greedy_edge_coloring,
    
    # Chromatic number
    compute_chromatic_number,
    compute_chromatic_index,
    
    # Helper functions
    build_adjacency_list,
    build_line_graph_adjacency,
)

from .mst import (
    # Union-Find data structure
    UnionFind,
    
    # MST algorithms
    kruskal_mst,
    prim_mst,
    compute_mst,
    
    # MST verification
    verify_spanning_tree,
    verify_mst,
    verify_mst_edges,
    
    # Disconnected graph handling
    compute_minimum_spanning_forest,
    
    # Visualization support
    get_mst_visualization,
    get_mst_animation_steps,
    
    # High-level API
    find_mst,
    evaluate_mst_submission,
    
    # Helper functions
    build_adjacency_list_weighted,
    is_graph_connected,
    count_components,
)

__all__ = [
    # Coloring - Verification
    "verify_vertex_coloring",
    "verify_edge_coloring",
    "detect_coloring_conflicts",
    "detect_edge_coloring_conflicts",
    
    # Coloring algorithms
    "greedy_coloring",
    "dsatur_coloring",
    "greedy_edge_coloring",
    
    # Chromatic number
    "compute_chromatic_number",
    "compute_chromatic_index",
    
    # Coloring - Helper functions
    "build_adjacency_list",
    "build_line_graph_adjacency",
    
    # MST - Union-Find
    "UnionFind",
    
    # MST algorithms
    "kruskal_mst",
    "prim_mst",
    "compute_mst",
    
    # MST verification
    "verify_spanning_tree",
    "verify_mst",
    "verify_mst_edges",
    
    # MST - Disconnected graph handling
    "compute_minimum_spanning_forest",
    
    # MST - Visualization support
    "get_mst_visualization",
    "get_mst_animation_steps",
    
    # MST - High-level API
    "find_mst",
    "evaluate_mst_submission",
    
    # MST - Helper functions
    "build_adjacency_list_weighted",
    "is_graph_connected",
    "count_components",
]
