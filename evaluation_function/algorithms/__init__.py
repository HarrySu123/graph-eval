"""
Graph Theory Algorithms

This package contains implementations of various graph algorithms.

Modules:
- coloring: Graph coloring algorithms (greedy, DSatur, chromatic number)
- mst: Minimum spanning tree algorithms (Kruskal, Prim, verification)
- path: Eulerian and Hamiltonian path/circuit algorithms
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

from .path import (
    # Eulerian existence checks
    check_eulerian_undirected,
    check_eulerian_directed,
    check_eulerian_existence,
    
    # Eulerian path/circuit finding
    find_eulerian_path_undirected,
    find_eulerian_path_directed,
    find_eulerian_path,
    find_eulerian_circuit,
    
    # Eulerian verification
    verify_eulerian_path,
    
    # Hamiltonian verification
    verify_hamiltonian_path,
    
    # Hamiltonian existence
    find_hamiltonian_path_backtrack,
    check_hamiltonian_existence,
    
    # High-level API
    evaluate_eulerian_path,
    evaluate_hamiltonian_path,
    
    # Feedback
    get_eulerian_feedback,
    get_hamiltonian_feedback,
    
    # Helper functions (path module)
    build_adjacency_multiset,
    get_in_out_degree,
    is_connected_undirected,
    is_weakly_connected_directed,
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
    
    # Path - Eulerian existence checks
    "check_eulerian_undirected",
    "check_eulerian_directed",
    "check_eulerian_existence",
    
    # Path - Eulerian path/circuit finding
    "find_eulerian_path_undirected",
    "find_eulerian_path_directed",
    "find_eulerian_path",
    "find_eulerian_circuit",
    
    # Path - Eulerian verification
    "verify_eulerian_path",
    
    # Path - Hamiltonian verification
    "verify_hamiltonian_path",
    
    # Path - Hamiltonian existence
    "find_hamiltonian_path_backtrack",
    "check_hamiltonian_existence",
    
    # Path - High-level API
    "evaluate_eulerian_path",
    "evaluate_hamiltonian_path",
    
    # Path - Feedback
    "get_eulerian_feedback",
    "get_hamiltonian_feedback",
    
    # Path - Helper functions
    "build_adjacency_multiset",
    "get_in_out_degree",
    "is_connected_undirected",
    "is_weakly_connected_directed",
]
