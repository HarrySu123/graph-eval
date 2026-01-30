"""
Graph Theory Algorithms

This package contains implementations of various graph algorithms.

Modules:
- coloring: Graph coloring algorithms (greedy, DSatur, chromatic number)
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

__all__ = [
    # Verification
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
    
    # Helper functions
    "build_adjacency_list",
    "build_line_graph_adjacency",
]
