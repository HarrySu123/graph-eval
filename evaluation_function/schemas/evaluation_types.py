"""
Evaluation Type Definitions

Defines all supported graph theory evaluation types and their descriptions.
"""

from typing import Literal


# =============================================================================
# CORE EVALUATION TYPES (Currently Implemented)
# =============================================================================

CoreEvaluationType = Literal[
    "connectivity",      # Check if graph is connected
    "shortest_path",     # Find shortest path between two nodes
    "bipartite",         # Check if graph is bipartite
    "graph_match",       # Compare student graph with answer graph
]


# =============================================================================
# EXTENDED EVALUATION TYPES (Proposed Features)
# =============================================================================

PathEvaluationType = Literal[
    "eulerian_path",     # Check/find Eulerian path (visits every edge once)
    "eulerian_circuit",  # Check/find Eulerian circuit (closed Eulerian path)
    "hamiltonian_path",  # Check/find Hamiltonian path (visits every vertex once)
    "hamiltonian_circuit",  # Check/find Hamiltonian circuit (closed Hamiltonian path)
]

CycleEvaluationType = Literal[
    "cycle_detection",   # Detect if cycles exist
    "find_all_cycles",   # Find all cycles in the graph
    "shortest_cycle",    # Find the shortest cycle (girth)
    "negative_cycle",    # Detect negative weight cycles
]

TreeEvaluationType = Literal[
    "is_tree",           # Check if graph is a tree
    "spanning_tree",     # Check if subgraph is a spanning tree
    "minimum_spanning_tree",  # Find/verify minimum spanning tree
    "tree_diameter",     # Find the diameter of a tree
    "tree_center",       # Find the center of a tree
]

ColoringEvaluationType = Literal[
    "graph_coloring",    # Find/verify a valid k-coloring
    "chromatic_number",  # Find the chromatic number
    "edge_coloring",     # Color edges so no adjacent edges share color
    "chromatic_index",   # Find the chromatic index (edge chromatic number)
]

FlowEvaluationType = Literal[
    "max_flow",          # Find maximum flow in a network
    "min_cut",           # Find minimum cut in a network
    "bipartite_matching",  # Find maximum bipartite matching
    "assignment_problem",  # Solve assignment/Hungarian algorithm
]

ComponentEvaluationType = Literal[
    "connected_components",      # Find all connected components
    "strongly_connected",        # Find strongly connected components (directed)
    "articulation_points",       # Find articulation points (cut vertices)
    "bridges",                   # Find bridges (cut edges)
    "biconnected_components",    # Find biconnected components
]

StructureEvaluationType = Literal[
    "degree_sequence",   # Verify degree sequence
    "is_planar",         # Check if graph is planar
    "is_complete",       # Check if graph is complete (K_n)
    "is_regular",        # Check if graph is k-regular
    "clique",            # Find maximum clique
    "independent_set",   # Find maximum independent set
    "vertex_cover",      # Find minimum vertex cover
    "dominating_set",    # Find minimum dominating set
]

OrderingEvaluationType = Literal[
    "topological_sort",  # Find topological ordering (DAG)
    "dfs_order",         # Verify DFS traversal order
    "bfs_order",         # Verify BFS traversal order
]

SpecialGraphEvaluationType = Literal[
    "is_dag",            # Check if directed acyclic graph
    "is_forest",         # Check if graph is a forest (acyclic)
    "is_tournament",     # Check if complete directed graph
    "is_wheel",          # Check if wheel graph
    "is_petersen",       # Check if Petersen graph
]


# =============================================================================
# COMBINED EVALUATION TYPE
# =============================================================================

EvaluationType = Literal[
    # Core types
    "connectivity",
    "shortest_path",
    "bipartite",
    "graph_match",
    # Path types
    "eulerian_path",
    "eulerian_circuit",
    "hamiltonian_path",
    "hamiltonian_circuit",
    # Cycle types
    "cycle_detection",
    "find_all_cycles",
    "shortest_cycle",
    "negative_cycle",
    # Tree types
    "is_tree",
    "spanning_tree",
    "minimum_spanning_tree",
    "tree_diameter",
    "tree_center",
    # Coloring types
    "graph_coloring",
    "chromatic_number",
    "edge_coloring",
    "chromatic_index",
    # Flow types
    "max_flow",
    "min_cut",
    "bipartite_matching",
    "assignment_problem",
    # Component types
    "connected_components",
    "strongly_connected",
    "articulation_points",
    "bridges",
    "biconnected_components",
    # Structure types
    "degree_sequence",
    "is_planar",
    "is_complete",
    "is_regular",
    "clique",
    "independent_set",
    "vertex_cover",
    "dominating_set",
    # Ordering types
    "topological_sort",
    "dfs_order",
    "bfs_order",
    # Special graph types
    "is_dag",
    "is_forest",
    "is_tournament",
    "is_wheel",
    "is_petersen",
]


# =============================================================================
# EVALUATION TYPE METADATA
# =============================================================================

EVALUATION_TYPE_INFO: dict[str, dict] = {
    # Core types
    "connectivity": {
        "name": "Connectivity Check",
        "description": "Check if the graph is connected (all vertices reachable from any vertex)",
        "category": "core",
        "complexity": "O(V + E)",
        "supports_directed": True,
        "supports_weighted": True,
    },
    "shortest_path": {
        "name": "Shortest Path",
        "description": "Find the shortest path between two vertices",
        "category": "core",
        "complexity": "O((V + E) log V) with Dijkstra",
        "supports_directed": True,
        "supports_weighted": True,
    },
    "bipartite": {
        "name": "Bipartite Check",
        "description": "Check if the graph can be 2-colored (vertices split into two groups with no edges within groups)",
        "category": "core",
        "complexity": "O(V + E)",
        "supports_directed": False,
        "supports_weighted": True,
    },
    "graph_match": {
        "name": "Graph Matching",
        "description": "Compare student's graph against the expected answer graph",
        "category": "core",
        "complexity": "O(V! worst case for isomorphism)",
        "supports_directed": True,
        "supports_weighted": True,
    },
    # Eulerian/Hamiltonian
    "eulerian_path": {
        "name": "Eulerian Path",
        "description": "Find a path that visits every edge exactly once",
        "category": "path",
        "complexity": "O(E)",
        "supports_directed": True,
        "supports_weighted": False,
    },
    "eulerian_circuit": {
        "name": "Eulerian Circuit",
        "description": "Find a closed path that visits every edge exactly once",
        "category": "path",
        "complexity": "O(E)",
        "supports_directed": True,
        "supports_weighted": False,
    },
    "hamiltonian_path": {
        "name": "Hamiltonian Path",
        "description": "Find a path that visits every vertex exactly once",
        "category": "path",
        "complexity": "O(n! worst case, NP-complete)",
        "supports_directed": True,
        "supports_weighted": False,
    },
    "hamiltonian_circuit": {
        "name": "Hamiltonian Circuit",
        "description": "Find a closed path that visits every vertex exactly once",
        "category": "path",
        "complexity": "O(n! worst case, NP-complete)",
        "supports_directed": True,
        "supports_weighted": False,
    },
    # Tree types
    "minimum_spanning_tree": {
        "name": "Minimum Spanning Tree",
        "description": "Find/verify a spanning tree with minimum total edge weight",
        "category": "tree",
        "complexity": "O(E log V) with Kruskal/Prim",
        "supports_directed": False,
        "supports_weighted": True,
    },
    "is_tree": {
        "name": "Tree Check",
        "description": "Check if the graph is a tree (connected and acyclic)",
        "category": "tree",
        "complexity": "O(V + E)",
        "supports_directed": False,
        "supports_weighted": True,
    },
    # Flow types
    "max_flow": {
        "name": "Maximum Flow",
        "description": "Find the maximum flow from source to sink in a flow network",
        "category": "flow",
        "complexity": "O(VE²) with Edmonds-Karp",
        "supports_directed": True,
        "supports_weighted": True,
    },
    "min_cut": {
        "name": "Minimum Cut",
        "description": "Find the minimum cut separating source and sink",
        "category": "flow",
        "complexity": "O(VE²) with Edmonds-Karp",
        "supports_directed": True,
        "supports_weighted": True,
    },
    "bipartite_matching": {
        "name": "Bipartite Matching",
        "description": "Find maximum matching in a bipartite graph",
        "category": "flow",
        "complexity": "O(V × E) with Hungarian",
        "supports_directed": False,
        "supports_weighted": True,
    },
    # Component types
    "strongly_connected": {
        "name": "Strongly Connected Components",
        "description": "Find all strongly connected components in a directed graph",
        "category": "component",
        "complexity": "O(V + E) with Tarjan/Kosaraju",
        "supports_directed": True,
        "supports_weighted": True,
    },
    "articulation_points": {
        "name": "Articulation Points",
        "description": "Find vertices whose removal disconnects the graph",
        "category": "component",
        "complexity": "O(V + E)",
        "supports_directed": False,
        "supports_weighted": False,
    },
    "bridges": {
        "name": "Bridges",
        "description": "Find edges whose removal disconnects the graph",
        "category": "component",
        "complexity": "O(V + E)",
        "supports_directed": False,
        "supports_weighted": False,
    },
    # Coloring types
    "graph_coloring": {
        "name": "Graph Coloring",
        "description": "Find/verify a valid k-coloring of the graph",
        "category": "coloring",
        "complexity": "O(k^V) worst case, NP-complete",
        "supports_directed": False,
        "supports_weighted": False,
    },
    "chromatic_number": {
        "name": "Chromatic Number",
        "description": "Find the minimum number of colors needed to color the graph",
        "category": "coloring",
        "complexity": "NP-hard",
        "supports_directed": False,
        "supports_weighted": False,
    },
    # Ordering types
    "topological_sort": {
        "name": "Topological Sort",
        "description": "Find a linear ordering of vertices in a DAG",
        "category": "ordering",
        "complexity": "O(V + E)",
        "supports_directed": True,
        "supports_weighted": False,
    },
    # Structure types
    "is_planar": {
        "name": "Planarity Check",
        "description": "Check if the graph can be drawn without edge crossings",
        "category": "structure",
        "complexity": "O(V) with proper algorithm",
        "supports_directed": False,
        "supports_weighted": False,
    },
    "clique": {
        "name": "Maximum Clique",
        "description": "Find the largest complete subgraph",
        "category": "structure",
        "complexity": "O(2^n) worst case, NP-complete",
        "supports_directed": False,
        "supports_weighted": False,
    },
    "independent_set": {
        "name": "Maximum Independent Set",
        "description": "Find the largest set of vertices with no edges between them",
        "category": "structure",
        "complexity": "O(2^n) worst case, NP-complete",
        "supports_directed": False,
        "supports_weighted": False,
    },
    "vertex_cover": {
        "name": "Minimum Vertex Cover",
        "description": "Find the smallest set of vertices that covers all edges",
        "category": "structure",
        "complexity": "O(2^n) worst case, NP-complete",
        "supports_directed": False,
        "supports_weighted": False,
    },
}
