"""
Evaluation Type Definitions

Defines all supported graph theory evaluation types and their descriptions.
"""

from typing import Literal
from enum import Enum
from pydantic import BaseModel, Field


# =============================================================================
# EVALUATION TYPE ENUMS
# =============================================================================

class EvaluationCategory(str, Enum):
    """Categories of evaluation types."""
    CORE = "core"
    PATH = "path"
    CYCLE = "cycle"
    TREE = "tree"
    COLORING = "coloring"
    FLOW = "flow"
    COMPONENT = "component"
    STRUCTURE = "structure"
    ORDERING = "ordering"
    SPECIAL = "special"


class CoreEvaluation(str, Enum):
    """Core evaluation types."""
    CONNECTIVITY = "connectivity"
    SHORTEST_PATH = "shortest_path"
    BIPARTITE = "bipartite"
    GRAPH_MATCH = "graph_match"


class PathEvaluation(str, Enum):
    """Path-related evaluation types."""
    EULERIAN_PATH = "eulerian_path"
    EULERIAN_CIRCUIT = "eulerian_circuit"
    HAMILTONIAN_PATH = "hamiltonian_path"
    HAMILTONIAN_CIRCUIT = "hamiltonian_circuit"


class CycleEvaluation(str, Enum):
    """Cycle-related evaluation types."""
    CYCLE_DETECTION = "cycle_detection"
    FIND_ALL_CYCLES = "find_all_cycles"
    SHORTEST_CYCLE = "shortest_cycle"
    NEGATIVE_CYCLE = "negative_cycle"


class TreeEvaluation(str, Enum):
    """Tree-related evaluation types."""
    IS_TREE = "is_tree"
    SPANNING_TREE = "spanning_tree"
    MINIMUM_SPANNING_TREE = "minimum_spanning_tree"
    TREE_DIAMETER = "tree_diameter"
    TREE_CENTER = "tree_center"


class ColoringEvaluation(str, Enum):
    """Coloring-related evaluation types."""
    GRAPH_COLORING = "graph_coloring"
    CHROMATIC_NUMBER = "chromatic_number"
    EDGE_COLORING = "edge_coloring"
    CHROMATIC_INDEX = "chromatic_index"


class FlowEvaluation(str, Enum):
    """Flow network evaluation types."""
    MAX_FLOW = "max_flow"
    MIN_CUT = "min_cut"
    BIPARTITE_MATCHING = "bipartite_matching"
    ASSIGNMENT_PROBLEM = "assignment_problem"


class ComponentEvaluation(str, Enum):
    """Component-related evaluation types."""
    CONNECTED_COMPONENTS = "connected_components"
    STRONGLY_CONNECTED = "strongly_connected"
    ARTICULATION_POINTS = "articulation_points"
    BRIDGES = "bridges"
    BICONNECTED_COMPONENTS = "biconnected_components"


class StructureEvaluation(str, Enum):
    """Graph structure evaluation types."""
    DEGREE_SEQUENCE = "degree_sequence"
    IS_PLANAR = "is_planar"
    IS_COMPLETE = "is_complete"
    IS_REGULAR = "is_regular"
    CLIQUE = "clique"
    INDEPENDENT_SET = "independent_set"
    VERTEX_COVER = "vertex_cover"
    DOMINATING_SET = "dominating_set"


class OrderingEvaluation(str, Enum):
    """Ordering evaluation types."""
    TOPOLOGICAL_SORT = "topological_sort"
    DFS_ORDER = "dfs_order"
    BFS_ORDER = "bfs_order"


class SpecialGraphEvaluation(str, Enum):
    """Special graph type evaluation."""
    IS_DAG = "is_dag"
    IS_FOREST = "is_forest"
    IS_TOURNAMENT = "is_tournament"
    IS_WHEEL = "is_wheel"
    IS_PETERSEN = "is_petersen"


# Combined type for all evaluations
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

class EvaluationTypeInfo(BaseModel):
    """Metadata about an evaluation type."""
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Description of what this evaluation does")
    category: EvaluationCategory = Field(..., description="Category of evaluation")
    complexity: str = Field(..., description="Time complexity")
    supports_directed: bool = Field(True, description="Whether it works with directed graphs")
    supports_weighted: bool = Field(True, description="Whether it works with weighted graphs")


EVALUATION_TYPE_INFO: dict[str, EvaluationTypeInfo] = {
    # Core types
    "connectivity": EvaluationTypeInfo(
        name="Connectivity Check",
        description="Check if the graph is connected (all vertices reachable from any vertex)",
        category=EvaluationCategory.CORE,
        complexity="O(V + E)",
        supports_directed=True,
        supports_weighted=True,
    ),
    "shortest_path": EvaluationTypeInfo(
        name="Shortest Path",
        description="Find the shortest path between two vertices",
        category=EvaluationCategory.CORE,
        complexity="O((V + E) log V) with Dijkstra",
        supports_directed=True,
        supports_weighted=True,
    ),
    "bipartite": EvaluationTypeInfo(
        name="Bipartite Check",
        description="Check if the graph can be 2-colored (vertices split into two groups with no edges within groups)",
        category=EvaluationCategory.CORE,
        complexity="O(V + E)",
        supports_directed=False,
        supports_weighted=True,
    ),
    "graph_match": EvaluationTypeInfo(
        name="Graph Matching",
        description="Compare student's graph against the expected answer graph",
        category=EvaluationCategory.CORE,
        complexity="O(V! worst case for isomorphism)",
        supports_directed=True,
        supports_weighted=True,
    ),
    # Eulerian/Hamiltonian
    "eulerian_path": EvaluationTypeInfo(
        name="Eulerian Path",
        description="Find a path that visits every edge exactly once",
        category=EvaluationCategory.PATH,
        complexity="O(E)",
        supports_directed=True,
        supports_weighted=False,
    ),
    "eulerian_circuit": EvaluationTypeInfo(
        name="Eulerian Circuit",
        description="Find a closed path that visits every edge exactly once",
        category=EvaluationCategory.PATH,
        complexity="O(E)",
        supports_directed=True,
        supports_weighted=False,
    ),
    "hamiltonian_path": EvaluationTypeInfo(
        name="Hamiltonian Path",
        description="Find a path that visits every vertex exactly once",
        category=EvaluationCategory.PATH,
        complexity="O(n! worst case, NP-complete)",
        supports_directed=True,
        supports_weighted=False,
    ),
    "hamiltonian_circuit": EvaluationTypeInfo(
        name="Hamiltonian Circuit",
        description="Find a closed path that visits every vertex exactly once",
        category=EvaluationCategory.PATH,
        complexity="O(n! worst case, NP-complete)",
        supports_directed=True,
        supports_weighted=False,
    ),
    # Tree types
    "minimum_spanning_tree": EvaluationTypeInfo(
        name="Minimum Spanning Tree",
        description="Find/verify a spanning tree with minimum total edge weight",
        category=EvaluationCategory.TREE,
        complexity="O(E log V) with Kruskal/Prim",
        supports_directed=False,
        supports_weighted=True,
    ),
    "is_tree": EvaluationTypeInfo(
        name="Tree Check",
        description="Check if the graph is a tree (connected and acyclic)",
        category=EvaluationCategory.TREE,
        complexity="O(V + E)",
        supports_directed=False,
        supports_weighted=True,
    ),
    "spanning_tree": EvaluationTypeInfo(
        name="Spanning Tree",
        description="Check if a subgraph is a valid spanning tree",
        category=EvaluationCategory.TREE,
        complexity="O(V + E)",
        supports_directed=False,
        supports_weighted=True,
    ),
    "tree_diameter": EvaluationTypeInfo(
        name="Tree Diameter",
        description="Find the longest path in a tree",
        category=EvaluationCategory.TREE,
        complexity="O(V)",
        supports_directed=False,
        supports_weighted=True,
    ),
    "tree_center": EvaluationTypeInfo(
        name="Tree Center",
        description="Find the center vertex(es) of a tree",
        category=EvaluationCategory.TREE,
        complexity="O(V)",
        supports_directed=False,
        supports_weighted=False,
    ),
    # Flow types
    "max_flow": EvaluationTypeInfo(
        name="Maximum Flow",
        description="Find the maximum flow from source to sink in a flow network",
        category=EvaluationCategory.FLOW,
        complexity="O(VE²) with Edmonds-Karp",
        supports_directed=True,
        supports_weighted=True,
    ),
    "min_cut": EvaluationTypeInfo(
        name="Minimum Cut",
        description="Find the minimum cut separating source and sink",
        category=EvaluationCategory.FLOW,
        complexity="O(VE²) with Edmonds-Karp",
        supports_directed=True,
        supports_weighted=True,
    ),
    "bipartite_matching": EvaluationTypeInfo(
        name="Bipartite Matching",
        description="Find maximum matching in a bipartite graph",
        category=EvaluationCategory.FLOW,
        complexity="O(V × E) with Hungarian",
        supports_directed=False,
        supports_weighted=True,
    ),
    # Component types
    "strongly_connected": EvaluationTypeInfo(
        name="Strongly Connected Components",
        description="Find all strongly connected components in a directed graph",
        category=EvaluationCategory.COMPONENT,
        complexity="O(V + E) with Tarjan/Kosaraju",
        supports_directed=True,
        supports_weighted=True,
    ),
    "connected_components": EvaluationTypeInfo(
        name="Connected Components",
        description="Find all connected components in an undirected graph",
        category=EvaluationCategory.COMPONENT,
        complexity="O(V + E)",
        supports_directed=False,
        supports_weighted=True,
    ),
    "articulation_points": EvaluationTypeInfo(
        name="Articulation Points",
        description="Find vertices whose removal disconnects the graph",
        category=EvaluationCategory.COMPONENT,
        complexity="O(V + E)",
        supports_directed=False,
        supports_weighted=False,
    ),
    "bridges": EvaluationTypeInfo(
        name="Bridges",
        description="Find edges whose removal disconnects the graph",
        category=EvaluationCategory.COMPONENT,
        complexity="O(V + E)",
        supports_directed=False,
        supports_weighted=False,
    ),
    # Coloring types
    "graph_coloring": EvaluationTypeInfo(
        name="Graph Coloring",
        description="Find/verify a valid k-coloring of the graph",
        category=EvaluationCategory.COLORING,
        complexity="O(k^V) worst case, NP-complete",
        supports_directed=False,
        supports_weighted=False,
    ),
    "chromatic_number": EvaluationTypeInfo(
        name="Chromatic Number",
        description="Find the minimum number of colors needed to color the graph",
        category=EvaluationCategory.COLORING,
        complexity="NP-hard",
        supports_directed=False,
        supports_weighted=False,
    ),
    # Cycle types
    "cycle_detection": EvaluationTypeInfo(
        name="Cycle Detection",
        description="Detect if the graph contains any cycles",
        category=EvaluationCategory.CYCLE,
        complexity="O(V + E)",
        supports_directed=True,
        supports_weighted=True,
    ),
    "negative_cycle": EvaluationTypeInfo(
        name="Negative Cycle Detection",
        description="Detect if the graph contains a negative weight cycle",
        category=EvaluationCategory.CYCLE,
        complexity="O(VE) with Bellman-Ford",
        supports_directed=True,
        supports_weighted=True,
    ),
    "shortest_cycle": EvaluationTypeInfo(
        name="Shortest Cycle (Girth)",
        description="Find the length of the shortest cycle in the graph",
        category=EvaluationCategory.CYCLE,
        complexity="O(VE)",
        supports_directed=False,
        supports_weighted=False,
    ),
    # Ordering types
    "topological_sort": EvaluationTypeInfo(
        name="Topological Sort",
        description="Find a linear ordering of vertices in a DAG",
        category=EvaluationCategory.ORDERING,
        complexity="O(V + E)",
        supports_directed=True,
        supports_weighted=False,
    ),
    "dfs_order": EvaluationTypeInfo(
        name="DFS Order",
        description="Verify a depth-first search traversal order",
        category=EvaluationCategory.ORDERING,
        complexity="O(V + E)",
        supports_directed=True,
        supports_weighted=False,
    ),
    "bfs_order": EvaluationTypeInfo(
        name="BFS Order",
        description="Verify a breadth-first search traversal order",
        category=EvaluationCategory.ORDERING,
        complexity="O(V + E)",
        supports_directed=True,
        supports_weighted=False,
    ),
    # Structure types
    "is_planar": EvaluationTypeInfo(
        name="Planarity Check",
        description="Check if the graph can be drawn without edge crossings",
        category=EvaluationCategory.STRUCTURE,
        complexity="O(V) with proper algorithm",
        supports_directed=False,
        supports_weighted=False,
    ),
    "degree_sequence": EvaluationTypeInfo(
        name="Degree Sequence",
        description="Verify or compute the degree sequence of the graph",
        category=EvaluationCategory.STRUCTURE,
        complexity="O(V + E)",
        supports_directed=True,
        supports_weighted=False,
    ),
    "clique": EvaluationTypeInfo(
        name="Maximum Clique",
        description="Find the largest complete subgraph",
        category=EvaluationCategory.STRUCTURE,
        complexity="O(2^n) worst case, NP-complete",
        supports_directed=False,
        supports_weighted=False,
    ),
    "independent_set": EvaluationTypeInfo(
        name="Maximum Independent Set",
        description="Find the largest set of vertices with no edges between them",
        category=EvaluationCategory.STRUCTURE,
        complexity="O(2^n) worst case, NP-complete",
        supports_directed=False,
        supports_weighted=False,
    ),
    "vertex_cover": EvaluationTypeInfo(
        name="Minimum Vertex Cover",
        description="Find the smallest set of vertices that covers all edges",
        category=EvaluationCategory.STRUCTURE,
        complexity="O(2^n) worst case, NP-complete",
        supports_directed=False,
        supports_weighted=False,
    ),
    "is_complete": EvaluationTypeInfo(
        name="Complete Graph Check",
        description="Check if the graph is a complete graph",
        category=EvaluationCategory.STRUCTURE,
        complexity="O(V²)",
        supports_directed=False,
        supports_weighted=False,
    ),
    "is_regular": EvaluationTypeInfo(
        name="Regular Graph Check",
        description="Check if all vertices have the same degree",
        category=EvaluationCategory.STRUCTURE,
        complexity="O(V + E)",
        supports_directed=False,
        supports_weighted=False,
    ),
    # Special graph types
    "is_dag": EvaluationTypeInfo(
        name="DAG Check",
        description="Check if the directed graph is acyclic",
        category=EvaluationCategory.SPECIAL,
        complexity="O(V + E)",
        supports_directed=True,
        supports_weighted=True,
    ),
    "is_forest": EvaluationTypeInfo(
        name="Forest Check",
        description="Check if the graph is a forest (collection of trees)",
        category=EvaluationCategory.SPECIAL,
        complexity="O(V + E)",
        supports_directed=False,
        supports_weighted=True,
    ),
}
