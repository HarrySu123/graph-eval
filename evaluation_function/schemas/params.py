"""
Evaluation Parameter Schemas

Defines the parameter schemas for each evaluation type.
"""

from typing import TypedDict, Literal


# =============================================================================
# CORE EVALUATION PARAMETERS
# =============================================================================

class ConnectivityParams(TypedDict, total=False):
    """
    Parameters for connectivity evaluation.
    
    Attributes:
        check_type: Type of connectivity check
            - "connected": Check if undirected graph is connected
            - "strongly_connected": Check if directed graph is strongly connected
            - "weakly_connected": Check if directed graph is weakly connected
        return_components: Whether to return connected components
    """
    check_type: Literal["connected", "strongly_connected", "weakly_connected"]
    return_components: bool


class ShortestPathParams(TypedDict, total=False):
    """
    Parameters for shortest path evaluation.
    
    Attributes:
        source_node: ID of the starting node (required)
        target_node: ID of the ending node (required)
        algorithm: Preferred algorithm (auto-selected if not specified)
            - "bfs": Breadth-first search (unweighted)
            - "dijkstra": Dijkstra's algorithm (non-negative weights)
            - "bellman_ford": Bellman-Ford (handles negative weights)
            - "floyd_warshall": All pairs shortest paths
        return_path: Whether to return the actual path (default: True)
        return_distance: Whether to return the distance (default: True)
        all_paths: Whether to return all shortest paths (if multiple exist)
    """
    source_node: str  # Required
    target_node: str  # Required
    algorithm: Literal["bfs", "dijkstra", "bellman_ford", "floyd_warshall", "auto"]
    return_path: bool
    return_distance: bool
    all_paths: bool


class BipartiteParams(TypedDict, total=False):
    """
    Parameters for bipartite evaluation.
    
    Attributes:
        return_partitions: Whether to return the two partitions if bipartite
        return_odd_cycle: Whether to return an odd cycle if not bipartite
        verify_partitions: Specific partitions to verify
    """
    return_partitions: bool
    return_odd_cycle: bool
    verify_partitions: list[list[str]]


class GraphMatchParams(TypedDict, total=False):
    """
    Parameters for graph matching evaluation.
    
    Attributes:
        match_type: Type of matching to perform
            - "exact": Nodes and edges must match exactly
            - "isomorphic": Graphs must be isomorphic
            - "subgraph": Student graph must be subgraph of answer
        ignore_labels: Whether to ignore node/edge labels in comparison
        ignore_weights: Whether to ignore edge weights in comparison
        tolerance: Numerical tolerance for weight comparison
        check_direction: Whether to check edge direction (for directed graphs)
    """
    match_type: Literal["exact", "isomorphic", "subgraph"]
    ignore_labels: bool
    ignore_weights: bool
    tolerance: float
    check_direction: bool


# =============================================================================
# PATH EVALUATION PARAMETERS
# =============================================================================

class EulerianParams(TypedDict, total=False):
    """
    Parameters for Eulerian path/circuit evaluation.
    
    Attributes:
        check_existence: Only check if path/circuit exists (don't find it)
        start_node: Required starting node (optional)
        end_node: Required ending node (for path, optional)
        return_path: Whether to return the actual path
    """
    check_existence: bool
    start_node: str
    end_node: str
    return_path: bool


class HamiltonianParams(TypedDict, total=False):
    """
    Parameters for Hamiltonian path/circuit evaluation.
    
    Attributes:
        check_existence: Only check if path/circuit exists
        start_node: Required starting node (optional)
        end_node: Required ending node (for path, optional)
        return_path: Whether to return the actual path
        timeout: Maximum computation time in seconds (NP-complete problem)
    """
    check_existence: bool
    start_node: str
    end_node: str
    return_path: bool
    timeout: float


# =============================================================================
# CYCLE EVALUATION PARAMETERS
# =============================================================================

class CycleDetectionParams(TypedDict, total=False):
    """
    Parameters for cycle detection evaluation.
    
    Attributes:
        find_all: Whether to find all cycles or just detect presence
        max_length: Maximum cycle length to consider
        return_cycles: Whether to return the actual cycles found
        min_length: Minimum cycle length to consider (default: 3)
    """
    find_all: bool
    max_length: int
    min_length: int
    return_cycles: bool


class NegativeCycleParams(TypedDict, total=False):
    """
    Parameters for negative cycle detection.
    
    Attributes:
        return_cycle: Whether to return the negative cycle if found
        source_node: Starting node for detection (optional)
    """
    return_cycle: bool
    source_node: str


# =============================================================================
# TREE EVALUATION PARAMETERS
# =============================================================================

class SpanningTreeParams(TypedDict, total=False):
    """
    Parameters for spanning tree evaluation.
    
    Attributes:
        check_mst: Whether to check if it's a minimum spanning tree
        expected_weight: Expected total weight of the MST
        weight_tolerance: Tolerance for weight comparison
        algorithm: Algorithm to use for MST computation
            - "kruskal": Kruskal's algorithm
            - "prim": Prim's algorithm
            - "auto": Automatic selection
    """
    check_mst: bool
    expected_weight: float
    weight_tolerance: float
    algorithm: Literal["kruskal", "prim", "auto"]


class TreeParams(TypedDict, total=False):
    """
    Parameters for tree-related evaluations.
    
    Attributes:
        root_node: Specified root node for rooted tree operations
        return_structure: Whether to return tree structure info
    """
    root_node: str
    return_structure: bool


# =============================================================================
# COLORING EVALUATION PARAMETERS
# =============================================================================

class GraphColoringParams(TypedDict, total=False):
    """
    Parameters for graph coloring evaluation.
    
    Attributes:
        num_colors: Number of colors allowed (k for k-coloring)
        verify_coloring: Specific coloring to verify
        find_optimal: Whether to find the chromatic number
        algorithm: Coloring algorithm to use
            - "greedy": Greedy coloring
            - "dsatur": DSatur algorithm
            - "backtracking": Exact backtracking
    """
    num_colors: int
    verify_coloring: dict[str, int]  # node_id -> color
    find_optimal: bool
    algorithm: Literal["greedy", "dsatur", "backtracking", "auto"]


# =============================================================================
# FLOW EVALUATION PARAMETERS
# =============================================================================

class MaxFlowParams(TypedDict, total=False):
    """
    Parameters for maximum flow evaluation.
    
    Attributes:
        source_node: Source node ID (required)
        sink_node: Sink node ID (required)
        algorithm: Flow algorithm to use
            - "ford_fulkerson": Ford-Fulkerson method
            - "edmonds_karp": Edmonds-Karp (BFS-based)
            - "dinic": Dinic's algorithm
        return_flow_assignment: Whether to return flow on each edge
        return_min_cut: Whether to also return the minimum cut
    """
    source_node: str  # Required
    sink_node: str  # Required
    algorithm: Literal["ford_fulkerson", "edmonds_karp", "dinic", "auto"]
    return_flow_assignment: bool
    return_min_cut: bool


class BipartiteMatchingParams(TypedDict, total=False):
    """
    Parameters for bipartite matching evaluation.
    
    Attributes:
        left_partition: Nodes in left partition (optional, auto-detect if not provided)
        right_partition: Nodes in right partition (optional)
        weighted: Whether to find maximum weight matching
        perfect: Whether to check for perfect matching
        return_matching: Whether to return the matching edges
    """
    left_partition: list[str]
    right_partition: list[str]
    weighted: bool
    perfect: bool
    return_matching: bool


# =============================================================================
# COMPONENT EVALUATION PARAMETERS
# =============================================================================

class ComponentParams(TypedDict, total=False):
    """
    Parameters for component-related evaluations.
    
    Attributes:
        return_components: Whether to return the actual components
        min_size: Minimum component size to report
    """
    return_components: bool
    min_size: int


class ArticulationParams(TypedDict, total=False):
    """
    Parameters for articulation points and bridges evaluation.
    
    Attributes:
        return_points: Whether to return articulation points
        return_bridges: Whether to return bridges
        return_biconnected: Whether to return biconnected components
    """
    return_points: bool
    return_bridges: bool
    return_biconnected: bool


# =============================================================================
# STRUCTURE EVALUATION PARAMETERS
# =============================================================================

class DegreeSequenceParams(TypedDict, total=False):
    """
    Parameters for degree sequence evaluation.
    
    Attributes:
        expected_sequence: Expected degree sequence (sorted)
        allow_permutation: Whether sequence can be in any order
        check_graphical: Whether to check if sequence is graphical
    """
    expected_sequence: list[int]
    allow_permutation: bool
    check_graphical: bool


class CliqueParams(TypedDict, total=False):
    """
    Parameters for clique evaluation.
    
    Attributes:
        min_size: Minimum clique size to find
        max_clique: Whether to find maximum clique
        all_cliques: Whether to find all maximal cliques
        verify_clique: Specific vertex set to verify as clique
        timeout: Maximum computation time (NP-complete)
    """
    min_size: int
    max_clique: bool
    all_cliques: bool
    verify_clique: list[str]
    timeout: float


class IndependentSetParams(TypedDict, total=False):
    """
    Parameters for independent set evaluation.
    
    Attributes:
        min_size: Minimum set size to find
        max_set: Whether to find maximum independent set
        verify_set: Specific vertex set to verify as independent
        timeout: Maximum computation time (NP-complete)
    """
    min_size: int
    max_set: bool
    verify_set: list[str]
    timeout: float


class VertexCoverParams(TypedDict, total=False):
    """
    Parameters for vertex cover evaluation.
    
    Attributes:
        max_size: Maximum cover size allowed
        min_cover: Whether to find minimum vertex cover
        verify_cover: Specific vertex set to verify as cover
        approximation: Use 2-approximation algorithm
        timeout: Maximum computation time (NP-complete)
    """
    max_size: int
    min_cover: bool
    verify_cover: list[str]
    approximation: bool
    timeout: float


# =============================================================================
# ORDERING EVALUATION PARAMETERS
# =============================================================================

class TopologicalSortParams(TypedDict, total=False):
    """
    Parameters for topological sort evaluation.
    
    Attributes:
        verify_order: Specific ordering to verify
        return_order: Whether to return a valid ordering
        all_orderings: Whether to return all valid orderings (warning: exponential)
    """
    verify_order: list[str]
    return_order: bool
    all_orderings: bool


class TraversalParams(TypedDict, total=False):
    """
    Parameters for DFS/BFS traversal evaluation.
    
    Attributes:
        start_node: Starting node for traversal (required)
        verify_order: Specific order to verify
        return_order: Whether to return the traversal order
        return_tree: Whether to return the traversal tree/forest
    """
    start_node: str  # Required
    verify_order: list[str]
    return_order: bool
    return_tree: bool


# =============================================================================
# MAIN EVALUATION PARAMS (COMBINED)
# =============================================================================

class EvaluationParams(TypedDict, total=False):
    """
    Combined parameters schema for the evaluation function.
    
    This is the `params` argument passed to evaluation_function().
    
    Attributes:
        evaluation_type: The type of evaluation to perform (required)
        [type-specific params]: Parameters for the specific evaluation
        partial_credit: Whether to award partial credit
        feedback_level: Level of detail in feedback
        timeout: Global timeout for computation
    """
    # Required
    evaluation_type: str
    
    # Core evaluation params
    connectivity: ConnectivityParams
    shortest_path: ShortestPathParams
    bipartite: BipartiteParams
    graph_match: GraphMatchParams
    
    # Path params
    eulerian: EulerianParams
    hamiltonian: HamiltonianParams
    
    # Cycle params
    cycle_detection: CycleDetectionParams
    negative_cycle: NegativeCycleParams
    
    # Tree params
    spanning_tree: SpanningTreeParams
    tree: TreeParams
    
    # Coloring params
    graph_coloring: GraphColoringParams
    
    # Flow params
    max_flow: MaxFlowParams
    bipartite_matching: BipartiteMatchingParams
    
    # Component params
    components: ComponentParams
    articulation: ArticulationParams
    
    # Structure params
    degree_sequence: DegreeSequenceParams
    clique: CliqueParams
    independent_set: IndependentSetParams
    vertex_cover: VertexCoverParams
    
    # Ordering params
    topological_sort: TopologicalSortParams
    traversal: TraversalParams
    
    # Global params
    partial_credit: bool
    feedback_level: Literal["minimal", "standard", "detailed"]
    timeout: float
