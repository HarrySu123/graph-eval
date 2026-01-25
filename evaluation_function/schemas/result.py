"""
Result Schemas

Defines the schemas for evaluation results and feedback.
"""

from typing import TypedDict, Literal
from .graph import EdgeSchema


# =============================================================================
# SPECIFIC RESULT SCHEMAS
# =============================================================================

class PathResult(TypedDict, total=False):
    """Result details for shortest path evaluation."""
    path: list[str]
    distance: float
    path_exists: bool
    algorithm_used: str
    all_paths: list[list[str]]  # If multiple shortest paths exist


class ConnectivityResult(TypedDict, total=False):
    """Result details for connectivity evaluation."""
    is_connected: bool
    num_components: int
    components: list[list[str]]
    connectivity_type: str  # "connected", "strongly_connected", etc.
    largest_component_size: int


class BipartiteResult(TypedDict, total=False):
    """Result details for bipartite evaluation."""
    is_bipartite: bool
    partitions: list[list[str]]
    odd_cycle: list[str]  # Proof if not bipartite


class GraphMatchResult(TypedDict, total=False):
    """Result details for graph matching evaluation."""
    is_match: bool
    match_type: str
    missing_nodes: list[str]
    extra_nodes: list[str]
    missing_edges: list[EdgeSchema]
    extra_edges: list[EdgeSchema]
    node_mapping: dict[str, str]  # For isomorphism


class EulerianResult(TypedDict, total=False):
    """Result details for Eulerian path/circuit evaluation."""
    exists: bool
    path: list[str]
    is_circuit: bool
    odd_degree_vertices: list[str]  # Vertices with odd degree


class HamiltonianResult(TypedDict, total=False):
    """Result details for Hamiltonian path/circuit evaluation."""
    exists: bool
    path: list[str]
    is_circuit: bool
    timed_out: bool


class CycleResult(TypedDict, total=False):
    """Result details for cycle detection evaluation."""
    has_cycle: bool
    cycles: list[list[str]]
    shortest_cycle: list[str]
    girth: int  # Length of shortest cycle
    has_negative_cycle: bool
    negative_cycle: list[str]


class TreeResult(TypedDict, total=False):
    """Result details for tree evaluations."""
    is_tree: bool
    is_spanning_tree: bool
    is_mst: bool
    total_weight: float
    edges: list[EdgeSchema]
    diameter: int
    center: list[str]
    root: str


class ColoringResult(TypedDict, total=False):
    """Result details for graph coloring evaluation."""
    is_valid_coloring: bool
    coloring: dict[str, int]
    num_colors_used: int
    chromatic_number: int
    conflicts: list[tuple[str, str]]  # Edges with same-color endpoints


class FlowResult(TypedDict, total=False):
    """Result details for flow network evaluation."""
    max_flow_value: float
    flow_assignment: dict[str, float]
    min_cut_nodes: list[str]
    min_cut_edges: list[EdgeSchema]
    min_cut_capacity: float
    is_valid_flow: bool


class MatchingResult(TypedDict, total=False):
    """Result details for matching evaluation."""
    matching: list[tuple[str, str]]
    matching_size: int
    is_perfect: bool
    is_maximum: bool
    unmatched_vertices: list[str]
    total_weight: float  # For weighted matching


class ComponentResult(TypedDict, total=False):
    """Result details for component-related evaluations."""
    components: list[list[str]]
    num_components: int
    articulation_points: list[str]
    bridges: list[EdgeSchema]
    biconnected_components: list[list[str]]


class StructureResult(TypedDict, total=False):
    """Result details for graph structure evaluations."""
    degree_sequence: list[int]
    is_regular: bool
    regularity: int
    is_planar: bool
    is_complete: bool
    clique: list[str]
    clique_size: int
    independent_set: list[str]
    independent_set_size: int
    vertex_cover: list[str]
    vertex_cover_size: int


class OrderingResult(TypedDict, total=False):
    """Result details for ordering evaluations."""
    is_valid_order: bool
    order: list[str]
    all_valid_orderings: list[list[str]]
    traversal_tree: dict[str, list[str]]  # parent -> children


# =============================================================================
# FEEDBACK SCHEMA
# =============================================================================

class FeedbackItem(TypedDict, total=False):
    """A single feedback item."""
    type: Literal["success", "error", "warning", "info", "hint"]
    message: str
    details: str
    location: str  # Reference to specific node/edge if applicable


class ComputationStep(TypedDict, total=False):
    """A step in the computation (for detailed feedback)."""
    step_number: int
    description: str
    state: dict  # Current state at this step
    highlight_nodes: list[str]
    highlight_edges: list[EdgeSchema]


# =============================================================================
# MAIN EVALUATION RESULT SCHEMA
# =============================================================================

class EvaluationDetails(TypedDict, total=False):
    """
    Detailed evaluation results.
    
    Contains type-specific results and feedback information.
    """
    # Type-specific results
    path_result: PathResult
    connectivity_result: ConnectivityResult
    bipartite_result: BipartiteResult
    graph_match_result: GraphMatchResult
    eulerian_result: EulerianResult
    hamiltonian_result: HamiltonianResult
    cycle_result: CycleResult
    tree_result: TreeResult
    coloring_result: ColoringResult
    flow_result: FlowResult
    matching_result: MatchingResult
    component_result: ComponentResult
    structure_result: StructureResult
    ordering_result: OrderingResult
    
    # Computed values (for display/verification)
    computed_answer: dict
    expected_answer: dict
    
    # Feedback
    feedback_items: list[FeedbackItem]
    computation_steps: list[ComputationStep]
    hints: list[str]
    
    # Scoring
    partial_score: float  # 0.0 to 1.0
    scoring_breakdown: dict[str, float]


# =============================================================================
# VISUALIZATION DATA
# =============================================================================

class VisualizationData(TypedDict, total=False):
    """
    Data for visualizing the result in the UI.
    
    Attributes:
        highlight_nodes: Nodes to highlight (e.g., path nodes)
        highlight_edges: Edges to highlight
        node_colors: Color assignments for nodes
        edge_colors: Color assignments for edges
        node_labels: Additional labels to show on nodes
        edge_labels: Additional labels to show on edges
        animation_steps: Steps for animated visualization
    """
    highlight_nodes: list[str]
    highlight_edges: list[EdgeSchema]
    node_colors: dict[str, str]
    edge_colors: dict[str, str]
    node_labels: dict[str, str]
    edge_labels: dict[str, str]
    animation_steps: list[dict]
    partitions: list[list[str]]  # For bipartite visualization
    tree_layout: dict[str, tuple[float, float]]  # Computed tree positions


# =============================================================================
# COMPLETE RESULT (extends lf_toolkit Result)
# =============================================================================

class ExtendedResult(TypedDict, total=False):
    """
    Extended result schema for graph evaluation.
    
    This extends the base Result from lf_toolkit with
    graph-specific evaluation details.
    
    Note: The actual return should use lf_toolkit.evaluation.Result,
    with these additional fields in a custom property.
    """
    is_correct: bool
    feedback: str
    
    # Extended graph evaluation data
    evaluation_details: EvaluationDetails
    visualization: VisualizationData
    
    # Warnings and errors
    warnings: list[str]
    errors: list[str]
