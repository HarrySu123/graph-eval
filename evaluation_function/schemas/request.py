"""
Request/Response Schemas

Defines the schemas for student responses and expected answers.
"""

from typing import TypedDict, Union
from .graph import GraphSchema, EdgeSchema


# =============================================================================
# STUDENT RESPONSE SCHEMA
# =============================================================================

class ResponseSchema(TypedDict, total=False):
    """
    Schema for student response.
    
    This is the `response` argument passed to evaluation_function().
    The response contains the student's submitted graph and/or answer.
    
    Students can submit:
    1. A graph they constructed/modified
    2. A specific answer (path, boolean, number, etc.)
    3. Both a graph and an answer
    
    Attributes:
        graph: The student's constructed/modified graph
        
        # Boolean answers
        is_connected: Answer for connectivity question
        is_bipartite: Answer for bipartite question
        is_tree: Answer for tree question
        is_planar: Answer for planarity question
        has_cycle: Answer for cycle existence question
        has_eulerian_path: Answer for Eulerian path existence
        has_hamiltonian_path: Answer for Hamiltonian path existence
        
        # Path/sequence answers
        path: A path (sequence of node IDs)
        cycle: A cycle (sequence of node IDs)
        ordering: An ordering (topological, DFS, BFS)
        
        # Numeric answers
        distance: Shortest distance answer
        flow_value: Maximum flow value
        chromatic_number: Chromatic number answer
        num_components: Number of connected components
        mst_weight: Total MST weight
        
        # Set answers
        partitions: Two-set partition for bipartite graphs
        components: Connected components (list of node lists)
        coloring: Graph coloring (node_id -> color mapping)
        matching: Matching edges
        vertex_cover: Vertex cover set
        independent_set: Independent set
        clique: Clique vertices
        articulation_points: Articulation points
        bridges: Bridge edges
        
        # Tree answers
        spanning_tree: Edges forming spanning tree
        mst: Minimum spanning tree edges
    """
    # Graph submission
    graph: GraphSchema
    
    # Boolean answers
    is_connected: bool
    is_bipartite: bool
    is_tree: bool
    is_planar: bool
    has_cycle: bool
    has_eulerian_path: bool
    has_eulerian_circuit: bool
    has_hamiltonian_path: bool
    has_hamiltonian_circuit: bool
    is_dag: bool
    
    # Path/sequence answers
    path: list[str]
    cycle: list[str]
    eulerian_path: list[str]
    hamiltonian_path: list[str]
    ordering: list[str]
    dfs_order: list[str]
    bfs_order: list[str]
    topological_order: list[str]
    
    # Numeric answers
    distance: float
    flow_value: float
    chromatic_number: int
    num_components: int
    mst_weight: float
    diameter: int
    girth: int  # Shortest cycle length
    
    # Set/partition answers
    partitions: list[list[str]]
    components: list[list[str]]
    strongly_connected_components: list[list[str]]
    coloring: dict[str, int]
    matching: list[tuple[str, str]]
    vertex_cover: list[str]
    independent_set: list[str]
    clique: list[str]
    dominating_set: list[str]
    articulation_points: list[str]
    bridges: list[EdgeSchema]
    
    # Tree structure answers
    spanning_tree: list[EdgeSchema]
    mst: list[EdgeSchema]
    tree_center: list[str]
    
    # Flow network answers
    flow_assignment: dict[str, float]  # edge_id -> flow
    min_cut: list[str]  # Nodes on source side of cut
    
    # Degree sequence
    degree_sequence: list[int]


# =============================================================================
# EXPECTED ANSWER SCHEMA
# =============================================================================

class AnswerSchema(TypedDict, total=False):
    """
    Schema for correct answer.
    
    This is the `answer` argument passed to evaluation_function().
    The answer contains the expected correct result.
    
    Attributes mirror ResponseSchema for comparison.
    Additional attributes allow for flexible answer specification.
    
    Attributes:
        graph: The expected/correct graph (for graph building questions)
        
        # All response fields are valid as answers
        [All fields from ResponseSchema]
        
        # Additional answer specification
        multiple_valid: Whether multiple answers are acceptable
        valid_answers: List of all valid answers (for non-unique solutions)
        answer_range: Acceptable range for numeric answers
        tolerance: Numerical tolerance for comparisons
    """
    # Graph answer
    graph: GraphSchema
    
    # Boolean answers
    is_connected: bool
    is_bipartite: bool
    is_tree: bool
    is_planar: bool
    has_cycle: bool
    has_eulerian_path: bool
    has_eulerian_circuit: bool
    has_hamiltonian_path: bool
    has_hamiltonian_circuit: bool
    is_dag: bool
    
    # Path/sequence answers
    path: list[str]
    shortest_path: list[str]
    cycle: list[str]
    eulerian_path: list[str]
    hamiltonian_path: list[str]
    ordering: list[str]
    topological_order: list[str]
    
    # Numeric answers
    distance: float
    shortest_distance: float
    flow_value: float
    max_flow: float
    chromatic_number: int
    num_components: int
    mst_weight: float
    diameter: int
    girth: int
    
    # Set/partition answers
    partitions: list[list[str]]
    components: list[list[str]]
    strongly_connected_components: list[list[str]]
    coloring: dict[str, int]
    matching: list[tuple[str, str]]
    max_matching_size: int
    vertex_cover: list[str]
    min_vertex_cover_size: int
    independent_set: list[str]
    max_independent_set_size: int
    clique: list[str]
    max_clique_size: int
    articulation_points: list[str]
    bridges: list[EdgeSchema]
    
    # Tree structure answers
    spanning_tree: list[EdgeSchema]
    mst: list[EdgeSchema]
    tree_center: list[str]
    
    # Flow network answers
    flow_assignment: dict[str, float]
    min_cut: list[str]
    min_cut_capacity: float
    
    # Degree sequence
    degree_sequence: list[int]
    
    # Answer flexibility
    multiple_valid: bool
    valid_answers: list[dict]  # List of valid answer configurations
    answer_range: tuple[float, float]  # (min, max) for numeric
    tolerance: float  # Numerical comparison tolerance
    
    # Custom answer for flexibility
    custom: Union[dict, list, str, int, float, bool]
