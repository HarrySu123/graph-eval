"""
Request/Response Schemas

Defines the schemas for student responses and expected answers using Pydantic.
"""

from typing import Optional, Union, Any
from pydantic import BaseModel, Field

from .graph import Graph, Edge


# =============================================================================
# STUDENT RESPONSE SCHEMA
# =============================================================================

class Response(BaseModel):
    """
    Schema for student response.
    
    This is the `response` argument passed to evaluation_function().
    The response contains the student's submitted graph and/or answer.
    
    Students can submit:
    1. A graph they constructed/modified
    2. A specific answer (path, boolean, number, etc.)
    3. Both a graph and an answer
    """
    # Graph submission
    graph: Optional[Graph] = Field(None, description="The student's constructed/modified graph")
    
    # Boolean answers
    is_connected: Optional[bool] = Field(None, description="Answer for connectivity question")
    is_bipartite: Optional[bool] = Field(None, description="Answer for bipartite question")
    is_tree: Optional[bool] = Field(None, description="Answer for tree question")
    is_planar: Optional[bool] = Field(None, description="Answer for planarity question")
    has_cycle: Optional[bool] = Field(None, description="Answer for cycle existence question")
    has_eulerian_path: Optional[bool] = Field(None, description="Answer for Eulerian path existence")
    has_eulerian_circuit: Optional[bool] = Field(None, description="Answer for Eulerian circuit existence")
    has_hamiltonian_path: Optional[bool] = Field(None, description="Answer for Hamiltonian path existence")
    has_hamiltonian_circuit: Optional[bool] = Field(None, description="Answer for Hamiltonian circuit existence")
    is_dag: Optional[bool] = Field(None, description="Answer for DAG question")
    
    # Path/sequence answers
    path: Optional[list[str]] = Field(None, description="A path (sequence of node IDs)")
    cycle: Optional[list[str]] = Field(None, description="A cycle (sequence of node IDs)")
    eulerian_path: Optional[list[str]] = Field(None, description="Eulerian path")
    hamiltonian_path: Optional[list[str]] = Field(None, description="Hamiltonian path")
    ordering: Optional[list[str]] = Field(None, description="An ordering")
    dfs_order: Optional[list[str]] = Field(None, description="DFS traversal order")
    bfs_order: Optional[list[str]] = Field(None, description="BFS traversal order")
    topological_order: Optional[list[str]] = Field(None, description="Topological ordering")
    
    # Numeric answers
    distance: Optional[float] = Field(None, description="Shortest distance answer")
    flow_value: Optional[float] = Field(None, description="Maximum flow value")
    chromatic_number: Optional[int] = Field(None, description="Chromatic number answer")
    num_components: Optional[int] = Field(None, description="Number of connected components")
    mst_weight: Optional[float] = Field(None, description="Total MST weight")
    diameter: Optional[int] = Field(None, description="Graph/tree diameter")
    girth: Optional[int] = Field(None, description="Shortest cycle length")
    
    # Set/partition answers
    partitions: Optional[list[list[str]]] = Field(None, description="Two-set partition for bipartite graphs")
    components: Optional[list[list[str]]] = Field(None, description="Connected components")
    strongly_connected_components: Optional[list[list[str]]] = Field(None, description="SCCs")
    coloring: Optional[dict[str, int]] = Field(None, description="Graph coloring (node_id -> color)")
    matching: Optional[list[tuple[str, str]]] = Field(None, description="Matching edges")
    vertex_cover: Optional[list[str]] = Field(None, description="Vertex cover set")
    independent_set: Optional[list[str]] = Field(None, description="Independent set")
    clique: Optional[list[str]] = Field(None, description="Clique vertices")
    dominating_set: Optional[list[str]] = Field(None, description="Dominating set")
    articulation_points: Optional[list[str]] = Field(None, description="Articulation points")
    bridges: Optional[list[Edge]] = Field(None, description="Bridge edges")
    
    # Tree structure answers
    spanning_tree: Optional[list[Edge]] = Field(None, description="Edges forming spanning tree")
    mst: Optional[list[Edge]] = Field(None, description="Minimum spanning tree edges")
    tree_center: Optional[list[str]] = Field(None, description="Tree center node(s)")
    
    # Flow network answers
    flow_assignment: Optional[dict[str, float]] = Field(None, description="Edge flow assignment")
    min_cut: Optional[list[str]] = Field(None, description="Nodes on source side of cut")
    
    # Degree sequence
    degree_sequence: Optional[list[int]] = Field(None, description="Degree sequence")

    class Config:
        extra = "allow"


# =============================================================================
# EXPECTED ANSWER SCHEMA
# =============================================================================

class Answer(BaseModel):
    """
    Schema for correct answer.
    
    This is the `answer` argument passed to evaluation_function().
    The answer contains the expected correct result.
    """
    # Graph answer
    graph: Optional[Graph] = Field(None, description="The expected/correct graph")
    
    # Boolean answers
    is_connected: Optional[bool] = None
    is_bipartite: Optional[bool] = None
    is_tree: Optional[bool] = None
    is_planar: Optional[bool] = None
    has_cycle: Optional[bool] = None
    has_eulerian_path: Optional[bool] = None
    has_eulerian_circuit: Optional[bool] = None
    has_hamiltonian_path: Optional[bool] = None
    has_hamiltonian_circuit: Optional[bool] = None
    is_dag: Optional[bool] = None
    
    # Path/sequence answers
    path: Optional[list[str]] = None
    shortest_path: Optional[list[str]] = None
    cycle: Optional[list[str]] = None
    eulerian_path: Optional[list[str]] = None
    hamiltonian_path: Optional[list[str]] = None
    ordering: Optional[list[str]] = None
    topological_order: Optional[list[str]] = None
    
    # Numeric answers
    distance: Optional[float] = None
    shortest_distance: Optional[float] = None
    flow_value: Optional[float] = None
    max_flow: Optional[float] = None
    chromatic_number: Optional[int] = None
    num_components: Optional[int] = None
    mst_weight: Optional[float] = None
    diameter: Optional[int] = None
    girth: Optional[int] = None
    
    # Set/partition answers
    partitions: Optional[list[list[str]]] = None
    components: Optional[list[list[str]]] = None
    strongly_connected_components: Optional[list[list[str]]] = None
    coloring: Optional[dict[str, int]] = None
    matching: Optional[list[tuple[str, str]]] = None
    max_matching_size: Optional[int] = None
    vertex_cover: Optional[list[str]] = None
    min_vertex_cover_size: Optional[int] = None
    independent_set: Optional[list[str]] = None
    max_independent_set_size: Optional[int] = None
    clique: Optional[list[str]] = None
    max_clique_size: Optional[int] = None
    articulation_points: Optional[list[str]] = None
    bridges: Optional[list[Edge]] = None
    
    # Tree structure answers
    spanning_tree: Optional[list[Edge]] = None
    mst: Optional[list[Edge]] = None
    tree_center: Optional[list[str]] = None
    
    # Flow network answers
    flow_assignment: Optional[dict[str, float]] = None
    min_cut: Optional[list[str]] = None
    min_cut_capacity: Optional[float] = None
    
    # Degree sequence
    degree_sequence: Optional[list[int]] = None
    
    # Answer flexibility
    multiple_valid: bool = Field(
        False,
        description="Whether multiple answers are acceptable"
    )
    valid_answers: Optional[list[dict[str, Any]]] = Field(
        None,
        description="List of valid answer configurations"
    )
    answer_range: Optional[tuple[float, float]] = Field(
        None,
        description="Acceptable range for numeric answers (min, max)"
    )
    tolerance: float = Field(
        1e-9,
        description="Numerical comparison tolerance"
    )
    
    # Custom answer for flexibility
    custom: Optional[Union[dict, list, str, int, float, bool]] = Field(
        None,
        description="Custom answer for flexible evaluation"
    )

    class Config:
        extra = "allow"
