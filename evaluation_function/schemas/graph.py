"""
Graph Structure Schemas

Defines the core data structures for representing graphs:
- Nodes (vertices)
- Edges
- Complete graph structures
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class Node(BaseModel):
    """
    Schema for a graph node/vertex.
    
    Attributes:
        id: Unique identifier for the node (required)
        label: Display label for the node (optional, defaults to id)
        x: X coordinate for visualization (optional)
        y: Y coordinate for visualization (optional)
        partition: For bipartite graphs, indicates which partition (0 or 1)
        color: Node color for graph coloring problems
        weight: Node weight for weighted vertex problems
        metadata: Additional custom data
    """
    id: str = Field(..., description="Unique identifier for the node")
    label: Optional[str] = Field(None, description="Display label for the node")
    x: Optional[float] = Field(None, description="X coordinate for visualization")
    y: Optional[float] = Field(None, description="Y coordinate for visualization")
    partition: Optional[int] = Field(None, description="Partition index for bipartite graphs (0 or 1)")
    color: Optional[int] = Field(None, description="Color index for graph coloring")
    weight: Optional[float] = Field(None, description="Node weight for weighted vertex problems")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Additional custom data")

    class Config:
        extra = "allow"


class Edge(BaseModel):
    """
    Schema for a graph edge.
    
    Attributes:
        source: ID of the source node (required)
        target: ID of the target node (required)
        weight: Edge weight for weighted graphs (optional, defaults to 1)
        capacity: Edge capacity for flow networks (optional)
        flow: Current flow value for flow networks (optional)
        label: Display label for the edge (optional)
        id: Unique edge identifier (optional, auto-generated if not provided)
        color: Edge color for visualization
        metadata: Additional custom data
    """
    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    weight: Optional[float] = Field(1.0, description="Edge weight for weighted graphs")
    capacity: Optional[float] = Field(None, description="Edge capacity for flow networks")
    flow: Optional[float] = Field(None, description="Current flow value for flow networks")
    label: Optional[str] = Field(None, description="Display label for the edge")
    id: Optional[str] = Field(None, description="Unique edge identifier")
    color: Optional[str] = Field(None, description="Edge color for visualization")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Additional custom data")

    class Config:
        extra = "allow"


class Graph(BaseModel):
    """
    Schema for representing a complete graph.
    
    Attributes:
        nodes: List of nodes in the graph (required)
        edges: List of edges in the graph (required)
        directed: Whether the graph is directed (default: False)
        weighted: Whether the graph is weighted (default: False)
        multigraph: Whether multiple edges between same nodes allowed (default: False)
        name: Optional name/identifier for the graph
        metadata: Additional custom graph properties
    """
    nodes: list[Node] = Field(..., description="List of nodes in the graph")
    edges: list[Edge] = Field(default_factory=list, description="List of edges in the graph")
    directed: bool = Field(False, description="Whether the graph is directed")
    weighted: bool = Field(False, description="Whether the graph is weighted")
    multigraph: bool = Field(False, description="Whether multiple edges between same nodes are allowed")
    name: Optional[str] = Field(None, description="Name/identifier for the graph")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Additional custom properties")

    class Config:
        extra = "allow"


# =============================================================================
# EXAMPLE GRAPHS
# =============================================================================

EXAMPLE_UNDIRECTED_GRAPH = Graph(
    nodes=[
        Node(id="A", label="Node A", x=0, y=0),
        Node(id="B", label="Node B", x=100, y=0),
        Node(id="C", label="Node C", x=50, y=100),
    ],
    edges=[
        Edge(source="A", target="B", weight=1),
        Edge(source="B", target="C", weight=2),
        Edge(source="A", target="C", weight=3),
    ],
    directed=False,
    weighted=True,
    name="Example Triangle Graph"
)

EXAMPLE_DIRECTED_GRAPH = Graph(
    nodes=[
        Node(id="1", label="Start"),
        Node(id="2", label="Middle"),
        Node(id="3", label="End"),
    ],
    edges=[
        Edge(source="1", target="2"),
        Edge(source="2", target="3"),
        Edge(source="1", target="3"),
    ],
    directed=True,
    weighted=False,
    name="Example DAG"
)

EXAMPLE_BIPARTITE_GRAPH = Graph(
    nodes=[
        Node(id="A", partition=0),
        Node(id="B", partition=0),
        Node(id="X", partition=1),
        Node(id="Y", partition=1),
    ],
    edges=[
        Edge(source="A", target="X"),
        Edge(source="A", target="Y"),
        Edge(source="B", target="X"),
    ],
    directed=False,
    weighted=False,
    name="Example Bipartite Graph"
)

EXAMPLE_FLOW_NETWORK = Graph(
    nodes=[
        Node(id="s", label="Source"),
        Node(id="a"),
        Node(id="b"),
        Node(id="t", label="Sink"),
    ],
    edges=[
        Edge(source="s", target="a", capacity=10),
        Edge(source="s", target="b", capacity=5),
        Edge(source="a", target="b", capacity=15),
        Edge(source="a", target="t", capacity=10),
        Edge(source="b", target="t", capacity=10),
    ],
    directed=True,
    weighted=False,
    name="Example Flow Network"
)
