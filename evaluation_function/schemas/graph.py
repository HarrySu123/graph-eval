"""
Graph Structure Schemas

Defines the core data structures for representing graphs:
- Nodes (vertices)
- Edges
- Complete graph structures
"""

from typing import TypedDict, Any


class NodeSchema(TypedDict, total=False):
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
    id: str  # Required
    label: str
    x: float
    y: float
    partition: int
    color: int
    weight: float
    metadata: dict[str, Any]


class EdgeSchema(TypedDict, total=False):
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
    source: str  # Required
    target: str  # Required
    weight: float
    capacity: float
    flow: float
    label: str
    id: str
    color: str
    metadata: dict[str, Any]


class GraphSchema(TypedDict, total=False):
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
    nodes: list[NodeSchema]  # Required
    edges: list[EdgeSchema]  # Required
    directed: bool
    weighted: bool
    multigraph: bool
    name: str
    metadata: dict[str, Any]


# =============================================================================
# EXAMPLE GRAPHS
# =============================================================================

EXAMPLE_UNDIRECTED_GRAPH: GraphSchema = {
    "nodes": [
        {"id": "A", "label": "Node A", "x": 0, "y": 0},
        {"id": "B", "label": "Node B", "x": 100, "y": 0},
        {"id": "C", "label": "Node C", "x": 50, "y": 100},
    ],
    "edges": [
        {"source": "A", "target": "B", "weight": 1},
        {"source": "B", "target": "C", "weight": 2},
        {"source": "A", "target": "C", "weight": 3},
    ],
    "directed": False,
    "weighted": True,
    "name": "Example Triangle Graph"
}

EXAMPLE_DIRECTED_GRAPH: GraphSchema = {
    "nodes": [
        {"id": "1", "label": "Start"},
        {"id": "2", "label": "Middle"},
        {"id": "3", "label": "End"},
    ],
    "edges": [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "1", "target": "3"},
    ],
    "directed": True,
    "weighted": False,
    "name": "Example DAG"
}

EXAMPLE_BIPARTITE_GRAPH: GraphSchema = {
    "nodes": [
        {"id": "A", "partition": 0},
        {"id": "B", "partition": 0},
        {"id": "X", "partition": 1},
        {"id": "Y", "partition": 1},
    ],
    "edges": [
        {"source": "A", "target": "X"},
        {"source": "A", "target": "Y"},
        {"source": "B", "target": "X"},
    ],
    "directed": False,
    "weighted": False,
    "name": "Example Bipartite Graph"
}

EXAMPLE_FLOW_NETWORK: GraphSchema = {
    "nodes": [
        {"id": "s", "label": "Source"},
        {"id": "a"},
        {"id": "b"},
        {"id": "t", "label": "Sink"},
    ],
    "edges": [
        {"source": "s", "target": "a", "capacity": 10},
        {"source": "s", "target": "b", "capacity": 5},
        {"source": "a", "target": "b", "capacity": 15},
        {"source": "a", "target": "t", "capacity": 10},
        {"source": "b", "target": "t", "capacity": 10},
    ],
    "directed": True,
    "weighted": False,
    "name": "Example Flow Network"
}
