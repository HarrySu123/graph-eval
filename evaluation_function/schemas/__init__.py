"""
Graph Theory Evaluation Schemas

This package contains all Pydantic schema definitions for the Graph Theory
Visualizer and Evaluator.

Modules:
- graph: Core graph structure schemas (Node, Edge, Graph)
- evaluation_types: Evaluation type definitions and metadata
- params: Evaluation parameter schemas
- request: Request/Response schemas
- result: Result and feedback schemas
"""

# Graph structures
from .graph import (
    Node,
    Edge,
    Graph,
    EXAMPLE_UNDIRECTED_GRAPH,
    EXAMPLE_DIRECTED_GRAPH,
    EXAMPLE_BIPARTITE_GRAPH,
    EXAMPLE_FLOW_NETWORK,
)

# Evaluation types
from .evaluation_types import (
    EvaluationType,
    EvaluationCategory,
    CoreEvaluation,
    PathEvaluation,
    CycleEvaluation,
    TreeEvaluation,
    ColoringEvaluation,
    FlowEvaluation,
    ComponentEvaluation,
    StructureEvaluation,
    OrderingEvaluation,
    SpecialGraphEvaluation,
    EvaluationTypeInfo,
    EVALUATION_TYPE_INFO,
)

# Parameters
from .params import (
    EvaluationParams,
    ConnectivityParams,
    ShortestPathParams,
    BipartiteParams,
    GraphMatchParams,
    EulerianParams,
    HamiltonianParams,
    CycleDetectionParams,
    NegativeCycleParams,
    SpanningTreeParams,
    TreeParams,
    GraphColoringParams,
    MaxFlowParams,
    BipartiteMatchingParams,
    ComponentParams,
    ArticulationParams,
    DegreeSequenceParams,
    CliqueParams,
    IndependentSetParams,
    VertexCoverParams,
    TopologicalSortParams,
    TraversalParams,
)

# Request/Response
from .request import (
    Response,
    Answer,
)

# Results
from .result import (
    PathResult,
    ConnectivityResult,
    BipartiteResult,
    GraphMatchResult,
    EulerianResult,
    HamiltonianResult,
    CycleResult,
    TreeResult,
    ColoringResult,
    FlowResult,
    MatchingResult,
    ComponentResult,
    StructureResult,
    OrderingResult,
    FeedbackItem,
    ComputationStep,
    EvaluationDetails,
    VisualizationData,
    EvaluationResult,
)

__all__ = [
    # Graph structures
    "Node",
    "Edge",
    "Graph",
    "EXAMPLE_UNDIRECTED_GRAPH",
    "EXAMPLE_DIRECTED_GRAPH",
    "EXAMPLE_BIPARTITE_GRAPH",
    "EXAMPLE_FLOW_NETWORK",
    
    # Evaluation types
    "EvaluationType",
    "EvaluationCategory",
    "CoreEvaluation",
    "PathEvaluation",
    "CycleEvaluation",
    "TreeEvaluation",
    "ColoringEvaluation",
    "FlowEvaluation",
    "ComponentEvaluation",
    "StructureEvaluation",
    "OrderingEvaluation",
    "SpecialGraphEvaluation",
    "EvaluationTypeInfo",
    "EVALUATION_TYPE_INFO",
    
    # Parameters
    "EvaluationParams",
    "ConnectivityParams",
    "ShortestPathParams",
    "BipartiteParams",
    "GraphMatchParams",
    "EulerianParams",
    "HamiltonianParams",
    "CycleDetectionParams",
    "NegativeCycleParams",
    "SpanningTreeParams",
    "TreeParams",
    "GraphColoringParams",
    "MaxFlowParams",
    "BipartiteMatchingParams",
    "ComponentParams",
    "ArticulationParams",
    "DegreeSequenceParams",
    "CliqueParams",
    "IndependentSetParams",
    "VertexCoverParams",
    "TopologicalSortParams",
    "TraversalParams",
    
    # Request/Response
    "Response",
    "Answer",
    
    # Results
    "PathResult",
    "ConnectivityResult",
    "BipartiteResult",
    "GraphMatchResult",
    "EulerianResult",
    "HamiltonianResult",
    "CycleResult",
    "TreeResult",
    "ColoringResult",
    "FlowResult",
    "MatchingResult",
    "ComponentResult",
    "StructureResult",
    "OrderingResult",
    "FeedbackItem",
    "ComputationStep",
    "EvaluationDetails",
    "VisualizationData",
    "EvaluationResult",
]
