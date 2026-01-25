"""
Graph Theory Evaluation Schemas

This package contains all schema definitions for the Graph Theory
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
    NodeSchema,
    EdgeSchema,
    GraphSchema,
    EXAMPLE_UNDIRECTED_GRAPH,
    EXAMPLE_DIRECTED_GRAPH,
    EXAMPLE_BIPARTITE_GRAPH,
    EXAMPLE_FLOW_NETWORK,
)

# Evaluation types
from .evaluation_types import (
    EvaluationType,
    CoreEvaluationType,
    PathEvaluationType,
    CycleEvaluationType,
    TreeEvaluationType,
    ColoringEvaluationType,
    FlowEvaluationType,
    ComponentEvaluationType,
    StructureEvaluationType,
    OrderingEvaluationType,
    SpecialGraphEvaluationType,
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
    ResponseSchema,
    AnswerSchema,
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
    ExtendedResult,
)

__all__ = [
    # Graph structures
    "NodeSchema",
    "EdgeSchema",
    "GraphSchema",
    "EXAMPLE_UNDIRECTED_GRAPH",
    "EXAMPLE_DIRECTED_GRAPH",
    "EXAMPLE_BIPARTITE_GRAPH",
    "EXAMPLE_FLOW_NETWORK",
    
    # Evaluation types
    "EvaluationType",
    "CoreEvaluationType",
    "PathEvaluationType",
    "CycleEvaluationType",
    "TreeEvaluationType",
    "ColoringEvaluationType",
    "FlowEvaluationType",
    "ComponentEvaluationType",
    "StructureEvaluationType",
    "OrderingEvaluationType",
    "SpecialGraphEvaluationType",
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
    "ResponseSchema",
    "AnswerSchema",
    
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
    "ExtendedResult",
]
