"""
Enhanced Intelligent Clustering Analysis Module

A comprehensive clustering analysis system with advanced algorithms, optimization, interpretability,
and scalability features. This module provides a complete workflow for clustering analysis with
state-of-the-art machine learning capabilities.

Main Components:
- IntelligentClusteringModule: Main GUI module with 6-stage workflow
- Advanced Algorithms: Fuzzy C-Means, Consensus K-Means, Enhanced Affinity Propagation
- Parameter Optimization: Bayesian optimization and adaptive parameter recommendation
- Evaluation Metrics: Comprehensive clustering evaluation with multiple validity measures
- Algorithm Selection: Intelligent algorithm selection based on data characteristics
- Interpretability: Cluster profiling, explanations, and visual interpretations
- Performance Optimization: Scalable algorithms for large datasets

Features:
- 10+ advanced clustering algorithms with ensemble methods
- Bayesian parameter optimization with Gaussian Process surrogate models
- Intelligent algorithm selection based on comprehensive data analysis
- Comprehensive evaluation metrics (internal, external, relative, stability)
- Clustering interpretability with natural language explanations
- Performance optimization for large-scale datasets
- Memory-efficient processing and streaming algorithms
- Real-time parameter adjustment with live clustering preview
- Professional report generation and export capabilities

Usage:
    from modules.clustering import IntelligentClusteringModule, IntelligentAlgorithmSelector
    
    # Main clustering GUI
    clustering_module = IntelligentClusteringModule()
    clustering_module.show()
    
    # Intelligent algorithm selection
    selector = IntelligentAlgorithmSelector()
    results = selector.recommend_algorithms(X, n_recommendations=3)
"""

# Main clustering module
from .intelligent_clustering_module import IntelligentClusteringModule

# Worker threads for background processing
from .clustering_workers import (
    ClusteringWorker,
    DimensionalityReductionWorker,
    EvaluationWorker
)

# Real-time preview functionality
from .realtime_preview import (
    RealtimePreviewWidget,
    RealtimePreviewWorker
)

# Parameter recommendation engine
from .parameter_recommendation import ParameterRecommendationEngine

# Advanced clustering algorithms
from .advanced_algorithms import (
    FuzzyCMeans,
    ConsensusKMeans,
    MiniBatchKMeansPlus,
    EnhancedAffinityPropagation,
    ADVANCED_ALGORITHMS
)

# Adaptive parameter optimization
from .adaptive_optimization import (
    BayesianClusterOptimizer,
    AdaptiveParameterRecommender,
    AdvancedParameterOptimizer
)

# Comprehensive evaluation metrics
from .evaluation_metrics import ComprehensiveClusteringEvaluator

# Intelligent algorithm selection
from .algorithm_selection import (
    DataCharacterizationEngine,
    AlgorithmSelector,
    IntelligentAlgorithmSelector
)

# Clustering interpretability
from .interpretability import (
    ClusterProfiler,
    ClusterExplainer,
    ClusterVisualExplainer,
    ComprehensiveClusteringInterpreter
)

# Performance optimization for large datasets
from .performance_optimization import (
    MemoryEfficientScaler,
    StreamingKMeans,
    ApproximateDBSCAN,
    ParallelAgglomerativeClustering,
    ChunkedDistanceMatrix,
    MemoryMonitor,
    ScalableClusteringPipeline,
    ParallelClusteringEvaluator,
    PerformanceOptimizedClusteringModule
)

# Main exports for easy importing
__all__ = [
    # Main module
    'IntelligentClusteringModule',
    
    # Workers
    'ClusteringWorker',
    'DimensionalityReductionWorker', 
    'EvaluationWorker',
    
    # Real-time preview
    'RealtimePreviewWidget',
    'RealtimePreviewWorker',
    
    # Parameter recommendation
    'ParameterRecommendationEngine',
    
    # Advanced algorithms
    'FuzzyCMeans',
    'ConsensusKMeans',
    'MiniBatchKMeansPlus',
    'EnhancedAffinityPropagation',
    'ADVANCED_ALGORITHMS',
    
    # Adaptive optimization
    'BayesianClusterOptimizer',
    'AdaptiveParameterRecommender',
    'AdvancedParameterOptimizer',
    
    # Evaluation metrics
    'ComprehensiveClusteringEvaluator',
    
    # Algorithm selection
    'DataCharacterizationEngine',
    'AlgorithmSelector',
    'IntelligentAlgorithmSelector',
    
    # Interpretability
    'ClusterProfiler',
    'ClusterExplainer',
    'ClusterVisualExplainer',
    'ComprehensiveClusteringInterpreter',
    
    # Performance optimization
    'MemoryEfficientScaler',
    'StreamingKMeans',
    'ApproximateDBSCAN',
    'ParallelAgglomerativeClustering',
    'ChunkedDistanceMatrix',
    'MemoryMonitor',
    'ScalableClusteringPipeline',
    'ParallelClusteringEvaluator',
    'PerformanceOptimizedClusteringModule'
]

# Module metadata
__version__ = '3.0.0'
__author__ = 'Enhanced Clustering Team'
__description__ = 'Comprehensive clustering analysis with advanced algorithms, optimization, interpretability, and scalability'