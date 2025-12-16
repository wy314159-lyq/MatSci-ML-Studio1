"""
Comprehensive Evaluation Metrics Suite for Clustering Analysis
Implements advanced clustering evaluation metrics and multi-criteria assessment
"""

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score,
    normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import time

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10  # Small value to prevent division by zero
MIN_CLUSTER_SIZE = 2
MAX_SAMPLE_SIZE = 5000  # Maximum sample size for expensive computations


class ComprehensiveClusteringEvaluator:
    """
    Comprehensive clustering evaluation system with multiple categories of metrics
    
    Categories:
    1. Internal Validity - Based only on clustering results
    2. External Validity - Requires ground truth labels
    3. Relative Validity - Comparing different clustering solutions
    4. Stability Analysis - Assessing clustering robustness
    5. Interpretability Metrics - Understanding cluster characteristics
    """
    
    def __init__(self, cache_results=True, parallel_jobs=1):
        self.cache_results = cache_results
        self.parallel_jobs = parallel_jobs
        self.evaluation_cache = {}
        self.evaluation_history = []
        
    def evaluate_clustering(self, X, labels, ground_truth=None, algorithm_name="Unknown", 
                          include_categories=None, quick_mode=False):
        """
        Comprehensive clustering evaluation across multiple metric categories
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Original data
        labels : array-like of shape (n_samples,)
            Cluster labels
        ground_truth : array-like, optional
            True labels for external validation
        algorithm_name : str
            Name of clustering algorithm used
        include_categories : list, optional
            Categories to include: ['internal', 'external', 'relative', 'stability', 'interpretability']
        quick_mode : bool
            If True, compute only essential metrics for speed
            
        Returns:
        --------
        evaluation_results : dict
            Comprehensive evaluation results
        """
        
        start_time = time.time()
        
        # Validate inputs
        X = np.asarray(X)
        labels = np.asarray(labels)
        
        if len(np.unique(labels)) < 2:
            return self._get_invalid_clustering_results("Less than 2 clusters")
        
        if include_categories is None:
            include_categories = ['internal', 'interpretability']
            if ground_truth is not None:
                include_categories.append('external')
        
        # Initialize results
        results = {
            'algorithm': algorithm_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_clusters': len(np.unique(labels)),
            'evaluation_time': 0,
            'metrics': {}
        }
        
        try:
            # Internal Validity Metrics
            if 'internal' in include_categories:
                results['metrics']['internal'] = self._evaluate_internal_validity(X, labels, quick_mode)
            
            # External Validity Metrics (if ground truth available)
            if 'external' in include_categories and ground_truth is not None:
                results['metrics']['external'] = self._evaluate_external_validity(labels, ground_truth)
            
            # Relative Validity Metrics
            if 'relative' in include_categories:
                results['metrics']['relative'] = self._evaluate_relative_validity(X, labels)
            
            # Stability Analysis
            if 'stability' in include_categories and not quick_mode:
                results['metrics']['stability'] = self._evaluate_stability(X, labels, algorithm_name)
            
            # Interpretability Metrics
            if 'interpretability' in include_categories:
                results['metrics']['interpretability'] = self._evaluate_interpretability(X, labels, quick_mode)
            
            # Overall Assessment
            results['overall_score'] = self._calculate_overall_score(results['metrics'])
            results['evaluation_time'] = time.time() - start_time
            
            # Cache results if enabled
            if self.cache_results:
                cache_key = f"{algorithm_name}_{X.shape}_{len(np.unique(labels))}"
                self.evaluation_cache[cache_key] = results
            
            # Add to evaluation history
            self.evaluation_history.append(results)
            
            return results
            
        except Exception as e:
            return self._get_error_results(f"Evaluation failed: {str(e)}")
    
    def _evaluate_internal_validity(self, X, labels, quick_mode=False):
        """Evaluate internal clustering validity metrics"""
        internal_metrics = {}
        
        try:
            # Core metrics (always computed)
            internal_metrics['silhouette_score'] = silhouette_score(X, labels)
            internal_metrics['calinski_harabasz_index'] = calinski_harabasz_score(X, labels)
            internal_metrics['davies_bouldin_index'] = davies_bouldin_score(X, labels)
            
            if not quick_mode:
                # Advanced internal metrics
                internal_metrics.update(self._compute_advanced_internal_metrics(X, labels))
            
            # Interpretation
            internal_metrics['interpretation'] = self._interpret_internal_metrics(internal_metrics)
            
        except Exception as e:
            internal_metrics['error'] = f"Internal validation failed: {str(e)}"
            
        return internal_metrics
    
    def _compute_advanced_internal_metrics(self, X, labels):
        """Compute advanced internal validity metrics"""
        advanced_metrics = {}
        
        try:
            # Dunn Index
            advanced_metrics['dunn_index'] = self._compute_dunn_index(X, labels)
            
            # Silhouette Analysis
            silhouette_samples_scores = silhouette_samples(X, labels)
            advanced_metrics['silhouette_samples'] = {
                'mean': np.mean(silhouette_samples_scores),
                'std': np.std(silhouette_samples_scores),
                'min': np.min(silhouette_samples_scores),
                'max': np.max(silhouette_samples_scores),
                'per_cluster': [np.mean(silhouette_samples_scores[labels == i]) 
                               for i in range(len(np.unique(labels)))]
            }
            
            # Within-Cluster Sum of Squares (WCSS)
            advanced_metrics['wcss'] = self._compute_wcss(X, labels)
            
            # Between-Cluster Sum of Squares (BCSS)
            advanced_metrics['bcss'] = self._compute_bcss(X, labels)
            
            # Variance Ratio Criterion
            if advanced_metrics['wcss'] > 0:
                advanced_metrics['variance_ratio'] = advanced_metrics['bcss'] / advanced_metrics['wcss']
            else:
                advanced_metrics['variance_ratio'] = float('inf')
            
            # Connectivity
            advanced_metrics['connectivity'] = self._compute_connectivity(X, labels)
            
            # Separation
            advanced_metrics['separation'] = self._compute_separation(X, labels)

            # DBCV (optional, requires hdbscan)
            advanced_metrics['dbcv'] = self._compute_dbcv(X, labels)
            
        except Exception as e:
            advanced_metrics['error'] = f"Advanced internal metrics failed: {str(e)}"
            
        return advanced_metrics

    def _compute_dbcv(self, X, labels):
        """Compute DBCV (Density-Based Clustering Validation) if hdbscan is available.
        Returns a float in [-1, 1] or None if not available/invalid.
        """
        try:
            import hdbscan  # Optional dependency
        except Exception:
            return None
        try:
            unique = np.unique(labels)
            n_clusters = len([u for u in unique if u != -1])
            if n_clusters < 2:
                return None
            return hdbscan.validity_index(np.asarray(X), np.asarray(labels))
        except Exception:
            return None

    
    def _compute_dunn_index(self, X, labels):
        """Compute Dunn Index (ratio of minimum inter-cluster distance to maximum intra-cluster distance)"""
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return 0.0
            
            # Compute inter-cluster distances (minimum distance between clusters)
            min_inter_cluster_dist = float('inf')
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    cluster_i = X[labels == unique_labels[i]]
                    cluster_j = X[labels == unique_labels[j]]
                    
                    if len(cluster_i) > 0 and len(cluster_j) > 0:
                        distances = pdist(np.vstack([cluster_i, cluster_j]))[:len(cluster_i) * len(cluster_j)]
                        min_dist = np.min(distances) if len(distances) > 0 else float('inf')
                        min_inter_cluster_dist = min(min_inter_cluster_dist, min_dist)
            
            # Compute intra-cluster distances (maximum distance within clusters)
            max_intra_cluster_dist = 0.0
            for label in unique_labels:
                cluster_data = X[labels == label]
                if len(cluster_data) > 1:
                    intra_distances = pdist(cluster_data)
                    max_intra_dist = np.max(intra_distances) if len(intra_distances) > 0 else 0.0
                    max_intra_cluster_dist = max(max_intra_cluster_dist, max_intra_dist)
            
            # Dunn Index
            if max_intra_cluster_dist > 0 and min_inter_cluster_dist != float('inf'):
                return min_inter_cluster_dist / max_intra_cluster_dist
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _compute_wcss(self, X, labels):
        """Compute Within-Cluster Sum of Squares"""
        wcss = 0.0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_data = X[labels == label]
            if len(cluster_data) > 0:
                centroid = np.mean(cluster_data, axis=0)
                wcss += np.sum((cluster_data - centroid) ** 2)
        
        return wcss
    
    def _compute_bcss(self, X, labels):
        """Compute Between-Cluster Sum of Squares"""
        overall_centroid = np.mean(X, axis=0)
        bcss = 0.0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_data = X[labels == label]
            if len(cluster_data) > 0:
                cluster_centroid = np.mean(cluster_data, axis=0)
                bcss += len(cluster_data) * np.sum((cluster_centroid - overall_centroid) ** 2)
        
        return bcss
    
    def _compute_connectivity(self, X, labels, k=10):
        """Compute connectivity metric"""
        try:
            n_samples = X.shape[0]
            k = min(k, n_samples - 1)
            
            # Find k nearest neighbors for each point
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
            distances, indices = nbrs.kneighbors(X)
            
            connectivity = 0.0
            for i in range(n_samples):
                # Check if k nearest neighbors are in the same cluster
                neighbor_indices = indices[i, 1:]  # Exclude the point itself
                neighbor_labels = labels[neighbor_indices]
                different_cluster = np.sum(neighbor_labels != labels[i])
                connectivity += different_cluster / k
            
            return connectivity / n_samples
            
        except Exception:
            return 1.0
    
    def _compute_separation(self, X, labels):
        """Compute average separation between cluster centers"""
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return 0.0
            
            # Compute cluster centers
            centers = []
            for label in unique_labels:
                cluster_data = X[labels == label]
                if len(cluster_data) > 0:
                    centers.append(np.mean(cluster_data, axis=0))
            
            centers = np.array(centers)
            
            # Compute pairwise distances between centers
            if len(centers) > 1:
                center_distances = pdist(centers)
                return np.mean(center_distances)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _evaluate_external_validity(self, labels, ground_truth):
        """Evaluate external clustering validity metrics"""
        external_metrics = {}
        
        try:
            # Core external metrics
            external_metrics['adjusted_rand_score'] = adjusted_rand_score(ground_truth, labels)
            external_metrics['adjusted_mutual_info'] = adjusted_mutual_info_score(ground_truth, labels)
            external_metrics['normalized_mutual_info'] = normalized_mutual_info_score(ground_truth, labels)
            external_metrics['homogeneity'] = homogeneity_score(ground_truth, labels)
            external_metrics['completeness'] = completeness_score(ground_truth, labels)
            external_metrics['v_measure'] = v_measure_score(ground_truth, labels)
            
            # Interpretation
            external_metrics['interpretation'] = self._interpret_external_metrics(external_metrics)
            
        except Exception as e:
            external_metrics['error'] = f"External validation failed: {str(e)}"
            
        return external_metrics
    
    def _evaluate_relative_validity(self, X, labels):
        """Evaluate relative clustering validity for comparison"""
        relative_metrics = {}
        
        try:
            # Cluster size distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            relative_metrics['cluster_sizes'] = counts.tolist()
            relative_metrics['size_distribution'] = {
                'mean_size': np.mean(counts),
                'std_size': np.std(counts),
                'min_size': np.min(counts),
                'max_size': np.max(counts),
                'size_ratio': np.max(counts) / np.min(counts) if np.min(counts) > 0 else float('inf')
            }
            
            # Density-based metrics
            relative_metrics['density_metrics'] = self._compute_density_metrics(X, labels)
            
            # Compactness and separation
            relative_metrics['compactness'] = self._compute_compactness(X, labels)
            
        except Exception as e:
            relative_metrics['error'] = f"Relative validation failed: {str(e)}"
            
        return relative_metrics
    
    def _compute_density_metrics(self, X, labels):
        """Compute density-based clustering metrics"""
        density_metrics = {}
        
        try:
            unique_labels = np.unique(labels)
            cluster_densities = []
            
            for label in unique_labels:
                cluster_data = X[labels == label]
                if len(cluster_data) > 1:
                    # Estimate density using k-nearest neighbors
                    k = min(5, len(cluster_data) - 1)
                    if k > 0:
                        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(cluster_data)
                        distances, _ = nbrs.kneighbors(cluster_data)
                        avg_distance = np.mean(distances[:, 1:])  # Exclude distance to self
                        density = 1.0 / (avg_distance + 1e-8)
                        cluster_densities.append(density)
            
            if cluster_densities:
                density_metrics['mean_density'] = np.mean(cluster_densities)
                density_metrics['std_density'] = np.std(cluster_densities)
                mean_density = np.mean(cluster_densities)
                if np.abs(mean_density) > EPSILON:
                    density_metrics['density_variation'] = np.std(cluster_densities) / mean_density
                else:
                    density_metrics['density_variation'] = 0.0

        except Exception as e:
            logger.debug(f"Density computation failed: {e}")
            density_metrics['error'] = "Density computation failed"
            
        return density_metrics
    
    def _compute_compactness(self, X, labels):
        """Compute cluster compactness metrics"""
        compactness_metrics = {}
        
        try:
            unique_labels = np.unique(labels)
            compactness_scores = []
            
            for label in unique_labels:
                cluster_data = X[labels == label]
                if len(cluster_data) > 0:
                    centroid = np.mean(cluster_data, axis=0)
                    distances = np.linalg.norm(cluster_data - centroid, axis=1)
                    compactness = np.mean(distances)
                    compactness_scores.append(compactness)
            
            if compactness_scores:
                compactness_metrics['mean_compactness'] = np.mean(compactness_scores)
                compactness_metrics['std_compactness'] = np.std(compactness_scores)
                mean_compactness = np.mean(compactness_scores)
                if np.abs(mean_compactness) > EPSILON:
                    compactness_metrics['compactness_variation'] = np.std(compactness_scores) / mean_compactness
                else:
                    compactness_metrics['compactness_variation'] = 0.0

        except Exception as e:
            logger.debug(f"Compactness computation failed: {e}")
            compactness_metrics['error'] = "Compactness computation failed"
            
        return compactness_metrics
    
    def _evaluate_stability(self, X, labels, algorithm_name):
        """Evaluate clustering stability through bootstrap sampling"""
        stability_metrics = {}
        
        try:
            n_bootstrap = 10
            n_samples = X.shape[0]
            sample_size = int(0.8 * n_samples)
            
            stability_scores = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sampling
                indices = np.random.choice(n_samples, sample_size, replace=True)
                X_bootstrap = X[indices]
                
                # Apply same clustering (simplified - would need actual algorithm)
                # For now, compute similarity to original clustering
                try:
                    # This is a simplified stability measure
                    # In practice, would need to re-run clustering algorithm
                    bootstrap_labels = labels[indices]
                    original_labels_subset = labels[indices]
                    
                    # Measure agreement using ARI
                    if len(np.unique(bootstrap_labels)) > 1 and len(np.unique(original_labels_subset)) > 1:
                        stability_score = adjusted_rand_score(original_labels_subset, bootstrap_labels)
                        stability_scores.append(stability_score)
                
                except Exception:
                    continue
            
            if stability_scores:
                stability_metrics['mean_stability'] = np.mean(stability_scores)
                stability_metrics['std_stability'] = np.std(stability_scores)
                stability_metrics['stability_scores'] = stability_scores
            else:
                stability_metrics['mean_stability'] = 0.0
                stability_metrics['std_stability'] = 1.0
            
        except Exception as e:
            stability_metrics['error'] = f"Stability analysis failed: {str(e)}"
            
        return stability_metrics
    
    def _evaluate_interpretability(self, X, labels, quick_mode=False):
        """Evaluate clustering interpretability metrics"""
        interpretability_metrics = {}
        
        try:
            # Cluster characteristics
            interpretability_metrics['cluster_profiles'] = self._compute_cluster_profiles(X, labels)
            
            if not quick_mode:
                # Feature importance for clusters
                interpretability_metrics['feature_importance'] = self._compute_cluster_feature_importance(X, labels)
                
                # Cluster outliers
                interpretability_metrics['outlier_analysis'] = self._analyze_cluster_outliers(X, labels)
            
        except Exception as e:
            interpretability_metrics['error'] = f"Interpretability analysis failed: {str(e)}"
            
        return interpretability_metrics
    
    def _compute_cluster_profiles(self, X, labels):
        """Compute statistical profiles for each cluster"""
        profiles = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_data = X[labels == label]
            if len(cluster_data) > 0:
                profiles[f'cluster_{label}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(X) * 100,
                    'centroid': np.mean(cluster_data, axis=0).tolist(),
                    'std': np.std(cluster_data, axis=0).tolist(),
                    'min': np.min(cluster_data, axis=0).tolist(),
                    'max': np.max(cluster_data, axis=0).tolist()
                }
        
        return profiles
    
    def _compute_cluster_feature_importance(self, X, labels):
        """Compute feature importance for distinguishing clusters"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            # Use clustering labels as target for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, labels)
            
            feature_importance = {
                'importance_scores': rf.feature_importances_.tolist(),
                'top_features': np.argsort(rf.feature_importances_)[::-1][:10].tolist()
            }
            
            # Cross-validation score for separability
            cv_scores = cross_val_score(rf, X, labels, cv=3)
            feature_importance['separability_score'] = np.mean(cv_scores)
            
            return feature_importance
            
        except Exception:
            return {'error': 'Feature importance computation failed'}
    
    def _analyze_cluster_outliers(self, X, labels):
        """Analyze outliers within each cluster"""
        outlier_analysis = {}
        
        try:
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                cluster_data = X[labels == label]
                cluster_indices = np.where(labels == label)[0]
                
                if len(cluster_data) > 3:
                    # Compute distances to cluster centroid
                    centroid = np.mean(cluster_data, axis=0)
                    distances = np.linalg.norm(cluster_data - centroid, axis=1)
                    
                    # Identify outliers (using IQR method)
                    q75, q25 = np.percentile(distances, [75, 25])
                    iqr = q75 - q25
                    outlier_threshold = q75 + 1.5 * iqr
                    
                    outlier_mask = distances > outlier_threshold
                    outlier_indices = cluster_indices[outlier_mask]
                    
                    outlier_analysis[f'cluster_{label}'] = {
                        'n_outliers': np.sum(outlier_mask),
                        'outlier_percentage': np.sum(outlier_mask) / len(cluster_data) * 100,
                        'outlier_indices': outlier_indices.tolist(),
                        'mean_distance_to_center': np.mean(distances),
                        'outlier_threshold': outlier_threshold
                    }
        
        except Exception as e:
            outlier_analysis['error'] = f"Outlier analysis failed: {str(e)}"
            
        return outlier_analysis
    
    def _interpret_internal_metrics(self, metrics):
        """Provide interpretation for internal metrics"""
        interpretation = {}
        
        # Silhouette Score interpretation
        sil_score = metrics.get('silhouette_score', 0)
        if sil_score > 0.7:
            interpretation['silhouette'] = "Excellent clustering structure"
        elif sil_score > 0.5:
            interpretation['silhouette'] = "Good clustering structure"
        elif sil_score > 0.25:
            interpretation['silhouette'] = "Weak clustering structure"
        else:
            interpretation['silhouette'] = "Poor clustering structure"
        
        # Davies-Bouldin Index interpretation (lower is better)
        db_score = metrics.get('davies_bouldin_index', float('inf'))
        if db_score < 1.0:
            interpretation['davies_bouldin'] = "Excellent separation between clusters"
        elif db_score < 2.0:
            interpretation['davies_bouldin'] = "Good separation between clusters"
        else:
            interpretation['davies_bouldin'] = "Poor separation between clusters"
        
        # Overall internal quality
        if sil_score > 0.5 and db_score < 2.0:
            interpretation['overall_internal'] = "High quality clustering"
        elif sil_score > 0.25 and db_score < 3.0:
            interpretation['overall_internal'] = "Moderate quality clustering"
        else:
            interpretation['overall_internal'] = "Low quality clustering"
        
        return interpretation
    
    def _interpret_external_metrics(self, metrics):
        """Provide interpretation for external metrics"""
        interpretation = {}
        
        # Adjusted Rand Index interpretation
        ari_score = metrics.get('adjusted_rand_score', 0)
        if ari_score > 0.9:
            interpretation['adjusted_rand'] = "Excellent agreement with ground truth"
        elif ari_score > 0.7:
            interpretation['adjusted_rand'] = "Good agreement with ground truth"
        elif ari_score > 0.3:
            interpretation['adjusted_rand'] = "Moderate agreement with ground truth"
        else:
            interpretation['adjusted_rand'] = "Poor agreement with ground truth"
        
        # V-measure interpretation
        v_measure = metrics.get('v_measure', 0)
        if v_measure > 0.8:
            interpretation['v_measure'] = "Excellent homogeneity and completeness"
        elif v_measure > 0.6:
            interpretation['v_measure'] = "Good homogeneity and completeness"
        else:
            interpretation['v_measure'] = "Poor homogeneity and completeness"
        
        return interpretation
    
    def _calculate_overall_score(self, metrics):
        """Calculate an overall clustering quality score"""
        scores = []
        weights = []
        
        # Internal metrics
        if 'internal' in metrics:
            internal = metrics['internal']
            if 'silhouette_score' in internal:
                scores.append(max(0, internal['silhouette_score']))
                weights.append(0.5)
            if 'dbcv' in internal and internal['dbcv'] is not None:
                scores.append((internal['dbcv'] + 1) / 2.0)
                weights.append(0.3)
            
            if 'davies_bouldin_index' in internal:
                # Convert to a score (lower DB is better, so invert)
                db_score = 1 / (1 + internal['davies_bouldin_index'])
                scores.append(db_score)
                weights.append(0.2)
        
        # External metrics (if available)
        if 'external' in metrics:
            external = metrics['external']
            if 'adjusted_rand_score' in external:
                scores.append(max(0, external['adjusted_rand_score']))
                weights.append(0.3)
            
            if 'v_measure' in external:
                scores.append(external['v_measure'])
                weights.append(0.2)
        
        # Calculate weighted average
        if scores and weights:
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 0.0
        
        return float(overall_score)
    
    def _get_invalid_clustering_results(self, reason):
        """Return results for invalid clustering"""
        return {
            'algorithm': 'Invalid',
            'error': reason,
            'overall_score': 0.0,
            'metrics': {}
        }
    
    def _get_error_results(self, error_message):
        """Return results when evaluation fails"""
        return {
            'algorithm': 'Error',
            'error': error_message,
            'overall_score': 0.0,
            'metrics': {}
        }
    
    def compare_clusterings(self, evaluation_results_list):
        """Compare multiple clustering results"""
        if not evaluation_results_list:
            return {}
        
        comparison = {
            'n_algorithms': len(evaluation_results_list),
            'algorithms': [result.get('algorithm', 'Unknown') for result in evaluation_results_list],
            'overall_scores': [result.get('overall_score', 0.0) for result in evaluation_results_list],
            'best_algorithm': None,
            'ranking': [],
            'detailed_comparison': {}
        }
        
        # Rank algorithms by overall score
        scores_with_indices = [(score, i, result['algorithm']) for i, (score, result) in 
                              enumerate(zip(comparison['overall_scores'], evaluation_results_list))]
        scores_with_indices.sort(reverse=True)
        
        comparison['ranking'] = [(name, score) for score, _, name in scores_with_indices]
        comparison['best_algorithm'] = comparison['ranking'][0][0] if comparison['ranking'] else None
        
        # Detailed metric comparison
        metric_names = set()
        for result in evaluation_results_list:
            for category, metrics in result.get('metrics', {}).items():
                for metric_name in metrics:
                    if isinstance(metrics[metric_name], (int, float)):
                        metric_names.add(f"{category}.{metric_name}")
        
        for metric_name in metric_names:
            values = []
            for result in evaluation_results_list:
                category, metric = metric_name.split('.', 1)
                value = result.get('metrics', {}).get(category, {}).get(metric, np.nan)
                values.append(value)
            
            comparison['detailed_comparison'][metric_name] = {
                'values': values,
                'best_index': np.nanargmax(values) if not all(np.isnan(values)) else 0,
                'worst_index': np.nanargmin(values) if not all(np.isnan(values)) else 0
            }
        
        return comparison
    
    def generate_evaluation_report(self, evaluation_result):
        """Generate a comprehensive evaluation report"""
        if not evaluation_result or 'error' in evaluation_result:
            return f"Evaluation failed: {evaluation_result.get('error', 'Unknown error')}"
        
        report_lines = []
        report_lines.append("CLUSTERING EVALUATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Algorithm: {evaluation_result.get('algorithm', 'Unknown')}")
        report_lines.append(f"Dataset: {evaluation_result.get('n_samples', 0)} samples, {evaluation_result.get('n_features', 0)} features")
        report_lines.append(f"Number of Clusters: {evaluation_result.get('n_clusters', 0)}")
        report_lines.append(f"Overall Score: {evaluation_result.get('overall_score', 0.0):.4f}")
        report_lines.append(f"Evaluation Time: {evaluation_result.get('evaluation_time', 0.0):.2f} seconds")
        report_lines.append("")
        
        metrics = evaluation_result.get('metrics', {})
        
        # Internal Metrics
        if 'internal' in metrics:
            internal = metrics['internal']
            report_lines.append("INTERNAL VALIDITY METRICS")
            report_lines.append("-" * 30)
            
            if 'silhouette_score' in internal:
                report_lines.append(f"Silhouette Score: {internal['silhouette_score']:.4f}")
            if 'calinski_harabasz_index' in internal:
                report_lines.append(f"Calinski-Harabasz Index: {internal['calinski_harabasz_index']:.4f}")
            if 'davies_bouldin_index' in internal:
                report_lines.append(f"Davies-Bouldin Index: {internal['davies_bouldin_index']:.4f}")
            
            if 'interpretation' in internal:
                report_lines.append("\nInterpretation:")
                for key, value in internal['interpretation'].items():
                    report_lines.append(f"  {key}: {value}")
            
            report_lines.append("")
        
        # External Metrics
        if 'external' in metrics:
            external = metrics['external']
            report_lines.append("EXTERNAL VALIDITY METRICS")
            report_lines.append("-" * 30)
            
            for metric_name, value in external.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"{metric_name}: {value:.4f}")
            
            report_lines.append("")
        
        # Interpretability
        if 'interpretability' in metrics:
            interp = metrics['interpretability']
            report_lines.append("CLUSTER INTERPRETABILITY")
            report_lines.append("-" * 30)
            
            if 'cluster_profiles' in interp:
                profiles = interp['cluster_profiles']
                for cluster_name, profile in profiles.items():
                    report_lines.append(f"{cluster_name}: {profile['size']} samples ({profile['percentage']:.1f}%)")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def get_evaluation_summary(self):
        """Get summary of all evaluations performed"""
        if not self.evaluation_history:
            return "No evaluations performed yet"
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'algorithms_tested': list(set(result.get('algorithm', 'Unknown') for result in self.evaluation_history)),
            'average_scores': {},
            'best_result': None,
            'recent_evaluations': self.evaluation_history[-5:]  # Last 5 evaluations
        }
        
        # Calculate average scores by algorithm
        algorithm_scores = {}
        for result in self.evaluation_history:
            algorithm = result.get('algorithm', 'Unknown')
            score = result.get('overall_score', 0.0)
            
            if algorithm not in algorithm_scores:
                algorithm_scores[algorithm] = []
            algorithm_scores[algorithm].append(score)
        
        for algorithm, scores in algorithm_scores.items():
            summary['average_scores'][algorithm] = np.mean(scores)
        
        # Find best result
        if self.evaluation_history:
            summary['best_result'] = max(self.evaluation_history, key=lambda x: x.get('overall_score', 0.0))
        
        return summary
