"""
Intelligent Algorithm Selection System for Clustering
Automatically selects the best clustering algorithm based on data characteristics
"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import shapiro, normaltest
from scipy.spatial.distance import pdist
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import time

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10  # Small value to prevent division by zero
MIN_SAMPLES_FOR_ANALYSIS = 10
MAX_SAMPLE_SIZE = 5000


class DataCharacterizationEngine:
    """
    Analyzes dataset characteristics to inform algorithm selection
    """
    
    def __init__(self, fast_mode=False):
        self.fast_mode = fast_mode
        self.characteristics = {}
        
    def analyze_dataset(self, X):
        """
        Comprehensive dataset analysis for algorithm selection
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input dataset
            
        Returns:
        --------
        characteristics : dict
            Dataset characteristics
        """
        
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        characteristics = {
            'basic_stats': self._compute_basic_statistics(X),
            'dimensionality': self._analyze_dimensionality(X),
            'distribution': self._analyze_distribution(X),
            'density': self._analyze_density(X),
            'structure': self._analyze_structure(X),
            'noise': self._estimate_noise(X),
            'scalability': self._assess_scalability(X),
            'complexity': self._estimate_complexity(X)
        }
        
        self.characteristics = characteristics
        return characteristics
    
    def _compute_basic_statistics(self, X):
        """Compute basic dataset statistics"""
        n_samples, n_features = X.shape
        
        basic_stats = {
            'n_samples': n_samples,
            'n_features': n_features,
            'data_size': n_samples * n_features,
            'memory_usage_mb': X.nbytes / (1024 * 1024),
            'missing_values': np.isnan(X).sum() if np.issubdtype(X.dtype, np.floating) else 0,
            'feature_ranges': np.ptp(X, axis=0),
            'mean_feature_range': np.mean(np.ptp(X, axis=0)),
            'feature_variance': np.var(X, axis=0),
            'total_variance': np.var(X)
        }
        
        return basic_stats
    
    def _analyze_dimensionality(self, X):
        """Analyze dimensionality characteristics"""
        n_samples, n_features = X.shape
        
        dimensionality_info = {
            'dimensionality_ratio': n_features / n_samples,
            'high_dimensional': n_features > n_samples / 10,
            'curse_of_dimensionality_risk': n_features > np.sqrt(n_samples)
        }
        
        # Estimate intrinsic dimensionality using PCA
        try:
            if not self.fast_mode and n_features > 1:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                pca = PCA()
                pca.fit(X_scaled)
                
                # Find number of components explaining 95% of variance
                cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
                intrinsic_dim = np.argmax(cumsum_ratio >= 0.95) + 1
                
                dimensionality_info.update({
                    'intrinsic_dimensionality': intrinsic_dim,
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'effective_dimensionality': intrinsic_dim / n_features,
                    'dimensionality_reduction_beneficial': intrinsic_dim < n_features * 0.8
                })
                
        except Exception as e:
            dimensionality_info['pca_error'] = str(e)
        
        return dimensionality_info
    
    def _analyze_distribution(self, X):
        """Analyze data distribution characteristics"""
        distribution_info = {
            'normality_tests': {},
            'skewness': [],
            'kurtosis': [],
            'multimodality_indicators': {}
        }
        
        try:
            # Test normality for each feature (sample if too many features)
            n_features = X.shape[1]
            feature_indices = range(n_features) if n_features <= 10 else np.random.choice(n_features, 10, replace=False)
            
            for i in feature_indices:
                feature_data = X[:, i]
                
                # Skip constant features
                if np.var(feature_data) > 1e-8:
                    # Shapiro-Wilk test (for small samples)
                    if len(feature_data) <= 5000:
                        stat, p_value = shapiro(feature_data)
                        distribution_info['normality_tests'][f'feature_{i}_shapiro'] = {
                            'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05
                        }
                    
                    # D'Agostino's test (for larger samples)
                    if len(feature_data) >= 20:
                        stat, p_value = normaltest(feature_data)
                        distribution_info['normality_tests'][f'feature_{i}_dagostino'] = {
                            'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05
                        }
                    
                    # Skewness and kurtosis
                    from scipy.stats import skew, kurtosis
                    distribution_info['skewness'].append(skew(feature_data))
                    distribution_info['kurtosis'].append(kurtosis(feature_data))
            
            # Overall distribution characteristics
            if distribution_info['skewness']:
                distribution_info['mean_skewness'] = np.mean(np.abs(distribution_info['skewness']))
                distribution_info['mean_kurtosis'] = np.mean(np.abs(distribution_info['kurtosis']))
                distribution_info['highly_skewed'] = distribution_info['mean_skewness'] > 1.0
                distribution_info['heavy_tailed'] = distribution_info['mean_kurtosis'] > 3.0
            
        except Exception as e:
            distribution_info['error'] = str(e)
        
        return distribution_info
    
    def _analyze_density(self, X):
        """Analyze data density characteristics"""
        density_info = {}
        
        try:
            n_samples = X.shape[0]
            sample_size = min(1000, n_samples) if self.fast_mode else min(5000, n_samples)
            
            if n_samples > sample_size:
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # K-nearest neighbor density estimation
            k_values = [3, 5, 10]
            density_estimates = []
            
            for k in k_values:
                if k < len(X_sample):
                    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_sample)
                    distances, _ = nbrs.kneighbors(X_sample)
                    
                    # Average distance to k-th nearest neighbor
                    avg_distance = np.mean(distances[:, k])
                    density_estimate = 1.0 / (avg_distance + 1e-8)
                    density_estimates.append(density_estimate)
            
            if density_estimates:
                density_info.update({
                    'density_estimates': density_estimates,
                    'mean_density': np.mean(density_estimates),
                    'density_variation': np.std(density_estimates) / np.mean(density_estimates),
                    'sparse_data': np.mean(density_estimates) < 0.1,
                    'dense_data': np.mean(density_estimates) > 10.0
                })
            
            # Local density variation
            if len(X_sample) > 50:
                nbrs = NearestNeighbors(n_neighbors=11).fit(X_sample)
                distances, _ = nbrs.kneighbors(X_sample)
                local_densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)
                
                density_info.update({
                    'local_density_variation': np.std(local_densities) / np.mean(local_densities),
                    'density_uniformity': 'uniform' if density_info.get('density_variation', 1) < 0.5 else 'varying'
                })
        
        except Exception as e:
            density_info['error'] = str(e)
        
        return density_info
    
    def _analyze_structure(self, X):
        """Analyze structural characteristics of the data"""
        structure_info = {}
        
        try:
            n_samples, n_features = X.shape
            sample_size = min(1000, n_samples) if self.fast_mode else n_samples
            
            if n_samples > sample_size:
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Cluster tendency analysis (Hopkins statistic)
            structure_info['hopkins_statistic'] = self._compute_hopkins_statistic(X_sample)
            structure_info['clusterable'] = structure_info['hopkins_statistic'] < 0.3 or structure_info['hopkins_statistic'] > 0.7
            
            # Connectivity analysis
            if len(X_sample) > 10:
                nbrs = NearestNeighbors(n_neighbors=min(11, len(X_sample))).fit(X_sample)
                connectivity_graph = nbrs.kneighbors_graph(X_sample, mode='connectivity')
                
                # Graph-based measures
                n_components = self._count_connected_components(connectivity_graph)
                structure_info.update({
                    'connected_components': n_components,
                    'well_connected': n_components == 1,
                    'fragmented': n_components > len(X_sample) * 0.1
                })
            
            # Estimate number of natural clusters using multiple methods
            structure_info['cluster_estimates'] = self._estimate_natural_clusters(X_sample)
            
        except Exception as e:
            structure_info['error'] = str(e)
        
        return structure_info
    
    def _compute_hopkins_statistic(self, X):
        """Compute Hopkins statistic for cluster tendency"""
        try:
            n_samples = X.shape[0]
            sample_size = min(50, n_samples // 4)
            
            if sample_size < 2:
                return 0.5
            
            # Sample random points from data
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[sample_indices]
            
            # Generate uniform random points in data space
            data_min = X.min(axis=0)
            data_max = X.max(axis=0)
            random_points = np.random.uniform(data_min, data_max, (sample_size, X.shape[1]))
            
            # Find nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=2).fit(X)
            
            # Distance from sample points to their nearest neighbors in X
            sample_distances, _ = nbrs.kneighbors(X_sample)
            w_distances = sample_distances[:, 1]  # Second nearest (first is itself)
            
            # Distance from random points to nearest neighbors in X
            random_distances, _ = nbrs.kneighbors(random_points)
            u_distances = random_distances[:, 0]
            
            # Hopkins statistic
            hopkins = np.sum(u_distances) / (np.sum(u_distances) + np.sum(w_distances))
            
            return hopkins
            
        except Exception:
            return 0.5
    
    def _count_connected_components(self, connectivity_graph):
        """Count connected components in connectivity graph"""
        try:
            from scipy.sparse.csgraph import connected_components
            n_components, _ = connected_components(connectivity_graph)
            return n_components
        except Exception:
            return 1
    
    def _estimate_natural_clusters(self, X):
        """Estimate number of natural clusters using multiple methods"""
        estimates = {}
        
        try:
            max_clusters = min(20, len(X) // 10)
            if max_clusters < 2:
                return {'error': 'Too few samples for cluster estimation'}
            
            # Elbow method with K-means
            inertias = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42)
                    kmeans.fit(X)
                    inertias.append(kmeans.inertia_)
                except Exception:
                    break
            
            if len(inertias) >= 3:
                # Find elbow using second derivative
                diffs = np.diff(inertias)
                diffs2 = np.diff(diffs)
                elbow_idx = np.argmax(diffs2) + 2
                estimates['elbow_method'] = min(k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1], max_clusters)
            
            # Gap statistic (simplified)
            if len(inertias) >= 2:
                # Simple heuristic: look for largest drop in inertia
                inertia_drops = -np.diff(inertias)
                gap_k = np.argmax(inertia_drops) + 2
                estimates['gap_statistic'] = min(gap_k, max_clusters)
            
            # Silhouette method
            silhouette_scores = []
            for k in range(2, min(11, max_clusters + 1)):
                try:
                    kmeans = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42)
                    labels = kmeans.fit_predict(X)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X, labels)
                        silhouette_scores.append((k, score))
                except Exception:
                    continue
            
            if silhouette_scores:
                best_k = max(silhouette_scores, key=lambda x: x[1])[0]
                estimates['silhouette_method'] = best_k
            
            # Rule of thumb estimates
            estimates['sqrt_rule'] = int(np.sqrt(len(X) / 2))
            estimates['log_rule'] = max(2, int(np.log2(len(X))))
            
        except Exception as e:
            estimates['error'] = str(e)
        
        return estimates
    
    def _estimate_noise(self, X):
        """Estimate noise level in the dataset"""
        noise_info = {}
        
        try:
            n_samples = X.shape[0]
            sample_size = min(500, n_samples) if self.fast_mode else n_samples
            
            if n_samples > sample_size:
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Local outlier factor based noise estimation
            if len(X_sample) > 20:
                from sklearn.neighbors import LocalOutlierFactor
                lof = LocalOutlierFactor(n_neighbors=min(20, len(X_sample) - 1), contamination='auto')
                outlier_scores = lof.fit_predict(X_sample)
                
                noise_info.update({
                    'outlier_fraction': np.sum(outlier_scores == -1) / len(outlier_scores),
                    'high_noise': np.sum(outlier_scores == -1) / len(outlier_scores) > 0.1,
                    'low_noise': np.sum(outlier_scores == -1) / len(outlier_scores) < 0.05
                })
            
            # Distance-based noise estimation
            if len(X_sample) > 10:
                nbrs = NearestNeighbors(n_neighbors=min(6, len(X_sample))).fit(X_sample)
                distances, _ = nbrs.kneighbors(X_sample)
                
                # Use distance to 5th nearest neighbor as noise indicator
                noise_distances = distances[:, -1]
                noise_threshold = np.percentile(noise_distances, 95)
                noise_points = noise_distances > noise_threshold
                
                noise_info.update({
                    'distance_based_noise_fraction': np.sum(noise_points) / len(noise_points),
                    'noise_threshold': noise_threshold,
                    'mean_neighbor_distance': np.mean(distances[:, 1:])
                })
        
        except Exception as e:
            noise_info['error'] = str(e)
        
        return noise_info
    
    def _assess_scalability(self, X):
        """Assess scalability requirements"""
        n_samples, n_features = X.shape
        
        scalability_info = {
            'dataset_size': n_samples * n_features,
            'large_dataset': n_samples > 10000,
            'very_large_dataset': n_samples > 100000,
            'high_dimensional': n_features > 100,
            'very_high_dimensional': n_features > 1000,
            'memory_intensive': (n_samples * n_features * 8) / (1024**3) > 1.0,  # > 1GB
            'scalability_requirements': []
        }
        
        # Determine scalability requirements
        if scalability_info['large_dataset']:
            scalability_info['scalability_requirements'].append('efficient_algorithms')
        if scalability_info['very_large_dataset']:
            scalability_info['scalability_requirements'].append('online_learning')
        if scalability_info['high_dimensional']:
            scalability_info['scalability_requirements'].append('dimensionality_reduction')
        if scalability_info['memory_intensive']:
            scalability_info['scalability_requirements'].append('memory_optimization')
        
        return scalability_info
    
    def _estimate_complexity(self, X):
        """Estimate dataset complexity for algorithm selection"""
        complexity_info = {}
        
        try:
            n_samples, n_features = X.shape
            
            # Feature correlation complexity
            if n_features > 1 and not self.fast_mode:
                # Sample features if too many
                if n_features > 50:
                    feature_indices = np.random.choice(n_features, 50, replace=False)
                    X_subset = X[:, feature_indices]
                else:
                    X_subset = X
                
                corr_matrix = np.corrcoef(X_subset.T)
                mean_correlation = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
                
                complexity_info.update({
                    'mean_feature_correlation': mean_correlation,
                    'highly_correlated_features': mean_correlation > 0.7,
                    'independent_features': mean_correlation < 0.3
                })
            
            # Linearity assessment using PCA
            if n_features > 1 and n_samples > n_features:
                try:
                    pca = PCA(n_components=min(10, n_features))
                    pca.fit(StandardScaler().fit_transform(X))
                    
                    # If first few components explain most variance, data is likely linear
                    first_two_components_var = np.sum(pca.explained_variance_ratio_[:2])
                    complexity_info.update({
                        'first_two_components_variance': first_two_components_var,
                        'linear_structure': first_two_components_var > 0.8,
                        'complex_structure': first_two_components_var < 0.5
                    })
                except Exception:
                    pass
            
            # Complexity score (higher = more complex)
            complexity_score = 0
            if complexity_info.get('highly_correlated_features', False):
                complexity_score += 1
            if complexity_info.get('complex_structure', False):
                complexity_score += 2
            if n_features > n_samples:
                complexity_score += 1
            if complexity_info.get('mean_feature_correlation', 0) > 0.5:
                complexity_score += 1
            
            complexity_info['complexity_score'] = complexity_score
            complexity_info['high_complexity'] = complexity_score >= 3
            
        except Exception as e:
            complexity_info['error'] = str(e)
        
        return complexity_info


class AlgorithmSelector:
    """
    Intelligent algorithm selection based on data characteristics
    """
    
    def __init__(self):
        self.algorithm_registry = self._build_algorithm_registry()
        self.selection_rules = self._build_selection_rules()
        
    def _build_algorithm_registry(self):
        """Build registry of available algorithms with their characteristics"""
        registry = {
            'K-Means': {
                'class': KMeans,
                'type': 'centroid_based',
                'scalability': 'excellent',
                'handles_noise': 'poor',
                'cluster_shapes': 'spherical',
                'parameter_sensitivity': 'medium',
                'requires_k': True,
                'memory_usage': 'low',
                'time_complexity': 'O(n*k*i)',
                'suitable_for': ['large_datasets', 'spherical_clusters', 'well_separated'],
                'not_suitable_for': ['noisy_data', 'irregular_shapes', 'varying_densities']
            },
            'DBSCAN': {
                'class': DBSCAN,
                'type': 'density_based',
                'scalability': 'good',
                'handles_noise': 'excellent',
                'cluster_shapes': 'arbitrary',
                'parameter_sensitivity': 'high',
                'requires_k': False,
                'memory_usage': 'medium',
                'time_complexity': 'O(n*log(n))',
                'suitable_for': ['noisy_data', 'arbitrary_shapes', 'varying_sizes'],
                'not_suitable_for': ['varying_densities', 'high_dimensional', 'very_large_datasets']
            },
            'Agglomerative': {
                'class': AgglomerativeClustering,
                'type': 'hierarchical',
                'scalability': 'poor',
                'handles_noise': 'medium',
                'cluster_shapes': 'flexible',
                'parameter_sensitivity': 'low',
                'requires_k': True,
                'memory_usage': 'high',
                'time_complexity': 'O(n^3)',
                'suitable_for': ['small_datasets', 'hierarchical_structure', 'irregular_shapes'],
                'not_suitable_for': ['large_datasets', 'high_dimensional', 'noisy_data']
            },
            'Spectral': {
                'class': SpectralClustering,
                'type': 'graph_based',
                'scalability': 'poor',
                'handles_noise': 'medium',
                'cluster_shapes': 'non_convex',
                'parameter_sensitivity': 'high',
                'requires_k': True,
                'memory_usage': 'high',
                'time_complexity': 'O(n^3)',
                'suitable_for': ['non_convex_clusters', 'manifold_data', 'complex_structures'],
                'not_suitable_for': ['large_datasets', 'noisy_data', 'high_dimensional']
            },
            'Gaussian Mixture': {
                'class': GaussianMixture,
                'type': 'model_based',
                'scalability': 'good',
                'handles_noise': 'medium',
                'cluster_shapes': 'elliptical',
                'parameter_sensitivity': 'medium',
                'requires_k': True,
                'memory_usage': 'medium',
                'time_complexity': 'O(n*k*i)',
                'suitable_for': ['overlapping_clusters', 'probabilistic_assignment', 'elliptical_shapes'],
                'not_suitable_for': ['very_large_datasets', 'irregular_shapes', 'high_noise']
            }
        }
        
        return registry
    
    def _build_selection_rules(self):
        """Build rules for algorithm selection based on data characteristics"""
        rules = [
            # Size-based rules
            {
                'condition': lambda chars: chars['basic_stats']['n_samples'] > 100000,
                'recommendations': ['K-Means', 'Mini-Batch K-Means'],
                'avoid': ['Agglomerative', 'Spectral'],
                'reason': 'Large dataset requires scalable algorithms'
            },
            
            # Noise-based rules
            {
                'condition': lambda chars: chars['noise'].get('high_noise', False),
                'recommendations': ['DBSCAN', 'HDBSCAN'],
                'avoid': ['K-Means'],
                'reason': 'High noise levels detected'
            },
            
            # Dimensionality rules
            {
                'condition': lambda chars: chars['dimensionality'].get('high_dimensional', False),
                'recommendations': ['K-Means', 'Gaussian Mixture'],
                'avoid': ['DBSCAN', 'Spectral'],
                'reason': 'High-dimensional data'
            },
            
            # Cluster structure rules
            {
                'condition': lambda chars: chars['structure'].get('well_connected', True) and 
                           chars['density'].get('density_uniformity', '') == 'uniform',
                'recommendations': ['K-Means', 'Gaussian Mixture'],
                'avoid': ['DBSCAN'],
                'reason': 'Uniform density and well-connected structure'
            },
            
            # Complex structure rules
            {
                'condition': lambda chars: chars['complexity'].get('complex_structure', False),
                'recommendations': ['Spectral', 'DBSCAN'],
                'avoid': ['K-Means'],
                'reason': 'Complex non-linear structure detected'
            },
            
            # Memory constraint rules
            {
                'condition': lambda chars: chars['scalability'].get('memory_intensive', False),
                'recommendations': ['K-Means', 'Mini-Batch K-Means'],
                'avoid': ['Agglomerative', 'Spectral'],
                'reason': 'Memory constraints require efficient algorithms'
            },
            
            # Clustering tendency rules
            {
                'condition': lambda chars: chars['structure'].get('hopkins_statistic', 0.5) > 0.3 and 
                           chars['structure'].get('hopkins_statistic', 0.5) < 0.7,
                'recommendations': [],
                'avoid': ['all'],
                'reason': 'Poor clustering tendency - consider dimensionality reduction first'
            }
        ]
        
        return rules
    
    def select_algorithms(self, data_characteristics, n_recommendations=3, include_parameters=True):
        """
        Select best algorithms based on data characteristics
        
        Parameters:
        -----------
        data_characteristics : dict
            Dataset characteristics from DataCharacterizationEngine
        n_recommendations : int
            Number of algorithms to recommend
        include_parameters : bool
            Whether to include parameter recommendations
            
        Returns:
        --------
        recommendations : list
            List of recommended algorithms with scores and parameters
        """
        
        # Score all algorithms
        algorithm_scores = {}
        
        for algo_name, algo_info in self.algorithm_registry.items():
            score = self._score_algorithm(algo_name, algo_info, data_characteristics)
            algorithm_scores[algo_name] = score
        
        # Apply selection rules
        rule_recommendations = {'recommend': [], 'avoid': []}
        
        for rule in self.selection_rules:
            try:
                if rule['condition'](data_characteristics):
                    rule_recommendations['recommend'].extend(rule['recommendations'])
                    rule_recommendations['avoid'].extend(rule['avoid'])
            except Exception:
                continue
        
        # Adjust scores based on rules
        for algo_name in rule_recommendations['recommend']:
            if algo_name in algorithm_scores:
                algorithm_scores[algo_name] += 0.3
        
        for algo_name in rule_recommendations['avoid']:
            if algo_name in algorithm_scores:
                if 'all' in rule_recommendations['avoid']:
                    algorithm_scores[algo_name] -= 0.8
                else:
                    algorithm_scores[algo_name] -= 0.2
        
        # Sort algorithms by score
        sorted_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build recommendations
        recommendations = []
        
        for i, (algo_name, score) in enumerate(sorted_algorithms[:n_recommendations]):
            if score > 0:  # Only recommend algorithms with positive scores
                recommendation = {
                    'algorithm': algo_name,
                    'score': score,
                    'confidence': min(1.0, score),
                    'rank': i + 1,
                    'algorithm_info': self.algorithm_registry[algo_name],
                    'reasoning': self._generate_reasoning(algo_name, data_characteristics, rule_recommendations)
                }
                
                if include_parameters:
                    recommendation['suggested_parameters'] = self._suggest_parameters(
                        algo_name, data_characteristics
                    )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _score_algorithm(self, algo_name, algo_info, characteristics):
        """Score an algorithm based on data characteristics"""
        score = 0.0  # Start with base score of 0 for better differentiation
        
        try:
            basic_stats = characteristics.get('basic_stats', {})
            scalability = characteristics.get('scalability', {})
            noise = characteristics.get('noise', {})
            structure = characteristics.get('structure', {})
            density = characteristics.get('density', {})
            dimensionality = characteristics.get('dimensionality', {})
            
            n_samples = basic_stats.get('n_samples', 0)
            n_features = basic_stats.get('n_features', 0)
            
            # Algorithm-specific base scoring
            if algo_name == 'K-Means':
                score = 0.7  # Generally good default
                # Excellent for spherical clusters and large datasets
                if n_samples > 1000:
                    score += 0.2
                if not noise.get('high_noise', False):
                    score += 0.1
                    
            elif algo_name == 'DBSCAN':
                score = 0.6  # Good for arbitrary shapes
                # Excellent for noise handling
                if noise.get('high_noise', False):
                    score += 0.3
                if density.get('density_variation', 0) > 0.5:
                    score += 0.2
                # Not great for high-dimensional data
                if dimensionality.get('high_dimensional', False):
                    score -= 0.2
                    
            elif algo_name == 'Agglomerative':
                score = 0.5  # Moderate baseline
                # Good for small to medium datasets
                if n_samples < 5000:
                    score += 0.2
                if n_samples > 10000:
                    score -= 0.3
                # Good connectivity structure
                if structure.get('connectivity', 0) > 0.7:
                    score += 0.2
                    
            elif algo_name == 'Spectral':
                score = 0.4  # Lower baseline due to complexity
                # Good for non-convex shapes
                if structure.get('non_convex', False):
                    score += 0.3
                # Memory intensive
                if n_samples > 5000:
                    score -= 0.2
                # Dimensionality sensitive
                if dimensionality.get('high_dimensional', False):
                    score -= 0.2
                    
            elif algo_name == 'Gaussian Mixture':
                score = 0.6  # Good probabilistic model
                # Good for overlapping clusters
                if structure.get('overlapping', False):
                    score += 0.3
                # Handles mixed cluster shapes well
                if not structure.get('well_separated', True):
                    score += 0.2
                # Good for moderate dimensionality
                if 2 <= n_features <= 20:
                    score += 0.1
            else:
                score = 0.3  # Unknown algorithm
            
            # Additional characteristic-based adjustments
            
            # Scalability scoring - more aggressive
            if n_samples > 10000:
                if algo_info.get('scalability') == 'excellent':
                    score += 0.2
                elif algo_info.get('scalability') == 'poor':
                    score -= 0.4
            
            # Noise handling - more decisive
            if noise.get('high_noise', False):
                if algo_info.get('handles_noise') == 'excellent':
                    score += 0.2
                elif algo_info.get('handles_noise') == 'poor':
                    score -= 0.3
            
            # Memory constraints - significant impact
            memory_mb = basic_stats.get('memory_usage_mb', 0)
            if memory_mb > 100:  # Large dataset
                if algo_info.get('memory_usage') == 'low':
                    score += 0.15
                elif algo_info.get('memory_usage') == 'high':
                    score -= 0.25
            
            # Cluster shape suitability
            if structure.get('spherical', False) and algo_info.get('cluster_shapes') == 'spherical':
                score += 0.15
            elif not structure.get('spherical', True) and algo_info.get('cluster_shapes') == 'any':
                score += 0.15
                
            # Ensure score stays in reasonable bounds
            score = max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"Warning: Error scoring algorithm {algo_name}: {e}")
            score = 0.3  # Fallback score
        
        return score
    
    def _generate_reasoning(self, algo_name, characteristics, rule_recommendations):
        """Generate reasoning for algorithm recommendation"""
        reasons = []
        
        basic_stats = characteristics.get('basic_stats', {})
        noise = characteristics.get('noise', {})
        dimensionality = characteristics.get('dimensionality', {})
        structure = characteristics.get('structure', {})
        density = characteristics.get('density', {})
        
        n_samples = basic_stats.get('n_samples', 0)
        n_features = basic_stats.get('n_features', 0)
        
        # Check why this algorithm was recommended or avoided by rules
        if algo_name in rule_recommendations['recommend']:
            reasons.append("Recommended by selection rules for this data type")
        
        # Algorithm-specific detailed reasoning
        if algo_name == 'K-Means':
            if n_samples > 1000:
                reasons.append(f"Excellent scalability for large datasets ({n_samples:,} samples)")
            if not noise.get('high_noise', False):
                reasons.append("Well-suited for clean data with low noise")
            if structure.get('spherical', False):
                reasons.append("Ideal for spherical cluster shapes")
            else:
                reasons.append("Assumes spherical clusters; may not capture complex shapes")
                
        elif algo_name == 'DBSCAN':
            if noise.get('high_noise', False):
                reasons.append("Excellent at handling noisy data and outliers")
            if density.get('density_variation', 0) > 0.3:
                reasons.append("Can find clusters of varying densities")
            if dimensionality.get('high_dimensional', False):
                reasons.append("May struggle with high-dimensional data (curse of dimensionality)")
            else:
                reasons.append("Good for arbitrary cluster shapes and sizes")
                
        elif algo_name == 'Agglomerative':
            if n_samples < 5000:
                reasons.append(f"Optimal for small to medium datasets ({n_samples:,} samples)")
            elif n_samples > 10000:
                reasons.append(f"May be computationally expensive for large datasets ({n_samples:,} samples)")
            if structure.get('connectivity', 0) > 0.7:
                reasons.append("Excellent for data with clear hierarchical structure")
            reasons.append("No assumptions about cluster shape; finds natural groupings")
            
        elif algo_name == 'Spectral':
            if structure.get('non_convex', False):
                reasons.append("Excellent for non-convex and complex cluster shapes")
            if n_samples > 5000:
                reasons.append(f"Memory intensive for large datasets ({n_samples:,} samples)")
            if dimensionality.get('high_dimensional', False):
                reasons.append("May not perform well in high-dimensional spaces")
            else:
                reasons.append("Good for manifold-like cluster structures")
                
        elif algo_name == 'Gaussian Mixture':
            if structure.get('overlapping', False):
                reasons.append("Handles overlapping clusters with probabilistic assignments")
            if 2 <= n_features <= 20:
                reasons.append(f"Well-suited for moderate dimensionality ({n_features} features)")
            elif n_features > 20:
                reasons.append(f"May struggle with high dimensionality ({n_features} features)")
            reasons.append("Provides cluster membership probabilities")
        
        # Additional data-specific reasoning
        if basic_stats.get('memory_usage_mb', 0) > 100:
            reasons.append(f"Dataset size: {basic_stats.get('memory_usage_mb', 0):.1f} MB")
            
        if dimensionality.get('intrinsic_dimensionality'):
            reasons.append(f"Estimated intrinsic dimensionality: {dimensionality['intrinsic_dimensionality']}")
        
        return reasons if reasons else [f"Baseline suitability for {n_samples:,} samples × {n_features} features dataset"]
    
    def _suggest_parameters(self, algo_name, characteristics):
        """Suggest parameters for the selected algorithm"""
        params = {}
        
        try:
            basic_stats = characteristics.get('basic_stats', {})
            structure = characteristics.get('structure', {})
            n_samples = basic_stats.get('n_samples', 1000)
            
            if algo_name == 'K-Means':
                # Suggest number of clusters
                cluster_estimates = structure.get('cluster_estimates', {})
                if cluster_estimates:
                    # Use silhouette method if available, otherwise elbow method
                    k = (cluster_estimates.get('silhouette_method') or 
                         cluster_estimates.get('elbow_method') or 
                         cluster_estimates.get('sqrt_rule', 3))
                    params['n_clusters'] = max(2, min(20, k))
                else:
                    params['n_clusters'] = max(2, int(np.sqrt(n_samples / 2)))
                
                # Other parameters
                params['n_init'] = 10 if n_samples < 10000 else 3
                params['max_iter'] = 300 if n_samples < 10000 else 100
                
            elif algo_name == 'DBSCAN':
                # Suggest eps and min_samples using heuristics
                # These would ideally be computed from actual data
                params['eps'] = 0.5  # Would need k-distance analysis
                params['min_samples'] = max(2, int(np.log(n_samples)))
                
            elif algo_name == 'Agglomerative':
                cluster_estimates = structure.get('cluster_estimates', {})
                if cluster_estimates:
                    k = (cluster_estimates.get('silhouette_method') or 
                         cluster_estimates.get('elbow_method') or 
                         cluster_estimates.get('sqrt_rule', 3))
                    params['n_clusters'] = max(2, min(20, k))
                else:
                    params['n_clusters'] = max(2, int(np.sqrt(n_samples / 2)))
                
                params['linkage'] = 'ward'
                
            elif algo_name == 'Spectral':
                cluster_estimates = structure.get('cluster_estimates', {})
                if cluster_estimates:
                    k = (cluster_estimates.get('silhouette_method') or 
                         cluster_estimates.get('elbow_method') or 
                         cluster_estimates.get('sqrt_rule', 3))
                    params['n_clusters'] = max(2, min(20, k))
                else:
                    params['n_clusters'] = max(2, int(np.sqrt(n_samples / 2)))
                
                # Gamma parameter based on data characteristics
                data_scale = characteristics.get('basic_stats', {}).get('total_variance', 1.0)
                params['gamma'] = 1.0 / max(data_scale, 0.001)
                
            elif algo_name == 'Gaussian Mixture':
                cluster_estimates = structure.get('cluster_estimates', {})
                if cluster_estimates:
                    k = (cluster_estimates.get('silhouette_method') or 
                         cluster_estimates.get('elbow_method') or 
                         cluster_estimates.get('sqrt_rule', 3))
                    params['n_components'] = max(2, min(20, k))
                else:
                    params['n_components'] = max(2, int(np.sqrt(n_samples / 2)))
                
                params['covariance_type'] = 'full'
                params['max_iter'] = 100
        
        except Exception:
            # Return default parameters if computation fails
            params = {'n_clusters': 3} if algo_name != 'DBSCAN' else {'eps': 0.5, 'min_samples': 5}
        
        return params


class IntelligentAlgorithmSelector:
    """
    Main interface for intelligent algorithm selection
    Combines data characterization with algorithm selection
    """
    
    def __init__(self, fast_mode=False):
        self.characterization_engine = DataCharacterizationEngine(fast_mode=fast_mode)
        self.algorithm_selector = AlgorithmSelector()
        self.selection_history = []
        
    def recommend_algorithms(self, X, n_recommendations=3, include_analysis=True):
        """
        Complete algorithm recommendation pipeline
        
        Parameters:
        -----------
        X : array-like
            Input dataset
        n_recommendations : int
            Number of algorithms to recommend
        include_analysis : bool
            Whether to include detailed data analysis
            
        Returns:
        --------
        results : dict
            Complete recommendation results
        """
        
        start_time = time.time()
        
        try:
            # Step 1: Characterize the dataset
            print("Analyzing dataset characteristics...")
            characteristics = self.characterization_engine.analyze_dataset(X)
            
            # Step 2: Select algorithms
            print("Selecting optimal algorithms...")
            recommendations = self.algorithm_selector.select_algorithms(
                characteristics, 
                n_recommendations=n_recommendations
            )
            
            # Step 3: Compile results
            results = {
                'recommendations': recommendations,
                'analysis_time': time.time() - start_time,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'top_choice': recommendations[0] if recommendations else None
            }
            
            if include_analysis:
                results['data_characteristics'] = characteristics
                results['analysis_summary'] = self._create_analysis_summary(characteristics)
            
            # Store in history
            self.selection_history.append({
                'timestamp': time.time(),
                'data_shape': X.shape,
                'recommendations': recommendations,
                'characteristics_summary': self._create_analysis_summary(characteristics)
            })
            
            return results
            
        except Exception as e:
            return {
                'error': f"Algorithm selection failed: {str(e)}",
                'recommendations': [],
                'analysis_time': time.time() - start_time
            }
    
    def _create_analysis_summary(self, characteristics):
        """Create a human-readable summary of data characteristics"""
        summary = []
        
        try:
            basic = characteristics.get('basic_stats', {})
            summary.append(f"Dataset: {basic.get('n_samples', 0)} samples, {basic.get('n_features', 0)} features")
            
            # Size assessment
            if basic.get('n_samples', 0) > 100000:
                summary.append("Large dataset - scalability is important")
            elif basic.get('n_samples', 0) < 1000:
                summary.append("Small dataset - complex algorithms may overfit")
            
            # Dimensionality assessment
            dimensionality = characteristics.get('dimensionality', {})
            if dimensionality.get('high_dimensional', False):
                summary.append("High-dimensional data detected")
            if dimensionality.get('curse_of_dimensionality_risk', False):
                summary.append("Risk of curse of dimensionality")
            
            # Noise assessment
            noise = characteristics.get('noise', {})
            if noise.get('high_noise', False):
                summary.append("High noise levels detected")
            elif noise.get('low_noise', False):
                summary.append("Clean dataset with low noise")
            
            # Structure assessment
            structure = characteristics.get('structure', {})
            if not structure.get('clusterable', True):
                summary.append("Weak clustering tendency - consider preprocessing")
            
            hopkins = structure.get('hopkins_statistic', 0.5)
            if hopkins < 0.3:
                summary.append("Strong clustering structure (Hopkins < 0.3)")
            elif hopkins > 0.7:
                summary.append("Random-like structure (Hopkins > 0.7)")
            
            # Complexity assessment
            complexity = characteristics.get('complexity', {})
            if complexity.get('high_complexity', False):
                summary.append("Complex data structure detected")
            if complexity.get('linear_structure', False):
                summary.append("Linear structure suitable for simple algorithms")
            
        except Exception:
            summary.append("Analysis summary generation failed")
        
        return summary
    
    def get_selection_history(self):
        """Get history of algorithm selections"""
        return self.selection_history
    
    def clear_history(self):
        """Clear selection history"""
        self.selection_history = []