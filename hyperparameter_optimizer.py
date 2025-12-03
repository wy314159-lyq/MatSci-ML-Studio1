"""
Automated Hyperparameter Optimization for Clustering
Provides automatic parameter search and optimization with visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from itertools import product
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.cluster import OPTICS
    OPTICS_AVAILABLE = True
except ImportError:
    OPTICS_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization"""
    algorithm: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_metric: str
    parameter_grid: Dict[str, List]
    best_labels: np.ndarray
    recommendations: List[str]


class ClusteringHyperparameterOptimizer:
    """Automated hyperparameter optimization for clustering algorithms"""
    
    def __init__(self, optimization_metric: str = 'silhouette', 
                 scoring_metrics: List[str] = None, n_jobs: int = 1):
        """
        Initialize optimizer
        
        Parameters:
        -----------
        optimization_metric : str
            Primary metric for optimization ('silhouette', 'davies_bouldin', 'calinski_harabasz')
        scoring_metrics : list
            Additional metrics to compute for comparison
        n_jobs : int
            Number of parallel jobs for computation
        """
        self.optimization_metric = optimization_metric
        self.scoring_metrics = scoring_metrics or ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        self.n_jobs = n_jobs
        self.results_cache = {}
    
    def optimize_kmeans(self, X: np.ndarray, k_range: Tuple[int, int] = (2, 10),
                       init_methods: List[str] = None, max_iter_values: List[int] = None,
                       random_states: List[int] = None) -> OptimizationResult:
        """
        Optimize K-Means parameters
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        k_range : tuple
            Range of k values to test (min_k, max_k)
        init_methods : list
            Initialization methods to test
        max_iter_values : list
            Maximum iterations to test
        random_states : list
            Random states for stability testing
            
        Returns:
        --------
        OptimizationResult
            Optimization results with best parameters
        """
        
        # Default parameter grid
        if init_methods is None:
            init_methods = ['k-means++', 'random']
        if max_iter_values is None:
            max_iter_values = [300]
        if random_states is None:
            random_states = [42]
        
        k_values = list(range(k_range[0], k_range[1] + 1))
        
        parameter_grid = {
            'n_clusters': k_values,
            'init': init_methods,
            'max_iter': max_iter_values,
            'random_state': random_states
        }
        
        print(f"Testing {len(list(product(*parameter_grid.values())))} parameter combinations...")
        
        all_results = []
        best_score = -np.inf if self.optimization_metric != 'davies_bouldin' else np.inf
        best_params = None
        best_labels = None
        
        for params in product(*parameter_grid.values()):
            param_dict = dict(zip(parameter_grid.keys(), params))
            
            try:
                # Fit K-Means with current parameters
                kmeans = KMeans(**param_dict)
                labels = kmeans.fit_predict(X)
                
                # Calculate metrics
                metrics = self._calculate_metrics(X, labels)
                
                # Store results
                result = {
                    'params': param_dict.copy(),
                    'labels': labels,
                    **metrics
                }
                all_results.append(result)
                
                # Check if this is the best result
                current_score = metrics[self.optimization_metric]
                
                is_better = False
                if self.optimization_metric == 'davies_bouldin':
                    is_better = current_score < best_score
                else:
                    is_better = current_score > best_score
                
                if is_better:
                    best_score = current_score
                    best_params = param_dict.copy()
                    best_labels = labels.copy()
                    
            except Exception as e:
                warnings.warn(f"Failed to fit with params {param_dict}: {str(e)}")
                continue
        
        # Generate recommendations
        recommendations = self._generate_kmeans_recommendations(all_results, k_values)
        
        return OptimizationResult(
            algorithm='K-Means',
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=self.optimization_metric,
            parameter_grid=parameter_grid,
            best_labels=best_labels,
            recommendations=recommendations
        )
    
    def optimize_dbscan(self, X: np.ndarray, eps_range: Tuple[float, float] = None,
                       min_samples_range: Tuple[int, int] = None,
                       n_eps_points: int = 20) -> OptimizationResult:
        """
        Optimize DBSCAN parameters using k-distance analysis
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        eps_range : tuple
            Range of eps values to test
        min_samples_range : tuple
            Range of min_samples values to test
        n_eps_points : int
            Number of eps values to test
        """
        
        # Auto-determine eps range using k-distance plot
        if eps_range is None:
            eps_range = self._estimate_eps_range(X)
        
        if min_samples_range is None:
            # Rule of thumb: min_samples = 2 * dimensions
            min_val = max(2, 2 * X.shape[1])
            max_val = min(20, max(5, int(0.1 * len(X))))
            min_samples_range = (min_val, max_val)
        
        eps_values = np.linspace(eps_range[0], eps_range[1], n_eps_points)
        min_samples_values = list(range(min_samples_range[0], min_samples_range[1] + 1))
        
        parameter_grid = {
            'eps': eps_values.tolist(),
            'min_samples': min_samples_values
        }
        
        print(f"Testing {len(eps_values) * len(min_samples_values)} parameter combinations...")
        
        all_results = []
        best_score = -np.inf if self.optimization_metric != 'davies_bouldin' else np.inf
        best_params = None
        best_labels = None
        
        for eps, min_samples in product(eps_values, min_samples_values):
            param_dict = {'eps': eps, 'min_samples': min_samples}
            
            try:
                # Fit DBSCAN
                dbscan = DBSCAN(**param_dict)
                labels = dbscan.fit_predict(X)
                
                # Skip if all points are noise or only one cluster
                n_clusters = len(np.unique(labels[labels != -1]))
                if n_clusters < 2:
                    continue
                
                # Calculate metrics (excluding noise points for some metrics)
                metrics = self._calculate_metrics(X, labels)
                metrics['n_clusters'] = n_clusters
                metrics['n_noise'] = (labels == -1).sum()
                metrics['noise_ratio'] = (labels == -1).sum() / len(labels)
                
                # Store results
                result = {
                    'params': param_dict.copy(),
                    'labels': labels,
                    **metrics
                }
                all_results.append(result)
                
                # Check if this is the best result
                current_score = metrics[self.optimization_metric]
                
                is_better = False
                if self.optimization_metric == 'davies_bouldin':
                    is_better = current_score < best_score
                else:
                    is_better = current_score > best_score
                
                if is_better and not np.isnan(current_score):
                    best_score = current_score
                    best_params = param_dict.copy()
                    best_labels = labels.copy()
                    
            except Exception as e:
                continue
        
        # Generate recommendations
        recommendations = self._generate_dbscan_recommendations(all_results, X)
        
        return OptimizationResult(
            algorithm='DBSCAN',
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=self.optimization_metric,
            parameter_grid=parameter_grid,
            best_labels=best_labels,
            recommendations=recommendations
        )
    
    def optimize_agglomerative(self, X: np.ndarray, k_range: Tuple[int, int] = (2, 10),
                             linkage_methods: List[str] = None) -> OptimizationResult:
        """Optimize Agglomerative Clustering parameters"""
        
        if linkage_methods is None:
            linkage_methods = ['ward', 'complete', 'average', 'single']
        
        k_values = list(range(k_range[0], k_range[1] + 1))
        
        parameter_grid = {
            'n_clusters': k_values,
            'linkage': linkage_methods
        }
        
        all_results = []
        best_score = -np.inf if self.optimization_metric != 'davies_bouldin' else np.inf
        best_params = None
        best_labels = None
        
        for n_clusters, linkage in product(k_values, linkage_methods):
            param_dict = {'n_clusters': n_clusters, 'linkage': linkage}
            
            try:
                # Fit Agglomerative Clustering
                agg = AgglomerativeClustering(**param_dict)
                labels = agg.fit_predict(X)
                
                # Calculate metrics
                metrics = self._calculate_metrics(X, labels)
                
                # Store results
                result = {
                    'params': param_dict.copy(),
                    'labels': labels,
                    **metrics
                }
                all_results.append(result)
                
                # Check if this is the best result
                current_score = metrics[self.optimization_metric]
                
                is_better = False
                if self.optimization_metric == 'davies_bouldin':
                    is_better = current_score < best_score
                else:
                    is_better = current_score > best_score
                
                if is_better:
                    best_score = current_score
                    best_params = param_dict.copy()
                    best_labels = labels.copy()
                    
            except Exception as e:
                warnings.warn(f"Failed to fit with params {param_dict}: {str(e)}")
                continue
        
        # Generate recommendations
        recommendations = self._generate_agglomerative_recommendations(all_results)
        
        return OptimizationResult(
            algorithm='Agglomerative',
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=self.optimization_metric,
            parameter_grid=parameter_grid,
            best_labels=best_labels,
            recommendations=recommendations
        )
    
    def optimize_gaussian_mixture(self, X: np.ndarray, k_range: Tuple[int, int] = (2, 10),
                                 covariance_types: List[str] = None) -> OptimizationResult:
        """Optimize Gaussian Mixture Model parameters"""
        
        if covariance_types is None:
            covariance_types = ['full', 'tied', 'diag', 'spherical']
        
        k_values = list(range(k_range[0], k_range[1] + 1))
        
        parameter_grid = {
            'n_components': k_values,
            'covariance_type': covariance_types,
            'random_state': [42]
        }
        
        all_results = []
        best_score = -np.inf if self.optimization_metric != 'davies_bouldin' else np.inf
        best_params = None
        best_labels = None
        
        for n_components, cov_type in product(k_values, covariance_types):
            param_dict = {
                'n_components': n_components, 
                'covariance_type': cov_type, 
                'random_state': 42
            }
            
            try:
                # Fit Gaussian Mixture Model
                gmm = GaussianMixture(**param_dict)
                gmm.fit(X)
                labels = gmm.predict(X)
                
                # Calculate metrics
                metrics = self._calculate_metrics(X, labels)
                
                # Add GMM-specific metrics
                metrics['aic'] = gmm.aic(X)
                metrics['bic'] = gmm.bic(X)
                metrics['log_likelihood'] = gmm.score(X)
                
                # Store results
                result = {
                    'params': param_dict.copy(),
                    'labels': labels,
                    **metrics
                }
                all_results.append(result)
                
                # Check if this is the best result
                current_score = metrics[self.optimization_metric]
                
                is_better = False
                if self.optimization_metric == 'davies_bouldin':
                    is_better = current_score < best_score
                else:
                    is_better = current_score > best_score
                
                if is_better:
                    best_score = current_score
                    best_params = param_dict.copy()
                    best_labels = labels.copy()
                    
            except Exception as e:
                continue
        
        # Generate recommendations
        recommendations = self._generate_gmm_recommendations(all_results)
        
        return OptimizationResult(
            algorithm='Gaussian Mixture',
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=self.optimization_metric,
            parameter_grid=parameter_grid,
            best_labels=best_labels,
            recommendations=recommendations
        )
    
    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering evaluation metrics"""
        metrics = {}
        
        # Filter out noise points for metrics that don't handle them
        mask = labels != -1
        if mask.sum() < 2:
            return {metric: np.nan for metric in self.scoring_metrics}
        
        filtered_X = X[mask]
        filtered_labels = labels[mask]
        
        if len(np.unique(filtered_labels)) < 2:
            return {metric: np.nan for metric in self.scoring_metrics}
        
        try:
            if 'silhouette' in self.scoring_metrics:
                metrics['silhouette'] = silhouette_score(filtered_X, filtered_labels)
        except:
            metrics['silhouette'] = np.nan
            
        try:
            if 'davies_bouldin' in self.scoring_metrics:
                metrics['davies_bouldin'] = davies_bouldin_score(filtered_X, filtered_labels)
        except:
            metrics['davies_bouldin'] = np.nan
            
        try:
            if 'calinski_harabasz' in self.scoring_metrics:
                metrics['calinski_harabasz'] = calinski_harabasz_score(filtered_X, filtered_labels)
        except:
            metrics['calinski_harabasz'] = np.nan
        
        return metrics
    
    def _estimate_eps_range(self, X: np.ndarray, k: int = 4) -> Tuple[float, float]:
        """Estimate good eps range using k-distance analysis"""
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        k_distances = np.sort(distances[:, k-1], axis=0)
        
        # Use percentiles to estimate range
        eps_min = np.percentile(k_distances, 10)
        eps_max = np.percentile(k_distances, 90)
        
        return (eps_min, eps_max)
    
    def _generate_kmeans_recommendations(self, all_results: List[Dict], 
                                       k_values: List[int]) -> List[str]:
        """Generate recommendations for K-Means results"""
        recommendations = []
        
        # Find elbow point
        if len(all_results) > 0:
            # Group by k value and get best score for each k
            k_scores = {}
            for result in all_results:
                k = result['params']['n_clusters']
                score = result[self.optimization_metric]
                if not np.isnan(score):
                    if k not in k_scores or score > k_scores[k]:
                        k_scores[k] = score
            
            if k_scores:
                sorted_k = sorted(k_scores.keys())
                scores = [k_scores[k] for k in sorted_k]
                
                # Simple elbow detection
                if len(scores) >= 3:
                    second_derivatives = []
                    for i in range(1, len(scores) - 1):
                        second_deriv = scores[i-1] - 2*scores[i] + scores[i+1]
                        second_derivatives.append(abs(second_deriv))
                    
                    if second_derivatives:
                        elbow_idx = np.argmax(second_derivatives)
                        elbow_k = sorted_k[elbow_idx + 1]
                        recommendations.append(f"Elbow method suggests k={elbow_k}")
                
                # Find k with best score
                best_k = max(k_scores.keys(), key=lambda x: k_scores[x])
                recommendations.append(f"Best {self.optimization_metric} score at k={best_k}")
        
        return recommendations
    
    def _generate_dbscan_recommendations(self, all_results: List[Dict], 
                                       X: np.ndarray) -> List[str]:
        """Generate recommendations for DBSCAN results"""
        recommendations = []
        
        if len(all_results) > 0:
            # Find configurations with reasonable cluster counts and low noise
            good_results = []
            for result in all_results:
                n_clusters = result.get('n_clusters', 0)
                noise_ratio = result.get('noise_ratio', 1.0)
                if 2 <= n_clusters <= 10 and noise_ratio < 0.5:
                    good_results.append(result)
            
            if good_results:
                # Sort by optimization metric
                if self.optimization_metric == 'davies_bouldin':
                    good_results.sort(key=lambda x: x[self.optimization_metric])
                else:
                    good_results.sort(key=lambda x: x[self.optimization_metric], reverse=True)
                
                best = good_results[0]
                recommendations.append(f"Best parameters: eps={best['params']['eps']:.3f}, "
                                     f"min_samples={best['params']['min_samples']}")
                recommendations.append(f"Results in {best['n_clusters']} clusters with "
                                     f"{best['noise_ratio']*100:.1f}% noise points")
        
        return recommendations
    
    def _generate_agglomerative_recommendations(self, all_results: List[Dict]) -> List[str]:
        """Generate recommendations for Agglomerative clustering"""
        recommendations = []
        
        if len(all_results) > 0:
            # Group by linkage method
            linkage_performance = {}
            for result in all_results:
                linkage = result['params']['linkage']
                score = result[self.optimization_metric]
                if not np.isnan(score):
                    if linkage not in linkage_performance:
                        linkage_performance[linkage] = []
                    linkage_performance[linkage].append(score)
            
            # Find best linkage method
            if linkage_performance:
                avg_scores = {linkage: np.mean(scores) 
                             for linkage, scores in linkage_performance.items()}
                
                if self.optimization_metric == 'davies_bouldin':
                    best_linkage = min(avg_scores.keys(), key=lambda x: avg_scores[x])
                else:
                    best_linkage = max(avg_scores.keys(), key=lambda x: avg_scores[x])
                
                recommendations.append(f"Best linkage method: {best_linkage}")
        
        return recommendations
    
    def _generate_gmm_recommendations(self, all_results: List[Dict]) -> List[str]:
        """Generate recommendations for Gaussian Mixture Model"""
        recommendations = []
        
        if len(all_results) > 0:
            # Find best based on BIC (lower is better)
            bic_results = [r for r in all_results if 'bic' in r and not np.isnan(r['bic'])]
            if bic_results:
                best_bic = min(bic_results, key=lambda x: x['bic'])
                recommendations.append(f"Best BIC score with {best_bic['params']['n_components']} components, "
                                     f"{best_bic['params']['covariance_type']} covariance")
            
            # Find best based on optimization metric
            valid_results = [r for r in all_results if not np.isnan(r[self.optimization_metric])]
            if valid_results:
                if self.optimization_metric == 'davies_bouldin':
                    best_metric = min(valid_results, key=lambda x: x[self.optimization_metric])
                else:
                    best_metric = max(valid_results, key=lambda x: x[self.optimization_metric])
                
                recommendations.append(f"Best {self.optimization_metric} with "
                                     f"{best_metric['params']['n_components']} components")
        
        return recommendations
    
    def create_optimization_report(self, result: OptimizationResult) -> str:
        """Create a detailed optimization report"""
        report = f"""
=== {result.algorithm} Hyperparameter Optimization Report ===

Optimization Metric: {result.optimization_metric}
Best Score: {result.best_score:.4f}
Best Parameters: {result.best_params}

Total Configurations Tested: {len(result.all_results)}
Parameter Grid: {result.parameter_grid}

Recommendations:
"""
        for i, rec in enumerate(result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
Top 5 Configurations:
"""
        # Sort results by optimization metric
        sorted_results = result.all_results.copy()
        if result.optimization_metric == 'davies_bouldin':
            sorted_results.sort(key=lambda x: x[result.optimization_metric] 
                               if not np.isnan(x[result.optimization_metric]) else np.inf)
        else:
            sorted_results.sort(key=lambda x: x[result.optimization_metric] 
                               if not np.isnan(x[result.optimization_metric]) else -np.inf, reverse=True)
        
        for i, result_item in enumerate(sorted_results[:5], 1):
            report += f"{i}. Params: {result_item['params']} -> "
            report += f"{result.optimization_metric}: {result_item[result.optimization_metric]:.4f}\n"
        
        return report
    
    def visualize_optimization_results(self, result: OptimizationResult) -> Figure:
        """Create visualization of optimization results"""
        if result.algorithm == 'K-Means':
            return self._visualize_kmeans_optimization(result)
        elif result.algorithm == 'DBSCAN':
            return self._visualize_dbscan_optimization(result)
        elif result.algorithm == 'Agglomerative':
            return self._visualize_agglomerative_optimization(result)
        elif result.algorithm == 'Gaussian Mixture':
            return self._visualize_gmm_optimization(result)
        else:
            return self._visualize_generic_optimization(result)
    
    def _visualize_kmeans_optimization(self, result: OptimizationResult) -> Figure:
        """Visualize K-Means optimization results"""
        fig = Figure(figsize=(15, 10))
        
        # Extract k values and scores
        k_values = []
        scores = []
        for res in result.all_results:
            k = res['params']['n_clusters']
            score = res[result.optimization_metric]
            if not np.isnan(score):
                k_values.append(k)
                scores.append(score)
        
        if not k_values:
            return fig
        
        # Create elbow plot
        ax1 = fig.add_subplot(2, 2, 1)
        
        # Group by k and get average scores
        k_to_scores = {}
        for k, score in zip(k_values, scores):
            if k not in k_to_scores:
                k_to_scores[k] = []
            k_to_scores[k].append(score)
        
        avg_k_values = sorted(k_to_scores.keys())
        avg_scores = [np.mean(k_to_scores[k]) for k in avg_k_values]
        std_scores = [np.std(k_to_scores[k]) for k in avg_k_values]
        
        ax1.plot(avg_k_values, avg_scores, 'bo-', linewidth=2, markersize=8)
        ax1.fill_between(avg_k_values, 
                        np.array(avg_scores) - np.array(std_scores),
                        np.array(avg_scores) + np.array(std_scores),
                        alpha=0.2)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel(f'{result.optimization_metric.title()} Score')
        ax1.set_title('Elbow Plot for K-Means Optimization')
        ax1.grid(True, alpha=0.3)
        
        # Mark best k
        best_k = result.best_params['n_clusters']
        best_score = result.best_score
        ax1.plot(best_k, best_score, 'ro', markersize=12, 
                markerfacecolor='none', markeredgewidth=3)
        ax1.annotate(f'Best k={best_k}', xy=(best_k, best_score), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Create heatmap of parameter combinations
        ax2 = fig.add_subplot(2, 2, 2)
        
        # Prepare data for heatmap
        init_methods = sorted(set(res['params']['init'] for res in result.all_results))
        k_vals = sorted(set(res['params']['n_clusters'] for res in result.all_results))
        
        heatmap_data = np.full((len(init_methods), len(k_vals)), np.nan)
        
        for res in result.all_results:
            if not np.isnan(res[result.optimization_metric]):
                i = init_methods.index(res['params']['init'])
                j = k_vals.index(res['params']['n_clusters'])
                heatmap_data[i, j] = res[result.optimization_metric]
        
        im = ax2.imshow(heatmap_data, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(k_vals)))
        ax2.set_xticklabels(k_vals)
        ax2.set_yticks(range(len(init_methods)))
        ax2.set_yticklabels(init_methods)
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Initialization Method')
        ax2.set_title('Parameter Combination Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label=f'{result.optimization_metric.title()} Score')
        
        # Score distribution
        ax3 = fig.add_subplot(2, 2, 3)
        valid_scores = [res[result.optimization_metric] for res in result.all_results 
                       if not np.isnan(res[result.optimization_metric])]
        ax3.hist(valid_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(result.best_score, color='red', linestyle='--', linewidth=2, 
                   label=f'Best Score: {result.best_score:.4f}')
        ax3.set_xlabel(f'{result.optimization_metric.title()} Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Score Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Recommendations text
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        rec_text = "Recommendations:\n\n"
        for i, rec in enumerate(result.recommendations, 1):
            rec_text += f"{i}. {rec}\n"
        
        ax4.text(0.05, 0.95, rec_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        fig.tight_layout()
        return fig
    
    def _visualize_dbscan_optimization(self, result: OptimizationResult) -> Figure:
        """Visualize DBSCAN optimization results"""
        fig = Figure(figsize=(15, 10))
        
        # Extract parameters and scores
        eps_values = []
        min_samples_values = []
        scores = []
        n_clusters = []
        noise_ratios = []
        
        for res in result.all_results:
            if not np.isnan(res[result.optimization_metric]):
                eps_values.append(res['params']['eps'])
                min_samples_values.append(res['params']['min_samples'])
                scores.append(res[result.optimization_metric])
                n_clusters.append(res.get('n_clusters', 0))
                noise_ratios.append(res.get('noise_ratio', 0))
        
        if not eps_values:
            return fig
        
        # 2D parameter space heatmap
        ax1 = fig.add_subplot(2, 2, 1)
        
        unique_eps = sorted(set(eps_values))
        unique_min_samples = sorted(set(min_samples_values))
        
        heatmap_data = np.full((len(unique_min_samples), len(unique_eps)), np.nan)
        
        for eps, min_samp, score in zip(eps_values, min_samples_values, scores):
            i = unique_min_samples.index(min_samp)
            j = unique_eps.index(eps)
            heatmap_data[i, j] = score
        
        im = ax1.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')
        ax1.set_xticks(range(0, len(unique_eps), max(1, len(unique_eps)//10)))
        ax1.set_xticklabels([f'{unique_eps[i]:.3f}' for i in range(0, len(unique_eps), max(1, len(unique_eps)//10))])
        ax1.set_yticks(range(len(unique_min_samples)))
        ax1.set_yticklabels(unique_min_samples)
        ax1.set_xlabel('eps')
        ax1.set_ylabel('min_samples')
        ax1.set_title(f'{result.optimization_metric.title()} Score Heatmap')
        plt.colorbar(im, ax=ax1)
        
        # Mark best parameters
        best_eps = result.best_params['eps']
        best_min_samples = result.best_params['min_samples']
        j_best = unique_eps.index(best_eps)
        i_best = unique_min_samples.index(best_min_samples)
        ax1.plot(j_best, i_best, 'ro', markersize=10, markerfacecolor='none', markeredgewidth=3)
        
        # Scatter plot: eps vs score
        ax2 = fig.add_subplot(2, 2, 2)
        scatter = ax2.scatter(eps_values, scores, c=noise_ratios, cmap='Reds', alpha=0.7)
        ax2.set_xlabel('eps')
        ax2.set_ylabel(f'{result.optimization_metric.title()} Score')
        ax2.set_title('eps vs Score (color = noise ratio)')
        plt.colorbar(scatter, ax=ax2, label='Noise Ratio')
        
        # Number of clusters distribution
        ax3 = fig.add_subplot(2, 2, 3)
        cluster_counts = {}
        for n in n_clusters:
            cluster_counts[n] = cluster_counts.get(n, 0) + 1
        
        bars = ax3.bar(cluster_counts.keys(), cluster_counts.values(), alpha=0.7, color='lightblue')
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Cluster Counts')
        
        # Recommendations
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        rec_text = "DBSCAN Optimization Results:\n\n"
        rec_text += f"Best Parameters:\n"
        rec_text += f"  eps: {result.best_params['eps']:.4f}\n"
        rec_text += f"  min_samples: {result.best_params['min_samples']}\n\n"
        
        rec_text += "Recommendations:\n"
        for i, rec in enumerate(result.recommendations, 1):
            rec_text += f"{i}. {rec}\n"
        
        ax4.text(0.05, 0.95, rec_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        fig.tight_layout()
        return fig
    
    def _visualize_agglomerative_optimization(self, result: OptimizationResult) -> Figure:
        """Visualize Agglomerative clustering optimization results"""
        return self._visualize_generic_optimization(result)
    
    def _visualize_gmm_optimization(self, result: OptimizationResult) -> Figure:
        """Visualize Gaussian Mixture Model optimization results"""
        fig = Figure(figsize=(15, 10))
        
        # Extract data
        n_components = []
        scores = []
        bic_scores = []
        aic_scores = []
        cov_types = []
        
        for res in result.all_results:
            if not np.isnan(res[result.optimization_metric]):
                n_components.append(res['params']['n_components'])
                scores.append(res[result.optimization_metric])
                bic_scores.append(res.get('bic', np.nan))
                aic_scores.append(res.get('aic', np.nan))
                cov_types.append(res['params']['covariance_type'])
        
        # Plot 1: Information Criteria
        ax1 = fig.add_subplot(2, 2, 1)
        
        unique_n_comp = sorted(set(n_components))
        unique_cov_types = sorted(set(cov_types))
        
        for cov_type in unique_cov_types:
            cov_bic = []
            cov_aic = []
            cov_n_comp = []
            
            for n_comp, bic, aic, ct in zip(n_components, bic_scores, aic_scores, cov_types):
                if ct == cov_type and not np.isnan(bic) and not np.isnan(aic):
                    cov_n_comp.append(n_comp)
                    cov_bic.append(bic)
                    cov_aic.append(aic)
            
            if cov_n_comp:
                ax1.plot(cov_n_comp, cov_bic, 'o-', label=f'{cov_type} (BIC)', alpha=0.7)
        
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('BIC Score')
        ax1.set_title('BIC Scores by Covariance Type')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Optimization metric by covariance type
        ax2 = fig.add_subplot(2, 2, 2)
        
        cov_type_scores = {}
        for score, cov_type in zip(scores, cov_types):
            if cov_type not in cov_type_scores:
                cov_type_scores[cov_type] = []
            cov_type_scores[cov_type].append(score)
        
        box_data = [cov_type_scores[ct] for ct in unique_cov_types]
        bp = ax2.boxplot(box_data, labels=unique_cov_types)
        ax2.set_xlabel('Covariance Type')
        ax2.set_ylabel(f'{result.optimization_metric.title()} Score')
        ax2.set_title('Score Distribution by Covariance Type')
        ax2.tick_params(axis='x', rotation=45)
        
        # Continue with other plots...
        # (Additional plots can be added here)
        
        fig.tight_layout()
        return fig
    
    def _visualize_generic_optimization(self, result: OptimizationResult) -> Figure:
        """Generic visualization for optimization results"""
        fig = Figure(figsize=(12, 8))
        
        # Score distribution
        ax1 = fig.add_subplot(2, 2, 1)
        valid_scores = [res[result.optimization_metric] for res in result.all_results 
                       if not np.isnan(res[result.optimization_metric])]
        ax1.hist(valid_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(result.best_score, color='red', linestyle='--', linewidth=2,
                   label=f'Best Score: {result.best_score:.4f}')
        ax1.set_xlabel(f'{result.optimization_metric.title()} Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Score Distribution')
        ax1.legend()
        
        # Parameter importance (if applicable)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.text(0.5, 0.5, 'Parameter Analysis\n(Implementation Specific)', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax2.set_title('Parameter Analysis')
        
        # Top configurations
        ax3 = fig.add_subplot(2, 1, 2)
        ax3.axis('off')
        
        # Sort results
        sorted_results = result.all_results.copy()
        if result.optimization_metric == 'davies_bouldin':
            sorted_results.sort(key=lambda x: x[result.optimization_metric] 
                               if not np.isnan(x[result.optimization_metric]) else np.inf)
        else:
            sorted_results.sort(key=lambda x: x[result.optimization_metric] 
                               if not np.isnan(x[result.optimization_metric]) else -np.inf, reverse=True)
        
        text = f"Top 10 Configurations for {result.algorithm}:\n\n"
        for i, res in enumerate(sorted_results[:10], 1):
            text += f"{i:2d}. {res['params']} -> {result.optimization_metric}: {res[result.optimization_metric]:.4f}\n"
        
        ax3.text(0.05, 0.95, text, transform=ax3.transAxes, verticalalignment='top', 
                fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        fig.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_blobs
    
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                          random_state=42, cluster_std=1.5)
    
    print("=== Automated Hyperparameter Optimization Demo ===")
    
    # Initialize optimizer
    optimizer = ClusteringHyperparameterOptimizer(optimization_metric='silhouette')
    
    # Optimize K-Means
    print("\n1. Optimizing K-Means...")
    kmeans_result = optimizer.optimize_kmeans(X, k_range=(2, 8))
    print(f"Best K-Means parameters: {kmeans_result.best_params}")
    print(f"Best score: {kmeans_result.best_score:.4f}")
    
    # Generate report
    report = optimizer.create_optimization_report(kmeans_result)
    print(report)
    
    # Optimize DBSCAN
    print("\n2. Optimizing DBSCAN...")
    dbscan_result = optimizer.optimize_dbscan(X)
    print(f"Best DBSCAN parameters: {dbscan_result.best_params}")
    print(f"Best score: {dbscan_result.best_score:.4f}")
    
    print("\n=== Optimization Features ===")
    print("✓ Automated parameter grid search")
    print("✓ Multiple evaluation metrics")
    print("✓ Statistical recommendations")
    print("✓ Comprehensive result visualization")
    print("✓ Algorithm-specific optimization strategies")