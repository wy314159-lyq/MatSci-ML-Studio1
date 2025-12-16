"""
Clustering Interpretability and Explanation System
Provides comprehensive explanations and interpretations of clustering results
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# Configure matplotlib to handle Unicode properly
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
import time

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10  # Small value to prevent division by zero
MIN_CLUSTER_SIZE = 2
DEFAULT_MAX_FEATURES = 10
DENSITY_SAMPLE_SIZE = 100


class ClusterProfiler:
    """
    Creates detailed profiles for each cluster including statistical summaries,
    feature importance, and characteristic patterns
    """
    
    def __init__(self):
        self.cluster_profiles = {}
        self.global_stats = {}
        
    def create_cluster_profiles(self, X, labels, feature_names=None):
        """
        Create comprehensive profiles for each cluster
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Original data
        labels : array-like of shape (n_samples,)
            Cluster labels
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        profiles : dict
            Comprehensive cluster profiles
        """
        
        X = np.asarray(X)
        labels = np.asarray(labels)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Global statistics for comparison
        self.global_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'median': np.median(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0)
        }
        
        unique_labels = np.unique(labels)
        profiles = {}
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_mask = labels == label
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            profile = self._create_single_cluster_profile(
                cluster_data, label, feature_names, X, labels
            )
            profiles[f'cluster_{label}'] = profile
        
        self.cluster_profiles = profiles
        return profiles
    
    def _create_single_cluster_profile(self, cluster_data, label, feature_names, X_full, labels_full):
        """Create profile for a single cluster"""
        n_samples_cluster = len(cluster_data)
        n_samples_total = len(X_full)
        
        profile = {
            'basic_info': {
                'label': label,
                'size': n_samples_cluster,
                'percentage': (n_samples_cluster / n_samples_total) * 100,
                'density_rank': self._calculate_density_rank(cluster_data, X_full, labels_full, label)
            },
            
            'statistical_summary': self._compute_statistical_summary(cluster_data, feature_names),
            
            'feature_characteristics': self._analyze_feature_characteristics(
                cluster_data, feature_names, self.global_stats
            ),
            
            'cluster_quality': self._assess_cluster_quality(cluster_data, X_full, labels_full, label),
            
            'interpretability_metrics': self._compute_interpretability_metrics(cluster_data, X_full, labels_full)
        }
        
        return profile
    
    def _compute_statistical_summary(self, cluster_data, feature_names):
        """Compute comprehensive statistical summary for cluster"""
        summary = {}

        for i, feature_name in enumerate(feature_names):
            feature_data = cluster_data[:, i]
            feature_mean = np.mean(feature_data)
            feature_std = np.std(feature_data)

            # Safe coefficient of variation calculation
            if np.abs(feature_mean) > EPSILON:
                cv = feature_std / np.abs(feature_mean)
            else:
                cv = 0.0

            summary[feature_name] = {
                'mean': feature_mean,
                'std': feature_std,
                'median': np.median(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'q25': np.percentile(feature_data, 25),
                'q75': np.percentile(feature_data, 75),
                'iqr': np.percentile(feature_data, 75) - np.percentile(feature_data, 25),
                'skewness': stats.skew(feature_data) if len(feature_data) > 2 else 0.0,
                'kurtosis': stats.kurtosis(feature_data) if len(feature_data) > 3 else 0.0,
                'coefficient_of_variation': cv
            }

        return summary

    def _analyze_feature_characteristics(self, cluster_data, feature_names, global_stats):
        """Analyze how cluster features compare to global statistics"""
        characteristics = {}

        for i, feature_name in enumerate(feature_names):
            cluster_mean = np.mean(cluster_data[:, i])
            global_mean = global_stats['mean'][i]
            global_std = global_stats['std'][i]

            # Z-score relative to global distribution with safe division
            z_score = (cluster_mean - global_mean) / max(global_std, EPSILON)

            # Feature importance for this cluster with safe division
            cluster_std = np.std(cluster_data[:, i])
            if global_std > EPSILON:
                importance = abs(z_score) * max(0, 1 - cluster_std / global_std)
            else:
                importance = 0.0

            characteristics[feature_name] = {
                'cluster_mean': cluster_mean,
                'global_mean': global_mean,
                'z_score': z_score,
                'relative_importance': importance,
                'distinguishing_feature': abs(z_score) > 1.5,
                'high_variability': cluster_std > global_std * 1.2 if global_std > EPSILON else False,
                'low_variability': cluster_std < global_std * 0.5 if global_std > EPSILON else False,
                'deviation_direction': 'high' if z_score > 0.5 else 'low' if z_score < -0.5 else 'average'
            }

        return characteristics
    
    def _assess_cluster_quality(self, cluster_data, X_full, labels_full, label):
        """Assess the quality and cohesion of the cluster"""
        quality_metrics = {}
        
        try:
            # Internal cohesion
            if len(cluster_data) > 1:
                centroid = np.mean(cluster_data, axis=0)
                distances_to_centroid = np.linalg.norm(cluster_data - centroid, axis=1)
                
                quality_metrics['cohesion'] = {
                    'mean_distance_to_center': np.mean(distances_to_centroid),
                    'std_distance_to_center': np.std(distances_to_centroid),
                    'max_distance_to_center': np.max(distances_to_centroid),
                    'compactness_score': 1.0 / (1.0 + np.mean(distances_to_centroid))
                }
            
            # Separation from other clusters
            other_clusters_data = X_full[labels_full != label]
            if len(other_clusters_data) > 0 and len(cluster_data) > 0:
                centroid = np.mean(cluster_data, axis=0)
                other_centroid = np.mean(other_clusters_data, axis=0)
                separation_distance = np.linalg.norm(centroid - other_centroid)
                
                quality_metrics['separation'] = {
                    'distance_to_others': separation_distance,
                    'separation_score': separation_distance / (separation_distance + np.mean(distances_to_centroid) if 'distances_to_centroid' in locals() else 1.0)
                }
            
            # Silhouette analysis for this cluster
            if len(np.unique(labels_full)) > 1:
                cluster_indices = labels_full == label
                if np.sum(cluster_indices) > 0:
                    silhouette_scores = silhouette_samples(X_full, labels_full)
                    cluster_silhouette = silhouette_scores[cluster_indices]
                    
                    quality_metrics['silhouette'] = {
                        'mean_silhouette': np.mean(cluster_silhouette),
                        'std_silhouette': np.std(cluster_silhouette),
                        'min_silhouette': np.min(cluster_silhouette),
                        'negative_silhouette_ratio': np.sum(cluster_silhouette < 0) / len(cluster_silhouette)
                    }
        
        except Exception as e:
            quality_metrics['error'] = f"Quality assessment failed: {str(e)}"
        
        return quality_metrics
    
    def _calculate_density_rank(self, cluster_data, X_full, labels_full, label):
        """Calculate density rank of cluster relative to others using memory-efficient sampling"""
        try:
            unique_labels = np.unique(labels_full)
            cluster_densities = []
            rng = np.random.default_rng(42)  # Use local RNG for reproducibility

            for other_label in unique_labels:
                if other_label == -1:  # Skip noise
                    continue

                other_cluster_data = X_full[labels_full == other_label]
                if len(other_cluster_data) > 1:
                    # Estimate density using sampled nearest neighbor distances (more memory efficient)
                    n_samples = min(DENSITY_SAMPLE_SIZE, len(other_cluster_data))
                    sample_indices = rng.choice(len(other_cluster_data), n_samples, replace=False)
                    sample_data = other_cluster_data[sample_indices]

                    # Use nearest neighbor distances instead of full pairwise matrix
                    from sklearn.neighbors import NearestNeighbors
                    k = min(5, n_samples - 1)
                    if k > 0:
                        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(sample_data)
                        distances, _ = nbrs.kneighbors(sample_data)
                        avg_distance = np.mean(distances[:, 1:])  # Exclude self
                        density = 1.0 / (avg_distance + EPSILON)
                        cluster_densities.append((other_label, density))

            # Sort by density and find rank
            cluster_densities.sort(key=lambda x: x[1], reverse=True)
            for rank, (cluster_label, density) in enumerate(cluster_densities):
                if cluster_label == label:
                    return rank + 1

            return len(cluster_densities)  # Fallback

        except Exception as e:
            logger.debug(f"Density rank calculation failed: {e}")
            return 0

    def _compute_interpretability_metrics(self, cluster_data, X_full, labels_full):
        """Compute metrics that help interpret the cluster"""
        metrics = {}

        try:
            # Homogeneity within cluster
            if len(cluster_data) > 1:
                # Feature-wise homogeneity with safe division
                feature_homogeneity = []
                for i in range(cluster_data.shape[1]):
                    feature_values = cluster_data[:, i]
                    feature_mean = np.mean(feature_values)
                    feature_std = np.std(feature_values)
                    if np.abs(feature_mean) > EPSILON:
                        cv = feature_std / np.abs(feature_mean)
                    else:
                        cv = 0.0
                    feature_homogeneity.append(1.0 / (1.0 + cv))  # Higher is more homogeneous

                metrics['homogeneity'] = {
                    'feature_homogeneity': feature_homogeneity,
                    'mean_homogeneity': np.mean(feature_homogeneity),
                    'least_homogeneous_feature': int(np.argmin(feature_homogeneity)),
                    'most_homogeneous_feature': int(np.argmax(feature_homogeneity))
                }

            # Outlier detection within cluster
            if len(cluster_data) > 3:
                centroid = np.mean(cluster_data, axis=0)
                distances = np.linalg.norm(cluster_data - centroid, axis=1)

                q75 = np.percentile(distances, 75)
                q25 = np.percentile(distances, 25)
                iqr = q75 - q25
                outlier_threshold = q75 + 1.5 * iqr
                
                outliers = distances > outlier_threshold
                metrics['outliers'] = {
                    'n_outliers': np.sum(outliers),
                    'outlier_percentage': np.sum(outliers) / len(cluster_data) * 100,
                    'outlier_indices': np.where(outliers)[0].tolist(),
                    'outlier_threshold': outlier_threshold
                }
        
        except Exception as e:
            metrics['error'] = f"Interpretability metrics failed: {str(e)}"
        
        return metrics


class ClusterExplainer:
    """
    Provides explanations for why data points belong to specific clusters
    and what makes clusters distinct
    """
    
    def __init__(self):
        self.explanation_models = {}
        self.feature_importance_global = None
        
    def explain_clustering(self, X, labels, feature_names=None, method='decision_tree'):
        """
        Create explanations for the clustering results
        
        Parameters:
        -----------
        X : array-like
            Original data
        labels : array-like
            Cluster labels
        feature_names : list, optional
            Feature names
        method : str
            Explanation method: 'decision_tree', 'random_forest', 'rules'
            
        Returns:
        --------
        explanations : dict
            Cluster explanations
        """
        
        X = np.asarray(X)
        labels = np.asarray(labels)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        explanations = {
            'method': method,
            'global_feature_importance': {},
            'cluster_rules': {},
            'decision_boundaries': {},
            'feature_distributions': {}
        }
        
        # Global feature importance for clustering
        explanations['global_feature_importance'] = self._compute_global_feature_importance(
            X, labels, feature_names, method
        )
        
        # Cluster-specific rules and explanations
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            explanations['cluster_rules'][f'cluster_{label}'] = self._explain_single_cluster(
                X, labels, label, feature_names, method
            )
        
        # Feature distributions by cluster
        explanations['feature_distributions'] = self._analyze_feature_distributions(
            X, labels, feature_names
        )
        
        return explanations
    
    def _compute_global_feature_importance(self, X, labels, feature_names, method):
        """Compute global feature importance for the clustering"""
        importance_results = {}
        
        try:
            if method == 'random_forest':
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, labels)
                
                # Feature importance from the model
                importances = rf.feature_importances_
                
                # Permutation importance for more robust estimates
                perm_importance = permutation_importance(rf, X, labels, n_repeats=5, random_state=42)
                
                importance_results = {
                    'feature_importance': dict(zip(feature_names, importances)),
                    'permutation_importance': dict(zip(feature_names, perm_importance.importances_mean)),
                    'importance_std': dict(zip(feature_names, perm_importance.importances_std)),
                    'top_features': [feature_names[i] for i in np.argsort(importances)[::-1][:10]]
                }
                
                self.explanation_models['global_rf'] = rf
                
            elif method == 'decision_tree':
                dt = DecisionTreeClassifier(max_depth=10, random_state=42)
                dt.fit(X, labels)
                
                importances = dt.feature_importances_
                
                importance_results = {
                    'feature_importance': dict(zip(feature_names, importances)),
                    'top_features': [feature_names[i] for i in np.argsort(importances)[::-1][:10]]
                }
                
                self.explanation_models['global_dt'] = dt
            
            # Statistical feature importance based on cluster separability
            statistical_importance = self._compute_statistical_importance(X, labels, feature_names)
            importance_results['statistical_importance'] = statistical_importance
            
        except Exception as e:
            importance_results['error'] = f"Global importance computation failed: {str(e)}"
        
        return importance_results
    
    def _compute_statistical_importance(self, X, labels, feature_names):
        """Compute statistical feature importance based on cluster separability"""
        importance_scores = {}
        
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return importance_scores
            
            for i, feature_name in enumerate(feature_names):
                feature_data = X[:, i]
                
                # Compute between-cluster variance vs within-cluster variance
                between_cluster_var = 0
                within_cluster_var = 0
                
                overall_mean = np.mean(feature_data)
                
                for label in unique_labels:
                    if label == -1:  # Skip noise
                        continue
                        
                    cluster_data = feature_data[labels == label]
                    if len(cluster_data) == 0:
                        continue
                    
                    cluster_mean = np.mean(cluster_data)
                    cluster_size = len(cluster_data)
                    
                    # Between-cluster variance
                    between_cluster_var += cluster_size * (cluster_mean - overall_mean) ** 2
                    
                    # Within-cluster variance
                    within_cluster_var += np.sum((cluster_data - cluster_mean) ** 2)
                
                # F-ratio as importance measure
                if within_cluster_var > 0:
                    f_ratio = (between_cluster_var / (n_clusters - 1)) / (within_cluster_var / (len(feature_data) - n_clusters))
                    importance_scores[feature_name] = f_ratio
                else:
                    importance_scores[feature_name] = 0
        
        except Exception as e:
            importance_scores['error'] = str(e)
        
        return importance_scores
    
    def _explain_single_cluster(self, X, labels, target_label, feature_names, method):
        """Create explanation for a single cluster"""
        cluster_explanation = {}
        
        try:
            # Create binary classification problem: this cluster vs all others
            binary_labels = (labels == target_label).astype(int)
            
            if method == 'decision_tree':
                dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
                dt.fit(X, binary_labels)
                
                # Extract rules from decision tree
                cluster_explanation['decision_rules'] = self._extract_tree_rules(dt, feature_names, target_label)
                cluster_explanation['feature_importance'] = dict(zip(feature_names, dt.feature_importances_))
                
                self.explanation_models[f'cluster_{target_label}_dt'] = dt
                
            elif method == 'random_forest':
                rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                rf.fit(X, binary_labels)
                
                cluster_explanation['feature_importance'] = dict(zip(feature_names, rf.feature_importances_))
                cluster_explanation['confidence'] = np.mean(rf.predict_proba(X[labels == target_label])[:, 1])
                
                self.explanation_models[f'cluster_{target_label}_rf'] = rf
            
            # Statistical rules
            cluster_explanation['statistical_rules'] = self._create_statistical_rules(
                X, labels, target_label, feature_names
            )
            
            # Cluster characteristics summary
            cluster_explanation['characteristics'] = self._summarize_cluster_characteristics(
                X, labels, target_label, feature_names
            )
            
        except Exception as e:
            cluster_explanation['error'] = f"Cluster explanation failed: {str(e)}"
        
        return cluster_explanation
    
    def _extract_tree_rules(self, tree_model, feature_names, target_label):
        """Extract human-readable rules from decision tree"""
        tree = tree_model.tree_
        rules = []
        
        def extract_rules_recursive(node, depth, condition):
            indent = "  " * depth
            
            if tree.feature[node] != -2:  # Not a leaf
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                # Left child (condition is true)
                left_condition = condition + f" AND {feature} <= {threshold:.3f}" if condition else f"{feature} <= {threshold:.3f}"
                extract_rules_recursive(tree.children_left[node], depth + 1, left_condition)
                
                # Right child (condition is false)
                right_condition = condition + f" AND {feature} > {threshold:.3f}" if condition else f"{feature} > {threshold:.3f}"
                extract_rules_recursive(tree.children_right[node], depth + 1, right_condition)
            else:
                # Leaf node
                samples = tree.n_node_samples[node]
                values = tree.value[node][0]
                
                if len(values) >= 2:
                    cluster_samples = values[1]  # Samples belonging to target cluster
                    confidence = cluster_samples / samples if samples > 0 else 0
                    
                    if confidence > 0.5:  # Only include rules that predict the target cluster
                        rule = {
                            'condition': condition if condition else "Always",
                            'confidence': confidence,
                            'support': samples,
                            'cluster_samples': int(cluster_samples)
                        }
                        rules.append(rule)
        
        extract_rules_recursive(0, 0, "")
        
        # Sort rules by confidence and support
        rules.sort(key=lambda x: (x['confidence'], x['support']), reverse=True)
        
        return rules[:10]  # Return top 10 rules
    
    def _create_statistical_rules(self, X, labels, target_label, feature_names):
        """Create statistical rules for cluster membership"""
        cluster_data = X[labels == target_label]
        other_data = X[labels != target_label]
        
        rules = []
        
        if len(cluster_data) == 0 or len(other_data) == 0:
            return rules
        
        for i, feature_name in enumerate(feature_names):
            cluster_values = cluster_data[:, i]
            other_values = other_data[:, i]
            
            # Statistical tests
            try:
                # T-test for mean difference
                t_stat, p_value = stats.ttest_ind(cluster_values, other_values)
                
                if p_value < 0.05:  # Significant difference
                    cluster_mean = np.mean(cluster_values)
                    other_mean = np.mean(other_values)
                    
                    if cluster_mean > other_mean:
                        direction = "higher"
                    else:
                        direction = "lower"
                    
                    rule = {
                        'feature': feature_name,
                        'type': 'mean_difference',
                        'description': f"{feature_name} is significantly {direction} in this cluster",
                        'cluster_mean': cluster_mean,
                        'other_mean': other_mean,
                        'p_value': p_value,
                        'effect_size': abs(cluster_mean - other_mean) / np.sqrt(np.var(other_values))
                    }
                    rules.append(rule)
                
                # Range-based rules
                cluster_q25, cluster_q75 = np.percentile(cluster_values, [25, 75])
                other_q25, other_q75 = np.percentile(other_values, [25, 75])
                
                # Check if cluster has a distinctive range
                if cluster_q75 < other_q25 or cluster_q25 > other_q75:
                    rule = {
                        'feature': feature_name,
                        'type': 'range_separation',
                        'description': f"{feature_name} has distinctive range [{cluster_q25:.3f}, {cluster_q75:.3f}] in this cluster",
                        'cluster_range': [cluster_q25, cluster_q75],
                        'other_range': [other_q25, other_q75]
                    }
                    rules.append(rule)
                
            except Exception:
                continue
        
        return rules
    
    def _summarize_cluster_characteristics(self, X, labels, target_label, feature_names):
        """Summarize key characteristics of the cluster"""
        cluster_data = X[labels == target_label]
        
        if len(cluster_data) == 0:
            return {}
        
        characteristics = {
            'size': len(cluster_data),
            'percentage_of_total': len(cluster_data) / len(X) * 100,
            'centroid': dict(zip(feature_names, np.mean(cluster_data, axis=0))),
            'spread': dict(zip(feature_names, np.std(cluster_data, axis=0))),
            'dominant_features': [],
            'distinctive_patterns': []
        }
        
        # Find dominant features (highest absolute z-scores from global mean)
        global_mean = np.mean(X, axis=0)
        global_std = np.std(X, axis=0)
        cluster_mean = np.mean(cluster_data, axis=0)
        
        z_scores = []
        for i, feature_name in enumerate(feature_names):
            if global_std[i] > 0:
                z_score = (cluster_mean[i] - global_mean[i]) / global_std[i]
                z_scores.append((feature_name, z_score))
        
        # Sort by absolute z-score
        z_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature_name, z_score in z_scores[:5]:  # Top 5 most distinctive features
            if abs(z_score) > 0.5:  # Only include meaningfully different features
                characteristics['dominant_features'].append({
                    'feature': feature_name,
                    'z_score': z_score,
                    'direction': 'high' if z_score > 0 else 'low',
                    'description': f"{feature_name} is {'above' if z_score > 0 else 'below'} average by {abs(z_score):.2f} standard deviations"
                })
        
        return characteristics
    
    def _analyze_feature_distributions(self, X, labels, feature_names):
        """Analyze feature distributions across clusters"""
        distributions = {}
        unique_labels = np.unique(labels)
        
        for i, feature_name in enumerate(feature_names):
            feature_distributions = {}
            
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                    
                cluster_data = X[labels == label, i]
                if len(cluster_data) > 0:
                    feature_distributions[f'cluster_{label}'] = {
                        'mean': np.mean(cluster_data),
                        'std': np.std(cluster_data),
                        'min': np.min(cluster_data),
                        'max': np.max(cluster_data),
                        'quartiles': np.percentile(cluster_data, [25, 50, 75]).tolist()
                    }
            
            distributions[feature_name] = feature_distributions
        
        return distributions


class ClusterVisualExplainer:
    """
    Creates visual explanations for clustering results
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        
    def create_explanation_visualizations(self, X, labels, feature_names=None, max_features=10):
        """
        Create comprehensive visual explanations
        
        Parameters:
        -----------
        X : array-like
            Original data
        labels : array-like
            Cluster labels
        feature_names : list, optional
            Feature names
        max_features : int
            Maximum number of features to visualize
            
        Returns:
        --------
        figures : dict
            Dictionary containing matplotlib figures
        """
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        figures = {}
        
        try:
            # 1. Cluster overview with dimensionality reduction
            figures['cluster_overview'] = self._create_cluster_overview(X, labels)
            
            # 2. Feature importance visualization
            figures['feature_importance'] = self._create_feature_importance_plot(X, labels, feature_names)
            
            # 3. Feature distributions by cluster
            n_features_to_plot = min(max_features, len(feature_names))
            figures['feature_distributions'] = self._create_feature_distribution_plots(
                X, labels, feature_names[:n_features_to_plot]
            )
            
            # 4. Cluster profiles radar chart
            figures['cluster_profiles'] = self._create_cluster_profiles_radar(
                X, labels, feature_names[:min(8, len(feature_names))]  # Limit for readability
            )
            
            # 5. Silhouette analysis
            if len(np.unique(labels)) > 1:
                figures['silhouette_analysis'] = self._create_silhouette_plot(X, labels)
            
        except Exception as e:
            print(f"Visualization creation failed: {str(e)}")
        
        return figures
    
    def _create_cluster_overview(self, X, labels):
        """Create cluster overview with 2D projection"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA projection
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
            explained_var = pca.explained_variance_ratio_
        else:
            X_pca = X
            explained_var = [1.0, 1.0]
        
        # Plot PCA
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        axes[0].set_title(f'PCA Projection\n(Explained Variance: {explained_var[0]:.2f}, {explained_var[1]:.2f})')
        axes[0].set_xlabel(f'PC1 ({explained_var[0]:.2f})')
        axes[0].set_ylabel(f'PC2 ({explained_var[1]:.2f})')
        plt.colorbar(scatter, ax=axes[0])
        
        # t-SNE projection (if data is not too large)
        if X.shape[0] <= 5000 and X.shape[1] > 2:
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
                X_tsne = tsne.fit_transform(StandardScaler().fit_transform(X))
                
                scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
                axes[1].set_title('t-SNE Projection')
                axes[1].set_xlabel('t-SNE 1')
                axes[1].set_ylabel('t-SNE 2')
                plt.colorbar(scatter2, ax=axes[1])
            except Exception:
                # Fallback to PCA if t-SNE fails
                axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
                axes[1].set_title('PCA Projection (t-SNE failed)')
        else:
            axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
            axes[1].set_title('PCA Projection')
        
        plt.tight_layout()
        return fig
    
    def _create_feature_importance_plot(self, X, labels, feature_names):
        """Create feature importance visualization"""
        # Compute feature importance using Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, labels)
            importance_scores = rf.feature_importances_
        except Exception:
            # Fallback to statistical importance
            importance_scores = self._compute_statistical_importance_scores(X, labels)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        # Plot top features
        n_features_to_plot = min(20, len(sorted_features))
        y_pos = np.arange(n_features_to_plot)
        
        bars = ax.barh(y_pos, sorted_scores[:n_features_to_plot])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features[:n_features_to_plot])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance for Clustering')
        
        # Color bars by importance level
        colors = plt.cm.viridis(sorted_scores[:n_features_to_plot] / np.max(sorted_scores))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        return fig
    
    def _compute_statistical_importance_scores(self, X, labels):
        """Compute statistical feature importance scores"""
        unique_labels = np.unique(labels)
        n_features = X.shape[1]
        importance_scores = np.zeros(n_features)
        
        for i in range(n_features):
            feature_data = X[:, i]
            between_var = 0
            within_var = 0
            total_mean = np.mean(feature_data)
            
            for label in unique_labels:
                if label == -1:
                    continue
                cluster_data = feature_data[labels == label]
                if len(cluster_data) > 0:
                    cluster_mean = np.mean(cluster_data)
                    between_var += len(cluster_data) * (cluster_mean - total_mean) ** 2
                    within_var += np.sum((cluster_data - cluster_mean) ** 2)
            
            if within_var > 0:
                importance_scores[i] = between_var / within_var
            
        # Normalize scores
        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)
        
        return importance_scores
    
    def _create_feature_distribution_plots(self, X, labels, feature_names):
        """Create feature distribution plots by cluster"""
        n_features = len(feature_names)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for idx, feature_name in enumerate(feature_names):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            for i, label in enumerate(unique_labels):
                if label == -1:  # Skip noise
                    continue
                    
                cluster_data = X[labels == label, idx]
                if len(cluster_data) > 0:
                    ax.hist(cluster_data, alpha=0.6, label=f'Cluster {label}', 
                           color=colors[i], bins=20, density=True)
            
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {feature_name}')
            ax.legend()
        
        # Hide empty subplots
        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _create_cluster_profiles_radar(self, X, labels, feature_names):
        """Create radar chart for cluster profiles"""
        unique_labels = np.unique(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        
        if n_clusters == 0:
            return None
        
        # Compute cluster centroids (normalized)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        cluster_profiles = {}
        for label in unique_labels:
            if label == -1:
                continue
            cluster_data = X_scaled[labels == label]
            if len(cluster_data) > 0:
                cluster_profiles[label] = np.mean(cluster_data, axis=0)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(cluster_profiles)))
        
        for i, (label, profile) in enumerate(cluster_profiles.items()):
            values = np.concatenate((profile, [profile[0]]))  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {label}', 
                   color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names)
        ax.set_ylim(-3, 3)  # Standardized values typically range from -3 to 3
        ax.set_title('Cluster Profiles (Standardized Features)', size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def _create_silhouette_plot(self, X, labels):
        """Create silhouette analysis plot"""
        silhouette_scores = silhouette_samples(X, labels)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_lower = 10
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Skip noise
                continue
                
            cluster_silhouette_values = silhouette_scores[labels == label]
            cluster_silhouette_values.sort()
            
            size_cluster = len(cluster_silhouette_values)
            y_upper = y_lower + size_cluster
            
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                           0, cluster_silhouette_values,
                           facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
            y_lower = y_upper + 10
        
        ax.set_xlabel('Silhouette coefficient values')
        ax.set_ylabel('Cluster label')
        ax.set_title('Silhouette Analysis')
        
        # Add vertical line for average silhouette score
        silhouette_avg = np.mean(silhouette_scores)
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                  label=f'Average Score: {silhouette_avg:.3f}')
        ax.legend()
        
        plt.tight_layout()
        return fig


class ComprehensiveClusteringInterpreter:
    """
    Main interface for comprehensive clustering interpretability
    """
    
    def __init__(self):
        self.profiler = ClusterProfiler()
        self.explainer = ClusterExplainer()
        self.visual_explainer = ClusterVisualExplainer()
        
    def interpret_clustering(self, X, labels, feature_names=None, 
                           include_visualizations=True, explanation_method='random_forest'):
        """
        Complete clustering interpretation pipeline
        
        Parameters:
        -----------
        X : array-like
            Original data
        labels : array-like
            Cluster labels
        feature_names : list, optional
            Feature names
        include_visualizations : bool
            Whether to create visualizations
        explanation_method : str
            Method for explanations
            
        Returns:
        --------
        interpretation_results : dict
            Complete interpretation results
        """
        
        interpretation_results = {
            'summary': {},
            'cluster_profiles': {},
            'explanations': {},
            'visualizations': {},
            'interpretation_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Basic summary
            unique_labels = np.unique(labels)
            n_clusters = len([l for l in unique_labels if l != -1])
            n_noise = np.sum(labels == -1) if -1 in unique_labels else 0
            
            interpretation_results['summary'] = {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'cluster_sizes': [np.sum(labels == label) for label in unique_labels if label != -1]
            }
            
            # Cluster profiling
            print("Creating cluster profiles...")
            interpretation_results['cluster_profiles'] = self.profiler.create_cluster_profiles(
                X, labels, feature_names
            )
            
            # Explanations
            print("Generating cluster explanations...")
            interpretation_results['explanations'] = self.explainer.explain_clustering(
                X, labels, feature_names, explanation_method
            )
            
            # Visualizations
            if include_visualizations:
                print("Creating visualizations...")
                interpretation_results['visualizations'] = self.visual_explainer.create_explanation_visualizations(
                    X, labels, feature_names
                )
            
            # Generate natural language summary
            interpretation_results['natural_language_summary'] = self._generate_natural_language_summary(
                interpretation_results
            )
            
            interpretation_results['interpretation_time'] = time.time() - start_time
            
        except Exception as e:
            interpretation_results['error'] = f"Interpretation failed: {str(e)}"
        
        return interpretation_results
    
    def _generate_natural_language_summary(self, results):
        """Generate natural language summary of clustering results"""
        summary_lines = []
        
        try:
            basic_info = results['summary']
            summary_lines.append(f"Clustering Analysis Summary:")
            summary_lines.append(f"- Dataset contains {basic_info['n_samples']} samples with {basic_info['n_features']} features")
            summary_lines.append(f"- Found {basic_info['n_clusters']} distinct clusters")
            
            if basic_info['n_noise_points'] > 0:
                summary_lines.append(f"- {basic_info['n_noise_points']} points identified as noise/outliers")
            
            # Cluster size analysis
            cluster_sizes = basic_info['cluster_sizes']
            if cluster_sizes:
                avg_size = np.mean(cluster_sizes)
                summary_lines.append(f"- Average cluster size: {avg_size:.0f} samples")
                
                if max(cluster_sizes) > 3 * min(cluster_sizes):
                    summary_lines.append("- Clusters have significantly different sizes")
                else:
                    summary_lines.append("- Clusters have relatively similar sizes")
            
            # Feature importance insights
            explanations = results.get('explanations', {})
            global_importance = explanations.get('global_feature_importance', {})
            
            if 'top_features' in global_importance:
                top_features = global_importance['top_features'][:3]
                summary_lines.append(f"- Most important features for clustering: {', '.join(top_features)}")
            
            # Cluster quality assessment
            profiles = results.get('cluster_profiles', {})
            if profiles:
                high_quality_clusters = 0
                for cluster_name, profile in profiles.items():
                    silhouette_info = profile.get('cluster_quality', {}).get('silhouette', {})
                    if silhouette_info.get('mean_silhouette', 0) > 0.5:
                        high_quality_clusters += 1
                
                if high_quality_clusters >= len(profiles) * 0.7:
                    summary_lines.append("- Overall clustering quality is good (high silhouette scores)")
                elif high_quality_clusters >= len(profiles) * 0.4:
                    summary_lines.append("- Clustering quality is moderate")
                else:
                    summary_lines.append("- Clustering quality may need improvement")
            
        except Exception:
            summary_lines = ["Natural language summary generation failed"]
        
        return "\n".join(summary_lines)