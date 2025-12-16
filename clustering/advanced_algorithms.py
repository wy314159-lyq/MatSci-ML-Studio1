"""
Advanced Clustering Algorithms Extension
Implements state-of-the-art clustering algorithms with full integration to the existing module
"""

import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.cluster import AffinityPropagation
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings
from typing import Optional, Union, Tuple, Dict, List, Any

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10  # Small value to prevent division by zero
MIN_CLUSTER_SIZE = 2


class FuzzyCMeans(BaseEstimator, ClusterMixin):
    """
    Fuzzy C-Means clustering algorithm implementation
    
    Parameters:
    -----------
    n_clusters : int, default=3
        The number of clusters to form
    m : float, default=2.0
        Fuzzy coefficient (fuzziness parameter)
    max_iter : int, default=300
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility
    """
    
    def __init__(self, n_clusters=3, m=2.0, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """
        Fit the Fuzzy C-Means clustering model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster
        y : Ignored
            Not used, present here for API consistency by convention
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        X = check_array(X, accept_sparse='csr')
        
        # Initialize random state
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize membership matrix U randomly
        self.membership_matrix_ = np.random.rand(n_samples, self.n_clusters)
        # Normalize so sum of memberships for each point equals 1
        self.membership_matrix_ = self.membership_matrix_ / self.membership_matrix_.sum(axis=1)[:, np.newaxis]
        
        self.cluster_centers_ = np.zeros((self.n_clusters, n_features))
        self.inertia_ = float('inf')
        
        for iteration in range(self.max_iter):
            # Update cluster centers
            um = self.membership_matrix_ ** self.m
            self.cluster_centers_ = um.T.dot(X) / um.sum(axis=0)[:, np.newaxis]
            
            # Calculate distances from each point to each cluster center
            distances = cdist(X, self.cluster_centers_)
            
            # Update membership matrix
            new_membership = np.zeros((n_samples, self.n_clusters))
            
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    if distances[i, j] == 0:
                        new_membership[i, j] = 1.0
                        for k in range(self.n_clusters):
                            if k != j:
                                new_membership[i, k] = 0.0
                        break
                    else:
                        sum_term = 0.0
                        for k in range(self.n_clusters):
                            if distances[i, k] > 0:
                                sum_term += (distances[i, j] / distances[i, k]) ** (2 / (self.m - 1))
                        new_membership[i, j] = 1.0 / sum_term
            
            # Check for convergence
            if np.linalg.norm(new_membership - self.membership_matrix_) < self.tol:
                break
                
            self.membership_matrix_ = new_membership
        
        # Assign crisp labels (highest membership)
        self.labels_ = np.argmax(self.membership_matrix_, axis=1)
        
        # Calculate final inertia
        self.inertia_ = self._calculate_inertia(X)
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict
            
        Returns:
        --------
        labels : array of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        X = check_array(X, accept_sparse='csr')
        
        # Calculate distances to cluster centers
        distances = cdist(X, self.cluster_centers_)
        
        # Calculate membership matrix for new data
        n_samples = X.shape[0]
        membership = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            for j in range(self.n_clusters):
                if distances[i, j] == 0:
                    membership[i, j] = 1.0
                    for k in range(self.n_clusters):
                        if k != j:
                            membership[i, k] = 0.0
                    break
                else:
                    sum_term = 0.0
                    for k in range(self.n_clusters):
                        if distances[i, k] > 0:
                            sum_term += (distances[i, j] / distances[i, k]) ** (2 / (self.m - 1))
                    membership[i, j] = 1.0 / sum_term
        
        # Return crisp labels
        return np.argmax(membership, axis=1)
    
    def _calculate_inertia(self, X):
        """Calculate within-cluster sum of squares"""
        inertia = 0.0
        for i in range(self.n_clusters):
            mask = self.labels_ == i
            if np.sum(mask) > 0:
                cluster_data = X[mask]
                center = self.cluster_centers_[i]
                inertia += np.sum((cluster_data - center) ** 2)
        return inertia


class ConsensusKMeans(BaseEstimator, ClusterMixin):
    """
    Consensus K-Means clustering using ensemble of multiple K-means runs
    
    Parameters:
    -----------
    n_clusters : int, default=3
        The number of clusters to form
    n_estimators : int, default=10
        Number of K-means runs for consensus
    consensus_method : str, default='voting'
        Method for consensus: 'voting', 'co-occurrence', or 'evidence_accumulation'
    sample_fraction : float, default=0.8
        Fraction of samples to use in each run (for bootstrap sampling)
    feature_fraction : float, default=1.0
        Fraction of features to use in each run
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility
    """
    
    def __init__(self, n_clusters=3, n_estimators=10, consensus_method='voting',
                 sample_fraction=0.8, feature_fraction=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.consensus_method = consensus_method
        self.sample_fraction = sample_fraction
        self.feature_fraction = feature_fraction
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """
        Fit the Consensus K-Means clustering model
        """
        from sklearn.cluster import KMeans
        from sklearn.utils import resample
        
        X = check_array(X, accept_sparse='csr')
        n_samples, n_features = X.shape
        
        # Initialize random state
        np.random.seed(self.random_state)
        
        # Store ensemble results
        self.estimators_ = []
        self.labels_ensemble_ = []
        self.sample_indices_ensemble_ = []
        self.feature_indices_ensemble_ = []
        
        # Generate ensemble of clusterings
        for i in range(self.n_estimators):
            # Bootstrap sampling
            n_sample_subset = int(n_samples * self.sample_fraction)
            n_feature_subset = int(n_features * self.feature_fraction)
            
            # Sample indices
            sample_indices = np.random.choice(n_samples, n_sample_subset, replace=True)
            feature_indices = np.random.choice(n_features, n_feature_subset, replace=False)
            
            # Get subset data
            X_subset = X[np.ix_(sample_indices, feature_indices)]
            
            # Fit K-means
            kmeans = KMeans(n_clusters=self.n_clusters, 
                          random_state=self.random_state + i if self.random_state else None,
                          n_init=10)
            kmeans.fit(X_subset)
            
            # Store results
            self.estimators_.append(kmeans)
            self.sample_indices_ensemble_.append(sample_indices)
            self.feature_indices_ensemble_.append(feature_indices)
            
            # Map back to original indices for consensus
            full_labels = np.full(n_samples, -1)
            full_labels[sample_indices] = kmeans.labels_
            self.labels_ensemble_.append(full_labels)
        
        # Generate consensus clustering
        if self.consensus_method == 'voting':
            self.labels_ = self._consensus_voting(X)
        elif self.consensus_method == 'co-occurrence':
            self.labels_ = self._consensus_co_occurrence(X)
        elif self.consensus_method == 'evidence_accumulation':
            self.labels_ = self._consensus_evidence_accumulation(X)
        else:
            raise ValueError(f"Unknown consensus method: {self.consensus_method}")
        
        # Calculate final cluster centers
        self.cluster_centers_ = np.array([
            X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)
        ])
        
        # Calculate inertia
        self.inertia_ = sum(
            np.sum((X[self.labels_ == i] - self.cluster_centers_[i]) ** 2)
            for i in range(self.n_clusters)
            if np.sum(self.labels_ == i) > 0
        )
        
        return self
    
    def _consensus_voting(self, X):
        """Simple voting consensus"""
        n_samples = X.shape[0]
        vote_matrix = np.zeros((n_samples, self.n_clusters))
        
        for labels in self.labels_ensemble_:
            for i in range(n_samples):
                if labels[i] != -1:
                    vote_matrix[i, labels[i]] += 1
        
        return np.argmax(vote_matrix, axis=1)
    
    def _consensus_co_occurrence(self, X):
        """Co-occurrence matrix based consensus"""
        n_samples = X.shape[0]
        co_occurrence_matrix = np.zeros((n_samples, n_samples))
        
        # Build co-occurrence matrix
        for labels in self.labels_ensemble_:
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                        co_occurrence_matrix[i, j] += 1
                        co_occurrence_matrix[j, i] += 1
        
        # Normalize
        co_occurrence_matrix /= self.n_estimators
        
        # Apply final clustering on co-occurrence matrix
        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering(n_clusters=self.n_clusters, 
                                    affinity='precomputed',
                                    random_state=self.random_state)
        return spectral.fit_predict(co_occurrence_matrix)
    
    def _consensus_evidence_accumulation(self, X):
        """Evidence accumulation clustering"""
        n_samples = X.shape[0]
        evidence_matrix = np.zeros((n_samples, n_samples))
        
        # Accumulate evidence
        for labels in self.labels_ensemble_:
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                        evidence_matrix[i, j] += 1
        
        # Normalize by number of estimators where both points were included
        for i in range(n_samples):
            for j in range(n_samples):
                count = sum(1 for labels in self.labels_ensemble_ 
                           if labels[i] != -1 and labels[j] != -1)
                if count > 0:
                    evidence_matrix[i, j] /= count
        
        # Apply hierarchical clustering on evidence matrix
        from sklearn.cluster import AgglomerativeClustering
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             linkage='average',
                                             metric='precomputed')
        
        # Convert evidence to distance (1 - evidence)
        distance_matrix = 1 - evidence_matrix
        return hierarchical.fit_predict(distance_matrix)
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        X = check_array(X, accept_sparse='csr')
        
        # Use simple nearest centroid prediction
        distances = cdist(X, self.cluster_centers_)
        return np.argmax(-distances, axis=1)


class MiniBatchKMeansPlus(BaseEstimator, ClusterMixin):
    """
    Enhanced Mini-Batch K-Means with smart initialization and adaptive batch sizes
    
    Parameters:
    -----------
    n_clusters : int, default=8
        The number of clusters to form
    init : str, default='k-means++'
        Initialization method
    max_iter : int, default=100
        Maximum number of iterations over the dataset
    batch_size : int, default='auto'
        Size of the mini batches. If 'auto', will be set to 100*n_clusters
    tol : float, default=0.0
        Control early stopping based on the relative center changes
    max_no_improvement : int, default=10
        Number of mini-batch iterations with no improvement to wait before early stopping
    init_size : int, default=None
        Number of samples to randomly sample for speeding up the initialization
    n_init : int, default=3
        Number of random initializations performed
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility
    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a center to be reassigned
    """
    
    def __init__(self, n_clusters=8, init='k-means++', max_iter=100, batch_size='auto',
                 tol=0.0, max_no_improvement=10, init_size=None, n_init=3,
                 random_state=None, reassignment_ratio=0.01):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.init_size = init_size
        self.n_init = n_init
        self.random_state = random_state
        self.reassignment_ratio = reassignment_ratio
        
    def fit(self, X, y=None):
        """Fit the Mini-Batch K-Means clustering model"""
        from sklearn.cluster import MiniBatchKMeans
        
        X = check_array(X, accept_sparse='csr')
        
        # Set adaptive batch size
        if self.batch_size == 'auto':
            batch_size = min(100 * self.n_clusters, X.shape[0] // 10)
            batch_size = max(batch_size, self.n_clusters * 2)
        else:
            batch_size = self.batch_size
        
        # Use sklearn's implementation with enhanced parameters
        self.estimator_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            max_iter=self.max_iter,
            batch_size=batch_size,
            tol=self.tol,
            max_no_improvement=self.max_no_improvement,
            init_size=self.init_size,
            n_init=self.n_init,
            random_state=self.random_state,
            reassignment_ratio=self.reassignment_ratio
        )
        
        self.estimator_.fit(X)
        
        # Copy attributes for compatibility
        self.cluster_centers_ = self.estimator_.cluster_centers_
        self.labels_ = self.estimator_.labels_
        self.inertia_ = self.estimator_.inertia_
        self.n_iter_ = self.estimator_.n_iter_
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.estimator_.predict(X)
    
    def partial_fit(self, X, y=None):
        """Update k means clustering with X"""
        return self.estimator_.partial_fit(X, y)


class EnhancedAffinityPropagation(BaseEstimator, ClusterMixin):
    """
    Enhanced Affinity Propagation with adaptive parameters and stability improvements
    
    Parameters:
    -----------
    damping : float, default=0.5
        Damping factor between 0.5 and 1
    max_iter : int, default=200
        Maximum number of iterations
    convergence_iter : int, default=15
        Number of iterations with no change in the number of estimated clusters
    copy : bool, default=True
        Whether to make a copy of input data
    preference : array-like, shape (n_samples,) or float, default=None
        Preferences for each point
    affinity : str, default='euclidean'
        Affinity to use: 'euclidean', 'precomputed'
    verbose : bool, default=False
        Whether to be verbose
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility
    adaptive_damping : bool, default=True
        Whether to use adaptive damping
    """
    
    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True,
                 preference=None, affinity='euclidean', verbose=False, random_state=None,
                 adaptive_damping=True):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.preference = preference
        self.affinity = affinity
        self.verbose = verbose
        self.random_state = random_state
        self.adaptive_damping = adaptive_damping
        
    def fit(self, X, y=None):
        """Fit the Enhanced Affinity Propagation clustering model"""
        X = check_array(X, accept_sparse='csr')
        
        # Adaptive preference setting
        if self.preference is None:
            if self.affinity == 'euclidean':
                # Use median of pairwise similarities
                S = -euclidean_distances(X, squared=True)
                preference = np.median(S)
            else:
                preference = None
        else:
            preference = self.preference
        
        # Adaptive damping
        damping = self.damping
        best_n_clusters = 0
        best_labels = None
        best_centers = None
        
        # Try multiple damping values if adaptive_damping is True
        if self.adaptive_damping:
            damping_values = [0.5, 0.6, 0.7, 0.8, 0.9]
            results = []
            
            for damp in damping_values:
                try:
                    ap = AffinityPropagation(
                        damping=damp,
                        max_iter=self.max_iter,
                        convergence_iter=self.convergence_iter,
                        copy=self.copy,
                        preference=preference,
                        affinity=self.affinity,
                        verbose=False,
                        random_state=self.random_state
                    )
                    ap.fit(X)
                    
                    n_clusters = len(ap.cluster_centers_indices_)
                    if n_clusters > 1:  # Valid clustering
                        results.append((damp, ap, n_clusters))
                except:
                    continue
            
            if results:
                # Choose the result with moderate number of clusters
                results.sort(key=lambda x: abs(x[2] - np.sqrt(X.shape[0])))
                damping, self.estimator_, best_n_clusters = results[0]
            else:
                # Fallback to default
                self.estimator_ = AffinityPropagation(
                    damping=self.damping,
                    max_iter=self.max_iter,
                    convergence_iter=self.convergence_iter,
                    copy=self.copy,
                    preference=preference,
                    affinity=self.affinity,
                    verbose=self.verbose,
                    random_state=self.random_state
                )
                self.estimator_.fit(X)
        else:
            self.estimator_ = AffinityPropagation(
                damping=damping,
                max_iter=self.max_iter,
                convergence_iter=self.convergence_iter,
                copy=self.copy,
                preference=preference,
                affinity=self.affinity,
                verbose=self.verbose,
                random_state=self.random_state
            )
            self.estimator_.fit(X)
        
        # Copy attributes for compatibility
        self.cluster_centers_indices_ = self.estimator_.cluster_centers_indices_
        self.cluster_centers_ = self.estimator_.cluster_centers_
        self.labels_ = self.estimator_.labels_
        self.affinity_matrix_ = self.estimator_.affinity_matrix_
        self.n_iter_ = self.estimator_.n_iter_
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        X = check_array(X, accept_sparse='csr')
        
        # Find nearest cluster centers
        distances = euclidean_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)


# Advanced clustering algorithms registry for integration
ADVANCED_ALGORITHMS = {
    'Fuzzy C-Means': {
        'class': FuzzyCMeans,
        'category': 'Soft Clustering',
        'description': 'Fuzzy clustering algorithm allowing soft cluster assignments',
        'pros': 'Provides membership probabilities, handles overlapping clusters well',
        'cons': 'Sensitive to initialization, requires specifying fuzziness parameter',
        'use_cases': 'Overlapping data patterns, soft cluster boundaries',
        'parameters': {
            'n_clusters': {'type': 'int', 'range': (2, 20), 'default': 3},
            'm': {'type': 'float', 'range': (1.1, 5.0), 'default': 2.0},
            'max_iter': {'type': 'int', 'range': (50, 500), 'default': 300},
            'tol': {'type': 'float', 'range': (1e-6, 1e-2), 'default': 1e-4}
        }
    },
    'Consensus K-Means': {
        'class': ConsensusKMeans,
        'category': 'Ensemble-based',
        'description': 'Ensemble clustering using multiple K-means runs with consensus',
        'pros': 'More robust than single K-means, reduces initialization sensitivity',
        'cons': 'Computationally expensive, still requires k specification',
        'use_cases': 'When robustness is critical, unstable clustering scenarios',
        'parameters': {
            'n_clusters': {'type': 'int', 'range': (2, 20), 'default': 3},
            'n_estimators': {'type': 'int', 'range': (5, 50), 'default': 10},
            'consensus_method': {'type': 'choice', 'options': ['voting', 'co-occurrence', 'evidence_accumulation'], 'default': 'voting'},
            'sample_fraction': {'type': 'float', 'range': (0.5, 1.0), 'default': 0.8}
        }
    },
    'Mini-Batch K-Means+': {
        'class': MiniBatchKMeansPlus,
        'category': 'Scalable',
        'description': 'Enhanced mini-batch K-means for large datasets with adaptive parameters',
        'pros': 'Fast on large datasets, adaptive batch sizing, memory efficient',
        'cons': 'Approximation algorithm, may not converge to global optimum',
        'use_cases': 'Large datasets, streaming data, memory-constrained environments',
        'parameters': {
            'n_clusters': {'type': 'int', 'range': (2, 50), 'default': 8},
            'batch_size': {'type': 'choice', 'options': ['auto', 100, 500, 1000], 'default': 'auto'},
            'max_iter': {'type': 'int', 'range': (50, 500), 'default': 100},
            'n_init': {'type': 'int', 'range': (1, 10), 'default': 3}
        }
    },
    'Enhanced Affinity Propagation': {
        'class': EnhancedAffinityPropagation,
        'category': 'Message Passing',
        'description': 'Improved affinity propagation with adaptive damping and stability',
        'pros': 'No need to specify number of clusters, finds exemplars automatically',
        'cons': 'Can be slow, sensitive to preference parameter',
        'use_cases': 'Unknown number of clusters, finding representative samples',
        'parameters': {
            'damping': {'type': 'float', 'range': (0.5, 0.95), 'default': 0.5},
            'max_iter': {'type': 'int', 'range': (100, 500), 'default': 200},
            'adaptive_damping': {'type': 'bool', 'default': True}
        }
    }
}

