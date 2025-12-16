"""
Performance Optimization for Large-Scale Clustering
Implements scalable clustering techniques and memory-efficient algorithms
"""

import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import time
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import joblib

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10  # Small value to prevent division by zero
DEFAULT_CHUNK_SIZE = 10000
MIN_MEMORY_THRESHOLD = 0.8  # 80% memory usage threshold


class MemoryEfficientScaler:
    """
    Memory-efficient data scaling for large datasets using chunked processing
    """
    
    def __init__(self, chunk_size=10000):
        self.chunk_size = chunk_size
        self.mean_ = None
        self.std_ = None
        self.n_samples_seen_ = 0
        
    def fit(self, X):
        """Fit scaler using chunked processing"""
        X = check_array(X, accept_sparse='csr')
        n_samples, n_features = X.shape
        
        # Initialize statistics
        self.mean_ = np.zeros(n_features)
        self.M2_ = np.zeros(n_features)
        self.n_samples_seen_ = 0
        
        # Process in chunks
        for start_idx in range(0, n_samples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_samples)
            chunk = X[start_idx:end_idx]
            
            # Update running statistics
            self._update_statistics(chunk)
        
        # Finalize statistics
        if self.n_samples_seen_ > 0:
            # Population variance (ddof=0) to match numpy std default
            self.std_ = np.sqrt(self.M2_ / self.n_samples_seen_)
            self.std_[self.std_ == 0] = 1.0  # Avoid division by zero
        return self
    
    def _update_statistics(self, chunk):
        """Update running mean and variance statistics using batched Welford algorithm"""
        if hasattr(chunk, 'toarray'):  # Sparse matrix
            chunk = chunk.toarray()
        chunk = np.asarray(chunk)
        chunk_size = chunk.shape[0]
        if chunk_size == 0:
            return
        # Batch stats
        chunk_mean = np.mean(chunk, axis=0)
        # Population variance (ddof=0) times count gives M2 for the batch
        chunk_var = np.var(chunk, axis=0)
        chunk_M2 = chunk_var * chunk_size
        # Combine with existing running stats
        n = self.n_samples_seen_
        new_n = n + chunk_size
        if n == 0:
            self.mean_ = chunk_mean
            self.M2_ = chunk_M2
            self.n_samples_seen_ = chunk_size
            return
        delta = chunk_mean - self.mean_
        self.mean_ = self.mean_ + delta * (chunk_size / new_n)
        self.M2_ = self.M2_ + chunk_M2 + (delta ** 2) * (n * chunk_size) / new_n
        self.n_samples_seen_ = new_n

    def transform(self, X):
        """Transform data using chunked processing"""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        X = check_array(X, accept_sparse='csr')
        n_samples = X.shape[0]
        
        # Transform in chunks to save memory
        transformed_chunks = []
        
        for start_idx in range(0, n_samples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_samples)
            chunk = X[start_idx:end_idx]
            
            if hasattr(chunk, 'toarray'):
                chunk = chunk.toarray()
            
            transformed_chunk = (chunk - self.mean_) / self.std_
            transformed_chunks.append(transformed_chunk)
        
        return np.vstack(transformed_chunks)
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)


class StreamingKMeans(BaseEstimator, ClusterMixin):
    """
    Streaming K-means for large datasets that don't fit in memory
    """
    
    def __init__(self, n_clusters=8, batch_size=1000, max_iter=100, 
                 tol=1e-4, random_state=None, reassignment_ratio=0.01):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.reassignment_ratio = reassignment_ratio
        
    def fit(self, X, y=None, sample_weight=None):
        """Fit streaming K-means"""
        X = check_array(X, accept_sparse='csr')
        n_samples, n_features = X.shape
        
        # Initialize with MiniBatchKMeans for robustness
        self.kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            reassignment_ratio=self.reassignment_ratio
        )
        
        # Fit using mini-batches
        self.kmeans_.fit(X)
        
        # Copy attributes for sklearn compatibility
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        self.labels_ = self.kmeans_.labels_
        self.inertia_ = self.kmeans_.inertia_
        self.n_iter_ = self.kmeans_.n_iter_
        
        return self
    
    def partial_fit(self, X, y=None, sample_weight=None):
        """Update the model with a new batch of data"""
        if not hasattr(self, 'kmeans_'):
            # Initialize on first call
            self.kmeans_ = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=self.batch_size,
                random_state=self.random_state,
                reassignment_ratio=self.reassignment_ratio
            )
        
        X = check_array(X, accept_sparse='csr')
        self.kmeans_.partial_fit(X, sample_weight=sample_weight)
        
        # Update attributes
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        self.inertia_ = self.kmeans_.inertia_
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        if not hasattr(self, 'kmeans_'):
            raise ValueError("Model has not been fitted yet.")
        
        return self.kmeans_.predict(X)


class ApproximateDBSCAN(BaseEstimator, ClusterMixin):
    """
    Approximate DBSCAN for large datasets using sampling and parallel processing
    """
    
    def __init__(self, eps=0.5, min_samples=5, sample_fraction=0.1, 
                 n_jobs=-1, random_state=None):
        self.eps = eps
        self.min_samples = min_samples
        self.sample_fraction = sample_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state
        
    def fit(self, X, y=None):

        """Fit approximate DBSCAN"""
        X = check_array(X, accept_sparse='csr')
        n_samples, n_features = X.shape

        np.random.seed(self.random_state)

        # Step 1: Sample data for core point identification
        sample_size = max(1000, int(n_samples * self.sample_fraction))
        sample_size = min(sample_size, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[sample_indices]

        # Step 2: Run DBSCAN on sample with adaptive fallback
        eps = float(self.eps)
        min_samples = int(self.min_samples)
        core_points = np.empty((0, X.shape[1]))
        core_labels = np.array([], dtype=int)
        sample_labels = np.full(sample_size, -1, dtype=int)

        for _ in range(3):
            dbscan_sample = DBSCAN(eps=eps, min_samples=min_samples)
            sample_labels = dbscan_sample.fit_predict(X_sample)
            # Determine core points
            core_indices = getattr(dbscan_sample, 'core_sample_indices_', None)
            if core_indices is not None:
                core_points = X_sample[core_indices]
                core_labels = sample_labels[core_indices]
            else:
                mask = sample_labels != -1
                core_points = X_sample[mask]
                core_labels = sample_labels[mask]
            has_cluster = (sample_labels != -1).any() and len(core_points) > 0
            if has_cluster:
                break
            # Adapt parameters
            eps *= 1.5
            min_samples = max(2, int(min_samples * 0.8))

        # Step 3: Assign labels to all points
        self.labels_ = np.full(n_samples, -1, dtype=int)
        # Use sample labels directly for sampled points
        self.labels_[sample_indices] = sample_labels

        # Process remaining points in chunks
        remaining_indices = np.setdiff1d(np.arange(n_samples), sample_indices)

        if len(core_points) > 0 and len(remaining_indices) > 0:
            chunk_size = 10000
            for start_idx in range(0, len(remaining_indices), chunk_size):
                end_idx = min(start_idx + chunk_size, len(remaining_indices))
                chunk_indices = remaining_indices[start_idx:end_idx]
                X_chunk = X[chunk_indices]
                # Find nearest core point for each point in chunk
                chunk_labels = self._assign_to_nearest_core(X_chunk, core_points, core_labels)
                self.labels_[chunk_indices] = chunk_labels

        return self

    
    def _assign_to_nearest_core(self, X_chunk, core_points, core_labels):
        """Assign points to nearest core point within eps distance"""
        chunk_labels = np.full(len(X_chunk), -1, dtype=int)
        
        # Use sklearn's efficient nearest neighbors
        nbrs = NearestNeighbors(radius=self.eps)
        nbrs.fit(core_points)
        
        # Find neighbors within eps
        distances, indices = nbrs.radius_neighbors(X_chunk)
        
        for i, (neighbor_indices, neighbor_distances) in enumerate(zip(indices, distances)):
            if len(neighbor_indices) > 0:
                # Assign to closest core point's cluster
                closest_idx = neighbor_indices[np.argmin(neighbor_distances)]
                chunk_labels[i] = core_labels[closest_idx]
        
        return chunk_labels
    
    def fit_predict(self, X, y=None):
        """Fit and predict in one step"""
        return self.fit(X, y).labels_


class ParallelAgglomerativeClustering(BaseEstimator, ClusterMixin):
    """
    Parallel implementation of agglomerative clustering for large datasets
    """
    
    def __init__(self, n_clusters=2, linkage='ward', sample_fraction=0.3, 
                 n_jobs=-1, random_state=None):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.sample_fraction = sample_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Fit parallel agglomerative clustering"""
        from sklearn.cluster import AgglomerativeClustering
        
        X = check_array(X, accept_sparse=False)  # Dense matrix required
        n_samples, n_features = X.shape
        
        np.random.seed(self.random_state)
        
        # Step 1: Hierarchical sampling approach
        if n_samples > 10000:
            # Use sample-based approach for very large datasets
            sample_size = max(1000, int(n_samples * self.sample_fraction))
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[sample_indices]
            
            # Fit on sample
            clustering_sample = AgglomerativeClustering(
                n_clusters=self.n_clusters, 
                linkage=self.linkage
            )
            sample_labels = clustering_sample.fit_predict(X_sample)
            
            # Find cluster centers from sample
            cluster_centers = []
            for cluster_id in range(self.n_clusters):
                cluster_mask = sample_labels == cluster_id
                if np.any(cluster_mask):
                    center = np.mean(X_sample[cluster_mask], axis=0)
                    cluster_centers.append(center)
            
            cluster_centers = np.array(cluster_centers)
            
            # Assign all points to nearest cluster center
            if len(cluster_centers) > 0:
                from scipy.spatial.distance import cdist
                distances = cdist(X, cluster_centers)
                self.labels_ = np.argmin(distances, axis=1)
            else:
                self.labels_ = np.zeros(n_samples, dtype=int)
        
        else:
            # Use standard algorithm for smaller datasets
            clustering = AgglomerativeClustering(
                n_clusters=self.n_clusters, 
                linkage=self.linkage
            )
            self.labels_ = clustering.fit_predict(X)
        
        return self
    
    def fit_predict(self, X, y=None):
        """Fit and predict in one step"""
        return self.fit(X, y).labels_


class ChunkedDistanceMatrix:
    """
    Memory-efficient distance matrix computation using chunked processing
    """
    
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
        
    def compute_pairwise_distances(self, X, Y=None, metric='euclidean', n_jobs=1):
        """Compute pairwise distances using chunked processing"""
        if Y is None:
            Y = X
        
        X = check_array(X, accept_sparse='csr')
        Y = check_array(Y, accept_sparse='csr')
        
        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]
        
        # Initialize distance matrix
        distances = np.zeros((n_samples_X, n_samples_Y))
        
        # Compute distances in chunks
        for i, chunk_slice in enumerate(self._get_chunks(n_samples_X)):
            start_i, end_i = chunk_slice
            X_chunk = X[start_i:end_i]
            
            for j, chunk_slice_j in enumerate(self._get_chunks(n_samples_Y)):
                start_j, end_j = chunk_slice_j
                Y_chunk = Y[start_j:end_j]
                
                # Compute distance for this chunk pair
                if hasattr(X_chunk, 'toarray'):
                    X_chunk = X_chunk.toarray()
                if hasattr(Y_chunk, 'toarray'):
                    Y_chunk = Y_chunk.toarray()
                
                chunk_distances = self._compute_chunk_distance(X_chunk, Y_chunk, metric)
                distances[start_i:end_i, start_j:end_j] = chunk_distances
                
                # Force garbage collection to manage memory
                if (i * len(list(self._get_chunks(n_samples_Y))) + j) % 10 == 0:
                    gc.collect()
        
        return distances
    
    def _get_chunks(self, n_samples):
        """Generate chunk indices"""
        for start in range(0, n_samples, self.chunk_size):
            end = min(start + self.chunk_size, n_samples)
            yield start, end
    
    def _compute_chunk_distance(self, X_chunk, Y_chunk, metric):
        """Compute distance for a single chunk pair"""
        from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
        
        if metric == 'euclidean':
            return euclidean_distances(X_chunk, Y_chunk)
        elif metric == 'manhattan':
            return manhattan_distances(X_chunk, Y_chunk)
        elif metric == 'cosine':
            return cosine_distances(X_chunk, Y_chunk)
        else:
            # Fallback to scipy
            from scipy.spatial.distance import cdist
            return cdist(X_chunk, Y_chunk, metric=metric)


class MemoryMonitor:
    """
    Memory usage monitor for clustering operations
    """
    
    def __init__(self, threshold_gb=4.0):
        self.threshold_bytes = threshold_gb * 1024**3
        self.monitoring = False
        self.peak_memory = 0
        
    def start_monitoring(self):
        """Start memory monitoring"""
        self.monitoring = True
        self.peak_memory = 0
        
        def monitor_memory():
            while self.monitoring:
                current_memory = psutil.Process().memory_info().rss
                self.peak_memory = max(self.peak_memory, current_memory)
                time.sleep(0.1)
        
        self.monitor_thread = threading.Thread(target=monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
    
    def get_memory_usage_gb(self):
        """Get current memory usage in GB"""
        return psutil.Process().memory_info().rss / (1024**3)
    
    def get_peak_memory_gb(self):
        """Get peak memory usage in GB"""
        return self.peak_memory / (1024**3)
    
    def is_memory_critical(self):
        """Check if memory usage is critical"""
        return psutil.Process().memory_info().rss > self.threshold_bytes


class ScalableClusteringPipeline:
    """
    Main pipeline for scalable clustering on large datasets
    """
    
    def __init__(self, memory_limit_gb=4.0, n_jobs=-1):
        self.memory_limit_gb = memory_limit_gb
        self.n_jobs = n_jobs
        self.memory_monitor = MemoryMonitor(threshold_gb=memory_limit_gb)
        self.preprocessing_strategy = 'auto'
        self.clustering_strategy = 'auto'
        
    def fit_predict(self, X, algorithm='auto', **algorithm_params):
        """
        Scalable clustering with automatic algorithm selection based on data size
        
        Parameters:
        -----------
        X : array-like
            Input data
        algorithm : str
            Clustering algorithm: 'auto', 'kmeans', 'dbscan', 'agglomerative'
        **algorithm_params : dict
            Algorithm-specific parameters
            
        Returns:
        --------
        labels : array
            Cluster labels
        metadata : dict
            Processing metadata
        """
        
        start_time = time.time()
        self.memory_monitor.start_monitoring()
        
        try:
            # Analyze data characteristics
            data_info = self._analyze_data_characteristics(X)
            
            # Determine optimal strategies
            preprocessing_strategy = self._determine_preprocessing_strategy(data_info)
            clustering_strategy = self._determine_clustering_strategy(data_info, algorithm)
            
            # Preprocess data
            X_processed = self._preprocess_data(X, preprocessing_strategy)
            
            # Apply clustering
            labels = self._apply_clustering(X_processed, clustering_strategy, algorithm_params)
            
            # Compile metadata
            processing_time = time.time() - start_time
            metadata = {
                'processing_time': processing_time,
                'data_characteristics': data_info,
                'preprocessing_strategy': preprocessing_strategy,
                'clustering_strategy': clustering_strategy,
                'peak_memory_gb': self.memory_monitor.get_peak_memory_gb(),
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_clusters': len(np.unique(labels))
            }
            
            return labels, metadata
            
        finally:
            self.memory_monitor.stop_monitoring()
    
    def _analyze_data_characteristics(self, X):
        """Analyze data characteristics for strategy selection"""
        X = check_array(X, accept_sparse='csr')
        n_samples, n_features = X.shape
        
        data_info = {
            'n_samples': n_samples,
            'n_features': n_features,
            'data_size_mb': X.nbytes / (1024**2),
            'is_sparse': hasattr(X, 'toarray'),
            'sparsity': None,
            'memory_usage_gb': self.memory_monitor.get_memory_usage_gb()
        }
        
        # Estimate sparsity for sparse matrices
        if hasattr(X, 'toarray'):
            data_info['sparsity'] = 1.0 - (X.nnz / (n_samples * n_features))
        
        # Categorize dataset size
        if n_samples <= 10000:
            data_info['size_category'] = 'small'
        elif n_samples <= 100000:
            data_info['size_category'] = 'medium'
        elif n_samples <= 1000000:
            data_info['size_category'] = 'large'
        else:
            data_info['size_category'] = 'very_large'
        
        return data_info
    
    def _determine_preprocessing_strategy(self, data_info):
        """Determine optimal preprocessing strategy"""
        if data_info['size_category'] in ['large', 'very_large']:
            return 'chunked_scaling'
        elif data_info['is_sparse']:
            return 'sparse_scaling'
        else:
            return 'standard_scaling'
    
    def _determine_clustering_strategy(self, data_info, algorithm):
        """Determine optimal clustering strategy"""
        if algorithm != 'auto':
            return algorithm
        
        n_samples = data_info['n_samples']
        size_category = data_info['size_category']
        
        if size_category == 'small':
            return 'standard_kmeans'
        elif size_category == 'medium':
            return 'minibatch_kmeans'
        elif size_category == 'large':
            return 'streaming_kmeans'
        else:  # very_large
            return 'approximate_clustering'
    
    def _preprocess_data(self, X, strategy):
        """Apply preprocessing based on strategy"""
        if strategy == 'chunked_scaling':
            scaler = MemoryEfficientScaler(chunk_size=10000)
            return scaler.fit_transform(X)
        
        elif strategy == 'sparse_scaling':
            # For sparse data, use sklearn's standard scaler which preserves sparsity
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler(with_mean=False)  # Don't center sparse data
            return scaler.fit_transform(X)
        
        elif strategy == 'standard_scaling':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        
        else:
            return X
    
    def _apply_clustering(self, X, strategy, algorithm_params):
        """Apply clustering based on strategy"""
        
        if strategy == 'standard_kmeans':
            from sklearn.cluster import KMeans
            n_clusters = algorithm_params.get('n_clusters', 8)
            clusterer = KMeans(n_clusters=n_clusters)
            return clusterer.fit_predict(X)
        
        elif strategy == 'minibatch_kmeans':
            n_clusters = algorithm_params.get('n_clusters', 8)
            batch_size = algorithm_params.get('batch_size', min(1000, X.shape[0] // 10))
            clusterer = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
            return clusterer.fit_predict(X)
        
        elif strategy == 'streaming_kmeans':
            n_clusters = algorithm_params.get('n_clusters', 8)
            batch_size = algorithm_params.get('batch_size', 1000)
            clusterer = StreamingKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size
            )
            return clusterer.fit_predict(X)
        
        elif strategy == 'approximate_clustering':
            # For very large datasets, use sampling-based approach
            return self._approximate_clustering(X, algorithm_params)
        
        elif strategy == 'dbscan':
            eps = algorithm_params.get('eps', 0.5)
            min_samples = algorithm_params.get('min_samples', 5)
            
            if X.shape[0] > 50000:
                # Use approximate DBSCAN for large datasets
                clusterer = ApproximateDBSCAN(eps=eps, min_samples=min_samples
                )
            else:
                clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            
            return clusterer.fit_predict(X)
        
        elif strategy == 'agglomerative':
            n_clusters = algorithm_params.get('n_clusters', 8)
            linkage = algorithm_params.get('linkage', 'ward')
            
            if X.shape[0] > 10000:
                # Use parallel/sampling approach for large datasets
                clusterer = ParallelAgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage,
                    n_jobs=self.n_jobs
                )
            else:
                from sklearn.cluster import AgglomerativeClustering
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage
                )
            
            return clusterer.fit_predict(X)
        
        else:
            # Fallback to streaming k-means
            clusterer = StreamingKMeans(n_clusters=algorithm_params.get('n_clusters', 8))
            return clusterer.fit_predict(X)
    
    def _approximate_clustering(self, X, algorithm_params):
        """Approximate clustering for very large datasets"""
        n_clusters = algorithm_params.get('n_clusters', 8)
        
        # Multi-level sampling approach
        n_samples = X.shape[0]
        
        # Level 1: Heavy sampling (10%)
        sample_size_1 = max(10000, int(n_samples * 0.1))
        indices_1 = np.random.choice(n_samples, sample_size_1, replace=False)
        X_sample_1 = X[indices_1]
        
        # Level 2: Moderate sampling from Level 1 (50%)
        sample_size_2 = max(1000, int(sample_size_1 * 0.5))
        indices_2 = np.random.choice(sample_size_1, sample_size_2, replace=False)
        X_sample_2 = X_sample_1[indices_2]
        
        # Cluster Level 2 sample
        clusterer = MiniBatchKMeans(n_clusters=n_clusters)
        sample_labels_2 = clusterer.fit_predict(X_sample_2)
        
        # Get cluster centers
        cluster_centers = clusterer.cluster_centers_
        
        # Assign all points to nearest cluster center using chunked processing
        labels = np.zeros(n_samples, dtype=int)
        chunk_size = 10000
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            X_chunk = X[start_idx:end_idx]
            
            # Find nearest cluster center for each point in chunk
            if hasattr(X_chunk, 'toarray'):
                X_chunk = X_chunk.toarray()
            
            from scipy.spatial.distance import cdist
            distances = cdist(X_chunk, cluster_centers)
            chunk_labels = np.argmin(distances, axis=1)
            labels[start_idx:end_idx] = chunk_labels
        
        return labels



    def get_memory_recommendations(self, X):
        """Get memory optimization recommendations (pipeline-level)"""
        data_info = self._analyze_data_characteristics(X)
        recommendations = []
        if data_info['data_size_mb'] > 1000:  # > 1GB
            recommendations.append("Consider using chunked processing for this dataset size")
        if data_info['n_samples'] > 100000:
            recommendations.append("Use streaming algorithms for large sample sizes")
        if data_info['n_features'] > 1000:
            recommendations.append("Consider dimensionality reduction before clustering")
        if data_info['memory_usage_gb'] > 2.0:
            recommendations.append("Current memory usage is high - monitor memory during clustering")
        return {
            'data_characteristics': data_info,
            'recommendations': recommendations
        }

class ParallelClusteringEvaluator:
    """
    Parallel evaluation of clustering results for large datasets
    """
    
    def __init__(self, n_jobs=-1, chunk_size=10000):
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        
    def evaluate_clustering_parallel(self, X, labels, metrics=['silhouette', 'calinski_harabasz']):
        """Evaluate clustering using parallel processing"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
            futures = {}
            
            # Submit evaluation tasks
            if 'silhouette' in metrics:
                futures['silhouette'] = executor.submit(self._compute_silhouette_parallel, X, labels)
            
            if 'calinski_harabasz' in metrics:
                futures['calinski_harabasz'] = executor.submit(self._compute_calinski_harabasz, X, labels)
            
            if 'davies_bouldin' in metrics:
                futures['davies_bouldin'] = executor.submit(self._compute_davies_bouldin, X, labels)
            
            # Collect results
            for metric_name, future in futures.items():
                try:
                    results[metric_name] = future.result(timeout=300)  # 5 minute timeout
                except Exception as e:
                    results[metric_name] = f"Error: {str(e)}"
        
        return results
    
    def _compute_silhouette_parallel(self, X, labels):
        """Compute silhouette score using sampling for large datasets"""
        n_samples = X.shape[0]
        
        if n_samples > 10000:
            # Use sampling for very large datasets
            sample_size = min(5000, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[sample_indices]
            labels_sample = labels[sample_indices]
            
            from sklearn.metrics import silhouette_score
            return silhouette_score(X_sample, labels_sample)
        else:
            from sklearn.metrics import silhouette_score
            return silhouette_score(X, labels)
    
    def _compute_calinski_harabasz(self, X, labels):
        """Compute Calinski-Harabasz score"""
        from sklearn.metrics import calinski_harabasz_score
        return calinski_harabasz_score(X, labels)
    
    def _compute_davies_bouldin(self, X, labels):
        """Compute Davies-Bouldin score"""
        from sklearn.metrics import davies_bouldin_score
        return davies_bouldin_score(X, labels)


# Integration with existing clustering module
    def get_memory_recommendations(self, X):
        """Get memory optimization recommendations (pipeline-level)"""
        data_info = self._analyze_data_characteristics(X)
        recommendations = []
        if data_info['data_size_mb'] > 1000:  # > 1GB
            recommendations.append("Consider using chunked processing for this dataset size")
        if data_info['n_samples'] > 100000:
            recommendations.append("Use streaming algorithms for large sample sizes")
        if data_info['n_features'] > 1000:
            recommendations.append("Consider dimensionality reduction before clustering")
        if data_info['memory_usage_gb'] > 2.0:
            recommendations.append("Current memory usage is high - monitor memory during clustering")
        return {
            'data_characteristics': data_info,
            'recommendations': recommendations
        }
class PerformanceOptimizedClusteringModule:
    """
    Main interface for performance-optimized clustering
    """
    
    def __init__(self, memory_limit_gb=4.0, n_jobs=-1):
        self.pipeline = ScalableClusteringPipeline(memory_limit_gb=memory_limit_gb, n_jobs=n_jobs)
        self.evaluator = ParallelClusteringEvaluator(n_jobs=n_jobs)
        
    def cluster_large_dataset(self, X, algorithm='auto', evaluate=True, **algorithm_params):
        """
        Complete clustering workflow for large datasets
        
        Parameters:
        -----------
        X : array-like
            Input data
        algorithm : str
            Clustering algorithm
        evaluate : bool
            Whether to evaluate results
        **algorithm_params : dict
            Algorithm parameters
            
        Returns:
        --------
        results : dict
            Complete clustering results
        """
        
        # Perform clustering
        labels, metadata = self.pipeline.fit_predict(X, algorithm=algorithm, **algorithm_params)
        
        results = {
            'labels': labels,
            'metadata': metadata,
            'evaluation': None
        }
        
        # Evaluate if requested
        if evaluate:
            print("Evaluating clustering results...")
            evaluation_results = self.evaluator.evaluate_clustering_parallel(X, labels)
            results['evaluation'] = evaluation_results
        
        return results
    
    def get_memory_recommendations(self, X):
        """Get memory optimization recommendations"""
        data_info = self.pipeline._analyze_data_characteristics(X)
        
        recommendations = []
        
        if data_info['data_size_mb'] > 1000:  # > 1GB
            recommendations.append("Consider using chunked processing for this dataset size")
        
        if data_info['n_samples'] > 100000:
            recommendations.append("Use streaming algorithms for large sample sizes")
        
        if data_info['n_features'] > 1000:
            recommendations.append("Consider dimensionality reduction before clustering")
        
        if data_info['memory_usage_gb'] > 2.0:
            recommendations.append("Current memory usage is high - monitor memory during clustering")
        
        return {
            'data_characteristics': data_info,
            'recommendations': recommendations
        }




