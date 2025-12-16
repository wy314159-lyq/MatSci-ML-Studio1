"""
Adaptive Parameter Optimization System
Implements Bayesian optimization and advanced parameter tuning for clustering algorithms
"""

import numpy as np
import pandas as pd
import hashlib
import logging
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import time
import json

# Configure module logger
logger = logging.getLogger(__name__)

# Constants for magic numbers
DEFAULT_NOISE_ESTIMATE = 0.1
MIN_CLUSTER_COUNT = 2
DEFAULT_CLUSTER_TENDENCY = 0.5
EPSILON = 1e-10  # Small value to prevent division by zero
CACHE_SAMPLE_SIZE = 100  # Number of samples to use for cache key hash
N_ACQUISITION_CANDIDATES = 100  # Number of candidates for acquisition function
MAX_HISTORY_SIZE = 100  # Maximum recommendation history size


class BayesianClusterOptimizer:
    """
    Bayesian optimization for clustering algorithm hyperparameters
    
    Uses Gaussian Process surrogate models to efficiently explore parameter space
    and find optimal clustering configurations.
    
    Parameters:
    -----------
    acquisition_function : str, default='expected_improvement'
        Acquisition function: 'expected_improvement', 'upper_confidence_bound', 'probability_improvement'
    n_initial_points : int, default=5
        Number of initial random evaluations
    n_optimization_steps : int, default=20
        Maximum number of optimization steps
    alpha : float, default=0.01
        Noise level in Gaussian process
    kappa : float, default=2.576
        Exploration parameter for Upper Confidence Bound
    xi : float, default=0.01
        Exploration parameter for Expected Improvement
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, acquisition_function='expected_improvement', n_initial_points=5,
                 n_optimization_steps=20, alpha=0.01, kappa=2.576, xi=0.01, random_state=None):
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        self.n_optimization_steps = n_optimization_steps
        self.alpha = alpha
        self.kappa = kappa
        self.xi = xi
        self.random_state = random_state

        # Initialize local random generator for thread safety
        self._rng = np.random.default_rng(random_state)

        # Initialize storage for optimization history
        self.optimization_history_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.gp_model_ = None
        
    def optimize(self, X, algorithm_class, param_space, scoring_function=None, 
                 cv_folds=3, n_jobs=1, verbose=False):
        """
        Optimize clustering algorithm parameters using Bayesian optimization
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        algorithm_class : class
            Clustering algorithm class
        param_space : dict
            Parameter space to optimize over
        scoring_function : callable, default=None
            Scoring function to maximize. If None, uses silhouette score
        cv_folds : int, default=3
            Number of cross-validation folds for robust evaluation
        n_jobs : int, default=1
            Number of parallel jobs
        verbose : bool, default=False
            Whether to print optimization progress
            
        Returns:
        --------
        best_params : dict
            Best parameters found
        best_score : float
            Best score achieved
        """
        
        if scoring_function is None:
            scoring_function = self._default_scoring_function

        # Input validation
        X = self._validate_input(X)

        # Convert parameter space to bounds
        bounds, param_names = self._convert_param_space(param_space)
        
        # Initialize with random points
        if verbose:
            logger.info(f"Starting Bayesian optimization with {self.n_initial_points} initial points...")

        for i in range(self.n_initial_points):
            params = self._sample_random_params(param_space)
            score = self._evaluate_params(X, algorithm_class, params, scoring_function, cv_folds)

            self.optimization_history_.append({
                'params': params,
                'score': score,
                'iteration': i,
                'type': 'initial'
            })

            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params.copy()

            if verbose:
                logger.info(f"Initial point {i+1}/{self.n_initial_points}: score = {score:.4f}")

        # Fit initial GP model
        self._fit_gp_model()

        # Optimization loop
        if verbose:
            logger.info(f"Starting optimization loop for {self.n_optimization_steps} steps...")

        for i in range(self.n_optimization_steps):
            # Find next point to evaluate using acquisition function
            next_params = self._get_next_params(param_space)
            score = self._evaluate_params(X, algorithm_class, next_params, scoring_function, cv_folds)

            self.optimization_history_.append({
                'params': next_params,
                'score': score,
                'iteration': self.n_initial_points + i,
                'type': 'optimization'
            })

            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = next_params.copy()

            # Update GP model
            self._fit_gp_model()

            if verbose:
                logger.info(f"Optimization step {i+1}/{self.n_optimization_steps}: score = {score:.4f}, best = {self.best_score_:.4f}")
        
        return self.best_params_, self.best_score_
    
    def _convert_param_space(self, param_space):
        """Convert parameter space to bounds for optimization"""
        bounds = []
        param_names = []
        
        for param_name, param_info in param_space.items():
            if param_info['type'] == 'int':
                bounds.append(param_info['range'])
            elif param_info['type'] == 'float':
                bounds.append(param_info['range'])
            elif param_info['type'] == 'choice':
                # For categorical parameters, we'll handle them separately
                bounds.append((0, len(param_info['options']) - 1))
            
            param_names.append(param_name)
        
        return bounds, param_names
    
    def _sample_random_params(self, param_space):
        """Sample random parameters from the space using local RNG"""
        params = {}

        for param_name, param_info in param_space.items():
            if param_info['type'] == 'int':
                low, high = param_info['range']
                params[param_name] = int(self._rng.integers(low, high + 1))
            elif param_info['type'] == 'float':
                low, high = param_info['range']
                params[param_name] = float(self._rng.uniform(low, high))
            elif param_info['type'] == 'choice':
                params[param_name] = self._rng.choice(param_info['options'])
            elif param_info['type'] == 'bool':
                params[param_name] = bool(self._rng.choice([True, False]))

        return params

    def _evaluate_params(self, X, algorithm_class, params, scoring_function, cv_folds):
        """Evaluate parameter configuration using cross-validation"""
        scores = []

        # Simple cross-validation by random sampling
        n_samples = X.shape[0]
        fold_size = n_samples // cv_folds

        # Create fold-specific RNG for reproducibility
        fold_rng = np.random.default_rng(self.random_state)

        for fold in range(cv_folds):
            # Create train/validation split using local RNG
            indices = fold_rng.permutation(n_samples)

            if fold < cv_folds - 1:
                val_indices = indices[fold * fold_size:(fold + 1) * fold_size]
                train_indices = np.concatenate([
                    indices[:fold * fold_size],
                    indices[(fold + 1) * fold_size:]
                ])
            else:
                val_indices = indices[fold * fold_size:]
                train_indices = indices[:fold * fold_size]

            if len(train_indices) < 2 or len(val_indices) < 2:
                continue

            try:
                # Fit on training data
                algorithm = algorithm_class(**params)
                algorithm.fit(X[train_indices])

                # Evaluate on validation data
                val_labels = algorithm.predict(X[val_indices])

                if len(np.unique(val_labels)) > 1:
                    score = scoring_function(X[val_indices], val_labels)
                    scores.append(score)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                # Handle specific algorithm failures
                logger.debug(f"Algorithm evaluation failed: {e}")
                scores.append(-1.0)
            except Exception as e:
                # Log unexpected errors
                logger.warning(f"Unexpected error during parameter evaluation: {e}")
                scores.append(-1.0)

        return np.mean(scores) if scores else -1.0

    def _validate_input(self, X):
        """Validate and prepare input data"""
        X = np.asarray(X, dtype=np.float64)

        # Check for NaN or Inf values
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError("Input data contains infinite values")

        # Check minimum samples
        if X.shape[0] < MIN_CLUSTER_COUNT:
            raise ValueError(f"Need at least {MIN_CLUSTER_COUNT} samples for clustering")

        return X
    
    def _fit_gp_model(self):
        """Fit Gaussian Process model to optimization history"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
            
            # Prepare training data
            X_train = []
            y_train = []
            
            for entry in self.optimization_history_:
                # Convert parameters to numerical vector
                x_vec = self._params_to_vector(entry['params'])
                X_train.append(x_vec)
                y_train.append(entry['score'])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Define kernel
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            
            # Fit GP
            self.gp_model_ = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.alpha,
                random_state=self.random_state
            )
            self.gp_model_.fit(X_train, y_train)
            
        except ImportError:
            warnings.warn("sklearn.gaussian_process not available. Using random sampling fallback.")
            self.gp_model_ = None
        except Exception as e:
            warnings.warn(f"GP fitting failed: {e}. Using random sampling fallback.")
            self.gp_model_ = None
    
    def _params_to_vector(self, params):
        """Convert parameter dictionary to numerical vector using stable hashing"""
        vector = []
        for key in sorted(params.keys()):
            val = params[key]
            if isinstance(val, bool):
                vector.append(float(val))
            elif isinstance(val, (int, float)):
                vector.append(float(val))
            elif isinstance(val, str):
                # Use stable hash function (MD5) for deterministic encoding
                hash_val = int(hashlib.md5(val.encode()).hexdigest()[:8], 16)
                vector.append(float(hash_val % 1000) / 1000.0)
            else:
                vector.append(0.0)
        return np.array(vector)

    def _get_next_params(self, param_space):
        """Get next parameter configuration using acquisition function"""
        if self.gp_model_ is None:
            # Fallback to random sampling
            return self._sample_random_params(param_space)

        # Use acquisition function to find next point
        best_acquisition = -np.inf
        best_params = None

        # Sample multiple candidates and choose best
        for _ in range(N_ACQUISITION_CANDIDATES):
            candidate_params = self._sample_random_params(param_space)
            candidate_vector = self._params_to_vector(candidate_params)

            # Compute acquisition function value
            try:
                acquisition_value = self._compute_acquisition(candidate_vector.reshape(1, -1))

                if acquisition_value > best_acquisition:
                    best_acquisition = acquisition_value
                    best_params = candidate_params
            except (ValueError, np.linalg.LinAlgError) as e:
                # Handle numerical errors gracefully
                logger.debug(f"Acquisition evaluation failed: {e}")
                continue
            except Exception as e:
                # Log unexpected errors
                logger.warning(f"Failed to evaluate candidate params: {e}")
                continue

        return best_params if best_params is not None else self._sample_random_params(param_space)

    def _compute_acquisition(self, X):
        """Compute acquisition function value with safe division"""
        if self.gp_model_ is None:
            return self._rng.random()

        try:
            mean, std = self.gp_model_.predict(X, return_std=True)

            # Prevent division by zero
            std = np.maximum(std, EPSILON)

            if self.acquisition_function == 'expected_improvement':
                z = (mean - self.best_score_ - self.xi) / std
                return (mean - self.best_score_ - self.xi) * norm.cdf(z) + std * norm.pdf(z)

            elif self.acquisition_function == 'upper_confidence_bound':
                return mean + self.kappa * std

            elif self.acquisition_function == 'probability_improvement':
                z = (mean - self.best_score_ - self.xi) / std
                return norm.cdf(z)

        except (ValueError, np.linalg.LinAlgError) as e:
            # Handle numerical errors gracefully
            logger.debug(f"Acquisition computation failed: {e}")
            return self._rng.random()

        return self._rng.random()

    def _default_scoring_function(self, X, labels):
        """Default scoring function using silhouette score"""
        if len(np.unique(labels)) < MIN_CLUSTER_COUNT:
            return -1.0
        try:
            return silhouette_score(X, labels)
        except Exception as e:
            logger.warning(f"Error computing silhouette score: {e}")
            return -1.0
    
    def get_optimization_summary(self):
        """Get summary of optimization process"""
        if not self.optimization_history_:
            return {}
        
        scores = [entry['score'] for entry in self.optimization_history_]
        
        return {
            'best_score': self.best_score_,
            'best_params': self.best_params_,
            'n_evaluations': len(self.optimization_history_),
            'score_improvement': self.best_score_ - min(scores) if scores else 0,
            'optimization_history': self.optimization_history_
        }


class AdaptiveParameterRecommender:
    """
    Adaptive parameter recommendation system that learns from data characteristics
    and past clustering results to suggest optimal parameters.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.recommendation_history_ = []
        self.data_characteristics_cache_ = {}
        
    def recommend_parameters(self, X, algorithm_name, n_recommendations=3, 
                           use_history=True, fast_mode=False):
        """
        Recommend optimal parameters for a given algorithm and dataset
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
        algorithm_name : str
            Name of the clustering algorithm
        n_recommendations : int, default=3
            Number of parameter recommendations to return
        use_history : bool, default=True
            Whether to use historical data for recommendations
        fast_mode : bool, default=False
            Whether to use fast approximations
            
        Returns:
        --------
        recommendations : list of dict
            List of recommended parameter configurations
        """
        
        # Get data characteristics
        data_chars = self._analyze_data_characteristics(X, fast_mode)
        
        # Algorithm-specific recommendations
        if algorithm_name.lower() in ['k-means', 'kmeans']:
            return self._recommend_kmeans_params(X, data_chars, n_recommendations, use_history)
        elif algorithm_name.lower() == 'dbscan':
            return self._recommend_dbscan_params(X, data_chars, n_recommendations, use_history)
        elif algorithm_name.lower() in ['spectral clustering', 'spectral']:
            return self._recommend_spectral_params(X, data_chars, n_recommendations, use_history)
        elif algorithm_name.lower() == 'mean shift':
            return self._recommend_meanshift_params(X, data_chars, n_recommendations, use_history)
        elif algorithm_name.lower() == 'fuzzy c-means':
            return self._recommend_fuzzy_params(X, data_chars, n_recommendations, use_history)
        elif algorithm_name.lower() in ['affinity propagation', 'affinity']:
            return self._recommend_affinity_params(X, data_chars, n_recommendations, use_history)
        else:
            return self._recommend_generic_params(X, data_chars, n_recommendations)
    
    def _analyze_data_characteristics(self, X, fast_mode=False):
        """Analyze data characteristics for parameter recommendation"""

        # Create an improved cache key with data fingerprint
        data_fingerprint = self._compute_data_fingerprint(X)
        cache_key = f"{X.shape}_{data_fingerprint}_{fast_mode}"

        if cache_key in self.data_characteristics_cache_:
            return self.data_characteristics_cache_[cache_key]

        n_samples, n_features = X.shape
        characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'density_estimate': self._estimate_density(X, fast_mode),
            'noise_estimate': self._estimate_noise(X, fast_mode),
            'dimensionality_ratio': n_features / max(n_samples, 1),
            'data_scale': np.std(X),
            'data_range': np.ptp(X, axis=0).mean(),
            'feature_correlation': self._estimate_feature_correlation(X, fast_mode),
            'cluster_tendency': self._estimate_cluster_tendency(X, fast_mode)
        }
        
        # Cache the results
        self.data_characteristics_cache_[cache_key] = characteristics
        
        return characteristics

    def _compute_data_fingerprint(self, X):
        """Compute a stable fingerprint for data to use in cache key"""
        # Sample a fixed number of data points for fingerprinting
        n_samples = min(CACHE_SAMPLE_SIZE, X.shape[0])
        # Use deterministic indices based on data shape
        step = max(1, X.shape[0] // n_samples)
        sample_indices = np.arange(0, X.shape[0], step)[:n_samples]
        X_sample = X[sample_indices].flatten()

        # Compute hash of sample data
        data_bytes = X_sample.tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:12]

    def _estimate_density(self, X, fast_mode=False):
        """Estimate data density"""
        if fast_mode or X.shape[0] > 1000:
            # Fast approximation using nearest neighbor distances
            from sklearn.neighbors import NearestNeighbors
            sample_size = min(100, X.shape[0])
            sample_indices = self._rng.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_indices]

            nbrs = NearestNeighbors(n_neighbors=min(5, sample_size-1)).fit(X_sample)
            distances, _ = nbrs.kneighbors(X_sample)
            mean_dist = np.mean(distances[:, 1:])
            return 1.0 / (mean_dist + EPSILON)
        else:
            # More accurate density estimation
            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde.fit(X[:min(500, X.shape[0])])
            return np.exp(kde.score_samples(X[:100]).mean())

    def _estimate_noise(self, X, fast_mode=False):
        """Estimate noise level in data with safe division"""
        if fast_mode:
            # Simple noise estimation using local variance
            sample_size = min(100, X.shape[0])
            sample_indices = self._rng.choice(X.shape[0], sample_size, replace=False)
            sample_data = X[sample_indices]
            sample_mean = np.mean(sample_data)
            sample_std = np.std(sample_data)
            # Safe division with EPSILON
            if np.abs(sample_mean) < EPSILON:
                return DEFAULT_NOISE_ESTIMATE
            return sample_std / np.abs(sample_mean)
        else:
            # More sophisticated noise estimation
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(10, X.shape[0]-1)).fit(X)
            distances, _ = nbrs.kneighbors(X)
            return np.std(distances[:, 1:])

    def _estimate_feature_correlation(self, X, fast_mode=False):
        """Estimate average feature correlation"""
        if X.shape[1] < 2:
            return 0.0

        try:
            if fast_mode or X.shape[1] > 50:
                # Sample subset of features
                n_features_sample = min(10, X.shape[1])
                feature_indices = self._rng.choice(X.shape[1], n_features_sample, replace=False)
                X_sample = X[:, feature_indices]
            else:
                X_sample = X

            corr_matrix = np.corrcoef(X_sample.T)
            # Handle NaN in correlation matrix
            if np.any(np.isnan(corr_matrix)):
                return 0.0
            # Return mean absolute correlation (excluding diagonal)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            return np.abs(corr_matrix[mask]).mean()
        except (ValueError, np.linalg.LinAlgError):
            return 0.0

    def _estimate_cluster_tendency(self, X, fast_mode=False):
        """Estimate clustering tendency using Hopkins statistic approximation"""
        sample_size = min(50, X.shape[0] // 10) if fast_mode else min(100, X.shape[0] // 5)

        if sample_size < 2:
            return DEFAULT_CLUSTER_TENDENCY

        try:
            # Sample random points using local RNG
            sample_indices = self._rng.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_indices]

            # Generate uniform random points in data space
            data_min = X.min(axis=0)
            data_max = X.max(axis=0)
            random_points = self._rng.uniform(data_min, data_max, (sample_size, X.shape[1]))

            # Calculate distances to nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=2).fit(X)

            # Distance from sample points to their nearest neighbors in X
            sample_distances, _ = nbrs.kneighbors(X_sample)
            w_distances = sample_distances[:, 1]  # Second nearest (first is itself)

            # Distance from random points to nearest neighbors in X
            random_distances, _ = nbrs.kneighbors(random_points)
            u_distances = random_distances[:, 0]

            # Hopkins statistic approximation with safe division
            total_dist = np.sum(u_distances) + np.sum(w_distances)
            if total_dist < EPSILON:
                return DEFAULT_CLUSTER_TENDENCY
            hopkins = np.sum(u_distances) / total_dist

            # Convert to cluster tendency (0 = no clusters, 1 = strong clusters)
            return abs(hopkins - 0.5) * 2
        except Exception as e:
            logger.debug(f"Cluster tendency estimation failed: {e}")
            return DEFAULT_CLUSTER_TENDENCY
    
    def _recommend_kmeans_params(self, X, data_chars, n_recommendations, use_history):
        """Recommend K-means parameters"""
        recommendations = []
        
        # Estimate optimal k using multiple methods
        k_estimates = []
        
        # Elbow method approximation
        max_k = min(20, X.shape[0] // 10)
        if max_k > 2:
            k_elbow = self._estimate_k_elbow(X, max_k)
            k_estimates.append(k_elbow)
        
        # Rule-of-thumb estimation
        k_rule = int(np.sqrt(X.shape[0] / 2))
        k_estimates.append(max(2, min(k_rule, max_k)))
        
        # Density-based estimation
        if data_chars['density_estimate'] > 0:
            k_density = max(2, int(np.log(data_chars['density_estimate']) + 3))
            k_estimates.append(min(k_density, max_k))
        
        # Generate recommendations
        k_values = sorted(set(k_estimates))[:n_recommendations]
        
        for i, k in enumerate(k_values):
            # Adaptive max_iter based on data size
            max_iter = 300 if X.shape[0] < 1000 else 100
            
            recommendation = {
                'n_clusters': k,
                'max_iter': max_iter,
                'n_init': 10 if X.shape[0] < 5000 else 3,
                'tol': 1e-4 if data_chars['noise_estimate'] < 0.1 else 1e-3,
                'confidence': 0.9 - (i * 0.2),
                'reasoning': f"Estimated using {'elbow method' if i == 0 else 'rule-of-thumb' if i == 1 else 'density analysis'}"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _estimate_k_elbow(self, X, max_k):
        """Fast elbow method estimation for k"""
        from sklearn.cluster import KMeans

        # Use sample for speed
        sample_size = min(500, X.shape[0])
        if sample_size < X.shape[0]:
            indices = self._rng.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        inertias = []
        k_range = range(2, min(max_k + 1, sample_size))

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=self.random_state)
                kmeans.fit(X_sample)
                inertias.append(kmeans.inertia_)
            except (ValueError, RuntimeError) as e:
                logger.debug(f"K-means failed for k={k}: {e}")
                break

        if len(inertias) < 3:
            return 3

        # Find elbow using second derivative
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_idx = np.argmax(diffs2) + 2  # +2 because we start from k=2 and took second diff

        return min(k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1], max_k)

    def _recommend_dbscan_params(self, X, data_chars, n_recommendations, use_history):
        """Recommend DBSCAN parameters"""
        from sklearn.neighbors import NearestNeighbors

        recommendations = []

        # Estimate eps using k-distance plot
        k = 4  # Standard choice
        sample_size = min(500, X.shape[0])
        if sample_size < X.shape[0]:
            indices = self._rng.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_sample)
        distances, _ = nbrs.kneighbors(X_sample)
        k_distances = np.sort(distances[:, k-1])
        
        # Find knee point (simplified)
        knee_idx = len(k_distances) // 3
        eps_base = k_distances[knee_idx]
        
        # Generate multiple eps values
        eps_values = [
            eps_base * 0.7,  # Conservative
            eps_base,        # Standard
            eps_base * 1.3   # Liberal
        ]
        
        # Adaptive min_samples
        min_samples_base = max(2, int(np.log(X.shape[0])))
        
        for i, eps in enumerate(eps_values[:n_recommendations]):
            min_samples = min_samples_base + i - 1  # Vary min_samples slightly
            min_samples = max(2, min_samples)
            
            recommendation = {
                'eps': eps,
                'min_samples': min_samples,
                'confidence': 0.8 - (i * 0.15),
                'reasoning': f"{'Conservative' if i == 0 else 'Standard' if i == 1 else 'Liberal'} k-distance estimation"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_spectral_params(self, X, data_chars, n_recommendations, use_history):
        """Recommend Spectral Clustering parameters"""
        recommendations = []
        
        # Estimate number of clusters
        k_estimates = [3, int(np.sqrt(X.shape[0] / 2)), min(10, X.shape[0] // 20)]
        k_estimates = sorted(set([max(2, k) for k in k_estimates]))
        
        # Gamma parameter based on data characteristics
        data_scale = data_chars['data_scale']
        gamma_base = 1.0 / (data_scale * X.shape[1]) if data_scale > 0 else 1.0
        
        gamma_values = [gamma_base * 0.1, gamma_base, gamma_base * 10]
        
        for i in range(min(n_recommendations, len(k_estimates))):
            recommendation = {
                'n_clusters': k_estimates[i],
                'gamma': gamma_values[i % len(gamma_values)],
                'confidence': 0.7 - (i * 0.1),
                'reasoning': f"Cluster count estimation with adaptive gamma scaling"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_meanshift_params(self, X, data_chars, n_recommendations, use_history):
        """Recommend Mean Shift parameters"""
        from sklearn.cluster import estimate_bandwidth

        recommendations = []

        try:
            # Estimate bandwidth using sklearn's method
            sample_size = min(300, X.shape[0])
            if sample_size < X.shape[0]:
                indices = self._rng.choice(X.shape[0], sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            bandwidth_base = estimate_bandwidth(X_sample, quantile=0.2, n_samples=sample_size)

            if bandwidth_base <= 0:
                bandwidth_base = max(data_chars['data_scale'] * 0.5, EPSILON)

        except (ValueError, RuntimeError) as e:
            logger.debug(f"Bandwidth estimation failed: {e}")
            bandwidth_base = max(data_chars['data_scale'] * 0.5, EPSILON)
        
        # Generate multiple bandwidth values
        bandwidth_values = [
            bandwidth_base * 0.5,  # Fine-grained
            bandwidth_base,        # Standard
            bandwidth_base * 2.0   # Coarse-grained
        ]
        
        for i, bandwidth in enumerate(bandwidth_values[:n_recommendations]):
            recommendation = {
                'bandwidth': bandwidth,
                'max_iter': 300,
                'confidence': 0.75 - (i * 0.15),
                'reasoning': f"{'Fine-grained' if i == 0 else 'Standard' if i == 1 else 'Coarse-grained'} bandwidth estimation"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_fuzzy_params(self, X, data_chars, n_recommendations, use_history):
        """Recommend Fuzzy C-Means parameters"""
        recommendations = []
        
        # Estimate k similar to K-means
        k_estimates = [3, max(2, int(np.sqrt(X.shape[0] / 2))), min(8, X.shape[0] // 15)]
        k_estimates = sorted(set(k_estimates))
        
        # Fuzziness parameter based on data characteristics
        if data_chars['noise_estimate'] > 0.5:
            m_values = [2.5, 3.0, 2.0]  # Higher fuzziness for noisy data
        else:
            m_values = [2.0, 2.5, 1.5]  # Lower fuzziness for clean data
        
        for i in range(min(n_recommendations, len(k_estimates))):
            recommendation = {
                'n_clusters': k_estimates[i],
                'm': m_values[i % len(m_values)],
                'max_iter': 300,
                'tol': 1e-4,
                'confidence': 0.7 - (i * 0.1),
                'reasoning': f"Adaptive fuzziness based on estimated noise level"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_affinity_params(self, X, data_chars, n_recommendations, use_history):
        """Recommend Enhanced Affinity Propagation parameters"""
        recommendations = []
        
        # Damping values based on data stability
        if data_chars['cluster_tendency'] > 0.7:
            damping_values = [0.5, 0.7, 0.6]  # More stable for well-clustered data
        else:
            damping_values = [0.8, 0.9, 0.7]  # Higher damping for difficult data
        
        for i, damping in enumerate(damping_values[:n_recommendations]):
            recommendation = {
                'damping': damping,
                'max_iter': 200 if X.shape[0] < 1000 else 100,
                'adaptive_damping': True,
                'confidence': 0.6 - (i * 0.1),
                'reasoning': f"Adaptive damping based on cluster tendency analysis"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_generic_params(self, X, data_chars, n_recommendations):
        """Generic parameter recommendations for unknown algorithms"""
        recommendations = []
        
        for i in range(n_recommendations):
            recommendation = {
                'n_clusters': max(2, int(np.sqrt(X.shape[0] / 2)) + i),
                'confidence': 0.5 - (i * 0.1),
                'reasoning': f"Generic rule-of-thumb estimation (variant {i+1})"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def update_history(self, algorithm_name, params, X, score, execution_time):
        """Update recommendation history with results"""
        entry = {
            'timestamp': time.time(),
            'algorithm': algorithm_name,
            'params': params,
            'data_shape': X.shape,
            'data_characteristics': self._analyze_data_characteristics(X, fast_mode=True),
            'score': score,
            'execution_time': execution_time
        }
        
        self.recommendation_history_.append(entry)
        
        # Keep only recent history (last MAX_HISTORY_SIZE entries)
        if len(self.recommendation_history_) > MAX_HISTORY_SIZE:
            self.recommendation_history_ = self.recommendation_history_[-MAX_HISTORY_SIZE:]
    
    def get_recommendation_statistics(self):
        """Get statistics about recommendation performance"""
        if not self.recommendation_history_:
            return {}
        
        scores = [entry['score'] for entry in self.recommendation_history_]
        times = [entry['execution_time'] for entry in self.recommendation_history_]
        
        return {
            'n_recommendations': len(self.recommendation_history_),
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'average_time': np.mean(times),
            'best_score': max(scores),
            'algorithms_used': list(set(entry['algorithm'] for entry in self.recommendation_history_))
        }


# Integration class for the main clustering module
class AdvancedParameterOptimizer:
    """
    Main interface for advanced parameter optimization in the clustering module
    Combines Bayesian optimization with adaptive recommendations
    """
    
    def __init__(self, random_state=None):
        self.bayesian_optimizer = BayesianClusterOptimizer(random_state=random_state)
        self.adaptive_recommender = AdaptiveParameterRecommender(random_state=random_state)
        self.optimization_results_ = {}
        
    def optimize_algorithm(self, X, algorithm_name, algorithm_class, param_space, 
                          optimization_method='adaptive', n_recommendations=3, 
                          bayesian_steps=15, verbose=False):
        """
        Optimize parameters for a clustering algorithm
        
        Parameters:
        -----------
        X : array-like
            Input data
        algorithm_name : str
            Name of the algorithm
        algorithm_class : class
            Algorithm class
        param_space : dict
            Parameter space definition
        optimization_method : str, default='adaptive'
            Optimization method: 'adaptive', 'bayesian', 'both'
        n_recommendations : int, default=3
            Number of recommendations for adaptive method
        bayesian_steps : int, default=15
            Number of Bayesian optimization steps
        verbose : bool, default=False
            Whether to print progress
            
        Returns:
        --------
        results : dict
            Optimization results
        """
        
        start_time = time.time()
        results = {
            'algorithm': algorithm_name,
            'optimization_method': optimization_method,
            'start_time': start_time
        }
        
        if optimization_method in ['adaptive', 'both']:
            # Get adaptive recommendations
            if verbose:
                logger.info(f"Getting adaptive recommendations for {algorithm_name}...")

            recommendations = self.adaptive_recommender.recommend_parameters(
                X, algorithm_name, n_recommendations, use_history=True
            )

            # Evaluate recommendations
            best_adaptive_score = -np.inf
            best_adaptive_params = None

            for i, rec in enumerate(recommendations):
                if verbose:
                    logger.info(f"Evaluating adaptive recommendation {i+1}/{len(recommendations)}...")

                try:
                    # Remove non-parameter keys
                    params = {k: v for k, v in rec.items()
                             if k not in ['confidence', 'reasoning']}

                    algorithm = algorithm_class(**params)
                    algorithm.fit(X)

                    if hasattr(algorithm, 'labels_'):
                        labels = algorithm.labels_
                        if len(np.unique(labels)) > 1:
                            score = silhouette_score(X, labels)

                            if score > best_adaptive_score:
                                best_adaptive_score = score
                                best_adaptive_params = params

                            # Update history
                            exec_time = time.time() - start_time
                            self.adaptive_recommender.update_history(
                                algorithm_name, params, X, score, exec_time
                            )
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    if verbose:
                        logger.debug(f"Recommendation {i+1} failed: {e}")
                    continue
                except Exception as e:
                    if verbose:
                        logger.warning(f"Unexpected error in recommendation {i+1}: {e}")
                    continue

            results['adaptive_results'] = {
                'recommendations': recommendations,
                'best_score': best_adaptive_score,
                'best_params': best_adaptive_params
            }

        if optimization_method in ['bayesian', 'both']:
            # Run Bayesian optimization
            if verbose:
                logger.info(f"Running Bayesian optimization for {algorithm_name}...")

            try:
                best_params, best_score = self.bayesian_optimizer.optimize(
                    X, algorithm_class, param_space,
                    n_optimization_steps=bayesian_steps,
                    verbose=verbose
                )

                results['bayesian_results'] = {
                    'best_score': best_score,
                    'best_params': best_params,
                    'optimization_summary': self.bayesian_optimizer.get_optimization_summary()
                }
            except (ValueError, RuntimeError) as e:
                if verbose:
                    logger.warning(f"Bayesian optimization failed: {e}")
                results['bayesian_results'] = None
        
        # Determine overall best result
        best_score = -np.inf
        best_params = None
        best_method = None
        
        if 'adaptive_results' in results and results['adaptive_results']['best_score'] > best_score:
            best_score = results['adaptive_results']['best_score']
            best_params = results['adaptive_results']['best_params']
            best_method = 'adaptive'
        
        if 'bayesian_results' in results and results['bayesian_results'] and \
           results['bayesian_results']['best_score'] > best_score:
            best_score = results['bayesian_results']['best_score']
            best_params = results['bayesian_results']['best_params']
            best_method = 'bayesian'
        
        results.update({
            'best_overall_score': best_score,
            'best_overall_params': best_params,
            'best_method': best_method,
            'total_time': time.time() - start_time
        })
        
        # Store results
        self.optimization_results_[algorithm_name] = results
        
        return results
    
    def get_optimization_history(self, algorithm_name=None):
        """Get optimization history"""
        if algorithm_name:
            return self.optimization_results_.get(algorithm_name, {})
        else:
            return self.optimization_results_
    
    def clear_history(self):
        """Clear optimization history"""
        self.optimization_results_.clear()
        self.adaptive_recommender.recommendation_history_.clear()
        self.bayesian_optimizer.optimization_history_.clear()