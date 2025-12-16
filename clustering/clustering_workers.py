"""
Enhanced clustering worker threads with progress tracking
"""

import numpy as np
import logging
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, OPTICS, MeanShift

# Handle BIRCH import (different versions use different capitalization)
try:
    from sklearn.cluster import BIRCH
except ImportError:
    try:
        from sklearn.cluster import Birch as BIRCH
    except ImportError:
        BIRCH = None

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)


class DimensionalityReductionWorker(QThread):
    """Worker thread for dimensionality reduction with progress tracking"""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, X, algorithm, n_components, parameters):
        super().__init__()
        self.X = X
        self.algorithm = algorithm
        self.n_components = n_components
        self.parameters = parameters
        self._should_stop = False
        self._mutex = QMutex()

    def stop(self):
        """Request cooperative stop of the worker thread"""
        self._mutex.lock()
        self._should_stop = True
        self._mutex.unlock()

    def _check_stop(self):
        """Check if stop has been requested"""
        self._mutex.lock()
        should_stop = self._should_stop
        self._mutex.unlock()
        return should_stop
        
    def run(self):
        """Run dimensionality reduction with progress updates"""
        try:
            self.status.emit(f"Initializing {self.algorithm}...")
            self.progress.emit(10)

            if self._check_stop():
                return

            # Initialize algorithm
            if self.algorithm == 'PCA':
                reducer = PCA(
                    n_components=self.n_components,
                    random_state=self.parameters.get('random_seed', 42)
                )
                self.status.emit("Computing principal components...")
                self.progress.emit(30)

            elif self.algorithm == 't-SNE':
                reducer = TSNE(
                    n_components=self.n_components,
                    perplexity=self.parameters.get('perplexity', 30),
                    random_state=self.parameters.get('random_seed', 42),
                    max_iter=1000,
                    n_iter_without_progress=300
                )
                self.status.emit("Optimizing t-SNE embedding... This may take a while...")
                self.progress.emit(20)

            elif self.algorithm == 'UMAP' and UMAP_AVAILABLE:
                reducer = umap.UMAP(
                    n_components=self.n_components,
                    n_neighbors=int(self.parameters.get('n_neighbors', 15)),
                    random_state=self.parameters.get('random_seed', 42)
                )
                self.status.emit("Building neighborhood graph for UMAP...")
                self.progress.emit(25)

            elif self.algorithm == 'Isomap':
                reducer = Isomap(
                    n_components=self.n_components,
                    n_neighbors=int(self.parameters.get('n_neighbors', 10))
                )
                self.status.emit("Computing geodesic distances...")
                self.progress.emit(25)

            if self._check_stop():
                return

            # Fit and transform
            self.progress.emit(50)
            self.status.emit(f"Applying {self.algorithm} transformation...")

            X_reduced = reducer.fit_transform(self.X)

            if self._check_stop():
                return

            self.progress.emit(90)
            self.status.emit("Finalizing results...")

            # Prepare results
            results = {
                'X_reduced': X_reduced,
                'reducer': reducer,
                'algorithm': self.algorithm,
                'parameters': self.parameters
            }

            if hasattr(reducer, 'explained_variance_ratio_'):
                results['explained_variance'] = reducer.explained_variance_ratio_

            self.progress.emit(100)
            self.status.emit(f"{self.algorithm} completed successfully!")
            self.finished.emit(results)

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            self.error.emit(f"{self.algorithm} failed: {str(e)}")
        except Exception as e:
            logger.warning(f"Unexpected error in {self.algorithm}: {e}")
            self.error.emit(f"{self.algorithm} failed: {str(e)}")


class ClusteringWorker(QThread):
    """Worker thread for clustering with progress tracking"""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, X, algorithm, parameters):
        super().__init__()
        self.X = X
        self.algorithm = algorithm
        self.parameters = parameters
        self._should_stop = False
        self._mutex = QMutex()

    def stop(self):
        """Request cooperative stop of the worker thread"""
        self._mutex.lock()
        self._should_stop = True
        self._mutex.unlock()

    def _check_stop(self):
        """Check if stop has been requested"""
        self._mutex.lock()
        should_stop = self._should_stop
        self._mutex.unlock()
        return should_stop

    def run(self):
        """Run clustering with progress updates"""
        try:
            self.status.emit(f"Initializing {self.algorithm}...")
            self.progress.emit(10)

            if self._check_stop():
                return

            # Initialize clustering algorithm
            model = None  # Initialize model to avoid UnboundLocalError
            
            if self.algorithm == 'K-Means':
                model = KMeans(
                    n_clusters=self.parameters['n_clusters'],
                    max_iter=self.parameters.get('max_iter', 300),
                    random_state=self.parameters.get('random_seed', 42),
                    n_init=10
                )
                self.status.emit(f"Running K-Means with {self.parameters['n_clusters']} clusters...")
                
            elif self.algorithm == 'DBSCAN':
                model = DBSCAN(
                    eps=self.parameters['eps'],
                    min_samples=self.parameters['min_samples']
                )
                self.status.emit("Identifying density-based clusters...")
                
            elif self.algorithm == 'Agglomerative':
                model = AgglomerativeClustering(
                    n_clusters=self.parameters['n_clusters'],
                    linkage=self.parameters.get('linkage', 'ward')
                )
                self.status.emit("Building hierarchical clusters...")
                
            elif self.algorithm == 'Gaussian Mixture':
                model = GaussianMixture(
                    n_components=self.parameters['n_components'],
                    covariance_type=self.parameters.get('covariance_type', 'full'),
                    random_state=self.parameters.get('random_seed', 42)
                )
                self.status.emit("Fitting Gaussian mixture model...")
                
            elif self.algorithm == 'Spectral Clustering':
                model = SpectralClustering(
                    n_clusters=self.parameters['n_clusters'],
                    gamma=self.parameters.get('gamma', 1.0),
                    random_state=self.parameters.get('random_seed', 42)
                )
                self.status.emit("Computing spectral embedding...")
                
            
            elif self.algorithm == 'OPTICS':
                model = OPTICS(
                    min_samples=self.parameters.get('min_samples', 5),
                    xi=self.parameters.get('xi', 0.05),
                    min_cluster_size=self.parameters.get('min_cluster_size', None)
                )
                self.status.emit("Running OPTICS clustering...")

            elif self.algorithm == 'HDBSCAN':
                try:
                    import hdbscan
                except ImportError:
                    self.error.emit("HDBSCAN is not available. Please install the hdbscan package.")
                    return
                model = hdbscan.HDBSCAN(
                    min_cluster_size=self.parameters.get('min_cluster_size', 5),
                    min_samples=self.parameters.get('min_samples', None)
                )
                self.status.emit("Running HDBSCAN clustering...")
            elif self.algorithm == 'BIRCH' and BIRCH is not None:
                model = BIRCH(
                    n_clusters=self.parameters['n_clusters'],
                    threshold=self.parameters.get('threshold', 0.5),
                    branching_factor=self.parameters.get('branching_factor', 50)
                )
                self.status.emit("Building CF tree for BIRCH...")
                
            elif self.algorithm == 'Mean Shift':
                from sklearn.cluster import estimate_bandwidth
                # Handle bandwidth parameter
                if self.parameters.get('bandwidth_mode') == 'auto':
                    bandwidth = estimate_bandwidth(self.X, quantile=0.2, n_samples=500)
                    if bandwidth <= 0:
                        bandwidth = 1.0  # Fallback bandwidth
                else:
                    bandwidth = self.parameters.get('custom_bandwidth', 1.0)
                    
                model = MeanShift(
                    bandwidth=bandwidth,
                    max_iter=self.parameters.get('max_iter', 300)
                )
                self.status.emit("Finding density modes with Mean Shift...")
                
            else:
                # Handle unsupported algorithms
                self.error.emit(f"Unsupported algorithm: {self.algorithm}")
                return
                
            # Ensure model was created
            if model is None:
                self.error.emit(f"Failed to initialize clustering model for {self.algorithm}")
                return

            if self._check_stop():
                return

            self.progress.emit(30)

            # Fit the model
            if self.algorithm == 'Gaussian Mixture':
                self.status.emit("Estimating mixture parameters...")
                self.progress.emit(50)
                labels = model.fit_predict(self.X)
            else:
                self.status.emit("Computing cluster assignments...")
                self.progress.emit(50)
                model.fit(self.X)
                labels = model.labels_

            if self._check_stop():
                return

            self.progress.emit(80)
            self.status.emit("Analyzing clustering results...")

            # Analyze results
            unique_labels = np.unique(labels[labels >= 0]) if -1 in labels else np.unique(labels)
            n_clusters = len(unique_labels)
            n_noise = np.sum(labels == -1) if -1 in labels else 0

            results = {
                'model': model,
                'labels': labels,
                'algorithm': self.algorithm,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'X_clustered': self.X
            }

            self.progress.emit(100)
            status_msg = f"{self.algorithm} completed: {n_clusters} clusters"
            if n_noise > 0:
                status_msg += f", {n_noise} noise points"
            self.status.emit(status_msg)

            self.finished.emit(results)

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            self.error.emit(f"Clustering failed: {str(e)}")
        except Exception as e:
            logger.warning(f"Unexpected error in clustering: {e}")
            self.error.emit(f"Clustering failed: {str(e)}")


class EvaluationWorker(QThread):
    """Worker thread for clustering evaluation with progress tracking"""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, X, labels):
        super().__init__()
        self.X = X
        self.labels = labels
        self._should_stop = False
        self._mutex = QMutex()

    def stop(self):
        """Request cooperative stop of the worker thread"""
        self._mutex.lock()
        self._should_stop = True
        self._mutex.unlock()

    def _check_stop(self):
        """Check if stop has been requested"""
        self._mutex.lock()
        should_stop = self._should_stop
        self._mutex.unlock()
        return should_stop

    def run(self):
        """Compute evaluation metrics with progress updates"""
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

            self.status.emit("Preparing data for evaluation...")
            self.progress.emit(10)

            # Remove noise points for evaluation
            mask = self.labels != -1
            X_clean = self.X[mask]
            labels_clean = self.labels[mask]

            if len(np.unique(labels_clean)) < 2:
                self.error.emit("Cannot compute metrics: insufficient clusters")
                return

            if self._check_stop():
                return

            self.status.emit("Computing silhouette score...")
            self.progress.emit(30)
            sil_score = silhouette_score(X_clean, labels_clean)

            if self._check_stop():
                return

            self.status.emit("Computing Davies-Bouldin index...")
            self.progress.emit(60)
            db_score = davies_bouldin_score(X_clean, labels_clean)

            if self._check_stop():
                return

            self.status.emit("Computing Calinski-Harabasz index...")
            self.progress.emit(90)
            ch_score = calinski_harabasz_score(X_clean, labels_clean)

            metrics = {
                'silhouette_score': sil_score,
                'davies_bouldin_score': db_score,
                'calinski_harabasz_score': ch_score
            }

            self.progress.emit(100)
            self.status.emit("Evaluation metrics computed successfully!")
            self.finished.emit(metrics)

        except (ValueError, RuntimeError) as e:
            self.error.emit(f"Evaluation failed: {str(e)}")
        except Exception as e:
            logger.warning(f"Unexpected error in evaluation: {e}")
            self.error.emit(f"Evaluation failed: {str(e)}")
