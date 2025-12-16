"""
Real-time parameter adjustment with preview functionality
"""

import numpy as np
import logging
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QMutex
from PyQt5.QtWidgets import QLabel, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QCheckBox
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, MeanShift

# Handle BIRCH import (different versions use different capitalization)
try:
    from sklearn.cluster import BIRCH
except ImportError:
    try:
        from sklearn.cluster import Birch as BIRCH
    except ImportError:
        BIRCH = None

from sklearn.metrics import silhouette_score
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Configure matplotlib to handle Unicode properly
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Configure module logger
logger = logging.getLogger(__name__)


class RealtimePreviewWorker(QThread):
    """Lightweight clustering for real-time preview with cooperative cancellation"""

    preview_ready = pyqtSignal(dict)

    def __init__(self, X_sample, algorithm, parameters):
        super().__init__()
        self.X_sample = X_sample
        self.algorithm = algorithm
        self.parameters = parameters
        self._should_stop = False
        self._mutex = QMutex()

    def request_stop(self):
        """Request cooperative stop of the worker"""
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
        try:
            # Check for early termination
            if self._check_stop():
                return

            if self.algorithm == 'K-Means':
                model = KMeans(
                    n_clusters=self.parameters['n_clusters'],
                    random_state=42,
                    n_init=3  # Reduced for speed
                )
                labels = model.fit_predict(self.X_sample)
                centers = model.cluster_centers_

            elif self.algorithm == 'DBSCAN':
                model = DBSCAN(
                    eps=self.parameters['eps'],
                    min_samples=self.parameters['min_samples']
                )
                labels = model.fit_predict(self.X_sample)
                centers = None

            elif self.algorithm == 'Spectral Clustering':
                model = SpectralClustering(
                    n_clusters=self.parameters.get('n_clusters', 3),
                    random_state=42,
                    gamma=self.parameters.get('gamma', 1.0),
                    n_init=3  # Reduced for speed
                )
                labels = model.fit_predict(self.X_sample)
                centers = None

            elif self.algorithm == 'BIRCH' and BIRCH is not None:
                model = BIRCH(
                    n_clusters=self.parameters.get('n_clusters', 3),
                    threshold=self.parameters.get('threshold', 0.5),
                    branching_factor=self.parameters.get('branching_factor', 50)
                )
                labels = model.fit_predict(self.X_sample)
                centers = None

            elif self.algorithm == 'Mean Shift':
                # For Mean Shift, use a simplified bandwidth calculation for preview
                from sklearn.cluster import estimate_bandwidth
                if self.parameters.get('bandwidth_mode') == 'auto':
                    try:
                        bandwidth = estimate_bandwidth(self.X_sample, quantile=0.2, n_samples=min(100, len(self.X_sample)))
                        if bandwidth <= 0:
                            bandwidth = 1.0  # Fallback
                    except (ValueError, RuntimeError):
                        bandwidth = 1.0
                else:
                    bandwidth = self.parameters.get('custom_bandwidth', 1.0)

                model = MeanShift(
                    bandwidth=bandwidth,
                    max_iter=50  # Reduced for speed
                )
                labels = model.fit_predict(self.X_sample)
                centers = model.cluster_centers_

            else:
                # Fallback for unsupported algorithms
                labels = np.zeros(len(self.X_sample), dtype=int)
                centers = None

            # Check for cancellation before computing metrics
            if self._check_stop():
                return

            # Compute basic metrics
            unique_labels = np.unique(labels[labels >= 0])
            n_clusters = len(unique_labels)
            n_noise = np.sum(labels == -1) if -1 in labels else 0

            # Quick silhouette score if we have enough clusters
            sil_score = None
            if n_clusters >= 2 and len(labels[labels >= 0]) >= 2:
                try:
                    clean_mask = labels != -1
                    if np.sum(clean_mask) >= 2:
                        sil_score = silhouette_score(self.X_sample[clean_mask], labels[clean_mask])
                except (ValueError, RuntimeError):
                    sil_score = None
                    
            result = {
                'labels': labels,
                'centers': centers,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette_score': sil_score
            }
            
            self.preview_ready.emit(result)
            
        except Exception as e:
            # If preview fails, just emit empty result
            self.preview_ready.emit({'error': str(e)})


class RealtimePreviewWidget(QWidget):
    """Widget for real-time parameter adjustment with preview"""
    
    def __init__(self, parent_module):
        super().__init__()
        self.parent_module = parent_module
        self.preview_enabled = True
        self.preview_worker = None
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_delay = 500  # 500ms delay for responsiveness
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the real-time preview UI"""
        layout = QHBoxLayout()
        
        # Left side: Parameter controls
        controls_layout = QVBoxLayout()
        
        # Preview toggle
        self.preview_toggle = QCheckBox("Enable Real-time Preview")
        self.preview_toggle.setChecked(True)
        self.preview_toggle.toggled.connect(self.toggle_preview)
        controls_layout.addWidget(self.preview_toggle)
        
        # Sample size for preview
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Preview Sample Size:"))
        self.sample_size_slider = QSlider()
        self.sample_size_slider.setOrientation(1)  # Horizontal
        self.sample_size_slider.setRange(100, 1000)
        self.sample_size_slider.setValue(200)
        self.sample_size_slider.valueChanged.connect(self.schedule_preview_update)
        self.sample_size_label = QLabel("200")
        sample_layout.addWidget(self.sample_size_slider)
        sample_layout.addWidget(self.sample_size_label)
        controls_layout.addLayout(sample_layout)
        
        # K-Means specific controls
        self.kmeans_controls = QWidget()
        kmeans_layout = QVBoxLayout(self.kmeans_controls)
        
        # K slider
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("K (clusters):"))
        self.k_slider = QSlider()
        self.k_slider.setOrientation(1)  # Horizontal
        self.k_slider.setRange(2, 10)
        self.k_slider.setValue(3)
        self.k_slider.valueChanged.connect(self.on_k_changed)
        self.k_label = QLabel("3")
        k_layout.addWidget(self.k_slider)
        k_layout.addWidget(self.k_label)
        kmeans_layout.addLayout(k_layout)
        
        controls_layout.addWidget(self.kmeans_controls)
        
        # DBSCAN specific controls
        self.dbscan_controls = QWidget()
        dbscan_layout = QVBoxLayout(self.dbscan_controls)
        
        # Eps slider
        eps_layout = QHBoxLayout()
        eps_layout.addWidget(QLabel("Epsilon:"))
        self.eps_slider = QSlider()
        self.eps_slider.setOrientation(1)  # Horizontal
        self.eps_slider.setRange(1, 100)  # Will be scaled to 0.01-1.0
        self.eps_slider.setValue(50)  # Default 0.5
        self.eps_slider.valueChanged.connect(self.on_eps_changed)
        self.eps_label = QLabel("0.50")
        eps_layout.addWidget(self.eps_slider)
        eps_layout.addWidget(self.eps_label)
        dbscan_layout.addLayout(eps_layout)
        
        # Min samples slider
        min_samples_layout = QHBoxLayout()
        min_samples_layout.addWidget(QLabel("Min Samples:"))
        self.min_samples_slider = QSlider()
        self.min_samples_slider.setOrientation(1)  # Horizontal
        self.min_samples_slider.setRange(2, 20)
        self.min_samples_slider.setValue(5)
        self.min_samples_slider.valueChanged.connect(self.on_min_samples_changed)
        self.min_samples_label = QLabel("5")
        min_samples_layout.addWidget(self.min_samples_slider)
        min_samples_layout.addWidget(self.min_samples_label)
        dbscan_layout.addLayout(min_samples_layout)
        
        controls_layout.addWidget(self.dbscan_controls)
        
        # Spectral Clustering specific controls
        self.spectral_controls = QWidget()
        spectral_layout = QVBoxLayout(self.spectral_controls)
        
        # K slider for spectral
        spectral_k_layout = QHBoxLayout()
        spectral_k_layout.addWidget(QLabel("Clusters:"))
        self.spectral_k_slider = QSlider()
        self.spectral_k_slider.setOrientation(1)  # Horizontal
        self.spectral_k_slider.setRange(2, 10)
        self.spectral_k_slider.setValue(3)
        self.spectral_k_slider.valueChanged.connect(self.on_spectral_k_changed)
        self.spectral_k_label = QLabel("3")
        spectral_k_layout.addWidget(self.spectral_k_slider)
        spectral_k_layout.addWidget(self.spectral_k_label)
        spectral_layout.addLayout(spectral_k_layout)
        
        # Gamma slider for spectral
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.gamma_slider = QSlider()
        self.gamma_slider.setOrientation(1)  # Horizontal
        self.gamma_slider.setRange(1, 100)  # Will be scaled to 0.01-10.0
        self.gamma_slider.setValue(10)  # Default 1.0
        self.gamma_slider.valueChanged.connect(self.on_gamma_changed)
        self.gamma_label = QLabel("1.0")
        gamma_layout.addWidget(self.gamma_slider)
        gamma_layout.addWidget(self.gamma_label)
        spectral_layout.addLayout(gamma_layout)
        
        controls_layout.addWidget(self.spectral_controls)
        
        # BIRCH specific controls
        self.birch_controls = QWidget()
        birch_layout = QVBoxLayout(self.birch_controls)
        
        # K slider for BIRCH
        birch_k_layout = QHBoxLayout()
        birch_k_layout.addWidget(QLabel("Clusters:"))
        self.birch_k_slider = QSlider()
        self.birch_k_slider.setOrientation(1)  # Horizontal
        self.birch_k_slider.setRange(2, 10)
        self.birch_k_slider.setValue(3)
        self.birch_k_slider.valueChanged.connect(self.on_birch_k_changed)
        self.birch_k_label = QLabel("3")
        birch_k_layout.addWidget(self.birch_k_slider)
        birch_k_layout.addWidget(self.birch_k_label)
        birch_layout.addLayout(birch_k_layout)
        
        # Threshold slider for BIRCH
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_slider = QSlider()
        self.threshold_slider.setOrientation(1)  # Horizontal
        self.threshold_slider.setRange(1, 100)  # Will be scaled to 0.1-10.0
        self.threshold_slider.setValue(5)  # Default 0.5
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.threshold_label = QLabel("0.5")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        birch_layout.addLayout(threshold_layout)
        
        controls_layout.addWidget(self.birch_controls)
        
        # Mean Shift specific controls
        self.meanshift_controls = QWidget()
        meanshift_layout = QVBoxLayout(self.meanshift_controls)
        
        # Bandwidth slider for Mean Shift
        bandwidth_layout = QHBoxLayout()
        bandwidth_layout.addWidget(QLabel("Bandwidth:"))
        self.bandwidth_slider = QSlider()
        self.bandwidth_slider.setOrientation(1)  # Horizontal
        self.bandwidth_slider.setRange(1, 100)  # Will be scaled to 0.1-10.0
        self.bandwidth_slider.setValue(10)  # Default 1.0
        self.bandwidth_slider.valueChanged.connect(self.on_bandwidth_changed)
        self.bandwidth_label = QLabel("1.0")
        bandwidth_layout.addWidget(self.bandwidth_slider)
        bandwidth_layout.addWidget(self.bandwidth_label)
        meanshift_layout.addLayout(bandwidth_layout)
        
        controls_layout.addWidget(self.meanshift_controls)
        
        # Initially hide all new algorithm controls
        self.spectral_controls.setVisible(False)
        self.birch_controls.setVisible(False)
        self.meanshift_controls.setVisible(False)
        
        # Initially hide DBSCAN controls
        self.dbscan_controls.setVisible(False)
        
        # Preview info
        self.preview_info = QLabel("Preview: Ready")
        self.preview_info.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        controls_layout.addWidget(self.preview_info)
        
        controls_layout.addStretch()
        
        # Right side: Preview plot
        self.preview_figure = Figure(figsize=(6, 4))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        
        # Add to main layout
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setMaximumWidth(300)
        
        layout.addWidget(controls_widget)
        layout.addWidget(self.preview_canvas)
        
        self.setLayout(layout)
        
    def set_algorithm(self, algorithm):
        """Set the current algorithm and show appropriate controls"""
        # Hide all controls first
        self.kmeans_controls.setVisible(False)
        self.dbscan_controls.setVisible(False)
        self.spectral_controls.setVisible(False)
        self.birch_controls.setVisible(False)
        self.meanshift_controls.setVisible(False)
        
        # Show appropriate controls based on algorithm
        if algorithm == 'K-Means':
            self.kmeans_controls.setVisible(True)
        elif algorithm == 'DBSCAN':
            self.dbscan_controls.setVisible(True)
        elif algorithm == 'Spectral Clustering':
            self.spectral_controls.setVisible(True)
        elif algorithm == 'BIRCH' and BIRCH is not None:
            self.birch_controls.setVisible(True)
        elif algorithm == 'Mean Shift':
            self.meanshift_controls.setVisible(True)
            
        self.current_algorithm = algorithm
        self.schedule_preview_update()
        
    def toggle_preview(self, enabled):
        """Toggle real-time preview on/off"""
        self.preview_enabled = enabled
        if enabled:
            self.schedule_preview_update()
        else:
            self.clear_preview()
            
    def clear_preview(self):
        """Clear the preview plot"""
        self.preview_figure.clear()
        self.preview_canvas.draw()
        self.preview_info.setText("Preview: Disabled")
        
    def on_k_changed(self, value):
        """Handle K slider change"""
        self.k_label.setText(str(value))
        if hasattr(self.parent_module, 'k_clusters_spin'):
            self.parent_module.k_clusters_spin.setValue(value)
        self.schedule_preview_update()
        
    def on_eps_changed(self, value):
        """Handle epsilon slider change"""
        eps_value = value / 100.0  # Scale to 0.01-1.0
        self.eps_label.setText(f"{eps_value:.2f}")
        if hasattr(self.parent_module, 'eps_spin'):
            self.parent_module.eps_spin.setValue(eps_value)
        self.schedule_preview_update()
        
    def on_min_samples_changed(self, value):
        """Handle min samples slider change"""
        self.min_samples_label.setText(str(value))
        if hasattr(self.parent_module, 'min_samples_spin'):
            self.parent_module.min_samples_spin.setValue(value)
        self.schedule_preview_update()
        
    def on_spectral_k_changed(self, value):
        """Handle Spectral Clustering K slider change"""
        self.spectral_k_label.setText(str(value))
        if hasattr(self.parent_module, 'spectral_n_clusters_spin'):
            self.parent_module.spectral_n_clusters_spin.setValue(value)
        self.schedule_preview_update()
        
    def on_gamma_changed(self, value):
        """Handle gamma slider change"""
        gamma_value = value / 10.0  # Scale to 0.1-10.0
        self.gamma_label.setText(f"{gamma_value:.1f}")
        if hasattr(self.parent_module, 'gamma_spin'):
            self.parent_module.gamma_spin.setValue(gamma_value)
        self.schedule_preview_update()
        
    def on_birch_k_changed(self, value):
        """Handle BIRCH K slider change"""
        self.birch_k_label.setText(str(value))
        if hasattr(self.parent_module, 'birch_n_clusters_spin'):
            self.parent_module.birch_n_clusters_spin.setValue(value)
        self.schedule_preview_update()
        
    def on_threshold_changed(self, value):
        """Handle threshold slider change"""
        threshold_value = value / 10.0  # Scale to 0.1-10.0
        self.threshold_label.setText(f"{threshold_value:.1f}")
        if hasattr(self.parent_module, 'threshold_spin'):
            self.parent_module.threshold_spin.setValue(threshold_value)
        self.schedule_preview_update()
        
    def on_bandwidth_changed(self, value):
        """Handle bandwidth slider change"""
        bandwidth_value = value / 10.0  # Scale to 0.1-10.0
        self.bandwidth_label.setText(f"{bandwidth_value:.1f}")
        if hasattr(self.parent_module, 'custom_bandwidth_spin'):
            self.parent_module.custom_bandwidth_spin.setValue(bandwidth_value)
            # Also set to custom mode
            if hasattr(self.parent_module, 'bandwidth_combo'):
                self.parent_module.bandwidth_combo.setCurrentText('Custom')
        self.schedule_preview_update()
        
    def schedule_preview_update(self):
        """Schedule a preview update with delay"""
        if self.preview_enabled:
            self.preview_timer.start(self.preview_delay)
            self.sample_size_label.setText(str(self.sample_size_slider.value()))
            
    def update_preview(self):
        """Update the preview visualization"""
        if not self.preview_enabled:
            return

        # Check if we have reduced dimensionality data
        if not hasattr(self.parent_module, 'dimensionality_reduction_results') or \
           not self.parent_module.dimensionality_reduction_results:
            self.preview_info.setText("Preview: No data available")
            return

        try:
            # Get sample of reduced data
            X_full = self.parent_module.dimensionality_reduction_results['X_reduced']
            sample_size = min(self.sample_size_slider.value(), len(X_full))

            # Random sample for preview using local RNG
            rng = np.random.default_rng(42)  # Consistent sampling
            indices = rng.choice(len(X_full), sample_size, replace=False)
            X_sample = X_full[indices]

            if not hasattr(self, 'current_algorithm'):
                return

            # Prepare parameters
            parameters = {}
            if self.current_algorithm == 'K-Means':
                parameters['n_clusters'] = self.k_slider.value()
            elif self.current_algorithm == 'DBSCAN':
                parameters['eps'] = self.eps_slider.value() / 100.0
                parameters['min_samples'] = self.min_samples_slider.value()
            elif self.current_algorithm == 'Spectral Clustering':
                parameters['n_clusters'] = self.spectral_k_slider.value()
                parameters['gamma'] = self.gamma_slider.value() / 10.0
            elif self.current_algorithm == 'BIRCH' and BIRCH is not None:
                parameters['n_clusters'] = self.birch_k_slider.value()
                parameters['threshold'] = self.threshold_slider.value() / 10.0
            elif self.current_algorithm == 'Mean Shift':
                parameters['bandwidth_mode'] = 'custom'
                parameters['custom_bandwidth'] = self.bandwidth_slider.value() / 10.0
            else:
                return
                
            # Stop previous worker if running using cooperative cancellation
            if self.preview_worker and self.preview_worker.isRunning():
                self.preview_worker.request_stop()
                self.preview_worker.wait(1000)  # Wait up to 1 second
                if self.preview_worker.isRunning():
                    # Only terminate as last resort
                    logger.warning("Preview worker did not stop cooperatively, forcing termination")
                    self.preview_worker.terminate()
                    self.preview_worker.wait()

            # Start preview worker
            self.preview_worker = RealtimePreviewWorker(X_sample, self.current_algorithm, parameters)
            self.preview_worker.preview_ready.connect(self.display_preview)
            self.preview_worker.start()
            
            self.preview_info.setText("Preview: Computing...")
            
        except Exception as e:
            self.preview_info.setText(f"Preview: Error - {str(e)}")
            
    def display_preview(self, result):
        """Display the preview result"""
        if 'error' in result:
            self.preview_info.setText(f"Preview: Error - {result['error']}")
            return

        try:
            self.preview_figure.clear()
            ax = self.preview_figure.add_subplot(111)

            # Get sample data for plotting using local RNG
            X_full = self.parent_module.dimensionality_reduction_results['X_reduced']
            sample_size = min(self.sample_size_slider.value(), len(X_full))

            rng = np.random.default_rng(42)
            indices = rng.choice(len(X_full), sample_size, replace=False)
            X_sample = X_full[indices]

            labels = result['labels']

            # Plot scatter
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                if label == -1:
                    # Noise points
                    mask = labels == label
                    ax.scatter(X_sample[mask, 0], X_sample[mask, 1],
                             c='black', marker='x', s=30, alpha=0.6, label='Noise')
                else:
                    mask = labels == label
                    ax.scatter(X_sample[mask, 0], X_sample[mask, 1],
                             c=[colors[i]], s=30, alpha=0.7, label=f'Cluster {label}')

            # Plot cluster centers if available
            if result['centers'] is not None:
                centers = result['centers']
                ax.scatter(centers[:, 0], centers[:, 1],
                         c='red', marker='x', s=100, linewidths=3, label='Centers')

            ax.set_title(f'Preview: {self.current_algorithm}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            self.preview_figure.tight_layout()
            self.preview_canvas.draw()

            # Update info
            info_text = f"Clusters: {result['n_clusters']}"
            if result['n_noise'] > 0:
                info_text += f", Noise: {result['n_noise']}"
            if result['silhouette_score'] is not None:
                info_text += f", Sil: {result['silhouette_score']:.3f}"

            self.preview_info.setText(f"Preview: {info_text}")

        except Exception as e:
            self.preview_info.setText(f"Preview: Display error - {str(e)}")