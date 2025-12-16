"""
Intelligent Clustering Analysis Module
A comprehensive clustering analysis system with advanced visualization and evaluation capabilities
"""

import sys
import os
import pandas as pd
import numpy as np
import sqlite3
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import pickle
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QProgressBar, QSlider,
    QCheckBox, QListWidget, QListWidgetItem, QScrollArea, QSplitter,
    QFileDialog, QMessageBox, QInputDialog, QFrame, QSizePolicy,
    QTreeWidget, QTreeWidgetItem, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

# Clustering algorithms
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering,
    OPTICS, MeanShift
)

# Handle BIRCH import (different versions use different capitalization)
try:
    from sklearn.cluster import BIRCH
except ImportError:
    try:
        from sklearn.cluster import Birch as BIRCH
    except ImportError:
        BIRCH = None
        
from sklearn.mixture import GaussianMixture
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Evaluation metrics
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)

# Preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Configure matplotlib to handle Unicode properly
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio

from .clustering_workers import DimensionalityReductionWorker, ClusteringWorker, EvaluationWorker
from .realtime_preview import RealtimePreviewWidget
from .parameter_recommendation import ParameterRecommendationEngine

# Enhanced clustering features
from .advanced_algorithms import FuzzyCMeans, ConsensusKMeans, MiniBatchKMeansPlus, EnhancedAffinityPropagation, ADVANCED_ALGORITHMS
from .adaptive_optimization import BayesianClusterOptimizer, AdaptiveParameterRecommender, AdvancedParameterOptimizer
from .evaluation_metrics import ComprehensiveClusteringEvaluator
from .algorithm_selection import IntelligentAlgorithmSelector
from .interpretability import ComprehensiveClusteringInterpreter
from .performance_optimization import PerformanceOptimizedClusteringModule

warnings.filterwarnings('ignore')


class DataAuditWorker(QThread):
    """Worker thread for data auditing and quality assessment"""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data
        
    def run(self):
        """Perform comprehensive data audit"""
        self.status.emit("Starting data audit...")
        self.progress.emit(10)
        
        audit_report = {}
        
        # Basic statistics
        self.status.emit("Computing basic statistics...")
        audit_report['basic_stats'] = {
            'n_samples': len(self.data),
            'n_features': len(self.data.columns),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': self.data.duplicated().sum()
        }
        self.progress.emit(25)
        
        # Feature analysis
        self.status.emit("Analyzing features...")
        feature_info = []
        for col in self.data.columns:
            info = {
                'name': col,
                'dtype': str(self.data[col].dtype),
                'missing_count': self.data[col].isnull().sum(),
                'missing_percentage': (self.data[col].isnull().sum() / len(self.data)) * 100,
                'unique_count': self.data[col].nunique(),
                'zero_count': (self.data[col] == 0).sum() if pd.api.types.is_numeric_dtype(self.data[col]) else 0
            }
            
            if pd.api.types.is_numeric_dtype(self.data[col]):
                info.update({
                    'mean': float(self.data[col].mean()) if not self.data[col].isnull().all() else None,
                    'std': float(self.data[col].std()) if not self.data[col].isnull().all() else None,
                    'min': float(self.data[col].min()) if not self.data[col].isnull().all() else None,
                    'max': float(self.data[col].max()) if not self.data[col].isnull().all() else None,
                    'skewness': float(self.data[col].skew()) if not self.data[col].isnull().all() else None,
                    'kurtosis': float(self.data[col].kurtosis()) if not self.data[col].isnull().all() else None
                })
            
            feature_info.append(info)
        
        audit_report['feature_analysis'] = feature_info
        self.progress.emit(50)
        
        # Correlation analysis
        self.status.emit("Computing correlations...")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            audit_report['high_correlations'] = high_corr_pairs
        else:
            audit_report['high_correlations'] = []
        self.progress.emit(75)
        
        # Data quality alerts
        self.status.emit("Generating quality alerts...")
        alerts = []
        
        for feature in feature_info:
            # High missing rate alert
            if feature['missing_percentage'] > 50:
                alerts.append({
                    'type': 'high_missing_rate',
                    'feature': feature['name'],
                    'message': f"Feature '{feature['name']}' has {feature['missing_percentage']:.1f}% missing values",
                    'severity': 'high'
                })
            
            # Zero variance alert
            if feature['unique_count'] == 1:
                alerts.append({
                    'type': 'zero_variance',
                    'feature': feature['name'],
                    'message': f"Feature '{feature['name']}' has zero variance",
                    'severity': 'high'
                })
            
            # High cardinality alert for categorical
            if not pd.api.types.is_numeric_dtype(self.data[feature['name']]) and feature['unique_count'] > 100:
                alerts.append({
                    'type': 'high_cardinality',
                    'feature': feature['name'],
                    'message': f"Categorical feature '{feature['name']}' has {feature['unique_count']} unique values",
                    'severity': 'medium'
                })
        
        # High correlation alert
        for corr_pair in audit_report['high_correlations']:
            alerts.append({
                'type': 'high_correlation',
                'feature': f"{corr_pair['feature1']} & {corr_pair['feature2']}",
                'message': f"High correlation ({corr_pair['correlation']:.3f}) between features",
                'severity': 'medium'
            })
        
        audit_report['quality_alerts'] = alerts
        
        self.progress.emit(100)
        self.status.emit("Data audit complete")
        self.finished.emit(audit_report)


class IntelligentClusteringModule(QWidget):
    """Main clustering analysis module with six-stage workflow"""
    
    # Signals
    data_processed = pyqtSignal(pd.DataFrame)
    clustering_complete = pyqtSignal(dict)
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        
        # Data state management
        self.raw_data = None
        self.processed_data = None
        self.current_data = None
        self.feature_columns = []
        self.selected_features = []
        self.preprocessing_history = []
        
        # Clustering state
        self.clustering_results = {}
        self.current_model = None
        self.cluster_labels = None
        self.dimensionality_reduction_results = {}
        
        # Configuration
        self.random_seed = 42
        self.export_settings = {
            'dpi': 300,
            'format': 'png',
            'width': 12,
            'height': 8
        }
        
        # Enhanced clustering components
        self.intelligent_algorithm_selector = None
        self.comprehensive_evaluator = None
        self.clustering_interpreter = None
        self.parameter_optimizer = None
        self.performance_optimizer = None
        self.current_algorithm_recommendation = None
        
        # Initialize UI
        self.init_ui()
        
        # Initialize enhanced components after UI is ready
        self.init_enhanced_features()
        
    def init_ui(self):
        """Initialize the user interface with six-stage workflow"""
        # Main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title_label = QLabel("Intelligent Clustering Analysis Module")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("QLabel { color: #2E86C1; margin: 10px; }")
        layout.addWidget(title_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to start clustering analysis")
        self.status_label.setStyleSheet("QLabel { color: #666; margin: 5px; }")
        layout.addWidget(self.status_label)
        
        # Main tab widget for six stages
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Stage 1: Data Input and Intelligent Audit
        self.setup_stage1_data_input()
        
        # Stage 2: Advanced Feature Engineering
        self.setup_stage2_feature_engineering()
        
        # Stage 3: Multi-Algorithm Clustering Engine  
        self.setup_stage3_clustering_engine()
        
        # Stage 4: Comprehensive Evaluation
        self.setup_stage4_evaluation()
        
        # Stage 5: Multi-Perspective Visualization
        self.setup_stage5_visualization()
        
        # Stage 6: Report Generation and Export
        self.setup_stage6_export()
        
        # Connect signals
        self.setup_signals()
        
    
    def setup_signals(self):
        """Connect basic signals to UI widgets (safe-guard)."""
        try:
            if hasattr(self, 'status_updated') and hasattr(self, 'status_label'):
                self.status_updated.connect(self.status_label.setText)
            if hasattr(self, 'progress_updated') and hasattr(self, 'progress_bar'):
                self.progress_updated.connect(self.progress_bar.setValue)
        except Exception:
            pass
    def setup_stage1_data_input(self):
        """Stage 1: Data Input and Intelligent Audit"""
        stage1_widget = QWidget()
        layout = QVBoxLayout(stage1_widget)
        
        # Data connection section
        connection_group = QGroupBox("1.1 Multi-Source Data Connector")
        connection_layout = QGridLayout(connection_group)
        
        # File input buttons
        self.load_csv_btn = QPushButton("Load CSV File")
        self.load_excel_btn = QPushButton("Load Excel File")
        self.load_database_btn = QPushButton("Connect to Database")
        
        connection_layout.addWidget(QLabel("Local Files:"), 0, 0)
        connection_layout.addWidget(self.load_csv_btn, 0, 1)
        connection_layout.addWidget(self.load_excel_btn, 0, 2)
        connection_layout.addWidget(QLabel("Database:"), 1, 0)
        connection_layout.addWidget(self.load_database_btn, 1, 1)
        
        layout.addWidget(connection_group)
        
        # Data preview and audit section
        audit_group = QGroupBox("1.2 Automated Data Preview and Health Report")
        audit_layout = QVBoxLayout(audit_group)
        
        # Data preview table
        self.data_preview_table = QTableWidget()
        self.data_preview_table.setMaximumHeight(200)
        audit_layout.addWidget(QLabel("Data Preview:"))
        audit_layout.addWidget(self.data_preview_table)
        
        # Audit report area
        audit_splitter = QSplitter(Qt.Horizontal)
        
        # Basic statistics
        self.basic_stats_text = QTextEdit()
        self.basic_stats_text.setMaximumWidth(300)
        self.basic_stats_text.setReadOnly(True)
        audit_splitter.addWidget(self.basic_stats_text)
        
        # Feature details table
        self.feature_details_table = QTableWidget()
        audit_splitter.addWidget(self.feature_details_table)
        
        # Quality alerts
        self.quality_alerts_text = QTextEdit()
        self.quality_alerts_text.setMaximumWidth(300)
        self.quality_alerts_text.setReadOnly(True)
        self.quality_alerts_text.setStyleSheet("QTextEdit { background-color: #FFF3CD; }")
        audit_splitter.addWidget(self.quality_alerts_text)
        
        audit_layout.addWidget(audit_splitter)
        
        # Audit control buttons
        audit_controls = QHBoxLayout()
        self.run_audit_btn = QPushButton("Run Data Audit")
        self.export_audit_btn = QPushButton("Export Audit Report")
        audit_controls.addWidget(self.run_audit_btn)
        audit_controls.addWidget(self.export_audit_btn)
        audit_controls.addStretch()
        audit_layout.addLayout(audit_controls)
        
        layout.addWidget(audit_group)
        
        self.tab_widget.addTab(stage1_widget, "Stage 1: Data Input & Audit")
        
    def setup_stage2_feature_engineering(self):
        """Stage 2: Advanced Feature Engineering Workbench"""
        stage2_widget = QWidget()
        layout = QVBoxLayout(stage2_widget)
        
        # Feature selection section
        selection_group = QGroupBox("2.1 Interactive Feature Selection")
        selection_layout = QHBoxLayout(selection_group)
        
        # Available features
        available_features_layout = QVBoxLayout()
        available_features_layout.addWidget(QLabel("Available Features:"))
        self.available_features_list = QListWidget()
        self.available_features_list.setMaximumWidth(250)
        available_features_layout.addWidget(self.available_features_list)
        
        # Selection controls
        controls_layout = QVBoxLayout()
        self.select_feature_btn = QPushButton("Select >")
        self.select_all_features_btn = QPushButton("Select All >>")
        self.deselect_feature_btn = QPushButton("< Deselect")
        self.deselect_all_features_btn = QPushButton("<< Deselect All")
        
        controls_layout.addStretch()
        controls_layout.addWidget(self.select_feature_btn)
        controls_layout.addWidget(self.select_all_features_btn)
        controls_layout.addWidget(self.deselect_feature_btn)
        controls_layout.addWidget(self.deselect_all_features_btn)
        controls_layout.addStretch()
        
        # Selected features
        selected_features_layout = QVBoxLayout()
        selected_features_layout.addWidget(QLabel("Selected Features:"))
        self.selected_features_list = QListWidget()
        self.selected_features_list.setMaximumWidth(250)
        selected_features_layout.addWidget(self.selected_features_list)
        
        selection_layout.addLayout(available_features_layout)
        selection_layout.addLayout(controls_layout)
        selection_layout.addLayout(selected_features_layout)
        
        layout.addWidget(selection_group)
        
        # Data preprocessing section
        preprocessing_group = QGroupBox("2.2 Data Preprocessing Workbench")
        preprocessing_layout = QGridLayout(preprocessing_group)
        
        # Column selection for preprocessing
        preprocessing_layout.addWidget(QLabel("Target Column:"), 0, 0)
        self.preprocessing_column_combo = QComboBox()
        preprocessing_layout.addWidget(self.preprocessing_column_combo, 0, 1)
        
        # Missing value handling
        preprocessing_layout.addWidget(QLabel("Missing Values:"), 1, 0)
        self.missing_value_combo = QComboBox()
        self.missing_value_combo.addItems(['None', 'Drop rows', 'Mean imputation', 'Median imputation', 'Mode imputation', 'KNN imputation'])
        preprocessing_layout.addWidget(self.missing_value_combo, 1, 1)
        
        # Categorical encoding
        preprocessing_layout.addWidget(QLabel("Categorical Encoding:"), 2, 0)
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(['None', 'Label encoding', 'One-hot encoding', 'Target encoding'])
        preprocessing_layout.addWidget(self.encoding_combo, 2, 1)
        
        # Numerical scaling
        preprocessing_layout.addWidget(QLabel("Numerical Scaling:"), 3, 0)
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(['None', 'Standard scaling', 'Min-Max scaling', 'Robust scaling'])
        preprocessing_layout.addWidget(self.scaling_combo, 3, 1)
        
        # Apply preprocessing button
        self.apply_preprocessing_btn = QPushButton("Apply Preprocessing")
        preprocessing_layout.addWidget(self.apply_preprocessing_btn, 4, 0, 1, 2)
        
        # Show processed data button
        self.show_processed_data_btn = QPushButton("Preview Processed Data")
        self.show_processed_data_btn.setStyleSheet("QPushButton { background-color: #17A2B8; color: white; font-weight: bold; }")
        preprocessing_layout.addWidget(self.show_processed_data_btn, 4, 2, 1, 1)
        
        # Preprocessing results
        self.preprocessing_results_text = QTextEdit()
        self.preprocessing_results_text.setMaximumHeight(150)
        self.preprocessing_results_text.setReadOnly(True)
        preprocessing_layout.addWidget(self.preprocessing_results_text, 5, 0, 1, 2)
        
        layout.addWidget(preprocessing_group)
        
        # Dimensionality reduction section
        dimred_group = QGroupBox("2.3 Dimensionality Reduction Algorithms")
        dimred_layout = QGridLayout(dimred_group)
        
        # Algorithm selection
        dimred_layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.dimred_combo = QComboBox()
        algorithms = ['PCA', 't-SNE', 'Isomap']
        if UMAP_AVAILABLE:
            algorithms.append('UMAP')
        self.dimred_combo.addItems(algorithms)
        dimred_layout.addWidget(self.dimred_combo, 0, 1)
        
        # Parameters
        dimred_layout.addWidget(QLabel("Components/Dimensions:"), 1, 0)
        self.dimred_components_spin = QSpinBox()
        self.dimred_components_spin.setRange(2, 10)
        self.dimred_components_spin.setValue(2)
        dimred_layout.addWidget(self.dimred_components_spin, 1, 1)
        
        # Algorithm-specific parameters
        self.dimred_param_label = QLabel("Perplexity:")
        self.dimred_param_spin = QDoubleSpinBox()
        self.dimred_param_spin.setRange(5, 100)
        self.dimred_param_spin.setValue(30)
        dimred_layout.addWidget(self.dimred_param_label, 2, 0)
        dimred_layout.addWidget(self.dimred_param_spin, 2, 1)
        
        # Random seed
        dimred_layout.addWidget(QLabel("Random Seed:"), 3, 0)
        self.dimred_seed_spin = QSpinBox()
        self.dimred_seed_spin.setRange(0, 9999)
        self.dimred_seed_spin.setValue(42)
        dimred_layout.addWidget(self.dimred_seed_spin, 3, 1)
        
        # Apply dimensionality reduction button
        dimred_controls_layout = QHBoxLayout()
        self.apply_dimred_btn = QPushButton("Apply Dimensionality Reduction")
        self.cancel_dimred_btn = QPushButton("Cancel")
        self.cancel_dimred_btn.setEnabled(False)
        self.cancel_dimred_btn.setStyleSheet("QPushButton { background-color: #dc3545; color: white; }")
        
        dimred_controls_layout.addWidget(self.apply_dimred_btn)
        dimred_controls_layout.addWidget(self.cancel_dimred_btn)
        dimred_layout.addLayout(dimred_controls_layout, 4, 0, 1, 2)
        
        # Algorithm descriptions
        self.dimred_description_text = QTextEdit()
        self.dimred_description_text.setMaximumHeight(100)
        self.dimred_description_text.setReadOnly(True)
        dimred_layout.addWidget(self.dimred_description_text, 5, 0, 1, 2)
        
        layout.addWidget(dimred_group)
        
        self.tab_widget.addTab(stage2_widget, "Stage 2: Feature Engineering")
        
    def setup_stage3_clustering_engine(self):
        """Stage 3: Multi-Algorithm Clustering Engine"""
        stage3_widget = QWidget()
        layout = QVBoxLayout(stage3_widget)
        
        # Algorithm selection with cards
        algorithm_group = QGroupBox("3.1 Extended Clustering Algorithm Library")
        algorithm_layout = QGridLayout(algorithm_group)
        
        # Intelligent Algorithm Selection Section
        intelligent_selection_layout = QHBoxLayout()
        self.intelligent_select_btn = QPushButton("Intelligent Algorithm Selection")
        self.intelligent_select_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        self.bayesian_optimize_btn = QPushButton("Bayesian Optimization")
        self.bayesian_optimize_btn.setStyleSheet("""
            QPushButton {
                background-color: #9B59B6; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
        """)
        self.auto_clustering_btn = QPushButton("Auto-Clustering")
        self.auto_clustering_btn.setStyleSheet("""
            QPushButton {
                background-color: #F39C12; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
        """)
        
        intelligent_selection_layout.addWidget(QLabel("Enhanced Features:"))
        intelligent_selection_layout.addWidget(self.intelligent_select_btn)
        intelligent_selection_layout.addWidget(self.bayesian_optimize_btn)  
        intelligent_selection_layout.addWidget(self.auto_clustering_btn)
        intelligent_selection_layout.addStretch()
        
        algorithm_layout.addLayout(intelligent_selection_layout, 0, 0, 1, 2)
        
        # Algorithm cards in a scroll area
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        
        # Create algorithm cards
        self.algorithm_cards = {}
        self.create_algorithm_cards(scroll_layout)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        algorithm_layout.addWidget(scroll_area, 1, 0, 1, 2)
        
        layout.addWidget(algorithm_group)
        
        # Parameter tuning section
        params_group = QGroupBox("3.2 Intelligent Parameter Assistance")
        params_layout = QGridLayout(params_group)
        
        # Selected algorithm display
        params_layout.addWidget(QLabel("Selected Algorithm:"), 0, 0)
        self.selected_algorithm_label = QLabel("None")
        self.selected_algorithm_label.setStyleSheet("QLabel { font-weight: bold; color: #2E86C1; }")
        params_layout.addWidget(self.selected_algorithm_label, 0, 1)
        
        # Parameter controls (dynamic based on algorithm)
        self.param_controls_widget = QWidget()
        self.param_controls_layout = QGridLayout(self.param_controls_widget)
        params_layout.addWidget(self.param_controls_widget, 1, 0, 1, 2)
        
        # Parameter assistance tools
        assistance_layout = QHBoxLayout()
        self.elbow_method_btn = QPushButton("Elbow Method")
        self.silhouette_analysis_btn = QPushButton("Silhouette Analysis")
        self.k_distance_plot_btn = QPushButton("K-Distance Plot")
        self.smart_params_btn = QPushButton("Smart Parameters")
        self.smart_params_btn.setStyleSheet("QPushButton { background-color: #6C5CE7; color: white; font-weight: bold; }")
        self.sync_preview_btn = QPushButton("Use Preview Parameters")
        self.sync_preview_btn.setStyleSheet("QPushButton { background-color: #28A745; color: white; font-weight: bold; }")
        
        assistance_layout.addWidget(self.elbow_method_btn)
        assistance_layout.addWidget(self.silhouette_analysis_btn)
        assistance_layout.addWidget(self.k_distance_plot_btn)
        assistance_layout.addWidget(self.smart_params_btn)
        assistance_layout.addWidget(self.sync_preview_btn)
        assistance_layout.addStretch()
        
        params_layout.addLayout(assistance_layout, 2, 0, 1, 2)
        
        # Run clustering button
        clustering_controls_layout = QHBoxLayout()
        self.run_clustering_btn = QPushButton("Run Clustering Analysis")
        self.run_clustering_btn.setStyleSheet("QPushButton { background-color: #2E86C1; color: white; font-weight: bold; padding: 10px; }")
        
        self.cancel_clustering_btn = QPushButton("Cancel")
        self.cancel_clustering_btn.setEnabled(False)
        self.cancel_clustering_btn.setStyleSheet("QPushButton { background-color: #dc3545; color: white; }")
        
        clustering_controls_layout.addWidget(self.run_clustering_btn)
        clustering_controls_layout.addWidget(self.cancel_clustering_btn)
        params_layout.addLayout(clustering_controls_layout, 3, 0, 1, 2)
        
        layout.addWidget(params_group)
        
        # Real-time preview section
        preview_group = QGroupBox("3.3 Real-time Parameter Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Create real-time preview widget
        self.realtime_preview = RealtimePreviewWidget(self)
        preview_layout.addWidget(self.realtime_preview)
        
        layout.addWidget(preview_group)
        
        self.tab_widget.addTab(stage3_widget, "Stage 3: Clustering Engine")
        
    def setup_stage4_evaluation(self):
        """Stage 4: Comprehensive Evaluation and Validation"""
        stage4_widget = QWidget()
        layout = QVBoxLayout(stage4_widget)
        
        # Enhanced evaluation metrics dashboard
        metrics_group = QGroupBox("4.1 Comprehensive Evaluation Dashboard")
        metrics_layout = QGridLayout(metrics_group)
        
        # Basic metrics display (existing)
        metrics_layout.addWidget(QLabel("Silhouette Score:"), 0, 0)
        self.silhouette_score_label = QLabel("N/A")
        metrics_layout.addWidget(self.silhouette_score_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Davies-Bouldin Index:"), 0, 2)
        self.davies_bouldin_label = QLabel("N/A")
        metrics_layout.addWidget(self.davies_bouldin_label, 0, 3)
        
        metrics_layout.addWidget(QLabel("Calinski-Harabasz Index:"), 1, 0)
        self.calinski_harabasz_label = QLabel("N/A")
        metrics_layout.addWidget(self.calinski_harabasz_label, 1, 1)
        
        # Enhanced evaluation controls
        enhanced_eval_layout = QHBoxLayout()
        self.run_comprehensive_eval_btn = QPushButton("Comprehensive Evaluation")
        self.run_comprehensive_eval_btn.setStyleSheet("""
            QPushButton {
                background-color: #17A2B8; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        
        self.export_eval_report_btn = QPushButton("Export Evaluation Report")
        self.export_eval_report_btn.setStyleSheet("""
            QPushButton {
                background-color: #6C757D; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #5A6268;
            }
        """)
        
        enhanced_eval_layout.addWidget(self.run_comprehensive_eval_btn)
        enhanced_eval_layout.addWidget(self.export_eval_report_btn)
        enhanced_eval_layout.addStretch()
        
        metrics_layout.addLayout(enhanced_eval_layout, 2, 0, 1, 4)
        
        # Enhanced metrics visualization area with tabs
        self.eval_tabs = QTabWidget()
        
        # Tab 1: Basic Metrics Visualization
        basic_viz_tab = QWidget()
        basic_viz_layout = QVBoxLayout(basic_viz_tab)
        self.metrics_canvas_widget = QWidget()
        self.metrics_canvas_layout = QVBoxLayout(self.metrics_canvas_widget)
        basic_viz_layout.addWidget(self.metrics_canvas_widget)
        self.eval_tabs.addTab(basic_viz_tab, "Basic Metrics")
        
        # Tab 2: Comprehensive Metrics
        comprehensive_tab = QWidget()
        comprehensive_layout = QVBoxLayout(comprehensive_tab)
        self.comprehensive_metrics_text = QTextEdit()
        self.comprehensive_metrics_text.setReadOnly(True)
        self.comprehensive_metrics_text.setMaximumHeight(400)
        comprehensive_layout.addWidget(self.comprehensive_metrics_text)
        self.eval_tabs.addTab(comprehensive_tab, "All Metrics")
        
        # Tab 3: Stability Analysis  
        stability_tab = QWidget()
        stability_layout = QVBoxLayout(stability_tab)
        
        # Bootstrap settings
        stability_controls = QGridLayout()
        stability_controls.addWidget(QLabel("Bootstrap Samples:"), 0, 0)
        self.bootstrap_samples_spin = QSpinBox()
        self.bootstrap_samples_spin.setRange(10, 1000)
        self.bootstrap_samples_spin.setValue(100)
        stability_controls.addWidget(self.bootstrap_samples_spin, 0, 1)
        
        stability_controls.addWidget(QLabel("Sample Fraction:"), 0, 2)
        self.sample_fraction_spin = QDoubleSpinBox()
        self.sample_fraction_spin.setRange(0.5, 0.9)
        self.sample_fraction_spin.setSingleStep(0.1)
        self.sample_fraction_spin.setValue(0.8)
        stability_controls.addWidget(self.sample_fraction_spin, 0, 3)
        
        # Run stability test
        self.run_stability_btn = QPushButton("Run Stability Test")
        stability_controls.addWidget(self.run_stability_btn, 1, 0, 1, 4)
        
        stability_layout.addLayout(stability_controls)
        
        # Stability results
        self.stability_results_text = QTextEdit()
        self.stability_results_text.setMaximumHeight(300)
        self.stability_results_text.setReadOnly(True)
        stability_layout.addWidget(self.stability_results_text)
        
        self.eval_tabs.addTab(stability_tab, "Stability")
        
        metrics_layout.addWidget(self.eval_tabs, 3, 0, 1, 4)
        
        layout.addWidget(metrics_group)
        
        self.tab_widget.addTab(stage4_widget, "Stage 4: Evaluation")
        
    def setup_stage5_visualization(self):
        """Stage 5: Multi-Perspective Visualization Insights"""
        stage5_widget = QWidget()
        layout = QVBoxLayout(stage5_widget)
        
        # Visualization controls
        controls_group = QGroupBox("5.1 Visualization Controls")
        controls_layout = QGridLayout(controls_group)
        
        # Plot type selection
        controls_layout.addWidget(QLabel("Visualization Type:"), 0, 0)
        self.visualization_type_combo = QComboBox()
        self.visualization_type_combo.addItems([
            'Scatter Plot (2D/3D)', 'Dendrogram', 'Parallel Coordinates',
            'Radar Chart', 'Feature Heatmap', 'Interactive Dashboard'
        ])
        controls_layout.addWidget(self.visualization_type_combo, 0, 1)
        
        # Export settings
        controls_layout.addWidget(QLabel("Export DPI:"), 1, 0)
        self.export_dpi_spin = QSpinBox()
        self.export_dpi_spin.setRange(72, 600)
        self.export_dpi_spin.setValue(300)
        controls_layout.addWidget(self.export_dpi_spin, 1, 1)
        
        controls_layout.addWidget(QLabel("Figure Size:"), 1, 2)
        size_layout = QHBoxLayout()
        self.export_width_spin = QDoubleSpinBox()
        self.export_width_spin.setRange(4, 20)
        self.export_width_spin.setValue(12)
        self.export_height_spin = QDoubleSpinBox()
        self.export_height_spin.setRange(4, 20)
        self.export_height_spin.setValue(8)
        size_layout.addWidget(self.export_width_spin)
        size_layout.addWidget(QLabel("Height:"))
        size_layout.addWidget(self.export_height_spin)
        controls_layout.addLayout(size_layout, 1, 3)
        
        # Generate visualization button
        self.generate_viz_btn = QPushButton("Generate Visualization")
        controls_layout.addWidget(self.generate_viz_btn, 2, 0, 1, 4)
        
        layout.addWidget(controls_group)
        
        # Visualization display area
        viz_group = QGroupBox("5.2 Visualization Display")
        viz_layout = QVBoxLayout(viz_group)
        
        self.visualization_widget = QWidget()
        self.visualization_layout = QVBoxLayout(self.visualization_widget)
        viz_layout.addWidget(self.visualization_widget)
        
        layout.addWidget(viz_group)
        
        # Automated insights section with enhanced interpretability
        insights_group = QGroupBox("5.3 Intelligent Cluster Analysis & Interpretability")
        insights_layout = QVBoxLayout(insights_group)
        
        # Enhanced cluster analysis controls
        analysis_controls_layout = QHBoxLayout()
        
        self.generate_insights_btn = QPushButton("Generate Cluster Profiles")
        self.generate_insights_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        self.generate_explanations_btn = QPushButton("Generate Explanations")
        self.generate_explanations_btn.setStyleSheet("""
            QPushButton {
                background-color: #6F42C1; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #5A32A3;
            }
        """)
        
        self.export_interpretability_btn = QPushButton("Export Analysis")
        self.export_interpretability_btn.setStyleSheet("""
            QPushButton {
                background-color: #FD7E14; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #E8590C;
            }
        """)
        
        analysis_controls_layout.addWidget(self.generate_insights_btn)
        analysis_controls_layout.addWidget(self.generate_explanations_btn)
        analysis_controls_layout.addWidget(self.export_interpretability_btn)
        analysis_controls_layout.addStretch()
        
        insights_layout.addLayout(analysis_controls_layout)
        
        # Enhanced cluster insights with tabs
        self.insights_tabs = QTabWidget()
        
        # Tab 1: Cluster Profiles
        profiles_tab = QWidget()
        profiles_layout = QVBoxLayout(profiles_tab)
        self.cluster_insights_text = QTextEdit()
        self.cluster_insights_text.setReadOnly(True)
        profiles_layout.addWidget(self.cluster_insights_text)
        self.insights_tabs.addTab(profiles_tab, "Cluster Profiles")
        
        # Tab 2: Natural Language Explanations
        explanations_tab = QWidget()
        explanations_layout = QVBoxLayout(explanations_tab)
        self.cluster_explanations_text = QTextEdit()
        self.cluster_explanations_text.setReadOnly(True)
        explanations_layout.addWidget(self.cluster_explanations_text)
        self.insights_tabs.addTab(explanations_tab, "Explanations")
        
        # Tab 3: Decision Rules
        rules_tab = QWidget()
        rules_layout = QVBoxLayout(rules_tab)
        self.cluster_rules_text = QTextEdit()
        self.cluster_rules_text.setReadOnly(True)
        rules_layout.addWidget(self.cluster_rules_text)
        self.insights_tabs.addTab(rules_tab, "Decision Rules")
        
        insights_layout.addWidget(self.insights_tabs)
        
        layout.addWidget(insights_group)
        
        self.tab_widget.addTab(stage5_widget, "Stage 5: Visualization")
        
    def setup_stage6_export(self):
        """Stage 6: Report Generation and Export"""
        stage6_widget = QWidget()
        layout = QVBoxLayout(stage6_widget)
        
        # Report generation section
        report_group = QGroupBox("6.1 Automated Analysis Report")
        report_layout = QGridLayout(report_group)
        
        # Report settings
        report_layout.addWidget(QLabel("Report Format:"), 0, 0)
        self.report_format_combo = QComboBox()
        self.report_format_combo.addItems(['PDF', 'HTML', 'Word Document'])
        report_layout.addWidget(self.report_format_combo, 0, 1)
        
        report_layout.addWidget(QLabel("Include Sections:"), 1, 0)
        
        # Checkboxes for report sections
        sections_widget = QWidget()
        sections_layout = QGridLayout(sections_widget)
        
        self.include_data_audit_cb = QCheckBox("Data Audit")
        self.include_preprocessing_cb = QCheckBox("Preprocessing Steps")
        self.include_clustering_cb = QCheckBox("Clustering Results")
        self.include_evaluation_cb = QCheckBox("Evaluation Metrics")
        self.include_visualizations_cb = QCheckBox("Visualizations")
        self.include_insights_cb = QCheckBox("Cluster Insights")
        
        sections_layout.addWidget(self.include_data_audit_cb, 0, 0)
        sections_layout.addWidget(self.include_preprocessing_cb, 0, 1)
        sections_layout.addWidget(self.include_clustering_cb, 1, 0)
        sections_layout.addWidget(self.include_evaluation_cb, 1, 1)
        sections_layout.addWidget(self.include_visualizations_cb, 2, 0)
        sections_layout.addWidget(self.include_insights_cb, 2, 1)
        
        # Set all checkboxes checked by default
        for cb in [self.include_data_audit_cb, self.include_preprocessing_cb, 
                self.include_clustering_cb, self.include_evaluation_cb,
                self.include_visualizations_cb, self.include_insights_cb]:
            cb.setChecked(True)
        
        report_layout.addWidget(sections_widget, 1, 1)
        
        # Generate report button
        self.generate_report_btn = QPushButton("Generate Complete Analysis Report")
        self.generate_report_btn.setStyleSheet("QPushButton { background-color: #28A745; color: white; font-weight: bold; padding: 10px; }")
        report_layout.addWidget(self.generate_report_btn, 2, 0, 1, 2)
        
        layout.addWidget(report_group)
        
        # Export options section
        export_group = QGroupBox("6.2 Data and Model Export")
        export_layout = QGridLayout(export_group)
        
        # Export buttons
        self.export_labeled_data_btn = QPushButton("Export Labeled Data (CSV)")
        self.export_cluster_profiles_btn = QPushButton("Export Cluster Profiles (Excel)")
        self.export_model_btn = QPushButton("Export Model Objects (.pkl)")
        
        export_layout.addWidget(self.export_labeled_data_btn, 0, 0)
        export_layout.addWidget(self.export_cluster_profiles_btn, 0, 1)
        export_layout.addWidget(self.export_model_btn, 1, 0, 1, 2)
        
        layout.addWidget(export_group)
        
        # Export status
        self.export_status_text = QTextEdit()
        self.export_status_text.setMaximumHeight(100)
        self.export_status_text.setReadOnly(True)
        layout.addWidget(self.export_status_text)
        
        self.tab_widget.addTab(stage6_widget, "Stage 6: Report & Export")
        
    def create_algorithm_cards(self, layout):
        """Create algorithm selection cards"""
        algorithms = [
            {
                'name': 'K-Means',
                'category': 'Partitioning-based',
                'description': 'Classic algorithm, fast, suitable for spherical clusters',
                'pros': 'Fast, simple, works well with globular clusters',
                'cons': 'Requires k specification, sensitive to initialization',
                'use_cases': 'Customer segmentation, image segmentation'
            },
            {
                'name': 'K-Medoids',
                'category': 'Partitioning-based', 
                'description': 'More robust K-Means variant using medoids',
                'pros': 'Robust to outliers, works with any distance metric',
                'cons': 'Slower than K-Means, still requires k specification',
                'use_cases': 'Noisy datasets, non-Euclidean distances'
            },
            {
                'name': 'Agglomerative',
                'category': 'Hierarchical-based',
                'description': 'Bottom-up hierarchical clustering',
                'pros': 'No need to specify k, creates hierarchy',
                'cons': 'Computationally expensive, sensitive to noise',
                'use_cases': 'Taxonomy creation, phylogenetic analysis'
            },
            {
                'name': 'DBSCAN',
                'category': 'Density-based',
                'description': 'Finds arbitrary-shaped clusters and identifies noise',
                'pros': 'Finds arbitrary shapes, identifies outliers',
                'cons': 'Sensitive to parameters, struggles with varying densities',
                'use_cases': 'Anomaly detection, spatial clustering'
            },
            {
                'name': 'OPTICS',
                'category': 'Density-based',
                'description': 'Improved DBSCAN for varying density clusters',
                'pros': 'Handles varying densities, no eps parameter',
                'cons': 'More complex, computationally expensive',
                'use_cases': 'Complex spatial data, varying cluster densities'
            },
            {
                'name': 'Spectral Clustering',
                'category': 'Graph-based',
                'description': 'Uses eigenvalues of similarity matrix for clustering',
                'pros': 'Handles non-convex shapes, works with similarity matrices',
                'cons': 'Computationally expensive, requires k specification',
                'use_cases': 'Image segmentation, social networks, non-convex clusters'
            },
            {
                'name': 'Mean Shift',
                'category': 'Density-based',
                'description': 'Finds dense regions by mode-seeking algorithm',
                'pros': 'No need to specify k, finds arbitrary shapes',
                'cons': 'Computationally expensive, sensitive to bandwidth',
                'use_cases': 'Computer vision, image processing, peak detection'
            },
            {
                'name': 'Gaussian Mixture',
                'category': 'Model-based',
                'description': 'Assumes data generated from mixture of Gaussians',
                'pros': 'Provides probability membership, handles overlapping clusters',
                'cons': 'Assumes Gaussian distributions, requires k specification',
                'use_cases': 'Soft clustering, overlapping clusters'
            },
            # Enhanced algorithms
            {
                'name': 'Fuzzy C-Means',
                'category': 'Soft Clustering',
                'description': 'Soft clustering with membership probabilities',
                'pros': 'Provides membership probabilities, handles overlapping clusters',
                'cons': 'Sensitive to initialization, requires fuzziness parameter',
                'use_cases': 'Overlapping data patterns, soft cluster boundaries'
            },
            {
                'name': 'Consensus K-Means',
                'category': 'Ensemble-based',
                'description': 'Robust ensemble clustering using multiple K-means runs',
                'pros': 'More robust than single K-means, reduces initialization sensitivity',
                'cons': 'Computationally expensive, still requires k specification',
                'use_cases': 'When robustness is critical, unstable clustering scenarios'
            },
            {
                'name': 'Mini-Batch K-Means+',
                'category': 'Scalable',
                'description': 'Enhanced mini-batch K-means for large datasets',
                'pros': 'Fast on large datasets, adaptive batch sizing, memory efficient',
                'cons': 'Approximation algorithm, may not converge to global optimum',
                'use_cases': 'Large datasets, streaming data, memory-constrained environments'
            },
            {
                'name': 'Enhanced Affinity Propagation',
                'category': 'Message Passing',
                'description': 'Improved affinity propagation with adaptive damping',
                'pros': 'No need to specify number of clusters, finds exemplars automatically',
                'cons': 'Can be slow, sensitive to preference parameter',
                'use_cases': 'Unknown number of clusters, finding representative samples'
            }
        ]
        
        if HDBSCAN_AVAILABLE:
            algorithms.append({
                'name': 'HDBSCAN',
                'category': 'Density-based',
                'description': 'Hierarchical density-based clustering',
                'pros': 'Handles varying densities, finds hierarchy',
                'cons': 'Complex parameter tuning',
                'use_cases': 'Complex cluster structures, outlier detection'
            })
            
        if BIRCH is not None:
            algorithms.append({
                'name': 'BIRCH',
                'category': 'Hierarchical-based',
                'description': 'Memory-efficient clustering for large datasets',
                'pros': 'Memory efficient, handles large datasets, incremental',
                'cons': 'Assumes spherical clusters, sensitive to branching factor',
                'use_cases': 'Large datasets, streaming data, memory constraints'
            })
        
        row, col = 0, 0
        for algo in algorithms:
            card = self.create_algorithm_card(algo)
            layout.addWidget(card, row, col)
            
            col += 1
            if col >= 3:  # 3 cards per row
                col = 0
                row += 1
                
    def create_algorithm_card(self, algo_info):
        """Create a single algorithm card"""
        card = QFrame()
        card.setFrameStyle(QFrame.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                border: 2px solid #E1E8ED;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
                background-color: #F8F9FA;
            }
            QFrame:hover {
                border-color: #2E86C1;
                background-color: #EBF3FD;
            }
        """)
        card.setMaximumWidth(300)
        card.setMinimumHeight(200)
        
        layout = QVBoxLayout(card)
        
        # Algorithm name and category
        name_label = QLabel(algo_info['name'])
        name_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        name_label.setStyleSheet("QLabel { color: #2E86C1; }")
        layout.addWidget(name_label)
        
        category_label = QLabel(f"({algo_info['category']})")
        category_label.setFont(QFont("Segoe UI", 9))
        category_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        layout.addWidget(category_label)
        
        # Description
        desc_label = QLabel(algo_info['description'])
        desc_label.setWordWrap(True)
        desc_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(desc_label)
        
        # Pros and cons (abbreviated)
        pros_label = QLabel(f"+ {algo_info['pros'][:50]}...")
        pros_label.setWordWrap(True)
        pros_label.setStyleSheet("QLabel { color: #28A745; font-size: 8pt; }")
        layout.addWidget(pros_label)

        cons_label = QLabel(f"- {algo_info['cons'][:50]}...")
        cons_label.setWordWrap(True)
        cons_label.setStyleSheet("QLabel { color: #DC3545; font-size: 8pt; }")
        layout.addWidget(cons_label)
        
        # Select button
        select_btn = QPushButton(f"Select {algo_info['name']}")
        select_btn.clicked.connect(lambda checked, name=algo_info['name']: self.select_algorithm(name))
        layout.addWidget(select_btn)
        
        self.algorithm_cards[algo_info['name']] = card
        
        return card
        
    def init_enhanced_features(self):
        """Initialize enhanced clustering features"""
        try:
            # Initialize intelligent algorithm selector
            self.intelligent_algorithm_selector = IntelligentAlgorithmSelector()
            
            # Initialize comprehensive evaluator
            self.comprehensive_evaluator = ComprehensiveClusteringEvaluator()
            
            # Initialize clustering interpreter
            self.clustering_interpreter = ComprehensiveClusteringInterpreter()
            
            # Initialize parameter optimizer
            self.parameter_optimizer = AdvancedParameterOptimizer(random_state=self.random_seed)
            
            # Initialize performance optimizer
            self.performance_optimizer = PerformanceOptimizedClusteringModule()
            
            # Initialize algorithm registry for parameter optimization
            self._init_algorithm_registry()
            
            self.status_updated.emit("Enhanced clustering features initialized")
            
        except Exception as e:
            print(f"Warning: Could not initialize enhanced features: {e}")
            self.status_updated.emit(f"Enhanced features initialization failed: {e}")
    
    def _init_algorithm_registry(self):
        """Initialize algorithm registry for parameter optimization"""
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
        from sklearn.mixture import GaussianMixture
        
        self.algorithm_registry = {
            'K-Means': KMeans,
            'DBSCAN': DBSCAN,
            'Agglomerative': AgglomerativeClustering,
            'Spectral': SpectralClustering,
            'Spectral Clustering': SpectralClustering,
            'Gaussian Mixture': GaussianMixture
        }
            
    def apply_optimal_parameters(self, algorithm_name, optimal_params):
        """Apply optimal parameters to the UI controls"""
        # Handle different possible return formats
        if 'best_overall_params' in optimal_params:
            params = optimal_params['best_overall_params']
        elif 'optimal_params' in optimal_params:
            params = optimal_params['optimal_params']
        else:
            params = optimal_params  # Assume it's already the params dict
        
        if not params:
            print(f"Warning: No parameters found for {algorithm_name}")
            return
        
        try:
            if algorithm_name == 'K-Means' and hasattr(self, 'k_clusters_spin'):
                if 'n_clusters' in params:
                    self.k_clusters_spin.setValue(int(params['n_clusters']))
                if 'max_iter' in params and hasattr(self, 'max_iter_spin'):
                    self.max_iter_spin.setValue(int(params['max_iter']))
                    
            elif algorithm_name == 'DBSCAN':
                if 'eps' in params and hasattr(self, 'eps_spin'):
                    self.eps_spin.setValue(float(params['eps']))
                if 'min_samples' in params and hasattr(self, 'min_samples_spin'):
                    self.min_samples_spin.setValue(int(params['min_samples']))
                    
            elif algorithm_name == 'Gaussian Mixture' and hasattr(self, 'n_components_spin'):
                if 'n_components' in params:
                    self.n_components_spin.setValue(int(params['n_components']))
                    
            elif algorithm_name == 'Spectral Clustering':
                if 'n_clusters' in params and hasattr(self, 'spectral_n_clusters_spin'):
                    self.spectral_n_clusters_spin.setValue(int(params['n_clusters']))
                if 'gamma' in params and hasattr(self, 'gamma_spin'):
                    self.gamma_spin.setValue(float(params['gamma']))
                    
        except Exception as e:
            print(f"Warning: Could not apply parameter {e}")
            
    def format_parameters(self, params_dict):
        """Format parameters dictionary for display"""
        formatted = []
        for key, value in params_dict.items():
            if isinstance(value, float):
                formatted.append(f"  {key}: {value:.4f}")
            else:
                formatted.append(f"  {key}: {value}")
        return "\n".join(formatted)
            

# Test the module functionality
def main():
    """Test function for the clustering module"""
    app = QApplication(sys.argv)
    
    # Create the clustering module
    clustering_module = IntelligentClusteringModule()
    clustering_module.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()






