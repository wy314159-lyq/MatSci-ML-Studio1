"""
SHAP Model Interpretability Analysis Module
Provides comprehensive model explanation using SHAP (SHapley Additive exPlanations)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QSpinBox, QTextEdit, QTabWidget,
                             QFileDialog, QMessageBox, QProgressBar, QGroupBox,
                             QGridLayout, QCheckBox, QSlider, QFrame, QSplitter,
                             QApplication, QDialog, QDialogButtonBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon
import shap
import joblib
import traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# 尝试导入额外的树模型
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# 导入独立可视化窗口类
from modules.shap_visualization_windows import (
    SHAPVisualizationWindow, 
    SHAPDependenceWindow, 
    SHAPLocalWindow
)

# 设置matplotlib后端
plt.switch_backend('Qt5Agg')

class SHAPCalculationWorker(QThread):
    """SHAP calculation worker thread"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    calculation_completed = pyqtSignal(dict)  # Return complete result dictionary
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model, background_data, explain_data, explainer_type, config, feature_names=None):
        super().__init__()
        self.model = model
        self.background_data = background_data
        self.explain_data = explain_data
        self.explainer_type = explainer_type
        self.config = config
        self.feature_names = feature_names
        
    def run(self):
        """Execute SHAP calculation using modern shap.Explainer interface"""
        try:
            self.status_updated.emit("🔍 Initializing SHAP explainer...")
            self.progress_updated.emit(10)
            
            # Use modern shap.Explainer interface for better Pipeline compatibility
            explainer = self._initialize_modern_explainer()
            self.progress_updated.emit(40)
            
            # Calculate SHAP values
            self.status_updated.emit("📊 Calculating SHAP values...")
            shap_values = self._calculate_modern_shap_values(explainer)
            self.progress_updated.emit(90)
            
            self.status_updated.emit("✅ SHAP calculation completed")
            self.progress_updated.emit(100)
            
            # Return complete result dictionary
            result = {
                'shap_values': shap_values,
                'explainer': explainer,
                'explain_data': self.explain_data,
                'feature_names': self.feature_names
            }
            
            self.calculation_completed.emit(result)
            
        except Exception as e:
            # Fallback to legacy method if modern approach fails
            try:
                self.status_updated.emit("🔄 Trying fallback method...")
                self.progress_updated.emit(20)
                
                explainer = self._initialize_explainer()
                self.progress_updated.emit(50)
                
                shap_values = self._calculate_shap_values(explainer)
                self.progress_updated.emit(90)
                
                self.status_updated.emit("✅ SHAP calculation completed (fallback)")
                self.progress_updated.emit(100)
                
                # Return complete result dictionary
                result = {
                    'shap_values': shap_values,
                    'explainer': explainer,
                    'explain_data': self.explain_data,
                    'feature_names': self.feature_names
                }
                
                self.calculation_completed.emit(result)
                
            except Exception as fallback_error:
                self.error_occurred.emit(f"SHAP calculation failed: {str(e)}. Fallback also failed: {str(fallback_error)}")
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer with enhanced error handling"""
        try:
            if self.explainer_type == "TreeExplainer":
                # For Pipeline models, avoid TreeExplainer due to feature_names_in_ issues
                if isinstance(self.model, Pipeline):
                    print("Pipeline detected, using KernelExplainer instead of TreeExplainer to avoid compatibility issues")
                    # Use KernelExplainer for Pipeline models
                    predict_fn = self.model.predict
                    background_array = self.background_data.iloc[:50].values
                    return shap.KernelExplainer(predict_fn, background_array)
                else:
                    # Extract the model from pipeline if needed
                    model = self._extract_model_from_pipeline(self.model)
                    return shap.TreeExplainer(model)
                
            elif self.explainer_type == "LinearExplainer":
                model = self._extract_model_from_pipeline(self.model)
                
                # Convert background data to numpy array to avoid feature name issues
                if hasattr(self.background_data, 'values'):
                    background_array = self.background_data.values
                else:
                    background_array = self.background_data
                    
                return shap.LinearExplainer(model, background_array)
                
            elif self.explainer_type == "KernelExplainer":
                # Use model.predict for kernel explainer
                predict_fn = self.model.predict if hasattr(self.model, 'predict') else self.model
                n_background = self.config.get('n_background', 100)
                
                # Sample background data safely
                if len(self.background_data) > n_background:
                    background_sample = self.background_data.sample(n_background, random_state=42)
                else:
                    background_sample = self.background_data.copy()
                
                # Convert to numpy array to avoid attribute issues
                if hasattr(background_sample, 'values'):
                    background_array = background_sample.values
                else:
                    background_array = background_sample
                    
                return shap.KernelExplainer(predict_fn, background_array)
                
            else:
                raise ValueError(f"Unsupported explainer type: {self.explainer_type}")
                
        except Exception as e:
            # Enhanced error handling - try simpler approach
            print(f"Primary explainer initialization failed: {e}")
            
            # Fallback to KernelExplainer with minimal configuration
            try:
                # Create safe wrapper for Pipeline models
                def safe_predict_wrapper(X):
                    try:
                        if hasattr(X, 'shape') and len(X.shape) == 1:
                            X = X.reshape(1, -1)
                        
                        # Convert numpy array to DataFrame with proper column names for Pipeline compatibility
                        if isinstance(X, np.ndarray) and hasattr(self, 'feature_names') and self.feature_names:
                            X_df = pd.DataFrame(X, columns=self.feature_names)
                        else:
                            X_df = X
                        
                        return self.model.predict(X_df)
                    except:
                        return np.zeros(X.shape[0] if hasattr(X, 'shape') else 1)
                
                # Use small background sample as numpy array
                background_sample = self.background_data.iloc[:10].values
                return shap.KernelExplainer(safe_predict_wrapper, background_sample)
                
            except Exception as fallback_error:
                raise Exception(f"All explainer initialization methods failed. Original: {e}, Fallback: {fallback_error}")
    
    def _initialize_modern_explainer(self):
        """Initialize SHAP explainer using modern shap.Explainer interface"""
        try:
            print("Attempting to use modern shap.Explainer interface...")
            
            # Prepare background data - use a reasonable sample size
            background_sample_size = min(100, len(self.background_data))
            if len(self.background_data) > background_sample_size:
                background_sample = self.background_data.sample(background_sample_size, random_state=42)
            else:
                background_sample = self.background_data.copy()
            
            print(f"Using background sample of size: {len(background_sample)}")
            print(f"Background data columns: {list(background_sample.columns)}")
            print(f"Model type: {type(self.model)}")
            
            # For Pipeline models, avoid the modern interface due to feature_names_in_ issues
            if isinstance(self.model, Pipeline):
                print("Detected Pipeline model, using safe approach to avoid feature_names_in_ errors...")
                
                # Extract the final estimator for explainer type detection
                final_estimator = self.model.named_steps[self.model.steps[-1][0]]
                estimator_name = type(final_estimator).__name__
                print(f"Final estimator: {estimator_name}")
                
                # Tree-based models work well with TreeExplainer even in Pipeline
                tree_models = ['RandomForestRegressor', 'RandomForestClassifier', 
                              'DecisionTreeRegressor', 'DecisionTreeClassifier',
                              'XGBRegressor', 'XGBClassifier', 'LGBMRegressor', 'LGBMClassifier']
                
                # For ALL Pipeline models, use a very conservative approach
                print("Using ultra-conservative KernelExplainer approach for Pipeline...")
                
                # Determine task type and create appropriate prediction wrapper
                # Check if this is a classification task by examining model methods
                is_classification = (hasattr(self.model, 'predict_proba') and 
                                   hasattr(self.model, 'classes_'))
                
                if is_classification:
                    print("Detected classification task, using predict_proba for SHAP")
                    def safe_predict_wrapper(X):
                        """Safe prediction wrapper for classification using predict_proba"""
                        try:
                            if hasattr(X, 'shape') and len(X.shape) == 1:
                                X = X.reshape(1, -1)
                            
                            # Convert numpy array to DataFrame with proper column names for Pipeline compatibility
                            if isinstance(X, np.ndarray) and hasattr(self, 'feature_names') and self.feature_names:
                                X_df = pd.DataFrame(X, columns=self.feature_names)
                            else:
                                X_df = X
                            
                            # For classification, use predict_proba to get class probabilities
                            proba = self.model.predict_proba(X_df)
                            # Return probabilities for positive class (binary) or all classes (multiclass)
                            return proba[:, 1] if proba.shape[1] == 2 else proba
                        except Exception as e:
                            print(f"Classification prediction wrapper error: {e}")
                            # Return dummy probabilities
                            if hasattr(X, 'shape'):
                                return np.full(X.shape[0], 0.5)
                            else:
                                return np.array([0.5])
                else:
                    print("Detected regression task, using predict for SHAP")
                    def safe_predict_wrapper(X):
                        """Safe prediction wrapper for regression using predict"""
                        try:
                            if hasattr(X, 'shape') and len(X.shape) == 1:
                                X = X.reshape(1, -1)
                            
                            # Convert numpy array to DataFrame with proper column names for Pipeline compatibility
                            if isinstance(X, np.ndarray) and hasattr(self, 'feature_names') and self.feature_names:
                                X_df = pd.DataFrame(X, columns=self.feature_names)
                            else:
                                X_df = X
                            
                            return self.model.predict(X_df)
                        except Exception as e:
                            print(f"Regression prediction wrapper error: {e}")
                            # Return dummy predictions
                            if hasattr(X, 'shape'):
                                return np.zeros(X.shape[0])
                            else:
                                return np.array([0])
                
                # Use minimal background data to reduce complexity
                minimal_background = background_sample.values[:10]  # Use only 10 samples
                
                try:
                    explainer = shap.KernelExplainer(safe_predict_wrapper, minimal_background)
                    print(f"Ultra-conservative KernelExplainer created successfully for {'classification' if is_classification else 'regression'}")
                    return explainer
                except Exception as kernel_error:
                    print(f"Even ultra-conservative approach failed: {kernel_error}")
                    raise kernel_error
            else:
                # For non-Pipeline models, try the modern interface but with error handling
                try:
                    explainer = shap.Explainer(self.model, background_sample)
                    print(f"Successfully created modern SHAP explainer: {type(explainer)}")
                    return explainer
                except Exception as modern_error:
                    print(f"Modern interface failed: {modern_error}")
                    # Fallback to KernelExplainer for non-Pipeline models too
                    print("Falling back to KernelExplainer for non-Pipeline model")
                    explainer = shap.KernelExplainer(self.model.predict, background_sample)
                    return explainer
            
        except Exception as e:
            print(f"Modern explainer initialization failed: {e}")
            raise e
    
    def _calculate_modern_shap_values(self, explainer):
        """Calculate SHAP values using modern explainer interface"""
        try:
            print("Calculating SHAP values with modern interface...")
            
            # Prepare explanation data - limit size for performance
            explain_sample_size = min(50, len(self.explain_data))
            if len(self.explain_data) > explain_sample_size:
                explain_sample = self.explain_data.iloc[:explain_sample_size].copy()
            else:
                explain_sample = self.explain_data.copy()
            
            print(f"Explaining {len(explain_sample)} samples")
            print(f"Explain data columns: {list(explain_sample.columns)}")
            print(f"Explainer type: {type(explainer)}")
            
            # Different calculation methods based on explainer type
            if isinstance(explainer, shap.TreeExplainer):
                print("Using TreeExplainer calculation...")
                # For TreeExplainer, use the data directly (should be rare now)
                shap_values = explainer.shap_values(explain_sample.values)
                
            elif isinstance(explainer, shap.KernelExplainer):
                print("Using KernelExplainer calculation...")
                # KernelExplainer works with the original data and handles Pipeline internally
                shap_values = explainer.shap_values(explain_sample.values)
                
            else:
                print("Using modern Explainer interface...")
                # Modern interface - try the callable approach with error handling
                try:
                    shap_explanation = explainer(explain_sample)
                    
                    # Extract values from the explanation object
                    if hasattr(shap_explanation, 'values'):
                        shap_values = shap_explanation.values
                    else:
                        raise Exception("SHAP explanation object does not contain 'values' attribute")
                except Exception as modern_calc_error:
                    print(f"Modern calculation failed: {modern_calc_error}")
                    # Try to convert to KernelExplainer as final fallback
                    print("Converting to KernelExplainer as final fallback...")
                    
                    # Create safe prediction wrapper for fallback
                    def fallback_predict_wrapper(X):
                        try:
                            if isinstance(X, np.ndarray) and hasattr(self, 'feature_names') and self.feature_names:
                                X_df = pd.DataFrame(X, columns=self.feature_names)
                            else:
                                X_df = X
                            return self.model.predict(X_df)
                        except:
                            return np.zeros(X.shape[0] if hasattr(X, 'shape') else 1)
                    
                    fallback_explainer = shap.KernelExplainer(fallback_predict_wrapper, self.background_data.sample(min(50, len(self.background_data))).values)
                    shap_values = fallback_explainer.shap_values(explain_sample.values)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class classification returns list of arrays
                if len(shap_values) == 2:
                    # Binary classification - use positive class
                    shap_values = shap_values[1]
                    print("Using positive class SHAP values for binary classification")
                else:
                    # Multi-class - use first class
                    shap_values = shap_values[0]
                    print("Using first class SHAP values for multi-class classification")
            
            # Ensure 2D array format
            if len(shap_values.shape) == 3:
                if shap_values.shape[2] == 2:
                    shap_values = shap_values[:, :, 1]
                    print("Converted 3D SHAP values to 2D (positive class)")
                else:
                    shap_values = shap_values[:, :, 0]
                    print("Converted 3D SHAP values to 2D (first class)")
            
            print(f"Final SHAP values shape: {shap_values.shape}")
            return shap_values
                
        except Exception as e:
            print(f"Modern SHAP calculation failed: {e}")
            raise e
    
    def _extract_model_from_pipeline(self, pipeline_or_model):
        """Extract the final estimator from pipeline"""
        if isinstance(pipeline_or_model, Pipeline):
            return pipeline_or_model.named_steps[pipeline_or_model.steps[-1][0]]
        return pipeline_or_model
    
    def _calculate_shap_values(self, explainer):
        """Calculate SHAP values with enhanced Pipeline handling"""
        try:
            if isinstance(explainer, shap.TreeExplainer):
                # For tree explainer, we need to transform data through pipeline preprocessing
                if isinstance(self.model, Pipeline):
                    # Get preprocessing steps (all except the last estimator)
                    if len(self.model.steps) > 1:
                        preprocessor = Pipeline(self.model.steps[:-1])
                        # Transform explain data through preprocessor
                        explain_data_transformed = preprocessor.transform(self.explain_data)
                        
                        # Convert to numpy array if it's still a DataFrame
                        if hasattr(explain_data_transformed, 'values'):
                            explain_data_transformed = explain_data_transformed.values
                    else:
                        # No preprocessing steps, use data as numpy array
                        explain_data_transformed = self.explain_data.values if hasattr(self.explain_data, 'values') else self.explain_data
                else:
                    # Direct model, convert to numpy array
                    explain_data_transformed = self.explain_data.values if hasattr(self.explain_data, 'values') else self.explain_data
                
                return explainer.shap_values(explain_data_transformed)
                
            else:
                # For LinearExplainer and KernelExplainer, use numpy arrays
                if hasattr(self.explain_data, 'values'):
                    explain_data_array = self.explain_data.values
                else:
                    explain_data_array = self.explain_data
                    
                return explainer.shap_values(explain_data_array)
                
        except Exception as e:
            # Multiple fallback strategies
            print(f"Primary SHAP calculation failed: {e}")
            
            # Fallback 1: Try with simple numpy arrays
            try:
                explain_data_values = self.explain_data.values if hasattr(self.explain_data, 'values') else self.explain_data
                return explainer.shap_values(explain_data_values)
                
            except Exception as e2:
                print(f"Fallback 1 failed: {e2}")
                
                # Fallback 2: Try with a smaller sample
                try:
                    if hasattr(self.explain_data, 'iloc'):
                        small_sample = self.explain_data.iloc[:10].values
                    else:
                        small_sample = self.explain_data[:10]
                    return explainer.shap_values(small_sample)
                    
                except Exception as e3:
                    raise Exception(f"All SHAP calculation methods failed. Primary: {e}, Fallback1: {e2}, Fallback2: {e3}")


class SHAPAnalysisModule(QWidget):
    """SHAP Model Interpretability Analysis Module"""
    
    # Signals
    analysis_completed = pyqtSignal()
    status_updated = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Set matplotlib backend for Qt compatibility
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode
        
        # Initialize attributes
        self.model = None
        self.background_data = None
        self.explain_data = None
        self.feature_names = None
        self.shap_values = None
        self.explainer = None
        self.training_data = None
        self.config = None
        self.base_value = 0
        
        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🧠 SHAP Model Interpretability Analysis")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { color: #2196F3; margin: 10px; }")
        layout.addWidget(title)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel - Configuration
        left_panel = self.create_configuration_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = self.create_results_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setStretchFactor(0, 1)  # Configuration panel
        main_splitter.setStretchFactor(1, 2)  # Results panel
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: #666666; margin: 5px; }")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
    def create_configuration_panel(self):
        """Create configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model and Data Loading
        model_group = QGroupBox("📁 Model & Data Loading")
        model_layout = QVBoxLayout(model_group)
        
        self.load_model_btn = QPushButton("📊 Load Trained Model")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        
        self.model_info_text = QTextEdit()
        self.model_info_text.setMaximumHeight(80)
        self.model_info_text.setReadOnly(True)
        model_layout.addWidget(self.model_info_text)
        
        self.load_background_btn = QPushButton("📋 Load Background Data")
        self.load_background_btn.clicked.connect(self.load_background_data)
        self.load_background_btn.setEnabled(False)
        model_layout.addWidget(self.load_background_btn)
        
        self.load_explain_btn = QPushButton("🎯 Load Data to Explain")
        self.load_explain_btn.clicked.connect(self.load_explain_data)
        self.load_explain_btn.setEnabled(False)
        model_layout.addWidget(self.load_explain_btn)
        
        # Add sample info display
        self.sample_info_text = QTextEdit()
        self.sample_info_text.setMaximumHeight(100)
        self.sample_info_text.setReadOnly(True)
        self.sample_info_text.setPlaceholderText("Training data information will appear here...")
        model_layout.addWidget(self.sample_info_text)
        
        layout.addWidget(model_group)
        
        # SHAP Configuration
        config_group = QGroupBox("⚙️ SHAP Configuration")
        config_layout = QVBoxLayout(config_group)
        
        explainer_layout = QHBoxLayout()
        explainer_layout.addWidget(QLabel("Explainer Type:"))
        self.explainer_combo = QComboBox()
        self.explainer_combo.addItems(["Auto-Select", "TreeExplainer", "LinearExplainer", "KernelExplainer"])
        explainer_layout.addWidget(self.explainer_combo)
        config_layout.addLayout(explainer_layout)
        
        background_layout = QHBoxLayout()
        background_layout.addWidget(QLabel("Background Samples:"))
        self.background_samples_spin = QSpinBox()
        self.background_samples_spin.setRange(10, 1000)
        self.background_samples_spin.setValue(100)
        background_layout.addWidget(self.background_samples_spin)
        config_layout.addLayout(background_layout)
        
        layout.addWidget(config_group)
        
        # Calculation Controls
        calc_group = QGroupBox("🚀 Calculation")
        calc_layout = QVBoxLayout(calc_group)
        
        self.calculate_btn = QPushButton("🔍 Calculate SHAP Values")
        self.calculate_btn.clicked.connect(self.calculate_shap_values)
        self.calculate_btn.setEnabled(False)
        calc_layout.addWidget(self.calculate_btn)
        
        layout.addWidget(calc_group)
        
        # Export Controls
        export_group = QGroupBox("💾 Export")
        export_layout = QVBoxLayout(export_group)
        
        self.export_values_btn = QPushButton("📊 Export SHAP Values")
        self.export_values_btn.clicked.connect(self.export_shap_values)
        self.export_values_btn.setEnabled(False)
        export_layout.addWidget(self.export_values_btn)
        
        self.export_plots_btn = QPushButton("🖼️ Export Current Plot")
        self.export_plots_btn.clicked.connect(self.export_current_plot)
        self.export_plots_btn.setEnabled(False)
        export_layout.addWidget(self.export_plots_btn)
        
        layout.addWidget(export_group)
        layout.addStretch()
        return widget
        
    def create_results_panel(self):
        """Create results panel with tabs"""
        self.tabs = QTabWidget()
        
        # Global Explanations tab
        self.create_global_explanations_tab()
        
        # Local Explanations tab
        self.create_local_explanations_tab()
        
        # Dependence Plots tab
        self.create_dependence_plots_tab()
        
        return self.tabs
        
    def create_global_explanations_tab(self):
        """Create global explanations tab with interactive matplotlib"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Plot Type:"))
        self.global_plot_combo = QComboBox()
        self.global_plot_combo.addItems(["Summary Plot", "Feature Importance", "Beeswarm Plot"])
        self.global_plot_combo.currentTextChanged.connect(self.update_summary_plot)
        controls_layout.addWidget(self.global_plot_combo)
        
        controls_layout.addWidget(QLabel("Max Features:"))
        self.max_features_spin = QSpinBox()
        self.max_features_spin.setRange(5, 50)
        self.max_features_spin.setValue(20)
        self.max_features_spin.valueChanged.connect(self.update_summary_plot)
        controls_layout.addWidget(self.max_features_spin)
        
        # 添加独立窗口按钮
        self.open_summary_window_btn = QPushButton("🔗 打开独立窗口")
        self.open_summary_window_btn.clicked.connect(self.open_independent_summary_window)
        self.open_summary_window_btn.setToolTip("在独立窗口中显示SHAP摘要图")
        controls_layout.addWidget(self.open_summary_window_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Create matplotlib figure and canvas
        self.global_figure = Figure(figsize=(12, 8))
        self.global_canvas = FigureCanvas(self.global_figure)
        
        # Add navigation toolbar for zoom, pan, save etc.
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.global_toolbar = NavigationToolbar(self.global_canvas, tab)
        
        layout.addWidget(self.global_toolbar)
        layout.addWidget(self.global_canvas)
        
        self.tabs.addTab(tab, "Global Explanations")
        
    def create_local_explanations_tab(self):
        """Create local explanations tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Sample Index:"))
        self.sample_index_spin = QSpinBox()
        self.sample_index_spin.setRange(0, 0)
        self.sample_index_spin.valueChanged.connect(self.update_local_plot)
        controls_layout.addWidget(self.sample_index_spin)
        
        controls_layout.addWidget(QLabel("Plot Type:"))
        self.local_plot_combo = QComboBox()
        self.local_plot_combo.addItems(["Force Plot", "Waterfall Plot"])
        self.local_plot_combo.currentTextChanged.connect(self.update_local_plot)
        controls_layout.addWidget(self.local_plot_combo)
        
        # 添加独立窗口按钮
        self.open_local_window_btn = QPushButton("🔗 打开独立窗口")
        self.open_local_window_btn.clicked.connect(self.open_independent_local_window)
        self.open_local_window_btn.setToolTip("在独立窗口中显示SHAP局部解释图")
        controls_layout.addWidget(self.open_local_window_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        self.sample_info_text = QTextEdit()
        self.sample_info_text.setMaximumHeight(100)
        self.sample_info_text.setReadOnly(True)
        layout.addWidget(self.sample_info_text)
        
        # Create matplotlib figure and canvas
        self.local_figure = Figure(figsize=(12, 6))
        self.local_canvas = FigureCanvas(self.local_figure)
        
        # Add navigation toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.local_toolbar = NavigationToolbar(self.local_canvas, tab)
        
        layout.addWidget(self.local_toolbar)
        layout.addWidget(self.local_canvas)
        
        self.tabs.addTab(tab, "Local Explanations")
        
    def create_dependence_plots_tab(self):
        """Create dependence plots tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Main Feature:"))
        self.main_feature_combo = QComboBox()
        self.main_feature_combo.currentTextChanged.connect(self.update_dependence_plot)
        controls_layout.addWidget(self.main_feature_combo)
        
        controls_layout.addWidget(QLabel("Interaction Feature:"))
        self.interaction_feature_combo = QComboBox()
        self.interaction_feature_combo.addItem("Auto")
        self.interaction_feature_combo.currentTextChanged.connect(self.update_dependence_plot)
        controls_layout.addWidget(self.interaction_feature_combo)
        
        # 添加独立窗口按钮
        self.open_dependence_window_btn = QPushButton("🔗 打开独立窗口")
        self.open_dependence_window_btn.clicked.connect(self.open_independent_dependence_window)
        self.open_dependence_window_btn.setToolTip("在独立窗口中显示SHAP依赖图")
        controls_layout.addWidget(self.open_dependence_window_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Create matplotlib figure and canvas
        self.dependence_figure = Figure(figsize=(12, 8))
        self.dependence_canvas = FigureCanvas(self.dependence_figure)
        
        # Add navigation toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.dependence_toolbar = NavigationToolbar(self.dependence_canvas, tab)
        
        layout.addWidget(self.dependence_toolbar)
        layout.addWidget(self.dependence_canvas)
        
        self.tabs.addTab(tab, "Dependence Plots")
        
    def load_model(self):
        """Load trained model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Trained Model", "", 
            "Joblib files (*.joblib);;Pickle files (*.pkl);;All files (*)"
        )
        
        if file_path:
            try:
                self.model = joblib.load(file_path)
                model_info = self._get_model_info(self.model)
                self.model_info_text.setText(model_info)
                
                self.explainer_type = self._auto_select_explainer()
                explainer_index = self.explainer_combo.findText(self.explainer_type)
                if explainer_index >= 0:
                    self.explainer_combo.setCurrentIndex(explainer_index)
                
                self.load_background_btn.setEnabled(True)
                self.load_explain_btn.setEnabled(True)
                self.status_label.setText(f"Model loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                
    def load_background_data(self):
        """Load background data for SHAP"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Background Data", "", 
            "CSV files (*.csv);;Excel files (*.xlsx);;All files (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.background_data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.background_data = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file format")
                
                self.feature_names = list(self.background_data.columns)
                self._update_feature_selections()
                self._check_calculation_readiness()
                self.status_label.setText(f"Background data loaded: {self.background_data.shape}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load background data: {str(e)}")
                
    def load_explain_data(self):
        """Load data to explain"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Data to Explain", "", 
            "CSV files (*.csv);;Excel files (*.xlsx);;All files (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.explain_data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.explain_data = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file format")
                
                self.sample_index_spin.setRange(0, len(self.explain_data) - 1)
                self._check_calculation_readiness()
                self.status_label.setText(f"Explain data loaded: {self.explain_data.shape}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load explain data: {str(e)}")
                
    def calculate_shap_values(self):
        """Calculate SHAP values"""
        if not self._validate_inputs():
            return
            
        explainer_type = self.explainer_combo.currentText()
        if explainer_type == "Auto-Select":
            explainer_type = self.explainer_type
            
        config = {'n_background': self.background_samples_spin.value()}
        
        self.progress_bar.setVisible(True)
        self.calculate_btn.setEnabled(False)
        
        self.worker = SHAPCalculationWorker(
            self.model, self.background_data, self.explain_data, 
            explainer_type, config, self.feature_names
        )
        
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.calculation_completed.connect(self.on_calculation_completed)
        self.worker.error_occurred.connect(self.on_calculation_error)
        self.worker.start()
        
    def on_calculation_completed(self, result):
        """Handle completed SHAP calculation - Fixed data handling"""
        print(f"🔍 === SHAP CALCULATION COMPLETED ===")
        
        try:
            if result is None:
                print(f"❌ Result is None")
                QMessageBox.warning(self, "Error", "SHAP calculation returned no results")
                return
            
            print(f"🔍 Result type: {type(result)}")
            print(f"🔍 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Extract results with validation
            self.shap_values = result.get('shap_values')
            self.explainer = result.get('explainer')
            self.explain_data = result.get('explain_data')
            self.feature_names = result.get('feature_names')
            
            print(f"🔍 SHAP values extracted: {self.shap_values is not None}")
            print(f"🔍 Explainer extracted: {self.explainer is not None}")
            print(f"🔍 Explain data extracted: {self.explain_data is not None}")
            print(f"🔍 Feature names extracted: {self.feature_names is not None}")
            
            # Validate SHAP values
            if self.shap_values is None:
                print(f"❌ SHAP values are None")
                QMessageBox.warning(self, "Error", "SHAP values calculation failed")
                return
            
            print(f"🔍 SHAP values shape: {self.shap_values.shape}")
            print(f"🔍 SHAP values type: {type(self.shap_values)}")
            
            # Validate explain data
            if self.explain_data is None:
                print(f"❌ Explain data is None")
                QMessageBox.warning(self, "Error", "Explanation data is missing")
                return
            
            print(f"🔍 Explain data shape: {self.explain_data.shape}")
            print(f"🔍 Explain data type: {type(self.explain_data)}")
            
            # Validate feature names
            if not self.feature_names:
                print(f"❌ Feature names are missing")
                if hasattr(self.explain_data, 'columns'):
                    self.feature_names = list(self.explain_data.columns)
                    print(f"🔍 Using DataFrame columns as feature names: {len(self.feature_names)}")
                else:
                    self.feature_names = [f"Feature_{i}" for i in range(self.shap_values.shape[1])]
                    print(f"🔍 Generated generic feature names: {len(self.feature_names)}")
            
            print(f"🔍 Feature names count: {len(self.feature_names)}")
            print(f"🔍 First few feature names: {self.feature_names[:5]}")
            
            # Ensure data consistency
            if self.shap_values.shape[1] != len(self.feature_names):
                print(f"❌ Shape mismatch: SHAP values {self.shap_values.shape[1]} vs feature names {len(self.feature_names)}")
                # Try to fix by truncating or extending
                if self.shap_values.shape[1] < len(self.feature_names):
                    self.feature_names = self.feature_names[:self.shap_values.shape[1]]
                    print(f"🔍 Truncated feature names to {len(self.feature_names)}")
                else:
                    # Extend feature names
                    missing_count = self.shap_values.shape[1] - len(self.feature_names)
                    self.feature_names.extend([f"Feature_{i}" for i in range(len(self.feature_names), self.shap_values.shape[1])])
                    print(f"🔍 Extended feature names by {missing_count}")
            
            # Extract base value for waterfall plots
            try:
                if hasattr(self.explainer, 'expected_value'):
                    if isinstance(self.explainer.expected_value, np.ndarray):
                        self.base_value = float(self.explainer.expected_value[0])
                    else:
                        self.base_value = float(self.explainer.expected_value)
                    print(f"🔍 Base value from explainer.expected_value: {self.base_value}")
                elif hasattr(self.explainer, 'base_value'):
                    self.base_value = float(self.explainer.base_value)
                    print(f"🔍 Base value from explainer.base_value: {self.base_value}")
                else:
                    # Calculate mean prediction as fallback
                    try:
                        sample_predictions = self.model.predict(self.explain_data.iloc[:100])
                        self.base_value = float(np.mean(sample_predictions))
                        print(f"🔍 Base value from mean predictions: {self.base_value}")
                    except:
                        self.base_value = 0.0
                        print(f"🔍 Base value set to default: {self.base_value}")
            except Exception as base_e:
                print(f"❌ Base value extraction failed: {base_e}")
                self.base_value = 0.0
            
            # Update UI components
            print(f"🔍 Updating UI components...")
            
            # Update feature selection combos
            self.main_feature_combo.clear()
            self.main_feature_combo.addItems(self.feature_names)
            print(f"🔍 Main feature combo updated with {len(self.feature_names)} items")
            
            self.interaction_feature_combo.clear()
            self.interaction_feature_combo.addItem("Auto")
            self.interaction_feature_combo.addItems(self.feature_names)
            print(f"🔍 Interaction feature combo updated")
            
            # Set default selections
            if self.feature_names:
                self.main_feature_combo.setCurrentText(self.feature_names[0])
                print(f"🔍 Default main feature set to: {self.feature_names[0]}")
            
            # Update sample index range
            max_samples = min(len(self.shap_values), len(self.explain_data))
            self.sample_index_spin.setMaximum(max_samples - 1)
            print(f"🔍 Sample index range set to 0-{max_samples-1}")
            
            # Enable UI elements
            self.main_feature_combo.setEnabled(True)
            self.interaction_feature_combo.setEnabled(True)
            self.sample_index_spin.setEnabled(True)
            self.local_plot_combo.setEnabled(True)
            
            # Update summary plot immediately
            print(f"🔍 Updating summary plot...")
            self.update_summary_plot()
            
            # Update other plots
            print(f"🔍 Updating dependence plot...")
            self.update_dependence_plot()
            
            print(f"🔍 Updating local plot...")
            self.update_local_plot()
            
            print(f"✅ SHAP calculation completed successfully!")
            QMessageBox.information(self, "Success", "SHAP analysis completed successfully!")
            
            # Update UI state
            self.progress_bar.setVisible(False)
            self.calculate_btn.setEnabled(True)
            self.export_values_btn.setEnabled(True)
            self.export_plots_btn.setEnabled(True)
            
            # Switch to the first results tab to show results
            self.tabs.setCurrentIndex(0)
            
            self.status_label.setText("✅ SHAP analysis completed! Check the visualization tabs.")
            
        except Exception as e:
            print(f"❌ Error in on_calculation_completed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to process SHAP results: {str(e)}")
            
            # Reset UI state on error
            self.progress_bar.setVisible(False)
            self.calculate_btn.setEnabled(True)
            self.status_label.setText("❌ SHAP calculation failed")
        
        print(f"🔍 === SHAP CALCULATION COMPLETED END ===")
    
    def on_calculation_error(self, error_msg):
        """Handle calculation error"""
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)
        QMessageBox.critical(self, "Calculation Error", error_msg)
        self.status_label.setText("SHAP calculation failed")
        
    def update_local_plot(self):
        """Update local SHAP plot using native SHAP visualizations"""
        print(f"🔍 === LOCAL PLOT UPDATE START ===")
        
        if self.shap_values is None:
            print(f"❌ No SHAP values available")
            self._show_message_on_figure(self.local_figure, "请先计算SHAP值")
            return
            
        try:
            sample_idx = self.sample_index_spin.value()
            plot_type = self.local_plot_combo.currentText()
            
            print(f"🔍 Sample: {sample_idx}, Type: {plot_type}")
            print(f"🔍 SHAP values shape: {self.shap_values.shape}")
            print(f"🔍 Explain data shape: {self.explain_data.shape}")
            
            # Validate sample index
            if sample_idx >= len(self.shap_values) or sample_idx >= len(self.explain_data):
                print(f"❌ Invalid sample index: {sample_idx}")
                self._show_message_on_figure(self.local_figure, f"样本索引 {sample_idx} 超出范围")
                return
            
            # Update sample info
            sample_data = self.explain_data.iloc[sample_idx]
            try:
                prediction = self.model.predict(sample_data.to_frame().T)[0]
                print(f"🔍 Prediction: {prediction}")
            except Exception as pred_e:
                print(f"❌ Prediction failed: {pred_e}")
                prediction = "N/A"
            
            # Update info text
            info_text = f"Sample {sample_idx} - Prediction: {prediction}\n"
            try:
                sample_items = list(sample_data.items())[:10]
                info_text += "\n".join([f"{feat}: {val:.4f}" if isinstance(val, (int, float)) else f"{feat}: {val}" 
                                       for feat, val in sample_items])
                if len(sample_data) > 10:
                    info_text += f"\n... and {len(sample_data) - 10} more features"
            except Exception as info_e:
                print(f"❌ Info text creation failed: {info_e}")
                info_text += "\nError displaying feature values"
            
            self.sample_info_text.setText(info_text)
            
            # Prepare data for SHAP
            explain_data_subset = self.explain_data.iloc[:len(self.shap_values)]
            if not isinstance(explain_data_subset, pd.DataFrame):
                explain_data_df = pd.DataFrame(explain_data_subset, columns=self.feature_names)
            else:
                explain_data_df = explain_data_subset.copy()
            
            # Create the plot
            print(f"🔍 Creating native {plot_type}...")
            
            if plot_type == "Waterfall Plot":
                success = self._create_native_waterfall_plot(sample_idx, explain_data_df)
            elif plot_type == "Force Plot":
                success = self._create_native_force_plot(sample_idx, explain_data_df)
            else:
                success = self._create_native_waterfall_plot(sample_idx, explain_data_df)
            
            if not success:
                print(f"🔄 Native {plot_type} failed, using fallback...")
                self._create_fallback_local_plot(sample_idx, plot_type)
            
            print(f"✅ Local plot created successfully")
            
        except Exception as e:
            print(f"❌ Error creating local plot: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            self._create_fallback_local_plot(sample_idx if 'sample_idx' in locals() else 0, "Waterfall Plot")
        
        print(f"🔍 === LOCAL PLOT UPDATE END ===")
    
    def update_summary_plot(self):
        """Update summary plot using native SHAP visualizations"""
        print(f"🔍 === SUMMARY PLOT UPDATE START ===")
        
        if self.shap_values is None:
            print(f"❌ No SHAP values available")
            self._show_message_on_figure(self.global_figure, "请先计算SHAP值")
            return
            
        try:
            plot_type = self.global_plot_combo.currentText()
            max_display = self.max_features_spin.value()
            
            print(f"🔍 Creating {plot_type}...")
            print(f"🔍 SHAP values shape: {self.shap_values.shape}")
            print(f"🔍 Feature names count: {len(self.feature_names) if self.feature_names else 0}")
            
            # Prepare data for SHAP
            n_samples = min(self.shap_values.shape[0], len(self.explain_data))
            shap_vals = self.shap_values[:n_samples]
            explain_data_subset = self.explain_data.iloc[:n_samples]
            
            # Ensure DataFrame format for SHAP compatibility
            if not isinstance(explain_data_subset, pd.DataFrame):
                explain_data_df = pd.DataFrame(explain_data_subset, columns=self.feature_names)
            else:
                explain_data_df = explain_data_subset.copy()
            
            print(f"🔍 Data prepared - SHAP: {shap_vals.shape}, Data: {explain_data_df.shape}")
            
            # Create SHAP plot based on type
            if plot_type == "Summary Plot":
                success = self._create_native_shap_plot(
                    lambda: shap.summary_plot(
                        shap_vals, explain_data_df,
                        feature_names=self.feature_names,
                        max_display=max_display, show=False
                    ),
                    self.global_figure,
                    "Summary Plot"
                )
                
            elif plot_type == "Feature Importance":
                success = self._create_native_shap_plot(
                    lambda: shap.summary_plot(
                        shap_vals, explain_data_df,
                        feature_names=self.feature_names,
                        plot_type="bar", max_display=max_display, show=False
                    ),
                    self.global_figure,
                    "Feature Importance"
                )
                
            elif plot_type == "Beeswarm Plot":
                def create_beeswarm():
                    if hasattr(shap, 'plots') and hasattr(shap.plots, 'beeswarm'):
                        # Modern SHAP API
                        shap_explanation = shap.Explanation(
                            values=shap_vals,
                            data=explain_data_df.values,
                            feature_names=self.feature_names
                        )
                        shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
                    else:
                        # Fallback to summary plot
                        shap.summary_plot(
                            shap_vals, explain_data_df,
                            feature_names=self.feature_names,
                            max_display=max_display, show=False
                        )
                
                success = self._create_native_shap_plot(
                    create_beeswarm,
                    self.global_figure,
                    "Beeswarm Plot"
                )
            
            if not success:
                print(f"🔄 Native SHAP failed, using fallback...")
                self._create_fallback_summary_plot(max_display)
            
            print(f"✅ Summary plot created successfully")
            
        except Exception as e:
            print(f"❌ Error creating summary plot: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            self._create_fallback_summary_plot(20)
        
        print(f"🔍 === SUMMARY PLOT UPDATE END ===")
    
    def _create_native_shap_plot(self, shap_plot_func, target_figure, plot_name):
        """Create native SHAP plot with reliable image conversion method"""
        try:
            print(f"🔍 Creating native {plot_name}...")
            
            # Clear target figure completely
            target_figure.clear()
            
            # Create temporary figure for SHAP plot
            import matplotlib.pyplot as plt
            temp_fig = plt.figure(figsize=(12, 8))
            
            # Execute SHAP plotting function
            print(f"🔍 Executing SHAP plotting function...")
            shap_plot_func()
            
            # Save to memory buffer with high quality
            import io
            buf = io.BytesIO()
            temp_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.1)
            buf.seek(0)
            
            # Close temporary figure
            plt.close(temp_fig)
            
            # Load image and display in target figure
            from PIL import Image
            import numpy as np
            
            img = Image.open(buf)
            img_array = np.array(img)
            
            print(f"🔍 Image loaded, shape: {img_array.shape}")
            
            # Create subplot in target figure with full size
            ax = target_figure.add_subplot(111)
            ax.imshow(img_array, aspect='auto', interpolation='bilinear')
            ax.axis('off')
            
            # Remove all margins and padding for full display
            ax.set_position([0, 0, 1, 1])
            target_figure.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
            # Determine and refresh the correct canvas
            canvas_to_refresh = None
            canvas_name = "unknown"
            if target_figure == self.global_figure:
                canvas_to_refresh = self.global_canvas
                canvas_name = "global"
            elif target_figure == self.local_figure:
                canvas_to_refresh = self.local_canvas
                canvas_name = "local"
            elif target_figure == self.dependence_figure:
                canvas_to_refresh = self.dependence_canvas
                canvas_name = "dependence"
            
            if canvas_to_refresh:
                print(f"🔍 Refreshing {canvas_name} canvas...")
                
                # Comprehensive refresh sequence
                canvas_to_refresh.draw()
                canvas_to_refresh.flush_events()
                QApplication.processEvents()
                
                # Force widget update
                canvas_to_refresh.update()
                canvas_to_refresh.repaint()
                QApplication.processEvents()
                
                # Final draw call
                canvas_to_refresh.draw_idle()
                QApplication.processEvents()
                
                print(f"🔍 {canvas_name} canvas refreshed successfully")
            else:
                print(f"⚠️ Canvas not found for {plot_name}")
                target_figure.canvas.draw()
                target_figure.canvas.flush_events()
                QApplication.processEvents()
            
            print(f"✅ Native {plot_name} created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Native {plot_name} failed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return False
    
    def _create_native_shap_plot_image_conversion(self, shap_plot_func, target_figure, plot_name):
        """Fallback: Create native SHAP plot with image conversion method"""
        try:
            print(f"🔍 Creating native {plot_name} with image conversion...")
            
            # Clear target figure
            target_figure.clear()
            
            # Create temporary figure for SHAP plot
            import matplotlib.pyplot as plt
            temp_fig = plt.figure(figsize=(12, 8))
            
            # Execute SHAP plotting function
            shap_plot_func()
            
            # Save to memory buffer with higher quality
            import io
            buf = io.BytesIO()
            temp_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.1)
            buf.seek(0)
            
            # Close temporary figure
            plt.close(temp_fig)
            
            # Load image and display in target figure
            from PIL import Image
            import numpy as np
            
            img = Image.open(buf)
            img_array = np.array(img)
            
            # Create subplot in target figure with full size
            ax = target_figure.add_subplot(111)
            ax.imshow(img_array, aspect='auto', interpolation='bilinear')
            ax.axis('off')
            
            # Remove all margins and padding
            ax.set_position([0, 0, 1, 1])
            target_figure.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
            # Enhanced canvas refresh with multiple attempts
            canvas_to_refresh = None
            if target_figure == self.global_figure:
                canvas_to_refresh = self.global_canvas
            elif target_figure == self.local_figure:
                canvas_to_refresh = self.local_canvas
            elif target_figure == self.dependence_figure:
                canvas_to_refresh = self.dependence_canvas
            
            if canvas_to_refresh:
                # Multiple refresh cycles
                for i in range(3):
                    canvas_to_refresh.draw()
                    canvas_to_refresh.flush_events()
                    QApplication.processEvents()
                    canvas_to_refresh.repaint()
                    QApplication.processEvents()
                    
                    if i < 2:  # Small delay between refreshes
                        import time
                        time.sleep(0.05)
                
                print(f"🔍 Canvas refreshed with image conversion for {plot_name}")
            
            print(f"✅ Native {plot_name} created successfully with image conversion")
            return True
            
        except Exception as e:
            print(f"❌ Native {plot_name} image conversion failed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return False
    
    def _create_fallback_summary_plot(self, max_display):
        """Fallback summary plot when native SHAP fails"""
        try:
            print(f"🔍 Creating fallback summary plot...")
            
            self.global_figure.clear()
            ax = self.global_figure.add_subplot(111)
            
            # Calculate feature importance
            feature_importance = np.mean(np.abs(self.shap_values), axis=0)
            
            # Sort and select top features
            sorted_indices = np.argsort(feature_importance)[-max_display:]
            sorted_importance = feature_importance[sorted_indices]
            sorted_names = [self.feature_names[i] for i in sorted_indices] if self.feature_names else [f"Feature_{i}" for i in sorted_indices]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(sorted_importance))
            bars = ax.barh(y_pos, sorted_importance, color='steelblue', alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('SHAP Feature Importance (Fallback)')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, sorted_importance):
                width = bar.get_width()
                ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', ha='left', va='center', fontsize=8)
            
            self.global_figure.tight_layout()
            self.global_figure.canvas.draw()
            self.global_figure.canvas.flush_events()
            
            print(f"✅ Fallback summary plot created")
            
        except Exception as e:
            print(f"❌ Fallback summary plot failed: {e}")
            self._show_message_on_figure(self.global_figure, f"绘图失败: {str(e)}")
      
    def _create_native_waterfall_plot(self, sample_idx, explain_data_df):
        """Create native SHAP waterfall plot"""
        try:
            print(f"🔍 Creating native waterfall plot for sample {sample_idx}")
            
            def create_waterfall():
                # Try modern SHAP waterfall plot
                if hasattr(shap, 'plots') and hasattr(shap.plots, 'waterfall'):
                    print(f"🔍 Using modern SHAP waterfall plot")
                    shap_explanation = shap.Explanation(
                        values=self.shap_values[sample_idx],
                        base_values=getattr(self, 'base_value', 0),
                        data=explain_data_df.iloc[sample_idx].values,
                        feature_names=self.feature_names
                    )
                    shap.plots.waterfall(shap_explanation, show=False)
                elif hasattr(shap, 'waterfall_plot'):
                    print(f"🔍 Using legacy SHAP waterfall plot")
                    shap.waterfall_plot(
                        getattr(self, 'base_value', 0),
                        self.shap_values[sample_idx],
                        explain_data_df.iloc[sample_idx],
                        feature_names=self.feature_names,
                        show=False
                    )
                else:
                    raise Exception("No native SHAP waterfall plot available")
            
            return self._create_native_shap_plot(
                create_waterfall,
                self.local_figure,
                "Waterfall Plot"
            )
            
        except Exception as e:
            print(f"❌ Native waterfall plot failed: {e}")
            return False
    
    def _create_native_force_plot(self, sample_idx, explain_data_df):
        """Create native SHAP force plot"""
        try:
            print(f"🔍 Creating native force plot for sample {sample_idx}")
            
            def create_force():
                if hasattr(shap, 'force_plot'):
                    print(f"🔍 Using SHAP force plot")
                    shap.force_plot(
                        getattr(self, 'base_value', 0),
                        self.shap_values[sample_idx],
                        explain_data_df.iloc[sample_idx],
                        feature_names=self.feature_names,
                        matplotlib=True,
                        show=False
                    )
                elif hasattr(shap, 'plots') and hasattr(shap.plots, 'force'):
                    print(f"🔍 Using modern SHAP force plot")
                    shap_explanation = shap.Explanation(
                        values=self.shap_values[sample_idx],
                        base_values=getattr(self, 'base_value', 0),
                        data=explain_data_df.iloc[sample_idx].values,
                        feature_names=self.feature_names
                    )
                    shap.plots.force(shap_explanation, matplotlib=True, show=False)
                else:
                    raise Exception("No native SHAP force plot available")
            
            return self._create_native_shap_plot(
                create_force,
                self.local_figure,
                "Force Plot"
            )
            
        except Exception as e:
            print(f"❌ Native force plot failed: {e}")
            return False
    
    def _create_fallback_local_plot(self, sample_idx, plot_type):
        """Fallback local plot when native SHAP fails"""
        try:
            print(f"🔍 Creating fallback {plot_type}...")
            
            self.local_figure.clear()
            ax = self.local_figure.add_subplot(111)
            
            # Get SHAP values for this sample
            sample_shap_vals = self.shap_values[sample_idx]
            sample_data = self.explain_data.iloc[sample_idx]
            
            if plot_type == "Waterfall Plot":
                self._create_simple_waterfall(ax, sample_shap_vals, sample_idx)
            elif plot_type == "Force Plot":
                self._create_simple_force_plot(ax, sample_shap_vals, sample_data, sample_idx)
            else:
                self._create_simple_waterfall(ax, sample_shap_vals, sample_idx)
            
            # Enhanced canvas refresh
            self.local_figure.tight_layout()
            self.local_canvas.draw()
            self.local_canvas.flush_events()
            QApplication.processEvents()
            
            # Force immediate repaint
            self.local_canvas.repaint()
            QApplication.processEvents()
            
            print(f"✅ Fallback {plot_type} created")
            
        except Exception as e:
            print(f"❌ Fallback {plot_type} failed: {e}")
            self._show_message_on_figure(self.local_figure, f"绘图失败: {str(e)}")
    
    def _create_simple_waterfall(self, ax, shap_vals, sample_idx):
        """Create a simple waterfall plot"""
        print(f"🔍 Creating simple waterfall plot")
        
        # Sort features by absolute SHAP value
        sorted_indices = np.argsort(np.abs(shap_vals))[-15:]  # Top 15 features
        
        # Get values and names for top features
        top_shap_vals = shap_vals[sorted_indices]
        top_feature_names = [self.feature_names[i] for i in sorted_indices]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_shap_vals))
        colors = ['red' if val < 0 else 'blue' for val in top_shap_vals]
        
        bars = ax.barh(y_pos, top_shap_vals, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feature_names)
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'SHAP Waterfall Plot - Sample {sample_idx}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, top_shap_vals):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)
        
        print(f"✅ Simple waterfall plot created")
    
    def _create_simple_force_plot(self, ax, shap_vals, sample_data, sample_idx):
        """Create a simple force plot visualization"""
        print(f"🔍 Creating simple force plot")
        
        # Sort features by absolute SHAP value
        sorted_indices = np.argsort(np.abs(shap_vals))[-10:]  # Top 10 features
        
        # Get values and names for top features
        top_shap_vals = shap_vals[sorted_indices]
        top_feature_names = [self.feature_names[i] for i in sorted_indices]
        top_feature_vals = [sample_data.iloc[i] for i in sorted_indices]
        
        # Create force plot style visualization
        base_value = getattr(self, 'base_value', 0)
        cumulative = base_value
        
        # Plot base value
        ax.barh(0, base_value, color='gray', alpha=0.5, label=f'Base Value: {base_value:.3f}')
        
        # Plot each feature's contribution
        for i, (shap_val, feat_name, feat_val) in enumerate(zip(top_shap_vals, top_feature_names, top_feature_vals)):
            color = 'red' if shap_val < 0 else 'blue'
            ax.barh(i+1, shap_val, left=cumulative, color=color, alpha=0.7)
            
            # Add text annotation
            ax.text(cumulative + shap_val/2, i+1, f'{feat_name}\n{feat_val:.3f}\n({shap_val:+.3f})', 
                   ha='center', va='center', fontsize=8, weight='bold')
            
            cumulative += shap_val
        
        # Final prediction line
        ax.axvline(x=cumulative, color='green', linestyle='--', linewidth=2, 
                  label=f'Prediction: {cumulative:.3f}')
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Features')
        ax.set_title(f'SHAP Force Plot - Sample {sample_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"✅ Simple force plot created")
    
    def update_dependence_plot(self):
        """Update dependence plot using native SHAP visualizations"""
        print(f"🔍 === DEPENDENCE PLOT UPDATE START ===")
        print(f"🔍 SHAP values exist: {self.shap_values is not None}")
        print(f"🔍 Feature names exist: {self.feature_names is not None}")
        
        if self.shap_values is not None:
            print(f"🔍 SHAP values shape: {self.shap_values.shape}")
        if self.feature_names:
            print(f"🔍 Feature names count: {len(self.feature_names)}")
            print(f"🔍 First few features: {self.feature_names[:3]}")
        
        # Check basic requirements
        if self.shap_values is None:
            print(f"❌ No SHAP values available")
            self._show_message_on_figure(self.dependence_figure, "请先计算SHAP值")
            return
            
        if not self.feature_names:
            print(f"❌ No feature names available")
            self._show_message_on_figure(self.dependence_figure, "特征名称缺失")
            return
        
        # Get current selections
        main_feature = self.main_feature_combo.currentText()
        interaction_feature = self.interaction_feature_combo.currentText()
        
        print(f"🔍 Selected features - Main: '{main_feature}', Interaction: '{interaction_feature}'")
        
        # Validate main feature
        if not main_feature:
            print(f"❌ No main feature selected")
            self._show_message_on_figure(self.dependence_figure, "请选择主要特征")
            return
            
        if main_feature not in self.feature_names:
            print(f"❌ Main feature '{main_feature}' not in feature list")
            self._show_message_on_figure(self.dependence_figure, f"特征 '{main_feature}' 不存在")
            return
        
        try:
            print(f"🔍 Creating native dependence plot...")
            
            # Prepare data for SHAP
            n_samples = min(self.shap_values.shape[0], len(self.explain_data))
            shap_vals = self.shap_values[:n_samples]
            explain_data_subset = self.explain_data.iloc[:n_samples]
            
            # Ensure DataFrame format for SHAP compatibility
            if not isinstance(explain_data_subset, pd.DataFrame):
                explain_data_df = pd.DataFrame(explain_data_subset, columns=self.feature_names)
            else:
                explain_data_df = explain_data_subset.copy()
            
            print(f"🔍 Data prepared - SHAP: {shap_vals.shape}, Data: {explain_data_df.shape}")
            
            # Get feature indices
            main_idx = self.feature_names.index(main_feature)
            print(f"🔍 Main feature index: {main_idx}")
            
            # Create native SHAP dependence plot
            def create_dependence():
                if interaction_feature == "Auto" or interaction_feature == main_feature:
                    print(f"🔍 Creating auto dependence plot")
                    shap.dependence_plot(
                        main_idx, shap_vals, explain_data_df,
                        feature_names=self.feature_names, show=False
                    )
                else:
                    if interaction_feature in self.feature_names:
                        interaction_idx = self.feature_names.index(interaction_feature)
                        print(f"🔍 Creating interaction dependence plot with index: {interaction_idx}")
                        shap.dependence_plot(
                            main_idx, shap_vals, explain_data_df,
                            feature_names=self.feature_names,
                            interaction_index=interaction_idx, show=False
                        )
                    else:
                        print(f"🔍 Interaction feature not found, using auto")
                        shap.dependence_plot(
                            main_idx, shap_vals, explain_data_df,
                            feature_names=self.feature_names, show=False
                        )
            
            # Try native SHAP plot first
            success = self._create_native_shap_plot(
                create_dependence,
                self.dependence_figure,
                "Dependence Plot"
            )
            
            if not success:
                print(f"🔄 Native dependence plot failed, using fallback...")
                self._create_fallback_dependence_plot(main_feature, interaction_feature)
            
            print(f"✅ Dependence plot created successfully")
            
        except Exception as e:
            print(f"❌ Error creating dependence plot: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            self._create_fallback_dependence_plot(main_feature, interaction_feature)
        
        print(f"🔍 === DEPENDENCE PLOT UPDATE END ===")
    
    def _create_fallback_dependence_plot(self, main_feature, interaction_feature):
        """Fallback dependence plot when native SHAP fails"""
        try:
            print(f"🔍 Creating fallback dependence plot...")
            
            self.dependence_figure.clear()
            ax = self.dependence_figure.add_subplot(111)
            
            # Get feature indices
            main_idx = self.feature_names.index(main_feature)
            
            # Prepare data
            n_samples = min(self.shap_values.shape[0], len(self.explain_data))
            shap_vals = self.shap_values[:n_samples, main_idx]
            feature_vals = self.explain_data.iloc[:n_samples, main_idx].values
            
            # Handle interaction feature
            if interaction_feature == "Auto" or interaction_feature == main_feature or interaction_feature not in self.feature_names:
                # Simple dependence plot colored by feature value
                print(f"🔍 Creating simple fallback dependence plot")
                
                # Handle boolean/categorical data
                if feature_vals.dtype == bool:
                    feature_vals_numeric = feature_vals.astype(int)
                else:
                    feature_vals_numeric = feature_vals
                
                scatter = ax.scatter(feature_vals_numeric, shap_vals, 
                                   c=feature_vals_numeric, cmap='viridis', alpha=0.7, s=30)
                ax.set_xlabel(f'{main_feature} (Feature Value)')
                ax.set_ylabel(f'SHAP Value for {main_feature}')
                ax.set_title(f'SHAP Dependence Plot: {main_feature} (Fallback)')
                
                # Add colorbar
                try:
                    cbar = self.dependence_figure.colorbar(scatter, ax=ax)
                    cbar.set_label(main_feature)
                except:
                    pass
                    
            else:
                # Interaction dependence plot
                print(f"🔍 Creating interaction fallback dependence plot with {interaction_feature}")
                int_idx = self.feature_names.index(interaction_feature)
                int_vals = self.explain_data.iloc[:n_samples, int_idx].values
                
                # Handle boolean/categorical data
                if feature_vals.dtype == bool:
                    feature_vals_numeric = feature_vals.astype(int)
                else:
                    feature_vals_numeric = feature_vals
                    
                if int_vals.dtype == bool:
                    int_vals_numeric = int_vals.astype(int)
                else:
                    int_vals_numeric = int_vals
                
                scatter = ax.scatter(feature_vals_numeric, shap_vals, 
                                   c=int_vals_numeric, cmap='viridis', alpha=0.7, s=30)
                ax.set_xlabel(f'{main_feature} (Feature Value)')
                ax.set_ylabel(f'SHAP Value for {main_feature}')
                ax.set_title(f'SHAP Dependence: {main_feature} vs {interaction_feature} (Fallback)')
                
                # Add colorbar
                try:
                    cbar = self.dependence_figure.colorbar(scatter, ax=ax)
                    cbar.set_label(interaction_feature)
                except:
                    pass
            
            # Add reference line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Enhanced canvas refresh
            self.dependence_figure.tight_layout()
            self.dependence_canvas.draw()
            self.dependence_canvas.flush_events()
            QApplication.processEvents()
            
            # Force immediate repaint
            self.dependence_canvas.repaint()
            QApplication.processEvents()
            
            print(f"✅ Fallback dependence plot created")
            
        except Exception as e:
            print(f"❌ Fallback dependence plot failed: {e}")
            self._show_message_on_figure(self.dependence_figure, f"绘图失败: {str(e)}")
    
    def _show_message_on_figure(self, figure, message):
        """Show a message on the figure when plots fail"""
        try:
            figure.clear()
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, message, ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            figure.tight_layout()
            
            # Enhanced canvas refresh based on figure type
            canvas_to_refresh = None
            if figure == self.global_figure:
                canvas_to_refresh = self.global_canvas
            elif figure == self.local_figure:
                canvas_to_refresh = self.local_canvas
            elif figure == self.dependence_figure:
                canvas_to_refresh = self.dependence_canvas
            
            if canvas_to_refresh:
                canvas_to_refresh.draw()
                canvas_to_refresh.flush_events()
                QApplication.processEvents()
                canvas_to_refresh.repaint()
                QApplication.processEvents()
            else:
                figure.canvas.draw()
                figure.canvas.flush_events()
                QApplication.processEvents()
                
        except Exception as e:
            print(f"❌ Failed to show message: {e}")
    
    def export_shap_values(self):
        """Export SHAP values to file"""
        if self.shap_values is None:
            QMessageBox.warning(self, "Warning", "No SHAP values to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export SHAP Values", "shap_values.csv", "CSV files (*.csv)"
        )
        
        if file_path:
            try:
                shap_df = pd.DataFrame(self.shap_values, columns=self.feature_names)
                shap_df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"SHAP values exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export SHAP values: {str(e)}")
                
    def export_current_plot(self):
        """Export current plot"""
        current_tab = self.tabs.currentIndex()
        tab_names = ["global", "local", "dependence"]
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", f"shap_{tab_names[current_tab]}_plot.png", 
            "PNG files (*.png);;SVG files (*.svg);;PDF files (*.pdf)"
        )
        
        if file_path:
            try:
                if current_tab == 0:
                    self.update_summary_plot()
                elif current_tab == 1:
                    self.update_local_plot()
                elif current_tab == 2:
                    self.update_dependence_plot()
                
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export plot: {str(e)}")
                
    def _get_model_info(self, model):
        """Get model information string"""
        if isinstance(model, Pipeline):
            model_type = type(model.named_steps[model.steps[-1][0]]).__name__
            preprocessing_steps = len(model.steps) - 1
            info = f"Pipeline with {preprocessing_steps} preprocessing steps\n"
            info += f"Final estimator: {model_type}"
        else:
            info = f"Model type: {type(model).__name__}"
        return info
        
    def _auto_select_explainer(self):
        """Auto-select appropriate SHAP explainer"""
        if isinstance(self.model, Pipeline):
            estimator = self.model.named_steps[self.model.steps[-1][0]]
        else:
            estimator = self.model
            
        estimator_name = type(estimator).__name__
        
        # Tree-based models
        tree_models = ['RandomForestRegressor', 'RandomForestClassifier', 
                      'DecisionTreeRegressor', 'DecisionTreeClassifier',
                      'XGBRegressor', 'XGBClassifier', 'LGBMRegressor', 'LGBMClassifier']
        
        if estimator_name in tree_models:
            return "TreeExplainer"
            
        # Linear models
        linear_models = ['LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso']
        if estimator_name in linear_models:
            return "LinearExplainer"
            
        return "KernelExplainer"
        
    def _update_feature_selections(self):
        """Update feature selection combo boxes"""
        print(f"🔍 _update_feature_selections called")
        print(f"🔍 Feature names available: {bool(self.feature_names)}")
        if self.feature_names:
            print(f"🔍 Feature names count: {len(self.feature_names)}")
            print(f"🔍 First few features: {self.feature_names[:5]}")
            
            print(f"🔍 Updating main feature combo...")
            self.main_feature_combo.clear()
            self.main_feature_combo.addItems(self.feature_names)
            print(f"🔍 Main feature combo updated, current: {self.main_feature_combo.currentText()}")
            
            print(f"🔍 Updating interaction feature combo...")
            self.interaction_feature_combo.clear()
            self.interaction_feature_combo.addItem("Auto")
            self.interaction_feature_combo.addItems(self.feature_names)
            print(f"🔍 Interaction feature combo updated, current: {self.interaction_feature_combo.currentText()}")
        else:
            print(f"❌ No feature names available for combo boxes")
        
    def _check_calculation_readiness(self):
        """Check if ready for SHAP calculation"""
        ready = (self.model is not None and 
                self.background_data is not None and 
                self.explain_data is not None)
        self.calculate_btn.setEnabled(ready)
        
    def _validate_inputs(self):
        """Validate inputs before calculation"""
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a trained model first")
            return False
        if self.background_data is None:
            QMessageBox.warning(self, "Warning", "Please load background data first")
            return False
        if self.explain_data is None:
            QMessageBox.warning(self, "Warning", "Please load data to explain first")
            return False
        return True
    
    def set_model(self, trained_model, feature_names=None, feature_info=None, X_train=None, y_train=None):
        """Set trained model from training module"""
        self.model = trained_model
        
        # Store feature names if provided
        if feature_names:
            self.feature_names = feature_names
        else:
            # Extract feature names from model safely
            self._extract_feature_names()
        
        # Extract model information
        model_info = self._get_model_info(self.model)
        self.model_info_text.setText(model_info)
        
        # Auto-select explainer type
        self.explainer_type = self._auto_select_explainer()
        explainer_index = self.explainer_combo.findText(self.explainer_type)
        if explainer_index >= 0:
            self.explainer_combo.setCurrentIndex(explainer_index)
        
        # Enable data loading buttons
        self.load_background_btn.setEnabled(True)
        self.load_explain_btn.setEnabled(True)
        
        # If training data is provided, store it
        if X_train is not None and y_train is not None:
            self.training_data = {'X': X_train.copy(), 'y': y_train.copy()}
            
            # Update sample info
            self.sample_info_text.setText(
                f"Training Data Loaded:\n"
                f"• Samples: {len(X_train)}\n"
                f"• Features: {len(feature_names) if feature_names else X_train.shape[1]}\n"
                f"• Target: {y_train.name if hasattr(y_train, 'name') else 'Unknown'}\n"
                f"• Feature names: {', '.join(feature_names[:5]) if feature_names else 'N/A'}{'...' if feature_names and len(feature_names) > 5 else ''}"
            )
            
            # Auto-setup the data
            self._auto_setup_data()
        
        self.status_label.setText("Model loaded from training session")
        
    def _extract_feature_names(self):
        """Safely extract feature names from model"""
        try:
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
            elif hasattr(self.model, 'steps'):
                # For pipeline, try to get feature names from final estimator
                final_estimator = self.model.steps[-1][1]
                if hasattr(final_estimator, 'feature_names_in_'):
                    self.feature_names = list(final_estimator.feature_names_in_)
                else:
                    # Use training data feature names if available
                    if self.training_data is not None:
                        self.feature_names = list(self.training_data['X'].columns)
                    else:
                        # Fallback to generic names
                        n_features = self._get_n_features_from_model()
                        self.feature_names = [f"feature_{i}" for i in range(n_features)]
            else:
                # Direct model
                if self.training_data is not None:
                    self.feature_names = list(self.training_data['X'].columns)
                else:
                    n_features = self._get_n_features_from_model()
                    self.feature_names = [f"feature_{i}" for i in range(n_features)]
        except Exception as e:
            print(f"Error extracting feature names: {e}")
            # Ultimate fallback
            if self.training_data is not None:
                self.feature_names = list(self.training_data['X'].columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(10)]
                
    def _get_n_features_from_model(self):
        """Get number of features from model"""
        try:
            if hasattr(self.model, 'n_features_in_'):
                return self.model.n_features_in_
            elif hasattr(self.model, 'steps'):
                final_estimator = self.model.steps[-1][1]
                if hasattr(final_estimator, 'n_features_in_'):
                    return final_estimator.n_features_in_
            return 10  # Default fallback
        except:
            return 10
        
    def set_data(self, X, y, config=None):
        """Set training data from feature selection module"""
        try:
            print(f"SHAP: Receiving data - X shape: {X.shape}, y shape: {y.shape}")
            self.training_data = {'X': X.copy(), 'y': y.copy()}
            self.feature_names = X.columns.tolist()
            
            # Store configuration if provided
            if config:
                self.config = config
            
            # Update sample info
            self.sample_info_text.setText(
                f"Training Data Loaded:\n"
                f"• Samples: {len(X)}\n"
                f"• Features: {len(X.columns)}\n"
                f"• Target: {y.name if hasattr(y, 'name') else 'Unknown'}\n"
                f"• Feature names: {', '.join(X.columns[:5])}{'...' if len(X.columns) > 5 else ''}"
            )
            
            # If we already have a model, auto-setup the data
            if self.model is not None:
                self._auto_setup_data()
                
            print("SHAP: Data successfully set")
            
        except Exception as e:
            print(f"Error setting SHAP data: {e}")
            self.status_label.setText(f"Error loading data: {str(e)}")
            
    def _auto_setup_data(self):
        """Automatically setup background and explain data from training data"""
        if self.training_data is None:
            return
            
        X = self.training_data['X']
        
        # Use a sample of training data as background data
        background_size = min(self.background_samples_spin.value(), len(X))
        sample_indices = np.random.choice(len(X), size=background_size, replace=False)
        self.background_data = X.iloc[sample_indices].copy()
        
        # Use first 100 samples (or less) as explanation data
        explain_size = min(100, len(X))
        self.explain_data = X.iloc[:explain_size].copy()
        
        # Update UI
        self.sample_index_spin.setRange(0, len(self.explain_data) - 1)
        self._update_feature_selections()
        self._check_calculation_readiness()
        
        self.status_label.setText(f"Auto-loaded: {len(self.background_data)} background samples, {len(self.explain_data)} explain samples")
        
    def reset(self):
        """Reset the module"""
        self.model = None
        self.background_data = None
        self.explain_data = None
        self.feature_names = None
        self.shap_values = None
        self.explainer = None
        self.training_data = None
        
        self.model_info_text.clear()
        self.sample_info_text.clear()
        
        self.load_background_btn.setEnabled(False)
        self.load_explain_btn.setEnabled(False)
        self.calculate_btn.setEnabled(False)
        self.export_values_btn.setEnabled(False)
        self.export_plots_btn.setEnabled(False)
        
        # Clear matplotlib figures
        if hasattr(self, 'global_figure'):
            self.global_figure.clear()
            self.global_canvas.draw()
        if hasattr(self, 'local_figure'):
            self.local_figure.clear()
            self.local_canvas.draw()
        if hasattr(self, 'dependence_figure'):
            self.dependence_figure.clear()
            self.dependence_canvas.draw()
        
        self.status_label.setText("Ready")
    
    def test_canvas_display(self):
        """Test method to verify canvas display works"""
        print(f"🔍 Testing canvas display...")
        
        # Test dependence canvas
        try:
            self.dependence_figure.clear()
            ax = self.dependence_figure.add_subplot(111)
            
            # Create simple test plot
            import numpy as np
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            
            ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Dependence Canvas Test')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Enhanced refresh
            self.dependence_figure.tight_layout()
            self.dependence_canvas.draw()
            self.dependence_canvas.flush_events()
            QApplication.processEvents()
            self.dependence_canvas.repaint()
            QApplication.processEvents()
            
            print(f"✅ Dependence canvas test completed")
            
        except Exception as e:
            print(f"❌ Dependence canvas test failed: {e}")
        
        # Test local canvas
        try:
            self.local_figure.clear()
            ax = self.local_figure.add_subplot(111)
            
            # Create simple test plot
            x = np.linspace(0, 10, 100)
            y = np.cos(x)
            
            ax.plot(x, y, 'r-', linewidth=2, label='cos(x)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Local Canvas Test')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Enhanced refresh
            self.local_figure.tight_layout()
            self.local_canvas.draw()
            self.local_canvas.flush_events()
            QApplication.processEvents()
            self.local_canvas.repaint()
            QApplication.processEvents()
            
            print(f"✅ Local canvas test completed")
            
        except Exception as e:
            print(f"❌ Local canvas test failed: {e}")
    
    def force_refresh_all_canvas(self):
        """强制刷新所有画布"""
        print(f"🔍 Force refreshing all canvas...")
        
        try:
            # 刷新全局画布
            if hasattr(self, 'global_canvas'):
                self.global_canvas.draw()
                self.global_canvas.flush_events()
                QApplication.processEvents()
                print(f"🔍 Global canvas force refreshed")
            
            # 刷新局部画布
            if hasattr(self, 'local_canvas'):
                self.local_canvas.draw()
                self.local_canvas.flush_events()
                QApplication.processEvents()
                print(f"🔍 Local canvas force refreshed")
            
            # 刷新依赖画布
            if hasattr(self, 'dependence_canvas'):
                self.dependence_canvas.draw()
                self.dependence_canvas.flush_events()
                QApplication.processEvents()
                print(f"🔍 Dependence canvas force refreshed")
            
            print(f"✅ All canvas force refreshed")
            
        except Exception as e:
            print(f"❌ Error force refreshing canvas: {e}")
    
    def open_independent_summary_window(self):
        """打开独立的摘要图窗口"""
        if self.shap_values is None:
            QMessageBox.warning(self, "警告", "请先计算SHAP值")
            return
        
        try:
            plot_type = self.global_plot_combo.currentText()
            window = SHAPVisualizationWindow(
                self.shap_values, self.explainer, self.explain_data, 
                self.feature_names, plot_type, self
            )
            window.show()
            print(f"✅ 独立摘要图窗口已打开: {plot_type}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开独立窗口失败:\n{str(e)}")
            print(f"❌ 打开独立摘要图窗口失败: {e}")
    
    def open_independent_dependence_window(self):
        """打开独立的依赖图窗口"""
        if self.shap_values is None:
            QMessageBox.warning(self, "警告", "请先计算SHAP值")
            return
        
        try:
            main_feature = self.main_feature_combo.currentText()
            interaction_feature = self.interaction_feature_combo.currentText()
            
            window = SHAPDependenceWindow(
                self.shap_values, self.explain_data, self.feature_names,
                main_feature, interaction_feature, self
            )
            window.show()
            print(f"✅ 独立依赖图窗口已打开: {main_feature} vs {interaction_feature}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开独立窗口失败:\n{str(e)}")
            print(f"❌ 打开独立依赖图窗口失败: {e}")
    
    def open_independent_local_window(self):
        """打开独立的局部解释窗口"""
        if self.shap_values is None:
            QMessageBox.warning(self, "警告", "请先计算SHAP值")
            return
        
        try:
            sample_idx = self.sample_index_spin.value()
            plot_type = self.local_plot_combo.currentText()
            
            window = SHAPLocalWindow(
                self.shap_values, self.explainer, self.explain_data,
                self.feature_names, sample_idx, plot_type, self
            )
            window.show()
            print(f"✅ 独立局部解释窗口已打开: {plot_type}, 样本 {sample_idx}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开独立窗口失败:\n{str(e)}")
            print(f"❌ 打开独立局部解释窗口失败: {e}")