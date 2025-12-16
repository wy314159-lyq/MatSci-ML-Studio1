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


# ============================================================================
# Configuration Constants - Extracted magic numbers for better maintainability
# ============================================================================
class SHAPConfig:
    """Configuration constants for SHAP analysis"""
    # Background data sampling
    DEFAULT_BACKGROUND_SAMPLE_SIZE = 100
    MIN_BACKGROUND_SAMPLE_SIZE = 10
    ULTRA_CONSERVATIVE_BACKGROUND_SIZE = 10

    # Explanation data sampling
    DEFAULT_EXPLAIN_SAMPLE_SIZE = 50
    MAX_EXPLAIN_SAMPLES = 100

    # Visualization settings
    DEFAULT_MAX_DISPLAY_FEATURES = 20
    MAX_WATERFALL_FEATURES = 15
    MAX_FORCE_PLOT_FEATURES = 10
    MAX_BEESWARM_FEATURES = 12

    # Image export settings
    DEFAULT_DPI = 100
    EXPORT_DPI = 300
    HIGH_QUALITY_DPI = 150

    # Fallback settings
    FALLBACK_SAMPLE_SIZE = 10

    # Thread safety
    WORKER_TIMEOUT_MS = 600000  # 10 minutes

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
    SHAPLocalWindow,
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
            
            # Return complete result dictionary with multi-class support
            result = {
                'shap_values': shap_values,
                'explainer': explainer,
                'explain_data': self.explain_data,
                'feature_names': self.feature_names,
                # Multi-class support
                'all_class_shap_values': getattr(self, '_all_class_shap_values', None),
                'n_classes': getattr(self, '_n_classes', 1),
                'selected_class': getattr(self, '_selected_class', 0)
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
                
                # Return complete result dictionary with multi-class support
                result = {
                    'shap_values': shap_values,
                    'explainer': explainer,
                    'explain_data': self.explain_data,
                    'feature_names': self.feature_names,
                    # Multi-class support (fallback may not have this info)
                    'all_class_shap_values': getattr(self, '_all_class_shap_values', None),
                    'n_classes': getattr(self, '_n_classes', 1),
                    'selected_class': getattr(self, '_selected_class', 0)
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
                    except Exception as pred_error:
                        print(f"Prediction wrapper error: {pred_error}")
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
            
            # Prepare background data - convert to DataFrame first for safe sampling
            bg_source = self.background_data
            if not isinstance(bg_source, pd.DataFrame):
                cols = self.feature_names or [f"feature_{i}" for i in range(bg_source.shape[1])]
                bg_source = pd.DataFrame(bg_source, columns=cols)

            background_sample_size = min(SHAPConfig.DEFAULT_BACKGROUND_SAMPLE_SIZE, len(bg_source))
            if len(bg_source) > background_sample_size:
                background_sample = bg_source.sample(background_sample_size, random_state=42)
            else:
                background_sample = bg_source.copy()
            
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
                minimal_background = background_sample.values[:SHAPConfig.ULTRA_CONSERVATIVE_BACKGROUND_SIZE]
                
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
            explain_source = self.explain_data
            if not isinstance(explain_source, pd.DataFrame):
                cols = self.feature_names or [f"feature_{i}" for i in range(explain_source.shape[1])]
                explain_source = pd.DataFrame(explain_source, columns=cols)

            # 保持 DataFrame 供后续 UI 使用
            self.explain_data = explain_source
            explain_sample_size = min(SHAPConfig.DEFAULT_EXPLAIN_SAMPLE_SIZE, len(explain_source))
            if len(explain_source) > explain_sample_size:
                # Random采样，减少顺序偏差
                explain_sample = explain_source.sample(explain_sample_size, random_state=42)
            else:
                explain_sample = explain_source.copy()
            
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
                        except Exception as fallback_pred_error:
                            print(f"Fallback prediction error: {fallback_pred_error}")
                            return np.zeros(X.shape[0] if hasattr(X, 'shape') else 1)

                    bg_source = self.background_data
                    if not isinstance(bg_source, pd.DataFrame):
                        cols = self.feature_names or [f"feature_{i}" for i in range(bg_source.shape[1])]
                        bg_source = pd.DataFrame(bg_source, columns=cols)
                    fallback_sample_size = min(SHAPConfig.DEFAULT_EXPLAIN_SAMPLE_SIZE, len(bg_source))
                    fallback_explainer = shap.KernelExplainer(fallback_predict_wrapper, bg_source.sample(fallback_sample_size).values)
                    shap_values = fallback_explainer.shap_values(explain_sample.values)
            
            # Handle different SHAP value formats with multi-class support
            all_class_shap_values = None
            n_classes = 1
            selected_class = 0

            if isinstance(shap_values, list):
                # Multi-class classification returns list of arrays
                n_classes = len(shap_values)
                all_class_shap_values = shap_values  # Store all classes
                if n_classes == 2:
                    # Binary classification - use positive class by default
                    selected_class = 1
                    shap_values = shap_values[1]
                    print(f"Binary classification detected. Using positive class (class 1) SHAP values")
                else:
                    # Multi-class - use first class by default, but store all
                    selected_class = 0
                    shap_values = shap_values[0]
                    print(f"Multi-class classification detected ({n_classes} classes). Using class 0 SHAP values by default")
                    print(f"Note: All class SHAP values are stored. Use class selector to view other classes.")

            # Ensure 2D array format
            if len(shap_values.shape) == 3:
                n_classes = shap_values.shape[2]
                all_class_shap_values = [shap_values[:, :, i] for i in range(n_classes)]
                if n_classes == 2:
                    selected_class = 1
                    shap_values = shap_values[:, :, 1]
                    print(f"Converted 3D SHAP values to 2D (positive class)")
                else:
                    selected_class = 0
                    shap_values = shap_values[:, :, 0]
                    print(f"Converted 3D SHAP values to 2D (class 0 of {n_classes})")

            print(f"Final SHAP values shape: {shap_values.shape}")

            # Store multi-class info in result
            self._all_class_shap_values = all_class_shap_values
            self._n_classes = n_classes
            self._selected_class = selected_class

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

        # Multi-class support attributes
        self.all_class_shap_values = None
        self.n_classes = 1
        self.selected_class = 0
        
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
        """Create results panel with tabs + class selector"""
        container = QWidget()
        container_layout = QVBoxLayout(container)

        # Class selector for多分类
        class_layout = QHBoxLayout()
        self.class_selector_label = QLabel("Class:")
        self.class_selector_label.setVisible(False)
        class_layout.addWidget(self.class_selector_label)

        self.class_selector = QComboBox()
        self.class_selector.setVisible(False)
        self.class_selector.currentIndexChanged.connect(self.on_class_changed)
        class_layout.addWidget(self.class_selector)

        class_layout.addStretch()
        container_layout.addLayout(class_layout)

        self.tabs = QTabWidget()
        
        # Global Explanations tab
        self.create_global_explanations_tab()
        
        # Local Explanations tab
        self.create_local_explanations_tab()
        
        # Dependence Plots tab
        self.create_dependence_plots_tab()

        # Advanced Analysis tab (新增)
        self.create_advanced_analysis_tab()

        # Statistics Summary tab (新增)
        self.create_statistics_tab()

        container_layout.addWidget(self.tabs)
        return container
        
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
        
        # Add independent window button
        self.open_summary_window_btn = QPushButton("🔗 Open Independent Window")
        self.open_summary_window_btn.clicked.connect(self.open_independent_summary_window)
        self.open_summary_window_btn.setToolTip("Display SHAP summary plot in independent window")
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
        
        # Add independent window button
        self.open_local_window_btn = QPushButton("🔗 Open Independent Window")
        self.open_local_window_btn.clicked.connect(self.open_independent_local_window)
        self.open_local_window_btn.setToolTip("Display SHAP local explanation plot in independent window")
        controls_layout.addWidget(self.open_local_window_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        self.local_sample_info_text = QTextEdit()
        self.local_sample_info_text.setMaximumHeight(100)
        self.local_sample_info_text.setReadOnly(True)
        layout.addWidget(self.local_sample_info_text)

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
        
        # Add independent window button
        self.open_dependence_window_btn = QPushButton("🔗 Open Independent Window")
        self.open_dependence_window_btn.clicked.connect(self.open_independent_dependence_window)
        self.open_dependence_window_btn.setToolTip("Display SHAP dependence plot in independent window")
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

    def create_advanced_analysis_tab(self):
        """Create advanced analysis tab with multiple visualization types"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Analysis Type:"))
        self.advanced_plot_combo = QComboBox()
        self.advanced_plot_combo.addItems([
            "Interaction Plot",
            "SHAP Heatmap",
            "Decision Plot",
            "Clustering Analysis",
            "Multi-Sample Comparison"
        ])
        self.advanced_plot_combo.currentTextChanged.connect(self.update_advanced_plot)
        controls_layout.addWidget(self.advanced_plot_combo)

        # Feature selection for interaction plot
        controls_layout.addWidget(QLabel("Feature 1:"))
        self.adv_feature1_combo = QComboBox()
        self.adv_feature1_combo.currentTextChanged.connect(self.update_advanced_plot)
        controls_layout.addWidget(self.adv_feature1_combo)

        controls_layout.addWidget(QLabel("Feature 2:"))
        self.adv_feature2_combo = QComboBox()
        self.adv_feature2_combo.currentTextChanged.connect(self.update_advanced_plot)
        controls_layout.addWidget(self.adv_feature2_combo)

        # Sample range for heatmap/decision plot
        controls_layout.addWidget(QLabel("Samples:"))
        self.adv_sample_start_spin = QSpinBox()
        self.adv_sample_start_spin.setRange(0, 0)
        self.adv_sample_start_spin.setValue(0)
        controls_layout.addWidget(self.adv_sample_start_spin)

        controls_layout.addWidget(QLabel("to"))
        self.adv_sample_end_spin = QSpinBox()
        self.adv_sample_end_spin.setRange(0, 50)
        self.adv_sample_end_spin.setValue(20)
        controls_layout.addWidget(self.adv_sample_end_spin)

        # Open independent window button
        self.open_advanced_window_btn = QPushButton("🔗 Open Independent Window")
        self.open_advanced_window_btn.clicked.connect(self.open_independent_advanced_window)
        controls_layout.addWidget(self.open_advanced_window_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create matplotlib figure and canvas
        self.advanced_figure = Figure(figsize=(12, 8))
        self.advanced_canvas = FigureCanvas(self.advanced_figure)

        # Add navigation toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.advanced_toolbar = NavigationToolbar(self.advanced_canvas, tab)

        layout.addWidget(self.advanced_toolbar)
        layout.addWidget(self.advanced_canvas)

        self.tabs.addTab(tab, "Advanced Analysis")

    def create_statistics_tab(self):
        """Create statistics summary tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Title
        title = QLabel("📊 SHAP Statistics Summary")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Statistics display area
        stats_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Feature importance ranking
        left_group = QGroupBox("Feature Importance Ranking")
        left_layout = QVBoxLayout(left_group)
        self.feature_ranking_text = QTextEdit()
        self.feature_ranking_text.setReadOnly(True)
        self.feature_ranking_text.setFont(QFont("Consolas", 10))
        left_layout.addWidget(self.feature_ranking_text)
        stats_splitter.addWidget(left_group)

        # Middle panel - SHAP statistics
        middle_group = QGroupBox("SHAP Value Statistics")
        middle_layout = QVBoxLayout(middle_group)
        self.shap_stats_text = QTextEdit()
        self.shap_stats_text.setReadOnly(True)
        self.shap_stats_text.setFont(QFont("Consolas", 10))
        middle_layout.addWidget(self.shap_stats_text)
        stats_splitter.addWidget(middle_group)

        # Right panel - Feature group analysis
        right_group = QGroupBox("Feature Group Analysis")
        right_layout = QVBoxLayout(right_group)

        # Feature grouping controls
        group_controls = QHBoxLayout()
        group_controls.addWidget(QLabel("Group by prefix:"))
        self.group_prefix_combo = QComboBox()
        self.group_prefix_combo.setEditable(True)
        self.group_prefix_combo.addItems(["Auto", "element_", "structure_", "composition_"])
        self.group_prefix_combo.currentTextChanged.connect(self.update_feature_group_analysis)
        group_controls.addWidget(self.group_prefix_combo)
        right_layout.addLayout(group_controls)

        self.feature_group_text = QTextEdit()
        self.feature_group_text.setReadOnly(True)
        self.feature_group_text.setFont(QFont("Consolas", 10))
        right_layout.addWidget(self.feature_group_text)
        stats_splitter.addWidget(right_group)

        layout.addWidget(stats_splitter)

        # Bottom panel - Visualization
        bottom_layout = QHBoxLayout()

        # Feature importance bar chart
        self.stats_figure = Figure(figsize=(8, 4))
        self.stats_canvas = FigureCanvas(self.stats_figure)
        bottom_layout.addWidget(self.stats_canvas)

        # Export buttons
        export_layout = QVBoxLayout()
        self.export_stats_btn = QPushButton("📄 Export Statistics Report")
        self.export_stats_btn.clicked.connect(self.export_statistics_report)
        export_layout.addWidget(self.export_stats_btn)

        self.refresh_stats_btn = QPushButton("🔄 Refresh Statistics")
        self.refresh_stats_btn.clicked.connect(self.update_statistics_tab)
        export_layout.addWidget(self.refresh_stats_btn)

        export_layout.addStretch()
        bottom_layout.addLayout(export_layout)

        layout.addLayout(bottom_layout)

        self.tabs.addTab(tab, "Statistics Summary")

    def update_feature_group_analysis(self):
        """Update feature group analysis based on selected prefix"""
        if self.shap_values is None or self.feature_names is None:
            self.feature_group_text.setText("No SHAP values available. Please calculate SHAP values first.")
            return

        prefix = self.group_prefix_combo.currentText()

        try:
            import numpy as np

            # Get SHAP values as numpy array
            if hasattr(self.shap_values, 'values'):
                shap_vals = self.shap_values.values
            else:
                shap_vals = np.array(self.shap_values)

            # Calculate mean absolute SHAP values per feature
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)

            # Group features by prefix
            groups = {}
            if prefix == "Auto":
                # Auto-detect prefixes
                for i, name in enumerate(self.feature_names):
                    parts = name.split('_')
                    group_name = parts[0] if len(parts) > 1 else "other"
                    if group_name not in groups:
                        groups[group_name] = []
                    groups[group_name].append((name, mean_abs_shap[i]))
            else:
                # Use specified prefix
                matched = []
                unmatched = []
                for i, name in enumerate(self.feature_names):
                    if name.startswith(prefix):
                        matched.append((name, mean_abs_shap[i]))
                    else:
                        unmatched.append((name, mean_abs_shap[i]))
                if matched:
                    groups[prefix.rstrip('_')] = matched
                if unmatched:
                    groups["other"] = unmatched

            # Generate report
            report = []
            report.append("=" * 50)
            report.append("Feature Group Analysis")
            report.append("=" * 50)

            group_importance = []
            for group_name, features in groups.items():
                total_importance = sum(f[1] for f in features)
                group_importance.append((group_name, total_importance, len(features), features))

            # Sort by total importance
            group_importance.sort(key=lambda x: x[1], reverse=True)

            for group_name, total_imp, count, features in group_importance:
                report.append(f"\n📁 {group_name.upper()} ({count} features)")
                report.append(f"   Total Importance: {total_imp:.4f}")
                report.append(f"   Average Importance: {total_imp/count:.4f}")
                report.append("   Top features:")
                # Show top 3 features in group
                sorted_features = sorted(features, key=lambda x: x[1], reverse=True)[:3]
                for fname, imp in sorted_features:
                    report.append(f"     - {fname}: {imp:.4f}")

            self.feature_group_text.setText("\n".join(report))

        except Exception as e:
            self.feature_group_text.setText(f"Error analyzing feature groups: {str(e)}")

    def update_statistics_tab(self):
        """Update all statistics displays"""
        if self.shap_values is None or self.feature_names is None:
            self.feature_ranking_text.setText("No SHAP values available.")
            self.shap_stats_text.setText("Please calculate SHAP values first.")
            return

        try:
            import numpy as np

            # Get SHAP values as numpy array
            if hasattr(self.shap_values, 'values'):
                shap_vals = self.shap_values.values
            else:
                shap_vals = np.array(self.shap_values)

            # Calculate statistics
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)

            # Feature ranking
            ranking = sorted(zip(self.feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)

            ranking_text = ["=" * 40, "Feature Importance Ranking", "=" * 40, ""]
            for i, (name, importance) in enumerate(ranking, 1):
                ranking_text.append(f"{i:3d}. {name}: {importance:.4f}")
            self.feature_ranking_text.setText("\n".join(ranking_text))

            # SHAP statistics
            stats_text = ["=" * 40, "SHAP Value Statistics", "=" * 40, ""]
            stats_text.append(f"Number of samples: {shap_vals.shape[0]}")
            stats_text.append(f"Number of features: {shap_vals.shape[1]}")
            stats_text.append(f"\nOverall Statistics:")
            stats_text.append(f"  Mean SHAP: {shap_vals.mean():.4f}")
            stats_text.append(f"  Std SHAP: {shap_vals.std():.4f}")
            stats_text.append(f"  Min SHAP: {shap_vals.min():.4f}")
            stats_text.append(f"  Max SHAP: {shap_vals.max():.4f}")
            stats_text.append(f"\nMean |SHAP| per feature:")
            stats_text.append(f"  Mean: {mean_abs_shap.mean():.4f}")
            stats_text.append(f"  Std: {mean_abs_shap.std():.4f}")
            self.shap_stats_text.setText("\n".join(stats_text))

            # Update feature group analysis
            self.update_feature_group_analysis()

            # Update visualization
            self._update_stats_figure(ranking[:20])

        except Exception as e:
            self.feature_ranking_text.setText(f"Error: {str(e)}")
            self.shap_stats_text.setText(f"Error updating statistics: {str(e)}")

    def _update_stats_figure(self, ranking):
        """Update the statistics figure with feature importance bar chart"""
        try:
            self.stats_figure.clear()
            ax = self.stats_figure.add_subplot(111)

            names = [r[0] for r in ranking]
            values = [r[1] for r in ranking]

            # Truncate long names
            names = [n[:20] + '...' if len(n) > 20 else n for n in names]

            y_pos = range(len(names))
            ax.barh(y_pos, values, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Top 20 Feature Importance')

            self.stats_figure.tight_layout()
            self.stats_canvas.draw()
        except Exception as e:
            print(f"Error updating stats figure: {e}")

    def export_statistics_report(self):
        """Export statistics report to file"""
        if self.shap_values is None:
            QMessageBox.warning(self, "Warning", "No SHAP values available to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics Report", "",
            "Text files (*.txt);;CSV files (*.csv);;All files (*)"
        )

        if not file_path:
            return

        try:
            import numpy as np

            # Get SHAP values
            if hasattr(self.shap_values, 'values'):
                shap_vals = self.shap_values.values
            else:
                shap_vals = np.array(self.shap_values)

            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            ranking = sorted(zip(self.feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)

            if file_path.endswith('.csv'):
                # Export as CSV
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Rank,Feature,Mean_Abs_SHAP\n")
                    for i, (name, importance) in enumerate(ranking, 1):
                        f.write(f"{i},{name},{importance:.6f}\n")
            else:
                # Export as text
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("SHAP Analysis Statistics Report\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Number of samples: {shap_vals.shape[0]}\n")
                    f.write(f"Number of features: {shap_vals.shape[1]}\n\n")
                    f.write("Feature Importance Ranking:\n")
                    f.write("-" * 40 + "\n")
                    for i, (name, importance) in enumerate(ranking, 1):
                        f.write(f"{i:3d}. {name}: {importance:.6f}\n")

            QMessageBox.information(self, "Success", f"Report exported to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export report:\n{str(e)}")

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
        """Calculate SHAP values with thread safety"""
        if not self._validate_inputs():
            return

        # Thread safety: Check if a worker is already running
        if hasattr(self, 'worker') and self.worker is not None:
            if self.worker.isRunning():
                QMessageBox.warning(self, "Warning",
                    "SHAP calculation is already in progress. Please wait for it to complete.")
                return
            # Clean up finished worker
            try:
                self.worker.deleteLater()
            except Exception:
                pass

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

            # Extract multi-class information
            self.all_class_shap_values = result.get('all_class_shap_values')
            self.n_classes = result.get('n_classes', 1)
            self.selected_class = result.get('selected_class', 0)

            print(f"🔍 SHAP values extracted: {self.shap_values is not None}")
            print(f"🔍 Multi-class info: n_classes={self.n_classes}, selected_class={self.selected_class}")
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
            # 初始/刷新多分类选择器
            self._refresh_class_selector()

            # Update UI components
            print(f"\U0001f50d Updating UI components...")
            self._update_feature_selections()

            # Update sample index range - calculate max_samples first
            max_samples = min(len(self.shap_values), len(self.explain_data))
            self.sample_index_spin.setMaximum(max_samples - 1)
            print(f"🔍 Sample index range set to 0-{max_samples-1}")

            # Sync advanced sample ranges
            if hasattr(self, "adv_sample_start_spin"):
                self.adv_sample_start_spin.setMaximum(max_samples - 1)
            if hasattr(self, "adv_sample_end_spin"):
                self.adv_sample_end_spin.setMaximum(max_samples - 1)
                if self.adv_sample_end_spin.value() >= max_samples:
                    self.adv_sample_end_spin.setValue(max_samples - 1)
            
            # Enable UI elements
            self.main_feature_combo.setEnabled(True)
            self.interaction_feature_combo.setEnabled(True)
            self.sample_index_spin.setEnabled(True)
            self.local_plot_combo.setEnabled(True)
            
            # Update summary plot immediately
            print(f"🔍 Updating summary plot...")
            try:
                self.update_summary_plot()
            except Exception as summary_error:
                print(f"❌ Summary plot update failed: {summary_error}")

            # Update other plots with better error handling
            print(f"🔍 Triggering other plot updates...")
            try:
                self.update_dependence_plot()
            except Exception as dep_error:
                print(f"❌ Dependence plot update failed: {dep_error}")

            try:
                self.update_local_plot()
            except Exception as local_error:
                print(f"❌ Local plot update failed: {local_error}")

            # Add force update method call
            QApplication.processEvents()
            self.force_update_all_plots()
            
            print(f"✅ SHAP calculation completed successfully!")
            QMessageBox.information(self, "Success", "SHAP analysis completed successfully!")
            
            # Update UI state
            self.progress_bar.setVisible(False)
            self.calculate_btn.setEnabled(True)
            self.export_values_btn.setEnabled(True)
            self.export_plots_btn.setEnabled(True)
            
            # Switch to the first results tab to show results
            self.tabs.setCurrentIndex(0)
            
            self.status_label.setText("SHAP analysis completed! Check visualization tabs.")
            
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
        
    def on_class_changed(self, index):
        """切换多分类可视化的类"""
        if self.all_class_shap_values is None:
            return
        if index < 0 or index >= len(self.all_class_shap_values):
            return
        try:
            self.selected_class = index
            self.shap_values = self.all_class_shap_values[index]
            self.status_label.setText(f"Class {index} selected for SHAP visualization")
            self.update_summary_plot()
            self.update_local_plot()
            self.update_dependence_plot()
            self.update_advanced_plot()
        except Exception as e:
            print(f"Class change failed: {e}")

    def update_local_plot(self):
        """Update local SHAP plot using native SHAP visualizations"""
        print(f"🔍 === LOCAL PLOT UPDATE START ===")
        
        if self.shap_values is None:
            print(f"❌ No SHAP values available")
            self._show_message_on_figure(self.local_figure, "Please calculate SHAP values first")
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
                self._show_message_on_figure(self.local_figure, f"Sample index {sample_idx} out of range")
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
            
            self.local_sample_info_text.setText(info_text)

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
            self._show_message_on_figure(self.global_figure, "Please calculate SHAP values first")
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
        """Create native SHAP plot with improved reliability and display"""
        try:
            print(f"🔍 Creating native {plot_name}...")

            # Clear target figure completely
            target_figure.clear()

            # Create temporary figure for SHAP plot with better settings
            import matplotlib.pyplot as plt
            temp_fig = plt.figure(figsize=(12, 8), facecolor='white')

            # Execute SHAP plotting function with error handling
            print(f"🔍 Executing SHAP plotting function...")
            try:
                shap_plot_func()
            except Exception as plot_error:
                print(f"❌ SHAP plot function failed: {plot_error}")
                plt.close(temp_fig)
                return False

            # Save to memory buffer with optimized settings
            import io
            buf = io.BytesIO()
            try:
                temp_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                               facecolor='white', edgecolor='none', pad_inches=0.2)
                buf.seek(0)
            except Exception as save_error:
                print(f"❌ Figure save failed: {save_error}")
                plt.close(temp_fig)
                return False

            # Close temporary figure immediately
            plt.close(temp_fig)

            # Load image and display in target figure
            try:
                from PIL import Image
                import numpy as np

                img = Image.open(buf)
                img_array = np.array(img)
                print(f"🔍 Image loaded successfully, shape: {img_array.shape}")

                # Create subplot in target figure with proper positioning
                ax = target_figure.add_subplot(111)
                ax.imshow(img_array, aspect='equal', interpolation='bilinear')
                ax.axis('off')

                # Optimize layout for better display
                target_figure.tight_layout()
                target_figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

            except Exception as display_error:
                print(f"❌ Image display failed: {display_error}")
                return False

            # Improved canvas refresh logic
            canvas_to_refresh = None
            if target_figure == self.global_figure:
                canvas_to_refresh = self.global_canvas
            elif target_figure == self.local_figure:
                canvas_to_refresh = self.local_canvas
            elif target_figure == self.dependence_figure:
                canvas_to_refresh = self.dependence_canvas

            if canvas_to_refresh:
                # Simplified but effective refresh sequence
                canvas_to_refresh.draw()
                QApplication.processEvents()
                canvas_to_refresh.flush_events()
                print(f"✅ Canvas refreshed for {plot_name}")

            print(f"✅ Native {plot_name} created successfully")
            return True

        except Exception as e:
            print(f"❌ Native {plot_name} creation failed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return False

    def _create_fallback_summary_plot(self, max_display):
        """Enhanced fallback summary plot with beeswarm-like visualization"""
        try:
            print(f"🔍 Creating enhanced fallback summary plot...")

            self.global_figure.clear()

            # Calculate feature importance
            feature_importance = np.mean(np.abs(self.shap_values), axis=0)

            # Sort and select top features
            sorted_indices = np.argsort(feature_importance)[-max_display:][::-1]
            n_features = len(sorted_indices)

            # Get plot type
            plot_type = self.global_plot_combo.currentText()

            if plot_type == "Feature Importance":
                # Bar plot for feature importance
                ax = self.global_figure.add_subplot(111)
                sorted_importance = feature_importance[sorted_indices]
                sorted_names = [self.feature_names[i] for i in sorted_indices]

                # Create gradient colors based on importance
                colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, n_features))

                y_pos = np.arange(n_features)
                bars = ax.barh(y_pos, sorted_importance, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

                ax.set_yticks(y_pos)
                ax.set_yticklabels(sorted_names, fontsize=9)
                ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
                ax.set_title('SHAP Feature Importance', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
                ax.invert_yaxis()

                # Add value labels
                max_val = sorted_importance.max()
                for bar, val in zip(bars, sorted_importance):
                    width = bar.get_width()
                    ax.text(width + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
                           f'{val:.4f}', ha='left', va='center', fontsize=8)

            else:
                # Beeswarm-like plot for Summary Plot
                ax = self.global_figure.add_subplot(111)
                sorted_names = [self.feature_names[i] for i in sorted_indices]

                # Prepare data for beeswarm
                for i, feat_idx in enumerate(sorted_indices):
                    shap_vals = self.shap_values[:, feat_idx]
                    feat_vals = self.explain_data.iloc[:, feat_idx].values if isinstance(self.explain_data, pd.DataFrame) else self.explain_data[:, feat_idx]

                    # Normalize feature values for coloring
                    if hasattr(feat_vals, 'dtype') and feat_vals.dtype == bool:
                        feat_vals = feat_vals.astype(float)

                    feat_min, feat_max = np.nanmin(feat_vals), np.nanmax(feat_vals)
                    if feat_max > feat_min:
                        feat_normalized = (feat_vals - feat_min) / (feat_max - feat_min)
                    else:
                        feat_normalized = np.zeros_like(feat_vals)

                    # Add jitter for y-axis
                    y_jitter = np.random.normal(0, 0.1, len(shap_vals))
                    y_positions = i + y_jitter

                    # Scatter plot with color based on feature value
                    scatter = ax.scatter(shap_vals, y_positions, c=feat_normalized,
                                        cmap='coolwarm', alpha=0.6, s=15, edgecolors='none')

                # Add colorbar
                cbar = self.global_figure.colorbar(scatter, ax=ax, shrink=0.6, aspect=30)
                cbar.set_label('Feature Value\n(normalized)', fontsize=9)

                ax.set_yticks(range(n_features))
                ax.set_yticklabels(sorted_names, fontsize=9)
                ax.set_xlabel('SHAP Value (impact on model output)', fontsize=11)
                ax.set_title('SHAP Summary Plot', fontsize=12, fontweight='bold')
                ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')

            self.global_figure.tight_layout()
            self.global_canvas.draw()
            QApplication.processEvents()

            print(f"✅ Enhanced fallback summary plot created")

        except Exception as e:
            print(f"❌ Enhanced fallback summary plot failed: {e}")
            import traceback
            traceback.print_exc()
            self._show_message_on_figure(self.global_figure, f"Plot failed: {str(e)}")
      
    def _create_native_waterfall_plot(self, sample_idx, explain_data_df):
        """Create native SHAP waterfall plot with improved compatibility"""
        try:
            print(f"🔍 Creating native waterfall plot for sample {sample_idx}")

            def create_waterfall():
                # Try multiple approaches for waterfall plot
                base_val = getattr(self, 'base_value', 0)
                shap_vals = self.shap_values[sample_idx]
                sample_data = explain_data_df.iloc[sample_idx]

                # Method 1: Modern SHAP waterfall with Explanation object
                if hasattr(shap, 'plots') and hasattr(shap.plots, 'waterfall'):
                    print(f"🔍 Trying modern SHAP waterfall plot")
                    try:
                        shap_explanation = shap.Explanation(
                            values=shap_vals,
                            base_values=base_val,
                            data=sample_data.values,
                            feature_names=self.feature_names
                        )
                        shap.plots.waterfall(shap_explanation, show=False, max_display=15)
                        return
                    except Exception as modern_error:
                        print(f"🔄 Modern waterfall failed: {modern_error}")

                # Method 2: Legacy waterfall plot
                if hasattr(shap, 'waterfall_plot'):
                    print(f"🔍 Trying legacy SHAP waterfall plot")
                    try:
                        shap.waterfall_plot(
                            base_val,
                            shap_vals,
                            sample_data,
                            feature_names=self.feature_names,
                            max_display=15,
                            show=False
                        )
                        return
                    except Exception as legacy_error:
                        print(f"🔄 Legacy waterfall failed: {legacy_error}")

                # Method 3: Manual waterfall using matplotlib directly
                print(f"🔍 Creating manual waterfall plot")
                self._create_manual_waterfall_in_temp_fig(shap_vals, sample_data, base_val)

            return self._create_native_shap_plot(
                create_waterfall,
                self.local_figure,
                "Waterfall Plot"
            )

        except Exception as e:
            print(f"❌ Native waterfall plot creation failed: {e}")
            return False
    
    def _create_native_force_plot(self, sample_idx, explain_data_df):
        """Create native SHAP force plot with improved compatibility"""
        try:
            print(f"🔍 Creating native force plot for sample {sample_idx}")

            def create_force():
                base_val = getattr(self, 'base_value', 0)
                shap_vals = self.shap_values[sample_idx]
                sample_data = explain_data_df.iloc[sample_idx]

                # Method 1: Traditional force plot with matplotlib
                if hasattr(shap, 'force_plot'):
                    print(f"🔍 Trying traditional SHAP force plot")
                    try:
                        shap.force_plot(
                            base_val,
                            shap_vals,
                            sample_data,
                            feature_names=self.feature_names,
                            matplotlib=True,
                            show=False
                        )
                        return
                    except Exception as force_error:
                        print(f"🔄 Traditional force plot failed: {force_error}")

                # Method 2: Modern force plot
                if hasattr(shap, 'plots') and hasattr(shap.plots, 'force'):
                    print(f"🔍 Trying modern SHAP force plot")
                    try:
                        shap_explanation = shap.Explanation(
                            values=shap_vals,
                            base_values=base_val,
                            data=sample_data.values,
                            feature_names=self.feature_names
                        )
                        shap.plots.force(shap_explanation, matplotlib=True, show=False)
                        return
                    except Exception as modern_error:
                        print(f"🔄 Modern force plot failed: {modern_error}")

                # Method 3: Manual force plot
                print(f"🔍 Creating manual force plot")
                self._create_manual_force_in_temp_fig(shap_vals, sample_data, base_val)

            return self._create_native_shap_plot(
                create_force,
                self.local_figure,
                "Force Plot"
            )

        except Exception as e:
            print(f"❌ Native force plot creation failed: {e}")
            return False
    
    def _create_fallback_local_plot(self, sample_idx, plot_type):
        """Fallback local plot when native SHAP fails"""
        try:
            print(f"🔍 Creating fallback {plot_type}...")

            self.local_figure.clear()
            ax = self.local_figure.add_subplot(111)

            # Get SHAP values for this sample with bounds checking
            if sample_idx >= len(self.shap_values):
                sample_idx = len(self.shap_values) - 1
                print(f"⚠️ Sample index adjusted to {sample_idx}")

            sample_shap_vals = self.shap_values[sample_idx]

            # Get sample data with bounds checking
            if sample_idx >= len(self.explain_data):
                sample_data_idx = len(self.explain_data) - 1
            else:
                sample_data_idx = sample_idx

            sample_data = self.explain_data.iloc[sample_data_idx]

            print(f"🔍 Processing sample {sample_idx}, SHAP shape: {sample_shap_vals.shape}")

            if plot_type == "Waterfall Plot":
                self._create_simple_waterfall(ax, sample_shap_vals, sample_idx)
            elif plot_type == "Force Plot":
                self._create_simple_force_plot(ax, sample_shap_vals, sample_data, sample_idx)
            else:
                self._create_simple_waterfall(ax, sample_shap_vals, sample_idx)

            # Enhanced canvas refresh with proper error handling
            try:
                self.local_figure.tight_layout()
                self.local_canvas.draw()
                QApplication.processEvents()
                self.local_canvas.flush_events()
                print(f"✅ Fallback {plot_type} created and displayed")
            except Exception as refresh_error:
                print(f"❌ Canvas refresh failed: {refresh_error}")
                # Try simple refresh
                try:
                    self.local_canvas.draw()
                except:
                    pass

        except Exception as e:
            print(f"❌ Fallback {plot_type} failed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            self._show_message_on_figure(self.local_figure, f"Local plot creation failed: {str(e)[:100]}")
    
    def _create_simple_waterfall(self, ax, shap_vals, sample_idx):
        """Create an enhanced waterfall plot with cumulative effect visualization"""
        try:
            print(f"🔍 Creating enhanced waterfall plot for sample {sample_idx}")

            # Validate inputs
            if shap_vals is None or len(shap_vals) == 0:
                ax.text(0.5, 0.5, 'No SHAP values available', ha='center', va='center', transform=ax.transAxes)
                return

            if not self.feature_names or len(self.feature_names) != len(shap_vals):
                feature_names = [f'Feature_{i}' for i in range(len(shap_vals))]
            else:
                feature_names = self.feature_names

            # Sort features by absolute SHAP value
            indices_and_vals = [(i, abs(shap_vals[i]), shap_vals[i]) for i in range(len(shap_vals))]
            indices_and_vals.sort(key=lambda x: x[1], reverse=True)

            # Take top features for cleaner visualization
            n_features = min(12, len(indices_and_vals))
            top_features = indices_and_vals[:n_features]

            # Extract data for plotting
            indices = [item[0] for item in top_features]
            values = [item[2] for item in top_features]
            names = [feature_names[i][:25] + '...' if len(feature_names[i]) > 25 else feature_names[i] for i in indices]

            # Get base value and calculate prediction
            base_value = getattr(self, 'base_value', 0)
            prediction = base_value + sum(shap_vals)

            # Create waterfall-style visualization
            y_pos = np.arange(len(values))

            # Color gradient based on value magnitude
            max_abs = max(abs(v) for v in values) if values else 1
            colors = []
            for val in values:
                if val < 0:
                    intensity = min(abs(val) / max_abs, 1.0)
                    colors.append(plt.cm.Reds(0.3 + 0.5 * intensity))
                else:
                    intensity = min(abs(val) / max_abs, 1.0)
                    colors.append(plt.cm.Blues(0.3 + 0.5 * intensity))

            bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5, height=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel('SHAP Value (contribution to prediction)', fontsize=10)
            ax.set_title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if abs(val) > 0.0001:
                    width = bar.get_width()
                    offset = max_abs * 0.03
                    label_x = width + offset if width >= 0 else width - offset
                    ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                           f'{val:+.4f}', ha='left' if width >= 0 else 'right', va='center', fontsize=8,
                           fontweight='bold' if abs(val) > max_abs * 0.5 else 'normal')

            # Add base value and prediction annotation
            ax.annotate(f'Base: {base_value:.4f}\nPrediction: {prediction:.4f}',
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       fontsize=9, ha='right', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=plt.cm.Blues(0.6), label='Positive impact'),
                             Patch(facecolor=plt.cm.Reds(0.6), label='Negative impact')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

            print(f"✅ Enhanced waterfall plot created with {n_features} features")

        except Exception as e:
            print(f"❌ Enhanced waterfall creation failed: {e}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, f'Waterfall plot failed: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
    
    def _create_simple_force_plot(self, ax, shap_vals, sample_data, sample_idx):
        """Create an enhanced force plot visualization showing cumulative contributions"""
        try:
            print(f"🔍 Creating enhanced force plot for sample {sample_idx}")

            # Validate inputs
            if shap_vals is None or len(shap_vals) == 0:
                ax.text(0.5, 0.5, 'No SHAP values available', ha='center', va='center', transform=ax.transAxes)
                return

            if not self.feature_names or len(self.feature_names) != len(shap_vals):
                feature_names = [f'Feature_{i}' for i in range(len(shap_vals))]
            else:
                feature_names = self.feature_names

            # Get base value and prediction
            base_value = getattr(self, 'base_value', 0)
            prediction = base_value + sum(shap_vals)

            # Sort features by SHAP value (positive first, then negative)
            pos_indices = [i for i in range(len(shap_vals)) if shap_vals[i] > 0]
            neg_indices = [i for i in range(len(shap_vals)) if shap_vals[i] < 0]

            pos_indices.sort(key=lambda i: shap_vals[i], reverse=True)
            neg_indices.sort(key=lambda i: shap_vals[i])

            # Take top features
            n_show = 6
            top_pos = pos_indices[:n_show]
            top_neg = neg_indices[:n_show]

            # Create force-style stacked bar visualization
            ax.set_xlim(-0.1, 1.1)

            # Calculate cumulative positions
            total_pos = sum(shap_vals[i] for i in top_pos)
            total_neg = sum(shap_vals[i] for i in top_neg)
            total_range = abs(total_pos) + abs(total_neg) if (total_pos or total_neg) else 1

            # Draw base value marker
            base_x = 0.5
            ax.axvline(x=base_x, color='gray', linestyle='--', alpha=0.5, linewidth=2)
            ax.text(base_x, 1.05, f'Base\n{base_value:.3f}', ha='center', va='bottom', fontsize=9,
                   transform=ax.get_xaxis_transform())

            # Draw positive contributions (pushing right)
            current_x = base_x
            bar_height = 0.3
            y_center = 0.5

            for i, idx in enumerate(top_pos):
                width = (shap_vals[idx] / total_range) * 0.4 if total_range else 0
                rect = plt.Rectangle((current_x, y_center - bar_height/2), width, bar_height,
                                     facecolor=plt.cm.Blues(0.4 + 0.4 * (1 - i/n_show)),
                                     edgecolor='white', linewidth=1, alpha=0.85)
                ax.add_patch(rect)

                # Add label
                name = feature_names[idx][:15] + '...' if len(feature_names[idx]) > 15 else feature_names[idx]
                if width > 0.03:
                    ax.text(current_x + width/2, y_center, f'{name}\n+{shap_vals[idx]:.3f}',
                           ha='center', va='center', fontsize=7, fontweight='bold')

                current_x += width

            # Draw negative contributions (pushing left)
            current_x = base_x
            for i, idx in enumerate(top_neg):
                width = abs(shap_vals[idx] / total_range) * 0.4 if total_range else 0
                current_x -= width
                rect = plt.Rectangle((current_x, y_center - bar_height/2), width, bar_height,
                                     facecolor=plt.cm.Reds(0.4 + 0.4 * (1 - i/n_show)),
                                     edgecolor='white', linewidth=1, alpha=0.85)
                ax.add_patch(rect)

                # Add label
                name = feature_names[idx][:15] + '...' if len(feature_names[idx]) > 15 else feature_names[idx]
                if width > 0.03:
                    ax.text(current_x + width/2, y_center, f'{name}\n{shap_vals[idx]:.3f}',
                           ha='center', va='center', fontsize=7, fontweight='bold')

            # Draw prediction marker
            pred_x = base_x + (total_pos / total_range) * 0.4 - (abs(total_neg) / total_range) * 0.4 if total_range else base_x
            ax.axvline(x=pred_x, color='green', linestyle='-', alpha=0.8, linewidth=3)
            ax.text(pred_x, -0.15, f'Prediction\n{prediction:.3f}', ha='center', va='top', fontsize=10,
                   fontweight='bold', color='green', transform=ax.get_xaxis_transform())

            # Styling
            ax.set_ylim(0, 1)
            ax.set_title(f'SHAP Force Plot - Sample {sample_idx}', fontsize=12, fontweight='bold', pad=20)
            ax.axis('off')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=plt.cm.Blues(0.6), label=f'Positive (+{total_pos:.3f})'),
                Patch(facecolor=plt.cm.Reds(0.6), label=f'Negative ({total_neg:.3f})')
            ]
            ax.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=8,
                     bbox_to_anchor=(0.5, -0.1))

            print(f"✅ Enhanced force plot created")

        except Exception as e:
            print(f"❌ Enhanced force plot creation failed: {e}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, f'Force plot failed: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
    
    def update_dependence_plot(self):
        """Update dependence plot using native SHAP visualizations"""
        print(f"🔍 === DEPENDENCE PLOT UPDATE START ===")

        # Check basic requirements
        if self.shap_values is None:
            self._show_message_on_figure(self.dependence_figure, "Please calculate SHAP values first")
            return

        if not self.feature_names:
            self._show_message_on_figure(self.dependence_figure, "Feature names missing")
            return

        # Get current selections
        main_feature = self.main_feature_combo.currentText()
        interaction_feature = self.interaction_feature_combo.currentText()

        # Validate main feature
        if not main_feature:
            self._show_message_on_figure(self.dependence_figure, "Please select main feature")
            return

        if main_feature not in self.feature_names:
            self._show_message_on_figure(self.dependence_figure, f"Feature '{main_feature}' does not exist")
            return

        try:
            # Prepare data
            n_samples = min(self.shap_values.shape[0], len(self.explain_data))
            shap_vals = self.shap_values[:n_samples]

            if isinstance(self.explain_data, pd.DataFrame):
                explain_data_df = self.explain_data.iloc[:n_samples].copy()
            else:
                explain_data_df = pd.DataFrame(self.explain_data[:n_samples], columns=self.feature_names)

            main_idx = self.feature_names.index(main_feature)

            # Try native SHAP dependence plot
            success = self._create_dependence_plot_native(
                main_idx, shap_vals, explain_data_df, main_feature, interaction_feature
            )

            if not success:
                print(f"🔄 Native dependence plot failed, using enhanced fallback...")
                self._create_dependence_plot_enhanced(main_idx, shap_vals, explain_data_df, main_feature, interaction_feature)

        except Exception as e:
            print(f"❌ Error creating dependence plot: {e}")
            import traceback
            traceback.print_exc()
            self._create_fallback_dependence_plot(main_feature, interaction_feature)

        print(f"🔍 === DEPENDENCE PLOT UPDATE END ===")

    def _create_dependence_plot_native(self, main_idx, shap_vals, explain_data_df, main_feature, interaction_feature):
        """Create native SHAP dependence plot"""
        try:
            import matplotlib.pyplot as plt
            import io
            from PIL import Image

            # Close all existing figures to avoid conflicts
            plt.close('all')

            # Create a new figure for SHAP
            fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')

            # Determine interaction index
            if interaction_feature == "Auto" or interaction_feature == main_feature or interaction_feature not in self.feature_names:
                interaction_idx = "auto"
            else:
                interaction_idx = self.feature_names.index(interaction_feature)

            # Call SHAP dependence_plot with ax parameter
            shap.dependence_plot(
                main_idx, shap_vals, explain_data_df,
                feature_names=self.feature_names,
                interaction_index=interaction_idx if interaction_idx != "auto" else None,
                ax=ax, show=False
            )

            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            buf.seek(0)
            plt.close(fig)

            # Display in target figure
            img = Image.open(buf)
            img_array = np.array(img)

            self.dependence_figure.clear()
            ax_target = self.dependence_figure.add_subplot(111)
            ax_target.imshow(img_array, aspect='auto')
            ax_target.axis('off')
            self.dependence_figure.tight_layout()

            # Refresh canvas
            self.dependence_canvas.draw()
            QApplication.processEvents()

            print(f"✅ Native dependence plot created successfully")
            return True

        except Exception as e:
            print(f"❌ Native dependence plot failed: {e}")
            return False

    def _create_dependence_plot_enhanced(self, main_idx, shap_vals, explain_data_df, main_feature, interaction_feature):
        """Create enhanced fallback dependence plot with better visualization"""
        try:
            self.dependence_figure.clear()
            ax = self.dependence_figure.add_subplot(111)

            # Get feature values
            feature_vals = explain_data_df.iloc[:, main_idx].values
            shap_feature_vals = shap_vals[:, main_idx]

            # Convert boolean to numeric
            if hasattr(feature_vals, 'dtype') and feature_vals.dtype == bool:
                feature_vals = feature_vals.astype(float)

            # Determine coloring
            if interaction_feature != "Auto" and interaction_feature != main_feature and interaction_feature in self.feature_names:
                int_idx = self.feature_names.index(interaction_feature)
                color_vals = explain_data_df.iloc[:, int_idx].values
                if hasattr(color_vals, 'dtype') and color_vals.dtype == bool:
                    color_vals = color_vals.astype(float)
                color_label = interaction_feature
            else:
                # Auto-select best interaction feature based on correlation
                color_vals = feature_vals
                color_label = main_feature

            # Create scatter plot with enhanced styling
            scatter = ax.scatter(
                feature_vals, shap_feature_vals,
                c=color_vals, cmap='coolwarm', alpha=0.7, s=40, edgecolors='white', linewidths=0.5
            )

            # Add colorbar
            cbar = self.dependence_figure.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(color_label, fontsize=10)

            # Add trend line
            try:
                z = np.polyfit(feature_vals, shap_feature_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(feature_vals.min(), feature_vals.max(), 100)
                ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2, label='Trend')
            except:
                pass

            # Styling
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
            ax.set_xlabel(main_feature, fontsize=11)
            ax.set_ylabel(f'SHAP value for {main_feature}', fontsize=11)
            ax.set_title(f'SHAP Dependence Plot: {main_feature}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

            # Add statistics annotation
            corr = np.corrcoef(feature_vals, shap_feature_vals)[0, 1]
            ax.annotate(f'Correlation: {corr:.3f}', xy=(0.02, 0.98), xycoords='axes fraction',
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            self.dependence_figure.tight_layout()
            self.dependence_canvas.draw()
            QApplication.processEvents()

            print(f"✅ Enhanced dependence plot created successfully")

        except Exception as e:
            print(f"❌ Enhanced dependence plot failed: {e}")
            import traceback
            traceback.print_exc()
            self._create_fallback_dependence_plot(main_feature, interaction_feature)
    def update_advanced_plot(self, target_figure=None, target_canvas=None):
        """Update advanced analysis plots"""
        # Handle signal connections that pass string arguments
        if isinstance(target_figure, str) or target_figure is None:
            target_figure = self.advanced_figure
        if isinstance(target_canvas, str) or target_canvas is None:
            target_canvas = self.advanced_canvas
        self._render_advanced_plot(target_figure, target_canvas)

    def _render_advanced_plot(self, figure, canvas):
        """Core rendering logic for advanced plots"""
        if self.shap_values is None:
            self._show_message_on_figure(figure, "Please calculate SHAP values first")
            return
        if not self.feature_names:
            self._show_message_on_figure(figure, "Feature names missing")
            return
        figure.clear()
        ax = figure.add_subplot(111)
        n_samples = min(len(self.shap_values), len(self.explain_data))
        if n_samples <= 0:
            self._show_message_on_figure(figure, "No samples to plot")
            return
        if isinstance(self.explain_data, pd.DataFrame):
            explain_df = self.explain_data.iloc[:n_samples].copy()
        else:
            explain_df = pd.DataFrame(self.explain_data[:n_samples], columns=self.feature_names)
        shap_vals = self.shap_values[:n_samples]
        start = min(self.adv_sample_start_spin.value(), n_samples - 1)
        end = min(self.adv_sample_end_spin.value(), n_samples - 1)
        if end < start:
            end = start
        shap_slice = shap_vals[start:end+1]
        explain_slice = explain_df.iloc[start:end+1]
        plot_type = self.advanced_plot_combo.currentText()
        try:
            if plot_type == "Interaction Plot":
                main_feature = self.adv_feature1_combo.currentText() or self.feature_names[0]
                interaction_feature = self.adv_feature2_combo.currentText()
                if main_feature not in self.feature_names:
                    main_feature = self.feature_names[0]
                main_idx = self.feature_names.index(main_feature)
                x_vals = explain_slice[main_feature] if main_feature in explain_slice else explain_slice.iloc[:, main_idx]
                color_vals = None
                if interaction_feature and interaction_feature != main_feature and interaction_feature in explain_slice:
                    color_vals = explain_slice[interaction_feature]
                scatter = ax.scatter(x_vals, shap_slice[:, main_idx],
                                     c=color_vals if color_vals is not None else x_vals,
                                     cmap="coolwarm", alpha=0.7, edgecolors="none")
                if color_vals is not None:
                    figure.colorbar(scatter, ax=ax, label=interaction_feature)
                ax.set_xlabel(main_feature)
                ax.set_ylabel("SHAP value")
                ax.set_title(f"Interaction: {main_feature} vs SHAP")
            elif plot_type == "SHAP Heatmap":
                import seaborn as sns
                mean_abs = np.abs(shap_slice).mean(axis=0)
                top_idx = np.argsort(mean_abs)[::-1][:min(25, len(self.feature_names))]
                heat_df = pd.DataFrame(shap_slice[:, top_idx], columns=[self.feature_names[i] for i in top_idx])
                sns.heatmap(heat_df.T, cmap="coolwarm", center=0, ax=ax)
                ax.set_ylabel("Feature")
                ax.set_xlabel("Sample")
                ax.set_title("SHAP Heatmap (top features)")
            elif plot_type == "Decision Plot":
                expected = getattr(self.explainer, "expected_value", self.base_value)
                try:
                    shap.decision_plot(expected, shap_slice, feature_names=self.feature_names, show=False, ax=ax)
                except Exception as plot_e:
                    cum = np.cumsum(shap_slice, axis=1)
                    ax.plot(cum.T, alpha=0.4)
                    ax.set_title(f"Decision (fallback) - {plot_e}")
                ax.set_xlabel("Features (ordered)")
                ax.set_ylabel("Contribution")
            elif plot_type == "Clustering Analysis":
                import seaborn as sns
                corr = np.corrcoef(shap_slice.T)
                sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                            xticklabels=self.feature_names, yticklabels=self.feature_names)
                ax.set_title("SHAP Feature Correlation")
            elif plot_type == "Multi-Sample Comparison":
                shap_df = pd.DataFrame(shap_slice, columns=self.feature_names)
                top_feats = shap_df.abs().mean().sort_values(ascending=False).head(8).index
                shap_df[top_feats].plot(kind="box", vert=False, ax=ax)
                ax.set_title("Top Feature SHAP Distributions")
                ax.set_xlabel("SHAP value")
            else:
                ax.text(0.5, 0.5, f"Plot type {plot_type} not implemented", ha="center", va="center", transform=ax.transAxes)
            figure.tight_layout()
            if canvas is not None:
                canvas.draw()
                canvas.flush_events()
                QApplication.processEvents()
        except Exception as e:
            print(f"Advanced plot failed: {e}")
            import traceback
            traceback.print_exc()
            self._show_message_on_figure(figure, f"Plot failed: {e}")
            if canvas is not None:
                canvas.draw()

    def open_independent_advanced_window(self):
        """Open advanced analysis in an independent window"""
        if self.shap_values is None:
            QMessageBox.warning(self, "Warning", "Please calculate SHAP values first")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("SHAP Advanced Analysis - Independent Window")
        layout = QVBoxLayout(dialog)
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        self._render_advanced_plot(fig, canvas)
        dialog.show()
        print("Opened independent advanced analysis window")


    def force_update_all_plots(self):
        """Force update all SHAP plots - useful for troubleshooting"""
        print(f"🔍 === FORCE UPDATE ALL PLOTS START ===")

        if self.shap_values is None:
            print(f"❌ No SHAP values available for plot update")
            return

        try:
            # Update all three plot types with error handling
            plots_to_update = [
                ("Summary Plot", self.update_summary_plot),
                ("Local Plot", self.update_local_plot),
                ("Dependence Plot", self.update_dependence_plot),
                ("Advanced Plot", self.update_advanced_plot)
            ]

            for plot_name, update_func in plots_to_update:
                try:
                    print(f"🔍 Force updating {plot_name}...")
                    update_func()
                    QApplication.processEvents()
                    print(f"✅ {plot_name} updated successfully")
                except Exception as e:
                    print(f"❌ {plot_name} update failed: {e}")

            # Force canvas refresh
            self.force_refresh_all_canvas()

            print(f"✅ All plots force updated")

        except Exception as e:
            print(f"❌ Force update all plots failed: {e}")

        print(f"🔍 === FORCE UPDATE ALL PLOTS END ===")

    def _create_manual_waterfall_in_temp_fig(self, shap_vals, sample_data, base_value):
        """Create a manual waterfall plot directly in the current matplotlib figure"""
        import matplotlib.pyplot as plt

        # Sort features by absolute SHAP value for better visualization
        feature_importance = [(i, abs(val), val, self.feature_names[i], sample_data.iloc[i])
                             for i, val in enumerate(shap_vals)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # Take top features for cleaner visualization
        top_features = feature_importance[:min(12, len(feature_importance))]

        # Create waterfall plot
        ax = plt.gca()
        y_positions = range(len(top_features) + 2)  # +2 for base and prediction

        # Calculate cumulative values
        cumulative = base_value
        positions = [0]
        values = [base_value]
        colors = ['gray']
        labels = ['Base Value']

        for i, (_, _, shap_val, feat_name, feat_val) in enumerate(top_features):
            values.append(shap_val)
            colors.append('red' if shap_val < 0 else 'blue')
            labels.append(f'{feat_name}\n= {feat_val:.3f}')
            positions.append(i + 1)
            cumulative += shap_val

        # Final prediction
        final_prediction = base_value + sum([item[2] for item in top_features])
        values.append(final_prediction)
        colors.append('green')
        labels.append('Prediction')
        positions.append(len(top_features) + 1)

        # Create horizontal bars
        bars = ax.barh(positions, values, color=colors, alpha=0.7)

        # Add value annotations
        for i, (bar, val) in enumerate(zip(bars, values)):
            if abs(val) > 0.001:
                x_pos = val + (0.01 * max(abs(min(values)), abs(max(values))) if val >= 0 else -0.01 * max(abs(min(values)), abs(max(values))))
                ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center', fontweight='bold')

        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'Waterfall Plot - Sample {self.sample_index_spin.value() if hasattr(self, "sample_index_spin") else "N/A"}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

    def _create_manual_force_in_temp_fig(self, shap_vals, sample_data, base_value):
        """Create a manual force plot directly in the current matplotlib figure"""
        import matplotlib.pyplot as plt

        # Sort features by absolute SHAP value
        indices = np.argsort(np.abs(shap_vals))[::-1]

        # Take top features for cleaner visualization
        top_n = min(10, len(indices))
        top_indices = indices[:top_n]

        ax = plt.gca()

        # Create horizontal bar chart
        y_pos = range(len(top_indices))
        values = shap_vals[top_indices]
        colors = ['red' if val < 0 else 'blue' for val in values]

        bars = ax.barh(y_pos, values, color=colors, alpha=0.7)

        # Set labels with feature names and values
        labels = [f"{self.feature_names[i]}\n= {sample_data.iloc[i]:.3f}" for i in top_indices]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)

        # Add value annotations
        for i, (bar, val) in enumerate(zip(bars, values)):
            if abs(val) > 0.001:
                x_pos = val/2
                ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', ha='center', va='center', fontweight='bold', color='white')

        ax.set_xlabel('SHAP Value')
        ax.set_title(f'Force Plot - Sample {self.sample_index_spin.value() if hasattr(self, "sample_index_spin") else "N/A"}\nBase: {base_value:.3f}, Prediction: {base_value + sum(shap_vals):.3f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
    
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

            print(f"🔍 Main feature: {main_feature}, samples: {n_samples}")

            # Handle interaction feature
            if interaction_feature == "Auto" or interaction_feature == main_feature or interaction_feature not in self.feature_names:
                # Simple dependence plot colored by feature value
                print(f"🔍 Creating simple fallback dependence plot")

                # Handle boolean/categorical data
                if hasattr(feature_vals, 'dtype') and feature_vals.dtype == bool:
                    feature_vals_numeric = feature_vals.astype(int)
                else:
                    feature_vals_numeric = feature_vals

                # Create scatter plot with proper error handling
                try:
                    scatter = ax.scatter(feature_vals_numeric, shap_vals,
                                       c=feature_vals_numeric, cmap='viridis', alpha=0.7, s=30)

                    # Add colorbar with error handling
                    try:
                        cbar = self.dependence_figure.colorbar(scatter, ax=ax)
                        cbar.set_label(main_feature)
                    except Exception as cbar_error:
                        print(f"⚠️ Colorbar creation failed: {cbar_error}")

                except Exception as scatter_error:
                    print(f"❌ Scatter plot creation failed: {scatter_error}")
                    # Create simple line plot as ultimate fallback
                    ax.plot(feature_vals_numeric, shap_vals, 'o', alpha=0.7)

                ax.set_xlabel(f'{main_feature} (Feature Value)')
                ax.set_ylabel(f'SHAP Value for {main_feature}')
                ax.set_title(f'SHAP Dependence Plot: {main_feature} (Fallback)')

            else:
                # Interaction dependence plot
                print(f"🔍 Creating interaction fallback dependence plot with {interaction_feature}")
                int_idx = self.feature_names.index(interaction_feature)
                int_vals = self.explain_data.iloc[:n_samples, int_idx].values

                # Handle boolean/categorical data
                if hasattr(feature_vals, 'dtype') and feature_vals.dtype == bool:
                    feature_vals_numeric = feature_vals.astype(int)
                else:
                    feature_vals_numeric = feature_vals

                if hasattr(int_vals, 'dtype') and int_vals.dtype == bool:
                    int_vals_numeric = int_vals.astype(int)
                else:
                    int_vals_numeric = int_vals

                try:
                    scatter = ax.scatter(feature_vals_numeric, shap_vals,
                                       c=int_vals_numeric, cmap='viridis', alpha=0.7, s=30)

                    # Add colorbar with error handling
                    try:
                        cbar = self.dependence_figure.colorbar(scatter, ax=ax)
                        cbar.set_label(interaction_feature)
                    except Exception as cbar_error:
                        print(f"⚠️ Interaction colorbar creation failed: {cbar_error}")

                except Exception as scatter_error:
                    print(f"❌ Interaction scatter plot creation failed: {scatter_error}")
                    # Create simple line plot as ultimate fallback
                    ax.plot(feature_vals_numeric, shap_vals, 'o', alpha=0.7)

                ax.set_xlabel(f'{main_feature} (Feature Value)')
                ax.set_ylabel(f'SHAP Value for {main_feature}')
                ax.set_title(f'SHAP Dependence: {main_feature} vs {interaction_feature} (Fallback)')

            # Add reference line and grid
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)

            # Enhanced canvas refresh with proper error handling
            try:
                self.dependence_figure.tight_layout()
                self.dependence_canvas.draw()
                QApplication.processEvents()
                self.dependence_canvas.flush_events()
                print(f"✅ Fallback dependence plot created and displayed")
            except Exception as refresh_error:
                print(f"❌ Canvas refresh failed: {refresh_error}")
                # Try simple refresh
                try:
                    self.dependence_canvas.draw()
                except:
                    pass

        except Exception as e:
            print(f"❌ Fallback dependence plot failed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            self._show_message_on_figure(self.dependence_figure, f"Dependence plot creation failed: {str(e)[:100]}")
    
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

            # Update advanced analysis feature combos
            if hasattr(self, 'adv_feature1_combo'):
                self.adv_feature1_combo.clear()
                self.adv_feature1_combo.addItems(self.feature_names)
            if hasattr(self, 'adv_feature2_combo'):
                self.adv_feature2_combo.clear()
                self.adv_feature2_combo.addItem("None")
                self.adv_feature2_combo.addItems(self.feature_names)

    def _refresh_class_selector(self):
        """展示/刷新多分类选择控件"""
        if not hasattr(self, "class_selector"):
            return
        has_multi = self.all_class_shap_values is not None and self.n_classes and self.n_classes > 1
        self.class_selector_label.setVisible(has_multi)
        self.class_selector.setVisible(has_multi)
        if not has_multi:
            self.class_selector.clear()
            return
        self.class_selector.blockSignals(True)
        self.class_selector.clear()
        for idx in range(self.n_classes):
            self.class_selector.addItem(f"Class {idx}")
        target_idx = min(self.selected_class if self.selected_class is not None else 0, self.n_classes - 1)
        self.class_selector.setCurrentIndex(target_idx)
        self.class_selector.blockSignals(False)

        
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

        try:
            X = self.training_data['X']

            # Use a sample of training data as background data
            background_size = min(self.background_samples_spin.value(), len(X))
            sample_indices = np.random.choice(len(X), size=background_size, replace=False)
            self.background_data = X.iloc[sample_indices].copy()

            # Use random subset for explanation data to avoid sequential bias
            explain_size = min(100, len(X))
            explain_indices = np.random.choice(len(X), size=explain_size, replace=False)
            self.explain_data = X.iloc[explain_indices].copy()

            # Update UI
            self.sample_index_spin.setRange(0, len(self.explain_data) - 1)
            self._update_feature_selections()
            self._check_calculation_readiness()

            self.status_label.setText(f"Auto-loaded: {len(self.background_data)} background samples, {len(self.explain_data)} explain samples")
            print(f"SHAP: Auto-setup completed - {len(self.background_data)} background, {len(self.explain_data)} explain samples")

        except Exception as e:
            print(f"Error in _auto_setup_data: {e}")
            self.status_label.setText(f"Error auto-loading data: {str(e)}")
        
    def reset(self):
        """Reset the module"""
        self.model = None
        self.background_data = None
        self.explain_data = None
        self.feature_names = None
        self.shap_values = None
        self.explainer = None
        self.training_data = None
        self.base_value = 0

        # Reset multi-class attributes
        self.all_class_shap_values = None
        self.n_classes = 1
        self.selected_class = 0

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
        """Force refresh all canvas"""
        print(f"🔍 Force refreshing all canvas...")

        try:
            # Refresh global canvas
            if hasattr(self, 'global_canvas'):
                self.global_canvas.draw()
                self.global_canvas.flush_events()
                QApplication.processEvents()
                print(f"🔍 Global canvas force refreshed")

            # Refresh local canvas
            if hasattr(self, 'local_canvas'):
                self.local_canvas.draw()
                self.local_canvas.flush_events()
                QApplication.processEvents()
                print(f"🔍 Local canvas force refreshed")

            # Refresh dependence canvas
            if hasattr(self, 'dependence_canvas'):
                self.dependence_canvas.draw()
                self.dependence_canvas.flush_events()
                QApplication.processEvents()
                print(f"🔍 Dependence canvas force refreshed")

            print(f"✅ All canvas force refreshed")

        except Exception as e:
            print(f"❌ Error force refreshing canvas: {e}")
    
    def open_independent_summary_window(self):
        """Open independent summary plot window"""
        if self.shap_values is None:
            QMessageBox.warning(self, "Warning", "Please calculate SHAP values first")
            return

        try:
            plot_type = self.global_plot_combo.currentText()
            window = SHAPVisualizationWindow(
                self.shap_values, self.explainer, self.explain_data,
                self.feature_names, plot_type, self
            )
            window.show()
            print(f"✅ Independent summary window opened: {plot_type}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open independent window:\n{str(e)}")
            print(f"❌ Failed to open independent summary window: {e}")
    
    def open_independent_dependence_window(self):
        """Open independent dependence plot window"""
        if self.shap_values is None:
            QMessageBox.warning(self, "Warning", "Please calculate SHAP values first")
            return

        try:
            main_feature = self.main_feature_combo.currentText()
            interaction_feature = self.interaction_feature_combo.currentText()

            window = SHAPDependenceWindow(
                self.shap_values, self.explain_data, self.feature_names,
                main_feature, interaction_feature, self
            )
            window.show()
            print(f"✅ Independent dependence window opened: {main_feature} vs {interaction_feature}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open independent window:\n{str(e)}")
            print(f"❌ Failed to open independent dependence window: {e}")
    
    def open_independent_local_window(self):
        """Open independent local explanation window"""
        if self.shap_values is None:
            QMessageBox.warning(self, "Warning", "Please calculate SHAP values first")
            return

        try:
            sample_idx = self.sample_index_spin.value()
            plot_type = self.local_plot_combo.currentText()

            window = SHAPLocalWindow(
                self.shap_values, self.explainer, self.explain_data,
                self.feature_names, sample_idx, plot_type, self
            )
            window.show()
            print(f"✅ Independent local window opened: {plot_type}, sample {sample_idx}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open independent window:\n{str(e)}")
            print(f"❌ Failed to open independent local window: {e}")
