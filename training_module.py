"""
Module 3: Model Training & Evaluation
Handles model training, hyperparameter optimization, and evaluation
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QPushButton, QMessageBox,
                            QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit,
                            QSplitter, QTabWidget, QCheckBox, QListWidget,
                            QListWidgetItem, QProgressBar, QFileDialog,
                            QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

# Add plotting imports
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("Warning: seaborn not available, some visualizations may be limited")

from utils.ml_utils import (get_available_models, get_default_hyperparameters,
                           create_preprocessing_pipeline, evaluate_classification_model,
                           evaluate_regression_model, get_cv_folds, get_scoring_metric,
                           create_model_with_params)
from utils.plot_utils import (plot_confusion_matrix, plot_roc_curve, 
                             plot_prediction_vs_actual, plot_residuals)


class TrainingModule(QWidget):
    """Model training and evaluation module"""
    
    # Signals
    model_ready = pyqtSignal(object, list, dict, object, object)  # Trained pipeline, feature_names, feature_info, X_train, y_train
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    training_started = pyqtSignal()  # Training start signal
    
    def __init__(self):
        super().__init__()
        self.X = None
        self.y = None
        self.y_original = None
        self.task_type = None
        self.trained_pipeline = None
        self.evaluation_results = None
        self.previous_model_selection = None  # Store model from feature selection
        self.custom_param_spaces = {}  # Store custom parameter spaces
        self.hpo_results = None  # Store hyperparameter optimization results
        self.label_encoder = None  # For classification target encoding
        self.class_names = None  # Store original class names
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Right panel for results
        right_panel = QTabWidget()
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 1000])
        
        # Data info
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout(info_group)
        self.data_info_label = QLabel("No data loaded")
        info_layout.addWidget(self.data_info_label)
        left_layout.addWidget(info_group)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        # Task type
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Task Type:"))
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["Classification", "Regression"])
        self.task_type_combo.currentTextChanged.connect(self.update_available_models)
        task_layout.addWidget(self.task_type_combo)
        model_layout.addLayout(task_layout)
        
        # Model selection
        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        model_selection_layout.addWidget(self.model_combo)
        model_layout.addLayout(model_selection_layout)
        
        left_layout.addWidget(model_group)
        
        # Training configuration
        config_group = QGroupBox("Training Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Test size
        test_size_layout = QHBoxLayout()
        test_size_layout.addWidget(QLabel("Test Set Ratio:"))
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setMinimum(0.1)
        self.test_size_spin.setMaximum(0.5)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.2)
        test_size_layout.addWidget(self.test_size_spin)
        config_layout.addLayout(test_size_layout)
        
        # Random state
        random_state_layout = QHBoxLayout()
        random_state_layout.addWidget(QLabel("Random Seed:"))
        self.random_state_spin = QSpinBox()
        self.random_state_spin.setMinimum(0)
        self.random_state_spin.setMaximum(999)
        self.random_state_spin.setValue(42)
        random_state_layout.addWidget(self.random_state_spin)
        config_layout.addLayout(random_state_layout)
        
        # Evaluation strategy
        eval_strategy_layout = QHBoxLayout()
        eval_strategy_layout.addWidget(QLabel("Evaluation Strategy:"))
        self.eval_strategy_combo = QComboBox()
        self.eval_strategy_combo.addItems([
            "Single Split (Fast)",
            "Multiple Splits (Recommended)",
            "Nested Cross-Validation (Most Reliable)"
        ])
        self.eval_strategy_combo.setCurrentIndex(1)  # Default to multiple splits
        eval_strategy_layout.addWidget(self.eval_strategy_combo)
        config_layout.addLayout(eval_strategy_layout)
        
        left_layout.addWidget(config_group)
        
        # HPO configuration
        hpo_group = QGroupBox("Hyperparameter Optimization")
        hpo_layout = QVBoxLayout(hpo_group)
        
        self.enable_hpo = QCheckBox("Enable HPO")
        self.enable_hpo.setChecked(False)  # Disabled by default for speed
        hpo_layout.addWidget(self.enable_hpo)
        
        # HPO method
        hpo_method_layout = QHBoxLayout()
        hpo_method_layout.addWidget(QLabel("Method:"))
        self.hpo_method_combo = QComboBox()
        self.hpo_method_combo.addItems(["Grid Search", "Random Search", "Bayesian Search"])
        hpo_method_layout.addWidget(self.hpo_method_combo)
        hpo_layout.addLayout(hpo_method_layout)
        
        # CV folds
        cv_layout = QHBoxLayout()
        cv_layout.addWidget(QLabel("CV Folds:"))
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setMinimum(3)
        self.cv_folds_spin.setMaximum(10)
        self.cv_folds_spin.setValue(5)
        cv_layout.addWidget(self.cv_folds_spin)
        hpo_layout.addLayout(cv_layout)
        
        # Parameter space configuration
        self.configure_params_btn = QPushButton("Configure Parameter Space")
        self.configure_params_btn.clicked.connect(self.configure_parameter_space)
        hpo_layout.addWidget(self.configure_params_btn)
        
        left_layout.addWidget(hpo_group)
        
        # Training buttons
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setEnabled(False)
        left_layout.addWidget(self.train_btn)
        
        # Save model
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        left_layout.addWidget(self.save_model_btn)
        
        # Proceed button
        self.proceed_btn = QPushButton("Proceed to Prediction")
        self.proceed_btn.clicked.connect(self.proceed_to_next_module)
        self.proceed_btn.setEnabled(False)
        left_layout.addWidget(self.proceed_btn)
        
        left_layout.addStretch()
        
        # Results panel
        self.results_tabs = {}
        self.right_panel = right_panel
        
        # Initially disable
        self.setEnabled(False)
    
    def configure_parameter_space(self):
        """Configure hyperparameter search space"""
        if not self.model_combo.currentText():
            QMessageBox.warning(self, "Warning", "Please select a model first.")
            return
            
        try:
            from modules.feature_module import ParameterSpaceDialog
            
            current_model = self.model_combo.currentText()
            task_type = self.task_type_combo.currentText().lower()
            
            dialog = ParameterSpaceDialog([current_model], task_type, self.custom_param_spaces)
            if dialog.exec_() == dialog.Accepted:
                self.custom_param_spaces = dialog.get_parameter_spaces()
                QMessageBox.information(
                    self, "Success", 
                    f"Parameter space configured for {current_model}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error configuring parameter space: {str(e)}")
    
    def show_quantile_input_dialog(self):
        """Show dialog for user to input quantile parameter"""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Quantile Regression Configuration")
            dialog.setModal(True)
            dialog.resize(450, 400)

            layout = QVBoxLayout(dialog)

            # Title
            title_label = QLabel("Quantile Regression Parameter Input")
            title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
            layout.addWidget(title_label)

            # Description
            desc_label = QLabel("Quantile Regression requires you to specify the quantile to predict.\n"
                               "This is a business decision parameter that depends on your specific needs:")
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("margin-bottom: 15px;")
            layout.addWidget(desc_label)

            # Input section
            input_group = QGroupBox("Quantile Value (0 < value < 1)")
            input_layout = QVBoxLayout(input_group)

            # Quantile input
            quantile_layout = QHBoxLayout()
            quantile_layout.addWidget(QLabel("Quantile:"))

            self.quantile_input = QDoubleSpinBox()
            self.quantile_input.setMinimum(0.01)
            self.quantile_input.setMaximum(0.99)
            self.quantile_input.setSingleStep(0.01)
            self.quantile_input.setDecimals(3)
            self.quantile_input.setValue(0.5)  # Default to median
            quantile_layout.addWidget(self.quantile_input)

            input_layout.addLayout(quantile_layout)
            layout.addWidget(input_group)

            # Examples section
            examples_group = QGroupBox("Common Use Cases")
            examples_layout = QVBoxLayout(examples_group)

            examples_text = """• 0.1 (10th percentile) - Conservative estimates, risk assessment
• 0.25 (25th percentile) - Lower quartile, pessimistic planning
• 0.5 (50th percentile) - Median prediction, most common choice
• 0.75 (75th percentile) - Upper quartile, optimistic planning
• 0.9 (90th percentile) - Optimistic estimates, upper bounds"""

            examples_label = QLabel(examples_text)
            examples_label.setStyleSheet("font-family: monospace; color: #555;")
            examples_layout.addWidget(examples_label)

            # Quick selection buttons
            quick_layout = QHBoxLayout()
            quick_layout.addWidget(QLabel("Quick select:"))

            quick_values = [
                ("Conservative (0.1)", 0.1),
                ("Lower Q (0.25)", 0.25),
                ("Median (0.5)", 0.5),
                ("Upper Q (0.75)", 0.75),
                ("Optimistic (0.9)", 0.9)
            ]

            for label, value in quick_values:
                btn = QPushButton(label)
                btn.clicked.connect(lambda checked, v=value: self.quantile_input.setValue(v))
                quick_layout.addWidget(btn)

            examples_layout.addLayout(quick_layout)
            layout.addWidget(examples_group)

            # Explanation section
            explain_group = QGroupBox("What Does This Mean?")
            explain_layout = QVBoxLayout(explain_group)

            explain_text = """Quantile regression predicts the specified percentile of the target distribution.

For example:
• Quantile 0.1: Predicts a value where 10% of actual values are below
• Quantile 0.5: Predicts the median (50% above, 50% below)
• Quantile 0.9: Predicts a value where 90% of actual values are below

This is different from regular regression which only predicts the mean."""

            explain_label = QLabel(explain_text)
            explain_label.setWordWrap(True)
            explain_label.setStyleSheet("color: #666; font-size: 11px;")
            explain_layout.addWidget(explain_label)
            layout.addWidget(explain_group)

            # Buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            # Show dialog and get result
            if dialog.exec_() == QDialog.Accepted:
                quantile_value = self.quantile_input.value()
                print(f"User selected quantile: {quantile_value}")
                return quantile_value
            else:
                print("User cancelled quantile input")
                return None

        except Exception as e:
            print(f"Error in quantile input dialog: {e}")
            QMessageBox.critical(self, "Error", f"Error showing quantile input dialog: {str(e)}")
            return None

    def create_model_instance(self, model_class, random_state):
        """Create model instance with appropriate parameters"""
        from utils.ml_utils import create_model_with_params, check_model_requires_user_input, get_user_input_requirements

        # Check if this model requires user input before creating
        model_name = model_class.__name__
        if 'QuantileRegressor' in model_name or check_model_requires_user_input(self.model_combo.currentText()):
            # Show user input dialog for quantile parameter
            quantile_value = self.show_quantile_input_dialog()
            if quantile_value is None:
                # User cancelled, raise an exception to stop training
                raise ValueError("User cancelled quantile input")

            # Create model with user-specified quantile
            return create_model_with_params(model_class, quantile=quantile_value, random_state=random_state)
        else:
            # Use the smart model creation function for other models
            return create_model_with_params(model_class, random_state=random_state)
    
    def set_data(self, X: pd.DataFrame, y: pd.Series, previous_config=None):
        """Set input data from feature selection module with optional configuration"""
        try:
            print(f"=== TRAINING MODULE DATA RECEPTION ===")
            
            # Input validation
            if X is None or y is None:
                raise ValueError("Received None data - X or y is None")
            
            if not isinstance(X, pd.DataFrame):
                raise TypeError(f"X must be a DataFrame, got {type(X)}")
            
            if not isinstance(y, pd.Series):
                raise TypeError(f"y must be a Series, got {type(y)}")
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError(f"Empty data received - X: {len(X)} samples, y: {len(y)} samples")
            
            if len(X) != len(y):
                raise ValueError(f"Data length mismatch - X: {len(X)} samples, y: {len(y)} samples")
            
            print(f"✓ Input validation passed")
            print(f"Received X shape: {X.shape}")
            print(f"Received y shape: {y.shape}")
            print(f"y dtype: {y.dtype}")
            print(f"y unique values: {sorted(y.unique())}")
            print(f"y name: {y.name}")
            
            # Safe copy operations
            try:
                self.X = X.copy()
                self.y_original = y.copy()  # Keep original for reference
                print(f"✓ Data copied successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to copy data: {str(e)}")
            
            # Store previous configuration if provided
            if previous_config:
                try:
                    self.previous_model_selection = previous_config.get('selected_model')
                    print(f"Previous model selection: {self.previous_model_selection}")
                    
                    # Also store other relevant config
                    self.previous_task_type = previous_config.get('task_type')
                    self.previous_scoring_metric = previous_config.get('scoring_metric')
                    print(f"Previous config - Task: {self.previous_task_type}, Scoring: {self.previous_scoring_metric}")
                    
                except Exception as e:
                    print(f"Warning: Could not process previous config: {str(e)}")
                    self.previous_model_selection = None
                    self.previous_task_type = None
                    self.previous_scoring_metric = None
            else:
                self.previous_model_selection = None
                self.previous_task_type = None
                self.previous_scoring_metric = None
                print("No previous configuration provided")
                
            # CRITICAL FIX: Use consistent task type detection across modules
            try:
                # Import the unified task type detection function
                from utils.data_utils import suggest_task_type
                
                # Use previous config task type if available, otherwise detect
                if hasattr(self, 'previous_task_type') and self.previous_task_type:
                    self.task_type = self.previous_task_type
                    print(f"✓ Using task type from previous module: {self.task_type}")
                else:
                    self.task_type = suggest_task_type(y)
                    print(f"✓ Detected task type using unified logic: {self.task_type}")
                
                if pd.api.types.is_numeric_dtype(y):
                    unique_values = y.nunique()
                    y_values = sorted(y.unique())
                    
                    # Check if this looks like already-encoded classification labels
                    is_likely_encoded_classification = (
                        self.task_type == "classification" and
                        unique_values <= 20 and 
                        all(isinstance(val, (int, np.integer)) for val in y_values) and
                        y_values == list(range(len(y_values))) and  # Sequential: 0, 1, 2, ...
                        min(y_values) == 0
                    )
                    
                    if is_likely_encoded_classification:
                        self.y = y.copy()  # Use as-is, already encoded
                        self.class_names = [f"Class_{i}" for i in range(unique_values)]
                        self.label_encoder = None  # No encoding needed
                        print(f"✓ Pre-encoded classification target with {unique_values} classes")
                        print(f"Class labels: {self.class_names}")
                    elif self.task_type == "classification" and unique_values < 20:
                        # Small number of unique values, classification but not sequential
                        self.y = y.copy()
                        self.class_names = sorted(y.unique())
                        self.label_encoder = None
                        print(f"✓ Numeric classification target: {self.class_names}")
                    else:
                        # Regression or other cases
                        self.y = y.copy()
                        self.class_names = None
                        self.label_encoder = None
                        if self.task_type == "regression":
                            print(f"✓ Regression target (range: {y.min():.3f} - {y.max():.3f})")
                        else:
                            print(f"✓ Target processed for {self.task_type} task")
                else:
                    # Non-numeric target - definitely classification that needs encoding
                    # This should rarely happen if DataModule did its job
                    print("WARNING: Received non-numeric target - this suggests DataModule didn't encode it")
                    self.task_type = "classification"
                    
                    from sklearn.preprocessing import LabelEncoder
                    self.label_encoder = LabelEncoder()
                    
                    # Ensure consistent encoding
                    y_str = y.astype(str)
                    encoded_y = self.label_encoder.fit_transform(y_str)
                    self.y = pd.Series(encoded_y, index=y.index, name=y.name)
                    
                    # Store the class names for later reference
                    self.class_names = self.label_encoder.classes_
                    
                    print(f"Applied encoding: {dict(zip(self.class_names, range(len(self.class_names))))}")
                    self.status_updated.emit(f"Target variable encoded: {dict(zip(self.class_names, range(len(self.class_names))))}")
                    
            except Exception as e:
                print(f"ERROR in target variable processing: {str(e)}")
                # Fallback: treat as classification with simple encoding
                self.task_type = "classification"
                self.y = y.copy()
                self.class_names = sorted(y.unique())
                self.label_encoder = None
                print(f"Fallback: treating as classification with classes: {self.class_names}")
                
            # Final validation
            print(f"Final task type: {self.task_type}")
            print(f"Final y dtype: {self.y.dtype}")
            print(f"Final y values: {sorted(self.y.unique())}")
            
            if self.task_type == "classification":
                try:
                    class_counts = pd.Series(self.y).value_counts().sort_index()
                    print(f"Class distribution:")
                    for class_val, count in class_counts.items():
                        class_name = self.class_names[class_val] if self.class_names and class_val < len(self.class_names) else f"Class_{class_val}"
                        print(f"  {class_name} ({class_val}): {count} samples ({count/len(self.y)*100:.1f}%)")
                except Exception as e:
                    print(f"Warning: Could not display class distribution: {str(e)}")
            
            print("=" * 50)
                
            # Update UI with error handling
            try:
                # Temporarily disconnect signal to avoid triggering update_available_models twice
                self.task_type_combo.currentTextChanged.disconnect()
                self.task_type_combo.setCurrentText(self.task_type.title())
                self.task_type_combo.currentTextChanged.connect(self.update_available_models)
                
                self.update_data_info()
                self.update_available_models()
                
                # Note: Model selection is now handled in update_available_models()
                    
                self.setEnabled(True)
                self.train_btn.setEnabled(True)
                
                self.status_updated.emit(f"Training data loaded: {X.shape[0]} samples, {X.shape[1]} features, Task: {self.task_type}")
                print(f"✓ UI updated successfully")
                
            except Exception as e:
                print(f"ERROR updating UI: {str(e)}")
                # Still enable basic functionality even if UI update fails
                self.setEnabled(True)
                self.train_btn.setEnabled(True)
                
            print("=== TRAINING MODULE DATA RECEPTION - SUCCESS ===")
            
        except Exception as e:
            print(f"CRITICAL ERROR in set_data: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Try to show error to user
            try:
                QMessageBox.critical(self, "Error", f"Failed to load training data: {str(e)}")
            except:
                print("Could not show error message to user")
                
            print("=== TRAINING MODULE DATA RECEPTION - FAILED ===")

    def _fix_quantile_regressor_params(self, pipeline):
        """
        Fix parameter types for QuantileRegressor after hyperparameter optimization

        Args:
            pipeline: Trained pipeline that may contain QuantileRegressor
        """
        try:
            if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
                model = pipeline.named_steps['model']
                model_name = model.__class__.__name__

                if 'QuantileRegressor' in model_name:
                    print(f"DEBUG: Fixing QuantileRegressor parameters...")

                    # Get current parameters
                    current_params = model.get_params()
                    print(f"DEBUG: Current parameters: {current_params}")

                    # Fix fit_intercept parameter type if needed
                    if 'fit_intercept' in current_params:
                        fit_intercept_value = current_params['fit_intercept']
                        if isinstance(fit_intercept_value, (int, np.integer, np.bool_)):
                            corrected_value = bool(fit_intercept_value)
                            model.set_params(fit_intercept=corrected_value)
                            print(f"DEBUG: Fixed fit_intercept: {fit_intercept_value} ({type(fit_intercept_value).__name__}) -> {corrected_value} ({type(corrected_value).__name__})")
                        elif not isinstance(fit_intercept_value, bool):
                            # Handle any other types that might need conversion
                            corrected_value = bool(fit_intercept_value)
                            model.set_params(fit_intercept=corrected_value)
                            print(f"DEBUG: Fixed fit_intercept: {fit_intercept_value} ({type(fit_intercept_value).__name__}) -> {corrected_value} ({type(corrected_value).__name__})")
                        else:
                            print(f"DEBUG: fit_intercept is already correct type: {fit_intercept_value} ({type(fit_intercept_value).__name__})")

                    # Verify all parameters are correct types
                    final_params = model.get_params()
                    print(f"DEBUG: Final parameters after fixing: {final_params}")

        except Exception as e:
            print(f"Warning: Could not fix QuantileRegressor parameters: {e}")

    def update_data_info(self):
        """Update data information display"""
        if self.X is not None and self.y is not None:
            info_text = f"Samples: {self.X.shape[0]}\n"
            info_text += f"Features: {self.X.shape[1]}\n"
            info_text += f"Task Type: {self.task_type.title()}\n"
            
            # Add target info
            if self.task_type == "classification":
                unique_classes = self.y.nunique()
                info_text += f"Classes: {unique_classes}\n"
                if hasattr(self, 'class_names'):
                    info_text += f"Labels: {', '.join(map(str, self.class_names[:5]))}"
                    if len(self.class_names) > 5:
                        info_text += "..."
                else:
                    info_text += f"Range: {self.y.min()}-{self.y.max()}"
            else:
                info_text += f"Target Range: {self.y.min():.3f} - {self.y.max():.3f}"
            
            self.data_info_label.setText(info_text)
            
    def update_available_models(self):
        """Update available models based on task type"""
        task_type = self.task_type_combo.currentText().lower()
        
        # Store current model selection before clearing
        current_model_selection = self.model_combo.currentText() if self.model_combo.count() > 0 else None
        
        if task_type != self.task_type and self.task_type:
            self.task_type = task_type
            
        try:
            models = get_available_models(task_type)
            
            self.model_combo.clear()
            self.model_combo.addItems(list(models.keys()))
            
            # CRITICAL FIX: Apply previous model selection after updating the model list
            # Priority 1: Use previous_model_selection from config (highest priority)
            model_to_set = None
            if hasattr(self, 'previous_model_selection') and self.previous_model_selection:
                model_to_set = self.previous_model_selection
                print(f"Trying to set model from previous config: {model_to_set}")
            # Priority 2: Use current selection if no previous config
            elif current_model_selection:
                model_to_set = current_model_selection
                print(f"Trying to preserve current model selection: {model_to_set}")
            
            if model_to_set:
                model_index = self.model_combo.findText(model_to_set)
                if model_index != -1:
                    self.model_combo.setCurrentIndex(model_index)
                    print(f"✓ Successfully set model to: {model_to_set}")
                else:
                    print(f"⚠️ Model '{model_to_set}' not found in available models")
                    print(f"Available models: {list(models.keys())}")
            else:
                print("No model selection to apply - using default")
            
        except Exception as e:
            self.status_updated.emit(f"Error updating models: {str(e)}")
            
    def train_model(self):
        """Train the selected model"""
        if self.X is None or self.y is None:
            QMessageBox.warning(self, "Warning", "No data available for training.")
            return
            
        try:
            # Emit training started signal
            self.training_started.emit()
            
            self.status_updated.emit("Starting model training...")
            self.progress_updated.emit(10)
            
            # Get model
            model_name = self.model_combo.currentText()
            task_type = self.task_type_combo.currentText().lower()
            models = get_available_models(task_type)
            
            if model_name not in models:
                QMessageBox.warning(self, "Warning", "Selected model not available.")
                return
                
            model_class = models[model_name]
            
            # Create preprocessing pipeline
            numeric_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # CRITICAL FIX: Identify boolean features (from one-hot encoding)
            # These are typically integer columns with only 0 and 1 values
            boolean_features = []
            for col in numeric_features:
                unique_vals = self.X[col].dropna().unique()
                # Check if column has only 0 and 1 values (typical one-hot encoding result)
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
                    boolean_features.append(col)
            
            # Also include actual boolean columns
            actual_bool_features = self.X.select_dtypes(include='bool').columns.tolist()
            boolean_features.extend(actual_bool_features)
            boolean_features = list(set(boolean_features))  # Remove duplicates
            
            # Ensure boolean features are not treated as standard numeric features
            numeric_features = [col for col in numeric_features if col not in boolean_features]
            
            print(f"=== PREPROCESSING PIPELINE SETUP ===")
            print(f"Total features in X: {len(self.X.columns)}")
            print(f"Numeric features to be scaled: {len(numeric_features)}")
            if numeric_features:
                print(f"  First 5 numeric features: {numeric_features[:5]}")
            print(f"Categorical features to be one-hot encoded: {len(categorical_features)}")
            if categorical_features:
                print(f"  First 5 categorical features: {categorical_features[:5]}")
            print(f"Boolean features to be passed through: {len(boolean_features)}")
            if boolean_features:
                print(f"  First 5 boolean features: {boolean_features[:5]}")
                # Show sample values for boolean features
                for col in boolean_features[:3]:
                    unique_vals = sorted(self.X[col].dropna().unique())
                    print(f"    {col}: unique values = {unique_vals}")
            
            # Verify all features are accounted for
            total_processed = len(numeric_features) + len(categorical_features) + len(boolean_features)
            if total_processed != len(self.X.columns):
                unaccounted = set(self.X.columns) - set(numeric_features) - set(categorical_features) - set(boolean_features)
                print(f"⚠️  WARNING: {len(unaccounted)} features unaccounted for: {list(unaccounted)[:5]}")
            else:
                print(f"✓ All {total_processed} features properly categorized")
            
            # Always use standard preprocessing - no secondary feature selection
            preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features, boolean_features)
            
            print("Using standard preprocessing pipeline (no secondary feature selection)")
            print("Feature selection is handled exclusively by the FeatureModule")
            print("=" * 50)
            
            # Create model with appropriate parameters and CONSISTENT random state
            random_state = self.random_state_spin.value()
            model_instance = self.create_model_instance(model_class, random_state)
            
            print(f"=== MODEL CONFIGURATION CHECK ===")
            print(f"Model: {model_name}")
            print(f"Task type: {task_type}")
            print(f"Model class: {model_class.__name__}")
            print(f"Random state: {random_state} (ensuring reproducibility)")
            
            # Debug: Print ALL model parameters for classification
            if task_type == 'classification':
                print(f"All model parameters:")
                model_params = model_instance.get_params()
                for param, value in sorted(model_params.items()):
                    print(f"  {param}: {value}")
                
                # Specifically check class_weight
                if hasattr(model_instance, 'class_weight'):
                    print(f"✓ class_weight parameter: {model_instance.class_weight}")
                else:
                    print("❌ No class_weight parameter found")
                
                # Print class distribution
                y_for_training = self.y.copy()
                class_counts = pd.Series(y_for_training).value_counts().sort_index()
                print(f"Target class distribution:")
                for class_val, count in class_counts.items():
                    print(f"  Class {class_val}: {count} samples ({count/len(y_for_training)*100:.1f}%)")
                
                # Calculate class imbalance ratio
                min_class_count = class_counts.min()
                max_class_count = class_counts.max()
                imbalance_ratio = max_class_count / min_class_count
                print(f"Class imbalance ratio: {imbalance_ratio:.2f} (max/min)")
                
                if imbalance_ratio > 2.0:
                    print("⚠️  High class imbalance detected! class_weight='balanced' is crucial.")
                else:
                    print("✓ Classes are relatively balanced")
            
            print("=" * 50)
            
            # Special handling for XGBoost to calculate scale_pos_weight for class imbalance
            if 'XGB' in model_name and task_type == 'classification':
                try:
                    # Calculate class distribution for binary classification
                    if len(y_for_training.unique()) == 2:
                        class_counts = y_for_training.value_counts().sort_index()
                        neg_class_count = class_counts.iloc[0]  # class 0
                        pos_class_count = class_counts.iloc[1]  # class 1
                        
                        # CRITICAL FIX: Prevent division by zero
                        if pos_class_count > 0:
                            scale_pos_weight = neg_class_count / pos_class_count
                            
                            # Update model parameters if it's an XGBoost model
                            if hasattr(model_instance, 'set_params'):
                                model_instance.set_params(scale_pos_weight=scale_pos_weight)
                                self.status_updated.emit(f"XGBoost scale_pos_weight set to: {scale_pos_weight:.2f}")
                                print(f"DEBUG: Set XGBoost scale_pos_weight = {scale_pos_weight:.2f}")
                                print(f"DEBUG: Class distribution - 0: {neg_class_count}, 1: {pos_class_count}")
                        else:
                            print("WARNING: Positive class count is 0, cannot calculate scale_pos_weight")
                            self.status_updated.emit("Warning: Cannot calculate XGBoost scale_pos_weight - no positive samples")
                except Exception as e:
                    print(f"Warning: Could not set scale_pos_weight for XGBoost: {e}")
                    self.status_updated.emit(f"Warning: Unable to set XGBoost class weights: {str(e)}")
            
            # Create full pipeline with consistent random state tracking
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model_instance)
            ])
            
            print(f"=== PIPELINE CREATION ===")
            print(f"Pipeline created with:")
            print(f"  - Preprocessor: {type(preprocessor).__name__}")
            print(f"  - Model: {type(model_instance).__name__}")
            print(f"  - Model random_state: {getattr(model_instance, 'random_state', 'Not set')}")
            print("=" * 50)
            
            self.progress_updated.emit(30)
            
            # Split data
            test_size = self.test_size_spin.value()
            random_state = self.random_state_spin.value()
            
            # CRITICAL FIX: Handle target variable encoding AFTER split to prevent data leakage
            y_for_training = self.y.copy()
            target_encoder = None
            
            # Debug: Print target variable info before encoding
            print(f"DEBUG: Target variable info before encoding:")
            print(f"  Type: {type(y_for_training)}")
            print(f"  Dtype: {y_for_training.dtype}")
            print(f"  Unique values: {sorted(y_for_training.unique())}")
            print(f"  Shape: {y_for_training.shape}")
            
            # Handle target encoding for classification if needed
            if task_type == 'classification' and (y_for_training.dtype == 'object' or y_for_training.dtype.name == 'category'):
                print("Target variable needs encoding - will encode after train/test split")
                
                # First split with original labels for stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, y_for_training, test_size=test_size, random_state=random_state,
                    stratify=y_for_training
                )
                
                # Now encode the target variable using only training data
                from sklearn.preprocessing import LabelEncoder
                target_encoder = LabelEncoder()
                
                # Fit encoder on training data only
                y_train_encoded = target_encoder.fit_transform(y_train.astype(str))
                y_test_encoded = target_encoder.transform(y_test.astype(str))
                
                # Convert back to pandas Series with proper index
                y_train = pd.Series(y_train_encoded, index=y_train.index, name=y_train.name)
                y_test = pd.Series(y_test_encoded, index=y_test.index, name=y_test.name)
                
                # Store class names for later use
                self.class_names = target_encoder.classes_
                self.label_encoder = target_encoder
                
                print(f"✓ Target encoded after split - Classes: {self.class_names}")
                print(f"✓ Encoding mapping: {dict(zip(self.class_names, target_encoder.transform(self.class_names)))}")
                
            else:
                # For numeric targets or regression, split normally
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, y_for_training, test_size=test_size, random_state=random_state,
                    stratify=y_for_training if task_type == 'classification' else None
                )
                print("✓ Target variable already numeric or regression task - no encoding needed")
            
            # CRITICAL FIX: Data consistency validation and repair
            print(f"=== DATA CONSISTENCY CHECK ===")
            
            # Validate feature consistency
            missing_in_train = set(self.X.columns) - set(X_train.columns)
            missing_in_test = set(self.X.columns) - set(X_test.columns)
            if missing_in_train or missing_in_test:
                print(f"⚠️  WARNING: Feature inconsistency detected!")
                print(f"  Missing in train: {missing_in_train}")
                print(f"  Missing in test: {missing_in_test}")
                # Fix by ensuring consistent column order
                X_train = X_train.reindex(columns=self.X.columns, fill_value=0)
                X_test = X_test.reindex(columns=self.X.columns, fill_value=0)
                print(f"✓ Fixed feature consistency")
            
            # Check for NaN values
            train_nas = X_train.isnull().sum().sum()
            test_nas = X_test.isnull().sum().sum()
            y_train_nas = y_train.isnull().sum()
            y_test_nas = y_test.isnull().sum()
            
            if train_nas > 0 or test_nas > 0 or y_train_nas > 0 or y_test_nas > 0:
                print(f"⚠️  WARNING: NaN values detected!")
                print(f"  X_train NaNs: {train_nas}, X_test NaNs: {test_nas}")
                print(f"  y_train NaNs: {y_train_nas}, y_test NaNs: {y_test_nas}")
                
                # Remove rows with NaN targets
                if y_train_nas > 0:
                    valid_train = ~y_train.isnull()
                    X_train = X_train[valid_train]
                    y_train = y_train[valid_train]
                    print(f"✓ Removed {y_train_nas} training samples with NaN targets")
                
                if y_test_nas > 0:
                    valid_test = ~y_test.isnull()
                    X_test = X_test[valid_test]
                    y_test = y_test[valid_test]
                    print(f"✓ Removed {y_test_nas} test samples with NaN targets")
            else:
                print(f"✓ No NaN values detected")
            
            # Debug: Print final split info
            print(f"FINAL: After validation and repair:")
            print(f"  X_train shape: {X_train.shape}")
            print(f"  X_test shape: {X_test.shape}")
            print(f"  y_train unique: {sorted(y_train.unique())}")
            print(f"  y_test unique: {sorted(y_test.unique())}")
            print(f"  y_train dtype: {y_train.dtype}")
            print(f"  y_test dtype: {y_test.dtype}")
            print(f"  Random state used: {random_state}")
            print("=" * 50)
            
            # Print class distribution after encoding
            if task_type == 'classification':
                train_class_counts = pd.Series(y_train).value_counts().sort_index()
                test_class_counts = pd.Series(y_test).value_counts().sort_index()
                print(f"Training set class distribution:")
                for class_val, count in train_class_counts.items():
                    class_name = self.class_names[class_val] if self.class_names is not None else str(class_val)
                    print(f"  Class {class_val} ({class_name}): {count} samples ({count/len(y_train)*100:.1f}%)")
                print(f"Test set class distribution:")
                for class_val, count in test_class_counts.items():
                    class_name = self.class_names[class_val] if self.class_names is not None else str(class_val)
                    print(f"  Class {class_val} ({class_name}): {count} samples ({count/len(y_test)*100:.1f}%)")
                
                # Calculate class imbalance ratio
                min_class_count = train_class_counts.min()
                max_class_count = train_class_counts.max()
                imbalance_ratio = max_class_count / min_class_count
                print(f"Training set class imbalance ratio: {imbalance_ratio:.2f} (max/min)")
                
                if imbalance_ratio > 2.0:
                    print("⚠️  High class imbalance detected! class_weight='balanced' is crucial.")
                else:
                    print("✓ Classes are relatively balanced")
            
            print("=" * 50)
            
            # Hyperparameter optimization
            if self.enable_hpo.isChecked():
                self.status_updated.emit("Optimizing hyperparameters...")
                self.progress_updated.emit(50)
                
                # Get hyperparameter grid - prefer custom spaces
                param_grid = None
                if model_name in self.custom_param_spaces:
                    param_grid = {}
                    for param_name, space_config in self.custom_param_spaces[model_name].items():
                        if isinstance(space_config, dict) and space_config.get('type') == 'range':
                            if self.hpo_method_combo.currentText() == "Grid Search":
                                # For grid search, create discrete values
                                min_val, max_val = space_config['min'], space_config['max']
                                if isinstance(min_val, int):
                                    param_grid[param_name] = list(range(int(min_val), int(max_val) + 1, max(1, (int(max_val) - int(min_val)) // 5)))
                                else:
                                    param_grid[param_name] = np.linspace(min_val, max_val, 6).tolist()
                            else:
                                # For random search, use distributions
                                from scipy.stats import uniform, randint
                                if isinstance(space_config['min'], int):
                                    param_grid[param_name] = randint(int(space_config['min']), int(space_config['max']) + 1)
                                else:
                                    param_grid[param_name] = uniform(space_config['min'], space_config['max'] - space_config['min'])
                        else:
                            param_grid[param_name] = space_config
                else:
                    # Use default hyperparameters
                    param_grid = get_default_hyperparameters(model_name, task_type)
                
                if param_grid:  # Only do HPO if parameters are available
                    # Add model prefix to parameters
                    param_grid_prefixed = {f'model__{k}': v for k, v in param_grid.items()}
                    
                    # Choose search method
                    search_method = self.hpo_method_combo.currentText()
                    
                    if search_method == "Grid Search":
                        search = GridSearchCV(
                            pipeline, param_grid_prefixed,
                            cv=self.cv_folds_spin.value(),
                            scoring='accuracy' if task_type == 'classification' else 'r2',
                            n_jobs=-1
                        )
                    elif search_method == "Random Search":
                        search = RandomizedSearchCV(
                            pipeline, param_grid_prefixed,
                            cv=self.cv_folds_spin.value(),
                            scoring='accuracy' if task_type == 'classification' else 'r2',
                            n_jobs=-1, n_iter=50,
                            random_state=random_state
                        )
                    else:  # Bayesian Search
                        try:
                            from skopt import BayesSearchCV
                            from skopt.space import Real, Integer, Categorical
                            
                            # Convert parameter grid to skopt format
                            skopt_space = {}
                            for param_name, param_values in param_grid_prefixed.items():
                                if isinstance(param_values, list):
                                    # CRITICAL FIX: Check for boolean values BEFORE checking for integers
                                    # This prevents [True, False] from being wrongly converted to Integer(0, 1)
                                    if all(isinstance(v, bool) for v in param_values):
                                        # Handle boolean parameters correctly
                                        skopt_space[param_name] = Categorical(param_values)
                                        print(f"DEBUG: Boolean parameter {param_name}: {param_values} -> Categorical")
                                    elif all(isinstance(v, (int, np.integer)) for v in param_values) and not all(isinstance(v, bool) for v in param_values):
                                        # Handle integer parameters (but exclude booleans)
                                        skopt_space[param_name] = Integer(min(param_values), max(param_values))
                                        print(f"DEBUG: Integer parameter {param_name}: {param_values} -> Integer({min(param_values)}, {max(param_values)})")
                                    elif all(isinstance(v, (float, np.floating)) for v in param_values):
                                        # Handle float parameters
                                        skopt_space[param_name] = Real(min(param_values), max(param_values))
                                        print(f"DEBUG: Float parameter {param_name}: {param_values} -> Real({min(param_values)}, {max(param_values)})")
                                    else:
                                        # Handle categorical/mixed parameters
                                        skopt_space[param_name] = Categorical(param_values)
                                        print(f"DEBUG: Categorical parameter {param_name}: {param_values} -> Categorical")
                                else:
                                    skopt_space[param_name] = param_values
                            
                            search = BayesSearchCV(
                                pipeline, skopt_space,
                                cv=self.cv_folds_spin.value(),
                                scoring='accuracy' if task_type == 'classification' else 'r2',
                                n_jobs=-1, n_iter=30,
                                random_state=random_state
                            )
                        except ImportError:
                            QMessageBox.warning(self, "Warning", "scikit-optimize not installed. Using Random Search instead.")
                            search = RandomizedSearchCV(
                                pipeline, param_grid_prefixed,
                                cv=self.cv_folds_spin.value(),
                                scoring='accuracy' if task_type == 'classification' else 'r2',
                                n_jobs=-1, n_iter=50,
                                random_state=random_state
                            )
                        
                    search.fit(X_train, y_train)
                    self.trained_pipeline = search.best_estimator_

                    # CRITICAL FIX: Ensure QuantileRegressor gets correct parameter types after HPO
                    self._fix_quantile_regressor_params(self.trained_pipeline)

                    # Store HPO results for visualization
                    self.hpo_results = {
                        'search_object': search,
                        'cv_results': search.cv_results_,
                        'best_params': search.best_params_,
                        'best_score': search.best_score_,
                        'search_method': search_method
                    }
                else:
                    # Train without HPO if no parameters available
                    pipeline.fit(X_train, y_train)
                    self.trained_pipeline = pipeline

            else:
                # Train without HPO
                self.status_updated.emit("Training model...")
                pipeline.fit(X_train, y_train)
                self.trained_pipeline = pipeline

            # CRITICAL FIX: Always fix QuantileRegressor parameters after training
            self._fix_quantile_regressor_params(self.trained_pipeline)
                
            self.progress_updated.emit(80)
            
            # Evaluate model
            self.status_updated.emit("Evaluating model...")
            
            # Predictions on test set
            y_pred = self.trained_pipeline.predict(X_test)
            
            if task_type == 'classification':
                try:
                    y_pred_proba = self.trained_pipeline.predict_proba(X_test)
                except:
                    y_pred_proba = None
                    
                metrics = evaluate_classification_model(y_test, y_pred, y_pred_proba)
            else:
                metrics = evaluate_regression_model(y_test, y_pred)
                
            # Also get training predictions for comparison
            y_train_pred = self.trained_pipeline.predict(X_train)
            
            # Calculate cross-validation score on the same data for comparison
            if self.enable_hpo.isChecked() and self.hpo_results:
                cv_score = self.hpo_results['best_score']
                test_score = metrics.get('accuracy', metrics.get('r2', 0))
                
                # Alert if there's a significant difference
                score_diff = abs(cv_score - test_score)
                if score_diff > 0.1:  # More than 10% difference
                    warning_msg = f"⚠️ Performance Difference Warning:\n"
                    warning_msg += f"Cross validation score: {cv_score:.4f}\n"
                    warning_msg += f"Test set score: {test_score:.4f}\n"
                    warning_msg += f"Difference: {score_diff:.4f}\n\n"
                    warning_msg += f"Possible reasons:\n"
                    warning_msg += f"1. Data distribution is not uniform\n"
                    warning_msg += f"2. Test set is too small\n"
                    warning_msg += f"3. Overfitting to cross-validation strategy\n\n"
                    warning_msg += f"Suggestions: Try different random_state or increase data size"
                    
                    QMessageBox.warning(self, "Performance Difference Warning", warning_msg)
                    self.status_updated.emit(f"CV score: {cv_score:.4f}, Test score: {test_score:.4f}")
            
            # Store results
            eval_results = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_train_pred': y_train_pred,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba if task_type == 'classification' else None,
                'metrics': metrics,
                'task_type': task_type,
                'model_name': model_name
            }
            
            # Enhanced evaluation based on strategy
            eval_strategy = self.eval_strategy_combo.currentText()
            if eval_strategy == "Multiple Split Validation (Recommended) ":
                self.status_updated.emit("Performing multiple split validation...")
                multiple_scores = self.perform_multiple_split_validation(
                    self.trained_pipeline, self.X, self.y, task_type
                )
                eval_results['multiple_scores'] = multiple_scores
                eval_results['robust_score'] = np.mean(multiple_scores)
                eval_results['score_std'] = np.std(multiple_scores)
                
                self.status_updated.emit(f"Multiple split validation average score: {eval_results['robust_score']:.4f} ± {eval_results['score_std']:.4f}")
                
            elif eval_strategy == "Nested Cross Validation (Most Reliable)":
                self.status_updated.emit("Performing nested cross validation...")
                nested_score = self.perform_nested_cv_validation(
                    pipeline, self.X, self.y, task_type, param_grid_prefixed if self.enable_hpo.isChecked() else None
                )
                eval_results['nested_cv_score'] = nested_score
                
                self.status_updated.emit(f"Nested cross validation score: {nested_score:.4f}")
            
            self.evaluation_results = eval_results
            
            # Create results tabs
            self.create_results_tabs()
            
            # Enable buttons
            self.save_model_btn.setEnabled(True)
            self.proceed_btn.setEnabled(True)
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Model training completed successfully")
            
            QMessageBox.information(self, "Success", "Model trained successfully!")
            
        except Exception as e:
            # Check if user cancelled quantile input
            if "User cancelled quantile input" in str(e):
                self.status_updated.emit("Training cancelled by user")
                self.progress_updated.emit(0)
                return
            else:
                QMessageBox.critical(self, "Error", f"Error during training: {str(e)}")
                self.progress_updated.emit(0)
                self.status_updated.emit("Training failed")
            
    def create_results_tabs(self):
        """Create results visualization tabs"""
        # Clear existing tabs
        self.right_panel.clear()
        self.results_tabs = {}
        
        # Metrics tab
        self.create_metrics_tab()
        
        # HPO visualization tab (if HPO was performed)
        if self.hpo_results:
            self.create_hpo_viz_tab()
        
        # Visualization tabs based on task type
        if self.evaluation_results['task_type'] == 'classification':
            self.create_classification_viz_tabs()
        else:
            self.create_regression_viz_tabs()
            
    def create_metrics_tab(self):
        """Create metrics display tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        metrics_text = QTextEdit()
        metrics_text.setReadOnly(True)
        
        # Format metrics
        metrics = self.evaluation_results['metrics']
        text = "Model Evaluation Metrics:\n"
        text += "=" * 40 + "\n\n"
        
        for metric, value in metrics.items():
            text += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
            
        metrics_text.setText(text)
        layout.addWidget(metrics_text)
        
        self.right_panel.addTab(widget, "Metrics")
        self.results_tabs["Metrics"] = widget
    
    def create_hpo_viz_tab(self):
        """Create hyperparameter optimization visualization tab"""
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # Export button
            export_btn = QPushButton("Export HPO Results")
            export_btn.clicked.connect(lambda: self.export_hpo_results())
            layout.addWidget(export_btn)
            
            # Create HPO visualization
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            
            fig = Figure(figsize=(15, 12))
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, widget)
            
            # Get results
            cv_results = self.hpo_results['cv_results']
            results_df = pd.DataFrame(cv_results)
            search_method = self.hpo_results['search_method']
            
            # 1. Optimization progress
            ax1 = fig.add_subplot(2, 3, 1)
            scores = results_df['mean_test_score'].values
            iterations = range(len(scores))
            ax1.plot(iterations, scores, 'b-', alpha=0.7, linewidth=2, label='CV Score')
            ax1.fill_between(iterations, 
                           scores - results_df['std_test_score'].values,
                           scores + results_df['std_test_score'].values,
                           alpha=0.3, color='blue')
            
            # Best score marker
            best_idx = np.argmax(scores)
            ax1.plot(best_idx, scores[best_idx], 'ro', markersize=8, 
                    label=f'Best: {scores[best_idx]:.4f}')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('CV Score')
            ax1.set_title(f'{search_method} Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Score distribution
            ax2 = fig.add_subplot(2, 3, 2)
            ax2.hist(scores, bins=min(20, len(scores)//2), alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(scores[best_idx], color='red', linestyle='--', linewidth=2, 
                       label=f'Best: {scores[best_idx]:.4f}')
            ax2.set_xlabel('CV Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Parameter importance (correlation with score)
            param_cols = [col for col in results_df.columns if col.startswith('param_')]
            if len(param_cols) > 0:
                ax3 = fig.add_subplot(2, 3, 3)
                param_importance = {}
                
                for param_col in param_cols[:10]:  # Limit to top 10 parameters
                    param_name = param_col.replace('param_model__', '').replace('param_', '')
                    param_values = results_df[param_col].values
                    
                    # Calculate correlation with scores
                    try:
                        # Handle different parameter types
                        if pd.api.types.is_numeric_dtype(param_values):
                            correlation = abs(np.corrcoef(param_values, scores)[0, 1])
                        else:
                            # For categorical parameters, use ANOVA F-statistic
                            from scipy.stats import f_oneway
                            groups = [scores[param_values == val] for val in np.unique(param_values)]
                            groups = [g for g in groups if len(g) > 0]
                            if len(groups) > 1:
                                f_stat, _ = f_oneway(*groups)
                                correlation = f_stat / (f_stat + 100)  # Normalize
                            else:
                                correlation = 0
                        
                        if not np.isnan(correlation):
                            param_importance[param_name] = correlation
                    except:
                        pass
                
                if param_importance:
                    params = list(param_importance.keys())
                    importances = list(param_importance.values())
                    y_pos = np.arange(len(params))
                    ax3.barh(y_pos, importances, color='lightcoral')
                    ax3.set_yticks(y_pos)
                    ax3.set_yticklabels(params)
                    ax3.set_xlabel('Importance Score')
                    ax3.set_title('Parameter Importance')
                    ax3.grid(True, alpha=0.3)
            
            # 4. Best parameters display
            ax4 = fig.add_subplot(2, 3, 4)
            ax4.axis('off')
            best_params = self.hpo_results['best_params']
            params_text = f"Best Parameters ({search_method}):\n\n"
            for i, (param, value) in enumerate(best_params.items()):
                param_clean = param.replace('model__', '')
                if i < 15:  # Limit display
                    params_text += f"{param_clean}: {value}\n"
                elif i == 15:
                    params_text += f"... and {len(best_params) - 15} more"
                    break
            
            ax4.text(0.05, 0.95, params_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 5. Parameter evolution (for first few parameters)
            if len(param_cols) > 0:
                ax5 = fig.add_subplot(2, 3, 5)
                
                # Show evolution of first numeric parameter
                numeric_param_col = None
                for col in param_cols[:5]:
                    if pd.api.types.is_numeric_dtype(results_df[col]):
                        numeric_param_col = col
                        break
                
                if numeric_param_col:
                    param_values = results_df[numeric_param_col].values
                    ax5.scatter(param_values, scores, alpha=0.6, c=iterations, cmap='viridis')
                    ax5.set_xlabel(numeric_param_col.replace('param_model__', '').replace('param_', ''))
                    ax5.set_ylabel('CV Score')
                    ax5.set_title('Parameter vs Score')
                    ax5.grid(True, alpha=0.3)
                    
                    # Add colorbar
                    cbar = fig.colorbar(ax5.collections[0], ax=ax5)
                    cbar.set_label('Iteration')
            
            # 6. Summary statistics
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.axis('off')
            
            summary_text = f"Optimization Summary:\n\n"
            summary_text += f"Method: {search_method}\n"
            summary_text += f"Total iterations: {len(results_df)}\n"
            summary_text += f"Best score: {self.hpo_results['best_score']:.4f}\n"
            summary_text += f"Score std: {scores.std():.4f}\n"
            summary_text += f"Score range: {scores.min():.4f} - {scores.max():.4f}\n"
            summary_text += f"Parameters optimized: {len([c for c in param_cols])}\n"
            
            # Add timing info if available
            if 'mean_fit_time' in results_df.columns:
                summary_text += f"Avg fit time: {results_df['mean_fit_time'].mean():.2f}s\n"
            
            ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            fig.tight_layout()
            
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            self.right_panel.addTab(widget, "HPO Results")
            self.results_tabs["HPO Results"] = widget
            
        except Exception as e:
            print(f"Error creating HPO visualization: {e}")
            QMessageBox.warning(self, "Warning", f"Could not create HPO visualization: {str(e)}")
    
    def export_hpo_results(self):
        """Export hyperparameter optimization results"""
        if not self.hpo_results:
            return
            
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export HPO Results", 
                f"hpo_results_{self.hpo_results['search_method'].lower().replace(' ', '_')}.csv",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
            )
            
            if file_path:
                results_df = pd.DataFrame(self.hpo_results['cv_results'])
                
                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        results_df.to_excel(writer, sheet_name='HPO_Results', index=False)
                        
                        # Add summary sheet
                        summary_data = {
                            'Metric': ['Search Method', 'Best Score', 'Total Iterations', 'Best Parameters'],
                            'Value': [
                                self.hpo_results['search_method'],
                                self.hpo_results['best_score'],
                                len(results_df),
                                str(self.hpo_results['best_params'])
                            ]
                        }
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                else:
                    results_df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"HPO results exported to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export HPO results: {str(e)}")
        
    def create_classification_viz_tabs(self):
        """Create comprehensive classification visualization tabs with interactive plots"""
        y_test = self.evaluation_results['y_test']
        y_pred = self.evaluation_results['y_pred']
        y_pred_proba = self.evaluation_results['y_pred_proba']
        model_name = self.evaluation_results['model_name']
        
        # 1. Enhanced Confusion Matrix with detailed metrics
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # Export button
            export_btn = QPushButton("Export Confusion Matrix")
            export_btn.clicked.connect(lambda: self.export_confusion_matrix())
            layout.addWidget(export_btn)
            
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from sklearn.metrics import confusion_matrix, classification_report
            import seaborn as sns
            
            fig = Figure(figsize=(14, 10))
            
            # Create labels for classes
            if hasattr(self, 'class_names') and self.class_names is not None:
                # Use original class names for display
                classes = self.class_names
                # But ensure y_test and y_pred use the same encoding
                y_test_display = y_test
                y_pred_display = y_pred
            else:
                # Use numeric labels
                classes = sorted(list(set(y_test) | set(y_pred)))
                y_test_display = y_test
                y_pred_display = y_pred
            
            # Confusion matrix heatmap
            ax1 = fig.add_subplot(221)
            cm = confusion_matrix(y_test_display, y_pred_display)
            
            # Plot heatmap
            if sns is not None:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=classes, yticklabels=classes, ax=ax1)
            else:
                # Fallback to matplotlib
                im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
                ax1.figure.colorbar(im, ax=ax1)
                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax1.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
                ax1.set_xticks(range(len(classes)))
                ax1.set_yticks(range(len(classes)))
                ax1.set_xticklabels(classes)
                ax1.set_yticklabels(classes)
            ax1.set_title(f'Confusion Matrix - {model_name}')
            ax1.set_xlabel('Predicted Label')
            ax1.set_ylabel('True Label')
            
            # Normalized confusion matrix
            ax2 = fig.add_subplot(222)
            cm_norm = confusion_matrix(y_test_display, y_pred_display, normalize='true')
            if sns is not None:
                sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Oranges',
                           xticklabels=classes, yticklabels=classes, ax=ax2)
            else:
                # Fallback to matplotlib
                im = ax2.imshow(cm_norm, interpolation='nearest', cmap='Oranges')
                ax2.figure.colorbar(im, ax=ax2)
                # Add text annotations
                thresh = cm_norm.max() / 2.
                for i in range(cm_norm.shape[0]):
                    for j in range(cm_norm.shape[1]):
                        ax2.text(j, i, format(cm_norm[i, j], '.3f'),
                               ha="center", va="center",
                               color="white" if cm_norm[i, j] > thresh else "black")
                ax2.set_xticks(range(len(classes)))
                ax2.set_yticks(range(len(classes)))
                ax2.set_xticklabels(classes)
                ax2.set_yticklabels(classes)
            ax2.set_title('Normalized Confusion Matrix (by True Class)')
            ax2.set_xlabel('Predicted Label')
            ax2.set_ylabel('True Label')
            
            # Classification report as text
            ax3 = fig.add_subplot(212)
            ax3.axis('off')
            
            # Get classification report as dict
            from sklearn.metrics import classification_report
            report = classification_report(y_test_display, y_pred_display, target_names=[str(c) for c in classes], output_dict=True)
            
            # Create formatted text
            report_text = "Classification Report:\n\n"
            report_text += f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
            report_text += "-" * 60 + "\n"
            
            for class_name in classes:
                class_str = str(class_name)
                if class_str in report:
                    metrics = report[class_str]
                    report_text += f"{class_str:<10} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {metrics['support']:<10.0f}\n"
            
            report_text += "-" * 60 + "\n"
            report_text += f"{'Accuracy':<10} {'':<10} {'':<10} {report['accuracy']:<10.3f} {report['macro avg']['support']:<10.0f}\n"
            report_text += f"{'Macro Avg':<10} {report['macro avg']['precision']:<10.3f} {report['macro avg']['recall']:<10.3f} {report['macro avg']['f1-score']:<10.3f} {report['macro avg']['support']:<10.0f}\n"
            report_text += f"{'Weighted Avg':<10} {report['weighted avg']['precision']:<10.3f} {report['weighted avg']['recall']:<10.3f} {report['weighted avg']['f1-score']:<10.3f} {report['weighted avg']['support']:<10.0f}\n"
            
            ax3.text(0.05, 0.95, report_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            
            # Add navigation toolbar for interactivity
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            toolbar = NavigationToolbar(canvas, widget)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            self.right_panel.addTab(widget, "Confusion Matrix & Report")
            self.results_tabs["Confusion Matrix & Report"] = widget
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            
        # 2. ROC Curves (Multi-class and Binary)
        if y_pred_proba is not None:
            try:
                widget = QWidget()
                layout = QVBoxLayout(widget)
                
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from sklearn.metrics import roc_curve, auc, roc_auc_score
                from sklearn.preprocessing import label_binarize
                import matplotlib.pyplot as plt
                
                fig = Figure(figsize=(14, 10))
                
                classes = sorted(list(set(y_test_display)))
                n_classes = len(classes)
                
                if n_classes == 2:
                    # Binary classification ROC
                    ax = fig.add_subplot(111)
                    
                    if y_pred_proba.ndim == 2:
                        fpr, tpr, _ = roc_curve(y_test_display, y_pred_proba[:, 1])
                        roc_auc = auc(fpr, tpr)
                    else:
                        fpr, tpr, _ = roc_curve(y_test_display, y_pred_proba)
                        roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC Curve (AUC = {roc_auc:.3f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                           label='Random Classifier')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'ROC Curve - {model_name}')
                    ax.legend(loc="lower right")
                    ax.grid(True, alpha=0.3)
                    
                else:
                    # Multi-class ROC
                    # Binarize the output
                    y_test_bin = label_binarize(y_test_display, classes=classes)
                    
                    # Compute ROC curve and ROC area for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
                    # Plot all ROC curves
                    ax = fig.add_subplot(111)
                    
                    # Plot micro-average ROC curve
                    ax.plot(fpr["micro"], tpr["micro"],
                           label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
                           color='deeppink', linestyle=':', linewidth=4)
                    
                    # Plot ROC curve for each class
                    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
                    for i, color in zip(range(n_classes), colors):
                        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                               label=f'Class {classes[i]} (AUC = {roc_auc[i]:.3f})')
                    
                    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'Multi-class ROC Curves - {model_name}')
                    ax.legend(loc="lower right")
                    ax.grid(True, alpha=0.3)
                
                fig.tight_layout()
                canvas = FigureCanvas(fig)
                
                # Add navigation toolbar
                toolbar = NavigationToolbar(canvas, widget)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
                
                self.right_panel.addTab(widget, "ROC Curves")
                self.results_tabs["ROC Curves"] = widget
            except Exception as e:
                print(f"Error creating ROC curves: {e}")
        
        # 3. Precision-Recall Curves
        if y_pred_proba is not None:
            try:
                widget = QWidget()
                layout = QVBoxLayout(widget)
                
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from sklearn.metrics import precision_recall_curve, average_precision_score
                from sklearn.preprocessing import label_binarize
                import matplotlib.pyplot as plt
                
                fig = Figure(figsize=(14, 10))
                
                classes = sorted(list(set(y_test_display)))
                n_classes = len(classes)
                
                if n_classes == 2:
                    # Binary classification
                    ax = fig.add_subplot(111)
                    
                    if y_pred_proba.ndim == 2:
                        precision, recall, _ = precision_recall_curve(y_test_display, y_pred_proba[:, 1])
                        avg_precision = average_precision_score(y_test_display, y_pred_proba[:, 1])
                    else:
                        precision, recall, _ = precision_recall_curve(y_test_display, y_pred_proba)
                        avg_precision = average_precision_score(y_test_display, y_pred_proba)
                    
                    ax.plot(recall, precision, color='darkorange', lw=2,
                           label=f'Precision-Recall (AP = {avg_precision:.3f})')
                    
                    # Random classifier baseline
                    baseline = len(y_test_display[y_test_display == 1]) / len(y_test_display)
                    ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                              label=f'Random Classifier (AP = {baseline:.3f})')
                    
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    ax.set_title(f'Precision-Recall Curve - {model_name}')
                    ax.legend(loc="lower left")
                    ax.grid(True, alpha=0.3)
                    
                else:
                    # Multi-class
                    y_test_bin = label_binarize(y_test_display, classes=classes)
                    
                    ax = fig.add_subplot(111)
                    
                    # Plot precision-recall curve for each class
                    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
                    for i, color in zip(range(n_classes), colors):
                        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
                        avg_precision = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
                        
                        ax.plot(recall, precision, color=color, lw=2,
                               label=f'Class {classes[i]} (AP = {avg_precision:.3f})')
                    
                    # Micro-average
                    precision_micro, recall_micro, _ = precision_recall_curve(y_test_bin.ravel(), y_pred_proba.ravel())
                    avg_precision_micro = average_precision_score(y_test_bin, y_pred_proba, average="micro")
                    ax.plot(recall_micro, precision_micro, color='gold', lw=2, linestyle=':',
                           label=f'Micro-average (AP = {avg_precision_micro:.3f})')
                    
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    ax.set_title(f'Multi-class Precision-Recall Curves - {model_name}')
                    ax.legend(loc="lower left")
                    ax.grid(True, alpha=0.3)
                
                fig.tight_layout()
                canvas = FigureCanvas(fig)
                
                # Add navigation toolbar
                toolbar = NavigationToolbar(canvas, widget)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
                
                self.right_panel.addTab(widget, "Precision-Recall Curves")
                self.results_tabs["Precision-Recall Curves"] = widget
            except Exception as e:
                print(f"Error creating precision-recall curves: {e}")
        
        # 4. Feature Importance (if available)
        try:
            model = self.trained_pipeline.named_steps['model']
            feature_names = self.X.columns.tolist()
            
            # Try to get feature importance
            importance = None
            importance_type = ""
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_type = "Feature Importances"
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                if len(model.coef_.shape) == 1:
                    importance = np.abs(model.coef_)
                else:
                    importance = np.abs(model.coef_).mean(axis=0)
                importance_type = "Coefficient Magnitudes"
            
            if importance is not None:
                widget = QWidget()
                layout = QVBoxLayout(widget)
                
                # Create importance DataFrame for processing
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                # Control buttons
                button_layout = QHBoxLayout()
                
                # Export button
                export_btn = QPushButton("Export Feature Importance")
                export_btn.clicked.connect(lambda: self.export_feature_importance(importance_df))
                button_layout.addWidget(export_btn)
                
                # Toggle aggregation button
                self.aggregate_training_features = True  # Default to aggregated view
                toggle_btn = QPushButton("Show Individual Features")
                toggle_btn.clicked.connect(lambda: self.toggle_training_feature_aggregation(widget, importance_df, toggle_btn, importance_type, model_name))
                button_layout.addWidget(toggle_btn)
                
                layout.addLayout(button_layout)
                
                # Create importance plot using the enhanced function
                from utils.plot_utils import plot_feature_importance
                fig, canvas = plot_feature_importance(importance_df, max_features=20, 
                                                    aggregate_encoded_features=self.aggregate_training_features)
                
                # Update title to include model info
                ax = fig.get_axes()[0]
                current_title = ax.get_title()
                ax.set_title(f'{current_title} - {model_name}')
                ax.set_xlabel(importance_type)
                
                # Add navigation toolbar
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                toolbar = NavigationToolbar(canvas, widget)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
                
                # Add summary information
                summary_text = QTextEdit()
                summary_text.setMaximumHeight(100)
                summary_text.setReadOnly(True)
                
                summary = f"Feature Importance Summary ({model_name}):\n"
                summary += f"• Total features: {len(importance_df)}\n"
                summary += f"• Top importance: {importance_df['importance'].max():.4f}\n"
                summary += f"• Average importance: {importance_df['importance'].mean():.4f}\n"
                summary += f"• Importance type: {importance_type}\n"
                
                # Check for encoded features
                encoded_features = [f for f in feature_names if '_' in f and f.split('_')[-1].isdigit()]
                if encoded_features:
                    summary += f"• Detected {len(encoded_features)} likely encoded features\n"
                    summary += f"• View mode: {'Aggregated by original features' if self.aggregate_training_features else 'Individual encoded features'}\n"
                
                summary_text.setText(summary)
                layout.addWidget(summary_text)
                
                self.right_panel.addTab(widget, "Feature Importance")
                self.results_tabs["Feature Importance"] = widget
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
    
    def toggle_training_feature_aggregation(self, widget, importance_df, toggle_btn, importance_type, model_name):
        """Toggle between aggregated and individual feature views in training module"""
        try:
            self.aggregate_training_features = not self.aggregate_training_features
            
            # Update button text
            if self.aggregate_training_features:
                toggle_btn.setText("Show Individual Features")
            else:
                toggle_btn.setText("Show Aggregated Features")
            
            # Recreate the plot
            from utils.plot_utils import plot_feature_importance
            fig, new_canvas = plot_feature_importance(importance_df, max_features=20, 
                                                    aggregate_encoded_features=self.aggregate_training_features)
            
            # Update title to include model info
            ax = fig.get_axes()[0]
            current_title = ax.get_title()
            ax.set_title(f'{current_title} - {model_name}')
            ax.set_xlabel(importance_type)
            
            # Replace the canvas in the layout
            layout = widget.layout()
            
            # Find and remove old canvas and toolbar
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and hasattr(item.widget(), 'figure'):
                    old_canvas = item.widget()
                    layout.removeWidget(old_canvas)
                    old_canvas.deleteLater()
                elif item and item.widget() and hasattr(item.widget(), 'canvas'):
                    old_toolbar = item.widget()
                    layout.removeWidget(old_toolbar)
                    old_toolbar.deleteLater()
            
            # Add new canvas and toolbar
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            new_toolbar = NavigationToolbar(new_canvas, widget)
            layout.insertWidget(-1, new_toolbar)
            layout.insertWidget(-1, new_canvas)
            
            # Update summary text
            summary_widget = None
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and isinstance(item.widget(), QTextEdit):
                    summary_widget = item.widget()
                    break
            
            if summary_widget:
                summary = f"Feature Importance Summary ({model_name}):\n"
                summary += f"• Total features: {len(importance_df)}\n"
                summary += f"• Top importance: {importance_df['importance'].max():.4f}\n"
                summary += f"• Average importance: {importance_df['importance'].mean():.4f}\n"
                summary += f"• Importance type: {importance_type}\n"
                
                encoded_features = [f for f in importance_df['feature'] if '_' in f and f.split('_')[-1].isdigit()]
                if encoded_features:
                    summary += f"• Detected {len(encoded_features)} likely encoded features\n"
                    summary += f"• View mode: {'Aggregated by original features' if self.aggregate_training_features else 'Individual encoded features'}\n"
                
                summary_widget.setText(summary)
            
        except Exception as e:
            print(f"Error toggling training feature aggregation: {e}")
    
    def export_feature_importance(self, importance_df):
        """Export feature importance data"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Feature Importance", 
                "feature_importance.xlsx",
                "Excel files (*.xlsx);;CSV files (*.csv)"
            )
            
            if file_path:
                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
                        
                        # Add aggregated version if applicable
                        try:
                            from utils.feature_name_utils import aggregate_feature_importance_by_original
                            aggregated_df = aggregate_feature_importance_by_original(importance_df)
                            aggregated_df.to_excel(writer, sheet_name='Aggregated_Importance', index=False)
                        except ImportError:
                            pass
                else:
                    importance_df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"Feature importance exported to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export feature importance: {str(e)}")
        
        # 5. Class Distribution and Prediction Analysis
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            fig = Figure(figsize=(14, 10))
            
            classes = sorted(list(set(y_test_display)))
            
            # Class distribution comparison
            ax1 = fig.add_subplot(221)
            
            unique_true, counts_true = np.unique(y_test_display, return_counts=True)
            unique_pred, counts_pred = np.unique(y_pred_display, return_counts=True)
            
            x = np.arange(len(classes))
            width = 0.35
            
            # Ensure all classes are represented
            true_counts = [counts_true[list(unique_true).index(c)] if c in unique_true else 0 for c in classes]
            pred_counts = [counts_pred[list(unique_pred).index(c)] if c in unique_pred else 0 for c in classes]
            
            ax1.bar(x - width/2, true_counts, width, label='True', alpha=0.7, color='lightblue')
            ax1.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7, color='orange')
            
            ax1.set_xlabel('Classes')
            ax1.set_ylabel('Count')
            ax1.set_title('Class Distribution: True vs Predicted')
            ax1.set_xticks(x)
            ax1.set_xticklabels(classes)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Per-class accuracy
            ax2 = fig.add_subplot(222)
            
            from sklearn.metrics import classification_report
            report = classification_report(y_test_display, y_pred_display, target_names=[str(c) for c in classes], output_dict=True)
            
            class_recalls = [report[str(c)]['recall'] for c in classes if str(c) in report]
            
            bars = ax2.bar(classes, class_recalls, alpha=0.7, color='lightgreen')
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('Recall (Sensitivity)')
            ax2.set_title('Per-Class Recall')
            ax2.set_ylim([0, 1])
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, recall in zip(bars, class_recalls):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{recall:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Prediction confidence distribution (if probabilities available)
            if y_pred_proba is not None:
                ax3 = fig.add_subplot(223)
                
                # Get maximum probability for each prediction
                max_probs = np.max(y_pred_proba, axis=1)
                
                ax3.hist(max_probs, bins=30, alpha=0.7, color='purple', edgecolor='black')
                ax3.set_xlabel('Maximum Prediction Probability')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Prediction Confidence Distribution')
                ax3.grid(True, alpha=0.3)
                
                # Add statistics
                ax3.axvline(np.mean(max_probs), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(max_probs):.3f}')
                ax3.axvline(np.median(max_probs), color='orange', linestyle='--', 
                           label=f'Median: {np.median(max_probs):.3f}')
                ax3.legend()
            
            # Error analysis by confidence (if probabilities available)
            if y_pred_proba is not None:
                ax4 = fig.add_subplot(224)
                
                max_probs = np.max(y_pred_proba, axis=1)
                correct_predictions = (y_test_display == y_pred_display)
                
                # Bin predictions by confidence
                confidence_bins = np.linspace(0, 1, 11)
                bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
                
                accuracy_by_confidence = []
                for i in range(len(confidence_bins) - 1):
                    mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
                    if mask.sum() > 0:
                        accuracy_by_confidence.append(correct_predictions[mask].mean())
                    else:
                        accuracy_by_confidence.append(0)
                
                ax4.plot(bin_centers, accuracy_by_confidence, 'o-', color='darkred', linewidth=2, markersize=6)
                ax4.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, label='Perfect Calibration')
                ax4.set_xlabel('Confidence')
                ax4.set_ylabel('Accuracy')
                ax4.set_title('Reliability Diagram')
                ax4.set_xlim([0, 1])
                ax4.set_ylim([0, 1])
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            
            # Add navigation toolbar
            toolbar = NavigationToolbar(canvas, widget)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            self.right_panel.addTab(widget, "Classification Analysis")
            self.results_tabs["Classification Analysis"] = widget
        except Exception as e:
            print(f"Error creating classification analysis: {e}")
    
    def create_regression_viz_tabs(self):
        """Create enhanced regression visualization tabs with train/test comparison"""
        y_train = self.evaluation_results['y_train']
        y_test = self.evaluation_results['y_test']
        y_train_pred = self.evaluation_results['y_train_pred']
        y_pred = self.evaluation_results['y_pred']
        model_name = self.evaluation_results['model_name']
        
        # Enhanced Prediction vs Actual plot with train/test separation
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # Export button
            export_btn = QPushButton("Export Regression Analysis")
            export_btn.clicked.connect(lambda: self.export_regression_results())
            layout.addWidget(export_btn)
            
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            fig = Figure(figsize=(15, 10))
            
            # Main prediction vs actual plot
            ax1 = fig.add_subplot(221)
            
            # Plot training set
            ax1.scatter(y_train, y_train_pred, alpha=0.6, c='blue', s=30, label='Training Set', edgecolors='navy', linewidth=0.5)
            # Plot test set
            ax1.scatter(y_test, y_pred, alpha=0.8, c='red', s=30, label='Test Set', edgecolors='darkred', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(min(y_train.min(), y_test.min()), min(y_train_pred.min(), y_pred.min()))
            max_val = max(max(y_train.max(), y_test.max()), max(y_train_pred.max(), y_pred.max()))
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
            
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title(f'Prediction vs Actual - {model_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Calculate R² for both sets
            from sklearn.metrics import r2_score
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_pred)
            ax1.text(0.05, 0.95, f'R² Train: {r2_train:.3f}\nR² Test: {r2_test:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Residuals plot
            ax2 = fig.add_subplot(222)
            residuals_train = y_train - y_train_pred
            residuals_test = y_test - y_pred
            
            ax2.scatter(y_train_pred, residuals_train, alpha=0.6, c='blue', s=30, label='Training Set', edgecolors='navy', linewidth=0.5)
            ax2.scatter(y_pred, residuals_test, alpha=0.8, c='red', s=30, label='Test Set', edgecolors='darkred', linewidth=0.5)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8, linewidth=2)
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals Plot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Distribution of residuals
            ax3 = fig.add_subplot(223)
            ax3.hist(residuals_train, bins=30, alpha=0.7, color='blue', label='Training Set', density=True)
            ax3.hist(residuals_test, bins=30, alpha=0.7, color='red', label='Test Set', density=True)
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Density')
            ax3.set_title('Residuals Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Error metrics comparison
            ax4 = fig.add_subplot(224)
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            mse_train = mean_squared_error(y_train, y_train_pred)
            mse_test = mean_squared_error(y_test, y_pred)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            mae_test = mean_absolute_error(y_test, y_pred)
            
            metrics_names = ['MSE', 'MAE', 'R²']
            train_metrics = [mse_train, mae_train, r2_train]
            test_metrics = [mse_test, mae_test, r2_test]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            ax4.bar(x - width/2, train_metrics, width, label='Training', color='blue', alpha=0.7)
            ax4.bar(x + width/2, test_metrics, width, label='Test', color='red', alpha=0.7)
            
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Values')
            ax4.set_title('Training vs Test Metrics')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics_names)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add values on bars
            for i, (train_val, test_val) in enumerate(zip(train_metrics, test_metrics)):
                ax4.text(i - width/2, train_val + max(train_metrics) * 0.01, f'{train_val:.3f}', 
                        ha='center', va='bottom', fontsize=8)
                ax4.text(i + width/2, test_val + max(test_metrics) * 0.01, f'{test_val:.3f}', 
                        ha='center', va='bottom', fontsize=8)
            
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            
            # Add navigation toolbar for interactivity
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            toolbar = NavigationToolbar(canvas, widget)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            self.right_panel.addTab(widget, "Regression Analysis")
            self.results_tabs["Regression Analysis"] = widget
            
        except Exception as e:
            print(f"Error creating regression analysis plot: {e}")
            
        # Additional detailed residuals plot
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            fig = Figure(figsize=(12, 8))
            
            # Q-Q plot for residuals normality check
            ax1 = fig.add_subplot(221)
            from scipy import stats
            residuals_test = y_test - y_pred
            stats.probplot(residuals_test, dist="norm", plot=ax1)
            ax1.set_title('Q-Q Plot (Test Set Residuals)')
            ax1.grid(True, alpha=0.3)
            
            # Scale-Location plot
            ax2 = fig.add_subplot(222)
            standardized_residuals = np.sqrt(np.abs(residuals_test / np.std(residuals_test)))
            ax2.scatter(y_pred, standardized_residuals, alpha=0.7, c='red', s=30)
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('√|Standardized Residuals|')
            ax2.set_title('Scale-Location Plot')
            ax2.grid(True, alpha=0.3)
            
            # Feature importance (if available)
            if hasattr(self.trained_pipeline.named_steps['model'], 'feature_importances_'):
                ax3 = fig.add_subplot(223)
                importances = self.trained_pipeline.named_steps['model'].feature_importances_
                feature_names = self.X.columns
                indices = np.argsort(importances)[::-1][:10]  # Top 10 features
                
                ax3.bar(range(len(indices)), importances[indices])
                ax3.set_xlabel('Features')
                ax3.set_ylabel('Importance')
                ax3.set_title('Top 10 Feature Importances')
                ax3.set_xticks(range(len(indices)))
                ax3.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                ax3.grid(True, alpha=0.3)
                
            # Learning curve comparison (simulated)
            ax4 = fig.add_subplot(224)
            train_sizes = np.linspace(0.1, 1.0, 10)
            ax4.plot(train_sizes, train_sizes * r2_train, 'b-', label='Training Score', alpha=0.7)
            ax4.plot(train_sizes, train_sizes * r2_test, 'r-', label='Validation Score', alpha=0.7)
            ax4.set_xlabel('Training Set Size')
            ax4.set_ylabel('Score')
            ax4.set_title('Learning Curve (Approximated)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            
            # Add navigation toolbar for interactivity
            toolbar = NavigationToolbar(canvas, widget)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            self.right_panel.addTab(widget, "Detailed Analysis")
            self.results_tabs["Detailed Analysis"] = widget
            
        except Exception as e:
            print(f"Error creating detailed analysis plot: {e}")
    
    def export_confusion_matrix(self):
        """Export confusion matrix data"""
        try:
            if not self.evaluation_results:
                return
                
            y_test = self.evaluation_results['y_test']
            y_pred = self.evaluation_results['y_pred']
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Confusion Matrix", "confusion_matrix.csv",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
            )
            
            if file_path:
                # Create DataFrame with proper labels
                unique_labels = sorted(y_test.unique())
                cm_df = pd.DataFrame(cm, 
                                   index=[f'Actual_{label}' for label in unique_labels],
                                   columns=[f'Predicted_{label}' for label in unique_labels])
                
                if file_path.endswith('.xlsx'):
                    cm_df.to_excel(file_path)
                else:
                    cm_df.to_csv(file_path)
                
                QMessageBox.information(self, "Success", f"Confusion matrix exported to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export confusion matrix: {str(e)}")
    
    def export_regression_results(self):
        """Export regression analysis results"""
        try:
            if not self.evaluation_results:
                return
                
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Regression Results", "regression_results.xlsx",
                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                # Prepare data
                y_train = self.evaluation_results['y_train']
                y_test = self.evaluation_results['y_test']
                y_train_pred = self.evaluation_results['y_train_pred']
                y_pred = self.evaluation_results['y_pred']
                
                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        # Training predictions
                        train_df = pd.DataFrame({
                            'Actual': y_train,
                            'Predicted': y_train_pred,
                            'Residuals': y_train - y_train_pred
                        })
                        train_df.to_excel(writer, sheet_name='Training_Results', index=False)
                        
                        # Test predictions
                        test_df = pd.DataFrame({
                            'Actual': y_test,
                            'Predicted': y_pred,
                            'Residuals': y_test - y_pred
                        })
                        test_df.to_excel(writer, sheet_name='Test_Results', index=False)
                        
                        # Metrics
                        metrics_df = pd.DataFrame([self.evaluation_results['metrics']])
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                else:
                    # For CSV, combine all data
                    combined_df = pd.DataFrame({
                        'Set': ['Train'] * len(y_train) + ['Test'] * len(y_test),
                        'Actual': list(y_train) + list(y_test),
                        'Predicted': list(y_train_pred) + list(y_pred),
                        'Residuals': list(y_train - y_train_pred) + list(y_test - y_pred)
                    })
                    combined_df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"Regression results exported to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export regression results: {str(e)}")
            
    def save_model(self):
        """Save the trained model"""
        if self.trained_pipeline is None:
            QMessageBox.warning(self, "Warning", "No trained model to save.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "Joblib Files (*.joblib);;All Files (*)"
        )
        
        if file_path:
            try:
                # Add .joblib extension if not present
                if not file_path.endswith('.joblib'):
                    file_path += '.joblib'
                    
                # CRITICAL FIX: Extract and save original feature ranges AND types for multi-objective optimization
                feature_bounds = {}
                feature_types = {}
                
                # Reproduce the same feature categorization logic used during training
                numeric_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
                categorical_features = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Debug: Print initial feature categorization
                print(f"[DEBUG] Initial numeric features: {len(numeric_features)}")
                print(f"[DEBUG] Initial categorical features: {len(categorical_features)}")
                
                # Identify boolean features (from one-hot encoding)
                boolean_features = []
                for col in numeric_features:
                    unique_vals = self.X[col].dropna().unique()
                    unique_set = set(unique_vals)
                    
                    print(f"[DEBUG] {col}: unique_vals = {sorted(unique_vals)}, set = {unique_set}")
                    
                    if len(unique_vals) <= 2 and unique_set.issubset({0, 1, 0.0, 1.0, True, False}):
                        boolean_features.append(col)
                        print(f"[DEBUG] -> Identified as BINARY: {col}")
                    else:
                        print(f"[DEBUG] -> Identified as CONTINUOUS: {col}")
                
                # Include actual boolean columns
                actual_bool_features = self.X.select_dtypes(include='bool').columns.tolist()
                boolean_features.extend(actual_bool_features)
                boolean_features = list(set(boolean_features))
                
                print(f"[DEBUG] Final boolean features: {boolean_features}")
                print(f"[DEBUG] Actual bool columns: {actual_bool_features}")
                
                # Remove boolean features from numeric features
                numeric_features = [col for col in numeric_features if col not in boolean_features]
                
                print(f"[DEBUG] Final numeric features: {len(numeric_features)}")
                print(f"[DEBUG] Final categorical features: {len(categorical_features)}")
                print(f"[DEBUG] Final boolean features: {len(boolean_features)}")
                
                # Save feature bounds and types
                for feature_name in self.X.columns:
                    min_val = float(self.X[feature_name].min())
                    max_val = float(self.X[feature_name].max())
                    feature_bounds[feature_name] = (min_val, max_val)
                    
                    # Determine feature type
                    if feature_name in numeric_features:
                        feature_types[feature_name] = 'continuous'
                    elif feature_name in categorical_features:
                        feature_types[feature_name] = 'categorical'
                    elif feature_name in boolean_features:
                        feature_types[feature_name] = 'binary'
                    else:
                        feature_types[feature_name] = 'continuous'  # fallback
                    
                    print(f"[DEBUG] {feature_name}: type={feature_types[feature_name]}, bounds=({min_val:.2f}, {max_val:.2f})")
                
                # Prepare metadata in the format expected by multi_objective_optimization.py
                metadata = {
                    'feature_names': self.X.columns.tolist(),
                    'task_type': self.task_type,
                    'target_name': self.y.name if self.y.name else 'target',  # 目标变量名称
                    'metrics': self.evaluation_results['metrics'] if hasattr(self, 'evaluation_results') and self.evaluation_results else None,
                    'class_names': self.class_names if hasattr(self, 'class_names') and self.class_names is not None else None,  # 分类模型的类名
                    'label_encoder': self.label_encoder if hasattr(self, 'label_encoder') and self.label_encoder is not None else None,  # 标签编码器
                    'n_features': len(self.X.columns),
                    'n_samples': len(self.X),
                    'save_timestamp': pd.Timestamp.now().isoformat(),  # 保存时间戳
                    'model_name': self.model_combo.currentText() if hasattr(self, 'model_combo') else 'unknown',  # 模型名称
                    'feature_bounds': feature_bounds,  # 原始特征边界 - 关键修复！
                    'feature_types': feature_types  # 特征类型 (continuous/categorical/binary) - 新增！
                }
                
                # Save model with structured metadata
                model_data = {
                    'pipeline': self.trained_pipeline,
                    'metadata': metadata
                }
                
                joblib.dump(model_data, file_path)
                
                QMessageBox.information(self, "Success", f"Model saved to {file_path}")
                self.status_updated.emit(f"Model saved to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving model: {str(e)}")
                
    def proceed_to_next_module(self):
        """Proceed to prediction module"""
        if self.trained_pipeline is None:
            QMessageBox.warning(self, "Warning", "No trained model available.")
            return
            
        try:
            # Prepare feature information for optimization
            feature_names = list(self.X.columns) if hasattr(self.X, 'columns') else None
            
            # Prepare feature type information
            feature_info = {}
            if hasattr(self.X, 'columns'):
                for col in self.X.columns:
                    # Detect feature type
                    if self.X[col].dtype == 'bool' or (set(self.X[col].unique()) <= {0, 1, 0.0, 1.0}):
                        feature_info[col] = {'type': 'binary', 'values': [0, 1]}
                    elif self.X[col].dtype in ['object', 'category']:
                        unique_vals = sorted(self.X[col].unique())
                        feature_info[col] = {'type': 'categorical', 'values': list(unique_vals)}
                    else:
                        feature_info[col] = {'type': 'continuous'}
            
            # Emit signal with trained model and training data for optimization
            self.model_ready.emit(
                self.trained_pipeline,  # Trained model
                feature_names,          # Feature names
                feature_info,           # Feature type information
                self.X,                 # Training data features
                self.y                  # Training data target
            )
            
            self.status_updated.emit("Model ready for prediction.")
            QMessageBox.information(self, "Success", "Model training completed! Ready for predictions.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error preparing model: {str(e)}")
    
    def perform_multiple_split_validation(self, model, X, y, task_type, n_splits=10):
        """Perform multiple train/test splits for robust evaluation"""
        scores = []
        
        for i in range(n_splits):
            # Use different random states for each split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size_spin.value(), 
                random_state=42+i,
                stratify=y if task_type == 'classification' else None
            )
            
            # Clone and fit the model
            try:
                from sklearn.base import clone
                model_clone = clone(model)
            except:
                # Fallback to deepcopy if sklearn clone fails
                import copy
                model_clone = copy.deepcopy(model)
            model_clone.fit(X_train, y_train)
            
            # Predict and score
            y_pred = model_clone.predict(X_test)
            
            if task_type == 'classification':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_test, y_pred)
            else:
                from sklearn.metrics import r2_score
                score = r2_score(y_test, y_pred)
                
            scores.append(score)
            
        return scores
    
    def perform_nested_cv_validation(self, pipeline, X, y, task_type, param_grid=None):
        """Perform nested cross-validation for unbiased performance estimate"""
        from sklearn.model_selection import cross_val_score
        
        # Outer CV for performance estimation
        if param_grid:
            # Use the same search method as HPO
            search_method = self.hpo_method_combo.currentText()
            
            if search_method == "Grid Search":
                search = GridSearchCV(
                    pipeline, param_grid,
                    cv=3,  # Inner CV
                    scoring='accuracy' if task_type == 'classification' else 'r2',
                    n_jobs=-1
                )
            else:
                search = RandomizedSearchCV(
                    pipeline, param_grid,
                    cv=3,  # Inner CV
                    scoring='accuracy' if task_type == 'classification' else 'r2',
                    n_jobs=-1, n_iter=20,
                    random_state=42
                )
            
            # Outer CV with inner HPO
            scores = cross_val_score(
                search, X, y,
                cv=5,  # Outer CV
                scoring='accuracy' if task_type == 'classification' else 'r2',
                n_jobs=-1
            )
        else:
            # Simple nested CV without HPO
            scores = cross_val_score(
                pipeline, X, y,
                cv=5,
                scoring='accuracy' if task_type == 'classification' else 'r2',
                n_jobs=-1
            )
            
        return np.mean(scores)

    def apply_wizard_config(self, config: dict):
        """Apply intelligent wizard configuration"""
        try:
            # Apply model selection configuration
            if 'selected_models' in config and config['selected_models']:
                model_name = config['selected_models'][0]
                # Find and set in model dropdown
                for i in range(self.model_combo.count()):
                    if self.model_combo.itemText(i) == model_name:
                        self.model_combo.setCurrentIndex(i)
                        break
            
            # Apply training configuration
            if 'test_size' in config:
                self.test_size_spin.setValue(config['test_size'])
            
            if 'random_state' in config:
                self.random_state_spin.setValue(config['random_state'])
            
            if 'cv_folds' in config:
                self.cv_folds_spin.setValue(config['cv_folds'])
                
        except Exception as e:
            self.status_updated.emit(f"Failed to apply wizard configuration: {str(e)}")
            
    def reset(self):
        """Reset the module"""
        self.X = None
        self.y = None
        self.y_original = None
        self.task_type = None
        self.trained_pipeline = None
        self.evaluation_results = None
        self.hpo_results = None
        self.label_encoder = None
        self.class_names = None
        
        # Reset UI
        self.data_info_label.setText("No data loaded")
        self.task_type_combo.setCurrentIndex(0)
        self.model_combo.clear()
        
        # Disable buttons
        self.train_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)
        self.proceed_btn.setEnabled(False)
        
        # Clear results
        self.right_panel.clear()
        self.results_tabs = {}
        
        self.setEnabled(False)
        self.status_updated.emit("Module reset")

    def show_evaluation_dialog(self):
        """Show comprehensive model evaluation dialog"""
        try:
            if self.trained_pipeline is None:
                QMessageBox.information(self, "No Model", "Please train a model first.")
                return
            
            if self.evaluation_results is None:
                QMessageBox.information(self, "No Results", "No evaluation results available.")
                return
            
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QTextEdit, QPushButton
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Model Evaluation Results")
            dialog.setModal(True)
            dialog.resize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Create tab widget for different evaluation views
            tab_widget = QTabWidget()
            layout.addWidget(tab_widget)
            
            # Metrics summary tab
            metrics_widget = QWidget()
            metrics_layout = QVBoxLayout(metrics_widget)
            
            metrics_text = QTextEdit()
            metrics_text.setReadOnly(True)
            
            # Generate metrics summary
            metrics_summary = "Model Evaluation Summary:\n"
            metrics_summary += "=" * 30 + "\n\n"
            
            if 'metrics' in self.evaluation_results:
                for metric, value in self.evaluation_results['metrics'].items():
                    if isinstance(value, float):
                        metrics_summary += f"{metric}: {value:.4f}\n"
                    else:
                        metrics_summary += f"{metric}: {value}\n"
            
            if 'cv_scores' in self.evaluation_results:
                cv_scores = self.evaluation_results['cv_scores']
                metrics_summary += f"\nCross-Validation Results:\n"
                metrics_summary += f"Mean CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n"
                metrics_summary += f"Individual CV Scores: {[f'{score:.4f}' for score in cv_scores]}\n"
            
            metrics_text.setPlainText(metrics_summary)
            metrics_layout.addWidget(metrics_text)
            
            tab_widget.addTab(metrics_widget, "Metrics Summary")
            
            # Feature importance tab (if available)
            if hasattr(self, 'feature_importance') and self.feature_importance is not None:
                importance_widget = QWidget()
                importance_layout = QVBoxLayout(importance_widget)
                
                # Create matplotlib figure for feature importance
                fig = Figure(figsize=(10, 6))
                canvas = FigureCanvas(fig)
                importance_layout.addWidget(canvas)
                
                ax = fig.add_subplot(111)
                
                # Plot top 15 features
                top_features = self.feature_importance.head(15)
                ax.barh(range(len(top_features)), top_features['importance'])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.set_xlabel('Importance')
                ax.set_title('Top 15 Feature Importances')
                ax.invert_yaxis()
                
                fig.tight_layout()
                canvas.draw()
                
                tab_widget.addTab(importance_widget, "Feature Importance")
            
            # Model details tab
            details_widget = QWidget()
            details_layout = QVBoxLayout(details_widget)
            
            details_text = QTextEdit()
            details_text.setReadOnly(True)
            
            # Generate model details
            details_summary = "Model Details:\n"
            details_summary += "=" * 20 + "\n\n"
            
            details_summary += f"Model Type: {self.model_combo.currentText()}\n"
            details_summary += f"Task Type: {self.task_type}\n"
            details_summary += f"Training Samples: {len(self.X)}\n"
            details_summary += f"Features: {len(self.X.columns)}\n"
            
            if self.task_type == 'classification' and hasattr(self, 'class_names'):
                details_summary += f"Classes: {self.class_names}\n"
            
            # Add hyperparameters if available
            if hasattr(self, 'hpo_results') and self.hpo_results:
                details_summary += "\nBest Hyperparameters:\n"
                for param, value in self.hpo_results['best_params'].items():
                    details_summary += f"  {param}: {value}\n"
            
            details_text.setPlainText(details_summary)
            details_layout.addWidget(details_text)
            
            tab_widget.addTab(details_widget, "Model Details")
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show evaluation dialog: {str(e)}")
    
    def show_learning_curves(self):
        """Show learning curves analysis"""
        try:
            if self.trained_pipeline is None:
                QMessageBox.information(self, "No Model", "Please train a model first.")
                return
            
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QProgressBar, QLabel
            from sklearn.model_selection import learning_curve
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Learning Curves Analysis")
            dialog.setModal(True)
            dialog.resize(900, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Progress bar
            progress_bar = QProgressBar()
            progress_label = QLabel("Generating learning curves...")
            layout.addWidget(progress_label)
            layout.addWidget(progress_bar)
            
            # Create matplotlib figure
            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.show()
            
            # Generate learning curves
            progress_bar.setValue(10)
            
            # Define training sizes
            train_sizes = np.linspace(0.1, 1.0, 10)
            
            # Generate learning curves
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
            
            progress_bar.setValue(30)
            
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.trained_pipeline, self.X, self.y,
                train_sizes=train_sizes,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                random_state=42
            )
            
            progress_bar.setValue(70)
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Plot learning curves
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
            ax1.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            ax1.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
            ax1.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            ax1.set_xlabel('Training Set Size')
            ax1.set_ylabel(f'{scoring.upper()} Score')
            ax1.set_title('Learning Curves')
            ax1.legend()
            ax1.grid(True)
            
            # Plot validation curve for a key hyperparameter
            ax2 = fig.add_subplot(2, 2, 2)
            
            # Try to plot validation curve for a relevant hyperparameter
            model_name = self.model_combo.currentText()
            if 'Random Forest' in model_name:
                from sklearn.model_selection import validation_curve
                param_name = 'n_estimators'
                param_range = [10, 50, 100, 200, 300]
                
                try:
                    train_scores_val, val_scores_val = validation_curve(
                        self.trained_pipeline, self.X, self.y,
                        param_name=param_name, param_range=param_range,
                        cv=3, scoring=scoring, n_jobs=-1
                    )
                    
                    train_mean_val = np.mean(train_scores_val, axis=1)
                    train_std_val = np.std(train_scores_val, axis=1)
                    val_mean_val = np.mean(val_scores_val, axis=1)
                    val_std_val = np.std(val_scores_val, axis=1)
                    
                    ax2.plot(param_range, train_mean_val, 'o-', color='blue', label='Training Score')
                    ax2.fill_between(param_range, train_mean_val - train_std_val, train_mean_val + train_std_val, alpha=0.1, color='blue')
                    ax2.plot(param_range, val_mean_val, 'o-', color='red', label='Validation Score')
                    ax2.fill_between(param_range, val_mean_val - val_std_val, val_mean_val + val_std_val, alpha=0.1, color='red')
                    ax2.set_xlabel(param_name)
                    ax2.set_ylabel(f'{scoring.upper()} Score')
                    ax2.set_title(f'Validation Curve ({param_name})')
                    ax2.legend()
                    ax2.grid(True)
                except:
                    ax2.text(0.5, 0.5, 'Validation curve not available', ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'Validation curve not available\nfor this model type', ha='center', va='center', transform=ax2.transAxes)
            
            # Plot training history (if available)
            ax3 = fig.add_subplot(2, 2, 3)
            if hasattr(self, 'training_history') and self.training_history:
                epochs = range(1, len(self.training_history) + 1)
                ax3.plot(epochs, self.training_history, 'o-', color='green')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Score')
                ax3.set_title('Training Progress')
                ax3.grid(True)
            else:
                ax3.text(0.5, 0.5, 'Training history not available', ha='center', va='center', transform=ax3.transAxes)
            
            # Plot feature importance (if available)
            ax4 = fig.add_subplot(2, 2, 4)
            if hasattr(self, 'feature_importance') and self.feature_importance is not None:
                top_features = self.feature_importance.head(10)
                ax4.barh(range(len(top_features)), top_features['importance'])
                ax4.set_yticks(range(len(top_features)))
                ax4.set_yticklabels(top_features['feature'])
                ax4.set_xlabel('Importance')
                ax4.set_title('Top 10 Feature Importances')
                ax4.invert_yaxis()
            else:
                ax4.text(0.5, 0.5, 'Feature importance not available', ha='center', va='center', transform=ax4.transAxes)
            
            progress_bar.setValue(100)
            progress_label.setText("Learning curves generated successfully!")
            
            fig.tight_layout()
            canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate learning curves: {str(e)}")
    
    def export_charts(self):
        """Export training charts and visualizations"""
        try:
            if self.trained_pipeline is None:
                QMessageBox.information(self, "No Model", "Please train a model first.")
                return
            
            from PyQt5.QtWidgets import QFileDialog
            import matplotlib.pyplot as plt
            
            # Get save directory
            save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Charts")
            if not save_dir:
                return
            
            # Export learning curves
            self.export_learning_curves_to_file(save_dir)
            
            # Export feature importance if available
            if hasattr(self, 'feature_importance') and self.feature_importance is not None:
                self.export_feature_importance_to_file(save_dir)
            
            # Export confusion matrix or regression plots
            if self.task_type == 'classification':
                self.export_confusion_matrix_to_file(save_dir)
            else:
                self.export_regression_plots_to_file(save_dir)
            
            QMessageBox.information(self, "Export Complete", f"Charts exported to {save_dir}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export charts: {str(e)}")
    
    def export_learning_curves_to_file(self, save_dir):
        """Export learning curves to file"""
        try:
            from sklearn.model_selection import learning_curve
            import matplotlib.pyplot as plt
            import os
            
            # Generate learning curves
            train_sizes = np.linspace(0.1, 1.0, 10)
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
            
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.trained_pipeline, self.X, self.y,
                train_sizes=train_sizes,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                random_state=42
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            plt.xlabel('Training Set Size')
            plt.ylabel(f'{scoring.upper()} Score')
            plt.title('Learning Curves')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            save_path = os.path.join(save_dir, 'learning_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error exporting learning curves: {e}")
    
    def export_feature_importance_to_file(self, save_dir):
        """Export feature importance plot to file"""
        try:
            import matplotlib.pyplot as plt
            import os
            
            # Create feature importance plot
            plt.figure(figsize=(10, 8))
            top_features = self.feature_importance.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.gca().invert_yaxis()
            
            # Save plot
            save_path = os.path.join(save_dir, 'feature_importance.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error exporting feature importance: {e}")
    
    def export_confusion_matrix_to_file(self, save_dir):
        """Export confusion matrix to file"""
        try:
            if 'confusion_matrix' not in self.evaluation_results:
                return
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            import os
            
            cm = self.evaluation_results['confusion_matrix']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            save_path = os.path.join(save_dir, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error exporting confusion matrix: {e}")
    
    def export_regression_plots_to_file(self, save_dir):
        """Export regression plots to file"""
        try:
            if 'y_pred' not in self.evaluation_results:
                return
            
            import matplotlib.pyplot as plt
            import os
            
            y_true = self.evaluation_results['y_true']
            y_pred = self.evaluation_results['y_pred']
            
            # Predicted vs Actual plot
            plt.figure(figsize=(10, 8))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predicted vs Actual Values')
            
            save_path = os.path.join(save_dir, 'predicted_vs_actual.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Residuals plot
            residuals = y_true - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            
            save_path = os.path.join(save_dir, 'residuals_plot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error exporting regression plots: {e}")