"""
Module 4: Enhanced Prediction & Results Export
Handles loading trained models and making predictions with detailed model info and dual prediction modes
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QPushButton, QMessageBox,
                            QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit,
                            QSplitter, QTabWidget, QCheckBox, QListWidget,
                            QListWidgetItem, QProgressBar, QTableWidget,
                            QTableWidgetItem, QFileDialog, QLineEdit,
                            QScrollArea, QFrame, QButtonGroup, QRadioButton)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont

import joblib
from utils.data_utils import load_csv_file, load_excel_file, load_clipboard_data


class PredictionModule(QWidget):
    """Enhanced prediction and results export module"""
    
    # Signals
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.trained_model = None
        self.model_metadata = None
        self.prediction_data = None
        self.predictions = None
        self.feature_names = []
        self.feature_inputs = {}  # Store feature input widgets
        self.feature_types = {}
        self.feature_bounds = {}
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the enhanced user interface"""
        layout = QVBoxLayout(self)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel for controls (with scroll area)
        left_scroll = QScrollArea()
        left_panel = QWidget()
        left_panel.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_panel)
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Right panel for results
        right_panel = QTabWidget()
        
        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([450, 1000])
        
        # === LEFT PANEL ===
        
        # Model Loading Section
        model_group = QGroupBox("Model Loading")
        model_layout = QVBoxLayout(model_group)
        
        # Load from current session
        self.load_current_btn = QPushButton("Use Current Session Model")
        self.load_current_btn.clicked.connect(self.use_current_model)
        self.load_current_btn.setEnabled(False)
        model_layout.addWidget(self.load_current_btn)
        
        # Load from file
        file_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select model file...")
        self.browse_model_btn = QPushButton("Browse")
        self.browse_model_btn.clicked.connect(self.browse_model_file)
        file_layout.addWidget(self.model_path_edit)
        file_layout.addWidget(self.browse_model_btn)
        model_layout.addLayout(file_layout)
        
        self.load_model_btn = QPushButton("Load Model from File")
        self.load_model_btn.clicked.connect(self.load_model_from_file)
        model_layout.addWidget(self.load_model_btn)
        
        left_layout.addWidget(model_group)
        
        # Model Information Section (详细信息)
        self.model_info_group = QGroupBox("Model Information")
        model_info_layout = QVBoxLayout(self.model_info_group)
        
        self.model_info_text = QTextEdit()
        self.model_info_text.setMinimumHeight(200)
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setPlaceholderText("Detailed model information will appear here...")
        model_info_layout.addWidget(self.model_info_text)
        
        self.model_info_group.setEnabled(False)
        left_layout.addWidget(self.model_info_group)
        
        # Prediction Mode Selection
        self.prediction_mode_group = QGroupBox("Prediction Mode")
        mode_layout = QVBoxLayout(self.prediction_mode_group)
        
        self.mode_button_group = QButtonGroup()
        
        self.import_mode_radio = QRadioButton("Import Data Mode")
        self.import_mode_radio.setChecked(True)
        self.import_mode_radio.toggled.connect(self.on_mode_changed)
        self.mode_button_group.addButton(self.import_mode_radio)
        mode_layout.addWidget(self.import_mode_radio)
        
        self.manual_mode_radio = QRadioButton("Manual Input Mode")
        self.manual_mode_radio.toggled.connect(self.on_mode_changed)
        self.mode_button_group.addButton(self.manual_mode_radio)
        mode_layout.addWidget(self.manual_mode_radio)
        
        self.prediction_mode_group.setEnabled(False)
        left_layout.addWidget(self.prediction_mode_group)
        
        # Data Import Section (导入模式)
        self.data_import_group = QGroupBox("Data Import")
        data_layout = QVBoxLayout(self.data_import_group)
        
        # File selection
        data_file_layout = QHBoxLayout()
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("Select data file for prediction...")
        self.browse_data_btn = QPushButton("Browse")
        self.browse_data_btn.clicked.connect(self.browse_data_file)
        data_file_layout.addWidget(self.data_path_edit)
        data_file_layout.addWidget(self.browse_data_btn)
        data_layout.addLayout(data_file_layout)
        
        # Import buttons
        data_btn_layout = QHBoxLayout()
        self.import_data_btn = QPushButton("Import Data")
        self.import_data_btn.clicked.connect(self.import_prediction_data)
        self.clipboard_data_btn = QPushButton("From Clipboard")
        self.clipboard_data_btn.clicked.connect(self.import_from_clipboard)
        data_btn_layout.addWidget(self.import_data_btn)
        data_btn_layout.addWidget(self.clipboard_data_btn)
        data_layout.addLayout(data_btn_layout)
        
        # Data info
        self.data_info_label = QLabel("No data loaded")
        data_layout.addWidget(self.data_info_label)
        
        self.data_import_group.setEnabled(False)
        left_layout.addWidget(self.data_import_group)
        
        # Manual Input Section (手动输入模式)
        self.manual_input_group = QGroupBox("Manual Feature Input")
        manual_layout = QVBoxLayout(self.manual_input_group)
        
        # Scroll area for feature inputs
        self.feature_scroll = QScrollArea()
        self.feature_widget = QWidget()
        self.feature_layout = QGridLayout(self.feature_widget)
        self.feature_scroll.setWidget(self.feature_widget)
        self.feature_scroll.setWidgetResizable(True)
        self.feature_scroll.setMaximumHeight(300)
        manual_layout.addWidget(self.feature_scroll)
        
        # Buttons for manual input
        manual_btn_layout = QHBoxLayout()
        self.reset_values_btn = QPushButton("Reset to Default")
        self.reset_values_btn.clicked.connect(self.reset_feature_values)
        self.load_sample_btn = QPushButton("Load Sample Values")
        self.load_sample_btn.clicked.connect(self.load_sample_values)
        manual_btn_layout.addWidget(self.reset_values_btn)
        manual_btn_layout.addWidget(self.load_sample_btn)
        manual_layout.addLayout(manual_btn_layout)
        
        self.manual_input_group.setEnabled(False)
        self.manual_input_group.setVisible(False)
        left_layout.addWidget(self.manual_input_group)
        
        # Prediction Section
        self.prediction_group = QGroupBox("Make Predictions")
        prediction_layout = QVBoxLayout(self.prediction_group)
        
        self.predict_btn = QPushButton("Generate Predictions")
        self.predict_btn.clicked.connect(self.make_predictions)
        self.predict_btn.setEnabled(False)
        prediction_layout.addWidget(self.predict_btn)
        
        # Prediction info
        self.prediction_info_label = QLabel("No predictions made")
        prediction_layout.addWidget(self.prediction_info_label)
        
        self.prediction_group.setEnabled(False)
        left_layout.addWidget(self.prediction_group)
        
        # Export Section
        self.export_group = QGroupBox("Export Results")
        export_layout = QVBoxLayout(self.export_group)
        
        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_csv_btn.setEnabled(False)
        export_layout.addWidget(self.export_csv_btn)
        
        self.export_excel_btn = QPushButton("Export to Excel")
        self.export_excel_btn.clicked.connect(self.export_to_excel)
        self.export_excel_btn.setEnabled(False)
        export_layout.addWidget(self.export_excel_btn)
        
        self.export_group.setEnabled(False)
        left_layout.addWidget(self.export_group)
        
        left_layout.addStretch()
        
        # === RIGHT PANEL ===
        
        # Model Details tab
        self.model_details_widget = QWidget()
        model_details_layout = QVBoxLayout(self.model_details_widget)
        
        self.model_details_text = QTextEdit()
        self.model_details_text.setReadOnly(True)
        model_details_layout.addWidget(QLabel("Complete Model Information:"))
        model_details_layout.addWidget(self.model_details_text)
        
        right_panel.addTab(self.model_details_widget, "Model Details")
        
        # Data preview tab
        self.data_preview_widget = QWidget()
        data_preview_layout = QVBoxLayout(self.data_preview_widget)
        
        self.data_table = QTableWidget()
        data_preview_layout.addWidget(QLabel("Data Preview:"))
        data_preview_layout.addWidget(self.data_table)
        
        right_panel.addTab(self.data_preview_widget, "Data Preview")
        
        # Results tab
        self.results_widget = QWidget()
        results_layout = QVBoxLayout(self.results_widget)
        
        self.results_table = QTableWidget()
        results_layout.addWidget(QLabel("Prediction Results:"))
        results_layout.addWidget(self.results_table)
        
        right_panel.addTab(self.results_widget, "Prediction Results")
        
        # Store reference to right panel
        self.right_panel = right_panel
        
        # Enable the module from start (allow independent use)
        self.setEnabled(True)
        
        # Enable model loading section immediately
        self.enable_model_loading_section()
        
    def set_model(self, trained_model):
        """Set the trained model for prediction with enhanced feature name handling"""
        self.trained_model = trained_model
        
        # **CRITICAL FIX**: Extract feature names properly from the model
        print(f"[DEBUG] PredictionModule: Setting model of type {type(trained_model)}")
        
        # Extract model information first
        self.extract_model_info()
        
        # **ENHANCED FEATURE NAME EXTRACTION**: Prioritize model's actual feature names
        model_feature_names = None
        
        # Try to get feature names directly from the model
        if hasattr(self.trained_model, 'feature_names_in_'):
            model_feature_names = list(self.trained_model.feature_names_in_)
            print(f"[DEBUG] PredictionModule: Got {len(model_feature_names)} feature names from model.feature_names_in_")
        elif hasattr(self.trained_model, 'named_steps'):
            # For Pipeline, try to get from the first step (usually preprocessor)
            try:
                first_step_name, first_step = list(self.trained_model.named_steps.items())[0]
                if hasattr(first_step, 'feature_names_in_'):
                    model_feature_names = list(first_step.feature_names_in_)
                    print(f"[DEBUG] PredictionModule: Got {len(model_feature_names)} feature names from {first_step_name}.feature_names_in_")
                elif hasattr(first_step, 'get_feature_names_out'):
                    # Try get_feature_names_out for transformers
                    try:
                        model_feature_names = list(first_step.get_feature_names_out())
                        print(f"[DEBUG] PredictionModule: Got {len(model_feature_names)} feature names from {first_step_name}.get_feature_names_out()")
                    except:
                        pass
            except Exception as e:
                print(f"[WARNING] PredictionModule: Could not extract feature names from Pipeline: {e}")
        
        # Set feature names
        if model_feature_names:
            self.feature_names = model_feature_names
            print(f"[SUCCESS] PredictionModule: Using model's feature names ({len(self.feature_names)} features)")
        else:
            # Fallback: try to get from model metadata or use generic names
            if hasattr(self, 'model_metadata') and self.model_metadata and 'n_features' in self.model_metadata:
                n_features = self.model_metadata['n_features']
                self.feature_names = [f"feature_{i}" for i in range(n_features)]
                print(f"[FALLBACK] PredictionModule: Using generic feature names ({len(self.feature_names)} features)")
            else:
                self.feature_names = []
                print(f"[WARNING] PredictionModule: No feature names available")
        
        # Update model info display
        self.update_model_info_display()
        
        # Setup manual input interface with new feature names
        self.setup_manual_input_interface()
        
        # Enable UI components
        self.model_info_group.setEnabled(True)
        self.prediction_mode_group.setEnabled(True)
        self.load_current_btn.setEnabled(True)  # Enable since we now have a current session model
        
        # Update mode visibility
        self.on_mode_changed()
        
        print(f"[INFO] PredictionModule: Model setup complete")
        print(f"[INFO] - Model type: {type(self.trained_model)}")
        print(f"[INFO] - Feature count: {len(self.feature_names)}")
        if self.feature_names:
            print(f"[INFO] - First few features: {self.feature_names[:3]}...")
        
        self.status_updated.emit("Model loaded successfully")
        
    def extract_model_info(self):
        """Extract detailed information from the trained model"""
        if self.trained_model is None:
            return
            
        try:
            # Initialize metadata dictionary
            self.model_metadata = {
                'model_type': 'Unknown',
                'feature_names': [],
                'n_features': 0,
                'pipeline_steps': [],
                'model_params': {},
                'preprocessing_info': {},
                'task_type': 'Unknown'
            }
            
            # Check if it's a pipeline
            if hasattr(self.trained_model, 'steps'):
                # It's a sklearn Pipeline
                self.model_metadata['pipeline_steps'] = [step[0] for step in self.trained_model.steps]
                
                # Get the final model (last step)
                final_model = self.trained_model.steps[-1][1]
                self.model_metadata['model_type'] = type(final_model).__name__
                self.model_metadata['model_params'] = final_model.get_params()
                
                # Get preprocessing info
                if len(self.trained_model.steps) > 1:
                    preprocessor = self.trained_model.steps[0][1]
                    if hasattr(preprocessor, 'transformers_'):
                        # ColumnTransformer
                        self.model_metadata['preprocessing_info'] = {
                            'type': 'ColumnTransformer',
                            'transformers': [t[0] for t in preprocessor.transformers_]
                        }
                
                # Try to get feature names
                if hasattr(self.trained_model, 'feature_names_in_'):
                    self.feature_names = list(self.trained_model.feature_names_in_)
                elif hasattr(final_model, 'feature_names_in_'):
                    self.feature_names = list(final_model.feature_names_in_)
                
            else:
                # Direct model (not pipeline)
                self.model_metadata['model_type'] = type(self.trained_model).__name__
                self.model_metadata['model_params'] = self.trained_model.get_params()
                
                if hasattr(self.trained_model, 'feature_names_in_'):
                    self.feature_names = list(self.trained_model.feature_names_in_)
            
            # Set feature count
            self.model_metadata['n_features'] = len(self.feature_names)
            self.model_metadata['feature_names'] = self.feature_names
            
            # Determine task type
            if hasattr(self.trained_model, 'predict_proba'):
                self.model_metadata['task_type'] = 'Classification'
            else:
                self.model_metadata['task_type'] = 'Regression'
                
        except Exception as e:
            print(f"Error extracting model info: {e}")
            
    def update_model_info_display(self):
        """Update model information display with detailed info"""
        if not self.model_metadata:
            return
            
        # Brief info for left panel
        brief_info = f"""Model Type: {self.model_metadata['model_type']}
Task Type: {self.model_metadata['task_type']}
Features: {self.model_metadata['n_features']}
Pipeline: {'Yes' if self.model_metadata['pipeline_steps'] else 'No'}"""
        
        self.model_info_text.setText(brief_info)
        
        # Detailed info for right panel
        detailed_info = self.generate_detailed_model_info()
        self.model_details_text.setText(detailed_info)
        
    def generate_detailed_model_info(self):
        """Generate comprehensive model information"""
        if not self.model_metadata:
            return "No model information available"
            
        info_parts = []
        
        # Basic Information
        info_parts.append("=" * 60)
        info_parts.append("MODEL INFORMATION")
        info_parts.append("=" * 60)
        info_parts.append(f"Model Type: {self.model_metadata['model_type']}")
        info_parts.append(f"Task Type: {self.model_metadata['task_type']}")
        info_parts.append(f"Number of Features: {self.model_metadata['n_features']}")
        info_parts.append("")
        
        # Pipeline Information
        if self.model_metadata['pipeline_steps']:
            info_parts.append("PIPELINE STRUCTURE:")
            info_parts.append("-" * 30)
            for i, step in enumerate(self.model_metadata['pipeline_steps'], 1):
                info_parts.append(f"{i}. {step}")
            info_parts.append("")
        
        # Enhanced Feature Information with Types and Bounds
        if self.feature_names:
            info_parts.append("FEATURE INFORMATION:")
            info_parts.append("-" * 40)
            
            # Feature type summary if available
            if self.feature_types:
                type_counts = {}
                for ftype in self.feature_types.values():
                    type_counts[ftype] = type_counts.get(ftype, 0) + 1
                
                info_parts.append("Feature Type Summary:")
                for ftype, count in sorted(type_counts.items()):
                    type_icon = {'continuous': '📈', 'binary': '🔘', 'categorical': '📝'}.get(ftype, '📊')
                    info_parts.append(f"  {type_icon} {ftype.title()}: {count} features")
                info_parts.append("")
            
            # Detailed feature list
            info_parts.append("Feature Details:")
            for i, feature in enumerate(self.feature_names, 1):
                # Get feature type and bounds if available
                ftype = self.feature_types.get(feature, 'unknown') if self.feature_types else 'unknown'
                bounds = self.feature_bounds.get(feature) if self.feature_bounds else None
                
                type_icon = {'continuous': '📈', 'binary': '🔘', 'categorical': '📝'}.get(ftype, '📊')
                
                if bounds:
                    bounds_str = f" [{bounds[0]:.2f} to {bounds[1]:.2f}]"
                else:
                    bounds_str = ""
                
                info_parts.append(f"{i:2d}. {type_icon} {feature} ({ftype}){bounds_str}")
            info_parts.append("")
        
        # Preprocessing Information
        if self.model_metadata['preprocessing_info']:
            info_parts.append("PREPROCESSING:")
            info_parts.append("-" * 30)
            prep_info = self.model_metadata['preprocessing_info']
            info_parts.append(f"Type: {prep_info.get('type', 'Unknown')}")
            if 'transformers' in prep_info:
                info_parts.append("Transformers:")
                for transformer in prep_info['transformers']:
                    info_parts.append(f"  - {transformer}")
            info_parts.append("")
        
        # Additional Metadata from Training Module
        if self.model_metadata:
            if 'save_timestamp' in self.model_metadata:
                info_parts.append("TRAINING METADATA:")
                info_parts.append("-" * 30)
                info_parts.append(f"Save Timestamp: {self.model_metadata['save_timestamp']}")
                
                if 'model_name' in self.model_metadata:
                    info_parts.append(f"Model Algorithm: {self.model_metadata['model_name']}")
                
                if 'target_name' in self.model_metadata:
                    info_parts.append(f"Target Variable: {self.model_metadata['target_name']}")
                
                if 'n_samples' in self.model_metadata:
                    info_parts.append(f"Training Samples: {self.model_metadata['n_samples']}")
                
                info_parts.append("")
        
        # Model Parameters
        if self.model_metadata['model_params']:
            info_parts.append("MODEL PARAMETERS:")
            info_parts.append("-" * 30)
            params = self.model_metadata['model_params']
            for key, value in sorted(params.items()):
                # Format parameter values nicely
                if isinstance(value, (int, float, str, bool, type(None))):
                    info_parts.append(f"{key}: {value}")
                elif isinstance(value, (list, tuple)) and len(value) < 10:
                    info_parts.append(f"{key}: {value}")
                else:
                    info_parts.append(f"{key}: {type(value).__name__}")
            info_parts.append("")
        
        # Model Capabilities
        info_parts.append("MODEL CAPABILITIES:")
        info_parts.append("-" * 30)
        if hasattr(self.trained_model, 'predict'):
            info_parts.append("✓ Prediction")
        if hasattr(self.trained_model, 'predict_proba'):
            info_parts.append("✓ Probability Prediction")
        if hasattr(self.trained_model, 'decision_function'):
            info_parts.append("✓ Decision Function")
        if hasattr(self.trained_model, 'score'):
            info_parts.append("✓ Scoring")
        
        return "\n".join(info_parts)
        
    def setup_manual_input_interface(self):
        """Setup manual input interface based on original feature names with enhanced error handling"""
        # Clear existing widgets
        for i in reversed(range(self.feature_layout.count())):
            self.feature_layout.itemAt(i).widget().setParent(None)
        
        self.feature_inputs = {}
        
        # **ENHANCED FIX**: Try to get original feature interface specification with detailed debugging
        try:
            from utils.prediction_utils import get_prediction_input_interface, feature_mapper
            
            # ENHANCED DEBUGGING: Print detailed mapper state
            print(f"[DEBUG] PredictionModule: Checking feature mapper state...")
            print(f"[DEBUG] - Has original_feature_names attr: {hasattr(feature_mapper, 'original_feature_names')}")
            if hasattr(feature_mapper, 'original_feature_names'):
                original_names = feature_mapper.original_feature_names
                print(f"[DEBUG] - Original feature names: {original_names}")
                print(f"[DEBUG] - Count: {len(original_names) if original_names else 0}")
            
            print(f"[DEBUG] - Has encoded_feature_names attr: {hasattr(feature_mapper, 'encoded_feature_names')}")
            if hasattr(feature_mapper, 'encoded_feature_names'):
                encoded_names = feature_mapper.encoded_feature_names
                print(f"[DEBUG] - Encoded feature names: {encoded_names}")
                print(f"[DEBUG] - Count: {len(encoded_names) if encoded_names else 0}")
            
            print(f"[DEBUG] - Has categorical_mappings: {hasattr(feature_mapper, 'categorical_mappings')}")
            if hasattr(feature_mapper, 'categorical_mappings'):
                print(f"[DEBUG] - Categorical mappings: {feature_mapper.categorical_mappings}")
            
            # Check if feature mapper is properly initialized
            if (hasattr(feature_mapper, 'original_feature_names') and 
                feature_mapper.original_feature_names and 
                len(feature_mapper.original_feature_names) > 0):
                
                print(f"[DEBUG] PredictionModule: Feature mapper has {len(feature_mapper.original_feature_names)} original features")
                interface_spec = get_prediction_input_interface()
                
                print(f"[DEBUG] Interface spec returned: {interface_spec}")
                
                # CRITICAL CHECK: Only use original interface if features match expected model features
                if interface_spec and len(interface_spec) > 0:
                    # Check if original features are actually only the model features (exclude target)
                    model_feature_count = len(self.feature_names) if hasattr(self, 'feature_names') and self.feature_names else 0
                    interface_feature_count = len(interface_spec)
                    
                    print(f"[DEBUG] Model expects {model_feature_count} features, interface has {interface_feature_count}")
                    
                    # Only use original interface if counts are reasonable
                    if model_feature_count > 0 and interface_feature_count <= model_feature_count + 5: # Allow some tolerance
                        self.setup_original_feature_interface(interface_spec)
                        print(f"[SUCCESS] PredictionModule: Using original feature interface with {len(interface_spec)} features")
                        return
                    else:
                        # This case indicates a significant mismatch, likely including the target column
                        print(f"[WARNING] PredictionModule: Mismatch detected between model features ({model_feature_count}) and interface spec ({interface_feature_count}). Falling back to default interface.")
                else:
                    print(f"[WARNING] PredictionModule: Interface spec is empty or invalid")
            else:
                print(f"[INFO] PredictionModule: Feature mapper not initialized or empty")
                
        except Exception as e:
            print(f"[WARNING] PredictionModule: Could not get original feature interface: {e}")
            import traceback
            traceback.print_exc()
        
        # **ENHANCED FALLBACK**: Use model's expected features if available
        if hasattr(self, 'trained_model') and self.trained_model is not None:
            try:
                # Try to get feature names from the model
                model_features = None
                if hasattr(self.trained_model, 'feature_names_in_'):
                    model_features = list(self.trained_model.feature_names_in_)
                    print(f"[DEBUG] PredictionModule: Using model.feature_names_in_ ({len(model_features)} features)")
                elif hasattr(self.trained_model, 'named_steps'):
                    # For Pipeline, try to get from first step
                    try:
                        first_step_name, first_step = list(self.trained_model.named_steps.items())[0]
                        if hasattr(first_step, 'feature_names_in_'):
                            model_features = list(first_step.feature_names_in_)
                            print(f"[DEBUG] PredictionModule: Using {first_step_name}.feature_names_in_ ({len(model_features)} features)")
                    except:
                        pass
                
                if model_features:
                    print(f"[INFO] PredictionModule: Setting up interface with model's feature names")
                    self.feature_names = model_features
                    
                    # Create input widgets for model features
                    for i, feature_name in enumerate(model_features):
                        # Feature label
                        label = QLabel(f"{feature_name}:")
                        label.setMinimumWidth(150)
                        
                        # Feature input (use QDoubleSpinBox for numeric input)
                        input_widget = QDoubleSpinBox()
                        input_widget.setMinimum(-999999.0)
                        input_widget.setMaximum(999999.0)
                        input_widget.setDecimals(6)
                        input_widget.setValue(0.0)
                        input_widget.setMinimumWidth(120)
                        
                        # Add to layout
                        row = i
                        self.feature_layout.addWidget(label, row, 0)
                        self.feature_layout.addWidget(input_widget, row, 1)
                        
                        # Store reference
                        self.feature_inputs[feature_name] = input_widget
                    
                    print(f"[SUCCESS] PredictionModule: Created interface with {len(model_features)} model features")
                    return
                    
            except Exception as e:
                print(f"[ERROR] PredictionModule: Failed to use model features: {e}")
        
        # **ULTIMATE FALLBACK**: Use self.feature_names if available
        if hasattr(self, 'feature_names') and self.feature_names:
            print(f"[FALLBACK] PredictionModule: Using stored feature_names ({len(self.feature_names)} features)")
            
            # Create input widgets for each stored feature
            for i, feature_name in enumerate(self.feature_names):
                # Feature label
                label = QLabel(f"{feature_name}:")
                label.setMinimumWidth(150)
                
                # Feature input (use QDoubleSpinBox for numeric input)
                input_widget = QDoubleSpinBox()
                input_widget.setMinimum(-999999.0)
                input_widget.setMaximum(999999.0)
                input_widget.setDecimals(6)
                input_widget.setValue(0.0)
                input_widget.setMinimumWidth(120)
                
                # Add to layout
                row = i
                self.feature_layout.addWidget(label, row, 0)
                self.feature_layout.addWidget(input_widget, row, 1)
                
                # Store reference
                self.feature_inputs[feature_name] = input_widget
            
            print(f"[SUCCESS] PredictionModule: Created interface with stored feature names")
            return
        
        # **LAST RESORT**: Create a minimal interface
        print(f"[WARNING] PredictionModule: No feature information available, creating minimal interface")
        
        # Create a simple message widget
        label = QLabel("No feature information available.\nPlease ensure the model is properly loaded.")
        label.setMinimumWidth(300)
        label.setWordWrap(True)
        self.feature_layout.addWidget(label, 0, 0, 1, 2)
        
        print(f"[INFO] PredictionModule: Manual input interface setup completed")
    
    def setup_original_feature_interface(self, interface_spec: Dict[str, Dict]):
        """Setup interface using original feature specifications"""
        from PyQt5.QtWidgets import QComboBox, QCheckBox
        
        for i, (feature_name, spec) in enumerate(interface_spec.items()):
            # Feature label with description
            label = QLabel(f"{feature_name}:")
            label.setMinimumWidth(150)
            
            # Create appropriate input widget based on feature type
            if spec['type'] == 'categorical':
                input_widget = QComboBox()
                input_widget.addItems([str(v) for v in spec['values']])
                if spec['default'] is not None:
                    input_widget.setCurrentText(str(spec['default']))
                input_widget.setMinimumWidth(120)
                
            elif spec['type'] == 'boolean':
                input_widget = QCheckBox()
                input_widget.setChecked(spec['default'])
                
            else:  # numeric
                input_widget = QDoubleSpinBox()
                input_widget.setMinimum(spec.get('min', -999999.0))
                input_widget.setMaximum(spec.get('max', 999999.0))
                input_widget.setDecimals(spec.get('decimals', 6))
                input_widget.setValue(spec.get('default', 0.0))
                input_widget.setMinimumWidth(120)
            
            # Add tooltip with feature description
            try:
                from utils.prediction_utils import get_feature_description
                description = get_feature_description(feature_name)
                input_widget.setToolTip(description)
                label.setToolTip(description)
            except:
                pass
            
            # Add to layout
            row = i
            self.feature_layout.addWidget(label, row, 0)
            self.feature_layout.addWidget(input_widget, row, 1)
            
            # Store reference
            self.feature_inputs[feature_name] = input_widget
            
    def on_mode_changed(self):
        """Handle prediction mode change"""
        if self.import_mode_radio.isChecked():
            # Import mode
            self.data_import_group.setVisible(True)
            self.manual_input_group.setVisible(False)
            self.data_import_group.setEnabled(True)
            self.manual_input_group.setEnabled(False)
        else:
            # Manual mode
            self.data_import_group.setVisible(False)
            self.manual_input_group.setVisible(True)
            self.data_import_group.setEnabled(False)
            self.manual_input_group.setEnabled(True)
            
        # Enable prediction if model is loaded
        if self.trained_model is not None:
            self.prediction_group.setEnabled(True)
            self.predict_btn.setEnabled(True)
            
    def reset_feature_values(self):
        """Reset all feature values to default"""
        from PyQt5.QtWidgets import QComboBox, QCheckBox, QDoubleSpinBox
        
        for input_widget in self.feature_inputs.values():
            if isinstance(input_widget, QDoubleSpinBox):
                input_widget.setValue(0.0)
            elif isinstance(input_widget, QCheckBox):
                input_widget.setChecked(False)
            elif isinstance(input_widget, QComboBox):
                input_widget.setCurrentIndex(0)
            
    def load_sample_values(self):
        """Load sample values for demonstration"""
        if not self.feature_inputs:
            return
        
        try:
            # Try to get sample values from prediction utils
            from utils.prediction_utils import get_sample_prediction_input
            sample_input = get_sample_prediction_input()
            
            if sample_input:
                self.set_manual_input_values(sample_input)
                QMessageBox.information(self, "Sample Values", "Sample values loaded from original feature mapping.")
                return
        except Exception as e:
            print(f"Could not get sample values from prediction utils: {e}")
        
        # Fallback: Generate sample values for encoded features
        from PyQt5.QtWidgets import QComboBox, QCheckBox, QDoubleSpinBox
        
        for i, (feature_name, input_widget) in enumerate(self.feature_inputs.items()):
            if isinstance(input_widget, QDoubleSpinBox):
                # Generate a sample value based on feature index
                sample_value = (i + 1) * 0.5
                input_widget.setValue(sample_value)
            elif isinstance(input_widget, QCheckBox):
                input_widget.setChecked(i % 2 == 0)  # Alternate True/False
            elif isinstance(input_widget, QComboBox):
                if input_widget.count() > 1:
                    input_widget.setCurrentIndex(1)  # Select second option
            
        QMessageBox.information(self, "Sample Values", "Sample values loaded for demonstration.")
    
    def set_manual_input_values(self, values_dict: Dict[str, Any]):
        """Set manual input values from dictionary"""
        from PyQt5.QtWidgets import QComboBox, QCheckBox, QDoubleSpinBox
        
        for feature_name, value in values_dict.items():
            if feature_name in self.feature_inputs:
                input_widget = self.feature_inputs[feature_name]
                
                if isinstance(input_widget, QDoubleSpinBox):
                    input_widget.setValue(float(value))
                elif isinstance(input_widget, QCheckBox):
                    input_widget.setChecked(bool(value))
                elif isinstance(input_widget, QComboBox):
                    input_widget.setCurrentText(str(value))
        
    def get_manual_input_data(self):
        """Get data from manual input widgets and convert to model format"""
        if not self.feature_inputs:
            return None
        
        from PyQt5.QtWidgets import QComboBox, QCheckBox, QDoubleSpinBox
        
        # Collect input values with enhanced type handling
        input_data = {}
        for feature_name, input_widget in self.feature_inputs.items():
            if isinstance(input_widget, QDoubleSpinBox):
                input_data[feature_name] = float(input_widget.value())
            elif isinstance(input_widget, QCheckBox):
                # Convert checkbox to 0/1 integer
                input_data[feature_name] = 1 if input_widget.isChecked() else 0
            elif isinstance(input_widget, QComboBox):
                # Convert combobox selection to appropriate numeric value
                try:
                    # Try to convert to integer first (for categorical features)
                    input_data[feature_name] = int(input_widget.currentText())
                except ValueError:
                    # If not integer, try float
                    try:
                        input_data[feature_name] = float(input_widget.currentText())
                    except ValueError:
                        # If neither, default to 0
                        input_data[feature_name] = 0
            else:
                input_data[feature_name] = 0.0  # Fallback
        
        print(f"[DEBUG] Manual input data collected: {input_data}")
        
        # **CRITICAL FIX**: Try to map original features to encoded format with enhanced error handling
        try:
            from utils.prediction_utils import map_original_input_for_prediction, validate_prediction_input
            
            # **FIX**: Only validate if we have meaningful feature mapping data
            # Check if the global feature mapper has been properly initialized
            from utils.prediction_utils import feature_mapper
            
            if (hasattr(feature_mapper, 'original_feature_names') and 
                feature_mapper.original_feature_names and 
                len(feature_mapper.original_feature_names) > 0):
                
                print(f"[DEBUG] PredictionModule: Attempting feature validation with {len(feature_mapper.original_feature_names)} original features")
                
                # Validate input first - but only if we have proper mapping
                is_valid, errors = validate_prediction_input(input_data)
                if not is_valid:
                    print(f"[WARNING] PredictionModule: Feature validation failed: {errors}")
                    
                    # **ENHANCED FIX**: Check if this is a critical error or just a mapping issue
                    critical_errors = [err for err in errors if "Unknown features" in err and len(input_data) > 0]
                    
                    if critical_errors:
                        print(f"[WARNING] PredictionModule: Critical feature validation errors detected")
                        # Only show error dialog if this is likely a user-initiated prediction
                        # (not triggered by optimization module or background processes)
                        
                        # Check if we're in manual input mode and user is actively using prediction module
                        if (hasattr(self, 'manual_mode_radio') and 
                            self.manual_mode_radio.isChecked() and 
                            self.isVisible() and 
                            self.prediction_group.isEnabled()):
                            
                            error_msg = "Input validation failed:\n" + "\n".join(errors)
                            QMessageBox.warning(self, "Input Validation Error", error_msg)
                            return None
                        else:
                            print(f"[INFO] PredictionModule: Skipping error dialog - likely background process")
                    else:
                        print(f"[INFO] PredictionModule: Non-critical validation errors, proceeding with fallback")
                
                # Try to map to encoded format
                encoded_data = map_original_input_for_prediction(input_data)
                if not encoded_data.empty:
                    print(f"[SUCCESS] PredictionModule: Successfully mapped {len(input_data)} original features to {len(encoded_data.columns)} encoded features")
                    return encoded_data
                else:
                    print(f"[WARNING] PredictionModule: Feature mapping returned empty DataFrame, using fallback")
            else:
                print(f"[INFO] PredictionModule: No feature mapping available, using direct input data")
                
        except Exception as e:
            print(f"[ERROR] PredictionModule: Could not map original features to encoded format: {e}")
            print(f"[INFO] PredictionModule: Proceeding with fallback approach")
        
        # **ENHANCED FALLBACK**: Create DataFrame with current feature names
        try:
            # If we have a trained model, try to use its expected feature names
            if (hasattr(self, 'trained_model') and self.trained_model is not None):
                
                # Try to get expected feature names from the model
                expected_features = None
                if hasattr(self.trained_model, 'feature_names_in_'):
                    expected_features = list(self.trained_model.feature_names_in_)
                    print(f"[DEBUG] PredictionModule: Model expects {len(expected_features)} features: {expected_features[:3]}...")
                elif hasattr(self.trained_model, 'named_steps'):
                    # For Pipeline, try to get from first step
                    try:
                        first_step_name, first_step = list(self.trained_model.named_steps.items())[0]
                        if hasattr(first_step, 'feature_names_in_'):
                            expected_features = list(first_step.feature_names_in_)
                            print(f"[DEBUG] PredictionModule: Pipeline {first_step_name} expects {len(expected_features)} features: {expected_features[:3]}...")
                    except:
                        pass
                
                # If we have expected features and they match our input count
                if expected_features and len(expected_features) == len(input_data):
                    print(f"[FIX] PredictionModule: Mapping input data to model's expected feature names")
                    
                    # Create ordered data based on expected feature names
                    ordered_data = {}
                    for expected_feature in expected_features:
                        # Try to find a matching input feature (case-insensitive, flexible matching)
                        matched_value = None
                        for input_feature, value in input_data.items():
                            if (input_feature == expected_feature or 
                                input_feature.lower() == expected_feature.lower() or
                                input_feature.replace('_', '').lower() == expected_feature.replace('_', '').lower()):
                                matched_value = value
                                break
                        
                        # Use matched value or default to 0
                        ordered_data[expected_feature] = matched_value if matched_value is not None else 0.0
                    
                    return pd.DataFrame([ordered_data])
                
                # If feature counts don't match, try to handle gracefully
                elif expected_features:
                    print(f"[WARNING] PredictionModule: Feature count mismatch - expected {len(expected_features)}, got {len(input_data)}")
                    
                    # Create DataFrame with expected features, filling missing ones with defaults
                    ordered_data = {}
                    input_features = list(input_data.keys())
                    
                    for i, expected_feature in enumerate(expected_features):
                        if i < len(input_features):
                            # Use input data in order
                            input_feature = input_features[i]
                            ordered_data[expected_feature] = input_data[input_feature]
                        else:
                            # Fill missing features with default values
                            ordered_data[expected_feature] = 0.0
                    
                    print(f"[FIX] PredictionModule: Created ordered data with {len(ordered_data)} features")
                    return pd.DataFrame([ordered_data])
            
            # Ultimate fallback: use input data as-is
            print(f"[FALLBACK] PredictionModule: Using input data as-is")
            return pd.DataFrame([input_data])
            
        except Exception as fallback_error:
            print(f"[ERROR] PredictionModule: Fallback failed: {fallback_error}")
            # Last resort: return empty DataFrame
            return pd.DataFrame()
        
    def use_current_model(self):
        """Use the model from current session"""
        if self.trained_model is None:
            QMessageBox.warning(self, "Warning", "No model available from current session.")
            return
            
        # Update model info display
        self.update_model_info_display()
        
        # Enable prediction mode selection
        self.prediction_mode_group.setEnabled(True)
        self.on_mode_changed()  # Update visibility based on current mode
        
        self.status_updated.emit("Using current session model")
        
    def browse_model_file(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "Joblib Files (*.joblib);;Pickle Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            self.model_path_edit.setText(file_path)
            
    def load_model_from_file(self):
        """Load model from file with enhanced metadata support"""
        file_path = self.model_path_edit.text()
        
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select a model file.")
            return
            
        try:
            self.status_updated.emit("Loading model from file...")
            self.progress_updated.emit(30)
            
            # Load model
            model_data = joblib.load(file_path)
            
            # Reset metadata
            self.model_metadata = {}
            self.feature_types = {}
            self.feature_bounds = {}
            
            # Handle different save formats
            if isinstance(model_data, dict):
                self.trained_model = model_data.get('pipeline', model_data.get('model'))
                
                # Extract metadata with enhanced support for training module format
                metadata = model_data.get('metadata', {})
                if metadata:
                    # Get feature names
                    if 'feature_names' in metadata:
                        self.feature_names = metadata['feature_names']
                        self.status_updated.emit("✅ Feature names loaded from metadata")
                    
                    # Get feature types (NEW!)
                    if 'feature_types' in metadata:
                        self.feature_types = metadata['feature_types']
                        self.status_updated.emit(f"✅ Feature types loaded: {len(self.feature_types)} features")
                        
                        # Log feature type summary
                        type_counts = {}
                        for ftype in self.feature_types.values():
                            type_counts[ftype] = type_counts.get(ftype, 0) + 1
                        print(f"[INFO] Feature types summary: {type_counts}")
                    
                    # Get feature bounds (NEW!)
                    if 'feature_bounds' in metadata:
                        self.feature_bounds = metadata['feature_bounds']
                        self.status_updated.emit(f"✅ Feature bounds loaded: {len(self.feature_bounds)} features")
                        
                        # Log some feature bounds examples
                        feature_names_sample = list(self.feature_bounds.keys())[:3]
                        for fname in feature_names_sample:
                            bounds = self.feature_bounds[fname]
                            print(f"[INFO] {fname}: {bounds[0]:.2f} to {bounds[1]:.2f}")
                    
                    # Store additional metadata
                    self.model_metadata.update(metadata)
                    
                # Fallback for older format
                elif 'feature_names' in model_data:
                    self.feature_names = model_data['feature_names']
            else:
                self.trained_model = model_data
                
            self.progress_updated.emit(70)
            
            # Extract model information
            self.extract_model_info()
            
            self.progress_updated.emit(90)
            
            # Update display with enhanced information
            self.update_model_info_display()
            
            # Setup manual input interface with feature type support
            self.setup_enhanced_manual_input_interface()
            
            # Enable UI components
            self.model_info_group.setEnabled(True)
            self.prediction_mode_group.setEnabled(True)
            self.load_current_btn.setEnabled(True)
            self.on_mode_changed()
            
            self.progress_updated.emit(100)
            self.status_updated.emit(f"Model loaded from {file_path}")
            QMessageBox.information(self, "Success", "Model loaded successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.progress_updated.emit(0)
            self.status_updated.emit("Model loading failed")
            
    def browse_data_file(self):
        """Browse for data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if file_path:
            self.data_path_edit.setText(file_path)
            
    def import_prediction_data(self):
        """Import data for prediction"""
        file_path = self.data_path_edit.text()
        
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select a data file.")
            return
            
        try:
            self.status_updated.emit("Importing prediction data...")
            self.progress_updated.emit(30)
            
            if file_path.endswith('.csv'):
                self.prediction_data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.prediction_data = pd.read_excel(file_path)
            else:
                QMessageBox.warning(self, "Warning", "Unsupported file format.")
                return
                
            self.progress_updated.emit(70)
            
            # Validate data
            self.validate_prediction_data()
            
            # Update display
            self.update_data_preview()
            self.update_data_info()
            
            # Enable prediction
            self.prediction_group.setEnabled(True)
            self.predict_btn.setEnabled(True)
            
            self.progress_updated.emit(100)
            self.status_updated.emit(f"Prediction data imported: {self.prediction_data.shape}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import data: {str(e)}")
            self.progress_updated.emit(0)
            self.status_updated.emit("Data import failed")
            
    def import_from_clipboard(self):
        """Import data from clipboard"""
        try:
            self.status_updated.emit("Importing from clipboard...")
            self.progress_updated.emit(50)
            
            self.prediction_data = pd.read_clipboard()
            
            # Validate data
            self.validate_prediction_data()
            
            # Update display
            self.update_data_preview()
            self.update_data_info()
            
            # Enable prediction
            self.prediction_group.setEnabled(True)
            self.predict_btn.setEnabled(True)
            
            self.progress_updated.emit(100)
            self.status_updated.emit(f"Data imported from clipboard: {self.prediction_data.shape}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import from clipboard: {str(e)}")
            self.progress_updated.emit(0)
            self.status_updated.emit("Clipboard import failed")
            
    def validate_prediction_data(self):
        """Validate prediction data against model requirements"""
        if self.prediction_data is None or not self.feature_names:
            return
            
        # Check if all required features are present
        missing_features = set(self.feature_names) - set(self.prediction_data.columns)
        extra_features = set(self.prediction_data.columns) - set(self.feature_names)
        
        if missing_features:
            QMessageBox.warning(
                self, "Missing Features", 
                f"Missing required features: {', '.join(missing_features)}"
            )
            
        if extra_features:
            QMessageBox.information(
                self, "Extra Features", 
                f"Extra features will be ignored: {', '.join(extra_features)}"
            )
            
        # Reorder columns to match model expectations
        try:
            self.prediction_data = self.prediction_data[self.feature_names]
        except KeyError as e:
            QMessageBox.critical(
                self, "Feature Mismatch", 
                f"Cannot match features with model requirements: {str(e)}"
            )
            
    def update_data_preview(self):
        """Update data preview table"""
        if self.prediction_data is None:
            return
            
        # Setup table
        self.data_table.setRowCount(min(100, len(self.prediction_data)))  # Show max 100 rows
        self.data_table.setColumnCount(len(self.prediction_data.columns))
        self.data_table.setHorizontalHeaderLabels(self.prediction_data.columns.tolist())
        
        # Fill table
        for i in range(min(100, len(self.prediction_data))):
            for j, col in enumerate(self.prediction_data.columns):
                value = str(self.prediction_data.iloc[i, j])
                self.data_table.setItem(i, j, QTableWidgetItem(value))
                
        # Resize columns
        self.data_table.resizeColumnsToContents()
        
    def update_data_info(self):
        """Update data information display"""
        if self.prediction_data is not None:
            info_text = f"Samples: {self.prediction_data.shape[0]}, Features: {self.prediction_data.shape[1]}"
            self.data_info_label.setText(info_text)
        else:
            self.data_info_label.setText("No data loaded")
            
    def make_predictions(self):
        """Make predictions using the loaded model"""
        if self.trained_model is None:
            QMessageBox.warning(self, "Warning", "No model loaded.")
            return
            
        try:
            self.status_updated.emit("Making predictions...")
            self.progress_updated.emit(30)
            
            # Get data based on mode
            if self.import_mode_radio.isChecked():
                # Import mode
                if self.prediction_data is None:
                    QMessageBox.warning(self, "Warning", "No prediction data loaded.")
                    return
                data_for_prediction = self.prediction_data.copy()
            else:
                # Manual mode
                data_for_prediction = self.get_manual_input_data()
                if data_for_prediction is None:
                    QMessageBox.warning(self, "Warning", "No manual input data available.")
                    return
                    
            self.progress_updated.emit(60)
            
            # Make predictions
            predictions = self.trained_model.predict(data_for_prediction)
            
            # Create results DataFrame
            results_data = data_for_prediction.copy()
            results_data['Prediction'] = predictions
            
            # Add probability predictions if available (classification)
            if hasattr(self.trained_model, 'predict_proba'):
                try:
                    probabilities = self.trained_model.predict_proba(data_for_prediction)
                    if probabilities.shape[1] == 2:  # Binary classification
                        results_data['Probability'] = probabilities[:, 1]
                    else:  # Multi-class
                        for i in range(probabilities.shape[1]):
                            results_data[f'Prob_Class_{i}'] = probabilities[:, i]
                except:
                    pass  # Skip if probability prediction fails
                    
            self.predictions = results_data
            
            self.progress_updated.emit(90)
            
            # Update results display
            self.update_results_display()
            
            # Enable export
            self.export_group.setEnabled(True)
            self.export_csv_btn.setEnabled(True)
            self.export_excel_btn.setEnabled(True)
            
            # Update info
            pred_count = len(predictions)
            self.prediction_info_label.setText(f"Predictions made: {pred_count} samples")
            
            self.progress_updated.emit(100)
            self.status_updated.emit(f"Predictions completed: {pred_count} samples")
            
            # Switch to results tab
            self.right_panel.setCurrentIndex(2)  # Results tab
            
            QMessageBox.information(self, "Success", f"Predictions completed for {pred_count} samples!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")
            self.progress_updated.emit(0)
            self.status_updated.emit("Prediction failed")
            
    def update_results_display(self):
        """Update prediction results table"""
        if self.predictions is None:
            return
            
        # Setup table
        self.results_table.setRowCount(len(self.predictions))
        self.results_table.setColumnCount(len(self.predictions.columns))
        self.results_table.setHorizontalHeaderLabels(self.predictions.columns.tolist())
        
        # Fill table
        for i in range(len(self.predictions)):
            for j, col in enumerate(self.predictions.columns):
                value = str(self.predictions.iloc[i, j])
                self.results_table.setItem(i, j, QTableWidgetItem(value))
                
        # Resize columns
        self.results_table.resizeColumnsToContents()
        
    def export_to_csv(self):
        """Export results to CSV"""
        if self.predictions is None:
            QMessageBox.warning(self, "Warning", "No predictions to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export to CSV", "predictions.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.predictions.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
                self.status_updated.emit(f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
                
    def export_to_excel(self):
        """Export results to Excel"""
        if self.predictions is None:
            QMessageBox.warning(self, "Warning", "No predictions to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export to Excel", "predictions.xlsx", "Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # Export predictions
                    self.predictions.to_excel(writer, sheet_name='Predictions', index=False)
                    
                    # Export model info if available
                    if self.model_metadata:
                        model_info_df = pd.DataFrame([
                            ['Model Type', self.model_metadata['model_type']],
                            ['Task Type', self.model_metadata['task_type']],
                            ['Number of Features', self.model_metadata['n_features']],
                            ['Pipeline Steps', ', '.join(self.model_metadata['pipeline_steps'])],
                        ], columns=['Property', 'Value'])
                        model_info_df.to_excel(writer, sheet_name='Model_Info', index=False)
                        
                        # Export feature names
                        if self.feature_names:
                            features_df = pd.DataFrame(self.feature_names, columns=['Feature_Names'])
                            features_df.to_excel(writer, sheet_name='Features', index=False)
                
                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
                self.status_updated.emit(f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
                
    def reset(self):
        """Reset the module"""
        self.trained_model = None
        self.model_metadata = None
        self.prediction_data = None
        self.predictions = None
        self.feature_names = []
        self.feature_inputs = {}
        self.feature_types = {}
        self.feature_bounds = {}
        
        # Reset UI
        self.model_info_text.clear()
        self.model_details_text.clear()
        self.data_info_label.setText("No data loaded")
        self.prediction_info_label.setText("No predictions made")
        
        # Clear tables
        self.data_table.setRowCount(0)
        self.results_table.setRowCount(0)
        
        # Disable groups
        self.model_info_group.setEnabled(False)
        self.prediction_mode_group.setEnabled(False)
        self.data_import_group.setEnabled(False)
        self.manual_input_group.setEnabled(False)
        self.prediction_group.setEnabled(False)
        self.export_group.setEnabled(False)
        
        # Reset buttons
        self.load_current_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.export_excel_btn.setEnabled(False)
        
        self.setEnabled(True)  # Keep module enabled for independent use
        
        # Re-enable model loading section
        self.enable_model_loading_section()
        
        self.status_updated.emit("Prediction module reset")
    
    def enable_model_loading_section(self):
        """Enable model loading section for independent use"""
        # Model loading is always available
        # Only disable prediction-related sections until model is loaded
        
        # Keep model loading section enabled
        # (model loading group is already enabled by default)
        
        # Initially disable other sections until model is loaded
        self.model_info_group.setEnabled(False)
        self.prediction_mode_group.setEnabled(False)
        self.data_import_group.setEnabled(False)
        self.manual_input_group.setEnabled(False)
        self.prediction_group.setEnabled(False)
        self.export_group.setEnabled(False)
        
        # Enable current session button only if there's a model
        self.load_current_btn.setEnabled(self.trained_model is not None)
        
        # Disable prediction-related buttons until model is ready
        self.predict_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.export_excel_btn.setEnabled(False)
    
    def setup_enhanced_manual_input_interface(self):
        """Setup enhanced manual input interface based on feature types and bounds"""
        # Clear existing widgets
        for i in reversed(range(self.feature_layout.count())):
            self.feature_layout.itemAt(i).widget().setParent(None)
        
        self.feature_inputs = {}
        
        # Check if we have feature types information
        if self.feature_types and self.feature_names:
            self.status_updated.emit("📋 Setting up interface with feature type information...")
            self.setup_typed_feature_interface()
        else:
            # Fallback to original setup
            self.status_updated.emit("📋 Setting up standard interface...")
            self.setup_manual_input_interface()
    
    def setup_typed_feature_interface(self):
        """Setup interface using feature type information"""
        from PyQt5.QtWidgets import QComboBox, QCheckBox
        
        # Debug: Print feature types and bounds for analysis
        print(f"[DEBUG] Setting up typed interface with {len(self.feature_names)} features")
        print(f"[DEBUG] Feature types available: {bool(self.feature_types)}")
        print(f"[DEBUG] Feature bounds available: {bool(self.feature_bounds)}")
        
        if self.feature_types:
            print(f"[DEBUG] Feature types summary:")
            type_counts = {}
            for fname, ftype in self.feature_types.items():
                type_counts[ftype] = type_counts.get(ftype, 0) + 1
                if fname in ['SMOKING_1', 'THROAT_DISCOMFORT_1', 'BREATHING_ISSUE_0', 'SMOKING_FAMILY_HISTORY_0']:
                    print(f"[DEBUG]   {fname}: {ftype}")
            print(f"[DEBUG] Type distribution: {type_counts}")
        
        for i, feature_name in enumerate(self.feature_names):
            # Get feature information
            feature_type = self.feature_types.get(feature_name, 'continuous')
            feature_bounds = self.feature_bounds.get(feature_name, (0.0, 1.0))
            
            print(f"[DEBUG] Feature {i}: {feature_name} -> type={feature_type}, bounds={feature_bounds}")
            
            # Create feature label with type indicator
            type_icon = {'continuous': '📈', 'binary': '🔘', 'categorical': '📝'}.get(feature_type, '📊')
            label = QLabel(f"{type_icon} {feature_name}:")
            label.setMinimumWidth(150)
            
            # Create appropriate input widget based on feature type
            if feature_type == 'binary':
                # Binary feature: use checkbox
                input_widget = QCheckBox()
                input_widget.setChecked(False)  # Default to False
                input_widget.setToolTip(f"Binary feature: 0 or 1\nRange: {feature_bounds[0]} to {feature_bounds[1]}")
                print(f"[DEBUG] Created CHECKBOX for {feature_name}")
                
            elif feature_type == 'categorical':
                # Categorical feature: use combobox with valid values
                input_widget = QComboBox()
                
                # Generate valid integer values within bounds
                min_val, max_val = feature_bounds
                valid_values = list(range(int(min_val), int(max_val) + 1))
                input_widget.addItems([str(v) for v in valid_values])
                input_widget.setCurrentIndex(0)  # Default to first value
                input_widget.setToolTip(f"Categorical feature\nValid values: {valid_values}\nRange: {feature_bounds[0]} to {feature_bounds[1]}")
                input_widget.setMinimumWidth(120)
                print(f"[DEBUG] Created COMBOBOX for {feature_name}")
                
            else:  # continuous
                # Continuous feature: use double spinbox
                input_widget = QDoubleSpinBox()
                min_val, max_val = feature_bounds
                
                # Set bounds with some padding
                padding = (max_val - min_val) * 0.1 if max_val > min_val else 1.0
                input_widget.setMinimum(min_val - padding)
                input_widget.setMaximum(max_val + padding)
                input_widget.setDecimals(6)
                
                # Set default value to middle of range
                default_val = (min_val + max_val) / 2
                input_widget.setValue(default_val)
                input_widget.setMinimumWidth(120)
                input_widget.setToolTip(f"Continuous feature\nOriginal range: {min_val:.2f} to {max_val:.2f}\nDefault: {default_val:.2f}")
                print(f"[DEBUG] Created DOUBLESPINBOX for {feature_name}")
            
            # Add to layout
            row = i
            self.feature_layout.addWidget(label, row, 0)
            self.feature_layout.addWidget(input_widget, row, 1)
            
            # Store reference
            self.feature_inputs[feature_name] = input_widget
        
        print(f"[SUCCESS] Created enhanced interface with {len(self.feature_names)} features")
        self.status_updated.emit(f"✅ Enhanced interface ready: {len(self.feature_names)} features with type information") 