"""
ALIGNN UI Widget
Integrated UI for ALIGNN module
Based on CGCNN UI style for consistency
"""

import os
import sys
from typing import Optional, Dict, Any
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QFileDialog,
    QTextEdit, QProgressBar, QMessageBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QSplitter, QCheckBox, QDialog, QDialogButtonBox,
    QApplication, QProgressDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .config import TrainingConfig, ALIGNNConfig

# Lazy imports for DGL-dependent modules
# These will be imported when actually needed (when user clicks train button)
# from .data_loader import ALIGNNDataLoader
# from .model import ALIGNN
# from .trainer import ALIGNNTrainer


class ColumnSelectionDialog(QDialog):
    """Dialog for selecting ID and property columns from data file."""

    def __init__(self, columns, data_preview=None, parent=None):
        """
        Initialize column selection dialog.

        Parameters
        ----------
        columns : list of str
            Available column names
        data_preview : pandas.DataFrame, optional
            First few rows for preview
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.columns = columns
        self.data_preview = data_preview
        self.id_column = None
        self.property_column = None

        self.setWindowTitle("Select Columns")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Please select which columns to use for matching CIF files and training.\n\n"
            "• ID Column: Used to match CIF filenames (e.g., 'structure_id', 'material_id')\n"
            "• Property Column: Target property for prediction (e.g., 'formation_energy', 'band_gap')"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Data preview
        if self.data_preview is not None:
            preview_label = QLabel("<b>Data Preview (first 5 rows):</b>")
            layout.addWidget(preview_label)

            preview_table = QTableWidget()
            preview_table.setRowCount(len(self.data_preview))
            preview_table.setColumnCount(len(self.columns))
            preview_table.setHorizontalHeaderLabels(self.columns)

            for i, row in enumerate(self.data_preview.itertuples(index=False)):
                for j, value in enumerate(row):
                    preview_table.setItem(i, j, QTableWidgetItem(str(value)))

            preview_table.resizeColumnsToContents()
            preview_table.setMaximumHeight(200)
            layout.addWidget(preview_table)

        # Column selection
        selection_group = QGroupBox("Column Selection")
        selection_layout = QVBoxLayout(selection_group)

        # ID column
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("ID Column (for matching CIF files):"))
        self.id_combo = QComboBox()
        self.id_combo.addItems(self.columns)
        # Try to auto-select common ID column names
        for i, col in enumerate(self.columns):
            if any(keyword in col.lower() for keyword in ['id', 'name', 'structure', 'jid', 'mp_id']):
                self.id_combo.setCurrentIndex(i)
                break
        id_layout.addWidget(self.id_combo)
        selection_layout.addLayout(id_layout)

        # Property column
        prop_layout = QHBoxLayout()
        prop_layout.addWidget(QLabel("Property Column (target value):"))
        self.property_combo = QComboBox()
        self.property_combo.addItems(self.columns)
        # Try to auto-select second column if available
        if len(self.columns) > 1:
            self.property_combo.setCurrentIndex(1)
        prop_layout.addWidget(self.property_combo)
        selection_layout.addLayout(prop_layout)

        layout.addWidget(selection_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def accept(self):
        """Handle OK button."""
        self.id_column = self.id_combo.currentText()
        self.property_column = self.property_combo.currentText()

        # Validate selection
        if self.id_column == self.property_column:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "ID column and Property column must be different!"
            )
            return

        super().accept()

    def get_selection(self):
        """Get selected columns."""
        return self.id_column, self.property_column


class TrainingThread(QThread):
    """Background thread for model training"""

    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, trainer, train_loader, val_loader, test_loader):
        super().__init__()
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def run(self):
        """Run training"""
        try:
            # Setup callback
            def progress_callback(info):
                self.progress.emit(info)

            # Train
            results = self.trainer.train(
                self.train_loader,
                self.val_loader,
                self.test_loader,
                progress_callback=progress_callback
            )

            self.finished.emit(results)

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class ALIGNNModule(QWidget):
    """
    ALIGNN Module Widget

    Main UI for ALIGNN training and prediction
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self.config = TrainingConfig()
        self.data_loader = None
        self.model = None
        self.trainer = None
        self.training_thread = None
        self.results = None
        self.prediction_results = None

        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset = None

        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("ALIGNN - Atomistic Line Graph Neural Network")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Main content
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Configuration
        left_panel = self.create_config_panel()
        splitter.addWidget(left_panel)

        # Right panel - Results
        right_panel = self.create_results_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 800])

        # Status bar
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        layout.addLayout(status_layout)

    def create_config_panel(self):
        """Create configuration panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Data section - CIF Folder + Property Table Workflow (like CGCNN)
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout(data_group)

        # Instructions
        instruction_label = QLabel(
            "<i>Please provide: (1) CIF files directory, (2) Property CSV/Excel file</i>"
        )
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("color: #666; font-size: 9pt;")
        data_layout.addWidget(instruction_label)

        # CIF directory
        cif_dir_layout = QHBoxLayout()
        cif_dir_layout.addWidget(QLabel("CIF Directory:"))
        self.cif_dir_edit = QLineEdit()
        self.cif_dir_edit.setPlaceholderText("Directory with .cif files")
        cif_dir_layout.addWidget(self.cif_dir_edit)
        browse_cif_btn = QPushButton("Browse")
        browse_cif_btn.clicked.connect(self.browse_cif_dir)
        cif_dir_layout.addWidget(browse_cif_btn)
        data_layout.addLayout(cif_dir_layout)

        # Property file
        prop_file_layout = QHBoxLayout()
        prop_file_layout.addWidget(QLabel("Property File:"))
        self.property_file_edit = QLineEdit()
        self.property_file_edit.setPlaceholderText("CSV/Excel with ID and property columns")
        prop_file_layout.addWidget(self.property_file_edit)
        browse_prop_btn = QPushButton("Browse")
        browse_prop_btn.clicked.connect(self.browse_property_file)
        prop_file_layout.addWidget(browse_prop_btn)
        data_layout.addLayout(prop_file_layout)

        # Auto-prepare button
        self.auto_prepare_btn = QPushButton("Prepare and Load Dataset")
        self.auto_prepare_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.auto_prepare_btn.setToolTip(
            "Click to prepare dataset from CIF files and CSV/Excel.\n"
            "You will be able to select which columns to use for ID and property."
        )
        self.auto_prepare_btn.clicked.connect(self.auto_prepare_and_load)
        data_layout.addWidget(self.auto_prepare_btn)

        # Dataset info
        self.data_info_label = QLabel("No dataset loaded")
        self.data_info_label.setStyleSheet("color: gray;")
        data_layout.addWidget(self.data_info_label)

        # Hidden fields for backward compatibility
        self.structure_file_edit = QLineEdit()
        self.structure_file_edit.setVisible(False)
        self.target_combo = QComboBox()
        self.target_combo.setVisible(False)

        layout.addWidget(data_group)

        # Model configuration
        config_group = QGroupBox("Model Configuration")
        config_layout = QVBoxLayout(config_group)

        # Preset selector
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Property Prediction (Regression)",
            "Formation Energy",
            "Band Gap",
            "Classification Task",
            "Quick Test",
            "Custom"
        ])
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)
        config_layout.addLayout(preset_layout)

        # Task type and classification settings
        task_layout = QVBoxLayout()

        # Task type
        task_type_layout = QHBoxLayout()
        task_type_layout.addWidget(QLabel("Task Type:"))
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["Regression", "Classification"])
        self.task_type_combo.currentTextChanged.connect(self.on_task_type_changed)
        task_type_layout.addWidget(self.task_type_combo)
        task_layout.addLayout(task_type_layout)

        # Classification settings (hidden by default)
        self.classification_group = QGroupBox("Classification Settings")
        classification_layout = QVBoxLayout(self.classification_group)

        # Number of classes
        classes_layout = QHBoxLayout()
        classes_layout.addWidget(QLabel("Number of Classes:"))
        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(2, 100)
        self.num_classes_spin.setValue(2)
        classes_layout.addWidget(self.num_classes_spin)
        classification_layout.addLayout(classes_layout)

        # Classification threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold (optional):"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setRange(-10.0, 10.0)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setSpecialValueText("Auto")
        threshold_layout.addWidget(self.threshold_spin)
        classification_layout.addLayout(threshold_layout)

        task_layout.addWidget(self.classification_group)
        self.classification_group.setVisible(False)  # Initially hidden

        config_layout.addLayout(task_layout)

        # Architecture
        arch_layout = QHBoxLayout()
        arch_layout.addWidget(QLabel("ALIGNN Layers:"))
        self.alignn_layers_spin = QSpinBox()
        self.alignn_layers_spin.setRange(1, 10)
        self.alignn_layers_spin.setValue(4)
        arch_layout.addWidget(self.alignn_layers_spin)
        arch_layout.addWidget(QLabel("GCN Layers:"))
        self.gcn_layers_spin = QSpinBox()
        self.gcn_layers_spin.setRange(1, 10)
        self.gcn_layers_spin.setValue(4)
        arch_layout.addWidget(self.gcn_layers_spin)
        config_layout.addLayout(arch_layout)

        # Training parameters
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(300)
        self.add_param_row(config_layout, "Epochs:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.01)
        self.add_param_row(config_layout, "Learning Rate:", self.lr_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(64)
        self.add_param_row(config_layout, "Batch Size:", self.batch_spin)

        # Data split
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel("Train/Val/Test:"))
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.9)
        self.train_ratio_spin.setSingleStep(0.1)
        self.train_ratio_spin.setValue(0.8)
        split_layout.addWidget(self.train_ratio_spin)
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.05, 0.5)
        self.val_ratio_spin.setSingleStep(0.05)
        self.val_ratio_spin.setValue(0.1)
        split_layout.addWidget(self.val_ratio_spin)
        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.05, 0.5)
        self.test_ratio_spin.setSingleStep(0.05)
        self.test_ratio_spin.setValue(0.1)
        split_layout.addWidget(self.test_ratio_spin)
        config_layout.addLayout(split_layout)

        # Advanced options
        self.use_gpu_check = QCheckBox("Use GPU (if available)")
        self.use_gpu_check.setChecked(True)
        config_layout.addWidget(self.use_gpu_check)

        # Random seed for reproducibility
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Random Seed:"))
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 999999)
        self.random_seed_spin.setValue(123)
        self.random_seed_spin.setToolTip("Random seed for reproducible training results")
        seed_layout.addWidget(self.random_seed_spin)
        seed_layout.addStretch()
        config_layout.addLayout(seed_layout)

        # Early Stopping settings
        early_stop_group = QGroupBox("Early Stopping (Optional)")
        early_stop_layout = QVBoxLayout(early_stop_group)

        # Enable early stopping
        self.enable_early_stopping_check = QCheckBox("Enable Early Stopping")
        self.enable_early_stopping_check.setChecked(False)
        self.enable_early_stopping_check.setToolTip(
            "Stop training when validation metric stops improving"
        )
        self.enable_early_stopping_check.stateChanged.connect(self.on_early_stopping_toggled)
        early_stop_layout.addWidget(self.enable_early_stopping_check)

        # Patience
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("Patience:"))
        self.early_stop_patience_spin = QSpinBox()
        self.early_stop_patience_spin.setRange(1, 500)
        self.early_stop_patience_spin.setValue(50)
        self.early_stop_patience_spin.setToolTip(
            "Number of epochs with no improvement after which training will be stopped"
        )
        self.early_stop_patience_spin.setEnabled(False)
        patience_layout.addWidget(self.early_stop_patience_spin)
        patience_layout.addStretch()
        early_stop_layout.addLayout(patience_layout)

        # Min delta
        min_delta_layout = QHBoxLayout()
        min_delta_layout.addWidget(QLabel("Min Delta:"))
        self.early_stop_min_delta_spin = QDoubleSpinBox()
        self.early_stop_min_delta_spin.setRange(0.0, 1.0)
        self.early_stop_min_delta_spin.setDecimals(6)
        self.early_stop_min_delta_spin.setSingleStep(0.0001)
        self.early_stop_min_delta_spin.setValue(0.0)
        self.early_stop_min_delta_spin.setToolTip(
            "Minimum change in validation metric to qualify as an improvement"
        )
        self.early_stop_min_delta_spin.setEnabled(False)
        min_delta_layout.addWidget(self.early_stop_min_delta_spin)
        min_delta_layout.addStretch()
        early_stop_layout.addLayout(min_delta_layout)

        config_layout.addWidget(early_stop_group)

        layout.addWidget(config_group)

        # Training section
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout(train_group)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        train_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        train_layout.addWidget(self.stop_btn)

        layout.addWidget(train_group)

        # Model Management section
        model_group = QGroupBox("Model Management")
        model_layout = QVBoxLayout(model_group)

        # Save model
        save_layout = QHBoxLayout()
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        save_layout.addWidget(self.save_model_btn)
        model_layout.addLayout(save_layout)

        # Load model
        load_layout = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        load_layout.addWidget(self.load_model_btn)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to model checkpoint")
        load_layout.addWidget(self.model_path_edit)
        browse_model_btn = QPushButton("Browse")
        browse_model_btn.clicked.connect(self.browse_model_file)
        load_layout.addWidget(browse_model_btn)
        model_layout.addLayout(load_layout)

        # Model status
        self.model_status_label = QLabel("No model loaded")
        model_layout.addWidget(self.model_status_label)

        layout.addWidget(model_group)

        # Prediction section
        pred_group = QGroupBox("Prediction")
        pred_layout = QVBoxLayout(pred_group)

        # Prediction file
        pred_file_layout = QHBoxLayout()
        pred_file_layout.addWidget(QLabel("Data File:"))
        self.pred_file_edit = QLineEdit()
        self.pred_file_edit.setPlaceholderText("Path to prediction data")
        pred_file_layout.addWidget(self.pred_file_edit)
        browse_pred_btn = QPushButton("Browse")
        browse_pred_btn.clicked.connect(self.browse_pred_file)
        pred_file_layout.addWidget(browse_pred_btn)
        pred_layout.addLayout(pred_file_layout)

        # Predict button
        self.predict_btn = QPushButton("Make Predictions")
        self.predict_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.predict_btn.clicked.connect(self.make_predictions)
        self.predict_btn.setEnabled(False)
        pred_layout.addWidget(self.predict_btn)

        # Save predictions
        save_pred_layout = QHBoxLayout()
        self.save_predictions_check = QCheckBox("Save predictions to:")
        self.save_predictions_check.setChecked(True)
        save_pred_layout.addWidget(self.save_predictions_check)
        self.pred_output_edit = QLineEdit()
        self.pred_output_edit.setText("predictions.csv")
        save_pred_layout.addWidget(self.pred_output_edit)
        pred_layout.addLayout(save_pred_layout)

        layout.addWidget(pred_group)

        layout.addStretch()

        return panel

    def create_results_panel(self):
        """Create results display panel"""
        tabs = QTabWidget()

        # System info tab
        system_info_widget = self.create_system_info_tab()
        tabs.addTab(system_info_widget, "System Info")

        # Training log tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        tabs.addTab(log_widget, "Training Log")

        # Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        results_layout.addWidget(self.metrics_table)

        # Plot
        self.figure = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)

        tabs.addTab(results_widget, "Results & Visualization")

        # Prediction Results tab
        pred_results_widget = QWidget()
        pred_results_layout = QVBoxLayout(pred_results_widget)

        # Prediction table
        self.pred_table = QTableWidget()
        self.pred_table.setColumnCount(2)
        self.pred_table.setHorizontalHeaderLabels(["ID", "Prediction"])
        pred_results_layout.addWidget(self.pred_table)

        # Export predictions button
        export_layout = QHBoxLayout()
        self.export_pred_btn = QPushButton("Export to CSV")
        self.export_pred_btn.clicked.connect(self.export_predictions)
        self.export_pred_btn.setEnabled(False)
        export_layout.addWidget(self.export_pred_btn)
        export_layout.addStretch()
        pred_results_layout.addLayout(export_layout)

        tabs.addTab(pred_results_widget, "Predictions")

        return tabs

    def create_system_info_tab(self):
        """Create system information tab with comprehensive ALIGNN-specific checks"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Refresh button
        refresh_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh System Info")
        refresh_btn.clicked.connect(self.refresh_system_info)
        refresh_layout.addWidget(refresh_btn)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)

        # System info text
        self.system_info_text = QTextEdit()
        self.system_info_text.setReadOnly(True)
        self.system_info_text.setFont(QFont("Consolas", 8))

        # Load initial system info
        self.refresh_system_info()

        layout.addWidget(self.system_info_text)

        return widget

    def refresh_system_info(self):
        """Refresh system information using ALIGNN-specific checker"""
        try:
            # Try to import and use the ALIGNN system info
            from .system_info import ALIGNNSystemInfo

            checker = ALIGNNSystemInfo()
            checker.check_all()
            system_info_text = checker.get_summary()

            # Add recommendations
            recommendations = checker.get_recommendations()
            if recommendations:
                system_info_text += "\n\n[ALIGNN Recommendations]\n"
                for i, rec in enumerate(recommendations, 1):
                    system_info_text += f"  {i}. {rec}\n"

            self.system_info_text.setText(system_info_text)

        except ImportError:
            # Fallback to original system info if new module not available
            self._get_fallback_system_info()
        except Exception as e:
            # Error handling
            error_text = f"Error loading system information: {e}\n\n"
            error_text += "Using fallback system information...\n\n"
            self.system_info_text.setText(error_text)
            self._get_fallback_system_info()

    def _get_fallback_system_info(self):
        """Fallback system info method (original implementation with ALIGNN element limitation added)"""
        info_lines = []
        info_lines.append("="*70)
        info_lines.append("ALIGNN Module - System Information & Environment")
        info_lines.append("="*70)

        try:
            import platform
            import psutil
            import sys

            # Platform Info
            info_lines.append("\n[Platform]")
            info_lines.append(f"  OS: {platform.system()} {platform.release()} {platform.version()}")
            info_lines.append(f"  Architecture: {platform.machine()}")
            info_lines.append(f"  Processor: {platform.processor()}")

            # Python Info
            info_lines.append("\n[Python]")
            info_lines.append(f"  Version: {platform.python_version()}")
            info_lines.append(f"  Executable: {sys.executable}")

            # PyTorch Info
            info_lines.append("\n[PyTorch]")
            import torch
            info_lines.append(f"  Version: {torch.__version__}")
            info_lines.append(f"  CUDA Compiled: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                info_lines.append(f"  CUDA Available: True")
                info_lines.append(f"  CUDA Version: {torch.version.cuda}")
                info_lines.append(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            else:
                info_lines.append(f"  CUDA Available: False (CPU mode only)")

            # GPU Info
            info_lines.append("\n[GPU]")
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                info_lines.append(f"  Count: {gpu_count}")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_memory_gb = gpu_props.total_memory / (1024**3)
                    info_lines.append(f"  GPU {i}: {gpu_name}")
                    info_lines.append(f"    Memory: {gpu_memory_gb:.2f} GB")

                    # ALIGNN-specific GPU recommendations
                    if gpu_memory_gb >= 16:
                        info_lines.append(f"    Status: Excellent for large ALIGNN datasets")
                    elif gpu_memory_gb >= 8:
                        info_lines.append(f"    Status: Good for medium ALIGNN datasets")
                    elif gpu_memory_gb >= 4:
                        info_lines.append(f"    Status: Suitable for small ALIGNN datasets")
                    else:
                        info_lines.append(f"    Status: May be insufficient for ALIGNN")
            else:
                info_lines.append(f"  No GPU detected - ALIGNN training will be VERY slow")

            # System Resources
            info_lines.append("\n[System Resources - ALIGNN Requirements]")
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            info_lines.append(f"  CPU Cores (Physical): {cpu_count}")
            info_lines.append(f"  CPU Cores (Logical): {cpu_count_logical}")

            memory = psutil.virtual_memory()
            ram_gb = memory.total / (1024**3)
            info_lines.append(f"  RAM Total: {ram_gb:.2f} GB")
            info_lines.append(f"  RAM Available: {memory.available / (1024**3):.2f} GB")
            info_lines.append(f"  RAM Usage: {memory.percent}%")

            # ALIGNN-specific RAM recommendations
            if ram_gb >= 32:
                info_lines.append(f"  RAM Status: Excellent for large ALIGNN graphs")
            elif ram_gb >= 16:
                info_lines.append(f"  RAM Status: Good for most ALIGNN tasks")
            elif ram_gb >= 8:
                info_lines.append(f"  RAM Status: Minimum for small ALIGNN datasets")
            else:
                info_lines.append(f"  RAM Status: Insufficient for ALIGNN training")

            # ALIGNN Element Support - THE KEY ADDITION
            info_lines.append("\n[ALIGNN ELEMENT SUPPORT LIMITATION]")
            info_lines.append("  *** IMPORTANT LIMITATION ***")
            info_lines.append("  Supported Elements: Z=1 (H) to Z=92 (U)")
            info_lines.append("  Max Atomic Number: 92")
            info_lines.append("  Feature Dimension: 92-dimensional CGCNN features")
            info_lines.append("  Coverage: 99.9% of real materials")
            info_lines.append("")
            info_lines.append("  WARNING: Materials with superheavy elements (Z>92) will fail!")
            info_lines.append("  - Elements like Z=93-118 will cause IndexError")
            info_lines.append("  - This is by design, not a bug")
            info_lines.append("  - Matches reference ALIGNN implementation")
            info_lines.append("  - Examples: Neptunium (Z=93), Oganesson (Z=118) not supported")
            info_lines.append("")
            info_lines.append("  Solution: Use only materials with elements Z <= 92")

            # ALIGNN Dependencies
            info_lines.append("\n[Critical ALIGNN Dependencies]")
            deps = {
                'torch': 'PyTorch (Required)',
                'dgl': 'DGL - Deep Graph Library (CRITICAL)',
                'jarvis': 'JARVIS-Tools (Required for features)',
                'numpy': 'NumPy',
                'pandas': 'Pandas',
                'pymatgen': 'Pymatgen (Material structures)',
                'sklearn': 'Scikit-learn',
                'pydantic': 'Pydantic (Configuration)',
            }

            for module_name, display_name in deps.items():
                try:
                    if module_name == 'dgl':
                        # Special handling for DGL
                        import dgl
                        version = getattr(dgl, '__version__', 'installed')
                        info_lines.append(f"  [OK] {display_name}: {version}")
                    elif module_name == 'sklearn':
                        import sklearn
                        version = sklearn.__version__
                        info_lines.append(f"  [OK] {display_name}: {version}")
                    elif module_name == 'jarvis':
                        import jarvis
                        version = getattr(jarvis, '__version__', 'installed')
                        info_lines.append(f"  [OK] {display_name}: {version}")
                    else:
                        mod = __import__(module_name)
                        version = getattr(mod, '__version__', 'installed')
                        info_lines.append(f"  [OK] {display_name}: {version}")
                except ImportError:
                    info_lines.append(f"  [MISSING] {display_name}")
                except Exception as e:
                    info_lines.append(f"  [ERROR] {display_name}: {str(e)[:40]}...")

            # ALIGNN Training Recommendations
            info_lines.append("\n[ALIGNN Training Recommendations]")
            info_lines.append("  Batch Size: Start with 32-64, adjust based on GPU memory")
            info_lines.append("  Learning Rate: 0.001-0.01 with OneCycleLR scheduler")
            info_lines.append("  Epochs: 200-500 depending on dataset size")
            info_lines.append("  Early Stopping: Recommended with patience=50")
            info_lines.append("  Graph Strategy: k-nearest (default) or radius (more accurate)")

            # Status Summary
            info_lines.append("\n" + "="*70)
            if torch.cuda.is_available():
                info_lines.append("GPU STATUS: Ready for ALIGNN training")
            else:
                info_lines.append("GPU STATUS: CPU only - training will be slow")

            # Final important reminders
            info_lines.append("\nIMPORTANT REMINDERS:")
            info_lines.append("1. ONLY use materials with elements Z=1 to Z=92")
            info_lines.append("2. Superheavy elements (Z>92) will cause errors")
            info_lines.append("3. This limitation matches reference ALIGNN")
            info_lines.append("4. 99.9% of real materials are supported")
            info_lines.append("="*70)

        except Exception as e:
            info_lines.append(f"\n[Error]")
            info_lines.append(f"  Failed to get system info: {e}")
            import traceback
            info_lines.append(f"\n{traceback.format_exc()}")

        self.system_info_text.setText("\n".join(info_lines))

    def add_param_row(self, layout, label_text, widget):
        """Add parameter row to layout"""
        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel(label_text))
        row_layout.addWidget(widget)
        row_layout.addStretch()
        layout.addLayout(row_layout)

    def browse_structure_file(self):
        """Browse for structure file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Structure File",
            "",
            "Data Files (*.csv *.xlsx);;All Files (*)"
        )
        if filename:
            self.structure_file_edit.setText(filename)

    def browse_property_file(self):
        """Browse for property file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Property File",
            "",
            "Data Files (*.csv *.xlsx);;All Files (*)"
        )
        if filename:
            self.property_file_edit.setText(filename)

    def browse_cif_dir(self):
        """Browse for CIF directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select CIF Directory",
            "",
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.cif_dir_edit.setText(directory)

    def auto_prepare_and_load(self):
        """Auto-prepare dataset from CIF directory and property file (CGCNN-style workflow)."""
        cif_dir = self.cif_dir_edit.text()
        prop_file = self.property_file_edit.text()

        # Validate inputs
        if not cif_dir or not os.path.exists(cif_dir):
            QMessageBox.warning(self, "Error", "Please select a valid CIF directory")
            return

        if not prop_file or not os.path.exists(prop_file):
            QMessageBox.warning(self, "Error", "Please select a valid property file")
            return

        try:
            # Read property file to show column selection dialog
            self.status_label.setText("Reading property file...")
            self.log_text.append(f"\n{'='*50}")
            self.log_text.append("Auto-preparing ALIGNN dataset...")

            if prop_file.endswith('.csv'):
                df = pd.read_csv(prop_file)
            elif prop_file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(prop_file)
            else:
                QMessageBox.warning(
                    self, "Error",
                    "Unsupported file format. Please use CSV or Excel files."
                )
                return

            # Show column selection dialog
            dialog = ColumnSelectionDialog(
                columns=list(df.columns),
                data_preview=df.head(5),
                parent=self
            )

            if dialog.exec_() != QDialog.Accepted:
                self.status_label.setText("Cancelled")
                return

            id_column, property_column = dialog.get_selection()

            self.log_text.append(f"CIF directory: {cif_dir}")
            self.log_text.append(f"Property file: {prop_file}")
            self.log_text.append(f"ID column: '{id_column}'")
            self.log_text.append(f"Property column: '{property_column}'")

            # Validate columns exist
            if id_column not in df.columns or property_column not in df.columns:
                QMessageBox.critical(
                    self, "Error",
                    f"Selected columns not found in file!"
                )
                return

            # Auto-create output directory
            cif_parent = os.path.dirname(cif_dir)
            output_dir = os.path.join(cif_parent, "alignn_prepared")

            # If directory exists, create a new one with timestamp
            if os.path.exists(output_dir):
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(cif_parent, f"alignn_prepared_{timestamp}")

            os.makedirs(output_dir, exist_ok=True)
            self.log_text.append(f"Output directory: {output_dir}")

            # Create progress dialog
            progress = QProgressDialog(
                "Preparing dataset...",
                "Cancel",
                0,
                100,
                self
            )
            progress.setWindowTitle("Dataset Preparation")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            # Update progress: Loading data
            progress.setLabelText("Loading property data...")
            progress.setValue(10)
            QApplication.processEvents()

            if progress.wasCanceled():
                self.status_label.setText("Cancelled")
                return

            self.status_label.setText("Preparing dataset...")

            # Update progress: Matching files
            progress.setLabelText(f"Matching {len(df)} structures with CIF files...")
            progress.setValue(30)
            QApplication.processEvents()

            if progress.wasCanceled():
                self.status_label.setText("Cancelled")
                return

            # Use DataMatcher to prepare dataset
            from .data_matcher import DataMatcher
            matcher = DataMatcher()

            try:
                prepared_dir, stats = matcher.prepare_dataset(
                    cif_dir=cif_dir,
                    property_file=prop_file,
                    output_dir=output_dir,
                    id_column=id_column,
                    property_column=property_column
                )

                # Update progress: Processing complete
                progress.setLabelText("Finalizing dataset preparation...")
                progress.setValue(90)
                QApplication.processEvents()

            except Exception as e:
                progress.close()
                QMessageBox.critical(self, "Error", f"Failed to prepare dataset:\n{str(e)}")
                self.status_label.setText("Error")
                return

            # Complete progress
            progress.setValue(100)
            progress.close()

            # Show statistics
            self.log_text.append("\nDataset Preparation Results:")
            self.log_text.append(f"  Total CIF files: {stats['total_cif_files']}")
            self.log_text.append(f"  Total property rows: {stats['total_property_rows']}")
            self.log_text.append(f"  Successfully matched: {stats['matched']}")
            self.log_text.append(f"  Missing CIF files: {stats['missing_cif']}")
            self.log_text.append(f"  CIF files without properties: {stats['missing_property']}")

            # Show matching strategy statistics
            if 'strategy_stats' in stats and stats['strategy_stats']:
                self.log_text.append("\nMatching Strategies Used:")
                for strategy, count in stats['strategy_stats'].items():
                    strategy_name = {
                        'exact': 'Exact match',
                        'remove_ext': 'After removing .cif extension',
                        'case_insensitive': 'Case-insensitive match',
                        'case_insensitive_no_ext': 'Case-insensitive without .cif',
                        'normalized': 'Normalized (no separators)',
                        'fuzzy': 'Fuzzy match (similarity-based)'
                    }.get(strategy, strategy)
                    self.log_text.append(f"  {strategy_name}: {count} matches")

            if stats['missing_cif'] > 0 and stats['missing_cif_samples']:
                self.log_text.append(f"\nFirst missing CIF samples: {stats['missing_cif_samples'][:5]}")
            if stats['missing_property'] > 0 and stats['missing_property_samples']:
                self.log_text.append(f"\nFirst CIF without properties: {stats['missing_property_samples'][:5]}")

            # Show summary dialog
            summary_msg = (
                f"Dataset prepared successfully!\n\n"
                f"Matched structures: {stats['matched']}\n"
                f"Missing CIF files: {stats['missing_cif']}\n"
                f"CIF files without properties: {stats['missing_property']}\n\n"
                f"Output directory: {prepared_dir}\n\n"
                f"Load this dataset now?"
            )

            reply = QMessageBox.question(
                self,
                "Dataset Prepared",
                summary_msg,
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Load the prepared dataset
                self._load_prepared_dataset(prepared_dir, property_column)
            else:
                self.status_label.setText("Dataset prepared (not loaded)")
                self.data_info_label.setText(f"Prepared: {stats['matched']} samples in {prepared_dir}")

        except Exception as e:
            import traceback
            error_msg = f"Failed to prepare dataset:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
            self.log_text.append(f"ERROR: {str(e)}")
            self.status_label.setText("Dataset preparation failed")

    def _load_prepared_dataset(self, prepared_dir, target_property):
        """Load prepared ALIGNN dataset from directory."""
        try:
            self.status_label.setText("Loading prepared dataset...")
            self.log_text.append(f"\n{'='*50}")
            self.log_text.append(f"Loading dataset from: {prepared_dir}")

            # Lazy import of data module
            from .data import CIFData, get_train_val_test_loader, collate_line_graph

            # Create dataset
            self.dataset = CIFData(
                root_dir=prepared_dir,
                max_num_nbr=self.batch_spin.value() if hasattr(self, 'max_neighbors_spin') else 12,
                radius=8.0,
                atom_features="cgcnn",
                compute_line_graph=True,
                neighbor_strategy="k-nearest"
            )

            # Get data loaders
            self.train_loader, self.val_loader, self.test_loader = get_train_val_test_loader(
                dataset=self.dataset,
                collate_fn=collate_line_graph,
                batch_size=self.batch_spin.value(),
                train_ratio=self.train_ratio_spin.value(),
                val_ratio=self.val_ratio_spin.value(),
                test_ratio=self.test_ratio_spin.value(),
                num_workers=0,
                pin_memory=False
            )

            # Store target column for later use
            self.target_column = target_property

            # Update UI
            total = len(self.dataset)
            train_size = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else int(total * self.train_ratio_spin.value())
            val_size = int(total * self.val_ratio_spin.value())
            test_size = int(total * self.test_ratio_spin.value())

            self.data_info_label.setText(
                f"Loaded: {total} samples | Train: {train_size} | Val: {val_size} | Test: {test_size}"
            )
            self.data_info_label.setStyleSheet("color: green; font-weight: bold;")
            self.train_btn.setEnabled(True)
            self.status_label.setText("Dataset loaded successfully")
            self.log_text.append(f"Dataset loaded: {total} samples")
            self.log_text.append(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

        except Exception as e:
            import traceback
            error_msg = f"Failed to load dataset:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
            self.log_text.append(f"ERROR: {str(e)}")
            self.status_label.setText("Failed to load dataset")

    def browse_data_file(self):
        """Browse for data file (backward compatibility)"""
        self.browse_structure_file()

    def on_task_type_changed(self, task_type):
        """Handle task type change"""
        is_classification = (task_type == "Classification")
        self.classification_group.setVisible(is_classification)

    def on_early_stopping_toggled(self, state):
        """Handle early stopping checkbox state change."""
        enabled = (state == Qt.Checked)
        self.early_stop_patience_spin.setEnabled(enabled)
        self.early_stop_min_delta_spin.setEnabled(enabled)

    def load_structures(self):
        """Step 1: Load structure file"""
        structure_file = self.structure_file_edit.text()

        if not structure_file:
            QMessageBox.warning(self, "Warning", "Please select a structure file")
            return

        if not os.path.exists(structure_file):
            QMessageBox.warning(self, "Warning", "Structure file does not exist")
            return

        try:
            self.status_label.setText("Loading structures...")
            self.log_text.append(f"\n{'='*50}")
            self.log_text.append(f"Step 1: Loading structures from: {structure_file}")

            # Load file
            if structure_file.endswith('.csv'):
                self.structure_df = pd.read_csv(structure_file)
            elif structure_file.endswith('.xlsx'):
                self.structure_df = pd.read_excel(structure_file)
            else:
                QMessageBox.warning(self, "Warning", "Unsupported file format. Use CSV or Excel.")
                return

            # Check for 'atoms' column
            if 'atoms' not in self.structure_df.columns:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Structure file must contain an 'atoms' column with Pymatgen Structure objects!"
                )
                return

            num_structures = len(self.structure_df)
            structure_columns = self.structure_df.columns.tolist()

            self.log_text.append(f"Loaded {num_structures} structures")
            self.log_text.append(f"Available columns: {', '.join(structure_columns)}")

            # Populate structure ID dropdown
            self.structure_id_combo.clear()
            self.structure_id_combo.addItems(structure_columns)

            # Auto-select ID column
            for col in structure_columns:
                if 'id' in col.lower() or 'name' in col.lower():
                    self.structure_id_combo.setCurrentText(col)
                    break

            # Update UI
            self.structure_info_label.setText(
                f"Loaded: {num_structures} structures | Columns: {len(structure_columns)}"
            )
            self.structure_info_label.setStyleSheet("color: green; font-weight: bold;")
            self.load_property_btn.setEnabled(True)
            self.data_info_label.setText("Step 1 Complete - Continue to Step 2")
            self.status_label.setText("Structures loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load structures:\n{str(e)}")
            self.log_text.append(f"Error: {str(e)}")
            self.status_label.setText("Failed to load structures")

    def load_properties(self):
        """Step 2: Load property table"""
        property_file = self.property_file_edit.text()

        if not property_file:
            QMessageBox.warning(self, "Warning", "Please select a property file")
            return

        if not os.path.exists(property_file):
            QMessageBox.warning(self, "Warning", "Property file does not exist")
            return

        try:
            self.status_label.setText("Loading properties...")
            self.log_text.append(f"\n{'='*50}")
            self.log_text.append(f"Step 2: Loading properties from: {property_file}")

            # Load file
            if property_file.endswith('.csv'):
                self.property_df = pd.read_csv(property_file)
            elif property_file.endswith('.xlsx'):
                self.property_df = pd.read_excel(property_file)
            else:
                QMessageBox.warning(self, "Warning", "Unsupported file format. Use CSV or Excel.")
                return

            num_properties = len(self.property_df)
            property_columns = self.property_df.columns.tolist()

            self.log_text.append(f"Loaded {num_properties} property records")
            self.log_text.append(f"Available columns: {', '.join(property_columns)}")

            # Populate property ID and target dropdowns
            self.property_id_combo.clear()
            self.property_id_combo.addItems(property_columns)
            self.target_combo.clear()
            self.target_combo.addItems(property_columns)

            # Auto-select ID column
            for col in property_columns:
                if 'id' in col.lower() or 'name' in col.lower():
                    self.property_id_combo.setCurrentText(col)

            # Auto-select target column
            for col in property_columns:
                if any(keyword in col.lower() for keyword in ['target', 'property', 'value', 'energy', 'gap']):
                    self.target_combo.setCurrentText(col)
                    break

            # Update UI
            self.property_info_label.setText(
                f"Loaded: {num_properties} records | Columns: {len(property_columns)}"
            )
            self.property_info_label.setStyleSheet("color: green; font-weight: bold;")
            self.column_selection_group.setVisible(True)
            self.match_group.setVisible(True)
            self.match_btn.setEnabled(True)
            self.data_info_label.setText("Step 2 Complete - Review Step 3 and continue to Step 4")
            self.status_label.setText("Properties loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load properties:\n{str(e)}")
            self.log_text.append(f"Error: {str(e)}")
            self.status_label.setText("Failed to load properties")

    def match_data(self):
        """Step 4: Match structures with properties"""
        if not hasattr(self, 'structure_df') or not hasattr(self, 'property_df'):
            QMessageBox.warning(self, "Warning", "Please load both structure and property files first")
            return

        structure_id_col = self.structure_id_combo.currentText()
        property_id_col = self.property_id_combo.currentText()
        target_col = self.target_combo.currentText()

        if not structure_id_col or not property_id_col or not target_col:
            QMessageBox.warning(self, "Warning", "Please select all required columns")
            return

        try:
            self.status_label.setText("Matching data...")
            self.log_text.append(f"\n{'='*50}")
            self.log_text.append("Step 4: Matching structures and properties")
            self.log_text.append(f"Structure ID column: {structure_id_col}")
            self.log_text.append(f"Property ID column: {property_id_col}")
            self.log_text.append(f"Target property: {target_col}")

            # Show progress
            self.match_progress.setVisible(True)
            self.match_progress.setRange(0, 0)  # Indeterminate progress

            # Get IDs
            structure_ids = set(self.structure_df[structure_id_col].astype(str))
            property_ids = set(self.property_df[property_id_col].astype(str))

            # Find matches
            matched_ids = structure_ids.intersection(property_ids)
            structure_only = structure_ids - property_ids
            property_only = property_ids - structure_ids

            # Create matched dataframe
            self.structure_df[structure_id_col] = self.structure_df[structure_id_col].astype(str)
            self.property_df[property_id_col] = self.property_df[property_id_col].astype(str)

            matched_structures = self.structure_df[self.structure_df[structure_id_col].isin(matched_ids)]
            matched_properties = self.property_df[self.property_df[property_id_col].isin(matched_ids)]

            # Merge on ID columns
            self.matched_df = pd.merge(
                matched_structures,
                matched_properties[[property_id_col, target_col]],
                left_on=structure_id_col,
                right_on=property_id_col,
                how='inner'
            )

            # Log results
            self.log_text.append(f"\nMatching Results:")
            self.log_text.append(f"  Total structures: {len(structure_ids)}")
            self.log_text.append(f"  Total properties: {len(property_ids)}")
            self.log_text.append(f"  Successfully matched: {len(matched_ids)}")
            self.log_text.append(f"  Structures without properties: {len(structure_only)}")
            self.log_text.append(f"  Properties without structures: {len(property_only)}")

            if len(structure_only) > 0:
                self.log_text.append(f"\nUnmatched structures (first 10): {list(structure_only)[:10]}")
            if len(property_only) > 0:
                self.log_text.append(f"Unmatched properties (first 10): {list(property_only)[:10]}")

            # Update UI
            match_rate = len(matched_ids) / max(len(structure_ids), 1) * 100
            self.match_info_label.setText(
                f"Matched: {len(matched_ids)} samples ({match_rate:.1f}% match rate)"
            )
            self.match_info_label.setStyleSheet("color: green; font-weight: bold;")
            self.match_progress.setVisible(False)

            if len(matched_ids) == 0:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No matches found! Please check:\n"
                    "- ID columns are correct\n"
                    "- ID values match between files\n"
                    "- ID formats are consistent (e.g., string vs number)"
                )
                self.status_label.setText("No matches found")
                return

            # Enable next step
            self.graph_group.setVisible(True)
            self.build_graphs_btn.setEnabled(True)
            self.data_info_label.setText(f"Step 4 Complete - {len(matched_ids)} samples ready - Continue to Step 5")
            self.status_label.setText("Data matched successfully")

            if match_rate < 50:
                QMessageBox.warning(
                    self,
                    "Low Match Rate",
                    f"Only {match_rate:.1f}% of structures were matched.\n"
                    f"Please verify the ID columns are correct."
                )

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", f"Failed to match data:\n{str(e)}")
            self.log_text.append(f"Error: {error_msg}")
            self.match_progress.setVisible(False)
            self.status_label.setText("Failed to match data")

    def build_graphs(self):
        """Step 5: Build graphs from matched data"""
        if not hasattr(self, 'matched_df'):
            QMessageBox.warning(self, "Warning", "Please match data first")
            return

        # Lazy import of DGL-dependent modules
        try:
            from .data_loader import ALIGNNDataLoader
        except Exception as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import ALIGNN modules. Please ensure DGL is properly installed.\n\nError: {str(e)}"
            )
            return

        try:
            self.status_label.setText("Building graphs...")
            self.log_text.append(f"\n{'='*50}")
            self.log_text.append("Step 5: Building graphs for training")

            # Show progress
            self.graph_progress.setVisible(True)
            self.graph_progress.setRange(0, len(self.matched_df))
            self.graph_progress.setValue(0)

            # Get column names
            target_col = self.target_combo.currentText()
            id_col = self.structure_id_combo.currentText()  # Use structure ID column

            self.log_text.append(f"Target: {target_col}")
            self.log_text.append(f"ID: {id_col}")

            # Create temporary CSV for data loader
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_path = temp_file.name
            self.matched_df.to_csv(temp_path, index=False)
            temp_file.close()

            # Create data loader
            self.data_loader = ALIGNNDataLoader(
                data_path=temp_path,
                target=target_col,
                id_tag=id_col,
                atom_features="cgcnn",
                cutoff=8.0,
                max_neighbors=12,
                compute_line_graph=True
            )

            # Use the matched dataframe directly
            self.data_loader.df = self.matched_df

            # Build graphs
            graphs = self.data_loader.build_graphs()
            self.log_text.append(f"Built {len(graphs)} graphs")

            # Update progress
            self.graph_progress.setValue(len(graphs))

            # Create dataset
            self.dataset = self.data_loader.create_dataset(
                graphs=graphs,
                classification=(self.task_type_combo.currentText() == "Classification")
            )

            # Get data loaders
            self.train_loader, self.val_loader, self.test_loader, self.dataset = \
                self.data_loader.get_data_loaders(
                    batch_size=self.batch_spin.value(),
                    train_ratio=self.train_ratio_spin.value(),
                    val_ratio=self.val_ratio_spin.value(),
                    test_ratio=self.test_ratio_spin.value(),
                    num_workers=2,
                    pin_memory=False
                )

            # Update UI
            self.graph_info_label.setText(
                f"Built {len(graphs)} graphs | "
                f"Train: {len(self.train_loader.dataset)} | "
                f"Val: {len(self.val_loader.dataset)} | "
                f"Test: {len(self.test_loader.dataset)}"
            )
            self.graph_info_label.setStyleSheet("color: green; font-weight: bold;")
            self.graph_progress.setVisible(False)

            self.train_btn.setEnabled(True)
            self.data_info_label.setText(
                f"All Steps Complete! Ready to train on {len(graphs)} samples"
            )
            self.data_info_label.setStyleSheet("font-weight: bold; color: green;")
            self.status_label.setText("Dataset ready for training")
            self.log_text.append("Dataset ready!")

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", f"Failed to build graphs:\n{str(e)}")
            self.log_text.append(f"Error: {error_msg}")
            self.graph_progress.setVisible(False)
            self.status_label.setText("Failed to build graphs")

    def load_file_preview(self):
        """Old method - kept for compatibility"""
        # Redirect to new workflow
        self.browse_structure_file()
        self.load_structures()

    def confirm_and_build(self):
        """Old method - kept for compatibility"""
        # Redirect to new workflow
        if hasattr(self, 'matched_df'):
            self.build_graphs()
        else:
            QMessageBox.information(
                self,
                "New Workflow",
                "Please use the new step-by-step workflow:\n"
                "1. Load structures\n"
                "2. Load properties\n"
                "3. Select columns\n"
                "4. Match data\n"
                "5. Build graphs"
            )

    def load_preset(self, preset_name):
        """Load configuration preset"""
        # Note: Target column is now selected by user from dropdown
        # Presets only configure training parameters
        if preset_name == "Formation Energy":
            self.task_type_combo.setCurrentText("Regression")
            self.epochs_spin.setValue(300)
            self.lr_spin.setValue(0.01)
            self.batch_spin.setValue(64)
        elif preset_name == "Band Gap":
            self.task_type_combo.setCurrentText("Regression")
            self.epochs_spin.setValue(300)
            self.lr_spin.setValue(0.01)
            self.batch_spin.setValue(64)
        elif preset_name == "Classification Task":
            self.task_type_combo.setCurrentText("Classification")
            self.num_classes_spin.setValue(2)
            self.epochs_spin.setValue(200)
            self.lr_spin.setValue(0.001)
            self.batch_spin.setValue(32)
        elif preset_name == "Property Prediction (Regression)":
            self.task_type_combo.setCurrentText("Regression")
            self.epochs_spin.setValue(300)
            self.lr_spin.setValue(0.01)
            self.batch_spin.setValue(64)
        elif preset_name == "Quick Test":
            self.task_type_combo.setCurrentText("Regression")
            self.epochs_spin.setValue(10)
            self.batch_spin.setValue(16)
            self.lr_spin.setValue(0.01)

    def start_training(self):
        """Start training"""
        # Lazy import of DGL-dependent modules
        try:
            from .model import ALIGNN
            from .trainer import ALIGNNTrainer
        except Exception as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import ALIGNN modules. Please ensure DGL is properly installed.\n\nError: {str(e)}"
            )
            return

        if self.train_loader is None:
            QMessageBox.warning(self, "Warning", "Please load dataset first")
            return

        try:
            self.log_text.append("\n" + "="*50)
            self.log_text.append("Starting training...")
            self.log_text.append("="*50)

            # Create configuration
            is_classification = (self.task_type_combo.currentText() == "Classification")

            model_config = ALIGNNConfig(
                name="alignn",
                alignn_layers=self.alignn_layers_spin.value(),
                gcn_layers=self.gcn_layers_spin.value(),
                atom_input_features=92,
                edge_input_features=80,
                triplet_input_features=40,
                embedding_features=64,
                hidden_features=256,
                output_features=self.num_classes_spin.value() if is_classification else 1,
                classification=is_classification,
                num_classes=self.num_classes_spin.value() if is_classification else 2
            )

            # Classification threshold (optional)
            classification_threshold = None
            if is_classification and self.threshold_spin.value() != self.threshold_spin.minimum():
                classification_threshold = self.threshold_spin.value()

            config = TrainingConfig(
                target=self.target_combo.currentText(),
                epochs=self.epochs_spin.value(),
                batch_size=self.batch_spin.value(),
                learning_rate=self.lr_spin.value(),
                model=model_config,
                output_dir="alignn_output",
                write_checkpoint=True,
                write_predictions=True,
                classification_threshold=classification_threshold,
                random_seed=self.random_seed_spin.value()
            )

            # Create output directory
            os.makedirs(config.output_dir, exist_ok=True)

            # Create model
            self.model = ALIGNN(model_config)

            # Create trainer with early stopping settings
            device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu_check.isChecked() else "cpu")

            # Get early stopping parameters
            early_stopping_patience = 0
            early_stopping_min_delta = 0.0
            if self.enable_early_stopping_check.isChecked():
                early_stopping_patience = self.early_stop_patience_spin.value()
                early_stopping_min_delta = self.early_stop_min_delta_spin.value()
                self.log_text.append(f"Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

            self.trainer = ALIGNNTrainer(
                config,
                self.model,
                device,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta
            )

            # Setup training thread
            self.training_thread = TrainingThread(
                self.trainer,
                self.train_loader,
                self.val_loader,
                self.test_loader
            )
            self.training_thread.progress.connect(self.on_training_progress)
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.error.connect(self.on_training_error)

            # Update UI
            self.train_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(config.epochs)
            self.status_label.setText("Training...")

            # Start training
            self.training_thread.start()

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", f"Failed to start training:\n{error_msg}")
            self.log_text.append(f"Error: {error_msg}")
            self.train_btn.setEnabled(True)

    def stop_training(self):
        """Stop training"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
            self.log_text.append("\nTraining stopped by user")
            self.status_label.setText("Training stopped")
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)

    def on_training_progress(self, info: Dict):
        """Handle training progress"""
        epoch = info['epoch']
        total_epochs = info['total_epochs']
        train_loss = info['train_loss']
        val_mae = info.get('val_mae', info.get('val_metric', 0))
        lr = info['learning_rate']

        # Update progress bar
        self.progress_bar.setValue(epoch)

        # Update log
        log_msg = (
            f"Epoch [{epoch}/{total_epochs}] "
            f"Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"LR: {lr:.6f}"
        )
        if info.get('is_best', False):
            log_msg += " [BEST]"

        self.log_text.append(log_msg)

        # Update status
        self.status_label.setText(
            f"Training: Epoch {epoch}/{total_epochs} - Loss: {train_loss:.4f}"
        )

    def on_training_finished(self, results: Dict):
        """Handle training finished"""
        self.log_text.append("\n" + "="*50)
        self.log_text.append("Training completed!")
        self.log_text.append(f"Best validation loss: {results['best_val_loss']:.4f}")
        self.log_text.append(f"Best validation MAE: {results.get('best_val_mae', 'N/A')}")
        self.log_text.append(f"Best epoch: {results.get('best_epoch', 'N/A')}")

        if results.get('stopped_early', False):
            self.log_text.append("Training stopped early due to early stopping criteria")

        if results['test_results']:
            test_mae = results['test_results']['mae']
            self.log_text.append(f"Test MAE: {test_mae:.4f}")

        self.log_text.append("="*50)

        self.results = results

        # Update UI
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Training completed")

        # Enable model saving and prediction
        self.save_model_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)
        self.model_status_label.setText("Model trained and ready")

        # Update metrics
        self.update_metrics(results)

        # Update plots
        self.update_plots(results)

        QMessageBox.information(self, "Success", "Training completed successfully!")

    def on_training_error(self, error_msg: str):
        """Handle training error"""
        self.log_text.append(f"\nERROR: {error_msg}")
        self.status_label.setText("Training failed")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        QMessageBox.critical(self, "Training Error", f"Training failed:\n{error_msg}")

    def update_metrics(self, results: Dict):
        """Update metrics table"""
        metrics = []

        metrics.append(("Best Val Loss", f"{results['best_val_loss']:.4f}"))

        # Add best MAE and epoch if available
        if 'best_val_mae' in results:
            metrics.append(("Best Val MAE", f"{results['best_val_mae']:.4f}"))
        if 'best_epoch' in results:
            metrics.append(("Best Epoch", f"{results['best_epoch']}"))
        if results.get('stopped_early', False):
            metrics.append(("Early Stopped", "Yes"))

        if results.get('test_results'):
            test_results = results['test_results']
            metrics.append(("Test MAE", f"{test_results['mae']:.4f}"))
            metrics.append(("Test RMSE", f"{test_results['rmse']:.4f}"))

        history = results['history']
        metrics.append(("Final Train Loss", f"{history['train_loss'][-1]:.4f}"))
        metrics.append(("Final Val Loss", f"{history['val_loss'][-1]:.4f}"))
        if 'val_mae' in history and history['val_mae']:
            metrics.append(("Final Val MAE", f"{history['val_mae'][-1]:.4f}"))

        self.metrics_table.setRowCount(len(metrics))
        for i, (metric, value) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))

        self.metrics_table.resizeColumnsToContents()

    def update_plots(self, results: Dict):
        """Update visualization plots"""
        self.figure.clear()

        history = results['history']
        test_results = results.get('test_results')

        if test_results:
            # Create 2x2 subplot
            ax1 = self.figure.add_subplot(2, 2, 1)
            ax2 = self.figure.add_subplot(2, 2, 2)
            ax3 = self.figure.add_subplot(2, 2, 3)
            ax4 = self.figure.add_subplot(2, 2, 4)
        else:
            # Create 1x2 subplot
            ax1 = self.figure.add_subplot(1, 2, 1)
            ax2 = self.figure.add_subplot(1, 2, 2)

        # Plot 1: Training curves
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Learning rate
        ax2.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)

        # Plot 3 & 4: Prediction scatter plots
        if test_results:
            targets = test_results['targets']
            predictions = test_results['predictions']

            ax3.scatter(targets, predictions, alpha=0.5)
            ax3.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
            ax3.set_xlabel('True Value')
            ax3.set_ylabel('Predicted Value')
            ax3.set_title('Test Set Predictions')
            ax3.grid(True, alpha=0.3)

            # Plot 4: Residuals
            residuals = [p - t for p, t in zip(predictions, targets)]
            ax4.scatter(targets, residuals, alpha=0.5)
            ax4.axhline(y=0, color='r', linestyle='--', lw=2)
            ax4.set_xlabel('True Value')
            ax4.set_ylabel('Residual')
            ax4.set_title('Prediction Residuals')
            ax4.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def browse_model_file(self):
        """Browse for model checkpoint file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Checkpoint",
            "",
            "PyTorch Model (*.pt *.pth);;All Files (*)"
        )
        if filename:
            self.model_path_edit.setText(filename)

    def browse_pred_file(self):
        """Browse for prediction data file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Prediction Data File",
            "",
            "Data Files (*.csv *.xlsx);;All Files (*)"
        )
        if filename:
            self.pred_file_edit.setText(filename)

    def save_model(self):
        """Save trained model"""
        if self.trainer is None:
            QMessageBox.warning(self, "Warning", "No trained model to save")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            "alignn_model.pt",
            "PyTorch Model (*.pt *.pth);;All Files (*)"
        )

        if filename:
            try:
                self.log_text.append(f"\nSaving model to: {filename}")
                self.trainer.save_model(filename)
                self.log_text.append("Model saved successfully!")
                self.status_label.setText("Model saved")
                QMessageBox.information(self, "Success", f"Model saved to {filename}")
            except Exception as e:
                error_msg = f"Failed to save model:\n{str(e)}"
                QMessageBox.critical(self, "Error", error_msg)
                self.log_text.append(f"Error: {error_msg}")

    def load_model(self):
        """Load model from checkpoint"""
        model_path = self.model_path_edit.text()

        if not model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file")
            return

        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", "Model file does not exist")
            return

        # Lazy import
        try:
            from .trainer import ALIGNNTrainer
        except Exception as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import ALIGNN modules:\n{str(e)}"
            )
            return

        try:
            self.status_label.setText("Loading model...")
            self.log_text.append(f"\nLoading model from: {model_path}")

            # Load trainer from checkpoint
            device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu_check.isChecked() else "cpu")
            self.trainer = ALIGNNTrainer.load_from_checkpoint(model_path, device)

            self.model = self.trainer.model
            self.log_text.append("Model loaded successfully!")
            self.model_status_label.setText(f"Model loaded: {os.path.basename(model_path)}")
            self.status_label.setText("Model loaded")

            # Enable prediction button
            self.predict_btn.setEnabled(True)
            self.save_model_btn.setEnabled(True)

            QMessageBox.information(self, "Success", "Model loaded successfully!")

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
            self.log_text.append(f"Error: {error_msg}")
            self.status_label.setText("Failed to load model")

    def make_predictions(self):
        """Make predictions on new data"""
        if self.trainer is None:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return

        pred_file = self.pred_file_edit.text()
        if not pred_file:
            QMessageBox.warning(self, "Warning", "Please select a prediction data file")
            return

        if not os.path.exists(pred_file):
            QMessageBox.warning(self, "Warning", "Prediction data file does not exist")
            return

        # Lazy import
        try:
            from .data_loader import ALIGNNDataLoader
        except Exception as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import ALIGNN modules:\n{str(e)}"
            )
            return

        try:
            self.status_label.setText("Preparing prediction data...")
            self.log_text.append(f"\n{'='*50}")
            self.log_text.append("Making predictions...")
            self.log_text.append(f"Data file: {pred_file}")

            # Create data loader for prediction
            # Use same settings as training data
            pred_data_loader = ALIGNNDataLoader(
                data_path=pred_file,
                target=self.target_combo.currentText(),
                id_tag=self.id_combo.currentText(),
                atom_features="cgcnn",
                cutoff=8.0,
                max_neighbors=12,
                compute_line_graph=True
            )

            # Load data
            df = pred_data_loader.load_data()
            self.log_text.append(f"Loaded {len(df)} structures for prediction")

            # Build graphs
            graphs = pred_data_loader.build_graphs()
            self.log_text.append(f"Built {len(graphs)} graphs")

            # Create dataset
            dataset = pred_data_loader.create_dataset(graphs=graphs, classification=False)

            # Create data loader (no split, use all data)
            from dgl.dataloading import GraphDataLoader
            pred_loader = GraphDataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=dataset.collate_line_graph,
                drop_last=False,
                num_workers=0,
                pin_memory=False
            )

            # Make predictions
            output_path = None
            if self.save_predictions_check.isChecked():
                output_path = self.pred_output_edit.text()

            self.status_label.setText("Making predictions...")
            results = self.trainer.predict(pred_loader, output_path=output_path)

            # Store results
            self.prediction_results = results

            # Update table
            self.update_prediction_table(results)

            # Enable export button
            self.export_pred_btn.setEnabled(True)

            self.log_text.append(f"Predictions completed: {len(results['predictions'])} samples")
            if output_path:
                self.log_text.append(f"Predictions saved to: {output_path}")
            self.status_label.setText("Predictions completed")

            QMessageBox.information(
                self,
                "Success",
                f"Predictions completed for {len(results['predictions'])} samples!"
            )

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", f"Failed to make predictions:\n{str(e)}")
            self.log_text.append(f"Error: {error_msg}")
            self.status_label.setText("Prediction failed")

    def update_prediction_table(self, results: Dict):
        """Update prediction results table"""
        ids = results['ids']
        predictions = results['predictions']

        self.pred_table.setRowCount(len(predictions))

        for i, (id_val, pred) in enumerate(zip(ids, predictions)):
            self.pred_table.setItem(i, 0, QTableWidgetItem(str(id_val)))
            self.pred_table.setItem(i, 1, QTableWidgetItem(f"{pred:.6f}"))

        self.pred_table.resizeColumnsToContents()

    def export_predictions(self):
        """Export predictions to CSV"""
        if not hasattr(self, 'prediction_results'):
            QMessageBox.warning(self, "Warning", "No predictions to export")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Predictions",
            "predictions.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            try:
                import pandas as pd
                results = self.prediction_results
                df = pd.DataFrame({
                    'id': results['ids'],
                    'prediction': results['predictions']
                })
                df.to_csv(filename, index=False)
                self.log_text.append(f"\nPredictions exported to: {filename}")
                QMessageBox.information(self, "Success", f"Predictions exported to {filename}")
            except Exception as e:
                error_msg = f"Failed to export predictions:\n{str(e)}"
                QMessageBox.critical(self, "Error", error_msg)
                self.log_text.append(f"Error: {error_msg}")

