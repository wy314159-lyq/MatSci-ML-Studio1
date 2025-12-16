"""
CGCNN UI Widget
Simplified interface for integration into main application
"""

import os
import sys
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QFileDialog,
    QTextEdit, QProgressBar, QMessageBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QSplitter, QDialog, QDialogButtonBox, QApplication,
    QHeaderView, QAbstractItemView, QProgressDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .model import CrystalGraphConvNet
from .data import CIFData, get_train_val_test_loader, collate_pool
from .trainer import CGCNNTrainer
from .utils import CGCNNConfig, prepare_sample_data, create_atom_init_json
from .system_info import SystemInfo
from .data_matcher import DataMatcher, prepare_cgcnn_data
from .enhanced_visualizer import EnhancedTrainingVisualizer, ProgressInfo


class PredictionResultsDialog(QDialog):
    """Dialog for displaying prediction results in a table."""

    def __init__(self, predictions, cif_ids, task_type='regression', parent=None):
        """
        Initialize prediction results dialog.

        Parameters
        ----------
        predictions : list
            List of prediction values
        cif_ids : list
            List of CIF file identifiers
        task_type : str
            'regression' or 'classification'
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.predictions = predictions
        self.cif_ids = cif_ids
        self.task_type = task_type

        self.setWindowTitle("Prediction Results")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)

        # Summary label
        summary = QLabel(f"<b>Total Predictions: {len(self.predictions)}</b>")
        layout.addWidget(summary)

        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['#', 'CIF ID', 'Prediction'])
        self.table.setRowCount(len(self.predictions))

        # Configure table
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(True)

        # Populate table
        for i, (cif_id, pred) in enumerate(zip(self.cif_ids, self.predictions)):
            # Row number
            item_num = QTableWidgetItem(str(i + 1))
            item_num.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, item_num)

            # CIF ID
            item_id = QTableWidgetItem(str(cif_id))
            self.table.setItem(i, 1, item_id)

            # Prediction value
            if self.task_type == 'regression':
                item_pred = QTableWidgetItem(f"{pred:.6f}")
            else:
                item_pred = QTableWidgetItem(f"{pred:.4f}")
            item_pred.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, 2, item_pred)

        # Adjust column widths
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

        layout.addWidget(self.table)

        # Statistics
        if self.task_type == 'regression' and len(self.predictions) > 0:
            import numpy as np
            preds_array = np.array(self.predictions)
            stats_text = (
                f"<b>Statistics:</b> "
                f"Mean = {np.mean(preds_array):.4f}, "
                f"Std = {np.std(preds_array):.4f}, "
                f"Min = {np.min(preds_array):.4f}, "
                f"Max = {np.max(preds_array):.4f}"
            )
            stats_label = QLabel(stats_text)
            layout.addWidget(stats_label)

        # Buttons
        button_layout = QHBoxLayout()

        # Export button
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(self.export_to_csv)
        button_layout.addWidget(export_btn)

        # Copy button
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(copy_btn)

        button_layout.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def export_to_csv(self):
        """Export results to CSV file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Predictions",
            "predictions.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['cif_id', 'prediction'])
                    for cif_id, pred in zip(self.cif_ids, self.predictions):
                        writer.writerow([cif_id, pred])

                QMessageBox.information(
                    self, "Success",
                    f"Predictions exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to export:\n{str(e)}"
                )

    def copy_to_clipboard(self):
        """Copy results to clipboard."""
        try:
            lines = ["cif_id\tprediction"]
            for cif_id, pred in zip(self.cif_ids, self.predictions):
                lines.append(f"{cif_id}\t{pred}")
            text = "\n".join(lines)

            clipboard = QApplication.clipboard()
            clipboard.setText(text)

            QMessageBox.information(
                self, "Success",
                f"Copied {len(self.predictions)} predictions to clipboard."
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to copy:\n{str(e)}"
            )


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
            if any(keyword in col.lower() for keyword in ['id', 'name', 'structure']):
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
    """Background thread for model training with enhanced progress tracking."""

    progress = pyqtSignal(dict)  # Changed to dict for more information
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    stopped = pyqtSignal()  # Signal when training is stopped by user
    batch_log = pyqtSignal(str)  # Signal for batch-level logging

    def __init__(self, trainer, train_loader, val_loader, test_loader, total_epochs):
        super().__init__()
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.total_epochs = total_epochs
        self._stop_requested = False  # Flag to safely stop training

    def request_stop(self):
        """Request the training to stop safely."""
        print("[TrainingThread] Stop requested by user")
        self._stop_requested = True

    def run(self):
        """Run training with enhanced progress reporting."""
        try:
            print(f"[TrainingThread] Starting training with {self.total_epochs} epochs")
            print(f"[TrainingThread] Train batches: {len(self.train_loader)}")
            print(f"[TrainingThread] Val batches: {len(self.val_loader)}")
            print(f"[TrainingThread] Test batches: {len(self.test_loader)}")

            # Setup callback for detailed progress
            def epoch_callback(info):
                try:
                    # Check if stop was requested
                    if self._stop_requested:
                        print("[TrainingThread] Stop detected in callback, raising StopIteration")
                        raise StopIteration("Training stopped by user")

                    # Get current learning rate
                    lr = self.trainer.optimizer.param_groups[0]['lr']

                    # Prepare progress info dictionary
                    progress_info = {
                        'epoch': info['epoch'],
                        'total_epochs': self.total_epochs,
                        'train_loss': info['train_loss'],
                        'val_metric': info['val_metric'],
                        'learning_rate': lr,
                        'is_best': info.get('is_best', False)
                    }

                    self.progress.emit(progress_info)
                except StopIteration:
                    raise  # Re-raise to stop training
                except Exception as e:
                    print(f"[TrainingThread] Error in epoch_callback: {e}")
                    import traceback
                    traceback.print_exc()

            self.trainer.epoch_callback = epoch_callback

            # Setup batch callback for detailed logging
            def batch_callback(info):
                try:
                    # Check if stop was requested
                    if self._stop_requested:
                        raise StopIteration("Training stopped by user")

                    # Emit log message to UI
                    self.batch_log.emit(info['message'])
                except StopIteration:
                    raise
                except Exception as e:
                    print(f"[TrainingThread] Error in batch_callback: {e}")

            self.trainer.batch_callback = batch_callback

            print("[TrainingThread] Starting trainer.train()...")

            # Train
            results = self.trainer.train(
                self.train_loader,
                self.val_loader,
                self.test_loader
            )

            print("[TrainingThread] Training completed successfully")
            self.finished.emit(results)

        except StopIteration as e:
            # Training was stopped by user - this is normal
            print(f"[TrainingThread] Training stopped by user: {e}")
            self.stopped.emit()
        except MemoryError as e:
            error_msg = f"内存不足！请尝试:\n1. 减小 batch size 到 8 或 16\n2. 减小训练集比例\n3. 关闭其他程序释放内存\n\n错误: {str(e)}"
            print(f"[TrainingThread] MemoryError: {error_msg}")
            self.error.emit(error_msg)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                error_msg = f"GPU显存不足！请尝试:\n1. 减小 batch size 到 8 或 16\n2. 使用CPU训练\n\n错误: {str(e)}"
            else:
                error_msg = f"运行时错误: {str(e)}"
            print(f"[TrainingThread] RuntimeError: {error_msg}")
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)
        except Exception as e:
            error_msg = f"训练失败: {type(e).__name__}: {str(e)}"
            print(f"[TrainingThread] Exception: {error_msg}")
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)


class CGCNNModule(QWidget):
    """
    CGCNN Module Widget.

    Simplified interface for CGCNN training and prediction.
    Supports drag and drop of CIF files/folders for prediction.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self.config = CGCNNConfig.formation_energy()
        self.dataset = None
        self.model = None
        self.trainer = None
        self.training_thread = None
        self.results = None
        self.visualizer = None  # Will be initialized with figure
        self.test_predictions = None  # Store for final visualization
        self.test_targets = None
        self._training_active = False  # Flag to track if training is active

        # Enable drag and drop
        self.setAcceptDrops(True)

        self.init_ui()

    def dragEnterEvent(self, event):
        """Handle drag enter event for CIF files/folders."""
        if event.mimeData().hasUrls():
            # Check if any URL is a CIF file or directory
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path) or path.lower().endswith('.cif'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Handle drop event for CIF files/folders."""
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        # Check if model is available
        if self.trainer is None or self.model is None:
            QMessageBox.warning(
                self, "No Model",
                "Please train or load a model before making predictions.\n\n"
                "You can drag and drop CIF files/folders after a model is ready."
            )
            event.ignore()
            return

        # Collect all CIF files from dropped items
        cif_files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                # Collect all CIF files from directory
                for filename in os.listdir(path):
                    if filename.lower().endswith('.cif'):
                        cif_files.append(os.path.join(path, filename))
            elif path.lower().endswith('.cif'):
                cif_files.append(path)

        if not cif_files:
            QMessageBox.warning(
                self, "No CIF Files",
                "No CIF files found in the dropped items.\n\n"
                "Please drop .cif files or folders containing .cif files."
            )
            event.ignore()
            return

        event.acceptProposedAction()

        # Confirm prediction
        reply = QMessageBox.question(
            self, "Confirm Prediction",
            f"Found {len(cif_files)} CIF file(s).\n\n"
            f"Do you want to make predictions on these files?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self._predict_dropped_files(cif_files)

    def _predict_dropped_files(self, cif_files):
        """Make predictions on dropped CIF files."""
        try:
            import tempfile
            import shutil

            self.status_label.setText("Preparing dropped files for prediction...")
            self.log("=" * 50)
            self.log(f"Making predictions on {len(cif_files)} dropped CIF file(s)...")

            # Create temporary directory with all CIF files
            temp_dir = tempfile.mkdtemp()
            for cif_path in cif_files:
                cif_name = os.path.basename(cif_path)
                # Handle duplicate filenames by adding suffix
                dest_path = os.path.join(temp_dir, cif_name)
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(cif_name)
                    dest_path = os.path.join(temp_dir, f"{name}_{counter}{ext}")
                    counter += 1
                shutil.copy(cif_path, dest_path)

            # Get atom_init.json
            atom_init_file = self._get_builtin_atom_init()
            if atom_init_file is None:
                if self.dataset is not None and hasattr(self.dataset, 'root_dir'):
                    test_path = os.path.join(self.dataset.root_dir, 'atom_init.json')
                    if os.path.exists(test_path):
                        atom_init_file = test_path

            if atom_init_file is None:
                QMessageBox.critical(
                    self, "Error",
                    "atom_init.json not found. Please ensure the CGCNN module is properly installed."
                )
                shutil.rmtree(temp_dir)
                return

            self.log(f"Using atom_init.json: {atom_init_file}")

            # Import the prediction dataset class
            from .data import CIFDataPredict

            # Load dataset
            pred_dataset = CIFDataPredict(
                cif_dir=temp_dir,
                atom_init_file=atom_init_file,
                max_num_nbr=self.config.max_num_nbr,
                radius=self.config.radius,
                dmin=self.config.dmin,
                step=self.config.step
            )

            if len(pred_dataset) == 0:
                QMessageBox.warning(
                    self, "No Data",
                    "Could not load any CIF files for prediction."
                )
                shutil.rmtree(temp_dir)
                return

            # Create data loader
            from torch.utils.data import DataLoader
            pred_loader = DataLoader(
                pred_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_pool,
                pin_memory=torch.cuda.is_available()
            )

            self.log(f"Loaded {len(pred_dataset)} CIF structures for prediction")

            # Create progress dialog
            progress = QProgressDialog(
                "Making predictions on dropped files...",
                "Cancel",
                0, len(pred_loader),
                self
            )
            progress.setWindowTitle("Prediction Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            # Make predictions
            self.model.eval()
            predictions = []
            cif_ids = []
            cancelled = False

            with torch.no_grad():
                for batch_idx, (input_data, target, batch_cif_ids) in enumerate(pred_loader):
                    if progress.wasCanceled():
                        cancelled = True
                        break

                    progress.setValue(batch_idx)
                    progress.setLabelText(
                        f"Processing batch {batch_idx + 1}/{len(pred_loader)}...\n"
                        f"Predicted: {len(predictions)} structures"
                    )
                    QApplication.processEvents()

                    # Move to device
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
                    atom_fea = atom_fea.to(self.trainer.device, non_blocking=True)
                    nbr_fea = nbr_fea.to(self.trainer.device, non_blocking=True)
                    nbr_fea_idx = nbr_fea_idx.to(self.trainer.device, non_blocking=True)
                    crystal_atom_idx = [
                        idx.to(self.trainer.device, non_blocking=True) for idx in crystal_atom_idx
                    ]

                    # Forward pass
                    output = self.model(
                        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx
                    )

                    # Denormalize predictions
                    if self.trainer.task == 'regression':
                        pred = self.trainer.normalizer.denorm(output.data.cpu())
                        predictions += pred.view(-1).tolist()
                    else:
                        pred = torch.exp(output.data.cpu())
                        predictions += pred[:, 1].tolist()

                    cif_ids += batch_cif_ids

            progress.setValue(len(pred_loader))
            progress.close()

            # Clean up temp directory
            shutil.rmtree(temp_dir)

            if cancelled:
                self.log("Prediction cancelled by user")
                self.status_label.setText("Prediction cancelled")
                return

            # Log results
            self.log(f"\nPrediction completed: {len(predictions)} structures")
            for cif_id, pred in zip(cif_ids, predictions):
                self.log(f"  {cif_id}: {pred:.6f}")

            # Show results in table dialog
            task_type = self.trainer.task
            dialog = PredictionResultsDialog(
                predictions=predictions,
                cif_ids=cif_ids,
                task_type=task_type,
                parent=self
            )
            dialog.exec_()

            self.status_label.setText("Prediction completed")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{str(e)}")
            self.log(f"ERROR: {str(e)}")
            self.status_label.setText("Prediction failed")

    def init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("CGCNN - Crystal Graph Convolutional Neural Network")
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
        """Create configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Data section
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout(data_group)

        # Hidden data_dir_edit for internal use (stores prepared dataset path)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setVisible(False)

        # Data loading: Load from CIF directory + Property file
        method_label = QLabel("<b>Load Data (Two Files):</b>")
        data_layout.addWidget(method_label)

        # Add instruction label
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
        self.prop_file_edit = QLineEdit()
        self.prop_file_edit.setPlaceholderText("CSV/Excel with any column names (will select columns in next step)")
        prop_file_layout.addWidget(self.prop_file_edit)
        browse_prop_btn = QPushButton("Browse")
        browse_prop_btn.clicked.connect(self.browse_prop_file)
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
        data_layout.addWidget(self.data_info_label)

        layout.addWidget(data_group)

        # Configuration section
        config_group = QGroupBox("Model Configuration")
        config_layout = QVBoxLayout(config_group)

        # Preset selector
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Formation Energy",
            "Band Gap",
            "Classification",
            "GPU Optimized",
            "Quick Test",
            "Custom"
        ])
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_layout.addWidget(self.preset_combo)
        config_layout.addLayout(preset_layout)

        # Task type selector
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Task Type:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Regression", "Classification"])
        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        task_layout.addWidget(self.task_combo)
        config_layout.addLayout(task_layout)

        # Number of classes (only for classification)
        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(2, 100)
        self.num_classes_spin.setValue(2)
        self.num_classes_row = QHBoxLayout()
        self.num_classes_label = QLabel("Number of Classes:")
        self.num_classes_row.addWidget(self.num_classes_label)
        self.num_classes_row.addWidget(self.num_classes_spin)
        config_layout.addLayout(self.num_classes_row)
        # Initially hidden for regression
        self.num_classes_label.setVisible(False)
        self.num_classes_spin.setVisible(False)

        # Key parameters
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.config.epochs)
        self.add_param_row(config_layout, "Epochs:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)  # 支持更精细的学习率，如1e-6
        self.lr_spin.setRange(0.000001, 10.0)  # 扩展到1e-6至10
        self.lr_spin.setSingleStep(0.0001)  # 更精细的步长
        self.lr_spin.setValue(self.config.learning_rate)
        self.add_param_row(config_layout, "Learning Rate:", self.lr_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 2048)  # 扩展到2048，支持更大batch
        self.batch_spin.setValue(self.config.batch_size)
        self.add_param_row(config_layout, "Batch Size:", self.batch_spin)

        # Data split ratios
        split_label = QLabel("<b>Data Split Ratios</b>")
        config_layout.addWidget(split_label)

        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setDecimals(3)  # 更精细的小数位
        self.train_ratio_spin.setRange(0.05, 0.95)  # 扩展范围，支持更极端的分割
        self.train_ratio_spin.setSingleStep(0.01)  # 更精细的步长
        self.train_ratio_spin.setValue(0.8)
        self.add_param_row(config_layout, "Train Ratio:", self.train_ratio_spin)

        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setDecimals(3)  # 更精细的小数位
        self.val_ratio_spin.setRange(0.0, 0.8)  # 扩展最大到0.8
        self.val_ratio_spin.setSingleStep(0.01)  # 更精细的步长
        self.val_ratio_spin.setValue(0.1)
        self.add_param_row(config_layout, "Validation Ratio:", self.val_ratio_spin)

        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setDecimals(3)  # 更精细的小数位
        self.test_ratio_spin.setRange(0.0, 0.8)  # 扩展最大到0.8
        self.test_ratio_spin.setSingleStep(0.01)  # 更精细的步长
        self.test_ratio_spin.setValue(0.1)
        self.add_param_row(config_layout, "Test Ratio:", self.test_ratio_spin)

        # Random seed for reproducibility
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 99999)
        self.random_seed_spin.setValue(42)
        self.random_seed_spin.setSpecialValueText("Random")  # 0 means random
        self.random_seed_spin.setToolTip(
            "Random seed for data splitting.\n"
            "Set to 0 for random shuffle each time.\n"
            "Set to any other value (e.g., 42) for reproducible splits."
        )
        self.add_param_row(config_layout, "Random Seed:", self.random_seed_spin)

        # Early Stopping section
        es_label = QLabel("<b>Early Stopping</b>")
        config_layout.addWidget(es_label)

        self.es_enabled_check = QComboBox()
        self.es_enabled_check.addItems(["Disabled", "Enabled"])
        self.es_enabled_check.setCurrentIndex(0)
        self.es_enabled_check.currentIndexChanged.connect(self.on_early_stopping_changed)
        self.add_param_row(config_layout, "Early Stopping:", self.es_enabled_check)

        self.es_patience_spin = QSpinBox()
        self.es_patience_spin.setRange(1, 100)
        self.es_patience_spin.setValue(10)
        self.es_patience_spin.setToolTip(
            "Number of epochs with no improvement after which training will be stopped.\n"
            "Larger values allow more time for the model to improve."
        )
        self.es_patience_spin.setEnabled(False)
        self.add_param_row(config_layout, "Patience:", self.es_patience_spin)

        self.es_min_delta_spin = QDoubleSpinBox()
        self.es_min_delta_spin.setDecimals(6)
        self.es_min_delta_spin.setRange(0.0, 1.0)
        self.es_min_delta_spin.setSingleStep(0.0001)
        self.es_min_delta_spin.setValue(0.0001)
        self.es_min_delta_spin.setToolTip(
            "Minimum change in validation metric to qualify as an improvement.\n"
            "For regression: decrease in MAE must exceed this value.\n"
            "For classification: increase in AUC must exceed this value."
        )
        self.es_min_delta_spin.setEnabled(False)
        self.add_param_row(config_layout, "Min Delta:", self.es_min_delta_spin)

        layout.addWidget(config_group)

        # Training section
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout(train_group)

        self.train_btn = QPushButton("Start Training")
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

        self.save_model_btn = QPushButton("Save Best Model As...")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        model_layout.addWidget(self.save_model_btn)

        self.load_model_btn = QPushButton("Load Model...")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)

        self.predict_btn = QPushButton("Predict on New Data...")
        self.predict_btn.clicked.connect(self.predict_new_data)
        self.predict_btn.setEnabled(False)
        model_layout.addWidget(self.predict_btn)

        layout.addWidget(model_group)

        # Results Export section
        export_group = QGroupBox("Results Export")
        export_layout = QVBoxLayout(export_group)

        self.export_data_btn = QPushButton("Export Training Data for Plotting...")
        self.export_data_btn.clicked.connect(self.export_training_data)
        self.export_data_btn.setEnabled(False)
        self.export_data_btn.setToolTip(
            "Export training history and prediction results to CSV files\n"
            "for creating custom plots in Excel, Origin, etc."
        )
        export_layout.addWidget(self.export_data_btn)

        layout.addWidget(export_group)

        layout.addStretch()

        return panel

    def create_results_panel(self):
        """Create results display panel."""
        tabs = QTabWidget()

        # System Info tab
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
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Initialize enhanced visualizer
        self.visualizer = EnhancedTrainingVisualizer(self.figure, self.canvas)

        results_layout.addWidget(self.toolbar)
        results_layout.addWidget(self.canvas)

        tabs.addTab(results_widget, "Results")

        return tabs

    def create_system_info_tab(self):
        """Create system information tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Refresh button
        refresh_btn = QPushButton("Refresh System Info")
        refresh_btn.clicked.connect(self.refresh_system_info)
        layout.addWidget(refresh_btn)

        # System info display
        self.system_info_text = QTextEdit()
        self.system_info_text.setReadOnly(True)
        self.system_info_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.system_info_text)

        # Load initial system info
        self.refresh_system_info()

        return widget

    def refresh_system_info(self):
        """Refresh and display system information."""
        try:
            self.system_info_text.clear()
            self.system_info_text.append("Checking system information...\n")

            # Create system checker
            checker = SystemInfo()
            results = checker.check_all()

            # Display summary
            summary = checker.get_summary()
            self.system_info_text.clear()
            self.system_info_text.append(summary)

            # Add recommendations
            recommendations = checker.get_recommendations()
            if recommendations:
                self.system_info_text.append("\n[Recommendations]")
                for i, rec in enumerate(recommendations, 1):
                    self.system_info_text.append(f"  {i}. {rec}")

            # Set text color based on status
            if results['status'] == 'error':
                self.system_info_text.append("\n\nWARNING: Please fix errors before training!")
            elif results['warnings']:
                self.system_info_text.append("\n\nNote: System has warnings but can still work.")
            else:
                self.system_info_text.append("\n\nSystem is ready for CGCNN training!")

        except Exception as e:
            self.system_info_text.clear()
            self.system_info_text.append(f"Error checking system info: {str(e)}")


    def add_param_row(self, layout, label, widget):
        """Add parameter row to layout."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(widget)
        layout.addLayout(row)

    def browse_cif_dir(self):
        """Browse for CIF directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select CIF Files Directory"
        )
        if directory:
            self.cif_dir_edit.setText(directory)

    def browse_prop_file(self):
        """Browse for property file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Property File",
            "",
            "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.prop_file_edit.setText(file_path)

    def _get_builtin_atom_init(self):
        """Get path to built-in atom_init.json in CGCNN module directory."""
        module_dir = os.path.dirname(os.path.abspath(__file__))
        atom_init_path = os.path.join(module_dir, 'atom_init.json')
        if os.path.exists(atom_init_path):
            return atom_init_path
        # Fallback to project root
        project_root = os.path.dirname(os.path.dirname(module_dir))
        atom_init_path = os.path.join(project_root, 'atom_init.json')
        if os.path.exists(atom_init_path):
            return atom_init_path
        return None

    def auto_prepare_and_load(self):
        """Auto-prepare dataset from separate CIF and property files."""
        cif_dir = self.cif_dir_edit.text()
        prop_file = self.prop_file_edit.text()

        # Validate inputs
        if not cif_dir or not os.path.exists(cif_dir):
            QMessageBox.warning(self, "Error", "Please select a valid CIF directory")
            return

        if not prop_file or not os.path.exists(prop_file):
            QMessageBox.warning(self, "Error", "Please select a valid property file")
            return

        try:
            # Read property file to show column selection dialog
            import pandas as pd

            self.status_label.setText("Reading property file...")

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

            self.log("=" * 50)
            self.log("Auto-preparing dataset from separate files...")
            self.log(f"CIF directory: {cif_dir}")
            self.log(f"Property file: {prop_file}")
            self.log(f"ID column: '{id_column}'")
            self.log(f"Property column: '{property_column}'")

            # Validate columns exist
            if id_column not in df.columns or property_column not in df.columns:
                QMessageBox.critical(
                    self, "Error",
                    f"Selected columns not found in file!"
                )
                return

            # Auto-create output directory in CIF directory parent folder
            # e.g., if CIF dir is "C:/data/cif_files", output will be "C:/data/cgcnn_prepared"
            cif_parent = os.path.dirname(cif_dir)
            output_dir = os.path.join(cif_parent, "cgcnn_prepared")

            # If directory exists, create a new one with timestamp
            if os.path.exists(output_dir):
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(cif_parent, f"cgcnn_prepared_{timestamp}")

            os.makedirs(output_dir, exist_ok=True)
            self.log(f"Output directory (auto-created): {output_dir}")

            # Prepare id_prop.csv with selected columns
            id_prop_data = df[[id_column, property_column]].copy()
            id_prop_data.columns = ['structure_id', 'property']  # Rename for DataMatcher

            # Save temporary id_prop.csv
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_id_prop = os.path.join(temp_dir, 'id_prop.csv')
            id_prop_data.to_csv(temp_id_prop, index=False)

            # Create progress dialog
            from PyQt5.QtWidgets import QProgressDialog
            from PyQt5.QtCore import Qt

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
                import shutil
                shutil.rmtree(temp_dir)
                self.status_label.setText("Cancelled")
                return

            self.status_label.setText("Preparing dataset...")

            # Update progress: Matching files
            progress.setLabelText(f"Matching {len(df)} structures with CIF files...")
            progress.setValue(30)
            QApplication.processEvents()

            if progress.wasCanceled():
                import shutil
                shutil.rmtree(temp_dir)
                self.status_label.setText("Cancelled")
                return

            # Use DataMatcher to prepare dataset
            matcher = DataMatcher()

            # Get built-in atom_init.json from CGCNN module directory
            builtin_atom_init = self._get_builtin_atom_init()
            if builtin_atom_init:
                self.log(f"Using built-in atom_init.json (100 elements, 92-dim features)")
                import shutil
                shutil.copy(builtin_atom_init, os.path.join(output_dir, 'atom_init.json'))
            else:
                self.log("WARNING: No atom_init.json found in CGCNN module, DataMatcher will create one")

            try:
                prepared_dir, stats = matcher.prepare_dataset(
                    cif_dir=cif_dir,
                    property_file=temp_id_prop,
                    output_dir=output_dir
                )

                # Update progress: Processing complete
                progress.setLabelText("Finalizing dataset preparation...")
                progress.setValue(90)
                QApplication.processEvents()

            except Exception as e:
                progress.close()
                import shutil
                shutil.rmtree(temp_dir)
                QMessageBox.critical(self, "Error", f"Failed to prepare dataset:\n{str(e)}")
                self.status_label.setText("Error")
                return

            # Clean up temp file
            import shutil
            shutil.rmtree(temp_dir)

            # Complete progress
            progress.setValue(100)
            progress.close()

            # Show statistics
            self.log("\nDataset Preparation Results:")
            self.log(f"  Total structures in property file: {stats['total']}")
            self.log(f"  Successfully matched: {stats['matched']}")
            self.log(f"  Missing CIF files: {stats['missing_cif']}")
            self.log(f"  CIF files without properties: {stats['missing_property']}")

            # Show matching strategy statistics
            if 'strategy_stats' in stats and stats['strategy_stats']:
                self.log("\nMatching Strategies Used:")
                for strategy, count in stats['strategy_stats'].items():
                    strategy_name = {
                        'exact': 'Exact match',
                        'remove_cif_ext': 'After removing .cif extension',
                        'case_insensitive': 'Case-insensitive match',
                        'case_insensitive_no_ext': 'Case-insensitive without .cif',
                        'normalized': 'Normalized (no separators)',
                        'fuzzy': 'Fuzzy match (similarity-based)'
                    }.get(strategy, strategy)
                    self.log(f"  {strategy_name}: {count} matches")

            # Show some match examples
            if 'match_details' in stats and stats['match_details']:
                self.log("\nExample Matches (first 10):")
                for detail in stats['match_details'][:10]:
                    if detail['original_id'] != detail['matched_id']:
                        self.log(f"  '{detail['original_id']}' -> '{detail['matched_id']}' ({detail['strategy']})")

            if stats['missing_cif'] > 0:
                self.log(f"\nFirst missing CIF files: {stats['missing_cif_list']}")
            if stats['missing_property'] > 0:
                self.log(f"\nFirst CIF files without properties: {stats['missing_property_list']}")

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
                self.data_dir_edit.setText(prepared_dir)
                self.load_dataset()
            else:
                self.status_label.setText("Dataset prepared (not loaded)")

        except Exception as e:
            error_msg = f"Failed to prepare dataset:\n{str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.log(f"ERROR: {str(e)}")
            self.status_label.setText("Dataset preparation failed")

    def load_preset(self, preset_name):
        """Load configuration preset."""
        if preset_name == "Formation Energy":
            self.config = CGCNNConfig.formation_energy()
        elif preset_name == "Band Gap":
            self.config = CGCNNConfig.band_gap()
        elif preset_name == "Classification":
            self.config = CGCNNConfig.classification_task()
        elif preset_name == "GPU Optimized":
            self.config = CGCNNConfig.gpu_optimized()
        elif preset_name == "Quick Test":
            self.config = CGCNNConfig.quick_test()
        else:
            return

        # Update UI
        self.epochs_spin.setValue(self.config.epochs)
        self.lr_spin.setValue(self.config.learning_rate)
        self.batch_spin.setValue(self.config.batch_size)

        # Update task type - block signals to prevent on_task_changed from overwriting config
        self.task_combo.blockSignals(True)
        if self.config.classification:
            self.task_combo.setCurrentText("Classification")
        else:
            self.task_combo.setCurrentText("Regression")
        self.task_combo.blockSignals(False)

        # Manually update UI elements for classification
        self.num_classes_label.setVisible(self.config.classification)
        self.num_classes_spin.setVisible(self.config.classification)

        self.log(f"Loaded preset: {preset_name} (batch_size={self.config.batch_size}, classification={self.config.classification})")

    def on_task_changed(self, task_type):
        """Handle task type change."""
        is_classification = (task_type == "Classification")

        # Show/hide number of classes
        self.num_classes_label.setVisible(is_classification)
        self.num_classes_spin.setVisible(is_classification)

        # Update config
        self.config.classification = is_classification

        self.log(f"Task type changed to: {task_type}")

    def on_early_stopping_changed(self, index):
        """Handle early stopping toggle."""
        enabled = (index == 1)  # 1 = Enabled
        self.es_patience_spin.setEnabled(enabled)
        self.es_min_delta_spin.setEnabled(enabled)

        if enabled:
            self.log("Early stopping enabled")
        else:
            self.log("Early stopping disabled")

    def load_dataset(self):
        """Load dataset from directory."""
        data_dir = self.data_dir_edit.text()

        if not data_dir or not os.path.exists(data_dir):
            QMessageBox.warning(self, "Error", "Please select a valid data directory")
            return

        try:
            self.status_label.setText("Loading dataset...")
            self.log("Loading dataset from: " + data_dir)

            # Load dataset
            self.dataset = CIFData(
                data_dir,
                max_num_nbr=self.config.max_num_nbr,
                radius=self.config.radius,
                dmin=self.config.dmin,
                step=self.config.step
            )

            # Update info
            info = f"Dataset loaded: {len(self.dataset)} structures"
            self.data_info_label.setText(info)
            self.log(info)

            # Show dataset statistics
            self.show_dataset_statistics()

            # Enable training
            self.train_btn.setEnabled(True)
            self.status_label.setText("Dataset ready")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{str(e)}")
            self.log(f"ERROR: {str(e)}")

    def show_dataset_statistics(self):
        """Show dataset statistics in log panel."""
        if self.dataset is None:
            return

        try:
            import numpy as np
            from collections import Counter

            self.log("\n" + "=" * 50)
            self.log("Dataset Statistics:")
            self.log("=" * 50)

            # Basic info
            self.log(f"  Total samples: {len(self.dataset)}")

            # Collect property values and elements
            targets = []
            element_counts = Counter()

            for i in range(min(len(self.dataset), 1000)):  # Sample first 1000 for efficiency
                try:
                    _, target, cif_id = self.dataset[i]
                    targets.append(target.item() if hasattr(target, 'item') else target)
                except Exception:
                    continue

            if targets:
                targets = np.array(targets)
                self.log(f"\nProperty Distribution:")
                self.log(f"  Min:    {np.min(targets):.4f}")
                self.log(f"  Max:    {np.max(targets):.4f}")
                self.log(f"  Mean:   {np.mean(targets):.4f}")
                self.log(f"  Median: {np.median(targets):.4f}")
                self.log(f"  Std:    {np.std(targets):.4f}")

                # Check for classification-like data (integer values)
                unique_values = np.unique(targets)
                if len(unique_values) <= 10 and all(v == int(v) for v in unique_values):
                    self.log(f"\nClass Distribution (potential classification task):")
                    for val in sorted(unique_values):
                        count = np.sum(targets == val)
                        pct = 100.0 * count / len(targets)
                        self.log(f"  Class {int(val)}: {count} samples ({pct:.1f}%)")

            # Try to get element info from id_prop_file
            id_prop_file = os.path.join(self.dataset.root_dir, 'id_prop.csv')
            if os.path.exists(id_prop_file):
                import csv
                with open(id_prop_file, 'r') as f:
                    reader = csv.reader(f)
                    sample_count = sum(1 for _ in reader) - 1  # Subtract header if exists
                    if sample_count > 0 and sample_count != len(self.dataset):
                        self.log(f"\n  Note: id_prop.csv has {sample_count} entries")

            self.log("=" * 50 + "\n")

        except Exception as e:
            self.log(f"  (Could not compute full statistics: {str(e)})")

    def start_training(self):
        """Start model training."""
        if self.dataset is None:
            QMessageBox.warning(self, "Error", "Please load a dataset first")
            return

        try:
            self.log("=" * 50)
            self.log("Starting training...")
            print("[start_training] Step 1: Reading configuration...")

            # Update config from UI
            self.config.epochs = self.epochs_spin.value()
            self.config.learning_rate = self.lr_spin.value()
            self.config.batch_size = self.batch_spin.value()

            print(f"[start_training] Config: epochs={self.config.epochs}, lr={self.config.learning_rate}, batch={self.config.batch_size}")

            # Get data split ratios from UI
            train_ratio = self.train_ratio_spin.value()
            val_ratio = self.val_ratio_spin.value()
            test_ratio = self.test_ratio_spin.value()

            # Validate ratios sum to 1.0
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 0.01:
                QMessageBox.warning(
                    self, "Invalid Ratios",
                    f"Train + Val + Test ratios must sum to 1.0\n"
                    f"Current sum: {total_ratio:.2f}\n"
                    f"Please adjust the ratios."
                )
                return

            print("[start_training] Step 2: Creating data loaders...")
            # Create data loaders
            self.log("Creating data loaders...")
            self.log(f"Data split: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
            self.log(f"Total structures: {len(self.dataset)}")

            # For large datasets (>5000), disable pin_memory to save RAM
            use_pin_memory = torch.cuda.is_available() and len(self.dataset) < 5000

            # Determine optimal num_workers based on system
            # Windows: num_workers > 0 can cause issues, keep at 0
            # Linux/Mac: Use 2-4 workers for better GPU utilization
            import platform
            if platform.system() == 'Windows':
                num_workers = 0
                self.log("[Data Loading] Windows detected - using num_workers=0 (Windows multiprocessing limitation)")
            else:
                # Linux/Mac can use multiple workers
                num_workers = min(4, torch.get_num_threads())
                self.log(f"[Data Loading] Using num_workers={num_workers} for faster data loading")

            # Warning for potentially problematic batch sizes (but don't auto-adjust)
            if len(self.dataset) >= 10000 and self.config.batch_size > 64:
                self.log(f"[Large Dataset Mode] Detected {len(self.dataset)} structures")
                self.log(f"  - Disabled pin_memory to save RAM")
                self.log(f"  - WARNING: Batch size {self.config.batch_size} is very large for {len(self.dataset)} structures!")
                self.log(f"  - If you encounter errors, consider reducing batch size to 16-32")

                # Show warning but let user decide
                reply = QMessageBox.warning(
                    self, "Large Batch Size Warning",
                    f"Your dataset has {len(self.dataset)} structures with batch size {self.config.batch_size}.\n\n"
                    f"This may cause:\n"
                    f"  • Memory errors (RAM/GPU)\n"
                    f"  • Data batching errors\n\n"
                    f"Recommended batch sizes:\n"
                    f"  • Good hardware (32GB+ RAM): 32-64\n"
                    f"  • Average hardware (16GB RAM): 16-32\n"
                    f"  • Limited hardware (<16GB RAM): 8-16\n\n"
                    f"Continue with current batch size {self.config.batch_size}?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    self.log("Training cancelled by user. Please adjust batch size and try again.")
                    return
            elif len(self.dataset) >= 5000:
                self.log(f"[Large Dataset Mode] Detected {len(self.dataset)} structures - optimizing memory usage")
                self.log(f"  - Disabled pin_memory to save RAM")
                if self.config.batch_size > 64:
                    self.log(f"  - NOTE: Batch size {self.config.batch_size} is quite large")
                    self.log(f"  - If you encounter errors, try reducing to 32 or lower")

            train_loader, val_loader, test_loader = get_train_val_test_loader(
                dataset=self.dataset,
                collate_fn=collate_pool,
                batch_size=self.config.batch_size,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                return_test=True,
                num_workers=num_workers,
                pin_memory=use_pin_memory,
                random_seed=self.random_seed_spin.value() if self.random_seed_spin.value() > 0 else None
            )

            print(f"[start_training] Data loaders created: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")

            print("[start_training] Step 3: Getting input dimensions...")
            # Get input dimensions
            structures, _, _ = self.dataset[0]
            orig_atom_fea_len = structures[0].shape[-1]
            nbr_fea_len = structures[1].shape[-1]
            print(f"[start_training] Input dims: orig_atom_fea={orig_atom_fea_len}, nbr_fea={nbr_fea_len}")

            print("[start_training] Step 4: Creating model...")
            # Create model
            n_classes = self.num_classes_spin.value() if self.config.classification else 2
            self.log(f"Creating model (atom_fea={orig_atom_fea_len}, nbr_fea={nbr_fea_len})...")
            self.log(f"Task: {'Classification' if self.config.classification else 'Regression'}")
            if self.config.classification:
                self.log(f"Number of classes: {n_classes}")

            self.model = CrystalGraphConvNet(
                orig_atom_fea_len=orig_atom_fea_len,
                nbr_fea_len=nbr_fea_len,
                atom_fea_len=self.config.atom_fea_len,
                n_conv=self.config.n_conv,
                h_fea_len=self.config.h_fea_len,
                n_h=self.config.n_h,
                classification=self.config.classification,
                n_classes=n_classes
            )

            # Log model output layer shape for verification
            output_shape = self.model.fc_out.weight.shape
            self.log(f"Model output layer shape: {output_shape}")
            print(f"[start_training] Model created successfully, output shape: {output_shape}")

            # Get early stopping parameters
            es_enabled = self.es_enabled_check.currentIndex() == 1
            es_patience = self.es_patience_spin.value() if es_enabled else 0
            es_min_delta = self.es_min_delta_spin.value() if es_enabled else 0.0

            print("[start_training] Step 5: Creating trainer...")
            # Create trainer
            self.trainer = CGCNNTrainer(
                model=self.model,
                task='classification' if self.config.classification else 'regression',
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum,
                optimizer_type=self.config.optimizer,
                lr_milestones=self.config.lr_milestones,
                epochs=self.config.epochs,
                early_stopping_patience=es_patience,
                early_stopping_min_delta=es_min_delta
            )
            print(f"[start_training] Trainer created, device: {self.trainer.device}")

            print("[start_training] Step 6: Starting visualizer...")
            # Reset and start visualizer
            self.visualizer.reset()
            self.visualizer.start_training()

            print("[start_training] Step 7: Starting training thread...")
            # Start training thread
            self.training_thread = TrainingThread(
                self.trainer, train_loader, val_loader, test_loader, self.config.epochs
            )
            self.training_thread.progress.connect(self.on_training_progress)
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.error.connect(self.on_training_error)
            self.training_thread.stopped.connect(self.on_training_stopped)
            self.training_thread.batch_log.connect(self.on_batch_log)
            self.training_thread.start()

            # Update UI and set training active flag
            self._training_active = True
            self.train_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(self.config.epochs)
            self.progress_bar.setValue(0)

        except MemoryError as e:
            error_msg = f"内存不足！\n\n请尝试:\n1. 减小 Batch Size 到 8 或 16\n2. 减小训练集比例\n3. 关闭其他程序释放内存\n\n详细错误: {str(e)}"
            print(f"[start_training] MemoryError: {error_msg}")
            QMessageBox.critical(self, "Memory Error", error_msg)
            self.log(f"ERROR: {error_msg}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                error_msg = f"GPU显存不足！\n\n请尝试:\n1. 减小 Batch Size 到 8 或 16\n2. 使用CPU训练（在System Info中检查）\n\n详细错误: {str(e)}"
            else:
                error_msg = f"运行时错误: {str(e)}"
            print(f"[start_training] RuntimeError: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Runtime Error", error_msg)
            self.log(f"ERROR: {error_msg}")
        except Exception as e:
            error_msg = f"训练启动失败: {type(e).__name__}: {str(e)}"
            print(f"[start_training] Exception: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to start training:\n{str(e)}")
            self.log(f"ERROR: {str(e)}")

    def stop_training(self):
        """Stop training safely."""
        # Clear training flag and disconnect signals immediately
        self._training_active = False
        try:
            if self.training_thread:
                self.training_thread.progress.disconnect(self.on_training_progress)
                self.training_thread.batch_log.disconnect(self.on_batch_log)
        except (TypeError, RuntimeError):
            pass

        if self.training_thread and self.training_thread.isRunning():
            self.log("Requesting training to stop...")
            self.training_thread.request_stop()

            # Wait for thread to finish naturally (max 5 seconds)
            if self.training_thread.wait(5000):  # Wait up to 5 seconds
                self.log("Training stopped successfully")
            else:
                # If thread doesn't stop in 5 seconds, force terminate as last resort
                self.log("Training taking too long to stop, force terminating...")
                self.training_thread.terminate()
                self.training_thread.wait()  # Wait for termination to complete
                self.log("Training force stopped")

            self.reset_ui()

    def on_training_progress(self, progress_info):
        """Handle training progress update with enhanced visualization."""
        # Skip if training is no longer active (avoid delayed signal processing)
        if not self._training_active:
            return

        epoch = progress_info['epoch']
        total_epochs = progress_info['total_epochs']
        train_loss = progress_info['train_loss']
        val_metric = progress_info['val_metric']
        learning_rate = progress_info['learning_rate']
        is_best = progress_info['is_best']

        # Update progress bar (always)
        self.progress_bar.setValue(epoch)

        # Determine update frequency based on total epochs to prevent UI freeze
        # For large epoch counts, update less frequently
        if total_epochs <= 50:
            update_interval = 1  # Update every epoch
        elif total_epochs <= 200:
            update_interval = 5  # Update every 5 epochs
        elif total_epochs <= 500:
            update_interval = 10  # Update every 10 epochs
        else:
            update_interval = 20  # Update every 20 epochs

        # Always update on first epoch, last epoch, best epoch, or at interval
        should_update_plot = (
            epoch == 1 or
            epoch == total_epochs or
            is_best or
            epoch % update_interval == 0
        )

        # Update visualizer (with controlled frequency)
        if should_update_plot:
            self.visualizer.update_progress(epoch, train_loss, val_metric, learning_rate, is_best)
            # Refresh canvas
            try:
                self.canvas.draw_idle()
            except Exception as e:
                print(f"[UI] Canvas refresh error: {e}")

        # Calculate ETA
        eta = self.visualizer.get_eta(epoch, total_epochs)
        elapsed = self.visualizer.get_elapsed_time()

        # Create progress info object
        progress = ProgressInfo(
            epoch=epoch,
            total_epochs=total_epochs,
            train_loss=train_loss,
            val_metric=val_metric,
            learning_rate=learning_rate,
            is_best=is_best,
            elapsed=elapsed,
            eta=eta
        )

        # Log progress (with controlled frequency for large epoch counts)
        if should_update_plot or is_best:
            self.log(progress.to_string())

        # Update status label (always, but lightweight)
        self.status_label.setText(f"Training: Epoch {epoch}/{total_epochs} | "
                                 f"Val: {val_metric:.4f} | ETA: {eta}")

        # Process events less frequently for large epoch counts
        if should_update_plot and self._training_active:
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()

    def on_training_finished(self, results):
        """Handle training completion with enhanced visualization."""
        # IMPORTANT: Disconnect progress signal FIRST to prevent any delayed signals
        self._training_active = False
        try:
            self.training_thread.progress.disconnect(self.on_training_progress)
            self.training_thread.batch_log.disconnect(self.on_batch_log)
        except (TypeError, RuntimeError):
            pass  # Already disconnected or no connection

        self.results = results
        self.log("=" * 50)
        self.log("Training completed!")
        self.log(f"Best metric: {results['best_metric']:.4f}")

        # Get training and test predictions for visualization
        train_predictions = None
        train_targets = None
        test_predictions = None
        test_targets = None

        # Extract training set predictions if available
        if 'train_results' in results and results['train_results']:
            import numpy as np
            train_predictions = np.array(results['train_results']['predictions'])
            train_targets = np.array(results['train_results']['targets'])

        # Extract test set predictions if available
        if 'test_results' in results and 'predictions' in results['test_results']:
            import numpy as np
            test_predictions = np.array(results['test_results']['predictions'])
            test_targets = np.array(results['test_results']['targets'])

        # Update metrics table
        self.update_metrics_table(results)

        # Determine task type for visualization
        task_type = 'classification' if self.config.classification else 'regression'

        # Plot comprehensive results with enhanced visualizer (both train and test)
        self.visualizer.plot_final_results(results, train_predictions, train_targets,
                                          test_predictions, test_targets, task_type)

        self.reset_ui()
        self.status_label.setText("Training completed")

        # Enable model saving and data export after training
        self.save_model_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)
        self.export_data_btn.setEnabled(True)

        # Prompt user to save the model
        self.prompt_save_model()

    def on_training_error(self, error_msg):
        """Handle training error."""
        # Disconnect signals first
        self._training_active = False
        try:
            self.training_thread.progress.disconnect(self.on_training_progress)
            self.training_thread.batch_log.disconnect(self.on_batch_log)
        except (TypeError, RuntimeError):
            pass
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Training Error", error_msg)
        self.reset_ui()

    def on_training_stopped(self):
        """Handle training stopped by user."""
        # Disconnect signals first
        self._training_active = False
        try:
            self.training_thread.progress.disconnect(self.on_training_progress)
            self.training_thread.batch_log.disconnect(self.on_batch_log)
        except (TypeError, RuntimeError):
            pass
        self.log("Training stopped by user")
        self.status_label.setText("Training stopped")
        # Don't need to call reset_ui() here as stop_training() already does it

    def on_batch_log(self, log_message):
        """Handle batch-level log messages."""
        self.log(log_message)

    def reset_ui(self):
        """Reset UI after training."""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    def update_metrics_table(self, results):
        """Update metrics table with results."""
        self.metrics_table.setRowCount(0)

        metrics = {
            "Best Metric": f"{results['best_metric']:.4f}",
            "Total Epochs": str(len(results['history']))
        }

        if results.get('test_results'):
            test = results['test_results']
            if 'mae' in test:
                metrics["Test MAE"] = f"{test['mae']:.4f}"
            if 'auc' in test:
                metrics["Test AUC"] = f"{test['auc']:.4f}"
            if 'accuracy' in test:
                metrics["Test Accuracy"] = f"{test['accuracy']:.4f}"
            if 'f1' in test:
                metrics["Test Macro F1"] = f"{test['f1']:.4f}"

        for i, (key, value) in enumerate(metrics.items()):
            self.metrics_table.insertRow(i)
            self.metrics_table.setItem(i, 0, QTableWidgetItem(key))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))

    def log(self, message):
        """Add message to log."""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def prompt_save_model(self):
        """Prompt user to save the trained model after training completes."""
        if self.trainer is None or self.trainer.best_model_state is None:
            return

        reply = QMessageBox.question(
            self,
            "Save Model",
            "Training completed!\n\n"
            "Would you like to save the best model now?\n\n"
            "Note: The model is currently in memory. If you close the application "
            "without saving, you will lose the trained model.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes  # Default to Yes
        )

        if reply == QMessageBox.Yes:
            self.save_model()

    def save_model(self):
        """Save the best trained model to a user-specified location."""
        # Check if trainer has a best model state in memory
        if self.trainer is None or self.trainer.best_model_state is None:
            QMessageBox.warning(
                self, "Error",
                "No trained model found.\n\n"
                "Please train a model first, then save it."
            )
            return

        # Ask user where to save
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Best Model",
            "cgcnn_model_best.pth.tar",
            "Model Files (*.pth.tar);;All Files (*)"
        )

        if file_path:
            try:
                if self.trainer.save_model(file_path):
                    self.log(f"Model saved to: {file_path}")
                    QMessageBox.information(
                        self, "Success",
                        f"Model saved successfully to:\n{file_path}"
                    )
                else:
                    QMessageBox.critical(
                        self, "Error",
                        "Failed to save model. Check console for details."
                    )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model:\n{str(e)}")
                self.log(f"ERROR: Failed to save model: {str(e)}")

    def load_model(self):
        """Load a previously trained model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model",
            "",
            "Model Files (*.pth.tar);;All Files (*)"
        )

        if file_path:
            try:
                self.log("Loading model checkpoint...")

                # Load checkpoint to inspect configuration
                checkpoint = torch.load(file_path, map_location='cpu')

                # Check if model_config exists in checkpoint
                if 'model_config' in checkpoint:
                    self.log("Model configuration found in checkpoint")
                    model_config = checkpoint['model_config']

                    # Extract configuration - MUST use checkpoint values, not current config
                    orig_atom_fea_len = model_config.get('orig_atom_fea_len')
                    nbr_fea_len = model_config.get('nbr_fea_len')
                    atom_fea_len = model_config.get('atom_fea_len')
                    n_conv = model_config.get('n_conv')
                    h_fea_len = model_config.get('h_fea_len')
                    n_h = model_config.get('n_h')
                    classification = model_config.get('classification', False)
                    n_classes = model_config.get('n_classes', 2)

                    self.log(f"Checkpoint model config:")
                    self.log(f"  orig_atom_fea_len={orig_atom_fea_len}")
                    self.log(f"  nbr_fea_len={nbr_fea_len}")
                    self.log(f"  atom_fea_len={atom_fea_len}")
                    self.log(f"  n_conv={n_conv}, h_fea_len={h_fea_len}, n_h={n_h}")
                    self.log(f"  classification={classification}, n_classes={n_classes}")

                    # Validate all required parameters are present
                    missing_params = []
                    if orig_atom_fea_len is None:
                        missing_params.append('orig_atom_fea_len')
                    if nbr_fea_len is None:
                        missing_params.append('nbr_fea_len')
                    if atom_fea_len is None:
                        missing_params.append('atom_fea_len')

                    if missing_params:
                        self.log(f"WARNING: Missing parameters in checkpoint: {missing_params}")
                        self.log("Attempting to infer from state_dict...")

                        # Infer from state_dict
                        state_dict = checkpoint['state_dict']

                        if atom_fea_len is None and 'embedding.weight' in state_dict:
                            atom_fea_len = state_dict['embedding.weight'].shape[0]
                            orig_atom_fea_len = state_dict['embedding.weight'].shape[1]
                            self.log(f"Inferred from embedding: atom_fea_len={atom_fea_len}, orig_atom_fea_len={orig_atom_fea_len}")

                        if nbr_fea_len is None and 'convs.0.fc_full.weight' in state_dict:
                            # fc_full input = 2*atom_fea_len + nbr_fea_len (see model.py ConvLayer)
                            conv_input_dim = state_dict['convs.0.fc_full.weight'].shape[1]
                            nbr_fea_len = conv_input_dim - 2 * atom_fea_len
                            self.log(f"Inferred nbr_fea_len={nbr_fea_len} from convs (conv_input={conv_input_dim}, atom_fea={atom_fea_len})")

                        if n_conv is None:
                            n_conv = 0
                            while f'convs.{n_conv}.fc_full.weight' in state_dict:
                                n_conv += 1
                            self.log(f"Inferred n_conv={n_conv}")

                        if h_fea_len is None and 'conv_to_fc.weight' in state_dict:
                            h_fea_len = state_dict['conv_to_fc.weight'].shape[0]
                            self.log(f"Inferred h_fea_len={h_fea_len}")

                        if n_h is None:
                            n_h = 1
                            while f'fcs.{n_h-1}.weight' in state_dict:
                                n_h += 1
                            self.log(f"Inferred n_h={n_h}")

                    # Final validation
                    if orig_atom_fea_len is None or nbr_fea_len is None or atom_fea_len is None:
                        raise ValueError(
                            f"Cannot determine model architecture from checkpoint!\n"
                            f"orig_atom_fea_len={orig_atom_fea_len}, "
                            f"nbr_fea_len={nbr_fea_len}, "
                            f"atom_fea_len={atom_fea_len}"
                        )

                    # Use default values for optional parameters
                    if n_conv is None:
                        n_conv = 3
                    if h_fea_len is None:
                        h_fea_len = 128
                    if n_h is None:
                        n_h = 1

                else:
                    self.log("Old checkpoint format detected - inferring architecture from state_dict")
                    # Old checkpoint format - infer from state_dict
                    state_dict = checkpoint['state_dict']

                    # Infer dimensions from state_dict
                    if 'embedding.weight' in state_dict:
                        atom_fea_len = state_dict['embedding.weight'].shape[0]  # Output dim
                        orig_atom_fea_len = state_dict['embedding.weight'].shape[1]  # Input dim
                    else:
                        raise ValueError("Cannot infer embedding dimensions from checkpoint")

                    if 'convs.0.fc_full.weight' in state_dict:
                        # fc_full input = 2*atom_fea_len + nbr_fea_len (see model.py ConvLayer)
                        conv_input_dim = state_dict['convs.0.fc_full.weight'].shape[1]
                        nbr_fea_len = conv_input_dim - 2 * atom_fea_len
                    else:
                        raise ValueError("Cannot infer neighbor feature dimensions from checkpoint")

                    # Infer n_conv
                    n_conv = 0
                    while f'convs.{n_conv}.fc_full.weight' in state_dict:
                        n_conv += 1

                    # Infer h_fea_len
                    if 'conv_to_fc.weight' in state_dict:
                        h_fea_len = state_dict['conv_to_fc.weight'].shape[0]
                    else:
                        h_fea_len = self.config.h_fea_len

                    # Infer n_h
                    n_h = 1
                    while f'fcs.{n_h-1}.weight' in state_dict:
                        n_h += 1

                    # Infer classification and n_classes
                    if 'fc_out.weight' in state_dict:
                        out_dim = state_dict['fc_out.weight'].shape[0]
                        if out_dim == 1:
                            classification = False
                            n_classes = 2
                        else:
                            classification = True
                            n_classes = out_dim
                    else:
                        classification = self.config.classification
                        n_classes = self.num_classes_spin.value() if classification else 2

                    self.log(f"Inferred architecture from state_dict:")
                    self.log(f"  orig_atom_fea_len={orig_atom_fea_len}, nbr_fea_len={nbr_fea_len}")
                    self.log(f"  atom_fea_len={atom_fea_len}, n_conv={n_conv}, h_fea_len={h_fea_len}, n_h={n_h}")
                    self.log(f"  classification={classification}, n_classes={n_classes}")

                # Create model with architecture
                self.log("="*50)
                self.log("Creating model with parameters:")
                self.log(f"  orig_atom_fea_len = {orig_atom_fea_len}")
                self.log(f"  nbr_fea_len = {nbr_fea_len}")
                self.log(f"  atom_fea_len = {atom_fea_len}")
                self.log(f"  n_conv = {n_conv}")
                self.log(f"  h_fea_len = {h_fea_len}")
                self.log(f"  n_h = {n_h}")
                self.log(f"  classification = {classification}")
                self.log(f"  n_classes = {n_classes}")
                self.log(f"Expected fc_full input dim = atom_fea_len + nbr_fea_len = {atom_fea_len} + {nbr_fea_len} = {atom_fea_len + nbr_fea_len}")
                self.log("="*50)

                self.model = CrystalGraphConvNet(
                    orig_atom_fea_len=orig_atom_fea_len,
                    nbr_fea_len=nbr_fea_len,
                    atom_fea_len=atom_fea_len,
                    n_conv=n_conv,
                    h_fea_len=h_fea_len,
                    n_h=n_h,
                    classification=classification,
                    n_classes=n_classes
                )

                if self.model is None:
                    raise RuntimeError("Failed to create model - model is None")

                self.log("Model created successfully")

                # Update UI to match loaded model
                self.task_combo.blockSignals(True)
                if classification:
                    self.task_combo.setCurrentText("Classification")
                    self.config.classification = True
                else:
                    self.task_combo.setCurrentText("Regression")
                    self.config.classification = False
                self.task_combo.blockSignals(False)
                self.num_classes_label.setVisible(classification)
                self.num_classes_spin.setVisible(classification)

                # Create trainer
                self.log("Creating trainer...")
                self.trainer = CGCNNTrainer(
                    model=self.model,
                    task='classification' if classification else 'regression',
                    learning_rate=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    momentum=self.config.momentum,
                    optimizer_type=self.config.optimizer,
                    lr_milestones=self.config.lr_milestones,
                    epochs=self.config.epochs
                )

                if self.trainer is None or self.trainer.model is None:
                    raise RuntimeError("Failed to create trainer or trainer.model is None")

                self.log("Trainer created successfully")

                # Load checkpoint using trainer's load_model method
                self.log(f"Loading checkpoint from {file_path}...")
                result = self.trainer.load_model(file_path)

                if result is None:
                    raise RuntimeError(f"Failed to load checkpoint from {file_path}")

                self.log(f"Model loaded from: {file_path}")
                self.predict_btn.setEnabled(True)
                self.save_model_btn.setEnabled(True)
                QMessageBox.information(self, "Success", "Model loaded successfully!")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
                self.log(f"ERROR: Failed to load model: {str(e)}")

    def predict_new_data(self):
        """Make predictions on new data using the trained model.

        Supports:
        - Single CIF file
        - Directory containing CIF files
        """
        if self.trainer is None or self.model is None:
            QMessageBox.warning(
                self, "Error",
                "No model available. Please train or load a model first."
            )
            return

        # Ask user to select CIF file or directory
        reply = QMessageBox.question(
            self, "Select Input Type",
            "What would you like to predict?\n\n"
            "Yes = Select a single CIF file\n"
            "No = Select a folder containing CIF files",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )

        if reply == QMessageBox.Cancel:
            return

        cif_files = []
        temp_dir = None

        if reply == QMessageBox.Yes:
            # Single CIF file
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select CIF File",
                "",
                "CIF Files (*.cif);;All Files (*)"
            )
            if not file_path:
                return

            # Create temporary directory with single file
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp()
            cif_name = os.path.basename(file_path)
            shutil.copy(file_path, os.path.join(temp_dir, cif_name))
            cif_dir = temp_dir
            self.log(f"Predicting single file: {file_path}")
        else:
            # Directory of CIF files
            cif_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory Containing CIF Files",
                ""
            )
            if not cif_dir:
                return
            self.log(f"Predicting directory: {cif_dir}")

        try:
            self.status_label.setText("Loading prediction data...")
            self.log("=" * 50)
            self.log("Making predictions on CIF structures...")

            # Get atom_init.json - use built-in first
            atom_init_file = self._get_builtin_atom_init()

            if atom_init_file is None:
                # Try from dataset
                if self.dataset is not None and hasattr(self.dataset, 'root_dir'):
                    test_path = os.path.join(self.dataset.root_dir, 'atom_init.json')
                    if os.path.exists(test_path):
                        atom_init_file = test_path

            if atom_init_file is None:
                QMessageBox.critical(
                    self, "Error",
                    "atom_init.json not found. Please ensure the CGCNN module is properly installed."
                )
                return

            self.log(f"Using atom_init.json: {atom_init_file}")

            # Import the prediction dataset class
            from .data import CIFDataPredict

            # Load dataset
            pred_dataset = CIFDataPredict(
                cif_dir=cif_dir,
                atom_init_file=atom_init_file,
                max_num_nbr=self.config.max_num_nbr,
                radius=self.config.radius,
                dmin=self.config.dmin,
                step=self.config.step
            )

            if len(pred_dataset) == 0:
                QMessageBox.warning(
                    self, "No Data",
                    "No CIF files found in the selected location."
                )
                return

            # Create data loader
            from torch.utils.data import DataLoader
            pred_loader = DataLoader(
                pred_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_pool,
                pin_memory=torch.cuda.is_available()
            )

            self.log(f"Loaded {len(pred_dataset)} CIF structures for prediction")

            # Create progress dialog
            progress = QProgressDialog(
                "Making predictions...",
                "Cancel",
                0, len(pred_loader),
                self
            )
            progress.setWindowTitle("Prediction Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            # Make predictions
            self.model.eval()
            predictions = []
            cif_ids = []
            cancelled = False

            with torch.no_grad():
                for batch_idx, (input_data, target, batch_cif_ids) in enumerate(pred_loader):
                    # Check for cancellation
                    if progress.wasCanceled():
                        cancelled = True
                        break

                    # Update progress
                    progress.setValue(batch_idx)
                    progress.setLabelText(
                        f"Processing batch {batch_idx + 1}/{len(pred_loader)}...\n"
                        f"Predicted: {len(predictions)} structures"
                    )
                    QApplication.processEvents()

                    # Move to device with async transfer
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
                    atom_fea = atom_fea.to(self.trainer.device, non_blocking=True)
                    nbr_fea = nbr_fea.to(self.trainer.device, non_blocking=True)
                    nbr_fea_idx = nbr_fea_idx.to(self.trainer.device, non_blocking=True)
                    crystal_atom_idx = [
                        idx.to(self.trainer.device, non_blocking=True) for idx in crystal_atom_idx
                    ]

                    # Forward pass
                    output = self.model(
                        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx
                    )

                    # Denormalize predictions
                    if self.trainer.task == 'regression':
                        pred = self.trainer.normalizer.denorm(output.data.cpu())
                        predictions += pred.view(-1).tolist()
                    else:
                        pred = torch.exp(output.data.cpu())
                        predictions += pred[:, 1].tolist()

                    cif_ids += batch_cif_ids

            progress.setValue(len(pred_loader))
            progress.close()

            # Handle cancellation
            if cancelled:
                self.log("Prediction cancelled by user")
                self.status_label.setText("Prediction cancelled")
                if temp_dir:
                    import shutil
                    shutil.rmtree(temp_dir)
                return

            # Clean up temp directory if used
            if temp_dir:
                import shutil
                shutil.rmtree(temp_dir)

            # Log results
            self.log(f"\nPrediction completed: {len(predictions)} structures")
            for cif_id, pred in zip(cif_ids, predictions):
                self.log(f"  {cif_id}: {pred:.6f}")

            # Show results in table dialog
            task_type = self.trainer.task
            dialog = PredictionResultsDialog(
                predictions=predictions,
                cif_ids=cif_ids,
                task_type=task_type,
                parent=self
            )
            dialog.exec_()

            self.status_label.setText("Prediction completed")

        except Exception as e:
            # Clean up temp directory on error
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{str(e)}")
            self.log(f"ERROR: {str(e)}")
            self.status_label.setText("Prediction failed")

    def export_training_data(self):
        """Export training data for custom plotting."""
        if self.results is None:
            QMessageBox.warning(
                self, "No Data",
                "No training results available. Please train a model first."
            )
            return

        # Ask user where to save
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save Export Files",
            ""
        )

        if not save_dir:
            return

        try:
            import pandas as pd
            import os

            self.log("="*50)
            self.log("Exporting training data for plotting...")

            # 1. Export training history
            history_file = os.path.join(save_dir, "training_history.csv")
            history_data = []
            for h in self.results['history']:
                row = {
                    'epoch': h['epoch'],
                    'train_loss': h['train'].get('loss', None),
                    'val_metric': h['val'],
                    'is_best': h['is_best']
                }
                # Add additional metrics if available
                if 'mae' in h['train']:
                    row['train_mae'] = h['train']['mae']
                if 'accuracy' in h['train']:
                    row['train_accuracy'] = h['train']['accuracy']
                if 'auc' in h['train']:
                    row['train_auc'] = h['train']['auc']
                history_data.append(row)

            history_df = pd.DataFrame(history_data)
            history_df.to_csv(history_file, index=False)
            self.log(f"✓ Saved training history: {history_file}")
            self.log(f"  Columns: {', '.join(history_df.columns)}")

            # 2. Export test predictions (if available)
            if 'test_results' in self.results and self.results['test_results']:
                test_res = self.results['test_results']
                if 'predictions' in test_res and 'targets' in test_res:
                    test_file = os.path.join(save_dir, "test_predictions.csv")
                    test_df = pd.DataFrame({
                        'actual': test_res['targets'],
                        'predicted': test_res['predictions']
                    })
                    test_df.to_csv(test_file, index=False)
                    self.log(f"✓ Saved test predictions: {test_file}")
                    self.log(f"  Samples: {len(test_df)}")

            # 3. Export train predictions (if available)
            if 'train_results' in self.results and self.results['train_results']:
                train_res = self.results['train_results']
                if 'predictions' in train_res and 'targets' in train_res:
                    train_file = os.path.join(save_dir, "train_predictions.csv")
                    train_df = pd.DataFrame({
                        'actual': train_res['targets'],
                        'predicted': train_res['predictions']
                    })
                    train_df.to_csv(train_file, index=False)
                    self.log(f"✓ Saved training predictions: {train_file}")
                    self.log(f"  Samples: {len(train_df)}")

            # 4. Export summary statistics
            summary_file = os.path.join(save_dir, "training_summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("CGCNN Training Summary\n")
                f.write("="*60 + "\n\n")

                f.write(f"Best Validation Metric: {self.results['best_metric']:.6f}\n")
                f.write(f"Total Epochs: {len(self.results['history'])}\n\n")

                if 'test_results' in self.results and self.results['test_results']:
                    f.write("Test Set Results:\n")
                    test_res = self.results['test_results']
                    if 'mae' in test_res:
                        f.write(f"  MAE: {test_res['mae']:.6f}\n")
                    if 'rmse' in test_res:
                        f.write(f"  RMSE: {test_res['rmse']:.6f}\n")
                    if 'auc' in test_res:
                        f.write(f"  AUC: {test_res['auc']:.6f}\n")

                    # Calculate R² if regression
                    if 'predictions' in test_res and 'targets' in test_res:
                        import numpy as np
                        targets = np.array(test_res['targets'])
                        preds = np.array(test_res['predictions'])
                        ss_res = np.sum((targets - preds) ** 2)
                        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        f.write(f"  R²: {r2:.6f}\n")

                f.write("\n")
                f.write("Exported Files:\n")
                f.write(f"  - training_history.csv: Training metrics per epoch\n")
                f.write(f"  - test_predictions.csv: Test set actual vs predicted\n")
                f.write(f"  - train_predictions.csv: Training set actual vs predicted\n")
                f.write(f"  - training_summary.txt: This summary file\n")

            self.log(f"✓ Saved summary: {summary_file}")

            self.log("="*50)
            self.log(f"Export completed! Files saved to: {save_dir}")

            # Show success message
            QMessageBox.information(
                self, "Export Successful",
                f"Training data exported successfully!\n\n"
                f"Location: {save_dir}\n\n"
                f"Files created:\n"
                f"  • training_history.csv\n"
                f"  • test_predictions.csv\n"
                f"  • train_predictions.csv\n"
                f"  • training_summary.txt\n\n"
                f"You can now use these files for custom plotting."
            )

        except Exception as e:
            error_msg = f"Failed to export data:\n{str(e)}"
            QMessageBox.critical(self, "Export Error", error_msg)
            self.log(f"ERROR: {error_msg}")

