"""
Module 1: Data Ingestion & Preprocessing
Handles data import, exploration, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QPushButton, QFileDialog, QMessageBox,
                            QTableWidget, QTableWidgetItem, QComboBox, QSpinBox,
                            QLineEdit, QTextEdit, QSplitter, QTabWidget, QCheckBox,
                            QListWidget, QListWidgetItem, QProgressBar, QSlider,
                            QDoubleSpinBox, QScrollArea, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont

from utils.data_utils import (load_csv_file, load_excel_file, get_excel_sheet_names,
                             load_clipboard_data, get_data_quality_report,
                             detect_outliers_iqr, detect_outliers_zscore,
                             validate_feature_target_selection, suggest_task_type,
                             safe_column_conversion)
from utils.plot_utils import (plot_missing_values_heatmap, plot_missing_values_bar,
                             plot_correlation_heatmap, plot_histogram, plot_boxplot,
                             plot_scatter, create_figure, create_subplots_figure)


class DataModule(QWidget):
    """Data ingestion and preprocessing module"""
    
    # Signals
    data_ready = pyqtSignal(pd.DataFrame, pd.Series)  # X, y
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.X = None
        self.y = None
        self.selected_features = []
        self.selected_target = ""
        self.final_df = None  # Store final processed dataframe state
        self.encoding_applied = False  # Track if encoding has been applied
        self.encoded_with_target = None  # Track which target was used for encoding
        self.label_encoder = None  # For target variable encoding
        self.original_classes = None  # Store original class names
        
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
        
        # Right panel for visualization
        right_panel = QTabWidget()
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 1000])
        
        # === LEFT PANEL ===
        
        # Data Import Section
        import_group = QGroupBox("Data Import")
        import_layout = QVBoxLayout(import_group)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select data file...")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_btn)
        import_layout.addLayout(file_layout)
        
        # Import options
        options_layout = QGridLayout()
        
        # Delimiter
        options_layout.addWidget(QLabel("Delimiter:"), 0, 0)
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems([",", ";", "\t", "|", " "])
        options_layout.addWidget(self.delimiter_combo, 0, 1)
        
        # Encoding
        options_layout.addWidget(QLabel("Encoding:"), 1, 0)
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(["utf-8", "latin-1", "ascii", "cp1252"])
        options_layout.addWidget(self.encoding_combo, 1, 1)
        
        # Header row
        options_layout.addWidget(QLabel("Header Row:"), 2, 0)
        self.header_spin = QSpinBox()
        self.header_spin.setMinimum(0)
        self.header_spin.setMaximum(10)
        options_layout.addWidget(self.header_spin, 2, 1)
        
        import_layout.addLayout(options_layout)
        
        # Import buttons
        import_btn_layout = QHBoxLayout()
        self.import_btn = QPushButton("Import Data")
        self.import_btn.clicked.connect(self.import_data)
        self.clipboard_btn = QPushButton("From Clipboard")
        self.clipboard_btn.clicked.connect(self.import_from_clipboard)
        import_btn_layout.addWidget(self.import_btn)
        import_btn_layout.addWidget(self.clipboard_btn)
        import_layout.addLayout(import_btn_layout)
        
        left_layout.addWidget(import_group)
        
        # Data Cleaning Section
        self.cleaning_group = QGroupBox("Data Cleaning")
        cleaning_layout = QVBoxLayout(self.cleaning_group)
        
        # Missing values handling
        missing_layout = QVBoxLayout()
        missing_layout.addWidget(QLabel("Missing Values:"))
        
        missing_method_layout = QHBoxLayout()
        missing_method_layout.addWidget(QLabel("Method:"))
        self.missing_method_combo = QComboBox()
        self.missing_method_combo.addItems([
            "Drop rows", "Drop columns", "Fill mean", "Fill median", 
            "Fill mode", "Fill constant"
        ])
        missing_method_layout.addWidget(self.missing_method_combo)
        missing_layout.addLayout(missing_method_layout)
        
        self.apply_missing_btn = QPushButton("Apply Missing Value Handling")
        self.apply_missing_btn.clicked.connect(self.handle_missing_values)
        missing_layout.addWidget(self.apply_missing_btn)
        
        cleaning_layout.addLayout(missing_layout)
        
        # Remove duplicates
        self.remove_duplicates_btn = QPushButton("Remove Duplicate Rows")
        self.remove_duplicates_btn.clicked.connect(self.remove_duplicates)
        cleaning_layout.addWidget(self.remove_duplicates_btn)
        
        self.cleaning_group.setEnabled(False)
        left_layout.addWidget(self.cleaning_group)
        
        # Feature & Target Selection
        self.selection_group = QGroupBox("Feature & Target Selection")
        selection_layout = QVBoxLayout(self.selection_group)
        
        # Feature selection
        selection_layout.addWidget(QLabel("Features (X):"))
        self.features_list = QListWidget()
        self.features_list.setSelectionMode(QListWidget.MultiSelection)
        self.features_list.setMaximumHeight(150)
        self.features_list.itemSelectionChanged.connect(self.check_selection_validity)
        selection_layout.addWidget(self.features_list)
        
        feature_btn_layout = QHBoxLayout()
        self.select_all_features_btn = QPushButton("Select All")
        self.select_all_features_btn.clicked.connect(self.select_all_features)
        self.clear_features_btn = QPushButton("Clear All")
        self.clear_features_btn.clicked.connect(self.clear_all_features)
        feature_btn_layout.addWidget(self.select_all_features_btn)
        feature_btn_layout.addWidget(self.clear_features_btn)
        selection_layout.addLayout(feature_btn_layout)
        
        # Target selection
        selection_layout.addWidget(QLabel("Target (y):"))
        self.target_combo = QComboBox()
        selection_layout.addWidget(self.target_combo)
        self.target_combo.currentTextChanged.connect(self.on_target_selection_changed)
        
        # Task type suggestion
        self.task_type_label = QLabel("Suggested Task: Not determined")
        selection_layout.addWidget(self.task_type_label)
        
        # Categorical encoding section
        encoding_group = QGroupBox("Categorical Feature Encoding")
        encoding_layout = QVBoxLayout(encoding_group)
        
        # Add informational text
        info_label = QLabel("ℹ️ Note: Target column will be automatically excluded from encoding")
        info_label.setStyleSheet("color: #666; font-style: italic; margin: 5px;")
        encoding_layout.addWidget(info_label)
        
        # Encoding method selection
        encoding_method_layout = QHBoxLayout()
        encoding_method_layout.addWidget(QLabel("Encoding Method:"))
        
        self.encoding_method_combo = QComboBox()
        self.encoding_method_combo.addItems([
            "One-Hot Encoding",
            "Label Encoding", 
            "Binary Encoding"
        ])
        self.encoding_method_combo.currentTextChanged.connect(self.on_encoding_method_changed)
        encoding_method_layout.addWidget(self.encoding_method_combo)
        encoding_layout.addLayout(encoding_method_layout)
        
        # Encoding method description
        self.encoding_description = QLabel()
        self.encoding_description.setWordWrap(True)
        self.encoding_description.setStyleSheet("""
            QLabel {
                background-color: #e3f2fd;
                border: 1px solid #bbdefb;
                border-radius: 4px;
                padding: 8px;
                color: #1565c0;
                font-size: 9pt;
            }
        """)
        self.update_encoding_description()
        encoding_layout.addWidget(self.encoding_description)
        
        # Categorical columns list
        encoding_layout.addWidget(QLabel("Categorical Columns (Features Only):"))
        self.categorical_list = QListWidget()
        self.categorical_list.setMaximumHeight(120)
        self.categorical_list.setSelectionMode(QListWidget.MultiSelection)  # Allow selection
        encoding_layout.addWidget(self.categorical_list)
        
        self.apply_encoding_btn = QPushButton("Apply Categorical Encoding")
        self.apply_encoding_btn.clicked.connect(self.apply_categorical_encoding)
        self.apply_encoding_btn.setEnabled(False)
        self.apply_encoding_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        encoding_layout.addWidget(self.apply_encoding_btn)
        
        selection_layout.addWidget(encoding_group)
        
        # Confirm selection button
        self.confirm_selection_btn = QPushButton("Confirm Feature & Target Selection")
        self.confirm_selection_btn.clicked.connect(self.confirm_selection)
        self.confirm_selection_btn.setEnabled(False)
        self.confirm_selection_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        selection_layout.addWidget(self.confirm_selection_btn)
        
        # Proceed button
        self.proceed_btn = QPushButton("Proceed to Feature Selection")
        self.proceed_btn.clicked.connect(self.proceed_to_next_module)
        self.proceed_btn.setEnabled(False)
        selection_layout.addWidget(self.proceed_btn)
        
        self.selection_group.setEnabled(False)
        left_layout.addWidget(self.selection_group)
        
        left_layout.addStretch()
        
        # === RIGHT PANEL ===
        
        # Data Overview Tab
        self.overview_widget = QWidget()
        overview_layout = QVBoxLayout(self.overview_widget)
        
        # Data info
        self.data_info_text = QTextEdit()
        self.data_info_text.setMaximumHeight(200)
        self.data_info_text.setReadOnly(True)
        overview_layout.addWidget(QLabel("Data Information:"))
        overview_layout.addWidget(self.data_info_text)
        
        # Data preview table
        self.data_table = QTableWidget()
        overview_layout.addWidget(QLabel("Data Preview:"))
        overview_layout.addWidget(self.data_table)
        
        right_panel.addTab(self.overview_widget, "Data Overview")
        
        # Store right panel reference for adding viz tabs
        self.right_panel = right_panel
        self.viz_tabs = {}
        
    def browse_file(self):
        """Browse for data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
            
    def import_data(self):
        """Import data from file"""
        file_path = self.file_path_edit.text()
        
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select a data file.")
            return
            
        try:
            self.status_updated.emit("Importing data...")
            self.progress_updated.emit(20)
            
            # Simplified data loading with better error handling
            if file_path.endswith('.csv'):
                delimiter = self.delimiter_combo.currentText()
                encoding = self.encoding_combo.currentText()
                header = self.header_spin.value() if self.header_spin.value() > 0 else 0
                
                # Use pandas with error handling
                try:
                    self.df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, header=header, low_memory=False)
                except UnicodeDecodeError:
                    # Try with different encoding
                    self.df = pd.read_csv(file_path, delimiter=delimiter, encoding='latin1', header=header, low_memory=False)
                    self.status_updated.emit("Importing data using Latin1 encoding")
                    
            elif file_path.endswith(('.xlsx', '.xls')):
                # Use pandas with error handling
                self.df = pd.read_excel(file_path, engine='openpyxl' if file_path.endswith('.xlsx') else 'xlrd')
            else:
                QMessageBox.warning(self, "Warning", "Unsupported file format.")
                return
            
            # Basic validation and cleanup
            if self.df is None or self.df.empty:
                raise ValueError("Imported data is empty")
                
            # Ensure DataFrame is properly formed and reset index
            self.df = pd.DataFrame(self.df).reset_index(drop=True)
            
            # Clean up column names (remove extra spaces, special characters)
            self.df.columns = [str(col).strip() for col in self.df.columns]
            
            # Replace any problematic characters in column names
            self.df.columns = [col.replace(' ', '_').replace('.', '_').replace('-', '_') for col in self.df.columns]
            
            self.progress_updated.emit(60)
            self.load_data_overview()
            self.progress_updated.emit(100)
            self.status_updated.emit(f"Data imported successfully. Shape: {self.df.shape}")
            
        except Exception as e:
            error_msg = f"Failed to import data: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.status_updated.emit("Import failed")
            self.progress_updated.emit(0)
            print(f"Debug - Import error: {e}")  # For debugging
            
    def import_from_clipboard(self):
        """Import data from clipboard"""
        try:
            self.status_updated.emit("Importing from clipboard...")
            self.progress_updated.emit(50)
            
            # Use pandas with error handling
            self.df = pd.read_clipboard(sep='\t')  # Try tab-separated first
            
            if self.df is None or self.df.empty:
                # Try comma-separated
                self.df = pd.read_clipboard(sep=',')
                
            if self.df is None or self.df.empty:
                raise ValueError("No valid data in clipboard")
            
            # Ensure DataFrame is properly formed and reset index
            self.df = pd.DataFrame(self.df).reset_index(drop=True)
            
            # Clean up column names
            self.df.columns = [str(col).strip() for col in self.df.columns]
            
            # Replace any problematic characters in column names
            self.df.columns = [col.replace(' ', '_').replace('.', '_').replace('-', '_') for col in self.df.columns]
            
            self.load_data_overview()
            
            self.progress_updated.emit(100)
            self.status_updated.emit(f"Data imported from clipboard successfully. Shape: {self.df.shape}")
            
        except Exception as e:
            error_msg = f"Failed to import data from clipboard: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.status_updated.emit("Import failed")
            self.progress_updated.emit(0)
            print(f"Debug - Clipboard import error: {e}")  # For debugging
            
    def load_data_overview(self):
        """Load data overview and enable controls"""
        if self.df is None:
            return
        
        # Validate that self.df is a DataFrame
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(self.df).__name__}. Data: {self.df}")
            
        # Update data info
        info_text = f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n\n"
        info_text += "Column Information:\n"
        info_text += f"{'Column':<20} {'Type':<15} {'Non-Null':<10} {'Unique':<10}\n"
        info_text += "-" * 55 + "\n"
        
        for col in self.df.columns:
            col_type = str(self.df[col].dtype)
            non_null = self.df[col].count()
            unique = self.df[col].nunique()
            info_text += f"{col[:19]:<20} {col_type:<15} {non_null:<10} {unique:<10}\n"
            
        self.data_info_text.setText(info_text)
        
        # Update data preview table
        self.update_data_table()
        
        # Update feature and target selection
        self.update_feature_target_lists()
        
        # Enable controls
        self.cleaning_group.setEnabled(True)
        self.selection_group.setEnabled(True)
        
        # Create visualization tabs
        self.create_visualization_tabs()
        
        # Analyze data quality
        self.analyze_data_quality()
        
    def update_data_table(self):
        """Update the data preview table"""
        if self.df is None:
            return
            
        # Show first 100 rows
        display_df = self.df.head(100)
        
        self.data_table.setRowCount(display_df.shape[0])
        self.data_table.setColumnCount(display_df.shape[1])
        self.data_table.setHorizontalHeaderLabels(display_df.columns.astype(str))
        
        for i in range(display_df.shape[0]):
            for j in range(display_df.shape[1]):
                value = display_df.iloc[i, j]
                if pd.isna(value):
                    item = QTableWidgetItem("NaN")
                else:
                    item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)
                
    def update_feature_target_lists(self):
        """Update feature and target selection lists"""
        if self.df is None:
            return
            
        columns = self.df.columns.tolist()
        
        # Store current target selection
        current_target = self.target_combo.currentText()
        
        # Update features list
        self.features_list.clear()
        for col in columns:
            item = QListWidgetItem(col)
            self.features_list.addItem(item)
            
        # Update target combo
        self.target_combo.clear()
        self.target_combo.addItem("")  # Empty option
        self.target_combo.addItems(columns)
        
        # Restore target selection if it was previously set
        if current_target and current_target in columns:
            target_index = self.target_combo.findText(current_target)
            if target_index >= 0:
                self.target_combo.setCurrentIndex(target_index)
        
        # Connect signal only if not already connected (avoid duplicate connections)
        try:
            self.target_combo.currentTextChanged.disconnect()
        except:
            pass  # Signal wasn't connected
        self.target_combo.currentTextChanged.connect(self.on_target_selection_changed)
        
        # Update categorical columns list
        self.update_categorical_columns()
        
    def update_categorical_columns(self):
        """Update categorical columns list, excluding target column"""
        if self.df is None:
            return
            
        # Get current target selection
        current_target = self.target_combo.currentText()
            
        # Identify categorical columns
        categorical_cols = []
        for col in self.df.columns:
            # Skip target column for encoding
            if current_target and col == current_target:
                continue
                
            if (self.df[col].dtype == 'object' or 
                self.df[col].dtype.name == 'category' or
                (self.df[col].dtype in ['int64', 'float64'] and self.df[col].nunique() < 20)):
                categorical_cols.append(col)
        
        # Update categorical list
        self.categorical_list.clear()
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            item_text = f"{col} ({unique_count} unique values)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, col)  # Store original column name
            self.categorical_list.addItem(item)
        
        # Enable encoding button if categorical columns exist
        self.apply_encoding_btn.setEnabled(len(categorical_cols) > 0)
        
        # Show info about excluded target
        if current_target and current_target in [col for col in self.df.columns if 
                                               (self.df[col].dtype == 'object' or 
                                                self.df[col].dtype.name == 'category')]:
            self.status_updated.emit(f"Target column '{current_target}' excluded from categorical encoding")
        else:
            self.status_updated.emit(f"Found {len(categorical_cols)} categorical columns for encoding")
        
    def create_visualization_tabs(self):
        """Create visualization tabs"""
        if self.df is None:
            return
            
        # Remove existing viz tabs
        for tab_name in list(self.viz_tabs.keys()):
            self.remove_viz_tab(tab_name)
            
        # Data Quality tab
        self.create_data_quality_tab()
        
        # Statistical Summary tab
        self.create_statistical_summary_tab()
        
        # Distributions tab
        self.create_distributions_tab()
        
        # Correlation tab
        self.create_correlation_tab()
        
        # Data Types tab
        self.create_data_types_tab()
        
    def create_data_quality_tab(self):
        """Create data quality visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create plots
        try:
            # Missing values heatmap
            fig1, canvas1 = plot_missing_values_heatmap(self.df)
            layout.addWidget(QLabel("Missing Values Heatmap:"))
            layout.addWidget(canvas1)
            
            # Missing values bar plot
            fig2, canvas2 = plot_missing_values_bar(self.df)
            layout.addWidget(QLabel("Missing Values by Column:"))
            layout.addWidget(canvas2)
            
        except Exception as e:
            error_label = QLabel(f"Error creating plots: {str(e)}")
            layout.addWidget(error_label)
            
        self.add_viz_tab(widget, "Data Quality")
        
    def create_statistical_summary_tab(self):
        """Create statistical summary visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        try:
            # Create scrollable area
            scroll_area = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            
            # Descriptive statistics table
            desc_stats = self.df.describe(include='all').round(3)
            
            # Create HTML table for better formatting
            html_content = "<h3>Descriptive Statistics</h3>"
            html_content += desc_stats.to_html(classes='table table-striped')
            
            # Add data types info
            html_content += "<h3>Data Types Summary</h3>"
            dtype_info = self.df.dtypes.value_counts()
            html_content += dtype_info.to_frame('Count').to_html()
            
            # Add missing values summary
            html_content += "<h3>Missing Values Summary</h3>"
            missing_info = self.df.isnull().sum()
            missing_pct = (self.df.isnull().sum() / len(self.df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Missing Count': missing_info,
                'Missing %': missing_pct
            })
            html_content += missing_df[missing_df['Missing Count'] > 0].to_html()
            
            # Add unique values info
            html_content += "<h3>Unique Values Count</h3>"
            unique_info = self.df.nunique().to_frame('Unique Values')
            html_content += unique_info.to_html()
            
            text_widget = QTextEdit()
            text_widget.setHtml(html_content)
            text_widget.setReadOnly(True)
            
            scroll_layout.addWidget(text_widget)
            scroll_area.setWidget(scroll_widget)
            layout.addWidget(scroll_area)
            
        except Exception as e:
            error_label = QLabel(f"Error creating statistical summary: {str(e)}")
            layout.addWidget(error_label)
            
        self.add_viz_tab(widget, "Statistical Summary")
        
    def create_distributions_tab(self):
        """Create interactive distributions visualization tab with dropdown selection"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        try:
            # Control panel
            control_panel = QWidget()
            control_layout = QHBoxLayout(control_panel)
            
            # Feature selection dropdown
            control_layout.addWidget(QLabel("Select Feature:"))
            self.feature_combo = QComboBox()
            
            # Get all columns (numeric and categorical)
            all_columns = list(self.df.columns)
            self.feature_combo.addItems(all_columns)
            
            # Connect to update function
            self.feature_combo.currentTextChanged.connect(self.update_distribution_plot)
            control_layout.addWidget(self.feature_combo)
            
            # Plot type selection for categorical features
            control_layout.addWidget(QLabel("Plot Type:"))
            self.plot_type_combo = QComboBox()
            self.plot_type_combo.addItems(["Auto", "Histogram", "Bar Chart", "Pie Chart"])
            self.plot_type_combo.currentTextChanged.connect(self.update_distribution_plot)
            control_layout.addWidget(self.plot_type_combo)
            
            # Bins control for histograms
            control_layout.addWidget(QLabel("Bins:"))
            self.bins_spin = QSpinBox()
            self.bins_spin.setMinimum(5)
            self.bins_spin.setMaximum(100)
            self.bins_spin.setValue(30)
            self.bins_spin.valueChanged.connect(self.update_distribution_plot)
            control_layout.addWidget(self.bins_spin)
            
            control_layout.addStretch()
            layout.addWidget(control_panel)
            
            # Plot area
            self.distribution_plot_widget = QWidget()
            self.distribution_plot_layout = QVBoxLayout(self.distribution_plot_widget)
            layout.addWidget(self.distribution_plot_widget)
            
            # Initial plot
            if len(all_columns) > 0:
                self.update_distribution_plot()
            else:
                info_label = QLabel("No columns found for distribution plots.")
                self.distribution_plot_layout.addWidget(info_label)
            
        except Exception as e:
            error_label = QLabel(f"Error creating distribution plots: {str(e)}")
            layout.addWidget(error_label)
            
        self.add_viz_tab(widget, "Distributions")
    
    def update_distribution_plot(self):
        """Update distribution plot based on selected feature"""
        try:
            # Clear previous plot
            for i in reversed(range(self.distribution_plot_layout.count())):
                child = self.distribution_plot_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            
            selected_feature = self.feature_combo.currentText()
            plot_type = self.plot_type_combo.currentText()
            bins = self.bins_spin.value()
            
            if not selected_feature or selected_feature not in self.df.columns:
                return
            
            # Get feature data
            feature_data = self.df[selected_feature].dropna()
            
            if len(feature_data) == 0:
                info_label = QLabel(f"No valid data for feature: {selected_feature}")
                self.distribution_plot_layout.addWidget(info_label)
                return
            
            # Determine if feature is numeric or categorical
            is_numeric = pd.api.types.is_numeric_dtype(feature_data)
            unique_values = feature_data.nunique()
            
            # Create figure
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            fig = Figure(figsize=(14, 8))
            
            # Auto-determine plot type if set to "Auto"
            if plot_type == "Auto":
                if is_numeric and unique_values > 10:
                    actual_plot_type = "Histogram"
                elif unique_values <= 20:
                    actual_plot_type = "Bar Chart"
                else:
                    actual_plot_type = "Histogram"
            else:
                actual_plot_type = plot_type
            
            # Create plots based on type
            if actual_plot_type == "Histogram" and is_numeric:
                # Two subplots: histogram and box plot
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                
                # Histogram
                ax1.hist(feature_data, bins=bins, alpha=0.7, edgecolor='black', color='skyblue')
                ax1.set_title(f'Distribution of {selected_feature}')
                ax1.set_xlabel(selected_feature)
                ax1.set_ylabel('Frequency')
                ax1.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = f'Mean: {feature_data.mean():.2f}\n'
                stats_text += f'Median: {feature_data.median():.2f}\n'
                stats_text += f'Std: {feature_data.std():.2f}\n'
                stats_text += f'Count: {len(feature_data)}'
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Box plot
                ax2.boxplot(feature_data, vert=True)
                ax2.set_title(f'Box Plot of {selected_feature}')
                ax2.set_ylabel(selected_feature)
                ax2.grid(True, alpha=0.3)
                
            elif actual_plot_type == "Bar Chart" or not is_numeric:
                # Bar chart for categorical or low-cardinality numeric data
                ax1 = fig.add_subplot(111)
                
                value_counts = feature_data.value_counts().head(20)  # Top 20 values
                
                bars = ax1.bar(range(len(value_counts)), value_counts.values, color='lightcoral', alpha=0.7)
                ax1.set_xticks(range(len(value_counts)))
                ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax1.set_title(f'Distribution of {selected_feature}')
                ax1.set_xlabel(selected_feature)
                ax1.set_ylabel('Count')
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + max(value_counts.values) * 0.01,
                            f'{int(height)}', ha='center', va='bottom', fontsize=9)
                
                # Add statistics text
                stats_text = f'Unique values: {unique_values}\n'
                stats_text += f'Most frequent: {value_counts.index[0]} ({value_counts.iloc[0]})\n'
                stats_text += f'Total count: {len(feature_data)}'
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            elif actual_plot_type == "Pie Chart":
                # Pie chart for categorical data
                ax1 = fig.add_subplot(111)
                
                value_counts = feature_data.value_counts().head(10)  # Top 10 values
                
                # If there are more than 10 categories, group the rest as "Others"
                if unique_values > 10:
                    others_count = len(feature_data) - value_counts.sum()
                    if others_count > 0:
                        value_counts['Others'] = others_count
                
                wedges, texts, autotexts = ax1.pie(value_counts.values, labels=value_counts.index, 
                                                  autopct='%1.1f%%', startangle=90)
                ax1.set_title(f'Distribution of {selected_feature}')
                
                # Improve text readability
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            
            # Add navigation toolbar
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            toolbar = NavigationToolbar(canvas, self.distribution_plot_widget)
            
            self.distribution_plot_layout.addWidget(toolbar)
            self.distribution_plot_layout.addWidget(canvas)
            
            # Add summary information
            summary_text = QTextEdit()
            summary_text.setMaximumHeight(80)
            summary_text.setReadOnly(True)
            
            summary = f"Feature: {selected_feature} | Type: {'Numeric' if is_numeric else 'Categorical'} | "
            summary += f"Unique values: {unique_values} | Missing: {self.df[selected_feature].isnull().sum()} | "
            summary += f"Plot type: {actual_plot_type}"
            
            summary_text.setText(summary)
            self.distribution_plot_layout.addWidget(summary_text)
            
        except Exception as e:
            error_label = QLabel(f"Error updating distribution plot: {str(e)}")
            self.distribution_plot_layout.addWidget(error_label)
            print(f"Error in update_distribution_plot: {e}")
        
    def create_correlation_tab(self):
        """Create correlation visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        try:
            fig, canvas = plot_correlation_heatmap(self.df)
            layout.addWidget(canvas)
        except Exception as e:
            error_label = QLabel(f"Error creating correlation plot: {str(e)}")
            layout.addWidget(error_label)
            
        self.add_viz_tab(widget, "Correlations")
        
    def create_data_types_tab(self):
        """Create data types visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        try:
            # Create pie chart for data types
            fig, ax = create_figure(figsize=(10, 6))
            
            # Count data types
            dtype_counts = self.df.dtypes.value_counts()
            
            # Create subplot layout
            fig.clear()
            
            # Pie chart for data types
            ax1 = fig.add_subplot(221)
            ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Data Types Distribution')
            
            # Bar chart for missing values
            ax2 = fig.add_subplot(222)
            missing_counts = self.df.isnull().sum()
            missing_counts = missing_counts[missing_counts > 0]
            if len(missing_counts) > 0:
                ax2.bar(range(len(missing_counts)), missing_counts.values)
                ax2.set_xticks(range(len(missing_counts)))
                ax2.set_xticklabels(missing_counts.index, rotation=45, ha='right')
                ax2.set_title('Missing Values by Column')
                ax2.set_ylabel('Missing Count')
            else:
                ax2.text(0.5, 0.5, 'No missing values', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Missing Values by Column')
            
            # Bar chart for unique values
            ax3 = fig.add_subplot(223)
            unique_counts = self.df.nunique().head(10)  # Top 10 columns
            ax3.bar(range(len(unique_counts)), unique_counts.values)
            ax3.set_xticks(range(len(unique_counts)))
            ax3.set_xticklabels(unique_counts.index, rotation=45, ha='right')
            ax3.set_title('Unique Values Count (Top 10 Columns)')
            ax3.set_ylabel('Unique Count')
            
            # Memory usage
            ax4 = fig.add_subplot(224)
            memory_usage = self.df.memory_usage(deep=True).drop('Index')
            memory_usage = memory_usage.head(10)  # Top 10 columns
            ax4.bar(range(len(memory_usage)), memory_usage.values / 1024)  # Convert to KB
            ax4.set_xticks(range(len(memory_usage)))
            ax4.set_xticklabels(memory_usage.index, rotation=45, ha='right')
            ax4.set_title('Memory Usage by Column (KB)')
            ax4.set_ylabel('Memory (KB)')
            
            fig.tight_layout()
            
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            
        except Exception as e:
            error_label = QLabel(f"Error creating data types visualization: {str(e)}")
            layout.addWidget(error_label)
            
        self.add_viz_tab(widget, "Data Types & Stats")
        
    def add_viz_tab(self, widget, name):
        """Add visualization tab"""
        self.right_panel.addTab(widget, name)
        self.viz_tabs[name] = widget
        
    def remove_viz_tab(self, name):
        """Remove visualization tab"""
        if name in self.viz_tabs:
            widget = self.viz_tabs[name]
            for i in range(self.right_panel.count()):
                if self.right_panel.widget(i) == widget:
                    self.right_panel.removeTab(i)
                    break
            del self.viz_tabs[name]
            
    def handle_missing_values(self):
        """Handle missing values based on selected method"""
        if self.df is None:
            return
            
        method = self.missing_method_combo.currentText()
        
        try:
            original_shape = self.df.shape
            
            if method == "Drop rows":
                self.df = self.df.dropna()
            elif method == "Drop columns":
                self.df = self.df.dropna(axis=1)
            elif method == "Fill mean":
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            elif method == "Fill median":
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
            elif method == "Fill mode":
                for col in self.df.columns:
                    mode_value = self.df[col].mode()
                    if len(mode_value) > 0:
                        self.df[col] = self.df[col].fillna(mode_value[0])
            elif method == "Fill constant":
                self.df = self.df.fillna(0)  # Simple constant fill
                
            self.status_updated.emit(f"Applied {method}. Shape: {original_shape} → {self.df.shape}")
            
            # Update displays
            self.load_data_overview()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error handling missing values: {str(e)}")
            
    def remove_duplicates(self):
        """Remove duplicate rows"""
        if self.df is None:
            return
            
        original_shape = self.df.shape
        duplicate_count = self.df.duplicated().sum()
        
        if duplicate_count == 0:
            QMessageBox.information(self, "Info", "No duplicate rows found.")
            return
            
        reply = QMessageBox.question(
            self, "Remove Duplicates",
            f"Found {duplicate_count} duplicate rows. Remove them?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.df = self.df.drop_duplicates()
            self.status_updated.emit(f"Removed {duplicate_count} duplicate rows. Shape: {original_shape} → {self.df.shape}")
            self.load_data_overview()
            
    def select_all_features(self):
        """Select all features"""
        for i in range(self.features_list.count()):
            item = self.features_list.item(i)
            item.setSelected(True)
            
    def clear_all_features(self):
        """Clear all feature selections"""
        self.features_list.clearSelection()
        
    def update_task_type_suggestion(self):
        """Update task type suggestion based on target selection"""
        target_col = self.target_combo.currentText()
        
        if target_col and target_col in self.df.columns:
            try:
                task_type = suggest_task_type(self.df[target_col])
                self.task_type_label.setText(f"Suggested Task: {task_type.title()}")
                
                # Enable proceed button if valid selection
                self.check_selection_validity()
                
            except Exception as e:
                self.task_type_label.setText("Suggested Task: Error determining task type")
        else:
            self.task_type_label.setText("Suggested Task: Not determined")
            self.proceed_btn.setEnabled(False)
            
    def check_selection_validity(self):
        """Check if feature and target selection is valid"""
        # Get selected features
        selected_items = self.features_list.selectedItems()
        selected_features = [item.text() for item in selected_items]
        
        target_col = self.target_combo.currentText()
        
        # Validate selection
        if self.df is not None:
            is_valid, error_msg = validate_feature_target_selection(
                self.df, selected_features, target_col
            )
            
            # Enable confirm button instead of proceed button directly
            self.confirm_selection_btn.setEnabled(is_valid)
            
            if not is_valid and error_msg:
                self.status_updated.emit(f"Selection error: {error_msg}")
            elif is_valid:
                self.status_updated.emit(f"Valid selection: {len(selected_features)} features, 1 target (click Confirm to proceed)")
                
        # Proceed button is only enabled after confirmation
        # (handled in confirm_selection method)
        
    def proceed_to_next_module(self):
        """Proceed to feature selection module"""
        # Get selected features
        selected_items = self.features_list.selectedItems()
        selected_features = [item.text() for item in selected_items]
        
        target_col = self.target_combo.currentText()
        
        # Validate and prepare data
        is_valid, error_msg = validate_feature_target_selection(
            self.df, selected_features, target_col
        )
        
        if not is_valid:
            QMessageBox.warning(self, "Selection Error", error_msg)
            return
            
        try:
            # CRITICAL FIX: Save the current dataframe state as final
            self.final_df = self.df.copy()
            
            print(f"=== DATA MODULE EXPORT CHECK ===")
            print(f"Original dataframe shape: {self.final_df.shape}")
            print(f"Selected features: {len(selected_features)}")
            print(f"Target column: {target_col}")
            
            # Prepare X and y from the final dataframe state
            self.X = self.final_df[selected_features].copy()
            self.y = self.final_df[target_col].copy()
            
            print(f"X shape before any processing: {self.X.shape}")
            print(f"y shape before any processing: {self.y.shape}")
            
            # Check for any missing values in the final selection
            x_missing = self.X.isnull().sum().sum()
            y_missing = self.y.isnull().sum()
            
            if x_missing > 0 or y_missing > 0:
                print(f"WARNING: Missing values detected - X: {x_missing}, y: {y_missing}")
                print("These will need to be handled consistently in downstream modules")
            else:
                print("✓ No missing values in final selection")
            
            # CRITICAL FIX: Remove target encoding to prevent data leakage
            # Target encoding will be handled in the training pipeline after train/test split
            task_type = suggest_task_type(self.y)
            
            # Store original target information for downstream modules
            self.label_encoder = None
            self.original_classes = None
            
            if task_type == 'classification':
                if self.y.dtype == 'object' or self.y.dtype.name == 'category':
                    # Store original classes but DO NOT encode yet
                    self.original_classes = self.y.unique()
                    print(f"Target classes detected: {self.original_classes}")
                    print("⚠️  Target encoding will be handled in training module to prevent data leakage")
                    self.status_updated.emit(f"Target classes detected: {len(self.original_classes)} classes. Encoding will be done during training.")
                else:
                    print(f"✓ Target already numeric: {self.y.unique()}")
            
            print(f"Final X shape: {self.X.shape}")
            print(f"Final y shape: {self.y.shape}")
            print(f"Task type: {task_type}")
            print("✓ Data prepared WITHOUT target encoding to prevent data leakage")
            
            # Store selections
            self.selected_features = selected_features
            self.selected_target = target_col
            
            # CRITICAL FIX: Record prediction encoding mapping NOW with only selected features
            if self.encoding_applied and hasattr(self, 'final_df'):
                try:
                    from utils.prediction_utils import record_feature_encoding_for_prediction
                    
                    # Use saved original DataFrame before encoding if available
                    if hasattr(self, 'original_df_before_encoding'):
                        original_source_df = self.original_df_before_encoding
                        print(f"✅ Using saved original DataFrame before encoding")
                    else:
                        # Fallback: use current df (may not be ideal but better than nothing)
                        original_source_df = self.df
                        print(f"⚠️ Using current DataFrame as original (encoding may have been applied)")
                    
                    # Create DataFrame with only selected features (exclude target from encoding mapping)
                    if hasattr(self, 'final_df'):
                        # Get original DataFrame state with only selected features (no target)
                        original_features_df = original_source_df[selected_features].copy()
                        # Get encoded DataFrame state with only selected features (no target)
                        encoded_features_df = self.final_df[selected_features].copy()
                        
                        # Find which features were actually encoded by comparing structure
                        encoded_categorical_cols = []
                        
                        # Check for one-hot encoding pattern (new columns created)
                        for col in selected_features:
                            if col in original_features_df.columns:
                                # Check if this original column spawned multiple encoded columns
                                related_encoded_cols = [enc_col for enc_col in encoded_features_df.columns 
                                                      if enc_col.startswith(f"{col}_")]
                                if related_encoded_cols:
                                    encoded_categorical_cols.append(col)
                                    print(f"🔍 Found one-hot encoding: {col} -> {related_encoded_cols}")
                                elif col in encoded_features_df.columns:
                                    # Check if values changed (label encoding)
                                    orig_unique = set(original_features_df[col].dropna().astype(str))
                                    encoded_unique = set(encoded_features_df[col].dropna().astype(str))
                                    if orig_unique != encoded_unique:
                                        encoded_categorical_cols.append(col)
                                        print(f"🔍 Found label encoding: {col}")
                        
                        if encoded_categorical_cols:
                            # CRITICAL FIX: Record mapping using ONLY FEATURES (exclude target from prediction mapping)
                            original_features_only = original_source_df[selected_features].copy()
                            encoded_features_only = self.final_df[selected_features].copy()
                            
                            print(f"🔍 Recording prediction mapping with:")
                            print(f"   - Original features shape: {original_features_only.shape}")
                            print(f"   - Encoded features shape: {encoded_features_only.shape}")
                            print(f"   - Categorical columns to map: {encoded_categorical_cols}")
                            print(f"   - Target column '{target_col}' is EXCLUDED from prediction mapping")
                            
                            record_feature_encoding_for_prediction(original_features_only, encoded_features_only, encoded_categorical_cols)
                            print(f"✅ Recorded prediction encoding mapping for {len(encoded_categorical_cols)} selected categorical features")
                            print(f"📝 Features mapped: {encoded_categorical_cols}")
                        else:
                            print("ℹ️ No categorical features found in selected features, no prediction mapping needed")
                    
                except Exception as e:
                    print(f"⚠️ Could not record prediction encoding mapping: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue anyway - this is not critical for basic functionality
            
            # Emit signal with RAW data (no target encoding)
            self.data_ready.emit(self.X, self.y)
            
            success_msg = f"Data prepared successfully!\n\nFeatures: {len(selected_features)}\nTarget: {target_col}\nSamples: {len(self.X)}\nTask Type: {task_type}"
            
            if self.original_classes is not None:
                success_msg += f"\nTarget Classes: {len(self.original_classes)} (will be encoded during training)"
            
            self.status_updated.emit("Data prepared successfully. Target encoding will be handled during training to prevent data leakage.")
            QMessageBox.information(self, "Success", success_msg)
            
        except Exception as e:
            print(f"ERROR in proceed_to_next_module: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error preparing data: {str(e)}")
            
    def reset(self):
        """Reset the module"""
        self.df = None
        self.X = None
        self.y = None
        self.selected_features = []
        self.selected_target = ""
        self.final_df = None  # Store final processed dataframe state
        self.encoding_applied = False  # Track if encoding has been applied
        self.encoded_with_target = None  # Track which target was used for encoding
        self.label_encoder = None
        self.original_classes = None
        
        # Remove saved original DataFrame if it exists
        if hasattr(self, 'original_df_before_encoding'):
            delattr(self, 'original_df_before_encoding')
        
        # Clear UI
        self.file_path_edit.clear()
        self.data_info_text.clear()
        self.data_table.clear()
        self.features_list.clear()
        self.target_combo.clear()
        self.categorical_list.clear()
        self.task_type_label.setText("Suggested Task: Not determined")
        
        # Reset encoding combo
        self.encoding_method_combo.setCurrentIndex(0)
        
        # Disable controls
        self.cleaning_group.setEnabled(False)
        self.selection_group.setEnabled(False)
        self.apply_encoding_btn.setEnabled(False)
        self.confirm_selection_btn.setEnabled(False)
        self.proceed_btn.setEnabled(False)
        
        # Remove viz tabs
        for tab_name in list(self.viz_tabs.keys()):
            self.remove_viz_tab(tab_name)
            
        self.status_updated.emit("Module reset")
        
    def apply_categorical_encoding(self):
        """Apply categorical encoding to selected columns (excluding target)"""
        if self.df is None:
            return
            
        method = self.encoding_method_combo.currentText()
        
        if method == "None (Keep as is)":
            QMessageBox.information(self, "Info", "No encoding applied.")
            return
            
        try:
            # Get current target to ensure exclusion and preserve selection
            current_target = self.target_combo.currentText()
            
            # Get selected categorical columns from the list
            selected_items = self.categorical_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Warning", "Please select categorical columns to encode.")
                return
            
            categorical_cols = []
            for item in selected_items:
                col_name = item.data(Qt.UserRole)
                
                # Double-check: exclude target column
                if current_target and col_name == current_target:
                    continue
                    
                categorical_cols.append(col_name)
            
            if not categorical_cols:
                QMessageBox.warning(self, "Warning", "No categorical columns found for encoding.")
                return
            
            # Verify target column still exists before encoding
            if current_target and current_target not in self.df.columns:
                QMessageBox.warning(self, "Warning", f"Target column '{current_target}' not found in data.")
                return
            
            original_shape = self.df.shape
            
            # Apply encoding and update tracking
            if method == "One-Hot Encoding":
                # Store original DataFrame for mapping
                original_df = self.df.copy()
                
                # Apply one-hot encoding only to feature columns
                encoded_df = pd.get_dummies(self.df, columns=categorical_cols, prefix=categorical_cols)
                self.df = encoded_df
                
                # Record encoding mapping for feature name restoration
                try:
                    from utils.feature_name_utils import record_feature_encoding
                    record_feature_encoding(original_df, encoded_df, categorical_cols, "One-Hot Encoding")
                    print(f"DEBUG: Recorded one-hot encoding mapping for {len(categorical_cols)} categorical features")
                except ImportError:
                    print("Warning: Could not import feature name utilities for mapping")
                
                # NOTE: Prediction encoding mapping will be recorded later in proceed_to_next_module
                # to ensure only final selected features are recorded, not all columns
                print(f"DEBUG: Categorical encoding applied, prediction mapping will be recorded after feature selection")
                
            elif method == "Label Encoding":
                from sklearn.preprocessing import LabelEncoder
                for col in categorical_cols:
                    if col in self.df.columns:
                        le = LabelEncoder()
                        self.df[col] = le.fit_transform(self.df[col].astype(str))
                        
            elif method == "Binary Encoding":
                try:
                    import category_encoders as ce
                    encoder = ce.BinaryEncoder(cols=categorical_cols)
                    encoded_df = encoder.fit_transform(self.df)
                    self.df = encoded_df
                except ImportError:
                    QMessageBox.warning(self, "Warning", "Binary encoding requires 'category_encoders' package. Using Label Encoding instead.")
                    from sklearn.preprocessing import LabelEncoder
                    for col in categorical_cols:
                        if col in self.df.columns:
                            le = LabelEncoder()
                            self.df[col] = le.fit_transform(self.df[col].astype(str))
            
            # CRITICAL FIX: Track encoding application and save original state
            self.encoding_applied = True
            self.encoded_with_target = current_target
            
            # Save original DataFrame state before encoding for prediction mapping
            if not hasattr(self, 'original_df_before_encoding'):
                self.original_df_before_encoding = original_df.copy()
            
            new_shape = self.df.shape
            self.status_updated.emit(f"Categorical encoding completed: {original_shape} -> {new_shape}")
            
            # Update UI
            self.update_categorical_columns()
            self.load_data_overview()
            
            QMessageBox.information(
                self, "Success", 
                f"Categorical encoding completed successfully!\n\n"
                f"Method: {method}\n"
                f"Columns encoded: {len(categorical_cols)}\n"
                f"Shape change: {original_shape} -> {new_shape}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying encoding: {str(e)}")
            
    def confirm_selection(self):
        """Confirm feature and target selection"""
        # Get selected features
        selected_items = self.features_list.selectedItems()
        selected_features = [item.text() for item in selected_items]
        
        target_col = self.target_combo.currentText()
        
        # Validate selection
        if self.df is not None:
            is_valid, error_msg = validate_feature_target_selection(
                self.df, selected_features, target_col
            )
            
            if not is_valid:
                QMessageBox.warning(self, "Selection Error", error_msg)
                return
                
            # Show confirmation dialog
            msg = f"Confirm Selection:\n\n"
            msg += f"Features ({len(selected_features)}):\n"
            for i, feature in enumerate(selected_features[:10]):  # Show first 10
                msg += f"  • {feature}\n"
            if len(selected_features) > 10:
                msg += f"  ... and {len(selected_features) - 10} more\n"
            msg += f"\nTarget: {target_col}\n"
            msg += f"Total samples: {len(self.df)}\n"
            
            reply = QMessageBox.question(
                self, "Confirm Selection", msg,
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.proceed_btn.setEnabled(True)
                self.status_updated.emit("Selection confirmed. Ready to proceed.")
                QMessageBox.information(self, "Success", "Feature and target selection confirmed!")
            else:
                self.proceed_btn.setEnabled(False) 

    def on_target_selection_changed(self, target_col):
        """Handle target column selection change"""
        if target_col and self.df is not None:
            # CRITICAL FIX: Warn if target changes after encoding
            if self.encoding_applied and self.encoded_with_target and self.encoded_with_target != target_col:
                reply = QMessageBox.warning(
                    self, 
                    "Target Change Warning",
                    f"Warning: You have changed the target column from '{self.encoded_with_target}' to '{target_col}' "
                    f"after applying categorical encoding.\n\n"
                    f"This may cause data inconsistency because the encoded features were created using "
                    f"information from the previous target column.\n\n"
                    f"Recommended actions:\n"
                    f"1. Reset the module and start over, or\n"
                    f"2. Change back to the original target: '{self.encoded_with_target}'\n\n"
                    f"Do you want to continue with the new target anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    # Revert to the previous target
                    self.target_combo.setCurrentText(self.encoded_with_target)
                    return
                else:
                    # User chose to continue - clear encoding tracking
                    self.encoding_applied = False
                    self.encoded_with_target = None
            
            self.update_task_type_suggestion()
            self.update_categorical_columns()
            self.check_selection_validity()
        
    def analyze_data_quality(self):
        """Analyze data quality and provide preprocessing recommendations"""
        if self.df is None:
            return
            
        results = []
        
        # Basic data info
        n_samples, n_features = self.df.shape
        results.append(f"数据维度: {n_samples} 样本 × {n_features} 特征")
        
        # Check for high-dimensional data (likely from one-hot encoding)
        if n_features > 50:
            results.append(f"⚠️ 检测到高维数据 ({n_features} 特征)")
            
            # Check for one-hot encoded patterns
            boolean_cols = self.df.select_dtypes(include=['bool']).columns
            if len(boolean_cols) > n_features * 0.7:
                results.append(f"📊 检测到大量布尔特征 ({len(boolean_cols)}个)，可能来自One-Hot编码")
                results.append("💡 建议:")
                results.append("   1. 使用特征选择减少维度")
                results.append("   2. 考虑使用目标编码替代One-Hot编码")
                results.append("   3. 应用PCA降维")
                results.append("   4. 使用正则化模型(如Ridge/Lasso)")
            
            # Check for sparse data
            if boolean_cols.any():
                sparsity = (self.df[boolean_cols] == False).sum().sum() / (len(boolean_cols) * n_samples)
                if sparsity > 0.9:
                    results.append(f"🔍 数据稀疏度: {sparsity:.1%} (大部分值为False)")
                    results.append("💡 稀疏数据建议:")
                    results.append("   1. 移除低方差特征")
                    results.append("   2. 使用支持稀疏数据的模型")
                    results.append("   3. 考虑特征聚合")
        
        # Check for class imbalance (if target is available)
        if hasattr(self, 'target_column') and self.target_column and self.target_column in self.df.columns:
            target = self.df[self.target_column]
            if target.dtype == 'object' or target.nunique() < 10:
                value_counts = target.value_counts()
                min_class_ratio = value_counts.min() / value_counts.max()
                if min_class_ratio < 0.3:
                    results.append(f"⚖️ 检测到类别不平衡:")
                    for cls, count in value_counts.items():
                        percentage = count / len(target) * 100
                        results.append(f"   {cls}: {count} ({percentage:.1f}%)")
                    results.append("💡 平衡建议: 已自动启用class_weight='balanced'")
        
        # Memory usage analysis
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 100:
            results.append(f"💾 内存使用: {memory_mb:.1f} MB")
            if memory_mb > 500:
                results.append("⚠️ 数据较大，建议使用增量学习或数据采样")
        
        # Display results
        analysis_text = "\n".join(results)
        
        # Create or update analysis display
        if not hasattr(self, 'analysis_text'):
            self.analysis_text = QTextEdit()
            self.analysis_text.setMaximumHeight(200)
            self.analysis_text.setStyleSheet("""
                QTextEdit {
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    padding: 8px;
                    font-family: 'Courier New', monospace;
                    font-size: 9pt;
                }
            """)
            
            # Add to overview widget layout instead of main layout
            try:
                # Get the overview widget's layout
                overview_layout = self.overview_widget.layout()
                if overview_layout:
                    # Add analysis before the data table
                    analysis_label = QLabel("📋 数据质量分析:")
                    analysis_label.setStyleSheet("font-weight: bold; color: #495057; margin-top: 10px;")
                    # Insert at the beginning of overview layout
                    overview_layout.insertWidget(0, analysis_label)
                    overview_layout.insertWidget(1, self.analysis_text)
            except Exception as e:
                # Fallback: just add to overview layout normally
                overview_layout = self.overview_widget.layout()
                if overview_layout:
                    analysis_label = QLabel("📋 数据质量分析:")
                    analysis_label.setStyleSheet("font-weight: bold; color: #495057; margin-top: 10px;")
                    overview_layout.addWidget(analysis_label)
                    overview_layout.addWidget(self.analysis_text)
        
        self.analysis_text.setPlainText(analysis_text) 

    def on_encoding_method_changed(self):
        """Handle encoding method change and update description"""
        self.update_encoding_description()

    def update_encoding_description(self):
        """Update the encoding method description"""
        method = self.encoding_method_combo.currentText()
        descriptions = {
            "One-Hot Encoding": "Creates binary columns for each category. Good for nominal data with few categories.",
            "Label Encoding": "Assigns integer values to categories. Good for ordinal data or high-cardinality features.",
            "Target Encoding": "Replaces categories with target mean. Good for high-cardinality features.",
            "Binary Encoding": "Uses binary representation. Memory-efficient for high-cardinality features.",
            "Frequency Encoding": "Replaces categories with their frequency. Good for capturing category importance."
        }
        
        self.encoding_description.setText(descriptions.get(method, ""))

    def load_training_data(self):
        """Load training data specifically - wrapper for import functionality"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Load Training Data", 
                "", 
                "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
            )
            
            if file_path:
                self.file_path_edit.setText(file_path)
                self.import_data()
                self.status_updated.emit("Training data loaded successfully")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load training data: {str(e)}")
    
    def load_virtual_data(self):
        """Load virtual screening data specifically - wrapper for import functionality"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Load Virtual Screening Data", 
                "", 
                "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
            )
            
            if file_path:
                self.file_path_edit.setText(file_path)
                self.import_data()
                self.status_updated.emit("Virtual screening data loaded successfully")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load virtual screening data: {str(e)}")
    
    def show_visualization_dialog(self):
        """Show comprehensive data visualization dialog"""
        try:
            if self.df is None:
                QMessageBox.information(self, "No Data", "Please load data first.")
                return
            
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QPushButton
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Data Visualization")
            dialog.setModal(True)
            dialog.resize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Create tab widget for different visualizations
            tab_widget = QTabWidget()
            layout.addWidget(tab_widget)
            
            # Distribution plots tab
            dist_widget = QWidget()
            dist_layout = QVBoxLayout(dist_widget)
            
            # Create matplotlib figure for distributions
            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvas(fig)
            dist_layout.addWidget(canvas)
            
            # Plot distributions of numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                
                for i, col in enumerate(numeric_cols[:9]):  # Limit to 9 plots
                    ax = fig.add_subplot(n_rows, n_cols, i + 1)
                    self.df[col].hist(bins=30, ax=ax, alpha=0.7)
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                
                fig.tight_layout()
                canvas.draw()
            
            tab_widget.addTab(dist_widget, "Distributions")
            
            # Correlation heatmap tab
            corr_widget = QWidget()
            corr_layout = QVBoxLayout(corr_widget)
            
            if len(numeric_cols) > 1:
                corr_fig = Figure(figsize=(10, 8))
                corr_canvas = FigureCanvas(corr_fig)
                corr_layout.addWidget(corr_canvas)
                
                ax = corr_fig.add_subplot(111)
                correlation_matrix = self.df[numeric_cols].corr()
                
                import seaborn as sns
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Feature Correlation Heatmap')
                
                corr_fig.tight_layout()
                corr_canvas.draw()
                
                tab_widget.addTab(corr_widget, "Correlations")
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show visualization: {str(e)}")
    
    def show_data_summary(self):
        """Show comprehensive data summary dialog"""
        try:
            if self.df is None:
                QMessageBox.information(self, "No Data", "Please load data first.")
                return
            
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QTabWidget
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Data Summary")
            dialog.setModal(True)
            dialog.resize(700, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Create tab widget for different summary views
            tab_widget = QTabWidget()
            layout.addWidget(tab_widget)
            
            # Basic info tab
            basic_info_widget = QWidget()
            basic_info_layout = QVBoxLayout(basic_info_widget)
            
            basic_info_text = QTextEdit()
            basic_info_text.setReadOnly(True)
            
            # Generate basic info
            info_text = f"""
Dataset Overview:
================
Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns
Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Column Information:
==================
"""
            
            for col in self.df.columns:
                dtype = self.df[col].dtype
                null_count = self.df[col].isnull().sum()
                null_pct = (null_count / len(self.df)) * 100
                unique_count = self.df[col].nunique()
                
                info_text += f"{col}:\n"
                info_text += f"  - Data Type: {dtype}\n"
                info_text += f"  - Missing Values: {null_count} ({null_pct:.1f}%)\n"
                info_text += f"  - Unique Values: {unique_count}\n\n"
            
            basic_info_text.setPlainText(info_text)
            basic_info_layout.addWidget(basic_info_text)
            
            tab_widget.addTab(basic_info_widget, "Basic Info")
            
            # Statistical summary tab
            stats_widget = QWidget()
            stats_layout = QVBoxLayout(stats_widget)
            
            stats_text = QTextEdit()
            stats_text.setReadOnly(True)
            
            # Generate statistical summary
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_summary = self.df[numeric_cols].describe()
                stats_text.setPlainText(stats_summary.to_string())
            else:
                stats_text.setPlainText("No numeric columns found for statistical summary.")
            
            stats_layout.addWidget(stats_text)
            tab_widget.addTab(stats_widget, "Statistics")
            
            # Data quality tab
            quality_widget = QWidget()
            quality_layout = QVBoxLayout(quality_widget)
            
            quality_text = QTextEdit()
            quality_text.setReadOnly(True)
            
            # Generate data quality report
            quality_report = f"""
Data Quality Assessment:
========================

Missing Values by Column:
-------------------------
"""
            
            for col in self.df.columns:
                missing_count = self.df[col].isnull().sum()
                missing_pct = (missing_count / len(self.df)) * 100
                quality_report += f"{col}: {missing_count} ({missing_pct:.1f}%)\n"
            
            quality_report += f"\nDuplicate Rows: {self.df.duplicated().sum()}\n"
            
            # Check for potential issues
            quality_report += "\nPotential Issues:\n"
            quality_report += "-----------------\n"
            
            for col in numeric_cols:
                if self.df[col].isnull().sum() > len(self.df) * 0.5:
                    quality_report += f"⚠️ {col}: High missing value rate (>{50}%)\n"
                
                if self.df[col].nunique() == 1:
                    quality_report += f"⚠️ {col}: Constant values (no variation)\n"
            
            quality_text.setPlainText(quality_report)
            quality_layout.addWidget(quality_text)
            
            tab_widget.addTab(quality_widget, "Data Quality")
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show data summary: {str(e)}")
    
    def copy_selection(self):
        """Copy selected data to clipboard"""
        try:
            if self.df is None:
                return
            
            # Copy the entire dataframe to clipboard (could be enhanced to copy only selected rows)
            self.df.to_clipboard(index=False)
            self.status_updated.emit("Data copied to clipboard")
        except Exception as e:
            print(f"Error copying data: {e}")
    
    def paste_data(self):
        """Paste data from clipboard"""
        try:
            # Try to read data from clipboard
            clipboard_data = pd.read_clipboard()
            if clipboard_data is not None and not clipboard_data.empty:
                self.df = clipboard_data
                self.load_data_overview()
                self.status_updated.emit("Data pasted from clipboard")
            else:
                QMessageBox.information(self, "Paste Data", "No valid data found in clipboard")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to paste data: {str(e)}")
    
    def find_text(self, search_text):
        """Find text in the data"""
        try:
            if self.df is None:
                return
            
            # Search in all string columns
            results = []
            for col in self.df.select_dtypes(include=['object']).columns:
                mask = self.df[col].astype(str).str.contains(search_text, case=False, na=False)
                if mask.any():
                    matching_rows = self.df[mask]
                    results.append(f"Found {mask.sum()} matches in column '{col}'")
            
            if results:
                result_text = "\n".join(results)
                QMessageBox.information(self, "Search Results", result_text)
            else:
                QMessageBox.information(self, "Search Results", f"No matches found for '{search_text}'")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to search: {str(e)}") 