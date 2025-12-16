"""
Main window for MatSci-ML Studio
"""

import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout,
                            QWidget, QMenuBar, QStatusBar, QAction, QMessageBox,
                            QFileDialog, QProgressBar, QHBoxLayout, QLabel)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

from modules.data_module import DataModule
from modules.feature_module import FeatureModule
from modules.training_module import TrainingModule
from modules.prediction_module import PredictionModule
from modules.intelligent_wizard import IntelligentWizard
from modules.performance_monitor import PerformanceMonitor
from modules.advanced_preprocessing import AdvancedPreprocessing
from modules.collaboration_version_control import CollaborationWidget
from modules.shap_analysis import SHAPAnalysisModule
from modules.target_optimization import TargetOptimizationModule
from modules.multi_objective_optimization import MultiObjectiveOptimizationModule
from modules.clustering import IntelligentClusteringModule

# ALIGNN module - graceful import with fallback
try:
    from modules.alignn import ALIGNNModule, ALIGNN_AVAILABLE, UI_AVAILABLE as ALIGNN_UI_AVAILABLE
except (ImportError, OSError, Exception) as e:
    # Catch ImportError, OSError (DLL loading failures on Windows), and other exceptions
    ALIGNN_AVAILABLE = False
    ALIGNN_UI_AVAILABLE = False
    # Create placeholder widget
    class ALIGNNModule(QWidget):
        """Placeholder when ALIGNN module cannot be imported."""
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setAlignment(Qt.AlignCenter)
            title = QLabel("ALIGNN Module Unavailable")
            title.setStyleSheet("font-size: 18px; font-weight: bold; color: #e74c3c;")
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)
            error_label = QLabel(f"Import error: {e}")
            error_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setWordWrap(True)
            layout.addWidget(error_label)
            instructions = QLabel(
                "\nTo enable ALIGNN functionality, please install:\n"
                "pip install torch dgl jarvis-tools"
            )
            instructions.setStyleSheet("font-size: 11px; color: #95a5a6;")
            instructions.setAlignment(Qt.AlignCenter)
            layout.addWidget(instructions)
            layout.addStretch()

# CGCNN module - graceful import with fallback
try:
    from modules.cgcnn import CGCNNModule, CGCNN_AVAILABLE, UI_AVAILABLE as CGCNN_UI_AVAILABLE
except (ImportError, OSError, Exception) as e:
    # Catch ImportError, OSError (DLL loading failures on Windows), and other exceptions
    CGCNN_AVAILABLE = False
    CGCNN_UI_AVAILABLE = False
    # Create placeholder widget
    class CGCNNModule(QWidget):
        """Placeholder when CGCNN module cannot be imported."""
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setAlignment(Qt.AlignCenter)
            title = QLabel("CGCNN Module Unavailable")
            title.setStyleSheet("font-size: 18px; font-weight: bold; color: #e74c3c;")
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)
            error_label = QLabel(f"Import error: {e}")
            error_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setWordWrap(True)
            layout.addWidget(error_label)
            instructions = QLabel(
                "\nTo enable CGCNN functionality, please install:\n"
                "pip install torch pymatgen"
            )
            instructions.setStyleSheet("font-size: 11px; color: #95a5a6;")
            instructions.setAlignment(Qt.AlignCenter)
            layout.addWidget(instructions)
            layout.addStretch()



class MatSciMLStudioWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.current_project_path = None  # Track current project file
        self.recent_projects = []  # Track recent projects
        self.auto_save_enabled = True  # Auto-save feature
        self.project_modified = False  # Track if project has been modified
        self.init_ui()
        self.setup_modules()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("AutoMatFlow v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application font
        font = QFont("Segoe UI", 9)
        self.setFont(font)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget for modules
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        layout.addWidget(self.tab_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Progress bar for status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def create_menu_bar(self):
        """Create the enhanced menu bar with comprehensive functionality"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # New submenu
        new_menu = file_menu.addMenu("New")
        new_project_action = QAction("Project", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self.new_project)
        new_menu.addAction(new_project_action)
        
        new_experiment_action = QAction("Experiment", self)
        new_experiment_action.setShortcut("Ctrl+Shift+N")
        new_experiment_action.triggered.connect(self.new_experiment)
        new_menu.addAction(new_experiment_action)
        
        file_menu.addSeparator()
        
        # Import Data submenu
        import_menu = file_menu.addMenu("Import Data")
        import_training_action = QAction("Training Data (.csv, .xlsx)", self)
        import_training_action.triggered.connect(self.import_training_data)
        import_menu.addAction(import_training_action)
        
        import_virtual_action = QAction("Virtual Screening Data", self)
        import_virtual_action.triggered.connect(self.import_virtual_data)
        import_menu.addAction(import_virtual_action)
        
        import_model_action = QAction("Pre-trained Model (.joblib)", self)
        import_model_action.triggered.connect(self.import_model)
        import_menu.addAction(import_model_action)
        
        # Export submenu
        export_menu = file_menu.addMenu("Export")
        export_results_action = QAction("Results (.csv)", self)
        export_results_action.triggered.connect(self.export_results)
        export_menu.addAction(export_results_action)
        
        export_charts_action = QAction("Charts (.png, .svg)", self)
        export_charts_action.triggered.connect(self.export_charts)
        export_menu.addAction(export_charts_action)
        
        export_model_action = QAction("Model (.joblib)", self)
        export_model_action.triggered.connect(self.export_model)
        export_menu.addAction(export_model_action)
        
        export_report_action = QAction("Optimization Report (.md, .pdf)", self)
        export_report_action.triggered.connect(self.export_report)
        export_menu.addAction(export_report_action)
        
        file_menu.addSeparator()
        
        # Recent Projects
        recent_menu = file_menu.addMenu("Recent Projects")
        self.update_recent_projects_menu(recent_menu)
        
        file_menu.addSeparator()
        
        # Settings
        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("Save Project", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        load_action = QAction("Load Project", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_project)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo_action)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.redo_action)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        copy_action = QAction("Copy", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.copy_data)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction("Paste", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self.paste_data)
        edit_menu.addAction(paste_action)
        
        edit_menu.addSeparator()
        
        find_action = QAction("Find", self)
        find_action.setShortcut("Ctrl+F")
        find_action.triggered.connect(self.show_find_dialog)
        edit_menu.addAction(find_action)
        
        # Data menu
        data_menu = menubar.addMenu("Data")
        
        data_import_action = QAction("Import Data", self)
        data_import_action.triggered.connect(self.switch_to_data_module)
        data_menu.addAction(data_import_action)
        
        data_preprocessing_action = QAction("Preprocessing", self)
        data_preprocessing_action.triggered.connect(self.switch_to_preprocessing_module)
        data_menu.addAction(data_preprocessing_action)
        
        feature_engineering_action = QAction("Feature Engineering", self)
        feature_engineering_action.triggered.connect(self.switch_to_feature_module)
        data_menu.addAction(feature_engineering_action)
        
        data_menu.addSeparator()
        
        data_visualization_action = QAction("Data Visualization", self)
        data_visualization_action.triggered.connect(self.show_data_visualization)
        data_menu.addAction(data_visualization_action)
        
        data_summary_action = QAction("Data Summary", self)
        data_summary_action.triggered.connect(self.show_data_summary)
        data_menu.addAction(data_summary_action)
        
        # Analyze menu
        analyze_menu = menubar.addMenu("Analyze")
        
        train_model_action = QAction("Train Model", self)
        train_model_action.triggered.connect(self.switch_to_training_module)
        analyze_menu.addAction(train_model_action)
        
        evaluate_model_action = QAction("Evaluate Model", self)
        evaluate_model_action.triggered.connect(self.show_model_evaluation)
        analyze_menu.addAction(evaluate_model_action)
        
        analyze_menu.addSeparator()
        
        shap_analysis_action = QAction("SHAP Analysis", self)
        shap_analysis_action.triggered.connect(self.switch_to_shap_module)
        analyze_menu.addAction(shap_analysis_action)
        
        learning_curve_action = QAction("Learning Curve Analysis", self)
        learning_curve_action.triggered.connect(self.show_learning_curve_analysis)
        analyze_menu.addAction(learning_curve_action)
        
        performance_monitor_action = QAction("Performance Monitor", self)
        performance_monitor_action.triggered.connect(self.switch_to_performance_module)
        analyze_menu.addAction(performance_monitor_action)
        
        analyze_menu.addSeparator()
        
        
        # Optimize menu
        optimize_menu = menubar.addMenu("Optimize")
        
        
        multi_objective_action = QAction("Multi-objective Optimization", self)
        multi_objective_action.triggered.connect(self.switch_to_multi_objective_module)
        optimize_menu.addAction(multi_objective_action)
        
        target_prediction_action = QAction("Target Prediction", self)
        target_prediction_action.triggered.connect(self.switch_to_prediction_module)
        optimize_menu.addAction(target_prediction_action)
        
        optimize_menu.addSeparator()
        
        target_optimization_action = QAction("Target Optimization", self)
        target_optimization_action.triggered.connect(self.switch_to_target_optimization_module)
        optimize_menu.addAction(target_optimization_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Panels submenu
        panels_menu = view_menu.addMenu("Panels")
        
        log_console_action = QAction("Log Console", self)
        log_console_action.setCheckable(True)
        log_console_action.setChecked(True)
        log_console_action.triggered.connect(self.toggle_log_console)
        panels_menu.addAction(log_console_action)
        
        data_viewer_action = QAction("Data Viewer", self)
        data_viewer_action.setCheckable(True)
        data_viewer_action.setChecked(True)
        data_viewer_action.triggered.connect(self.toggle_data_viewer)
        panels_menu.addAction(data_viewer_action)
        
        charts_panel_action = QAction("Charts Panel", self)
        charts_panel_action.setCheckable(True)
        charts_panel_action.setChecked(True)
        charts_panel_action.triggered.connect(self.toggle_charts_panel)
        panels_menu.addAction(charts_panel_action)
        
        view_menu.addSeparator()
        
        # Appearance submenu
        appearance_menu = view_menu.addMenu("Appearance")
        
        theme_action = QAction("Toggle Light/Dark Theme", self)
        theme_action.triggered.connect(self.toggle_theme)
        appearance_menu.addAction(theme_action)
        
        font_size_menu = appearance_menu.addMenu("Font Size")
        
        small_font_action = QAction("Small", self)
        small_font_action.triggered.connect(lambda: self.set_font_size(8))
        font_size_menu.addAction(small_font_action)
        
        normal_font_action = QAction("Normal", self)
        normal_font_action.triggered.connect(lambda: self.set_font_size(9))
        font_size_menu.addAction(normal_font_action)
        
        large_font_action = QAction("Large", self)
        large_font_action.triggered.connect(lambda: self.set_font_size(11))
        font_size_menu.addAction(large_font_action)
        
        view_menu.addSeparator()
        
        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)
        
        fullscreen_action = QAction("Toggle Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        welcome_guide_action = QAction("Welcome Guide", self)
        welcome_guide_action.triggered.connect(self.show_welcome_guide)
        help_menu.addAction(welcome_guide_action)
        
        online_docs_action = QAction("Online Documentation", self)
        online_docs_action.triggered.connect(self.show_online_documentation)
        help_menu.addAction(online_docs_action)
        
        help_menu.addSeparator()
        
        view_logs_action = QAction("View Logs", self)
        view_logs_action.triggered.connect(self.show_logs)
        help_menu.addAction(view_logs_action)
        
        report_issue_action = QAction("Report Issue", self)
        report_issue_action.triggered.connect(self.report_issue)
        help_menu.addAction(report_issue_action)
        
        help_menu.addSeparator()
        
        user_guide_action = QAction("User Guide", self)
        user_guide_action.triggered.connect(self.show_user_guide)
        help_menu.addAction(user_guide_action)
        
        about_action = QAction("About AutoMatFlow", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_modules(self):
        """Setup all application modules"""
        # Module 1: Data Ingestion & Preprocessing
        self.data_module = DataModule()
        self.tab_widget.addTab(self.data_module, "📊 Data Management")
        
        # Module 2: Intelligent Wizard
        self.intelligent_wizard = IntelligentWizard()
        self.tab_widget.addTab(self.intelligent_wizard, "🧙‍♂️ Intelligent Wizard")
        
        # Module 3: Advanced Preprocessing
        self.advanced_preprocessing = AdvancedPreprocessing()
        self.tab_widget.addTab(self.advanced_preprocessing, "🧬 Advanced Preprocessing")
        
        # Module 4: Feature Engineering & Selection
        self.feature_module = FeatureModule()
        self.tab_widget.addTab(self.feature_module, "🎯 Feature Selection")
        
        # Module 5: Model Training & Evaluation
        self.training_module = TrainingModule()
        self.tab_widget.addTab(self.training_module, "🔬 Model Training")
        
        # Module 6: Prediction & Export
        self.prediction_module = PredictionModule()
        self.tab_widget.addTab(self.prediction_module, "🎯 Model Prediction")
        
        # Module 7: Performance Monitor
        self.performance_monitor = PerformanceMonitor()
        self.tab_widget.addTab(self.performance_monitor, "📊 Performance Monitor")
        
        # Module 8: Collaboration & Version Control
        self.collaboration_widget = CollaborationWidget()
        self.tab_widget.addTab(self.collaboration_widget, "🤝 Collaboration")
        
        # Module 9: SHAP Analysis
        self.shap_analysis = SHAPAnalysisModule()
        self.tab_widget.addTab(self.shap_analysis, "🧠 SHAP Analysis")
        
        # Module 10: Target Optimization
        self.target_optimization = TargetOptimizationModule()
        self.tab_widget.addTab(self.target_optimization, "🎯 Target Optimization")
        
        # Module 11: ALIGNN Neural Network
        self.alignn_module = ALIGNNModule()
        # Show availability status in tab title
        try:
            alignn_available = ALIGNN_AVAILABLE
        except NameError:
            alignn_available = False
        alignn_title = "🧠 ALIGNN Neural Network" if alignn_available else "⚠️ ALIGNN (Unavailable)"
        self.tab_widget.addTab(self.alignn_module, alignn_title)

        # Module 12: Single & Multi-Objective Optimization
        self.multi_objective_optimization = MultiObjectiveOptimizationModule()
        self.tab_widget.addTab(self.multi_objective_optimization, "🔄 Optimization")

        # Module 13: Intelligent Clustering Analysis
        self.intelligent_clustering = IntelligentClusteringModule()
        self.tab_widget.addTab(self.intelligent_clustering, "🎯 Clustering Analysis")

        # Module 14: CGCNN Neural Network
        self.cgcnn_module = CGCNNModule()
        # Show availability status in tab title
        try:
            cgcnn_available = CGCNN_AVAILABLE
        except NameError:
            cgcnn_available = False
        cgcnn_title = "🔮 CGCNN Neural Network" if cgcnn_available else "⚠️ CGCNN (Unavailable)"
        self.tab_widget.addTab(self.cgcnn_module, cgcnn_title)



        # Connect modules
        self.connect_modules()
        
        # Initially disable some modules
        self.set_module_enabled(3, False)  # Feature Selection
        self.set_module_enabled(4, False)  # Model Training
        # self.set_module_enabled(5, False)  # Prediction - Keep enabled for independent use
        self.set_module_enabled(8, False)  # SHAP Analysis
        self.set_module_enabled(9, False)  # Target Optimization
        # Prediction and Multi-Objective Optimization are always enabled (independent modules)
        
    def connect_modules(self):
        """Connect signals between modules"""
        # Data module -> Intelligent Wizard and Advanced Preprocessing
        self.data_module.data_ready.connect(self.intelligent_wizard.set_data)
        self.data_module.data_ready.connect(self.advanced_preprocessing.set_data)
        self.data_module.data_ready.connect(lambda: self.set_module_enabled(3, True))  # Enable Feature Selection
        
        # Intelligent Wizard -> Other modules
        self.intelligent_wizard.configuration_ready.connect(self.apply_wizard_configuration)
        
        # Advanced Preprocessing -> Feature module (with safe wrapper)
        self.advanced_preprocessing.preprocessing_completed.connect(self.safe_set_feature_data)
        
        # Data/Preprocessing -> Feature module
        self.data_module.data_ready.connect(self.feature_module.set_data)
        
        # Data module -> Clustering module (connect raw data)
        self.data_module.data_ready.connect(self.pass_data_to_clustering)
        
        # Feature module -> Training module (with safe wrapper)
        self.feature_module.features_ready.connect(self.safe_set_training_data)
        self.feature_module.features_ready.connect(lambda: self.set_module_enabled(4, True))  # Enable Training
        
        # Training module -> Prediction module (pass model to already-enabled prediction module)
        self.training_module.model_ready.connect(self.prediction_module.set_model)
        # Note: Prediction module is already enabled for independent use
        
        # Training module -> SHAP Analysis and Target Optimization
        self.training_module.model_ready.connect(self.shap_analysis.set_model)
        # Modified to pass training data to target optimization
        self.training_module.model_ready.connect(
            lambda model, feature_names, feature_info, X_train, y_train: 
            self.target_optimization.set_model(model, feature_names, feature_info, X_train)
        )
        self.training_module.model_ready.connect(lambda: self.set_module_enabled(8, True))  # Enable SHAP Analysis
        self.training_module.model_ready.connect(lambda: self.set_module_enabled(9, True))  # Enable Target Optimization
        # Multi-Objective Optimization is independent and doesn't need model_ready signal
        
        # Also pass training data to SHAP analysis (with safe wrapper)
        self.feature_module.features_ready.connect(self.safe_set_shap_data)
        
        # Performance Monitor connections
        self.feature_module.selection_started.connect(
            lambda: self.performance_monitor.start_task("feature_selection", "Feature Selection", 100)
        )
        self.training_module.training_started.connect(
            lambda: self.performance_monitor.start_task("model_training", "Model Training", 100)
        )
        
        # Progress updates
        self.data_module.progress_updated.connect(self.update_progress)
        self.feature_module.progress_updated.connect(self.update_progress)
        self.training_module.progress_updated.connect(self.update_progress)
        self.prediction_module.progress_updated.connect(self.update_progress)
        
        
        # Progress updates to performance monitor
        self.feature_module.progress_updated.connect(
            lambda value: self.performance_monitor.update_task_progress("feature_selection", value)
        )
        self.training_module.progress_updated.connect(
            lambda value: self.performance_monitor.update_task_progress("model_training", value)
        )
        
        # Status updates
        self.data_module.status_updated.connect(self.update_status)
        self.feature_module.status_updated.connect(self.update_status)
        self.training_module.status_updated.connect(self.update_status)
        self.prediction_module.status_updated.connect(self.update_status)
        self.shap_analysis.status_updated.connect(self.update_status)
        self.target_optimization.status_updated.connect(self.update_status)
        self.intelligent_wizard.wizard_completed.connect(lambda: self.update_status("Intelligent configuration applied"))
        self.advanced_preprocessing.preprocessing_completed.connect(lambda: self.update_status("Advanced preprocessing completed"))
        
        # Performance alerts
        self.performance_monitor.performance_alert.connect(self.handle_performance_alert)
    
    def safe_set_feature_data(self, X, y):
        """Safely set feature data with error handling"""
        try:
            print("=== SAFE FEATURE DATA TRANSFER FROM PREPROCESSING ===")
            print(f"Transferring data: X shape {X.shape}, y shape {y.shape}")
            self.feature_module.set_data(X, y)
            print("✓ Feature data transfer from preprocessing successful")
        except Exception as e:
            print(f"ERROR in safe_set_feature_data: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Data Transfer Error", 
                               f"Failed to transfer preprocessed data to feature module:\n{str(e)}\n\n"
                               f"Please try the following:\n"
                               f"1. Check if preprocessing was completed successfully\n"
                               f"2. Restart the application if the issue persists")
            
    def pass_data_to_clustering(self, X, y):
        """Pass data to clustering module"""
        try:
            print("=== DATA TRANSFER TO CLUSTERING MODULE ===")
            print(f"Transferring data: X shape {X.shape}, y shape {y.shape}")
            
            # Convert to DataFrame if it's numpy array
            if hasattr(X, 'shape'):
                import pandas as pd
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                    
            # Set the raw data in clustering module
            self.intelligent_clustering.raw_data = X
            self.intelligent_clustering.current_data = X.copy()
            
            # Update the data preview and populate feature lists
            self.intelligent_clustering.update_data_preview()
            self.intelligent_clustering.populate_feature_lists()
            
            print("✓ Data transfer to clustering module successful")
            
        except Exception as e:
            print(f"ERROR in pass_data_to_clustering: {str(e)}")
            # Non-critical error - clustering module can work independently
            pass
        
    def safe_set_training_data(self, X, y, config):
        """Safely set training data with error handling"""
        try:
            print("=== SAFE TRAINING DATA TRANSFER ===")
            print(f"Transferring data: X shape {X.shape}, y shape {y.shape}")
            self.training_module.set_data(X, y, config)
            print("✓ Training data transfer successful")
        except Exception as e:
            print(f"ERROR in safe_set_training_data: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Data Transfer Error", 
                               f"Failed to transfer data to training module:\n{str(e)}\n\nPlease try again or restart the application.")
    
    def safe_set_shap_data(self, X, y, config):
        """Safely set SHAP data with error handling"""
        try:
            print("=== SAFE SHAP DATA TRANSFER ===")
            print(f"Transferring data to SHAP: X shape {X.shape}, y shape {y.shape}")
            self.shap_analysis.set_data(X, y, config)
            print("✓ SHAP data transfer successful")
        except Exception as e:
            print(f"ERROR in safe_set_shap_data: {str(e)}")
            # Don't show error for SHAP as it's not critical
            print("SHAP data transfer failed, but continuing...")
    
    def set_module_enabled(self, module_index: int, enabled: bool):
        """Enable/disable a module tab"""
        self.tab_widget.setTabEnabled(module_index, enabled)
        
    def update_progress(self, value: int):
        """Update progress bar"""
        if value == 0:
            self.progress_bar.setVisible(False)
        else:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(value)
            if value >= 100:
                self.progress_bar.setVisible(False)
                
    def update_status(self, message: str):
        """Update status bar message"""
        self.status_bar.showMessage(message)
    
    def apply_wizard_configuration(self, config: dict):
        """Apply intelligent wizard configuration"""
        try:
            # Apply feature selection configuration
            if 'feature_selection' in config:
                self.feature_module.apply_wizard_config(config['feature_selection'])
            
            # Apply model configuration
            if 'selected_models' in config:
                self.training_module.apply_wizard_config(config)
            
            self.update_status("Intelligent wizard configuration applied")
            
        except Exception as e:
            self.update_status(f"Failed to apply configuration: {str(e)}")
    
    def handle_performance_alert(self, alert_type: str, message: str):
        """Handle performance alerts"""
        if alert_type == 'cpu_high':
            QMessageBox.warning(self, "Performance Warning", f"High CPU usage!\n{message}\nRecommend pausing compute-intensive tasks.")
        elif alert_type == 'memory_high':
            QMessageBox.warning(self, "Performance Warning", f"High memory usage!\n{message}\nRecommend reducing dataset size or enabling batch processing.")
        elif alert_type == 'disk_full':
            QMessageBox.critical(self, "Storage Warning", f"Low disk space!\n{message}\nPlease free up disk space.")
        
    def new_project(self):
        """Create new project"""
        reply = QMessageBox.question(
            self, "New Project", 
            "This will clear all current work. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reset all modules
            self.data_module.reset()
            self.feature_module.reset()
            self.training_module.reset()
            self.prediction_module.reset()
            self.shap_analysis.reset()
            self.target_optimization.reset()
            
            # Disable modules 3-9 (except Prediction which stays enabled)
            self.set_module_enabled(3, False)  # Feature Selection
            self.set_module_enabled(4, False)  # Model Training
            # self.set_module_enabled(5, False)  # Prediction - Keep enabled for independent use
            self.set_module_enabled(8, False)  # SHAP Analysis
            self.set_module_enabled(9, False)  # Target Optimization
            
            # Switch to first tab
            self.tab_widget.setCurrentIndex(0)
            
            self.update_status("New project created")
            
    def save_project(self):
        """Save project with complete state information"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "AutoMatFlow Projects (*.mml)"
        )
        
        if file_path:
            try:
                # Collect project state from all modules
                project_state = self._collect_project_state()
                
                # Save to JSON format
                import json
                import os
                from datetime import datetime
                
                # Add metadata
                project_state['metadata'] = {
                    'version': '1.0',
                    'created_date': datetime.now().isoformat(),
                    'software_version': 'AutoMatFlow v1.0',
                    'project_name': os.path.splitext(os.path.basename(file_path))[0]
                }
                
                # Write to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(project_state, f, indent=2, ensure_ascii=False)
                
                # Store current project path and add to recent projects
                self.current_project_path = file_path
                self._add_to_recent_projects(file_path)
                self._mark_project_saved()
                
                self.update_status(f"Project saved to {file_path}")
                QMessageBox.information(self, "Success", 
                                      f"Project saved successfully!\n\nLocation: {file_path}\n"
                                      f"Modules saved: {len(project_state.get('modules', {}))}")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Save project error: {error_details}")
                QMessageBox.critical(self, "Error", 
                                   f"Failed to save project: {str(e)}\n\n"
                                   f"Please ensure you have write permissions to the selected location.")
                
    def load_project(self):
        """Load project and restore complete state"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "AutoMatFlow Projects (*.mml)"
        )
        
        if file_path:
            try:
                # Read project file
                import json
                import os
                
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Project file not found: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    project_state = json.load(f)
                
                # Validate project format
                if not isinstance(project_state, dict):
                    raise ValueError("Invalid project file format")
                
                # Check version compatibility
                metadata = project_state.get('metadata', {})
                project_version = metadata.get('version', '1.0')
                
                # Restore project state
                self._restore_project_state(project_state, file_path)
                
                # Store current project path and add to recent projects
                self.current_project_path = file_path
                self._add_to_recent_projects(file_path)
                self._mark_project_saved()
                
                # Update project name in title
                project_name = metadata.get('project_name', os.path.splitext(os.path.basename(file_path))[0])
                self.setWindowTitle(f"AutoMatFlow v1.0 - {project_name}")
                
                self.update_status(f"Project loaded from {file_path}")
                QMessageBox.information(self, "Success", 
                                      f"Project loaded successfully!\n\n"
                                      f"Project: {project_name}\n"
                                      f"Created: {metadata.get('created_date', 'Unknown')}\n"
                                      f"Modules restored: {len(project_state.get('modules', {}))}")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Load project error: {error_details}")
                QMessageBox.critical(self, "Error", 
                                   f"Failed to load project: {str(e)}\n\n"
                                   f"Please ensure the file is a valid AutoMatFlow project file.")
                
    def reset_layout(self):
        """Reset window layout"""
        self.resize(1400, 900)
        self.center_window()
        
    def center_window(self):
        """Center the window on screen"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>AutoMatFlow v1.0</h2>
        This software is developed by Dr. Yu Wang from Sichuan University. Welcome to use! If you have any questions or bugs, please contact the email 1255201958@qq.com
        """
        
        QMessageBox.about(self, "About AutoMatFlow", about_text)
        
        
    def show_user_guide(self):
        """Show user guide"""
        guide_text = """
        <h2>AutoMatFlow User Guide</h2>
        
        <h3>Module 1: Data & Preprocessing</h3>
        <p>Import your materials data from CSV or Excel files. Explore data quality, 
        handle missing values, and prepare features and target variables.</p>
        
        <h3>Module 2: Feature Selection</h3>
        <p>Select the best features using multiple strategies: importance-based filtering,
        correlation analysis, and advanced wrapper methods.</p>
        
        <h3>Module 3: Model Training</h3>
        <p>Train machine learning models with hyperparameter optimization.
        Evaluate model performance with comprehensive metrics and visualizations.</p>
        
        <h3>Module 4: Prediction</h3>
        <p>Apply trained models to new data and export predictions.</p>
        
        <p><b>Workflow:</b> Follow the modules in order for the best experience.
        Each module builds upon the previous one.</p>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("User Guide")
        msg.setText(guide_text)
        msg.setTextFormat(Qt.RichText)
        msg.exec_()
        
    def closeEvent(self, event):
        """Handle application close event"""
        reply = QMessageBox.question(self, 'Exit Application', 
                                   'Are you sure you want to exit?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Save current state if needed
            try:
                # Add any cleanup code here
                pass
            except Exception as e:
                print(f"Error during cleanup: {e}")
            event.accept()
        else:
            event.ignore()

    # Enhanced Menu Action Methods
    def new_experiment(self):
        """Create a new experiment"""
        try:
            from PyQt5.QtWidgets import QInputDialog
            experiment_name, ok = QInputDialog.getText(self, 'New Experiment', 'Enter experiment name:')
            if ok and experiment_name:
                self.update_status(f"Created new experiment: {experiment_name}")
                QMessageBox.information(self, "New Experiment", f"Experiment '{experiment_name}' created successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create experiment: {str(e)}")
    
    def import_training_data(self):
        """Import training data"""
        try:
            self.tab_widget.setCurrentIndex(0)  # Switch to Data Management tab
            self.data_module.load_training_data()
            self.update_status("Training data import initiated")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to import training data: {str(e)}")
    
    def import_virtual_data(self):
        """Import virtual screening data"""
        try:
            self.tab_widget.setCurrentIndex(0)  # Switch to Data Management tab
            self.data_module.load_virtual_data()
            self.update_status("Virtual data import initiated")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to import virtual data: {str(e)}")
    
    def import_model(self):
        """Import pre-trained model"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Import Model", "", "Joblib Files (*.joblib)")
            if file_path:
                # Load model and set it to prediction module
                import joblib
                model = joblib.load(file_path)
                self.prediction_module.set_model(model)
                self.update_status(f"Model imported from {file_path}")
                QMessageBox.information(self, "Import Model", "Model imported successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to import model: {str(e)}")
    
    def export_results(self):
        """Export analysis results"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'export_results'):
                current_tab.export_results()
                self.update_status("Results exported successfully")
            else:
                QMessageBox.information(self, "Export Results", "No results available to export from current tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export results: {str(e)}")
    
    def export_charts(self):
        """Export charts and visualizations"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'export_charts'):
                current_tab.export_charts()
                self.update_status("Charts exported successfully")
            else:
                QMessageBox.information(self, "Export Charts", "No charts available to export from current tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export charts: {str(e)}")
    
    def export_model(self):
        """Export trained model"""
        try:
            if hasattr(self.training_module, 'model') and self.training_module.model is not None:
                file_path, _ = QFileDialog.getSaveFileName(self, "Export Model", "", "Joblib Files (*.joblib)")
                if file_path:
                    import joblib
                    joblib.dump(self.training_module.model, file_path)
                    self.update_status(f"Model exported to {file_path}")
                    QMessageBox.information(self, "Export Model", "Model exported successfully!")
            else:
                QMessageBox.information(self, "Export Model", "No trained model available to export")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export model: {str(e)}")
    
    def export_report(self):
        """Export optimization report"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'export_report'):
                current_tab.export_report()
                self.update_status("Report exported successfully")
            else:
                QMessageBox.information(self, "Export Report", "No report available to export from current tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export report: {str(e)}")
    
    def update_recent_projects_menu(self, menu):
        """Update recent projects menu with actual recent project files"""
        try:
            menu.clear()
            
            # Load recent projects from settings
            self._load_recent_projects()
            
            if self.recent_projects:
                for project_path in self.recent_projects[:10]:  # Show max 10 recent projects
                    project_name = os.path.basename(project_path)
                    action = QAction(project_name, self)
                    action.setToolTip(project_path)  # Show full path as tooltip
                    action.triggered.connect(lambda checked, p=project_path: self.load_recent_project(p))
                    menu.addAction(action)
                
                menu.addSeparator()
                clear_action = QAction("Clear Recent Projects", self)
                clear_action.triggered.connect(self.clear_recent_projects)
                menu.addAction(clear_action)
            else:
                no_projects_action = QAction("No Recent Projects", self)
                no_projects_action.setEnabled(False)
                menu.addAction(no_projects_action)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update recent projects: {str(e)}")
    
    def load_recent_project(self, project_path):
        """Load a recent project from file path"""
        try:
            import json
            import os
            
            if os.path.exists(project_path):
                # Use the existing load_project logic
                with open(project_path, 'r', encoding='utf-8') as f:
                    project_state = json.load(f)
                
                self._restore_project_state(project_state, project_path)
                self.current_project_path = project_path
                self._add_to_recent_projects(project_path)
                
                project_name = os.path.splitext(os.path.basename(project_path))[0]
                self.update_status(f"Loaded recent project: {project_name}")
                QMessageBox.information(self, "Load Project", f"Project '{project_name}' loaded successfully!")
            else:
                QMessageBox.warning(self, "Project Not Found", 
                                  f"The project file could not be found:\n{project_path}\n\n"
                                  f"It may have been moved or deleted.")
                self._remove_from_recent_projects(project_path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load recent project: {str(e)}")
    
    def clear_recent_projects(self):
        """Clear recent projects list"""
        try:
            reply = QMessageBox.question(self, 'Clear Recent Projects', 
                                       'Are you sure you want to clear the recent projects list?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.recent_projects.clear()
                self._save_recent_projects()
                self.update_status("Recent projects list cleared")
                QMessageBox.information(self, "Clear Recent Projects", "Recent projects list cleared!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to clear recent projects: {str(e)}")
    
    def show_settings(self):
        """Show application settings dialog"""
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox, QPushButton, QTabWidget, QComboBox
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Settings")
            dialog.setModal(True)
            dialog.resize(500, 400)
            
            layout = QVBoxLayout(dialog)
            
            # Create tab widget for settings categories
            tab_widget = QTabWidget()
            layout.addWidget(tab_widget)
            
            # General settings tab
            general_tab = QWidget()
            general_layout = QVBoxLayout(general_tab)
            
            # Auto-save settings
            auto_save_cb = QCheckBox("Enable auto-save")
            auto_save_cb.setChecked(True)
            general_layout.addWidget(auto_save_cb)
            
            # Default paths
            general_layout.addWidget(QLabel("Default Data Path:"))
            # Add path selection widget here
            
            tab_widget.addTab(general_tab, "General")
            
            # Appearance settings tab
            appearance_tab = QWidget()
            appearance_layout = QVBoxLayout(appearance_tab)
            
            # Theme selection
            appearance_layout.addWidget(QLabel("Theme:"))
            theme_combo = QComboBox()
            theme_combo.addItems(["Light", "Dark", "Auto"])
            appearance_layout.addWidget(theme_combo)
            
            # Font size
            font_layout = QHBoxLayout()
            font_layout.addWidget(QLabel("Font Size:"))
            font_spin = QSpinBox()
            font_spin.setRange(8, 16)
            font_spin.setValue(9)
            font_layout.addWidget(font_spin)
            appearance_layout.addLayout(font_layout)
            
            tab_widget.addTab(appearance_tab, "Appearance")
            
            # Button layout
            button_layout = QHBoxLayout()
            ok_button = QPushButton("OK")
            cancel_button = QPushButton("Cancel")
            
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            button_layout.addStretch()
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            
            layout.addLayout(button_layout)
            
            if dialog.exec_() == QDialog.Accepted:
                self.update_status("Settings updated")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show settings: {str(e)}")
    
    # Edit menu methods
    def undo_action(self):
        """Undo last action"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'undo'):
                current_tab.undo()
                self.update_status("Action undone")
            else:
                QMessageBox.information(self, "Undo", "No action to undo in current tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to undo: {str(e)}")
    
    def redo_action(self):
        """Redo last undone action"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'redo'):
                current_tab.redo()
                self.update_status("Action redone")
            else:
                QMessageBox.information(self, "Redo", "No action to redo in current tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to redo: {str(e)}")
    
    def copy_data(self):
        """Copy selected data"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'copy_selection'):
                current_tab.copy_selection()
                self.update_status("Data copied to clipboard")
            else:
                QMessageBox.information(self, "Copy", "No data selected to copy")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to copy data: {str(e)}")
    
    def paste_data(self):
        """Paste data from clipboard"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'paste_data'):
                current_tab.paste_data()
                self.update_status("Data pasted from clipboard")
            else:
                QMessageBox.information(self, "Paste", "Current tab doesn't support paste operation")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to paste data: {str(e)}")
    
    def show_find_dialog(self):
        """Show find dialog"""
        try:
            from PyQt5.QtWidgets import QInputDialog
            search_text, ok = QInputDialog.getText(self, 'Find', 'Enter search text:')
            if ok and search_text:
                current_tab = self.tab_widget.currentWidget()
                if hasattr(current_tab, 'find_text'):
                    current_tab.find_text(search_text)
                    self.update_status(f"Searching for: {search_text}")
                else:
                    QMessageBox.information(self, "Find", "Current tab doesn't support text search")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show find dialog: {str(e)}")
    
    # Navigation methods for switching between modules
    def switch_to_data_module(self):
        """Switch to data management module"""
        self.tab_widget.setCurrentIndex(0)
        self.update_status("Switched to Data Management")
    
    def switch_to_preprocessing_module(self):
        """Switch to preprocessing module"""
        self.tab_widget.setCurrentIndex(2)  # Advanced Preprocessing tab
        self.update_status("Switched to Advanced Preprocessing")
    
    def switch_to_feature_module(self):
        """Switch to feature engineering module"""
        self.tab_widget.setCurrentIndex(3)
        self.update_status("Switched to Feature Selection")
    
    def switch_to_training_module(self):
        """Switch to model training module"""
        self.tab_widget.setCurrentIndex(4)
        self.update_status("Switched to Model Training")
    
    def switch_to_shap_module(self):
        """Switch to SHAP analysis module"""
        self.tab_widget.setCurrentIndex(8)
        self.update_status("Switched to SHAP Analysis")
    
    def switch_to_performance_module(self):
        """Switch to performance monitor module"""
        self.tab_widget.setCurrentIndex(6)
        self.update_status("Switched to Performance Monitor")
    
    
    
    def switch_to_multi_objective_module(self):
        """Switch to multi-objective optimization module"""
        self.tab_widget.setCurrentIndex(10)
        self.update_status("Switched to Multi-objective Optimization")
    
    def switch_to_prediction_module(self):
        """Switch to prediction module"""
        self.tab_widget.setCurrentIndex(5)
        self.update_status("Switched to Model Prediction")
    
    def switch_to_target_optimization_module(self):
        """Switch to target optimization module"""
        self.tab_widget.setCurrentIndex(9)
        self.update_status("Switched to Target Optimization")
    
    
    # Analysis and visualization methods
    def show_data_visualization(self):
        """Show data visualization dialog"""
        try:
            if hasattr(self.data_module, 'show_visualization_dialog'):
                self.data_module.show_visualization_dialog()
            else:
                QMessageBox.information(self, "Data Visualization", "Please load data first in the Data Management tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show data visualization: {str(e)}")
    
    def show_data_summary(self):
        """Show data summary dialog"""
        try:
            if hasattr(self.data_module, 'show_data_summary'):
                self.data_module.show_data_summary()
            else:
                QMessageBox.information(self, "Data Summary", "Please load data first in the Data Management tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show data summary: {str(e)}")
    
    def show_model_evaluation(self):
        """Show model evaluation dialog"""
        try:
            if hasattr(self.training_module, 'show_evaluation_dialog'):
                self.training_module.show_evaluation_dialog()
            else:
                QMessageBox.information(self, "Model Evaluation", "Please train a model first in the Model Training tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show model evaluation: {str(e)}")
    
    def show_learning_curve_analysis(self):
        """Show learning curve analysis"""
        try:
            if hasattr(self.training_module, 'show_learning_curves'):
                self.training_module.show_learning_curves()
            else:
                QMessageBox.information(self, "Learning Curve Analysis", "Please train a model first in the Model Training tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show learning curve analysis: {str(e)}")
    
    # View menu methods
    def toggle_log_console(self, checked):
        """Toggle log console visibility"""
        try:
            # Implementation would show/hide log console panel
            self.update_status(f"Log console {'shown' if checked else 'hidden'}")
        except Exception as e:
            print(f"Error toggling log console: {e}")
    
    def toggle_data_viewer(self, checked):
        """Toggle data viewer panel"""
        try:
            # Implementation would show/hide data viewer panel
            self.update_status(f"Data viewer {'shown' if checked else 'hidden'}")
        except Exception as e:
            print(f"Error toggling data viewer: {e}")
    
    def toggle_charts_panel(self, checked):
        """Toggle charts panel"""
        try:
            # Implementation would show/hide charts panel
            self.update_status(f"Charts panel {'shown' if checked else 'hidden'}")
        except Exception as e:
            print(f"Error toggling charts panel: {e}")
    
    def toggle_theme(self):
        """Toggle between light and dark theme"""
        try:
            # Simple theme toggle implementation
            current_style = self.styleSheet()
            if "background-color: #2b2b2b" in current_style:
                # Switch to light theme
                self.setStyleSheet("")
                self.update_status("Switched to light theme")
            else:
                # Switch to dark theme
                dark_style = """
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QTabWidget::pane {
                    background-color: #3c3c3c;
                    border: 1px solid #555555;
                }
                QTabBar::tab {
                    background-color: #404040;
                    color: #ffffff;
                    padding: 8px 16px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #555555;
                }
                """
                self.setStyleSheet(dark_style)
                self.update_status("Switched to dark theme")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle theme: {str(e)}")
    
    def set_font_size(self, size):
        """Set application font size"""
        try:
            font = QFont("Segoe UI", size)
            self.setFont(font)
            self.update_status(f"Font size set to {size}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to set font size: {str(e)}")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        try:
            if self.isFullScreen():
                self.showNormal()
                self.update_status("Exited fullscreen mode")
            else:
                self.showFullScreen()
                self.update_status("Entered fullscreen mode")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle fullscreen: {str(e)}")
    
    # Help menu methods
    def show_welcome_guide(self):
        """Show welcome guide dialog"""
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Welcome to AutoMatFlow")
            dialog.setModal(True)
            dialog.resize(600, 500)
            
            layout = QVBoxLayout(dialog)
            
            # Welcome content
            welcome_text = """
            <h2>Welcome to AutoMatFlow v1.0</h2>
            
            <h3>Getting Started:</h3>
            <ol>
                <li><b>Load Data:</b> Start by importing your training data in the Data Management tab</li>
                <li><b>Preprocess:</b> Use Advanced Preprocessing to clean and prepare your data</li>
                <li><b>Feature Engineering:</b> Select and engineer features in the Feature Selection tab</li>
                <li><b>Train Models:</b> Build and evaluate models in the Model Training tab</li>
                <li><b>Optimize:</b> Use Multi-objective Optimization for advanced analysis</li>
            </ol>
            
            <h3>Key Features:</h3>
            <ul>
                <li>🧙‍♂️ Intelligent Wizard for automated workflow guidance</li>
                <li>🧬 Advanced preprocessing with multiple algorithms</li>
                <li>🎯 Comprehensive feature engineering and selection</li>
                <li>🔬 Multiple machine learning algorithms</li>
                <li>🧠 SHAP analysis for model interpretability</li>
                <li>🔄 Multi-objective optimization</li>
                <li>📊 Rich visualizations and performance monitoring</li>
            </ul>
            
            <h3>Navigation:</h3>
            <p>Use the menu bar to quickly access different functions:</p>
            <ul>
                <li><b>File:</b> Project management and data import/export</li>
                <li><b>Data:</b> Data-related operations</li>
                <li><b>Analyze:</b> Model training and analysis</li>
                <li><b>Optimize:</b> Advanced optimization techniques</li>
                <li><b>View:</b> Interface customization</li>
            </ul>
            """
            
            text_browser = QTextBrowser()
            text_browser.setHtml(welcome_text)
            layout.addWidget(text_browser)
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show welcome guide: {str(e)}")
    
    def show_online_documentation(self):
        """Open online documentation"""
        try:
            import webbrowser
            webbrowser.open("https://github.com/wy314159-lyq/ML-stdio.git")
            self.update_status("Opened online documentation")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open online documentation: {str(e)}")
    
    def show_logs(self):
        """Show application logs"""
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Application Logs")
            dialog.setModal(True)
            dialog.resize(700, 500)
            
            layout = QVBoxLayout(dialog)
            
            # Log content (in a real application, this would read from log files)
            log_text = QTextEdit()
            log_text.setReadOnly(True)
            log_text.setPlainText("""
[2024-01-01 10:00:00] INFO: Application started
[2024-01-01 10:00:01] INFO: Modules loaded successfully
[2024-01-01 10:00:02] INFO: UI initialized
[2024-01-01 10:05:30] INFO: Training data loaded: 1000 samples
[2024-01-01 10:06:15] INFO: Model training completed
[2024-01-01 10:07:00] INFO: Analysis results exported
            """)
            layout.addWidget(log_text)
            
            # Button layout
            button_layout = QHBoxLayout()
            refresh_button = QPushButton("Refresh")
            clear_button = QPushButton("Clear Logs")
            close_button = QPushButton("Close")
            
            refresh_button.clicked.connect(lambda: self.update_status("Logs refreshed"))
            clear_button.clicked.connect(lambda: log_text.clear())
            close_button.clicked.connect(dialog.accept)
            
            button_layout.addWidget(refresh_button)
            button_layout.addWidget(clear_button)
            button_layout.addStretch()
            button_layout.addWidget(close_button)
            
            layout.addLayout(button_layout)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show logs: {str(e)}")
    
    def report_issue(self):
        """Open issue reporting interface"""
        try:
            import webbrowser
            webbrowser.open("https://github.com/your-repo/automatflow-studio/issues/new")
            self.update_status("Opened issue reporting page")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open issue reporting: {str(e)}")

    # ========== PROJECT STATE MANAGEMENT ==========
    
    def _collect_project_state(self):
        """Collect comprehensive project state from all modules"""
        project_state = {
            'modules': {},
            'ui_state': {},
            'settings': {}
        }
        
        try:
            # Data Module State
            if hasattr(self, 'data_module') and self.data_module:
                data_state = {}
                if hasattr(self.data_module, 'file_path_edit') and self.data_module.file_path_edit.text():
                    data_state['data_file_path'] = self.data_module.file_path_edit.text()
                if hasattr(self.data_module, 'selected_target') and self.data_module.selected_target:
                    data_state['selected_target'] = self.data_module.selected_target
                if hasattr(self.data_module, 'selected_features') and self.data_module.selected_features:
                    data_state['selected_features'] = self.data_module.selected_features
                if hasattr(self.data_module, 'encoding_applied'):
                    data_state['encoding_applied'] = self.data_module.encoding_applied
                
                # Save current data info if available
                if hasattr(self.data_module, 'df') and self.data_module.df is not None:
                    data_state['data_shape'] = list(self.data_module.df.shape)
                    data_state['column_names'] = list(self.data_module.df.columns)
                    data_state['data_types'] = {col: str(dtype) for col, dtype in self.data_module.df.dtypes.items()}
                
                project_state['modules']['data_module'] = data_state
            
            # Feature Module State
            if hasattr(self, 'feature_module') and self.feature_module:
                feature_state = {}
                if hasattr(self.feature_module, 'selected_features') and self.feature_module.selected_features:
                    feature_state['selected_features'] = self.feature_module.selected_features
                if hasattr(self.feature_module, 'feature_importance_results'):
                    feature_state['has_feature_analysis'] = True
                
                project_state['modules']['feature_module'] = feature_state
            
            # Training Module State
            if hasattr(self, 'training_module') and self.training_module:
                training_state = {}
                if hasattr(self.training_module, 'trained_pipeline') and self.training_module.trained_pipeline:
                    # Save model to temporary location and store path
                    import tempfile
                    import joblib
                    import os
                    import time
                    
                    temp_dir = tempfile.gettempdir()
                    model_filename = f"automat_model_{int(time.time())}.joblib"
                    model_path = os.path.join(temp_dir, model_filename)
                    
                    try:
                        joblib.dump({
                            'model': self.training_module.trained_pipeline,
                            'feature_names': getattr(self.training_module, 'feature_names', []),
                            'task_type': getattr(self.training_module, 'task_type', 'regression')
                        }, model_path)
                        training_state['model_path'] = model_path
                        training_state['has_trained_model'] = True
                    except Exception as e:
                        print(f"Warning: Could not save model state: {e}")
                        training_state['has_trained_model'] = False
                
                if hasattr(self.training_module, 'task_type'):
                    training_state['task_type'] = self.training_module.task_type
                if hasattr(self.training_module, 'model_combo') and self.training_module.model_combo.currentText():
                    training_state['selected_model'] = self.training_module.model_combo.currentText()
                
                project_state['modules']['training_module'] = training_state
            
            # Prediction Module State
            if hasattr(self, 'prediction_module') and self.prediction_module:
                prediction_state = {}
                if hasattr(self.prediction_module, 'model_path_edit') and self.prediction_module.model_path_edit.text():
                    prediction_state['model_path'] = self.prediction_module.model_path_edit.text()
                if hasattr(self.prediction_module, 'data_path_edit') and self.prediction_module.data_path_edit.text():
                    prediction_state['prediction_data_path'] = self.prediction_module.data_path_edit.text()
                
                project_state['modules']['prediction_module'] = prediction_state
            
            
            # UI State
            project_state['ui_state'] = {
                'current_tab_index': self.tab_widget.currentIndex(),
                'window_geometry': {
                    'width': self.width(),
                    'height': self.height(),
                    'x': self.x(),
                    'y': self.y()
                },
                'enabled_tabs': [self.tab_widget.isTabEnabled(i) for i in range(self.tab_widget.count())]
            }
            
            return project_state
            
        except Exception as e:
            print(f"Error collecting project state: {e}")
            import traceback
            traceback.print_exc()
            return {'modules': {}, 'ui_state': {}, 'settings': {}}
    
    def _restore_project_state(self, project_state, project_file_path):
        """Restore project state to all modules"""
        import time
        import os
        
        try:
            modules_state = project_state.get('modules', {})
            ui_state = project_state.get('ui_state', {})
            
            # Start with data module restoration
            if 'data_module' in modules_state:
                self._restore_data_module_state(modules_state['data_module'], project_file_path)
                # Allow time for data loading
                QApplication.processEvents()
                time.sleep(0.5)
            
            # Restore feature module state
            if 'feature_module' in modules_state:
                self._restore_feature_module_state(modules_state['feature_module'])
                QApplication.processEvents()
            
            # Restore training module state
            if 'training_module' in modules_state:
                self._restore_training_module_state(modules_state['training_module'])
                QApplication.processEvents()
            
            # Restore prediction module state
            if 'prediction_module' in modules_state:
                self._restore_prediction_module_state(modules_state['prediction_module'])
                QApplication.processEvents()
            
            # Restore UI state
            if ui_state:
                self._restore_ui_state(ui_state)
            
            self.update_status("Project state restored successfully")
            
        except Exception as e:
            print(f"Error restoring project state: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Partial Load", 
                              f"Project loaded with some limitations: {str(e)}\n\n"
                              f"Some modules may need to be reconfigured manually.")
    
    def _restore_data_module_state(self, data_state, project_file_path):
        """Restore data module state"""
        try:
            if 'data_file_path' in data_state and os.path.exists(data_state['data_file_path']):
                # Restore data file path and load data
                self.data_module.file_path_edit.setText(data_state['data_file_path'])
                self.data_module.import_data()
                
                # Restore target and feature selections after data loads
                if 'selected_target' in data_state:
                    # Find and set target in combo box
                    target_combo = getattr(self.data_module, 'target_combo', None)
                    if target_combo:
                        index = target_combo.findText(data_state['selected_target'])
                        if index >= 0:
                            target_combo.setCurrentIndex(index)
                            self.data_module.selected_target = data_state['selected_target']
                
                print(f"✓ Data module state restored: {data_state.get('data_file_path', 'N/A')}")
            else:
                print(f"⚠ Data file not found: {data_state.get('data_file_path', 'N/A')}")
                
        except Exception as e:
            print(f"Error restoring data module: {e}")
    
    def _restore_feature_module_state(self, feature_state):
        """Restore feature module state"""
        try:
            if 'selected_features' in feature_state:
                # This will be handled when data flows through the pipeline
                print(f"✓ Feature selections available for restoration")
        except Exception as e:
            print(f"Error restoring feature module: {e}")
    
    def _restore_training_module_state(self, training_state):
        """Restore training module state"""
        try:
            if 'model_path' in training_state and os.path.exists(training_state['model_path']):
                # Load saved model
                import joblib
                model_data = joblib.load(training_state['model_path'])
                
                if hasattr(self.training_module, 'trained_pipeline'):
                    self.training_module.trained_pipeline = model_data.get('model')
                    
                # Set task type if available
                if 'task_type' in training_state:
                    if hasattr(self.training_module, 'task_type_combo'):
                        self.training_module.task_type_combo.setCurrentText(training_state['task_type'].title())
                
                # Set selected model if available
                if 'selected_model' in training_state:
                    if hasattr(self.training_module, 'model_combo'):
                        index = self.training_module.model_combo.findText(training_state['selected_model'])
                        if index >= 0:
                            self.training_module.model_combo.setCurrentIndex(index)
                
                print(f"✓ Training module state restored")
                
        except Exception as e:
            print(f"Error restoring training module: {e}")
    
    def _restore_prediction_module_state(self, prediction_state):
        """Restore prediction module state"""
        try:
            if 'model_path' in prediction_state and os.path.exists(prediction_state['model_path']):
                self.prediction_module.model_path_edit.setText(prediction_state['model_path'])
                
            if 'prediction_data_path' in prediction_state and os.path.exists(prediction_state['prediction_data_path']):
                self.prediction_module.data_path_edit.setText(prediction_state['prediction_data_path'])
                
            print(f"✓ Prediction module state restored")
            
        except Exception as e:
            print(f"Error restoring prediction module: {e}")
    
    def _restore_ui_state(self, ui_state):
        """Restore UI state"""
        try:
            # Restore window geometry
            if 'window_geometry' in ui_state:
                geom = ui_state['window_geometry']
                self.setGeometry(geom.get('x', 100), geom.get('y', 100), 
                               geom.get('width', 1400), geom.get('height', 900))
            
            # Restore enabled tabs
            if 'enabled_tabs' in ui_state:
                enabled_tabs = ui_state['enabled_tabs']
                for i, enabled in enumerate(enabled_tabs):
                    if i < self.tab_widget.count():
                        self.tab_widget.setTabEnabled(i, enabled)
            
            # Restore current tab (do this last)
            if 'current_tab_index' in ui_state:
                tab_index = ui_state['current_tab_index']
                if 0 <= tab_index < self.tab_widget.count():
                    self.tab_widget.setCurrentIndex(tab_index)
            
            print(f"✓ UI state restored")
            
        except Exception as e:
            print(f"Error restoring UI state: {e}")

    # ========== RECENT PROJECTS MANAGEMENT ==========
    
    def _add_to_recent_projects(self, project_path):
        """Add a project to the recent projects list"""
        try:
            import os
            
            # Convert to absolute path
            project_path = os.path.abspath(project_path)
            
            # Remove if already exists
            if project_path in self.recent_projects:
                self.recent_projects.remove(project_path)
            
            # Add to beginning of list
            self.recent_projects.insert(0, project_path)
            
            # Keep only the last 10 projects
            self.recent_projects = self.recent_projects[:10]
            
            # Save to persistent storage
            self._save_recent_projects()
            
        except Exception as e:
            print(f"Error adding to recent projects: {e}")
    
    def _remove_from_recent_projects(self, project_path):
        """Remove a project from the recent projects list"""
        try:
            if project_path in self.recent_projects:
                self.recent_projects.remove(project_path)
                self._save_recent_projects()
        except Exception as e:
            print(f"Error removing from recent projects: {e}")
    
    def _load_recent_projects(self):
        """Load recent projects from settings file"""
        try:
            import json
            import os
            
            settings_dir = os.path.expanduser("~/.automatflow")
            settings_file = os.path.join(settings_dir, "recent_projects.json")
            
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.recent_projects = data.get('recent_projects', [])
                    
                    # Filter out non-existent files
                    self.recent_projects = [p for p in self.recent_projects if os.path.exists(p)]
            else:
                self.recent_projects = []
                
        except Exception as e:
            print(f"Error loading recent projects: {e}")
            self.recent_projects = []
    
    def _save_recent_projects(self):
        """Save recent projects to settings file"""
        try:
            import json
            import os
            
            settings_dir = os.path.expanduser("~/.automatflow")
            if not os.path.exists(settings_dir):
                os.makedirs(settings_dir)
            
            settings_file = os.path.join(settings_dir, "recent_projects.json")
            
            data = {
                'recent_projects': self.recent_projects,
                'auto_save_enabled': self.auto_save_enabled
            }
            
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving recent projects: {e}")

    # ========== AUTO-SAVE FUNCTIONALITY ==========
    
    def _mark_project_modified(self):
        """Mark the current project as modified"""
        self.project_modified = True
        
        # Update window title to show modification
        if self.current_project_path:
            project_name = os.path.splitext(os.path.basename(self.current_project_path))[0]
            self.setWindowTitle(f"MatSci-ML Studio v1.0 - {project_name} *")
        else:
            self.setWindowTitle("MatSci-ML Studio v1.0 - Untitled Project *")
    
    def _mark_project_saved(self):
        """Mark the current project as saved"""
        self.project_modified = False
        
        # Update window title to remove modification indicator
        if self.current_project_path:
            project_name = os.path.splitext(os.path.basename(self.current_project_path))[0]
            self.setWindowTitle(f"MatSci-ML Studio v1.0 - {project_name}")
        else:
            self.setWindowTitle("MatSci-ML Studio v1.0")
    
    def _auto_save_project(self):
        """Automatically save the project if auto-save is enabled"""
        try:
            if self.auto_save_enabled and self.project_modified and self.current_project_path:
                # Perform auto-save without user interaction
                project_state = self._collect_project_state()
                
                # Add metadata
                from datetime import datetime
                project_state['metadata'] = {
                    'version': '1.0',
                    'last_modified': datetime.now().isoformat(),
                    'software_version': 'MatSci-ML Studio v1.0',
                    'project_name': os.path.splitext(os.path.basename(self.current_project_path))[0],
                    'auto_saved': True
                }
                
                # Create backup file
                backup_path = self.current_project_path + '.backup'
                import shutil
                if os.path.exists(self.current_project_path):
                    shutil.copy2(self.current_project_path, backup_path)
                
                # Save to file
                with open(self.current_project_path, 'w', encoding='utf-8') as f:
                    json.dump(project_state, f, indent=2, ensure_ascii=False)
                
                self._mark_project_saved()
                self.update_status("Project auto-saved")
                
        except Exception as e:
            print(f"Auto-save failed: {e}")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("MatSci-ML Studio")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MatSciMLStudioWindow()
    window.show()
    window.center_window()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 