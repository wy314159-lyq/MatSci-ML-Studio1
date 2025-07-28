import sys
import os
import pandas as pd
import numpy as np
import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

# Add the modules directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.active_learning_optimizer import ActiveLearningOptimizer


class ActiveLearningSession:
    """
    Manages the state of an iterative active learning session, including
    the original data, newly acquired points, and iteration history.
    """
    
    def __init__(self, original_training_df):
        """
        Initialize the session with original training data.
        
        Parameters:
        -----------
        original_training_df : pandas.DataFrame
            The initial training dataset
        """
        self.original_training_df = original_training_df.copy()
        self.newly_acquired_points = pd.DataFrame(columns=original_training_df.columns)
        self.iteration_history = []
        self.current_iteration = 0
        
        # Performance tracking for visualization
        self.performance_history = []  # Model performance metrics over iterations
        self.feature_importance_history = []  # Feature importance evolution
        self.best_values_history = []  # Best values found in each iteration
        self.uncertainty_history = []  # Uncertainty metrics over iterations
        self.acquisition_history = []  # Acquisition function values over iterations
    
    def add_new_point(self, features, targets):
        """
        Add a new, validated data point to the session.
        
        Parameters:
        -----------
        features : pd.Series or dict
            The feature values of the new point
        targets : dict
            A dictionary mapping target column names to their measured values
        """
        # Convert features to series if it's a dict
        if isinstance(features, dict):
            features = pd.Series(features)
        
        # Create new point data by combining features and targets
        new_point_data = features.copy()
        for target_name, target_value in targets.items():
            new_point_data[target_name] = target_value
        
        # Add to newly acquired points
        self.newly_acquired_points = pd.concat(
            [self.newly_acquired_points, new_point_data.to_frame().T],
            ignore_index=True
        )
        
        # Add to iteration history with metadata
        history_entry = {
            'iteration': self.current_iteration + 1,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': new_point_data.to_dict()
        }
        self.iteration_history.append(history_entry)
        self.current_iteration += 1
    
    def get_combined_training_data(self):
        """
        Return the full, up-to-date training dataset by combining
        the original data with all newly acquired points.
        
        Returns:
        --------
        pandas.DataFrame
            Combined training data
        """
        if self.newly_acquired_points.empty:
            return self.original_training_df.copy()
        
        return pd.concat([self.original_training_df, self.newly_acquired_points], 
                        ignore_index=True)
    
    def get_newly_acquired_data(self):
        """
        Return the DataFrame of newly acquired points.
        
        Returns:
        --------
        pandas.DataFrame
            Newly acquired data points
        """
        return self.newly_acquired_points.copy()
    
    def get_iteration_count(self):
        """
        Get the current iteration number.
        
        Returns:
        --------
        int
            Current iteration count
        """
        return self.current_iteration
    
    def get_session_summary(self):
        """
        Get a summary of the current session.
        
        Returns:
        --------
        dict
            Session summary statistics
        """
        return {
            'original_data_points': len(self.original_training_df),
            'newly_acquired_points': len(self.newly_acquired_points),
            'total_data_points': len(self.get_combined_training_data()),
            'iterations_completed': self.current_iteration,
            'session_start': self.iteration_history[0]['timestamp'] if self.iteration_history else None,
            'last_update': self.iteration_history[-1]['timestamp'] if self.iteration_history else None
        }
    
    def add_performance_record(self, model_performance, feature_importance, best_value, 
                             uncertainty_metrics, acquisition_stats):
        """
        Add performance metrics for the current iteration.
        
        Parameters:
        -----------
        model_performance : dict
            Model performance metrics (R2, RMSE, etc.)
        feature_importance : dict or pd.Series
            Feature importance values
        best_value : float
            Best target value found so far
        uncertainty_metrics : dict
            Uncertainty statistics (mean, std, etc.)
        acquisition_stats : dict
            Acquisition function statistics
        """
        self.performance_history.append({
            'iteration': self.current_iteration,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_performance': model_performance,
            'best_value': best_value,
            'uncertainty_metrics': uncertainty_metrics,
            'acquisition_stats': acquisition_stats
        })
        
        self.feature_importance_history.append({
            'iteration': self.current_iteration,
            'importance': feature_importance
        })
    
    def get_performance_history(self):
        """
        Get the complete performance history for visualization.
        
        Returns:
        --------
        list
            List of performance records
        """
        return self.performance_history
    
    def get_feature_importance_evolution(self):
        """
        Get feature importance evolution over iterations.
        
        Returns:
        --------
        list
            List of feature importance records
        """
        return self.feature_importance_history
    
    def reset(self):
        """
        Reset the session by clearing all acquired points and history.
        """
        self.newly_acquired_points = pd.DataFrame(columns=self.original_training_df.columns)
        self.iteration_history = []
        self.current_iteration = 0
        self.performance_history = []
        self.feature_importance_history = []
        self.best_values_history = []
        self.uncertainty_history = []
        self.acquisition_history = []


class AnalysisWorker(QObject):
    """
    Worker class for performing analysis in a separate thread to prevent UI freezing.
    """
    # Signals for communication with main thread
    finished = pyqtSignal(object)  # Emits results_data when analysis completes
    error = pyqtSignal(str)        # Emits error message when analysis fails
    progress = pyqtSignal(str)     # Emits progress messages during analysis
    
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.analysis_params = {}
        self.should_stop = False
    
    def setup_analysis(self, training_data, virtual_data, target_columns, feature_columns, 
                      goal_directions, is_multi_objective, model_type, acquisition_function, 
                      acquisition_config, n_iterations):
        """
        Setup analysis parameters.
        """
        self.analysis_params = {
            'training_data': training_data,
            'virtual_data': virtual_data,
            'target_columns': target_columns,
            'feature_columns': feature_columns,
            'goal_directions': goal_directions,
            'is_multi_objective': is_multi_objective,
            'model_type': model_type,
            'acquisition_function': acquisition_function,
            'acquisition_config': acquisition_config,
            'n_iterations': n_iterations
        }
    
    def run_analysis(self):
        """
        Execute the analysis in the worker thread.
        """
        try:
            # Check if stopped before starting
            if self.should_stop:
                self.error.emit("Analysis cancelled before starting")
                return
            
            # Emit progress signal
            self.progress.emit("Initializing enhanced active learning optimizer...")
            
            # Create optimizer
            self.optimizer = ActiveLearningOptimizer(
                model_type=self.analysis_params['model_type'],
                acquisition_function=self.analysis_params['acquisition_function'],
                random_state=42
            )
            
            # Check if stopped after initialization
            if self.should_stop:
                self.error.emit("Analysis cancelled during initialization")
                return
            
            # Run analysis based on mode
            self.progress.emit("Running enhanced active learning analysis...")
            
            if self.analysis_params['is_multi_objective']:
                # Multi-objective optimization
                results_data = self.optimizer.run_multi_objective(
                    training_df=self.analysis_params['training_data'],
                    virtual_df=self.analysis_params['virtual_data'],
                    target_columns=self.analysis_params['target_columns'],
                    feature_columns=self.analysis_params['feature_columns'],
                    goal_directions=self.analysis_params['goal_directions'],
                    acquisition_config=self.analysis_params['acquisition_config'],
                    n_iterations_bootstrap=self.analysis_params['n_iterations']
                )
            else:
                # Single-objective optimization
                results_data = self.optimizer.run(
                    training_df=self.analysis_params['training_data'],
                    virtual_df=self.analysis_params['virtual_data'],
                    target_column=self.analysis_params['target_columns'][0],
                    feature_columns=self.analysis_params['feature_columns'],
                    goal=self.analysis_params['goal_directions'][0],
                    acquisition_config=self.analysis_params['acquisition_config'],
                    n_iterations_bootstrap=self.analysis_params['n_iterations']
                )
            
            # Check if stopped after analysis
            if self.should_stop:
                self.error.emit("Analysis cancelled during execution")
                return
            
            # Emit success signal
            self.progress.emit("Analysis completed successfully!")
            self.finished.emit(results_data)
            
        except Exception as e:
            # Check if it's a cancellation
            if self.should_stop:
                self.error.emit("Analysis cancelled by user")
            else:
                # Emit error signal
                self.error.emit(f"Error occurred during analysis: {str(e)}")
    
    def stop_analysis(self):
        """
        Request the worker to stop the analysis.
        """
        self.should_stop = True
    
    def get_optimizer(self):
        """
        Get the optimizer instance (for feature importance, etc.)
        """
        return self.optimizer


class CustomTableWidget(QTableWidget):

    
    def __init__(self):
        super().__init__()
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSortingEnabled(True)
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #d6d9dc;
                background-color: white;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #3daee9;
                color: white;
            }
        """)


class MatplotlibWidget(QWidget):
    """Enhanced matplotlib widget with interactive toolbar and export functionality."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Start with a reasonable default size, but make it adaptable
        self.figure = Figure(figsize=(8, 6), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        
        # Create navigation toolbar for interactivity
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # Add export button to toolbar
        export_action = QAction('💾 Export', self)
        export_action.setToolTip('Export chart as image')
        export_action.triggered.connect(self.export_chart)
        self.toolbar.addAction(export_action)
        
        # Layout with toolbar on top
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Add margins
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)  # Give canvas stretch factor
        self.setLayout(layout)
        
        # Set matplotlib style for better appearance
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Enable tight layout for better spacing
        self.figure.set_tight_layout(True)
        
        # Enable auto-resize
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def resizeEvent(self, event):
        """Handle resize events to adjust figure size dynamically."""
        super().resizeEvent(event)
        if hasattr(self, 'canvas') and self.canvas:
            # Get widget size in inches (accounting for DPI)
            width_inches = (self.width() - 20) / self.figure.dpi  # Account for margins
            height_inches = (self.height() - 60) / self.figure.dpi  # Account for toolbar
            
            # Set minimum size constraints
            width_inches = max(width_inches, 4)
            height_inches = max(height_inches, 3)
            
            # Update figure size
            self.figure.set_size_inches(width_inches, height_inches)
            self.canvas.draw_idle()
    
    def clear(self):
        """Clear the figure and redraw."""
        self.figure.clear()
        self.canvas.draw()
    
    def export_chart(self):
        """Export the current chart as an image file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Chart", "chart.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*)"
            )
            
            if file_path:
                # Save with high DPI for quality
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight', 
                                  facecolor='white', edgecolor='none')
                
                # Show success message
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", f"Chart exported successfully to:\n{file_path}")
                
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to export chart:\n{str(e)}")
    
    def set_title(self, title):
        """Set a title for the entire widget."""
        self.setWindowTitle(title)


class ActiveLearningWindow(QMainWindow):
    """
Main window for the Active Learning & Optimization module.

This window provides a user-friendly interface for Bayesian optimization,
allowing users to load data, configure models, and visualize results.

Key Features:
- Non-blocking analysis execution using QThread workers
- Real-time progress updates and cancellation support
- Comprehensive multi-objective optimization with Pareto analysis
- Interactive data visualization and model insights
- Advanced candidate set generation with constraints

Threading Architecture:
- Main UI thread handles user interactions and display updates
- Analysis worker thread performs computationally intensive tasks
- Signal-slot communication ensures thread-safe UI updates
"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Active Learning & Optimization")
        
        # 🔧 实现自适应窗口大小和DPI感知
        self._setup_adaptive_window_size()
        
        # Data storage
        self.training_data = None
        self.virtual_data = None
        self.results_data = None
        self.optimizer = None
        
        # Multi-objective support
        self.is_multi_objective = False
        self.target_columns = []
        self.goal_directions = []
        
        # Threading for non-blocking analysis
        self.analysis_thread = None
        self.analysis_worker = None
        self.is_analysis_running = False
        
        # Iterative active learning session management
        self.session = None
        self.last_recommendation = None
        self.feedback_inputs = {}
        self.feedback_box = None
        
        # Enhanced recommendation system
        self.recommendation_mode = "single"  # "single" or "batch"
        self.batch_size = 5
        self.confidence_threshold = 0.5
        self.current_batch_recommendations = []
        
        # Dynamic budget and early stopping
        self.enable_early_stopping = True
        self.convergence_threshold = 0.01
        self.max_stagnation_iterations = 3
        self.budget_allocation = "adaptive"
        self.performance_history = []
        self.cost_benefit_data = []
        
        # Constraint optimization
        self.hard_constraints = {}
        self.soft_constraints = {}
        self.feasibility_weights = {}
        self.enable_constraint_optimization = False
        
        # Advanced uncertainty quantification
        self.uncertainty_method = "ensemble"  # "ensemble", "dropout", "gaussian"
        self.uncertainty_components = {}
        
        # UI components
        self.init_ui()
        
    def _setup_adaptive_window_size(self):
        """设置自适应窗口大小，支持不同分辨率和DPI设置"""
        from PyQt5.QtWidgets import QApplication, QDesktopWidget
        from PyQt5.QtCore import Qt
        import sys
        
        # 获取屏幕信息
        desktop = QApplication.desktop()
        screen_rect = desktop.screenGeometry()
        screen_width = screen_rect.width()
        screen_height = screen_rect.height()
        
        # 获取DPI信息
        screen = QApplication.primaryScreen()
        dpi_ratio = screen.devicePixelRatio() if hasattr(screen, 'devicePixelRatio') else 1.0
        logical_dpi = screen.logicalDotsPerInch() if hasattr(screen, 'logicalDotsPerInch') else 96
        
        # 基于屏幕尺寸计算适当的窗口大小
        # 使用屏幕的70-85%作为窗口大小，确保在不同分辨率下都有良好的显示效果
        if screen_width <= 1366:  # 小屏幕/笔记本
            window_width = int(screen_width * 0.85)
            window_height = int(screen_height * 0.80)
            self.adaptive_left_panel_width = 280
        elif screen_width <= 1920:  # 标准1080p
            window_width = int(screen_width * 0.75)
            window_height = int(screen_height * 0.75)
            self.adaptive_left_panel_width = 320
        elif screen_width <= 2560:  # 2K分辨率
            window_width = int(screen_width * 0.70)
            window_height = int(screen_height * 0.70)
            self.adaptive_left_panel_width = 380
        else:  # 4K和更高分辨率
            window_width = int(screen_width * 0.65)
            window_height = int(screen_height * 0.65)
            self.adaptive_left_panel_width = 420
        
        # DPI调整：对于高DPI屏幕，适当增加组件大小
        if logical_dpi > 120:  # 高DPI屏幕
            dpi_scale_factor = logical_dpi / 96.0
            self.adaptive_left_panel_width = int(self.adaptive_left_panel_width * min(dpi_scale_factor, 1.5))
        
        # 确保窗口不会太小
        min_width = 1000
        min_height = 700
        window_width = max(window_width, min_width)
        window_height = max(window_height, min_height)
        
        # 计算居中位置
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # 设置窗口几何
        self.setGeometry(x, y, window_width, window_height)
        
        # 设置最小尺寸
        self.setMinimumSize(min_width, min_height)
        
        # 启用自适应布局
        self.setAttribute(Qt.WA_DontShowOnScreen, False)
        
        print(f"🖥️ 屏幕信息: {screen_width}×{screen_height}, DPI: {logical_dpi}, 比例: {dpi_ratio}")
        print(f"📐 窗口设置: {window_width}×{window_height}, 左侧面板宽度: {self.adaptive_left_panel_width}")
    
    def resizeEvent(self, event):
        """处理窗口大小改变事件，实现响应式布局"""
        super().resizeEvent(event)
        
        # 获取当前窗口大小
        current_width = self.width()
        current_height = self.height()
        
        # 根据当前窗口大小动态调整左侧面板宽度
        if hasattr(self, 'adaptive_left_panel_width'):
            # 计算适当的左侧面板宽度比例
            if current_width < 1200:
                panel_ratio = 0.28  # 小窗口时使用更大比例
            elif current_width < 1600:
                panel_ratio = 0.25  # 中等窗口
            else:
                panel_ratio = 0.22  # 大窗口时使用较小比例
            
            new_panel_width = min(int(current_width * panel_ratio), self.adaptive_left_panel_width)
            new_panel_width = max(new_panel_width, 250)  # 最小宽度
            
            # 如果左侧面板存在，更新其宽度
            if hasattr(self, 'centralWidget') and self.centralWidget():
                layout = self.centralWidget().layout()
                if layout and layout.count() > 0:
                    left_panel = layout.itemAt(0).widget()
                    if left_panel:
                        left_panel.setMaximumWidth(new_panel_width)
        
        # 更新matplotlib图表大小以适应新的窗口尺寸
        self._update_plots_for_resize()
    
    def _update_plots_for_resize(self):
        """更新所有matplotlib图表以适应新的窗口尺寸"""
        try:
            # 更新主要的matplotlib组件
            plots_to_update = []
            
            # 收集所有需要更新的图表
            if hasattr(self, 'exploration_plot') and self.exploration_plot:
                plots_to_update.append(self.exploration_plot)
            if hasattr(self, 'importance_plot') and self.importance_plot:
                plots_to_update.append(self.importance_plot)
            if hasattr(self, 'correlation_plot') and self.correlation_plot:
                plots_to_update.append(self.correlation_plot)
            if hasattr(self, 'uncertainty_plot') and self.uncertainty_plot:
                plots_to_update.append(self.uncertainty_plot)
            if hasattr(self, 'design_space_plot') and self.design_space_plot:
                plots_to_update.append(self.design_space_plot)
            if hasattr(self, 'pareto_plot') and self.pareto_plot:
                plots_to_update.append(self.pareto_plot)
            
            # 更新图表
            for plot in plots_to_update:
                if plot and hasattr(plot, 'figure') and plot.figure:
                    plot.figure.tight_layout()
                    if hasattr(plot, 'canvas') and plot.canvas:
                        plot.canvas.draw()
                        
        except Exception as e:
                         # 静默处理错误，避免影响UI操作
             print(f"🔧 图表更新警告: {e}")
    
    def showEvent(self, event):
        """处理窗口显示事件，确保初始布局正确"""
        super().showEvent(event)
        
        # 延迟执行自适应调整，确保所有组件都已创建
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._apply_adaptive_adjustments)
    
    def _apply_adaptive_adjustments(self):
        """应用额外的自适应调整"""
        try:
            # 调整表格列宽
            self._adjust_table_columns()
            
            # 调整滚动区域
            self._adjust_scroll_areas()
            
        except Exception as e:
            print(f"🔧 自适应调整警告: {e}")
    
    def _adjust_table_columns(self):
        """调整表格列宽以适应窗口大小"""
        tables = []
        
        # 收集所有表格
        if hasattr(self, 'results_table') and self.results_table:
            tables.append(self.results_table)
        if hasattr(self, 'history_table') and self.history_table:
            tables.append(self.history_table)
        if hasattr(self, 'selected_targets_table') and self.selected_targets_table:
            tables.append(self.selected_targets_table)
        if hasattr(self, 'pareto_table') and self.pareto_table:
            tables.append(self.pareto_table)
        
        for table in tables:
            if table and hasattr(table, 'horizontalHeader'):
                header = table.horizontalHeader()
                if hasattr(header, 'setStretchLastSection'):
                    header.setStretchLastSection(True)
                if hasattr(header, 'setSectionResizeMode'):
                    from PyQt5.QtWidgets import QHeaderView
                    # 设置列宽自适应模式
                    for i in range(table.columnCount()):
                        if i == table.columnCount() - 1:  # 最后一列拉伸
                            header.setSectionResizeMode(i, QHeaderView.Stretch)
                        else:  # 其他列根据内容调整
                            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
    
    def _adjust_scroll_areas(self):
        """调整滚动区域的行为"""
        # 确保特征列表有合适的滚动行为
        if hasattr(self, 'feature_list') and self.feature_list:
            # 根据窗口高度调整特征列表的最大高度
            window_height = self.height()
            if window_height > 800:
                max_height = min(250, window_height // 4)
            else:
                max_height = min(200, window_height // 5)
            
            self.feature_list.setMaximumHeight(max_height)
        
        # Apply modern font styling consistent with the provided UI design
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                color: #212529;
            }
            QGroupBox {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-weight: 600;
                font-size: 10pt;
                color: #343a40;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                background-color: #ffffff;
                color: #495057;
            }
            QPushButton {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                font-weight: 500;
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px 16px;
                color: #495057;
                min-height: 18px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
                color: #212529;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
                border-color: #adb5bd;
            }
            QPushButton:disabled {
                background-color: #f8f9fa;
                color: #6c757d;
                border-color: #e9ecef;
            }
            QLabel {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                color: #495057;
            }
            QComboBox {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                color: #495057;
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #adb5bd;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #6c757d;
                margin-right: 5px;
            }
            QListWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                color: #495057;
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                selection-background-color: #007bff;
                selection-color: #ffffff;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #f8f9fa;
            }
            QListWidget::item:hover {
                background-color: #f8f9fa;
            }
            QTableWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                color: #495057;
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                gridline-color: #e9ecef;
                selection-background-color: #007bff;
                selection-color: #ffffff;
            }
            QTableWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #f8f9fa;
            }
            QHeaderView::section {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                font-weight: 600;
                background-color: #f8f9fa;
                color: #495057;
                border: 1px solid #dee2e6;
                padding: 8px;
            }
            QRadioButton {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                color: #495057;
                spacing: 8px;
            }
            QCheckBox {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                color: #495057;
                spacing: 8px;
            }
            QSpinBox, QDoubleSpinBox {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                color: #495057;
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 20px;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                background-color: #ffffff;
                border-radius: 4px;
            }
            QTabBar::tab {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                font-weight: 500;
                color: #6c757d;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #495057;
                border-bottom: 2px solid #007bff;
            }
            QTabBar::tab:hover {
                background-color: #e9ecef;
                color: #495057;
            }
            QTextEdit {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Consolas', monospace;
                font-size: 9pt;
                color: #495057;
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
            }
            QProgressBar {
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;
                font-size: 9pt;
                color: #495057;
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #ced4da;
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #007bff;
                border: 1px solid #0056b3;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #0056b3;
            }
        """)
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout: horizontal split
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Controls (使用自适应宽度)
        left_panel = self.create_control_panel()
        adaptive_width = getattr(self, 'adaptive_left_panel_width', 350)
        left_panel.setMaximumWidth(adaptive_width)
        left_panel.setMinimumWidth(min(280, adaptive_width))
        
        # Right panel: Results
        right_panel = self.create_results_panel()
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 0)
        main_layout.addWidget(right_panel, 1)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready to load data")
    
    # Simplified progress handling - no complex callbacks
    
    def update_pareto_visualization(self, pareto_front, pareto_indices):
        """Update Pareto front visualization for multi-objective results."""
        if not self.is_multi_objective:
            return
        
        # Check if pareto plot exists in results tabs
        pareto_tab_exists = False
        for i in range(self.results_tabs.count()):
            if self.results_tabs.tabText(i) == "Pareto Front":
                pareto_tab_exists = True
                break
        
        # Add a new tab for Pareto visualization if it doesn't exist
        if not pareto_tab_exists:
            self.pareto_tab = QWidget()
            pareto_layout = QVBoxLayout(self.pareto_tab)
            
            # Create matplotlib widget for Pareto plot
            self.pareto_plot = MatplotlibWidget()
            pareto_layout.addWidget(self.pareto_plot, 2)  # Give more space to plot
            
            # Create Pareto optimal solutions table
            pareto_table_label = QLabel("Pareto Optimal Solutions Details:")
            pareto_table_label.setStyleSheet("font-weight: 600; font-size: 10pt; margin-top: 10px; color: #495057; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
            pareto_layout.addWidget(pareto_table_label)
            
            self.pareto_table = CustomTableWidget()
            self.pareto_table.setMaximumHeight(200)  # Limit table height
            self.pareto_table.setToolTip("Display feature values and target values for all Pareto optimal solutions")
            pareto_layout.addWidget(self.pareto_table, 1)  # Give less space to table
            
            # Add tab to results
            self.results_tabs.addTab(self.pareto_tab, "Pareto Front")
        
        # Update Pareto plot
        self.update_pareto_plot()
    
    def update_pareto_plot(self):
        """Update the Pareto front plot with support for 2D, 3D, and projection views."""
        if not self.is_multi_objective or self.results_data is None:
            return
        
        targets, goals = self.get_selected_targets_and_goals()
        if len(targets) < 2:
            return
        
        # Get objective data columns
        obj_cols = [f'{target}_predicted_mean' for target in targets]
        missing_cols = [col for col in obj_cols if col not in self.results_data.columns]
        
        if missing_cols:
            print(f"Missing objective columns: {missing_cols}")
            return
        
        # Get valid data
        valid_mask = ~self.results_data[obj_cols].isnull().any(axis=1)
        valid_data = self.results_data[valid_mask]
        
        if len(valid_data) == 0:
            print("No valid data for Pareto plot")
            return
        
        self.pareto_plot.clear()
        
        if len(targets) == 2:
            # 2D Pareto front
            self._plot_pareto_2d(valid_data, targets, goals, obj_cols)
        elif len(targets) == 3:
            # 3D Pareto front with projections
            self._plot_pareto_3d_with_projections(valid_data, targets, goals, obj_cols)
        else:
            # For 4+ objectives, show parallel coordinates and pairwise projections
            self._plot_pareto_high_dimensional(valid_data, targets, goals, obj_cols)
        
        self.pareto_plot.figure.tight_layout()
        self.pareto_plot.canvas.draw()
        
        # Update Pareto table
        try:
            self.update_pareto_table()
        except Exception as e:
            print(f"Error updating Pareto table: {e}")
    
    def _plot_pareto_2d(self, valid_data, targets, goals, obj_cols):
        """Plot 2D Pareto front."""
        ax = self.pareto_plot.figure.add_subplot(111)
        
        obj1_data = valid_data[obj_cols[0]]
        obj2_data = valid_data[obj_cols[1]]
        
        # All candidate points
        ax.scatter(obj1_data, obj2_data, alpha=0.6, s=30, c='lightblue', 
                  label='All Candidates', edgecolors='blue', linewidth=0.5)
        
        # Pareto optimal points
        if 'is_pareto_optimal' in valid_data.columns:
            pareto_mask = valid_data['is_pareto_optimal'].fillna(False)
            if pareto_mask.any():
                pareto_data = valid_data[pareto_mask]
                ax.scatter(pareto_data[obj_cols[0]], pareto_data[obj_cols[1]], 
                          s=60, c='red', label='Pareto Optimal', marker='D', 
                          edgecolors='darkred', linewidth=1)
        
        # Top recommendation
        top_result = valid_data.iloc[0]
        ax.scatter(top_result[obj_cols[0]], top_result[obj_cols[1]], 
                  s=100, c='gold', marker='*', edgecolors='orange', linewidth=2,
                  label='Top Recommendation', zorder=5)
        
        # Labels and formatting
        goal1_text = "↑" if goals[0] == "maximize" else "↓"
        goal2_text = "↑" if goals[1] == "maximize" else "↓"
        
        ax.set_xlabel(f'{targets[0]} {goal1_text}')
        ax.set_ylabel(f'{targets[1]} {goal2_text}')
        ax.set_title('2D Pareto Front Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_pareto_3d_with_projections(self, valid_data, targets, goals, obj_cols):
        """Plot 3D Pareto front with 2D projections."""
        # Create 2x2 subplot layout
        # Top-left: 3D plot
        # Top-right: XY projection
        # Bottom-left: XZ projection  
        # Bottom-right: YZ projection
        
        # 3D plot
        ax3d = self.pareto_plot.figure.add_subplot(2, 2, 1, projection='3d')
        
        obj1_data = valid_data[obj_cols[0]]
        obj2_data = valid_data[obj_cols[1]] 
        obj3_data = valid_data[obj_cols[2]]
        
        # All candidate points in 3D
        ax3d.scatter(obj1_data, obj2_data, obj3_data, alpha=0.4, s=20, c='lightblue', 
                    label='All Candidates')
        
        # Pareto optimal points in 3D
        if 'is_pareto_optimal' in valid_data.columns:
            pareto_mask = valid_data['is_pareto_optimal'].fillna(False)
            if pareto_mask.any():
                pareto_data = valid_data[pareto_mask]
                ax3d.scatter(pareto_data[obj_cols[0]], pareto_data[obj_cols[1]], pareto_data[obj_cols[2]], 
                            s=60, c='red', marker='D', label='Pareto Optimal')
        
        # Top recommendation in 3D
        top_result = valid_data.iloc[0]
        ax3d.scatter(top_result[obj_cols[0]], top_result[obj_cols[1]], top_result[obj_cols[2]], 
                    s=100, c='gold', marker='*', label='Top Recommendation')
        
        # 3D labels
        goal1_text = "↑" if goals[0] == "maximize" else "↓"
        goal2_text = "↑" if goals[1] == "maximize" else "↓"
        goal3_text = "↑" if goals[2] == "maximize" else "↓"
        
        ax3d.set_xlabel(f'{targets[0]} {goal1_text}')
        ax3d.set_ylabel(f'{targets[1]} {goal2_text}')
        ax3d.set_zlabel(f'{targets[2]} {goal3_text}')
        ax3d.set_title('3D Pareto Front')
        ax3d.legend()
        
        # 2D projections
        projections = [
            (1, 2, 2, f'{targets[0]} vs {targets[1]}'),  # XY projection
            (1, 3, 3, f'{targets[0]} vs {targets[2]}'),  # XZ projection  
            (2, 3, 4, f'{targets[1]} vs {targets[2]}')   # YZ projection
        ]
        
        for i, j, subplot_idx, title in projections:
            ax = self.pareto_plot.figure.add_subplot(2, 2, subplot_idx)
            
            # All points
            ax.scatter(valid_data[obj_cols[i-1]], valid_data[obj_cols[j-1]], 
                      alpha=0.6, s=20, c='lightblue', label='All Candidates')
            
            # Pareto points
            if 'is_pareto_optimal' in valid_data.columns:
                pareto_mask = valid_data['is_pareto_optimal'].fillna(False)
                if pareto_mask.any():
                    pareto_data = valid_data[pareto_mask]
                    ax.scatter(pareto_data[obj_cols[i-1]], pareto_data[obj_cols[j-1]], 
                              s=40, c='red', marker='D', label='Pareto Optimal')
            
            # Top recommendation
            ax.scatter(top_result[obj_cols[i-1]], top_result[obj_cols[j-1]], 
                      s=80, c='gold', marker='*', label='Top Recommendation')
            
            ax.set_xlabel(f'{targets[i-1]} {[goal1_text, goal2_text, goal3_text][i-1]}')
            ax.set_ylabel(f'{targets[j-1]} {[goal1_text, goal2_text, goal3_text][j-1]}')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if subplot_idx == 2:  # Only show legend on one subplot to save space
                ax.legend(fontsize=8)
    
    def _plot_pareto_high_dimensional(self, valid_data, targets, goals, obj_cols):
        """Plot high-dimensional Pareto front using parallel coordinates."""
        ax = self.pareto_plot.figure.add_subplot(111)
        
        # Normalize data for parallel coordinates
        normalized_data = valid_data[obj_cols].copy()
        for col in obj_cols:
            min_val, max_val = normalized_data[col].min(), normalized_data[col].max()
            if max_val > min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        x_positions = range(len(targets))
        
        # Plot all candidates
        for idx, row in normalized_data.iterrows():
            ax.plot(x_positions, row[obj_cols], alpha=0.3, color='lightblue', linewidth=0.5)
        
        # Plot Pareto optimal solutions
        if 'is_pareto_optimal' in valid_data.columns:
            pareto_mask = valid_data['is_pareto_optimal'].fillna(False)
            if pareto_mask.any():
                pareto_normalized = normalized_data[pareto_mask]
                for idx, row in pareto_normalized.iterrows():
                    ax.plot(x_positions, row[obj_cols], alpha=0.8, color='red', linewidth=2)
        
        # Plot top recommendation
        top_normalized = normalized_data.iloc[0]
        ax.plot(x_positions, top_normalized[obj_cols], alpha=1.0, color='gold', linewidth=3, marker='o')
        
        # Formatting
        ax.set_xticks(x_positions)
        goal_texts = ["↑" if goal == "maximize" else "↓" for goal in goals]
        ax.set_xticklabels([f'{target} {goal_text}' for target, goal_text in zip(targets, goal_texts)], 
                          rotation=45, ha='right')
        ax.set_ylabel('Normalized Objective Values')
        ax.set_title(f'{len(targets)}D Pareto Front (Parallel Coordinates)')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lightblue', alpha=0.3, label='All Candidates'),
            Line2D([0], [0], color='red', alpha=0.8, label='Pareto Optimal'),
            Line2D([0], [0], color='gold', alpha=1.0, label='Top Recommendation')
        ]
        ax.legend(handles=legend_elements)
    
    def update_pareto_table(self):
        """Update Pareto optimal solutions table"""
        if not hasattr(self, 'pareto_table') or self.results_data is None:
            return
        
        # Get Pareto optimal solutions
        if 'is_pareto_optimal' not in self.results_data.columns:
            self.pareto_table.setRowCount(1)
            self.pareto_table.setColumnCount(1)
            self.pareto_table.setHorizontalHeaderLabels(['Information'])
            item = QTableWidgetItem("No Pareto optimal solution data found")
            self.pareto_table.setItem(0, 0, item)
            return
        
        pareto_mask = self.results_data['is_pareto_optimal'].fillna(False)
        pareto_solutions = self.results_data[pareto_mask].copy()
        
        if len(pareto_solutions) == 0:
            self.pareto_table.setRowCount(1)
            self.pareto_table.setColumnCount(1)
            self.pareto_table.setHorizontalHeaderLabels(['Information'])
            item = QTableWidgetItem("No Pareto optimal solutions found")
            self.pareto_table.setItem(0, 0, item)
            return
        
        # Get target and feature columns
        targets, _ = self.get_selected_targets_and_goals()
        
        # Select columns to display: target predictions, uncertainty, acquisition score, and important features
        display_columns = []
        
        # Add target columns
        for target in targets:
            mean_col = f'{target}_predicted_mean'
            std_col = f'{target}_uncertainty_std'
            if mean_col in pareto_solutions.columns:
                display_columns.append(mean_col)
            if std_col in pareto_solutions.columns:
                display_columns.append(std_col)
        
        # Add acquisition score
        if 'acquisition_score' in pareto_solutions.columns:
            display_columns.append('acquisition_score')
        
        # Add some important feature columns (prioritize user-selected features)
        feature_columns = []
        selected_features = self.get_selected_features()
        
        # First add user-selected features
        for col in selected_features:
            if col in pareto_solutions.columns and col not in display_columns:
                feature_columns.append(col)
                if len(feature_columns) >= 5:  # Limit number of features displayed
                    break
        
        # If not enough features, add other numeric columns
        if len(feature_columns) < 5:
            for col in pareto_solutions.columns:
                if (col not in display_columns and 
                    col != 'is_pareto_optimal' and 
                    col not in feature_columns):
                    # Check if it's a numeric column
                    if pareto_solutions[col].dtype in ['int64', 'float64']:
                        feature_columns.append(col)
                        if len(feature_columns) >= 5:  # Limit number of features displayed
                            break
        
        display_columns.extend(feature_columns)
        
        # Set up table
        pareto_display = pareto_solutions[display_columns]
        
        self.pareto_table.setRowCount(len(pareto_display))
        self.pareto_table.setColumnCount(len(pareto_display.columns))
        
        # Set column headers (simplified display)
        simplified_headers = []
        for col in pareto_display.columns:
            if '_predicted_mean' in col:
                simplified_headers.append(col.replace('_predicted_mean', '_Pred'))
            elif '_uncertainty_std' in col:
                simplified_headers.append(col.replace('_uncertainty_std', '_Uncertainty'))
            elif col == 'acquisition_score':
                simplified_headers.append('Acq_Score')
            else:
                simplified_headers.append(col)
        
        self.pareto_table.setHorizontalHeaderLabels(simplified_headers)
        
        # Fill table data
        for i, (_, row) in enumerate(pareto_display.iterrows()):
            for j, value in enumerate(row):
                if pd.isna(value):
                    item = QTableWidgetItem("N/A")
                elif isinstance(value, (int, float)):
                    # Format numeric display
                    if abs(value) < 0.001:
                        item = QTableWidgetItem(f"{value:.2e}")
                    elif abs(value) < 1:
                        item = QTableWidgetItem(f"{value:.4f}")
                    elif abs(value) < 1000:
                        item = QTableWidgetItem(f"{value:.3f}")
                    else:
                        item = QTableWidgetItem(f"{value:.1f}")
                else:
                    item = QTableWidgetItem(str(value))
                
                # Highlight target columns
                col_name = pareto_display.columns[j]
                if any(target in col_name for target in targets) or col_name == 'acquisition_score':
                    item.setBackground(QColor(255, 240, 240))  # Light red background
                
                self.pareto_table.setItem(i, j, item)
        
        # Adjust column widths
        self.pareto_table.resizeColumnsToContents()
        
        # Add sorting functionality
        self.pareto_table.setSortingEnabled(True)
        
        # Sort by acquisition score in descending order (if column exists)
        if 'acquisition_score' in pareto_display.columns:
            score_col_idx = list(pareto_display.columns).index('acquisition_score')
            self.pareto_table.sortItems(score_col_idx, Qt.DescendingOrder)
    
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Export results action
        export_action = QAction('Export Results', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        # Add separator
        file_menu.addSeparator()
        
        # Reset active learning session action
        reset_session_action = QAction('Reset Active Learning Session', self)
        reset_session_action.setShortcut('Ctrl+R')
        reset_session_action.triggered.connect(self.reset_session)
        reset_session_action.setToolTip('Clear all iteratively acquired data points and start fresh')
        file_menu.addAction(reset_session_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_control_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Data loading section
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout(data_group)
        
        # Training data
        self.training_button = QPushButton("Load Training Data")
        self.training_button.clicked.connect(self.load_training_data)
        self.training_label = QLabel("No training data loaded")
        self.training_label.setStyleSheet("color: #6c757d; font-size: 9pt; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
        
        data_layout.addWidget(self.training_button)
        data_layout.addWidget(self.training_label)
        
        # Virtual data
        self.virtual_button = QPushButton("Load Candidate Data")
        self.virtual_button.clicked.connect(self.load_virtual_data)
        self.virtual_label = QLabel("No candidate data loaded")
        self.virtual_label.setStyleSheet("color: #6c757d; font-size: 9pt; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
        
        data_layout.addWidget(self.virtual_button)
        data_layout.addWidget(self.virtual_label)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        data_layout.addWidget(separator)
        
        # Candidate set generation
        generate_label = QLabel("Or Generate Candidate Set:")
        generate_label.setStyleSheet("color: #495057; font-weight: 600; font-size: 9pt; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
        data_layout.addWidget(generate_label)
        
        self.generate_button = QPushButton("🎯 Generate Candidates")
        self.generate_button.clicked.connect(self.show_candidate_generator)
        self.generate_button.setToolTip("Generate candidate set using advanced sampling methods")
        self.generate_button.setEnabled(False)  # Enabled only when training data is loaded
        data_layout.addWidget(self.generate_button)
        
        layout.addWidget(data_group)
        
        # Configuration section
        config_group = QGroupBox("Core Configuration")
        config_layout = QFormLayout(config_group)
        
        # Optimization mode selection
        self.mode_group = QButtonGroup()
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        
        self.single_obj_radio = QRadioButton("Single Objective")
        self.multi_obj_radio = QRadioButton("Multi Objective")
        self.single_obj_radio.setChecked(True)
        self.single_obj_radio.toggled.connect(self.on_mode_changed)
        
        self.mode_group.addButton(self.single_obj_radio)
        self.mode_group.addButton(self.multi_obj_radio)
        
        mode_layout.addWidget(self.single_obj_radio)
        mode_layout.addWidget(self.multi_obj_radio)
        
        config_layout.addRow("Optimization Mode:", mode_widget)
        
        # Single objective widgets (shown by default)
        self.single_obj_widget = QWidget()
        single_obj_layout = QFormLayout(self.single_obj_widget)
        single_obj_layout.setContentsMargins(0, 0, 0, 0)
        
        self.target_combo = QComboBox()
        self.target_combo.setToolTip("Select target variable to optimize")
        single_obj_layout.addRow("Target Variable:", self.target_combo)
        
        # Goal selection for single objective
        goal_widget = QWidget()
        goal_layout = QHBoxLayout(goal_widget)
        goal_layout.setContentsMargins(0, 0, 0, 0)
        
        self.goal_group = QButtonGroup()
        self.maximize_radio = QRadioButton("Maximize")
        self.minimize_radio = QRadioButton("Minimize")
        self.maximize_radio.setChecked(True)
        
        self.goal_group.addButton(self.maximize_radio)
        self.goal_group.addButton(self.minimize_radio)
        
        goal_layout.addWidget(self.maximize_radio)
        goal_layout.addWidget(self.minimize_radio)
        
        single_obj_layout.addRow("Goal:", goal_widget)
        
        config_layout.addRow(self.single_obj_widget)
        
        # Multi objective widgets (hidden by default)
        self.multi_obj_widget = QWidget()
        multi_obj_layout = QVBoxLayout(self.multi_obj_widget)
        multi_obj_layout.setContentsMargins(0, 0, 0, 0)
        
        # Multi-objective target selection
        multi_targets_label = QLabel("Target Variables:")
        multi_targets_label.setStyleSheet("font-weight: 600; font-size: 9pt; color: #495057; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
        multi_obj_layout.addWidget(multi_targets_label)
        
        # Available targets list
        self.available_targets_list = QListWidget()
        self.available_targets_list.setMaximumHeight(80)
        self.available_targets_list.setToolTip("Available target variables")
        multi_obj_layout.addWidget(self.available_targets_list)
        
        # Add/Remove buttons
        button_layout = QHBoxLayout()
        self.add_target_button = QPushButton("Add →")
        self.add_target_button.clicked.connect(self.add_target)
        self.remove_target_button = QPushButton("← Remove")
        self.remove_target_button.clicked.connect(self.remove_target)
        
        button_layout.addWidget(self.add_target_button)
        button_layout.addWidget(self.remove_target_button)
        multi_obj_layout.addLayout(button_layout)
        
        # Selected targets with goals
        selected_label = QLabel("Selected Objectives:")
        selected_label.setStyleSheet("font-weight: 600; font-size: 9pt; color: #495057; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
        multi_obj_layout.addWidget(selected_label)
        
        self.selected_targets_table = QTableWidget()
        self.selected_targets_table.setColumnCount(2)
        self.selected_targets_table.setHorizontalHeaderLabels(["Target Variable", "Goal"])
        self.selected_targets_table.horizontalHeader().setStretchLastSection(True)
        self.selected_targets_table.setMaximumHeight(120)
        multi_obj_layout.addWidget(self.selected_targets_table)
        
        config_layout.addRow(self.multi_obj_widget)
        self.multi_obj_widget.setVisible(False)
        

        
        # Feature selection
        feature_widget = QWidget()
        feature_layout = QVBoxLayout(feature_widget)
        feature_layout.setContentsMargins(0, 0, 0, 0)
        
        feature_buttons = QWidget()
        feature_buttons_layout = QHBoxLayout(feature_buttons)
        feature_buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.select_all_features)
        self.select_none_button = QPushButton("Select None")
        self.select_none_button.clicked.connect(self.select_no_features)
        
        feature_buttons_layout.addWidget(self.select_all_button)
        feature_buttons_layout.addWidget(self.select_none_button)
        
        # 添加特征排序控件 (使用下拉菜单替代滑块，避免重叠问题)
        sort_layout = QHBoxLayout()
        sort_layout.setContentsMargins(0, 0, 0, 0)
        sort_layout.setSpacing(5)
        
        sort_label = QLabel("特征排序:")
        sort_label.setStyleSheet("font-weight: bold;")
        sort_layout.addWidget(sort_label)
        
        self.sort_methods = [
            "元素优先",
            "工艺优先", 
            "A-Z", 
            "Z-A",
            "原始顺序"
        ]
        
        self.feature_sort_combo = QComboBox()
        self.feature_sort_combo.addItems(self.sort_methods)
        self.feature_sort_combo.setCurrentText("元素优先")
        self.feature_sort_combo.currentIndexChanged.connect(self.on_feature_sort_changed)
        self.feature_sort_combo.setToolTip("""
            元素优先: 按元素、材料属性、工艺参数排序
            工艺优先: 按工艺参数、元素、材料属性排序
            A-Z: 字母升序排列
            Z-A: 字母降序排列
            原始顺序: 保持数据源顺序
        """)
        self.feature_sort_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #2E86AB;
                border-radius: 3px;
                padding: 2px 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f8f8, stop:1 #e8e8e8);
                min-width: 100px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left: 1px solid #2E86AB;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #2E86AB;
                selection-background-color: #2E86AB;
            }
        """)
        sort_layout.addWidget(self.feature_sort_combo)
        sort_layout.addStretch()
        
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.feature_list.setMinimumHeight(150)  # Increased from 120 to 150
        self.feature_list.setMaximumHeight(200)  # Added maximum height to prevent too much expansion
        
        feature_layout.addWidget(feature_buttons)
        feature_layout.addLayout(sort_layout)
        feature_layout.addWidget(self.feature_list)
        
        config_layout.addRow("Feature Variables:", feature_widget)
        
        layout.addWidget(config_group)
        
        # Advanced settings (collapsible)
        self.advanced_group = QGroupBox("Advanced Settings")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QFormLayout(self.advanced_group)
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "RandomForest", "XGBoost", "GaussianProcess", 
            "LightGBM", "CatBoost", "ExtraTrees", 
            "SVR", "MLPRegressor", "BayesianRidge"
        ])
        self.model_combo.setCurrentText("RandomForest")
        self.model_combo.setToolTip("Select surrogate model type:\n"
                                   "• RandomForest: Robust ensemble, good for mixed data\n"
                                   "• XGBoost: High performance gradient boosting\n"
                                   "• GaussianProcess: Native uncertainty, good for small datasets\n"
                                   "• LightGBM: Fast gradient boosting, memory efficient\n"
                                   "• CatBoost: Optimized for categorical features\n"
                                   "• ExtraTrees: Faster than Random Forest\n"
                                   "• SVR: Effective for high-dimensional data\n"
                                   "• MLPRegressor: Neural network for complex patterns\n"
                                   "• BayesianRidge: Bayesian linear with uncertainty")
        advanced_layout.addRow("Surrogate Model:", self.model_combo)
        
        # Acquisition function
        self.acquisition_combo = QComboBox()
        self.acquisition_combo.addItems(["ExpectedImprovement", "UpperConfidenceBound"])
        self.acquisition_combo.setCurrentText("ExpectedImprovement")
        self.acquisition_combo.currentTextChanged.connect(self.on_acquisition_changed)
        advanced_layout.addRow("Acquisition Strategy:", self.acquisition_combo)
        
        # Bootstrap iterations
        self.bootstrap_spin = QSpinBox()
        self.bootstrap_spin.setRange(10, 2000)
        self.bootstrap_spin.setValue(100)
        self.bootstrap_spin.setToolTip("Bootstrap iterations (for RandomForest and XGBoost only)")
        advanced_layout.addRow("Bootstrap Iterations:", self.bootstrap_spin)
        
        # UCB Kappa (only visible when UCB is selected)
        self.kappa_widget = QWidget()
        kappa_layout = QHBoxLayout(self.kappa_widget)
        kappa_layout.setContentsMargins(0, 0, 0, 0)
        
        self.kappa_slider = QSlider(Qt.Horizontal)
        self.kappa_slider.setRange(1, 50)  # 0.1 to 5.0 with 0.1 step
        self.kappa_slider.setValue(26)  # 2.6
        self.kappa_slider.valueChanged.connect(self.update_kappa_label)
        
        self.kappa_label = QLabel("2.6")
        self.kappa_label.setMinimumWidth(30)
        
        kappa_layout.addWidget(self.kappa_slider)
        kappa_layout.addWidget(self.kappa_label)
        
        advanced_layout.addRow("Exploration Factor κ:", self.kappa_widget)
        self.kappa_widget.setVisible(False)
        
        # Iterations spinner
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100)
        self.iterations_spin.setValue(10)
        self.iterations_spin.setToolTip("Number of optimization iterations to perform")
        advanced_layout.addRow("Iterations:", self.iterations_spin)
        
        layout.addWidget(self.advanced_group)
        
        # Enhanced recommendation controls
        recommendation_group = QGroupBox("Enhanced Recommendation Settings")
        rec_layout = QFormLayout(recommendation_group)
        
        # Recommendation mode
        rec_mode_widget = QWidget()
        rec_mode_layout = QHBoxLayout(rec_mode_widget)
        rec_mode_layout.setContentsMargins(0, 0, 0, 0)
        
        self.single_rec_radio = QRadioButton("Single Best")
        self.batch_rec_radio = QRadioButton("Batch")
        self.single_rec_radio.setChecked(True)
        self.single_rec_radio.toggled.connect(self.on_recommendation_mode_changed)
        
        rec_mode_layout.addWidget(self.single_rec_radio)
        rec_mode_layout.addWidget(self.batch_rec_radio)
        rec_layout.addRow("Recommendation Mode:", rec_mode_widget)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(2, 20)
        self.batch_size_spin.setValue(5)
        self.batch_size_spin.setEnabled(False)
        self.batch_size_spin.setToolTip("Number of candidates to recommend in batch mode")
        rec_layout.addRow("Batch Size:", self.batch_size_spin)
        
        # Confidence threshold
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setToolTip("Minimum confidence threshold for recommendations")
        self.confidence_label = QLabel("0.50")
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        
        conf_widget = QWidget()
        conf_layout = QHBoxLayout(conf_widget)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(self.confidence_label)
        rec_layout.addRow("Confidence Threshold:", conf_widget)
        
        layout.addWidget(recommendation_group)
        
        # Dynamic budget and early stopping
        budget_group = QGroupBox("Dynamic Budget & Early Stopping")
        budget_layout = QFormLayout(budget_group)
        
        # Enable early stopping
        self.early_stopping_checkbox = QCheckBox("Enable Early Stopping")
        self.early_stopping_checkbox.setChecked(True)
        self.early_stopping_checkbox.setToolTip("Automatically stop when convergence is detected")
        budget_layout.addRow(self.early_stopping_checkbox)
        
        # Convergence threshold
        self.convergence_spin = QDoubleSpinBox()
        self.convergence_spin.setRange(0.001, 1.0)
        self.convergence_spin.setValue(0.01)
        self.convergence_spin.setSingleStep(0.001)
        self.convergence_spin.setDecimals(3)
        self.convergence_spin.setToolTip("Threshold for detecting convergence")
        budget_layout.addRow("Convergence Threshold:", self.convergence_spin)
        
        # Max stagnation iterations
        self.stagnation_spin = QSpinBox()
        self.stagnation_spin.setRange(2, 20)
        self.stagnation_spin.setValue(3)
        self.stagnation_spin.setToolTip("Maximum iterations without improvement before stopping")
        budget_layout.addRow("Max Stagnation:", self.stagnation_spin)
        
        # Budget allocation strategy
        self.budget_combo = QComboBox()
        self.budget_combo.addItems(["adaptive", "uniform", "frontloaded"])
        self.budget_combo.setToolTip("Strategy for allocating computational budget")
        budget_layout.addRow("Budget Strategy:", self.budget_combo)
        
        layout.addWidget(budget_group)
        
        # Constraint optimization
        constraint_group = QGroupBox("Constraint Optimization")
        constraint_layout = QFormLayout(constraint_group)
        
        # Enable constraints
        self.enable_constraints_checkbox = QCheckBox("Enable Constraint Optimization")
        self.enable_constraints_checkbox.setToolTip("Apply constraints during optimization")
        constraint_layout.addRow(self.enable_constraints_checkbox)
        
        # Constraint configuration button
        self.configure_constraints_button = QPushButton("Configure Constraints")
        self.configure_constraints_button.clicked.connect(self.show_constraint_dialog)
        self.configure_constraints_button.setEnabled(False)
        self.enable_constraints_checkbox.toggled.connect(
            self.configure_constraints_button.setEnabled
        )
        constraint_layout.addRow(self.configure_constraints_button)
        
        layout.addWidget(constraint_group)
        
        # Advanced uncertainty quantification
        uncertainty_group = QGroupBox("Advanced Uncertainty Quantification")
        uncertainty_layout = QFormLayout(uncertainty_group)
        
        # Uncertainty method
        self.uncertainty_combo = QComboBox()
        self.uncertainty_combo.addItems([
            "ensemble", "mc_dropout", "gaussian_process", "deep_ensemble"
        ])
        self.uncertainty_combo.setToolTip(
            "Method for uncertainty quantification:\n"
            "• ensemble: Bootstrap ensemble\n"
            "• mc_dropout: Monte Carlo Dropout\n"
            "• gaussian_process: GP-based uncertainty\n"
            "• deep_ensemble: Deep ensemble method"
        )
        uncertainty_layout.addRow("Uncertainty Method:", self.uncertainty_combo)
        
        # Ensemble size
        self.ensemble_size_spin = QSpinBox()
        self.ensemble_size_spin.setRange(3, 50)
        self.ensemble_size_spin.setValue(10)
        self.ensemble_size_spin.setToolTip("Number of models in ensemble")
        uncertainty_layout.addRow("Ensemble Size:", self.ensemble_size_spin)
        
        layout.addWidget(uncertainty_group)
        
        # Run button
        self.run_button = QPushButton("Start Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        
        # Cancel button (initially hidden)
        self.cancel_button = QPushButton("Cancel Analysis")
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.cancel_button.setVisible(False)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch at the end
        layout.addStretch()
        
        return panel
    
    def create_results_panel(self):
        """Create the right results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget
        self.results_tabs = QTabWidget()
        
        # Tab 1: Recommendation & Visualization
        self.recommendation_tab = self.create_recommendation_tab()
        self.results_tabs.addTab(self.recommendation_tab, "Core Recommendation")
        
        # Tab 2: Data Explorer
        self.data_tab = self.create_data_explorer_tab()
        self.results_tabs.addTab(self.data_tab, "Data Explorer")
        
        # Tab 3: Model Insights
        self.insights_tab = self.create_model_insights_tab()
        self.results_tabs.addTab(self.insights_tab, "Model Insights")
        
        # Tab 4: Learning Process (real-time visualization)
        self.history_tab = self.create_history_tab()
        self.results_tabs.addTab(self.history_tab, "Learning Process")
        
        layout.addWidget(self.results_tabs)
        
        return panel
    
    def create_recommendation_tab(self):
        """Create the recommendation and visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Step 1: Current recommendation
        layout.addWidget(QLabel("Step 1: Current Best Recommendation:"))
        self.recommendation_text = QTextEdit()
        self.recommendation_text.setMaximumHeight(100)
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setStyleSheet("""
            QTextEdit {
                background-color: #e8f4fd;
                border: 2px solid #007bff;
                border-radius: 8px;
                font-size: 13px;
                padding: 10px;
            }
        """)
        self.recommendation_text.setText("Load data and run analysis to see the best recommendation here...")
        
        layout.addWidget(self.recommendation_text)
        
        # Step 2: Iterative feedback section (initially hidden)
        self.feedback_box = QGroupBox("Step 2: Provide Experimental Results and Continue Learning")
        self.feedback_layout = QFormLayout()
        self.feedback_box.setLayout(self.feedback_layout)
        self.feedback_box.setStyleSheet("""
            QGroupBox {
                background-color: #f8f9fa;
                border: 2px solid #28a745;
                border-radius: 8px;
                font-weight: bold;
                padding-top: 15px;
            }
            QGroupBox::title {
                color: #28a745;
                font-size: 14px;
            }
        """)
        self.feedback_box.hide()  # Initially hidden
        layout.addWidget(self.feedback_box)
        
        # Session status section
        self.session_status_box = QGroupBox("Session Status")
        session_layout = QFormLayout()
        self.session_status_box.setLayout(session_layout)
        
        self.session_status_label = QLabel("No active session")
        self.session_status_label.setStyleSheet("color: #6c757d; font-size: 9pt; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
        session_layout.addRow("Status:", self.session_status_label)
        
        layout.addWidget(self.session_status_box)
        
        # Exploration map
        layout.addWidget(QLabel("Exploration Map:"))
        self.exploration_plot = MatplotlibWidget()
        layout.addWidget(self.exploration_plot)
        
        return tab
    
    def create_data_explorer_tab(self):
        """Create the data explorer tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Search and filter controls
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search data...")
        self.search_box.textChanged.connect(self.filter_table)
        
        self.show_top_spin = QSpinBox()
        self.show_top_spin.setRange(10, 1000)
        self.show_top_spin.setValue(50)
        self.show_top_spin.setPrefix("Show top ")
        self.show_top_spin.setSuffix(" rows")
        self.show_top_spin.valueChanged.connect(self.update_table_display)
        
        controls_layout.addWidget(QLabel("Search:"))
        controls_layout.addWidget(self.search_box)
        controls_layout.addStretch()
        controls_layout.addWidget(self.show_top_spin)
        
        layout.addWidget(controls_widget)
        
        # Results table
        self.results_table = CustomTableWidget()
        layout.addWidget(self.results_table)
        
        return tab
    
    def create_model_insights_tab(self):
        """Create the model insights tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create sub-tabs for different types of analysis
        insights_tabs = QTabWidget()
        
        # Tab 1: Feature Importance
        importance_tab = QWidget()
        importance_layout = QVBoxLayout(importance_tab)
        
        importance_layout.addWidget(QLabel("Feature Importance:"))
        self.importance_plot = MatplotlibWidget()
        importance_layout.addWidget(self.importance_plot)
        
        # Model reliability section
        reliability_widget = QWidget()
        reliability_layout = QHBoxLayout(reliability_widget)
        
        self.reliability_button = QPushButton("Assess Model Reliability")
        self.reliability_button.clicked.connect(self.assess_model_reliability)
        self.reliability_button.setEnabled(False)
        
        self.reliability_label = QLabel("Not assessed")
        self.reliability_label.setStyleSheet("font-size: 9pt; color: #6c757d; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
        
        reliability_layout.addWidget(self.reliability_button)
        reliability_layout.addWidget(self.reliability_label)
        reliability_layout.addStretch()
        
        importance_layout.addWidget(reliability_widget)
        insights_tabs.addTab(importance_tab, "Feature Importance")
        
        # Tab 2: Feature Correlation
        correlation_tab = QWidget()
        correlation_layout = QVBoxLayout(correlation_tab)
        
        # Correlation controls
        correlation_controls = QWidget()
        correlation_controls_layout = QHBoxLayout(correlation_controls)
        
        correlation_button = QPushButton("Generate Correlation Matrix")
        correlation_button.clicked.connect(self.update_correlation_plot)
        correlation_controls_layout.addWidget(correlation_button)
        
        correlation_controls_layout.addStretch()
        correlation_layout.addWidget(correlation_controls)
        
        correlation_layout.addWidget(QLabel("Feature Correlation Matrix:"))
        self.correlation_plot = MatplotlibWidget()
        correlation_layout.addWidget(self.correlation_plot)
        
        insights_tabs.addTab(correlation_tab, "Feature Correlation")
        
        # Tab 3: Uncertainty Analysis
        uncertainty_tab = QWidget()
        uncertainty_layout = QVBoxLayout(uncertainty_tab)
        
        # Uncertainty controls
        uncertainty_controls = QWidget()
        uncertainty_controls_layout = QHBoxLayout(uncertainty_controls)
        
        uncertainty_button = QPushButton("Generate Uncertainty Analysis")
        uncertainty_button.clicked.connect(self.update_uncertainty_analysis)
        uncertainty_controls_layout.addWidget(uncertainty_button)
        
        uncertainty_controls_layout.addStretch()
        uncertainty_layout.addWidget(uncertainty_controls)
        
        uncertainty_layout.addWidget(QLabel("Model Uncertainty Analysis:"))
        self.uncertainty_plot = MatplotlibWidget()
        uncertainty_layout.addWidget(self.uncertainty_plot)
        
        insights_tabs.addTab(uncertainty_tab, "Uncertainty Analysis")
        
        # Tab 4: Design Space Visualization
        design_space_tab = QWidget()
        design_space_layout = QVBoxLayout(design_space_tab)
        
        # Design space controls
        design_controls = QWidget()
        design_controls_layout = QHBoxLayout(design_controls)
        
        # Feature selectors for X and Y axes
        design_controls_layout.addWidget(QLabel("X-axis:"))
        self.x_feature_combo = QComboBox()
        design_controls_layout.addWidget(self.x_feature_combo)
        
        design_controls_layout.addWidget(QLabel("Y-axis:"))
        self.y_feature_combo = QComboBox()
        design_controls_layout.addWidget(self.y_feature_combo)
        
        # Update button
        design_space_button = QPushButton("Update Design Space")
        design_space_button.clicked.connect(self.update_design_space_plot)
        design_controls_layout.addWidget(design_space_button)
        
        design_controls_layout.addStretch()
        design_space_layout.addWidget(design_controls)
        
        design_space_layout.addWidget(QLabel("Interactive Design Space Exploration:"))
        self.design_space_plot = MatplotlibWidget()
        design_space_layout.addWidget(self.design_space_plot)
        
        insights_tabs.addTab(design_space_tab, "Design Space")
        
        layout.addWidget(insights_tabs)
        
        return tab
    
    def create_history_tab(self):
        """Create the iterative learning history and process visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create tabs for different types of history
        history_tabs = QTabWidget()
        
        # Tab 1: Iterative Session History
        session_tab = QWidget()
        session_layout = QVBoxLayout(session_tab)
        
        # Session controls
        session_controls = QWidget()
        session_controls_layout = QHBoxLayout(session_controls)
        
        refresh_session_button = QPushButton("🔄 Refresh Session History")
        refresh_session_button.clicked.connect(self.update_history_tab)
        session_controls_layout.addWidget(refresh_session_button)
        
        session_controls_layout.addStretch()
        session_layout.addWidget(session_controls)
        
        # Iterative history table
        session_layout.addWidget(QLabel("Iteratively Acquired Data Points:"))
        self.history_table = CustomTableWidget()
        session_layout.addWidget(self.history_table)
        
        history_tabs.addTab(session_tab, "Session History")
        
        # Tab 2: Learning Curves
        learning_tab = QWidget()
        learning_layout = QVBoxLayout(learning_tab)
        
        # Learning curves controls
        learning_controls = QWidget()
        learning_controls_layout = QHBoxLayout(learning_controls)
        
        refresh_learning_button = QPushButton("🔄 Update Learning Curves")
        refresh_learning_button.clicked.connect(self.update_learning_curves)
        learning_controls_layout.addWidget(refresh_learning_button)
        
        # Add status info
        self.learning_status_label = QLabel("状态：等待迭代数据")
        self.learning_status_label.setStyleSheet("color: #6c757d; font-size: 9pt; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
        learning_controls_layout.addWidget(self.learning_status_label)
        
        learning_controls_layout.addStretch()
        learning_layout.addWidget(learning_controls)
        
        # Learning curves plot
        learning_layout.addWidget(QLabel("Model Performance Evolution:"))
        self.learning_curves_plot = MatplotlibWidget()
        learning_layout.addWidget(self.learning_curves_plot)
        
        history_tabs.addTab(learning_tab, "📈 Learning Curves")
        
        # Tab 3: Feature Evolution
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)
        
        # Feature evolution controls
        feature_controls = QWidget()
        feature_controls_layout = QHBoxLayout(feature_controls)
        
        refresh_feature_button = QPushButton("🔄 Update Feature Evolution")
        refresh_feature_button.clicked.connect(self.update_feature_evolution)
        feature_controls_layout.addWidget(refresh_feature_button)
        
        feature_controls_layout.addStretch()
        feature_layout.addWidget(feature_controls)
        
        # Feature importance evolution plot
        feature_layout.addWidget(QLabel("Feature Importance Evolution:"))
        self.feature_evolution_plot = MatplotlibWidget()
        feature_layout.addWidget(self.feature_evolution_plot)
        
        history_tabs.addTab(feature_tab, "🎯 Feature Evolution")
        
        # Tab 4: Exploration Analysis
        exploration_tab = QWidget()
        exploration_layout = QVBoxLayout(exploration_tab)
        
        # Exploration analysis controls
        exploration_controls = QWidget()
        exploration_controls_layout = QHBoxLayout(exploration_controls)
        
        refresh_exploration_button = QPushButton("🔄 Update Exploration Analysis")
        refresh_exploration_button.clicked.connect(self.update_exploration_analysis)
        exploration_controls_layout.addWidget(refresh_exploration_button)
        
        exploration_controls_layout.addStretch()
        exploration_layout.addWidget(exploration_controls)
        
        # Exploration vs exploitation plot
        exploration_layout.addWidget(QLabel("Exploration vs Exploitation Analysis:"))
        self.exploration_analysis_plot = MatplotlibWidget()
        exploration_layout.addWidget(self.exploration_analysis_plot)
        
        history_tabs.addTab(exploration_tab, "🔍 Exploration Analysis")
        
        # Tab 5: Real-time Process Visualization
        process_tab = QWidget()
        process_layout = QVBoxLayout(process_tab)
        
        # Process controls
        process_controls = QWidget()
        process_controls_layout = QHBoxLayout(process_controls)
        
        # Real-time toggle
        self.realtime_checkbox = QCheckBox("Real-time Updates")
        self.realtime_checkbox.setChecked(True)
        self.realtime_checkbox.setToolTip("Enable real-time process visualization during analysis")
        process_controls_layout.addWidget(self.realtime_checkbox)
        
        # Refresh button
        refresh_button = QPushButton("🔄 Refresh Process View")
        refresh_button.clicked.connect(self.update_process_visualization)
        process_controls_layout.addWidget(refresh_button)
        
        # Clear history button
        clear_button = QPushButton("🗑️ Clear History")
        clear_button.clicked.connect(self.clear_process_history)
        process_controls_layout.addWidget(clear_button)
        
        process_controls_layout.addStretch()
        process_layout.addWidget(process_controls)
        
        # Progress information
        self.progress_info = QLabel("No analysis running. Start an analysis to see the learning process.")
        self.progress_info.setStyleSheet("font-size: 9pt; color: #6c757d; padding: 10px; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
        process_layout.addWidget(self.progress_info)
        
        # Process visualization plot
        process_layout.addWidget(QLabel("Learning Process Visualization:"))
        self.process_plot = MatplotlibWidget()
        process_layout.addWidget(self.process_plot)
        
        history_tabs.addTab(process_tab, "Process Visualization")
        
        layout.addWidget(history_tabs)
        
        # Initialize process tracking
        self.process_history = {
            'iteration': [],
            'best_score': [],
            'current_score': [],
            'uncertainty': [],
            'acquisition_score': [],
            'timestamp': []
        }
        
        # Initialize empty state visualizations
        QTimer.singleShot(100, self.initialize_learning_visualizations)
        
        return tab
    
    def initialize_learning_visualizations(self):
        """Initialize learning visualizations with empty state."""
        self.update_learning_curves()
        self.update_feature_evolution()
        self.update_exploration_analysis()
    
    def load_training_data(self):
        """Load training data from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Data File", "", 
            "CSV files (*.csv);;Excel files (*.xlsx *.xls);;All files (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.training_data = pd.read_csv(file_path)
                else:
                    self.training_data = pd.read_excel(file_path)
                
                # Update UI
                filename = os.path.basename(file_path)
                self.training_label.setText(
                    f"✅ {filename}\n({len(self.training_data)} rows × {len(self.training_data.columns)} cols)"
                )
                self.training_label.setStyleSheet("color: #28a745; font-size: 9pt; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
                
                # Enable candidate generation button
                self.generate_button.setEnabled(True)
                
                # Update column selectors
                self.update_column_selectors()
                
                # Update run button state
                self.update_run_button_state()
                
                self.status_bar.showMessage(f"Training data loaded successfully: {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load training data:\n{str(e)}")
    
    def load_virtual_data(self):
        """Load virtual/candidate data from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Candidate Data File", "", 
            "CSV files (*.csv);;Excel files (*.xlsx *.xls);;All files (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.virtual_data = pd.read_csv(file_path)
                else:
                    self.virtual_data = pd.read_excel(file_path)
                
                # Update UI
                filename = os.path.basename(file_path)
                self.virtual_label.setText(
                    f"✅ {filename}\n({len(self.virtual_data)} rows × {len(self.virtual_data.columns)} cols)"
                )
                self.virtual_label.setStyleSheet("color: #28a745; font-size: 9pt; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
                
                # Update column selectors
                self.update_column_selectors()
                
                # Update run button state
                self.update_run_button_state()
                
                self.status_bar.showMessage(f"Candidate data loaded successfully: {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load candidate data:\n{str(e)}")
    
    def update_column_selectors(self):
        """Update target and feature column selectors."""
        # 🔍 添加调用栈跟踪
        import traceback
        print(f"\n=== 🔍 UPDATE_COLUMN_SELECTORS CALLED ===")
        print(f"Call stack (last 3 frames):")
        for line in traceback.format_stack()[-4:-1]:  # 排除当前frame
            print(f"  {line.strip()}")
        
        if self.training_data is not None:
            # Update target selector
            current_target = self.target_combo.currentText()
            self.target_combo.clear()
            numeric_columns = self.training_data.select_dtypes(include=[np.number]).columns.tolist()
            self.target_combo.addItems(numeric_columns)
            
            # Restore previous selection if possible
            if current_target in numeric_columns:
                self.target_combo.setCurrentText(current_target)
            
            # Update available targets for multi-objective mode
            self.update_available_targets()
            
            # Update feature selector
            if self.virtual_data is not None:
                # Find common columns between training and virtual data
                common_columns = list(set(self.training_data.columns) & set(self.virtual_data.columns))
                # 🔧 扩展数据类型支持，包含更多数值类型
                common_numeric_columns = []
                for col in common_columns:
                    dtype = self.training_data[col].dtype
                    # 支持更多的数值类型
                    if pd.api.types.is_numeric_dtype(dtype):
                        common_numeric_columns.append(col)
                        print(f"  ✓ {col}: {dtype} (numeric)")
                    else:
                        print(f"  ✗ {col}: {dtype} (non-numeric, excluded)")
                
                # 🔍 添加详细的调试信息
                print(f"\n=== 🔍 UPDATE_COLUMN_SELECTORS DEBUG ===")
                print(f"Training data columns ({len(self.training_data.columns)}): {list(self.training_data.columns)}")
                print(f"Virtual data columns ({len(self.virtual_data.columns)}): {list(self.virtual_data.columns)}")
                print(f"Common columns ({len(common_columns)}): {common_columns}")
                print(f"Common numeric columns ({len(common_numeric_columns)}): {common_numeric_columns}")
                
                # Remove target columns from features
                excluded_targets = set()
                
                # For single-objective mode
                if not self.is_multi_objective:
                    current_target = self.target_combo.currentText()
                    if current_target:
                        excluded_targets.add(current_target)
                        print(f"Single-objective target to exclude: {current_target}")
                        
                        # 🔍 调试信息：检查目标选择状态
                        print(f"Target combo has {self.target_combo.count()} items:")
                        for i in range(self.target_combo.count()):
                            item_text = self.target_combo.itemText(i)
                            is_current = (i == self.target_combo.currentIndex())
                            print(f"  [{i}] {item_text} {'← CURRENT' if is_current else ''}")
                
                # For multi-objective mode, exclude all selected targets
                if self.is_multi_objective:
                    targets, _ = self.get_selected_targets_and_goals()
                    excluded_targets.update(targets)
                    print(f"Multi-objective targets to exclude: {targets}")
                
                # 🔧 修复：只排除实际选择的目标列，不硬编码排除潜在的特征
                # 移除硬编码的common_targets过滤，这会错误地排除有用的特征
                # 如果用户真的选择了'sigma'等作为目标，会在上面的逻辑中被正确排除
                print(f"NOT applying hardcoded target exclusions - allowing all valid features")
                
                print(f"All excluded targets: {excluded_targets}")
                
                feature_columns = [col for col in common_numeric_columns if col not in excluded_targets]
                
                # 🆕 新增：智能特征排序功能
                feature_columns = self._sort_features(feature_columns)
                
                print(f"Final feature columns ({len(feature_columns)}): {feature_columns}")
                print(f"Expected feature count: {len(feature_columns)}")  # 🔧 修复：使用实际计算的特征数量
                
                # Update feature list
                self.feature_list.clear()
                for col in feature_columns:
                    item = QListWidgetItem(col)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked)
                    self.feature_list.addItem(item)
                    
                print(f"Added {len(feature_columns)} features to UI feature list")
                
                # Update design space feature selectors
                if hasattr(self, 'x_feature_combo') and hasattr(self, 'y_feature_combo'):
                    current_x = self.x_feature_combo.currentText()
                    current_y = self.y_feature_combo.currentText()
                    
                    self.x_feature_combo.clear()
                    self.y_feature_combo.clear()
                    
                    self.x_feature_combo.addItems(feature_columns)
                    self.y_feature_combo.addItems(feature_columns)
                    
                    # Restore previous selections if possible
                    if current_x in feature_columns:
                        self.x_feature_combo.setCurrentText(current_x)
                    elif feature_columns:
                        self.x_feature_combo.setCurrentIndex(0)
                    
                    if current_y in feature_columns:
                        self.y_feature_combo.setCurrentText(current_y)
                    elif len(feature_columns) > 1:
                        self.y_feature_combo.setCurrentIndex(1)
                    elif feature_columns:
                        self.y_feature_combo.setCurrentIndex(0)
    
    def select_all_features(self):
        """Select all features."""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setCheckState(Qt.Checked)
    
    def select_no_features(self):
        """Deselect all features."""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setCheckState(Qt.Unchecked)
    
    def get_selected_features(self):
        """Get list of selected features."""
        selected_features = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_features.append(item.text())
        return selected_features
    
    def _sort_features(self, feature_columns):
        """Sort features according to the selected sorting method."""
        if not hasattr(self, 'feature_sort_combo'):
            # If sorting combo doesn't exist yet, return as-is
            return feature_columns
            
        sort_method = self.feature_sort_combo.currentText()
        
        if sort_method == "A-Z":
            return sorted(feature_columns)
        elif sort_method == "Z-A":
            return sorted(feature_columns, reverse=True)
        elif sort_method == "元素优先":
            return self._sort_elements_first(feature_columns)
        elif sort_method == "工艺优先":
            return self._sort_process_first(feature_columns)
        else:  # 原始顺序
            return feature_columns
    
    def _sort_elements_first(self, feature_columns):
        """Sort with chemical elements first, followed by other features."""
        # 定义化学元素和工艺参数
        element_names = ['Al', 'Co', 'Cr', 'Fe', 'Ni', 'Ta', 'Mo', 'W', 'Ti', 'V', 'Nb', 'Mn', 'Cu', 'Zn', 'Si', 'C', 'N', 'O', 'H', 'S', 'P']
        process_params = [
            'recrystalize_K', 'annealing_K', 'aging_K', 'homogenize_K', 
            'h_time_h', 'ag_time_h', 'temp', 'time'
        ]
        material_props = [
            'sigma', 'ys_Mpa', 'vec', 'delta_r', 'd_r', 'eeta', 'wf_sixsq', 'CR_01'
        ]
        
        # 分类特征
        elements = []
        processes = []
        materials = []
        others = []
        
        for col in feature_columns:
            if any(elem in col for elem in element_names):
                elements.append(col)
            elif any(proc in col for proc in process_params):
                processes.append(col)
            elif any(mat in col for mat in material_props):
                materials.append(col)
            else:
                others.append(col)
        
        # 分别排序后合并
        return sorted(elements) + sorted(materials) + sorted(processes) + sorted(others)
    
    def _sort_process_first(self, feature_columns):
        """Sort with process parameters first, followed by other features."""
        # 定义工艺参数
        process_params = [
            'recrystalize_K', 'annealing_K', 'aging_K', 'homogenize_K', 
            'h_time_h', 'ag_time_h', 'temp', 'time'
        ]
        element_names = ['Al', 'Co', 'Cr', 'Fe', 'Ni', 'Ta', 'Mo', 'W', 'Ti', 'V', 'Nb', 'Mn', 'Cu', 'Zn', 'Si', 'C', 'N', 'O', 'H', 'S', 'P']
        material_props = [
            'sigma', 'ys_Mpa', 'vec', 'delta_r', 'd_r', 'eeta', 'wf_sixsq', 'CR_01'
        ]
        
        # 分类特征
        processes = []
        elements = []
        materials = []
        others = []
        
        for col in feature_columns:
            if any(proc in col for proc in process_params):
                processes.append(col)
            elif any(elem in col for elem in element_names):
                elements.append(col)
            elif any(mat in col for mat in material_props):
                materials.append(col)
            else:
                others.append(col)
        
        # 分别排序后合并
        return sorted(processes) + sorted(elements) + sorted(materials) + sorted(others)
    
    def on_feature_sort_changed(self):
        """Handle feature sorting change."""
        if hasattr(self, 'feature_sort_combo'):
            current_method = self.feature_sort_combo.currentText()
            print(f"🔄 特征排序已切换到: {current_method}")
            
            # Apply new sorting if data is loaded
            if hasattr(self, 'training_data') and self.training_data is not None:
                # 保存当前选择状态
                selected_features = self.get_selected_features()
                
                # 重新更新列选择器（会应用新的排序）
                self.update_column_selectors()
                
                # 恢复选择状态
                for i in range(self.feature_list.count()):
                    item = self.feature_list.item(i)
                    if item.text() in selected_features:
                        item.setCheckState(Qt.Checked)
                    else:
                        item.setCheckState(Qt.Unchecked)
    
    def on_acquisition_changed(self):
        """Handle acquisition function change."""
        is_ucb = self.acquisition_combo.currentText() == "UpperConfidenceBound"
        self.kappa_widget.setVisible(is_ucb)
    
    def update_kappa_label(self):
        """Update kappa value label."""
        value = self.kappa_slider.value() / 10.0
        self.kappa_label.setText(f"{value:.1f}")
    
    def on_recommendation_mode_changed(self):
        """Handle recommendation mode change."""
        is_batch = self.batch_rec_radio.isChecked()
        self.batch_size_spin.setEnabled(is_batch)
        self.recommendation_mode = "batch" if is_batch else "single"
    
    def update_confidence_label(self):
        """Update confidence threshold label."""
        value = self.confidence_slider.value() / 100.0
        self.confidence_label.setText(f"{value:.2f}")
        self.confidence_threshold = value
    
    def on_mode_changed(self):
        """Handle optimization mode change between single and multi-objective."""
        self.is_multi_objective = self.multi_obj_radio.isChecked()
        
        # Show/hide appropriate widgets
        self.single_obj_widget.setVisible(not self.is_multi_objective)
        self.multi_obj_widget.setVisible(self.is_multi_objective)
        
        # Update available targets for multi-objective mode
        if self.is_multi_objective:
            self.update_available_targets()
        
        # Update run button state
        self.update_run_button_state()
    
    def add_target(self):
        """Add selected target to multi-objective optimization."""
        current_item = self.available_targets_list.currentItem()
        if current_item is None:
            return
        
        target_name = current_item.text()
        
        # Check if already added
        for row in range(self.selected_targets_table.rowCount()):
            if self.selected_targets_table.item(row, 0).text() == target_name:
                return
        
        # Add to selected targets table
        row = self.selected_targets_table.rowCount()
        self.selected_targets_table.insertRow(row)
        
        self.selected_targets_table.setItem(row, 0, QTableWidgetItem(target_name))
        
        # Add goal selection combobox
        goal_combo = QComboBox()
        goal_combo.addItems(["maximize", "minimize"])
        goal_combo.setCurrentText("maximize")
        self.selected_targets_table.setCellWidget(row, 1, goal_combo)
        
        self.update_run_button_state()
    
    def remove_target(self):
        """Remove selected target from multi-objective optimization."""
        current_row = self.selected_targets_table.currentRow()
        if current_row >= 0:
            self.selected_targets_table.removeRow(current_row)
            self.update_run_button_state()
    
    def update_available_targets(self):
        """Update the available targets list for multi-objective mode."""
        self.available_targets_list.clear()
        
        if self.training_data is not None:
            # Get numeric columns (potential targets)
            numeric_columns = self.training_data.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_columns:
                self.available_targets_list.addItem(col)
    
    def get_selected_targets_and_goals(self):
        """Get selected targets and their goals for multi-objective optimization."""
        targets = []
        goals = []
        
        for row in range(self.selected_targets_table.rowCount()):
            target = self.selected_targets_table.item(row, 0).text()
            goal_widget = self.selected_targets_table.cellWidget(row, 1)
            goal = goal_widget.currentText()
            
            targets.append(target)
            goals.append(goal)
        
        return targets, goals

    def update_run_button_state(self):
        """Update the state of the run button based on data availability."""
        has_training = self.training_data is not None
        has_virtual = self.virtual_data is not None
        has_features = len(self.get_selected_features()) > 0
        
        if self.is_multi_objective:
            has_targets = self.selected_targets_table.rowCount() > 0
        else:
            has_targets = self.target_combo.currentText() != ""
        
        self.run_button.setEnabled(has_training and has_virtual and has_targets and has_features)
    
    def run_analysis(self):
        """Start the analysis in a separate thread to prevent UI freezing."""
        # Prevent multiple simultaneous analyses
        if self.is_analysis_running:
            QMessageBox.information(self, "Information", "Analysis is already running. Please wait for completion.")
            return
        
        try:
            # Initialize session if it doesn't exist
            if self.session is None:
                self.session = ActiveLearningSession(self.training_data)
                self.update_session_status()
            
            # Get configuration
            feature_columns = self.get_selected_features()
            model_type = self.model_combo.currentText()
            acquisition_function = self.acquisition_combo.currentText()
            n_iterations = self.bootstrap_spin.value()
            
            # 🔍 添加详细的特征选择调试信息
            print(f"\n=== 🔍 UI FEATURE SELECTION DEBUG ===")
            print(f"Training data shape: {self.training_data.shape if self.training_data is not None else 'None'}")
            print(f"Virtual data shape: {self.virtual_data.shape if self.virtual_data is not None else 'None'}")
            print(f"Total items in feature_list: {self.feature_list.count()}")
            print(f"Selected features count: {len(feature_columns)}")
            print(f"Selected features: {feature_columns}")
            
            # 检查特征列表的详细状态
            print(f"Feature list details:")
            for i in range(self.feature_list.count()):
                item = self.feature_list.item(i)
                is_checked = item.checkState() == Qt.Checked
                print(f"  [{i}] {item.text()}: {'✓' if is_checked else '✗'}")
            
            # 检查训练数据中的列
            if self.training_data is not None:
                print(f"Training data columns: {list(self.training_data.columns)}")
                print(f"Training data column count: {len(self.training_data.columns)}")
                
                # 检查特征是否存在于训练数据中
                missing_features = [f for f in feature_columns if f not in self.training_data.columns]
                if missing_features:
                    print(f"⚠️ WARNING: Features not found in training data: {missing_features}")
                else:
                    print(f"✅ All selected features found in training data")
            
            if not feature_columns:
                QMessageBox.warning(self, "Warning", "Please select at least one feature variable")
                return
            
            # Get targets and goals based on mode
            if self.is_multi_objective:
                target_columns, goal_directions = self.get_selected_targets_and_goals()
                if not target_columns:
                    QMessageBox.warning(self, "Warning", "Please select at least one target variable for multi-objective optimization")
                    return
            else:
                target_columns = [self.target_combo.currentText()]
                goal_directions = ['maximize' if self.maximize_radio.isChecked() else 'minimize']
                
                if not target_columns[0]:
                    QMessageBox.warning(self, "Warning", "Please select a target variable")
                    return
            
            # Prepare acquisition configuration
            acquisition_config = {}
            if acquisition_function == "UpperConfidenceBound":
                acquisition_config['kappa'] = self.kappa_slider.value() / 10.0
            
            # Clear previous process history
            self.clear_process_history()
            
            # Set analysis state
            self.is_analysis_running = True
            self.run_button.setEnabled(False)
            self.run_button.setText("Analyzing...")
            self.cancel_button.setVisible(True)
            
            # Create worker thread
            self.analysis_thread = QThread()
            self.analysis_worker = AnalysisWorker()
            
            # Use combined training data from session (includes iteratively acquired points)
            combined_training_data = self.session.get_combined_training_data()
            
            # Setup analysis parameters
            self.analysis_worker.setup_analysis(
                training_data=combined_training_data,  # Use combined data instead of original
                virtual_data=self.virtual_data,
                target_columns=target_columns,
                feature_columns=feature_columns,
                goal_directions=goal_directions,
                is_multi_objective=self.is_multi_objective,
                model_type=model_type,
                acquisition_function=acquisition_function,
                acquisition_config=acquisition_config,
                n_iterations=n_iterations
            )
            
            # Move worker to thread
            self.analysis_worker.moveToThread(self.analysis_thread)
            
            # Connect signals
            self.analysis_thread.started.connect(self.analysis_worker.run_analysis)
            self.analysis_worker.finished.connect(self.handle_analysis_completion)
            self.analysis_worker.error.connect(self.handle_analysis_error)
            self.analysis_worker.progress.connect(self.update_progress_info)
            
            # Ensure thread cleanup
            self.analysis_worker.finished.connect(self.analysis_thread.quit)
            self.analysis_worker.finished.connect(self.analysis_worker.deleteLater)
            self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)
            
            # Start the thread
            self.analysis_thread.start()
            
            # Update status
            self.status_bar.showMessage("Running enhanced active learning analysis...")
            self.update_progress_info("Starting analysis in background thread...")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start analysis:\n{str(e)}")
            self.reset_analysis_state()
    
    def handle_analysis_completion(self, results_data):
        """Handle successful analysis completion."""
        try:
            # Store results and optimizer
            self.results_data = results_data
            self.optimizer = self.analysis_worker.get_optimizer()
            
            # Check for poor model performance and trigger auto-tuning if needed
            if self.session is not None:
                try:
                    combined_data = self.session.get_combined_training_data()
                    feature_columns = self.get_selected_features()
                    
                    if self.is_multi_objective:
                        targets_list, goals_list = self.get_selected_targets_and_goals()
                        target_col = targets_list[0] if targets_list else None
                    else:
                        target_col = self.target_combo.currentText()
                    
                    if target_col and target_col in combined_data.columns:
                        # Quick R² assessment
                        current_r2 = self.optimizer.assess_model_reliability(
                            combined_data, target_col, feature_columns
                        )
                        
                        print(f"Post-analysis R² assessment: {current_r2:.4f}")
                        
                        # Trigger tuning for very poor performance
                        if current_r2 is not None and current_r2 < 0.0:
                            print(f"CRITICAL: Negative R² detected ({current_r2:.4f}), triggering immediate optimization...")
                            
                            tuned_params = self.optimizer.auto_tune_model(
                                training_df=combined_data,
                                target_columns=target_col,
                                feature_columns=feature_columns
                            )
                            
                            if tuned_params:
                                self.optimizer.apply_optimized_parameters(tuned_params)
                                print("Applied emergency hyperparameter optimization")
                                
                except Exception as e:
                    print(f"Warning: Post-analysis tuning check failed: {e}")
            
            # Update UI with results
            self.update_results_display()
            self.update_progress_info("Analysis completed successfully!")
            self.status_bar.showMessage("Analysis completed!")
            
            # Store the top recommendation and set up feedback UI
            if not results_data.empty:
                self.last_recommendation = results_data.iloc[0]
                self.setup_batch_feedback_ui()
            else:
                self.feedback_box.hide()
            
            # Update session status and history
            self.update_session_status()
            self.update_history_tab()
            
            # Show completion message
            QMessageBox.information(self, "Success", "Analysis completed successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error updating results:\n{str(e)}")
        
        finally:
            self.reset_analysis_state()
    
    def handle_analysis_error(self, error_message):
        """Handle analysis error."""
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.status_bar.showMessage("Analysis failed")
        self.update_progress_info(f"Analysis failed: {error_message}")
        self.reset_analysis_state()
    
    def cancel_analysis(self):
        """Cancel the running analysis."""
        if not self.is_analysis_running:
            return
        
        reply = QMessageBox.question(
            self, "Cancel Analysis", 
            "Are you sure you want to cancel the running analysis?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.analysis_worker:
                # Request graceful stop
                self.analysis_worker.stop_analysis()
            
            if self.analysis_thread and self.analysis_thread.isRunning():
                # Give the thread some time to stop gracefully
                self.analysis_thread.quit()
                if not self.analysis_thread.wait(5000):  # Wait up to 5 seconds
                    # Force terminate if still running
                    self.analysis_thread.terminate()
                    self.analysis_thread.wait()
            
            self.status_bar.showMessage("Analysis cancelled by user")
            self.update_progress_info("Analysis cancelled by user")
            self.reset_analysis_state()
    
    def reset_analysis_state(self):
        """Reset the analysis state and UI elements."""
        self.is_analysis_running = False
        self.run_button.setEnabled(True)
        self.run_button.setText("Start Analysis")
        self.cancel_button.setVisible(False)
        
        # Clean up thread references
        self.analysis_thread = None
        self.analysis_worker = None
    
    def update_results_display(self):
        """Update all result displays with individual error handling."""
        if self.results_data is None:
            return
        
        # Update each component with individual error handling
        update_methods = [
            ("recommendation text", self.update_recommendation_text),
            ("exploration plot", self.update_exploration_plot),
            ("data table", self.update_data_table),
            ("feature importance plot", self.update_feature_importance_plot),
            ("correlation matrix", self.update_correlation_plot),
            ("uncertainty analysis", self.update_uncertainty_analysis),
            ("design space visualization", self.update_design_space_plot),
        ]
        
        for name, method in update_methods:
            try:
                method()
            except Exception as e:
                print(f"Error updating {name}: {e}")
                # Continue with other updates
        
        # Update Pareto visualization for multi-objective results
        if self.is_multi_objective:
            try:
                # Only update if pareto plot exists
                if hasattr(self, 'pareto_plot'):
                    self.update_pareto_plot()
                else:
                    # Create pareto visualization if it doesn't exist
                    self.update_pareto_visualization(None, None)
            except Exception as e:
                print(f"Error updating Pareto visualization: {e}")
        
        # Enable reliability assessment
        self.reliability_button.setEnabled(True)
    
    def update_recommendation_text(self):
        """Update the recommendation text with enhanced features."""
        if self.results_data is None or len(self.results_data) == 0:
            self.recommendation_text.setText("No recommendations available. Please run analysis first.")
            return
        
        try:
            # Apply confidence filtering
            confidence_threshold = getattr(self, 'confidence_threshold', 0.0)
            filtered_results = self.results_data.copy()
            
            if confidence_threshold > 0 and 'acquisition_score' in filtered_results.columns:
                max_score = filtered_results['acquisition_score'].max()
                min_score = filtered_results['acquisition_score'].min()
                score_range = max_score - min_score
                
                if score_range > 0:
                    normalized_scores = (filtered_results['acquisition_score'] - min_score) / score_range
                    filtered_results = filtered_results[normalized_scores >= confidence_threshold]
                
                if len(filtered_results) == 0:
                    self.recommendation_text.setText(
                        f"No candidates meet the confidence threshold of {confidence_threshold:.2f}. "
                        "Consider lowering the threshold or running more iterations."
                    )
                    return
            
            # Check recommendation mode
            if getattr(self, 'recommendation_mode', 'single') == 'batch':
                batch_size = getattr(self, 'batch_size_spin', type('', (), {'value': lambda: 5})).value()
                recommendations = self.generate_batch_recommendations(filtered_results, batch_size)
                self._display_batch_recommendations(recommendations)
            else:
                # Single recommendation mode
                top_result = filtered_results.iloc[0]
                self._display_single_recommendation(top_result)
        
        except Exception as e:
            self.recommendation_text.setText(f"Error generating recommendations: {str(e)}")
    
    def _display_single_recommendation(self, top_result):
        """Display single best recommendation with enhanced features."""
        feature_columns = self.get_selected_features()
        
        if self.is_multi_objective:
            # Multi-objective recommendation
            targets, goals = self.get_selected_targets_and_goals()
            
            recommendation = f"🎯 BEST MULTI-OBJECTIVE RECOMMENDATION:\n\n"
            recommendation += f"• Sample ID: Row {top_result.name + 1}\n"
            recommendation += f"• Acquisition Score: {top_result['acquisition_score']:.6f}\n\n"
            
            # Add predictions for each objective
            recommendation += "Predicted Objectives:\n"
            for target, goal in zip(targets, goals):
                mean_col = f'{target}_predicted_mean'
                std_col = f'{target}_uncertainty_std'
                if mean_col in top_result.index and std_col in top_result.index:
                    recommendation += f"  • {target} ({goal}): {top_result[mean_col]:.4f} ± {top_result[std_col]:.4f}\n"
            
            # Add Pareto information if available
            if 'is_pareto_optimal' in self.results_data.columns:
                pareto_count = self.results_data['is_pareto_optimal'].sum()
                is_pareto = top_result.get('is_pareto_optimal', False)
                recommendation += f"\n{'✓ Pareto optimal' if is_pareto else '✗ Not Pareto optimal'}\n"
                recommendation += f"Total Pareto solutions: {pareto_count}\n"
            
        else:
            # Single-objective recommendation
            goal = 'maximize' if self.maximize_radio.isChecked() else 'minimize'
            target = self.target_combo.currentText()
            
            recommendation = f"🎯 BEST SINGLE-OBJECTIVE RECOMMENDATION:\n\n"
            recommendation += f"• Target: {target} ({goal})\n"
            recommendation += f"• Sample ID: Row {top_result.name + 1}\n"
            recommendation += f"• Acquisition Score: {top_result['acquisition_score']:.6f}\n\n"
            
            # Handle both new and old column naming
            if 'predicted_mean' in top_result.index:
                pred_mean = top_result['predicted_mean']
                uncertainty = top_result['uncertainty_std']
            else:
                mean_col = f'{target}_predicted_mean'
                std_col = f'{target}_uncertainty_std'
                pred_mean = top_result[mean_col] if mean_col in top_result.index else 'N/A'
                uncertainty = top_result[std_col] if std_col in top_result.index else 'N/A'
            
            pred_mean_str = f"{pred_mean:.4f}" if isinstance(pred_mean, (int, float)) else str(pred_mean)
            uncertainty_str = f"{uncertainty:.4f}" if isinstance(uncertainty, (int, float)) else str(uncertainty)
            
            recommendation += f"• Predicted {target}: {pred_mean_str}\n"
            recommendation += f"• Prediction Uncertainty: {uncertainty_str}\n"
        
        # Add feature values
        recommendation += f"\nRecommended Feature Values:\n"
        for feature in feature_columns[:5]:  # Show top 5 features
            if feature in top_result.index:
                value = top_result[feature]
                recommendation += f"  • {feature}: {value:.4f}\n"
        
        if len(feature_columns) > 5:
            recommendation += f"  ... and {len(feature_columns) - 5} more features\n"
        
        # Add feasibility score if constraints are enabled
        if getattr(self, 'enable_constraint_optimization', False):
            feasibility_scores = self.calculate_feasibility_scores(pd.DataFrame([top_result]))
            recommendation += f"\n• Feasibility Score: {feasibility_scores[0]:.3f}\n"
        
        # Add uncertainty method info
        uncertainty_method = getattr(self, 'uncertainty_combo', None)
        if uncertainty_method and hasattr(uncertainty_method, 'currentText'):
            method = uncertainty_method.currentText()
            recommendation += f"\n• Uncertainty Method: {method}\n"
        
        recommendation += f"\nThis design is recommended for maximum information gain.\n\n"
        recommendation += "📋 NEXT STEPS:\n"
        recommendation += "1. Perform experiment using the recommended formulation above\n"
        recommendation += "2. Record your measured target values in the Batch Feedback table below\n"
        recommendation += "3. Click 'Submit All Results' to add data and get next recommendations\n"
        
        self.recommendation_text.setText(recommendation)
    
    def _display_batch_recommendations(self, recommendations):
        """Display batch recommendations with enhanced features."""
        feature_columns = self.get_selected_features()
        
        if self.is_multi_objective:
            targets, goals = self.get_selected_targets_and_goals()
            target_text = ", ".join([f"{t} ({g})" for t, g in zip(targets, goals)])
        else:
            target = self.target_combo.currentText()
            goal = 'maximize' if self.maximize_radio.isChecked() else 'minimize'
            target_text = f"{target} ({goal})"
        
        recommendation = f"🎯 BATCH RECOMMENDATIONS (Top {len(recommendations)}):\n\n"
        recommendation += f"Targets: {target_text}\n"
        recommendation += f"Selection: Diversity-based for exploration coverage\n\n"
        
        for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
            recommendation += f"--- Recommendation #{i} ---\n"
            recommendation += f"Acquisition Score: {rec['acquisition_score']:.6f}\n"
            
            # Show top 3 features to save space
            important_features = feature_columns[:3] if len(feature_columns) > 3 else feature_columns
            for feature in important_features:
                if feature in rec:
                    recommendation += f"  {feature}: {rec[feature]:.4f}\n"
            
            if len(feature_columns) > 3:
                recommendation += f"  ... and {len(feature_columns) - 3} more features\n"
            
            # Add prediction
            if self.is_multi_objective:
                targets, _ = self.get_selected_targets_and_goals()
                if targets:
                    target = targets[0]
                    mean_col = f'{target}_predicted_mean'
                    if mean_col in rec.index:
                        recommendation += f"Predicted {target}: {rec[mean_col]:.4f}\n"
            else:
                if 'predicted_mean' in rec.index:
                    recommendation += f"Predicted: {rec['predicted_mean']:.4f}\n"
            
            # Add feasibility if enabled
            if getattr(self, 'enable_constraint_optimization', False):
                feasibility_scores = self.calculate_feasibility_scores(pd.DataFrame([rec]))
                recommendation += f"Feasibility: {feasibility_scores[0]:.3f}\n"
            
            recommendation += "\n"
        
        recommendation += "💡 See Data Explorer tab for complete feature values.\n\n"
        recommendation += "📋 NEXT STEPS:\n"
        recommendation += "1. Perform experiments using the recommended formulations above\n"
        recommendation += "2. Record your measured target values in the Batch Feedback table below\n"
        recommendation += "3. Click 'Submit All Results' to add data and get next recommendations\n"
        
        self.recommendation_text.setText(recommendation)
    
    def update_exploration_plot(self):
        """Update the exploration map plot."""
        if self.results_data is None:
            return
        
        self.exploration_plot.clear()
        ax = self.exploration_plot.figure.add_subplot(111)
        
        # Determine column names based on optimization mode
        if self.is_multi_objective:
            # For multi-objective, use the first target for visualization
            targets, _ = self.get_selected_targets_and_goals()
            if not targets:
                ax.text(0.5, 0.5, 'No targets selected', ha='center', va='center', transform=ax.transAxes)
                self.exploration_plot.canvas.draw()
                return
            target = targets[0]
            mean_col = f'{target}_predicted_mean'
            std_col = f'{target}_uncertainty_std'
        else:
            # For single-objective, use standard column names
            target = self.target_combo.currentText()
            mean_col = 'predicted_mean'
            std_col = 'uncertainty_std'
        
        # Check if required columns exist
        required_cols = [mean_col, std_col, 'acquisition_score']
        missing_cols = [col for col in required_cols if col not in self.results_data.columns]
        
        if missing_cols:
            ax.text(0.5, 0.5, f'Missing columns: {missing_cols}', ha='center', va='center', transform=ax.transAxes)
            self.exploration_plot.canvas.draw()
            return
        
        # Filter out NaN values
        valid_data = self.results_data.dropna(subset=required_cols)
        
        if len(valid_data) == 0:
            ax.text(0.5, 0.5, 'No valid data for plotting', ha='center', va='center', transform=ax.transAxes)
            self.exploration_plot.canvas.draw()
            return
        
        # Create scatter plot
        scatter = ax.scatter(
            valid_data[mean_col], 
            valid_data[std_col],
            c=valid_data['acquisition_score'],
            s=50,
            alpha=0.7,
            cmap='viridis'
        )
        
        # Highlight top point
        top_point = valid_data.iloc[0]
        ax.scatter(
            top_point[mean_col], 
            top_point[std_col],
            c='red', s=100, marker='*', 
            label='Recommended Point', edgecolors='darkred', linewidth=1
        )
        
        # Labels and title
        ax.set_xlabel(f'Predicted Mean ({target})')
        ax.set_ylabel('Prediction Uncertainty (Std Dev)')
        ax.set_title('Exploration Map: Performance vs Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = self.exploration_plot.figure.colorbar(scatter, ax=ax)
        cbar.set_label('Acquisition Score')
        
        self.exploration_plot.figure.tight_layout()
        self.exploration_plot.canvas.draw()
    
    def update_data_table(self):
        """Update the data explorer table."""
        if self.results_data is None:
            return
        
        # Show top N rows
        n_rows = self.show_top_spin.value()
        display_data = self.results_data.head(n_rows)
        
        # Set up table
        self.results_table.setRowCount(len(display_data))
        self.results_table.setColumnCount(len(display_data.columns))
        self.results_table.setHorizontalHeaderLabels(display_data.columns.tolist())
        
        # Populate table
        for i, row in display_data.iterrows():
            for j, value in enumerate(row):
                if pd.isna(value):
                    item = QTableWidgetItem("N/A")
                elif isinstance(value, (int, float)):
                    item = QTableWidgetItem(f"{value:.6f}" if abs(value) < 1000 else f"{value:.2e}")
                else:
                    item = QTableWidgetItem(str(value))
                
                # Highlight important columns
                col_name = display_data.columns[j]
                important_cols = ['acquisition_score']
                
                # Add prediction columns based on mode
                if self.is_multi_objective:
                    # For multi-objective, check for any target prediction columns
                    if '_predicted_mean' in col_name or '_uncertainty_std' in col_name:
                        important_cols.append(col_name)
                else:
                    # For single-objective, use standard names
                    important_cols.extend(['predicted_mean', 'uncertainty_std'])
                
                if col_name in important_cols:
                    item.setBackground(QColor(240, 248, 255))
                
                self.results_table.setItem(i, j, item)
        
        # Resize columns
        self.results_table.resizeColumnsToContents()
    
    def update_feature_importance_plot(self):
        """Update the feature importance plot."""
        if self.optimizer is None:
            return
        
        try:
            importance = self.optimizer.get_feature_importance()
            print(f"Feature importance type: {type(importance)}")
            if isinstance(importance, dict):
                print(f"Importance keys: {list(importance.keys())}")
                for key, value in importance.items():
                    print(f"  {key}: {type(value)}, length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
            else:
                print(f"Importance length: {len(importance) if hasattr(importance, '__len__') else 'N/A'}")
            
            self.importance_plot.clear()
            ax = self.importance_plot.figure.add_subplot(111)
            
            # Handle both single-objective (Series) and multi-objective (dict) cases
            if isinstance(importance, dict):
                # Multi-objective case: importance is a dict with target names as keys
                if len(importance) == 1:
                    # Single objective in dict format
                    importance_data = next(iter(importance.values()))
                else:
                    # Multiple objectives: calculate average importance across all targets
                    import pandas as pd
                    # Convert dict of Series to DataFrame, then calculate row means
                    importance_df = pd.DataFrame(importance)
                    importance_data = importance_df.mean(axis=1)
                    
                print(f"Multi-objective feature importance: {len(importance)} targets, {len(importance_data)} features")
            else:
                # Single-objective case: already a Series
                importance_data = importance
                print(f"Single-objective feature importance: {len(importance_data)} features")
            
            # Ensure we have valid data
            if importance_data is None or len(importance_data) == 0:
                ax.text(0.5, 0.5, 'No feature importance data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Importance (No Data)')
                self.importance_plot.canvas.draw()
                return
            
            # Sort by importance
            importance_sorted = importance_data.sort_values(ascending=True)
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(importance_sorted)), importance_sorted.values)
            ax.set_yticks(range(len(importance_sorted)))
            ax.set_yticklabels(importance_sorted.index)
            ax.set_xlabel('Feature Importance')
            
            # Set appropriate title
            if isinstance(importance, dict) and len(importance) > 1:
                ax.set_title('Average Feature Importance (Multi-Objective)')
            else:
                ax.set_title('Model Feature Importance Analysis')
            
            # Color bars
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.viridis(i / len(bars)))
            
            ax.grid(True, alpha=0.3, axis='x')
            
            self.importance_plot.figure.tight_layout()
            self.importance_plot.canvas.draw()
            
        except Exception as e:
            print(f"Error updating feature importance plot: {e}")
            # Show error message in plot
            self.importance_plot.clear()
            ax = self.importance_plot.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance (Error)')
            self.importance_plot.canvas.draw()
    
    def filter_table(self):
        """Filter the data table based on search text."""
        # This is a simple implementation - could be enhanced
        search_text = self.search_box.text().lower()
        
        for row in range(self.results_table.rowCount()):
            should_show = False
            if not search_text:
                should_show = True
            else:
                for col in range(self.results_table.columnCount()):
                    item = self.results_table.item(row, col)
                    if item and search_text in item.text().lower():
                        should_show = True
                        break
            
            self.results_table.setRowHidden(row, not should_show)
    
    def update_table_display(self):
        """Update the number of rows displayed in the table."""
        self.update_data_table()
    
    def assess_model_reliability(self):
        """Assess and display model reliability."""
        if self.optimizer is None or self.training_data is None:
            return
        
        try:
            target_column = self.target_combo.currentText()
            feature_columns = self.get_selected_features()
            
            self.reliability_button.setEnabled(False)
            self.reliability_button.setText("Assessing...")
            QApplication.processEvents()
            
            score = self.optimizer.assess_model_reliability(
                self.training_data, target_column, feature_columns
            )
            
            self.reliability_label.setText(f"Model R² Score: {score:.4f}")
            if score > 0.8:
                color = "#28a745"  # Green
                assessment = "Excellent"
            elif score > 0.6:
                color = "#ffc107"  # Yellow
                assessment = "Good"
            else:
                color = "#dc3545"  # Red
                assessment = "Needs Improvement"
            
            self.reliability_label.setStyleSheet(f"font-size: 9pt; color: {color}; font-weight: 600; font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif;")
            self.reliability_label.setText(f"Model Reliability: {assessment} (R² = {score:.4f})")
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Model reliability assessment failed:\n{str(e)}")
        
        finally:
            self.reliability_button.setEnabled(True)
            self.reliability_button.setText("Assess Model Reliability")
    
    def update_correlation_plot(self):
        """Update the feature correlation matrix plot."""
        if self.training_data is None:
            QMessageBox.warning(self, "Warning", "Please load training data first")
            return
        
        try:
            # Get selected features
            feature_columns = self.get_selected_features()
            
            if len(feature_columns) < 2:
                QMessageBox.warning(self, "Warning", "Please select at least 2 features for correlation analysis")
                return
            
            # Clear the plot
            self.correlation_plot.clear()
            
            # Create correlation matrix
            corr_data = self.training_data[feature_columns]
            correlation_matrix = corr_data.corr()
            
            # Create the plot with better spacing
            ax = self.correlation_plot.figure.add_subplot(111)
            
            # Calculate appropriate figure size based on number of features
            n_features = len(feature_columns)
            if n_features > 10:
                # For many features, use smaller annotation font
                annot_fontsize = 8
                cbar_shrink = 0.6
            else:
                annot_fontsize = 10
                cbar_shrink = 0.8
            
            # Create heatmap using seaborn with improved formatting
            sns.heatmap(correlation_matrix, 
                       annot=True,  # Show correlation values
                       cmap='RdBu_r',  # Red-Blue color scheme
                       center=0,  # Center colormap at 0
                       square=True,  # Square cells
                       fmt='.2f',  # Format numbers to 2 decimal places
                       cbar_kws={"shrink": cbar_shrink, "aspect": 20},  # Better colorbar
                       annot_kws={"size": annot_fontsize},  # Annotation font size
                       linewidths=0.5,  # Add grid lines
                       ax=ax)
            
            # Customize the plot
            ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            # Improve label readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
            
            # Adjust layout with extra space for labels
            self.correlation_plot.figure.subplots_adjust(bottom=0.15, left=0.15)
            self.correlation_plot.canvas.draw()
            
            self.status_bar.showMessage(f"Correlation matrix updated with {len(feature_columns)} features")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate correlation matrix:\n{str(e)}")
            print(f"Correlation plot error: {e}")  # Debug info
    
    def update_uncertainty_analysis(self):
        """Update the uncertainty analysis visualization."""
        if self.results_data is None:
            QMessageBox.warning(self, "Warning", "Please run analysis first to generate uncertainty data")
            return
        
        try:
            # Determine column names based on optimization mode
            if self.is_multi_objective:
                # For multi-objective, use the first target for uncertainty analysis
                targets, _ = self.get_selected_targets_and_goals()
                if not targets:
                    QMessageBox.warning(self, "Warning", "No targets selected for multi-objective optimization")
                    return
                target = targets[0]
                mean_col = f'{target}_predicted_mean'
                std_col = f'{target}_uncertainty_std'
            else:
                # For single-objective, use standard column names
                mean_col = 'predicted_mean'
                std_col = 'uncertainty_std'
            
            # Check if required columns exist
            required_columns = [mean_col, std_col, 'acquisition_score']
            missing_columns = [col for col in required_columns if col not in self.results_data.columns]
            
            if missing_columns:
                QMessageBox.warning(self, "Warning", 
                                   f"Missing required columns: {', '.join(missing_columns)}\n"
                                   "Please run the analysis first.")
                return
            
            # Clear the plot
            self.uncertainty_plot.clear()
            
            # Create figure with subplots
            fig = self.uncertainty_plot.figure
            
            # Three subplots in a row
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)  
            ax3 = fig.add_subplot(133)
            
            # Filter out NaN values
            valid_data = self.results_data.dropna(subset=required_columns)
            
            if len(valid_data) == 0:
                ax2.text(0.5, 0.5, 'No valid uncertainty data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                self.uncertainty_plot.canvas.draw()
                return
            
            # Plot 1: Uncertainty Distribution
            uncertainty_values = valid_data[std_col]
            ax1.hist(uncertainty_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Prediction Uncertainty (Std)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Uncertainty Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics
            mean_unc = uncertainty_values.mean()
            std_unc = uncertainty_values.std()
            ax1.axvline(mean_unc, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_unc:.3f}')
            ax1.legend()
            
            # Plot 2: Uncertainty vs Predicted Value
            scatter = ax2.scatter(valid_data[mean_col], 
                                valid_data[std_col],
                                c=valid_data['acquisition_score'], 
                                cmap='plasma', alpha=0.7, s=40)
            ax2.set_xlabel('Predicted Mean Value')
            ax2.set_ylabel('Prediction Uncertainty')
            ax2.set_title('Uncertainty vs Prediction')
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar for acquisition score
            cbar = fig.colorbar(scatter, ax=ax2, shrink=0.8)
            cbar.set_label('Acquisition Score', rotation=270, labelpad=15)
            
            # Highlight top 5 points
            top_5 = valid_data.head(5)
            ax2.scatter(top_5[mean_col], top_5[std_col],
                       s=100, facecolors='none', edgecolors='red', linewidths=2,
                       label='Top 5 Candidates')
            ax2.legend()
            
            # Plot 3: Acquisition Score Distribution
            acquisition_values = valid_data['acquisition_score']
            ax3.hist(acquisition_values, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax3.set_xlabel('Acquisition Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Acquisition Score Distribution')
            ax3.grid(True, alpha=0.3)
            
            # Add statistics
            mean_acq = acquisition_values.mean()
            top_5_threshold = acquisition_values.quantile(0.95)
            ax3.axvline(mean_acq, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_acq:.4f}')
            ax3.axvline(top_5_threshold, color='green', linestyle='--', linewidth=2,
                       label=f'95th %ile: {top_5_threshold:.4f}')
            ax3.legend()
            
            # Adjust layout
            fig.suptitle('Model Uncertainty Analysis', fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            # Refresh canvas
            self.uncertainty_plot.canvas.draw()
            
            # Update status
            self.status_bar.showMessage(f"Uncertainty analysis updated with {len(valid_data)} data points")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate uncertainty analysis:\n{str(e)}")
            print(f"Uncertainty analysis error: {e}")  # Debug info
    
    def update_design_space_plot(self):
        """Update the design space visualization."""
        if self.results_data is None:
            QMessageBox.warning(self, "Warning", "Please run analysis first to generate design space data")
            return
        
        try:
            # Get selected features for X and Y axes
            x_feature = self.x_feature_combo.currentText()
            y_feature = self.y_feature_combo.currentText()
            
            if not x_feature or not y_feature:
                QMessageBox.warning(self, "Warning", "Please select features for both X and Y axes")
                return
            
            if x_feature not in self.results_data.columns or y_feature not in self.results_data.columns:
                QMessageBox.warning(self, "Warning", f"Selected features not found in results data")
                return
            
            # Clear the plot
            self.design_space_plot.clear()
            
            # Create the plot
            ax = self.design_space_plot.figure.add_subplot(111)
            
            # Filter out NaN values
            valid_data = self.results_data.dropna(subset=[x_feature, y_feature, 'acquisition_score'])
            
            if len(valid_data) == 0:
                ax.text(0.5, 0.5, 'No valid data available for selected features', 
                       ha='center', va='center', transform=ax.transAxes)
                self.design_space_plot.canvas.draw()
                return
            
            # Create scatter plot colored by acquisition score
            scatter = ax.scatter(valid_data[x_feature], 
                               valid_data[y_feature],
                               c=valid_data['acquisition_score'], 
                               cmap='viridis', 
                               alpha=0.7, 
                               s=60,
                               edgecolors='black',
                               linewidth=0.5)
            
            # Add colorbar
            cbar = self.design_space_plot.figure.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Acquisition Score', rotation=270, labelpad=15)
            
            # Highlight top candidates
            top_candidates = valid_data.head(10)  # Top 10 candidates
            ax.scatter(top_candidates[x_feature], 
                      top_candidates[y_feature],
                      s=120, 
                      facecolors='none', 
                      edgecolors='red', 
                      linewidths=2,
                      label=f'Top 10 Candidates')
            
            # Highlight the best candidate
            best_candidate = valid_data.iloc[0]
            ax.scatter(best_candidate[x_feature], 
                      best_candidate[y_feature],
                      s=200, 
                      marker='*', 
                      c='gold', 
                      edgecolors='black',
                      linewidth=2,
                      label='Best Candidate')
            
            # Add training data if available
            if self.training_data is not None and x_feature in self.training_data.columns and y_feature in self.training_data.columns:
                train_valid = self.training_data.dropna(subset=[x_feature, y_feature])
                if len(train_valid) > 0:
                    ax.scatter(train_valid[x_feature], 
                              train_valid[y_feature],
                              alpha=0.5, 
                              s=30, 
                              c='gray', 
                              marker='^',
                              label='Training Data')
            
            # Customize the plot
            ax.set_xlabel(f'{x_feature}', fontsize=12)
            ax.set_ylabel(f'{y_feature}', fontsize=12)
            ax.set_title(f'Design Space: {x_feature} vs {y_feature}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Adjust layout
            self.design_space_plot.figure.tight_layout()
            
            # Refresh canvas
            self.design_space_plot.canvas.draw()
            
            # Update status
            self.status_bar.showMessage(f"Design space updated: {x_feature} vs {y_feature} ({len(valid_data)} points)")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate design space plot:\n{str(e)}")
            print(f"Design space plot error: {e}")  # Debug info
    
    def simulate_progressive_analysis(self, target_column, feature_columns, goal, acquisition_config, n_iterations):
        """Simulate progressive analysis with real-time updates."""
        import time
        import datetime
        
        # Simulate different phases of the analysis
        phases = [
            ("Preprocessing data...", 0.5),
            ("Training initial model...", 1.0),
            ("Calculating uncertainties...", 0.8),
            ("Computing acquisition scores...", 0.7),
            ("Optimizing candidates...", 1.2),
            ("Validating results...", 0.5)
        ]
        
        total_phases = len(phases)
        best_score_so_far = None
        
        for i, (phase_name, duration) in enumerate(phases):
            # Update progress info
            progress_percent = int((i / total_phases) * 100)
            self.update_progress_info(f"[{progress_percent}%] {phase_name}")
            
            # Simulate some work and generate mock progress data
            steps_per_phase = 5
            for step in range(steps_per_phase):
                time.sleep(duration / steps_per_phase)
                
                # Generate mock data for this step
                iteration = i * steps_per_phase + step + 1
                
                # Simulate improving performance over time
                if goal == 'maximize':
                    current_score = np.random.normal(5 + iteration * 0.1, 0.5)
                    if best_score_so_far is None or current_score > best_score_so_far:
                        best_score_so_far = current_score
                else:
                    current_score = np.random.normal(10 - iteration * 0.05, 0.3)
                    if best_score_so_far is None or current_score < best_score_so_far:
                        best_score_so_far = current_score
                
                uncertainty = max(0.1, np.random.exponential(0.5) * (1 - iteration/30))
                acquisition_score = uncertainty + abs(current_score - best_score_so_far) * 0.1
                
                # Add to process history
                self.process_history['iteration'].append(iteration)
                self.process_history['best_score'].append(best_score_so_far)
                self.process_history['current_score'].append(current_score)
                self.process_history['uncertainty'].append(uncertainty)
                self.process_history['acquisition_score'].append(acquisition_score)
                self.process_history['timestamp'].append(datetime.datetime.now())
                
                # Update visualization in real-time if enabled
                if hasattr(self, 'realtime_checkbox') and self.realtime_checkbox.isChecked():
                    self.update_process_visualization()
                    QApplication.processEvents()
        
        # Final update
        self.update_progress_info("Finalizing results...")
        self.update_process_visualization()
    
    def update_progress_info(self, message):
        """Update the progress information label."""
        if hasattr(self, 'progress_info'):
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.progress_info.setText(f"[{timestamp}] {message}")
            QApplication.processEvents()
    
    def clear_process_history(self):
        """Clear the process history data."""
        self.process_history = {
            'iteration': [],
            'best_score': [],
            'current_score': [],
            'uncertainty': [],
            'acquisition_score': [],
            'timestamp': []
        }
        if hasattr(self, 'process_plot'):
            self.process_plot.clear()
        self.update_progress_info("Process history cleared.")
    
    def update_process_visualization(self):
        """Update the real-time process visualization."""
        if not hasattr(self, 'process_plot') or len(self.process_history['iteration']) == 0:
            return
        
        try:
            # Clear the plot
            self.process_plot.clear()
            
            # Create subplots
            fig = self.process_plot.figure
            
            # Create 2x2 subplot layout
            ax1 = fig.add_subplot(221)  # Top left: Score progression
            ax2 = fig.add_subplot(222)  # Top right: Uncertainty evolution
            ax3 = fig.add_subplot(223)  # Bottom left: Acquisition scores
            ax4 = fig.add_subplot(224)  # Bottom right: Learning speed
            
            iterations = self.process_history['iteration']
            best_scores = self.process_history['best_score']
            current_scores = self.process_history['current_score']
            uncertainties = self.process_history['uncertainty']
            acquisition_scores = self.process_history['acquisition_score']
            
            # Plot 1: Score progression
            ax1.plot(iterations, best_scores, 'g-', linewidth=2, label='Best Score', marker='o', markersize=4)
            ax1.plot(iterations, current_scores, 'b-', alpha=0.6, label='Current Score', linewidth=1)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Score')
            ax1.set_title('Score Progression')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Uncertainty evolution
            ax2.plot(iterations, uncertainties, 'r-', linewidth=2, marker='s', markersize=3)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Uncertainty')
            ax2.set_title('Uncertainty Evolution')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Acquisition scores
            ax3.bar(iterations[-10:], acquisition_scores[-10:], alpha=0.7, color='orange')
            ax3.set_xlabel('Recent Iterations')
            ax3.set_ylabel('Acquisition Score')
            ax3.set_title('Recent Acquisition Scores')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Learning speed (improvement per iteration)
            if len(best_scores) > 1:
                improvements = [abs(best_scores[i] - best_scores[i-1]) for i in range(1, len(best_scores))]
                ax4.plot(iterations[1:], improvements, 'm-', linewidth=2, marker='^', markersize=3)
                ax4.set_xlabel('Iteration')
                ax4.set_ylabel('Improvement')
                ax4.set_title('Learning Speed')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax4.transAxes)
            
            # Adjust layout
            fig.suptitle('Real-time Learning Process', fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            # Refresh canvas
            self.process_plot.canvas.draw()
            
        except Exception as e:
            print(f"Process visualization error: {e}")
    
    def export_results(self):
        """Export results to file."""
        if self.results_data is None:
            QMessageBox.information(self, "Information", "No results data to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "active_learning_results.csv",
            "CSV files (*.csv);;Excel files (*.xlsx);;All files (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.results_data.to_csv(file_path, index=False)
                else:
                    self.results_data.to_excel(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"Results exported to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About", 
                         "Active Learning & Optimization Module\n\n"
                         "Intelligent experimental design assistant based on Bayesian optimization\n"
                         "Helps you efficiently find optimal solutions\n\n"
                         "🤖 Supported Surrogate Models (9 total):\n"
                         "• Tree-based: RandomForest, XGBoost, LightGBM, CatBoost, ExtraTrees\n"
                         "• Probabilistic: GaussianProcess, BayesianRidge\n"
                         "• Neural Networks: MLPRegressor\n"
                         "• Kernel Methods: SVR\n\n"
                         "🎯 Features:\n"
                         "• Bayesian & Grid Search hyperparameter optimization\n"
                         "• Single & Multi-objective optimization\n"
                         "• Intelligent candidate generation\n"
                         "• Rich visualization and analysis capabilities")
    
    def show_candidate_generator(self):
        """Show the candidate set generator dialog."""
        if self.training_data is None:
            QMessageBox.warning(self, "Warning", "Please load training data first.")
            return
        
        # Get all numeric columns as potential targets and features
        numeric_columns = self.training_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            QMessageBox.warning(self, "Warning", "No numeric columns found for candidate generation.")
            return
        
        # Create and show optimized generator dialog
        dialog = OptimizedCandidateDialog(self.training_data, numeric_columns, self)
        if dialog.exec_() == QDialog.Accepted:
            candidate_df = dialog.get_generated_candidates()
            generation_settings = dialog.get_generation_settings()
            
            if candidate_df is not None:
                # Set as virtual data
                self.virtual_data = candidate_df
                
                # Update UI
                self.virtual_label.setText(
                    f"✅ Generated Candidates\n({len(self.virtual_data)} rows × {len(self.virtual_data.columns)} cols)"
                )
                self.virtual_label.setStyleSheet("color: #28a745; font-size: 11px;")
                
                # 🎯 先应用候选生成器的设置到主界面（这会设置正确的目标）
                if generation_settings:
                    self.apply_generation_settings(generation_settings)
                
                # 然后更新列选择器（此时目标已正确设置）
                self.update_column_selectors()
                
                # Update run button state
                self.update_run_button_state()
                
                self.status_bar.showMessage(f"Generated {len(candidate_df)} candidate samples successfully")
                
                # 重要：显示详细信息确认数据已正确加载
                info_message = f"Successfully generated and loaded {len(candidate_df)} candidate samples!\n\n"
                info_message += f"Data dimensions: {len(candidate_df)} rows × {len(candidate_df.columns)} columns\n"
                info_message += f"Column names: {', '.join(candidate_df.columns.tolist())}\n\n"
                
                # 显示自动应用的设置信息
                if generation_settings:
                    info_message += "🎯 The following settings have been automatically applied to the main interface:\n"
                    info_message += f"• Target variables: {', '.join(generation_settings['targets'])}\n"
                    info_message += f"• Number of features: {len(generation_settings['features'])} features\n"
                    info_message += f"• Optimization mode: {'Single-objective' if generation_settings['target_mode'] == 'single' else 'Multi-objective'}\n\n"
                
                # 如果数据量少于1000，可能是约束过滤的结果
                if len(candidate_df) < 1000:
                    info_message += "⚠️ Note: If you expected more samples, it might be due to overly strict constraints.\n"
                    info_message += "💡 Suggestion: Enable the 'Ensure Exact Sample Count' option in the candidate generator.\n\n"
                
                info_message += "Data has been set as a virtual dataset, and the main interface settings have been synchronized, allowing direct start of analysis."
                
                QMessageBox.information(self, "Candidate Data Generation Success", info_message)
            else:
                QMessageBox.warning(self, "Error", "Candidate data generation failed!")
        else:
            QMessageBox.information(self, "Cancel", "Candidate data generation cancelled.")
    
    def apply_generation_settings(self, settings):
      
        try:
            print(f"Applying generation settings: {settings}")
            
            # 1. 设置优化模式
            if settings['target_mode'] == 'single':
                self.single_obj_radio.setChecked(True)
            else:
                self.multi_obj_radio.setChecked(True)
            
            # 触发模式切换
            self.on_mode_changed()
            
            # 2. 设置目标变量
            targets = settings.get('targets', [])
            if targets:
                if settings['target_mode'] == 'single':
                    # 单目标模式：设置下拉框和优化方向
                    if hasattr(self, 'target_combo') and targets[0] in [self.target_combo.itemText(i) for i in range(self.target_combo.count())]:
                        index = self.target_combo.findText(targets[0])
                        if index >= 0:
                            self.target_combo.setCurrentIndex(index)
                            
                            # 设置优化方向
                            goal = settings['optimization_goals'].get(targets[0], 'maximize')
                            if hasattr(self, 'maximize_radio') and hasattr(self, 'minimize_radio'):
                                if goal == 'minimize':
                                    self.minimize_radio.setChecked(True)
                                else:
                                    self.maximize_radio.setChecked(True)
                            
                            print(f"Set single target: {targets[0]} ({goal})")
                else:
                    # 多目标模式：清除现有目标并添加新目标到表格
                    if hasattr(self, 'selected_targets_table'):
                        # 清空表格
                        self.selected_targets_table.setRowCount(0)
                        
                        # 更新可用目标列表
                        self.update_available_targets()
                        
                        # 添加新目标到表格
                        for i, target in enumerate(targets):
                            goal = settings['optimization_goals'].get(target, 'maximize')
                            self.add_target_to_table(target, goal)
                        
                        print(f"Set multi targets: {targets}")
            
            # 3. 设置特征选择
            features = settings.get('features', [])
            if features and hasattr(self, 'feature_list'):
                print(f"\n=== 🔍 APPLYING FEATURE SELECTION FROM GENERATOR ===")
                print(f"Features from generator settings: {len(features)} features")
                print(f"Feature list: {features}")
                print(f"UI feature_list widget has {self.feature_list.count()} items")
                
                # 🔍 显示当前UI中所有可用的特征（在清除选择之前）
                available_features = []
                for i in range(self.feature_list.count()):
                    item = self.feature_list.item(i)
                    available_features.append(item.text())
                print(f"Available features in UI: {available_features}")
                
                # 🔍 检查缺失的特征（在应用选择之前就检查）
                features_in_ui = set(available_features)
                features_from_generator = set(features)
                missing_in_ui = features_from_generator - features_in_ui
                extra_in_ui = features_in_ui - features_from_generator
                
                if missing_in_ui:
                    print(f"⚠️ CRITICAL: Features in settings but missing from UI: {list(missing_in_ui)}")
                    print(f"   This suggests update_column_selectors() filtered out these features")
                if extra_in_ui:
                    print(f"ℹ️ INFO: Features in UI but not in generator settings: {list(extra_in_ui)}")
                
                # 先取消所有选择
                for i in range(self.feature_list.count()):
                    item = self.feature_list.item(i)
                    item.setCheckState(Qt.Unchecked)
                
                # 选择指定的特征并详细记录
                selected_count = 0
                for i in range(self.feature_list.count()):
                    item = self.feature_list.item(i)
                    if item.text() in features:
                        item.setCheckState(Qt.Checked)
                        selected_count += 1
                        print(f"  ✓ Selected: {item.text()}")
                
                # 检查是否有特征在设置中但不在UI列表中
                if missing_in_ui:
                    print(f"⚠️ WARNING: Features in settings but not in UI: {list(missing_in_ui)}")
                
                print(f"Successfully selected {selected_count} out of {len(features)} features from generation settings")
                print(f"Expected: {len(features)}, Actually selected: {selected_count}")
                
                if selected_count < len(features):
                    print(f"⚠️ WARNING: {len(features) - selected_count} features could not be selected!")
                    print(f"Missing features: {list(missing_in_ui)}")
            
            # 4. 在设置目标后立即更新列选择器，确保特征过滤使用正确的目标
            print("🔄 Updating column selectors after target settings...")
            # 注意：这里可能导致重复调用，但确保目标设置后特征过滤正确
            
            # 5. 更新UI状态
            self.update_run_button_state()
            
            print("Generation settings applied successfully")
            
        except Exception as e:
            print(f"Error applying generation settings: {e}")
            QMessageBox.warning(self, "Failed to apply settings", f"Failed to apply candidate generator settings to main interface:\n{str(e)}") 
    
    def add_target_to_list(self, target_name, goal='maximize'):
        """添加目标到多目标列表"""
        try:
            if hasattr(self, 'target_list') and hasattr(self, 'target_combo'):
                # 检查目标是否已存在
                for i in range(self.target_list.count()):
                    if self.target_list.item(i).text().split(' (')[0] == target_name:
                        return  # 已存在，不重复添加
                
                # 添加到列表
                goal_text = "maximize" if goal == 'maximize' else "minimize"
                item_text = f"{target_name} ({goal_text})"
                self.target_list.addItem(item_text)
                
                # 从单目标下拉框中移除（如果存在）
                index = self.target_combo.findText(target_name)
                if index >= 0:
                    self.target_combo.removeItem(index)
                
                print(f"Added target to multi-objective list: {item_text}")
                
        except Exception as e:
            print(f"Error adding target to list: {e}")
    
    def add_target_to_table(self, target_name, goal='maximize'):
        """添加目标到多目标表格"""
        try:
            if hasattr(self, 'selected_targets_table'):
                # 检查目标是否已存在
                for row in range(self.selected_targets_table.rowCount()):
                    if self.selected_targets_table.item(row, 0).text() == target_name:
                        return  # 已存在，不重复添加
                
                # 添加新行到表格
                row_count = self.selected_targets_table.rowCount()
                self.selected_targets_table.insertRow(row_count)
                
                # 设置目标变量名
                target_item = QTableWidgetItem(target_name)
                self.selected_targets_table.setItem(row_count, 0, target_item)
                
                # 设置优化目标下拉框
                goal_combo = QComboBox()
                goal_combo.addItem("maximize", "maximize")
                goal_combo.addItem("minimize", "minimize")
                
                # 设置当前值
                if goal == "minimize":
                    goal_combo.setCurrentIndex(1)
                else:
                    goal_combo.setCurrentIndex(0)
                
                self.selected_targets_table.setCellWidget(row_count, 1, goal_combo)
                
                # 从可用目标列表中移除
                if hasattr(self, 'available_targets_list'):
                    for i in range(self.available_targets_list.count()):
                        item = self.available_targets_list.item(i)
                        if item and item.text() == target_name:
                            self.available_targets_list.takeItem(i)
                            break
                
                print(f"Added target to multi-objective table: {target_name} ({goal})")
                
        except Exception as e:
            print(f"Error adding target to table: {e}")
    
    def reset_session(self):
        """Reset the active learning session."""
        if not self.session:
            QMessageBox.information(self, "Info", "No active session to reset.")
            return

        reply = QMessageBox.question(self, 'Reset Session',
                                     "Are you sure you want to discard all iteratively acquired data points and reset the session?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.session.reset()
            self.last_recommendation = None
            self.feedback_box.hide()
            self.update_session_status()
            self.update_history_tab()
            self.update_recommendation_text()  # Clear the recommendation
            QMessageBox.information(self, "Success", "Active learning session has been reset.")
    
    def setup_feedback_ui(self):
        """Set up feedback UI - delegates to batch feedback for better UX."""
        self.setup_batch_feedback_ui()

    def setup_batch_feedback_ui(self):
        """Set up batch feedback UI for multiple experimental results."""
        # Clear any previous widgets
        while self.feedback_layout.count():
            child = self.feedback_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if self.results_data is None or len(self.results_data) == 0:
            self.feedback_box.hide()
            return

        # Get batch size from current setting or default
        if hasattr(self, 'current_batch_spin'):
            batch_size = self.current_batch_spin.value()
        else:
            batch_size = getattr(self, 'batch_size_spin', type('', (), {'value': lambda: 5})).value()
        top_recommendations = self.results_data.head(batch_size)
        
        if len(top_recommendations) == 0:
            self.feedback_box.hide()
            return

        # Get target columns based on current mode
        if self.is_multi_objective:
            targets, _ = self.get_selected_targets_and_goals()
        else:
            targets = [self.target_combo.currentText()]

        # Create title
        title_label = QLabel(f"📊 Batch Experimental Results Entry")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
                padding: 5px;
            }
        """)
        self.feedback_layout.addRow(title_label)

        # Batch size control
        batch_control_layout = QHBoxLayout()
        batch_control_layout.addWidget(QLabel("Batch Size:"))
        
        self.current_batch_spin = QSpinBox()
        self.current_batch_spin.setRange(1, min(50, len(self.results_data)))
        self.current_batch_spin.setValue(batch_size)
        self.current_batch_spin.setToolTip("Number of recommendations to show for experiments")
        self.current_batch_spin.valueChanged.connect(self.update_batch_size)
        batch_control_layout.addWidget(self.current_batch_spin)
        
        batch_control_layout.addStretch()
        batch_control_widget = QWidget()
        batch_control_widget.setLayout(batch_control_layout)
        self.feedback_layout.addRow(batch_control_widget)

        # Instructions
        instructions = QLabel(
            f"Enter experimental results for the {len(top_recommendations)} recommended formulations below.\n"
            f"Complete your experiments, then fill in the measured values for each target property.\n"
            f"💡 Adjust batch size above to show more/fewer recommendations."
        )
        instructions.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 11px;
                margin-bottom: 10px;
                padding: 5px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }
        """)
        instructions.setWordWrap(True)
        self.feedback_layout.addRow(instructions)

        # 创建一个滚动区域来容纳表格，避免表格过大导致界面重叠
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setMinimumHeight(250)
        scroll_area.setMaximumHeight(350)  # 限制最大高度，避免挤压其他控件
        
        # 创建表格容器
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建批量反馈表格
        self.batch_feedback_table = QTableWidget()
        self.batch_feedback_table.setAlternatingRowColors(True)
        self.batch_feedback_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)  # 平滑滚动
        self.batch_feedback_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        
        feature_columns = self.get_selected_features()
        # 限制显示的特征数量，优先显示重要特征
        display_features = feature_columns[:4] if len(feature_columns) > 4 else feature_columns
        
        # 使用更简洁的列标题
        all_columns = ['#'] + display_features + [f"测量值_{target}" for target in targets] + ['状态']
        
        self.batch_feedback_table.setColumnCount(len(all_columns))
        self.batch_feedback_table.setRowCount(len(top_recommendations))
        self.batch_feedback_table.setHorizontalHeaderLabels(all_columns)
        
        # 存储批处理数据
        self.batch_recommendations = top_recommendations.copy()
        self.batch_target_inputs = {}
        
        # 填充表格
        for row_idx, (_, rec) in enumerate(top_recommendations.iterrows()):
            # 推荐编号
            rec_item = QTableWidgetItem(f"{row_idx + 1}")
            rec_item.setFlags(rec_item.flags() & ~Qt.ItemIsEditable)
            rec_item.setTextAlignment(Qt.AlignCenter)
            self.batch_feedback_table.setItem(row_idx, 0, rec_item)
            
            # 特征值（显示特征）
            for col_idx, feature in enumerate(display_features):
                if feature in rec.index:
                    # 使用更紧凑的格式显示数值
                    value = rec[feature]
                    if abs(value) < 0.001 or abs(value) >= 10000:
                        formatted_value = f"{value:.2e}"  # 科学计数法
                    elif abs(value) < 0.01:
                        formatted_value = f"{value:.4f}"  # 小数点后4位
                    elif abs(value) < 1:
                        formatted_value = f"{value:.3f}"  # 小数点后3位
                    else:
                        formatted_value = f"{value:.2f}"  # 小数点后2位
                    
                    feature_item = QTableWidgetItem(formatted_value)
                    feature_item.setFlags(feature_item.flags() & ~Qt.ItemIsEditable)
                    feature_item.setTextAlignment(Qt.AlignCenter)
                    feature_item.setBackground(QColor(240, 248, 255))
                    
                    # 添加包含所有特征的工具提示
                    all_features_info = f"推荐 #{row_idx + 1} - 所有特征:\n"
                    for feat in feature_columns:
                        if feat in rec.index:
                            all_features_info += f"{feat}: {rec[feat]:.4f}\n"
                    feature_item.setToolTip(all_features_info.strip())
                    
                    self.batch_feedback_table.setItem(row_idx, col_idx + 1, feature_item)
            
            # 目标输入字段
            self.batch_target_inputs[row_idx] = {}
            for target_idx, target in enumerate(targets):
                # 使用LineEdit替代TableWidgetItem，提供更好的输入体验
                input_widget = QLineEdit()
                input_widget.setAlignment(Qt.AlignCenter)
                input_widget.setStyleSheet("""
                    QLineEdit {
                        background-color: #fff8f0;
                        border: 1px solid #ddd;
                        border-radius: 2px;
                        padding: 2px;
                    }
                    QLineEdit:focus {
                        border: 1px solid #2E86AB;
                        background-color: #fffaf5;
                    }
                """)
                input_widget.setToolTip(f"Enter measured value for {target}")
                input_widget.setPlaceholderText("Enter value")
                
                # 使用QDoubleValidator确保只能输入数值
                validator = QDoubleValidator()
                validator.setNotation(QDoubleValidator.StandardNotation)
                input_widget.setValidator(validator)
                
                col_pos = len(display_features) + 1 + target_idx
                self.batch_feedback_table.setCellWidget(row_idx, col_pos, input_widget)
                self.batch_target_inputs[row_idx][target] = input_widget
            
            # 状态列
            status_item = QTableWidgetItem("Pending")
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setBackground(QColor(255, 255, 224))
            self.batch_feedback_table.setItem(row_idx, len(all_columns) - 1, status_item)

        # 设置表格样式
        self.batch_feedback_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                border: 1px solid #ccc;
                background-color: white;
                font-size: 10px;  /* 更小的字体 */
            }
            QTableWidget::item {
                padding: 4px;  /* 更小的内边距 */
                border: 1px solid #ddd;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 4px;  /* 更小的内边距 */
                border: 1px solid #ddd;
                font-weight: bold;
                font-size: 10px;  /* 更小的字体 */
            }
        """)
        
        # 设置列宽
        self.batch_feedback_table.setColumnWidth(0, 30)  # 编号列宽度
        for col in range(1, self.batch_feedback_table.columnCount()):
            self.batch_feedback_table.setColumnWidth(col, 80)  # 其他列宽度
        
        # 将表格添加到容器
        table_layout.addWidget(self.batch_feedback_table)
        
        # 设置滚动区域的内容
        scroll_area.setWidget(table_container)
        
        # 添加到布局
        self.feedback_layout.addRow(scroll_area)

        # Control buttons
        button_layout = QHBoxLayout()
        
        # View full features button
        view_features_btn = QPushButton("📋 View All Features")
        view_features_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        view_features_btn.clicked.connect(self.show_full_features_dialog)
        button_layout.addWidget(view_features_btn)
        
        # Validate button
        validate_btn = QPushButton("🔍 Validate Inputs")
        validate_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        validate_btn.clicked.connect(self.validate_batch_inputs)
        button_layout.addWidget(validate_btn)
        
        # Submit button
        submit_btn = QPushButton("✅ Submit All Results & Start Next Iteration")
        submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        submit_btn.clicked.connect(self.submit_batch_results)
        button_layout.addWidget(submit_btn)
        
        # Clear button
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        clear_btn.clicked.connect(self.clear_batch_inputs)
        button_layout.addWidget(clear_btn)
        
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        self.feedback_layout.addRow(button_widget)

        self.feedback_box.show()

    def update_batch_size(self):
        """Update batch size and refresh the feedback table."""
        if hasattr(self, 'current_batch_spin'):
            self.setup_batch_feedback_ui()

    def validate_batch_inputs(self):
        """验证所有批量输入值并更新状态。"""
        if not hasattr(self, 'batch_target_inputs'):
            return
            
        valid_count = 0
        total_count = len(self.batch_target_inputs)
        
        for row_idx, target_inputs in self.batch_target_inputs.items():
            all_valid = True
            values = {}
            
            for target, input_widget in target_inputs.items():
                value_str = input_widget.text().strip()
                
                if not value_str:
                    all_valid = False
                    input_widget.setStyleSheet("""
                        QLineEdit {
                            background-color: #fff0f0;
                            border: 1px solid #ffb6b6;
                            border-radius: 2px;
                            padding: 2px;
                        }
                    """)  # 浅红色背景
                    input_widget.setToolTip(f"缺少{target}的值")
                    continue
                    
                try:
                    value = float(value_str)
                    values[target] = value
                    input_widget.setStyleSheet("""
                        QLineEdit {
                            background-color: #f0fff0;
                            border: 1px solid #90EE90;
                            border-radius: 2px;
                            padding: 2px;
                        }
                    """)  # 浅绿色背景
                    input_widget.setToolTip(f"有效值: {value}")
                except ValueError:
                    all_valid = False
                    input_widget.setStyleSheet("""
                        QLineEdit {
                            background-color: #fff0f0;
                            border: 1px solid #ffb6b6;
                            border-radius: 2px;
                            padding: 2px;
                        }
                    """)  # 浅红色背景
                    input_widget.setToolTip(f"Invalid number: '{value_str}'")
            
            # 更新状态列
            status_col = self.batch_feedback_table.columnCount() - 1
            status_item = self.batch_feedback_table.item(row_idx, status_col)
            
            if all_valid:
                status_item.setText("✓ Valid")
                status_item.setBackground(QColor(240, 255, 240))
                valid_count += 1
            else:
                status_item.setText("✗ Invalid")
                status_item.setBackground(QColor(255, 240, 240))
        
        # 显示验证摘要
        if valid_count == total_count:
            QMessageBox.information(self, "Verification Complete", 
                                   f"✅ All {total_count} entries are valid and ready for submission!")
        else:
            QMessageBox.warning(self, "Validation Issues", 
                               f"⚠️ {valid_count}/{total_count} entries are valid. "
                               f"Please fix the highlighted issues.")

    def clear_batch_inputs(self):
        """清除所有批量输入值。"""
        if not hasattr(self, 'batch_target_inputs'):
            return
            
        reply = QMessageBox.question(self, "Clear All Inputs", 
                                   "Are you sure you want to clear all input values?",
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            for row_idx, target_inputs in self.batch_target_inputs.items():
                for target, input_widget in target_inputs.items():
                    input_widget.setText("")
                    input_widget.setStyleSheet("""
                        QLineEdit {
                            background-color: #fff8f0;
                            border: 1px solid #ddd;
                            border-radius: 2px;
                            padding: 2px;
                        }
                    """)  # 恢复原始样式
                    input_widget.setToolTip(f"Enter measured value for {target}")
                
                # 重置状态
                status_col = self.batch_feedback_table.columnCount() - 1
                status_item = self.batch_feedback_table.item(row_idx, status_col)
                status_item.setText("Pending")
                status_item.setBackground(QColor(255, 255, 224))

    def submit_batch_results(self):
        """提交所有批量实验结果并开始下一轮迭代。"""
        if not hasattr(self, 'batch_target_inputs') or not hasattr(self, 'batch_recommendations'):
            QMessageBox.warning(self, "Warning", "No batch data available to submit.")
            return

        # 收集并验证所有结果
        batch_results = []
        feature_columns = self.get_selected_features()
        
        for row_idx, target_inputs in self.batch_target_inputs.items():
            row_data = {}
            
            # 从推荐中获取特征值
            rec = self.batch_recommendations.iloc[row_idx]
            for feature in feature_columns:
                if feature in rec.index:
                    row_data[feature] = rec[feature]
            
            # 从用户输入获取目标值
            target_values = {}
            for target, input_widget in target_inputs.items():
                value_str = input_widget.text().strip()
                
                if not value_str:
                    QMessageBox.critical(self, "Input Error", 
                                       f"Missing value for {target} in recommendation #{row_idx + 1}")
                    return
                
                try:
                    target_values[target] = float(value_str)
                except ValueError:
                    QMessageBox.critical(self, "Input Error", 
                                       f"Invalid number '{value_str}' for {target} in recommendation #{row_idx + 1}")
                    return
            
            row_data['targets'] = target_values
            batch_results.append(row_data)

        # 确认提交
        reply = QMessageBox.question(self, "Submit Batch Results", 
                                   f"Submit {len(batch_results)} experimental results and start next iteration?",
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.Yes)
        
        if reply != QMessageBox.Yes:
            return

        try:
            # Add all results to session
            for result in batch_results:
                # Extract features and targets
                feature_series = pd.Series({k: v for k, v in result.items() if k != 'targets'})
                target_dict = result['targets']
                
                # Add to session
                self.session.add_new_point(feature_series, target_dict)

            # Record performance metrics
            if self.optimizer and self.results_data is not None:
                try:
                    training_data = self.session.get_combined_training_data()
                    
                    if self.is_multi_objective:
                        targets_list, goals_list = self.get_selected_targets_and_goals()
                        target_col = targets_list[0] if targets_list else None
                    else:
                        target_col = self.target_combo.currentText()
                        goals_list = ['maximize' if self.maximize_radio.isChecked() else 'minimize']
                    
                    if target_col and target_col in training_data.columns:
                        # Assess model performance
                        reliability_score = self.optimizer.assess_model_reliability(
                            training_data, target_col, feature_columns
                        )
                        
                        # Get feature importance
                        feature_importance = self.optimizer.get_feature_importance()
                        if isinstance(feature_importance, dict):
                            feature_importance = feature_importance.get(target_col, {})
                        
                        # Calculate best value found so far
                        if goals_list[0] == 'maximize':
                            best_value = training_data[target_col].max()
                        else:
                            best_value = training_data[target_col].min()
                        
                        # Calculate uncertainty metrics
                        uncertainty_metrics = {
                            'mean_uncertainty': self.results_data['uncertainty_std'].mean(),
                            'max_uncertainty': self.results_data['uncertainty_std'].max(),
                            'min_uncertainty': self.results_data['uncertainty_std'].min()
                        }
                        
                        # Calculate acquisition function statistics
                        acquisition_stats = {
                            'max_acquisition': self.results_data['acquisition_score'].max(),
                            'mean_acquisition': self.results_data['acquisition_score'].mean(),
                            'top_10_mean': self.results_data.head(10)['acquisition_score'].mean()
                        }
                        
                        # Add performance record
                        self.session.add_performance_record(
                            model_performance={'r2_score': reliability_score},
                            feature_importance=feature_importance,
                            best_value=best_value,
                            uncertainty_metrics=uncertainty_metrics,
                            acquisition_stats=acquisition_stats
                        )
                        
                        # Auto-tune model if needed
                        iteration_count = self.session.get_iteration_count()
                        if reliability_score is not None and isinstance(reliability_score, (int, float)):
                            if reliability_score < 0.1 or iteration_count % 5 == 0:
                                try:
                                    print("Starting automatic hyperparameter optimization...")
                                    tuned_params = self.optimizer.auto_tune_model(
                                        training_df=training_data,
                                        target_columns=target_col,
                                        feature_columns=feature_columns
                                    )
                                    
                                    if tuned_params:
                                        self.optimizer.apply_optimized_parameters(tuned_params)
                                        print(f"Applied optimized parameters: {tuned_params}")
                                
                                except Exception as tune_error:
                                    print(f"Warning: Auto-tuning failed: {tune_error}")
                        
                except Exception as e:
                    print(f"Warning: Could not record performance metrics: {e}")

            # Clean up UI
            self.feedback_box.hide()
            
            # Update all visualizations
            self.update_session_status()
            self.update_history_tab()
            self.update_learning_curves()
            self.update_feature_evolution()
            self.update_exploration_analysis()
            
            # Show success message
            iteration_count = self.session.get_iteration_count()
            QMessageBox.information(self, "Batch Submission Successful", 
                                   f"✅ {len(batch_results)} experimental results added successfully!\n"
                                   f"Starting iteration {iteration_count + 1}...")
            
            # Automatically start next analysis
            self.run_analysis()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to submit batch results: {str(e)}")

    def show_full_features_dialog(self):
        """Show a dialog with all features for all recommendations."""
        if not hasattr(self, 'batch_recommendations'):
            QMessageBox.warning(self, "Warning", "No recommendations available to display.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Complete Feature Values for All Recommendations")
        dialog.setModal(True)
        dialog.resize(1000, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Create table with all features
        feature_columns = self.get_selected_features()
        table = QTableWidget()
        table.setRowCount(len(self.batch_recommendations))
        table.setColumnCount(len(feature_columns) + 1)  # +1 for recommendation number
        
        headers = ['Rec#'] + feature_columns
        table.setHorizontalHeaderLabels(headers)
        
        # Populate table
        for row_idx, (_, rec) in enumerate(self.batch_recommendations.iterrows()):
            # Recommendation number
            rec_item = QTableWidgetItem(f"#{row_idx + 1}")
            rec_item.setTextAlignment(Qt.AlignCenter)
            rec_item.setBackground(QColor(240, 248, 255))
            table.setItem(row_idx, 0, rec_item)
            
            # All feature values
            for col_idx, feature in enumerate(feature_columns):
                if feature in rec.index:
                    value_item = QTableWidgetItem(f"{rec[feature]:.6f}")
                    value_item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(row_idx, col_idx + 1, value_item)
        
        table.setAlternatingRowColors(True)
        table.resizeColumnsToContents()
        
        # Style the table
        table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                border: 1px solid #ccc;
                background-color: white;
                font-size: 10px;
            }
            QTableWidget::item {
                padding: 4px;
                border: 1px solid #ddd;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 6px;
                border: 1px solid #ddd;
                font-weight: bold;
                font-size: 10px;
            }
        """)
        
        layout.addWidget(QLabel(f"Complete feature values for {len(self.batch_recommendations)} recommendations:"))
        layout.addWidget(table)
        
        # Export button
        export_btn = QPushButton("📄 Export to CSV")
        export_btn.clicked.connect(lambda: self.export_recommendations_csv())
        layout.addWidget(export_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()

    def export_recommendations_csv(self):
        """Export current recommendations to CSV file."""
        if not hasattr(self, 'batch_recommendations'):
            QMessageBox.warning(self, "Warning", "No recommendations available to export.")
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recommendations_{timestamp}.csv"
        
        try:
            # Add recommendation number
            export_data = self.batch_recommendations.copy()
            export_data.insert(0, 'Recommendation_Number', range(1, len(export_data) + 1))
            
            export_data.to_csv(filename, index=False)
            QMessageBox.information(self, "Export Successful", 
                                   f"Recommendations exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def add_result_and_rerun(self):
        """Add the user's experimental result and start the next iteration."""
        if self.last_recommendation is None:
            QMessageBox.warning(self, "Warning", "No recommendation available to add results for.")
            return

        feature_columns = self.get_selected_features()
        recommendation_features = self.last_recommendation[feature_columns]

        try:
            # Validate and collect input values
            new_targets = {}
            for target_name, line_edit in self.feedback_inputs.items():
                value_str = line_edit.text().strip()
                if not value_str:
                    raise ValueError(f"Result for {target_name} cannot be empty.")
                try:
                    new_targets[target_name] = float(value_str)
                except ValueError:
                    raise ValueError(f"Result for {target_name} must be a valid number.")

            # Add the validated point to the session
            self.session.add_new_point(recommendation_features, new_targets)

            # Record performance metrics if available
            if self.optimizer and self.results_data is not None:
                try:
                    # Calculate model performance metrics
                    training_data = self.session.get_combined_training_data()
                    feature_columns = self.get_selected_features()
                    
                    if self.is_multi_objective:
                        targets_list, goals_list = self.get_selected_targets_and_goals()
                        target_col = targets_list[0] if targets_list else None
                    else:
                        target_col = self.target_combo.currentText()
                        goals_list = ['maximize' if self.maximize_radio.isChecked() else 'minimize']
                    
                    if target_col and target_col in training_data.columns:
                        # Assess model performance
                        print(f"DEBUG: Calculating R2 for target '{target_col}' with {len(training_data)} data points")
                        print(f"DEBUG: Feature columns: {feature_columns}")
                        print(f"DEBUG: Training data shape: {training_data.shape}")
                        
                        reliability_score = self.optimizer.assess_model_reliability(
                            training_data, target_col, feature_columns
                        )
                        print(f"DEBUG: Calculated R2 score: {reliability_score}")
                        
                        # Get feature importance
                        feature_importance = self.optimizer.get_feature_importance()
                        if isinstance(feature_importance, dict):
                            # Multi-objective case
                            feature_importance = feature_importance.get(target_col, {})
                        
                        # Calculate best value found so far
                        if goals_list[0] == 'maximize':
                            best_value = training_data[target_col].max()
                        else:
                            best_value = training_data[target_col].min()
                        
                        # Calculate uncertainty metrics
                        uncertainty_metrics = {
                            'mean_uncertainty': self.results_data['uncertainty_std'].mean(),
                            'max_uncertainty': self.results_data['uncertainty_std'].max(),
                            'min_uncertainty': self.results_data['uncertainty_std'].min()
                        }
                        
                        # Calculate acquisition function statistics
                        acquisition_stats = {
                            'max_acquisition': self.results_data['acquisition_score'].max(),
                            'mean_acquisition': self.results_data['acquisition_score'].mean(),
                            'top_10_mean': self.results_data.head(10)['acquisition_score'].mean()
                        }
                        
                        # Add performance record
                        self.session.add_performance_record(
                            model_performance={'r2_score': reliability_score},
                            feature_importance=feature_importance,
                            best_value=best_value,
                            uncertainty_metrics=uncertainty_metrics,
                            acquisition_stats=acquisition_stats
                        )
                        
                        # Auto-tune model if R² is poor or sufficient iterations completed
                        iteration_count = self.session.get_iteration_count()
                        should_tune = False
                        
                        # Trigger tuning conditions
                        if reliability_score is not None and isinstance(reliability_score, (int, float)):
                            if reliability_score < 0.1:  # Poor R² score
                                should_tune = True
                                print(f"AUTO-TUNE: R² = {reliability_score:.4f} is poor, triggering hyperparameter optimization...")
                            elif iteration_count % 5 == 0:  # Every 5 iterations
                                should_tune = True
                                print(f"AUTO-TUNE: Regular tuning at iteration {iteration_count}...")
                        
                        if should_tune:
                            try:
                                print("Starting automatic hyperparameter optimization...")
                                tuned_params = self.optimizer.auto_tune_model(
                                    training_df=training_data,
                                    target_columns=target_col,
                                    feature_columns=feature_columns
                                )
                                
                                if tuned_params:
                                    self.optimizer.apply_optimized_parameters(tuned_params)
                                    print(f"Applied optimized parameters: {tuned_params}")
                                    
                                    # Re-assess model with new parameters
                                    new_reliability = self.optimizer.assess_model_reliability(
                                        training_data, target_col, feature_columns
                                    )
                                    print(f"Model performance after tuning: R² = {new_reliability:.4f}")
                                    
                                    if new_reliability > reliability_score:
                                        improvement = new_reliability - reliability_score
                                        print(f"✅ Hyperparameter tuning improved R² by {improvement:.4f}")
                                    else:
                                        print("⚠️ No significant improvement from tuning")
                                else:
                                    print("⚠️ Auto-tuning did not find better parameters")
                                    
                            except Exception as tune_error:
                                print(f"Warning: Auto-tuning failed: {tune_error}")
                        
                except Exception as e:
                    print(f"Warning: Could not record performance metrics: {e}")

            # Hide feedback box and show success message
            self.feedback_box.hide()
            self.last_recommendation = None
            
            # Update session status and history immediately
            self.update_session_status()
            self.update_history_tab()
            
            # Update learning visualizations
            self.update_learning_curves()
            self.update_feature_evolution()
            self.update_exploration_analysis()
            
            # Update status labels
            if hasattr(self, 'learning_status_label'):
                iteration_count = self.session.get_iteration_count()
                self.learning_status_label.setText(f"Status: {iteration_count} iterations completed")
            
            # Show success message and automatically start next iteration
            iteration_count = self.session.get_iteration_count()
            QMessageBox.information(self, "Success", 
                                   f"New data point added successfully! Starting iteration {iteration_count + 1}...")
            
            # Automatically start the next analysis iteration
            self.run_analysis()

        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")
    
    def update_session_status(self):
        """Update the session status display."""
        if not self.session:
            self.session_status_label.setText("No active session")
            self.session_status_label.setStyleSheet("color: #666; font-size: 12px;")
            return
        
        summary = self.session.get_session_summary()
        status_text = f"Active session: {summary['iterations_completed']} iterations completed\n"
        status_text += f"Original data: {summary['original_data_points']} points\n"
        status_text += f"Acquired data: {summary['newly_acquired_points']} points\n"
        status_text += f"Total data: {summary['total_data_points']} points"
        
        if summary['last_update']:
            status_text += f"\nLast update: {summary['last_update']}"
        
        self.session_status_label.setText(status_text)
        self.session_status_label.setStyleSheet("color: #28a745; font-size: 11px; font-family: monospace;")
    
    def update_history_tab(self):
        """Update the history tab with iteratively acquired data points."""
        if not self.session:
            self.history_table.clear()
            self.history_table.setRowCount(0)
            self.history_table.setColumnCount(0)
            return

        history_df = self.session.get_newly_acquired_data()
        
        if history_df.empty:
            self.history_table.clear()
            self.history_table.setRowCount(0)
            self.history_table.setColumnCount(0)
            return
        
        # Set up the table
        self.history_table.setRowCount(history_df.shape[0])
        self.history_table.setColumnCount(history_df.shape[1])
        self.history_table.setHorizontalHeaderLabels(history_df.columns.tolist())
        
        # Populate the table
        for i in range(history_df.shape[0]):
            for j in range(history_df.shape[1]):
                value = history_df.iat[i, j]
                # Format numeric values
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        item_text = f"{value:.4f}"
                    else:
                        item_text = str(value)
                else:
                    item_text = str(value)
                
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)
                self.history_table.setItem(i, j, item)
        
        # Resize columns to content
        self.history_table.resizeColumnsToContents()
        
        # Highlight the table header to show it has new data
        if history_df.shape[0] > 0:
            self.history_table.setStyleSheet("""
                QTableWidget::horizontalHeader {
                    background-color: #28a745;
                    color: white;
                    font-weight: bold;
                }
            """)
    
    def update_learning_curves(self):
        """Update the learning curves visualization."""
        if not self.session or not hasattr(self, 'learning_curves_plot'):
            return
        
        performance_history = self.session.get_performance_history()
        
        self.learning_curves_plot.clear()
        fig = self.learning_curves_plot.figure
        
        # Handle empty or insufficient data
        if not performance_history:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Learning Curves\n\nNo iteration data available yet\n\nPlease complete a few iteration cycles:\n1. Run analysis to get recommendations\n2. Input experimental results\n3. Repeat steps 1-2\n\nAt least 2 iterations needed for curves', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            fig.tight_layout()
            self.learning_curves_plot.canvas.draw()
            return
        
        # Handle single data point
        if len(performance_history) == 1:
            ax = fig.add_subplot(111)
            record = performance_history[0]
            r2_score = record['model_performance'].get('r2_score', None)
            print(f"UI DEBUG: R2 score from record: {r2_score} (type: {type(r2_score)})")
            
            if r2_score is not None and isinstance(r2_score, (int, float)) and not np.isnan(r2_score):
                r2_text = f"{r2_score:.4f}"
            else:
                r2_text = "Calculating..."
            
            best_value = record.get('best_value', 0)
            uncertainty = record.get('uncertainty_metrics', {}).get('mean_uncertainty', 0)
            
            info_text = f"""First Iteration Completed!
            
Iteration: {record['iteration']}
Model R2 Score: {r2_text}
Best Value: {best_value:.4f}
Mean Uncertainty: {uncertainty:.4f}

Learning curves will appear after more iterations"""
            
            ax.text(0.5, 0.5, info_text, ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            fig.tight_layout()
            self.learning_curves_plot.canvas.draw()
            return
        
        # Extract data
        iterations = [record['iteration'] for record in performance_history]
        r2_scores = [record['model_performance'].get('r2_score', 0) for record in performance_history]
        best_values = [record['best_value'] for record in performance_history]
        mean_uncertainties = [record['uncertainty_metrics']['mean_uncertainty'] for record in performance_history]
        max_acquisitions = [record['acquisition_stats']['max_acquisition'] for record in performance_history]
        
        # Create 2x2 subplots
        axes = fig.subplots(2, 2)
        
        # 1. Model Performance (R² Score)
        axes[0, 0].plot(iterations, r2_scores, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Model Performance Evolution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Best Value Found
        axes[0, 1].plot(iterations, best_values, 'g-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Best Value Discovery', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Best Target Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Uncertainty Reduction
        axes[1, 0].plot(iterations, mean_uncertainties, 'r-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Uncertainty Reduction', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Mean Prediction Uncertainty')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Acquisition Function Evolution
        axes[1, 1].plot(iterations, max_acquisitions, 'm-o', linewidth=2, markersize=6)
        axes[1, 1].set_title('Acquisition Function Evolution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Max Acquisition Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        self.learning_curves_plot.canvas.draw()
    
    def update_feature_evolution(self):
        """Update the feature importance evolution visualization."""
        if not self.session or not hasattr(self, 'feature_evolution_plot'):
            return
        
        feature_history = self.session.get_feature_importance_evolution()
        
        self.feature_evolution_plot.clear()
        fig = self.feature_evolution_plot.figure
        
        if len(feature_history) < 2:
            ax = fig.add_subplot(111)
            if len(feature_history) == 0:
                message = 'Feature Importance Evolution\n\nNo iteration data available yet\n\nPlease complete a few iteration cycles\nto view feature importance changes'
                color = "lightblue"
            else:
                message = 'Feature Importance Evolution\n\nFirst iteration completed\n\nMore iterations needed\nto show importance trends'
                color = "lightgreen"
            
            ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            fig.tight_layout()
            self.feature_evolution_plot.canvas.draw()
            return
        
        ax = fig.add_subplot(111)
        
        # Extract feature importance data
        iterations = [record['iteration'] for record in feature_history]
        
        # Get all feature names
        all_features = set()
        for record in feature_history:
            if hasattr(record['importance'], 'index'):
                all_features.update(record['importance'].index)
            elif isinstance(record['importance'], dict):
                all_features.update(record['importance'].keys())
        
        all_features = list(all_features)[:10]  # Limit to top 10 features for readability
        
        # Plot feature importance evolution
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_features)))
        
        for i, feature in enumerate(all_features):
            importance_values = []
            for record in feature_history:
                if hasattr(record['importance'], 'get'):
                    importance_values.append(record['importance'].get(feature, 0))
                elif isinstance(record['importance'], dict):
                    importance_values.append(record['importance'].get(feature, 0))
                else:
                    importance_values.append(0)
            
            ax.plot(iterations, importance_values, 'o-', 
                   color=colors[i], label=feature, linewidth=2, markersize=4)
        
        ax.set_title('Feature Importance Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Feature Importance')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.tight_layout()
        self.feature_evolution_plot.canvas.draw()
    
    def update_exploration_analysis(self):
        """Update the exploration vs exploitation analysis."""
        if not self.session or not hasattr(self, 'exploration_analysis_plot'):
            return
        
        performance_history = self.session.get_performance_history()
        
        self.exploration_analysis_plot.clear()
        fig = self.exploration_analysis_plot.figure
        
        if len(performance_history) < 2:
            ax = fig.add_subplot(111)
            if len(performance_history) == 0:
                message = 'Exploration Analysis\n\nNo iteration data available yet\n\nPlease complete a few iteration cycles\nto analyze exploration vs exploitation balance'
                color = "lightblue"
            else:
                message = 'Exploration Analysis\n\nFirst iteration completed\n\nMore iterations needed\nto show exploration analysis charts'
                color = "lightgreen"
            
            ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            fig.tight_layout()
            self.exploration_analysis_plot.canvas.draw()
            return
        
        # Create subplots
        axes = fig.subplots(2, 2)
        
        # Extract data
        iterations = [record['iteration'] for record in performance_history]
        mean_acquisitions = [record['acquisition_stats']['mean_acquisition'] for record in performance_history]
        max_acquisitions = [record['acquisition_stats']['max_acquisition'] for record in performance_history]
        top_10_means = [record['acquisition_stats']['top_10_mean'] for record in performance_history]
        mean_uncertainties = [record['uncertainty_metrics']['mean_uncertainty'] for record in performance_history]
        
        # 1. Acquisition Score Distribution
        axes[0, 0].plot(iterations, mean_acquisitions, 'b-o', label='Mean Acquisition', linewidth=2)
        axes[0, 0].plot(iterations, max_acquisitions, 'r-o', label='Max Acquisition', linewidth=2)
        axes[0, 0].plot(iterations, top_10_means, 'g-o', label='Top 10 Mean', linewidth=2)
        axes[0, 0].set_title('Acquisition Function Statistics', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Acquisition Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Exploration Efficiency
        if len(iterations) > 1:
            exploration_efficiency = np.diff(mean_uncertainties) / np.diff(iterations)
            axes[0, 1].plot(iterations[1:], exploration_efficiency, 'purple', linewidth=2, marker='o')
            axes[0, 1].set_title('Exploration Efficiency', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Uncertainty Reduction Rate')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Uncertainty vs Acquisition Relationship
        axes[1, 0].scatter(mean_uncertainties, mean_acquisitions, 
                          c=iterations, cmap='viridis', s=60, alpha=0.7)
        axes[1, 0].set_title('Uncertainty vs Acquisition', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Mean Uncertainty')
        axes[1, 0].set_ylabel('Mean Acquisition Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative Learning Progress
        if len(performance_history) > 0:
            cumulative_points = list(range(1, len(iterations) + 1))
            r2_scores = [record['model_performance'].get('r2_score', 0) for record in performance_history]
            
            axes[1, 1].plot(cumulative_points, r2_scores, 'orange', linewidth=3, marker='o')
            axes[1, 1].set_title('Cumulative Learning Progress', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Number of Added Points')
            axes[1, 1].set_ylabel('Model R² Score')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
        
        fig.tight_layout()
        self.exploration_analysis_plot.canvas.draw()
    
    def test_recommendation_display(self):
        """Test method to force update recommendation display."""
        print("DEBUG: Testing recommendation display...")
        if self.results_data is not None:
            test_text = f"TEST: Found {len(self.results_data)} results with top acquisition score: {self.results_data.iloc[0]['acquisition_score']:.6f}"
            self.recommendation_text.setText(test_text)
            print(f"DEBUG: Set test text: {test_text}")
        else:
            self.recommendation_text.setText("TEST: No results data available")
            print("DEBUG: No results data for test")
    
    def show_constraint_dialog(self):
        """Show constraint configuration dialog."""
        dialog = ConstraintConfigDialog(self.training_data, self)
        if dialog.exec_() == QDialog.Accepted:
            self.hard_constraints, self.soft_constraints, self.feasibility_weights = dialog.get_constraints()
            self.enable_constraint_optimization = True
    
    def calculate_feasibility_scores(self, candidates):
        """Calculate feasibility scores for candidates based on constraints."""
        if not self.enable_constraint_optimization or not self.hard_constraints:
            return np.ones(len(candidates))
        
        scores = np.ones(len(candidates))
        
        for i, candidate in candidates.iterrows():
            score = 1.0
            
            # Check hard constraints
            for feature, (min_val, max_val) in self.hard_constraints.items():
                if feature in candidate:
                    value = candidate[feature]
                    if not (min_val <= value <= max_val):
                        score = 0.0
                        break
            
            # Apply soft constraints
            if score > 0:
                for feature, (min_val, max_val) in self.soft_constraints.items():
                    if feature in candidate:
                        value = candidate[feature]
                        if not (min_val <= value <= max_val):
                            penalty = self.feasibility_weights.get(feature, 0.5)
                            score *= (1.0 - penalty)
            
            scores[i] = score
        
        return scores
    
    def apply_advanced_uncertainty_quantification(self, model, X_train, y_train, X_virtual):
        """Apply advanced uncertainty quantification methods."""
        method = self.uncertainty_combo.currentText()
        ensemble_size = self.ensemble_size_spin.value()
        
        uncertainties = {}
        
        if method == "ensemble":
            predictions = []
            for i in range(ensemble_size):
                # Bootstrap sampling
                bootstrap_indices = np.random.choice(len(X_train), len(X_train), replace=True)
                X_boot = X_train.iloc[bootstrap_indices]
                y_boot = y_train.iloc[bootstrap_indices]
                
                # Clone and train model
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_boot, y_boot)
                pred = model_clone.predict(X_virtual)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            uncertainties['ensemble_mean'] = np.mean(predictions, axis=0)
            uncertainties['ensemble_std'] = np.std(predictions, axis=0)
            uncertainties['ensemble_epistemic'] = np.var(predictions, axis=0)
            
        elif method == "mc_dropout":
            # Simplified MC Dropout implementation
            # This would require special model architecture in practice
            n_samples = ensemble_size
            predictions = []
            
            for _ in range(n_samples):
                # Add noise to simulate dropout effect
                noise_factor = 0.1
                X_noisy = X_virtual + np.random.normal(0, noise_factor, X_virtual.shape)
                pred = model.predict(X_noisy)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            uncertainties['mc_dropout_mean'] = np.mean(predictions, axis=0)
            uncertainties['mc_dropout_std'] = np.std(predictions, axis=0)
            
        elif method == "gaussian_process":
            # Enhanced GP uncertainty
            if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                try:
                    mean_pred, std_pred = model.predict(X_virtual, return_std=True)
                    uncertainties['gp_mean'] = mean_pred
                    uncertainties['gp_std'] = std_pred
                    uncertainties['gp_confidence_interval'] = 1.96 * std_pred
                except:
                    # Fallback for non-GP models
                    pred = model.predict(X_virtual)
                    uncertainties['gp_mean'] = pred
                    uncertainties['gp_std'] = np.std(pred) * np.ones_like(pred)
        
        elif method == "deep_ensemble":
            # Deep ensemble with different initializations
            predictions = []
            for i in range(ensemble_size):
                # Different random states for diversity
                model_copy = clone(model)
                if hasattr(model_copy, 'random_state'):
                    model_copy.random_state = i
                
                model_copy.fit(X_train, y_train)
                pred = model_copy.predict(X_virtual)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            uncertainties['deep_ensemble_mean'] = np.mean(predictions, axis=0)
            uncertainties['deep_ensemble_std'] = np.std(predictions, axis=0)
            uncertainties['deep_ensemble_disagreement'] = np.max(predictions, axis=0) - np.min(predictions, axis=0)
        
        return uncertainties
    
    def check_early_stopping_criteria(self, current_performance, iteration):
        """Check if early stopping criteria are met."""
        if not self.early_stopping_checkbox.isChecked():
            return False
        
        self.performance_history.append(current_performance)
        
        # Need at least 3 iterations to check convergence
        if len(self.performance_history) < 3:
            return False
        
        # Check for convergence
        recent_performance = self.performance_history[-self.stagnation_spin.value():]
        if len(recent_performance) >= self.stagnation_spin.value():
            improvement = max(recent_performance) - min(recent_performance)
            if improvement < self.convergence_spin.value():
                return True
        
        return False
    
    def calculate_cost_benefit_ratio(self, iteration, performance_gain, computational_cost):
        """Calculate cost-benefit ratio for dynamic budget allocation."""
        if computational_cost == 0:
            return float('inf')
        
        ratio = performance_gain / computational_cost
        self.cost_benefit_data.append({
            'iteration': iteration,
            'performance_gain': performance_gain,
            'computational_cost': computational_cost,
            'ratio': ratio
        })
        
        return ratio
    
    def generate_batch_recommendations(self, results_data, batch_size):
        if len(results_data) < batch_size:
            return results_data
        
        # Sort by acquisition function value
        sorted_data = results_data.sort_values('acquisition_value', ascending=False)
        
        # Select diverse candidates
        selected_indices = []
        candidates = sorted_data.copy()
        
        # Select first candidate (highest acquisition)
        selected_indices.append(candidates.index[0])
        selected_candidates = candidates.iloc[[0]]
        
        # Select remaining candidates based on diversity
        feature_columns = self.get_selected_features()
        
        for _ in range(batch_size - 1):
            if len(candidates) <= 1:
                break
            
            # Remove already selected candidates
            remaining_candidates = candidates.drop(selected_indices)
            if len(remaining_candidates) == 0:
                break
            
            # Calculate diversity scores
            diversity_scores = []
            
            for idx, candidate in remaining_candidates.iterrows():
                # Calculate minimum distance to selected candidates
                min_distance = float('inf')
                
                for _, selected in selected_candidates.iterrows():
                    # Euclidean distance in feature space
                    distance = np.sqrt(np.sum((candidate[feature_columns] - selected[feature_columns]) ** 2))
                    min_distance = min(min_distance, distance)
                
                # Combine acquisition value and diversity
                acquisition_score = candidate['acquisition_value']
                diversity_score = min_distance
                combined_score = acquisition_score * 0.7 + diversity_score * 0.3
                
                diversity_scores.append(combined_score)
            
            # Select candidate with highest combined score
            best_idx = np.argmax(diversity_scores)
            selected_idx = remaining_candidates.index[best_idx]
            selected_indices.append(selected_idx)
            
            # Add to selected candidates
            new_candidate = remaining_candidates.iloc[[best_idx]]
            selected_candidates = pd.concat([selected_candidates, new_candidate])
        
        return sorted_data.loc[selected_indices]
    
    def update_cost_benefit_visualization(self):

        if not self.cost_benefit_data:
            return
        
        # Create new tab for cost-benefit analysis if not exists
        cost_benefit_tab_exists = False
        for i in range(self.results_tabs.count()):
            if self.results_tabs.tabText(i) == "Cost-Benefit Analysis":
                cost_benefit_tab_exists = True
                break
        
        if not cost_benefit_tab_exists:
            self.cost_benefit_tab = QWidget()
            cost_benefit_layout = QVBoxLayout(self.cost_benefit_tab)
            
            self.cost_benefit_plot = MatplotlibWidget()
            cost_benefit_layout.addWidget(self.cost_benefit_plot)
            
            self.results_tabs.addTab(self.cost_benefit_tab, "Cost-Benefit Analysis")
        
        # Plot cost-benefit analysis
        self.cost_benefit_plot.clear()
        fig = self.cost_benefit_plot.figure
        
        # Create subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Performance over iterations
        ax1 = fig.add_subplot(gs[0, 0])
        iterations = [d['iteration'] for d in self.cost_benefit_data]
        performance_gains = [d['performance_gain'] for d in self.cost_benefit_data]
        ax1.plot(iterations, performance_gains, 'bo-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Performance Gain')
        ax1.set_title('Performance Improvement')
        ax1.grid(True, alpha=0.3)
        
        # Cost over iterations
        ax2 = fig.add_subplot(gs[0, 1])
        computational_costs = [d['computational_cost'] for d in self.cost_benefit_data]
        ax2.plot(iterations, computational_costs, 'ro-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Computational Cost')
        ax2.set_title('Computational Cost')
        ax2.grid(True, alpha=0.3)
        
        # Cost-benefit ratio
        ax3 = fig.add_subplot(gs[1, 0])
        ratios = [d['ratio'] for d in self.cost_benefit_data]
        ax3.plot(iterations, ratios, 'go-', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Benefit/Cost Ratio')
        ax3.set_title('Cost-Benefit Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Cumulative performance vs cost
        ax4 = fig.add_subplot(gs[1, 1])
        cumulative_performance = np.cumsum(performance_gains)
        cumulative_cost = np.cumsum(computational_costs)
        ax4.plot(cumulative_cost, cumulative_performance, 'mo-', linewidth=2)
        ax4.set_xlabel('Cumulative Cost')
        ax4.set_ylabel('Cumulative Performance')
        ax4.set_title('Performance vs Cost Trade-off')
        ax4.grid(True, alpha=0.3)
        
        self.cost_benefit_plot.canvas.draw()


class CandidateGeneratorDialog(QDialog):
    """Simplified dialog for generating candidate sets - redesigned for performance."""
    
    def __init__(self, training_data, numeric_columns, parent=None):
        super().__init__(parent)
        self.training_data = training_data
        # 🔧 不再限制列数，保持所有数值列
        self.numeric_columns = numeric_columns
        self.generated_candidates = None
        
        self.setWindowTitle("Candidate Set Generator")
        self.setModal(True)
        self.setFixedSize(700, 600)  # 稍微增大以容纳更多特征信息
        
        print(f"Initializing candidate generator with {len(self.numeric_columns)} columns")
        print(f"Data shape: {training_data.shape}")
        if len(self.numeric_columns) > 100:
            print(f"⚠️ Large feature set detected ({len(self.numeric_columns)} features)")
        self.init_simple_ui()
    
    def init_simple_ui(self):

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # 标题
        title = QLabel("🎯 Candidate Set Generator")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 使用标签页来分步骤
        self.tab_widget = QTabWidget()
        
        # 第一步：基本设置
        self.create_basic_tab()
        
        # 第二步：高级选项（可选）
        self.create_advanced_tab()
        
        layout.addWidget(self.tab_widget)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("📊 Preview Settings")
        self.preview_btn.clicked.connect(self.preview_settings)
        
        self.generate_btn = QPushButton("🎯 Generate Candidate Set")
        self.generate_btn.clicked.connect(self.quick_generate)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        self.use_btn = QPushButton("✅ Use Data")
        self.use_btn.clicked.connect(self.accept)
        self.use_btn.setEnabled(False)
        
        button_layout.addWidget(self.preview_btn)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.use_btn)
        
        layout.addLayout(button_layout)
        
        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
    
    def create_basic_tab(self):
        """创建基本设置标签页"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
       
        self.target_combo = QComboBox()
        self.target_combo.addItems(self.numeric_columns)
        layout.addRow("Target Variable:", self.target_combo)
        
        # 生成方法 - 简化选项
        self.method_combo = QComboBox()
        methods = [
            ("latin_hypercube", "Latin Hypercube Sampling (Recommended)"),
            ("random_sampling", "Random Sampling"),
            ("sobol", "Sobol Sequence"),
            ("grid_search", "Grid Search")
        ]
        for value, text in methods:
            self.method_combo.addItem(text, value)
        layout.addRow("Generation Method:", self.method_combo)
        
        # 批次大小设置
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1000, 10000)
        self.batch_size_spin.setValue(5000)
        self.batch_size_spin.setToolTip("Number of samples generated per iteration")
        layout.addRow("Batch Size:", self.batch_size_spin)
        
        # 样本数量
        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(100, 50000)
        self.n_samples_spin.setValue(5000)
        layout.addRow("Number of Samples:", self.n_samples_spin)
        
        # 特征选择 - 简化显示
        feature_info = QLabel(f"All {len(self.numeric_columns)} numeric features will be used (excluding target)")
        feature_info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addRow("Feature Selection:", feature_info)
        
        # 范围扩展
        self.expansion_spin = QDoubleSpinBox()
        self.expansion_spin.setRange(0.5, 3.0)
        self.expansion_spin.setSingleStep(0.1)
        self.expansion_spin.setValue(1.2)
        layout.addRow("Expansion Factor:", self.expansion_spin)
        
        self.tab_widget.addTab(widget, "🔧 Basic Settings")
    
    def create_advanced_tab(self):
        """创建高级选项标签页"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # 随机种子
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        layout.addRow("Random Seed:", self.seed_spin)
        
        # 约束条件 - 简化为文本输入
        constraint_label = QLabel("Constraints (optional, one per line):")
        layout.addRow(constraint_label)
        
        self.constraints_text = QTextEdit()
        self.constraints_text.setMaximumHeight(100)
        self.constraints_text.setPlaceholderText("For example: feature1 >= 0.1\nfeature2 <= 0.9")
        layout.addRow(self.constraints_text)
        
        # 生成模式
        self.guarantee_mode = QCheckBox("Ensure Exact Sample Count (Ignore Constraint Filtering)")
        layout.addRow(self.guarantee_mode)
        
        self.tab_widget.addTab(widget, "⚙️ Advanced Options")
    
    def preview_settings(self):
        """预览当前设置"""
        target = self.target_combo.currentText()
        method = self.method_combo.currentData()
        n_samples = self.n_samples_spin.value()
        expansion = self.expansion_spin.value()
        
        available_features = [col for col in self.numeric_columns if col != target]
        
        msg = f"Preview of Generation Settings:\n\n"
        msg += f"Target Variable: {target}\n"
        msg += f"Generation Method: {self.method_combo.currentText()}\n"
        msg += f"Number of Samples: {n_samples:,}\n"
        msg += f"Available Features: {len(available_features)} 个\n"
        msg += f"Expansion Factor: {expansion}x\n"
        
        constraints = self.get_constraints_simple()
        if constraints:
            msg += f"Constraints: {len(constraints)} 个\n"
        
        if method == "grid_search":
            estimated = 10 ** len(available_features)
            msg += f"\n⚠️ Grid Search: {estimated:,} samples"
        
        QMessageBox.information(self, "Preview of Generation Settings", msg)
    
    def get_constraints_simple(self):
        """Get simplified constraints"""
        text = self.constraints_text.toPlainText().strip()
        if not text:
            return []
        
        constraints = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                constraints.append(line)
        
        return constraints
    
    def quick_generate(self):
        """Quickly generate candidate set"""
        try:
            # Show progress
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)  # 不确定进度
            self.generate_btn.setEnabled(False)
            
            # 强制更新界面
            QApplication.processEvents()
            
            # 获取参数
            target = self.target_combo.currentText()
            method = self.method_combo.currentData()
            n_samples = self.n_samples_spin.value()
            expansion = self.expansion_spin.value()
            seed = self.seed_spin.value()
            constraints = self.get_constraints_simple()
            
            available_features = [col for col in self.numeric_columns if col != target]
            
            if not available_features:
                QMessageBox.warning(self, "Error", "No available feature variables!")
                return
            
            print(f"Generating {n_samples} candidates using {method}...")
            
            # 创建优化器
            from modules.active_learning_optimizer import ActiveLearningOptimizer
            optimizer = ActiveLearningOptimizer(random_state=seed)
            
            # 生成候选集 - 使用更简单的参数
            generation_params = {}
            if method == "grid_search":
                generation_params['grid_points'] = 10  # 固定网格点数
            else:
                generation_params['n_samples'] = n_samples
            
            self.progress.setFormat("Generating candidate set...")
            QApplication.processEvents()
            
            # 检查是否需要迭代生成
            use_iterative = (constraints and len(constraints) > 0 and method != "grid_search")
            
            if use_iterative:
                # 创建并显示进度对话框
                progress_dialog = IterativeProgressDialog(self)
                progress_dialog.show()
                QApplication.processEvents()
                
                # 设置进度回调
                def progress_callback(iteration, valid_samples, total_attempts, success_rate, batch_size, target_samples):
                    progress_dialog.update_progress(iteration, valid_samples, total_attempts, success_rate, batch_size, target_samples)
                    QApplication.processEvents()
                
                optimizer.set_progress_callback(progress_callback)
                # 设置批次大小
                if hasattr(self, 'batch_size_spin'):
                    optimizer.set_batch_size(self.batch_size_spin.value())
                
                try:
                    # 调用生成方法
                    self.generated_candidates = optimizer.generate_candidate_set_from_training(
                        training_df=self.training_data,
                        feature_columns=available_features,
                        method=method,
                        expansion_factor=expansion,
                        constraints=constraints if constraints else None,
                        random_seed=seed,
                        **generation_params
                    )
                finally:
                    optimizer.set_progress_callback(None)
                    progress_dialog.close()
            else:
                # 标准生成
                self.generated_candidates = optimizer.generate_candidate_set_from_training(
                    training_df=self.training_data,
                    feature_columns=available_features,
                    method=method,
                    expansion_factor=expansion,
                    constraints=constraints if constraints else None,
                    random_seed=seed,
                    **generation_params
                )
            
            # 完成
            self.progress.setVisible(False)
            self.generate_btn.setEnabled(True)
            self.use_btn.setEnabled(True)
            
            # 显示结果
            result_msg = f"✅ Successfully generated {len(self.generated_candidates)} candidate samples!\n\n"
            result_msg += f"Data Dimension: {len(self.generated_candidates)} × {len(self.generated_candidates.columns)}\n"
            result_msg += f"Feature Columns: {', '.join(available_features[:5])}"
            if len(available_features) > 5:
                result_msg += f" ... (+{len(available_features)-5} more)"
            
            QMessageBox.information(self, "Generation Completed", result_msg)
            
        except Exception as e:
            self.progress.setVisible(False)
            self.generate_btn.setEnabled(True)
            
            error_msg = f"Generation Failed: {str(e)}\n\n"
            error_msg += "Suggestions:\n"
            error_msg += "• Check constraint syntax\n"
            error_msg += "• Reduce sample size\n"
            error_msg += "• Try other generation methods"
            
            QMessageBox.critical(self, "Generation Error", error_msg)
    
    def get_generated_candidates(self):
        """Return generated candidate data"""
        return self.generated_candidates


class DataPreviewDialog(QDialog):
    """Dialog for previewing generated candidate data with statistics and visualizations."""
    
    def __init__(self, candidate_data, training_data, parent=None):
        super().__init__(parent)
        self.candidate_data = candidate_data
        self.training_data = training_data
        
        self.setWindowTitle("Generated Data Preview & Statistics")
        self.setModal(True)
        self.setFixedSize(1200, 800)  # Much larger for better visualization
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the preview dialog UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📊 Generated Candidate Data Preview")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Summary statistics
        summary_group = QGroupBox("📈 Summary Statistics")
        summary_layout = QVBoxLayout(summary_group)
        
        # Basic info
        info_text = f"""
        <b>Generated Samples:</b> {len(self.candidate_data):,} rows × {len(self.candidate_data.columns)} columns<br>
        <b>Training Data:</b> {len(self.training_data):,} rows × {len(self.training_data.columns)} columns<br>
        <b>Coverage Expansion:</b> Generated data explores beyond training boundaries
        """
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #333; font-size: 11px; margin: 10px;")
        summary_layout.addWidget(info_label)
        
        layout.addWidget(summary_group)
        
        # Tabbed visualization area
        tab_widget = QTabWidget()
        
        # Data table tab
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        
        # Sample data table
        self.data_table = QTableWidget()
        self.update_data_table()
        table_layout.addWidget(self.data_table)
        
        tab_widget.addTab(table_tab, "📋 Data Table")
        
        # Statistics comparison tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        self.stats_widget = MatplotlibWidget()
        self.update_statistics_plot()
        stats_layout.addWidget(self.stats_widget)
        
        tab_widget.addTab(stats_tab, "📊 Statistics")
        
        # Distribution comparison tab
        dist_tab = QWidget()
        dist_layout = QVBoxLayout(dist_tab)
        
        # Feature selection for distribution
        dist_control_layout = QHBoxLayout()
        
        dist_label = QLabel("Select Feature:")
        dist_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        dist_control_layout.addWidget(dist_label)
        
        self.feature_combo = QComboBox()
        self.feature_combo.setStyleSheet("font-size: 11px;")
        # Get common numeric features
        common_cols = list(set(self.candidate_data.columns) & set(self.training_data.columns))
        numeric_cols = []
        for col in common_cols:
            if self.candidate_data[col].dtype in ['int64', 'float64'] and self.training_data[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
        self.feature_combo.addItems(numeric_cols)
        self.feature_combo.currentTextChanged.connect(self.update_single_distribution_plot)
        dist_control_layout.addWidget(self.feature_combo)
        
        # Show all features button
        self.show_all_dist_btn = QPushButton("📊 Show All Features")
        self.show_all_dist_btn.setStyleSheet("font-size: 11px;")
        self.show_all_dist_btn.clicked.connect(self.update_distribution_plot)
        dist_control_layout.addWidget(self.show_all_dist_btn)
        
        dist_control_layout.addStretch()
        dist_layout.addLayout(dist_control_layout)
        
        self.dist_widget = MatplotlibWidget()
        self.initialize_distribution_plot()  # Start with welcome message
        dist_layout.addWidget(self.dist_widget)
        
        # Trigger initial plot if there are features available
        if self.feature_combo.count() > 0:
            # Set first item as selected and trigger update
            self.feature_combo.setCurrentIndex(0)
            self.update_single_distribution_plot()
        
        tab_widget.addTab(dist_tab, "📈 Distributions")
        
        # Coverage analysis tab
        coverage_tab = QWidget()
        coverage_layout = QVBoxLayout(coverage_tab)
        
        self.coverage_widget = MatplotlibWidget()
        self.update_coverage_plot()
        coverage_layout.addWidget(self.coverage_widget)
        
        tab_widget.addTab(coverage_tab, "🎯 Coverage Analysis")
        
        layout.addWidget(tab_widget)
        
        # Export buttons
        button_layout = QHBoxLayout()
        
        export_csv_btn = QPushButton("💾 Export CSV")
        export_csv_btn.clicked.connect(self.export_csv)
        export_csv_btn.setStyleSheet("font-size: 11px;")
        
        export_plot_btn = QPushButton("📸 Export Plots")
        export_plot_btn.clicked.connect(self.export_plots)
        export_plot_btn.setStyleSheet("font-size: 11px;")
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; font-size: 11px; padding: 8px 16px;")
        
        button_layout.addWidget(export_csv_btn)
        button_layout.addWidget(export_plot_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def update_data_table(self):
        """Update the data preview table."""
        # Show first 100 rows
        display_data = self.candidate_data.head(100)
        
        self.data_table.setRowCount(len(display_data))
        self.data_table.setColumnCount(len(display_data.columns))
        self.data_table.setHorizontalHeaderLabels(display_data.columns.tolist())
        
        # Fill table with data
        for i, row in enumerate(display_data.itertuples(index=False)):
            for j, value in enumerate(row):
                item = QTableWidgetItem(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
                self.data_table.setItem(i, j, item)
        
        # Resize columns
        self.data_table.resizeColumnsToContents()
        
        # Add info label if showing partial data
        if len(self.candidate_data) > 100:
            info_label = QLabel(f"Showing first 100 of {len(self.candidate_data):,} rows")
            info_label.setStyleSheet("color: #666; font-style: italic; margin: 5px;")
    
    def update_statistics_plot(self):
        """Update the statistics comparison plot."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Clear widget figure and create subplots directly on it
            self.stats_widget.figure.clear()
            axes = []
            for i in range(4):
                ax = self.stats_widget.figure.add_subplot(2, 2, i+1)
                axes.append(ax)
            axes = np.array(axes).reshape(2, 2)
            self.stats_widget.figure.suptitle('Generated vs Training Data Statistics', fontsize=14, fontweight='bold')
            
            # Get numeric columns common to both datasets
            common_cols = list(set(self.candidate_data.columns) & set(self.training_data.columns))
            numeric_cols = []
            for col in common_cols:
                if self.candidate_data[col].dtype in ['int64', 'float64'] and self.training_data[col].dtype in ['int64', 'float64']:
                    numeric_cols.append(col)
            
            if len(numeric_cols) == 0:
                axes[0, 0].text(0.5, 0.5, 'No common numeric columns found', 
                              ha='center', va='center', transform=axes[0, 0].transAxes)
                return
            
            # Select up to 6 features for comparison
            selected_cols = numeric_cols[:6]
            
            # 1. Mean comparison
            train_means = [self.training_data[col].mean() for col in selected_cols]
            gen_means = [self.candidate_data[col].mean() for col in selected_cols]
            
            x = np.arange(len(selected_cols))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, train_means, width, label='Training Data', alpha=0.8, color='skyblue')
            axes[0, 0].bar(x + width/2, gen_means, width, label='Generated Data', alpha=0.8, color='lightcoral')
            axes[0, 0].set_title('Mean Values Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels([col[:8] + '...' if len(col) > 8 else col for col in selected_cols], rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Standard deviation comparison
            train_stds = [self.training_data[col].std() for col in selected_cols]
            gen_stds = [self.candidate_data[col].std() for col in selected_cols]
            
            axes[0, 1].bar(x - width/2, train_stds, width, label='Training Data', alpha=0.8, color='skyblue')
            axes[0, 1].bar(x + width/2, gen_stds, width, label='Generated Data', alpha=0.8, color='lightcoral')
            axes[0, 1].set_title('Standard Deviation Comparison')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels([col[:8] + '...' if len(col) > 8 else col for col in selected_cols], rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Range comparison
            train_ranges = [self.training_data[col].max() - self.training_data[col].min() for col in selected_cols]
            gen_ranges = [self.candidate_data[col].max() - self.candidate_data[col].min() for col in selected_cols]
            
            axes[1, 0].bar(x - width/2, train_ranges, width, label='Training Data', alpha=0.8, color='skyblue')
            axes[1, 0].bar(x + width/2, gen_ranges, width, label='Generated Data', alpha=0.8, color='lightcoral')
            axes[1, 0].set_title('Range Comparison')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([col[:8] + '...' if len(col) > 8 else col for col in selected_cols], rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Coverage expansion ratio
            expansion_ratios = []
            for col in selected_cols:
                train_range = self.training_data[col].max() - self.training_data[col].min()
                gen_range = self.candidate_data[col].max() - self.candidate_data[col].min()
                ratio = gen_range / train_range if train_range > 0 else 1.0
                expansion_ratios.append(ratio)
            
            colors = ['green' if r >= 1.0 else 'orange' for r in expansion_ratios]
            axes[1, 1].bar(x, expansion_ratios, color=colors, alpha=0.7)
            axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Training Range')
            axes[1, 1].set_title('Range Expansion Ratio')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels([col[:8] + '...' if len(col) > 8 else col for col in selected_cols], rotation=45)
            axes[1, 1].set_ylabel('Generated Range / Training Range')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            self.stats_widget.figure.tight_layout()
            self.stats_widget.canvas.draw()
            self.stats_widget.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating statistics plot: {e}")
    
    def initialize_distribution_plot(self):
        """Initialize distribution plot with default message."""
        try:
            # Clear widget figure and draw directly on it
            self.dist_widget.figure.clear()
            ax = self.dist_widget.figure.add_subplot(111)
            
            ax.text(0.5, 0.5, 'Welcome to Distribution Analysis!\n\nPlease:\n1. Select a feature from the dropdown above\n2. Or click "Show All Features" for overview', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Feature Distribution Analysis', fontsize=16, fontweight='bold')
            
            self.dist_widget.canvas.draw()
            self.dist_widget.canvas.flush_events()
            
        except Exception as e:
            print(f"Error initializing distribution plot: {e}")

    def update_distribution_plot(self):
        """Update the distribution comparison plot for multiple features."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get numeric columns common to both datasets
            common_cols = list(set(self.candidate_data.columns) & set(self.training_data.columns))
            numeric_cols = []
            for col in common_cols:
                if self.candidate_data[col].dtype in ['int64', 'float64'] and self.training_data[col].dtype in ['int64', 'float64']:
                    numeric_cols.append(col)
            
            if len(numeric_cols) == 0:
                self.dist_widget.figure.clear()
                ax = self.dist_widget.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No common numeric columns found for distribution comparison', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title('Distribution Analysis', fontsize=14, fontweight='bold')
                self.dist_widget.canvas.draw()
                return
            
            # Select up to 6 features for distribution comparison
            selected_cols = numeric_cols[:6]
            n_cols = len(selected_cols)
            
            # Calculate subplot layout
            if n_cols <= 2:
                rows, cols = 1, 2
            elif n_cols <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 2, 3
            
            # Clear widget figure and create subplots directly on it
            self.dist_widget.figure.clear()
            self.dist_widget.figure.suptitle(f'Distribution Comparison: Training vs Generated Data ({n_cols} Features)', 
                        fontsize=14, fontweight='bold')
            
            axes = []
            for i in range(rows * cols):
                ax = self.dist_widget.figure.add_subplot(rows, cols, i+1)
                axes.append(ax)
            
            # axes is already a flat list, no need to flatten
            
            for i, col in enumerate(selected_cols):
                if i >= len(axes):
                    break
                
                # Get data and remove NaN values
                train_data = self.training_data[col].dropna()
                gen_data = self.candidate_data[col].dropna()
                
                # Plot histograms with better styling
                axes[i].hist(train_data, bins=25, alpha=0.7, label='Training Data', 
                           color='skyblue', density=True, edgecolor='navy', linewidth=0.5)
                axes[i].hist(gen_data, bins=25, alpha=0.7, label='Generated Data', 
                           color='lightcoral', density=True, edgecolor='darkred', linewidth=0.5)
                
                # Add statistics text
                train_mean = train_data.mean()
                gen_mean = gen_data.mean()
                train_std = train_data.std()
                gen_std = gen_data.std()
                
                stats_text = f'Train: μ={train_mean:.3f}, σ={train_std:.3f}\nGen: μ={gen_mean:.3f}, σ={gen_std:.3f}'
                axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Formatting
                title = col if len(col) <= 12 else col[:12] + '...'
                axes[i].set_title(title, fontsize=11, fontweight='bold')
                axes[i].set_xlabel('Value', fontsize=10)
                axes[i].set_ylabel('Density', fontsize=10)
                axes[i].legend(fontsize=9)
                axes[i].grid(True, alpha=0.3)
                
                # Set tick label size
                axes[i].tick_params(labelsize=9)
            
            # Hide unused subplots
            total_subplots = rows * cols
            for i in range(len(selected_cols), total_subplots):
                if i < len(axes):
                    axes[i].set_visible(False)
            
            self.dist_widget.figure.tight_layout()
            self.dist_widget.canvas.draw()
            self.dist_widget.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating distribution plot: {e}")
            # Show error message directly on widget's figure
            self.dist_widget.figure.clear()
            ax = self.dist_widget.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error displaying distribution plot:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Error in Distribution Analysis', fontsize=14, fontweight='bold')
            self.dist_widget.canvas.draw()
            self.dist_widget.canvas.flush_events()
    
    def update_single_distribution_plot(self):
        """Update the distribution plot for a single selected feature."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            selected_feature = self.feature_combo.currentText()
            if not selected_feature:
                # Show default message directly on widget's figure
                self.dist_widget.figure.clear()
                ax = self.dist_widget.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Please select a feature from the dropdown menu above', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title('Feature Distribution Analysis', fontsize=16, fontweight='bold')
                self.dist_widget.canvas.draw()
                return
            
            # Check if feature exists in both datasets
            if selected_feature not in self.candidate_data.columns or selected_feature not in self.training_data.columns:
                return
            
            # Check if feature is numeric
            if (self.candidate_data[selected_feature].dtype not in ['int64', 'float64'] or 
                self.training_data[selected_feature].dtype not in ['int64', 'float64']):
                return
            
            # Clear widget figure and create subplots directly on it
            self.dist_widget.figure.clear()
            axes = []
            for i in range(4):
                ax = self.dist_widget.figure.add_subplot(2, 2, i+1)
                axes.append(ax)
            axes = np.array(axes).reshape(2, 2)
            self.dist_widget.figure.suptitle(f'Distribution Analysis: {selected_feature}', fontsize=14, fontweight='bold')
            
            # Get data
            train_data = self.training_data[selected_feature].dropna()
            gen_data = self.candidate_data[selected_feature].dropna()
            
            # 1. Histogram comparison
            axes[0, 0].hist(train_data, bins=30, alpha=0.6, label='Training Data', 
                           color='skyblue', density=True, edgecolor='black', linewidth=0.5)
            axes[0, 0].hist(gen_data, bins=30, alpha=0.6, label='Generated Data', 
                           color='lightcoral', density=True, edgecolor='black', linewidth=0.5)
            axes[0, 0].set_title('Histogram Comparison')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Box plot comparison
            box_data = [train_data, gen_data]
            box_labels = ['Training', 'Generated']
            bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('skyblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            axes[0, 1].set_title('Box Plot Comparison')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Cumulative distribution
            train_sorted = np.sort(train_data)
            gen_sorted = np.sort(gen_data)
            train_cdf = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
            gen_cdf = np.arange(1, len(gen_sorted) + 1) / len(gen_sorted)
            
            axes[1, 0].plot(train_sorted, train_cdf, label='Training Data', color='blue', linewidth=2)
            axes[1, 0].plot(gen_sorted, gen_cdf, label='Generated Data', color='red', linewidth=2)
            axes[1, 0].set_title('Cumulative Distribution')
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Cumulative Probability')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Statistics comparison
            train_stats = {
                'Mean': train_data.mean(),
                'Std': train_data.std(),
                'Min': train_data.min(),
                'Max': train_data.max(),
                'Median': train_data.median(),
                'Q25': train_data.quantile(0.25),
                'Q75': train_data.quantile(0.75)
            }
            
            gen_stats = {
                'Mean': gen_data.mean(),
                'Std': gen_data.std(),
                'Min': gen_data.min(),
                'Max': gen_data.max(),
                'Median': gen_data.median(),
                'Q25': gen_data.quantile(0.25),
                'Q75': gen_data.quantile(0.75)
            }
            
            # Create statistics table
            stats_text = f"""
Statistics Comparison for {selected_feature}

{'Metric':<10} {'Training':<12} {'Generated':<12} {'Ratio':<8}
{'-'*50}
{'Mean':<10} {train_stats['Mean']:<12.4f} {gen_stats['Mean']:<12.4f} {gen_stats['Mean']/train_stats['Mean'] if train_stats['Mean'] != 0 else 0:<8.3f}
{'Std':<10} {train_stats['Std']:<12.4f} {gen_stats['Std']:<12.4f} {gen_stats['Std']/train_stats['Std'] if train_stats['Std'] != 0 else 0:<8.3f}
{'Min':<10} {train_stats['Min']:<12.4f} {gen_stats['Min']:<12.4f} {gen_stats['Min']/train_stats['Min'] if train_stats['Min'] != 0 else 0:<8.3f}
{'Max':<10} {train_stats['Max']:<12.4f} {gen_stats['Max']:<12.4f} {gen_stats['Max']/train_stats['Max'] if train_stats['Max'] != 0 else 0:<8.3f}
{'Median':<10} {train_stats['Median']:<12.4f} {gen_stats['Median']:<12.4f} {gen_stats['Median']/train_stats['Median'] if train_stats['Median'] != 0 else 0:<8.3f}
{'Q25':<10} {train_stats['Q25']:<12.4f} {gen_stats['Q25']:<12.4f} {gen_stats['Q25']/train_stats['Q25'] if train_stats['Q25'] != 0 else 0:<8.3f}
{'Q75':<10} {train_stats['Q75']:<12.4f} {gen_stats['Q75']:<12.4f} {gen_stats['Q75']/train_stats['Q75'] if train_stats['Q75'] != 0 else 0:<8.3f}

Range Expansion: {(gen_stats['Max'] - gen_stats['Min']) / (train_stats['Max'] - train_stats['Min']) if (train_stats['Max'] - train_stats['Min']) != 0 else 0:.3f}x
Coverage: Generated data covers {100 * (min(gen_stats['Max'], train_stats['Max']) - max(gen_stats['Min'], train_stats['Min'])) / (train_stats['Max'] - train_stats['Min']) if (train_stats['Max'] - train_stats['Min']) != 0 else 0:.1f}% of training range
            """
            
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Statistical Summary')
            
            self.dist_widget.figure.tight_layout()
            self.dist_widget.canvas.draw()
            self.dist_widget.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating single distribution plot: {e}")
            # Show error message directly on widget's figure
            self.dist_widget.figure.clear()
            ax = self.dist_widget.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error displaying distribution for {self.feature_combo.currentText()}:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Error in Distribution Analysis', fontsize=14, fontweight='bold')
            self.dist_widget.canvas.draw()
            self.dist_widget.canvas.flush_events()
    
    def update_coverage_plot(self):
        """Update the coverage analysis plot."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.patches import Rectangle
            
            # Get numeric columns common to both datasets
            common_cols = list(set(self.candidate_data.columns) & set(self.training_data.columns))
            numeric_cols = []
            for col in common_cols:
                if self.candidate_data[col].dtype in ['int64', 'float64'] and self.training_data[col].dtype in ['int64', 'float64']:
                    numeric_cols.append(col)
            
            if len(numeric_cols) < 2:
                return
            
            # Select first two features for 2D coverage plot
            col1, col2 = numeric_cols[0], numeric_cols[1]
            
            # Clear widget figure and create subplots directly on it
            self.coverage_widget.figure.clear()
            axes = []
            for i in range(2):
                ax = self.coverage_widget.figure.add_subplot(1, 2, i+1)
                axes.append(ax)
            self.coverage_widget.figure.suptitle(f'Coverage Analysis: {col1} vs {col2}', fontsize=14, fontweight='bold')
            
            # Plot 1: Scatter plot comparison
            axes[0].scatter(self.training_data[col1], self.training_data[col2], 
                          alpha=0.6, s=20, color='skyblue', label='Training Data')
            axes[0].scatter(self.candidate_data[col1], self.candidate_data[col2], 
                          alpha=0.6, s=20, color='lightcoral', label='Generated Data')
            
            # Add training data boundary rectangle
            train_min1, train_max1 = self.training_data[col1].min(), self.training_data[col1].max()
            train_min2, train_max2 = self.training_data[col2].min(), self.training_data[col2].max()
            
            rect = Rectangle((train_min1, train_min2), train_max1 - train_min1, train_max2 - train_min2,
                           linewidth=2, edgecolor='blue', facecolor='none', linestyle='--',
                           label='Training Boundary')
            axes[0].add_patch(rect)
            
            axes[0].set_xlabel(col1)
            axes[0].set_ylabel(col2)
            axes[0].set_title('Data Coverage Comparison')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Coverage metrics
            metrics = []
            labels = []
            
            # Calculate coverage metrics
            for col in numeric_cols[:6]:  # Limit to 6 features
                train_min, train_max = self.training_data[col].min(), self.training_data[col].max()
                gen_min, gen_max = self.candidate_data[col].min(), self.candidate_data[col].max()
                
                # Coverage expansion ratio
                train_range = train_max - train_min
                gen_range = gen_max - gen_min
                expansion = gen_range / train_range if train_range > 0 else 1.0
                
                metrics.append(expansion)
                labels.append(col[:10] + '...' if len(col) > 10 else col)
            
            colors = ['green' if m >= 1.0 else 'orange' for m in metrics]
            bars = axes[1].bar(range(len(metrics)), metrics, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, metric in zip(bars, metrics):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{metric:.2f}x', ha='center', va='bottom', fontsize=9)
            
            axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Training Range')
            axes[1].set_title('Range Expansion by Feature')
            axes[1].set_xlabel('Features')
            axes[1].set_ylabel('Expansion Ratio')
            axes[1].set_xticks(range(len(labels)))
            axes[1].set_xticklabels(labels, rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            self.coverage_widget.figure.tight_layout()
            self.coverage_widget.canvas.draw()
            self.coverage_widget.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating coverage plot: {e}")
    
    def export_csv(self):
        """Export generated candidates to CSV."""
        try:
            from PyQt5.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Generated Candidates", 
                "generated_candidates.csv", 
                "CSV Files (*.csv)"
            )
            
            if file_path:
                self.candidate_data.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data:\n{str(e)}")
    
    def export_plots(self):
        """Export all plots as images."""
        try:
            from PyQt5.QtWidgets import QFileDialog
            import os
            
            folder_path = QFileDialog.getExistingDirectory(self, "Select Export Folder")
            
            if folder_path:
                # Export statistics plot
                if hasattr(self.stats_widget, 'figure'):
                    stats_path = os.path.join(folder_path, "candidate_statistics.png")
                    self.stats_widget.figure.savefig(stats_path, dpi=300, bbox_inches='tight')
                
                # Export distribution plot
                if hasattr(self.dist_widget, 'figure'):
                    dist_path = os.path.join(folder_path, "candidate_distributions.png")
                    self.dist_widget.figure.savefig(dist_path, dpi=300, bbox_inches='tight')
                
                # Export coverage plot
                if hasattr(self.coverage_widget, 'figure'):
                    coverage_path = os.path.join(folder_path, "candidate_coverage.png")
                    self.coverage_widget.figure.savefig(coverage_path, dpi=300, bbox_inches='tight')
                
                QMessageBox.information(self, "Success", f"Plots exported to:\n{folder_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export plots:\n{str(e)}")


class IterativeProgressDialog(QDialog):
    """Dialog to show real-time progress for iterative candidate generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔄 Iterative Generation Progress")
        self.setModal(False)  # Non-modal so user can see the main dialog
        self.setFixedSize(600, 400)
        
        # Data for tracking progress
        self.iteration_data = {
            'iterations': [],
            'valid_samples': [],
            'total_attempts': [],
            'success_rates': [],
            'batch_sizes': []
        }
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the progress dialog UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🔄 Real-time Iterative Generation Progress")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Status info
        self.status_label = QLabel("Preparing to start iterative generation...")
        self.status_label.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(self.status_label)
        
        # Progress visualization
        self.progress_widget = MatplotlibWidget()
        layout.addWidget(self.progress_widget)
        
        # Log area
        log_label = QLabel("📋 Generation Log:")
        log_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.log_text)
        
        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("🛑 Cancel Generation")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_generation)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        # Initialize plot
        self.init_progress_plot()
    
    def init_progress_plot(self):
        """Initialize the progress visualization plot."""
        self.progress_widget.clear()
        ax = self.progress_widget.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Waiting for iterative generation to start...', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Iterative Generation Progress', fontsize=14, fontweight='bold')
        self.progress_widget.canvas.draw()
    
    def update_progress(self, iteration, valid_samples, total_attempts, success_rate, batch_size, target_samples):
        """Update the progress visualization with new data."""
        # Add data
        self.iteration_data['iterations'].append(iteration)
        self.iteration_data['valid_samples'].append(valid_samples)
        self.iteration_data['total_attempts'].append(total_attempts)
        self.iteration_data['success_rates'].append(success_rate * 100)  # Convert to percentage
        self.iteration_data['batch_sizes'].append(batch_size)
        
        # Update status
        self.status_label.setText(
            f"Iteration {iteration}: {valid_samples}/{target_samples} valid samples "
            f"({success_rate*100:.1f}% success rate, {total_attempts} total attempts)"
        )
        
        # Update plot
        self.update_plot()
    
    def update_plot(self):
        """Update the progress plot."""
        try:
            self.progress_widget.clear()
            
            if len(self.iteration_data['iterations']) == 0:
                return
            
            # Create subplots
            fig = self.progress_widget.figure
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            
            iterations = self.iteration_data['iterations']
            valid_samples = self.iteration_data['valid_samples']
            total_attempts = self.iteration_data['total_attempts']
            success_rates = self.iteration_data['success_rates']
            batch_sizes = self.iteration_data['batch_sizes']
            
            # Plot 1: Valid samples over iterations
            ax1.plot(iterations, valid_samples, 'o-', color='green', linewidth=2, markersize=4)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Valid Samples')
            ax1.set_title('Valid Samples Progress')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Success rate over iterations
            ax2.plot(iterations, success_rates, 'o-', color='orange', linewidth=2, markersize=4)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Constraint Success Rate')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Total attempts (cumulative)
            ax3.plot(iterations, total_attempts, 'o-', color='blue', linewidth=2, markersize=4)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Total Attempts')
            ax3.set_title('Cumulative Attempts')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Batch sizes
            ax4.bar(iterations, batch_sizes, alpha=0.7, color='purple')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Batch Size')
            ax4.set_title('Adaptive Batch Sizes')
            ax4.grid(True, alpha=0.3)
            
            fig.suptitle('Iterative Generation Real-time Analytics', fontsize=12, fontweight='bold')
            fig.tight_layout()
            self.progress_widget.canvas.draw()
            
        except Exception as e:
            print(f"Error updating progress plot: {e}")
    
    def add_log(self, message):
        """Add a message to the log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
        
        # Process events to update display
        QApplication.processEvents()
    
    def cancel_generation(self):
        """Cancel the generation process."""
        reply = QMessageBox.question(self, "Confirm Cancel", 
                                   "Are you sure you want to cancel the current candidate set generation?\nThe generated data will be lost.",
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 设置取消标志
            if hasattr(self.parent(), '_cancel_generation'):
                self.parent()._cancel_generation = True
            
            self.add_log("🛑 User requested to cancel generation...")
            self.status_label.setText("🛑 Canceling generation...")
            self.cancel_button.setText("Canceling...")
            self.cancel_button.setEnabled(False)
            
            # 关闭对话框
            QTimer.singleShot(1000, self.reject)


class OptimizedCandidateDialog(QDialog):
    """Optimized candidate generator - keep full functionality but solve UI freezing issues"""
    
    def __init__(self, training_data, numeric_columns, parent=None):
        super().__init__(parent)
        self.training_data = training_data
        # 🔧 修复：移除100列限制，支持大数据集
        self.numeric_columns = numeric_columns
        
        # 为大数据集提供性能优化，但不限制列数
        if len(numeric_columns) > 200:
            print(f"📊 Large dataset detected: {len(numeric_columns)} features")
            print("🚀 Enabling performance optimizations for UI responsiveness")
            self._large_dataset_mode = True
            self._batch_ui_updates = True  # 批量更新UI以提高性能
        else:
            self._large_dataset_mode = False
            self._batch_ui_updates = False
        
        self.available_features = []
        self.generated_candidates = None
        
        self.setWindowTitle("Candidate Set Generator")
        self.setModal(True)
        self.setMinimumSize(800, 700)
        self.resize(900, 750)
        
        # 异步初始化UI
        self.init_optimized_ui()
    
    def init_optimized_ui(self):
        """创建优化的全功能UI - 分步加载避免卡死"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # 标题
        title = QLabel("🔬 Candidate Set Generator")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 使用标签页但优化加载
        self.tab_widget = QTabWidget()
        
        # 第一步：立即创建基础标签页
        self.create_optimized_basic_tab()
        
        # 第二步：延迟创建高级功能
        QTimer.singleShot(50, self.create_optimized_advanced_tab)
        
        layout.addWidget(self.tab_widget)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 按钮区域
        self.create_button_panel(layout)
    
    def init_method_params(self):
        """初始化所有方法的参数控件"""
        # 清除现有控件
        for i in reversed(range(self.method_params_layout.count())): 
            self.method_params_layout.itemAt(i).widget().setParent(None)
        
        # 拉丁超立方参数
        self.lhs_scramble = QCheckBox("Enable Scramble")
        self.lhs_scramble.setChecked(True)
        self.lhs_scramble.setToolTip("Scramble can improve sample quality")
        
        # Sobol参数
        self.sobol_scramble = QCheckBox("Enable Scramble")
        self.sobol_scramble.setChecked(True)
        self.sobol_skip = QSpinBox()
        self.sobol_skip.setRange(0, 10000)
        self.sobol_skip.setValue(0)
        self.sobol_skip.setToolTip("Skip first N samples")
        
        # Halton参数
        self.halton_scramble = QCheckBox("Enable Scramble")
        self.halton_scramble.setChecked(True)
        
        # CVT参数
        self.cvt_iterations = QSpinBox()
        self.cvt_iterations.setRange(5, 50)
        self.cvt_iterations.setValue(20)
        self.cvt_iterations.setToolTip("K-means iterations")
        
        self.cvt_oversample = QSpinBox()
        self.cvt_oversample.setRange(5, 50)
        self.cvt_oversample.setValue(10)
        self.cvt_oversample.setToolTip("Oversampling factor")
        
        # Maximin参数
        self.maximin_pool_factor = QSpinBox()
        self.maximin_pool_factor.setRange(10, 100)
        self.maximin_pool_factor.setValue(20)
        self.maximin_pool_factor.setToolTip("Candidate pool factor")
        
        # 网格搜索参数
        self.grid_points_spin = QSpinBox()
        self.grid_points_spin.setRange(3, 20)
        self.grid_points_spin.setValue(10)
        self.grid_points_spin.setToolTip("Number of grid points per feature")
        
        # 默认显示拉丁超立方参数
        self.on_method_changed()
    
    def on_method_changed(self):
        """当采样方法改变时更新参数控件"""
        # 清除现有控件
        for i in reversed(range(self.method_params_layout.count())): 
            self.method_params_layout.itemAt(i).widget().setParent(None)
        
        method = self.method_combo.currentData()
        
        if method == "latin_hypercube":
            self.method_params_layout.addRow("Scramble:", self.lhs_scramble)
            help_text = QLabel("Latin Hypercube Sampling ensures uniform distribution across dimensions")
            help_text.setStyleSheet("color: #666; font-size: 10px;")
            self.method_params_layout.addRow(help_text)
            
        elif method == "sobol":
            self.method_params_layout.addRow("Scramble:", self.sobol_scramble)
            self.method_params_layout.addRow("Skip Samples:", self.sobol_skip)
            help_text = QLabel("Sobol sequence provides excellent high-dimensional uniformity")
            help_text.setStyleSheet("color: #666; font-size: 10px;")
            self.method_params_layout.addRow(help_text)
            
        elif method == "halton":
            self.method_params_layout.addRow("Scramble:", self.halton_scramble)
            help_text = QLabel("Halton sequence generates quickly, suitable for large samples")
            help_text.setStyleSheet("color: #666; font-size: 10px;")
            self.method_params_layout.addRow(help_text)
            
        elif method == "cvt":
            self.method_params_layout.addRow("K-means iterations:", self.cvt_iterations)
            self.method_params_layout.addRow("Oversampling factor:", self.cvt_oversample)
            help_text = QLabel("Centroidal Voronoi provides best spatial filling")
            help_text.setStyleSheet("color: #666; font-size: 10px;")
            self.method_params_layout.addRow(help_text)
            
        elif method == "maximin":
            self.method_params_layout.addRow("Candidate pool factor:", self.maximin_pool_factor)
            help_text = QLabel("Maximin design, maximizing sample-to-sample distance")
            help_text.setStyleSheet("color: #666; font-size: 10px;")
            self.method_params_layout.addRow(help_text)
            
        elif method == "grid_search":
            self.method_params_layout.addRow("Number of grid points:", self.grid_points_spin)
            help_text = QLabel("Systematic exploration of the design space")
            help_text.setStyleSheet("color: #666; font-size: 10px;")
            self.method_params_layout.addRow(help_text)
            
        elif method == "random_sampling":
            help_text = QLabel("Pure random sampling, as a baseline method")
            help_text.setStyleSheet("color: #666; font-size: 10px;")
            self.method_params_layout.addRow(help_text)
    
    def get_method_params(self):
        """Get current method parameters"""
        method = self.method_combo.currentData()
        params = {}
        
        if method == "latin_hypercube":
            params['scramble'] = self.lhs_scramble.isChecked()
            
        elif method == "sobol":
            params['scramble'] = self.sobol_scramble.isChecked()
            params['skip'] = self.sobol_skip.value()
            
        elif method == "halton":
            params['scramble'] = self.halton_scramble.isChecked()
            
        elif method == "cvt":
            params['iterations'] = self.cvt_iterations.value()
            params['oversample_factor'] = self.cvt_oversample.value()
            
        elif method == "maximin":
            params['pool_factor'] = self.maximin_pool_factor.value()
            
        elif method == "grid_search":
            params['grid_points'] = self.grid_points_spin.value()
            
        return params
    
    def create_optimized_basic_tab(self):
        """创建基础设置标签页 - 立即加载"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 目标变量选择
        target_group = QGroupBox("🎯 Target Variable Settings")
        target_layout = QVBoxLayout(target_group)
        
        # 模式选择
        mode_layout = QHBoxLayout()
        self.target_mode_group = QButtonGroup(self)
        self.single_target_radio = QRadioButton("Single-objective Optimization")
        self.multi_target_radio = QRadioButton("Multi-objective Optimization")
        self.single_target_radio.setChecked(True)
        
        self.target_mode_group.addButton(self.single_target_radio)
        self.target_mode_group.addButton(self.multi_target_radio)
        
        mode_layout.addWidget(self.single_target_radio)
        mode_layout.addWidget(self.multi_target_radio)
        mode_layout.addStretch()
        target_layout.addLayout(mode_layout)
        
        # 单目标选择
        self.single_target_widget = QWidget()
        single_layout = QFormLayout(self.single_target_widget)
        
        self.target_combo = QComboBox()
        self.target_combo.addItems(self.numeric_columns)
        single_layout.addRow("Target Variable:", self.target_combo)
        
        # 单目标的优化方向
        self.single_goal_combo = QComboBox()
        self.single_goal_combo.addItem("Maximize", "maximize")
        self.single_goal_combo.addItem("Minimize", "minimize")
        single_layout.addRow("Optimization Goal:", self.single_goal_combo)
        
        target_layout.addWidget(self.single_target_widget)
        
        # 多目标选择
        self.multi_target_widget = QWidget()
        multi_layout = QVBoxLayout(self.multi_target_widget)
        multi_layout.addWidget(QLabel("Select multiple target variables:"))
        
        # 创建一个更复杂的多目标选择界面
        self.target_scroll = QScrollArea()
        self.target_scroll.setMaximumHeight(150)
        self.target_scroll.setWidgetResizable(True)
        
        self.target_container = QWidget()
        self.target_container_layout = QVBoxLayout(self.target_container)
        self.target_checkboxes = {}  # 存储复选框
        self.target_goal_combos = {}  # 存储目标方向下拉框
        
        for col in self.numeric_columns:
            # 创建每个目标的行
            target_row = QWidget()
            target_row_layout = QHBoxLayout(target_row)
            target_row_layout.setContentsMargins(5, 2, 5, 2)
            
            # 复选框
            checkbox = QCheckBox(col)
            checkbox.setMinimumWidth(150)
            self.target_checkboxes[col] = checkbox
            target_row_layout.addWidget(checkbox)
            
            # 优化方向选择
            goal_combo = QComboBox()
            goal_combo.addItem("maximize", "maximize")
            goal_combo.addItem("minimize", "minimize")
            goal_combo.setEnabled(False)  # 默认禁用
            goal_combo.setMinimumWidth(80)
            self.target_goal_combos[col] = goal_combo
            target_row_layout.addWidget(goal_combo)
            
            # 连接信号：当复选框状态改变时启用/禁用目标方向选择
            checkbox.toggled.connect(lambda checked, combo=goal_combo: combo.setEnabled(checked))
            checkbox.toggled.connect(self.update_available_features_optimized)
            
            self.target_container_layout.addWidget(target_row)
        
        self.target_scroll.setWidget(self.target_container)
        multi_layout.addWidget(self.target_scroll)
        
        target_layout.addWidget(self.multi_target_widget)
        self.multi_target_widget.setVisible(False)
        
        # 连接信号
        self.single_target_radio.toggled.connect(self.on_target_mode_changed)
        self.target_combo.currentTextChanged.connect(self.update_available_features_optimized)
        
        layout.addWidget(target_group)
        
        # 生成方法选择
        method_group = QGroupBox("📊 Generation Method")
        method_layout = QFormLayout(method_group)
        
        self.method_combo = QComboBox()
        methods = [
            ("latin_hypercube", "Latin Hypercube Sampling (Recommended)"),
            ("sobol", "Sobol Sequence (High-dimensional Optimization)"),
            ("halton", "Halton Sequence (Fast Generation)"),
            ("random_sampling", "Random Sampling (Baseline Method)"),
            ("cvt", "Centroidal Voronoi Distribution (High Quality)"),
            ("maximin", "Maximin Distance (Small Sample)"),
            ("grid_search", "Grid Search (Systematic Exploration)")
        ]
        
        for method_id, display_name in methods:
            self.method_combo.addItem(display_name, method_id)
        
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addRow("Sampling Method:", self.method_combo)
        
        # 方法参数配置区域
        self.method_params_group = QGroupBox("📋 Method Parameters")
        self.method_params_layout = QFormLayout(self.method_params_group)
        method_layout.addRow(self.method_params_group)
        
        # 初始化参数控件
        self.init_method_params()
        
        # 样本数量
        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(100, 100000)
        self.n_samples_spin.setValue(5000)
        self.n_samples_spin.setSuffix(" samples")
        method_layout.addRow("Number of Samples:", self.n_samples_spin)
        
        # 范围扩展
        self.expansion_spin = QDoubleSpinBox()
        self.expansion_spin.setRange(0.5, 5.0)
        self.expansion_spin.setSingleStep(0.1)
        self.expansion_spin.setValue(1.2)
        self.expansion_spin.setSuffix("x")
        method_layout.addRow("Expansion Factor:", self.expansion_spin)
        
        layout.addWidget(method_group)
        
        # 特征选择 - 优化版本
        self.create_optimized_feature_selection(layout)
        
        self.tab_widget.addTab(widget, "🔧 Basic Settings")
    
    def create_optimized_feature_selection(self, layout):
        """创建优化的特征选择区域"""
        feature_group = QGroupBox("🎯Feature Selection")
        feature_layout = QVBoxLayout(feature_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        select_all_btn.clicked.connect(self.select_all_features_optimized)
        select_none_btn.clicked.connect(self.select_none_features_optimized)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(select_none_btn)
        button_layout.addStretch()
        
        self.feature_info_label = QLabel("Features will be automatically updated based on target variables")
        self.feature_info_label.setStyleSheet("color: #666; font-size: 11px;")
        button_layout.addWidget(self.feature_info_label)
        
        feature_layout.addLayout(button_layout)
        
        # 特征列表 - 使用虚拟化提高性能
        self.feature_list = QListWidget()
        self.feature_list.setMaximumHeight(200)
        feature_layout.addWidget(self.feature_list)
        
        layout.addWidget(feature_group)
        
        # 初始化特征列表
        self.update_available_features_optimized()
    
    def create_optimized_advanced_tab(self):
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 约束条件
        constraint_group = QGroupBox("📐 Constraints (Optional)")
        constraint_layout = QVBoxLayout(constraint_group)
        
        # 约束输入
        constraint_layout.addWidget(QLabel("Input constraints (one per line):"))
        self.constraints_text = QTextEdit()
        self.constraints_text.setMaximumHeight(120)
        self.constraints_text.setPlaceholderText(
            "For example:\n"
            "feature1 >= 0.1\n"
            "feature2 <= 0.9\n"
            "feature1 + feature2 <= 1.0"
        )
        constraint_layout.addWidget(self.constraints_text)
        
        # 约束模式
        self.guarantee_mode = QCheckBox("Ensure Exact Sample Count (Enable Iterative Generation)")
        self.guarantee_mode.setToolTip("Enable iterative generation to ensure obtaining a specified number of valid samples")
        constraint_layout.addWidget(self.guarantee_mode)
        
        layout.addWidget(constraint_group)
        
        # 高级参数
        advanced_group = QGroupBox("⚙️ Advanced Parameters")
        advanced_layout = QFormLayout(advanced_group)
        
        # 随机种子
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        advanced_layout.addRow("Random Seed:", self.seed_spin)
        
        # 网格搜索参数
        self.grid_points_spin = QSpinBox()
        self.grid_points_spin.setRange(2, 20)
        self.grid_points_spin.setValue(10)
        advanced_layout.addRow("Number of Grid Points per Feature:", self.grid_points_spin)
        
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        
        self.tab_widget.addTab(widget, "⚙️ Advanced Options")
    
    def create_button_panel(self, layout):
        """创建按钮面板"""
        button_layout = QHBoxLayout()
        
        # 预览按钮
        self.preview_btn = QPushButton("📊 Preview Settings")
        self.preview_btn.clicked.connect(self.preview_generation_optimized)
        
        # 数据预览按钮
        self.data_preview_btn = QPushButton("📈 Data Preview")
        self.data_preview_btn.clicked.connect(self.show_data_preview_optimized)
        self.data_preview_btn.setEnabled(False)
        
        # 主要操作按钮
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        self.generate_btn = QPushButton("🎯 Generate Candidate Set")
        self.generate_btn.clicked.connect(self.optimized_generate)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        
        self.use_btn = QPushButton("✅ Use Data")
        self.use_btn.clicked.connect(self.accept)
        self.use_btn.setEnabled(False)
        self.use_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover:enabled {
                background-color: #218838;
            }
        """)
        
        button_layout.addWidget(self.preview_btn)
        button_layout.addWidget(self.data_preview_btn)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.use_btn)
        
        layout.addLayout(button_layout)
    
    def on_target_mode_changed(self):
        """Target mode switch"""
        is_multi = self.multi_target_radio.isChecked()
        self.single_target_widget.setVisible(not is_multi)
        self.multi_target_widget.setVisible(is_multi)
        self.update_available_features_optimized()
    
    def update_available_features_optimized(self):
        """Optimized feature update method"""
        # 防止递归调用
        if getattr(self, '_updating_features', False):
            return
        
        self._updating_features = True
        try:
            # 获取选中的目标
            selected_targets = set()
            if self.single_target_radio.isChecked():
                if hasattr(self, 'target_combo'):
                    target = self.target_combo.currentText()
                    if target:
                        selected_targets.add(target)
            else:
                # 多目标模式：从复选框获取选中的目标
                if hasattr(self, 'target_checkboxes'):
                    for col, checkbox in self.target_checkboxes.items():
                        if checkbox.isChecked():
                            selected_targets.add(col)
            
            # 更新可用特征
            self.available_features = [col for col in self.numeric_columns if col not in selected_targets]
            
            # 批量更新特征列表（大数据集性能优化）
            if hasattr(self, 'feature_list'):
                self.feature_list.blockSignals(True)
                self.feature_list.clear()
                
                # 根据数据集大小调整批次处理
                if self._large_dataset_mode:
                    # 大数据集：使用较大批次和进度指示
                    batch_size = 50
                    total_features = len(self.available_features)
                    
                    if hasattr(self, 'feature_info_label'):
                        self.feature_info_label.setText(f"Loading {total_features} features...")
                    
                    for i in range(0, total_features, batch_size):
                        batch = self.available_features[i:i+batch_size]
                        for feature in batch:
                            item = QListWidgetItem(feature)
                            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                            item.setCheckState(Qt.Checked)
                            self.feature_list.addItem(item)
                        
                        # 大数据集：显示加载进度
                        if hasattr(self, 'feature_info_label'):
                            progress = min(i + batch_size, total_features)
                            self.feature_info_label.setText(f"Loading features: {progress}/{total_features}")
                        
                        # 每批处理后更新UI
                        QApplication.processEvents()
                else:
                    # 小数据集：正常处理
                    batch_size = 20
                    for i in range(0, len(self.available_features), batch_size):
                        batch = self.available_features[i:i+batch_size]
                        for feature in batch:
                            item = QListWidgetItem(feature)
                            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                            item.setCheckState(Qt.Checked)
                            self.feature_list.addItem(item)
                        
                        # 每批处理后更新UI
                        QApplication.processEvents()
                
                self.feature_list.blockSignals(False)
            
            # 更新信息标签
            if hasattr(self, 'feature_info_label'):
                self.feature_info_label.setText(f"Available features: {len(self.available_features)}")
                
        except Exception as e:
            print(f"Error updating features: {e}")
        finally:
            self._updating_features = False
    
    def select_all_features_optimized(self):
        """全选特征"""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setCheckState(Qt.Checked)
    
    def select_none_features_optimized(self):
        """取消全选特征"""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setCheckState(Qt.Unchecked)
    
    def get_selected_features_optimized(self):
        
        selected = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected
    
    def get_selected_targets_optimized(self):
        
        if self.single_target_radio.isChecked():
            return [self.target_combo.currentText()] if self.target_combo.currentText() else []
        else:
            # 从复选框中获取选中的目标
            selected_targets = []
            for col, checkbox in self.target_checkboxes.items():
                if checkbox.isChecked():
                    selected_targets.append(col)
            return selected_targets
    
    def get_constraints_optimized(self):
        """获取约束条件"""
        if not hasattr(self, 'constraints_text'):
            return []
        
        text = self.constraints_text.toPlainText().strip()
        if not text:
            return []
        
        constraints = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                constraints.append(line)
        return constraints
    
    def preview_generation_optimized(self):
        """预览生成设置"""
        targets = self.get_selected_targets_optimized()
        features = self.get_selected_features_optimized()
        method = self.method_combo.currentData()
        n_samples = self.n_samples_spin.value()
        constraints = self.get_constraints_optimized()
        
        if not targets:
            QMessageBox.warning(self, "Warning", "Please select at least one target variable")
            return
        
        if not features:
            QMessageBox.warning(self, "Warning", "Please select at least one feature")
            return
        
        msg = f"Preview of Generation Settings:\n\n"
        msg += f"Target Variables: {len(targets)} \n"
        msg += f"  {', '.join(targets)}\n\n"
        msg += f"Generation Method: {self.method_combo.currentText()}\n"
        msg += f"Number of Samples: {n_samples:,}\n"
        msg += f"Number of Features: {len(features)}\n"
        msg += f"Expansion Factor: {self.expansion_spin.value()}x\n"
        
        if constraints:
            msg += f"Constraints: {len(constraints)} \n"
            if hasattr(self, 'guarantee_mode') and self.guarantee_mode.isChecked():
                msg += "Generation Mode: Iterative Generation (Ensure Exact Count)\n"
        
        if method == "grid_search":
            estimated = self.grid_points_spin.value() ** len(features)
            msg += f"\n⚠️ Grid Search Estimated Samples: {estimated:,} "
            if estimated > 100000:
                msg += "\n🚨 Warning: Sample size is very large, may take a long time"
        
        QMessageBox.information(self, "Preview of Generation Settings", msg)
    
    def show_data_preview_optimized(self):
        """Show data preview"""
        if self.generated_candidates is not None:
            preview_dialog = DataPreviewDialog(self.generated_candidates, self.training_data, self)
            preview_dialog.exec_()
    
    def optimized_generate(self):
       
        try:
            # 获取参数
            targets = self.get_selected_targets_optimized()
            features = self.get_selected_features_optimized()
            method = self.method_combo.currentData()
            n_samples = self.n_samples_spin.value()
            expansion = self.expansion_spin.value()
            constraints = self.get_constraints_optimized()
            seed = self.seed_spin.value() if hasattr(self, 'seed_spin') else 42
            
            # 验证输入
            if not targets:
                QMessageBox.warning(self, "Warning", "Please select at least one target variable")
                return
            
            if not features:
                QMessageBox.warning(self, "Warning", "Please select at least one feature")
                return
            
            # 显示进度
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Generating candidate set...")
            self.generate_btn.setEnabled(False)
            
            # 强制更新UI
            QApplication.processEvents()
            
            # 创建优化器
            from modules.active_learning_optimizer import ActiveLearningOptimizer
            optimizer = ActiveLearningOptimizer(random_state=seed)
            
            # 准备参数
            generation_params = {}
            method_params = self.get_method_params()
            
            if method == "grid_search":
                generation_params['grid_points'] = method_params.get('grid_points', 10)
            else:
                generation_params['n_samples'] = n_samples
            
            # 添加方法特定参数
            generation_params.update(method_params)
            
            # 检查是否使用迭代生成 - 当有约束条件时自动启用
            use_iterative = (constraints and len(constraints) > 0 and method != "grid_search")
            
            # 如果用户明确启用了保证模式，也使用迭代生成
            if hasattr(self, 'guarantee_mode') and self.guarantee_mode.isChecked():
                use_iterative = True
            
            if use_iterative:
                # 使用迭代生成
                self.progress_bar.setFormat("Using iterative generation to ensure exact count...")
                QApplication.processEvents()
                
                # 创建并显示进度对话框
                progress_dialog = IterativeProgressDialog(self)
                progress_dialog.show()
                QApplication.processEvents()
                
                # 设置进度回调
                def progress_callback(iteration, valid_samples, total_attempts, success_rate, batch_size, target_samples):
                    progress_dialog.update_progress(iteration, valid_samples, total_attempts, success_rate, batch_size, target_samples)
                    QApplication.processEvents()
                
                optimizer.set_progress_callback(progress_callback)
                # 设置批次大小为5000
                optimizer.set_batch_size(5000)
                
                try:
                    # 调用迭代生成方法
                    self.generated_candidates = optimizer.generate_candidate_set_from_training(
                        training_df=self.training_data,
                        feature_columns=features,
                        method=method,
                        expansion_factor=expansion,
                        constraints=constraints,
                        random_seed=seed,
                        **generation_params
                    )
                finally:
                    optimizer.set_progress_callback(None)  # 清除回调
                    progress_dialog.close()
            else:
                # 标准生成
                self.generated_candidates = optimizer.generate_candidate_set_from_training(
                    training_df=self.training_data,
                    feature_columns=features,
                    method=method,
                    expansion_factor=expansion,
                    constraints=constraints if constraints else None,
                    random_seed=seed,
                    **generation_params
                )
            
            # 完成
            self.progress_bar.setVisible(False)
            self.generate_btn.setEnabled(True)
            self.use_btn.setEnabled(True)
            self.data_preview_btn.setEnabled(True)
            
            # 更新按钮文本
            self.generate_btn.setText("🔄 Re-generate")
            
            # 显示成功消息
            result_msg = f"✅ Successfully generated {len(self.generated_candidates)} candidate samples!\n\n"
            result_msg += f"Target Variables: {', '.join(targets)}\n"
            result_msg += f"Number of Features: {len(features)}\n"
            result_msg += f"Generation Method: {self.method_combo.currentText()}\n"
            result_msg += f"Data Dimension: {len(self.generated_candidates)} × {len(self.generated_candidates.columns)}"
            
            if constraints:
                result_msg += f"\nConstraints: {len(constraints)} applied"
            
            QMessageBox.information(self, "Generation Completed", result_msg)
            
        except Exception as e:
            # 恢复UI状态
            self.progress_bar.setVisible(False)
            self.generate_btn.setEnabled(True)
            
            # 显示详细错误信息
            error_msg = f"Candidate Set Generation Failed:\n\n"
            error_msg += f"Error Type: {type(e).__name__}\n"
            error_msg += f"Error Message: {str(e)}\n\n"
            error_msg += f"Suggested Solutions:\n"
            error_msg += f"• Check constraint syntax\n"
            error_msg += f"• Reduce sample size or feature count\n"
            error_msg += f"• Try other generation methods\n"
            error_msg += f"• Check if training data is complete"
            
            QMessageBox.critical(self, "Generation Error", error_msg)
    
    def get_generated_candidates(self):
        """Return generated data"""
        return self.generated_candidates
    
    def get_generation_settings(self):
        """Return generation settings, used for synchronization to main interface"""
        if not hasattr(self, 'generated_candidates') or self.generated_candidates is None:
            return None
        
        settings = {
            'targets': self.get_selected_targets_optimized(),
            'features': self.get_selected_features_optimized(),
            'target_mode': 'single' if self.single_target_radio.isChecked() else 'multi',
            'optimization_goals': {},  # 目标的优化方向
            'method': self.method_combo.currentData(),
            'method_params': self.get_method_params()
        }
        
        # 获取优化目标的方向
        if settings['target_mode'] == 'single':
            # 单目标模式，从下拉框获取优化方向
            target = settings['targets'][0] if settings['targets'] else None
            if target:
                goal_value = self.single_goal_combo.currentData()
                settings['optimization_goals'][target] = goal_value
        else:
            # 多目标模式，从下拉框中获取每个目标的方向
            for target in settings['targets']:
                if target in self.target_goal_combos:
                    goal_combo = self.target_goal_combos[target]
                    goal_value = goal_combo.currentData()
                    settings['optimization_goals'][target] = goal_value
                else:
                    settings['optimization_goals'][target] = 'maximize'  # 默认最大化
        
        return settings


class ConstraintConfigDialog(QDialog):
    """Dialog for configuring optimization constraints."""
    
    def __init__(self, training_data, parent=None):
        super().__init__(parent)
        self.training_data = training_data
        self.hard_constraints = {}
        self.soft_constraints = {}
        self.feasibility_weights = {}
        
        self.setWindowTitle("Constraint Configuration")
        self.setModal(True)
        self.setFixedSize(600, 500)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the constraint configuration UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Optimization Constraint Configuration")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Hard constraints tab
        hard_tab = self.create_hard_constraints_tab()
        tabs.addTab(hard_tab, "Hard Constraints")
        
        # Soft constraints tab
        soft_tab = self.create_soft_constraints_tab()
        tabs.addTab(soft_tab, "Soft Constraints")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        preview_btn = QPushButton("Preview Constraints")
        preview_btn.clicked.connect(self.preview_constraints)
        
        ok_btn = QPushButton("Apply Constraints")
        ok_btn.clicked.connect(self.accept)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(preview_btn)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
    
    def create_hard_constraints_tab(self):
        """Create hard constraints configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel("Hard constraints must be satisfied. Candidates violating these will be excluded.")
        desc.setStyleSheet("color: #666; font-style: italic; margin: 5px;")
        layout.addWidget(desc)
        
        # Feature selection
        feature_layout = QHBoxLayout()
        feature_layout.addWidget(QLabel("Feature:"))
        
        self.hard_feature_combo = QComboBox()
        if self.training_data is not None:
            numeric_columns = self.training_data.select_dtypes(include=[np.number]).columns.tolist()
            self.hard_feature_combo.addItems(numeric_columns)
        feature_layout.addWidget(self.hard_feature_combo)
        
        layout.addLayout(feature_layout)
        
        # Range specification
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Min Value:"))
        
        self.hard_min_spin = QDoubleSpinBox()
        self.hard_min_spin.setRange(-999999, 999999)
        self.hard_min_spin.setDecimals(4)
        range_layout.addWidget(self.hard_min_spin)
        
        range_layout.addWidget(QLabel("Max Value:"))
        
        self.hard_max_spin = QDoubleSpinBox()
        self.hard_max_spin.setRange(-999999, 999999)
        self.hard_max_spin.setDecimals(4)
        range_layout.addWidget(self.hard_max_spin)
        
        add_hard_btn = QPushButton("Add Hard Constraint")
        add_hard_btn.clicked.connect(self.add_hard_constraint)
        range_layout.addWidget(add_hard_btn)
        
        layout.addLayout(range_layout)
        
        # Hard constraints list
        layout.addWidget(QLabel("Current Hard Constraints:"))
        self.hard_constraints_list = QListWidget()
        layout.addWidget(self.hard_constraints_list)
        
        # Remove button
        remove_hard_btn = QPushButton("Remove Selected")
        remove_hard_btn.clicked.connect(self.remove_hard_constraint)
        layout.addWidget(remove_hard_btn)
        
        return widget
    
    def create_soft_constraints_tab(self):
        """Create soft constraints configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel("Soft constraints apply penalties but don't exclude candidates completely.")
        desc.setStyleSheet("color: #666; font-style: italic; margin: 5px;")
        layout.addWidget(desc)
        
        # Feature selection
        feature_layout = QHBoxLayout()
        feature_layout.addWidget(QLabel("Feature:"))
        
        self.soft_feature_combo = QComboBox()
        if self.training_data is not None:
            numeric_columns = self.training_data.select_dtypes(include=[np.number]).columns.tolist()
            self.soft_feature_combo.addItems(numeric_columns)
        feature_layout.addWidget(self.soft_feature_combo)
        
        layout.addLayout(feature_layout)
        
        # Range and penalty specification
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Min Value:"))
        
        self.soft_min_spin = QDoubleSpinBox()
        self.soft_min_spin.setRange(-999999, 999999)
        self.soft_min_spin.setDecimals(4)
        range_layout.addWidget(self.soft_min_spin)
        
        range_layout.addWidget(QLabel("Max Value:"))
        
        self.soft_max_spin = QDoubleSpinBox()
        self.soft_max_spin.setRange(-999999, 999999)
        self.soft_max_spin.setDecimals(4)
        range_layout.addWidget(self.soft_max_spin)
        
        range_layout.addWidget(QLabel("Penalty:"))
        
        self.penalty_spin = QDoubleSpinBox()
        self.penalty_spin.setRange(0.01, 1.0)
        self.penalty_spin.setValue(0.5)
        self.penalty_spin.setDecimals(2)
        self.penalty_spin.setSingleStep(0.1)
        range_layout.addWidget(self.penalty_spin)
        
        add_soft_btn = QPushButton("Add Soft Constraint")
        add_soft_btn.clicked.connect(self.add_soft_constraint)
        range_layout.addWidget(add_soft_btn)
        
        layout.addLayout(range_layout)
        
        # Soft constraints list
        layout.addWidget(QLabel("Current Soft Constraints:"))
        self.soft_constraints_list = QListWidget()
        layout.addWidget(self.soft_constraints_list)
        
        # Remove button
        remove_soft_btn = QPushButton("Remove Selected")
        remove_soft_btn.clicked.connect(self.remove_soft_constraint)
        layout.addWidget(remove_soft_btn)
        
        return widget
    
    def add_hard_constraint(self):
        """Add a hard constraint."""
        feature = self.hard_feature_combo.currentText()
        min_val = self.hard_min_spin.value()
        max_val = self.hard_max_spin.value()
        
        if min_val >= max_val:
            QMessageBox.warning(self, "Warning", "Minimum value must be less than maximum value.")
            return
        
        self.hard_constraints[feature] = (min_val, max_val)
        
        constraint_text = f"{feature}: [{min_val:.4f}, {max_val:.4f}]"
        self.hard_constraints_list.addItem(constraint_text)
    
    def add_soft_constraint(self):
        """Add a soft constraint."""
        feature = self.soft_feature_combo.currentText()
        min_val = self.soft_min_spin.value()
        max_val = self.soft_max_spin.value()
        penalty = self.penalty_spin.value()
        
        if min_val >= max_val:
            QMessageBox.warning(self, "Warning", "Minimum value must be less than maximum value.")
            return
        
        self.soft_constraints[feature] = (min_val, max_val)
        self.feasibility_weights[feature] = penalty
        
        constraint_text = f"{feature}: [{min_val:.4f}, {max_val:.4f}] (penalty: {penalty:.2f})"
        self.soft_constraints_list.addItem(constraint_text)
    
    def remove_hard_constraint(self):
        """Remove selected hard constraint."""
        current_item = self.hard_constraints_list.currentItem()
        if current_item:
            # Extract feature name from the constraint text
            constraint_text = current_item.text()
            feature = constraint_text.split(':')[0]
            
            if feature in self.hard_constraints:
                del self.hard_constraints[feature]
            
            self.hard_constraints_list.takeItem(self.hard_constraints_list.row(current_item))
    
    def remove_soft_constraint(self):
        """Remove selected soft constraint."""
        current_item = self.soft_constraints_list.currentItem()
        if current_item:
            # Extract feature name from the constraint text
            constraint_text = current_item.text()
            feature = constraint_text.split(':')[0]
            
            if feature in self.soft_constraints:
                del self.soft_constraints[feature]
            if feature in self.feasibility_weights:
                del self.feasibility_weights[feature]
            
            self.soft_constraints_list.takeItem(self.soft_constraints_list.row(current_item))
    
    def preview_constraints(self):
        """Preview the configured constraints."""
        msg = "Configured Constraints:\n\n"
        
        if self.hard_constraints:
            msg += "Hard Constraints:\n"
            for feature, (min_val, max_val) in self.hard_constraints.items():
                msg += f"  {feature}: [{min_val:.4f}, {max_val:.4f}]\n"
            msg += "\n"
        
        if self.soft_constraints:
            msg += "Soft Constraints:\n"
            for feature, (min_val, max_val) in self.soft_constraints.items():
                penalty = self.feasibility_weights.get(feature, 0.5)
                msg += f"  {feature}: [{min_val:.4f}, {max_val:.4f}] (penalty: {penalty:.2f})\n"
        
        if not self.hard_constraints and not self.soft_constraints:
            msg += "No constraints configured."
        
        QMessageBox.information(self, "Constraint Preview", msg)
    
    def get_constraints(self):
        """Get the configured constraints."""
        return self.hard_constraints, self.soft_constraints, self.feasibility_weights


def main():
    """Main function to run the application with DPI awareness and adaptive UI."""
    import os
    
    # 🔧 设置DPI感知 - 必须在创建QApplication之前设置
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Windows平台的额外DPI设置
    if os.name == 'nt':  # Windows
        try:
            from PyQt5.QtWinExtras import QtWin
            QtWin.setCurrentProcessExplicitAppUserModelID("ActiveLearningOptimizer.1.0")
        except ImportError:
            pass
        
        # 设置DPI感知级别
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except:
                pass
    
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Active Learning Optimizer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("AIResearch")
    app.setOrganizationDomain("airesearch.com")
    
    # 设置应用程序图标（如果存在的话）
    # app.setWindowIcon(QIcon('icon.png'))
    
    # 设置全局字体策略以支持不同DPI
    font = app.font()
    if hasattr(QApplication, 'primaryScreen'):
        screen = app.primaryScreen()
        if screen and hasattr(screen, 'logicalDotsPerInch'):
            dpi = screen.logicalDotsPerInch()
            if dpi > 120:  # 高DPI屏幕
                font_size = max(8, int(9 * (dpi / 96.0)))
                font.setPointSize(min(font_size, 12))  # 限制最大字体大小
                app.setFont(font)
    
    print(f"🚀 Start the Active Learning Optimizer")
    print(f"📱 Application DPI setting: {app.devicePixelRatio() if hasattr(app, 'devicePixelRatio') else 'Unknown'}")
    
    # Create and show window
    window = ActiveLearningWindow()
    window.show()
    
    # 确保窗口在屏幕中心显示
    window.activateWindow()
    window.raise_()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 