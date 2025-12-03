"""
Independent SHAP visualization window
Used to display SHAP plots in a separate window, solving Qt-matplotlib compatibility issues
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
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
plt.switch_backend('Qt5Agg')

class SHAPVisualizationWindow(QDialog):
    """Independent SHAP visualization window"""
    
    def __init__(self, shap_values, explainer, explain_data, feature_names, plot_type="Summary Plot", parent=None):
        super().__init__(parent)
        self.shap_values = shap_values
        self.explainer = explainer
        self.explain_data = explain_data
        self.feature_names = feature_names
        self.plot_type = plot_type
        
        self.setWindowTitle(f"SHAP {plot_type} - Independent Visualization Window")
        self.setGeometry(100, 100, 1200, 800)
        self.setModal(False)  # Non-modal window, can open multiple
        
        self.init_ui()
        self.create_plot()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel(f"SHAP {self.plot_type}")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh Plot")
        self.refresh_btn.clicked.connect(self.create_plot)
        button_layout.addWidget(self.refresh_btn)
        
        self.export_btn = QPushButton("Export Image")
        self.export_btn.clicked.connect(self.export_plot)
        button_layout.addWidget(self.export_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def create_plot(self):
        """Create SHAP plot using image save/load method"""
        try:
            print(f"Creating independent window {self.plot_type}...")
            
            # Clear previous figure
            self.figure.clear()
            
            # Prepare data
            n_samples = min(self.shap_values.shape[0], len(self.explain_data))
            shap_vals = self.shap_values[:n_samples]
            
            if isinstance(self.explain_data, pd.DataFrame):
                explain_data_df = self.explain_data.iloc[:n_samples].copy()
            else:
                explain_data_df = pd.DataFrame(self.explain_data[:n_samples], columns=self.feature_names)
            
            print(f"Data prepared - SHAP: {shap_vals.shape}, Data: {explain_data_df.shape}")
            
            # Create SHAP plot in temporary figure and save as image
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            
            temp_fig = plt.figure(figsize=(12, 8))
            
            if self.plot_type == "Summary Plot":
                shap.summary_plot(shap_vals, explain_data_df, 
                                feature_names=self.feature_names, 
                                show=False)
                
            elif self.plot_type == "Feature Importance":
                shap.summary_plot(shap_vals, explain_data_df,
                                feature_names=self.feature_names,
                                plot_type="bar", show=False)
                
            elif self.plot_type == "Beeswarm Plot":
                # Create manual beeswarm plot
                ax = temp_fig.add_subplot(111)
                self._create_manual_beeswarm(ax, shap_vals, explain_data_df)
            
            # Save to memory buffer
            buf = io.BytesIO()
            temp_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            buf.seek(0)
            plt.close(temp_fig)
            
            # Load image and display in our figure
            img = Image.open(buf)
            img_array = np.array(img)
            
            # Display image in our figure
            ax = self.figure.add_subplot(111)
            ax.imshow(img_array, aspect='auto', interpolation='bilinear')
            ax.axis('off')
            ax.set_position([0, 0, 1, 1])  # Full figure
            
            # Adjust layout
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
            # Refresh canvas
            self.canvas.draw()
            self.canvas.flush_events()
            QApplication.processEvents()
            
            print(f"Independent window {self.plot_type} created successfully")
            
        except Exception as e:
            print(f"Failed to create independent window plot: {e}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            
            # Show error message
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot creation failed:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.canvas.draw()
    
    def _create_manual_beeswarm(self, ax, shap_vals, explain_data_df):
        """Create manual beeswarm-like plot"""
        # Calculate feature importance
        feature_importance = np.abs(shap_vals).mean(0)
        sorted_idx = np.argsort(feature_importance)
        
        # Plot each feature
        for i, idx in enumerate(sorted_idx):
            y_pos = i
            values = shap_vals[:, idx]
            
            # Add some jitter for beeswarm effect
            y_jitter = np.random.normal(0, 0.1, len(values))
            
            # Color by feature value
            colors = explain_data_df.iloc[:, idx].values
            scatter = ax.scatter(values, y_pos + y_jitter, c=colors, 
                               alpha=0.6, s=30, cmap='coolwarm')
        
        # Set labels
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([self.feature_names[i] for i in sorted_idx])
        ax.set_xlabel('SHAP Value')
        ax.set_title('SHAP Beeswarm Plot')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    def export_plot(self):
        """Export image"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, f"Export {self.plot_type}", 
                f"shap_{self.plot_type.lower().replace(' ', '_')}.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
            )
            
            if file_path:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Export Successful", f"Plot saved to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting image:\n{str(e)}")


class SHAPDependenceWindow(QDialog):
    """Independent SHAP dependence plot window"""
    
    def __init__(self, shap_values, explain_data, feature_names, main_feature=None, interaction_feature=None, parent=None):
        super().__init__(parent)
        self.shap_values = shap_values
        self.explain_data = explain_data
        self.feature_names = feature_names
        self.main_feature = main_feature or feature_names[0]
        self.interaction_feature = interaction_feature
        
        self.setWindowTitle("SHAP Dependence Plot - Independent Visualization Window")
        self.setGeometry(150, 150, 1200, 800)
        self.setModal(False)
        
        self.init_ui()
        self.create_plot()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("SHAP Dependence Plot")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("Main Feature:"))
        self.main_feature_combo = QComboBox()
        self.main_feature_combo.addItems(self.feature_names)
        self.main_feature_combo.setCurrentText(self.main_feature)
        self.main_feature_combo.currentTextChanged.connect(self.on_feature_changed)
        control_layout.addWidget(self.main_feature_combo)
        
        control_layout.addWidget(QLabel("Interaction Feature:"))
        self.interaction_combo = QComboBox()
        self.interaction_combo.addItem("Auto")
        self.interaction_combo.addItems(self.feature_names)
        if self.interaction_feature:
            self.interaction_combo.setCurrentText(self.interaction_feature)
        self.interaction_combo.currentTextChanged.connect(self.on_feature_changed)
        control_layout.addWidget(self.interaction_combo)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh Plot")
        self.refresh_btn.clicked.connect(self.create_plot)
        button_layout.addWidget(self.refresh_btn)
        
        self.export_btn = QPushButton("Export Image")
        self.export_btn.clicked.connect(self.export_plot)
        button_layout.addWidget(self.export_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def on_feature_changed(self):
        """Update plot when feature selection changes"""
        self.main_feature = self.main_feature_combo.currentText()
        self.interaction_feature = self.interaction_combo.currentText()
        self.create_plot()
    
    def create_plot(self):
        """Create dependence plot using image save/load method"""
        try:
            print(f"Creating independent dependence plot window...")
            
            # Clear previous figure
            self.figure.clear()
            
            # Prepare data
            n_samples = min(self.shap_values.shape[0], len(self.explain_data))
            shap_vals = self.shap_values[:n_samples]
            
            if isinstance(self.explain_data, pd.DataFrame):
                explain_data_df = self.explain_data.iloc[:n_samples].copy()
            else:
                explain_data_df = pd.DataFrame(self.explain_data[:n_samples], columns=self.feature_names)
            
            # Get feature indices
            main_idx = self.feature_names.index(self.main_feature)
            interaction_idx = None
            if self.interaction_feature and self.interaction_feature != "Auto":
                interaction_idx = self.feature_names.index(self.interaction_feature)
            
            print(f"Creating dependence plot - Main feature: {self.main_feature}, Interaction feature: {self.interaction_feature}")
            
            # Create SHAP dependence plot in temporary figure and save as image
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            
            temp_fig = plt.figure(figsize=(12, 8))
            
            if interaction_idx is not None:
                shap.dependence_plot(main_idx, shap_vals, explain_data_df,
                                   feature_names=self.feature_names,
                                   interaction_index=interaction_idx,
                                   show=False)
            else:
                shap.dependence_plot(main_idx, shap_vals, explain_data_df,
                                   feature_names=self.feature_names,
                                   show=False)
            
            # Save to memory buffer
            buf = io.BytesIO()
            temp_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            buf.seek(0)
            plt.close(temp_fig)
            
            # Load image and display in our figure
            img = Image.open(buf)
            img_array = np.array(img)
            
            # Display image in our figure
            ax = self.figure.add_subplot(111)
            ax.imshow(img_array, aspect='auto', interpolation='bilinear')
            ax.axis('off')
            ax.set_position([0, 0, 1, 1])  # Full figure
            
            # Adjust layout
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
            # Refresh canvas
            self.canvas.draw()
            self.canvas.flush_events()
            QApplication.processEvents()
            
            print(f"Independent dependence plot window created successfully")
            
        except Exception as e:
            print(f"Failed to create independent dependence plot: {e}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            
            # Show error message
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Dependence plot creation failed:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.canvas.draw()
    
    def export_plot(self):
        """Export image"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Dependence Plot", 
                f"shap_dependence_{self.main_feature}.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
            )
            
            if file_path:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Export Successful", f"Plot saved to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting image:\n{str(e)}")


class SHAPLocalWindow(QDialog):
    """Independent SHAP local explanation window"""
    
    def __init__(self, shap_values, explainer, explain_data, feature_names, sample_idx=0, plot_type="Waterfall Plot", parent=None):
        super().__init__(parent)
        self.shap_values = shap_values
        self.explainer = explainer
        self.explain_data = explain_data
        self.feature_names = feature_names
        self.sample_idx = sample_idx
        self.plot_type = plot_type
        
        self.setWindowTitle(f"SHAP {plot_type} - Independent Visualization Window")
        self.setGeometry(200, 200, 1200, 800)
        self.setModal(False)
        
        self.init_ui()
        self.create_plot()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel(f"SHAP {self.plot_type}")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("Sample Index:"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(0, len(self.shap_values) - 1)
        self.sample_spin.setValue(self.sample_idx)
        self.sample_spin.valueChanged.connect(self.on_sample_changed)
        control_layout.addWidget(self.sample_spin)
        
        control_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Waterfall Plot", "Force Plot"])
        self.plot_type_combo.setCurrentText(self.plot_type)
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)
        control_layout.addWidget(self.plot_type_combo)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh Plot")
        self.refresh_btn.clicked.connect(self.create_plot)
        button_layout.addWidget(self.refresh_btn)
        
        self.export_btn = QPushButton("Export Image")
        self.export_btn.clicked.connect(self.export_plot)
        button_layout.addWidget(self.export_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def on_sample_changed(self):
        """Update plot when sample index changes"""
        self.sample_idx = self.sample_spin.value()
        self.create_plot()
    
    def on_plot_type_changed(self):
        """Update plot when plot type changes"""
        self.plot_type = self.plot_type_combo.currentText()
        self.create_plot()
    
    def create_plot(self):
        """Create local explanation plot"""
        try:
            print(f"Creating independent local explanation window - {self.plot_type}, sample: {self.sample_idx}")
            
            # Clear previous figure
            self.figure.clear()
            
            # Prepare data
            if isinstance(self.explain_data, pd.DataFrame):
                explain_data_df = self.explain_data.copy()
            else:
                explain_data_df = pd.DataFrame(self.explain_data, columns=self.feature_names)
            
            sample_shap = self.shap_values[self.sample_idx]
            sample_data = explain_data_df.iloc[self.sample_idx]
            
            # Get base value
            if hasattr(self.explainer, 'expected_value'):
                if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                    base_value = self.explainer.expected_value[0]
                else:
                    base_value = self.explainer.expected_value
            else:
                base_value = 0.0
            
            print(f"Sample data prepared - SHAP: {sample_shap.shape}, base value: {base_value}")
            
            # Create different types of plots
            if self.plot_type == "Waterfall Plot":
                # Use manual waterfall plot to avoid SHAP library bugs
                ax = self.figure.add_subplot(111)
                self._create_manual_waterfall(ax, sample_shap, sample_data, base_value)
                    
            elif self.plot_type == "Force Plot":
                # Try SHAP force plot first, fallback to manual if needed
                try:
                    shap.force_plot(base_value, sample_shap, sample_data, 
                                  feature_names=self.feature_names, 
                                  matplotlib=True, show=False)
                except:
                    # Manual force plot
                    ax = self.figure.add_subplot(111)
                    self._create_manual_force_plot(ax, sample_shap, sample_data, base_value)
            
            # Adjust layout
            self.figure.tight_layout()
            
            # Refresh canvas
            self.canvas.draw()
            self.canvas.flush_events()
            QApplication.processEvents()
            
            print(f"Independent local explanation window created successfully")
            
        except Exception as e:
            print(f"Failed to create independent local explanation plot: {e}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            
            # Show error message
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot creation failed:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.canvas.draw()
    
    def _create_manual_waterfall(self, ax, shap_vals, sample_data, base_value):
        """Create manual waterfall plot with improved visualization"""
        # Calculate cumulative values
        cumulative = [base_value]
        for val in shap_vals:
            cumulative.append(cumulative[-1] + val)
        
        # Sort features by absolute SHAP value for better visualization
        feature_importance = [(i, abs(val), val, self.feature_names[i], sample_data.iloc[i]) 
                             for i, val in enumerate(shap_vals)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Take top features for cleaner visualization
        top_features = feature_importance[:min(10, len(feature_importance))]
        
        # Create waterfall plot
        y_pos = range(len(top_features) + 2)  # +2 for base and prediction
        values = [base_value] + [item[2] for item in top_features] + [sum(shap_vals)]
        colors = ['gray'] + ['red' if item[2] < 0 else 'blue' for item in top_features] + ['green']
        
        # Create horizontal bar chart
        bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
        
        # Add cumulative line
        cumulative_values = [base_value]
        running_sum = base_value
        for item in top_features:
            running_sum += item[2]
            cumulative_values.append(running_sum)
        cumulative_values.append(base_value + sum(shap_vals))
        
        # Set labels
        labels = ['Base Value'] + [f"{item[3]} = {item[4]:.3f}" for item in top_features] + ['Prediction']
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        
        # Add value annotations
        for i, (bar, val) in enumerate(zip(bars, values)):
            if abs(val) > 0.001:  # Only show significant values
                ax.text(val/2 if val > 0 else val/2, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'Waterfall Plot - Sample {self.sample_idx}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adjust layout
        ax.margins(y=0.02)
    
    def _create_manual_force_plot(self, ax, shap_vals, sample_data, base_value):
        """Create manual force plot with improved visualization"""
        # Sort features by absolute SHAP value
        indices = np.argsort(np.abs(shap_vals))[::-1]
        
        # Take top features for cleaner visualization
        top_n = min(10, len(indices))
        top_indices = indices[:top_n]
        
        # Create horizontal bar chart
        y_pos = range(len(top_indices))
        values = shap_vals[top_indices]
        colors = ['red' if val < 0 else 'blue' for val in values]
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
        
        # Set labels with feature names and values
        labels = [f"{self.feature_names[i]} = {sample_data.iloc[i]:.3f}" for i in top_indices]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        
        # Add value annotations
        for i, (bar, val) in enumerate(zip(bars, values)):
            if abs(val) > 0.001:  # Only show significant values
                ax.text(val/2 if val > 0 else val/2, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', ha='center', va='center', fontweight='bold', color='white')
        
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'Force Plot - Sample {self.sample_idx}\nBase Value: {base_value:.3f}, Prediction: {base_value + sum(shap_vals):.3f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adjust layout
        ax.margins(y=0.02)
    
    def export_plot(self):
        """Export image"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, f"Export {self.plot_type}", 
                f"shap_{self.plot_type.lower().replace(' ', '_')}_sample_{self.sample_idx}.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
            )
            
            if file_path:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Export Successful", f"Plot saved to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting image:\n{str(e)}") 