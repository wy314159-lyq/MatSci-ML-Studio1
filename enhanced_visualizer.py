"""
Enhanced Interactive Visualization System for Clustering Module
Provides interactive charts with export capabilities and customization options
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QPushButton, QFileDialog, QMessageBox,
                            QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider,
                            QDialog, QDialogButtonBox, QFormLayout, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont

# Plotly imports for interactive visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    from plotly.io import write_image, write_html
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly kaleido")


class ExportDialog(QDialog):
    """Dialog for configuring chart export options"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Chart")
        self.setModal(True)
        self.resize(400, 300)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QFormLayout(self)
        
        # File format selection
        self.format_combo = QComboBox()
        if PLOTLY_AVAILABLE:
            self.format_combo.addItems(['PNG', 'SVG', 'PDF', 'HTML', 'JPEG'])
        else:
            self.format_combo.addItems(['PNG', 'SVG', 'PDF', 'JPEG'])
        layout.addRow("Format:", self.format_combo)
        
        # Width and height
        self.width_spin = QSpinBox()
        self.width_spin.setRange(200, 4000)
        self.width_spin.setValue(1200)
        self.width_spin.setSuffix(" px")
        layout.addRow("Width:", self.width_spin)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(200, 4000)
        self.height_spin.setValue(800)
        self.height_spin.setSuffix(" px")
        layout.addRow("Height:", self.height_spin)
        
        # DPI
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        layout.addRow("DPI:", self.dpi_spin)
        
        # Quality for JPEG
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(50, 100)
        self.quality_slider.setValue(95)
        self.quality_label = QLabel("95%")
        self.quality_slider.valueChanged.connect(
            lambda v: self.quality_label.setText(f"{v}%"))
        
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(self.quality_slider)
        quality_layout.addWidget(self.quality_label)
        layout.addRow("JPEG Quality:", quality_layout)
        
        # Transparent background
        self.transparent_check = QCheckBox("Transparent Background")
        layout.addRow("", self.transparent_check)
        
        # Title and filename
        self.title_edit = QLineEdit("Clustering Analysis")
        layout.addRow("Chart Title:", self.title_edit)
        
        self.filename_edit = QLineEdit("clustering_chart")
        layout.addRow("Filename:", self.filename_edit)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow("", button_box)
    
    def get_export_config(self):
        """Get export configuration"""
        return {
            'format': self.format_combo.currentText().lower(),
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'dpi': self.dpi_spin.value(),
            'quality': self.quality_slider.value(),
            'transparent': self.transparent_check.isChecked(),
            'title': self.title_edit.text(),
            'filename': self.filename_edit.text()
        }


class InteractiveVisualizer:
    """Enhanced visualization system with interactive charts and export capabilities"""
    
    def __init__(self):
        self.current_figure = None
        self.current_plotly_fig = None
        self.export_dialog = None
    
    def create_interactive_scatter_plot(self, data: np.ndarray, labels: np.ndarray, 
                                      title: str = "Clustering Results",
                                      feature_names: List[str] = None,
                                      size_feature: np.ndarray = None,
                                      color_feature: np.ndarray = None) -> go.Figure:
        """Create interactive scatter plot using Plotly"""
        if not PLOTLY_AVAILABLE:
            return self._create_matplotlib_scatter(data, labels, title, feature_names, 
                                                 size_feature, color_feature)
        
        # Prepare data
        df_plot = pd.DataFrame()
        
        if data.shape[1] >= 2:
            df_plot['x'] = data[:, 0]
            df_plot['y'] = data[:, 1]
            if data.shape[1] >= 3:
                df_plot['z'] = data[:, 2]
        
        df_plot['cluster'] = labels
        df_plot['cluster_name'] = [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
        
        # Add size feature
        if size_feature is not None:
            df_plot['size'] = size_feature
            size_col = 'size'
        else:
            size_col = None
        
        # Add color feature
        if color_feature is not None:
            df_plot['color'] = color_feature
            color_col = 'color'
        else:
            color_col = 'cluster_name'
        
        # Create hover text
        hover_data = []
        for i in range(len(df_plot)):
            hover_text = f"Point {i}<br>"
            hover_text += f"Cluster: {df_plot.iloc[i]['cluster_name']}<br>"
            if feature_names and len(feature_names) >= 2:
                hover_text += f"{feature_names[0]}: {df_plot.iloc[i]['x']:.3f}<br>"
                hover_text += f"{feature_names[1]}: {df_plot.iloc[i]['y']:.3f}<br>"
            if size_feature is not None:
                hover_text += f"Size: {df_plot.iloc[i]['size']:.3f}<br>"
            if color_feature is not None:
                hover_text += f"Color: {df_plot.iloc[i]['color']:.3f}"
            hover_data.append(hover_text)
        
        df_plot['hover_text'] = hover_data
        
        # Create 3D or 2D scatter plot
        if data.shape[1] >= 3:
            # 3D scatter plot
            fig = px.scatter_3d(
                df_plot, x='x', y='y', z='z',
                color=color_col,
                size=size_col if size_col else None,
                hover_name='hover_text',
                title=title,
                labels={
                    'x': feature_names[0] if feature_names else 'Component 1',
                    'y': feature_names[1] if feature_names else 'Component 2',
                    'z': feature_names[2] if feature_names else 'Component 3'
                }
            )
        else:
            # 2D scatter plot
            fig = px.scatter(
                df_plot, x='x', y='y',
                color=color_col,
                size=size_col if size_col else None,
                hover_name='hover_text',
                title=title,
                labels={
                    'x': feature_names[0] if feature_names else 'Component 1',
                    'y': feature_names[1] if feature_names else 'Component 2'
                }
            )
        
        # Customize layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            hovermode='closest',
            template='plotly_white'
        )
        
        # Add custom controls
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [{"visible": [True] * len(fig.data)}],
                            "label": "Show All",
                            "method": "update"
                        },
                        {
                            "args": [{"visible": [i % 2 == 0 for i in range(len(fig.data))]}],
                            "label": "Show Even Clusters",
                            "method": "update"
                        }
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": 1.02,
                    "y": 1.02
                }
            ]
        )
        
        self.current_plotly_fig = fig
        return fig
    
    def _create_matplotlib_scatter(self, data: np.ndarray, labels: np.ndarray,
                                 title: str, feature_names: List[str] = None,
                                 size_feature: np.ndarray = None,
                                 color_feature: np.ndarray = None) -> Figure:
        """Fallback matplotlib scatter plot"""
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            label_name = f'Cluster {label}' if label != -1 else 'Noise'
            
            sizes = size_feature[mask] * 50 if size_feature is not None else 50
            
            scatter = ax.scatter(data[mask, 0], data[mask, 1], 
                               c=[color], s=sizes, alpha=0.7, 
                               label=label_name, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(feature_names[0] if feature_names else 'Component 1')
        ax.set_ylabel(feature_names[1] if feature_names else 'Component 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.current_figure = fig
        return fig
    
    def create_interactive_parallel_coordinates(self, data: pd.DataFrame, labels: np.ndarray,
                                              feature_subset: List[str] = None) -> go.Figure:
        """Create interactive parallel coordinates plot"""
        if not PLOTLY_AVAILABLE:
            return self._create_matplotlib_parallel_coordinates(data, labels, feature_subset)
        
        df = data.copy()
        df['Cluster'] = labels
        df['Cluster_Name'] = df['Cluster'].apply(lambda x: f'Cluster {x}' if x != -1 else 'Noise')
        
        # Select features
        if feature_subset:
            plot_cols = feature_subset
        else:
            plot_cols = [col for col in df.columns if col not in ['Cluster', 'Cluster_Name']]
            plot_cols = plot_cols[:8]  # Limit to 8 features for readability
        
        # Create dimensions for parallel coordinates
        dimensions = []
        for col in plot_cols:
            dimensions.append(
                dict(
                    range=[df[col].min(), df[col].max()],
                    label=col,
                    values=df[col]
                )
            )
        
        # Create parallel coordinates plot
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=df['Cluster'], 
                         colorscale='Viridis',
                         showscale=True,
                         colorbar=dict(title="Cluster")),
                dimensions=dimensions
            )
        )
        
        fig.update_layout(
            title="Interactive Parallel Coordinates Plot",
            template='plotly_white'
        )
        
        self.current_plotly_fig = fig
        return fig
    
    def _create_matplotlib_parallel_coordinates(self, data: pd.DataFrame, labels: np.ndarray,
                                              feature_subset: List[str] = None) -> Figure:
        """Fallback matplotlib parallel coordinates"""
        df = data.copy()
        df['Cluster'] = labels
        df['Cluster'] = df['Cluster'].apply(lambda x: f'Cluster {x}' if x != -1 else 'Noise')
        
        if feature_subset:
            plot_cols = feature_subset + ['Cluster']
            df = df[plot_cols]
        
        fig = Figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        
        # Normalize data
        feature_cols = [col for col in df.columns if col != 'Cluster']
        df_norm = df.copy()
        df_norm[feature_cols] = (df[feature_cols] - df[feature_cols].min()) / (df[feature_cols].max() - df[feature_cols].min())
        
        unique_clusters = df['Cluster'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster, color in zip(unique_clusters, colors):
            cluster_data = df_norm[df_norm['Cluster'] == cluster]
            for _, row in cluster_data.iterrows():
                ax.plot(range(len(feature_cols)), row[feature_cols], 
                       color=color, alpha=0.6, linewidth=1)
        
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Parallel Coordinates Plot by Cluster')
        ax.grid(True, alpha=0.3)
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=color, lw=2, label=cluster) 
                         for cluster, color in zip(unique_clusters, colors)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        self.current_figure = fig
        return fig
    
    def create_interactive_heatmap(self, data: pd.DataFrame, labels: np.ndarray,
                                 title: str = "Cluster Feature Heatmap") -> go.Figure:
        """Create interactive heatmap"""
        if not PLOTLY_AVAILABLE:
            return self._create_matplotlib_heatmap(data, labels, title)
        
        df = data.copy()
        df['Cluster'] = labels
        
        # Calculate cluster means
        cluster_means = df.groupby('Cluster').mean()
        
        # Remove noise cluster if present
        if -1 in cluster_means.index:
            cluster_means = cluster_means.drop(-1)
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cluster_means.values,
            x=cluster_means.columns.tolist(),
            y=[f'Cluster {i}' for i in cluster_means.index],
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                         '<b>%{x}</b><br>' +
                         'Value: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Clusters",
            template='plotly_white'
        )
        
        self.current_plotly_fig = fig
        return fig
    
    def _create_matplotlib_heatmap(self, data: pd.DataFrame, labels: np.ndarray,
                                 title: str) -> Figure:
        """Fallback matplotlib heatmap"""
        df = data.copy()
        df['Cluster'] = labels
        
        # Calculate cluster means
        cluster_means = df.groupby('Cluster').mean()
        
        # Remove noise cluster if present
        if -1 in cluster_means.index:
            cluster_means = cluster_means.drop(-1)
        
        fig = Figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Create heatmap
        im = ax.imshow(cluster_means.values, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(cluster_means.columns)))
        ax.set_xticklabels(cluster_means.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(cluster_means.index)))
        ax.set_yticklabels([f'Cluster {i}' for i in cluster_means.index])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Feature Value')
        
        # Add text annotations
        for i in range(len(cluster_means.index)):
            for j in range(len(cluster_means.columns)):
                text = ax.text(j, i, f'{cluster_means.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title)
        
        self.current_figure = fig
        return fig
    
    def export_chart(self, parent_widget=None):
        """Export current chart with custom options"""
        if not self.export_dialog:
            self.export_dialog = ExportDialog(parent_widget)
        
        if self.export_dialog.exec_() == QDialog.Accepted:
            config = self.export_dialog.get_export_config()
            
            # Get file path
            file_filter = f"{config['format'].upper()} Files (*.{config['format']})"
            file_path, _ = QFileDialog.getSaveFileName(
                parent_widget, "Export Chart", 
                f"{config['filename']}.{config['format']}", 
                file_filter
            )
            
            if file_path:
                try:
                    self._do_export(file_path, config)
                    QMessageBox.information(parent_widget, "Success", 
                                          f"Chart exported to {file_path}")
                except Exception as e:
                    QMessageBox.critical(parent_widget, "Error", 
                                       f"Export failed: {str(e)}")
    
    def _do_export(self, file_path: str, config: Dict[str, Any]):
        """Perform the actual export"""
        if PLOTLY_AVAILABLE and self.current_plotly_fig:
            # Export Plotly figure
            if config['format'] == 'html':
                self.current_plotly_fig.write_html(file_path)
            else:
                # Update title if provided
                if config['title']:
                    self.current_plotly_fig.update_layout(title=config['title'])
                
                # Export static image
                write_image(
                    self.current_plotly_fig, 
                    file_path,
                    width=config['width'],
                    height=config['height'],
                    scale=config['dpi']/96  # Convert DPI to scale
                )
        
        elif self.current_figure:
            # Export matplotlib figure
            if config['title']:
                self.current_figure.suptitle(config['title'])
            
            # Set figure size
            self.current_figure.set_size_inches(
                config['width']/config['dpi'], 
                config['height']/config['dpi']
            )
            
            # Export with custom settings
            export_kwargs = {
                'dpi': config['dpi'],
                'bbox_inches': 'tight',
                'transparent': config['transparent']
            }
            
            if config['format'] == 'jpeg':
                export_kwargs['quality'] = config['quality']
            
            self.current_figure.savefig(file_path, **export_kwargs)
        
        else:
            raise ValueError("No figure available for export")
    
    def create_comparison_dashboard(self, results_dict: Dict[str, Any]) -> go.Figure:
        """Create a comparison dashboard for multiple clustering results"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for comparison dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Scatter Plot', 'Metrics Comparison', 
                           'Cluster Sizes', 'Silhouette Scores'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add plots for each algorithm
        colors = px.colors.qualitative.Set1
        
        for i, (algorithm, result) in enumerate(results_dict.items()):
            color = colors[i % len(colors)]
            
            # Scatter plot (row 1, col 1)
            data = result['data']
            labels = result['labels']
            
            fig.add_trace(
                go.Scatter(
                    x=data[:, 0], y=data[:, 1],
                    mode='markers',
                    marker=dict(color=labels, colorscale='Viridis', size=5),
                    name=f'{algorithm}',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add metrics comparison and other plots...
        # (Implementation continues...)
        
        fig.update_layout(
            title="Clustering Algorithms Comparison Dashboard",
            showlegend=True,
            template='plotly_white',
            height=800
        )
        
        return fig