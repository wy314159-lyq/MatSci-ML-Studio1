"""
Plotting utilities for MatSci-ML Studio
Provides functions for creating various visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve


# Set style
plt.style.use('default')
sns.set_palette("husl")


def create_figure(figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, FigureCanvas]:
    """
    Create matplotlib figure and canvas for embedding in PyQt
    
    Args:
        figsize: Figure size (width, height)
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig = Figure(figsize=figsize, dpi=100)
    canvas = FigureCanvas(fig)
    return fig, canvas


def create_subplots_figure(nrows: int, ncols: int, figsize: Tuple[int, int] = (12, 8)) -> Tuple[Figure, np.ndarray]:
    """
    Create matplotlib figure with subplots
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size (width, height)
        
    Returns:
        Tuple of (Figure, axes array)
    """
    fig = Figure(figsize=figsize, dpi=100)
    
    # Handle single subplot case
    if nrows == 1 and ncols == 1:
        ax = fig.add_subplot(111)
        return fig, np.array([[ax]])
    elif nrows == 1:
        axes = np.array([fig.add_subplot(1, ncols, i+1) for i in range(ncols)])
        return fig, axes.reshape(1, -1)
    elif ncols == 1:
        axes = np.array([fig.add_subplot(nrows, 1, i+1) for i in range(nrows)])
        return fig, axes.reshape(-1, 1)
    else:
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(nrows, ncols, i*ncols + j + 1)
        return fig, axes


def plot_missing_values_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> Tuple[Figure, FigureCanvas]:
    """
    Create heatmap of missing values
    
    Args:
        df: Input DataFrame
        figsize: Figure size
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    # Create missing values matrix
    missing_matrix = df.isnull()
    
    if missing_matrix.any().any():
        sns.heatmap(missing_matrix, yticklabels=False, cbar=True, 
                   cmap='viridis', ax=ax)
        ax.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Rows', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No Missing Values Found', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=16)
        ax.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    return fig, canvas


def plot_missing_values_bar(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> Tuple[Figure, FigureCanvas]:
    """
    Create bar plot of missing values by column
    
    Args:
        df: Input DataFrame
        figsize: Figure size
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    if len(missing_counts) > 0:
        missing_counts.plot(kind='bar', ax=ax)
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Number of Missing Values', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No Missing Values Found', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=16)
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    return fig, canvas


def plot_correlation_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10),
                           use_readable_labels: bool = True) -> Tuple[Figure, FigureCanvas]:
    """
    Create correlation heatmap for numeric columns with readable labels
    
    Args:
        df: Input DataFrame
        figsize: Figure size
        use_readable_labels: Whether to use readable feature names
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        
        # Create readable labels if requested
        if use_readable_labels:
            try:
                from utils.feature_name_utils import create_readable_correlation_labels
                readable_labels = create_readable_correlation_labels(corr_matrix.columns.tolist())
                
                # Adjust figure size based on label length
                max_label_length = max(len(label) for label in readable_labels)
                if max_label_length > 20:
                    adjusted_figsize = (max(figsize[0], max_label_length * 0.2), 
                                      max(figsize[1], max_label_length * 0.2))
                    if adjusted_figsize != figsize:
                        fig, canvas = create_figure(adjusted_figsize)
                        ax = fig.add_subplot(111)
                
            except ImportError:
                print("Warning: Could not import feature name utilities for correlation labels")
                readable_labels = corr_matrix.columns.tolist()
        else:
            readable_labels = corr_matrix.columns.tolist()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Determine annotation and label settings based on matrix size
        n_features = len(corr_matrix)
        show_annotations = n_features <= 15  # Only show values for smaller matrices
        label_fontsize = max(6, min(10, 100 // n_features))
        
        sns.heatmap(corr_matrix, mask=mask, annot=show_annotations, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, ax=ax,
                   xticklabels=readable_labels, yticklabels=readable_labels,
                   fmt='.2f' if show_annotations else '')
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=label_fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fontsize)
        
        title = 'Feature Correlations Before Filtering'
        if use_readable_labels:
            title += ' (Readable Labels)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add feature count to title
        ax.set_title(f'{title}\n({n_features} features)', fontsize=14, fontweight='bold')
        
    else:
        ax.text(0.5, 0.5, 'Insufficient Numeric Columns for Correlation', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=16)
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    return fig, canvas


def plot_histogram(series: pd.Series, bins: int = 30, 
                  figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, FigureCanvas]:
    """
    Create histogram for a numeric series
    
    Args:
        series: Input series
        bins: Number of bins
        figsize: Figure size
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    series.hist(bins=bins, ax=ax, alpha=0.7, edgecolor='black')
    ax.set_title(f'Histogram of {series.name}', fontsize=14, fontweight='bold')
    ax.set_xlabel(series.name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, canvas


def plot_boxplot(series: pd.Series, figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, FigureCanvas]:
    """
    Create boxplot for a numeric series
    
    Args:
        series: Input series
        figsize: Figure size
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    series.plot.box(ax=ax)
    ax.set_title(f'Boxplot of {series.name}', fontsize=14, fontweight='bold')
    ax.set_ylabel(series.name, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, canvas


def plot_scatter(x: pd.Series, y: pd.Series, figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, FigureCanvas]:
    """
    Create scatter plot
    
    Args:
        x: X-axis series
        y: Y-axis series
        figsize: Figure size
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    ax.scatter(x, y, alpha=0.6)
    ax.set_title(f'Scatter Plot: {x.name} vs {y.name}', fontsize=14, fontweight='bold')
    ax.set_xlabel(x.name, fontsize=12)
    ax.set_ylabel(y.name, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, canvas


def plot_feature_importance(importance_df: pd.DataFrame, max_features: int = 20,
                          figsize: Tuple[int, int] = (10, 8), 
                          aggregate_encoded_features: bool = True) -> Tuple[Figure, FigureCanvas]:
    """
    Plot feature importance with support for aggregated encoded features
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        max_features: Maximum number of features to display
        figsize: Figure size
        aggregate_encoded_features: Whether to aggregate one-hot encoded features
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    # Process feature names for better readability
    plot_data = importance_df.copy()
    
    if aggregate_encoded_features:
        try:
            from utils.feature_name_utils import aggregate_feature_importance_by_original
            plot_data = aggregate_feature_importance_by_original(importance_df)
            
            # Use readable feature names if available
            if 'readable_feature' in plot_data.columns:
                feature_labels = plot_data['readable_feature'].tolist()
            else:
                feature_labels = plot_data['feature'].tolist()
                
        except ImportError:
            print("Warning: Could not import feature name utilities, using original names")
            feature_labels = plot_data['feature'].tolist()
    else:
        # Use readable names without aggregation
        try:
            from utils.feature_name_utils import get_readable_feature_names
            feature_labels = get_readable_feature_names(plot_data['feature'].tolist())
        except ImportError:
            feature_labels = plot_data['feature'].tolist()
    
    # Take top features
    plot_data = plot_data.head(max_features)
    feature_labels = feature_labels[:max_features]
    
    # Adjust figure size based on number of features and label length
    max_label_length = max(len(label) for label in feature_labels) if feature_labels else 20
    adjusted_figsize = (max(figsize[0], max_label_length * 0.15), 
                       max(figsize[1], len(plot_data) * 0.4))
    
    if adjusted_figsize != figsize:
        fig, canvas = create_figure(adjusted_figsize)
        ax = fig.add_subplot(111)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(plot_data)), plot_data['importance'])
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(feature_labels, fontsize=max(8, min(12, 120 // len(feature_labels))))
    ax.set_xlabel('Importance', fontsize=12)
    
    # Update title based on aggregation
    title = 'Feature Importance'
    if aggregate_encoded_features:
        title += ' (Aggregated by Original Features)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(plot_data['importance']) * 0.01, 
               bar.get_y() + bar.get_height()/2, 
               f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    fig.tight_layout()
    return fig, canvas


def plot_confusion_matrix(y_true, y_pred, class_names: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (8, 6)) -> Tuple[Figure, FigureCanvas]:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    
    fig.tight_layout()
    return fig, canvas


def plot_roc_curve(y_true, y_pred_proba, class_names: Optional[List[str]] = None,
                  figsize: Tuple[int, int] = (10, 8)) -> Tuple[Figure, FigureCanvas]:
    """
    Plot ROC curve for classification
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    # Get unique classes
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        auc_score = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    else:
        # Multi-class classification
        y_true_bin = label_binarize(y_true, classes=classes)
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc_score = np.trapz(tpr, fpr)
            class_name = class_names[i] if class_names else f'Class {classes[i]}'
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, canvas


def plot_prediction_vs_actual(y_true, y_pred, figsize: Tuple[int, int] = (10, 8)) -> Tuple[Figure, FigureCanvas]:
    """
    Plot predicted vs actual values for regression
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    # Calculate R²
    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'Predicted vs Actual (R² = {r2:.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, canvas


def plot_residuals(y_true, y_pred, figsize: Tuple[int, int] = (10, 8)) -> Tuple[Figure, FigureCanvas]:
    """
    Plot residuals for regression
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    residuals = y_true - y_pred
    
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, canvas


def plot_learning_curve(estimator, X, y, cv=5, figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, FigureCanvas]:
    """
    Plot learning curve
    
    Args:
        estimator: ML model
        X: Feature matrix
        y: Target vector
        cv: Cross-validation folds
        figsize: Figure size
        
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig, canvas = create_figure(figsize)
    ax = fig.add_subplot(111)
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    
    ax.plot(train_sizes, val_mean, 'o-', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, canvas 