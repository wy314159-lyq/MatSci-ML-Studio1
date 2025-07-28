"""
Enhanced Visualization Module for Active Learning System
Provides additional charts and analysis tools beyond the basic interface.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class EnhancedVisualizations:
    """Enhanced visualization capabilities for active learning analysis."""
    
    def __init__(self):
        """Initialize the visualization module."""
        self.optimization_history = []
        self.feature_evolution = []
        
    def add_iteration_data(self, iteration_data):
        """Add data from each optimization iteration for tracking."""
        self.optimization_history.append(iteration_data)
        
    def create_correlation_heatmap(self, data, feature_columns, figsize=(10, 8)):
        """Create feature correlation heatmap."""
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Calculate correlation matrix
        corr_matrix = data[feature_columns].corr()
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(feature_columns)))
        ax.set_yticks(range(len(feature_columns)))
        ax.set_xticklabels(feature_columns, rotation=45, ha='right')
        ax.set_yticklabels(feature_columns)
        
        # Add correlation values
        for i in range(len(feature_columns)):
            for j in range(len(feature_columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Feature Correlation Matrix')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        fig.tight_layout()
        return fig
    
    def create_pca_visualization(self, data, feature_columns, target_column=None, figsize=(12, 5)):
        """Create PCA analysis visualization."""
        fig = Figure(figsize=figsize)
        
        # Prepare data
        X = data[feature_columns].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot 1: Explained variance
        ax1 = fig.add_subplot(121)
        explained_var = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, label='Individual')
        ax1.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-', label='Cumulative')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Explained Variance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: First two components
        ax2 = fig.add_subplot(122)
        if target_column and target_column in data.columns:
            scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=data[target_column], cmap='viridis', alpha=0.7)
            cbar = fig.colorbar(scatter, ax=ax2)
            cbar.set_label(target_column)
        else:
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
        
        ax2.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        ax2.set_title('PCA: First Two Components')
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def create_uncertainty_analysis(self, results_data, figsize=(15, 5)):
        """Create uncertainty analysis visualization."""
        fig = Figure(figsize=figsize)
        
        # Plot 1: Uncertainty distribution
        ax1 = fig.add_subplot(131)
        ax1.hist(results_data['uncertainty_std'], bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Prediction Uncertainty (Std)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Uncertainty Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty vs Predicted Value
        ax2 = fig.add_subplot(132)
        scatter = ax2.scatter(results_data['predicted_mean'], 
                            results_data['uncertainty_std'],
                            c=results_data['acquisition_score'], 
                            cmap='plasma', alpha=0.7)
        ax2.set_xlabel('Predicted Mean Value')
        ax2.set_ylabel('Prediction Uncertainty')
        ax2.set_title('Uncertainty vs Prediction')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Acquisition Score Distribution
        ax3 = fig.add_subplot(133)
        ax3.hist(results_data['acquisition_score'], bins=30, alpha=0.7, 
                color='orange', edgecolor='black')
        ax3.set_xlabel('Acquisition Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Acquisition Score Distribution')
        ax3.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def create_design_space_slice(self, results_data, feature_columns, 
                                 x_feature, y_feature, figsize=(10, 8)):
        """Create 2D design space slice visualization."""
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Get feature indices
        if x_feature not in feature_columns or y_feature not in feature_columns:
            ax.text(0.5, 0.5, 'Selected features not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create scatter plot
        x_data = results_data[x_feature] if x_feature in results_data.columns else np.random.randn(len(results_data))
        y_data = results_data[y_feature] if y_feature in results_data.columns else np.random.randn(len(results_data))
        
        scatter = ax.scatter(x_data, y_data,
                           c=results_data['predicted_mean'], 
                           s=results_data['acquisition_score'] * 100,
                           cmap='viridis', alpha=0.7, edgecolors='black')
        
        # Highlight top recommendations
        top_indices = results_data.head(5).index
        ax.scatter(x_data[top_indices], y_data[top_indices],
                  s=200, facecolors='none', edgecolors='red', linewidths=2,
                  label='Top 5 Recommendations')
        
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'Design Space: {x_feature} vs {y_feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Value')
        
        fig.tight_layout()
        return fig
    
    def create_optimization_history(self, figsize=(15, 5)):
        """Create optimization history visualization."""
        if not self.optimization_history:
            fig = Figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No optimization history available\nRun multiple iterations to see progress',
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig = Figure(figsize=figsize)
        
        # Extract history data
        iterations = list(range(1, len(self.optimization_history) + 1))
        best_values = [data.get('best_value', 0) for data in self.optimization_history]
        mean_uncertainty = [data.get('mean_uncertainty', 0) for data in self.optimization_history]
        acquisition_max = [data.get('max_acquisition', 0) for data in self.optimization_history]
        
        # Plot 1: Best value evolution
        ax1 = fig.add_subplot(131)
        ax1.plot(iterations, best_values, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Predicted Value')
        ax1.set_title('Optimization Progress')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty evolution
        ax2 = fig.add_subplot(132)
        ax2.plot(iterations, mean_uncertainty, 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Mean Uncertainty')
        ax2.set_title('Model Uncertainty Reduction')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Acquisition function maximum
        ax3 = fig.add_subplot(133)
        ax3.plot(iterations, acquisition_max, 'ro-', linewidth=2, markersize=6)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Max Acquisition Score')
        ax3.set_title('Exploration Intensity')
        ax3.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def create_feature_radar_chart(self, importance_data, top_n=8, figsize=(8, 8)):
        """Create radar chart for feature importance."""
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # Get top N features
        top_features = importance_data.nlargest(top_n)
        features = list(top_features.index)
        values = list(top_features.values)
        
        # Number of variables
        N = len(features)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values
        values += values[:1]  # Complete the circle
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label='Feature Importance')
        ax.fill(angles, values, alpha=0.25)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        
        # Set title
        ax.set_title('Feature Importance Radar Chart', size=16, pad=20)
        
        # Add grid
        ax.grid(True)
        
        return fig
    
    def create_acquisition_landscape(self, virtual_data, feature_columns, 
                                   acquisition_scores, x_feature, y_feature, 
                                   figsize=(10, 8)):
        """Create acquisition function landscape."""
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        if x_feature not in feature_columns or y_feature not in feature_columns:
            ax.text(0.5, 0.5, 'Selected features not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create contour plot
        x_data = virtual_data[x_feature]
        y_data = virtual_data[y_feature]
        
        # Create a grid for contour plotting
        xi = np.linspace(x_data.min(), x_data.max(), 50)
        yi = np.linspace(y_data.min(), y_data.max(), 50)
        
        # Create scatter plot with acquisition scores
        scatter = ax.scatter(x_data, y_data, c=acquisition_scores, 
                           cmap='hot', s=50, alpha=0.7)
        
        # Highlight top acquisition points
        top_idx = np.argsort(acquisition_scores)[-10:]
        ax.scatter(x_data.iloc[top_idx], y_data.iloc[top_idx], 
                  s=100, facecolors='none', edgecolors='blue', 
                  linewidths=2, label='Top 10 Candidates')
        
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'Acquisition Function Landscape: {x_feature} vs {y_feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Acquisition Score')
        
        fig.tight_layout()
        return fig
    
    def create_model_performance_metrics(self, optimizer, training_data, 
                                       target_column, feature_columns, 
                                       figsize=(12, 8)):
        """Create comprehensive model performance visualization."""
        fig = Figure(figsize=figsize)
        
        try:
            # Get cross-validation scores
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import r2_score, mean_squared_error
            
            X = training_data[feature_columns]
            y = training_data[target_column]
            
            # Get the trained model from optimizer
            model = optimizer.model
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            # Predictions
            y_pred = model.predict(X)
            
            # Plot 1: CV scores
            ax1 = fig.add_subplot(221)
            ax1.bar(range(1, 6), cv_scores, alpha=0.7)
            ax1.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                       label=f'Mean: {cv_scores.mean():.3f}')
            ax1.set_xlabel('Fold')
            ax1.set_ylabel('R² Score')
            ax1.set_title('Cross-Validation Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Predicted vs Actual
            ax2 = fig.add_subplot(222)
            ax2.scatter(y, y_pred, alpha=0.7)
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax2.set_xlabel('Actual Values')
            ax2.set_ylabel('Predicted Values')
            ax2.set_title(f'Predicted vs Actual (R² = {r2_score(y, y_pred):.3f})')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Residuals
            ax3 = fig.add_subplot(223)
            residuals = y - y_pred
            ax3.scatter(y_pred, residuals, alpha=0.7)
            ax3.axhline(y=0, color='red', linestyle='--')
            ax3.set_xlabel('Predicted Values')
            ax3.set_ylabel('Residuals')
            ax3.set_title('Residual Plot')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Learning curve (if data is available)
            ax4 = fig.add_subplot(224)
            if hasattr(optimizer, 'learning_curve_data'):
                train_sizes, train_scores, val_scores = optimizer.learning_curve_data
                ax4.plot(train_sizes, train_scores, 'o-', label='Training')
                ax4.plot(train_sizes, val_scores, 'o-', label='Validation')
                ax4.set_xlabel('Training Set Size')
                ax4.set_ylabel('R² Score')
                ax4.set_title('Learning Curve')
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'Learning curve data\nnot available', 
                        ha='center', va='center', transform=ax4.transAxes)
            ax4.grid(True, alpha=0.3)
            
        except Exception as e:
            # Fallback if model performance analysis fails
            ax1 = fig.add_subplot(111)
            ax1.text(0.5, 0.5, f'Model performance analysis failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        fig.tight_layout()
        return fig


# Example usage and integration helper
def create_enhanced_visualization_tabs(results_data, optimizer, training_data, 
                                     virtual_data, feature_columns, target_column):
    """Create all enhanced visualizations for integration into the UI."""
    
    visualizer = EnhancedVisualizations()
    
    visualizations = {
        'correlation_heatmap': visualizer.create_correlation_heatmap(
            training_data, feature_columns),
        'pca_analysis': visualizer.create_pca_visualization(
            training_data, feature_columns, target_column),
        'uncertainty_analysis': visualizer.create_uncertainty_analysis(results_data),
        'feature_radar': visualizer.create_feature_radar_chart(
            optimizer.get_feature_importance()),
        'model_performance': visualizer.create_model_performance_metrics(
            optimizer, training_data, target_column, feature_columns)
    }
    
    return visualizations 