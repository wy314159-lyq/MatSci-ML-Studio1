"""
Specialized evaluation and visualization utilities for Quantile Regression
Extends ml_utils.py with quantile-specific functionality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def evaluate_quantile_regression(y_true: np.ndarray,
                                predictions: Dict[float, np.ndarray],
                                quantiles: List[float]) -> Dict[str, float]:
    """
    Comprehensive evaluation for quantile regression models

    Args:
        y_true: True target values
        predictions: Dictionary mapping quantiles to predictions
        quantiles: List of quantile values

    Returns:
        Dictionary of quantile-specific metrics
    """
    metrics = {}

    # 1. Quantile Loss (Pinball Loss)
    for q in quantiles:
        if q in predictions:
            pred = predictions[q]
            residuals = y_true - pred
            loss = np.maximum(q * residuals, (q - 1) * residuals)
            metrics[f'quantile_loss_{q}'] = np.mean(loss)

    # 2. Coverage metrics (for prediction intervals)
    if len(quantiles) >= 2:
        sorted_q = sorted(quantiles)
        for i in range(len(sorted_q) - 1):
            q_low, q_high = sorted_q[i], sorted_q[-1-i]
            if q_low in predictions and q_high in predictions:
                # Check if true values fall within the interval
                in_interval = (y_true >= predictions[q_low]) & (y_true <= predictions[q_high])
                coverage = np.mean(in_interval)
                expected_coverage = q_high - q_low
                metrics[f'coverage_{q_low}_{q_high}'] = coverage
                metrics[f'coverage_error_{q_low}_{q_high}'] = coverage - expected_coverage

    # 3. Interval width metrics
    if 0.1 in predictions and 0.9 in predictions:
        interval_width = predictions[0.9] - predictions[0.1]
        metrics['mean_interval_width_80'] = np.mean(interval_width)
        metrics['std_interval_width_80'] = np.std(interval_width)

    if 0.25 in predictions and 0.75 in predictions:
        interval_width = predictions[0.75] - predictions[0.25]
        metrics['mean_interval_width_50'] = np.mean(interval_width)
        metrics['std_interval_width_50'] = np.std(interval_width)

    # 4. Median-specific metrics (if 0.5 quantile available)
    if 0.5 in predictions:
        median_pred = predictions[0.5]
        metrics['median_mae'] = mean_absolute_error(y_true, median_pred)
        metrics['median_mse'] = mean_squared_error(y_true, median_pred)

        # Median bias
        metrics['median_bias'] = np.median(median_pred - y_true)

    # 5. Quantile crossing check
    crossing_violations = 0
    total_comparisons = 0

    sorted_quantiles = sorted([q for q in quantiles if q in predictions])
    for i in range(len(sorted_quantiles) - 1):
        q1, q2 = sorted_quantiles[i], sorted_quantiles[i + 1]
        violations = np.sum(predictions[q1] > predictions[q2])
        crossing_violations += violations
        total_comparisons += len(predictions[q1])

    if total_comparisons > 0:
        metrics['quantile_crossing_rate'] = crossing_violations / total_comparisons

    return metrics

def create_quantile_comparison_report(models: Dict[str, object],
                                    X_test: np.ndarray,
                                    y_test: np.ndarray,
                                    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]) -> pd.DataFrame:
    """
    Create comprehensive comparison report for multiple quantile models

    Args:
        models: Dictionary mapping model names to fitted models
        X_test: Test features
        y_test: Test targets
        quantiles: List of quantiles to evaluate

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    for model_name, model in models.items():
        predictions = {}

        # Get predictions for each quantile
        if hasattr(model, 'quantile'):
            # Single quantile model
            q = model.quantile
            predictions[q] = model.predict(X_test)
            eval_quantiles = [q]
        else:
            # Multiple quantile models (assume it's a dict of models)
            for q in quantiles:
                if q in model:
                    predictions[q] = model[q].predict(X_test)
            eval_quantiles = list(predictions.keys())

        # Evaluate
        metrics = evaluate_quantile_regression(y_test, predictions, eval_quantiles)
        metrics['model'] = model_name
        metrics['quantiles_evaluated'] = str(eval_quantiles)

        results.append(metrics)

    return pd.DataFrame(results)

def plot_quantile_predictions(y_true: np.ndarray,
                             predictions: Dict[float, np.ndarray],
                             quantiles: List[float],
                             title: str = "Quantile Regression Predictions",
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create comprehensive visualization for quantile regression predictions

    Args:
        y_true: True target values
        predictions: Dictionary mapping quantiles to predictions
        quantiles: List of quantile values
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Sort data by true values for better visualization
    sort_idx = np.argsort(y_true)
    y_sorted = y_true[sort_idx]

    # 1. Prediction intervals plot
    ax1 = axes[0, 0]

    # Plot prediction intervals
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)))
    for i, q in enumerate(sorted(quantiles)):
        if q in predictions:
            pred_sorted = predictions[q][sort_idx]
            ax1.plot(pred_sorted, alpha=0.7, color=colors[i], label=f'Q{q}')

    ax1.plot(y_sorted, 'k-', alpha=0.8, linewidth=2, label='True values')
    ax1.set_title('Prediction Intervals')
    ax1.set_xlabel('Sorted samples')
    ax1.set_ylabel('Target value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals plot (for median if available)
    ax2 = axes[0, 1]
    if 0.5 in predictions:
        residuals = y_true - predictions[0.5]
        ax2.scatter(predictions[0.5], residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax2.set_title('Residuals (Median Quantile)')
        ax2.set_xlabel('Predicted values')
        ax2.set_ylabel('Residuals')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Median quantile\nnot available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Residuals Plot')

    # 3. Quantile loss comparison
    ax3 = axes[1, 0]
    quantile_losses = []
    quantile_labels = []

    for q in sorted(quantiles):
        if q in predictions:
            pred = predictions[q]
            residuals = y_true - pred
            loss = np.maximum(q * residuals, (q - 1) * residuals)
            quantile_losses.append(np.mean(loss))
            quantile_labels.append(f'Q{q}')

    if quantile_losses:
        bars = ax3.bar(quantile_labels, quantile_losses, alpha=0.7)
        ax3.set_title('Quantile Loss by Quantile')
        ax3.set_ylabel('Mean Quantile Loss')
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, loss in zip(bars, quantile_losses):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(quantile_losses),
                    f'{loss:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. Coverage analysis
    ax4 = axes[1, 1]
    if len(quantiles) >= 2:
        coverage_data = []
        interval_labels = []

        sorted_q = sorted(quantiles)
        for i in range(len(sorted_q) - 1):
            q_low, q_high = sorted_q[i], sorted_q[-1-i]
            if q_low in predictions and q_high in predictions:
                in_interval = (y_true >= predictions[q_low]) & (y_true <= predictions[q_high])
                actual_coverage = np.mean(in_interval)
                expected_coverage = q_high - q_low

                coverage_data.append([expected_coverage, actual_coverage])
                interval_labels.append(f'{q_low}-{q_high}')

        if coverage_data:
            coverage_array = np.array(coverage_data)
            x_pos = np.arange(len(interval_labels))

            width = 0.35
            ax4.bar(x_pos - width/2, coverage_array[:, 0], width,
                   label='Expected Coverage', alpha=0.7)
            ax4.bar(x_pos + width/2, coverage_array[:, 1], width,
                   label='Actual Coverage', alpha=0.7)

            ax4.set_title('Coverage Analysis')
            ax4.set_ylabel('Coverage Rate')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(interval_labels)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_quantile_training_report(X: np.ndarray,
                                   y: np.ndarray,
                                   model_class,
                                   quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
                                   test_size: float = 0.3,
                                   random_state: int = 42) -> Dict:
    """
    Create a comprehensive training and evaluation report for quantile regression

    Args:
        X: Feature matrix
        y: Target vector
        model_class: Quantile regression model class
        quantiles: List of quantiles to train
        test_size: Test set proportion
        random_state: Random state for reproducibility

    Returns:
        Dictionary containing models, metrics, and visualizations
    """
    from sklearn.model_selection import train_test_split
    from utils.ml_utils import create_model_with_params

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train models for each quantile
    models = {}
    predictions = {}

    print(f"Training quantile regression models for {len(quantiles)} quantiles...")

    for q in quantiles:
        print(f"  Training quantile {q}...")
        model = create_model_with_params(model_class, quantile=q)
        model.fit(X_train, y_train)
        models[q] = model
        predictions[q] = model.predict(X_test)

    # Evaluate
    metrics = evaluate_quantile_regression(y_test, predictions, quantiles)

    # Create visualizations
    fig = plot_quantile_predictions(y_test, predictions, quantiles,
                                   title="Quantile Regression Analysis")

    # Summary statistics
    summary = {
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'n_features': X.shape[1],
        'target_range': (y.min(), y.max()),
        'quantiles_trained': quantiles
    }

    return {
        'models': models,
        'predictions': predictions,
        'metrics': metrics,
        'summary': summary,
        'visualization': fig,
        'test_data': (X_test, y_test)
    }

def quantile_feature_importance_analysis(models: Dict[float, object],
                                       feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze feature importance across different quantiles

    Args:
        models: Dictionary mapping quantiles to fitted models
        feature_names: Optional list of feature names

    Returns:
        DataFrame with feature importance across quantiles
    """
    if not feature_names:
        feature_names = [f'Feature_{i}' for i in range(len(models[list(models.keys())[0]].coef_))]

    importance_data = []

    for quantile, model in models.items():
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
                importance_data.append({
                    'quantile': quantile,
                    'feature': feature,
                    'coefficient': coef,
                    'abs_coefficient': abs(coef),
                    'feature_index': i
                })

    importance_df = pd.DataFrame(importance_data)

    # Add ranking within each quantile
    importance_df['rank'] = importance_df.groupby('quantile')['abs_coefficient'].rank(
        method='dense', ascending=False
    ).astype(int)

    return importance_df

# Additional utility functions for quantile regression
def calculate_prediction_intervals(models: Dict[float, object],
                                 X: np.ndarray,
                                 confidence_levels: List[float] = [0.5, 0.8, 0.9]) -> Dict:
    """
    Calculate prediction intervals from quantile models

    Args:
        models: Dictionary of quantile models
        X: Features for prediction
        confidence_levels: List of confidence levels

    Returns:
        Dictionary with prediction intervals
    """
    intervals = {}

    for confidence in confidence_levels:
        alpha = 1 - confidence
        q_low = alpha / 2
        q_high = 1 - alpha / 2

        if q_low in models and q_high in models:
            lower = models[q_low].predict(X)
            upper = models[q_high].predict(X)

            intervals[f'{confidence}_interval'] = {
                'lower': lower,
                'upper': upper,
                'width': upper - lower,
                'confidence': confidence
            }

    return intervals

def export_quantile_results(results: Dict,
                           filename: str = "quantile_regression_results"):
    """
    Export quantile regression results to files

    Args:
        results: Results dictionary from create_quantile_training_report
        filename: Base filename for exports
    """
    # Export metrics
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(f"{filename}_metrics.csv", index=False)

    # Export predictions
    predictions_df = pd.DataFrame(results['predictions'])
    predictions_df.to_csv(f"{filename}_predictions.csv", index=False)

    # Save visualization
    results['visualization'].savefig(f"{filename}_visualization.png",
                                   dpi=300, bbox_inches='tight')

    print(f"Results exported:")
    print(f"  - {filename}_metrics.csv")
    print(f"  - {filename}_predictions.csv")
    print(f"  - {filename}_visualization.png")

print("Quantile regression specialized utilities loaded successfully!")
print("Available functions:")
print("  - evaluate_quantile_regression()")
print("  - create_quantile_comparison_report()")
print("  - plot_quantile_predictions()")
print("  - create_quantile_training_report()")
print("  - quantile_feature_importance_analysis()")
print("  - calculate_prediction_intervals()")
print("  - export_quantile_results()")