"""
Enhanced Visualization for CGCNN Training
Provides real-time training progress and comprehensive result plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Dict, List, Optional
import time


class EnhancedTrainingVisualizer:
    """
    Enhanced visualizer for CGCNN training with real-time updates
    and comprehensive result plots.
    """

    def __init__(self, figure: Figure, canvas: FigureCanvas):
        """
        Initialize visualizer.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure object for plotting
        canvas : FigureCanvasQTAgg
            Canvas for rendering
        """
        self.figure = figure
        self.canvas = canvas

        # Training state
        self.train_losses = []
        self.val_metrics = []
        self.epochs = []
        self.learning_rates = []
        self.epoch_times = []

        # Start time for ETA calculation
        self.start_time = None
        self.last_epoch_time = None

    def reset(self):
        """Reset all tracking data."""
        self.train_losses = []
        self.val_metrics = []
        self.epochs = []
        self.learning_rates = []
        self.epoch_times = []
        self.start_time = None
        self.last_epoch_time = None

    def start_training(self):
        """Mark start of training."""
        self.start_time = time.time()
        self.last_epoch_time = time.time()

    def update_progress(self, epoch: int, train_loss: float, val_metric: float,
                       learning_rate: float, is_best: bool = False):
        """
        Update training progress with new epoch data.

        Parameters
        ----------
        epoch : int
            Current epoch number
        train_loss : float
            Training loss for this epoch
        val_metric : float
            Validation metric for this epoch
        learning_rate : float
            Current learning rate
        is_best : bool
            Whether this is the best model so far
        """
        # Record epoch time
        current_time = time.time()
        epoch_time = current_time - self.last_epoch_time
        self.last_epoch_time = current_time

        # Store data
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_metrics.append(val_metric)
        self.learning_rates.append(learning_rate)
        self.epoch_times.append(epoch_time)

        # Update plot
        self._plot_training_progress(is_best)

    def _plot_training_progress(self, mark_best: bool = False):
        """Plot real-time training progress with multiple subplots."""
        self.figure.clear()

        if len(self.epochs) == 0:
            return

        # Create 2x2 subplot layout
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Subplot 1: Training and Validation Metrics
        ax1 = self.figure.add_subplot(gs[0, :])

        # Plot with both lines and markers for better visibility
        ax1.plot(self.epochs, self.train_losses, 'b-o', label='Training Loss',
                linewidth=2, markersize=6, markerfacecolor='blue', markeredgecolor='white', markeredgewidth=1)
        ax1.plot(self.epochs, self.val_metrics, 'r-s', label='Validation Metric',
                linewidth=2, markersize=6, markerfacecolor='red', markeredgecolor='white', markeredgewidth=1)

        # Mark best epoch
        if mark_best and len(self.val_metrics) > 0:
            best_idx = np.argmin(self.val_metrics)
            ax1.plot(self.epochs[best_idx], self.val_metrics[best_idx],
                    'g*', markersize=20, label=f'Best (Epoch {self.epochs[best_idx]})', zorder=5)

        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Loss / Metric', fontsize=10)
        ax1.set_title('Training Progress', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Learning Rate Schedule
        ax2 = self.figure.add_subplot(gs[1, 0])
        ax2.plot(self.epochs, self.learning_rates, 'g-o', linewidth=2, markersize=5)
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Learning Rate', fontsize=10)
        ax2.set_title('Learning Rate Schedule', fontsize=11, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Epoch Time
        ax3 = self.figure.add_subplot(gs[1, 1])
        if len(self.epoch_times) > 0:
            ax3.bar(self.epochs, self.epoch_times, color='orange', alpha=0.7)
            ax3.axhline(np.mean(self.epoch_times), color='r', linestyle='--',
                       linewidth=2, label=f'Avg: {np.mean(self.epoch_times):.2f}s')
        ax3.set_xlabel('Epoch', fontsize=10)
        ax3.set_ylabel('Time (seconds)', fontsize=10)
        ax3.set_title('Training Speed', fontsize=11, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)

        if self.canvas:
            # Use draw_idle() for thread-safe updates
            self.canvas.draw_idle()
            # Force immediate processing of GUI events
            try:
                self.canvas.flush_events()
            except NotImplementedError:
                # flush_events() not available in all backends
                pass

    def plot_final_results(self, results: Dict,
                          train_predictions: Optional[np.ndarray] = None,
                          train_targets: Optional[np.ndarray] = None,
                          test_predictions: Optional[np.ndarray] = None,
                          test_targets: Optional[np.ndarray] = None,
                          task_type: str = 'regression'):
        """
        Plot comprehensive final results after training completion.

        Parameters
        ----------
        results : dict
            Training results dictionary with history and test results
        train_predictions : np.ndarray, optional
            Training set predictions for scatter plot (regression) or probabilities (classification)
        train_targets : np.ndarray, optional
            Training set ground truth values
        test_predictions : np.ndarray, optional
            Test set predictions for scatter plot (regression) or probabilities (classification)
        test_targets : np.ndarray, optional
            Test set ground truth values
        task_type : str, optional
            'regression' or 'classification' (default: 'regression')
        """
        self.figure.clear()

        history = results.get('history', [])
        if len(history) == 0:
            return

        # Extract data
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train'].get('loss', 0) for h in history]
        val_metrics = [h['val'] for h in history]

        # Determine if we have prediction data
        has_train_pred = (train_predictions is not None and train_targets is not None)
        has_test_pred = (test_predictions is not None and test_targets is not None)
        has_prediction_data = has_train_pred or has_test_pred

        if has_prediction_data:
            # 2x3 layout with prediction plot
            gs = self.figure.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
        else:
            # 2x2 layout without prediction plot
            gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # ==========================================
        # Plot 1: Training and Validation Curves
        # ==========================================
        ax1 = self.figure.add_subplot(gs[0, 0:2] if has_prediction_data else gs[0, :])

        # Plot with markers for better visibility
        ax1.plot(epochs, train_losses, 'b-o', label='Training Loss',
                linewidth=2, alpha=0.8, markersize=5, markerfacecolor='blue',
                markeredgecolor='white', markeredgewidth=0.5)
        ax1.plot(epochs, val_metrics, 'r-s', label='Validation Metric',
                linewidth=2, alpha=0.8, markersize=5, markerfacecolor='red',
                markeredgecolor='white', markeredgewidth=0.5)

        # Mark best epoch
        best_idx = np.argmin(val_metrics)
        best_epoch = epochs[best_idx]
        best_val = val_metrics[best_idx]
        ax1.plot(best_epoch, best_val, 'g*', markersize=20,
                label=f'Best: {best_val:.4f} @ Epoch {best_epoch}', zorder=5)

        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss / Metric', fontsize=11)
        ax1.set_title('Training Curves', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ==========================================
        # Plot 2: Loss Improvement Rate
        # ==========================================
        ax2 = self.figure.add_subplot(gs[0, 2] if has_prediction_data else gs[1, 0])

        if len(val_metrics) > 1:
            improvements = np.diff(val_metrics)
            ax2.bar(epochs[1:], improvements, color=['g' if x < 0 else 'r' for x in improvements],
                   alpha=0.7)
            ax2.axhline(0, color='k', linestyle='-', linewidth=1)
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Metric Change', fontsize=10)
            ax2.set_title('Epoch-to-Epoch Improvement', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # ==========================================
        # Plot 3: Training Statistics
        # ==========================================
        ax3 = self.figure.add_subplot(gs[1, 0] if has_prediction_data else gs[1, 1])

        # Calculate improvement percentage with zero check
        total_improvement = val_metrics[0] - best_val
        if abs(val_metrics[0]) > 1e-10:  # Avoid division by zero
            improvement_pct = (total_improvement / val_metrics[0] * 100)
            improvement_str = f"Improvement %: {improvement_pct:.1f}%"
        else:
            improvement_str = "Improvement %: N/A (initial metric near zero)"

        stats_text = [
            f"Total Epochs: {len(epochs)}",
            f"Best Val Metric: {best_val:.4f}",
            f"Best Epoch: {best_epoch}",
            f"Final Val Metric: {val_metrics[-1]:.4f}",
            f"Initial Val Metric: {val_metrics[0]:.4f}",
            f"Total Improvement: {total_improvement:.4f}",
            improvement_str
        ]

        if 'test_results' in results:
            test_res = results['test_results']
            stats_text.append(f"\n--- Test Set ---")
            if 'mae' in test_res:
                stats_text.append(f"Test MAE: {test_res['mae']:.4f}")
            if 'rmse' in test_res:
                stats_text.append(f"Test RMSE: {test_res['rmse']:.4f}")
            if 'auc' in test_res:
                stats_text.append(f"Test AUC: {test_res['auc']:.4f}")

        ax3.text(0.1, 0.95, '\n'.join(stats_text),
                transform=ax3.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax3.axis('off')
        ax3.set_title('Training Summary', fontsize=11, fontweight='bold')

        # ==========================================
        # Plot 4 & 5: Task-specific visualizations
        # ==========================================
        if has_prediction_data:
            if task_type == 'classification':
                # Classification visualizations
                self._plot_classification_results(
                    gs, train_predictions, train_targets,
                    test_predictions, test_targets,
                    has_train_pred, has_test_pred
                )
            else:
                # Regression visualizations
                self._plot_regression_results(
                    gs, train_predictions, train_targets,
                    test_predictions, test_targets,
                    has_train_pred, has_test_pred
                )

        if self.canvas:
            self.canvas.draw()

    def _plot_regression_results(self, gs, train_predictions, train_targets,
                                 test_predictions, test_targets,
                                 has_train_pred, has_test_pred):
        """Plot regression-specific visualizations: scatter plot and residuals."""
        ax4 = self.figure.add_subplot(gs[1, 1])

        # Collect all values to determine plot range
        all_targets = []
        all_predictions = []

        # Plot training set (blue)
        if has_train_pred:
            ax4.scatter(train_targets, train_predictions, alpha=0.5, s=40,
                       color='blue', edgecolors='k', linewidth=0.5, label='Training Set')
            all_targets.extend(train_targets)
            all_predictions.extend(train_predictions)

        # Plot test set (red)
        if has_test_pred:
            ax4.scatter(test_targets, test_predictions, alpha=0.5, s=40,
                       color='red', edgecolors='k', linewidth=0.5, label='Test Set')
            all_targets.extend(test_targets)
            all_predictions.extend(test_predictions)

        # Perfect prediction line
        if len(all_targets) > 0 and len(all_predictions) > 0:
            min_val = min(min(all_targets), min(all_predictions))
            max_val = max(max(all_targets), max(all_predictions))
            ax4.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, label='Perfect Prediction')

        # Calculate R² for both train and test sets
        r2_info = []
        if has_train_pred:
            train_ss_res = np.sum((train_targets - train_predictions) ** 2)
            train_ss_tot = np.sum((train_targets - np.mean(train_targets)) ** 2)
            train_r2 = 1 - (train_ss_res / train_ss_tot) if train_ss_tot > 0 else 0
            r2_info.append(f"Train R² = {train_r2:.3f}")

        if has_test_pred:
            test_ss_res = np.sum((test_targets - test_predictions) ** 2)
            test_ss_tot = np.sum((test_targets - np.mean(test_targets)) ** 2)
            test_r2 = 1 - (test_ss_res / test_ss_tot) if test_ss_tot > 0 else 0
            r2_info.append(f"Test R² = {test_r2:.3f}")

        r2_label = " (" + ", ".join(r2_info) + ")" if r2_info else ""

        ax4.set_xlabel('Actual Values', fontsize=10)
        ax4.set_ylabel('Predicted Values', fontsize=10)
        ax4.set_title(f'Prediction Accuracy{r2_label}', fontsize=11, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)

        # ==========================================
        # Plot 5: Residual Plot
        # ==========================================
        ax5 = self.figure.add_subplot(gs[1, 2])

        # Plot training set residuals (blue)
        if has_train_pred:
            train_residuals = train_targets - train_predictions
            ax5.scatter(train_predictions, train_residuals, alpha=0.5, s=40,
                       color='blue', edgecolors='k', linewidth=0.5, label='Training Set')

        # Plot test set residuals (red)
        if has_test_pred:
            test_residuals = test_targets - test_predictions
            ax5.scatter(test_predictions, test_residuals, alpha=0.5, s=40,
                       color='red', edgecolors='k', linewidth=0.5, label='Test Set')

        ax5.axhline(0, color='g', linestyle='--', linewidth=2)

        ax5.set_xlabel('Predicted Values', fontsize=10)
        ax5.set_ylabel('Residuals', fontsize=10)
        ax5.set_title('Residual Plot', fontsize=11, fontweight='bold')
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.3)

    def _plot_classification_results(self, gs, train_predictions, train_targets,
                                     test_predictions, test_targets,
                                     has_train_pred, has_test_pred):
        """Plot classification-specific visualizations: confusion matrix and metrics."""
        from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

        # Use test set if available, otherwise training set
        if has_test_pred:
            predictions = test_predictions
            targets = test_targets
            dataset_name = "Test Set"
        elif has_train_pred:
            predictions = train_predictions
            targets = train_targets
            dataset_name = "Training Set"
        else:
            return

        # For binary classification, predictions are probabilities [prob_class_0, prob_class_1]
        # For multi-class, predictions are probabilities for all classes

        # Determine number of classes
        if len(predictions.shape) == 1:
            # Binary classification with single probability
            n_classes = 2
            pred_classes = (predictions > 0.5).astype(int)
            pred_probs = predictions
        else:
            # Multi-class or binary with 2-column probabilities
            n_classes = predictions.shape[1]
            pred_classes = np.argmax(predictions, axis=1)
            pred_probs = predictions

        true_classes = targets.astype(int)

        # ==========================================
        # Plot 4: Confusion Matrix
        # ==========================================
        ax4 = self.figure.add_subplot(gs[1, 1])

        cm = confusion_matrix(true_classes, pred_classes)

        # Plot confusion matrix as heatmap
        im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
        ax4.figure.colorbar(im, ax=ax4)

        # Add labels
        tick_marks = np.arange(n_classes)
        ax4.set_xticks(tick_marks)
        ax4.set_yticks(tick_marks)
        ax4.set_xticklabels([f'Class {i}' for i in range(n_classes)])
        ax4.set_yticklabels([f'Class {i}' for i in range(n_classes)])

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax4.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)

        accuracy = accuracy_score(true_classes, pred_classes)
        ax4.set_xlabel('Predicted Class', fontsize=10)
        ax4.set_ylabel('True Class', fontsize=10)
        ax4.set_title(f'Confusion Matrix - {dataset_name}\n(Accuracy: {accuracy:.3f})',
                     fontsize=11, fontweight='bold')

        # ==========================================
        # Plot 5: ROC Curve (binary) or Per-Class Accuracy (multi-class)
        # ==========================================
        ax5 = self.figure.add_subplot(gs[1, 2])

        if n_classes == 2:
            # Binary classification: Plot ROC curve
            if len(pred_probs.shape) == 1:
                # Single probability column
                y_score = pred_probs
            else:
                # Two probability columns, use positive class
                y_score = pred_probs[:, 1]

            try:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(true_classes, y_score)
                auc_score = roc_auc_score(true_classes, y_score)

                # Plot ROC curve
                ax5.plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC curve (AUC = {auc_score:.3f})')
                ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                        label='Random classifier')

                ax5.set_xlim([0.0, 1.0])
                ax5.set_ylim([0.0, 1.05])
                ax5.set_xlabel('False Positive Rate', fontsize=10)
                ax5.set_ylabel('True Positive Rate', fontsize=10)
                ax5.set_title('ROC Curve', fontsize=11, fontweight='bold')
                ax5.legend(loc="lower right", fontsize=9)
                ax5.grid(True, alpha=0.3)
            except Exception as e:
                # If ROC calculation fails, show error message
                ax5.text(0.5, 0.5, f'ROC curve unavailable:\n{str(e)}',
                        ha='center', va='center', fontsize=10)
                ax5.set_title('ROC Curve', fontsize=11, fontweight='bold')
                ax5.axis('off')
        else:
            # Multi-class: Plot per-class accuracy
            per_class_acc = []
            class_labels = []

            for i in range(n_classes):
                mask = (true_classes == i)
                if mask.sum() > 0:
                    class_acc = (pred_classes[mask] == i).sum() / mask.sum()
                    per_class_acc.append(class_acc)
                    class_labels.append(f'Class {i}')

            # Bar plot
            bars = ax5.bar(range(len(per_class_acc)), per_class_acc,
                          color='steelblue', edgecolor='black', linewidth=1)

            # Color bars based on accuracy
            for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
                if acc >= 0.9:
                    bar.set_color('green')
                elif acc >= 0.7:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')

            ax5.set_xticks(range(len(per_class_acc)))
            ax5.set_xticklabels(class_labels, rotation=45 if n_classes > 5 else 0)
            ax5.set_ylim([0, 1.0])
            ax5.set_ylabel('Accuracy', fontsize=10)
            ax5.set_title('Per-Class Accuracy', fontsize=11, fontweight='bold')
            ax5.grid(True, axis='y', alpha=0.3)

            # Add value labels on bars
            for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.2f}',
                        ha='center', va='bottom', fontsize=9)

    def get_eta(self, current_epoch: int, total_epochs: int) -> str:
        """
        Calculate estimated time remaining.

        Parameters
        ----------
        current_epoch : int
            Current epoch number
        total_epochs : int
            Total number of epochs

        Returns
        -------
        str
            Formatted ETA string
        """
        if len(self.epoch_times) == 0 or current_epoch >= total_epochs:
            return "N/A"

        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = total_epochs - current_epoch
        eta_seconds = avg_epoch_time * remaining_epochs

        # Format ETA
        if eta_seconds < 60:
            return f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            minutes = int(eta_seconds / 60)
            seconds = int(eta_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(eta_seconds / 3600)
            minutes = int((eta_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def get_elapsed_time(self) -> str:
        """
        Get elapsed time since training start.

        Returns
        -------
        str
            Formatted elapsed time string
        """
        if self.start_time is None:
            return "N/A"

        elapsed = time.time() - self.start_time

        if elapsed < 60:
            return f"{int(elapsed)}s"
        elif elapsed < 3600:
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed / 3600)
            minutes = int((elapsed % 3600) / 60)
            return f"{hours}h {minutes}m"


class ProgressInfo:
    """Container for detailed progress information."""

    def __init__(self, epoch: int, total_epochs: int, train_loss: float,
                 val_metric: float, learning_rate: float, is_best: bool,
                 elapsed: str, eta: str):
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.train_loss = train_loss
        self.val_metric = val_metric
        self.learning_rate = learning_rate
        self.is_best = is_best
        self.elapsed = elapsed
        self.eta = eta

    def to_string(self) -> str:
        """Format progress info as string."""
        best_marker = " [BEST]" if self.is_best else ""
        return (f"Epoch {self.epoch}/{self.total_epochs}{best_marker} | "
                f"Train Loss: {self.train_loss:.4f} | "
                f"Val Metric: {self.val_metric:.4f} | "
                f"LR: {self.learning_rate:.6f} | "
                f"Elapsed: {self.elapsed} | ETA: {self.eta}")
