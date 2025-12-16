"""
ALIGNN Trainer
Training loop and utilities for ALIGNN models
Based on reference_alignn/alignn/train.py
"""

import os
import json
import time
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score
from tqdm import tqdm

from .model import ALIGNN
from .config import TrainingConfig, ALIGNNConfig


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a validation metric and stops training when
    no improvement is seen for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.0,
        mode: str = 'min',
        baseline: float = None,
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Parameters
        ----------
        patience : int
            Number of epochs with no improvement after which training will be stopped.
            Default: 50
        min_delta : float
            Minimum change in monitored metric to qualify as an improvement.
            For 'min' mode, decrease must be larger than min_delta.
            For 'max' mode, increase must be larger than min_delta.
            Default: 0.0
        mode : str
            One of 'min' or 'max'.
            'min': Training stops when metric stops decreasing (e.g., MAE, loss)
            'max': Training stops when metric stops increasing (e.g., AUC, accuracy)
            Default: 'min'
        baseline : float, optional
            Baseline value for the monitored metric. Training will stop if
            the model doesn't show improvement over the baseline.
            Default: None
        verbose : bool
            Whether to print early stopping messages.
            Default: True
        """
        if patience < 1:
            raise ValueError("Patience must be at least 1")
        if mode not in ('min', 'max'):
            raise ValueError("Mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.baseline = baseline
        self.verbose = verbose

        # Internal state
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

        # Set comparison function based on mode
        if mode == 'min':
            self.is_improvement = lambda current, best: current < best - self.min_delta
            self.best_score = float('inf') if baseline is None else baseline
        else:
            self.is_improvement = lambda current, best: current > best + self.min_delta
            self.best_score = float('-inf') if baseline is None else baseline

    def __call__(self, epoch: int, metric: float) -> bool:
        """
        Check if training should stop.

        Parameters
        ----------
        epoch : int
            Current epoch number
        metric : float
            Current validation metric value

        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        if self.is_improvement(metric, self.best_score):
            # Improvement detected
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Improvement detected at epoch {epoch}: "
                      f"{metric:.6f} (best so far)")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs. "
                      f"Best: {self.best_score:.6f} at epoch {self.best_epoch}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] Stopping training. "
                          f"No improvement for {self.patience} epochs.")

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.early_stop = False
        if self.mode == 'min':
            self.best_score = float('inf') if self.baseline is None else self.baseline
        else:
            self.best_score = float('-inf') if self.baseline is None else self.baseline
        self.best_epoch = 0

    def get_status(self) -> dict:
        """Get current early stopping status."""
        return {
            'counter': self.counter,
            'patience': self.patience,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'should_stop': self.early_stop
        }


class ALIGNNTrainer:
    """
    ALIGNN Model Trainer

    Handles training loop, validation, and model checkpointing
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0
    ):
        """
        Initialize trainer

        Args:
            config: Training configuration
            model: Pre-initialized model (optional)
            device: Device to use (optional)
            early_stopping_patience: Early stopping patience (0 to disable)
            early_stopping_min_delta: Minimum improvement for early stopping
        """
        self.config = config
        self.device = device or self._get_device()

        # Set dtype for tensors (aligned with official ALIGNN)
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        self.dtype = dtype_map.get(getattr(config, 'dtype', 'float32'), torch.float32)

        # Set random seed for reproducibility
        if config.random_seed is not None:
            self._set_random_seed(config.random_seed)

        # Auto-enable classification mode when classification_threshold is set
        # Aligned with official ALIGNN (reference_alignn/alignn/train.py:156-159)
        if config.classification_threshold is not None:
            if hasattr(config.model, 'classification'):
                config.model.classification = True
                config.model.num_classes = 2  # Binary classification
                print(f"[Trainer] Classification mode enabled (threshold={config.classification_threshold})")

        # Initialize model
        if model is None:
            if isinstance(config.model, ALIGNNConfig):
                self.model = ALIGNN(config.model)
            else:
                raise ValueError("Model config must be ALIGNNConfig")
        else:
            self.model = model

        self.model.to(self.device)

        # Loss function
        self.criterion = self._get_criterion()

        # Optimizer
        self.optimizer = self._get_optimizer()

        # Learning rate scheduler
        self.scheduler = None  # Will be set in train()

        # Early stopping
        self.early_stopping = None
        if early_stopping_patience > 0:
            # Use MAE for early stopping metric (lower is better)
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode='min',
                verbose=True
            )
            print(f"[Trainer] Early stopping enabled: patience={early_stopping_patience}, "
                  f"min_delta={early_stopping_min_delta}")

        # Training state
        self.best_loss = float('inf')
        self.best_mae = float('inf')
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rate': []
        }

        # Callbacks
        self.epoch_callback = None

    def _set_random_seed(self, seed: int):
        """Set random seed for reproducibility"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For reproducibility on CUDA
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"[Trainer] Random seed set to {seed}")

    def _get_device(self) -> torch.device:
        """Get device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _get_criterion(self) -> nn.Module:
        """Get loss criterion"""
        if self.config.classification_threshold is not None:
            return nn.NLLLoss()
        else:
            criterion_map = {
                "mse": nn.MSELoss(),
                "l1": nn.L1Loss(),
                "mae": nn.L1Loss()
            }
            return criterion_map.get(self.config.criterion, nn.L1Loss())

    def _get_optimizer(self):
        """
        Get optimizer with parameter groups.

        Aligned with official ALIGNN implementation:
        - Bias and LayerNorm/BatchNorm parameters should not have weight decay
        - Other parameters have weight decay applied
        """
        # Group parameters: no decay for bias and normalization layers
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias',
                    'BatchNorm', 'bn.weight', 'bn.bias']

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]

        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _get_scheduler(self, train_loader):
        """
        Get learning rate scheduler.

        Aligned with official ALIGNN implementation:
        - Supports warmup_steps for gradual learning rate increase
        - OneCycleLR for cyclic learning rate schedule
        - Step decay scheduler
        """
        steps_per_epoch = len(train_loader)
        total_steps = self.config.epochs * steps_per_epoch
        warmup_steps = getattr(self.config, 'warmup_steps', 0)

        if self.config.scheduler == "onecycle":
            # OneCycleLR with pct_start based on warmup_steps
            pct_start = warmup_steps / total_steps if warmup_steps > 0 else 0.3
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        else:
            # None scheduler with optional warmup
            if warmup_steps > 0:
                def warmup_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    return 1.0
                return torch.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_lambda)
            else:
                return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0)

    def train_epoch(self, train_loader) -> float:
        """
        Train one epoch

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Handle different batch formats from collate functions
            # Format 1: ((g, lg, lat), target, cif_ids) - from collate_line_graph
            # Format 2: (g, lg, lat, target) - legacy format
            if len(batch) == 3 and isinstance(batch[0], tuple):
                # New format: ((g, lg, lat), target, cif_ids)
                (g, lg, lat), target, _ = batch
                g = g.to(self.device)
                lg = lg.to(self.device)
                lat = lat.to(self.device)
                target = target.to(self.device)
                inputs = (g, lg, lat)
            elif len(batch) == 4:
                # Legacy format: (g, lg, lat, target)
                g, lg, lat, target = batch
                g = g.to(self.device)
                lg = lg.to(self.device)
                lat = lat.to(self.device)
                target = target.to(self.device)
                inputs = (g, lg, lat)
            else:
                raise ValueError(f"Unexpected batch format: got {len(batch)} elements")

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(inputs)

            # Compute loss
            loss = self.criterion(output, target)

            # Normalize loss by number of graphs if configured
            # Aligned with official ALIGNN (reference_alignn/alignn/train.py)
            if getattr(self.config, 'normalize_graph_level_loss', False):
                num_graphs = g.batch_size if hasattr(g, 'batch_size') else 1
                loss = loss / num_graphs

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update scheduler if using OneCycleLR
            if self.config.scheduler == "onecycle":
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate model

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        predictions = []
        targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Handle different batch formats
                if len(batch) == 3 and isinstance(batch[0], tuple):
                    # New format: ((g, lg, lat), target, cif_ids)
                    (g, lg, lat), target, _ = batch
                    g = g.to(self.device)
                    lg = lg.to(self.device)
                    lat = lat.to(self.device)
                    target = target.to(self.device)
                    inputs = (g, lg, lat)
                elif len(batch) == 4:
                    # Legacy format: (g, lg, lat, target)
                    g, lg, lat, target = batch
                    g = g.to(self.device)
                    lg = lg.to(self.device)
                    lat = lat.to(self.device)
                    target = target.to(self.device)
                    inputs = (g, lg, lat)
                else:
                    continue

                # Forward pass
                output = self.model(inputs)

                # Compute loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                num_batches += 1

                # Store predictions and targets
                predictions.extend(output.cpu().numpy().tolist())
                targets.extend(target.cpu().numpy().tolist())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute MAE
        predictions = np.array(predictions)
        targets = np.array(targets)
        mae = mean_absolute_error(targets, predictions) if len(targets) > 0 else 0.0

        return {
            'loss': avg_loss,
            'mae': mae
        }

    def train(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            progress_callback: Callback for progress updates

        Returns:
            Training results dictionary
        """
        print(f"\n{'='*50}")
        print(f"Starting ALIGNN Training")
        print(f"{'='*50}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Random seed: {self.config.random_seed}")
        if self.early_stopping:
            print(f"Early stopping: patience={self.early_stopping.patience}")
        print(f"{'='*50}\n")

        # Save config to JSON (aligned with official ALIGNN)
        self._save_config()

        # Compute MAD and baseline MAE (aligned with official ALIGNN)
        target_stats = self._compute_target_statistics(train_loader)
        if target_stats:
            self.history['mad'] = target_stats.get('mad')
            self.history['baseline_mae'] = target_stats.get('baseline_mae')
            self.history['target_mean'] = target_stats.get('mean')
            self.history['target_std'] = target_stats.get('std')
            print(f"[Target Statistics]")
            print(f"  Mean: {target_stats['mean']:.4f}")
            print(f"  Std: {target_stats['std']:.4f}")
            print(f"  MAD (Mean Absolute Deviation): {target_stats['mad']:.4f}")
            print(f"  Baseline MAE (predict mean): {target_stats['baseline_mae']:.4f}")
            print(f"{'='*50}\n")

        # Initialize scheduler
        self.scheduler = self._get_scheduler(train_loader)

        # Reset early stopping if exists
        if self.early_stopping:
            self.early_stopping.reset()

        stopped_early = False

        # Training loop
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            val_mae = val_metrics['mae']

            # Update learning rate (for non-OneCycle schedulers)
            if self.config.scheduler != "onecycle":
                self.scheduler.step()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rate'].append(current_lr)

            # Check if best model (based on MAE for consistency)
            is_best = val_mae < self.best_mae
            if is_best:
                self.best_mae = val_mae
                self.best_loss = val_loss
                self.best_epoch = epoch + 1
                self._save_model('best_model.pt')

            # Save current model
            self._save_model('current_model.pt')

            # Epoch time
            epoch_time = time.time() - epoch_start_time

            # Print progress
            print(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
                f"{' [BEST]' if is_best else ''}"
            )

            # Callback
            if self.epoch_callback:
                self.epoch_callback({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_metric': val_mae,
                    'is_best': is_best
                })

            # Progress callback
            if progress_callback:
                progress_callback({
                    'epoch': epoch + 1,
                    'total_epochs': self.config.epochs,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'learning_rate': current_lr,
                    'is_best': is_best,
                    'best_epoch': self.best_epoch,
                    'best_mae': self.best_mae
                })

            # Early stopping (new mechanism)
            if self.early_stopping is not None:
                if self.early_stopping(epoch + 1, val_mae):
                    print(f"\n[EarlyStopping] Training stopped at epoch {epoch + 1}")
                    print(f"[EarlyStopping] Best MAE: {self.best_mae:.4f} at epoch {self.best_epoch}")
                    stopped_early = True
                    break

            # Legacy early stopping (from config.n_early_stopping)
            elif self.config.n_early_stopping is not None:
                if epoch > self.config.n_early_stopping:
                    recent_losses = self.history['val_loss'][-self.config.n_early_stopping:]
                    if all(loss >= self.best_loss for loss in recent_losses):
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        stopped_early = True
                        break

        # Test evaluation
        test_results = None
        if test_loader is not None:
            print("\nEvaluating on test set...")
            test_results = self.evaluate(test_loader)
            print(f"Test MAE: {test_results['mae']:.4f}")

        # Save history
        self._save_history()

        # Save final model
        self._save_model('final_model.pt')

        return {
            'best_val_loss': self.best_loss,
            'best_val_mae': self.best_mae,
            'best_epoch': self.best_epoch,
            'stopped_early': stopped_early,
            'history': self.history,
            'test_results': test_results
        }

    def evaluate(self, test_loader) -> Dict[str, Any]:
        """
        Evaluate model on test set

        Args:
            test_loader: Test data loader

        Returns:
            Test results dictionary
        """
        self.model.eval()
        predictions = []
        targets = []
        ids = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # Handle different batch formats
                if len(batch) == 3 and isinstance(batch[0], tuple):
                    # New format: ((g, lg, lat), target, cif_ids)
                    (g, lg, lat), target, cif_ids = batch
                    g = g.to(self.device)
                    lg = lg.to(self.device)
                    lat = lat.to(self.device)

                    output = self.model((g, lg, lat))

                    predictions.extend(output.cpu().numpy().tolist())
                    targets.extend(target.cpu().numpy().tolist())
                    ids.extend(cif_ids)

                elif len(batch) == 4:
                    # Legacy format: (g, lg, lat, target)
                    g, lg, lat, target = batch
                    g = g.to(self.device)
                    lg = lg.to(self.device)
                    lat = lat.to(self.device)

                    output = self.model((g, lg, lat))

                    predictions.extend(output.cpu().numpy().tolist())
                    targets.extend(target.cpu().numpy().tolist())

                    # Try to get ID
                    try:
                        id_val = test_loader.dataset.dataset.ids[
                            test_loader.dataset.indices[i]
                        ]
                        ids.append(id_val)
                    except:
                        ids.append(f"sample_{i}")

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Compute metrics
        mae = mean_absolute_error(targets, predictions)
        mse = np.mean((targets - predictions) ** 2)
        rmse = np.sqrt(mse)

        return {
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'ids': ids
        }

    def _save_model(self, filename: str):
        """Save model checkpoint"""
        if self.config.write_checkpoint:
            filepath = os.path.join(self.config.output_dir, filename)
            torch.save(self.model.state_dict(), filepath)

    def _save_config(self):
        """Save training configuration to JSON (aligned with official ALIGNN)."""
        if self.config.store_outputs:
            os.makedirs(self.config.output_dir, exist_ok=True)
            filepath = os.path.join(self.config.output_dir, "config.json")
            try:
                # Try to convert config to dict
                if hasattr(self.config, 'dict'):
                    config_dict = self.config.dict()
                elif hasattr(self.config, 'model_dump'):
                    config_dict = self.config.model_dump()
                else:
                    config_dict = vars(self.config)

                # Handle nested model config
                if hasattr(self.config, 'model'):
                    if hasattr(self.config.model, 'dict'):
                        config_dict['model'] = self.config.model.dict()
                    elif hasattr(self.config.model, 'model_dump'):
                        config_dict['model'] = self.config.model.model_dump()

                with open(filepath, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
                print(f"[Config] Saved to: {filepath}")
            except Exception as e:
                print(f"[Config] Warning: Could not save config: {e}")

    def _compute_target_statistics(self, train_loader) -> Dict[str, float]:
        """
        Compute target statistics from training data.

        Aligned with official ALIGNN implementation:
        - Computes mean, std, MAD (mean absolute deviation)
        - Computes baseline MAE (predicting mean for all samples)

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary with target statistics
        """
        try:
            targets = []

            for batch in train_loader:
                # Handle different batch formats
                if len(batch) == 3 and isinstance(batch[0], tuple):
                    _, target, _ = batch
                elif len(batch) == 4:
                    _, _, _, target = batch
                else:
                    continue

                targets.extend(target.cpu().numpy().flatten().tolist())

            if len(targets) == 0:
                return None

            targets = np.array(targets)

            # Compute statistics
            mean_val = float(np.mean(targets))
            std_val = float(np.std(targets))
            mad = float(np.mean(np.abs(targets - mean_val)))
            baseline_mae = float(np.mean(np.abs(targets - mean_val)))

            return {
                'mean': mean_val,
                'std': std_val,
                'mad': mad,
                'baseline_mae': baseline_mae,
                'n_samples': len(targets)
            }
        except Exception as e:
            print(f"[Warning] Could not compute target statistics: {e}")
            return None

    def _save_history(self):
        """Save training history"""
        if self.config.store_outputs:
            filepath = os.path.join(self.config.output_dir, "training_history.json")
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=2)

    def load_model(self, filepath: str):
        """Load model from checkpoint"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        print(f"Loaded model from {filepath}")

    def predict(self, data_loader, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Make predictions on new data

        Args:
            data_loader: Data loader for prediction
            output_path: Optional path to save predictions (CSV)

        Returns:
            Dictionary with predictions and IDs
        """
        self.model.eval()
        predictions = []
        ids = []

        print(f"Making predictions on {len(data_loader.dataset)} samples...")

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="Predicting")):
                # Handle different batch formats
                if len(batch) == 3 and isinstance(batch[0], tuple):
                    # New format: ((g, lg, lat), target, cif_ids)
                    (g, lg, lat), _, cif_ids = batch
                    g = g.to(self.device)
                    lg = lg.to(self.device)
                    lat = lat.to(self.device)

                    output = self.model((g, lg, lat))
                    predictions.extend(output.cpu().numpy().tolist())
                    ids.extend(cif_ids)

                elif len(batch) == 4:
                    # Legacy format: (g, lg, lat, target)
                    g, lg, lat, _ = batch
                    g = g.to(self.device)
                    lg = lg.to(self.device)
                    lat = lat.to(self.device)

                    output = self.model((g, lg, lat))
                    predictions.extend(output.cpu().numpy().tolist())

                    # Try to get ID
                    try:
                        if hasattr(data_loader.dataset, 'dataset'):
                            # Subset
                            id_val = data_loader.dataset.dataset.ids[
                                data_loader.dataset.indices[i]
                            ]
                        else:
                            # Direct dataset
                            id_val = data_loader.dataset.ids[i]
                        ids.append(id_val)
                    except:
                        ids.append(f"sample_{i}")

        predictions = np.array(predictions)

        # Save predictions if output path provided
        if output_path:
            import pandas as pd
            results_df = pd.DataFrame({
                'id': ids,
                'prediction': predictions
            })
            results_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")

        return {
            'predictions': predictions.tolist(),
            'ids': ids
        }

    def save_model(self, filepath: str):
        """
        Save model checkpoint with full state

        Args:
            filepath: Path to save model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.dict(),
            'model_config': self.config.model.dict(),
            'best_loss': self.best_loss,
            'best_mae': self.best_mae,
            'best_epoch': self.best_epoch,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_from_checkpoint(filepath: str, device: Optional[torch.device] = None):
        """
        Load trainer from checkpoint

        Args:
            filepath: Path to checkpoint
            device: Device to use (optional)

        Returns:
            ALIGNNTrainer instance
        """
        checkpoint = torch.load(filepath, map_location=device or torch.device('cpu'))

        # Reconstruct config
        config = TrainingConfig(**checkpoint['config'])
        model_config = ALIGNNConfig(**checkpoint['model_config'])
        config.model = model_config

        # Create model
        model = ALIGNN(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Create trainer
        trainer = ALIGNNTrainer(config, model, device)
        trainer.best_loss = checkpoint.get('best_loss', float('inf'))
        trainer.best_mae = checkpoint.get('best_mae', float('inf'))
        trainer.best_epoch = checkpoint.get('best_epoch', 0)
        trainer.history = checkpoint.get('history', {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rate': []
        })

        print(f"Loaded model from {filepath}")
        print(f"Best validation loss: {trainer.best_loss:.4f}")
        print(f"Best validation MAE: {trainer.best_mae:.4f}")
        print(f"Best epoch: {trainer.best_epoch}")

        return trainer
