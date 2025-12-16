"""
CGCNN Trainer
Handles model training, evaluation, and checkpointing

Features:
- Automatic device selection (GPU/CPU)
- Progress monitoring
- Early stopping
- Model checkpointing
- Metrics tracking
"""

import os
import time
import shutil
from typing import Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn import metrics


class Normalizer:
    """Normalize target values and restore them later."""

    def __init__(self, tensor):
        """
        Initialize normalizer from sample tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Sample tensor to calculate mean and std
        """
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        """Normalize tensor."""
        # Ensure mean and std are on the same device as input tensor
        mean = self.mean.to(tensor.device) if self.mean.device != tensor.device else self.mean
        std = self.std.to(tensor.device) if self.std.device != tensor.device else self.std
        return (tensor - mean) / std

    def denorm(self, normed_tensor):
        """Denormalize tensor."""
        # Ensure mean and std are on the same device as input tensor
        mean = self.mean.to(normed_tensor.device) if self.mean.device != normed_tensor.device else self.mean
        std = self.std.to(normed_tensor.device) if self.std.device != normed_tensor.device else self.std
        return normed_tensor * std + mean

    def state_dict(self):
        """Return state dict."""
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class AverageMeter:
    """Computes and stores average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.

    Implements a scientifically rigorous early stopping strategy with:
    - Configurable patience (number of epochs to wait)
    - Minimum delta threshold for improvement
    - Support for both minimization (regression) and maximization (classification)
    - Optional baseline threshold

    Reference:
        Prechelt, L. (1998). Early Stopping - But When?
        Neural Networks: Tricks of the Trade, Springer.
    """

    def __init__(
        self,
        patience: int = 10,
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
            Default: 10
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


class CGCNNTrainer:
    """
    Trainer for CGCNN models.

    Handles the complete training workflow with monitoring and checkpointing.
    """

    def __init__(
        self,
        model,
        task='regression',
        learning_rate=0.01,
        weight_decay=0,
        momentum=0.9,
        optimizer_type='SGD',
        lr_milestones=[100],
        epochs=30,
        device=None,
        print_freq=10,
        early_stopping_patience=0,
        early_stopping_min_delta=0.0
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : nn.Module
            CGCNN model to train
        task : str
            'regression' or 'classification'
        learning_rate : float
            Initial learning rate
        weight_decay : float
            Weight decay (L2 penalty)
        momentum : float
            Momentum for SGD
        optimizer_type : str
            'SGD' or 'Adam'
        lr_milestones : list
            Epochs to reduce learning rate
        epochs : int
            Number of training epochs
        device : str, optional
            Device to use (default: auto-select)
        print_freq : int
            Print frequency
        early_stopping_patience : int
            Patience for early stopping (0 = disabled)
        early_stopping_min_delta : float
            Minimum improvement threshold for early stopping
        """
        self.model = model
        self.task = task
        self.epochs = epochs
        self.print_freq = print_freq
        self.epoch_callback = None  # Optional callback for epoch completion
        self.batch_callback = None  # Optional callback for batch progress

        # Device setup
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Loss function
        if task == 'classification':
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.MSELoss()

        # Optimizer
        if optimizer_type == 'SGD':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Learning rate scheduler
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=lr_milestones,
            gamma=0.1
        )

        # Tracking
        self.normalizer = None
        self.best_metric = 1e10 if task == 'regression' else 0.
        self.training_history = []

        # Best model state (kept in memory, not saved to disk automatically)
        self.best_model_state = None

        # Early stopping
        self.early_stopping = None
        if early_stopping_patience > 0:
            es_mode = 'min' if task == 'regression' else 'max'
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode=es_mode,
                verbose=True
            )
            print(f"[Trainer] Early stopping enabled: patience={early_stopping_patience}, "
                  f"min_delta={early_stopping_min_delta}, mode={es_mode}")

    def train(self, train_loader, val_loader, test_loader=None):
        """
        Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        test_loader : DataLoader, optional
            Test data loader

        Returns
        -------
        results : dict
            Training results including best metrics and history

        Raises
        ------
        StopIteration
            When training is stopped by user via callback
        """
        # Setup normalizer
        self._setup_normalizer(train_loader)

        # Training loop
        try:
            for epoch in range(self.epochs):
                # Train one epoch
                train_metrics = self._train_epoch(train_loader, epoch)

                # Validate
                val_metric = self._validate(val_loader, epoch)

                # Check for NaN
                if val_metric != val_metric:
                    print('Exit due to NaN')
                    break

                # Update learning rate
                self.scheduler.step()

                # Check if best model
                if self.task == 'regression':
                    is_best = val_metric < self.best_metric
                    self.best_metric = min(val_metric, self.best_metric)
                else:
                    is_best = val_metric > self.best_metric
                    self.best_metric = max(val_metric, self.best_metric)

                # Build checkpoint state with model config
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_metric': self.best_metric,
                    'optimizer': self.optimizer.state_dict(),
                    'normalizer': self.normalizer.state_dict(),
                    'task': self.task,
                    'model_config': {
                        'orig_atom_fea_len': self.model.embedding.weight.shape[1],
                        'atom_fea_len': self.model.embedding.weight.shape[0],
                        'n_conv': len(self.model.convs) if hasattr(self.model, 'convs') else 3,
                        'h_fea_len': self.model.conv_to_fc.out_features if hasattr(self.model, 'conv_to_fc') else 128,
                        'n_h': (len(self.model.fcs) + 1) if hasattr(self.model, 'fcs') else 1,
                        'classification': self.model.classification if hasattr(self.model, 'classification') else False,
                        'n_classes': self.model.n_classes if hasattr(self.model, 'n_classes') else 2,
                    }
                }

                # Try to extract nbr_fea_len from first conv layer
                if hasattr(self.model, 'convs') and len(self.model.convs) > 0:
                    first_conv = self.model.convs[0]
                    if hasattr(first_conv, 'fc_full'):
                        checkpoint_state['model_config']['nbr_fea_len'] = (
                            first_conv.fc_full.in_features - 2 * checkpoint_state['model_config']['atom_fea_len']
                        )

                # Keep best model state in memory (not saved to disk)
                if is_best:
                    import copy
                    self.best_model_state = copy.deepcopy(checkpoint_state)
                    print(f"[Trainer] Best model updated at epoch {epoch + 1} (metric: {self.best_metric:.4f})")

                # Store history
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train': train_metrics,
                    'val': val_metric,
                    'is_best': is_best
                })

                # Callback
                if self.epoch_callback:
                    print(f"[Trainer] Calling epoch_callback for epoch {epoch + 1}")
                    try:
                        self.epoch_callback({
                            'epoch': epoch + 1,
                            'train_loss': train_metrics['loss'],
                            'val_metric': val_metric,
                            'is_best': is_best
                        })
                        print(f"[Trainer] epoch_callback completed")
                    except Exception as e:
                        print(f"[Trainer] ERROR in epoch_callback: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[Trainer] WARNING: epoch_callback is None!")

                # Check early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(epoch + 1, val_metric):
                        print(f"[Trainer] Early stopping triggered at epoch {epoch + 1}")
                        break

        except StopIteration as e:
            # Training stopped by user - re-raise to propagate
            print(f"[Trainer] Training stopped by user: {e}")
            raise

        # Test with best model (load from memory)
        train_results = None
        if test_loader is not None:
            print('-' * 50)
            print('Evaluating best model on test set...')
            # Load best model from memory
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state['state_dict'])
                self.normalizer.load_state_dict(self.best_model_state['normalizer'])
            test_results = self._test(test_loader)

            # Also evaluate on training set for visualization
            print('Evaluating on training set for visualization...')
            train_results = self._test_for_viz(train_loader)
        else:
            test_results = None

        return {
            'best_metric': self.best_metric,
            'history': self.training_history,
            'test_results': test_results,
            'train_results': train_results
        }

    def _setup_normalizer(self, train_loader):
        """Setup target normalizer from training data."""
        if self.task == 'classification':
            self.normalizer = Normalizer(torch.zeros(2))
            self.normalizer.load_state_dict({'mean': 0., 'std': 1.})
        else:
            # Sample targets from training set
            sample_targets = []
            for i, (_, target, _) in enumerate(train_loader):
                sample_targets.append(target)
                if i >= 10:  # Sample first few batches
                    break
            sample_target = torch.cat(sample_targets, dim=0)
            self.normalizer = Normalizer(sample_target)

    def _train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        if self.task == 'regression':
            mae_errors = AverageMeter()
        else:
            accuracies = AverageMeter()
            auc_scores = AverageMeter()

        # Switch to train mode
        self.model.train()

        end = time.time()
        for i, (input_data, target, _) in enumerate(train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)

            # Move to device
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
            # Use non_blocking=True for async transfer when pin_memory is enabled
            atom_fea = atom_fea.to(self.device, non_blocking=True)
            nbr_fea = nbr_fea.to(self.device, non_blocking=True)
            nbr_fea_idx = nbr_fea_idx.to(self.device, non_blocking=True)
            crystal_atom_idx = [idx.to(self.device, non_blocking=True) for idx in crystal_atom_idx]

            # Normalize target
            if self.task == 'regression':
                target_normed = self.normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()

            target_var = target_normed.to(self.device, non_blocking=True)

            # Forward pass
            output = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            loss = self.criterion(output, target_var)

            # Measure accuracy
            if self.task == 'regression':
                mae_error = self._mae(
                    self.normalizer.denorm(output.data.cpu()), target
                )
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
            else:
                accuracy, auc_score = self._class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                auc_scores.update(auc_score, target.size(0))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print progress and call batch callback
            if i % self.print_freq == 0:
                if self.task == 'regression':
                    log_msg = (f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                              f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                              f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')
                    print(log_msg)

                    # Batch callback with detailed info
                    if self.batch_callback:
                        self.batch_callback({
                            'epoch': epoch,
                            'batch': i,
                            'total_batches': len(train_loader),
                            'batch_time': batch_time.val,
                            'data_time': data_time.val,
                            'loss': losses.val,
                            'mae': mae_errors.val,
                            'avg_loss': losses.avg,
                            'avg_mae': mae_errors.avg,
                            'message': log_msg
                        })
                else:
                    log_msg = (f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                              f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                              f'Acc {accuracies.val:.3f} ({accuracies.avg:.3f})\t'
                              f'AUC {auc_scores.val:.3f} ({auc_scores.avg:.3f})')
                    print(log_msg)

                    # Batch callback with detailed info
                    if self.batch_callback:
                        self.batch_callback({
                            'epoch': epoch,
                            'batch': i,
                            'total_batches': len(train_loader),
                            'batch_time': batch_time.val,
                            'data_time': data_time.val,
                            'loss': losses.val,
                            'accuracy': accuracies.val,
                            'auc': auc_scores.val,
                            'avg_loss': losses.avg,
                            'avg_accuracy': accuracies.avg,
                            'avg_auc': auc_scores.avg,
                            'message': log_msg
                        })

        if self.task == 'regression':
            return {'loss': losses.avg, 'mae': mae_errors.avg}
        else:
            return {'loss': losses.avg, 'accuracy': accuracies.avg,
                    'auc': auc_scores.avg}

    def _validate(self, val_loader, epoch):
        """Validate the model."""
        losses = AverageMeter()

        if self.task == 'regression':
            mae_errors = AverageMeter()
        else:
            accuracies = AverageMeter()
            auc_scores = AverageMeter()

        # Switch to eval mode
        self.model.eval()

        with torch.no_grad():
            for i, (input_data, target, _) in enumerate(val_loader):
                # Move to device
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
                atom_fea = atom_fea.to(self.device, non_blocking=True)
                nbr_fea = nbr_fea.to(self.device, non_blocking=True)
                nbr_fea_idx = nbr_fea_idx.to(self.device, non_blocking=True)
                crystal_atom_idx = [
                    idx.to(self.device, non_blocking=True) for idx in crystal_atom_idx
                ]

                # Normalize target
                if self.task == 'regression':
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target.view(-1).long()

                target_var = target_normed.to(self.device, non_blocking=True)

                # Forward pass
                output = self.model(
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx
                )
                loss = self.criterion(output, target_var)

                # Measure accuracy
                if self.task == 'regression':
                    mae_error = self._mae(
                        self.normalizer.denorm(output.data.cpu()), target
                    )
                    losses.update(loss.data.cpu().item(), target.size(0))
                    mae_errors.update(mae_error, target.size(0))
                else:
                    accuracy, auc_score = self._class_eval(
                        output.data.cpu(), target
                    )
                    losses.update(loss.data.cpu().item(), target.size(0))
                    accuracies.update(accuracy, target.size(0))
                    auc_scores.update(auc_score, target.size(0))

        if self.task == 'regression':
            print(f'* Val MAE {mae_errors.avg:.3f}')
            return mae_errors.avg
        else:
            print(f'* Val AUC {auc_scores.avg:.3f}')
            return auc_scores.avg

    def _test(self, test_loader):
        """Test the model and save predictions."""
        test_targets = []
        test_preds = []
        test_cif_ids = []

        self.model.eval()

        with torch.no_grad():
            for input_data, target, batch_cif_ids in test_loader:
                # Move to device
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
                atom_fea = atom_fea.to(self.device, non_blocking=True)
                nbr_fea = nbr_fea.to(self.device, non_blocking=True)
                nbr_fea_idx = nbr_fea_idx.to(self.device, non_blocking=True)
                crystal_atom_idx = [
                    idx.to(self.device, non_blocking=True) for idx in crystal_atom_idx
                ]

                # Forward pass
                output = self.model(
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx
                )

                # Collect predictions
                if self.task == 'regression':
                    test_pred = self.normalizer.denorm(output.data.cpu())
                    test_preds += test_pred.view(-1).tolist()
                    test_targets += target.view(-1).tolist()
                else:
                    # Classification: convert log-probabilities to probabilities
                    test_pred = torch.exp(output.data.cpu())
                    n_classes = test_pred.shape[1]
                    if n_classes == 2:
                        # Binary classification: return probability of class 1
                        test_preds += test_pred[:, 1].tolist()
                    else:
                        # Multi-class: return all class probabilities as array
                        test_preds.append(test_pred.numpy())
                    test_targets += target.view(-1).tolist()

                test_cif_ids += batch_cif_ids

        # Save predictions
        import csv
        with open('test_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cif_id', 'target', 'prediction'])
            for cif_id, target, pred in zip(
                test_cif_ids, test_targets, test_preds
            ):
                # For multi-class, pred is an array - convert to string or take argmax
                if isinstance(pred, np.ndarray):
                    pred_str = np.argmax(pred)  # Save predicted class
                else:
                    pred_str = pred
                writer.writerow([cif_id, target, pred_str])

        # Calculate metrics
        if self.task == 'regression':
            mae = np.mean(np.abs(np.array(test_targets) - np.array(test_preds)))
            print(f'** Test MAE: {mae:.3f}')
            return {'mae': mae, 'predictions': test_preds, 'targets': test_targets}
        else:
            # Handle multi-class classification
            if len(test_preds) > 0 and isinstance(test_preds[0], np.ndarray):
                # Multi-class: test_preds is list of arrays, concatenate them
                test_preds_array = np.vstack(test_preds)
                n_classes = test_preds_array.shape[1]
                pred_labels = np.argmax(test_preds_array, axis=1)

                # Calculate accuracy
                accuracy = metrics.accuracy_score(test_targets, pred_labels)
                print(f'** Test Accuracy: {accuracy:.3f}')

                # Calculate macro F1
                f1 = metrics.f1_score(test_targets, pred_labels, average='macro', zero_division=0)
                print(f'** Test Macro F1: {f1:.3f}')

                # Try to calculate multi-class AUC
                try:
                    from sklearn.preprocessing import label_binarize
                    targets_onehot = label_binarize(test_targets, classes=range(n_classes))
                    auc = metrics.roc_auc_score(targets_onehot, test_preds_array,
                                                multi_class='ovr', average='macro')
                    print(f'** Test AUC (OVR): {auc:.3f}')
                except (ValueError, Exception) as e:
                    auc = f1  # Use F1 as fallback
                    print(f'** Test AUC unavailable: {e}')

                return {'auc': auc, 'accuracy': accuracy, 'f1': f1,
                        'predictions': test_preds_array, 'targets': test_targets}
            else:
                # Binary classification
                auc = metrics.roc_auc_score(test_targets, test_preds)
                print(f'** Test AUC: {auc:.3f}')
                return {'auc': auc, 'predictions': test_preds, 'targets': test_targets}

    def _test_for_viz(self, data_loader):
        """Evaluate model on a dataset for visualization (no CSV saving)."""
        preds = []
        targets_list = []

        self.model.eval()

        with torch.no_grad():
            for input_data, target, _ in data_loader:
                # Move to device
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
                atom_fea = atom_fea.to(self.device, non_blocking=True)
                nbr_fea = nbr_fea.to(self.device, non_blocking=True)
                nbr_fea_idx = nbr_fea_idx.to(self.device, non_blocking=True)
                crystal_atom_idx = [
                    idx.to(self.device, non_blocking=True) for idx in crystal_atom_idx
                ]

                # Forward pass
                output = self.model(
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx
                )

                # Collect predictions
                if self.task == 'regression':
                    pred = self.normalizer.denorm(output.data.cpu())
                    preds += pred.view(-1).tolist()
                    targets_list += target.view(-1).tolist()
                else:
                    # For classification, return all class probabilities for visualization
                    pred = torch.exp(output.data.cpu())  # Convert log-probabilities to probabilities
                    preds.append(pred.numpy())  # Keep as 2D array
                    targets_list += target.view(-1).tolist()

        # For classification, concatenate all prediction batches
        if self.task == 'classification' and len(preds) > 0:
            preds = np.vstack(preds)

        return {'predictions': preds, 'targets': targets_list}

    def _mae(self, prediction, target):
        """Calculate mean absolute error."""
        return torch.mean(torch.abs(target - prediction)).item()

    def _class_eval(self, prediction, target):
        """
        Evaluate classification performance.

        Supports both binary and multi-class classification.

        Parameters
        ----------
        prediction : torch.Tensor
            Log-softmax predictions from model
        target : torch.Tensor
            Ground truth labels

        Returns
        -------
        accuracy : float
            Classification accuracy
        auc_or_f1 : float
            For binary: ROC-AUC score
            For multi-class: Macro F1 score
        """
        prediction = np.exp(prediction.numpy())  # Convert log-prob to prob
        target = target.numpy()
        pred_label = np.argmax(prediction, axis=1)
        target_label = np.squeeze(target)

        if not target_label.shape:
            target_label = np.asarray([target_label])

        n_classes = prediction.shape[1]

        if n_classes == 2:
            # Binary classification: use ROC-AUC
            try:
                auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
            except ValueError:
                # ROC-AUC undefined when only one class present in batch
                auc_score = 0.5
            accuracy = metrics.accuracy_score(target_label, pred_label)
            return accuracy, auc_score
        else:
            # Multi-class classification: use macro F1 and weighted AUC
            accuracy = metrics.accuracy_score(target_label, pred_label)

            # Calculate macro F1 score
            f1_score = metrics.f1_score(target_label, pred_label, average='macro', zero_division=0)

            # Try to calculate multi-class AUC (one-vs-rest)
            try:
                # For multi-class AUC, we need one-hot encoded targets
                from sklearn.preprocessing import label_binarize
                n_unique = len(np.unique(target_label))
                if n_unique >= 2:
                    target_onehot = label_binarize(target_label, classes=range(n_classes))
                    auc_score = metrics.roc_auc_score(
                        target_onehot, prediction,
                        multi_class='ovr', average='macro'
                    )
                else:
                    auc_score = f1_score  # Fallback to F1 if AUC not computable
            except (ValueError, Exception):
                # If AUC fails, use F1 as the metric
                auc_score = f1_score

            return accuracy, auc_score

    def save_model(self, filepath):
        """
        Save the best model to a user-specified file path.

        Parameters
        ----------
        filepath : str
            Path where to save the model

        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        if self.best_model_state is None:
            print("[Trainer] No best model state available to save")
            return False

        try:
            torch.save(self.best_model_state, filepath)
            print(f"[Trainer] Best model saved to: {filepath}")
            return True
        except Exception as e:
            print(f"[Trainer] Failed to save model: {e}")
            return False

    def load_model(self, filepath):
        """
        Load a model from a file.

        Parameters
        ----------
        filepath : str
            Path to the model file

        Returns
        -------
        dict or None
            The loaded checkpoint dict, or None if loading failed
        """
        if not os.path.isfile(filepath):
            print(f"[Trainer] No checkpoint found at '{filepath}'")
            return None

        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])

            # Initialize normalizer if not already set
            if self.normalizer is None:
                self.normalizer = Normalizer(torch.zeros(2))

            self.normalizer.load_state_dict(checkpoint['normalizer'])

            # Also store as best model state
            self.best_model_state = checkpoint

            print(f"[Trainer] Loaded checkpoint from '{filepath}'")
            return checkpoint
        except Exception as e:
            print(f"[Trainer] Failed to load checkpoint: {e}")
            return None

