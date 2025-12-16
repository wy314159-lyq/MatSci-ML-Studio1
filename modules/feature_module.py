"""
Module 2: Advanced Feature Selection & Model Configuration
Implements comprehensive feature selection strategies with model-based evaluation
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QPushButton, QMessageBox,
                            QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit,
                            QSplitter, QTabWidget, QCheckBox, QListWidget,
                            QListWidgetItem, QProgressBar, QScrollArea,
                            QTableWidget, QTableWidgetItem, QSlider,
                            QLineEdit, QFrame, QDialog, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error

# Parallel processing imports
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing
import time

from utils.ml_utils import (get_available_models, get_default_hyperparameters,
                           remove_correlated_features, perform_hyperparameter_optimization,
                           get_hyperparameter_distributions, create_correlation_matrix_comparison,
                           visualize_hyperparameter_optimization_results, create_model_with_params)
from utils.plot_utils import (plot_feature_importance, create_figure, 
                             plot_correlation_heatmap, create_subplots_figure)


def evaluate_feature_subset_parallel(args):
    """
    Parallel wrapper function for evaluating feature subsets
    Args: (features, combined_data, model_params, cv_folds, scoring, y_name)
    """
    features, combined_data, model_params, cv_folds, scoring, y_name = args
    
    if len(features) == 0:
        return -np.inf, features
        
    try:
        # 从组合数据中提取所需列
        required_columns = features + [y_name]
        current_data = combined_data[required_columns].copy()
        
        # 处理缺失值，保持 X-y 对齐
        if current_data.isnull().any().any():
            current_data = current_data.dropna()
            
            # 检查剩余样本数量
            if len(current_data) < max(10, cv_folds):
                return -np.inf, features
        
        # 分离 X 和 y
        X_subset = current_data[features]
        y_subset = current_data[y_name]
        
        # 重建模型
        task_type, model_name = model_params['task_type'], model_params['model_name']
        available_models = get_available_models(task_type)
        
        if model_name in available_models:
            model_class = available_models[model_name]
            model = create_model_with_params(model_class, random_state=42)
        else:
            # 后备模型
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # 交叉验证评估
        scores = cross_val_score(model, X_subset, y_subset, cv=cv_folds, scoring=scoring)
        return np.mean(scores), features
        
    except Exception as e:
        print(f"Parallel evaluation error {features}: {e}")
        return -np.inf, features

def create_parallel_tasks(feature_combinations, combined_data, model_params, cv_folds, scoring, y_name):
    """Create parallel task list with proper data alignment"""
    tasks = []
    for features in feature_combinations:
        task = (list(features), combined_data, model_params, cv_folds, scoring, y_name)
        tasks.append(task)
    
    return tasks


class FeatureSelectionWorker(QThread):
    """Worker thread for feature selection operations"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    step_completed = pyqtSignal(str, dict)  # step_name, results
    error_occurred = pyqtSignal(str)
    
    def __init__(self, X, y, config):
        super().__init__()
        self.X = X.copy()
        self.y = y.copy()
        self.config = config
        self.current_features = list(X.columns)
        self.y_name = y.name if hasattr(y, 'name') and y.name else 'target'
        
        # Initialize optimized_model attribute to prevent AttributeError
        self.optimized_model = None
        
        # CRITICAL FIX: Split data early to prevent data leakage
        self.split_data_early()
        
        # Store combined data for feature selection (TRAINING DATA ONLY)
        self.data = self.X_train_fs.copy()
        self.data[self.y_name] = self.y_train_fs.copy()
        
        # Store final aligned data for module handoff
        self.final_data = None
        
        print(f"FeatureSelectionWorker initialized:")
        print(f"  Original data: X={X.shape}, y={y.shape}")
        print(f"  Training data for FS: X={self.X_train_fs.shape}, y={self.y_train_fs.shape}")
        print(f"  Test data (held out): X={self.X_test_fs.shape}, y={self.y_test_fs.shape}")
        print(f"  Features: {len(self.current_features)}")
        
    def split_data_early(self):
        """Split data early to prevent data leakage in feature selection"""
        from sklearn.model_selection import train_test_split
        
        # Get split ratio from config (default 0.8 for training)
        train_ratio = self.config.get('train_test_split', 0.8)
        test_size = 1.0 - train_ratio
        random_state = self.config.get('random_seed', 42)
        
        # Perform stratified split for classification
        task_type = self.config.get('task_type', 'classification')
        stratify = self.y if task_type == 'classification' else None
        
        try:
            self.X_train_fs, self.X_test_fs, self.y_train_fs, self.y_test_fs = train_test_split(
                self.X, self.y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify
            )
            
            print(f"✓ Data split successfully:")
            print(f"  Training: {self.X_train_fs.shape[0]} samples ({train_ratio*100:.1f}%)")
            print(f"  Test: {self.X_test_fs.shape[0]} samples ({test_size*100:.1f}%)")
            
            if task_type == 'classification':
                train_dist = pd.Series(self.y_train_fs).value_counts().sort_index()
                test_dist = pd.Series(self.y_test_fs).value_counts().sort_index()
                print(f"  Training class distribution: {dict(train_dist)}")
                print(f"  Test class distribution: {dict(test_dist)}")
                
        except Exception as e:
            print(f"ERROR in data splitting: {e}")
            # Fallback: use all data (no split)
            self.X_train_fs = self.X.copy()
            self.X_test_fs = self.X.iloc[:0].copy()  # Empty test set
            self.y_train_fs = self.y.copy()
            self.y_test_fs = self.y.iloc[:0].copy()  # Empty test set
            print("WARNING: Using all data for feature selection (no split)")
    
    def prepare_final_data(self):
        """Prepare final aligned data by combining training and test sets with selected features"""
        try:
            # Combine training and test data with selected features only
            X_train_selected = self.X_train_fs[self.current_features].copy()
            X_test_selected = self.X_test_fs[self.current_features].copy()
            
            # Combine back into full dataset
            X_full_selected = pd.concat([X_train_selected, X_test_selected], axis=0, ignore_index=False)
            y_full = pd.concat([self.y_train_fs, self.y_test_fs], axis=0, ignore_index=False)
            
            # Create final aligned dataset
            self.final_data = X_full_selected.copy()
            self.final_data[self.y_name] = y_full
            
            # Sort by index to maintain original order
            self.final_data = self.final_data.sort_index()
            
            print(f"✓ Final data prepared:")
            print(f"  Shape: {self.final_data.shape}")
            print(f"  Selected features: {len(self.current_features)}")
            print(f"  Columns: {list(self.final_data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"ERROR preparing final data: {e}")
            return False
    
    def apply_data_scaling(self, X):
        """
        DEPRECATED: Data scaling removed to prevent data leakage.
        All scaling operations should be handled in the training module's Pipeline.
        """
        # Return original data without any scaling to prevent data leakage
        print("Warning: Data scaling in feature module is disabled to prevent data leakage.")
        print("All scaling operations will be handled in the training module's Pipeline.")
        return X
    
    def run(self):
        """Execute the feature selection pipeline"""
        try:
            results = {}
            
            # Step 0: Hyperparameter optimization (if enabled)
            if self.config.get('enable_hyperopt', False):
                self.status_updated.emit("Step 0: Hyperparameter optimization...")
                self.progress_updated.emit(10)
                hyperopt_results = self.hyperparameter_optimization()
                results['hyperparameter_optimization'] = hyperopt_results
                self.step_completed.emit("hyperparameter_optimization", hyperopt_results)
            
            # Step 1: Model-based importance filtering
            if self.config.get('enable_importance_filter', True):
                self.status_updated.emit("Step 1: Model importance-based feature filtering...")
                self.progress_updated.emit(30)
                importance_results = self.importance_based_filtering()
                results['importance_filtering'] = importance_results
                self.step_completed.emit("importance_filtering", importance_results)
                
            # Step 2: Correlation-based redundancy filtering
            if self.config.get('enable_correlation_filter', True):
                self.status_updated.emit("Step 2: Correlation-based redundant feature filtering...")
                self.progress_updated.emit(60)
                correlation_results = self.correlation_based_filtering()
                results['correlation_filtering'] = correlation_results
                self.step_completed.emit("correlation_filtering", correlation_results)
                
            # Step 3: Advanced subset search (optional)
            if self.config.get('enable_advanced_search', False):
                self.status_updated.emit("Step 3: Advanced feature subset search...")
                self.progress_updated.emit(85)
                advanced_results = self.advanced_subset_search()
                results['advanced_search'] = advanced_results
                self.step_completed.emit("advanced_search", advanced_results)
                
            self.progress_updated.emit(100)
            
            # CRITICAL FIX: Prepare final aligned data by combining train/test with selected features
            if self.prepare_final_data():
                self.status_updated.emit("Feature selection completed successfully!")
                print(f"✓ Feature selection completed with {len(self.current_features)} features")
            else:
                self.status_updated.emit("Feature selection completed with data preparation issues")
                print("⚠️ Feature selection completed but final data preparation failed")
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def hyperparameter_optimization(self):
        """Step 0: Optimize hyperparameters for the selected model with early stopping"""
        try:
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from scipy.stats import uniform, randint
            import warnings
            warnings.filterwarnings('ignore')
            
            # Get base model
            task_type = self.config['task_type']
            models = self.config['selected_models']
            model_name = models[0] if models else 'Random Forest'
            
            # Get model class
            available_models = get_available_models(task_type)
            if model_name in available_models:
                model_class = available_models[model_name]
                base_model = model_class(random_state=42) if 'random_state' in model_class().get_params() else model_class()
            else:
                # Fallback
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                base_model = RandomForestClassifier(random_state=42) if task_type == 'classification' else RandomForestRegressor(random_state=42)
            
            # Prepare data with proper X-y alignment
            required_columns = self.current_features + [self.y_name]
            current_data = self.data[required_columns].copy()
            
            if current_data.isnull().any().any():
                current_data = current_data.dropna()
            
            X_current = current_data[self.current_features]
            y_current = current_data[self.y_name]
            
            # Get hyperparameter space
            search_method = self.config.get('search_method', 'Random Search')
            custom_spaces = self.config.get('custom_param_spaces', {})
            
            if model_name in custom_spaces:
                param_grid = {}
                for param_name, space_config in custom_spaces[model_name].items():
                    if isinstance(space_config, dict) and space_config.get('type') == 'range':
                        if search_method == 'Grid Search':
                            # For grid search, create discrete values
                            min_val, max_val = space_config['min'], space_config['max']
                            if isinstance(min_val, int):
                                param_grid[param_name] = list(range(int(min_val), int(max_val) + 1, max(1, (int(max_val) - int(min_val)) // 5)))
                            else:
                                param_grid[param_name] = np.linspace(min_val, max_val, 6).tolist()
                        else:
                            # For random search, use distributions
                            if isinstance(space_config['min'], int):
                                param_grid[param_name] = randint(int(space_config['min']), int(space_config['max']) + 1)
                            else:
                                param_grid[param_name] = uniform(space_config['min'], space_config['max'] - space_config['min'])
                    else:
                        param_grid[param_name] = space_config
            else:
                # Use default hyperparameters
                default_params = get_default_hyperparameters(model_name, task_type)
                param_grid = default_params or {}
            
            if not param_grid:
                # No hyperparameters to optimize
                self.optimized_model = base_model
                return {
                    'message': 'No hyperparameters to optimize',
                    'best_params': {},
                    'best_score': 0.0,
                    'model_name': model_name
                }
            
            # Setup cross-validation
            cv_folds = self.config.get('cv_folds', 5)
            scoring = self.config.get('scoring_metric', 'accuracy')
            n_iter = self.config.get('n_iter', 50)
            
            # Early stopping parameters
            early_stopping_enabled = self.config.get('enable_early_stopping', True)
            patience = self.config.get('early_stopping_patience', 10)
            min_improvement = self.config.get('early_stopping_min_improvement', 0.001)
            
            # Custom Early Stopping Search CV
            class EarlyStoppingSearchCV:
                def __init__(self, estimator, param_grid, cv=5, scoring='accuracy', 
                           n_iter=50, patience=10, min_improvement=0.001, random_state=42):
                    self.estimator = estimator
                    self.param_grid = param_grid
                    self.cv = cv
                    self.scoring = scoring
                    self.n_iter = n_iter
                    self.patience = patience
                    self.min_improvement = min_improvement
                    self.random_state = random_state
                    self.best_score_ = -np.inf
                    self.best_params_ = {}
                    self.best_estimator_ = None
                    self.cv_results_ = {
                        'params': [],
                        'mean_test_score': [],
                        'std_test_score': [],
                        'rank_test_score': [],
                        'iteration': [],
                        'early_stopped': False,
                        'stopping_iteration': None
                    }
                    
                def _generate_param_combinations(self):
                    """Generate parameter combinations for random search"""
                    np.random.seed(self.random_state)
                    combinations = []
                    
                    for _ in range(self.n_iter):
                        params = {}
                        for param_name, param_space in self.param_grid.items():
                            if hasattr(param_space, 'rvs'):  # scipy distribution
                                params[param_name] = param_space.rvs()
                            elif isinstance(param_space, list):
                                params[param_name] = np.random.choice(param_space)
                            else:
                                params[param_name] = param_space
                        combinations.append(params)
                    return combinations
                
                def fit(self, X, y):
                    """Fit with early stopping"""
                    param_combinations = self._generate_param_combinations()
                    
                    best_scores_history = []
                    no_improvement_count = 0
                    
                    for i, params in enumerate(param_combinations):
                        # Create model with current parameters
                        model = self.estimator.__class__(**{**self.estimator.get_params(), **params})
                        
                        # Perform cross-validation
                        try:
                            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
                            mean_score = np.mean(scores)
                            std_score = np.std(scores)
                        except Exception as e:
                            print(f"Error evaluating params {params}: {e}")
                            mean_score = -np.inf
                            std_score = 0
                        
                        # Store results
                        self.cv_results_['params'].append(params)
                        self.cv_results_['mean_test_score'].append(mean_score)
                        self.cv_results_['std_test_score'].append(std_score)
                        self.cv_results_['iteration'].append(i)
                        
                        # Update best score
                        if mean_score > self.best_score_:
                            improvement = mean_score - self.best_score_
                            self.best_score_ = mean_score
                            self.best_params_ = params.copy()
                            self.best_estimator_ = model
                            
                            # Reset patience if significant improvement
                            if improvement >= self.min_improvement:
                                no_improvement_count = 0
                            else:
                                no_improvement_count += 1
                        else:
                            no_improvement_count += 1
                        
                        best_scores_history.append(self.best_score_)
                        
                        # Early stopping check
                        if no_improvement_count >= self.patience and i >= self.patience:
                            self.cv_results_['early_stopped'] = True
                            self.cv_results_['stopping_iteration'] = i
                            print(f"Early stopping at iteration {i+1}/{self.n_iter} (patience={self.patience})")
                            break
                        
                        # Progress update
                        if hasattr(self, 'progress_callback'):
                            self.progress_callback(i + 1, len(param_combinations))
                    
                    # Rank results
                    scores = np.array(self.cv_results_['mean_test_score'])
                    ranks = len(scores) - np.argsort(np.argsort(scores))
                    self.cv_results_['rank_test_score'] = ranks.tolist()
                    
                    return self
            
            # Perform hyperparameter optimization with early stopping
            if early_stopping_enabled and search_method != 'Grid Search':
                # Use custom early stopping search for Random Search
                search = EarlyStoppingSearchCV(
                    base_model, param_grid,
                    cv=cv_folds, scoring=scoring, n_iter=n_iter,
                    patience=patience, min_improvement=min_improvement,
                    random_state=42
                )
                
                # Add progress callback
                def progress_callback(current, total):
                    progress = int((current / total) * 100)
                    self.status_updated.emit(f"Hyperparameter optimization: {current}/{total} iterations")
                
                search.progress_callback = progress_callback
                search.fit(X_current, y_current)
                
            elif search_method == 'Grid Search':
                # Use standard GridSearchCV (early stopping not applicable)
                search = GridSearchCV(
                    base_model, param_grid,
                    cv=cv_folds, scoring=scoring,
                    n_jobs=-1, verbose=0
                )
                search.fit(X_current, y_current)
                
            else:
                # Use standard RandomizedSearchCV without early stopping
                search = RandomizedSearchCV(
                    base_model, param_grid,
                    n_iter=n_iter, cv=cv_folds, scoring=scoring,
                    n_jobs=-1, verbose=0, random_state=42
                )
                search.fit(X_current, y_current)
            
            # Store optimized model
            self.optimized_model = search.best_estimator_
            
            # Prepare results
            results_df = pd.DataFrame(search.cv_results_)
            
            # Add early stopping information to results
            early_stopping_info = {}
            if hasattr(search, 'cv_results_') and isinstance(search.cv_results_, dict):
                early_stopping_info = {
                    'early_stopped': search.cv_results_.get('early_stopped', False),
                    'stopping_iteration': search.cv_results_.get('stopping_iteration', None),
                    'total_iterations': len(search.cv_results_.get('params', [])),
                    'patience_used': patience,
                    'min_improvement_threshold': min_improvement
                }
            
            return {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_,
                'results_df': results_df,
                'model_name': model_name,
                'search_method': search_method,
                'optimization_results': search.cv_results_,
                'early_stopping_info': early_stopping_info,
                'early_stopping_enabled': early_stopping_enabled
            }
            
        except Exception as e:
            print(f"Error in hyperparameter optimization: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to base model
            try:
                # Try to create a simple fallback model
                task_type = self.config.get('task_type', 'classification')
                if task_type == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    self.optimized_model = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    self.optimized_model = RandomForestRegressor(n_estimators=50, random_state=42)
            except:
                self.optimized_model = None
                
            return {
                'error': str(e),
                'message': 'Hyperparameter optimization failed, using default parameters'
            }
            
    def importance_based_filtering(self):
        """Step 1: Filter features based on model importance with proper X-y alignment"""
        # Get model from config using the same logic as evaluation
        model = self.get_evaluation_model()
        
        # Extract current features and target from unified data source
        required_columns = self.current_features + [self.y_name]
        current_data = self.data[required_columns].copy()
        
        # Handle NaN values while maintaining X-y alignment
        if current_data.isnull().any().any():
            current_data = current_data.dropna()
            
            # If too few samples remain, use a fallback approach
            if len(current_data) < 10:
                print("Warning: Too many missing values for importance calculation")
                # Return all features as equally important
                importances = np.ones(len(self.current_features))
            else:
                # Separate X and y from aligned data
                X_current = current_data[self.current_features]
                y_current = current_data[self.y_name]
                
                # Fit model and get importance
                try:
                    model.fit(X_current, y_current)
                    
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        importances = np.abs(model.coef_).flatten()
                    else:
                        # Use permutation importance as fallback
                        from sklearn.inspection import permutation_importance
                        perm_importance = permutation_importance(model, X_current, y_current, random_state=42)
                        importances = perm_importance.importances_mean
                except Exception as e:
                    print(f"Error in importance calculation: {e}")
                    # Fallback to permutation importance
                    from sklearn.inspection import permutation_importance
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    
                    fallback_model = RandomForestClassifier(n_estimators=50, random_state=42) if self.config['task_type'] == 'classification' else RandomForestRegressor(n_estimators=50, random_state=42)
                    fallback_model.fit(X_current, y_current)
                    perm_importance = permutation_importance(fallback_model, X_current, y_current, random_state=42)
                    importances = perm_importance.importances_mean
        else:
            # No missing values, proceed normally
            X_current = current_data[self.current_features]
            y_current = current_data[self.y_name]
            
            # Fit model and get importance
            try:
                model.fit(X_current, y_current)
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_).flatten()
                else:
                    # Use permutation importance as fallback
                    from sklearn.inspection import permutation_importance
                    perm_importance = permutation_importance(model, X_current, y_current, random_state=42)
                    importances = perm_importance.importances_mean
            except Exception as e:
                print(f"Error in importance calculation: {e}")
                # Fallback to permutation importance
                from sklearn.inspection import permutation_importance
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                fallback_model = RandomForestClassifier(n_estimators=50, random_state=42) if self.config['task_type'] == 'classification' else RandomForestRegressor(n_estimators=50, random_state=42)
                fallback_model.fit(X_current, y_current)
                perm_importance = permutation_importance(fallback_model, X_current, y_current, random_state=42)
                importances = perm_importance.importances_mean
        
        # Create importance dataframe with One-Hot group aggregation
        importance_df = pd.DataFrame({
            'feature': self.current_features,
            'importance': importances
        })
        
        # ENHANCEMENT: Aggregate One-Hot group importances
        if hasattr(self, 'one_hot_groups') and self.one_hot_groups:
            aggregated_importances = self._aggregate_one_hot_importances(importance_df)
            print(f"=== ONE-HOT IMPORTANCE AGGREGATION ===")
            for group_name, (group_importance, group_features) in aggregated_importances.items():
                individual_scores = [importance_df[importance_df['feature'] == f]['importance'].iloc[0] 
                                   for f in group_features if f in importance_df['feature'].values]
                print(f"Group '{group_name}': {group_importance:.6f} (sum of {individual_scores})")
            print("=" * 50)
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Apply filtering based on config
        filter_type = self.config.get('importance_filter_type', 'top_k_features')
        
        if filter_type in ['top_k', 'top_k_features']:
            n_features = self.config.get('importance_top_k', 50)
            selected_features = importance_df.head(n_features)['feature'].tolist()
        elif filter_type in ['threshold', 'importance_threshold']:
            threshold = self.config.get('importance_threshold', 0.01)
            selected_features = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()
        else:  # cumulative percentage
            cum_importance = importance_df['importance'].cumsum() / importance_df['importance'].sum()
            threshold = self.config.get('importance_cumulative', 0.95)
            selected_features = importance_df[cum_importance <= threshold]['feature'].tolist()
        
        self.current_features = selected_features
        
        return {
            'importance_scores': importance_df,
            'selected_features': selected_features,
            'n_features_before': len(importance_df),
            'n_features_after': len(selected_features)
        }
    
    def correlation_based_filtering(self):
        """Step 2: Remove highly correlated features with before/after visualization"""
        # Extract current features from unified data source
        required_columns = self.current_features + [self.y_name]
        current_data = self.data[required_columns].copy()
        
        # Handle missing values while maintaining alignment
        if current_data.isnull().any().any():
            current_data = current_data.dropna()
        
        X_current = current_data[self.current_features]
        
        # Calculate correlation matrix BEFORE filtering
        corr_matrix_before = X_current.corr().abs()
        
        # Find highly correlated pairs
        threshold = self.config.get('correlation_threshold', 0.95)
        highly_corr_pairs = []
        
        for i in range(len(corr_matrix_before.columns)):
            for j in range(i+1, len(corr_matrix_before.columns)):
                if corr_matrix_before.iloc[i, j] >= threshold:
                    highly_corr_pairs.append((corr_matrix_before.columns[i], corr_matrix_before.columns[j], corr_matrix_before.iloc[i, j]))
        
        # Remove features based on correlation strategy
        features_to_remove = set()
        removal_method = self.config.get('correlation_removal_method', 'model_based')
        
        if removal_method == 'model_based':
            # Use model performance to decide which feature to keep
            for feat1, feat2, corr_val in highly_corr_pairs:
                if feat1 in features_to_remove or feat2 in features_to_remove:
                    continue
                    
                # Compare model performance with each feature removed
                features_without_feat1 = [f for f in self.current_features if f != feat1 and f not in features_to_remove]
                features_without_feat2 = [f for f in self.current_features if f != feat2 and f not in features_to_remove]
                
                score1 = self.evaluate_feature_subset(features_without_feat1)
                score2 = self.evaluate_feature_subset(features_without_feat2)
                
                # Keep the feature that results in better performance when the other is removed
                if score1 > score2:  # removing feat1 gives better performance, so keep feat2
                    features_to_remove.add(feat1)
                else:  # removing feat2 gives better performance, so keep feat1
                    features_to_remove.add(feat2)
        else:
            # Simple method: remove feature with lower correlation to target
            y_current = current_data[self.y_name]
            for feat1, feat2, corr_val in highly_corr_pairs:
                if feat1 in features_to_remove or feat2 in features_to_remove:
                    continue
                    
                corr1 = abs(X_current[feat1].corr(y_current))
                corr2 = abs(X_current[feat2].corr(y_current))
                
                if corr1 < corr2:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
        
        # Update current features
        n_features_before = len(self.current_features)
        self.current_features = [f for f in self.current_features if f not in features_to_remove]
        
        # Calculate correlation matrix AFTER filtering
        if len(self.current_features) > 1:
            # Extract updated features from aligned data
            after_columns = self.current_features + [self.y_name]
            after_data = self.data[after_columns].copy()
            if after_data.isnull().any().any():
                after_data = after_data.dropna()
            X_after = after_data[self.current_features]
            corr_matrix_after = X_after.corr().abs()
        else:
            corr_matrix_after = None
        
        return {
            'highly_corr_pairs': highly_corr_pairs,
            'features_removed': list(features_to_remove),
            'n_features_before': n_features_before,
            'n_features_after': len(self.current_features),
            'correlation_before': corr_matrix_before,
            'correlation_after': corr_matrix_after,
            'threshold': threshold
        }
    
    def advanced_subset_search(self):
        """Step 3: Advanced feature subset search"""
        search_method = self.config.get('advanced_search_method', 'sequential_feature_selection')
        
        # 修复方法名映射问题
        if search_method == 'sequential_feature_selection':
            return self.sequential_feature_selection()
        elif search_method == 'genetic_algorithm':
            return self.genetic_algorithm_search()
        elif search_method == 'exhaustive_search':
            return self.exhaustive_search()
        else:
            # 兼容旧的映射方式
            if search_method == 'sequential':
                return self.sequential_feature_selection()
            elif search_method == 'genetic':
                return self.genetic_algorithm_search()
            else:  # exhaustive
                return self.exhaustive_search()
    
    def sequential_feature_selection(self):
        """Sequential feature selection (forward or backward) - simplified implementation"""
        try:
            # Try to use mlxtend if available
            from mlxtend.feature_selection import SequentialFeatureSelector as SFS
            
            # Get model for evaluation
            model = self.get_evaluation_model()
            
            direction = self.config.get('sfs_direction', 'forward')
            k_features = self.config.get('sfs_k_features', 10)
            
            sfs = SFS(model, 
                      k_features=k_features,
                      forward=(direction == 'forward'),
                      floating=False,
                      scoring=self.config.get('scoring_metric', 'accuracy'),
                      cv=self.config.get('cv_folds', 5))
            
            X_current = self.X[self.current_features]
            sfs.fit(X_current, self.y)
            
            selected_indices = list(sfs.k_feature_idx_)
            selected_features = [self.current_features[i] for i in selected_indices]
            
            self.current_features = selected_features
            
            return {
                'method': 'Sequential Feature Selection',
                'direction': direction,
                'selected_features': selected_features,
                'n_features_after': len(selected_features),
                'best_score': sfs.k_score_
            }
            
        except ImportError:
            # Fallback to simple greedy selection if mlxtend is not available
            return self.simple_sequential_selection()
    
    def simple_sequential_selection(self):
        """Simple sequential feature selection implementation"""
        direction = self.config.get('sfs_direction', 'forward')
        k_features = min(self.config.get('sfs_k_features', 10), len(self.current_features))
        
        if direction == 'forward':
            # Forward selection
            selected_features = []
            remaining_features = self.current_features.copy()
            best_score = -np.inf
            
            for _ in range(k_features):
                best_feature = None
                best_current_score = -np.inf
                
                for feature in remaining_features:
                    test_features = selected_features + [feature]
                    score = self.evaluate_feature_subset(test_features)
                    
                    if score > best_current_score:
                        best_current_score = score
                        best_feature = feature
                
                if best_feature is not None:
                    selected_features.append(best_feature)
                    remaining_features.remove(best_feature)
                    best_score = best_current_score
                else:
                    break
        else:
            # Backward elimination
            selected_features = self.current_features.copy()
            best_score = self.evaluate_feature_subset(selected_features)
            
            while len(selected_features) > k_features:
                worst_feature = None
                best_current_score = -np.inf
                
                for feature in selected_features:
                    test_features = [f for f in selected_features if f != feature]
                    if len(test_features) > 0:
                        score = self.evaluate_feature_subset(test_features)
                        
                        if score > best_current_score:
                            best_current_score = score
                            worst_feature = feature
                
                if worst_feature is not None:
                    selected_features.remove(worst_feature)
                    best_score = best_current_score
                else:
                    break
        
        self.current_features = selected_features
        
        return {
            'method': 'Sequential Feature Selection',
            'direction': direction,
            'selected_features': selected_features,
            'n_features_after': len(selected_features),
            'best_score': best_score
        }
    
    def genetic_algorithm_search(self):
        """Genetic algorithm for feature selection with parallel evaluation and visualization data"""
        from sklearn.model_selection import cross_val_score
        
        population_size = self.config.get('ga_population_size', 50)
        generations = self.config.get('ga_generations', 20)
        mutation_rate = self.config.get('ga_mutation_rate', 0.1)
        crossover_rate = self.config.get('ga_crossover_rate', 0.8)
        tournament_size = self.config.get('ga_tournament_size', 3)
        elite_ratio = self.config.get('ga_elite_ratio', 0.1)
        
        # 启用并行处理
        use_parallel = self.config.get('use_parallel', True) and population_size > 10
        n_workers = min(multiprocessing.cpu_count(), self.config.get('n_workers', 4))
        
        # Initialize random population (向量化)
        n_features = len(self.current_features)
        population = np.random.choice([True, False], size=(population_size, n_features), p=[0.3, 0.7])
        
        # Ensure at least one feature is selected for each individual
        for i in range(population_size):
            if not any(population[i]):
                population[i][np.random.randint(n_features)] = True
        
        model = self.get_evaluation_model()
        best_score = -np.inf
        best_features = None
        
        # Store evolution history for visualization
        self.ga_evolution_history = []
        
        for generation in range(generations):
            self.status_updated.emit(f"Genetic Algorithm: Generation {generation+1}/{generations}")
            
            # Evaluate population (并行或单线程)
            if use_parallel:
                scores = self._evaluate_population_parallel(population, n_workers)
            else:
                scores = []
                for individual in population:
                    selected_features = [self.current_features[i] for i, selected in enumerate(individual) if selected]
                    if len(selected_features) > 0:
                        score = self.evaluate_feature_subset(selected_features)
                    else:
                        score = -np.inf
                    scores.append(score)
            
            # Store generation statistics
            generation_stats = {
                'generation': generation,
                'best_score': max(scores),
                'avg_score': np.mean(scores),
                'worst_score': min(scores),
                'avg_features': np.mean([np.sum(ind) for ind in population]),
                'population_diversity': self.calculate_population_diversity(population)
            }
            self.ga_evolution_history.append(generation_stats)
            
            # Find best individual
            best_idx = np.argmax(scores)
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_features = [self.current_features[i] for i, selected in enumerate(population[best_idx]) if selected]
            
            # Elite selection - keep best individuals
            elite_count = int(population_size * elite_ratio)
            elite_indices = np.argsort(scores)[-elite_count:]
            elites = [population[i].copy() for i in elite_indices]
            
            # Selection (tournament selection) for remaining population
            new_population = elites.copy()
            remaining_slots = population_size - elite_count
            
            for _ in range(remaining_slots // 2):
                parent1 = self.tournament_selection(population, scores, tournament_size)
                parent2 = self.tournament_selection(population, scores, tournament_size)
                
                # Crossover
                if np.random.random() < crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)
                
                new_population.extend([child1, child2])
            
            # Handle odd population size
            if len(new_population) < population_size:
                parent = self.tournament_selection(population, scores, tournament_size)
                child = self.mutate(parent.copy(), mutation_rate)
                new_population.append(child)
            
            # Keep population size constant
            population = new_population[:population_size]
        
        self.current_features = best_features
        
        return {
            'method': 'Genetic Algorithm',
            'selected_features': best_features,
            'n_features_after': len(best_features),
            'best_score': best_score,
            'evolution_history': self.ga_evolution_history,
            'parameters': {
                'population_size': population_size,
                'generations': generations,
                'mutation_rate': mutation_rate,
                'crossover_rate': crossover_rate,
                'tournament_size': tournament_size,
                'elite_ratio': elite_ratio
            }
        }
    
    def calculate_population_diversity(self, population):
        """Calculate population diversity (average Hamming distance)"""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0
        count = 0
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                distance = np.sum(population[i] != population[j]) / len(population[i])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def tournament_selection(self, population, scores, tournament_size=3):
        """Tournament selection for genetic algorithm"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        # Ensure at least one feature is selected
        if not any(child1):
            child1[np.random.randint(len(child1))] = True
        if not any(child2):
            child2[np.random.randint(len(child2))] = True
            
        return child1, child2
    
    def mutate(self, individual, mutation_rate):
        """Bit-flip mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = not mutated[i]
        
        # Ensure at least one feature is selected
        if not any(mutated):
            mutated[np.random.randint(len(mutated))] = True
            
        return mutated
    
    def exhaustive_search(self):
        """Exhaustive search with parallel processing and progress tracking"""
        if len(self.current_features) > 15:
            raise ValueError("Exhaustive search is only feasible for ≤15 features")
        
        from itertools import combinations
        from math import comb
        
        min_features = self.config.get('exhaustive_min_features', 1)
        max_features = self.config.get('exhaustive_max_features', len(self.current_features))
        
        # Validate parameters
        min_features = max(1, min(min_features, len(self.current_features)))
        max_features = max(min_features, min(max_features, len(self.current_features)))
        
        if min_features > max_features:
            min_features = max_features
        
        best_score = -np.inf
        best_features = None
        
        # Store search history for visualization
        self.exhaustive_search_history = []
        total_combinations = sum(comb(len(self.current_features), k) 
                               for k in range(min_features, max_features + 1))
        
        # 准备并行处理
        use_parallel = self.config.get('use_parallel', True) and total_combinations > 10
        n_workers = min(multiprocessing.cpu_count(), self.config.get('n_workers', 4))
        
        combination_count = 0
        
        if use_parallel:
            self.status_updated.emit(f"Exhaustive search: Using {n_workers} parallel workers")
            
            # 收集所有特征组合
            all_combinations = []
            for k in range(min_features, max_features + 1):
                all_combinations.extend(combinations(self.current_features, k))
            
            # 准备模型参数
            models = self.config['selected_models']
            model_name = models[0] if models else 'Random Forest'
            model_params = {
                'task_type': self.config['task_type'],
                'model_name': model_name
            }
            
            # 创建并行任务
            batch_size = max(1, len(all_combinations) // (n_workers * 4))  # 动态批次大小
            batches = [all_combinations[i:i + batch_size] for i in range(0, len(all_combinations), batch_size)]
            
            # 并行执行
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_batch = {}
                
                for batch in batches:
                    future = executor.submit(
                        self._evaluate_combination_batch,
                        batch, model_params
                    )
                    future_to_batch[future] = batch
                
                # 收集结果
                for future in as_completed(future_to_batch):
                    batch_results = future.result()
                    
                    for features, score in batch_results:
                        combination_count += 1
                        
                        # Store combination result
                        self.exhaustive_search_history.append({
                            'features': list(features),
                            'n_features': len(features),
                            'score': score,
                            'combination_id': combination_count
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_features = list(features)
                    
                    # Update progress
                    progress = int((combination_count / total_combinations) * 15) + 85
                    self.progress_updated.emit(min(progress, 100))
        else:
            # 单线程执行（用于小数据集）
            for k in range(min_features, max_features + 1):
                self.status_updated.emit(f"Exhaustive search: Testing {k}-feature combinations")
                
                for feature_combo in combinations(self.current_features, k):
                    combination_count += 1
                    score = self.evaluate_feature_subset(list(feature_combo))
                    
                    # Store combination result
                    self.exhaustive_search_history.append({
                        'features': list(feature_combo),
                        'n_features': len(feature_combo),
                        'score': score,
                        'combination_id': combination_count
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_features = list(feature_combo)
                    
                    # Update progress
                    progress = int((combination_count / total_combinations) * 15) + 85
                    self.progress_updated.emit(min(progress, 100))
        
        # Ensure we have valid features selected
        if best_features is None or len(best_features) == 0:
            # Fallback to minimum feature combination that worked
            if self.exhaustive_search_history:
                valid_combinations = [h for h in self.exhaustive_search_history if h['score'] > -np.inf]
                if valid_combinations:
                    best_combo = max(valid_combinations, key=lambda x: x['score'])
                    best_features = best_combo['features']
                    best_score = best_combo['score']
                else:
                    # No valid combinations found, keep at least one feature
                    best_features = [self.current_features[0]] if self.current_features else []
            else:
                best_features = [self.current_features[0]] if self.current_features else []
        
        self.current_features = best_features
        
        return {
            'method': 'Exhaustive Search',
            'selected_features': best_features,
            'n_features_after': len(best_features) if best_features else 0,
            'best_score': best_score,
            'search_history': self.exhaustive_search_history,
            'total_combinations': total_combinations,
            'parameters': {
                'min_features': min_features,
                'max_features': max_features
            }
        }
    
    def get_evaluation_model(self):
        """Get model for evaluation based on user selection, preferring optimized model"""
        from utils.ml_utils import create_model_with_params
        
        # If we have an optimized model from hyperparameter optimization, use it
        if hasattr(self, 'optimized_model') and self.optimized_model is not None:
            return self.optimized_model
        
        task_type = self.config['task_type']
        models = self.config['selected_models']
        
        if not models:
            # Fallback if no models selected
            if task_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                return create_model_with_params(RandomForestClassifier, n_estimators=50, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                return create_model_with_params(RandomForestRegressor, n_estimators=50, random_state=42)
        
        # Use first selected model
        model_name = models[0]
        
        try:
            # Get model from ml_utils
            available_models = get_available_models(task_type)
            if model_name in available_models:
                model_class = available_models[model_name]
                # Create model with smart parameter handling
                return create_model_with_params(model_class, random_state=42)
            else:
                # Fallback to Random Forest
                if task_type == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    return create_model_with_params(RandomForestClassifier, n_estimators=50, random_state=42)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    return create_model_with_params(RandomForestRegressor, n_estimators=50, random_state=42)
                    
        except Exception as e:
            print(f"Error creating model {model_name}: {e}")
            # Fallback to Random Forest
            if task_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                return create_model_with_params(RandomForestClassifier, n_estimators=50, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                return create_model_with_params(RandomForestRegressor, n_estimators=50, random_state=42)
    
    def evaluate_feature_subset(self, features):
        """Evaluate a subset of features with proper X-y alignment"""
        if len(features) == 0:
            return -np.inf
            
        try:
            # 1. Extract current features and target from the unified data source
            required_columns = features + [self.y_name]
            current_data = self.data[required_columns].copy()
            
            # 2. Handle missing values on the combined data to maintain alignment
            if current_data.isnull().any().any():
                # Drop rows with any missing values to maintain X-y alignment
                current_data = current_data.dropna()
                
                # If too few samples remain after dropping NaN, return poor score
                if len(current_data) < max(10, self.config.get('cv_folds', 5)):
                    return -np.inf
            
            # 3. Separate X and y from the aligned data
            X_subset = current_data[features]
            y_subset = current_data[self.y_name]
            
            # 4. Safety check
            if X_subset.empty or y_subset.empty:
                return -np.inf
            
            model = self.get_evaluation_model()
            
            # Use cross-validation with aligned X and y
            cv_folds = self.config.get('cv_folds', 5)
            scoring = self.config.get('scoring_metric', 'accuracy')
            
            scores = cross_val_score(model, X_subset, y_subset, cv=cv_folds, scoring=scoring)
            return np.mean(scores)
        except Exception as e:
            print(f"Error evaluating feature subset {features}: {e}")
            return -np.inf
    
    def _evaluate_combination_batch(self, combinations, model_params):
        """批量评估特征组合 - 修复了 X-y 对齐问题"""
        results = []
        
        try:
            # 获取评估模型
            task_type = model_params['task_type']
            model_name = model_params['model_name']
            
            available_models = get_available_models(task_type)
            if model_name in available_models:
                model_class = available_models[model_name]
                model = create_model_with_params(model_class, random_state=42)
            else:
                # 后备模型
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                if task_type == 'classification':
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            cv_folds = self.config.get('cv_folds', 5)
            scoring = self.config.get('scoring_metric', 'accuracy')
            
            # 评估每个组合
            for features in combinations:
                features_list = list(features)
                if len(features_list) == 0:
                    results.append((features_list, -np.inf))
                    continue
                
                try:
                    # 从统一数据源提取特征和目标，保持对齐
                    required_columns = features_list + [self.y_name]
                    current_data = self.data[required_columns].copy()
                    
                    # 处理缺失值，保持 X-y 对齐
                    if current_data.isnull().any().any():
                        current_data = current_data.dropna()
                        
                        # 检查剩余样本数量
                        if len(current_data) < max(10, cv_folds):
                            results.append((features_list, -np.inf))
                            continue
                    
                    # 分离 X 和 y
                    X_subset = current_data[features_list]
                    y_subset = current_data[self.y_name]
                    
                    # 交叉验证评估
                    scores = cross_val_score(model, X_subset, y_subset, cv=cv_folds, scoring=scoring)
                    score = np.mean(scores)
                    results.append((features_list, score))
                    
                except Exception as e:
                    print(f"Batch evaluation error for {features_list}: {e}")
                    results.append((features_list, -np.inf))
                    
        except Exception as e:
            print(f"Batch processing error: {e}")
            # 返回失败结果
            for features in combinations:
                results.append((list(features), -np.inf))
        
        return results
    
    def _evaluate_population_parallel(self, population, n_workers):
        """并行评估遗传算法种群"""
        scores = []
        
        # 准备评估任务
        feature_combinations = []
        for individual in population:
            selected_features = [self.current_features[i] for i, selected in enumerate(individual) if selected]
            feature_combinations.append(selected_features if len(selected_features) > 0 else [])
        
        # 分批并行处理
        batch_size = max(1, len(feature_combinations) // n_workers)
        batches = [feature_combinations[i:i + batch_size] for i in range(0, len(feature_combinations), batch_size)]
        
        # 准备模型参数
        models = self.config['selected_models']
        model_name = models[0] if models else 'Random Forest'
        model_params = {
            'task_type': self.config['task_type'],
            'model_name': model_name
        }
        
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_batch = {}
                
                for batch in batches:
                    future = executor.submit(
                        self._evaluate_combination_batch,
                        batch, model_params
                    )
                    future_to_batch[future] = batch
                
                # 收集结果
                batch_results = []
                for future in as_completed(future_to_batch):
                    batch_result = future.result()
                    batch_results.extend(batch_result)
                
                # 按原始顺序排序结果
                for features, score in batch_results:
                    scores.append(score)
                    
        except Exception as e:
            print(f"Parallel population evaluation failed: {e}")
            # 回退到单线程
            for individual in population:
                selected_features = [self.current_features[i] for i, selected in enumerate(individual) if selected]
                if len(selected_features) > 0:
                    score = self.evaluate_feature_subset(selected_features)
                else:
                    score = -np.inf
                scores.append(score)
        
        return scores

class FeatureModule(QWidget):
    """Advanced feature selection and model configuration module"""
    
    # Signals
    features_ready = pyqtSignal(pd.DataFrame, pd.Series, dict)  # X_selected, y, config
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    selection_started = pyqtSignal()  # Feature selection start signal
    
    def __init__(self):
        super().__init__()
        self.X = None
        self.y = None
        self.task_type = None
        self.selected_features = []
        self.selection_results = {}
        self.worker = None  # Ensure worker is properly initialized
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel for controls (with scroll area)
        left_scroll = QScrollArea()
        left_panel = QWidget()
        left_panel.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_panel)
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Right panel for visualization
        right_panel = QTabWidget()
        
        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([450, 1200])
        
        # === LEFT PANEL ===
        
        # 1. Data Information
        self.create_data_info_section(left_layout)
        
        # 2. Task Type & Model Selection
        self.create_task_model_section(left_layout)
        
        # 3. Training Configuration
        self.create_training_config_section(left_layout)
        
        # 3.5. Performance Configuration
        self.create_performance_config_section(left_layout)
        
        # 4. Feature Selection Strategy
        self.create_feature_selection_section(left_layout)
        
        # 5. Execute & Results
        self.create_execution_section(left_layout)
        
        left_layout.addStretch()
        
        # === RIGHT PANEL ===
        self.right_panel = right_panel
        self.viz_tabs = {}
        
        # Initially disable the module
        self.setEnabled(False)
        
    def create_data_info_section(self, parent_layout):
        """Create data information section"""
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout(info_group)
        
        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setWordWrap(True)
        info_layout.addWidget(self.data_info_label)
        
        parent_layout.addWidget(info_group)
        
    def create_task_model_section(self, parent_layout):
        """Create task type and model selection section"""
        task_group = QGroupBox("Task Type & Model Selection")
        task_layout = QVBoxLayout(task_group)
        
        # Task type selection
        task_type_layout = QHBoxLayout()
        task_type_layout.addWidget(QLabel("Task Type:"))
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["Auto-detect", "Classification", "Regression"])
        self.task_type_combo.currentTextChanged.connect(self.update_available_models)
        task_type_layout.addWidget(self.task_type_combo)
        task_layout.addLayout(task_type_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model for Feature Evaluation:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        model_layout.addWidget(self.model_combo)
        task_layout.addLayout(model_layout)
        
        parent_layout.addWidget(task_group)
        
    def create_training_config_section(self, parent_layout):
        """Create training configuration section"""
        config_group = QGroupBox("Training Configuration")
        config_layout = QGridLayout(config_group)
        
        # Train/test split
        config_layout.addWidget(QLabel("Train/Test Split:"), 0, 0)
        self.train_test_split_spin = QDoubleSpinBox()
        self.train_test_split_spin.setMinimum(0.5)
        self.train_test_split_spin.setMaximum(0.9)
        self.train_test_split_spin.setSingleStep(0.05)
        self.train_test_split_spin.setValue(0.8)
        config_layout.addWidget(self.train_test_split_spin, 0, 1)
        
        # Random seed
        config_layout.addWidget(QLabel("Random Seed:"), 1, 0)
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setMinimum(0)
        self.random_seed_spin.setMaximum(9999)
        self.random_seed_spin.setValue(42)
        config_layout.addWidget(self.random_seed_spin, 1, 1)
        
        # Cross-validation folds
        config_layout.addWidget(QLabel("CV Folds:"), 2, 0)
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setMinimum(3)
        self.cv_folds_spin.setMaximum(10)
        self.cv_folds_spin.setValue(5)
        config_layout.addWidget(self.cv_folds_spin, 2, 1)
        
        # Scoring metric
        config_layout.addWidget(QLabel("Scoring Metric:"), 3, 0)
        self.scoring_combo = QComboBox()
        self.scoring_combo.addItems(["accuracy", "f1_macro", "roc_auc", "r2", "neg_mean_squared_error"])
        config_layout.addWidget(self.scoring_combo, 3, 1)
        
        # Data scaling
        config_layout.addWidget(QLabel("Data Scaling:"), 4, 0)
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "None", 
            "StandardScaler (Z-score)", 
            "MinMaxScaler (0-1)", 
            "RobustScaler", 
            "Normalizer",
            "QuantileTransformer",
            "PowerTransformer"
        ])
        config_layout.addWidget(self.scaling_combo, 4, 1)
        
        parent_layout.addWidget(config_group)
    
    def create_performance_config_section(self, parent_layout):
        """Create performance configuration section"""
        group = QGroupBox("🚀 Performance Configuration")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QGridLayout()
        
        # Enable parallel processing
        self.enable_parallel_cb = QCheckBox("Enable Parallel Processing")
        self.enable_parallel_cb.setChecked(True)
        self.enable_parallel_cb.setToolTip("使用多线程加速特征选择过程")
        layout.addWidget(self.enable_parallel_cb, 0, 0, 1, 2)
        
        # Number of workers
        layout.addWidget(QLabel("Worker Threads:"), 1, 0)
        self.n_workers_spin = QSpinBox()
        self.n_workers_spin.setMinimum(1)
        self.n_workers_spin.setMaximum(multiprocessing.cpu_count())
        self.n_workers_spin.setValue(min(4, multiprocessing.cpu_count()))
        self.n_workers_spin.setToolTip(f"并行处理线程数 (最大: {multiprocessing.cpu_count()})")
        layout.addWidget(self.n_workers_spin, 1, 1)
        
        # Auto-reduce CV folds for speed
        self.auto_reduce_cv_cb = QCheckBox("Auto-reduce CV for large datasets")
        self.auto_reduce_cv_cb.setChecked(True)
        self.auto_reduce_cv_cb.setToolTip("大数据集时自动减少交叉验证折数以提升速度")
        layout.addWidget(self.auto_reduce_cv_cb, 2, 0, 1, 2)
        
        # Cache model results
        self.cache_results_cb = QCheckBox("Cache Evaluation Results")
        self.cache_results_cb.setChecked(True)
        self.cache_results_cb.setToolTip("缓存特征组合评估结果以避免重复计算")
        layout.addWidget(self.cache_results_cb, 3, 0, 1, 2)
        
        # Performance info
        cpu_count = multiprocessing.cpu_count()
        info_label = QLabel(f"📊 System: {cpu_count} CPU cores detected")
        info_label.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(info_label, 4, 0, 1, 2)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
        # Add hyperparameter optimization section
        self.create_hyperopt_section(parent_layout)
    
    def create_hyperopt_section(self, parent_layout):
        """Create hyperparameter optimization section"""
        hyperopt_group = QGroupBox("Hyperparameter Optimization")
        hyperopt_layout = QVBoxLayout(hyperopt_group)
        
        # Enable hyperparameter optimization
        self.enable_hyperopt_cb = QCheckBox("Enable Hyperparameter Optimization")
        self.enable_hyperopt_cb.stateChanged.connect(self.on_hyperopt_toggled)
        hyperopt_layout.addWidget(self.enable_hyperopt_cb)
        
        # Hyperopt controls widget
        self.hyperopt_controls = QWidget()
        hyperopt_controls_layout = QGridLayout(self.hyperopt_controls)
        
        # Search method
        hyperopt_controls_layout.addWidget(QLabel("Search Method:"), 0, 0)
        self.search_method_combo = QComboBox()
        self.search_method_combo.addItems(['Grid Search', 'Random Search', 'Bayesian Search'])
        self.search_method_combo.setCurrentText('Random Search')
        self.search_method_combo.currentTextChanged.connect(self.on_search_method_changed)
        hyperopt_controls_layout.addWidget(self.search_method_combo, 0, 1)
        
        # Number of iterations (for Random/Bayesian)
        hyperopt_controls_layout.addWidget(QLabel("Max Iterations:"), 1, 0)
        self.n_iter_spin = QSpinBox()
        self.n_iter_spin.setRange(10, 1000)
        self.n_iter_spin.setValue(100)
        hyperopt_controls_layout.addWidget(self.n_iter_spin, 1, 1)
        
        # Early stopping section
        early_stopping_group = QGroupBox("Early Stopping (Overfitting Prevention)")
        early_stopping_layout = QGridLayout(early_stopping_group)
        
        # Enable early stopping
        self.enable_early_stopping_cb = QCheckBox("Enable Early Stopping")
        self.enable_early_stopping_cb.setChecked(True)
        self.enable_early_stopping_cb.setToolTip("Stop optimization early if no improvement is observed")
        self.enable_early_stopping_cb.stateChanged.connect(self.on_early_stopping_toggled)
        early_stopping_layout.addWidget(self.enable_early_stopping_cb, 0, 0, 1, 2)
        
        # Early stopping controls
        self.early_stopping_controls = QWidget()
        early_stopping_controls_layout = QGridLayout(self.early_stopping_controls)
        
        # Patience
        early_stopping_controls_layout.addWidget(QLabel("Patience:"), 0, 0)
        self.early_stopping_patience_spin = QSpinBox()
        self.early_stopping_patience_spin.setRange(3, 50)
        self.early_stopping_patience_spin.setValue(10)
        self.early_stopping_patience_spin.setToolTip("Number of iterations without improvement before stopping")
        early_stopping_controls_layout.addWidget(self.early_stopping_patience_spin, 0, 1)
        
        # Minimum improvement threshold
        early_stopping_controls_layout.addWidget(QLabel("Min Improvement:"), 1, 0)
        self.early_stopping_min_improvement_spin = QDoubleSpinBox()
        self.early_stopping_min_improvement_spin.setRange(0.0001, 0.1)
        self.early_stopping_min_improvement_spin.setSingleStep(0.0001)
        self.early_stopping_min_improvement_spin.setValue(0.001)
        self.early_stopping_min_improvement_spin.setDecimals(4)
        self.early_stopping_min_improvement_spin.setToolTip("Minimum score improvement to reset patience counter")
        early_stopping_controls_layout.addWidget(self.early_stopping_min_improvement_spin, 1, 1)
        
        early_stopping_layout.addWidget(self.early_stopping_controls, 1, 0, 1, 2)
        hyperopt_controls_layout.addWidget(early_stopping_group, 2, 0, 1, 2)
        
        # Custom parameter space button
        self.custom_params_btn = QPushButton("Configure Parameter Space")
        self.custom_params_btn.clicked.connect(self.configure_parameter_space)
        hyperopt_controls_layout.addWidget(self.custom_params_btn, 3, 0, 1, 2)
        
        # Initially disable hyperopt controls
        self.hyperopt_controls.setEnabled(False)
        hyperopt_layout.addWidget(self.hyperopt_controls)
        
        # Custom parameter space storage
        self.custom_param_spaces = {}
        
        parent_layout.addWidget(hyperopt_group)
    
    def create_feature_selection_section(self, parent_layout):
        """Create feature selection strategy section"""
        strategy_group = QGroupBox("Feature Selection Strategy")
        strategy_layout = QVBoxLayout(strategy_group)
        
        # Step 1: Model-based importance filtering
        step1_group = QGroupBox("Step 1: Model-based Importance Filtering")
        step1_layout = QVBoxLayout(step1_group)
        
        self.enable_importance_filter = QCheckBox("Enable importance-based filtering")
        self.enable_importance_filter.setChecked(True)
        step1_layout.addWidget(self.enable_importance_filter)
        
        # Importance filter type
        importance_type_layout = QHBoxLayout()
        importance_type_layout.addWidget(QLabel("Filter Type:"))
        self.importance_filter_type = QComboBox()
        self.importance_filter_type.addItems(["Top K features", "Importance threshold", "Cumulative percentage"])
        self.importance_filter_type.currentTextChanged.connect(self.update_importance_controls)
        importance_type_layout.addWidget(self.importance_filter_type)
        step1_layout.addLayout(importance_type_layout)
        
        # Dynamic controls for importance filtering
        self.importance_controls_widget = QWidget()
        self.importance_controls_layout = QHBoxLayout(self.importance_controls_widget)
        self.importance_controls_layout.setContentsMargins(0, 0, 0, 0)
        step1_layout.addWidget(self.importance_controls_widget)
        
        self.update_importance_controls()
        strategy_layout.addWidget(step1_group)
        
        # Step 2: Correlation-based filtering
        step2_group = QGroupBox("Step 2: Correlation-based Redundancy Filtering")
        step2_layout = QVBoxLayout(step2_group)
        
        self.enable_correlation_filter = QCheckBox("Enable correlation-based filtering")
        self.enable_correlation_filter.setChecked(True)
        step2_layout.addWidget(self.enable_correlation_filter)
        
        # Correlation threshold
        corr_threshold_layout = QHBoxLayout()
        corr_threshold_layout.addWidget(QLabel("Correlation Threshold:"))
        self.correlation_threshold_spin = QDoubleSpinBox()
        self.correlation_threshold_spin.setMinimum(0.5)
        self.correlation_threshold_spin.setMaximum(1.0)
        self.correlation_threshold_spin.setSingleStep(0.05)
        self.correlation_threshold_spin.setValue(0.95)
        corr_threshold_layout.addWidget(self.correlation_threshold_spin)
        step2_layout.addLayout(corr_threshold_layout)
        
        # Correlation removal method
        corr_method_layout = QHBoxLayout()
        corr_method_layout.addWidget(QLabel("Removal Method:"))
        self.correlation_method_combo = QComboBox()
        self.correlation_method_combo.addItems(["Model-based performance", "Target correlation"])
        corr_method_layout.addWidget(self.correlation_method_combo)
        step2_layout.addLayout(corr_method_layout)
        
        strategy_layout.addWidget(step2_group)
        
        # Step 3: Advanced subset search (optional)
        step3_group = QGroupBox("Step 3: Advanced Feature Subset Search (Optional)")
        step3_layout = QVBoxLayout(step3_group)
        
        self.enable_advanced_search = QCheckBox("Enable advanced subset search")
        self.enable_advanced_search.setChecked(False)
        self.enable_advanced_search.toggled.connect(self.toggle_advanced_search)
        step3_layout.addWidget(self.enable_advanced_search)
        
        # Advanced search method
        self.advanced_search_widget = QWidget()
        advanced_search_layout = QVBoxLayout(self.advanced_search_widget)
        advanced_search_layout.setContentsMargins(0, 0, 0, 0)
        
        search_method_layout = QHBoxLayout()
        search_method_layout.addWidget(QLabel("Search Method:"))
        self.advanced_search_method = QComboBox()
        self.advanced_search_method.addItems(["Sequential Feature Selection", "Genetic Algorithm", "Exhaustive Search"])
        self.advanced_search_method.currentTextChanged.connect(self.update_advanced_search_controls)
        search_method_layout.addWidget(self.advanced_search_method)
        advanced_search_layout.addLayout(search_method_layout)
        
        # Dynamic controls for advanced search
        self.advanced_controls_widget = QWidget()
        self.advanced_controls_layout = QVBoxLayout(self.advanced_controls_widget)
        self.advanced_controls_layout.setContentsMargins(0, 0, 0, 0)
        advanced_search_layout.addWidget(self.advanced_controls_widget)
        
        step3_layout.addWidget(self.advanced_search_widget)
        self.advanced_search_widget.setEnabled(False)
        
        self.update_advanced_search_controls()
        strategy_layout.addWidget(step3_group)
        
        parent_layout.addWidget(strategy_group)
        
    def create_execution_section(self, parent_layout):
        """Create execution and results section"""
        # Execute button
        self.execute_btn = QPushButton("Execute Feature Selection")
        self.execute_btn.clicked.connect(self.execute_feature_selection)
        self.execute_btn.setEnabled(False)
        self.execute_btn.setMinimumHeight(40)
        self.execute_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        parent_layout.addWidget(self.execute_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        parent_layout.addWidget(self.progress_bar)
        
        # Results section
        results_group = QGroupBox("Feature Selection Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        # Save/Load configuration
        config_btn_layout = QHBoxLayout()
        self.save_config_btn = QPushButton("Save Configuration")
        self.save_config_btn.clicked.connect(self.save_configuration)
        self.load_config_btn = QPushButton("Load Configuration")
        self.load_config_btn.clicked.connect(self.load_configuration)
        config_btn_layout.addWidget(self.save_config_btn)
        config_btn_layout.addWidget(self.load_config_btn)
        results_layout.addLayout(config_btn_layout)
        
        # Proceed button
        self.proceed_btn = QPushButton("Proceed to Model Training")
        self.proceed_btn.clicked.connect(self.proceed_to_next_module)
        self.proceed_btn.setEnabled(False)
        self.proceed_btn.setMinimumHeight(35)
        self.proceed_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        results_layout.addWidget(self.proceed_btn)
        
        parent_layout.addWidget(results_group)
        
    def update_available_models(self):
        """Update available models based on task type"""
        task_type_text = self.task_type_combo.currentText()
        
        print(f"DEBUG: update_available_models called")
        print(f"DEBUG: task_type_combo text: '{task_type_text}'")
        print(f"DEBUG: self.task_type: '{getattr(self, 'task_type', 'NOT_SET')}'")
        
        self.model_combo.clear()
        
        # Determine task type
        if task_type_text == "Classification":
            task_type = "classification"
        elif task_type_text == "Regression":
            task_type = "regression"
        elif task_type_text == "Auto-detect" and self.task_type:
            task_type = self.task_type
        else:
            task_type = "classification"  # Default
        
        print(f"DEBUG: Determined task_type: '{task_type}'")
        
        # CRITICAL FIX: If user manually selected a task type (not Auto-detect),
        # override the internal task_type to ensure consistency throughout the module
        if task_type_text != "Auto-detect":
            old_task_type = getattr(self, 'task_type', None)
            self.task_type = task_type
            print(f"DEBUG: Task type MANUALLY OVERRIDDEN from '{old_task_type}' to '{self.task_type}'")
            
            # Also update the scoring metric to match the new task type
            if self.task_type == 'classification':
                self.scoring_combo.setCurrentText('accuracy')
            else:
                self.scoring_combo.setCurrentText('r2')
            print(f"DEBUG: Updated scoring metric for task type '{self.task_type}'")
        
        # Get available models for this task type
        try:
            from utils.ml_utils import get_available_models
            models = get_available_models(task_type)
            
            # Add models to combo box
            for model_name in models.keys():
                self.model_combo.addItem(model_name)
                
            print(f"DEBUG: Added {len(models)} models for task type '{task_type}'")
            
        except Exception as e:
            print(f"ERROR: Failed to get models for task type '{task_type}': {e}")
            # Add some default models as fallback
            if task_type == "classification":
                self.model_combo.addItems(["Random Forest", "Logistic Regression"])
            else:
                self.model_combo.addItems(["Random Forest", "Linear Regression"])
    
    def update_importance_controls(self):
        """Update importance filtering controls based on selected type"""
        # Clear existing controls
        for i in reversed(range(self.importance_controls_layout.count())):
            self.importance_controls_layout.itemAt(i).widget().setParent(None)
        
        filter_type = self.importance_filter_type.currentText()
        
        if filter_type == "Top K features":
            self.importance_controls_layout.addWidget(QLabel("Number of features:"))
            self.importance_top_k_spin = QSpinBox()
            self.importance_top_k_spin.setMinimum(1)
            self.importance_top_k_spin.setMaximum(1000)
            self.importance_top_k_spin.setValue(50)
            self.importance_controls_layout.addWidget(self.importance_top_k_spin)
            
        elif filter_type == "Importance threshold":
            self.importance_controls_layout.addWidget(QLabel("Threshold:"))
            self.importance_threshold_spin = QDoubleSpinBox()
            self.importance_threshold_spin.setMinimum(0.0)
            self.importance_threshold_spin.setMaximum(1.0)
            self.importance_threshold_spin.setSingleStep(0.01)
            self.importance_threshold_spin.setValue(0.01)
            self.importance_threshold_spin.setDecimals(3)
            self.importance_controls_layout.addWidget(self.importance_threshold_spin)
            
        else:  # Cumulative percentage
            self.importance_controls_layout.addWidget(QLabel("Cumulative %:"))
            self.importance_cumulative_spin = QDoubleSpinBox()
            self.importance_cumulative_spin.setMinimum(0.5)
            self.importance_cumulative_spin.setMaximum(1.0)
            self.importance_cumulative_spin.setSingleStep(0.05)
            self.importance_cumulative_spin.setValue(0.95)
            self.importance_controls_layout.addWidget(self.importance_cumulative_spin)
    
    def toggle_advanced_search(self, enabled):
        """Toggle advanced search options"""
        self.advanced_search_widget.setEnabled(enabled)
    
    def update_advanced_search_controls(self):
        """Update advanced search controls based on selected method"""
        # Clear existing controls
        for i in reversed(range(self.advanced_controls_layout.count())):
            self.advanced_controls_layout.itemAt(i).widget().setParent(None)
        
        method = self.advanced_search_method.currentText()
        
        if method == "Sequential Feature Selection":
            # Direction
            direction_layout = QHBoxLayout()
            direction_layout.addWidget(QLabel("Direction:"))
            self.sfs_direction_combo = QComboBox()
            self.sfs_direction_combo.addItems(["Forward", "Backward"])
            self.sfs_direction_combo.setToolTip("Forward: Start from empty set and add features\nBackward: Start from full set and remove features")
            direction_layout.addWidget(self.sfs_direction_combo)
            
            direction_widget = QWidget()
            direction_widget.setLayout(direction_layout)
            self.advanced_controls_layout.addWidget(direction_widget)
            
            # Number of features
            k_layout = QHBoxLayout()
            k_layout.addWidget(QLabel("Target features:"))
            self.sfs_k_features_spin = QSpinBox()
            self.sfs_k_features_spin.setMinimum(1)
            self.sfs_k_features_spin.setMaximum(100)
            self.sfs_k_features_spin.setValue(10)
            self.sfs_k_features_spin.setToolTip("Final number of features to select")
            k_layout.addWidget(self.sfs_k_features_spin)
            
            k_widget = QWidget()
            k_widget.setLayout(k_layout)
            self.advanced_controls_layout.addWidget(k_widget)
            
        elif method == "Genetic Algorithm":
            # Population size
            pop_layout = QHBoxLayout()
            pop_layout.addWidget(QLabel("Population size:"))
            self.ga_population_spin = QSpinBox()
            self.ga_population_spin.setMinimum(10)
            self.ga_population_spin.setMaximum(200)
            self.ga_population_spin.setValue(50)
            self.ga_population_spin.setToolTip("Number of individuals in population, affects search diversity")
            pop_layout.addWidget(self.ga_population_spin)
            
            pop_widget = QWidget()
            pop_widget.setLayout(pop_layout)
            self.advanced_controls_layout.addWidget(pop_widget)
            
            # Generations
            gen_layout = QHBoxLayout()
            gen_layout.addWidget(QLabel("Generations:"))
            self.ga_generations_spin = QSpinBox()
            self.ga_generations_spin.setMinimum(5)
            self.ga_generations_spin.setMaximum(500)
            self.ga_generations_spin.setValue(20)
            self.ga_generations_spin.setToolTip("Number of generations to evolve, more generations may find better solutions")
            gen_layout.addWidget(self.ga_generations_spin)
            
            gen_widget = QWidget()
            gen_widget.setLayout(gen_layout)
            self.advanced_controls_layout.addWidget(gen_widget)
            
            # Mutation rate
            mut_layout = QHBoxLayout()
            mut_layout.addWidget(QLabel("Mutation rate:"))
            self.ga_mutation_spin = QDoubleSpinBox()
            self.ga_mutation_spin.setMinimum(0.01)
            self.ga_mutation_spin.setMaximum(0.5)
            self.ga_mutation_spin.setSingleStep(0.01)
            self.ga_mutation_spin.setValue(0.1)
            self.ga_mutation_spin.setDecimals(2)
            self.ga_mutation_spin.setToolTip("Probability of gene mutation, adds randomness to search")
            mut_layout.addWidget(self.ga_mutation_spin)
            
            mut_widget = QWidget()
            mut_widget.setLayout(mut_layout)
            self.advanced_controls_layout.addWidget(mut_widget)
            
            # Crossover rate
            cross_layout = QHBoxLayout()
            cross_layout.addWidget(QLabel("Crossover rate:"))
            self.ga_crossover_spin = QDoubleSpinBox()
            self.ga_crossover_spin.setMinimum(0.5)
            self.ga_crossover_spin.setMaximum(1.0)
            self.ga_crossover_spin.setSingleStep(0.05)
            self.ga_crossover_spin.setValue(0.8)
            self.ga_crossover_spin.setDecimals(2)
            self.ga_crossover_spin.setToolTip("Probability of parent gene crossover, affects inheritance")
            cross_layout.addWidget(self.ga_crossover_spin)
            
            cross_widget = QWidget()
            cross_widget.setLayout(cross_layout)
            self.advanced_controls_layout.addWidget(cross_widget)
            
            # Tournament size
            tour_layout = QHBoxLayout()
            tour_layout.addWidget(QLabel("Tournament size:"))
            self.ga_tournament_spin = QSpinBox()
            self.ga_tournament_spin.setMinimum(2)
            self.ga_tournament_spin.setMaximum(10)
            self.ga_tournament_spin.setValue(3)
            self.ga_tournament_spin.setToolTip("Number of individuals competing in tournament selection")
            tour_layout.addWidget(self.ga_tournament_spin)
            
            tour_widget = QWidget()
            tour_widget.setLayout(tour_layout)
            self.advanced_controls_layout.addWidget(tour_widget)
            
            # Elite ratio
            elite_layout = QHBoxLayout()
            elite_layout.addWidget(QLabel("Elite ratio:"))
            self.ga_elite_spin = QDoubleSpinBox()
            self.ga_elite_spin.setMinimum(0.0)
            self.ga_elite_spin.setMaximum(0.3)
            self.ga_elite_spin.setSingleStep(0.05)
            self.ga_elite_spin.setValue(0.1)
            self.ga_elite_spin.setDecimals(2)
            self.ga_elite_spin.setToolTip("Proportion of best individuals preserved to next generation")
            elite_layout.addWidget(self.ga_elite_spin)
            
            elite_widget = QWidget()
            elite_widget.setLayout(elite_layout)
            self.advanced_controls_layout.addWidget(elite_widget)
            
        else:  # Exhaustive Search
            # Min features
            min_layout = QHBoxLayout()
            min_layout.addWidget(QLabel("Min features:"))
            self.exhaustive_min_spin = QSpinBox()
            self.exhaustive_min_spin.setMinimum(1)
            self.exhaustive_min_spin.setMaximum(15)
            self.exhaustive_min_spin.setValue(1)
            self.exhaustive_min_spin.setToolTip("Minimum number of features to search")
            min_layout.addWidget(self.exhaustive_min_spin)
            
            min_widget = QWidget()
            min_widget.setLayout(min_layout)
            self.advanced_controls_layout.addWidget(min_widget)
            
            # Max features
            max_layout = QHBoxLayout()
            max_layout.addWidget(QLabel("Max features:"))
            self.exhaustive_max_spin = QSpinBox()
            self.exhaustive_max_spin.setMinimum(1)
            self.exhaustive_max_spin.setMaximum(15)
            self.exhaustive_max_spin.setValue(10)
            self.exhaustive_max_spin.setToolTip("Maximum number of features to search")
            max_layout.addWidget(self.exhaustive_max_spin)
            
            max_widget = QWidget()
            max_widget.setLayout(max_layout)
            self.advanced_controls_layout.addWidget(max_widget)
            
            # Warning label
            warning_label = QLabel("⚠️ Only feasible for ≤15 features")
            warning_label.setStyleSheet("color: orange; font-weight: bold;")
            self.advanced_controls_layout.addWidget(warning_label)
    
    def save_configuration(self):
        """Save current configuration to file"""
        # Implementation for saving configuration
        QMessageBox.information(self, "Info", "Configuration save functionality will be implemented.")
    
    def load_configuration(self):
        """Load configuration from file"""
        # Implementation for loading configuration
        QMessageBox.information(self, "Info", "Configuration load functionality will be implemented.")

    def set_data(self, X: pd.DataFrame, y: pd.Series):
        """Set data from previous module with comprehensive validation and One-Hot feature grouping"""
        print(f"=== FEATURE MODULE DATA RECEPTION ===")
        print(f"Received X shape: {X.shape}")
        print(f"Received y shape: {y.shape}")
        print(f"X columns: {list(X.columns)}")
        print(f"y name: {y.name}")
        print(f"y dtype: {y.dtype}")
        print(f"y unique values: {y.unique()}")
        
        # Check for missing values in received data
        x_missing = X.isnull().sum().sum()
        y_missing = y.isnull().sum()
        
        if x_missing > 0 or y_missing > 0:
            print(f"WARNING: Received data contains missing values - X: {x_missing}, y: {y_missing}")
        else:
            print("✓ Received data is clean (no missing values)")
        
        # Verify index alignment
        if not X.index.equals(y.index):
            print("ERROR: X and y indices are not aligned!")
            print(f"X index range: {X.index.min()} - {X.index.max()}")
            print(f"y index range: {y.index.min()} - {y.index.max()}")
            # Force realignment
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            print("Forced index reset to align X and y")
        else:
            print("✓ X and y indices are properly aligned")
        
        # Store the data copies
        self.X = X.copy()
        self.y = y.copy()
        
        # CRITICAL ENHANCEMENT: Detect and group One-Hot encoded features
        self.one_hot_groups = self._detect_one_hot_feature_groups(X.columns.tolist())
        if self.one_hot_groups:
            print(f"=== ONE-HOT FEATURE GROUPS DETECTED ===")
            for group_name, features in self.one_hot_groups.items():
                print(f"Group '{group_name}': {features}")
            print("These groups will be treated as units during feature selection")
            print("=" * 50)
        
        # CRITICAL: Create unified dataframe immediately and ensure consistency
        self.y_name = y.name if y.name else 'target'
        combined_data = pd.concat([X, y.rename(self.y_name)], axis=1)
        
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Combined data columns: {list(combined_data.columns)}")
        
        # Store this as the single source of truth
        self.data = combined_data.copy()
        
        # Verify the combined data integrity
        combined_missing = self.data.isnull().sum().sum()
        if combined_missing > 0:
            print(f"WARNING: Combined data has missing values: {combined_missing}")
        
        # Initialize current features to all features (excluding target)
        self.current_features = [col for col in self.data.columns if col != self.y_name]
        
        # Detect task type from y values using improved logic
        from utils.data_utils import suggest_task_type
        self.task_type = suggest_task_type(y)
            
        print(f"Detected task type: {self.task_type}")
        print(f"Initial feature count: {len(self.current_features)}")
        print(f"✓ Data successfully loaded into FeatureModule")
        print("=" * 50)
        
        # Update UI
        self.update_data_info()
        
        # CRITICAL FIX: Ensure task type combo reflects detected type when in Auto-detect mode
        if self.task_type_combo.currentText() == "Auto-detect":
            # Update displayed type and scoring metric based on detected task type
            if self.task_type == 'classification':
                self.scoring_combo.setCurrentText('accuracy')
            else:
                self.scoring_combo.setCurrentText('r2')
        
        # Update available models based on detected task type
        self.update_available_models()
        
        # Enable execution
        self.execute_btn.setEnabled(True)
    
    def _detect_one_hot_feature_groups(self, feature_names):
        """
        Detect One-Hot encoded feature groups based on naming patterns and data characteristics
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dict mapping group names to lists of feature names
        """
        groups = {}
        
        for feature_name in feature_names:
            if '_' in feature_name:
                # Split by last underscore to get potential base name and value
                parts = feature_name.rsplit('_', 1)
                if len(parts) == 2:
                    base_name, value_part = parts
                    
                    # Check if this looks like a One-Hot encoded feature
                    # Criteria: value part is alphabetic or short non-numeric
                    if (value_part.isalpha() or 
                        (len(value_part) <= 3 and not value_part.isdigit())):
                        
                        if base_name not in groups:
                            groups[base_name] = []
                        groups[base_name].append(feature_name)
        
        # Only keep groups with multiple features and validate they are actually One-Hot
        validated_groups = {}
        for group_name, group_features in groups.items():
            if len(group_features) > 1:
                # Additional validation: check if features are binary and mutually exclusive
                if self._validate_one_hot_group(group_features):
                    validated_groups[group_name] = group_features
        
        return validated_groups
    
    def _validate_one_hot_group(self, group_features):
        """
        Validate that a group of features forms a valid One-Hot encoding
        
        Args:
            group_features: List of feature names in the group
            
        Returns:
            bool: True if the group is a valid One-Hot encoding
        """
        if not hasattr(self, 'X') or self.X is None:
            return True  # Cannot validate without data, assume true
        
        try:
            # Check if all features exist in the data
            if not all(feature in self.X.columns for feature in group_features):
                return False
            
            # Check if all features are binary (0/1 values only)
            for feature in group_features:
                unique_vals = set(self.X[feature].dropna().unique())
                if not unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                    return False
            
            # Check mutual exclusivity (sum should be <= 1 for each row)
            group_data = self.X[group_features]
            row_sums = group_data.sum(axis=1)
            if (row_sums > 1.1).any():  # Allow small floating point errors
                return False
            
            # Check if the group actually represents different categories
            # (not just correlated binary features)
            if (row_sums == 0).all():  # All zeros - not a valid One-Hot
                return False
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not validate One-Hot group {group_features}: {e}")
            return False
    
    def _aggregate_one_hot_importances(self, importance_df):
        """
        Aggregate importance scores for One-Hot encoded feature groups
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            
        Returns:
            Dict mapping group names to (aggregated_importance, group_features)
        """
        aggregated_importances = {}
        
        if not hasattr(self, 'one_hot_groups') or not self.one_hot_groups:
            return aggregated_importances
        
        for group_name, group_features in self.one_hot_groups.items():
            # Find features in this group that are in the current feature set
            current_group_features = [f for f in group_features if f in importance_df['feature'].values]
            
            if current_group_features:
                # Sum the importance scores for this group
                group_importance = 0.0
                for feature in current_group_features:
                    feature_importance = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
                    group_importance += feature_importance
                
                aggregated_importances[group_name] = (group_importance, current_group_features)
        
        return aggregated_importances
    
    def update_data_info(self):
        """Update data information display"""
        if self.X is not None and self.y is not None:
            info_text = f"Samples: {self.X.shape[0]}\n"
            info_text += f"Features: {self.X.shape[1]}\n"
            info_text += f"Task Type: {self.task_type.title()}\n"
            
            if self.task_type == 'classification':
                info_text += f"Target Classes: {self.y.nunique()}\n"
                info_text += f"Class Distribution: {dict(self.y.value_counts().head(3))}"
            else:
                info_text += f"Target Range: {self.y.min():.2f} - {self.y.max():.2f}\n"
                info_text += f"Target Mean: {self.y.mean():.2f} ± {self.y.std():.2f}"
            
            self.data_info_label.setText(info_text)
            
    def execute_feature_selection(self):
        """Execute the feature selection pipeline"""
        if self.X is None or self.y is None:
            QMessageBox.warning(self, "Warning", "No data available for feature selection.")
            return
        
        # Validate model selection
        selected_model = self.model_combo.currentText()
        if not selected_model:
            QMessageBox.warning(self, "Warning", "Please select a model for evaluation.")
            return
            
        try:
            # Prepare configuration
            config = self.get_selection_config()
            config['selected_models'] = [selected_model]
            
            # Emit selection started signal
            self.selection_started.emit()
            
            # Show progress
            self.progress_bar.setVisible(True)
            self.execute_btn.setEnabled(False)
            
            # Start worker thread
            self.worker = FeatureSelectionWorker(self.X, self.y, config)
            self.worker.progress_updated.connect(self.progress_bar.setValue)
            self.worker.status_updated.connect(self.status_updated.emit)
            self.worker.step_completed.connect(self.on_step_completed)
            self.worker.error_occurred.connect(self.on_error_occurred)
            self.worker.finished.connect(self.on_selection_completed)
            
            self.worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error starting feature selection: {str(e)}")
            self.progress_bar.setVisible(False)
            self.execute_btn.setEnabled(True)
    
    def get_selection_config(self):
        """Get current selection configuration"""
        config = {
            'task_type': self.task_type,
            'train_test_split': self.train_test_split_spin.value(),
            'random_seed': self.random_seed_spin.value(),
            'cv_folds': self.cv_folds_spin.value(),
            'scoring_metric': self.scoring_combo.currentText(),
            'scaling_method': self.scaling_combo.currentText(),
            
            # Hyperparameter optimization
            'enable_hyperopt': self.enable_hyperopt_cb.isChecked(),
            'search_method': self.search_method_combo.currentText(),
            'n_iter': self.n_iter_spin.value(),
            'custom_param_spaces': self.custom_param_spaces,
            
            # Early stopping for hyperparameter optimization
            'enable_early_stopping': self.enable_early_stopping_cb.isChecked(),
            'early_stopping_patience': self.early_stopping_patience_spin.value(),
            'early_stopping_min_improvement': self.early_stopping_min_improvement_spin.value(),
            
            # Performance configuration
            'use_parallel': self.enable_parallel_cb.isChecked(),
            'n_workers': self.n_workers_spin.value(),
            'auto_reduce_cv': self.auto_reduce_cv_cb.isChecked(),
            'cache_results': self.cache_results_cb.isChecked(),
            
            # Step 1: Importance filtering
            'enable_importance_filter': self.enable_importance_filter.isChecked(),
            'importance_filter_type': self.importance_filter_type.currentText().lower().replace(' ', '_'),
            
            # Step 2: Correlation filtering
            'enable_correlation_filter': self.enable_correlation_filter.isChecked(),
            'correlation_threshold': self.correlation_threshold_spin.value(),
            'correlation_removal_method': 'model_based' if self.correlation_method_combo.currentText() == 'Model-based performance' else 'target_correlation',
            
            # Step 3: Advanced search
            'enable_advanced_search': self.enable_advanced_search.isChecked(),
            'advanced_search_method': self.advanced_search_method.currentText().lower().replace(' ', '_')
        }
        
        # Add importance filter specific config
        if hasattr(self, 'importance_top_k_spin'):
            config['importance_top_k'] = self.importance_top_k_spin.value()
        if hasattr(self, 'importance_threshold_spin'):
            config['importance_threshold'] = self.importance_threshold_spin.value()
        if hasattr(self, 'importance_cumulative_spin'):
            config['importance_cumulative'] = self.importance_cumulative_spin.value()
        
        # Add advanced search specific config
        if config['enable_advanced_search']:
            method = self.advanced_search_method.currentText()
            if method == "Sequential Feature Selection" and hasattr(self, 'sfs_direction_combo'):
                config['sfs_direction'] = self.sfs_direction_combo.currentText().lower()
                config['sfs_k_features'] = self.sfs_k_features_spin.value()
            elif method == "Genetic Algorithm" and hasattr(self, 'ga_population_spin'):
                config['ga_population_size'] = self.ga_population_spin.value()
                config['ga_generations'] = self.ga_generations_spin.value()
                config['ga_mutation_rate'] = self.ga_mutation_spin.value() if hasattr(self, 'ga_mutation_spin') else 0.1
                config['ga_crossover_rate'] = self.ga_crossover_spin.value() if hasattr(self, 'ga_crossover_spin') else 0.8
                config['ga_tournament_size'] = self.ga_tournament_spin.value() if hasattr(self, 'ga_tournament_spin') else 3
                config['ga_elite_ratio'] = self.ga_elite_spin.value() if hasattr(self, 'ga_elite_spin') else 0.1
            elif method == "Exhaustive Search" and hasattr(self, 'exhaustive_min_spin'):
                config['exhaustive_min_features'] = self.exhaustive_min_spin.value()
                config['exhaustive_max_features'] = self.exhaustive_max_spin.value()
        
        return config
    
    def on_step_completed(self, step_name, results):
        """Handle completion of a selection step"""
        self.selection_results[step_name] = results
        self.update_results_display()
        
        # Create/update visualizations
        if step_name == "hyperparameter_optimization":
            viz_widget = self.create_hyperopt_visualization(results)
            if viz_widget:
                self.add_viz_tab(viz_widget, "Hyperparameter Optimization")
        elif step_name == "correlation_filtering":
            # Create separate tabs for before and after correlation
            before_widget = self.create_correlation_before_visualization(results)
            after_widget = self.create_correlation_after_visualization(results)
            if before_widget:
                self.add_viz_tab(before_widget, "Correlation Before")
            if after_widget:
                self.add_viz_tab(after_widget, "Correlation After")
        else:
            self.create_step_visualization(step_name, results)
    
    def on_error_occurred(self, error_message):
        """Handle error in feature selection"""
        QMessageBox.critical(self, "Feature Selection Error", error_message)
        self.progress_bar.setVisible(False)
        self.execute_btn.setEnabled(True)
    
    def on_selection_completed(self):
        """Handle completion of entire feature selection process"""
        self.progress_bar.setVisible(False)
        self.execute_btn.setEnabled(True)
        
        if self.worker and hasattr(self.worker, 'current_features'):
            self.selected_features = self.worker.current_features
            self.proceed_btn.setEnabled(True)
            
            # Final results display
            self.update_final_results()
            
            QMessageBox.information(
                self, "Feature Selection Complete", 
                f"Feature selection completed successfully!\n\n"
                f"Selected {len(self.selected_features)} features from {self.X.shape[1]} original features."
            )
    
    def update_results_display(self):
        """Update results text display"""
        results_text = "Feature Selection Progress:\n\n"
        
        for step_name, results in self.selection_results.items():
            if step_name == 'hyperparameter_optimization':
                results_text += f"✓ Step 0: Hyperparameter Optimization\n"
                if 'error' in results:
                    results_text += f"  Status: Failed - {results.get('message', 'Unknown error')}\n\n"
                else:
                    results_text += f"  Model: {results.get('model_name', 'Unknown')}\n"
                    results_text += f"  Method: {results.get('search_method', 'Unknown')}\n"
                    results_text += f"  Best Score: {results.get('best_score', 0):.4f}\n"
                    if results.get('best_params'):
                        results_text += f"  Best Parameters: {len(results['best_params'])} optimized\n\n"
                    else:
                        results_text += f"  No parameters optimized\n\n"
                        
            elif step_name == 'importance_filtering':
                results_text += f"✓ Step 1: Importance Filtering\n"
                results_text += f"  Features: {results['n_features_before']} → {results['n_features_after']}\n"
                results_text += f"  Removed: {results['n_features_before'] - results['n_features_after']} features\n\n"
                
            elif step_name == 'correlation_filtering':
                results_text += f"✓ Step 2: Correlation Filtering\n"
                results_text += f"  Features: {results['n_features_before']} → {results['n_features_after']}\n"
                results_text += f"  Highly correlated pairs: {len(results['highly_corr_pairs'])}\n"
                results_text += f"  Removed: {len(results['features_removed'])} features\n\n"
                
            elif step_name == 'advanced_search':
                results_text += f"✓ Step 3: Advanced Search ({results['method']})\n"
                results_text += f"  Final features: {results['n_features_after']}\n"
                if 'best_score' in results:
                    results_text += f"  Best CV score: {results['best_score']:.4f}\n\n"
        
        self.results_text.setText(results_text)
    
    def update_final_results(self):
        """Update final results display"""
        if not self.selected_features:
            return
            
        results_text = self.results_text.toPlainText()
        results_text += "\n" + "="*50 + "\n"
        results_text += "FINAL RESULTS:\n\n"
        results_text += f"Original features: {self.X.shape[1]}\n"
        results_text += f"Selected features: {len(self.selected_features)}\n"
        results_text += f"Reduction: {(1 - len(self.selected_features)/self.X.shape[1])*100:.1f}%\n\n"
        results_text += "Selected Features:\n"
        
        for i, feature in enumerate(self.selected_features[:10], 1):
            results_text += f"  {i}. {feature}\n"
        
        if len(self.selected_features) > 10:
            results_text += f"  ... and {len(self.selected_features) - 10} more\n"
        
        self.results_text.setText(results_text)
    
    def create_step_visualization(self, step_name, results):
        """Create visualization for a completed step"""
        if step_name == 'importance_filtering' and 'importance_scores' in results:
            self.create_importance_visualization(results['importance_scores'])
        elif step_name == 'correlation_filtering':
            self.create_correlation_comparison_visualization(results)
        elif step_name == 'hyperparameter_optimization':
            self.create_hyperopt_visualization(results)
        elif step_name == 'advanced_search':
            self.create_advanced_search_visualization(results)
    
    def create_advanced_search_visualization(self, results):
        """Create visualization for advanced search methods"""
        method = results.get('method', '')
        
        if method == 'Genetic Algorithm':
            self.create_genetic_algorithm_visualization(results)
        elif method == 'Exhaustive Search':
            self.create_exhaustive_search_visualization(results)
        elif method == 'Sequential Feature Selection':
            self.create_sequential_selection_visualization(results)
    
    def create_genetic_algorithm_visualization(self, results):
        """Create genetic algorithm evolution visualization"""
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # Parameters summary
            params = results.get('parameters', {})
            summary_text = QTextEdit()
            summary_text.setMaximumHeight(120)
            summary_text.setReadOnly(True)
            
            summary = f"Genetic Algorithm Parameters:\n"
            summary += f"• Population size: {params.get('population_size', 50)}\n"
            summary += f"• Generations: {params.get('generations', 20)}\n"
            summary += f"• Mutation rate: {params.get('mutation_rate', 0.1):.2f}\n"
            summary += f"• Crossover rate: {params.get('crossover_rate', 0.8):.2f}\n"
            summary += f"• Tournament size: {params.get('tournament_size', 3)}\n"
            summary += f"• Elite ratio: {params.get('elite_ratio', 0.1):.2f}\n"
            summary += f"• Final selected features: {results.get('n_features_after', 0)}\n"
            summary += f"• Best score: {results.get('best_score', 0):.4f}"
            
            summary_text.setText(summary)
            layout.addWidget(summary_text)
            
            # Evolution plots
            evolution_history = results.get('evolution_history', [])
            if evolution_history:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                
                fig = Figure(figsize=(15, 10))
                
                # Plot 1: Score evolution
                ax1 = fig.add_subplot(2, 3, 1)
                generations = [h['generation'] for h in evolution_history]
                best_scores = [h['best_score'] for h in evolution_history]
                avg_scores = [h['avg_score'] for h in evolution_history]
                worst_scores = [h['worst_score'] for h in evolution_history]
                
                ax1.plot(generations, best_scores, 'g-', label='Best Score', linewidth=2)
                ax1.plot(generations, avg_scores, 'b-', label='Average Score', linewidth=1)
                ax1.plot(generations, worst_scores, 'r-', label='Worst Score', linewidth=1)
                ax1.set_xlabel('Generation')
                ax1.set_ylabel('Score')
                ax1.set_title('Score Evolution Curve')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Feature count evolution
                ax2 = fig.add_subplot(2, 3, 2)
                avg_features = [h['avg_features'] for h in evolution_history]
                ax2.plot(generations, avg_features, 'purple', linewidth=2)
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Average Features')
                ax2.set_title('Feature Count Evolution')
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Population diversity
                ax3 = fig.add_subplot(2, 3, 3)
                diversity = [h['population_diversity'] for h in evolution_history]
                ax3.plot(generations, diversity, 'orange', linewidth=2)
                ax3.set_xlabel('Generation')
                ax3.set_ylabel('Population Diversity')
                ax3.set_title('Population Diversity Change')
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Score distribution (final generation)
                ax4 = fig.add_subplot(2, 3, 4)
                final_gen = evolution_history[-1]
                ax4.bar(['Best', 'Average', 'Worst'], 
                       [final_gen['best_score'], final_gen['avg_score'], final_gen['worst_score']],
                       color=['green', 'blue', 'red'], alpha=0.7)
                ax4.set_ylabel('Score')
                ax4.set_title('Final Generation Score Distribution')
                ax4.grid(True, alpha=0.3)
                
                # Plot 5: Convergence analysis
                ax5 = fig.add_subplot(2, 3, 5)
                score_improvement = np.diff([0] + best_scores)
                ax5.plot(generations[1:], score_improvement[1:], 'darkgreen', linewidth=2)
                ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax5.set_xlabel('Generation')
                ax5.set_ylabel('Score Improvement')
                ax5.set_title('Convergence Analysis')
                ax5.grid(True, alpha=0.3)
                
                # Plot 6: Selected features visualization
                ax6 = fig.add_subplot(2, 3, 6)
                selected_features = results.get('selected_features', [])
                if len(selected_features) <= 20:
                    ax6.barh(range(len(selected_features)), [1] * len(selected_features), 
                            color='lightblue', alpha=0.7)
                    ax6.set_yticks(range(len(selected_features)))
                    ax6.set_yticklabels(selected_features, fontsize=8)
                    ax6.set_xlabel('Selected Status')
                    ax6.set_title(f'Final Selected Features ({len(selected_features)} features)')
                else:
                    ax6.text(0.5, 0.5, f'Selected {len(selected_features)} features\n(Too many to display)', 
                            ha='center', va='center', transform=ax6.transAxes, fontsize=12)
                    ax6.set_title('Final Selected Features')
                
                fig.tight_layout()
                canvas = FigureCanvas(fig)
                
                # Add navigation toolbar
                toolbar = NavigationToolbar(canvas, widget)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
            
            # Export button
            export_btn = QPushButton("Export Genetic Algorithm Results")
            export_btn.clicked.connect(lambda: self.export_advanced_search_data(results))
            layout.addWidget(export_btn)
            
            self.add_viz_tab(widget, "Genetic Algorithm Visualization")
            
        except Exception as e:
            print(f"Error creating genetic algorithm visualization: {e}")
    
    def create_exhaustive_search_visualization(self, results):
        """Create exhaustive search results visualization"""
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # Parameters summary
            params = results.get('parameters', {})
            summary_text = QTextEdit()
            summary_text.setMaximumHeight(100)
            summary_text.setReadOnly(True)
            
            summary = f"Exhaustive Search Parameters:\n"
            summary += f"• Minimum Features: {params.get('min_features', 1)}\n"
            summary += f"• Maximum Features: {params.get('max_features', 10)}\n"
            summary += f"• Total Combinations: {results.get('total_combinations', 0)}\n"
            summary += f"• Final Selected Features: {results.get('n_features_after', 0)}\n"
            summary += f"• Best Score: {results.get('best_score', 0):.4f}"
            
            summary_text.setText(summary)
            layout.addWidget(summary_text)
            
            # Search results plots
            search_history = results.get('search_history', [])
            if search_history:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                
                fig = Figure(figsize=(15, 8))
                
                # Plot 1: Score vs feature count
                ax1 = fig.add_subplot(2, 3, 1)
                feature_counts = [h['n_features'] for h in search_history]
                scores = [h['score'] for h in search_history]
                
                scatter = ax1.scatter(feature_counts, scores, alpha=0.6, c=scores, cmap='viridis')
                ax1.set_xlabel('Feature Count')
                ax1.set_ylabel('Score')
                ax1.set_title('Score vs Feature Count')
                ax1.grid(True, alpha=0.3)
                fig.colorbar(scatter, ax=ax1, label='Score')
                
                # Plot 2: Best score by feature count
                ax2 = fig.add_subplot(2, 3, 2)
                unique_counts = sorted(set(feature_counts))
                best_scores_by_count = []
                for count in unique_counts:
                    best_score = max([h['score'] for h in search_history if h['n_features'] == count])
                    best_scores_by_count.append(best_score)
                
                ax2.plot(unique_counts, best_scores_by_count, 'bo-', linewidth=2, markersize=6)
                ax2.set_xlabel('Feature Count')
                ax2.set_ylabel('Best Score')
                ax2.set_title('Best Score by Feature Count')
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Score distribution
                ax3 = fig.add_subplot(2, 3, 3)
                ax3.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(results.get('best_score', 0), color='red', linestyle='--', 
                           linewidth=2, label=f'Best Score: {results.get("best_score", 0):.4f}')
                ax3.set_xlabel('Score')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Score Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Feature count distribution
                ax4 = fig.add_subplot(2, 3, 4)
                ax4.hist(feature_counts, bins=range(min(feature_counts), max(feature_counts)+2), 
                        alpha=0.7, color='lightgreen', edgecolor='black')
                ax4.set_xlabel('Feature Count')
                ax4.set_ylabel('Combination Count')
                ax4.set_title('Feature Count Distribution')
                ax4.grid(True, alpha=0.3)
                
                # Plot 5: Top combinations
                ax5 = fig.add_subplot(2, 3, 5)
                top_10 = sorted(search_history, key=lambda x: x['score'], reverse=True)[:10]
                top_scores = [h['score'] for h in top_10]
                top_counts = [h['n_features'] for h in top_10]
                
                bars = ax5.bar(range(len(top_10)), top_scores, 
                              color=['red' if i == 0 else 'lightcoral' for i in range(len(top_10))])
                ax5.set_xlabel('Rank')
                ax5.set_ylabel('Score')
                ax5.set_title('Top 10 Best Combinations')
                ax5.set_xticks(range(len(top_10)))
                ax5.set_xticklabels([f'{i+1}\n({c} features)' for i, c in enumerate(top_counts)], fontsize=8)
                ax5.grid(True, alpha=0.3)
                
                # Plot 6: Selected features
                ax6 = fig.add_subplot(2, 3, 6)
                selected_features = results.get('selected_features', [])
                if len(selected_features) <= 15:
                    ax6.barh(range(len(selected_features)), [1] * len(selected_features), 
                            color='gold', alpha=0.7)
                    ax6.set_yticks(range(len(selected_features)))
                    ax6.set_yticklabels(selected_features, fontsize=8)
                    ax6.set_xlabel('Selected Status')
                    ax6.set_title(f'Final Selected Features ({len(selected_features)} features)')
                else:
                    ax6.text(0.5, 0.5, f'Selected {len(selected_features)} features\n(Too many to display)', 
                            ha='center', va='center', transform=ax6.transAxes, fontsize=12)
                    ax6.set_title('Final Selected Features')
                
                fig.tight_layout()
                canvas = FigureCanvas(fig)
                
                # Add navigation toolbar
                toolbar = NavigationToolbar(canvas, widget)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
            
            # Export button
            export_btn = QPushButton("Export Exhaustive Search Results")
            export_btn.clicked.connect(lambda: self.export_advanced_search_data(results))
            layout.addWidget(export_btn)
            
            self.add_viz_tab(widget, "Exhaustive Search Visualization")
            
        except Exception as e:
            print(f"Error creating exhaustive search visualization: {e}")
    
    def create_sequential_selection_visualization(self, results):
        """Create sequential feature selection visualization"""
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # Summary
            summary_text = QTextEdit()
            summary_text.setMaximumHeight(100)
            summary_text.setReadOnly(True)
            
            summary = f"Sequential Feature Selection Results:\n"
            summary += f"• Direction: {results.get('direction', 'forward')}\n"
            summary += f"• Final Feature Count: {results.get('n_features_after', 0)}\n"
            summary += f"• Best Score: {results.get('best_score', 0):.4f}\n"
            
            selected_features = results.get('selected_features', [])
            if selected_features:
                summary += f"• Selected Features: {', '.join(selected_features[:5])}"
                if len(selected_features) > 5:
                    summary += f" ... and {len(selected_features)} more"
            
            summary_text.setText(summary)
            layout.addWidget(summary_text)
            
            # Simple visualization
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            if len(selected_features) <= 20:
                ax.barh(range(len(selected_features)), [1] * len(selected_features), 
                       color='lightblue', alpha=0.7)
                ax.set_yticks(range(len(selected_features)))
                ax.set_yticklabels(selected_features, fontsize=10)
                ax.set_xlabel('Selected Status')
                ax.set_title(f'Sequential Feature Selection Results ({len(selected_features)} features)')
            else:
                ax.text(0.5, 0.5, f'Selected {len(selected_features)} features\n(Too many to display)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Sequential Feature Selection Results')
            
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            
            # Export button
            export_btn = QPushButton("Export Sequential Selection Results")
            export_btn.clicked.connect(lambda: self.export_advanced_search_data(results))
            layout.addWidget(export_btn)
            
            self.add_viz_tab(widget, "Sequential Selection Visualization")
            
        except Exception as e:
            print(f"Error creating sequential selection visualization: {e}")
    
    def export_advanced_search_data(self, results):
        """Export advanced search results"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Advanced Search Results", f"advanced_search_{results.get('method', 'results').lower().replace(' ', '_')}.xlsx",
                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                method = results.get('method', '')
                
                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        # Basic results
                        basic_results = pd.DataFrame([{
                            'Method': method,
                            'Selected_Features_Count': results.get('n_features_after', 0),
                            'Best_Score': results.get('best_score', 0),
                            'Selected_Features': ', '.join(results.get('selected_features', []))
                        }])
                        basic_results.to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Method-specific data
                        if method == 'Genetic Algorithm' and 'evolution_history' in results:
                            evolution_df = pd.DataFrame(results['evolution_history'])
                            evolution_df.to_excel(writer, sheet_name='Evolution_History', index=False)
                        elif method == 'Exhaustive Search' and 'search_history' in results:
                            search_df = pd.DataFrame(results['search_history'])
                            search_df.to_excel(writer, sheet_name='Search_History', index=False)
                        
                        # Parameters
                        if 'parameters' in results:
                            params_df = pd.DataFrame([results['parameters']])
                            params_df.to_excel(writer, sheet_name='Parameters', index=False)
                else:
                    # CSV export (basic results only)
                    basic_results = pd.DataFrame([{
                        'Method': method,
                        'Selected_Features_Count': results.get('n_features_after', 0),
                        'Best_Score': results.get('best_score', 0),
                        'Selected_Features': ', '.join(results.get('selected_features', []))
                    }])
                    basic_results.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"Advanced search results exported to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export advanced search results: {str(e)}")
    
    def create_importance_visualization(self, importance_df):
        """Create feature importance visualization with readable labels"""
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # Control buttons
            button_layout = QHBoxLayout()
            
            # Export button
            export_btn = QPushButton("Export Feature Importance")
            export_btn.clicked.connect(lambda: self.export_importance_data(importance_df))
            button_layout.addWidget(export_btn)
            
            # Toggle aggregation button
            self.aggregate_features = True  # Default to aggregated view
            toggle_btn = QPushButton("Show Individual Features")
            toggle_btn.clicked.connect(lambda: self.toggle_feature_aggregation(widget, importance_df, toggle_btn))
            button_layout.addWidget(toggle_btn)
            
            layout.addLayout(button_layout)
            
            # Create importance plot with aggregation
            fig, canvas = plot_feature_importance(importance_df, max_features=20, 
                                                aggregate_encoded_features=self.aggregate_features)
            
            # Store canvas for updates
            self.importance_canvas = canvas
            
            # Add navigation toolbar for interactivity
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            toolbar = NavigationToolbar(canvas, widget)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            # Add summary information
            summary_text = QTextEdit()
            summary_text.setMaximumHeight(100)
            summary_text.setReadOnly(True)
            
            summary = f"Feature Importance Summary:\n"
            summary += f"• Total features analyzed: {len(importance_df)}\n"
            summary += f"• Top feature importance: {importance_df['importance'].max():.4f}\n"
            summary += f"• Average importance: {importance_df['importance'].mean():.4f}\n"
            
            # Check if features are likely encoded
            encoded_features = [f for f in importance_df['feature'] if '_' in f and f.split('_')[-1].isdigit()]
            if encoded_features:
                summary += f"• Detected {len(encoded_features)} likely encoded features\n"
                summary += f"• View mode: {'Aggregated by original features' if self.aggregate_features else 'Individual encoded features'}\n"
            
            summary_text.setText(summary)
            layout.addWidget(summary_text)
            
            self.add_viz_tab(widget, "Feature Importance")
            
        except Exception as e:
            print(f"Error creating importance visualization: {e}")
    
    def toggle_feature_aggregation(self, widget, importance_df, toggle_btn):
        """Toggle between aggregated and individual feature views"""
        try:
            self.aggregate_features = not self.aggregate_features
            
            # Update button text
            if self.aggregate_features:
                toggle_btn.setText("Show Individual Features")
            else:
                toggle_btn.setText("Show Aggregated Features")
            
            # Recreate the plot
            fig, new_canvas = plot_feature_importance(importance_df, max_features=20, 
                                                    aggregate_encoded_features=self.aggregate_features)
            
            # Replace the canvas in the layout
            layout = widget.layout()
            
            # Find and remove old canvas and toolbar
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and hasattr(item.widget(), 'figure'):
                    # This is the canvas
                    old_canvas = item.widget()
                    layout.removeWidget(old_canvas)
                    old_canvas.deleteLater()
                elif item and item.widget() and hasattr(item.widget(), 'canvas'):
                    # This is the toolbar
                    old_toolbar = item.widget()
                    layout.removeWidget(old_toolbar)
                    old_toolbar.deleteLater()
            
            # Add new canvas and toolbar
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            new_toolbar = NavigationToolbar(new_canvas, widget)
            layout.insertWidget(-1, new_toolbar)  # Insert before summary text
            layout.insertWidget(-1, new_canvas)   # Insert before summary text
            
            # Update summary text
            summary_widget = None
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and isinstance(item.widget(), QTextEdit):
                    summary_widget = item.widget()
                    break
            
            if summary_widget:
                summary = f"Feature Importance Summary:\n"
                summary += f"• Total features analyzed: {len(importance_df)}\n"
                summary += f"• Top feature importance: {importance_df['importance'].max():.4f}\n"
                summary += f"• Average importance: {importance_df['importance'].mean():.4f}\n"
                
                encoded_features = [f for f in importance_df['feature'] if '_' in f and f.split('_')[-1].isdigit()]
                if encoded_features:
                    summary += f"• Detected {len(encoded_features)} likely encoded features\n"
                    summary += f"• View mode: {'Aggregated by original features' if self.aggregate_features else 'Individual encoded features'}\n"
                
                summary_widget.setText(summary)
            
        except Exception as e:
            print(f"Error toggling feature aggregation: {e}")
    
    def export_importance_data(self, importance_df):
        """Export feature importance data"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Feature Importance", "feature_importance.csv",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
            )
            
            if file_path:
                if file_path.endswith('.xlsx'):
                    importance_df.to_excel(file_path, index=False)
                else:
                    importance_df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"Feature importance exported to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export feature importance: {str(e)}")
    
    def create_correlation_comparison_visualization(self, results):
        """Create correlation before/after comparison visualization"""
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            corr_before = results.get('correlation_before')
            corr_after = results.get('correlation_after')
            
            if corr_before is not None:
                # Create figure with subplots
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                
                if corr_after is not None:
                    fig = Figure(figsize=(16, 8))
                    
                    # Before filtering
                    ax1 = fig.add_subplot(121)
                    im1 = ax1.imshow(corr_before.values, cmap='coolwarm', vmin=0, vmax=1)
                    ax1.set_title(f'Correlations Before Filtering\n({corr_before.shape[0]} features)')
                    
                    # Only show labels if not too many features
                    if corr_before.shape[0] <= 20:
                        ax1.set_xticks(range(len(corr_before.columns)))
                        ax1.set_yticks(range(len(corr_before.index)))
                        ax1.set_xticklabels(corr_before.columns, rotation=45, ha='right', fontsize=8)
                        ax1.set_yticklabels(corr_before.index, fontsize=8)
                    else:
                        ax1.set_xticks([])
                        ax1.set_yticks([])
                    
                    # After filtering
                    ax2 = fig.add_subplot(122)
                    im2 = ax2.imshow(corr_after.values, cmap='coolwarm', vmin=0, vmax=1)
                    ax2.set_title(f'Correlations After Filtering\n({corr_after.shape[0]} features)')
                    
                    # Only show labels if not too many features
                    if corr_after.shape[0] <= 20:
                        ax2.set_xticks(range(len(corr_after.columns)))
                        ax2.set_yticks(range(len(corr_after.index)))
                        ax2.set_xticklabels(corr_after.columns, rotation=45, ha='right', fontsize=8)
                        ax2.set_yticklabels(corr_after.index, fontsize=8)
                    else:
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                    
                    # Add colorbar
                    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, pad=0.02)
                    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
                else:
                    # Only before filtering (if only 1 feature remains)
                    fig = Figure(figsize=(10, 8))
                    ax = fig.add_subplot(111)
                    im = ax.imshow(corr_before.values, cmap='coolwarm', vmin=0, vmax=1)
                    ax.set_title(f'Feature Correlations Before Filtering\n({corr_before.shape[0]} features)')
                    
                    # Only show labels if not too many features
                    if corr_before.shape[0] <= 20:
                        ax.set_xticks(range(len(corr_before.columns)))
                        ax.set_yticks(range(len(corr_before.index)))
                        ax.set_xticklabels(corr_before.columns, rotation=45, ha='right', fontsize=8)
                        ax.set_yticklabels(corr_before.index, fontsize=8)
                    else:
                        ax.set_xticks([])
                        ax.set_yticks([])
                    
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
                
                # Use subplots_adjust instead of tight_layout for better control
                fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
            
            # Add summary text
            summary_text = QTextEdit()
            summary_text.setMaximumHeight(150)
            summary_text.setReadOnly(True)
            
            summary = f"Correlation Filtering Summary:\n"
            summary += f"• Threshold: {results.get('threshold', 0.95)}\n"
            summary += f"• Features before: {results.get('n_features_before', 0)}\n"
            summary += f"• Features after: {results.get('n_features_after', 0)}\n"
            summary += f"• Features removed: {len(results.get('features_removed', []))}\n"
            summary += f"• Highly correlated pairs found: {len(results.get('highly_corr_pairs', []))}\n"
            
            if results.get('features_removed'):
                summary += f"\nRemoved features: {', '.join(results['features_removed'][:5])}"
                if len(results['features_removed']) > 5:
                    summary += f" ... and {len(results['features_removed']) - 5} more"
            
            summary_text.setText(summary)
            layout.addWidget(summary_text)
            
            self.add_viz_tab(widget, "Correlation Filtering")
            
        except Exception as e:
            print(f"Error creating correlation comparison visualization: {e}")
    
    def create_correlation_before_visualization(self, results):
        """Create correlation before filtering visualization with readable labels"""
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            corr_before = results.get('correlation_before')
            if corr_before is not None:
                # Control buttons
                button_layout = QHBoxLayout()
                
                # Export button
                export_btn = QPushButton("Export Correlation Matrix")
                export_btn.clicked.connect(lambda: self.export_correlation_data(results, 'before'))
                button_layout.addWidget(export_btn)
                
                # Toggle readable labels button
                self.use_readable_corr_labels = True  # Default to readable labels
                toggle_labels_btn = QPushButton("Show Original Names")
                toggle_labels_btn.clicked.connect(lambda: self.toggle_correlation_labels(widget, corr_before, toggle_labels_btn))
                button_layout.addWidget(toggle_labels_btn)
                
                layout.addLayout(button_layout)
                
                # Create correlation plot with readable labels
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                
                # Get readable labels
                if self.use_readable_corr_labels:
                    try:
                        from utils.feature_name_utils import create_readable_correlation_labels
                        readable_labels = create_readable_correlation_labels(corr_before.columns.tolist())
                    except ImportError:
                        readable_labels = corr_before.columns.tolist()
                else:
                    readable_labels = corr_before.columns.tolist()
                
                # Adjust figure size based on number of features and label length
                n_features = corr_before.shape[0]
                max_label_length = max(len(label) for label in readable_labels) if readable_labels else 20
                
                figsize = (max(12, max_label_length * 0.2), max(10, max_label_length * 0.2))
                fig = Figure(figsize=figsize)
                ax = fig.add_subplot(111)
                
                # Create heatmap
                import seaborn as sns
                mask = np.triu(np.ones_like(corr_before, dtype=bool))
                
                # Determine settings based on matrix size
                show_annotations = n_features <= 15
                label_fontsize = max(6, min(10, 100 // n_features))
                
                sns.heatmap(corr_before, mask=mask, annot=show_annotations, cmap='coolwarm', 
                           center=0, square=True, linewidths=0.5, ax=ax,
                           xticklabels=readable_labels, yticklabels=readable_labels,
                           fmt='.2f' if show_annotations else '')
                
                # Rotate labels for better readability
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=label_fontsize)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fontsize)
                
                title = f'Feature Correlations Before Filtering\n({n_features} features)'
                if self.use_readable_corr_labels:
                    title += ' - Readable Labels'
                ax.set_title(title, fontsize=14, fontweight='bold')
                
                fig.tight_layout()
                canvas = FigureCanvas(fig)
                
                # Store canvas for updates
                self.correlation_canvas = canvas
                
                # Add navigation toolbar for interactivity
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                toolbar = NavigationToolbar(canvas, widget)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
            
            # Add summary text
            summary_text = QTextEdit()
            summary_text.setMaximumHeight(120)
            summary_text.setReadOnly(True)
            
            summary = f"Before Correlation Filtering:\n"
            summary += f"• Total features: {results.get('n_features_before', 0)}\n"
            summary += f"• Correlation threshold: {results.get('threshold', 0.95)}\n"
            summary += f"• Highly correlated pairs found: {len(results.get('highly_corr_pairs', []))}\n"
            
            # Check for encoded features
            if corr_before is not None:
                encoded_features = [f for f in corr_before.columns if '_' in f and f.split('_')[-1].isdigit()]
                if encoded_features:
                    summary += f"• Detected {len(encoded_features)} likely encoded features\n"
                    summary += f"• Label mode: {'Readable labels' if self.use_readable_corr_labels else 'Original names'}\n"
            
            summary_text.setText(summary)
            layout.addWidget(summary_text)
            
            return widget
            
        except Exception as e:
            print(f"Error creating correlation before visualization: {e}")
            return None
    
    def toggle_correlation_labels(self, widget, corr_matrix, toggle_btn):
        """Toggle between readable and original correlation labels"""
        try:
            self.use_readable_corr_labels = not self.use_readable_corr_labels
            
            # Update button text
            if self.use_readable_corr_labels:
                toggle_btn.setText("Show Original Names")
            else:
                toggle_btn.setText("Show Readable Labels")
            
            # Get labels
            if self.use_readable_corr_labels:
                try:
                    from utils.feature_name_utils import create_readable_correlation_labels
                    readable_labels = create_readable_correlation_labels(corr_matrix.columns.tolist())
                except ImportError:
                    readable_labels = corr_matrix.columns.tolist()
            else:
                readable_labels = corr_matrix.columns.tolist()
            
            # Recreate the plot
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            n_features = corr_matrix.shape[0]
            max_label_length = max(len(label) for label in readable_labels) if readable_labels else 20
            
            figsize = (max(12, max_label_length * 0.2), max(10, max_label_length * 0.2))
            fig = Figure(figsize=figsize)
            ax = fig.add_subplot(111)
            
            # Create heatmap
            import seaborn as sns
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            show_annotations = n_features <= 15
            label_fontsize = max(6, min(10, 100 // n_features))
            
            sns.heatmap(corr_matrix, mask=mask, annot=show_annotations, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, ax=ax,
                       xticklabels=readable_labels, yticklabels=readable_labels,
                       fmt='.2f' if show_annotations else '')
            
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=label_fontsize)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fontsize)
            
            title = f'Feature Correlations Before Filtering\n({n_features} features)'
            if self.use_readable_corr_labels:
                title += ' - Readable Labels'
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            fig.tight_layout()
            new_canvas = FigureCanvas(fig)
            
            # Replace the canvas in the layout
            layout = widget.layout()
            
            # Find and remove old canvas and toolbar
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and hasattr(item.widget(), 'figure'):
                    old_canvas = item.widget()
                    layout.removeWidget(old_canvas)
                    old_canvas.deleteLater()
                elif item and item.widget() and hasattr(item.widget(), 'canvas'):
                    old_toolbar = item.widget()
                    layout.removeWidget(old_toolbar)
                    old_toolbar.deleteLater()
            
            # Add new canvas and toolbar
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            new_toolbar = NavigationToolbar(new_canvas, widget)
            layout.insertWidget(-1, new_toolbar)
            layout.insertWidget(-1, new_canvas)
            
            # Update summary text
            summary_widget = None
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and isinstance(item.widget(), QTextEdit):
                    summary_widget = item.widget()
                    break
            
            if summary_widget:
                summary = summary_widget.toPlainText()
                # Update the label mode line
                lines = summary.split('\n')
                for i, line in enumerate(lines):
                    if 'Label mode:' in line:
                        lines[i] = f"• Label mode: {'Readable labels' if self.use_readable_corr_labels else 'Original names'}"
                        break
                summary_widget.setText('\n'.join(lines))
            
        except Exception as e:
            print(f"Error toggling correlation labels: {e}")
    
    def toggle_correlation_after_labels(self, widget, corr_matrix, toggle_btn):
        """Toggle between readable and original feature labels in correlation after plot"""
        try:
            # Toggle the state
            self.corr_after_use_readable = not self.corr_after_use_readable
            
            # Update button text
            if self.corr_after_use_readable:
                toggle_btn.setText("Show Original Names")
            else:
                toggle_btn.setText("Show Readable Names")
            
            # Get the appropriate labels
            try:
                from utils.plot_utils import get_readable_feature_labels
                readable_labels = get_readable_feature_labels(list(corr_matrix.columns))
                display_labels = readable_labels if self.corr_after_use_readable else list(corr_matrix.columns)
                print(f"Correlation after: Switching to {'readable' if self.corr_after_use_readable else 'original'} labels")
            except Exception as e:
                print(f"Warning: Could not get readable labels: {e}")
                display_labels = list(corr_matrix.columns)
            
            # Find and update the matplotlib figure
            layout = widget.layout()
            canvas = None
            
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget() and hasattr(item.widget(), 'figure'):
                    canvas = item.widget()
                    break
            
            if canvas and hasattr(canvas, 'figure'):
                # Clear and redraw the figure
                fig = canvas.figure
                fig.clear()
                ax = fig.add_subplot(111)
                
                # Recreate the correlation plot with new labels
                im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=0, vmax=1)
                ax.set_title(f'Feature Correlations After Filtering\n({corr_matrix.shape[0]} features)')
                
                # Only show labels if not too many features
                if corr_matrix.shape[0] <= 20:
                    ax.set_xticks(range(len(corr_matrix.columns)))
                    ax.set_yticks(range(len(corr_matrix.index)))
                    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=8)
                    ax.set_yticklabels(display_labels, fontsize=8)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                # Add colorbar
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
                
                fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
                canvas.draw()
                
                print(f"Updated correlation after plot with {'readable' if self.corr_after_use_readable else 'original'} labels")
            
        except Exception as e:
            print(f"Error toggling correlation after labels: {e}")
    
    def create_correlation_after_visualization(self, results):
        """Create correlation after filtering visualization with readable feature names"""
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            corr_after = results.get('correlation_after')
            if corr_after is not None and len(corr_after) > 1:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                
                fig = Figure(figsize=(12, 10))
                ax = fig.add_subplot(111)
                
                # Control panel for label toggle
                control_panel = QWidget()
                control_layout = QHBoxLayout(control_panel)
                
                # Toggle button for readable labels
                self.corr_after_use_readable = True
                toggle_btn = QPushButton("Show Original Names")
                toggle_btn.clicked.connect(lambda: self.toggle_correlation_after_labels(widget, corr_after, toggle_btn))
                control_layout.addWidget(toggle_btn)
                control_layout.addStretch()
                
                layout.addWidget(control_panel)
                
                # Get readable labels
                try:
                    from utils.plot_utils import get_readable_feature_labels
                    readable_labels = get_readable_feature_labels(list(corr_after.columns))
                    display_labels = readable_labels if self.corr_after_use_readable else list(corr_after.columns)
                    print(f"Correlation after: Using readable labels for {len(readable_labels)} features")
                except Exception as e:
                    print(f"Warning: Could not get readable labels for correlation after plot: {e}")
                    display_labels = list(corr_after.columns)
                
                im = ax.imshow(corr_after.values, cmap='coolwarm', vmin=0, vmax=1)
                ax.set_title(f'Feature Correlations After Filtering\n({corr_after.shape[0]} features)')
                
                # Only show labels if not too many features
                if corr_after.shape[0] <= 20:
                    ax.set_xticks(range(len(corr_after.columns)))
                    ax.set_yticks(range(len(corr_after.index)))
                    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=8)
                    ax.set_yticklabels(display_labels, fontsize=8)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
                
                fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
                canvas = FigureCanvas(fig)
                
                # Add navigation toolbar for interactivity
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                toolbar = NavigationToolbar(canvas, widget)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
            else:
                # Show message if only 1 feature remains
                from PyQt5.QtCore import Qt
                label = QLabel("Only 1 feature remains after correlation filtering.\nNo correlation matrix to display.")
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("font-size: 14px; color: #666; padding: 50px;")
                layout.addWidget(label)
            
            # Export button
            export_btn = QPushButton("Export Correlation Results")
            export_btn.clicked.connect(lambda: self.export_correlation_data(results, 'after'))
            layout.addWidget(export_btn)
            
            # Add summary text
            summary_text = QTextEdit()
            summary_text.setMaximumHeight(120)
            summary_text.setReadOnly(True)
            
            summary = f"After Correlation Filtering:\n"
            summary += f"• Remaining features: {results.get('n_features_after', 0)}\n"
            summary += f"• Features removed: {len(results.get('features_removed', []))}\n"
            
            if results.get('features_removed'):
                summary += f"\nRemoved features:\n"
                for i, feature in enumerate(results['features_removed'][:5], 1):
                    summary += f"  {i}. {feature}\n"
                if len(results['features_removed']) > 5:
                    summary += f"  ... and {len(results['features_removed']) - 5} more"
            
            summary_text.setText(summary)
            layout.addWidget(summary_text)
            
            return widget
            
        except Exception as e:
            print(f"Error creating correlation after visualization: {e}")
            return None
    
    def export_correlation_data(self, results, stage):
        """Export correlation filtering data"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, f"Export Correlation Data ({stage.title()})", 
                f"correlation_{stage}.xlsx",
                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        # Correlation matrix
                        if stage == 'before' and 'correlation_before' in results:
                            results['correlation_before'].to_excel(writer, sheet_name='Correlation_Matrix')
                        elif stage == 'after' and 'correlation_after' in results:
                            if results['correlation_after'] is not None:
                                results['correlation_after'].to_excel(writer, sheet_name='Correlation_Matrix')
                        
                        # Highly correlated pairs
                        if 'highly_corr_pairs' in results:
                            pairs_df = pd.DataFrame(results['highly_corr_pairs'], 
                                                  columns=['Feature_1', 'Feature_2', 'Correlation'])
                            pairs_df.to_excel(writer, sheet_name='Correlated_Pairs', index=False)
                        
                        # Summary
                        summary_data = {
                            'Metric': ['Features Before', 'Features After', 'Threshold', 'Pairs Found', 'Features Removed'],
                            'Value': [
                                results.get('n_features_before', 0),
                                results.get('n_features_after', 0),
                                results.get('threshold', 0.95),
                                len(results.get('highly_corr_pairs', [])),
                                len(results.get('features_removed', []))
                            ]
                        }
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                else:
                    # For CSV, export correlation matrix only
                    if stage == 'before' and 'correlation_before' in results:
                        results['correlation_before'].to_csv(file_path)
                    elif stage == 'after' and 'correlation_after' in results:
                        if results['correlation_after'] is not None:
                            results['correlation_after'].to_csv(file_path)
                
                QMessageBox.information(self, "Success", f"Correlation data exported to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export correlation data: {str(e)}")
    
    def create_hyperopt_visualization(self, results):
        """Create hyperparameter optimization visualization with early stopping info"""
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # Get optimization results
            opt_results = results.get('optimization_results', {})
            results_df = results.get('results_df')
            early_stopping_info = results.get('early_stopping_info', {})
            
            if results_df is not None and len(results_df) > 0:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                
                fig = Figure(figsize=(15, 10))
                
                # Score evolution plot
                ax1 = fig.add_subplot(221)
                scores = results_df['mean_test_score'].values
                iterations = range(len(scores))
                ax1.plot(iterations, scores, 'b-', alpha=0.7, linewidth=2)
                ax1.fill_between(iterations, 
                               scores - results_df['std_test_score'].values,
                               scores + results_df['std_test_score'].values,
                               alpha=0.3, color='blue')
                
                # Mark early stopping point if applicable
                if early_stopping_info.get('early_stopped', False):
                    stopping_iter = early_stopping_info.get('stopping_iteration', len(scores))
                    ax1.axvline(stopping_iter, color='red', linestyle='--', alpha=0.7, 
                               label=f'Early stopped at iteration {stopping_iter + 1}')
                
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('CV Score')
                ax1.set_title('Hyperparameter Optimization Progress')
                ax1.grid(True, alpha=0.3)
                
                # Best score marker
                best_idx = np.argmax(scores)
                ax1.plot(best_idx, scores[best_idx], 'ro', markersize=8, label=f'Best: {scores[best_idx]:.4f}')
                ax1.legend()
                
                # Parameter importance (if available)
                param_cols = [col for col in results_df.columns if col.startswith('param_')]
                if len(param_cols) > 0:
                    ax2 = fig.add_subplot(222)
                    param_importance = {}
                    
                    for param_col in param_cols[:8]:  # Limit to top 8 parameters
                        param_name = param_col.replace('param_', '')
                        param_values = results_df[param_col].values
                        
                        # Calculate correlation with scores
                        try:
                            numeric_values = pd.to_numeric(param_values, errors='coerce')
                            if not numeric_values.isna().all():
                                correlation = abs(np.corrcoef(numeric_values, scores)[0, 1])
                                if not np.isnan(correlation):
                                    param_importance[param_name] = correlation
                        except:
                            pass
                    
                    if param_importance:
                        params = list(param_importance.keys())
                        importances = list(param_importance.values())
                        ax2.barh(params, importances)
                        ax2.set_xlabel('Absolute Correlation with Score')
                        ax2.set_title('Parameter Importance')
                        ax2.grid(True, alpha=0.3)
                
                # Score distribution
                ax3 = fig.add_subplot(223)
                ax3.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(scores[best_idx], color='red', linestyle='--', linewidth=2, label=f'Best: {scores[best_idx]:.4f}')
                ax3.set_xlabel('CV Score')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Score Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Best parameters display
                ax4 = fig.add_subplot(224)
                ax4.axis('off')
                best_params = results.get('best_params', {})
                params_text = "Best Parameters:\n\n"
                for i, (param, value) in enumerate(best_params.items()):
                    if i < 10:  # Limit display
                        params_text += f"{param}: {value}\n"
                    elif i == 10:
                        params_text += f"... and {len(best_params) - 10} more"
                        break
                
                ax4.text(0.1, 0.9, params_text, transform=ax4.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                fig.tight_layout()
                canvas = FigureCanvas(fig)
                
                # Add navigation toolbar for interactivity
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                toolbar = NavigationToolbar(canvas, widget)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
            
            # Summary text with early stopping information
            summary_text = QTextEdit()
            summary_text.setMaximumHeight(150)
            summary_text.setReadOnly(True)
            
            summary = f"Hyperparameter Optimization Summary:\n"
            summary += f"• Search method: {results.get('search_method', 'N/A')}\n"
            summary += f"• Total iterations: {len(results_df) if results_df is not None else 0}\n"
            summary += f"• Best score: {results.get('best_score', 0):.4f}\n"
            summary += f"• Model: {results.get('model_name', 'N/A')}\n"
            
            # Add early stopping information
            if early_stopping_info:
                summary += f"\nEarly Stopping Information:\n"
                summary += f"• Early stopping enabled: {results.get('early_stopping_enabled', False)}\n"
                if early_stopping_info.get('early_stopped', False):
                    summary += f"• Early stopped: Yes (iteration {early_stopping_info.get('stopping_iteration', 0) + 1})\n"
                    summary += f"• Patience used: {early_stopping_info.get('patience_used', 0)}\n"
                    summary += f"• Min improvement threshold: {early_stopping_info.get('min_improvement_threshold', 0):.4f}\n"
                else:
                    summary += f"• Early stopped: No (completed all iterations)\n"
            
            summary_text.setText(summary)
            layout.addWidget(summary_text)
            
            return widget
            
        except Exception as e:
            print(f"Error creating hyperparameter optimization visualization: {e}")
            return None
    
    def add_viz_tab(self, widget, name):
        """Add visualization tab"""
        # Remove existing tab with same name
        for i in range(self.right_panel.count()):
            if self.right_panel.tabText(i) == name:
                self.right_panel.removeTab(i)
                break
        
        self.right_panel.addTab(widget, name)
        self.viz_tabs[name] = widget
    
    def proceed_to_next_module(self):
        """Proceed to model training module"""
        if not self.selected_features:
            QMessageBox.warning(self, "Warning", "No features selected.")
            return
            
        try:
            print("=== PROCEED TO NEXT MODULE - START ===")
            
            # Safety check: Ensure we have basic data
            if self.X is None or self.y is None:
                QMessageBox.warning(self, "Warning", "No data available. Please load data first.")
                return
            
            print(f"Basic data check passed - X: {self.X.shape}, y: {self.y.shape}")
            
            # Initialize variables with safe defaults
            X_selected = None
            y_selected = None
            
            # CRITICAL FIX: Use final aligned data from worker instead of original data
            if (self.worker and 
                hasattr(self.worker, 'final_data') and 
                self.worker.final_data is not None and
                not self.worker.final_data.empty):
                
                try:
                    # Use the processed, aligned data from feature selection
                    final_data = self.worker.final_data
                    
                    print(f"Using worker final_data with shape: {final_data.shape}")
                    print(f"Available columns: {list(final_data.columns)}")
                    print(f"Selected features: {self.selected_features}")
                    print(f"Worker y_name: {self.worker.y_name}")
                    
                    # Verify all selected features exist in final_data
                    missing_features = [f for f in self.selected_features if f not in final_data.columns]
                    if missing_features:
                        print(f"ERROR: Missing features in final_data: {missing_features}")
                        raise ValueError(f"Selected features not found in processed data: {missing_features}")
                    
                    # Verify target column exists
                    if self.worker.y_name not in final_data.columns:
                        print(f"ERROR: Target column '{self.worker.y_name}' not found in final_data")
                        raise ValueError(f"Target column '{self.worker.y_name}' not found in processed data")
                    
                    # Extract data safely
                    X_selected = final_data[self.selected_features].copy()
                    y_selected = final_data[self.worker.y_name].copy()
                    
                    print(f"=== DATA ALIGNMENT CHECK ===")
                    print(f"Original data shape: {self.X.shape[0]} samples")
                    print(f"Final aligned data shape: {len(final_data)} samples")
                    print(f"X_selected shape: {X_selected.shape}")
                    print(f"y_selected shape: {y_selected.shape}")
                    print(f"Selected features: {len(self.selected_features)}")
                    
                    # Verify no NaN values
                    x_nan_count = X_selected.isnull().sum().sum()
                    y_nan_count = y_selected.isnull().sum()
                    
                    if x_nan_count > 0:
                        print(f"WARNING: X_selected contains {x_nan_count} NaN values!")
                    if y_nan_count > 0:
                        print(f"WARNING: y_selected contains {y_nan_count} NaN values!")
                    
                    if x_nan_count == 0 and y_nan_count == 0:
                        print(f"✓ All data is clean (no NaN values)")
                    
                    print(f"✓ Using aligned data: X shape {X_selected.shape}, y shape {y_selected.shape}")
                    
                except Exception as e:
                    print(f"ERROR in worker data extraction: {str(e)}")
                    print("Falling back to original data method")
                    X_selected = None
                    y_selected = None
            
            # Fallback to original method if worker data failed
            if X_selected is None or y_selected is None:
                print("WARNING: Using fallback data method - potential alignment issues!")
                
                # Verify selected features exist in original data
                missing_features = [f for f in self.selected_features if f not in self.X.columns]
                if missing_features:
                    QMessageBox.critical(self, "Error", f"Selected features not found: {missing_features}")
                    return
                
                X_selected = self.X[self.selected_features].copy()
                y_selected = self.y.copy()
                
                print(f"Fallback data - X: {X_selected.shape}, y: {y_selected.shape}")
            
            # Final safety checks
            if X_selected is None or y_selected is None:
                QMessageBox.critical(self, "Error", "Failed to prepare data for next module.")
                return
            
            if len(X_selected) != len(y_selected):
                QMessageBox.critical(self, "Error", f"Data alignment error: X has {len(X_selected)} samples, y has {len(y_selected)} samples.")
                return
            
            if len(X_selected) == 0:
                QMessageBox.critical(self, "Error", "No samples available after feature selection.")
                return
            
            # Prepare configuration to pass to next module
            config = {
                'selected_model': self.model_combo.currentText() if self.model_combo.currentText() else 'Random Forest',
                'task_type': self.task_type if self.task_type else 'classification',
                'scaling_method': self.scaling_combo.currentText() if hasattr(self, 'scaling_combo') else 'StandardScaler',
                'random_seed': self.random_seed_spin.value() if hasattr(self, 'random_seed_spin') else 42,
                'cv_folds': self.cv_folds_spin.value() if hasattr(self, 'cv_folds_spin') else 5,
                'scoring_metric': self.scoring_combo.currentText() if hasattr(self, 'scoring_combo') else 'accuracy'
            }
            
            print(f"Configuration prepared: {config}")
            
            # Emit signal to next module with error handling
            try:
                print("Emitting features_ready signal...")
                self.features_ready.emit(X_selected, y_selected, config)
                print("Signal emitted successfully")
            except Exception as e:
                print(f"ERROR emitting signal: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to send data to next module: {str(e)}")
                return
            
            self.status_updated.emit(f"Proceeding with {len(self.selected_features)} selected features")
            
            # Show success message
            try:
                QMessageBox.information(
                    self, "Success", 
                    f"Proceeding to training with:\n\n"
                    f"Features: {len(self.selected_features)}\n"
                    f"Samples: {len(X_selected)}\n"
                    f"Model: {config['selected_model']}\n"
                    f"Scaling: {config['scaling_method']}"
                )
            except Exception as e:
                print(f"Warning: Could not show success message: {str(e)}")
            
            print("=== PROCEED TO NEXT MODULE - SUCCESS ===")
            
        except Exception as e:
            print(f"CRITICAL ERROR in proceed_to_next_module: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Critical error preparing data for next module: {str(e)}")
            print("=== PROCEED TO NEXT MODULE - FAILED ===")
    
    def apply_wizard_config(self, config: dict):
        """Apply intelligent wizard configuration"""
        try:
            # Apply feature selection configuration
            if 'feature_selection_method' in config:
                method = config['feature_selection_method']
                if method == 'correlation':
                    self.enable_correlation_filter.setChecked(True)
                elif method == 'importance':
                    self.enable_importance_filter.setChecked(True)
                elif method == 'advanced':
                    self.enable_advanced_search.setChecked(True)
            
            # Apply other configuration parameters
            if 'correlation_threshold' in config:
                self.correlation_threshold_spin.setValue(config['correlation_threshold'])
            
            if 'importance_top_k' in config and hasattr(self, 'importance_top_k_spin'):
                self.importance_top_k_spin.setValue(config['importance_top_k'])
                
        except Exception as e:
            self.status_updated.emit(f"Failed to apply wizard configuration: {str(e)}")
    
    def reset(self):
        """Reset the module"""
        self.X = None
        self.y = None
        self.task_type = None
        self.selected_features = []
        self.selection_results = {}
        
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker = None
        
        # Reset UI
        self.data_info_label.setText("No data loaded")
        self.results_text.clear()
        self.model_combo.clear()
        
        # Reset controls to defaults
        self.task_type_combo.setCurrentIndex(0)
        self.train_test_split_spin.setValue(0.8)
        self.random_seed_spin.setValue(42)
        self.cv_folds_spin.setValue(5)
        self.scoring_combo.setCurrentIndex(0)
        
        # Reset filters
        self.enable_importance_filter.setChecked(True)
        self.enable_correlation_filter.setChecked(True)
        self.enable_advanced_search.setChecked(False)
        
        # Disable controls
        self.execute_btn.setEnabled(False)
        self.proceed_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # Clear visualizations
        for tab_name in list(self.viz_tabs.keys()):
            for i in range(self.right_panel.count()):
                if self.right_panel.tabText(i) == tab_name:
                    self.right_panel.removeTab(i)
                    break
        self.viz_tabs.clear()
        
        self.setEnabled(False)
        self.status_updated.emit("Feature selection module reset")
    
    def on_early_stopping_toggled(self, checked):
        """Toggle early stopping controls"""
        self.early_stopping_controls.setEnabled(checked)
    
    def on_hyperopt_toggled(self, checked):
        """Toggle hyperparameter optimization controls"""
        self.hyperopt_controls.setEnabled(checked)
    
    def on_search_method_changed(self):
        """Handle search method change"""
        method = self.search_method_combo.currentText()
        # Grid search doesn't support early stopping
        if method == 'Grid Search':
            self.enable_early_stopping_cb.setEnabled(False)
            self.early_stopping_controls.setEnabled(False)
        else:
            self.enable_early_stopping_cb.setEnabled(True)
            self.early_stopping_controls.setEnabled(self.enable_early_stopping_cb.isChecked())
    
    def configure_parameter_space(self):
        """Open parameter space configuration dialog"""
        try:
            selected_model = self.model_combo.currentText()
            if not selected_model:
                QMessageBox.warning(self, "Warning", "Please select a model first.")
                return
                
            dialog = ParameterSpaceDialog([selected_model], self.task_type, self.custom_param_spaces)
            if dialog.exec_() == QDialog.Accepted:
                self.custom_param_spaces = dialog.get_parameter_spaces()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening parameter configuration dialog: {str(e)}")


class ParameterSpaceDialog(QDialog):
    """Dialog for configuring hyperparameter spaces"""
    
    def __init__(self, models, task_type, existing_spaces=None):
        super().__init__()
        self.models = models if isinstance(models, list) else [models]
        self.model = self.models[0]  # Use first model
        self.task_type = task_type
        self.parameter_spaces = existing_spaces or {}
        try:
            self.init_ui()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error initializing dialog: {str(e)}")
    
    def init_ui(self):
        self.setWindowTitle("Configure Parameter Spaces")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Configure Hyperparameter Search Spaces")
        header_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(header_label)
        
        # Scrollable area for models
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.param_controls = {}
        
        # Get default parameters with error handling
        try:
            default_params = get_default_hyperparameters(self.model, self.task_type)
        except Exception as e:
            print(f"Error getting parameters for {self.model}: {e}")
            default_params = {}
        
        # Create parameter controls
        for param_name, param_values in default_params.items():
            try:
                param_widget = QWidget()
                param_layout = QHBoxLayout(param_widget)
                param_layout.setContentsMargins(0, 0, 0, 0)
                
                # Parameter name
                param_layout.addWidget(QLabel(f"{param_name}:"))
                
                # Parameter type detection and control creation
                if param_values and len(param_values) > 0 and isinstance(param_values[0], (int, float)):
                    # Numeric parameter - use range controls
                    min_spin = QDoubleSpinBox()
                    max_spin = QDoubleSpinBox()
                    
                    if isinstance(param_values[0], int):
                        min_spin.setDecimals(0)
                        max_spin.setDecimals(0)
                        min_spin.setRange(0, 100000)
                        max_spin.setRange(0, 100000)
                    else:
                        min_spin.setDecimals(6)
                        max_spin.setDecimals(6)
                        min_spin.setRange(0.000001, 100000)
                        max_spin.setRange(0.000001, 100000)
                    
                    # Safely get min/max values
                    try:
                        min_val = min(param_values)
                        max_val = max(param_values)
                        min_spin.setValue(min_val)
                        max_spin.setValue(max_val)
                    except:
                        min_spin.setValue(0.1)
                        max_spin.setValue(10.0)
                    
                    param_layout.addWidget(QLabel("Min:"))
                    param_layout.addWidget(min_spin)
                    param_layout.addWidget(QLabel("Max:"))
                    param_layout.addWidget(max_spin)
                    
                    self.param_controls[param_name] = ('range', min_spin, max_spin)
                else:
                    # Categorical parameter - use list
                    values_edit = QLineEdit()
                    try:
                        values_edit.setText(", ".join(map(str, param_values)))
                    except:
                        values_edit.setText("auto")
                    
                    param_layout.addWidget(QLabel("Values:"))
                    param_layout.addWidget(values_edit)
                    
                    self.param_controls[param_name] = ('categorical', values_edit)
                
                scroll_layout.addWidget(param_widget)
            except Exception as e:
                print(f"Error creating control for parameter {param_name}: {e}")
                continue
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        
        # Load existing spaces if available
        try:
            self.load_existing_spaces()
        except Exception as e:
            print(f"Error loading existing spaces: {e}")
    
    def load_existing_spaces(self):
        """Load existing parameter spaces into the dialog"""
        try:
            for param_name, space_config in self.parameter_spaces.items():
                if param_name in self.param_controls:
                    control_type, *widgets = self.param_controls[param_name]
                    if control_type == 'range' and len(widgets) == 2:
                        min_widget, max_widget = widgets
                        if isinstance(space_config, dict) and 'min' in space_config:
                            min_widget.setValue(space_config['min'])
                            max_widget.setValue(space_config['max'])
                    elif control_type == 'categorical' and len(widgets) == 1:
                        values_widget = widgets[0]
                        if isinstance(space_config, list):
                            values_widget.setText(", ".join(map(str, space_config)))
        except Exception as e:
            print(f"Error in load_existing_spaces: {e}")
    
    def reset_to_defaults(self):
        """Reset all parameters to default values"""
        try:
            # Get default parameters
            default_params = get_default_hyperparameters(self.model, self.task_type)
            
            for param_name, control_info in self.param_controls.items():
                try:
                    if param_name in default_params:
                        param_values = default_params[param_name]
                        control_type, *widgets = control_info
                        if control_type == 'range' and len(widgets) == 2:
                            min_widget, max_widget = widgets
                            min_widget.setValue(min(param_values))
                            max_widget.setValue(max(param_values))
                        elif control_type == 'categorical' and len(widgets) == 1:
                            values_widget = widgets[0]
                            values_widget.setText(", ".join(map(str, param_values)))
                except Exception as e:
                    print(f"Error resetting parameter {param_name}: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error resetting to defaults: {str(e)}")
    
    def get_parameter_spaces(self):
        """Extract parameter spaces from the dialog"""
        spaces = {}
        
        try:
            for param_name, control_info in self.param_controls.items():
                try:
                    control_type, *widgets = control_info
                    
                    if control_type == 'range' and len(widgets) == 2:
                        min_widget, max_widget = widgets
                        spaces[param_name] = {
                            'type': 'range',
                            'min': min_widget.value(),
                            'max': max_widget.value()
                        }
                    elif control_type == 'categorical' and len(widgets) == 1:
                        values_widget = widgets[0]
                        values_text = values_widget.text().strip()
                        if values_text:
                            try:
                                # Try to parse as appropriate types
                                values = [v.strip() for v in values_text.split(',')]
                                # Attempt to convert to numbers if possible
                                converted_values = []
                                for v in values:
                                    try:
                                        # Try int first, then float
                                        if '.' in v:
                                            converted_values.append(float(v))
                                        else:
                                            converted_values.append(int(v))
                                    except ValueError:
                                        # Keep as string
                                        converted_values.append(v)
                                
                                spaces[param_name] = converted_values
                            except:
                                # Fallback to string list
                                spaces[param_name] = [v.strip() for v in values_text.split(',')]
                except Exception as e:
                    print(f"Error extracting parameter {param_name}: {e}")
                    continue
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error extracting parameter spaces: {str(e)}")
        
        return spaces 