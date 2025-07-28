import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from scipy.stats import norm, qmc
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
import warnings
import gc
import itertools
import random
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Bayesian Optimization imports
try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


class ActiveLearningOptimizer:
    """
    A comprehensive Bayesian Optimization engine for active learning with multi-objective support.
    
    Available Surrogate Models:
    --------------------------
    
    Tree-Based Models (with Bootstrap uncertainty):
    • RandomForest: Robust ensemble method, good for mixed data types and noisy data
    • XGBoost: Gradient boosting, excellent performance on structured data
    • LightGBM: Fast gradient boosting, memory efficient, handles categorical features well  
    • CatBoost: Gradient boosting optimized for categorical features, reduces overfitting
    • ExtraTrees: Extra randomized trees, faster than Random Forest with lower variance
    
    Probabilistic Models (with native uncertainty):
    • GaussianProcess: Provides principled uncertainty quantification, good for small datasets
    • BayesianRidge: Bayesian linear regression with uncertainty estimates
    
    Neural Networks (with Monte Carlo uncertainty):
    • MLPRegressor: Multi-layer perceptron, handles complex non-linear relationships
    
    Kernel Methods (with Bootstrap uncertainty):
    • SVR: Support Vector Regression, effective for high-dimensional data and small samples
    
    Model Selection Guidelines:
    --------------------------
    • Small datasets (<100 samples): GaussianProcess, BayesianRidge, SVR
    • Large datasets (>1000 samples): LightGBM, XGBoost, CatBoost
    • Categorical features: CatBoost, LightGBM  
    • High-dimensional data: SVR, BayesianRidge
    • Complex non-linear patterns: MLPRegressor, XGBoost
    • Fast training needed: ExtraTrees, LightGBM
    • Interpretability important: RandomForest, BayesianRidge
    
    Hyperparameter Optimization:
    ---------------------------
    • Grid Search: Exhaustive search over predefined parameter grids
    • Bayesian Optimization: Intelligent search using Gaussian Process, Random Forest, or Gradient Boosting
      - More efficient than grid search
      - Better for expensive evaluations
      - Automatic exploration vs exploitation balance
    
    Acquisition Functions: ExpectedImprovement, UpperConfidenceBound
    Supports both single-objective and multi-objective optimization with Pareto analysis.
    
    Example Usage:
    -------------
    # Basic usage with Bayesian optimization
    optimizer = ActiveLearningOptimizer(model_type='XGBoost')
    
    # Bayesian hyperparameter optimization
    best_params = optimizer.optimize_hyperparameters_bayesian(
        training_df=train_data,
        target_columns=['strength'],
        feature_columns=['composition', 'temperature', 'time'],
        n_calls=50,
        optimizer='gp'  # or 'forest', 'gbrt'
    )
    
    # Apply optimized parameters and run active learning
    optimizer.apply_optimized_parameters(best_params)
    results = optimizer.run(train_data, candidates, 'strength', feature_columns)
    """
    
    def __init__(self, model_type='RandomForest', acquisition_function='ExpectedImprovement', 
                 random_state=42):
        """
        Initialize the Active Learning Optimizer.
        
        Parameters:
        -----------
        model_type : str, default='RandomForest'
            Type of surrogate model to use: 
            'RandomForest', 'XGBoost', 'GaussianProcess', 'LightGBM', 'CatBoost', 
            'ExtraTrees', 'SVR', 'MLPRegressor', 'BayesianRidge'
        acquisition_function : str, default='ExpectedImprovement'
            Acquisition function: 'ExpectedImprovement', 'UpperConfidenceBound'
        random_state : int, default=42
            Random state for reproducibility
        """
        self.model_type = model_type
        self.acquisition_function = acquisition_function
        self.random_state = random_state
        self.model = None
        self.models = {}  # For multi-objective
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scalers_X = {}  # For multi-objective
        self.feature_names = None
        self.is_fitted = False
        self._progress_callback = None  # For iterative generation progress
        self._batch_size = None  # Custom batch size for iterative generation
    
        # Iterative optimization state
        self._performance_history = []
        self._current_training_data = None
        self._current_virtual_data = None
        self._iteration_count = 0
        self._is_iterative_mode = False
    
    def set_progress_callback(self, callback):
        """设置迭代生成的进度回调函数"""
        self._progress_callback = callback
    
    def set_batch_size(self, batch_size):
        """设置迭代生成的批次大小"""
        self._batch_size = batch_size
        
    def _create_model(self, model_config=None):
        """Create and return a model instance based on the specified type."""
        if model_config is None:
            model_config = {}
            
        # Use optimized parameters if available
        if hasattr(self, '_optimized_params') and self._optimized_params is not None:
            optimized_config = self._optimized_params.copy()
            # Override with any explicit config
            optimized_config.update(model_config.get('params', {}))
            model_config = {'params': optimized_config}
            
        if self.model_type == 'RandomForest':
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': self.random_state
            }
            default_params.update(model_config.get('params', {}))
            return RandomForestRegressor(**default_params)
            
        elif self.model_type == 'XGBoost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Please install it using 'pip install xgboost'")
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.random_state,
                'verbosity': 0  # Reduce output noise
            }
            default_params.update(model_config.get('params', {}))
            return xgb.XGBRegressor(**default_params)
            
        elif self.model_type == 'GaussianProcess':
            # Use a combination of RBF and White noise kernels
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
            default_params = {
                'kernel': kernel,
                'alpha': 1e-6,
                'random_state': self.random_state,
                'normalize_y': True
            }
            default_params.update(model_config.get('params', {}))
            return GaussianProcessRegressor(**default_params)
            
        elif self.model_type == 'LightGBM':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not installed. Please install it using 'pip install lightgbm'")
            default_params = {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'random_state': self.random_state,
                'verbosity': -1,  # Reduce output noise
                'force_row_wise': True  # Avoid warnings
            }
            default_params.update(model_config.get('params', {}))
            return lgb.LGBMRegressor(**default_params)
            
        elif self.model_type == 'CatBoost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost is not installed. Please install it using 'pip install catboost'")
            default_params = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_state': self.random_state,
                'verbose': False,  # Reduce output noise
                'allow_writing_files': False  # Prevent file creation
            }
            default_params.update(model_config.get('params', {}))
            return cb.CatBoostRegressor(**default_params)
            
        elif self.model_type == 'ExtraTrees':
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': self.random_state
            }
            default_params.update(model_config.get('params', {}))
            return ExtraTreesRegressor(**default_params)
            
        elif self.model_type == 'SVR':
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            }
            default_params.update(model_config.get('params', {}))
            return SVR(**default_params)
            
        elif self.model_type == 'MLPRegressor':
            default_params = {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 500,
                'alpha': 0.001,
                'solver': 'adam',
                'learning_rate': 'adaptive',
                'random_state': self.random_state,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
            default_params.update(model_config.get('params', {}))
            return MLPRegressor(**default_params)
            
        elif self.model_type == 'BayesianRidge':
            default_params = {
                'alpha_1': 1e-6,
                'alpha_2': 1e-6,
                'lambda_1': 1e-6,
                'lambda_2': 1e-6,
                'compute_score': True
            }
            default_params.update(model_config.get('params', {}))
            return BayesianRidge(**default_params)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Available models: 'RandomForest', 'XGBoost', 'GaussianProcess', "
                           f"'LightGBM', 'CatBoost', 'ExtraTrees', 'SVR', 'MLPRegressor', 'BayesianRidge'")
    
    def _calculate_expected_improvement(self, mean, std, best_so_far, goal='maximize', xi=0.01):
        """
        Calculate Expected Improvement acquisition function.
        
        Parameters:
        -----------
        mean : array-like
            Predicted means
        std : array-like
            Predicted standard deviations
        best_so_far : float
            Current best observed value
        goal : str
            'maximize' or 'minimize'
        xi : float
            Exploration parameter (trade-off between exploration and exploitation)
            
        Returns:
        --------
        ei : array-like
            Expected improvement values
        """
        mean = np.array(mean)
        std = np.array(std)
        
        # Avoid division by zero
        std = np.maximum(std, 1e-9)
        
        if goal == 'maximize':
            improvement = mean - best_so_far - xi
        else:
            improvement = best_so_far - mean - xi
            
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        # Set EI to 0 where std is effectively 0
        ei[std < 1e-9] = 0
        
        return ei
    
    def _calculate_upper_confidence_bound(self, mean, std, goal='maximize', kappa=2.576):
        """
        Calculate Upper Confidence Bound acquisition function.
        
        Parameters:
        -----------
        mean : array-like
            Predicted means
        std : array-like
            Predicted standard deviations
        goal : str
            'maximize' or 'minimize'
        kappa : float
            Exploration parameter (higher values encourage more exploration)
            
        Returns:
        --------
        ucb : array-like
            Upper confidence bound values
        """
        mean = np.array(mean)
        std = np.array(std)
        
        if goal == 'maximize':
            ucb = mean + kappa * std
        else:
            ucb = -(mean - kappa * std)  # For minimization, we want lower values
            
        return ucb
    
    def _is_pareto_optimal(self, costs):
        """
        Determine Pareto optimal points efficiently.
        
        Parameters:
        -----------
        costs : numpy.ndarray
            Cost matrix (n_points x n_objectives)
            
        Returns:
        --------
        is_efficient : numpy.ndarray
            Boolean array indicating Pareto optimal points
        """
        costs = np.array(costs)
        if costs.ndim == 1:
            costs = costs.reshape(-1, 1)
        
        n_points = costs.shape[0]
        is_efficient = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            if is_efficient[i]:
                # Remove dominated points
                dominated = np.all(costs >= costs[i], axis=1) & np.any(costs > costs[i], axis=1)
                is_efficient[dominated] = False
        
        return is_efficient
    
    def _scalarize_objectives(self, predictions, weights, goal_directions):
        """
        Scalarize multi-objective predictions using weighted sum.
        
        Parameters:
        -----------
        predictions : dict
            Dictionary of predictions for each objective
        weights : numpy.ndarray
            Weights for each objective
        goal_directions : list
            List of 'maximize' or 'minimize' for each objective
            
        Returns:
        --------
        scalarized : numpy.ndarray
            Scalarized values
        """
        scalarized = np.zeros(len(next(iter(predictions.values()))['mean']))
        
        for i, (obj_name, pred) in enumerate(predictions.items()):
            mean = pred['mean']
            if goal_directions[i] == 'minimize':
                mean = -mean  # Convert to maximization
            scalarized += weights[i] * mean
            
        return scalarized

    def _run_single_iteration(self, training_df, virtual_df, target_column, feature_columns, 
            goal='maximize', model_config=None, acquisition_config=None, 
            n_iterations_bootstrap=100):
        """
        Execute the complete active learning optimization workflow for single objective.
        
        Parameters:
        -----------
        training_df : pandas.DataFrame
            Existing experimental data
        virtual_df : pandas.DataFrame
            Candidate data points to evaluate
        target_column : str
            Name of the target column to optimize
        feature_columns : list
            List of feature column names
        goal : str, default='maximize'
            Optimization goal: 'maximize' or 'minimize'
        model_config : dict, optional
            Model configuration parameters
        acquisition_config : dict, optional
            Acquisition function configuration
        n_iterations_bootstrap : int, default=100
            Number of bootstrap iterations (for RF and XGBoost)
            
        Returns:
        --------
        results_df : pandas.DataFrame
            Virtual data augmented with predictions and acquisition scores
        """
        if model_config is None:
            model_config = {}
        if acquisition_config is None:
            acquisition_config = {}
            
        # Validate inputs
        if target_column not in training_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in training data")
        
        missing_features = set(feature_columns) - set(training_df.columns)
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in training data")
            
        missing_virtual_features = set(feature_columns) - set(virtual_df.columns)
        if missing_virtual_features:
            raise ValueError(f"Features {missing_virtual_features} not found in virtual data")
        
        # 🔍 添加详细的调试信息  
        print(f"\n=== 🔍 ACTIVE LEARNING RUN DEBUG ===")
        print(f"Training data shape: {training_df.shape}")
        print(f"Virtual data shape: {virtual_df.shape}")
        print(f"Available columns in training_df: {list(training_df.columns)}")
        print(f"Available columns in virtual_df: {list(virtual_df.columns)}")
        print(f"Feature columns requested: {feature_columns}")
        print(f"Number of feature columns: {len(feature_columns)}")
        print(f"Target column: {target_column}")
        
        # Prepare data
        X_train = training_df[feature_columns].copy()
        y_train = training_df[target_column].copy()
        X_virtual = virtual_df[feature_columns].copy()
        
        print(f"X_train shape after selection: {X_train.shape}")
        print(f"y_train shape after selection: {y_train.shape}")
        print(f"X_virtual shape after selection: {X_virtual.shape}")
        print(f"X_train columns: {list(X_train.columns)}")
        
        # Store feature names for later use
        self.feature_names = feature_columns
        
        # Remove any rows with missing values
        mask_train = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        X_train = X_train[mask_train]
        y_train = y_train[mask_train]
        
        mask_virtual = ~X_virtual.isnull().any(axis=1)
        X_virtual = X_virtual[mask_virtual]
        
        if len(X_train) == 0:
            raise ValueError("No valid training data after removing missing values")
        if len(X_virtual) == 0:
            raise ValueError("No valid virtual data after removing missing values")
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_virtual_scaled = self.scaler_X.transform(X_virtual)
        
        # Determine current best value
        if goal == 'maximize':
            best_so_far = y_train.max()
        else:
            best_so_far = y_train.min()
        
        print(f"Current best value: {best_so_far:.4f}")
        
        # Uncertainty quantification
        if self.model_type == 'GaussianProcess':
            # Gaussian Process can directly provide mean and std
            print("Using Gaussian Process for uncertainty quantification...")
            model = self._create_model(model_config)
            
            # Scale target for GP
            y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            model.fit(X_train_scaled, y_train_scaled)
            
            # Predict with uncertainty
            mean_scaled, std_scaled = model.predict(X_virtual_scaled, return_std=True)
            
            # Transform back to original scale
            mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
            # For std, we need to scale by the std of the scaler
            std = std_scaled * self.scaler_y.scale_[0]
            
            self.model = model
            
        elif self.model_type == 'BayesianRidge':
            # Bayesian Ridge can provide uncertainty estimates
            print("Using Bayesian Ridge for uncertainty quantification...")
            model = self._create_model(model_config)
            model.fit(X_train_scaled, y_train)
            
            # Predict with uncertainty
            mean, std = model.predict(X_virtual_scaled, return_std=True)
            
            self.model = model
            
        elif self.model_type in ['MLPRegressor']:
            # For neural networks, use Monte Carlo Dropout for uncertainty
            print(f"Using {self.model_type} with Monte Carlo Dropout for uncertainty quantification...")
            predictions = []
            
            # Create multiple models with different dropout realizations
            n_mc_samples = min(n_iterations_bootstrap, 50)  # Limit MC samples for speed
            
            for i in range(n_mc_samples):
                # Create model with dropout enabled during prediction
                model_config_mc = model_config.copy() if model_config else {}
                if 'params' not in model_config_mc:
                    model_config_mc['params'] = {}
                
                # Enable early stopping for faster training
                model_config_mc['params'].update({
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'n_iter_no_change': 10
                })
                
                model = self._create_model(model_config_mc)
                model.fit(X_train_scaled, y_train)
                
                # For MC Dropout, we would need to modify the model architecture
                # For now, use bootstrap sampling as approximation
                pred = model.predict(X_virtual_scaled)
                predictions.append(pred)
                
                if (i + 1) % max(1, n_mc_samples // 5) == 0:
                    print(f"MC iteration {i + 1}/{n_mc_samples} completed")
            
            # Calculate mean and std from MC predictions
            predictions = np.array(predictions)
            mean = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
            
            self.model = model
            
        else:
            # Bootstrap for tree-based and other models
            # (RandomForest, XGBoost, LightGBM, CatBoost, ExtraTrees, SVR)
            print(f"Using {self.model_type} with {n_iterations_bootstrap} bootstrap iterations...")
            predictions = []
            
            for i in range(n_iterations_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X_train_scaled), size=len(X_train_scaled), replace=True)
                X_boot = X_train_scaled[indices]
                y_boot = y_train.iloc[indices]
                
                # Train model on bootstrap sample
                model = self._create_model(model_config)
                model.fit(X_boot, y_boot)
                
                # Predict on virtual data
                pred = model.predict(X_virtual_scaled)
                predictions.append(pred)
                
                if (i + 1) % max(1, n_iterations_bootstrap // 10) == 0:
                    print(f"Bootstrap iteration {i + 1}/{n_iterations_bootstrap} completed")
            
            # Calculate mean and std from bootstrap predictions
            predictions = np.array(predictions)
            mean = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
            
            # Store the last trained model for feature importance
            self.model = model
        
        self.is_fitted = True
        
        # Calculate acquisition function
        if self.acquisition_function == 'ExpectedImprovement':
            xi = acquisition_config.get('xi', 0.01)
            acquisition_scores = self._calculate_expected_improvement(
                mean, std, best_so_far, goal, xi
            )
        elif self.acquisition_function == 'UpperConfidenceBound':
            kappa = acquisition_config.get('kappa', 2.576)
            acquisition_scores = self._calculate_upper_confidence_bound(
                mean, std, goal, kappa
            )
        else:
            raise ValueError(f"Unsupported acquisition function: {self.acquisition_function}")
        
        # Prepare results
        results_df = virtual_df.copy()
        
        # Add predictions and scores only for valid rows
        results_df['predicted_mean'] = np.nan
        results_df['uncertainty_std'] = np.nan
        results_df['acquisition_score'] = np.nan
        
        results_df.loc[mask_virtual, 'predicted_mean'] = mean
        results_df.loc[mask_virtual, 'uncertainty_std'] = std
        results_df.loc[mask_virtual, 'acquisition_score'] = acquisition_scores
        
        # Sort by acquisition score (descending)
        results_df = results_df.sort_values('acquisition_score', ascending=False, na_position='last')
        results_df = results_df.reset_index(drop=True)
        
        print(f"Analysis completed. Top acquisition score: {results_df['acquisition_score'].iloc[0]:.6f}")
        
        return results_df

    def _run_multi_objective_single_iteration(self, training_df, virtual_df, target_columns, feature_columns, 
                          goal_directions, model_config=None, acquisition_config=None, 
                          n_iterations_bootstrap=30, n_scalarizations=8):
        """
        Execute multi-objective optimization with Pareto analysis.
        
        Parameters:
        -----------
        training_df : pandas.DataFrame
            Existing experimental data
        virtual_df : pandas.DataFrame
            Candidate data points to evaluate
        target_columns : list
            List of target column names to optimize
        feature_columns : list
            List of feature column names
        goal_directions : list
            List of 'maximize' or 'minimize' for each target
        model_config : dict, optional
            Model configuration parameters
        acquisition_config : dict, optional
            Acquisition function configuration
        n_iterations_bootstrap : int, default=30
            Number of bootstrap iterations (reduced for multi-objective)
        n_scalarizations : int, default=8
            Number of scalarization weights to try
            
        Returns:
        --------
        results_df : pandas.DataFrame
            Virtual data with multi-objective predictions and Pareto analysis
        """
        if model_config is None:
            model_config = {}
        if acquisition_config is None:
            acquisition_config = {}
            
        print(f"Starting multi-objective optimization for {len(target_columns)} objectives")
        print(f"Targets: {target_columns}")
        print(f"Goals: {goal_directions}")
        
        # Validate inputs
        if len(target_columns) != len(goal_directions):
            raise ValueError("Number of target columns must match number of goal directions")
        
        for target_col in target_columns:
            if target_col not in training_df.columns:
                raise ValueError(f"Target column '{target_col}' not found in training data")
        
        missing_features = set(feature_columns) - set(training_df.columns)
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in training data")
            
        missing_virtual_features = set(feature_columns) - set(virtual_df.columns)
        if missing_virtual_features:
            raise ValueError(f"Features {missing_virtual_features} not found in virtual data")
        
        # Prepare data
        X_train = training_df[feature_columns].copy()
        X_virtual = virtual_df[feature_columns].copy()
        
        # Store feature names
        self.feature_names = feature_columns
        
        # Remove rows with missing values
        mask_train = ~X_train.isnull().any(axis=1)
        for target_col in target_columns:
            mask_train &= ~training_df[target_col].isnull()
        
        X_train = X_train[mask_train]
        
        mask_virtual = ~X_virtual.isnull().any(axis=1)
        X_virtual = X_virtual[mask_virtual]
        
        if len(X_train) == 0:
            raise ValueError("No valid training data after removing missing values")
        if len(X_virtual) == 0:
            raise ValueError("No valid virtual data after removing missing values")
        
        # Train models for each objective
        predictions = {}
        
        for i, target_col in enumerate(target_columns):
            print(f"Processing objective {i+1}/{len(target_columns)}: {target_col}")
            
            y_train = training_df.loc[mask_train, target_col].copy()
            
            # Scale features for this objective
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_virtual_scaled = scaler_X.transform(X_virtual)
            self.scalers_X[target_col] = scaler_X
            
            # Train model and get predictions
            if self.model_type == 'GaussianProcess':
                model = self._create_model(model_config)
                scaler_y = StandardScaler()
                y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                model.fit(X_train_scaled, y_train_scaled)
                
                mean_scaled, std_scaled = model.predict(X_virtual_scaled, return_std=True)
                mean = scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
                std = std_scaled * scaler_y.scale_[0]
                
                self.models[target_col] = model
                
            elif self.model_type == 'BayesianRidge':
                # Bayesian Ridge can provide uncertainty estimates
                model = self._create_model(model_config)
                model.fit(X_train_scaled, y_train)
                
                mean, std = model.predict(X_virtual_scaled, return_std=True)
                
                self.models[target_col] = model
                
            elif self.model_type in ['MLPRegressor']:
                # For neural networks, use reduced MC sampling in multi-objective mode
                print(f"Training with {min(n_iterations_bootstrap, 20)} MC iterations")
                mc_predictions = []
                
                for j in range(min(n_iterations_bootstrap, 20)):  # Reduced for multi-objective
                    # Create model with variation
                    model_config_mc = model_config.copy() if model_config else {}
                    if 'params' not in model_config_mc:
                        model_config_mc['params'] = {}
                    
                    # Add some variation to training
                    model_config_mc['params'].update({
                        'early_stopping': True,
                        'validation_fraction': 0.1,
                        'n_iter_no_change': 5  # Faster convergence for multi-objective
                    })
                    
                    model = self._create_model(model_config_mc)
                    model.fit(X_train_scaled, y_train)
                    
                    pred = model.predict(X_virtual_scaled)
                    mc_predictions.append(pred)
                
                # Calculate statistics
                mc_predictions = np.array(mc_predictions)
                mean = np.mean(mc_predictions, axis=0)
                std = np.std(mc_predictions, axis=0)
                
                self.models[target_col] = model
                
            else:
                # Bootstrap approach for tree-based and other models
                # (RandomForest, XGBoost, LightGBM, CatBoost, ExtraTrees, SVR)
                print(f"Training with {n_iterations_bootstrap} bootstrap iterations")
                bootstrap_predictions = []
                
                for j in range(n_iterations_bootstrap):
                    # Bootstrap sample
                    indices = np.random.choice(len(X_train_scaled), size=len(X_train_scaled), replace=True)
                    X_boot = X_train_scaled[indices]
                    y_boot = y_train.iloc[indices]
                    
                    # Train model
                    model = self._create_model(model_config)
                    model.fit(X_boot, y_boot)
                    
                    # Predict
                    pred = model.predict(X_virtual_scaled)
                    bootstrap_predictions.append(pred)
                
                # Calculate statistics
                bootstrap_predictions = np.array(bootstrap_predictions)
                mean = np.mean(bootstrap_predictions, axis=0)
                std = np.std(bootstrap_predictions, axis=0)
                
                self.models[target_col] = model
            
            # Store predictions
            predictions[target_col] = {
                'mean': mean,
                'std': std
            }
            
            print(f"Completed {target_col}: mean range [{mean.min():.3f}, {mean.max():.3f}]")
            
            # Clean up memory
            gc.collect()
        
        # Calculate acquisition scores using scalarization
        print("Calculating acquisition scores...")
        
        # Generate different weight combinations
        if len(target_columns) == 2:
            # For 2 objectives, use evenly spaced weights
            weight_combinations = []
            for w1 in np.linspace(0.1, 0.9, n_scalarizations):
                w2 = 1.0 - w1
                weight_combinations.append([w1, w2])
        else:
            # For more objectives, use random weights
            np.random.seed(self.random_state)
            weight_combinations = []
            for _ in range(n_scalarizations):
                weights = np.random.random(len(target_columns))
                weights = weights / weights.sum()
                weight_combinations.append(weights)
        
        # Calculate acquisition scores for each weight combination
        all_acquisition_scores = []
        
        for weights in weight_combinations:
            scalarized = self._scalarize_objectives(predictions, weights, goal_directions)
            
            # Calculate acquisition function on scalarized values
            # Use the mean of scalarized as "best so far"
            best_so_far = np.mean(scalarized)
            
            if self.acquisition_function == 'ExpectedImprovement':
                xi = acquisition_config.get('xi', 0.01)
                # For scalarized objectives, use combined uncertainty
                combined_std = np.sqrt(sum(predictions[col]['std']**2 for col in target_columns))
                acquisition_scores = self._calculate_expected_improvement(
                    scalarized, combined_std, best_so_far, 'maximize', xi
                )
            else:  # UpperConfidenceBound
                kappa = acquisition_config.get('kappa', 2.576)
                combined_std = np.sqrt(sum(predictions[col]['std']**2 for col in target_columns))
                acquisition_scores = self._calculate_upper_confidence_bound(
                    scalarized, combined_std, 'maximize', kappa
                )
            
            all_acquisition_scores.append(acquisition_scores)
        
        # Combine acquisition scores (take maximum across all weight combinations)
        final_acquisition_scores = np.max(all_acquisition_scores, axis=0)
        
        # Prepare results
        print("Creating results...")
        results_df = virtual_df.copy()
        
        # Add predictions for each objective
        for target_col in target_columns:
            mean_col = f'{target_col}_predicted_mean'
            std_col = f'{target_col}_uncertainty_std'
            
            results_df[mean_col] = np.nan
            results_df[std_col] = np.nan
            
            results_df.loc[mask_virtual, mean_col] = predictions[target_col]['mean']
            results_df.loc[mask_virtual, std_col] = predictions[target_col]['std']
        
        # Add acquisition scores
        results_df['acquisition_score'] = np.nan
        results_df.loc[mask_virtual, 'acquisition_score'] = final_acquisition_scores
        
        # Pareto analysis
        print("Performing Pareto analysis...")
        if len(target_columns) > 1:
            # Build objective matrix for Pareto analysis
            objective_matrix = []
            for i, target_col in enumerate(target_columns):
                mean_col = f'{target_col}_predicted_mean'
                values = results_df.loc[mask_virtual, mean_col].values
                
                # Convert to costs (lower is better) for Pareto analysis
                if goal_directions[i] == 'maximize':
                    values = -values  # Convert maximization to minimization
                
                objective_matrix.append(values)
            
            objective_matrix = np.column_stack(objective_matrix)
            
            # Find Pareto optimal points
            is_pareto = self._is_pareto_optimal(objective_matrix)
            
            # Add Pareto information to results
            results_df['is_pareto_optimal'] = False
            results_df.loc[mask_virtual, 'is_pareto_optimal'] = is_pareto
            
            n_pareto = np.sum(is_pareto)
            print(f"Pareto analysis: {n_pareto} optimal solutions")
        else:
            results_df['is_pareto_optimal'] = True  # All points are "Pareto optimal" for single objective
        
        # Sort by acquisition score
        results_df = results_df.sort_values('acquisition_score', ascending=False, na_position='last')
        results_df = results_df.reset_index(drop=True)
        
        self.is_fitted = True
        
        print(f"Multi-objective optimization completed: {len(results_df)} results")
        
        return results_df
    
    def run(self, training_df, virtual_df, target_column, feature_columns, 
            goal='maximize', model_config=None, acquisition_config=None, 
            n_iterations_bootstrap=100):
     
        return self._run_single_iteration(
            training_df, virtual_df, target_column, feature_columns,
            goal, model_config, acquisition_config, n_iterations_bootstrap
        )
    
    def run_multi_objective(self, training_df, virtual_df, target_columns, feature_columns, 
                          goal_directions, model_config=None, acquisition_config=None, 
                          n_iterations_bootstrap=30, n_scalarizations=8):
      
        return self._run_multi_objective_single_iteration(
            training_df, virtual_df, target_columns, feature_columns,
            goal_directions, model_config, acquisition_config, 
            n_iterations_bootstrap, n_scalarizations
        )
    
    def start_iterative_optimization(self, training_df, virtual_df, target_columns, feature_columns,
                                   goal_directions=None, max_iterations=10, batch_size=5,
                                   model_config=None, acquisition_config=None,
                                   n_iterations_bootstrap=50, n_scalarizations=6,
                                   enable_adaptive_sampling=True, enable_intelligent_stopping=True,
                                   enable_batch_optimization=True, stopping_config=None):
        
        # Normalize inputs
        if isinstance(target_columns, str):
            target_columns = [target_columns]
            
        if goal_directions is None:
            goal_directions = ['maximize'] * len(target_columns)
        elif isinstance(goal_directions, str):
            goal_directions = [goal_directions] * len(target_columns)
            
        if len(goal_directions) != len(target_columns):
            raise ValueError("Number of goal_directions must match number of target_columns")
            
        # Initialize stopping configuration
        if stopping_config is None:
            stopping_config = {
                'min_iterations': 3,
                'patience': 3,
                'min_improvement': 0.001,
                'convergence_window': 3
            }
        
        # Initialize iterative state
        self._is_iterative_mode = True
        self._iteration_count = 0
        self._performance_history = []
        self._current_training_data = training_df.copy()
        self._current_virtual_data = virtual_df.copy()
        
        # Initialize acquisition config
        if acquisition_config is None:
            acquisition_config = {}
            
        current_acquisition_config = acquisition_config.copy()
        
        print(f"Starting iterative optimization for {len(target_columns)} target(s)")
        print(f"Targets: {target_columns}")
        print(f"Goals: {goal_directions}")
        print(f"Max iterations: {max_iterations}, Batch size: {batch_size}")
        print(f"Advanced features: Adaptive={enable_adaptive_sampling}, "
              f"Stopping={enable_intelligent_stopping}, BatchOpt={enable_batch_optimization}")
        
        # Main optimization loop
        for iteration in range(max_iterations):
            self._iteration_count = iteration + 1
            
            print(f"\n=== Iteration {self._iteration_count}/{max_iterations} ===")
            
            # Step 1: Adaptive sampling strategy
            if enable_adaptive_sampling and len(self._performance_history) > 0:
                try:
                    # Create predictions_data structure for adaptive sampling
                    predictions_data = {
                        'iteration_count': self._iteration_count,
                        'performance_history': self._performance_history
                    }
                    
                    adaptive_result = self.adaptive_sampling_strategy(
                        predictions_data, self._iteration_count, self._performance_history
                    )
                    
                    # Update acquisition config based on adaptive strategy
                    if adaptive_result and 'acquisition_config' in adaptive_result:
                        current_acquisition_config.update(adaptive_result['acquisition_config'])
                        print(f"Adaptive sampling updated acquisition config: {adaptive_result['acquisition_config']}")
                        
                except Exception as e:
                    print(f"Warning: Adaptive sampling failed: {str(e)}")
            
            # Step 2: Run single iteration analysis
            try:
                if len(target_columns) == 1:
                    # Single objective
                    iteration_results = self._run_single_iteration(
                        self._current_training_data, self._current_virtual_data,
                        target_columns[0], feature_columns, goal_directions[0],
                        model_config, current_acquisition_config, n_iterations_bootstrap
                    )
                else:
                    # Multi-objective
                    iteration_results = self._run_multi_objective_single_iteration(
                        self._current_training_data, self._current_virtual_data,
                        target_columns, feature_columns, goal_directions,
                        model_config, current_acquisition_config, 
                        n_iterations_bootstrap, n_scalarizations
                    )
                    
            except Exception as e:
                print(f"Error in iteration {self._iteration_count}: {str(e)}")
                yield {
                    'iteration': self._iteration_count,
                    'recommendations': pd.DataFrame(),
                    'performance_metrics': None,
                    'should_stop': True,
                    'stop_reason': f'Analysis failed: {str(e)}',
                    'acquisition_config': current_acquisition_config
                }
                break
            
            # Step 3: Optimized batch sampling
            if enable_batch_optimization and len(iteration_results) > batch_size:
                try:
                    # Prepare predictions data for batch sampling
                    predictions_data = {
                        'acquisition_values': iteration_results['acquisition_score'].values,
                        'uncertainty': iteration_results.get('uncertainty_std', 
                                                            iteration_results.get(f'{target_columns[0]}_uncertainty_std')),
                        'predicted_mean': iteration_results.get('predicted_mean', 
                                                              iteration_results.get(f'{target_columns[0]}_predicted_mean'))
                    }
                    
                    # Apply optimized batch sampling
                    batch_indices = self.optimized_batch_sampling(
                        predictions_data, batch_size, 
                        diversity_weight=0.3, uncertainty_threshold=0.1
                    )
                    
                    recommendations = iteration_results.iloc[batch_indices].copy()
                    print(f"Optimized batch sampling selected {len(recommendations)} diverse samples")
                    
                except Exception as e:
                    print(f"Warning: Batch optimization failed, using top samples: {str(e)}")
                    recommendations = iteration_results.head(batch_size).copy()
            else:
                recommendations = iteration_results.head(batch_size).copy()
            
            # Step 4: Calculate performance metrics
            try:
                performance_metrics = self.assess_model_reliability(
                    self._current_training_data, target_columns, feature_columns
                )
                
                # Store performance history
                if isinstance(performance_metrics, dict):
                    # Multi-objective: use average R² score
                    avg_r2 = np.mean([score for score in performance_metrics.values() if score is not None])
                else:
                    # Single objective: use the R² score directly
                    avg_r2 = performance_metrics if performance_metrics is not None else 0.0
                
                self._performance_history.append(avg_r2)
                
            except Exception as e:
                print(f"Warning: Performance assessment failed: {str(e)}")
                performance_metrics = None
            
            # Step 5: Intelligent stopping criteria
            should_stop = False
            stop_reason = None
            
            if enable_intelligent_stopping and len(self._performance_history) >= stopping_config['min_iterations']:
                try:
                    stopping_result = self.intelligent_stopping_criteria(
                        self._performance_history,
                        min_iterations=stopping_config['min_iterations'],
                        patience=stopping_config['patience'],
                        min_improvement=stopping_config['min_improvement'],
                        convergence_window=stopping_config['convergence_window']
                    )
                    
                    if stopping_result['should_stop']:
                        should_stop = True
                        stop_reason = stopping_result['reason']
                        print(f"Intelligent stopping triggered: {stop_reason}")
                        
                except Exception as e:
                    print(f"Warning: Stopping criteria evaluation failed: {str(e)}")
            
            # Step 6: Simulate adding recommendations to training data
            # (In real application, this would be done after experimental validation)
            if len(recommendations) > 0:
                # For simulation, we can add the predicted values as "new experimental data"
                new_training_rows = recommendations[feature_columns].copy()
                
                # Add predicted values as simulated experimental results
                for i, target_col in enumerate(target_columns):
                    if len(target_columns) == 1:
                        pred_col = 'predicted_mean'
                    else:
                        pred_col = f'{target_col}_predicted_mean'
                    
                    if pred_col in recommendations.columns:
                        new_training_rows[target_col] = recommendations[pred_col]
                
                # Update training data for next iteration
                self._current_training_data = pd.concat([
                    self._current_training_data, new_training_rows
                ], ignore_index=True)
                
                # Remove recommended samples from virtual data
                virtual_indices = recommendations.index
                self._current_virtual_data = self._current_virtual_data.drop(virtual_indices).reset_index(drop=True)
                
                print(f"Updated training data: {len(self._current_training_data)} samples")
                print(f"Remaining virtual data: {len(self._current_virtual_data)} samples")
            
            # Yield iteration results
            yield {
                'iteration': self._iteration_count,
                'recommendations': recommendations,
                'performance_metrics': performance_metrics,
                'should_stop': should_stop,
                'stop_reason': stop_reason,
                'acquisition_config': current_acquisition_config,
                'training_size': len(self._current_training_data),
                'virtual_size': len(self._current_virtual_data)
            }
            
            # Check stopping conditions
            if should_stop:
                break
                
            if len(self._current_virtual_data) < batch_size:
                print("Insufficient virtual data for next iteration")
                break
        
        print(f"\nIterative optimization completed after {self._iteration_count} iterations")
        self._is_iterative_mode = False
    
    def get_optimization_state(self):
        """
        Get current state of iterative optimization.
        
        Returns:
        --------
        state : dict
            Dictionary containing current optimization state
        """
        return {
            'is_iterative_mode': self._is_iterative_mode,
            'iteration_count': self._iteration_count,
            'performance_history': self._performance_history.copy(),
            'training_data_size': len(self._current_training_data) if self._current_training_data is not None else 0,
            'virtual_data_size': len(self._current_virtual_data) if self._current_virtual_data is not None else 0
        }
    
    def reset_optimization_state(self):
        """
        Reset the iterative optimization state.
        """
        self._is_iterative_mode = False
        self._iteration_count = 0
        self._performance_history = []
        self._current_training_data = None
        self._current_virtual_data = None
        print("Optimization state reset")
    
    def get_feature_importance(self, objective=None):
        """
        Get feature importance from trained models.
        
        Parameters:
        -----------
        objective : str, optional
            Specific objective to get importance for. If None, returns average importance.
            
        Returns:
        --------
        importance : pandas.Series or dict
            Feature importance(s)
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before getting feature importance")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        if objective is not None:
            # Single objective
            if objective in self.models:
                model = self.models[objective]
            elif self.model is not None:
                model = self.model
            else:
                raise ValueError(f"Model for objective '{objective}' not found")
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models: RandomForest, XGBoost, LightGBM, CatBoost, ExtraTrees
                importance = model.feature_importances_
            elif self.model_type == 'GaussianProcess':
                # Gaussian Process: use inverse length scales
                if hasattr(model.kernel_, 'length_scale'):
                    length_scales = model.kernel_.length_scale
                    if np.isscalar(length_scales):
                        length_scales = np.full(len(self.feature_names), length_scales)
                    importance = 1.0 / (length_scales + 1e-10)
                    importance = importance / importance.sum()
                else:
                    importance = np.ones(len(self.feature_names)) / len(self.feature_names)
            elif self.model_type == 'SVR':
                # Support Vector Regression: use feature weights (for linear kernel) or uniform
                if hasattr(model, 'coef_') and model.coef_ is not None:
                    importance = np.abs(model.coef_[0])
                    importance = importance / importance.sum()
                else:
                    # For non-linear kernels, feature importance is not directly available
                    importance = np.ones(len(self.feature_names)) / len(self.feature_names)
            elif self.model_type == 'MLPRegressor':
                # Neural Network: approximate importance using first layer weights
                if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
                    # Use mean absolute weights from input to first hidden layer
                    first_layer_weights = model.coefs_[0]  # Shape: (n_features, n_hidden)
                    importance = np.mean(np.abs(first_layer_weights), axis=1)
                    importance = importance / importance.sum()
                else:
                    importance = np.ones(len(self.feature_names)) / len(self.feature_names)
            elif self.model_type == 'BayesianRidge':
                # Bayesian Ridge: use coefficient magnitudes
                if hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_)
                    importance = importance / importance.sum()
                else:
                    importance = np.ones(len(self.feature_names)) / len(self.feature_names)
            else:
                # Fallback for unknown models
                print(f"Warning: Feature importance not directly available for {self.model_type}, using uniform weights")
                importance = np.ones(len(self.feature_names)) / len(self.feature_names)
            
            return pd.Series(importance, index=self.feature_names, name=f'importance_{objective}')
        
        else:
            # Multi-objective or single objective
            if self.models:
                # Multi-objective case
                all_importance = {}
                for obj_name, model in self.models.items():
                    if hasattr(model, 'feature_importances_'):
                        # Tree-based models: RandomForest, XGBoost, LightGBM, CatBoost, ExtraTrees
                        importance = model.feature_importances_
                    elif self.model_type == 'GaussianProcess':
                        # Gaussian Process: use inverse length scales
                        if hasattr(model.kernel_, 'length_scale'):
                            length_scales = model.kernel_.length_scale
                            if np.isscalar(length_scales):
                                length_scales = np.full(len(self.feature_names), length_scales)
                            importance = 1.0 / (length_scales + 1e-10)
                            importance = importance / importance.sum()
                        else:
                            importance = np.ones(len(self.feature_names)) / len(self.feature_names)
                    elif self.model_type == 'SVR':
                        # Support Vector Regression: use feature weights (for linear kernel) or uniform
                        if hasattr(model, 'coef_') and model.coef_ is not None:
                            importance = np.abs(model.coef_[0])
                            importance = importance / importance.sum()
                        else:
                            # For non-linear kernels, feature importance is not directly available
                            importance = np.ones(len(self.feature_names)) / len(self.feature_names)
                    elif self.model_type == 'MLPRegressor':
                        # Neural Network: approximate importance using first layer weights
                        if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
                            # Use mean absolute weights from input to first hidden layer
                            first_layer_weights = model.coefs_[0]  # Shape: (n_features, n_hidden)
                            importance = np.mean(np.abs(first_layer_weights), axis=1)
                            importance = importance / importance.sum()
                        else:
                            importance = np.ones(len(self.feature_names)) / len(self.feature_names)
                    elif self.model_type == 'BayesianRidge':
                        # Bayesian Ridge: use coefficient magnitudes
                        if hasattr(model, 'coef_'):
                            importance = np.abs(model.coef_)
                            importance = importance / importance.sum()
                        else:
                            importance = np.ones(len(self.feature_names)) / len(self.feature_names)
                    else:
                        # Fallback for unknown models
                        print(f"Warning: Feature importance not directly available for {self.model_type}, using uniform weights")
                        importance = np.ones(len(self.feature_names)) / len(self.feature_names)
                    
                    all_importance[obj_name] = pd.Series(importance, index=self.feature_names)
                
                return all_importance
            
            elif self.model is not None:
                # Single objective case
                return self.get_feature_importance(objective='single')
            
            else:
                raise ValueError("No models available for feature importance")
    
    def assess_model_reliability(self, training_df, target_columns, feature_columns, cv_folds=5):
        """
        Assess model reliability using cross-validation for all objectives.
        
        Parameters:
        -----------
        training_df : pandas.DataFrame
            Training data
        target_columns : list or str
            Target column name(s)
        feature_columns : list
            Feature column names
        cv_folds : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        reliability_scores : dict or float
            Cross-validation R² scores for each objective
        """
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        
        reliability_scores = {}
        
        # 🔍 添加详细的调试信息
        print(f"\n=== 🔍 MODEL RELIABILITY ASSESSMENT DEBUG ===")
        print(f"Training data shape: {training_df.shape}")
        print(f"Available columns in training_df: {list(training_df.columns)}")
        print(f"Feature columns requested: {feature_columns}")
        print(f"Number of feature columns: {len(feature_columns)}")
        print(f"Target columns: {target_columns}")
        
        # 检查特征列是否都存在
        missing_features = [col for col in feature_columns if col not in training_df.columns]
        if missing_features:
            print(f"❌ ERROR: Missing feature columns: {missing_features}")
            raise ValueError(f"Feature columns not found in training data: {missing_features}")
        
        # 检查目标列是否存在
        missing_targets = [col for col in target_columns if col not in training_df.columns]
        if missing_targets:
            print(f"❌ ERROR: Missing target columns: {missing_targets}")
            raise ValueError(f"Target columns not found in training data: {missing_targets}")
        
        print(f"✅ All requested columns found in training data")
        
        for target_col in target_columns:
            print(f"\n=== Calculating R² for target: {target_col} ===")
            
            # Prepare data
            X = training_df[feature_columns].copy()
            y = training_df[target_col].copy()
            
            print(f"Initial data shape: X={X.shape}, y={y.shape}")
            print(f"Missing values in X: {X.isnull().sum().sum()}")
            print(f"Missing values in y: {y.isnull().sum()}")
            
            # 🔍 添加特征列详细信息
            print(f"Feature columns actually used: {list(X.columns)}")
            print(f"Expected {len(feature_columns)} features, got {len(X.columns)} features")
            if len(feature_columns) != len(X.columns):
                print(f"⚠️ WARNING: Feature count mismatch!")
                print(f"Requested: {feature_columns}")
                print(f"Actually used: {list(X.columns)}")
                print(f"Missing: {set(feature_columns) - set(X.columns)}")
                print(f"Extra: {set(X.columns) - set(feature_columns)}")
            
            # Remove missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            print(f"After removing missing values: X={X.shape}, y={y.shape}")
            
            if len(X) == 0:
                print(f"ERROR: No valid data points after removing missing values!")
                reliability_scores[target_col] = None
                continue
            
            if len(X) == 1:
                print(f"WARNING: Only 1 data point available. R² cannot be calculated meaningfully.")
                reliability_scores[target_col] = None
                continue
            
            # Calculate R² using cross-validation (standard approach for adequate data)
            try:
                scaler_X = StandardScaler()
                X_scaled = scaler_X.fit_transform(X)
                model = self._create_model()
                
                print(f"Calculating R² with {len(X)} total samples")
                print(f"Target range: [{y.min():.4f}, {y.max():.4f}], mean: {y.mean():.4f}")
                
                # Use appropriate CV strategy based on data size
                if len(X) >= cv_folds:
                    # Standard k-fold cross-validation
                    print(f"Using {cv_folds}-fold cross-validation")
                    cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
                    mean_r2 = cv_scores.mean()
                    std_r2 = cv_scores.std()
                    
                    print(f"CV scores: {cv_scores}")
                    print(f"CV R² = {mean_r2:.6f} ± {std_r2:.6f}")
                    
                    reliability_scores[target_col] = mean_r2
                    
                elif len(X) >= 3:
                    # For smaller datasets, use leave-one-out or stratified CV
                    from sklearn.model_selection import LeaveOneOut
                    print(f"Using Leave-One-Out CV for smaller dataset (n={len(X)})")
                    
                    loo_cv = LeaveOneOut()
                    loo_scores = cross_val_score(model, X_scaled, y, cv=loo_cv, scoring='r2')
                    mean_r2 = loo_scores.mean()
                    std_r2 = loo_scores.std()
                    
                    print(f"LOO CV scores: {loo_scores}")
                    print(f"LOO R² = {mean_r2:.6f} ± {std_r2:.6f}")
                    
                    reliability_scores[target_col] = mean_r2
                
                else:
                    # Fallback for very small datasets (should be rare in your case)
                    print(f"Dataset too small for CV (n={len(X)}), using training R² with caution")
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                    r2 = r2_score(y, y_pred)
                    
                    print(f"Training R² = {r2:.6f} (note: may be optimistic)")
                    reliability_scores[target_col] = r2
                    
            except Exception as e:
                print(f"ERROR during R² calculation: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Even if there's an error, try to provide a basic score
                try:
                    # Simple correlation-based R² as fallback
                    if len(np.unique(y)) > 1:  # Check if y has variance
                        # Use a very simple linear model as fallback
                        from sklearn.linear_model import LinearRegression
                        simple_model = LinearRegression()
                        simple_model.fit(X_scaled, y)
                        y_pred_simple = simple_model.predict(X_scaled)
                        fallback_r2 = r2_score(y, y_pred_simple)
                        print(f"Fallback R² (Linear Regression): {fallback_r2:.6f}")
                        reliability_scores[target_col] = fallback_r2
                    else:
                        print(f"Target has no variance (all values the same). Setting R² = 0")
                        reliability_scores[target_col] = 0.0
                except:
                    print(f"Fallback calculation also failed. Setting R² = None")
                    reliability_scores[target_col] = None
        
        print(f"\n=== R² Calculation Summary ===")
        for target, score in reliability_scores.items():
            if score is not None:
                print(f"{target}: R² = {score:.6f}")
            else:
                print(f"{target}: R² = None (calculation failed)")
        
        if len(reliability_scores) == 1:
            return next(iter(reliability_scores.values()))
        else:
            return reliability_scores
    
    def generate_candidate_set(self, feature_ranges, method='latin_hypercube', 
                             n_samples=1000, grid_points=None, constraints=None, 
                             random_seed=None, **method_params):
        """
        Generate candidate set using various sampling strategies.
        
        Parameters:
        -----------
        feature_ranges : dict
            Dictionary mapping feature names to [min, max] ranges.
            Example: {'feature1': [0.1, 0.9], 'feature2': [800, 1200]}
        method : str, default='latin_hypercube'
            Sampling method: 'grid_search', 'random_sampling', 'latin_hypercube', 'sobol', 'halton', 'cvt', 'maximin'
        n_samples : int, default=1000
            Number of samples to generate (for random_sampling, latin_hypercube, sobol, halton, cvt, maximin)
        grid_points : dict, optional
            For grid_search: number of points per feature.
            Example: {'feature1': 5, 'feature2': 3} generates 5*3=15 combinations
        constraints : list of str, optional
            List of constraint expressions to filter candidates.
            Example: ['feature1 + feature2 <= 1.0', 'feature2 > 500']
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        candidate_df : pandas.DataFrame
            Generated candidate set with specified features
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        feature_names = list(feature_ranges.keys())
        n_features = len(feature_names)
        
        if n_features == 0:
            raise ValueError("At least one feature range must be specified")
        
        print(f"Generating candidate set using {method} method...")
        print(f"Features: {feature_names}")
        print(f"Ranges: {feature_ranges}")
        
        if method == 'grid_search':
            # Grid Search Implementation
            if grid_points is None:
                # Default: 10 points per feature for up to 3 features, fewer for higher dimensions
                if n_features <= 3:
                    default_points = 10
                elif n_features <= 5:
                    default_points = 5
                else:
                    default_points = 3
                grid_points = {name: default_points for name in feature_names}
            
            # Generate grid points for each feature
            feature_grids = []
            total_combinations = 1
            
            for feature_name in feature_names:
                min_val, max_val = feature_ranges[feature_name]
                n_points = grid_points.get(feature_name, 10)
                
                if n_points == 1:
                    grid = np.array([min_val])
                else:
                    grid = np.linspace(min_val, max_val, n_points)
                
                feature_grids.append(grid)
                total_combinations *= len(grid)
            
            print(f"Grid configuration: {[(name, len(grid)) for name, grid in zip(feature_names, feature_grids)]}")
            print(f"Total combinations: {total_combinations}")
            
            if total_combinations > 100000:
                print(f"Warning: Large number of combinations ({total_combinations}). Consider reducing grid_points.")
            
            # Generate all combinations using Cartesian product
            combinations = list(itertools.product(*feature_grids))
            candidate_array = np.array(combinations)
            
        elif method == 'random_sampling':
            # Random Sampling Implementation
            print(f"Generating {n_samples} random samples...")
            
            candidate_array = np.zeros((n_samples, n_features))
            
            for i, feature_name in enumerate(feature_names):
                min_val, max_val = feature_ranges[feature_name]
                candidate_array[:, i] = np.random.uniform(min_val, max_val, n_samples)
        
        elif method == 'latin_hypercube':
            # Latin Hypercube Sampling Implementation
            scramble = method_params.get('scramble', True)
            print(f"Generating {n_samples} Latin Hypercube samples (scramble={scramble})...")
            
            # Use scipy's quasi-Monte Carlo for Latin Hypercube Sampling
            sampler = qmc.LatinHypercube(d=n_features, scramble=scramble, seed=random_seed)
            lhs_samples = sampler.random(n=n_samples)
            
            # Transform from [0,1] to actual feature ranges
            candidate_array = np.zeros((n_samples, n_features))
            
            for i, feature_name in enumerate(feature_names):
                min_val, max_val = feature_ranges[feature_name]
                candidate_array[:, i] = min_val + lhs_samples[:, i] * (max_val - min_val)
        
        elif method == 'sobol':
            # Sobol Sequence Implementation
            scramble = method_params.get('scramble', True)
            skip = method_params.get('skip', 0)
            print(f"Generating {n_samples} Sobol sequence samples (scramble={scramble}, skip={skip})...")
            
            # Use Sobol sequence for better uniformity than LHS
            sampler = qmc.Sobol(d=n_features, scramble=scramble, seed=random_seed)
            if skip > 0:
                sampler.fast_forward(skip)
            sobol_samples = sampler.random(n=n_samples)
            
            # Transform from [0,1] to actual feature ranges
            l_bounds = [feature_ranges[name][0] for name in feature_names]
            u_bounds = [feature_ranges[name][1] for name in feature_names]
            candidate_array = qmc.scale(sobol_samples, l_bounds, u_bounds)
            
        elif method == 'halton':
            # Halton Sequence Implementation
            scramble = method_params.get('scramble', True)
            print(f"Generating {n_samples} Halton sequence samples (scramble={scramble})...")
            
            # Use Halton sequence for good uniformity with fast generation
            sampler = qmc.Halton(d=n_features, scramble=scramble, seed=random_seed)
            halton_samples = sampler.random(n=n_samples)
            
            # Transform from [0,1] to actual feature ranges
            l_bounds = [feature_ranges[name][0] for name in feature_names]
            u_bounds = [feature_ranges[name][1] for name in feature_names]
            candidate_array = qmc.scale(halton_samples, l_bounds, u_bounds)
            
        elif method == 'cvt':
            # Centroidal Voronoi Tessellation Implementation with performance limits
            iterations = method_params.get('iterations', 20)
            oversample_factor = method_params.get('oversample_factor', 10)
            print(f"Generating {n_samples} Centroidal Voronoi Tessellation (CVT) samples (iter={iterations}, oversample={oversample_factor}x)...")
            
            # Enforce limits to prevent UI freezing
            if n_samples > 2000:
                print(f"Warning: CVT with {n_samples} samples may be slow. Reducing to 2000 for performance.")
                n_samples = 2000
            
            # Generate many random points for initial clustering with controlled size
            pool_size = min(50000, n_samples * oversample_factor)  # Limit total pool size
            initial_points = np.random.rand(pool_size, n_features)
            
            # Use K-Means to find cluster centers (CVT approximation) with timeout protection
            try:
                print(f"CVT: K-means clustering {pool_size} points into {n_samples} clusters...")
                candidate_array_norm, _ = kmeans(initial_points, n_samples, iter=iterations)
                
                # Transform from [0,1] to actual feature ranges
                l_bounds = [feature_ranges[name][0] for name in feature_names]
                u_bounds = [feature_ranges[name][1] for name in feature_names]
                candidate_array = qmc.scale(candidate_array_norm, l_bounds, u_bounds)
                print("CVT generation completed successfully")
                
            except Exception as e:
                print(f"Warning: CVT generation failed ({str(e)}), falling back to Latin Hypercube")
                # Fallback to LHS if CVT fails
                sampler = qmc.LatinHypercube(d=n_features, seed=random_seed)
                lhs_samples = sampler.random(n=n_samples)
                l_bounds = [feature_ranges[name][0] for name in feature_names]
                u_bounds = [feature_ranges[name][1] for name in feature_names]
                candidate_array = qmc.scale(lhs_samples, l_bounds, u_bounds)
            
        elif method == 'maximin':
            # Maximin Design Implementation with performance limits
            pool_factor = method_params.get('pool_factor', 20)
            print(f"Generating {n_samples} Maximin design samples (pool_factor={pool_factor}x)...")
            
            # Enforce stricter limits to prevent UI freezing
            if n_samples > 300:
                print(f"Warning: Maximin with {n_samples} samples may be slow. Reducing to 300 for performance.")
                n_samples = 300
            
            if n_features > 10:
                print(f"Warning: Maximin with {n_features} features may be slow. Consider using LHS instead.")
            
            # Generate a smaller, more manageable pool
            pool_size = min(10000, max(500, pool_factor * n_samples))  # Use user-defined pool factor
            pool = np.random.rand(pool_size, n_features)
            
            # Transform pool to actual feature ranges
            l_bounds = [feature_ranges[name][0] for name in feature_names]
            u_bounds = [feature_ranges[name][1] for name in feature_names]
            pool = qmc.scale(pool, l_bounds, u_bounds)
            
            # Greedy algorithm for maximin design with early termination
            candidate_list = []
            
            # Select first point randomly
            first_idx = np.random.randint(pool_size)
            candidate_list.append(pool[first_idx])
            pool = np.delete(pool, first_idx, axis=0)
            
            # Select remaining points to maximize minimum distance
            max_iterations = min(n_samples - 1, 500)  # Limit iterations
            
            for i in range(1, max_iterations + 1):
                if len(pool) == 0:
                    print(f"Warning: Pool exhausted at {i} samples. Generating remaining with LHS.")
                    # Fill remaining with LHS if pool is exhausted
                    remaining = n_samples - i
                    sampler = qmc.LatinHypercube(d=n_features, seed=random_seed + i)
                    lhs_samples = sampler.random(n=remaining)
                    lhs_scaled = qmc.scale(lhs_samples, l_bounds, u_bounds)
                    candidate_list.extend(lhs_scaled.tolist())
                    break
                
                try:
                    # Calculate minimum distance from each pool point to selected points
                    distances = cdist(pool, candidate_list)
                    min_distances = np.min(distances, axis=1)
                    
                    # Select point with maximum minimum distance
                    best_idx = np.argmax(min_distances)
                    candidate_list.append(pool[best_idx])
                    pool = np.delete(pool, best_idx, axis=0)
                    
                    # Progress indicator for large samples
                    if i % max(1, max_iterations // 10) == 0:
                        print(f"  Selected {i}/{max_iterations} points...")
                        
                except MemoryError:
                    print(f"Memory error at {i} samples. Switching to LHS for remaining samples.")
                    remaining = n_samples - i
                    sampler = qmc.LatinHypercube(d=n_features, seed=random_seed + i)
                    lhs_samples = sampler.random(n=remaining)
                    lhs_scaled = qmc.scale(lhs_samples, l_bounds, u_bounds)
                    candidate_list.extend(lhs_scaled.tolist())
                    break
            
            candidate_array = np.array(candidate_list[:n_samples])  # Ensure exact count
        
        else:
            raise ValueError(f"Unsupported sampling method: {method}. "
                           f"Choose from: 'grid_search', 'random_sampling', 'latin_hypercube', 'sobol', 'halton', 'cvt', 'maximin'")
        
        # Create DataFrame
        candidate_df = pd.DataFrame(candidate_array, columns=feature_names)
        
        print(f"Generated {len(candidate_df)} initial candidates")
        
        # Apply constraints if specified
        if constraints:
            print(f"Applying {len(constraints)} constraints...")
            initial_count = len(candidate_df)
            
            # Get allowed feature names from the DataFrame
            allowed_names = list(candidate_df.columns)
            
            for constraint in constraints:
                try:
                    # Validate constraint string before evaluation
                    self._validate_constraint_string(constraint, allowed_names)
                    
                    # Create a safe evaluation environment with only the candidate data
                    # This allows expressions like "feature1 + feature2 <= 1.0"
                    mask = candidate_df.eval(constraint)
                    candidate_df = candidate_df[mask].reset_index(drop=True)
                    
                    remaining_count = len(candidate_df)
                    filtered_count = initial_count - remaining_count
                    print(f"Constraint '{constraint}': filtered {filtered_count} candidates, {remaining_count} remaining")
                    initial_count = remaining_count
                    
                except ValueError as e:
                    print(f"Warning: Constraint validation failed for '{constraint}': {str(e)}")
                    continue
                except Exception as e:
                    print(f"Warning: Could not apply constraint '{constraint}': {str(e)}")
                    continue
            
            # 如果约束条件过滤掉了所有样本，自动触发迭代生成
            if len(candidate_df) == 0:
                print("\n⚠️ 所有候选样本都被约束条件过滤，启动智能迭代生成...")
                return self._generate_with_constraints_iterative(
                    feature_ranges={name: [candidate_array[:, i].min(), candidate_array[:, i].max()] 
                                   for i, name in enumerate(feature_names)},
                    method=method,
                    target_samples=n_samples,
                    grid_points=grid_points,
                    constraints=constraints,
                    random_seed=random_seed,
                    progress_callback=getattr(self, '_progress_callback', None),
                    **method_params
                )
        
        if len(candidate_df) == 0:
            raise ValueError("All candidates were filtered out by constraints. "
                           "Please relax constraints or expand feature ranges.")
        
        print(f"Final candidate set: {len(candidate_df)} samples")
        print(f"Feature statistics:")
        for feature_name in feature_names:
            values = candidate_df[feature_name]
            print(f"  {feature_name}: [{values.min():.4f}, {values.max():.4f}], "
                  f"mean={values.mean():.4f}, std={values.std():.4f}")
        
        return candidate_df
    
    def generate_candidate_set_from_training(self, training_df, feature_columns, 
                                           method='latin_hypercube', n_samples=1000,
                                           expansion_factor=1.2, grid_points=None,
                                           constraints=None, random_seed=None, **method_params):
        """
        Generate candidate set based on feature ranges observed in training data.
        
        This is a convenience method that automatically determines feature ranges
        from the training data and optionally expands them.
        
        Parameters:
        -----------
        training_df : pandas.DataFrame
            Training data to determine feature ranges from
        feature_columns : list
            List of feature column names to include
        method : str, default='latin_hypercube'
            Sampling method: 'grid_search', 'random_sampling', 'latin_hypercube'
        n_samples : int, default=1000
            Number of samples to generate
        expansion_factor : float, default=1.2
            Factor to expand the observed ranges. 1.0 = no expansion, 1.2 = 20% expansion
        grid_points : dict, optional
            For grid_search: number of points per feature
        constraints : list of str, optional
            List of constraint expressions
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        candidate_df : pandas.DataFrame
            Generated candidate set
        """
        print("Analyzing training data to determine feature ranges...")
        
        # Validate feature columns
        missing_features = set(feature_columns) - set(training_df.columns)
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in training data")
        
        # Calculate feature ranges from training data
        feature_ranges = {}
        
        for feature in feature_columns:
            values = training_df[feature].dropna()
            
            if len(values) == 0:
                raise ValueError(f"No valid values found for feature '{feature}'")
            
            min_val = values.min()
            max_val = values.max()
            
            if min_val == max_val:
                # Handle constant features
                print(f"Warning: Feature '{feature}' has constant value {min_val}. "
                      f"Using small range around this value.")
                range_width = abs(min_val) * 0.1 if min_val != 0 else 0.1
                min_val = min_val - range_width
                max_val = max_val + range_width
            else:
                # Expand range if requested
                if expansion_factor != 1.0:
                    center = (min_val + max_val) / 2
                    half_range = (max_val - min_val) / 2
                    expanded_half_range = half_range * expansion_factor
                    min_val = center - expanded_half_range
                    max_val = center + expanded_half_range
            
            feature_ranges[feature] = [min_val, max_val]
            print(f"  {feature}: [{min_val:.4f}, {max_val:.4f}] "
                  f"(expansion factor: {expansion_factor})")
        
        # Generate candidate set using the determined ranges
        # 如果有约束条件，使用智能迭代生成
        if constraints:
            return self._generate_with_constraints_iterative(
                feature_ranges=feature_ranges,
                method=method,
                target_samples=n_samples,
                grid_points=grid_points,
                constraints=constraints,
                random_seed=random_seed,
                progress_callback=getattr(self, '_progress_callback', None),
                **method_params
            )
        else:
            return self.generate_candidate_set(
                feature_ranges=feature_ranges,
                method=method,
                n_samples=n_samples,
                grid_points=grid_points,
                constraints=constraints,
                random_seed=random_seed,
                **method_params
            )
    
    def _generate_with_constraints_iterative(self, feature_ranges, method, target_samples, 
                                           grid_points=None, constraints=None, random_seed=None,
                                           progress_callback=None, **method_params):
        """
        迭代生成候选集，每次生成固定数量样本并应用约束，直到获得足够样本
        
        Parameters:
        -----------
        progress_callback : callable, optional
            回调函数用于更新进度，格式: callback(iteration, valid_samples, total_attempts, success_rate, batch_size, target_samples)
        """
        all_valid_candidates = []
        total_attempts = 0
        iteration = 0
        # 移除最大迭代次数限制，直到达到目标为止
        
        # 每次迭代生成的样本数，可以自定义或使用默认值
        if self._batch_size is not None:
            batch_size = self._batch_size
        else:
            batch_size = min(5000, max(target_samples, 1000))  # 最少1000，最多5000
        
        while len(all_valid_candidates) < target_samples:
            iteration += 1
            
            print(f"Generating {batch_size} {method} samples...")
            
            # 生成一批候选样本（不应用约束）
            try:
                batch_candidates = self.generate_candidate_set(
                    feature_ranges=feature_ranges,
                    method=method,
                    n_samples=batch_size,
                    grid_points=grid_points,
                    constraints=None,  # 先不应用约束
                    random_seed=random_seed + iteration if random_seed else None,
                    **method_params
                )
                
                total_attempts += len(batch_candidates)
                print(f"Generated {len(batch_candidates)} initial candidates")
                
                # 应用约束条件过滤 - 使用原来的逐个约束应用方式
                valid_candidates = self._apply_constraints_original_style(batch_candidates, constraints)
                
                if len(valid_candidates) > 0:
                    # 收集有效样本
                    needed = target_samples - len(all_valid_candidates)
                    candidates_to_add = valid_candidates.iloc[:needed]
                    all_valid_candidates.extend(candidates_to_add.values.tolist())
                    
                    print(f"Added {len(candidates_to_add)} valid candidates to collection")
                    print(f"Progress: {len(all_valid_candidates)}/{target_samples} candidates collected")
                    
                    # 更新进度回调
                    if progress_callback:
                        success_rate = len(valid_candidates) / len(batch_candidates)
                        progress_callback(iteration, len(all_valid_candidates), total_attempts, 
                                        success_rate, batch_size, target_samples)
                else:
                    print("No valid candidates found in this iteration")
                    print(f"Progress: {len(all_valid_candidates)}/{target_samples} candidates collected")
                    
                    # 更新进度回调（成功率为0）
                    if progress_callback:
                        progress_callback(iteration, len(all_valid_candidates), total_attempts, 
                                        0.0, batch_size, target_samples)
                
            except Exception as e:
                print(f"Iteration {iteration} failed: {str(e)}")
                continue
        
        # 创建最终结果 - 由于移除了迭代限制，一定能达到目标数量
        final_candidates_array = np.array(all_valid_candidates[:target_samples])
        final_candidates = pd.DataFrame(final_candidates_array, columns=list(feature_ranges.keys()))
        print(f"Successfully generated {target_samples} candidates with constraints after {iteration} iterations")
        
        return final_candidates
    
    def _apply_constraints_original_style(self, candidate_df, constraints):
        """
        Apply constraints with original style formatting, including security validation.
        """
        if not constraints:
            return candidate_df
        
        result_df = candidate_df.copy()
        print(f"Applying {len(constraints)} constraints...")
        initial_count = len(result_df)
        
        # Get allowed feature names from the DataFrame
        allowed_names = list(candidate_df.columns)
        
        for constraint in constraints:
            try:
                # Validate constraint string before evaluation
                self._validate_constraint_string(constraint, allowed_names)
                
                before_count = len(result_df)
                mask = result_df.eval(constraint)
                result_df = result_df[mask].reset_index(drop=True)
                after_count = len(result_df)
                filtered_count = before_count - after_count
                
                print(f"Constraint '{constraint}': filtered {filtered_count} candidates, {after_count} remaining")
                
                if after_count == 0:
                    break
                    
            except ValueError as e:
                print(f"Warning: Constraint validation failed for '{constraint}': {str(e)}")
                continue
            except Exception as e:
                print(f"Warning: Could not apply constraint '{constraint}': {str(e)}")
                continue
        
        return result_df
    
    def _validate_constraint_string(self, constraint_string, allowed_names):
        """
        Validate constraint string to prevent code injection attacks.
        
        Parameters:
        -----------
        constraint_string : str
            The constraint expression to validate
        allowed_names : list
            List of allowed variable names (feature names)
            
        Returns:
        --------
        bool : True if constraint is safe, raises ValueError if unsafe
        """
        import re
        
        # Remove whitespace for easier parsing
        cleaned = re.sub(r'\s+', '', constraint_string)
        
        # Define allowed patterns
        allowed_operators = r'[+\-*/()&|<>=!]'
        allowed_numbers = r'\d+\.?\d*'
        
        # Create pattern for allowed names (escape special regex characters)
        escaped_names = [re.escape(name) for name in allowed_names]
        allowed_names_pattern = '|'.join(escaped_names) if escaped_names else 'NONE'
        
        # Complete allowed pattern
        allowed_pattern = f'^({allowed_names_pattern}|{allowed_numbers}|{allowed_operators})+$'
        
        # Check if the constraint matches only allowed patterns
        if not re.match(allowed_pattern, cleaned):
            raise ValueError(f"Constraint contains unsafe characters: '{constraint_string}'")
        
        # Additional security checks for dangerous patterns
        dangerous_patterns = [
            r'__\w+__',  # Double underscore methods
            r'import\s+',  # Import statements
            r'exec\s*\(',  # Exec function
            r'eval\s*\(',  # Eval function
            r'open\s*\(',  # File operations
            r'os\.',  # OS module
            r'sys\.',  # Sys module
            r'subprocess',  # Subprocess module
            r'[;\n]',  # Multiple statements
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, constraint_string, re.IGNORECASE):
                raise ValueError(f"Constraint contains potentially dangerous pattern: '{constraint_string}'")
        
        return True
    
    def _apply_constraints_safe(self, candidate_df, constraints):
        """
        Safely apply constraints with validation to prevent code injection.
        """
        if not constraints:
            return candidate_df
        
        result_df = candidate_df.copy()
        initial_count = len(result_df)
        
        print(f"   🔍 apply {len(constraints)} constraints...")
        
        # Get allowed feature names from the DataFrame
        allowed_names = list(candidate_df.columns)
        
        for i, constraint in enumerate(constraints):
            try:
                # Validate constraint string before evaluation
                self._validate_constraint_string(constraint, allowed_names)
                
                before_count = len(result_df)
                mask = result_df.eval(constraint)
                result_df = result_df[mask].reset_index(drop=True)
                after_count = len(result_df)
                filtered_count = before_count - after_count
                
                print(f"      constraint {i+1}: '{constraint}' -> filtered {filtered_count}, remaining {after_count}")
                
                if after_count == 0:
                    print(f"      ⚠️ constraint {i+1} filtered out all samples")
                    break
                    
            except ValueError as e:
                print(f"      ❌ constraint {i+1} validation failed: {str(e)}")
                continue
            except Exception as e:
                print(f"      ❌ constraint {i+1} evaluation failed: {str(e)}")
                continue
        
        return result_df
    
    def optimize_hyperparameters(self, training_df, target_columns, feature_columns, 
                                 optimization_mode='adaptive', cv_folds=5, n_trials=20):
        """
        Optimize model hyperparameters for active learning.
        
        Parameters:
        -----------
        training_df : pandas.DataFrame
            Training data for hyperparameter optimization
        target_columns : list or str
            Target column name(s)
        feature_columns : list
            Feature column names
        optimization_mode : str, default='adaptive'
            'adaptive': Automatic optimization based on data size
            'aggressive': Comprehensive search (for larger datasets)
            'conservative': Limited search (for small datasets)
        cv_folds : int, default=5
            Cross-validation folds for evaluation
        n_trials : int, default=20
            Number of trials for optimization
            
        Returns:
        --------
        best_params : dict
            Optimized hyperparameters
        """
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        
        print(f"Starting hyperparameter optimization in {optimization_mode} mode...")
        
        # Prepare data
        X = training_df[feature_columns].copy()
        
        # Remove missing values for all targets
        mask = ~X.isnull().any(axis=1)
        for target_col in target_columns:
            mask &= ~training_df[target_col].isnull()
        
        X = X[mask]
        data_size = len(X)
        n_features = len(feature_columns)
        
        print(f"Optimization data: {data_size} samples, {n_features} features")
        
        # Adaptive mode: adjust strategy based on data characteristics
        if optimization_mode == 'adaptive':
            if data_size < 20:
                optimization_mode = 'conservative'
                print("Switching to conservative mode due to small dataset")
            elif data_size < 100:
                optimization_mode = 'moderate'
                print("Using moderate optimization for medium dataset")
            else:
                optimization_mode = 'aggressive'
                print("Using aggressive optimization for large dataset")
        
        # Define parameter search spaces based on model type
        if self.model_type == 'RandomForest':
            if optimization_mode == 'conservative':
                param_space = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            elif optimization_mode == 'moderate':
                param_space = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:  # aggressive
                param_space = {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [None, 3, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None]
                }
                
        elif self.model_type == 'XGBoost':
            if optimization_mode == 'conservative':
                param_space = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            elif optimization_mode == 'moderate':
                param_space = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 6, 9, 12],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:  # aggressive
                param_space = {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [3, 6, 9, 12, 15],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
                }
                
        elif self.model_type == 'GaussianProcess':
            if optimization_mode == 'conservative':
                length_scales = [0.1, 1.0, 10.0]
                noise_levels = [1e-6, 1e-5, 1e-4]
            elif optimization_mode == 'moderate':
                length_scales = [0.1, 0.5, 1.0, 2.0, 10.0]
                noise_levels = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
            else:  # aggressive
                length_scales = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                noise_levels = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
            
            # GP parameter space is handled differently
            param_space = {
                'length_scales': length_scales,
                'noise_levels': noise_levels
            }
        
        elif self.model_type == 'LightGBM':
            if optimization_mode == 'conservative':
                param_space = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [-1, 5, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [15, 31, 63]
                }
            elif optimization_mode == 'moderate':
                param_space = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [-1, 3, 5, 10, 15],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [15, 31, 63, 127],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:  # aggressive
                param_space = {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [-1, 3, 5, 10, 15, 20],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'num_leaves': [15, 31, 63, 127, 255],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
                }
                
        elif self.model_type == 'CatBoost':
            if optimization_mode == 'conservative':
                param_space = {
                    'iterations': [50, 100, 200],
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            elif optimization_mode == 'moderate':
                param_space = {
                    'iterations': [50, 100, 200, 300],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5, 7, 9]
                }
            else:  # aggressive
                param_space = {
                    'iterations': [50, 100, 200, 300, 500],
                    'depth': [4, 6, 8, 10, 12],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128, 255]
                }
                
        elif self.model_type == 'ExtraTrees':
            if optimization_mode == 'conservative':
                param_space = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            elif optimization_mode == 'moderate':
                param_space = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:  # aggressive
                param_space = {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [None, 3, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None]
                }
                
        elif self.model_type == 'SVR':
            if optimization_mode == 'conservative':
                param_space = {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1, 0.2]
                }
            elif optimization_mode == 'moderate':
                param_space = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'epsilon': [0.01, 0.1, 0.2, 0.5]
                }
            else:  # aggressive
                param_space = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5]
                }
                
        elif self.model_type == 'MLPRegressor':
            if optimization_mode == 'conservative':
                param_space = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [200, 500]
                }
            elif optimization_mode == 'moderate':
                param_space = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200,), (200, 100)],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'max_iter': [200, 500, 1000],
                    'solver': ['adam', 'lbfgs']
                }
            else:  # aggressive
                param_space = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200,), (200, 100), (300, 150), (100, 100, 50)],
                    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'max_iter': [200, 500, 1000, 2000],
                    'solver': ['adam', 'lbfgs'],
                    'activation': ['relu', 'tanh', 'logistic']
                }
                
        elif self.model_type == 'BayesianRidge':
            if optimization_mode == 'conservative':
                param_space = {
                    'alpha_1': [1e-6, 1e-4, 1e-2],
                    'alpha_2': [1e-6, 1e-4, 1e-2],
                    'lambda_1': [1e-6, 1e-4, 1e-2],
                    'lambda_2': [1e-6, 1e-4, 1e-2]
                }
            elif optimization_mode == 'moderate':
                param_space = {
                    'alpha_1': [1e-8, 1e-6, 1e-4, 1e-2, 1e-1],
                    'alpha_2': [1e-8, 1e-6, 1e-4, 1e-2, 1e-1],
                    'lambda_1': [1e-8, 1e-6, 1e-4, 1e-2, 1e-1],
                    'lambda_2': [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
                }
            else:  # aggressive
                param_space = {
                    'alpha_1': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0],
                    'alpha_2': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0],
                    'lambda_1': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0],
                    'lambda_2': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
                }
        
        else:
            raise ValueError(f"Hyperparameter optimization not implemented for {self.model_type}")
        
        # Perform optimization for each target
        best_params_all = {}
        
        for target_col in target_columns:
            print(f"Optimizing hyperparameters for target: {target_col}")
            
            y = training_df.loc[mask, target_col].copy()
            
            if self.model_type == 'GaussianProcess':
                # Special handling for GP
                best_score = -np.inf
                best_params = None
                
                for length_scale in param_space['length_scales']:
                    for noise_level in param_space['noise_levels']:
                        try:
                            kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
                            model = GaussianProcessRegressor(
                                kernel=kernel,
                                alpha=1e-6,
                                random_state=self.random_state,
                                normalize_y=True
                            )
                            
                            # Cross-validation
                            scaler_X = StandardScaler()
                            X_scaled = scaler_X.fit_transform(X)
                            
                            if len(X) >= cv_folds:
                                scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
                                score = scores.mean()
                            else:
                                model.fit(X_scaled, y)
                                score = model.score(X_scaled, y)
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'kernel': kernel,
                                    'alpha': 1e-6,
                                    'normalize_y': True
                                }
                                
                        except Exception as e:
                            print(f"Warning: GP optimization failed for length_scale={length_scale}, noise_level={noise_level}: {e}")
                            continue
                
            else:
                # Grid search for all other models
                param_grid = list(ParameterGrid(param_space))
                
                # Limit trials if too many combinations
                if len(param_grid) > n_trials:
                    random.seed(self.random_state)
                    param_grid = random.sample(param_grid, n_trials)
                
                print(f"Testing {len(param_grid)} parameter combinations...")
                
                for i, params in enumerate(param_grid):
                    try:
                        # Add fixed parameters
                        full_params = params.copy()
                        
                        # Add random_state for models that support it
                        if self.model_type in ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees', 'MLPRegressor']:
                            if self.model_type == 'CatBoost':
                                full_params['random_state'] = self.random_state
                            else:
                                full_params['random_state'] = self.random_state
                        
                        # Add verbosity control
                        if self.model_type == 'XGBoost':
                            full_params['verbosity'] = 0
                        elif self.model_type == 'LightGBM':
                            full_params['verbosity'] = -1
                            full_params['force_row_wise'] = True
                        elif self.model_type == 'CatBoost':
                            full_params['verbose'] = False
                            full_params['allow_writing_files'] = False
                        
                        # Create model using the factory method
                        model_config = {'params': full_params}
                        model = self._create_model(model_config)
                        
                        # Cross-validation
                        scaler_X = StandardScaler()
                        X_scaled = scaler_X.fit_transform(X)
                        
                        if len(X) >= cv_folds:
                            scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
                            score = scores.mean()
                        else:
                            model.fit(X_scaled, y)
                            score = model.score(X_scaled, y)
                        
                        if score > best_score:
                            best_score = score
                            best_params = full_params
                        
                        if (i + 1) % max(1, len(param_grid) // 5) == 0:
                            print(f"  Progress: {i + 1}/{len(param_grid)}, best R²: {best_score:.4f}")
                            
                    except Exception as e:
                        print(f"Warning: Parameter combination {i+1} failed: {e}")
                        continue
            
            if best_params is not None:
                best_params_all[target_col] = {
                    'params': best_params,
                    'score': best_score
                }
                print(f"Best parameters for {target_col}: R² = {best_score:.4f}")
                print(f"Parameters: {best_params}")
            else:
                print(f"Warning: No valid parameters found for {target_col}")
                best_params_all[target_col] = None
        
        # Return results
        if len(target_columns) == 1:
            result = best_params_all[target_columns[0]]
            return result['params'] if result else None
        else:
            return best_params_all
    
    def auto_tune_model(self, training_df, target_columns, feature_columns, 
                       min_data_threshold=15, retune_interval=10, use_bayesian=True):
        """
        Automatically tune model parameters when conditions are met.
        
        Parameters:
        -----------
        training_df : pandas.DataFrame
            Training data
        target_columns : list or str
            Target column name(s)
        feature_columns : list
            Feature column names
        min_data_threshold : int, default=15
            Minimum data points required for tuning
        retune_interval : int, default=10
            Retune every N new data points
        use_bayesian : bool, default=True
            Use Bayesian optimization if available, otherwise use grid search
            
        Returns:
        --------
        tuned_params : dict or None
            Tuned parameters if tuning was performed
        """
        data_size = len(training_df)
        
        # Check if tuning is warranted
        if data_size < min_data_threshold:
            print(f"Auto-tune skipped: only {data_size} samples (need {min_data_threshold})")
            return None
        
        # Check if it's time to retune (every retune_interval samples)
        if hasattr(self, '_last_tune_size'):
            if data_size - self._last_tune_size < retune_interval:
                print(f"Auto-tune skipped: only {data_size - self._last_tune_size} new samples since last tune")
                return None
        
        print(f"Auto-tuning triggered with {data_size} samples...")
        
        # Choose optimization method
        if use_bayesian and SKOPT_AVAILABLE:
            print("Using Bayesian optimization for hyperparameter tuning...")
            tuned_params = self.optimize_hyperparameters_bayesian(
                training_df=training_df,
                target_columns=target_columns,
                feature_columns=feature_columns,
                optimization_mode='adaptive',
                n_calls=30,  # Reduced for auto-tuning
                verbose=False  # Less verbose for auto-tuning
            )
        else:
            if use_bayesian and not SKOPT_AVAILABLE:
                print("scikit-optimize not available, falling back to grid search...")
            else:
                print("Using grid search for hyperparameter tuning...")
            
            tuned_params = self.optimize_hyperparameters(
                training_df=training_df,
                target_columns=target_columns,
                feature_columns=feature_columns,
                optimization_mode='adaptive'
            )
        
        # Update last tune size
        self._last_tune_size = data_size
        
        return tuned_params
    
    def apply_optimized_parameters(self, optimized_params):
        """
        Apply optimized hyperparameters to the model.
        
        Parameters:
        -----------
        optimized_params : dict
            Optimized hyperparameters from optimize_hyperparameters()
        """
        if optimized_params is not None:
            self._optimized_params = optimized_params
            print(f"Applied optimized parameters: {optimized_params}")
        else:
            print("No optimized parameters to apply")
    
    def optimize_hyperparameters_bayesian(self, training_df, target_columns, feature_columns,
                                         optimization_mode='adaptive', cv_folds=5, n_calls=50, 
                                         optimizer='gp', random_state=None, verbose=True):
        """
        Optimize model hyperparameters using Bayesian optimization.
        
        Parameters:
        -----------
        training_df : pandas.DataFrame
            Training data for hyperparameter optimization
        target_columns : list or str
            Target column name(s)
        feature_columns : list
            Feature column names
        optimization_mode : str, default='adaptive'
            'conservative': Limited search space (for small datasets)
            'moderate': Medium search space
            'aggressive': Full search space (for large datasets)
            'adaptive': Automatic selection based on data size
        cv_folds : int, default=5
            Cross-validation folds for evaluation
        n_calls : int, default=50
            Number of Bayesian optimization iterations
        optimizer : str, default='gp'
            Bayesian optimizer: 'gp' (Gaussian Process), 'forest' (Random Forest), 'gbrt' (Gradient Boosting)
        random_state : int, optional
            Random state for reproducibility
        verbose : bool, default=True
            Print optimization progress
            
        Returns:
        --------
        best_params : dict
            Optimized hyperparameters
        """
        if not SKOPT_AVAILABLE:
            print("Warning: scikit-optimize not available. Falling back to grid search...")
            return self.optimize_hyperparameters(training_df, target_columns, feature_columns,
                                                optimization_mode, cv_folds)
        
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        
        if random_state is None:
            random_state = self.random_state
        
        if verbose:
            print(f"Starting Bayesian hyperparameter optimization using {optimizer} in {optimization_mode} mode...")
        
        # Prepare data
        X = training_df[feature_columns].copy()
        
        # Remove missing values for all targets
        mask = ~X.isnull().any(axis=1)
        for target_col in target_columns:
            mask &= ~training_df[target_col].isnull()
        
        X = X[mask]
        data_size = len(X)
        n_features = len(feature_columns)
        
        if verbose:
            print(f"Optimization data: {data_size} samples, {n_features} features")
        
        # Adaptive mode: adjust strategy based on data characteristics
        if optimization_mode == 'adaptive':
            if data_size < 20:
                optimization_mode = 'conservative'
                n_calls = min(n_calls, 20)  # Reduce calls for small datasets
                if verbose:
                    print("Switching to conservative mode due to small dataset")
            elif data_size < 100:
                optimization_mode = 'moderate'
                n_calls = min(n_calls, 35)
                if verbose:
                    print("Using moderate optimization for medium dataset")
            else:
                optimization_mode = 'aggressive'
                if verbose:
                    print("Using aggressive optimization for large dataset")
        
        # Define search spaces based on model type and optimization mode
        search_space = self._get_bayesian_search_space(optimization_mode)
        
        # Perform optimization for each target
        best_params_all = {}
        
        for target_col in target_columns:
            if verbose:
                print(f"\nOptimizing hyperparameters for target: {target_col}")
            
            y = training_df.loc[mask, target_col].copy()
            
            # Define objective function for Bayesian optimization
            @use_named_args(search_space)
            def objective(**params):
                try:
                    # Special handling for Gaussian Process
                    if self.model_type == 'GaussianProcess':
                        length_scale = params.get('length_scale', 1.0)
                        noise_level = params.get('noise_level', 1e-5)
                        alpha = params.get('alpha', 1e-6)
                        
                        # Create kernel with optimized parameters
                        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
                        model_params = {
                            'kernel': kernel,
                            'alpha': alpha,
                            'normalize_y': True,
                            'random_state': self.random_state
                        }
                        model_config = {'params': model_params}
                    else:
                        # Standard parameter handling for other models
                        model_config = {'params': params}
                    
                    # Create model with current parameters
                    model = self._create_model(model_config)
                    
                    # Cross-validation
                    scaler_X = StandardScaler()
                    X_scaled = scaler_X.fit_transform(X)
                    
                    if len(X) >= cv_folds:
                        scores = cross_val_score(model, X_scaled, y, cv=cv_folds, 
                                               scoring='r2', n_jobs=-1)
                        score = scores.mean()
                    else:
                        # For small datasets, use simple validation
                        model.fit(X_scaled, y)
                        score = model.score(X_scaled, y)
                    
                    # Return negative score (skopt minimizes)
                    return -score
                    
                except Exception as e:
                    if verbose:
                        print(f"Warning: Evaluation failed with params {params}: {e}")
                    return 1.0  # High penalty for failed evaluations
            
            # Choose optimizer
            if optimizer == 'gp':
                optimizer_func = gp_minimize
            elif optimizer == 'forest':
                optimizer_func = forest_minimize
            elif optimizer == 'gbrt':
                optimizer_func = gbrt_minimize
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}. Choose from 'gp', 'forest', 'gbrt'")
            
            # Run Bayesian optimization
            if verbose:
                print(f"Running Bayesian optimization with {n_calls} calls...")
            
            result = optimizer_func(
                func=objective,
                dimensions=search_space,
                n_calls=n_calls,
                random_state=random_state,
                n_initial_points=min(10, n_calls // 3),  # Initial random points
                acq_func='EI'  # Expected Improvement
            )
            
            # Extract best parameters
            best_params = dict(zip([dim.name for dim in search_space], result.x))
            best_score = -result.fun  # Convert back to positive score
            
            # Store results
            best_params_all[target_col] = {
                'params': best_params,
                'score': best_score,
                'n_calls': len(result.func_vals),
                'convergence': result.func_vals
            }
            
            if verbose:
                print(f"Best parameters for {target_col}: R² = {best_score:.6f}")
                print(f"Best parameters: {best_params}")
                print(f"Optimization converged after {len(result.func_vals)} evaluations")
        
        # Return results
        if len(target_columns) == 1:
            result = best_params_all[target_columns[0]]
            return result['params'] if result else None
        else:
            return best_params_all
    
    def _get_bayesian_search_space(self, optimization_mode):
        """
        Define Bayesian optimization search spaces for different models and modes.
        
        Parameters:
        -----------
        optimization_mode : str
            Optimization intensity: 'conservative', 'moderate', 'aggressive'
            
        Returns:
        --------
        search_space : list
            List of skopt dimension objects
        """
        if self.model_type == 'RandomForest':
            if optimization_mode == 'conservative':
                return [
                    Integer(50, 200, name='n_estimators'),
                    Integer(5, 20, name='max_depth'),
                    Integer(2, 10, name='min_samples_split'),
                    Integer(1, 4, name='min_samples_leaf')
                ]
            elif optimization_mode == 'moderate':
                return [
                    Integer(50, 300, name='n_estimators'),
                    Integer(3, 30, name='max_depth'),
                    Integer(2, 20, name='min_samples_split'),
                    Integer(1, 8, name='min_samples_leaf'),
                    Categorical(['sqrt', 'log2'], name='max_features')
                ]
            else:  # aggressive
                return [
                    Integer(50, 500, name='n_estimators'),
                    Integer(3, 50, name='max_depth'),
                    Integer(2, 30, name='min_samples_split'),
                    Integer(1, 15, name='min_samples_leaf'),
                    Categorical(['sqrt', 'log2', None], name='max_features'),
                    Real(0.1, 1.0, name='max_samples')
                ]
                
        elif self.model_type == 'XGBoost':
            if optimization_mode == 'conservative':
                return [
                    Integer(50, 200, name='n_estimators'),
                    Integer(3, 9, name='max_depth'),
                    Real(0.01, 0.3, prior='log-uniform', name='learning_rate')
                ]
            elif optimization_mode == 'moderate':
                return [
                    Integer(50, 300, name='n_estimators'),
                    Integer(3, 12, name='max_depth'),
                    Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
                    Real(0.6, 1.0, name='subsample'),
                    Real(0.6, 1.0, name='colsample_bytree')
                ]
            else:  # aggressive
                return [
                    Integer(50, 500, name='n_estimators'),
                    Integer(3, 15, name='max_depth'),
                    Real(0.001, 0.3, prior='log-uniform', name='learning_rate'),
                    Real(0.5, 1.0, name='subsample'),
                    Real(0.5, 1.0, name='colsample_bytree'),
                    Real(0.001, 10, prior='log-uniform', name='reg_alpha'),
                    Real(0.001, 10, prior='log-uniform', name='reg_lambda')
                ]
                
        elif self.model_type == 'LightGBM':
            if optimization_mode == 'conservative':
                return [
                    Integer(50, 200, name='n_estimators'),
                    Integer(-1, 15, name='max_depth'),
                    Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
                    Integer(10, 100, name='num_leaves')
                ]
            elif optimization_mode == 'moderate':
                return [
                    Integer(50, 300, name='n_estimators'),
                    Integer(-1, 20, name='max_depth'),
                    Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
                    Integer(10, 200, name='num_leaves'),
                    Real(0.6, 1.0, name='subsample'),
                    Real(0.6, 1.0, name='colsample_bytree')
                ]
            else:  # aggressive
                return [
                    Integer(50, 500, name='n_estimators'),
                    Integer(-1, 25, name='max_depth'),
                    Real(0.005, 0.3, prior='log-uniform', name='learning_rate'),
                    Integer(10, 300, name='num_leaves'),
                    Real(0.5, 1.0, name='subsample'),
                    Real(0.5, 1.0, name='colsample_bytree'),
                    Real(0.001, 10, prior='log-uniform', name='reg_alpha'),
                    Real(0.001, 10, prior='log-uniform', name='reg_lambda')
                ]
                
        elif self.model_type == 'CatBoost':
            if optimization_mode == 'conservative':
                return [
                    Integer(50, 200, name='iterations'),
                    Integer(4, 8, name='depth'),
                    Real(0.01, 0.3, prior='log-uniform', name='learning_rate')
                ]
            elif optimization_mode == 'moderate':
                return [
                    Integer(50, 300, name='iterations'),
                    Integer(4, 10, name='depth'),
                    Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
                    Real(1, 10, name='l2_leaf_reg')
                ]
            else:  # aggressive
                return [
                    Integer(50, 500, name='iterations'),
                    Integer(4, 12, name='depth'),
                    Real(0.005, 0.3, prior='log-uniform', name='learning_rate'),
                    Real(1, 20, name='l2_leaf_reg'),
                    Integer(32, 255, name='border_count'),
                    Real(0.5, 1.0, name='subsample')
                ]
                
        elif self.model_type == 'ExtraTrees':
            if optimization_mode == 'conservative':
                return [
                    Integer(50, 200, name='n_estimators'),
                    Integer(5, 20, name='max_depth'),
                    Integer(2, 10, name='min_samples_split'),
                    Integer(1, 4, name='min_samples_leaf')
                ]
            elif optimization_mode == 'moderate':
                return [
                    Integer(50, 300, name='n_estimators'),
                    Integer(3, 30, name='max_depth'),
                    Integer(2, 20, name='min_samples_split'),
                    Integer(1, 8, name='min_samples_leaf'),
                    Categorical(['sqrt', 'log2'], name='max_features')
                ]
            else:  # aggressive
                return [
                    Integer(50, 500, name='n_estimators'),
                    Integer(3, 50, name='max_depth'),
                    Integer(2, 30, name='min_samples_split'),
                    Integer(1, 15, name='min_samples_leaf'),
                    Categorical(['sqrt', 'log2', None], name='max_features'),
                    Real(0.1, 1.0, name='max_samples')
                ]
                
        elif self.model_type == 'SVR':
            if optimization_mode == 'conservative':
                return [
                    Real(0.1, 100, prior='log-uniform', name='C'),
                    Categorical(['rbf', 'poly'], name='kernel'),
                    Real(0.01, 1.0, name='epsilon')
                ]
            elif optimization_mode == 'moderate':
                return [
                    Real(0.01, 1000, prior='log-uniform', name='C'),
                    Categorical(['linear', 'rbf', 'poly'], name='kernel'),
                    Real(0.001, 1.0, prior='log-uniform', name='gamma'),
                    Real(0.001, 1.0, name='epsilon')
                ]
            else:  # aggressive
                return [
                    Real(0.001, 10000, prior='log-uniform', name='C'),
                    Categorical(['linear', 'rbf', 'poly', 'sigmoid'], name='kernel'),
                    Real(0.0001, 10, prior='log-uniform', name='gamma'),
                    Real(0.001, 2.0, name='epsilon'),
                    Integer(2, 5, name='degree')  # For polynomial kernel
                ]
                
        elif self.model_type == 'MLPRegressor':
            if optimization_mode == 'conservative':
                return [
                    Categorical([(50,), (100,), (100, 50)], name='hidden_layer_sizes'),
                    Real(0.0001, 0.1, prior='log-uniform', name='alpha'),
                    Categorical(['constant', 'adaptive'], name='learning_rate'),
                    Integer(200, 500, name='max_iter')
                ]
            elif optimization_mode == 'moderate':
                return [
                    Categorical([(50,), (100,), (100, 50), (200,), (200, 100)], name='hidden_layer_sizes'),
                    Real(0.00001, 0.1, prior='log-uniform', name='alpha'),
                    Categorical(['constant', 'invscaling', 'adaptive'], name='learning_rate'),
                    Integer(200, 1000, name='max_iter'),
                    Categorical(['adam', 'lbfgs'], name='solver')
                ]
            else:  # aggressive
                return [
                    Categorical([(50,), (100,), (100, 50), (200,), (200, 100), 
                               (300, 150), (100, 100, 50)], name='hidden_layer_sizes'),
                    Real(0.000001, 0.1, prior='log-uniform', name='alpha'),
                    Categorical(['constant', 'invscaling', 'adaptive'], name='learning_rate'),
                    Integer(200, 2000, name='max_iter'),
                    Categorical(['adam', 'lbfgs'], name='solver'),
                    Categorical(['relu', 'tanh', 'logistic'], name='activation')
                ]
                
        elif self.model_type == 'BayesianRidge':
            if optimization_mode == 'conservative':
                return [
                    Real(1e-8, 1e-2, prior='log-uniform', name='alpha_1'),
                    Real(1e-8, 1e-2, prior='log-uniform', name='alpha_2'),
                    Real(1e-8, 1e-2, prior='log-uniform', name='lambda_1'),
                    Real(1e-8, 1e-2, prior='log-uniform', name='lambda_2')
                ]
            else:  # moderate and aggressive
                return [
                    Real(1e-12, 1.0, prior='log-uniform', name='alpha_1'),
                    Real(1e-12, 1.0, prior='log-uniform', name='alpha_2'),
                    Real(1e-12, 1.0, prior='log-uniform', name='lambda_1'),
                    Real(1e-12, 1.0, prior='log-uniform', name='lambda_2')
                ]
                
        elif self.model_type == 'GaussianProcess':
            # For GP, we use a different approach with kernel parameters
            if optimization_mode == 'conservative':
                return [
                    Real(0.1, 10.0, prior='log-uniform', name='length_scale'),
                    Real(1e-8, 1e-3, prior='log-uniform', name='noise_level')
                ]
            else:  # moderate and aggressive
                return [
                    Real(0.01, 100.0, prior='log-uniform', name='length_scale'),
                    Real(1e-10, 1e-2, prior='log-uniform', name='noise_level'),
                    Real(1e-8, 1e-4, prior='log-uniform', name='alpha')
                ]
        
        else:
            raise ValueError(f"Bayesian optimization not implemented for {self.model_type}")
    
    def adaptive_sampling_strategy(self, predictions_data, iteration_count, performance_history=None):
        """
        Adaptive sampling strategy that adjusts acquisition function parameters based on learning progress.
        
        Parameters:
        -----------
        predictions_data : dict
            Predictions from current model(s)
        iteration_count : int
            Current iteration number
        performance_history : list, optional
            History of performance metrics
            
        Returns:
        --------
        adjusted_config : dict
            Adjusted acquisition function configuration
        """
        base_config = {'xi': 0.01, 'kappa': 2.576}
        
        # Early exploration phase (first 20% of iterations)
        if iteration_count <= 5:
            # High exploration
            base_config['xi'] = 0.1  # More exploration for EI
            base_config['kappa'] = 3.0  # More exploration for UCB
            strategy = 'high_exploration'
            
        # Learning phase (20%-80% of iterations)
        elif iteration_count <= 20:
            # Balanced exploration-exploitation
            if performance_history and len(performance_history) >= 3:
                # Check improvement rate
                recent_improvements = [
                    performance_history[i] - performance_history[i-1] 
                    for i in range(-3, 0) if i < len(performance_history)
                ]
                avg_improvement = np.mean(recent_improvements) if recent_improvements else 0
                
                if avg_improvement > 0.01:  # Still improving
                    base_config['xi'] = 0.05
                    base_config['kappa'] = 2.5
                    strategy = 'moderate_exploration'
                else:  # Slow improvement, more exploration
                    base_config['xi'] = 0.08
                    base_config['kappa'] = 2.8
                    strategy = 'increased_exploration'
            else:
                base_config['xi'] = 0.05
                base_config['kappa'] = 2.5
                strategy = 'moderate_exploration'
                
        # Refinement phase (80%+ of iterations)
        else:
            # High exploitation
            base_config['xi'] = 0.001
            base_config['kappa'] = 1.5
            strategy = 'high_exploitation'
        
        # Uncertainty-based adjustment
        if predictions_data and 'uncertainty' in predictions_data:
            uncertainties = predictions_data['uncertainty']
            if hasattr(uncertainties, '__len__') and len(uncertainties) > 0:
                avg_uncertainty = np.mean(uncertainties)
                max_uncertainty = np.max(uncertainties)
                
                # If uncertainty is very high, increase exploration
                if avg_uncertainty > 0.3 or max_uncertainty > 0.8:
                    base_config['xi'] *= 1.5
                    base_config['kappa'] *= 1.2
                    strategy += '_high_uncertainty'
                
                # If uncertainty is very low, increase exploitation
                elif avg_uncertainty < 0.1 and max_uncertainty < 0.3:
                    base_config['xi'] *= 0.5
                    base_config['kappa'] *= 0.8
                    strategy += '_low_uncertainty'
        
        return {
            'acquisition_config': base_config,
            'strategy': strategy,
            'iteration': iteration_count,
            'reasoning': f"Applied {strategy} strategy at iteration {iteration_count}"
        }
    
    def multi_source_uncertainty_fusion(self, X_virtual, ensemble_size=5, methods=['bootstrap', 'mc_dropout', 'deep_ensemble']):
        """
        Fuse multiple uncertainty quantification methods for more robust uncertainty estimation.
        
        Parameters:
        -----------
        X_virtual : pd.DataFrame or np.ndarray
            Virtual candidates for uncertainty estimation
        ensemble_size : int, default=5
            Size of ensemble for bootstrap and deep ensemble methods
        methods : list, default=['bootstrap', 'mc_dropout', 'deep_ensemble']
            Uncertainty quantification methods to combine
            
        Returns:
        --------
        fused_uncertainty : dict
            Combined uncertainty estimates with individual method contributions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before uncertainty quantification")
        
        X_virtual_array = X_virtual.values if hasattr(X_virtual, 'values') else X_virtual
        X_virtual_scaled = self.scaler_X.transform(X_virtual_array)
        
        uncertainty_estimates = {}
        weights = {}
        
        # Method 1: Bootstrap Ensemble
        if 'bootstrap' in methods:
            try:
                bootstrap_uncertainties = []
                for i in range(ensemble_size):
                    # Create bootstrap sample
                    if hasattr(self, '_X_train_scaled') and hasattr(self, '_y_train'):
                        n_samples = len(self._X_train_scaled)
                        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                        X_bootstrap = self._X_train_scaled[bootstrap_indices]
                        y_bootstrap = self._y_train.iloc[bootstrap_indices] if hasattr(self._y_train, 'iloc') else self._y_train[bootstrap_indices]
                        
                        # Train bootstrap model
                        bootstrap_model = self._create_model()
                        bootstrap_model.fit(X_bootstrap, y_bootstrap)
                        predictions = bootstrap_model.predict(X_virtual_scaled)
                        bootstrap_uncertainties.append(predictions)
                
                if bootstrap_uncertainties:
                    bootstrap_uncertainties = np.array(bootstrap_uncertainties)
                    uncertainty_estimates['bootstrap'] = np.std(bootstrap_uncertainties, axis=0)
                    weights['bootstrap'] = 0.4
                    
            except Exception as e:
                print(f"Bootstrap uncertainty estimation failed: {e}")
        
        # Method 2: Monte Carlo Dropout (approximation using multiple predictions)
        if 'mc_dropout' in methods:
            try:
                mc_predictions = []
                for i in range(ensemble_size):
                    # Add small random noise to simulate dropout effect
                    X_noisy = X_virtual_scaled + np.random.normal(0, 0.01, X_virtual_scaled.shape)
                    predictions = self.model.predict(X_noisy)
                    mc_predictions.append(predictions)
                
                mc_predictions = np.array(mc_predictions)
                uncertainty_estimates['mc_dropout'] = np.std(mc_predictions, axis=0)
                weights['mc_dropout'] = 0.3
                
            except Exception as e:
                print(f"MC Dropout uncertainty estimation failed: {e}")
        
        # Method 3: Deep Ensemble (multiple independent models)
        if 'deep_ensemble' in methods:
            try:
                ensemble_predictions = []
                for i in range(ensemble_size):
                    # Train independent model with different random state
                    ensemble_model = self._create_model({'params': {'random_state': self.random_state + i}})
                    if hasattr(self, '_X_train_scaled') and hasattr(self, '_y_train'):
                        ensemble_model.fit(self._X_train_scaled, self._y_train)
                        predictions = ensemble_model.predict(X_virtual_scaled)
                        ensemble_predictions.append(predictions)
                
                if ensemble_predictions:
                    ensemble_predictions = np.array(ensemble_predictions)
                    uncertainty_estimates['deep_ensemble'] = np.std(ensemble_predictions, axis=0)
                    weights['deep_ensemble'] = 0.3
                    
            except Exception as e:
                print(f"Deep ensemble uncertainty estimation failed: {e}")
        
        # Fuse uncertainties using weighted average
        if uncertainty_estimates:
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in weights.items()}
            else:
                normalized_weights = {k: 1.0/len(weights) for k in weights.keys()}
            
            # Compute weighted fusion
            fused_uncertainty = np.zeros(len(X_virtual_scaled))
            for method, uncertainty in uncertainty_estimates.items():
                weight = normalized_weights.get(method, 0)
                fused_uncertainty += weight * uncertainty
            
            # Additional confidence metrics
            method_agreement = 1.0
            if len(uncertainty_estimates) > 1:
                # Calculate agreement between methods (inverse of variance across methods)
                uncertainties_matrix = np.array(list(uncertainty_estimates.values()))
                method_variance = np.var(uncertainties_matrix, axis=0)
                method_agreement = 1.0 / (1.0 + method_variance)
            
            return {
                'fused_uncertainty': fused_uncertainty,
                'individual_uncertainties': uncertainty_estimates,
                'weights': normalized_weights,
                'method_agreement': method_agreement,
                'confidence_score': np.mean(method_agreement),
                'methods_used': list(uncertainty_estimates.keys())
            }
        else:
            # Fallback to simple prediction uncertainty
            predictions = self.model.predict(X_virtual_scaled)
            fallback_uncertainty = np.full_like(predictions, 0.1)  # Default uncertainty
            return {
                'fused_uncertainty': fallback_uncertainty,
                'individual_uncertainties': {'fallback': fallback_uncertainty},
                'weights': {'fallback': 1.0},
                'method_agreement': np.ones_like(fallback_uncertainty),
                'confidence_score': 1.0,
                'methods_used': ['fallback']
            }
    
    def online_learning_update(self, new_X, new_y, update_strategy='incremental', retrain_threshold=10):
        """
        Online learning capability for incremental model updates.
        
        Parameters:
        -----------
        new_X : pd.DataFrame or np.ndarray
            New training features
        new_y : pd.Series or np.ndarray
            New training targets
        update_strategy : str, default='incremental'
            Update strategy: 'incremental', 'retrain', 'adaptive'
        retrain_threshold : int, default=10
            Number of new samples before full retraining
            
        Returns:
        --------
        update_result : dict
            Results of the online update
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before online updates")
        
        # Store original training data if not already stored
        if not hasattr(self, '_training_history'):
            self._training_history = []
            self._update_count = 0
        
        # Convert new data to appropriate format
        new_X_array = new_X.values if hasattr(new_X, 'values') else new_X
        new_y_array = new_y.values if hasattr(new_y, 'values') else new_y
        
        # Scale new features
        new_X_scaled = self.scaler_X.transform(new_X_array)
        
        # Add to training history
        self._training_history.append({
            'X': new_X_scaled,
            'y': new_y_array,
            'timestamp': pd.Timestamp.now(),
            'batch_size': len(new_X_scaled)
        })
        self._update_count += len(new_X_scaled)
        
        update_result = {
            'strategy_used': update_strategy,
            'samples_added': len(new_X_scaled),
            'total_updates': self._update_count,
            'update_successful': False,
            'performance_change': None
        }
        
        try:
            if update_strategy == 'incremental':
                # For tree-based models, incremental learning is limited
                # We use a sliding window approach
                if self.model_type in ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees']:
                    # Combine recent data and retrain
                    recent_history = self._training_history[-5:]  # Keep last 5 batches
                    all_X = np.vstack([batch['X'] for batch in recent_history])
                    all_y = np.concatenate([batch['y'] for batch in recent_history])
                    
                    # Add original training data
                    if hasattr(self, '_X_train_scaled') and hasattr(self, '_y_train'):
                        all_X = np.vstack([self._X_train_scaled, all_X])
                        all_y_original = self._y_train.values if hasattr(self._y_train, 'values') else self._y_train
                        all_y = np.concatenate([all_y_original, all_y])
                    
                    # Retrain with combined data
                    self.model.fit(all_X, all_y)
                    update_result['strategy_used'] = 'incremental_retrain'
                    
                elif self.model_type in ['MLPRegressor', 'BayesianRidge']:
                    # These models support incremental learning
                    if hasattr(self.model, 'partial_fit'):
                        self.model.partial_fit(new_X_scaled, new_y_array)
                        update_result['strategy_used'] = 'true_incremental'
                    else:
                        # Fallback to retrain
                        if hasattr(self, '_X_train_scaled') and hasattr(self, '_y_train'):
                            combined_X = np.vstack([self._X_train_scaled, new_X_scaled])
                            combined_y_original = self._y_train.values if hasattr(self._y_train, 'values') else self._y_train
                            combined_y = np.concatenate([combined_y_original, new_y_array])
                            self.model.fit(combined_X, combined_y)
                            update_result['strategy_used'] = 'fallback_retrain'
                        
            elif update_strategy == 'retrain':
                # Full retrain with all historical data
                if hasattr(self, '_X_train_scaled') and hasattr(self, '_y_train'):
                    all_X = [self._X_train_scaled]
                    all_y = [self._y_train.values if hasattr(self._y_train, 'values') else self._y_train]
                    
                    for batch in self._training_history:
                        all_X.append(batch['X'])
                        all_y.append(batch['y'])
                    
                    combined_X = np.vstack(all_X)
                    combined_y = np.concatenate(all_y)
                    
                    self.model.fit(combined_X, combined_y)
                    update_result['strategy_used'] = 'full_retrain'
                    
            elif update_strategy == 'adaptive':
                # Adaptive strategy based on update count
                if self._update_count >= retrain_threshold:
                    # Full retrain
                    return self.online_learning_update(new_X, new_y, 'retrain', retrain_threshold)
                else:
                    # Incremental update
                    return self.online_learning_update(new_X, new_y, 'incremental', retrain_threshold)
            
            update_result['update_successful'] = True
            
            # Optionally evaluate performance change
            if hasattr(self, '_X_train_scaled') and hasattr(self, '_y_train'):
                try:
                    # Quick performance check on new data
                    predictions = self.model.predict(new_X_scaled)
                    mse = np.mean((predictions - new_y_array) ** 2)
                    update_result['new_data_mse'] = mse
                    update_result['performance_change'] = f"MSE on new data: {mse:.6f}"
                except:
                    pass
            
        except Exception as e:
            update_result['error'] = str(e)
            print(f"Online learning update failed: {e}")
        
        return update_result
    
    def intelligent_stopping_criteria(self, performance_history, min_iterations=5, patience=3, 
                                    min_improvement=0.001, convergence_window=3):
        """
        Intelligent early stopping based on multiple convergence criteria.
        
        Parameters:
        -----------
        performance_history : list
            History of performance metrics (higher is better)
        min_iterations : int, default=5
            Minimum number of iterations before considering stopping
        patience : int, default=3
            Number of iterations to wait after last improvement
        min_improvement : float, default=0.001
            Minimum improvement threshold
        convergence_window : int, default=3
            Window size for convergence detection
            
        Returns:
        --------
        stopping_decision : dict
            Decision and reasoning for stopping
        """
        if len(performance_history) < min_iterations:
            return {
                'should_stop': False,
                'reason': f'Need at least {min_iterations} iterations (current: {len(performance_history)})',
                'confidence': 0.0,
                'iterations_remaining': min_iterations - len(performance_history)
            }
        
        # Convert to numpy array for easier manipulation
        history = np.array(performance_history)
        current_performance = history[-1]
        
        # Criterion 1: Lack of improvement
        recent_best = np.max(history[-patience:])
        historical_best = np.max(history[:-patience]) if len(history) > patience else -np.inf
        
        improvement_criterion = recent_best - historical_best >= min_improvement
        
        # Criterion 2: Performance convergence
        if len(history) >= convergence_window:
            recent_window = history[-convergence_window:]
            window_std = np.std(recent_window)
            window_mean = np.mean(recent_window)
            
            # Coefficient of variation
            cv = window_std / (abs(window_mean) + 1e-8)
            convergence_criterion = cv < 0.01  # Very low relative variation
        else:
            convergence_criterion = False
        
        # Criterion 3: Diminishing returns
        if len(history) >= 2 * convergence_window:
            first_half = history[-2*convergence_window:-convergence_window]
            second_half = history[-convergence_window:]
            
            first_half_mean = np.mean(first_half)
            second_half_mean = np.mean(second_half)
            
            relative_improvement = (second_half_mean - first_half_mean) / (abs(first_half_mean) + 1e-8)
            diminishing_returns_criterion = relative_improvement < min_improvement
        else:
            diminishing_returns_criterion = False
        
        # Criterion 4: Trend analysis
        if len(history) >= convergence_window:
            # Fit linear trend to recent performance
            x = np.arange(len(history[-convergence_window:]))
            y = history[-convergence_window:]
            slope = np.polyfit(x, y, 1)[0]
            
            # Very small or negative slope indicates no improvement
            trend_criterion = abs(slope) < min_improvement / convergence_window
        else:
            trend_criterion = False
        
        # Combine criteria
        criteria_met = sum([
            not improvement_criterion,
            convergence_criterion,
            diminishing_returns_criterion,
            trend_criterion
        ])
        
        # Decision making
        total_criteria = 4
        confidence = criteria_met / total_criteria
        
        should_stop = criteria_met >= 2  # At least 2 criteria must be met
        
        # Generate detailed reasoning
        reasons = []
        if not improvement_criterion:
            reasons.append(f"No significant improvement in last {patience} iterations")
        if convergence_criterion:
            reasons.append(f"Performance converged (CV < 1%)")
        if diminishing_returns_criterion:
            reasons.append(f"Diminishing returns detected")
        if trend_criterion:
            reasons.append(f"Flat performance trend")
        
        if should_stop:
            main_reason = f"Multiple stopping criteria met ({criteria_met}/{total_criteria}): " + "; ".join(reasons)
        else:
            main_reason = f"Continuing optimization ({criteria_met}/{total_criteria} criteria met)"
        
        return {
            'should_stop': should_stop,
            'reason': main_reason,
            'confidence': confidence,
            'criteria_details': {
                'improvement_criterion': improvement_criterion,
                'convergence_criterion': convergence_criterion,
                'diminishing_returns_criterion': diminishing_returns_criterion,
                'trend_criterion': trend_criterion
            },
            'current_performance': current_performance,
            'best_performance': np.max(history),
            'iterations_run': len(history)
        }
    
    def optimized_batch_sampling(self, predictions_data, batch_size, diversity_weight=0.3, 
                                uncertainty_threshold=0.1, method='greedy_diverse'):
        """
        Optimized batch sampling that balances acquisition value and sample diversity.
        
        Parameters:
        -----------
        predictions_data : dict
            Predictions with uncertainty estimates
        batch_size : int
            Number of samples to select in batch
        diversity_weight : float, default=0.3
            Weight for diversity vs acquisition value (0=pure acquisition, 1=pure diversity)
        uncertainty_threshold : float, default=0.1
            Minimum uncertainty threshold for consideration
        method : str, default='greedy_diverse'
            Batch selection method: 'greedy_diverse', 'clustering', 'determinantal'
            
        Returns:
        --------
        batch_indices : list
            Indices of selected samples
        batch_info : dict
            Information about the batch selection process
        """
        if 'acquisition_values' not in predictions_data:
            raise ValueError("Acquisition values required for batch sampling")
        
        acquisition_values = np.array(predictions_data['acquisition_values'])
        uncertainties = np.array(predictions_data.get('uncertainty', np.ones_like(acquisition_values)))
        
        # Filter by uncertainty threshold
        valid_mask = uncertainties >= uncertainty_threshold
        if not np.any(valid_mask):
            # If no samples meet threshold, select top uncertainty samples
            valid_mask = uncertainties >= np.percentile(uncertainties, 90)
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < batch_size:
            # Not enough valid samples, use all available
            return list(valid_indices), {
                'method': method,
                'samples_selected': len(valid_indices),
                'diversity_achieved': 0.0,
                'average_acquisition': np.mean(acquisition_values[valid_indices]),
                'average_uncertainty': np.mean(uncertainties[valid_indices])
            }
        
        valid_acquisition = acquisition_values[valid_indices]
        valid_uncertainties = uncertainties[valid_indices]
        
        if method == 'greedy_diverse':
            selected_indices = []
            remaining_indices = list(range(len(valid_indices)))
            
            # First sample: highest acquisition value
            best_idx = np.argmax(valid_acquisition)
            selected_indices.append(valid_indices[best_idx])
            remaining_indices.remove(best_idx)
            
            # Additional samples: balance acquisition and diversity
            for _ in range(min(batch_size - 1, len(remaining_indices))):
                if not remaining_indices:
                    break
                
                scores = []
                for idx in remaining_indices:
                    candidate_idx = valid_indices[idx]
                    
                    # Acquisition component
                    acq_score = valid_acquisition[idx]
                    
                    # Diversity component (minimum distance to selected samples)
                    if 'features' in predictions_data:
                        features = predictions_data['features']
                        candidate_features = features[candidate_idx]
                        selected_features = features[selected_indices]
                        
                        # Calculate minimum distance to selected samples
                        if len(selected_features) > 0:
                            distances = [
                                np.linalg.norm(candidate_features - sel_features)
                                for sel_features in selected_features
                            ]
                            min_distance = min(distances)
                        else:
                            min_distance = 1.0
                    else:
                        # Use uncertainty difference as diversity proxy
                        candidate_uncertainty = valid_uncertainties[idx]
                        selected_uncertainties = uncertainties[selected_indices]
                        min_distance = min([
                            abs(candidate_uncertainty - sel_unc)
                            for sel_unc in selected_uncertainties
                        ]) if len(selected_uncertainties) > 0 else 1.0
                    
                    # Combined score
                    combined_score = (1 - diversity_weight) * acq_score + diversity_weight * min_distance
                    scores.append(combined_score)
                
                # Select best combined score
                best_remaining_idx = remaining_indices[np.argmax(scores)]
                selected_indices.append(valid_indices[best_remaining_idx])
                remaining_indices.remove(best_remaining_idx)
        
        elif method == 'clustering':
            from sklearn.cluster import KMeans
            
            # Use features if available, otherwise use acquisition values and uncertainties
            if 'features' in predictions_data:
                feature_data = predictions_data['features'][valid_indices]
            else:
                feature_data = np.column_stack([valid_acquisition, valid_uncertainties])
            
            # Cluster samples
            n_clusters = min(batch_size * 2, len(valid_indices))
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_data)
            
            # Select best sample from each cluster
            selected_indices = []
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.any(cluster_mask):
                    cluster_indices = valid_indices[cluster_mask]
                    cluster_acquisition = acquisition_values[cluster_indices]
                    best_in_cluster = cluster_indices[np.argmax(cluster_acquisition)]
                    selected_indices.append(best_in_cluster)
                    
                    if len(selected_indices) >= batch_size:
                        break
        
        elif method == 'determinantal':
            # Simplified determinantal point process approximation
            # Select samples that maximize determinant (diversity)
            selected_indices = []
            
            if 'features' in predictions_data:
                feature_data = predictions_data['features'][valid_indices]
                
                # Start with highest acquisition sample
                best_idx = np.argmax(valid_acquisition)
                selected_indices.append(valid_indices[best_idx])
                
                # Greedily add samples that maximize determinant
                for _ in range(min(batch_size - 1, len(valid_indices) - 1)):
                    best_det = -np.inf
                    best_candidate = None
                    
                    for idx in valid_indices:
                        if idx in selected_indices:
                            continue
                        
                        # Form matrix with current selection + candidate
                        candidate_features = [feature_data[i] for i, orig_idx in enumerate(valid_indices) 
                                           if orig_idx in selected_indices]
                        candidate_features.append(feature_data[valid_indices.tolist().index(idx)])
                        
                        try:
                            feature_matrix = np.array(candidate_features)
                            if feature_matrix.shape[0] <= feature_matrix.shape[1]:
                                # Use Gram matrix for overdetermined case
                                gram_matrix = np.dot(feature_matrix, feature_matrix.T)
                                det = np.linalg.det(gram_matrix)
                            else:
                                det = 1.0  # Fallback
                                
                            # Weight by acquisition value
                            weighted_det = det * valid_acquisition[valid_indices.tolist().index(idx)]
                            
                            if weighted_det > best_det:
                                best_det = weighted_det
                                best_candidate = idx
                        except:
                            continue
                    
                    if best_candidate is not None:
                        selected_indices.append(best_candidate)
            else:
                # Fallback to greedy_diverse
                return self.optimized_batch_sampling(
                    predictions_data, batch_size, diversity_weight, 
                    uncertainty_threshold, 'greedy_diverse'
                )
        
        # Calculate diversity metrics
        if len(selected_indices) > 1 and 'features' in predictions_data:
            selected_features = predictions_data['features'][selected_indices]
            pairwise_distances = []
            for i in range(len(selected_features)):
                for j in range(i+1, len(selected_features)):
                    dist = np.linalg.norm(selected_features[i] - selected_features[j])
                    pairwise_distances.append(dist)
            diversity_achieved = np.mean(pairwise_distances) if pairwise_distances else 0.0
        else:
            diversity_achieved = 0.0
        
        batch_info = {
            'method': method,
            'samples_selected': len(selected_indices),
            'diversity_achieved': diversity_achieved,
            'average_acquisition': np.mean(acquisition_values[selected_indices]),
            'average_uncertainty': np.mean(uncertainties[selected_indices]),
            'diversity_weight_used': diversity_weight,
            'uncertainty_threshold_used': uncertainty_threshold
        }
        
        return selected_indices, batch_info
    
    def create_experiment_manager(self, experiment_name="active_learning_exp"):
        """
        Create experiment manager for tracking active learning experiments.
        
        Parameters:
        -----------
        experiment_name : str, default="active_learning_exp"
            Name of the experiment
            
        Returns:
        --------
        experiment_manager : ActiveLearningExperimentManager
            Experiment manager instance
        """
        return ActiveLearningExperimentManager(experiment_name, self)
    
    def analyze_learning_curve(self, performance_history, metrics=['r2', 'mse'], save_plot=False):
        """
        Analyze learning curve and provide insights.
        
        Parameters:
        -----------
        performance_history : list or dict
            Performance metrics history over iterations
        metrics : list, default=['r2', 'mse']
            Metrics to analyze
        save_plot : bool, default=False
            Whether to save the learning curve plot
            
        Returns:
        --------
        analysis_result : dict
            Learning curve analysis results
        """
        import matplotlib.pyplot as plt
        
        if isinstance(performance_history, list):
            # Single metric case
            history = {'main_metric': performance_history}
        else:
            history = performance_history
        
        analysis = {
            'convergence_point': None,
            'learning_phases': [],
            'recommendations': [],
            'trend_analysis': {},
            'optimal_stopping_point': None
        }
        
        for metric_name, values in history.items():
            if len(values) < 3:
                continue
                
            values = np.array(values)
            iterations = np.arange(len(values))
            
            # Trend analysis
            slope, intercept = np.polyfit(iterations, values, 1)
            r_squared = np.corrcoef(iterations, values)[0, 1] ** 2
            
            # Find change points using sliding window
            change_points = []
            window_size = max(3, len(values) // 10)
            
            for i in range(window_size, len(values) - window_size):
                before_window = values[max(0, i-window_size):i]
                after_window = values[i:min(len(values), i+window_size)]
                
                before_trend = np.polyfit(range(len(before_window)), before_window, 1)[0]
                after_trend = np.polyfit(range(len(after_window)), after_window, 1)[0]
                
                trend_change = abs(after_trend - before_trend)
                if trend_change > 0.01:  # Significant trend change
                    change_points.append(i)
            
            # Identify learning phases
            phases = []
            last_point = 0
            
            for cp in change_points[:3]:  # Limit to 3 major phases
                phase_values = values[last_point:cp]
                if len(phase_values) > 2:
                    phase_trend = np.polyfit(range(len(phase_values)), phase_values, 1)[0]
                    phase_type = 'rapid_improvement' if phase_trend > 0.01 else ('slow_improvement' if phase_trend > 0 else 'plateau')
                    phases.append({
                        'start': last_point,
                        'end': cp,
                        'type': phase_type,
                        'improvement': phase_values[-1] - phase_values[0],
                        'duration': cp - last_point
                    })
                last_point = cp
            
            # Final phase
            if last_point < len(values) - 1:
                final_values = values[last_point:]
                final_trend = np.polyfit(range(len(final_values)), final_values, 1)[0]
                final_type = 'rapid_improvement' if final_trend > 0.01 else ('slow_improvement' if final_trend > 0 else 'plateau')
                phases.append({
                    'start': last_point,
                    'end': len(values) - 1,
                    'type': final_type,
                    'improvement': final_values[-1] - final_values[0],
                    'duration': len(values) - 1 - last_point
                })
            
            analysis['trend_analysis'][metric_name] = {
                'overall_slope': slope,
                'r_squared': r_squared,
                'trend_direction': 'improving' if slope > 0.001 else ('declining' if slope < -0.001 else 'stable'),
                'change_points': change_points,
                'learning_phases': phases
            }
            
            # Find optimal stopping point
            if len(values) >= 5:
                # Use elbow method to find optimal stopping
                improvements = np.diff(values)
                cumulative_improvement = np.cumsum(improvements)
                
                # Find point where improvement rate significantly decreases
                if len(improvements) >= 3:
                    improvement_slopes = []
                    for i in range(2, len(improvements)):
                        recent_improvements = improvements[max(0, i-2):i+1]
                        slope_improvement = np.polyfit(range(len(recent_improvements)), recent_improvements, 1)[0]
                        improvement_slopes.append(slope_improvement)
                    
                    if improvement_slopes:
                        # Find first point where improvement slope becomes very small
                        for i, imp_slope in enumerate(improvement_slopes):
                            if imp_slope < 0.001:  # Very small improvement rate
                                analysis['optimal_stopping_point'] = i + 3  # Adjust for window offset
                                break
            
            # Convergence detection
            if len(values) >= 5:
                recent_values = values[-5:]
                if np.std(recent_values) < 0.01 * np.mean(recent_values):
                    analysis['convergence_point'] = len(values) - 5
        
        # Generate recommendations
        if analysis['convergence_point']:
            analysis['recommendations'].append(f"Model converged around iteration {analysis['convergence_point']}")
        
        if analysis['optimal_stopping_point']:
            analysis['recommendations'].append(f"Optimal stopping point detected at iteration {analysis['optimal_stopping_point']}")
        
        main_trend = analysis['trend_analysis'].get('main_metric', analysis['trend_analysis'].get(list(analysis['trend_analysis'].keys())[0]))
        if main_trend and main_trend['trend_direction'] == 'plateau':
            analysis['recommendations'].append("Consider increasing exploration or changing acquisition function")
        elif main_trend and main_trend['trend_direction'] == 'declining':
            analysis['recommendations'].append("Model performance is declining - check for overfitting or data issues")
        
        # Create plot if requested
        if save_plot:
            try:
                fig, axes = plt.subplots(len(history), 1, figsize=(10, 4*len(history)))
                if len(history) == 1:
                    axes = [axes]
                
                for i, (metric_name, values) in enumerate(history.items()):
                    ax = axes[i] if len(history) > 1 else axes[0]
                    iterations = range(len(values))
                    
                    ax.plot(iterations, values, 'b-', linewidth=2, label=f'{metric_name}')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel(metric_name)
                    ax.set_title(f'Learning Curve - {metric_name}')
                    ax.grid(True, alpha=0.3)
                    
                    # Mark change points
                    if metric_name in analysis['trend_analysis']:
                        change_points = analysis['trend_analysis'][metric_name]['change_points']
                        for cp in change_points:
                            ax.axvline(x=cp, color='r', linestyle='--', alpha=0.7, label='Change Point')
                    
                    # Mark convergence point
                    if analysis['convergence_point']:
                        ax.axvline(x=analysis['convergence_point'], color='g', linestyle=':', 
                                 alpha=0.7, label='Convergence Point')
                    
                    # Mark optimal stopping point
                    if analysis['optimal_stopping_point']:
                        ax.axvline(x=analysis['optimal_stopping_point'], color='orange', linestyle='-.',
                                 alpha=0.7, label='Optimal Stop')
                    
                    ax.legend()
                
                plt.tight_layout()
                plot_filename = f"learning_curve_analysis.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                analysis['plot_saved'] = plot_filename
            except Exception as e:
                analysis['plot_error'] = str(e)
        
        return analysis
    
    def diagnose_model_performance(self, X_test=None, y_test=None, detailed=True):
        """
        Comprehensive model performance diagnosis.
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.ndarray, optional
            Test features for evaluation
        y_test : pd.Series or np.ndarray, optional
            Test targets for evaluation
        detailed : bool, default=True
            Whether to perform detailed diagnosis
            
        Returns:
        --------
        diagnosis : dict
            Comprehensive model diagnosis results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before diagnosis")
        
        diagnosis = {
            'model_info': {
                'model_type': self.model_type,
                'acquisition_function': self.acquisition_function,
                'is_fitted': self.is_fitted,
                'feature_count': len(self.feature_names) if self.feature_names else 'Unknown'
            },
            'training_data_analysis': {},
            'model_complexity': {},
            'prediction_analysis': {},
            'recommendations': []
        }
        
        # Training data analysis
        if hasattr(self, '_X_train_scaled') and hasattr(self, '_y_train'):
            X_train = self._X_train_scaled
            y_train = self._y_train
            
            diagnosis['training_data_analysis'] = {
                'sample_count': len(X_train),
                'feature_count': X_train.shape[1],
                'target_range': [float(y_train.min()), float(y_train.max())],
                'target_std': float(y_train.std()),
                'data_coverage': 'Good' if len(X_train) > X_train.shape[1] * 10 else 'Limited'
            }
            
            # Check for potential issues
            if len(X_train) < X_train.shape[1] * 5:
                diagnosis['recommendations'].append("Training data may be insufficient for reliable modeling")
            
            if y_train.std() < 0.01:
                diagnosis['recommendations'].append("Target variable has very low variance - check data quality")
        
        # Model complexity analysis
        if hasattr(self.model, 'get_params'):
            params = self.model.get_params()
            diagnosis['model_complexity']['parameters'] = {k: v for k, v in params.items() if not callable(v)}
            
            # Complexity assessment for different model types
            if self.model_type == 'RandomForest':
                n_trees = params.get('n_estimators', 100)
                max_depth = params.get('max_depth', None)
                complexity_score = n_trees * (max_depth if max_depth else 20) / 1000
                diagnosis['model_complexity']['complexity_score'] = complexity_score
                diagnosis['model_complexity']['complexity_level'] = (
                    'Low' if complexity_score < 2 else 'Medium' if complexity_score < 10 else 'High'
                )
                
                if complexity_score > 15:
                    diagnosis['recommendations'].append("Model complexity is very high - consider regularization")
            
            elif self.model_type in ['XGBoost', 'LightGBM', 'CatBoost']:
                n_trees = params.get('n_estimators', 100)
                learning_rate = params.get('learning_rate', 0.1)
                complexity_score = n_trees * (1 / learning_rate) / 1000
                diagnosis['model_complexity']['complexity_score'] = complexity_score
                diagnosis['model_complexity']['complexity_level'] = (
                    'Low' if complexity_score < 1 else 'Medium' if complexity_score < 5 else 'High'
                )
        
        # Prediction analysis on training data
        if hasattr(self, '_X_train_scaled') and hasattr(self, '_y_train'):
            try:
                train_predictions = self.model.predict(self._X_train_scaled)
                train_residuals = self._y_train - train_predictions
                
                diagnosis['prediction_analysis']['train_metrics'] = {
                    'r2': float(1 - np.var(train_residuals) / np.var(self._y_train)),
                    'mse': float(np.mean(train_residuals ** 2)),
                    'mae': float(np.mean(np.abs(train_residuals))),
                    'residual_std': float(np.std(train_residuals))
                }
                
                # Check for overfitting indicators
                r2_train = diagnosis['prediction_analysis']['train_metrics']['r2']
                if r2_train > 0.99:
                    diagnosis['recommendations'].append("Very high training R² - potential overfitting")
                elif r2_train < 0.1:
                    diagnosis['recommendations'].append("Low training R² - model may be underfitting")
                
                # Residual analysis
                residual_skewness = float((train_residuals - train_residuals.mean()).skew() if hasattr(train_residuals, 'skew') else 0)
                if abs(residual_skewness) > 1:
                    diagnosis['recommendations'].append("Residuals are skewed - consider data transformation")
                    
            except Exception as e:
                diagnosis['prediction_analysis']['train_error'] = str(e)
        
        # Test data evaluation if provided
        if X_test is not None and y_test is not None:
            try:
                X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
                y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
                
                X_test_scaled = self.scaler_X.transform(X_test_array)
                test_predictions = self.model.predict(X_test_scaled)
                test_residuals = y_test_array - test_predictions
                
                diagnosis['prediction_analysis']['test_metrics'] = {
                    'r2': float(1 - np.var(test_residuals) / np.var(y_test_array)),
                    'mse': float(np.mean(test_residuals ** 2)),
                    'mae': float(np.mean(np.abs(test_residuals))),
                    'residual_std': float(np.std(test_residuals))
                }
                
                # Compare train vs test performance
                if 'train_metrics' in diagnosis['prediction_analysis']:
                    train_r2 = diagnosis['prediction_analysis']['train_metrics']['r2']
                    test_r2 = diagnosis['prediction_analysis']['test_metrics']['r2']
                    
                    performance_gap = train_r2 - test_r2
                    if performance_gap > 0.1:
                        diagnosis['recommendations'].append(f"Significant overfitting detected (train R²: {train_r2:.3f}, test R²: {test_r2:.3f})")
                    elif performance_gap < -0.05:
                        diagnosis['recommendations'].append("Test performance better than training - check data leakage")
                    else:
                        diagnosis['recommendations'].append("Good generalization performance")
                        
            except Exception as e:
                diagnosis['prediction_analysis']['test_error'] = str(e)
        
        # Feature importance analysis if available
        if detailed and hasattr(self.model, 'feature_importances_'):
            try:
                importances = self.model.feature_importances_
                if self.feature_names:
                    feature_importance = dict(zip(self.feature_names, importances))
                    # Sort by importance
                    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    diagnosis['feature_importance'] = {
                        'top_features': sorted_importance[:10],
                        'importance_concentration': float(np.sum(importances[:3]) / np.sum(importances)),
                        'low_importance_features': [name for name, imp in sorted_importance if imp < 0.01]
                    }
                    
                    # Recommendations based on feature importance
                    if diagnosis['feature_importance']['importance_concentration'] > 0.8:
                        diagnosis['recommendations'].append("Feature importance highly concentrated - consider feature selection")
                    
                    if len(diagnosis['feature_importance']['low_importance_features']) > len(self.feature_names) // 2:
                        diagnosis['recommendations'].append("Many features have low importance - consider dimensionality reduction")
                        
            except Exception as e:
                diagnosis['feature_importance_error'] = str(e)
        
        # Overall health score
        health_factors = []
        
        if 'train_metrics' in diagnosis['prediction_analysis']:
            train_r2 = diagnosis['prediction_analysis']['train_metrics']['r2']
            health_factors.append(min(1.0, max(0.0, train_r2)))
        
        if 'test_metrics' in diagnosis['prediction_analysis']:
            test_r2 = diagnosis['prediction_analysis']['test_metrics']['r2']
            health_factors.append(min(1.0, max(0.0, test_r2)))
        
        # Penalty for too many recommendations (issues)
        issue_penalty = max(0, (len(diagnosis['recommendations']) - 2) * 0.1)
        
        if health_factors:
            health_score = np.mean(health_factors) - issue_penalty
            diagnosis['overall_health'] = {
                'score': max(0.0, min(1.0, health_score)),
                'level': 'Excellent' if health_score > 0.8 else ('Good' if health_score > 0.6 else ('Fair' if health_score > 0.4 else 'Poor')),
                'issues_count': len(diagnosis['recommendations'])
            }
        
        return diagnosis

class ActiveLearningExperimentManager:
    """
    Experiment manager for active learning experiments.
    """
    
    def __init__(self, experiment_name, optimizer=None):
        self.experiment_name = experiment_name
        self.optimizer = optimizer
        self.experiments = []
        self.current_experiment = None
        
    def start_experiment(self, experiment_config):
        """Start a new experiment with given configuration."""
        experiment = {
            'id': len(self.experiments),
            'name': f"{self.experiment_name}_{len(self.experiments)}",
            'config': experiment_config,
            'start_time': pd.Timestamp.now(),
            'iterations': [],
            'performance_history': [],
            'status': 'running',
            'metadata': {}
        }
        
        self.experiments.append(experiment)
        self.current_experiment = experiment
        return experiment['id']
    
    def log_iteration(self, iteration_data):
        """Log data for current iteration."""
        if self.current_experiment:
            self.current_experiment['iterations'].append({
                'iteration': len(self.current_experiment['iterations']),
                'timestamp': pd.Timestamp.now(),
                'data': iteration_data
            })
            
            if 'performance' in iteration_data:
                self.current_experiment['performance_history'].append(iteration_data['performance'])
    
    def finish_experiment(self, final_results=None):
        """Finish current experiment."""
        if self.current_experiment:
            self.current_experiment['end_time'] = pd.Timestamp.now()
            self.current_experiment['duration'] = (
                self.current_experiment['end_time'] - self.current_experiment['start_time']
            ).total_seconds()
            self.current_experiment['status'] = 'completed'
            
            if final_results:
                self.current_experiment['final_results'] = final_results
            
            self.current_experiment = None
    
    def get_experiment_summary(self, experiment_id=None):
        """Get summary of experiment(s)."""
        if experiment_id is not None:
            if experiment_id < len(self.experiments):
                return self._summarize_experiment(self.experiments[experiment_id])
            else:
                raise ValueError(f"Experiment {experiment_id} not found")
        else:
            return [self._summarize_experiment(exp) for exp in self.experiments]
    
    def _summarize_experiment(self, experiment):
        """Create summary of single experiment."""
        summary = {
            'id': experiment['id'],
            'name': experiment['name'],
            'status': experiment['status'],
            'duration': experiment.get('duration', 0),
            'iterations_count': len(experiment['iterations']),
            'config': experiment['config']
        }
        
        if experiment['performance_history']:
            summary['performance'] = {
                'initial': experiment['performance_history'][0],
                'final': experiment['performance_history'][-1],
                'best': max(experiment['performance_history']),
                'improvement': experiment['performance_history'][-1] - experiment['performance_history'][0]
            }
        
        return summary
    
    def compare_experiments(self, experiment_ids=None):
        """Compare multiple experiments."""
        if experiment_ids is None:
            experiment_ids = list(range(len(self.experiments)))
        
        comparison = {
            'experiments': [],
            'best_experiment': None,
            'performance_comparison': {}
        }
        
        best_performance = -np.inf
        best_exp_id = None
        
        for exp_id in experiment_ids:
            if exp_id >= len(self.experiments):
                continue
                
            exp = self.experiments[exp_id]
            summary = self._summarize_experiment(exp)
            comparison['experiments'].append(summary)
            
            if 'performance' in summary and summary['performance']['best'] > best_performance:
                best_performance = summary['performance']['best']
                best_exp_id = exp_id
        
        if best_exp_id is not None:
            comparison['best_experiment'] = best_exp_id
            
        return comparison
    
    def export_results(self, filename=None):
        """Export experiment results to file."""
        if filename is None:
            filename = f"{self.experiment_name}_results.json"
        
        export_data = {
            'experiment_name': self.experiment_name,
            'export_time': pd.Timestamp.now().isoformat(),
            'experiments': self.experiments
        }
        
        # Convert timestamps to strings for JSON serialization
        for exp in export_data['experiments']:
            if 'start_time' in exp:
                exp['start_time'] = exp['start_time'].isoformat()
            if 'end_time' in exp:
                exp['end_time'] = exp['end_time'].isoformat()
            
            for iteration in exp['iterations']:
                iteration['timestamp'] = iteration['timestamp'].isoformat()
        
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filename