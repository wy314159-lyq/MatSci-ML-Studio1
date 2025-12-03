"""
Machine Learning utilities for MatSci-ML Studio
Provides functions for model selection, feature selection, and evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression, chi2
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import uniform, randint

# Classification Models
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
                             ExtraTreesClassifier, BaggingClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Regression Models
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
                                 SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, TheilSenRegressor, QuantileRegressor)
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,
                             ExtraTreesRegressor, BaggingRegressor, VotingRegressor)
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error, mean_squared_error,
    r2_score, explained_variance_score, median_absolute_error, max_error
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Gaussian Process models
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C


def get_available_models(task_type: str) -> Dict[str, Any]:
    """
    Get available models for specified task type
    
    Args:
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary of model names and classes
    """
    if task_type.lower() == 'classification':
        models = {
            # Linear Models
            'Logistic Regression': LogisticRegression,
            'Ridge Classifier': RidgeClassifier,
            'SGD Classifier': SGDClassifier,
            'Passive Aggressive Classifier': PassiveAggressiveClassifier,
            
            # SVM Models
            'Support Vector Classifier': SVC,
            'Nu-Support Vector Classifier': NuSVC,
            
            # Tree Models
            'Decision Tree': DecisionTreeClassifier,
            'Extra Tree': ExtraTreeClassifier,
            
            # Ensemble Models
            'Random Forest': RandomForestClassifier,
            'Extra Trees': ExtraTreesClassifier,
            'AdaBoost': AdaBoostClassifier,
            'Gradient Boosting': GradientBoostingClassifier,
            'Bagging Classifier': BaggingClassifier,
            
            # Nearest Neighbors
            'K-Nearest Neighbors': KNeighborsClassifier,
            'Radius Neighbors': RadiusNeighborsClassifier,
            
            # Naive Bayes
            'Gaussian Naive Bayes': GaussianNB,
            'Multinomial Naive Bayes': MultinomialNB,
            'Bernoulli Naive Bayes': BernoulliNB,
            
            # Discriminant Analysis
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis,
            'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis,
            
            # Neural Networks
            'MLP Classifier': MLPClassifier,
            
            # Gaussian Process Models
            'Gaussian Process Classifier': GaussianProcessClassifier
        }
        
        if XGB_AVAILABLE:
            models['XGBoost Classifier'] = xgb.XGBClassifier
        if LGB_AVAILABLE:
            models['LightGBM Classifier'] = lgb.LGBMClassifier
        if CATBOOST_AVAILABLE:
            models['CatBoost Classifier'] = cb.CatBoostClassifier
            
    elif task_type.lower() == 'regression':
        models = {
            # Linear Models
            'Linear Regression': LinearRegression,
            'Ridge Regression': Ridge,
            'Lasso Regression': Lasso,
            'ElasticNet Regression': ElasticNet,
            'Bayesian Ridge': BayesianRidge,
            'SGD Regressor': SGDRegressor,
            'Passive Aggressive Regressor': PassiveAggressiveRegressor,
            'Huber Regressor': HuberRegressor,
            'Theil-Sen Regressor': TheilSenRegressor,
            'Quantile Regression': QuantileRegressor,

            # SVM Models
            'Support Vector Regression': SVR,
            'Nu-Support Vector Regression': NuSVR,

            # Tree Models
            'Decision Tree Regressor': DecisionTreeRegressor,
            'Extra Tree Regressor': ExtraTreeRegressor,

            # Ensemble Models
            'Random Forest Regressor': RandomForestRegressor,
            'Extra Trees Regressor': ExtraTreesRegressor,
            'AdaBoost Regressor': AdaBoostRegressor,
            'Gradient Boosting Regressor': GradientBoostingRegressor,
            'Bagging Regressor': BaggingRegressor,

            # Nearest Neighbors
            'K-Nearest Neighbors Regressor': KNeighborsRegressor,
            'Radius Neighbors Regressor': RadiusNeighborsRegressor,

            # Neural Networks
            'MLP Regressor': MLPRegressor,

            # Gaussian Process Models
            'Gaussian Process Regressor': GaussianProcessRegressor
        }
        
        if XGB_AVAILABLE:
            models['XGBoost Regressor'] = xgb.XGBRegressor
        if LGB_AVAILABLE:
            models['LightGBM Regressor'] = lgb.LGBMRegressor
        if CATBOOST_AVAILABLE:
            models['CatBoost Regressor'] = cb.CatBoostRegressor
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return models


def get_default_hyperparameters(model_name: str, task_type: str) -> Dict[str, List]:
    """
    Get default hyperparameter grids for models
    
    Args:
        model_name: Name of the model
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary of hyperparameters
    """
    # Common parameters for tree-based models
    tree_params = {
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None, 0.5, 0.8]
    }
    
    ensemble_params = {
        **tree_params,
        'n_estimators': [50, 100, 200, 300, 500]
    }
    
    params = {}
    
    # Linear Models
    if model_name == 'Logistic Regression':
        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga', 'lbfgs'],
            'max_iter': [100, 200, 500, 1000],
            'class_weight': ['balanced']
        }
    elif model_name == 'Ridge Classifier':
        params = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
        }
    elif model_name == 'SGD Classifier':
        params = {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
        }
    elif model_name == 'Passive Aggressive Classifier':
        params = {
            'C': [0.01, 0.1, 1, 10, 100],
            'loss': ['hinge', 'squared_hinge']
        }
    
    # SVM Models
    elif model_name in ['Support Vector Classifier', 'Linear SVC']:
        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4, 5],
            'class_weight': ['balanced']
        }
    elif model_name == 'Nu-Support Vector Classifier':
        params = {
            'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'class_weight': ['balanced']
        }
    
    # Tree Models
    elif model_name in ['Decision Tree', 'Decision Tree Regressor']:
        params = tree_params
    elif model_name in ['Extra Tree', 'Extra Tree Regressor']:
        params = tree_params
    
    # Ensemble Models
    elif model_name in ['Random Forest', 'Random Forest Regressor']:
        params = ensemble_params.copy()
        # Add class_weight for classification
        if 'Regressor' not in model_name:
            params['class_weight'] = ['balanced', None]
    elif model_name in ['Extra Trees', 'Extra Trees Regressor']:
        params = ensemble_params.copy()
        # Add class_weight for classification
        if 'Regressor' not in model_name:
            params['class_weight'] = ['balanced', None]
    elif model_name in ['Bagging Classifier', 'Bagging Regressor']:
        params = {
            'n_estimators': [10, 50, 100, 200],
            'max_samples': [0.5, 0.7, 0.8, 1.0],
            'max_features': [0.5, 0.7, 0.8, 1.0]
        }
        # Only add class_weight for classification
        if 'Classifier' in model_name:
            params['class_weight'] = ['balanced']
    elif model_name == 'AdaBoost':
        params = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0],
            'algorithm': ['SAMME']  # SAMME.R deprecated in recent sklearn versions
        }
    elif model_name == 'AdaBoost Regressor':
        params = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0],
            'loss': ['linear', 'square', 'exponential']
        }
    elif model_name in ['Gradient Boosting', 'Gradient Boosting Regressor']:
        params = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.6, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
    
    # Nearest Neighbors
    elif model_name in ['K-Nearest Neighbors', 'K-Nearest Neighbors Regressor']:
        params = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    elif model_name in ['Radius Neighbors', 'Radius Neighbors Regressor']:
        params = {
            'radius': [0.5, 1.0, 1.5, 2.0, 3.0],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    
    # Naive Bayes (only for classification)
    elif model_name == 'Gaussian Naive Bayes':
        params = {
            'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        }
    elif model_name == 'Multinomial Naive Bayes':
        params = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
        }
    elif model_name == 'Bernoulli Naive Bayes':
        params = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'binarize': [0.0, 0.1, 0.5, 1.0]
        }
    
    # Discriminant Analysis
    elif model_name == 'Linear Discriminant Analysis':
        params = {
            'solver': ['svd', 'lsqr', 'eigen'],
            'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
        }
    elif model_name == 'Quadratic Discriminant Analysis':
        params = {
            'reg_param': [0.0, 0.01, 0.1, 0.5, 1.0]
        }
    
    # Neural Networks
    elif model_name in ['MLP Classifier', 'MLP Regressor']:
        params = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [200, 500, 1000]
        }
    
    # Regression-specific models
    elif model_name in ['Ridge', 'Ridge Regression']:
        params = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
        }
    elif model_name in ['Lasso', 'Lasso Regression']:
        params = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'selection': ['cyclic', 'random'],
            'max_iter': [500, 1000, 2000]
        }
    elif model_name in ['ElasticNet', 'ElasticNet Regression']:
        params = {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'selection': ['cyclic', 'random']
        }
    elif model_name == 'Bayesian Ridge':
        params = {
            'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],
            'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],
            'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3],
            'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3]
        }
    elif model_name == 'SGD Regressor':
        params = {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
        }
    elif model_name == 'Huber Regressor':
        params = {
            'epsilon': [1.1, 1.35, 1.5, 2.0, 3.0],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
        }
    elif model_name == 'Theil-Sen Regressor':
        params = {
            'max_subpopulation': [1e4, 1e5, 1e6],
            'n_subsamples': [None, 100, 500, 1000]
        }
    elif model_name == 'Quantile Regression':
        params = {
            # NOTE: quantile is NOT included in hyperparameter optimization
            # quantile should be specified by user based on business needs
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['highs-ds', 'highs-ipm', 'highs', 'interior-point'],
            'fit_intercept': [True, False]
        }
    
    # SVM Regression
    elif model_name in ['SVR', 'Support Vector Regression']:
        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
    elif model_name == 'Nu-Support Vector Regression':
        params = {
            'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    
    # XGBoost and LightGBM
    elif model_name in ['XGBoost', 'XGBoost Classifier', 'XGBoost Regressor'] and XGB_AVAILABLE:
        params = {
            'n_estimators': [50, 100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 5, 7, 10, 15],
            'subsample': [0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0, 0.01, 0.1, 1],
            'min_child_weight': [1, 3, 5, 7]
            # Note: XGBoost doesn't support class_weight parameter
            # Use scale_pos_weight for binary classification instead
        }
        # Add scale_pos_weight for binary classification only
        if task_type == 'classification':
            params['scale_pos_weight'] = [1, 2, 3, 5, 10]
            
    elif model_name in ['LightGBM', 'LightGBM Classifier', 'LightGBM Regressor'] and LGB_AVAILABLE:
        params = {
            'n_estimators': [50, 100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 5, 7, 10, 15],
            'subsample': [0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0, 0.01, 0.1, 1],
            'min_child_samples': [5, 10, 20, 50],
            'num_leaves': [15, 31, 63, 127]
            # Note: LightGBM uses class_weight differently, handled in model creation
        }
        # Add class balancing for LightGBM classification
        if task_type == 'classification':
            params['is_unbalance'] = [True, False]
    
    # CatBoost Models
    elif model_name in ['CatBoost', 'CatBoost Classifier', 'CatBoost Regressor'] and CATBOOST_AVAILABLE:
        params = {
            'iterations': [100, 200, 300, 500, 800],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            'depth': [3, 4, 5, 6, 7, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 10, 100],
            'border_count': [32, 64, 128, 254],
            'bagging_temperature': [0, 0.5, 1, 2, 5],
            'random_strength': [0, 1, 2, 5, 10],
            'od_type': ['IncToDec', 'Iter'],
            'od_wait': [10, 20, 50, 100]
        }
        # Add task-specific parameters
        if task_type == 'classification':
            params.update({
                'class_weights': [[1, 1], [1, 2], [1, 3], [1, 5]],
                'auto_class_weights': ['Balanced', 'SqrtBalanced']
            })
        else:  # regression
            params.update({
                'loss_function': ['RMSE', 'MAE', 'Quantile', 'LogLinQuantile']
            })
    
    # Gaussian Process Models
    elif model_name == 'Gaussian Process Classifier':
        params = {
            'kernel': ['RBF', 'Matern', 'RBF + WhiteKernel', 'Matern + WhiteKernel'],
            'n_restarts_optimizer': [0, 1, 2, 5, 10],
            'max_iter_predict': [100, 200, 500, 1000],
            'warm_start': [True, False],
            'copy_X_train': [True, False],
            'random_state': [42],
            'multi_class': ['one_vs_rest', 'one_vs_one']
        }
    elif model_name == 'Gaussian Process Regressor':
        params = {
            'kernel': ['RBF', 'Matern', 'RBF + WhiteKernel', 'Matern + WhiteKernel'],
            'alpha': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1],
            'n_restarts_optimizer': [0, 1, 2, 5, 10],
            'normalize_y': [True, False],
            'copy_X_train': [True, False],
            'random_state': [42]
        }
    
    return params


def create_preprocessing_pipeline(numeric_features: List[str], 
                                categorical_features: List[str],
                                boolean_features: List[str] = None) -> ColumnTransformer:
    """
    Create preprocessing pipeline for features with feature selection for high-dimensional data
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names (from one-hot encoding)
        
    Returns:
        ColumnTransformer pipeline
    """
    transformers = []
    
    if numeric_features:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_transformer, numeric_features))
    
    if categorical_features:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            # Add feature selection for high-dimensional categorical data
            ('feature_selection', SelectKBest(chi2, k='all'))  # Will be configured later
        ])
        transformers.append(('categorical', categorical_transformer, categorical_features))
    
    # CRITICAL FIX: Add support for boolean features (from one-hot encoding)
    if boolean_features:
        # Boolean features don't need transformation, just pass through
        from sklearn.preprocessing import FunctionTransformer
        boolean_transformer = FunctionTransformer(validate=False)  # Pass through as-is
        transformers.append(('boolean', boolean_transformer, boolean_features))
        print(f"DEBUG: Added boolean transformer for {len(boolean_features)} features: {boolean_features[:5]}...")
    
    return ColumnTransformer(transformers=transformers)


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained sklearn model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance scores
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    except Exception:
        return None


def perform_feature_selection_importance(X: pd.DataFrame, y: pd.Series, 
                                       model, method: str = 'top_k', 
                                       k: int = 10, threshold: float = 0.01) -> List[str]:
    """
    Perform feature selection based on feature importance
    
    Args:
        X: Feature matrix
        y: Target vector
        model: ML model to use for importance calculation
        method: Selection method ('top_k', 'threshold', 'cumulative')
        k: Number of features to select (for top_k)
        threshold: Importance threshold (for threshold method)
        
    Returns:
        List of selected feature names
    """
    # Fit model
    model.fit(X, y)
    
    # Get feature importance
    importance_df = get_feature_importance(model, X.columns.tolist())
    if importance_df is None:
        return X.columns.tolist()
    
    if method == 'top_k':
        selected_features = importance_df.head(k)['feature'].tolist()
    elif method == 'threshold':
        selected_features = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()
    elif method == 'cumulative':
        # Select features until cumulative importance reaches threshold (e.g., 0.95)
        importance_df['cumulative'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
        selected_features = importance_df[importance_df['cumulative'] <= threshold]['feature'].tolist()
        if not selected_features:  # Ensure at least one feature
            selected_features = [importance_df.iloc[0]['feature']]
    else:
        selected_features = X.columns.tolist()
    
    return selected_features


def remove_correlated_features(X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Remove highly correlated features
    
    Args:
        X: Feature matrix
        threshold: Correlation threshold
        
    Returns:
        List of features to keep
    """
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Get upper triangle of correlation matrix
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Return features to keep
    features_to_keep = [col for col in X.columns if col not in to_drop]
    return features_to_keep


def evaluate_classification_model(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
    """
    Comprehensive evaluation for classification models
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Add ROC AUC if probabilities are available
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Multiclass
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')
        except Exception:
            pass
    
    return metrics


def evaluate_regression_model(y_true, y_pred) -> Dict[str, float]:
    """
    Comprehensive evaluation for regression models
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred),
        'median_absolute_error': median_absolute_error(y_true, y_pred),
        'max_error': max_error(y_true, y_pred)
    }
    
    return metrics


def get_cv_folds(task_type: str, y: pd.Series, n_splits: int = 5, 
                 random_state: int = 42) -> Union[StratifiedKFold, KFold]:
    """
    Get appropriate cross-validation folds for task type
    
    Args:
        task_type: 'classification' or 'regression'
        y: Target variable
        n_splits: Number of folds
        random_state: Random state
        
    Returns:
        CV fold generator
    """
    if task_type.lower() == 'classification':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def get_scoring_metric(task_type: str, metric_name: str) -> str:
    """
    Get sklearn scoring string for metric
    
    Args:
        task_type: 'classification' or 'regression'
        metric_name: Name of metric
        
    Returns:
        Sklearn scoring string
    """
    classification_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': 'roc_auc'
    }
    
    regression_metrics = {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',
        'mse': 'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error'
    }
    
    if task_type.lower() == 'classification':
        return classification_metrics.get(metric_name.lower(), 'accuracy')
    else:
        return regression_metrics.get(metric_name.lower(), 'r2')


def perform_hyperparameter_optimization(X: pd.DataFrame, y: pd.Series, model, param_grid: Dict[str, List],
                                      search_method: str = 'grid', scoring: str = 'r2', cv: int = 5,
                                      n_iter: int = 100, random_state: int = 42) -> Tuple[Any, Dict[str, float], pd.DataFrame]:
    """
    Perform hyperparameter optimization using different search strategies
    
    Args:
        X: Feature matrix
        y: Target vector
        model: Model to optimize
        param_grid: Parameter grid or space
        search_method: 'grid', 'random', or 'bayesian'
        scoring: Scoring metric
        cv: Cross-validation folds
        n_iter: Number of iterations for random/bayesian search
        random_state: Random state
        
    Returns:
        Tuple of (best_model, optimization_results, results_df)
    """
    if search_method == 'grid':
        search = GridSearchCV(
            model, param_grid, scoring=scoring, cv=cv,
            n_jobs=-1, verbose=1, return_train_score=True
        )
    elif search_method == 'random':
        search = RandomizedSearchCV(
            model, param_grid, scoring=scoring, cv=cv,
            n_iter=n_iter, n_jobs=-1, verbose=1,
            random_state=random_state, return_train_score=True
        )
    else:
        # For bayesian, we'll use RandomizedSearchCV with continuous distributions
        # In the future, this could be extended to use scikit-optimize or optuna
        search = RandomizedSearchCV(
            model, param_grid, scoring=scoring, cv=cv,
            n_iter=n_iter, n_jobs=-1, verbose=1,
            random_state=random_state, return_train_score=True
        )
    
    search.fit(X, y)
    
    # Extract optimization results
    results_dict = {
        'best_score': search.best_score_,
        'best_params': search.best_params_,
        'cv_scores': search.cv_results_
    }
    
    # Create results DataFrame for visualization
    results_df = pd.DataFrame(search.cv_results_)
    
    return search.best_estimator_, results_dict, results_df


def get_hyperparameter_distributions(model_name: str, task_type: str) -> Dict[str, Any]:
    """
    Get continuous/discrete distributions for randomized search
    
    Args:
        model_name: Name of the model
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary of parameter distributions
    """
    distributions = {}
    
    # Get base grid parameters
    base_params = get_default_hyperparameters(model_name, task_type)
    
    # Convert discrete lists to distributions for continuous parameters
    if model_name == 'Logistic Regression':
        distributions = {
            'C': uniform(0.001, 1000),
            'penalty': base_params.get('penalty', ['l2']),
            'solver': base_params.get('solver', ['lbfgs']),
            'max_iter': randint(100, 2000),
            'class_weight': base_params.get('class_weight', ['balanced'])
        }
    elif model_name in ['Ridge Regression', 'Ridge Classifier']:
        distributions = {
            'alpha': uniform(0.001, 1000),
            'solver': base_params.get('solver', ['auto']),
            'class_weight': base_params.get('class_weight', ['balanced'])
        }
    elif model_name in ['Random Forest', 'Random Forest Regressor']:
        distributions = {
            'n_estimators': randint(50, 500),
            'max_depth': randint(3, 20),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 20),
            'max_features': uniform(0.1, 0.9),
            'class_weight': base_params.get('class_weight', ['balanced'])
        }
    elif model_name in ['Support Vector Classifier', 'Support Vector Regression']:
        distributions = {
            'C': uniform(0.001, 1000),
            'gamma': uniform(0.001, 1),
            'kernel': base_params.get('kernel', ['rbf']),
            'degree': randint(2, 6),
            'class_weight': base_params.get('class_weight', ['balanced'])
        }
    elif model_name in ['XGBoost Classifier', 'XGBoost Regressor'] and XGB_AVAILABLE:
        distributions = {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 15),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(0, 2),
            'min_child_weight': randint(1, 10),
            'class_weight': base_params.get('class_weight', ['balanced'])
        }
    elif model_name in ['LightGBM Classifier', 'LightGBM Regressor'] and LGB_AVAILABLE:
        distributions = {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 15),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(0, 2),
            'min_child_samples': randint(5, 100),
            'num_leaves': randint(15, 300),
            'is_unbalance': base_params.get('is_unbalance', [False])
        }
    elif model_name in ['CatBoost Classifier', 'CatBoost Regressor'] and CATBOOST_AVAILABLE:
        distributions = {
            'iterations': randint(100, 800),
            'learning_rate': uniform(0.01, 0.29),
            'depth': randint(3, 10),
            'l2_leaf_reg': uniform(1, 99),
            'border_count': [32, 64, 128, 254],
            'bagging_temperature': uniform(0, 5),
            'random_strength': uniform(0, 10),
            'od_type': base_params.get('od_type', ['IncToDec']),
            'od_wait': randint(10, 100)
        }
        # Add task-specific distributions
        if task_type == 'classification':
            distributions.update({
                'class_weights': base_params.get('class_weights', [[1, 1]]),
                'auto_class_weights': base_params.get('auto_class_weights', ['Balanced'])
            })
        else:
            distributions.update({
                'loss_function': base_params.get('loss_function', ['RMSE'])
            })
    elif model_name == 'Gaussian Process Classifier':
        distributions = {
            'kernel': base_params.get('kernel', ['RBF']),
            'n_restarts_optimizer': randint(0, 10),
            'max_iter_predict': randint(100, 1000),
            'warm_start': base_params.get('warm_start', [False]),
            'copy_X_train': base_params.get('copy_X_train', [True]),
            'multi_class': base_params.get('multi_class', ['one_vs_rest'])
        }
    elif model_name == 'Gaussian Process Regressor':
        distributions = {
            'kernel': base_params.get('kernel', ['RBF']),
            'alpha': uniform(1e-10, 1e-1),
            'n_restarts_optimizer': randint(0, 10),
            'normalize_y': base_params.get('normalize_y', [False]),
            'copy_X_train': base_params.get('copy_X_train', [True])
        }
    else:
        # For models without specific distributions, use the original grid
        distributions = base_params
    
    return distributions


def create_correlation_matrix_comparison(X_before: pd.DataFrame, X_after: pd.DataFrame, 
                                       threshold: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Create correlation matrices before and after filtering and identify removed features
    
    Args:
        X_before: Feature matrix before filtering
        X_after: Feature matrix after filtering
        threshold: Correlation threshold used for filtering
        
    Returns:
        Tuple of (correlation_before, correlation_after, removed_features)
    """
    # Calculate correlation matrices
    corr_before = X_before.corr()
    corr_after = X_after.corr()
    
    # Identify removed features
    removed_features = [col for col in X_before.columns if col not in X_after.columns]
    
    return corr_before, corr_after, removed_features


def visualize_hyperparameter_optimization_results(results_df: pd.DataFrame, param_names: List[str], 
                                                 scoring_metric: str) -> Dict[str, Any]:
    """
    Prepare data for visualizing hyperparameter optimization results
    
    Args:
        results_df: DataFrame with optimization results
        param_names: List of parameter names to visualize
        scoring_metric: Scoring metric used
        
    Returns:
        Dictionary with visualization data
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    viz_data = {}
    
    # Prepare score evolution data
    viz_data['scores'] = results_df[f'mean_test_score'].values
    viz_data['score_std'] = results_df[f'std_test_score'].values
    viz_data['iterations'] = range(len(viz_data['scores']))
    
    # Parameter importance analysis
    param_importance = {}
    for param in param_names:
        if f'param_{param}' in results_df.columns:
            param_values = results_df[f'param_{param}'].values
            scores = results_df[f'mean_test_score'].values
            
            # Calculate correlation between parameter values and scores
            # For categorical parameters, use different approach
            try:
                # Try to convert to numeric
                numeric_values = pd.to_numeric(param_values, errors='coerce')
                if not numeric_values.isna().all():
                    correlation = np.corrcoef(numeric_values, scores)[0, 1]
                    param_importance[param] = abs(correlation) if not np.isnan(correlation) else 0
                else:
                    # For categorical, use variance in scores across categories
                    param_df = pd.DataFrame({'param': param_values, 'score': scores})
                    variance = param_df.groupby('param')['score'].var().mean()
                    param_importance[param] = variance if not np.isnan(variance) else 0
            except:
                param_importance[param] = 0
    
    viz_data['param_importance'] = param_importance
    viz_data['best_score'] = max(viz_data['scores'])
    viz_data['best_iteration'] = np.argmax(viz_data['scores'])
    
    return viz_data


def create_model_with_params(model_class, **kwargs):
    """
    Create model instance with compatible parameters and better defaults
    
    Args:
        model_class: Model class to instantiate
        **kwargs: Parameters to pass to the model
        
    Returns:
        Model instance with optimized parameters
    """
    try:
        # Get model's valid parameters
        temp_model = model_class()
        valid_params = temp_model.get_params().keys()
        
        # Filter kwargs to only include valid parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        # Add better default parameters for specific models
        model_name = model_class.__name__
        
        if 'RandomForest' in model_name:
            # Better defaults for Random Forest
            default_rf_params = {
                'n_estimators': 100,  # More trees for better performance
                'max_depth': 10,      # Prevent overfitting
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,    # Out-of-bag scoring
                'n_jobs': -1          # Use all available cores
            }
            
            # Add class balancing for classification
            if 'Classifier' in model_name:
                default_rf_params['class_weight'] = 'balanced'
            
            # Only add defaults if not already specified
            for param, value in default_rf_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'DecisionTree' in model_name:
            # Better defaults for Decision Tree
            default_dt_params = {
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt'
            }
            
            for param, value in default_dt_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'GradientBoosting' in model_name:
            # Better defaults for Gradient Boosting
            default_gb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'subsample': 0.8
            }
            
            for param, value in default_gb_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'ExtraTrees' in model_name:
            # Better defaults for Extra Trees
            default_et_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'n_jobs': -1
            }
            
            # Add class balancing for classification
            if 'Classifier' in model_name:
                default_et_params['class_weight'] = 'balanced'
            
            for param, value in default_et_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'LogisticRegression' in model_name:
            # Better defaults for Logistic Regression
            default_lr_params = {
                'C': 1.0,
                'max_iter': 1000,  # Prevent convergence warnings
                'solver': 'lbfgs',
                'class_weight': 'balanced'  # Handle imbalanced classes
            }
            
            for param, value in default_lr_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'SVC' in model_name or 'SVM' in model_name:
            # Better defaults for SVM
            default_svm_params = {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,  # Enable probability estimates for better predictions
                'class_weight': 'balanced'  # Handle imbalanced classes
            }
            
            for param, value in default_svm_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'MLPClassifier' in model_name or 'MLPRegressor' in model_name:
            # Better defaults for Neural Networks
            default_mlp_params = {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 500,
                'alpha': 0.001,
                'solver': 'adam',
                'learning_rate': 'adaptive'
            }
            
            # Only add class_weight for classification
            if 'Classifier' in model_name:
                default_mlp_params['class_weight'] = 'balanced'
            
            for param, value in default_mlp_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'AdaBoost' in model_name:
            # Better defaults for AdaBoost
            default_ada_params = {
                'n_estimators': 100,
                'learning_rate': 1.0,
                'algorithm': 'SAMME'  # Use SAMME instead of deprecated SAMME.R
            }
            
            for param, value in default_ada_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'XGB' in model_name:
            # Better defaults for XGBoost
            default_xgb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'min_child_weight': 1,
                'random_state': 42
            }
            
            # For classification, handle class imbalance with scale_pos_weight
            # Note: XGBoost doesn't use class_weight parameter
            if 'Classifier' in model_name:
                # Will be calculated based on actual class distribution during training
                default_xgb_params['objective'] = 'binary:logistic'
                # scale_pos_weight will be set dynamically if needed
            
            for param, value in default_xgb_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'LGB' in model_name or 'LightGBM' in model_name:
            # Better defaults for LightGBM
            default_lgb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'min_child_samples': 20,
                'num_leaves': 31,
                'random_state': 42
            }
            
            # For classification, handle class imbalance
            if 'Classifier' in model_name:
                default_lgb_params['is_unbalance'] = True
                default_lgb_params['objective'] = 'binary'
                
            for param, value in default_lgb_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'CatBoost' in model_name:
            # Better defaults for CatBoost
            default_catboost_params = {
                'iterations': 200,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'border_count': 128,
                'bagging_temperature': 1,
                'random_strength': 1,
                'od_type': 'IncToDec',
                'od_wait': 20,
                'random_seed': 42,
                'logging_level': 'Silent',  # Reduce verbose output
                'allow_writing_files': False  # Prevent writing temp files
            }
            
            
            
            # For classification, handle class imbalance
            if 'Classifier' in model_name:
                default_catboost_params['auto_class_weights'] = 'Balanced'
                # Remove class_weights if auto_class_weights is used
                if 'class_weights' in filtered_kwargs and 'auto_class_weights' in filtered_kwargs:
                    del filtered_kwargs['class_weights']
            
            for param, value in default_catboost_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
                    
        elif 'GaussianProcess' in model_name:
            # Better defaults for Gaussian Process
            default_gp_params = {
                'n_restarts_optimizer': 2,
                'random_state': 42,
                'copy_X_train': True
            }
            
            # Add task-specific defaults
            if 'Classifier' in model_name:
                default_gp_params.update({
                    'max_iter_predict': 200,
                    'warm_start': False,
                    'multi_class': 'one_vs_rest'
                })
            else:  # Regressor
                default_gp_params.update({
                    'alpha': 1e-6,
                    'normalize_y': False
                })
            
            # Handle special kernel parameter processing
            if 'kernel' in filtered_kwargs:
                kernel_name = filtered_kwargs['kernel']
                if isinstance(kernel_name, str):
                    # Convert string kernel names to actual kernel objects
                    if kernel_name == 'RBF':
                        filtered_kwargs['kernel'] = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
                    elif kernel_name == 'Matern':
                        filtered_kwargs['kernel'] = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
                    elif kernel_name == 'RBF + WhiteKernel':
                        filtered_kwargs['kernel'] = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-5, (1e-10, 1e-1))
                    elif kernel_name == 'Matern + WhiteKernel':
                        filtered_kwargs['kernel'] = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-5, (1e-10, 1e-1))
                    else:
                        # Default to RBF if unknown kernel
                        filtered_kwargs['kernel'] = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            else:
                # Set default kernel if not specified
                filtered_kwargs['kernel'] = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            
            for param, value in default_gp_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
        
 
        elif 'QuantileRegressor' in model_name:
            # SPECIAL HANDLING: Quantile Regression requires user input for quantile parameter
            # quantile should be specified by user based on business needs, NOT have a default

            # Check if quantile is provided by user
            if 'quantile' not in filtered_kwargs:
                # Quantile parameter is required but not provided
                # This should trigger UI prompt for user input
                raise ValueError(
                    "QUANTILE_INPUT_REQUIRED: "
                    "Quantile Regression requires user to specify the quantile parameter. "
                    "Please choose a quantile value between 0 and 1 based on your business needs:\n"
                    "- 0.1 = 10th percentile (conservative estimate)\n"
                    "- 0.25 = 25th percentile (lower quartile)\n"
                    "- 0.5 = 50th percentile (median)\n"
                    "- 0.75 = 75th percentile (upper quartile)\n"
                    "- 0.9 = 90th percentile (optimistic estimate)"
                )

            # Validate quantile value
            quantile_value = filtered_kwargs['quantile']
            if not (0 < quantile_value < 1):
                raise ValueError(f"Quantile value must be between 0 and 1, got: {quantile_value}")

            # CRITICAL FIX: Ensure fit_intercept is boolean type, not integer
            # QuantileRegressor strictly requires bool type for fit_intercept parameter
            if 'fit_intercept' in filtered_kwargs:
                # Convert to boolean if it's an integer (1 or 0) or numpy bool
                fit_intercept_value = filtered_kwargs['fit_intercept']
                if isinstance(fit_intercept_value, (int, np.integer, np.bool_)):
                    filtered_kwargs['fit_intercept'] = bool(fit_intercept_value)
                    print(f"DEBUG: Converted fit_intercept from {type(fit_intercept_value).__name__} ({fit_intercept_value}) to bool ({filtered_kwargs['fit_intercept']})")
                elif not isinstance(fit_intercept_value, bool):
                    # Handle any other types that might need conversion
                    filtered_kwargs['fit_intercept'] = bool(fit_intercept_value)
                    print(f"DEBUG: Converted fit_intercept from {type(fit_intercept_value).__name__} ({fit_intercept_value}) to bool ({filtered_kwargs['fit_intercept']})")

            # Set other default parameters (technical parameters only)
            default_qr_params = {
                'alpha': 1.0,           # Regularization strength
                'solver': 'highs',      # Recommended solver
                'fit_intercept': True   # Include intercept (boolean type)
            }

            for param, value in default_qr_params.items():
                if param in valid_params and param not in filtered_kwargs:
                    filtered_kwargs[param] = value
        # Create model with optimized parameters
        model = model_class(**filtered_kwargs)
        
        # Print the parameters being used for debugging
        if 'random_state' in filtered_kwargs:
            print(f"Created {model_name} with parameters: {filtered_kwargs}")
        
        return model
        
    except Exception as e:
        # Check if this is a user input requirement error that should be propagated
        if "QUANTILE_INPUT_REQUIRED" in str(e) or "Quantile value must be between 0 and 1" in str(e):
            # These are special errors that UI should handle, re-raise them
            raise e

        print(f"Error creating model {model_class.__name__} with params {kwargs}: {e}")
        # Fallback: create model without any parameters
        try:
            return model_class()
        except Exception as e2:
            print(f"Error creating model {model_class.__name__} without params: {e2}")
            raise e2


def get_model_parameter_compatibility():
    """
    Get parameter compatibility information for all models

    Returns:
        Dictionary mapping model names to their supported parameters
    """
    compatibility = {}

    # Models that support random_state
    random_state_models = [
        'Random Forest', 'Random Forest Regressor',
        'Extra Trees', 'Extra Trees Regressor',
        'Decision Tree', 'Decision Tree Regressor',
        'Extra Tree', 'Extra Tree Regressor',
        'AdaBoost', 'AdaBoost Regressor',
        'Gradient Boosting', 'Gradient Boosting Regressor',
        'Bagging Classifier', 'Bagging Regressor',
        'SGD Classifier', 'SGD Regressor',
        'Logistic Regression',
        'MLP Classifier', 'MLP Regressor',
        'K-Nearest Neighbors', 'K-Nearest Neighbors Regressor'
    ]

    # Models that do NOT support random_state
    no_random_state_models = [
        'Linear Regression', 'Ridge', 'Ridge Regression', 'Ridge Classifier',
        'Lasso', 'Lasso Regression', 'ElasticNet', 'ElasticNet Regression',
        'Bayesian Ridge', 'Huber Regressor', 'Theil-Sen Regressor', 'Quantile Regression',
        'Support Vector Classifier', 'Linear SVC', 'Nu-Support Vector Classifier',
        'SVR', 'Support Vector Regression',
        'Gaussian Naive Bayes', 'Multinomial Naive Bayes', 'Bernoulli Naive Bayes',
        'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis',
        'Passive Aggressive Classifier', 'Radius Neighbors', 'Radius Neighbors Regressor'
    ]

    # Build compatibility dictionary
    for model in random_state_models:
        compatibility[model] = {'supports_random_state': True}

    for model in no_random_state_models:
        compatibility[model] = {'supports_random_state': False}

    return compatibility


def get_models_requiring_user_input():
    """
    Get models that require user input for specific parameters

    Returns:
        Dictionary mapping model names to their required user input parameters
    """
    models_requiring_input = {
        'Quantile Regression': {
            'required_params': ['quantile'],
            'param_info': {
                'quantile': {
                    'description': 'Please specify the quantile to predict (business decision parameter)',
                    'type': 'float',
                    'range': (0, 1),
                    'examples': {
                        0.1: '10th percentile (conservative estimate)',
                        0.25: '25th percentile (lower quartile)',
                        0.5: '50th percentile (median prediction)',
                        0.75: '75th percentile (upper quartile)',
                        0.9: '90th percentile (optimistic estimate)'
                    },
                    'validation_message': 'Quantile value must be between 0 and 1'
                }
            }
        }
    }

    return models_requiring_input


def check_model_requires_user_input(model_name: str) -> bool:
    """
    Check if a model requires user input for parameters

    Args:
        model_name: Name of the model

    Returns:
        True if model requires user input, False otherwise
    """
    models_requiring_input = get_models_requiring_user_input()
    return model_name in models_requiring_input


def get_user_input_requirements(model_name: str) -> dict:
    """
    Get user input requirements for a specific model

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with parameter requirements, or empty dict if no input required
    """
    models_requiring_input = get_models_requiring_user_input()
    return models_requiring_input.get(model_name, {})


def validate_model_parameters(model_name, parameters):
    """
    Validate parameters for a specific model
    
    Args:
        model_name: Name of the model
        parameters: Dictionary of parameters to validate
        
    Returns:
        Dictionary of valid parameters only
    """
    try:
        # Get model class
        task_type = 'classification' if any(x in model_name for x in ['Classifier', 'Classification']) else 'regression'
        available_models = get_available_models(task_type)
        
        if model_name not in available_models:
            print(f"Model {model_name} not found in available models")
            return {}
        
        model_class = available_models[model_name]
        
        # Get valid parameters
        temp_model = model_class()
        valid_params = temp_model.get_params().keys()
        
        # Filter parameters
        valid_parameters = {k: v for k, v in parameters.items() if k in valid_params}
        invalid_parameters = {k: v for k, v in parameters.items() if k not in valid_params}
        
        if invalid_parameters:
            print(f"Invalid parameters for {model_name}: {list(invalid_parameters.keys())}")
        
        return valid_parameters
        
    except Exception as e:
        print(f"Error validating parameters for {model_name}: {e}")
        return {} 