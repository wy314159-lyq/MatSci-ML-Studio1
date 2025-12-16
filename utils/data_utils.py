"""
Data utilities for MatSci-ML Studio
Provides functions for data import, validation, and basic preprocessing
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import io


def load_csv_file(file_path: str, delimiter: str = ',', encoding: str = 'utf-8', 
                  header: Optional[int] = 0) -> pd.DataFrame:
    """
    Load CSV file with specified parameters
    
    Args:
        file_path: Path to CSV file
        delimiter: Column delimiter
        encoding: File encoding
        header: Row to use as header
        
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, header=header)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")


def load_excel_file(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load Excel file with optional sheet selection
    
    Args:
        file_path: Path to Excel file
        sheet_name: Name of sheet to load (None for first sheet)
        
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        raise ValueError(f"Error loading Excel file: {str(e)}")


def get_excel_sheet_names(file_path: str) -> List[str]:
    """
    Get list of sheet names from Excel file
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        List of sheet names
    """
    try:
        xl_file = pd.ExcelFile(file_path)
        return xl_file.sheet_names
    except Exception as e:
        raise ValueError(f"Error reading Excel file sheets: {str(e)}")


def load_clipboard_data() -> pd.DataFrame:
    """
    Load data from clipboard
    
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_clipboard()
        return df
    except Exception as e:
        raise ValueError(f"Error loading clipboard data: {str(e)}")


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing quality metrics
    """
    report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicated_rows': df.duplicated().sum(),
        'unique_values': df.nunique().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    return report


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        series: Input series
        factor: IQR factor for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method
    
    Args:
        series: Input series
        threshold: Z-score threshold
        
    Returns:
        Boolean series indicating outliers
    """
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold


def validate_feature_target_selection(df: pd.DataFrame, feature_cols: List[str], 
                                     target_col: str) -> Tuple[bool, str]:
    """
    Validate feature and target column selection
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not feature_cols:
        return False, "No feature columns selected"
    
    if not target_col:
        return False, "No target column selected"
    
    if target_col in feature_cols:
        return False, "Target column cannot be in feature columns"
    
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        return False, f"Feature columns not found: {missing_features}"
    
    if target_col not in df.columns:
        return False, f"Target column not found: {target_col}"
    
    return True, ""


def suggest_task_type(target_series: pd.Series) -> str:
    """
    Suggest ML task type based on target variable
    
    Args:
        target_series: Target variable series
        
    Returns:
        Suggested task type ('classification' or 'regression')
    """
    # Check if target is numeric
    if pd.api.types.is_numeric_dtype(target_series):
        # Check number of unique values
        unique_values = target_series.nunique()
        total_values = len(target_series)
        
        # If target is float type, it's almost always regression
        if pd.api.types.is_float_dtype(target_series):
            return "regression"
        
        # For integer types, apply more sophisticated logic
        if unique_values < 10:
            # Check if values look like class labels (0,1,2,3...) or actual measurements
            sorted_unique = sorted(target_series.unique())
            # If values are consecutive integers starting from 0 or 1, likely classification
            if (sorted_unique == list(range(min(sorted_unique), max(sorted_unique) + 1)) and 
                min(sorted_unique) in [0, 1]):
                return "classification"
            # If very few unique values relative to sample size, also likely classification
            elif (unique_values / total_values) < 0.05:
                return "classification"
            else:
                return "regression"
        elif unique_values < 20 and (unique_values / total_values) < 0.05:
            # Many samples with few unique values - likely classification
            return "classification"
        else:
            return "regression"
    else:
        return "classification"


def safe_column_conversion(series: pd.Series, target_type: str) -> Tuple[pd.Series, bool, str]:
    """
    Safely convert series to target data type
    
    Args:
        series: Input series
        target_type: Target data type ('numeric', 'category', 'datetime')
        
    Returns:
        Tuple of (converted_series, success, error_message)
    """
    try:
        if target_type == 'numeric':
            converted = pd.to_numeric(series, errors='coerce')
            return converted, True, ""
        elif target_type == 'category':
            converted = series.astype('category')
            return converted, True, ""
        elif target_type == 'datetime':
            converted = pd.to_datetime(series, errors='coerce')
            return converted, True, ""
        else:
            return series, False, f"Unknown target type: {target_type}"
    except Exception as e:
        return series, False, f"Conversion error: {str(e)}" 