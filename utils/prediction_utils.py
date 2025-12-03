"""
Prediction utilities for handling original feature input and encoding mapping
Provides functions to map original features to encoded features for prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import re


class OriginalFeatureMapper:
    """
    Class to handle mapping between original features and encoded features for prediction
    """
    
    def __init__(self):
        self.original_to_encoded_map = {}  # original_feature -> [encoded_features]
        self.encoded_to_original_map = {}  # encoded_feature -> original_feature
        self.categorical_mappings = {}     # original_feature -> {value: encoded_columns}
        self.feature_types = {}           # original_feature -> type (numeric/categorical/boolean)
        self.original_feature_names = []
        self.encoded_feature_names = []
        
    def record_encoding_mapping(self, original_df: pd.DataFrame, encoded_df: pd.DataFrame, 
                               categorical_columns: List[str] = None):
        """
        Record the mapping between original and encoded features
        
        Args:
            original_df: DataFrame before encoding
            encoded_df: DataFrame after encoding
            categorical_columns: List of categorical columns that were encoded
        """
        try:
            self.original_feature_names = list(original_df.columns)
            self.encoded_feature_names = list(encoded_df.columns)
            
            if categorical_columns is None:
                categorical_columns = []
            
            # Identify feature types
            for col in original_df.columns:
                if col in categorical_columns:
                    self.feature_types[col] = 'categorical'
                elif pd.api.types.is_numeric_dtype(original_df[col]):
                    # Check if it's actually a boolean disguised as numeric
                    unique_vals = set(original_df[col].dropna().unique())
                    if unique_vals.issubset({0, 1, 0.0, 1.0}):
                        self.feature_types[col] = 'boolean'
                    else:
                        self.feature_types[col] = 'numeric'
                elif original_df[col].dtype == 'bool':
                    self.feature_types[col] = 'boolean'
                else:
                    # Check if it's a binary categorical that should be boolean
                    unique_vals = set(original_df[col].dropna().unique())
                    if len(unique_vals) == 2 and unique_vals.issubset({'Yes', 'No', 'True', 'False', 'yes', 'no', 'true', 'false', 'Y', 'N', 'y', 'n'}):
                        self.feature_types[col] = 'boolean'
                    else:
                        self.feature_types[col] = 'categorical'
            
            # Map features
            for original_col in original_df.columns:
                if original_col in categorical_columns:
                    # Find encoded columns for this categorical feature
                    encoded_cols = [col for col in encoded_df.columns if col.startswith(f"{original_col}_")]
                    self.original_to_encoded_map[original_col] = encoded_cols
                    
                    # Create value mapping for categorical features
                    unique_values = original_df[original_col].dropna().unique()
                    value_mapping = {}
                    
                    for value in unique_values:
                        # Find the corresponding encoded column
                        encoded_col = f"{original_col}_{value}"
                        if encoded_col in encoded_df.columns:
                            value_mapping[value] = encoded_col
                    
                    self.categorical_mappings[original_col] = value_mapping
                    
                    # Reverse mapping
                    for encoded_col in encoded_cols:
                        self.encoded_to_original_map[encoded_col] = original_col
                        
                else:
                    # Numeric or boolean features remain the same
                    if original_col in encoded_df.columns:
                        self.original_to_encoded_map[original_col] = [original_col]
                        self.encoded_to_original_map[original_col] = original_col
            
            print(f"Feature mapping recorded: {len(self.original_feature_names)} original -> {len(self.encoded_feature_names)} encoded")
            
        except Exception as e:
            print(f"Error recording encoding mapping: {e}")
    
    def create_prediction_input_interface(self) -> Dict[str, Dict]:
        """
        Create input interface specification for original features
        
        Returns:
            Dictionary with feature specifications for UI creation
        """
        interface_spec = {}
        
        for feature in self.original_feature_names:
            feature_type = self.feature_types.get(feature, 'numeric')
            
            if feature_type == 'categorical':
                # Get possible values from categorical mappings
                possible_values = list(self.categorical_mappings.get(feature, {}).keys())
                interface_spec[feature] = {
                    'type': 'categorical',
                    'widget': 'combobox',
                    'values': possible_values,
                    'default': possible_values[0] if possible_values else None
                }
            elif feature_type == 'boolean':
                interface_spec[feature] = {
                    'type': 'boolean',
                    'widget': 'checkbox',
                    'values': [True, False],
                    'default': False
                }
            else:  # numeric
                interface_spec[feature] = {
                    'type': 'numeric',
                    'widget': 'spinbox',
                    'min': -999999.0,
                    'max': 999999.0,
                    'default': 0.0,
                    'decimals': 6
                }
        
        return interface_spec
    
    def map_original_to_encoded(self, original_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Map original feature input to encoded format for model prediction
        
        Args:
            original_input: Dictionary with original feature names and values
            
        Returns:
            DataFrame with encoded features ready for model prediction
        """
        try:
            # Initialize encoded data with zeros
            encoded_data = pd.DataFrame(0, index=[0], columns=self.encoded_feature_names, dtype=float)
            
            for original_feature, value in original_input.items():
                if original_feature not in self.original_to_encoded_map:
                    print(f"Warning: Unknown original feature '{original_feature}'")
                    continue
                
                feature_type = self.feature_types.get(original_feature, 'numeric')
                encoded_features = self.original_to_encoded_map[original_feature]
                
                if feature_type == 'categorical':
                    # Handle categorical features with type-safe matching
                    if original_feature in self.categorical_mappings:
                        value_mapping = self.categorical_mappings[original_feature]
                        
                        # Try exact match first
                        if value in value_mapping:
                            encoded_col = value_mapping[value]
                            if encoded_col in encoded_data.columns:
                                encoded_data[encoded_col] = 1.0
                        else:
                            # Try string-based matching for type compatibility
                            str_value = str(value)
                            matched = False
                            for map_key, encoded_col in value_mapping.items():
                                if str(map_key) == str_value:
                                    if encoded_col in encoded_data.columns:
                                        encoded_data[encoded_col] = 1.0
                                        matched = True
                                        break
                            
                            if not matched:
                                print(f"Warning: Unknown value '{value}' for categorical feature '{original_feature}'")
                    
                elif feature_type == 'numeric':
                    # Handle numeric features (direct mapping)
                    for encoded_feature in encoded_features:
                        if encoded_feature in encoded_data.columns:
                            encoded_data[encoded_feature] = float(value)
                            
                elif feature_type == 'boolean':
                    # Handle boolean features with proper conversion
                    bool_value = self._convert_to_boolean(value)
                    for encoded_feature in encoded_features:
                        if encoded_feature in encoded_data.columns:
                            encoded_data[encoded_feature] = float(bool_value)
            
            return encoded_data
            
        except Exception as e:
            print(f"Error mapping original to encoded features: {e}")
            return pd.DataFrame()
    
    def get_feature_description(self, original_feature: str) -> str:
        """
        Get human-readable description of a feature
        
        Args:
            original_feature: Original feature name
            
        Returns:
            Human-readable description
        """
        feature_type = self.feature_types.get(original_feature, 'unknown')
        
        if feature_type == 'categorical':
            possible_values = list(self.categorical_mappings.get(original_feature, {}).keys())
            return f"Categorical feature. Possible values: {', '.join(map(str, possible_values))}"
        elif feature_type == 'boolean':
            return "Boolean feature. Choose Yes/No or True/False"
        elif feature_type == 'numeric':
            return "Numeric feature. Enter a numerical value"
        else:
            return "Unknown feature type"
    
    def validate_original_input(self, original_input: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate original feature input
        
        Args:
            original_input: Dictionary with original feature names and values
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for missing required features
        missing_features = set(self.original_feature_names) - set(original_input.keys())
        if missing_features:
            errors.append(f"Missing required features: {', '.join(missing_features)}")
        
        # Check for unknown features
        unknown_features = set(original_input.keys()) - set(self.original_feature_names)
        if unknown_features:
            errors.append(f"Unknown features: {', '.join(unknown_features)}")
        
        # Validate individual feature values
        for feature, value in original_input.items():
            if feature not in self.feature_types:
                continue
                
            feature_type = self.feature_types[feature]
            
            if feature_type == 'categorical':
                if feature in self.categorical_mappings:
                    valid_values = list(self.categorical_mappings[feature].keys())
                    # Convert both input value and valid values to strings for comparison
                    # This handles type mismatches between user input and stored values
                    valid_values_str = [str(v) for v in valid_values]
                    if str(value) not in valid_values_str:
                        errors.append(f"Invalid value '{value}' for categorical feature '{feature}'. Valid values: {valid_values}")
            
            elif feature_type == 'boolean':
                # Use the enhanced boolean conversion to validate
                try:
                    self._convert_to_boolean(value)
                except:
                    valid_bool_values = ['True', 'False', 'true', 'false', 'Yes', 'No', 'yes', 'no', 'Y', 'N', 'y', 'n', '1', '0', 'on', 'off', 'enabled', 'disabled']
                    errors.append(f"Invalid boolean value '{value}' for feature '{feature}'. Valid values: {', '.join(valid_bool_values)}")
            
            elif feature_type == 'numeric':
                try:
                    float(value)
                except (ValueError, TypeError):
                    errors.append(f"Invalid numeric value '{value}' for feature '{feature}'")
        
        return len(errors) == 0, errors
    
    def _convert_to_boolean(self, value) -> bool:
        """Convert various boolean representations to Python boolean"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, (int, float)):
            # Consider 0 as False, any other number as True
            return bool(value)
        elif isinstance(value, str):
            value_lower = value.lower().strip()
            # Extended boolean representations
            true_values = ['true', 'yes', 'y', '1', 'on', 'enabled', 'active', 'positive']
            false_values = ['false', 'no', 'n', '0', 'off', 'disabled', 'inactive', 'negative']
            
            if value_lower in true_values:
                return True
            elif value_lower in false_values:
                return False
            else:
                # If we can't match, try to convert to number first
                try:
                    return bool(float(value))
                except (ValueError, TypeError):
                    return bool(value)
        else:
            # For other types, try string conversion first
            try:
                str_value = str(value).lower().strip()
                if str_value in ['true', 'yes', 'y', '1', 'on']:
                    return True
                elif str_value in ['false', 'no', 'n', '0', 'off']:
                    return False
            except:
                pass
            return bool(value)
    
    def get_sample_input(self) -> Dict[str, Any]:
        """
        Generate sample input values for demonstration
        
        Returns:
            Dictionary with sample values for all original features
        """
        sample_input = {}
        
        for feature in self.original_feature_names:
            feature_type = self.feature_types.get(feature, 'numeric')
            
            if feature_type == 'categorical':
                possible_values = list(self.categorical_mappings.get(feature, {}).keys())
                sample_input[feature] = possible_values[0] if possible_values else 'Unknown'
            elif feature_type == 'boolean':
                sample_input[feature] = False
            else:  # numeric
                sample_input[feature] = 0.0
        
        return sample_input


# Global instance for the application
feature_mapper = OriginalFeatureMapper()


def record_feature_encoding_for_prediction(original_df: pd.DataFrame, encoded_df: pd.DataFrame, 
                                         categorical_columns: List[str] = None):
    """
    Record feature encoding mapping for prediction module
    
    Args:
        original_df: DataFrame before encoding
        encoded_df: DataFrame after encoding  
        categorical_columns: List of categorical columns that were encoded
    """
    global feature_mapper
    feature_mapper.record_encoding_mapping(original_df, encoded_df, categorical_columns)


def get_prediction_input_interface() -> Dict[str, Dict]:
    """
    Get input interface specification for prediction
    
    Returns:
        Dictionary with feature specifications for UI creation
    """
    global feature_mapper
    return feature_mapper.create_prediction_input_interface()


def map_original_input_for_prediction(original_input: Dict[str, Any]) -> pd.DataFrame:
    """
    Map original feature input to encoded format for model prediction
    
    Args:
        original_input: Dictionary with original feature names and values
        
    Returns:
        DataFrame with encoded features ready for model prediction
    """
    global feature_mapper
    return feature_mapper.map_original_to_encoded(original_input)


def validate_prediction_input(original_input: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate original feature input for prediction
    
    Args:
        original_input: Dictionary with original feature names and values
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    global feature_mapper
    return feature_mapper.validate_original_input(original_input)


def get_sample_prediction_input() -> Dict[str, Any]:
    """
    Generate sample input values for prediction demonstration
    
    Returns:
        Dictionary with sample values for all original features
    """
    global feature_mapper
    return feature_mapper.get_sample_input()


def get_feature_description(feature_name: str) -> str:
    """
    Get human-readable description of a feature
    
    Args:
        feature_name: Original feature name
        
    Returns:
        Human-readable description
    """
    global feature_mapper
    return feature_mapper.get_feature_description(feature_name) 