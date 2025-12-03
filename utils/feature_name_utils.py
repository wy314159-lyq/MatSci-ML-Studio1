"""
Feature name utilities for handling encoded feature names and their restoration
Provides functions to map encoded feature names back to original meaningful names
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any


class FeatureNameMapper:
    """
    Class to handle mapping between original and encoded feature names
    """
    
    def __init__(self):
        self.original_to_encoded_map = {}  # original_feature -> [encoded_features]
        self.encoded_to_original_map = {}  # encoded_feature -> original_feature
        self.encoding_method = None
        self.categorical_columns = []
        self.original_feature_names = []
        
    def record_encoding(self, original_df: pd.DataFrame, encoded_df: pd.DataFrame, 
                       categorical_columns: List[str], encoding_method: str = "One-Hot Encoding"):
        """
        Record the mapping between original and encoded feature names
        
        Args:
            original_df: DataFrame before encoding
            encoded_df: DataFrame after encoding
            categorical_columns: List of categorical columns that were encoded
            encoding_method: Method used for encoding
        """
        self.encoding_method = encoding_method
        self.categorical_columns = categorical_columns
        self.original_feature_names = original_df.columns.tolist()
        
        # Clear existing mappings
        self.original_to_encoded_map.clear()
        self.encoded_to_original_map.clear()
        
        # Map numeric features (unchanged)
        numeric_features = [col for col in original_df.columns if col not in categorical_columns]
        for feature in numeric_features:
            if feature in encoded_df.columns:
                self.original_to_encoded_map[feature] = [feature]
                self.encoded_to_original_map[feature] = feature
        
        # Map categorical features (encoded)
        if encoding_method == "One-Hot Encoding":
            for original_feature in categorical_columns:
                encoded_features = [col for col in encoded_df.columns 
                                  if col.startswith(f"{original_feature}_")]
                if encoded_features:
                    self.original_to_encoded_map[original_feature] = encoded_features
                    for encoded_feature in encoded_features:
                        self.encoded_to_original_map[encoded_feature] = original_feature
        
        elif encoding_method == "Label Encoding":
            for feature in categorical_columns:
                if feature in encoded_df.columns:
                    self.original_to_encoded_map[feature] = [feature]
                    self.encoded_to_original_map[feature] = feature
    
    def get_readable_feature_name(self, encoded_name: str) -> str:
        """
        Convert encoded feature name to readable format
        
        Args:
            encoded_name: Encoded feature name (e.g., "SMOKING_1")
            
        Returns:
            Readable feature name (e.g., "Smoking: Yes")
        """
        # If it's a direct mapping, return the original name
        if encoded_name in self.encoded_to_original_map:
            original_name = self.encoded_to_original_map[encoded_name]
            
            # For one-hot encoded features, create readable labels
            if self.encoding_method == "One-Hot Encoding" and "_" in encoded_name:
                # Extract the value part (e.g., "1" from "SMOKING_1")
                parts = encoded_name.split("_")
                if len(parts) >= 2:
                    value_part = parts[-1]
                    feature_part = "_".join(parts[:-1])
                    
                    # Create readable format
                    readable_feature = self._format_feature_name(feature_part)
                    readable_value = self._format_value_name(value_part)
                    
                    return f"{readable_feature}: {readable_value}"
            
            # For other encoding methods or direct features
            return self._format_feature_name(original_name)
        
        # Fallback: try to infer from the name itself
        return self._infer_readable_name(encoded_name)
    
    def _format_feature_name(self, name: str) -> str:
        """Format feature name to be more readable"""
        # Replace underscores with spaces and title case
        formatted = name.replace("_", " ").title()
        
        # Handle common abbreviations
        replacements = {
            "Id": "ID",
            "Url": "URL",
            "Api": "API",
            "Ui": "UI",
            "Db": "Database",
            "Avg": "Average",
            "Max": "Maximum",
            "Min": "Minimum",
            "Std": "Standard Deviation",
            "Pct": "Percentage"
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _format_value_name(self, value: str) -> str:
        """Format value name to be more readable"""
        # Common value mappings
        value_mappings = {
            "0": "No",
            "1": "Yes",
            "false": "No",
            "true": "Yes",
            "0.0": "No",
            "1.0": "Yes"
        }
        
        return value_mappings.get(value.lower(), value.title())
    
    def _infer_readable_name(self, encoded_name: str) -> str:
        """Infer readable name when no mapping is available"""
        # Try to detect one-hot encoding pattern
        if "_" in encoded_name:
            parts = encoded_name.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                feature_part = "_".join(parts[:-1])
                value_part = parts[-1]
                
                readable_feature = self._format_feature_name(feature_part)
                readable_value = self._format_value_name(value_part)
                
                return f"{readable_feature}: {readable_value}"
        
        # Default formatting
        return self._format_feature_name(encoded_name)
    
    def aggregate_feature_importance(self, importance_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate feature importance for encoded features back to original features
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            
        Returns:
            DataFrame with aggregated importance for original features
        """
        if not self.original_to_encoded_map:
            # No mapping available, return as-is with readable names
            result_df = importance_df.copy()
            result_df['readable_feature'] = result_df['feature'].apply(self.get_readable_feature_name)
            return result_df
        
        # Aggregate importance by original features
        aggregated_importance = {}
        
        for _, row in importance_df.iterrows():
            encoded_feature = row['feature']
            importance = row['importance']
            
            # Find the original feature
            original_feature = self.encoded_to_original_map.get(encoded_feature, encoded_feature)
            
            # Aggregate importance
            if original_feature in aggregated_importance:
                aggregated_importance[original_feature] += importance
            else:
                aggregated_importance[original_feature] = importance
        
        # Create result DataFrame
        result_df = pd.DataFrame([
            {
                'feature': original_feature,
                'importance': importance,
                'readable_feature': self._format_feature_name(original_feature)
            }
            for original_feature, importance in aggregated_importance.items()
        ]).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return result_df
    
    def create_readable_correlation_labels(self, feature_names: List[str]) -> List[str]:
        """
        Create readable labels for correlation matrix
        
        Args:
            feature_names: List of encoded feature names
            
        Returns:
            List of readable feature names
        """
        return [self.get_readable_feature_name(name) for name in feature_names]


# Global instance for easy access
feature_name_mapper = FeatureNameMapper()


def record_feature_encoding(original_df: pd.DataFrame, encoded_df: pd.DataFrame, 
                           categorical_columns: List[str], encoding_method: str = "One-Hot Encoding"):
    """
    Record feature encoding mapping globally
    
    Args:
        original_df: DataFrame before encoding
        encoded_df: DataFrame after encoding
        categorical_columns: List of categorical columns that were encoded
        encoding_method: Method used for encoding
    """
    feature_name_mapper.record_encoding(original_df, encoded_df, categorical_columns, encoding_method)


def get_readable_feature_names(encoded_names: List[str]) -> List[str]:
    """
    Get readable feature names for a list of encoded names
    
    Args:
        encoded_names: List of encoded feature names
        
    Returns:
        List of readable feature names
    """
    return [feature_name_mapper.get_readable_feature_name(name) for name in encoded_names]


def aggregate_feature_importance_by_original(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate feature importance by original features
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        
    Returns:
        DataFrame with aggregated importance
    """
    return feature_name_mapper.aggregate_feature_importance(importance_df)


def create_readable_correlation_labels(feature_names: List[str]) -> List[str]:
    """
    Create readable labels for correlation matrix
    
    Args:
        feature_names: List of encoded feature names
        
    Returns:
        List of readable feature names
    """
    return feature_name_mapper.create_readable_correlation_labels(feature_names) 