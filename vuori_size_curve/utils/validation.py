"""
Validation utilities for size curve analysis.
"""
from typing import Optional, Dict, List, Any
import pandas as pd
from datetime import datetime


def validate_date_format(date_str: str) -> bool:
    """
    Validate that a date string is in YYYY-MM-DD format.
    
    Args:
        date_str (str): The date string to validate
        
    Returns:
        bool: True if the date is valid, False otherwise
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def validate_min_sales(min_sales: Any) -> int:
    """
    Validate and convert minimum sales value.
    
    Args:
        min_sales (Any): The min_sales value to validate
        
    Returns:
        int: The validated min_sales value
    """
    try:
        return max(0, int(min_sales))
    except (ValueError, TypeError):
        # Default to 0 if conversion fails
        return 0


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains the required columns.
    
    Args:
        df (pd.DataFrame): The DataFrame to validate
        required_columns (List[str]): List of required column names
        
    Returns:
        bool: True if all required columns exist, False otherwise
    """
    if df is None or df.empty:
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0