"""
Date helper utilities for size curve analysis.
"""
from datetime import datetime, timedelta
from typing import Optional


def get_default_start_date(months_back: int = 12) -> str:
    """
    Get a default start date (X months from current date).
    
    Args:
        months_back (int): Number of months to subtract from current date
    
    Returns:
        str: Default start date in YYYY-MM-DD format
    """
    today = datetime.now()
    # Subtract months from the current date
    start_date = today.replace(day=1) - timedelta(days=1)
    for _ in range(months_back - 1):
        start_date = start_date.replace(day=1) - timedelta(days=1)
    
    return start_date.strftime('%Y-%m-%d')


def format_date_for_output(date_str: Optional[str] = None) -> str:
    """
    Format a date string for output directory names.
    
    Args:
        date_str (Optional[str]): Date string in YYYY-MM-DD format
    
    Returns:
        str: Formatted date string (YYYYMMDD)
    """
    if date_str:
        # Parse the date from YYYY-MM-DD format
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj.strftime('%Y%m%d')
        except ValueError:
            pass
    
    # Fallback to current date
    return datetime.now().strftime('%Y%m%d')


def get_timestamp_str() -> str:
    """
    Get a timestamp string for filenames.
    
    Returns:
        str: Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")