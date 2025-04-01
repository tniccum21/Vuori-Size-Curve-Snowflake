"""
Filter components for Streamlit UI.
"""
from typing import Dict, List, Tuple, Optional, Any, Callable
import streamlit as st
import datetime
import pandas as pd
from vuori_size_curve.config.app_config import DEFAULT_START_DATE, ANALYSIS_LEVELS
from vuori_size_curve.data.models.sales import FilterCriteria


def create_date_filter(default_start_date: str = DEFAULT_START_DATE) -> datetime.date:
    """
    Create a date filter widget.
    
    Args:
        default_start_date (str): Default start date in YYYY-MM-DD format
    
    Returns:
        datetime.date: Selected date
    """
    try:
        default_date = datetime.datetime.strptime(default_start_date, '%Y-%m-%d').date()
    except ValueError:
        default_date = datetime.date(2022, 1, 1)
    
    return st.sidebar.date_input(
        "Start Date",
        value=default_date,
        min_value=datetime.date(2020, 1, 1),
        max_value=datetime.date.today()
    )


def create_channel_filter(channels: List[str]) -> str:
    """
    Create a channel filter widget.
    
    Args:
        channels (List[str]): List of available channels
    
    Returns:
        str: Selected channel
    """
    options = ['ALL'] + sorted(channels)
    return st.sidebar.selectbox(
        "Distribution Channel",
        options=options
    )


def create_gender_filter(genders: List[str]) -> str:
    """
    Create a gender filter widget.
    
    Args:
        genders (List[str]): List of available genders
    
    Returns:
        str: Selected gender
    """
    options = ['ALL'] + sorted(genders)
    return st.sidebar.selectbox(
        "Gender",
        options=options
    )


def create_class_filter(classes: List[str]) -> str:
    """
    Create a class filter widget.
    
    Args:
        classes (List[str]): List of available classes
    
    Returns:
        str: Selected class
    """
    options = ['ALL'] + sorted(classes)
    return st.sidebar.selectbox(
        "Product Class",
        options=options
    )


def create_subclass_filter(subclasses: List[str], filter_by_class: Optional[str] = None, sales_data: Optional[pd.DataFrame] = None) -> str:
    """
    Create a subclass filter widget, optionally filtered by class.
    
    Args:
        subclasses (List[str]): List of available subclasses
        filter_by_class (Optional[str]): Class to filter by
        sales_data (Optional[pd.DataFrame]): Sales data for filtering
    
    Returns:
        str: Selected subclass
    """
    filtered_subclasses = ['ALL']
    
    if filter_by_class != 'ALL' and filter_by_class and sales_data is not None and 'PRODUCT_CLASS_TEXT' in sales_data.columns:
        # Filter subclasses by the selected class
        filtered_subclasses += sorted(sales_data[
            sales_data['PRODUCT_CLASS_TEXT'] == filter_by_class
        ]['PRODUCT_SUB_CLASS_TEXT'].unique().tolist())
    else:
        filtered_subclasses += sorted(subclasses)
    
    return st.sidebar.selectbox(
        "Product Subclass",
        options=filtered_subclasses
    )


def create_style_filter(
    styles: List[str], 
    filter_by_subclass: Optional[str] = None,
    filter_by_class: Optional[str] = None,
    sales_data: Optional[pd.DataFrame] = None
) -> str:
    """
    Create a style filter widget, optionally filtered by subclass or class.
    
    Args:
        styles (List[str]): List of available styles
        filter_by_subclass (Optional[str]): Subclass to filter by
        filter_by_class (Optional[str]): Class to filter by
        sales_data (Optional[pd.DataFrame]): Sales data for filtering
    
    Returns:
        str: Selected style
    """
    filtered_styles = ['ALL']
    
    if sales_data is not None:
        if filter_by_subclass != 'ALL' and filter_by_subclass and 'PRODUCT_SUB_CLASS_TEXT' in sales_data.columns:
            # Filter styles by the selected subclass
            filtered_styles += sorted(sales_data[
                sales_data['PRODUCT_SUB_CLASS_TEXT'] == filter_by_subclass
            ]['STYLE_CODE'].unique().tolist())
        elif filter_by_class != 'ALL' and filter_by_class and 'PRODUCT_CLASS_TEXT' in sales_data.columns:
            # Filter styles by the selected class
            filtered_styles += sorted(sales_data[
                sales_data['PRODUCT_CLASS_TEXT'] == filter_by_class
            ]['STYLE_CODE'].unique().tolist())
        else:
            filtered_styles += sorted(styles)
    else:
        filtered_styles += sorted(styles)
    
    return st.sidebar.selectbox(
        "Style Code",
        options=filtered_styles
    )


def create_min_sales_filter(default_value: int = 10) -> int:
    """
    Create a minimum sales filter widget.
    
    Args:
        default_value (int): Default minimum sales value
        
    Returns:
        int: Selected minimum sales value
    """
    return st.sidebar.number_input(
        "Minimum Sales Quantity",
        min_value=0,
        value=default_value,
        step=10
    )


def create_analysis_level_filter() -> str:
    """
    Create an analysis level filter widget.
    
    Returns:
        str: Selected analysis level
    """
    options = [level.capitalize() for level in ANALYSIS_LEVELS]
    
    # Create a unique key for this widget that includes the session ID
    # This ensures the radio button will trigger callbacks when changed
    key = f"analysis_level_{id(st.session_state)}"
    
    selected = st.sidebar.radio(
        "Analysis Level",
        options=options,
        index=0,
        key=key,
        on_change=lambda: st.session_state.update({"level_changed": True})
    )
    
    return selected.lower()


def create_run_button() -> bool:
    """
    Create a run analysis button.
    
    Returns:
        bool: True if the button was clicked
    """
    return st.sidebar.button("Run Analysis", type="primary")


def create_all_filters(sales_data: Optional[pd.DataFrame] = None) -> Tuple[FilterCriteria, str]:
    """
    Create all filter widgets and return filter criteria.
    
    Args:
        sales_data (Optional[pd.DataFrame]): Sales data for populating filters
    
    Returns:
        Tuple[FilterCriteria, str]: Filter criteria and selected analysis level
    """
    st.sidebar.header("Filters")
    
    # Extract filter values from sales data
    channels = []
    genders = []
    classes = []
    subclasses = []
    styles = []
    collections = []
    
    if sales_data is not None and not sales_data.empty:
        sales_cols = sales_data.columns.tolist()
        
        if 'DISTRIBUTION_CHANNEL_CODE' in sales_cols:
            channels = sorted(sales_data['DISTRIBUTION_CHANNEL_CODE'].unique().tolist())
        
        if 'GENDER_CODE' in sales_cols:
            genders = sorted(sales_data['GENDER_CODE'].unique().tolist())
        
        if 'PRODUCT_CLASS_TEXT' in sales_cols:
            # Filter out N/A classes
            valid_classes = sales_data['PRODUCT_CLASS_TEXT'].dropna()
            valid_classes = valid_classes[~valid_classes.str.contains('_N/A', na=False)]
            classes = sorted(valid_classes.unique().tolist())
        
        if 'PRODUCT_SUB_CLASS_TEXT' in sales_cols:
            # Filter out N/A subclasses
            valid_subclasses = sales_data['PRODUCT_SUB_CLASS_TEXT'].dropna()
            valid_subclasses = valid_subclasses[~valid_subclasses.str.contains('_N/A', na=False)]
            subclasses = sorted(valid_subclasses.unique().tolist())
        
        if 'COLLECTION_TEXT' in sales_cols:
            # Filter out N/A collections
            valid_collections = sales_data['COLLECTION_TEXT'].dropna()
            valid_collections = valid_collections[~valid_collections.str.contains('_N/A', na=False)]
            collections = sorted(valid_collections.unique().tolist())
        
        if 'STYLE_CODE' in sales_cols:
            styles = sorted(sales_data['STYLE_CODE'].unique().tolist())
    
    # Create filter widgets
    start_date = create_date_filter()
    selected_channel = create_channel_filter(channels)
    selected_gender = create_gender_filter(genders)
    selected_class = create_class_filter(classes)
    selected_subclass = create_subclass_filter(subclasses, selected_class, sales_data)
    selected_style = create_style_filter(styles, selected_subclass, selected_class, sales_data)
    min_sales = create_min_sales_filter()
    analysis_level = create_analysis_level_filter()
    
    # Convert start_date to string format
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    # Create filter criteria
    filter_criteria = FilterCriteria(
        start_date=start_date_str,
        min_sales=min_sales,
        channels_filter=[selected_channel] if selected_channel != 'ALL' else None,
        gender_filter=selected_gender if selected_gender != 'ALL' else None,
        class_filter=selected_class if selected_class != 'ALL' else None,
        subclass_filter=selected_subclass if selected_subclass != 'ALL' else None,
        style_filter=selected_style if selected_style != 'ALL' else None
    )
    
    return filter_criteria, analysis_level