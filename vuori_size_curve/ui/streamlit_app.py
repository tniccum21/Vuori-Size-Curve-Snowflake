"""
Streamlit web interface for Vuori Size Curve Analyzer.
"""
import streamlit as st
import pandas as pd
import datetime
import traceback
import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Any

# Add the parent directory to the path so we can import the package
# This is only needed when running the script directly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from vuori_size_curve.data.connectors.snowflake_connector import SnowflakeConnector
from vuori_size_curve.data.repositories.product_repository import ProductRepository
from vuori_size_curve.data.repositories.sales_repository import SalesRepository
from vuori_size_curve.data.models.sales import FilterCriteria
from vuori_size_curve.analysis.analyzer_factory import AnalyzerFactory
from vuori_size_curve.config.app_config import LEVEL_COLUMN_MAP
from vuori_size_curve.ui.components.filters import (
    create_all_filters,
    create_run_button
)
from vuori_size_curve.ui.components.visualizations import (
    create_metrics,
    create_results_table,
    create_item_selection,
    create_size_curve_plot
)
from vuori_size_curve.ui.find_styles import find_contributing_styles


# Set page configuration
st.set_page_config(
    page_title="Vuori Size Curve Analyzer",
    page_icon="📊",
    layout="wide"
)

# Page title and description
st.title("Vuori Size Curve Analyzer")
st.markdown("This application analyzes sales data to generate size curves for fashion retail products.")


# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.connector = None
    st.session_state.product_repository = None
    st.session_state.sales_repository = None
    st.session_state.sales_data = None
    st.session_state.product_data = None
    st.session_state.analyzer_factory = None
    st.session_state.results = pd.DataFrame()
    st.session_state.filtered_data = None


# Initialize repositories
@st.cache_resource
def initialize_repositories():
    """Initialize database connection and repositories."""
    connector = SnowflakeConnector()
    product_repository = ProductRepository(connector)
    sales_repository = SalesRepository(connector)
    
    return connector, product_repository, sales_repository


def apply_filters_in_memory(full_data, filter_criteria):
    """
    Apply filters to the in-memory dataset instead of requerying the database.
    
    Args:
        full_data (pd.DataFrame): The full dataset to filter
        filter_criteria (FilterCriteria): The filter criteria to apply
        
    Returns:
        pd.DataFrame: The filtered dataset
    """
    filtered_data = full_data.copy()
    
    # Filter by channels if specified
    if filter_criteria.channels_filter and 'DISTRIBUTION_CHANNEL_CODE' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['DISTRIBUTION_CHANNEL_CODE'].isin(filter_criteria.channels_filter)]
    
    # Filter by gender if specified
    if filter_criteria.gender_filter and filter_criteria.gender_filter != 'ALL' and 'GENDER_CODE' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['GENDER_CODE'] == filter_criteria.gender_filter]
    
    # Filter by class if specified
    if filter_criteria.class_filter and filter_criteria.class_filter != 'ALL' and 'PRODUCT_CLASS_TEXT' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['PRODUCT_CLASS_TEXT'] == filter_criteria.class_filter]
    
    # Filter by subclass if specified
    if filter_criteria.subclass_filter and filter_criteria.subclass_filter != 'ALL' and 'PRODUCT_SUB_CLASS_TEXT' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['PRODUCT_SUB_CLASS_TEXT'] == filter_criteria.subclass_filter]
    
    # Filter by style if specified
    if filter_criteria.style_filter and filter_criteria.style_filter != 'ALL' and 'STYLE_CODE' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['STYLE_CODE'] == filter_criteria.style_filter]
    
    # Filter out invalid size data
    if 'SIZE_CODE' in filtered_data.columns and 'SIZE_ORDER' in filtered_data.columns:
        original_count = len(filtered_data)
        filtered_data = filtered_data[
            (~filtered_data['SIZE_CODE'].str.contains('_N/A', na=False)) & 
            (filtered_data['SIZE_ORDER'] > 0) & 
            (filtered_data['SIZE_ORDER'].notnull())
        ]
    
    return filtered_data


# Load initial data
if not st.session_state.initialized:
    # Show extended loading message with progress bar
    loading_text = st.empty()
    loading_text.text("Initializing data connections...")
    progress_bar = st.progress(0)
    
    try:
        # Step 1: Initialize repositories (25%)
        loading_text.text("Connecting to Snowflake...")
        connector, product_repository, sales_repository = initialize_repositories()
        progress_bar.progress(25)
        
        st.session_state.connector = connector
        st.session_state.product_repository = product_repository
        st.session_state.sales_repository = sales_repository
        
        # Step 2: Fetch all sales data (50%)
        loading_text.text("Loading full sales data from Snowflake (this may take a while)...")
        base_filter_criteria = FilterCriteria()
        st.session_state.full_sales_data = sales_repository.get_raw_data(base_filter_criteria)
        st.session_state.sales_data = st.session_state.full_sales_data.copy()
        progress_bar.progress(50)
        
        # Step 3: Load product data (75%)
        loading_text.text("Loading product data...")
        st.session_state.product_data = product_repository.get_raw_data()
        progress_bar.progress(75)
        
        # Step 4: Create analyzer and cache templates (100%)
        loading_text.text("Creating analyzers and caching templates...")
        st.session_state.analyzer_factory = AnalyzerFactory(
            product_repository=product_repository,
            sales_repository=sales_repository
        )
        
        # Cache size templates from product repository to avoid requerying
        style_templates = product_repository.determine_size_templates()
        st.session_state.style_templates = style_templates
        progress_bar.progress(100)
        
        loading_text.text("Data loading complete! The app will now be much faster when changing filters.")
        st.session_state.initialized = True
        
    except Exception as e:
        st.error(f"Error loading initial data: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        st.session_state.initialization_error = str(e)


# Create filters and track if level changed
if st.session_state.initialized and 'sales_data' in st.session_state:
    # Initialize level tracking if not already in session state
    if 'previous_analysis_level' not in st.session_state:
        st.session_state.previous_analysis_level = None
        
    # Get filters and current level selection
    filter_criteria, analysis_level = create_all_filters(st.session_state.sales_data)
    
    # Check if level has changed
    level_changed = (st.session_state.previous_analysis_level is not None and 
                     st.session_state.previous_analysis_level != analysis_level)
    
    # Update previous level for next time
    st.session_state.previous_analysis_level = analysis_level
    
    # Show run button
    run_button = create_run_button()
    
    # Auto-run if level changed
    if level_changed:
        run_button = True
else:
    filter_criteria = None
    analysis_level = "style"
    run_button = False


# Run analysis when button is clicked
if run_button and st.session_state.initialized:
    with st.spinner("Running analysis..."):
        try:
            # Get the analyzer for the selected level
            analyzer = st.session_state.analyzer_factory.get_analyzer(analysis_level)
            
            # Use our helper function to filter the data in memory
            filtered_data = apply_filters_in_memory(st.session_state.full_sales_data, filter_criteria)
                
            # Use cached templates instead of requerying
            style_templates = st.session_state.style_templates
            
            # Add TEMPLATE column to filtered_data based on PRODUCT definitions, not sales data
            filtered_data['TEMPLATE'] = ''
            
            # Use product_repository to get the complete template definitions
            product_repo = st.session_state.product_repository
            
            # Check if we have style+color templates
            if hasattr(product_repo, 'style_color_templates') and product_repo.style_color_templates:
                # Create a temporary column with style+color key for easy matching
                if 'COLOR_CODE' in filtered_data.columns:
                    filtered_data['STYLE_COLOR_KEY'] = filtered_data['STYLE_CODE'] + '_' + filtered_data['COLOR_CODE']
                    
                    # Create a mapping dictionary for fast vectorized assignment
                    template_mapping = {
                        key: obj.size_template.name 
                        for key, obj in product_repo.style_color_templates.items()
                    }
                    
                    # Use map function instead of iterating (much faster)
                    mapped_templates = filtered_data['STYLE_COLOR_KEY'].map(template_mapping)
                    # Only update where we have matches (keep '' where there's no match)
                    filtered_data.loc[mapped_templates.notna(), 'TEMPLATE'] = mapped_templates[mapped_templates.notna()]
                    
                    # Drop the temporary key column
                    filtered_data.drop(columns=['STYLE_COLOR_KEY'], inplace=True)
                
                # Use style-only templates with faster vectorized approach
                style_templates_dict = style_templates
                
                # Create a mapping dictionary for style templates
                style_template_mapping = {
                    style: obj.size_template.name 
                    for style, obj in style_templates_dict.items()
                }
                
                # Only update rows with empty templates
                empty_template_mask = filtered_data['TEMPLATE'] == ''
                if empty_template_mask.any():
                    # Apply mapping only to rows with empty templates
                    empty_template_styles = filtered_data.loc[empty_template_mask, 'STYLE_CODE']
                    mapped_templates = empty_template_styles.map(style_template_mapping)
                    # Only update cells where we have a match
                    filtered_data.loc[
                        empty_template_mask & mapped_templates.notna(), 
                        'TEMPLATE'
                    ] = mapped_templates[mapped_templates.notna()]
            
            # Before analysis, ensure we capture all possible templates for each collection
            # Get all styles and their product-defined templates for each collection
            all_style_templates = {}
            
            # Create a dictionary of style -> collection mapping using vectorized operations
            style_to_collection = {}
            if 'COLLECTION_TEXT' in filtered_data.columns and 'STYLE_CODE' in filtered_data.columns:
                # More efficient groupby that processes all at once
                style_collection_groups = filtered_data.groupby('STYLE_CODE')['COLLECTION_TEXT'].unique()
                
                # Process all at once without nested loops
                for style, collections in style_collection_groups.items():
                    # Filter out None/empty collections
                    valid_collections = [c for c in collections if c]
                    if valid_collections:
                        style_to_collection[style] = valid_collections
            
            # Now use this mapping to associate templates with collections more efficiently
            if hasattr(product_repo, 'style_color_templates'):
                # First create a mapping of style to template for all style+colors
                style_to_template = {}
                for style_color_key, template_obj in product_repo.style_color_templates.items():
                    style = style_color_key.split('_')[0]  # Extract style from style_color_key
                    template_name = template_obj.size_template.name
                    
                    if style not in style_to_template:
                        style_to_template[style] = set()
                    style_to_template[style].add(template_name)
                
                # Now process all collections in one pass
                for style, collections in style_to_collection.items():
                    if style in style_to_template:
                        templates = style_to_template[style]
                        for collection in collections:
                            if collection not in all_style_templates:
                                all_style_templates[collection] = set()
                            all_style_templates[collection].update(templates)
            
            # Store collection templates for later use
            st.session_state.collection_templates = all_style_templates
                            
            # Run the analysis
            results = analyzer.analyze(
                sales_data=filtered_data,
                style_templates=style_templates,
                channel=filter_criteria.channels_filter[0] if filter_criteria.channels_filter else None,
                product_repository=st.session_state.product_repository,  # Pass the repository for style+color templates
                all_collection_templates=all_style_templates  # Pass all possible templates for each collection
            )
            
            # Prepare output dataframe
            results_df = analyzer.prepare_output_dataframe(results)
            
            # Store results
            st.session_state.results = results_df
            st.session_state.filtered_data = filtered_data
            
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.error(f"Detailed error: {traceback.format_exc()}")
            if 'results' in st.session_state:
                del st.session_state.results


# Display results
if 'results' in st.session_state and st.session_state.results is not None and isinstance(st.session_state.results, pd.DataFrame) and not st.session_state.results.empty:
    results_df = st.session_state.results
    filtered_data = st.session_state.filtered_data
    
    # Display metrics
    create_metrics(filtered_data)
    
    # Display results table
    create_results_table(results_df, analysis_level)
    
    # Visualization section
    st.write("#### Size Curve Visualization")
    
    
    # Get the appropriate column name for the level
    level_to_col = {
        'style': 'STYLE_CODE',
        'class': 'PRODUCT_CLASS_TEXT',
        'subclass': 'PRODUCT_SUB_CLASS_TEXT',
        'collection': 'COLLECTION_TEXT'
    }
    
    level_column = level_to_col.get(analysis_level, 'STYLE_CODE')
    
    # Attempt to map column names that might be slightly different
    column_mapping = {
        'STYLE_CODE': ['style_code', 'STYLE_CODE', 'style'],
        'PRODUCT_CLASS_TEXT': ['product_class_text', 'PRODUCT_CLASS_TEXT', 'class'],
        'PRODUCT_SUB_CLASS_TEXT': ['product_sub_class_text', 'PRODUCT_SUB_CLASS_TEXT', 'subclass'],
        'COLLECTION_TEXT': ['collection_text', 'COLLECTION_TEXT', 'collection']
    }
    
    # Determine if we should use a direct level name
    actual_column = None
    
    # First try the level_column directly
    if level_column in results_df.columns:
        actual_column = level_column
    # Then try alternatives
    else:
        for possible_name in column_mapping.get(level_column, []):
            if possible_name in results_df.columns:
                actual_column = possible_name
                break
        
        # If still not found, look for level-specific columns based on analysis_level
        if actual_column is None:
            level_specific_columns = {
                'style': ['STYLE_CODE', 'style_code', 'style'],
                'class': ['PRODUCT_CLASS_TEXT', 'product_class_text', 'class'],
                'subclass': ['PRODUCT_SUB_CLASS_TEXT', 'product_sub_class_text', 'subclass'],
                'collection': ['COLLECTION_TEXT', 'collection_text', 'collection']
            }
            
            for col in level_specific_columns.get(analysis_level, []):
                if col in results_df.columns:
                    actual_column = col
                    break
    
    if actual_column:
        # Item selection
        selected_items = create_item_selection(
            results_df, 
            actual_column, 
            allow_multiple=True
        )
        
        # Create plot
        create_size_curve_plot(
            results_df,
            selected_items,
            actual_column
        )
        
        # Show styles/colors section for all levels
        if selected_items:
            if analysis_level != 'style':
                st.write("### Styles in Selection")
            else:
                st.write("### Color Analysis for Selected Style")
            
            # Define the size template column name
            size_template_col = None
            if 'size_template' in results_df.columns:
                size_template_col = 'size_template'
            elif 'SIZE_TEMPLATE' in results_df.columns:
                size_template_col = 'SIZE_TEMPLATE'
            
            # Extract selected item, gender, and template info
            for item_id in selected_items:
                parts = item_id.split(' | ')
                if len(parts) == 2:
                    item_with_gender, template = parts
                    
                    # Extract gender if present (format: "VESTS (M)" or "VESTS")
                    gender_code = None
                    if '(' in item_with_gender and ')' in item_with_gender:
                        # Extract item_code and gender
                        item_code = item_with_gender.split(' (')[0].strip()
                        gender_code = item_with_gender.split('(')[1].split(')')[0].strip()
                    else:
                        item_code = item_with_gender
                    
                    if gender_code:
                        st.write(f"#### Styles in {item_code} ({gender_code}) with template {template}")
                    else:
                        st.write(f"#### Styles in {item_code} with template {template}")
                    
                    # Process the filtered data
                    if 'STYLE_CODE' in filtered_data.columns:
                        # Check if TEMPLATE column is available (this is the most direct way to match)
                        if 'TEMPLATE' in filtered_data.columns:
                            # Get all styles that match the level and gender
                            level_to_col = {
                                'style': 'STYLE_CODE',
                                'class': 'PRODUCT_CLASS_TEXT',
                                'subclass': 'PRODUCT_SUB_CLASS_TEXT',
                                'collection': 'COLLECTION_TEXT'
                            }
                            level_column = level_to_col.get(analysis_level, 'STYLE_CODE')
                            
                            # Filter by level, gender, and template
                            match_mask = (filtered_data['TEMPLATE'] == template) & (filtered_data[level_column] == item_code)
                            if gender_code and 'GENDER_CODE' in filtered_data.columns:
                                match_mask = match_mask & (filtered_data['GENDER_CODE'] == gender_code)
                            
                            level_template_matches = filtered_data[match_mask]
                            
                            # Get unique style+colors for this combination
                            if 'COLOR_CODE' in level_template_matches.columns:
                                style_colors = level_template_matches[['STYLE_CODE', 'COLOR_CODE']].drop_duplicates()
                                
                                # Style colors are available but we don't need to show a sample
                        else:
                            # Use alternative method when TEMPLATE column is not available
                            level_to_col = {
                                'style': 'STYLE_CODE',
                                'class': 'PRODUCT_CLASS_TEXT',
                                'subclass': 'PRODUCT_SUB_CLASS_TEXT',
                                'collection': 'COLLECTION_TEXT'
                            }
                            level_column = level_to_col.get(analysis_level, 'STYLE_CODE')
                            
                            if level_column in filtered_data.columns:
                                level_matches = filtered_data[filtered_data[level_column] == item_code]
                                
                                # Further filter by gender if provided
                                if gender_code and 'GENDER_CODE' in filtered_data.columns:
                                    level_gender_matches = level_matches[level_matches['GENDER_CODE'] == gender_code]
                                    
                                    # Get unique styles at this level+gender
                                    level_gender_styles = level_gender_matches['STYLE_CODE'].unique()
                                    
                                    # Show a sample of these styles
                                    if len(level_gender_styles) > 0:
                                        st.write(f"Sample styles: {level_gender_styles[:5].tolist()}")
                                        
                                        # Template information note removed
                            
                        # Template explanation removed
                        
                        
                        # Template information removed
                    
                    # Find the contributing styles using our updated utility
                    all_item_styles = find_contributing_styles(
                        filtered_data=filtered_data,
                        results_df=results_df,
                        item_code=item_code,
                        template=template,
                        gender_code=gender_code,
                        analysis_level=analysis_level
                    )
                    
                    if all_item_styles:
                        # Check if we need to display style-color combinations
                        if isinstance(all_item_styles, tuple) and len(all_item_styles) == 2:
                            styles, style_colors = all_item_styles
                            
                            # Sort styles for consistent display
                            styles.sort()
                            
                            # Show a toggle for simple vs detailed view with a unique key
                            checkbox_key = f"{item_code}_{template}_{gender_code}".replace(" ", "_")
                            show_details = st.checkbox(f"Show color details for {item_code}", 
                                                     value=False, 
                                                     key=checkbox_key)
                            
                            # Store the color preference in session state for use in visualization
                            st.session_state.show_color_details = show_details
                            st.session_state.current_style_colors = style_colors if show_details else None
                            
                            if show_details:
                                st.write("### Color Analysis")
                                # Create a size curve by color visualization
                                from vuori_size_curve.ui.components.visualizations import create_size_curve_by_color
                                
                                # No more debug output needed
                                
                                # Generate size curve by color chart
                                color_curve_fig = create_size_curve_by_color(
                                    filtered_data=filtered_data,
                                    item_code=item_code,
                                    template=template,
                                    gender_code=gender_code,
                                    analysis_level=analysis_level
                                )
                                st.plotly_chart(color_curve_fig, use_container_width=True)
                                
                                st.write("### Style-Color combinations that match this template:")
                                
                                # Also show the detailed list
                                for i, combo in enumerate(style_colors[:50]):  # Limit to first 50
                                    style, color = combo
                                    st.write(f"- {style} ({color})")
                                
                                if len(style_colors) > 50:
                                    st.info(f"Showing 50 of {len(style_colors)} style-color combinations.")
                                
                                st.write(f"Total unique styles: {len(styles)}")
                            else:
                                # Create a simplified display with just styles
                                num_cols = 3
                                cols = st.columns(num_cols)
                                for i, style in enumerate(styles[:30]):  # Limit to first 30
                                    cols[i % num_cols].write(f"- {style}")
                                
                                if len(styles) > 30:
                                    st.info(f"Showing 30 of {len(styles)} styles.")
                                else:
                                    st.write(f"Total Styles: {len(styles)}")
                        else:
                            # Sortable list for consistent display
                            if isinstance(all_item_styles, list):
                                all_item_styles.sort()
                                
                            # Fallback to original display
                            num_cols = 3
                            cols = st.columns(num_cols)
                            for i, style in enumerate(all_item_styles[:30]):  # Limit to first 30
                                cols[i % num_cols].write(f"- {style}")
                            
                            if len(all_item_styles) > 30:
                                st.info(f"Showing 30 of {len(all_item_styles)} styles.")
                            else:
                                st.write(f"Total Styles: {len(all_item_styles)}")
                    else:
                        st.info("No styles found for this selection.")
    else:
        st.info("Please click 'Run Analysis' to generate results for this level.")

else:
    # Initial state message
    st.info("Use the filters in the sidebar and click 'Run Analysis' to generate size curves.")
    
    # Data summary
    if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
        sales_data = st.session_state.sales_data
        st.write("### Data Summary")
        st.write(f"Total Records: {len(sales_data):,}")
        
        if 'STYLE_CODE' in sales_data.columns:
            st.write(f"Unique Styles: {sales_data['STYLE_CODE'].nunique():,}")


# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Snowflake")