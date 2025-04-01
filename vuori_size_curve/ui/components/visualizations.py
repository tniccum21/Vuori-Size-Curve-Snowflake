"""
Visualization components for the size curve analysis.
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter


def create_metrics(filtered_data: pd.DataFrame) -> None:
    """
    Display key metrics as Streamlit metrics.
    
    Args:
        filtered_data (pd.DataFrame): Filtered sales data
    """
    # KPI row
    metrics_container = st.container()
    col1, col2, col3 = metrics_container.columns(3)
    
    # Check if the expected columns exist in the dataframe
    if 'STYLE_CODE' in filtered_data.columns:
        col1.metric("Total Styles", f"{filtered_data['STYLE_CODE'].nunique():,}")
    
    if 'ORDER_ID' in filtered_data.columns:
        col2.metric("Orders", f"{filtered_data['ORDER_ID'].nunique():,}")
    
    if 'QTY' in filtered_data.columns:
        col3.metric("Total Units", f"{filtered_data['QTY'].sum():,}")


def create_results_table(results_df: pd.DataFrame, analysis_level: str) -> None:
    """
    Create and display the results table.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame
        analysis_level (str): Analysis level (style, class, subclass, collection)
    """
    st.write("#### Size Curve Analysis Results")
    
    # Determine which columns to show based on analysis level
    display_columns = []
    
    # Map analysis level to column name
    level_to_col = {
        'style': 'STYLE_CODE',
        'class': 'PRODUCT_CLASS_TEXT',
        'subclass': 'PRODUCT_SUB_CLASS_TEXT', 
        'collection': 'COLLECTION_TEXT'
    }
    
    # Include level-specific column
    level_column = level_to_col.get(analysis_level, 'STYLE_CODE')
    
    # Check if column exists (in upper or lower case)
    for col in [level_column, level_column.lower()]:
        if col in results_df.columns:
            display_columns.append(col)
            break
    
    # Add other important columns
    meta_columns = ['size_template', 'SIZE_TEMPLATE', 'GENDER_CODE', 'gender_code']
    for col in meta_columns:
        if col in results_df.columns:
            display_columns.append(col)
    
    # Add size columns
    size_columns = [col for col in results_df.columns if col not in display_columns and not col.startswith('item_')]
    display_columns.extend(size_columns)
    
    # Show table
    st.dataframe(results_df[display_columns])


def create_item_selection(
    results_df: pd.DataFrame, 
    level_column: str, 
    allow_multiple: bool = False
) -> List[str]:
    """
    Create an item selection widget.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame
        level_column (str): Column name for the level
        allow_multiple (bool): Whether to allow multiple selection
    
    Returns:
        List[str]: Selected item IDs
    """
    # Normalize column names to handle case differences
    size_template_col = 'SIZE_TEMPLATE'  # Standardized to uppercase
    if size_template_col not in results_df.columns and 'size_template' in results_df.columns:
        size_template_col = 'size_template'
    
    # Create a combined identifier that includes item name, gender, and size template
    if level_column in results_df.columns and size_template_col is not None:
        # Check if we have gender information
        gender_col = 'GENDER_CODE'
        if gender_col not in results_df.columns:
            gender_col = 'gender_code' if 'gender_code' in results_df.columns else None
        
        if gender_col:
            # Filter out rows with null gender codes - don't create dummy UNISEX entries
            valid_results = results_df[results_df[gender_col].notna()].copy()
            
            # Include gender in the display ID (don't use fillna anymore)
            valid_results['item_template_id'] = (
                valid_results[level_column] + ' (' + 
                valid_results[gender_col] + ') | ' + 
                valid_results[size_template_col].fillna('Unknown')
            )
            
            # Replace results_df with the filtered version
            results_df = valid_results
        else:
            # Fall back to just item and template
            results_df['item_template_id'] = results_df[level_column] + ' | ' + results_df[size_template_col].fillna('Unknown')
        
        # Create a dictionary to map the combined IDs back to the original components
        id_cols = ['item_template_id', level_column, size_template_col]
        if gender_col:
            id_cols.append(gender_col)
        id_template_map = results_df[id_cols].drop_duplicates().set_index('item_template_id').to_dict('index')
        
        # Create a selection mode option
        if allow_multiple:
            selection_mode = st.radio(
                "Selection Mode:",
                ["Single Item", "Multiple Items Comparison"],
                index=0
            )
        else:
            selection_mode = "Single Item"
        
        # Item selection based on mode
        # Get only the combinations with actual data in the results
        valid_items = []
        for item_id in results_df['item_template_id'].unique():
            parts = item_id.split(' | ')
            if len(parts) == 2:
                item_with_gender, template = parts
                
                # Extract gender if present
                gender_code = None
                if '(' in item_with_gender and ')' in item_with_gender:
                    item_code = item_with_gender.split(' (')[0].strip()
                    gender_code = item_with_gender.split('(')[1].split(')')[0].strip()
                else:
                    item_code = item_with_gender
                
                # Check if there's actual data for this item+gender+template
                mask = results_df['item_template_id'] == item_id
                if mask.any() and sum(mask) > 0:
                    valid_items.append(item_id)
        
        # Sort for consistent display
        items = sorted(valid_items)
        
        if selection_mode == "Single Item":
            # Single item selection with template
            selected_item_with_template = st.selectbox(
                "Select item and template to visualize", 
                items
            )
            return [selected_item_with_template]
        else:
            # Multi-select for items with templates
            selected_items_with_templates = st.multiselect(
                "Select items and templates to compare",
                items,
                default=[items[0]] if items else []
            )
            
            if not selected_items_with_templates:
                st.warning("Please select at least one item to visualize.")
                return []
                
            return selected_items_with_templates
    else:
        st.warning(f"Required columns not found in results. Needed: {level_column}, SIZE_TEMPLATE")
        return []


def create_size_curve_plot(
    results_df: pd.DataFrame,
    selected_items: List[str],
    level_column: str
) -> None:
    """
    Create a size curve plot.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame
        selected_items (List[str]): Selected item IDs
        level_column (str): Column name for the level
    """
    if not selected_items:
        return
        
    # Normalize column names
    size_template_col = None
    if 'size_template' in results_df.columns:
        size_template_col = 'size_template'
    elif 'SIZE_TEMPLATE' in results_df.columns:
        size_template_col = 'SIZE_TEMPLATE'
    
    # Create a map from item_template_id to its components
    id_map = {}
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
                
            id_map[item_id] = {
                'item_code': item_code,
                'gender_code': gender_code,
                'template': template
            }
    
    # Group templates into letter sizes and numeric sizes
    letter_templates = []
    numeric_templates = []
    
    for item_id, details in id_map.items():
        template = details['template']
        # Check if the first size in the template is numeric or letter
        first_size = template.split('-')[0] if '-' in template else template
        if first_size and first_size[0].isdigit():
            numeric_templates.append(item_id)
        else:
            letter_templates.append(item_id)
    
    # Determine if we need one or two plots
    has_letter_sizes = len(letter_templates) > 0
    has_numeric_sizes = len(numeric_templates) > 0
    
    # We'll display the color chart separately in streamlit_app.py
    
    # Display the standard size curve plot
    if has_letter_sizes and has_numeric_sizes:
        # Create a dual-template plot with letter sizes and numeric sizes
        fig = create_letter_numeric_plot(results_df, letter_templates, numeric_templates, id_map, level_column, size_template_col)
    else:
        # Create a single plot for all items (they're all the same type)
        fig = create_single_template_plot(results_df, selected_items, id_map, level_column, size_template_col)
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


def create_single_template_plot(
    results_df: pd.DataFrame, 
    selected_items: List[str], 
    id_map: Dict[str, Dict[str, str]], 
    level_column: str, 
    size_template_col: str
) -> go.Figure:
    """
    Create a plot for items with the same template.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame
        selected_items (List[str]): Selected item IDs
        id_map (Dict[str, Dict[str, str]]): Map of item IDs to details
        level_column (str): Column name for the level
        size_template_col (str): Column name for size template
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Identify gender column if it exists
    gender_col = None
    for possible_col in ['GENDER_CODE', 'gender_code', 'Gender']:
        if possible_col in results_df.columns:
            gender_col = possible_col
            break
    
    # Plot each selected item
    for item_id in selected_items:
        if item_id not in id_map:
            continue
            
        item_code = id_map[item_id]['item_code']
        template = id_map[item_id]['template']
        gender_code = id_map[item_id].get('gender_code')
        
        # Create a display name that includes gender if available
        display_name = item_code
        if gender_code:
            display_name = f"{item_code} ({gender_code})"
        
        # Get data for this item
        item_data = None
        if size_template_col is not None:
            # Create the filter mask
            mask = (results_df[level_column] == item_code) & (results_df[size_template_col] == template)
            
            # Add gender filter if both gender code and gender column exist
            if gender_code and gender_col:
                mask = mask & (results_df[gender_col] == gender_code)
                
            item_data = results_df[mask]
        
        if item_data is None or item_data.empty:
            continue
        
        # Get template sizes in order
        template_sizes = template.split('-')
        
        # Extract percentages for each size
        x_values = []
        y_values = []
        
        for size in template_sizes:
            if size in item_data.columns:
                x_values.append(size)
                y_values.append(item_data[size].iloc[0])
        
        # Add line to figure
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=display_name
            )
        )
    
    # Set layout
    title = "Size Curve Comparison" if len(selected_items) > 1 else f"Size Curve for {id_map[selected_items[0]]['item_code']}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Size",
        yaxis_title="Percentage (%)",
        legend_title="Items"
    )
    
    # Set y-axis range from 0-100%
    max_y = 0
    for trace in fig.data:
        if len(trace.y) > 0:  # Check if there are any y values
            current_max = max(trace.y)
            if current_max > max_y:
                max_y = current_max
    
    y_max = max(100, max_y * 1.1)
    fig.update_yaxes(range=[0, y_max])
    
    return fig


def create_size_curve_by_color(
    filtered_data: pd.DataFrame,
    item_code: str,
    template: str, 
    gender_code: str = None,
    analysis_level: str = 'style'
) -> go.Figure:
    """
    Create a size curve plot showing data for each color of a specific style or category.
    
    Args:
        filtered_data (pd.DataFrame): Raw sales data
        item_code (str): Style code or category code to analyze
        template (str): Size template (e.g., "XS-S-M-L-XL")
        gender_code (str, optional): Gender code for filtering
        analysis_level (str, optional): Analysis level (style, class, subclass, collection)
        
    Returns:
        go.Figure: Plotly figure with size curves by color
    """
    import plotly.express as px
    
    # Create a figure with placeholder data (empty plot)
    fig = go.Figure()
    
    # Add a placeholder trace to ensure x-axis shows correctly
    # This is important to ensure sizes are displayed properly
    template_sizes = template.split('-')
    fig.add_trace(
        go.Scatter(
            x=template_sizes,
            y=[0] * len(template_sizes),
            mode='lines',
            name='Placeholder',
            opacity=0
        )
    )
    
    # Map analysis level to column name
    level_to_col = {
        'style': 'STYLE_CODE',
        'class': 'PRODUCT_CLASS_TEXT',
        'subclass': 'PRODUCT_SUB_CLASS_TEXT',
        'collection': 'COLLECTION_TEXT'
    }
    level_column = level_to_col.get(analysis_level, 'STYLE_CODE')
    
    # Check if necessary columns exist
    if level_column not in filtered_data.columns:
        fig.update_layout(
            title=f"Cannot create color analysis - {level_column} column not found",
            xaxis_title="Size", 
            yaxis_title="Percentage (%)"
        )
        return fig
    
    # Use the appropriate column based on analysis level
    mask = filtered_data[level_column] == item_code
    
    # Add template filter if provided
    if template and 'TEMPLATE' in filtered_data.columns:
        mask = mask & (filtered_data['TEMPLATE'] == template)
    
    # Add gender filter if provided
    if gender_code and 'GENDER_CODE' in filtered_data.columns:
        mask = mask & (filtered_data['GENDER_CODE'] == gender_code)
    
    # Get data for this level/template/gender
    filtered_level_data = filtered_data[mask]
    
    if filtered_level_data.empty:
        fig.update_layout(
            title=f"No data found for {item_code} with template {template}",
            xaxis_title="Size",
            yaxis_title="Percentage (%)"
        )
        return fig
    
    # Get unique colors for this level
    if 'COLOR_CODE' not in filtered_level_data.columns:
        fig.update_layout(
            title="Cannot create color analysis - COLOR_CODE column not found",
            xaxis_title="Size", 
            yaxis_title="Percentage (%)"
        )
        return fig
        
    colors = filtered_level_data['COLOR_CODE'].unique()
    
    # Create a dataframe to hold the size distributions by color
    color_size_data = {}
    
    # Check if we have SIZE_CODE in the data
    size_col = None
    for possible_col in ['SIZE_CODE', 'SIZE', 'size_code', 'size', 'SIZE_DESC', 'size_desc', 'SIZE_DESCRIPTION', 'size_description']:
        if possible_col in filtered_level_data.columns:
            size_col = possible_col
            break
    
    # We know from debug output that 'TOTAL_SALES' is the quantity column
    qty_col = 'TOTAL_SALES'
    
    # If TOTAL_SALES isn't in the columns, try other possible names
    if qty_col not in filtered_level_data.columns:
        for possible_col in ['QTY', 'QUANTITY', 'qty', 'quantity', 'total_sales', 
                            'UNITS', 'units', 'UNIT_SALES', 'unit_sales', 'SALES_QTY', 'sales_qty',
                            'SALES_UNITS', 'sales_units', 'QTY_SOLD', 'qty_sold']:
            if possible_col in filtered_level_data.columns:
                qty_col = possible_col
                break
                
        # If we still don't have a quantity column, look for any column that might contain numeric sales data
        if qty_col not in filtered_level_data.columns:
            for col in filtered_level_data.columns:
                if col not in ['STYLE_CODE', 'COLOR_CODE', 'SIZE_CODE', 'GENDER_CODE', 'TEMPLATE'] and \
                pd.api.types.is_numeric_dtype(filtered_level_data[col]) and \
                'ID' not in col.upper() and 'DATE' not in col.upper() and 'ORDER' not in col.upper():
                    # Check if column has positive values that could represent quantities
                    if filtered_level_data[col].max() > 0:
                        qty_col = col
                        break
    
    if size_col and qty_col:
        # Get total quantities for all sizes for each color
        for color in colors:
            color_data = filtered_level_data[filtered_level_data['COLOR_CODE'] == color]
            total_qty = color_data[qty_col].sum()
            
            if total_qty == 0:
                continue
                
            # Group by size and calculate percentages
            size_groups = color_data.groupby(size_col)[qty_col].sum()
            
            # Convert to percentages
            size_percentages = {}
            for size in template_sizes:
                if size in size_groups.index:
                    size_percentages[size] = (size_groups[size] / total_qty) * 100
                else:
                    size_percentages[size] = 0
                    
            color_size_data[color] = size_percentages
    
        # Plot size curves for each color
        for color, size_data in color_size_data.items():
            x_values = []
            y_values = []
            
            for size in template_sizes:
                if size in size_data:
                    x_values.append(size)
                    y_values.append(size_data[size])
            
            # Skip if no data
            if not x_values:
                continue
                
            # Add line to figure
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=color
                )
            )
    
    # Set layout
    if len(fig.data) > 1:  # More than just the placeholder trace
        title = f"Size Curve by Color for {item_code} ({analysis_level.title()})"
    elif size_col is None or qty_col is None:
        if size_col is None:
            title = "Cannot create color analysis - Size column not found"
            # Add debug info to see what columns are available
            columns_str = ", ".join(filtered_level_data.columns.tolist()[:10])  # Show first 10 columns
            if len(filtered_level_data.columns) > 10:
                columns_str += f"... (and {len(filtered_level_data.columns)-10} more)"
            fig.add_annotation(
                text=f"Available columns: {columns_str}",
                xref="paper", yref="paper",
                x=0.5, y=0.1,
                showarrow=False
            )
        else:
            title = f"Cannot create color analysis - Quantity column not found (Size col: {size_col})"
            # Add debug info to see what columns are available
            columns_str = ", ".join(filtered_level_data.columns.tolist()[:10])  # Show first 10 columns
            if len(filtered_level_data.columns) > 10:
                columns_str += f"... (and {len(filtered_level_data.columns)-10} more)"
            fig.add_annotation(
                text=f"Available columns: {columns_str}",
                xref="paper", yref="paper",
                x=0.5, y=0.1,
                showarrow=False
            )
    else:
        title = f"No size data available for {item_code} colors ({analysis_level.title()})"
    
    fig.update_layout(
        title=title,
        xaxis_title="Size",
        yaxis_title="Percentage (%)",
        legend_title="Colors",
        xaxis=dict(
            type='category',  # Force categorical axis for sizes
            categoryorder='array',
            categoryarray=template_sizes  # Preserve the order of sizes
        )
    )
    
    # Set y-axis range from 0-100%
    max_y = 0
    for trace in fig.data[1:]:  # Skip the placeholder trace
        if hasattr(trace, 'y') and len(trace.y) > 0:  # Check if there are any y values
            current_max = max(trace.y)
            if current_max > max_y:
                max_y = current_max
    
    y_max = max(100, max_y * 1.1) if max_y > 0 else 100
    fig.update_yaxes(range=[0, y_max])
    
    # Hide the placeholder trace in the legend
    if len(fig.data) > 0:
        fig.data[0].showlegend = False
    
    return fig


def create_color_breakdown_chart(style_colors: List[Tuple[str, str]]) -> go.Figure:
    """
    Create a chart showing the breakdown of styles by color.
    
    Args:
        style_colors (List[Tuple[str, str]]): List of (style_code, color_code) tuples
    
    Returns:
        go.Figure: Plotly figure with color breakdown
    """
    from plotly.subplots import make_subplots
    import pandas as pd
    
    # Handle different input formats
    if isinstance(style_colors, pd.DataFrame):
        # Convert DataFrame to list of tuples
        style_colors = [
            (row['STYLE_CODE'], row['COLOR_CODE'])
            for _, row in style_colors.iterrows()
        ]
    
    # Count styles by color
    color_counts = Counter([color for _, color in style_colors])
    
    # Create sorted lists of colors and their counts
    colors = []
    counts = []
    for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
        colors.append(color)
        counts.append(count)
    
    # Create a figure with 1 row and 2 columns for bar chart and pie chart
    fig = make_subplots(
        rows=1, 
        cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=("Number of Styles by Color", "Percentage Distribution"),
        column_widths=[0.6, 0.4]
    )
    
    # Add bar chart trace
    fig.add_trace(
        go.Bar(
            x=colors,
            y=counts,
            name="Styles per Color",
            text=counts,
            textposition="auto"
        ),
        row=1, col=1
    )
    
    # Add pie chart trace
    fig.add_trace(
        go.Pie(
            labels=colors,
            values=counts,
            textinfo="percent",
            hoverinfo="label+percent+value",
            hole=0.4,
            sort=False
        ),
        row=1, col=2
    )
    
    # Improve layout
    fig.update_layout(
        title={
            'text': 'Distribution of Styles by Color',
            'y':0.97,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=450,
        margin=dict(l=50, r=50, t=80, b=100),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update bar chart x-axis
    fig.update_xaxes(
        title_text="Color Code",
        tickangle=45,
        row=1, col=1
    )
    
    # Update bar chart y-axis
    fig.update_yaxes(
        title_text="Number of Styles",
        row=1, col=1
    )
    
    return fig


def create_letter_numeric_plot(
    results_df: pd.DataFrame, 
    letter_templates: List[str], 
    numeric_templates: List[str], 
    id_map: Dict[str, Dict[str, str]], 
    level_column: str, 
    size_template_col: str
) -> go.Figure:
    """
    Create a dual plot with letter sizes and numeric sizes.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame
        letter_templates (List[str]): Item IDs with letter size templates
        numeric_templates (List[str]): Item IDs with numeric size templates
        id_map (Dict[str, Dict[str, str]]): Map of item IDs to details
        level_column (str): Column name for the level
        size_template_col (str): Column name for size template
        
    Returns:
        go.Figure: Plotly figure
    """
    from plotly.subplots import make_subplots
    
    # Identify gender column if it exists
    gender_col = None
    for possible_col in ['GENDER_CODE', 'gender_code', 'Gender']:
        if possible_col in results_df.columns:
            gender_col = possible_col
            break
    
    # Create two subplots
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=("Letter Sizes", "Numeric Sizes"),
        shared_yaxes=True
    )
    
    # Process letter size templates
    for item_id in letter_templates:
        if item_id not in id_map:
            continue
            
        item_code = id_map[item_id]['item_code']
        template = id_map[item_id]['template']
        gender_code = id_map[item_id].get('gender_code')
        
        # Create a display name that includes gender if available
        display_name = item_code
        if gender_code:
            display_name = f"{item_code} ({gender_code})"
        
        # Get data for this item
        item_data = None
        if size_template_col is not None:
            # Create the filter mask
            mask = (results_df[level_column] == item_code) & (results_df[size_template_col] == template)
            
            # Add gender filter if both gender code and gender column exist
            if gender_code and gender_col:
                mask = mask & (results_df[gender_col] == gender_code)
                
            item_data = results_df[mask]
        
        if item_data is None or item_data.empty:
            continue
        
        # Get template sizes in order
        template_sizes = template.split('-')
        
        # Extract percentages for each size
        x_values = []
        y_values = []
        
        for size in template_sizes:
            if size in item_data.columns:
                x_values.append(size)
                y_values.append(item_data[size].iloc[0])
        
        # Add line to left subplot (letter sizes)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=display_name
            ),
            row=1, col=1
        )
    
    # Process numeric size templates
    for item_id in numeric_templates:
        if item_id not in id_map:
            continue
            
        item_code = id_map[item_id]['item_code']
        template = id_map[item_id]['template']
        gender_code = id_map[item_id].get('gender_code')
        
        # Create a display name that includes gender if available
        display_name = item_code
        if gender_code:
            display_name = f"{item_code} ({gender_code})"
        
        # Get data for this item
        item_data = None
        if size_template_col is not None:
            # Create the filter mask
            mask = (results_df[level_column] == item_code) & (results_df[size_template_col] == template)
            
            # Add gender filter if both gender code and gender column exist
            if gender_code and gender_col:
                mask = mask & (results_df[gender_col] == gender_code)
                
            item_data = results_df[mask]
        
        if item_data is None or item_data.empty:
            continue
        
        # Get template sizes in order
        template_sizes = template.split('-')
        
        # Extract percentages for each size
        x_values = []
        y_values = []
        
        for size in template_sizes:
            if size in item_data.columns:
                x_values.append(size)
                y_values.append(item_data[size].iloc[0])
        
        # Add line to right subplot (numeric sizes)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=display_name
            ),
            row=1, col=2
        )
    
    # Set layout
    fig.update_layout(
        title="Size Curve Comparison (Mixed Size Types)",
        yaxis_title="Percentage (%)",
        legend_title="Items"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Size", row=1, col=1)
    fig.update_xaxes(title_text="Size", row=1, col=2)
    
    # Set y-axis range from 0-100%
    max_y = 0
    for trace in fig.data:
        if len(trace.y) > 0:  # Check if there are any y values
            current_max = max(trace.y)
            if current_max > max_y:
                max_y = current_max
    
    y_max = max(100, max_y * 1.1)
    fig.update_yaxes(range=[0, y_max])
    
    return fig