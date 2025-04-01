"""
Simple utility to find which styles contribute to a specific size curve.
"""
import pandas as pd

def find_contributing_styles(
    filtered_data: pd.DataFrame, 
    results_df: pd.DataFrame,
    item_code: str, 
    template: str, 
    analysis_level: str,
    gender_code: str = None
):
    """
    Find styles that contribute to a specific size curve.
    
    Args:
        filtered_data: Raw sales data with STYLE_CODE, COLOR_CODE, etc.
        results_df: Processed results data with size curves
        item_code: The category/class/subclass code
        template: The size template (e.g., 'XXS-XS-S-M-L-XL-XXL')
        analysis_level: The level of analysis (style, class, subclass, collection)
        gender_code: Optional gender code to filter by (e.g., 'M', 'W')
        
    Returns:
        List of style codes that contribute to this curve
    """
    # For style level, the style is the item itself
    if analysis_level == 'style':
        # If filtered_data is available, get color information for this style
        if 'COLOR_CODE' in filtered_data.columns:
            # Filter to just this style
            style_mask = filtered_data['STYLE_CODE'] == item_code
            
            # Add template filter if provided
            if template and 'TEMPLATE' in filtered_data.columns:
                style_mask = style_mask & (filtered_data['TEMPLATE'] == template)
                
            # Add gender filter if provided
            if gender_code and 'GENDER_CODE' in filtered_data.columns:
                style_mask = style_mask & (filtered_data['GENDER_CODE'] == gender_code)
            
            # Get matching records
            style_matches = filtered_data[style_mask]
            
            if len(style_matches) > 0:
                # Get unique color combinations using a faster vectorized approach
                style_colors = style_matches[['STYLE_CODE', 'COLOR_CODE']].drop_duplicates()
                # Convert directly to a list of tuples without iterating rows
                style_color_tuples = list(zip(style_colors['STYLE_CODE'], style_colors['COLOR_CODE']))
                style_color_tuples.sort()
                
                return ([item_code], style_color_tuples)
        
        # Default return if no color data
        return [item_code]
    
    # Map analysis level to column name
    level_to_col = {
        'style': 'STYLE_CODE',
        'class': 'PRODUCT_CLASS_TEXT',
        'subclass': 'PRODUCT_SUB_CLASS_TEXT',
        'collection': 'COLLECTION_TEXT'
    }
    level_column = level_to_col.get(analysis_level, 'STYLE_CODE')
    
    # The most authoritative way to find contributing styles is to use the TEMPLATE column
    # which was assigned based on the product repository (VW_PRODUCT)
    if 'TEMPLATE' in filtered_data.columns:
        # Find matching records using the exact template assigned from product data
        mask = filtered_data['TEMPLATE'] == template
        
        # Add level filter
        if level_column in filtered_data.columns:
            mask = mask & (filtered_data[level_column] == item_code)
        
        # Add gender filter if provided
        if gender_code and 'GENDER_CODE' in filtered_data.columns:
            mask = mask & (filtered_data['GENDER_CODE'] == gender_code)
        
        # Get matching records - these are the exact style+colors that have this template
        matches = filtered_data[mask]
        
        # Extract unique style codes
        if len(matches) > 0:
            styles = matches['STYLE_CODE'].unique().tolist()
            styles.sort()
            
            # If we have color codes, return style+color tuples too
            if 'COLOR_CODE' in matches.columns:
                style_colors = matches[['STYLE_CODE', 'COLOR_CODE']].drop_duplicates()
                # Use zip for faster tuple creation
                style_color_tuples = list(zip(style_colors['STYLE_CODE'], style_colors['COLOR_CODE']))
                style_color_tuples.sort()
                
                return (styles, style_color_tuples)
            
            return styles
    
    # Fallback approach if the TEMPLATE column isn't available
    # This is a pure fallback and shouldn't be needed normally
    if len(results_df) > 0:
        # Determine column names in results_df
        template_col = None
        for col in ['size_template', 'SIZE_TEMPLATE', 'template', 'TEMPLATE']:
            if col in results_df.columns:
                template_col = col
                break
        
        level_col = None
        level_mapping = {
            'style': ['style_code', 'STYLE_CODE'],
            'class': ['product_class_text', 'PRODUCT_CLASS_TEXT', 'class', 'CLASS'],
            'subclass': ['product_sub_class_text', 'PRODUCT_SUB_CLASS_TEXT', 'subclass', 'SUBCLASS'],
            'collection': ['collection_text', 'COLLECTION_TEXT', 'collection', 'COLLECTION']
        }
        
        for possible_col in level_mapping.get(analysis_level, []):
            if possible_col in results_df.columns:
                level_col = possible_col
                break
        
        gender_col = None
        for col in ['gender_code', 'GENDER_CODE']:
            if col in results_df.columns:
                gender_col = col
                break
        
        # If we found the necessary columns, filter the results
        if template_col and level_col:
            # Find the specific row in results_df that we're looking at
            mask = (results_df[template_col] == template) & (results_df[level_col] == item_code)
            
            if gender_code and gender_col:
                mask = mask & (results_df[gender_col] == gender_code)
            
            result_matches = results_df[mask]
            
            if not result_matches.empty:
                # Get styles that match this level, gender, and were used in the analysis
                level_gender_mask = filtered_data[level_column] == item_code
                if gender_code and 'GENDER_CODE' in filtered_data.columns:
                    level_gender_mask = level_gender_mask & (filtered_data['GENDER_CODE'] == gender_code)
                
                matches = filtered_data[level_gender_mask]
                if len(matches) > 0:
                    styles = matches['STYLE_CODE'].unique().tolist()
                    styles.sort()
                    
                    # If we have color codes, return style+color tuples too
                    if 'COLOR_CODE' in matches.columns:
                        style_colors = matches[['STYLE_CODE', 'COLOR_CODE']].drop_duplicates()
                        # Use zip for faster tuple creation
                        style_color_tuples = list(zip(style_colors['STYLE_CODE'], style_colors['COLOR_CODE']))
                        style_color_tuples.sort()
                        
                        return (styles, style_color_tuples)
                    
                    return styles
    
    # If we get here, we couldn't find any styles
    return []