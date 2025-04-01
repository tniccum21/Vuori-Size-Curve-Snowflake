"""
Subclass-level size curve analyzer.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
from vuori_size_curve.analysis.base_analyzer import BaseAnalyzer
from vuori_size_curve.data.models.product import StyleTemplate
from vuori_size_curve.data.models.sales import SizeCurve
from vuori_size_curve.config.app_config import LEVEL_COLUMN_MAP


class SubclassAnalyzer(BaseAnalyzer):
    """
    Analyzer for subclass-level size curves.
    """
    
    def analyze(
        self,
        sales_data: pd.DataFrame,
        style_templates: Dict[str, StyleTemplate],
        **kwargs
    ) -> Dict[str, SizeCurve]:
        """
        Analyze sales data to generate subclass-level size curves.
        
        Args:
            sales_data (pd.DataFrame): The sales data to analyze
            style_templates (Dict[str, StyleTemplate]): Style templates to use
            **kwargs: Additional arguments, including:
                channel (Optional[str]): Distribution channel to filter by
        
        Returns:
            Dict[str, SizeCurve]: Dictionary mapping subclass identifiers to size curves
        """
        result = {}
        
        # Get the subclass column name
        level_column = LEVEL_COLUMN_MAP.get('subclass')
        if level_column not in sales_data.columns:
            print(f"Column {level_column} not found in sales data. Available columns: {sales_data.columns.tolist()}")
            # Try to find alternative columns that might match
            subclass_candidates = [col for col in sales_data.columns if 'sub' in col.lower() and 'class' in col.lower()]
            if subclass_candidates:
                level_column = subclass_candidates[0]
                print(f"Using alternative column for subclass: {level_column}")
            else:
                return result
        
        # Filter out invalid size data and _N/A subclasses
        filtered_sales_data = sales_data[
            (~sales_data['SIZE_CODE'].str.contains('_N/A', na=False)) & 
            (sales_data['SIZE_ORDER'] > 0) & 
            (sales_data['SIZE_ORDER'].notnull()) &
            (~sales_data[level_column].str.contains('_N/A', na=False))
        ].copy()
        
        # Apply channel filter if provided
        channel = kwargs.get('channel')
        if channel and channel != 'ALL' and 'DISTRIBUTION_CHANNEL_CODE' in filtered_sales_data.columns:
            filtered_sales_data = filtered_sales_data[
                filtered_sales_data['DISTRIBUTION_CHANNEL_CODE'] == channel
            ]
        
        # Assign templates based on PRODUCT definitions from the repository
        filtered_sales_data['TEMPLATE'] = ''
        
        # Get templates from product repository
        product_repo = kwargs.get('product_repository')
        if product_repo:
            print("Using product data for template definitions (not sales data)")
            
            # Ensure we have template data
            if not hasattr(product_repo, 'style_color_templates') or not product_repo.style_color_templates:
                product_repo.determine_size_templates()
                
            # Check if we have style+color templates available
            if hasattr(product_repo, 'style_color_templates') and product_repo.style_color_templates:
                # Create a temporary column with style+color key
                if 'COLOR_CODE' in filtered_sales_data.columns:
                    filtered_sales_data['STYLE_COLOR_KEY'] = filtered_sales_data['STYLE_CODE'] + '_' + filtered_sales_data['COLOR_CODE']
                    
                    # Create a mapping dictionary for faster template assignment
                    template_mapping = {
                        key: obj.size_template.name 
                        for key, obj in product_repo.style_color_templates.items()
                    }
                    
                    # Use map function instead of iterating through each key (much faster)
                    mapped_templates = filtered_sales_data['STYLE_COLOR_KEY'].map(template_mapping)
                    # Only update where we have matches (keep '' where there's no match)
                    filtered_sales_data.loc[mapped_templates.notna(), 'TEMPLATE'] = mapped_templates[mapped_templates.notna()]
                    
                    # Drop the temporary key column
                    filtered_sales_data.drop(columns=['STYLE_COLOR_KEY'], inplace=True)
            
            # Fall back to style-only templates with faster vectorized approach
            # Create a mapping dictionary for style templates
            style_template_mapping = {
                style: obj.size_template.name 
                for style, obj in style_templates.items()
            }
            
            # Only update rows with empty templates
            empty_template_mask = filtered_sales_data['TEMPLATE'] == ''
            if empty_template_mask.any():
                # Apply mapping only to rows with empty templates
                empty_template_styles = filtered_sales_data.loc[empty_template_mask, 'STYLE_CODE']
                mapped_templates = empty_template_styles.map(style_template_mapping)
                # Only update cells where we have a match
                filtered_data_update_mask = empty_template_mask & mapped_templates.notna()
                if filtered_data_update_mask.any():
                    filtered_sales_data.loc[filtered_data_update_mask, 'TEMPLATE'] = mapped_templates[mapped_templates.notna()]
        else:
            # Fallback to style templates if no product_repo available - use vectorized operations
            style_template_mapping = {
                style: template.size_template.name 
                for style, template in style_templates.items()
            }
            # Use map for faster assignment
            filtered_sales_data['TEMPLATE'] = filtered_sales_data['STYLE_CODE'].map(style_template_mapping).fillna('')
        
        # Group by subclass and gender
        for (subclass_value, gender_code), subclass_group in filtered_sales_data.groupby([level_column, 'GENDER_CODE']):
            # Skip groups with insufficient sales
            if subclass_group['TOTAL_SALES'].sum() < self.min_sales:
                continue
            
            # Further group by template from the product data
            for template_name, template_group in subclass_group.groupby('TEMPLATE'):
                if not template_name:  # Skip empty templates
                    continue
                
                template_sizes = template_name.split('-')
                if not template_sizes:  # Skip empty template lists
                    continue
                
                # Calculate total sales for this subclass, gender, and template
                total_template_sales = template_group['TOTAL_SALES'].sum()
                
                if total_template_sales > 0:
                    # Calculate size percentages using vectorized operations
                    # Use Series aggregation instead of iterating through groups
                    size_sales = template_group.groupby('SIZE_CODE')['TOTAL_SALES'].sum()
                    # Calculate percentages in one vectorized operation
                    size_percentages_series = (size_sales / total_template_sales) * 100
                    
                    # Convert to dictionary for the API
                    size_percentages = size_percentages_series.to_dict()
                    
                    # Fill in zeros for missing sizes in this template using dict comprehension
                    size_percentages = {
                        size: size_percentages.get(size, 0.0) 
                        for size in template_sizes
                    }
                    
                    # Create a composite key for the result
                    composite_key = f"{subclass_value}|{gender_code}|{template_name}"
                    
                    # Create a SizeCurve object
                    size_curve = SizeCurve(
                        item_code=subclass_value,
                        gender_code=gender_code,
                        size_template_name=template_name,
                        level_type="SUBCLASS",
                        channel=channel,
                        size_percentages=size_percentages
                    )
                    
                    result[composite_key] = size_curve
        
        print(f"Generated {len(result)} subclass-level size curves.")
        return result