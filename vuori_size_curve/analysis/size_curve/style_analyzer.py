"""
Style-level size curve analyzer.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
from vuori_size_curve.analysis.base_analyzer import BaseAnalyzer
from vuori_size_curve.data.models.product import StyleTemplate
from vuori_size_curve.data.models.sales import SizeCurve


class StyleAnalyzer(BaseAnalyzer):
    """
    Analyzer for style-level size curves.
    """
    
    def analyze(
        self,
        sales_data: pd.DataFrame,
        style_templates: Dict[str, StyleTemplate],
        **kwargs
    ) -> Dict[str, SizeCurve]:
        """
        Analyze sales data to generate style-level size curves.
        
        Args:
            sales_data (pd.DataFrame): The sales data to analyze
            style_templates (Dict[str, StyleTemplate]): Style templates to use
            **kwargs: Additional arguments, including:
                channel (Optional[str]): Distribution channel to filter by
        
        Returns:
            Dict[str, SizeCurve]: Dictionary mapping style identifiers to size curves
        """
        result = {}
        
        # Filter out invalid size data
        filtered_sales_data = sales_data[
            (~sales_data['SIZE_CODE'].str.contains('_N/A', na=False)) & 
            (sales_data['SIZE_ORDER'] > 0) & 
            (sales_data['SIZE_ORDER'].notnull())
        ].copy()
        
        # Apply channel filter if provided
        channel = kwargs.get('channel')
        if channel and channel != 'ALL' and 'DISTRIBUTION_CHANNEL_CODE' in filtered_sales_data.columns:
            filtered_sales_data = filtered_sales_data[
                filtered_sales_data['DISTRIBUTION_CHANNEL_CODE'] == channel
            ]
        
        # Group by style code and gender
        for (style_code, gender_code), style_group in filtered_sales_data.groupby(['STYLE_CODE', 'GENDER_CODE']):
            # Skip groups with insufficient sales
            if style_group['TOTAL_SALES'].sum() < self.min_sales:
                continue
            
            # Determine the template to use
            template_sizes = []
            template_name = ""
            
            # If this style is in our templates, use its template
            if style_code in style_templates:
                template = style_templates[style_code].size_template
                template_sizes = template.sizes
                template_name = template.name
            else:
                # Otherwise, determine sizes from this group and sort by SIZE_ORDER
                sizes_with_order = style_group[['SIZE_CODE', 'SIZE_ORDER']].drop_duplicates()
                # Filter out any problematic SIZE_ORDER values
                sizes_with_order = sizes_with_order[
                    (sizes_with_order['SIZE_ORDER'] > 0) & 
                    (sizes_with_order['SIZE_ORDER'].notnull())
                ]
                # Ensure we're getting unique sizes only
                sizes_with_order = sizes_with_order.drop_duplicates('SIZE_CODE')
                if sizes_with_order.empty:
                    continue  # Skip if no valid sizes
                
                sizes_sorted = sizes_with_order.sort_values('SIZE_ORDER')
                template_sizes = sizes_sorted['SIZE_CODE'].tolist()
                template_name = "-".join(template_sizes)
            
            # Skip if no valid template sizes
            if not template_sizes:
                continue
                
            # Calculate total sales for this style and gender
            total_style_sales = style_group['TOTAL_SALES'].sum()
            
            if total_style_sales > 0:
                # Calculate size percentages
                size_percentages = {}
                
                # Group by size and calculate percentage
                for size_code, size_group in style_group.groupby('SIZE_CODE'):
                    size_sales = size_group['TOTAL_SALES'].sum()
                    size_percentages[size_code] = (size_sales / total_style_sales) * 100
                
                # Fill in zeros for missing sizes
                for size in template_sizes:
                    if size not in size_percentages:
                        size_percentages[size] = 0.0
                
                # Create a composite key and size curve object
                composite_key = f"{style_code}|{gender_code}"
                
                # Create a SizeCurve object
                size_curve = SizeCurve(
                    item_code=style_code,
                    gender_code=gender_code,
                    size_template_name=template_name,
                    level_type="STYLE",
                    channel=channel,
                    size_percentages=size_percentages
                )
                
                result[composite_key] = size_curve
        
        print(f"Generated {len(result)} style-level size curves.")
        return result