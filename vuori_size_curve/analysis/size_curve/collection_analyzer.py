"""
Collection-level size curve analyzer.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
from vuori_size_curve.analysis.base_analyzer import BaseAnalyzer
from vuori_size_curve.data.models.product import StyleTemplate
from vuori_size_curve.data.models.sales import SizeCurve
from vuori_size_curve.config.app_config import LEVEL_COLUMN_MAP


class CollectionAnalyzer(BaseAnalyzer):
    """
    Analyzer for collection-level size curves.
    """
    
    def analyze(
        self,
        sales_data: pd.DataFrame,
        style_templates: Dict[str, StyleTemplate],
        **kwargs
    ) -> Dict[str, SizeCurve]:
        """
        Analyze sales data to generate collection-level size curves.
        
        Args:
            sales_data (pd.DataFrame): The sales data to analyze
            style_templates (Dict[str, StyleTemplate]): Style templates to use
            **kwargs: Additional arguments, including:
                channel (Optional[str]): Distribution channel to filter by
                all_collection_templates (Dict[str, Set[str]]): All templates that should appear for each collection
        
        Returns:
            Dict[str, SizeCurve]: Dictionary mapping collection identifiers to size curves
        """
        result = {}
        
        # Get the collection column name
        level_column = LEVEL_COLUMN_MAP.get('collection')
        
        # If collection column doesn't exist, try to find alternatives
        if level_column not in sales_data.columns:
            print(f"Column {level_column} not found in sales data. Available columns: {sales_data.columns.tolist()}")
            # Try to find alternative columns that might match
            collection_candidates = [col for col in sales_data.columns if 'collection' in col.lower()]
            
            if collection_candidates:
                level_column = collection_candidates[0]
                print(f"Using alternative column for collection: {level_column}")
            elif 'GENDER_CODE' in sales_data.columns:
                print(f"No collection column found, falling back to GENDER_CODE.")
                level_column = 'GENDER_CODE'
            else:
                print(f"No suitable collection or gender column found.")
                return result
        
        # Filter out invalid size data and _N/A collections
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
                    filtered_sales_data['TEMPLATE'] = filtered_sales_data['STYLE_COLOR_KEY'].map(template_mapping).fillna('')
                    
                    # Drop the temporary key column
                    filtered_sales_data.drop(columns=['STYLE_COLOR_KEY'], inplace=True)
            
            # Fall back to style-only templates using a faster vectorized approach
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
                filtered_sales_data.loc[empty_template_mask, 'TEMPLATE'] = empty_template_styles.map(style_template_mapping).fillna('')
        
        # Check if we have all_collection_templates from kwargs
        all_collection_templates = kwargs.get('all_collection_templates', {})
        
        # First process collections based on sales data
        processed_templates = set()  # Keep track of which templates we've processed
        
        # Group by collection and gender
        for (collection_value, gender_code), collection_group in filtered_sales_data.groupby([level_column, 'GENDER_CODE']):
            # Skip groups with insufficient sales
            if collection_group['TOTAL_SALES'].sum() < self.min_sales:
                continue
            
            # Further group by template from the product data
            for template_name, template_group in collection_group.groupby('TEMPLATE'):
                if not template_name:  # Skip empty templates
                    continue
                
                template_sizes = template_name.split('-')
                if not template_sizes:  # Skip empty template lists
                    continue
                
                # Calculate total sales for this collection, gender, and template
                total_template_sales = template_group['TOTAL_SALES'].sum()
                
                if total_template_sales > 0:
                    # Calculate size percentages
                    size_percentages = {}
                    
                    # Group by size and calculate percentage
                    for size_code, size_group in template_group.groupby('SIZE_CODE'):
                        size_sales = size_group['TOTAL_SALES'].sum()
                        size_percentages[size_code] = (size_sales / total_template_sales) * 100
                    
                    # Fill in zeros for missing sizes in this template
                    for size in template_sizes:
                        if size not in size_percentages:
                            size_percentages[size] = 0.0
                    
                    # Create a composite key for the result
                    composite_key = f"{collection_value}|{gender_code}|{template_name}"
                    
                    # Create a SizeCurve object
                    size_curve = SizeCurve(
                        item_code=collection_value,
                        gender_code=gender_code,
                        size_template_name=template_name,
                        level_type="COLLECTION",
                        channel=channel,
                        size_percentages=size_percentages
                    )
                    
                    result[composite_key] = size_curve
                    
                    # Mark this template as processed
                    processed_templates.add((collection_value, gender_code, template_name))
        
        # Now add any additional templates from all_collection_templates
        if all_collection_templates:
            
            # Get the product repository to check valid collection+gender combinations
            product_repo = kwargs.get('product_repository')
            
            # Get valid collection to gender mappings from product data using vectorized operations
            valid_collection_genders = {}
            if product_repo and level_column in filtered_sales_data.columns and 'GENDER_CODE' in filtered_sales_data.columns:
                # Filter to rows with valid templates
                valid_templates = filtered_sales_data[filtered_sales_data['TEMPLATE'] != '']
                
                # Group by collection and get unique genders
                if not valid_templates.empty:
                    collection_genders = valid_templates.groupby(level_column)['GENDER_CODE'].unique()
                    for collection, genders in collection_genders.items():
                        if collection:
                            valid_collection_genders[collection] = set(genders)
            
            # For each collection, add any templates we haven't processed yet
            for collection, templates in all_collection_templates.items():
                # Get valid genders for this collection
                valid_genders = []
                if collection in valid_collection_genders:
                    valid_genders = list(valid_collection_genders[collection])
                
                # If no valid genders found, skip this collection
                if not valid_genders:
                    continue
                
                for template_name in templates:
                    for gender_code in valid_genders:
                        # Skip if we've already processed this combination
                        if (collection, gender_code, template_name) in processed_templates:
                            continue
                        
                        # Create an empty size curve for this template
                        template_sizes = template_name.split('-')
                        size_percentages = {size: 0.0 for size in template_sizes}
                        
                        # Create a composite key for the result
                        composite_key = f"{collection}|{gender_code}|{template_name}"
                        
                        # Create a SizeCurve object (with zero percentages)
                        size_curve = SizeCurve(
                            item_code=collection,
                            gender_code=gender_code,
                            size_template_name=template_name,
                            level_type="COLLECTION",
                            channel=channel,
                            size_percentages=size_percentages
                        )
                        
                        result[composite_key] = size_curve
        
        print(f"Generated {len(result)} collection-level size curves.")
        return result