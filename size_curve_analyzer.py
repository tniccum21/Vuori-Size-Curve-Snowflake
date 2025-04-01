import pandas as pd
import numpy as np
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, sum as sum_
import os
from datetime import datetime
from snowflake_config import SNOWFLAKE_CONFIG

class SizeCurveAnalyzer:
    def __init__(self):
        """Initialize the Size Curve Analyzer with Snowflake connection"""
        self.session = self._connect_to_snowflake()
        self.sales_data = None
        self.start_date = '2022-01-01'
        self.channels_filter = None
        self.min_sales = 0
        self.output_dir = None  # Will be set by CLI or defaulted in run_analysis
        
    def _connect_to_snowflake(self):
        """Connect to Snowflake database"""
        # Create new session using config
        session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
        print("Snowflake connection established.")
        return session
        
    def __del__(self):
        """Destructor to ensure Snowflake connection is closed properly"""
        try:
            if hasattr(self, 'session') and self.session:
                self.session.close()
                print("Snowflake connection closed.")
        except Exception as e:
            print(f"Error closing Snowflake connection: {str(e)}")
        
    def fetch_sales_data(self, custom_start_date=None):
        """Fetch sales data from Snowflake with optional custom start date"""
        # Use the provided custom start date or fall back to the default
        start_date_to_use = custom_start_date if custom_start_date else self.start_date
        
        query = f"""
        SELECT 
            p.STYLE_CODE, 
            p.GENDER_CODE, 
            p.COLOR_CODE, 
            p.COLLECTION_TEXT,
            c.NRF_COLOR_TEXT, 
            p.SIZE_CODE, 
            p.SIZE_ORDER, 
            p.PRODUCT_CLASS_TEXT, 
            p.PRODUCT_SUB_CLASS_TEXT,
            s.DISTRIBUTION_CHANNEL_CODE,
            SUM(s.SALES_QUANTITY) AS TOTAL_SALES
        FROM 
            PROD_DL.CURATED.VW_SALES_TRANSACTION_DAILY s
        JOIN 
            PROD_DL.CURATED.VW_PRODUCT p ON s.PRODUCT_KEY = p.PRODUCT_KEY
        JOIN 
            PROD_DL.CURATED.VW_COLOR c ON p.COLOR_CODE = c.COLOR_CODE
        WHERE 
            s.ORDER_DATE >= '{start_date_to_use}'
            AND p.SIZE_CODE IS NOT NULL
            AND p.SIZE_CODE NOT LIKE '%_N/A%'
            AND p.SIZE_ORDER IS NOT NULL
        GROUP BY 
            p.STYLE_CODE, 
            p.GENDER_CODE, 
            p.COLOR_CODE, 
            p.COLLECTION_TEXT,
            c.NRF_COLOR_TEXT, 
            p.SIZE_CODE, 
            p.SIZE_ORDER, 
            p.PRODUCT_CLASS_TEXT, 
            p.PRODUCT_SUB_CLASS_TEXT,
            s.DISTRIBUTION_CHANNEL_CODE
        """
        
        print(f"Fetching sales data from {start_date_to_use}...")
        sales_data = self.session.sql(query).to_pandas()
        print(f"Retrieved {len(sales_data)} sales records.")
        return sales_data
        
    def fetch_product_data(self):
        """Fetch product data to determine size templates"""
        query = """
        SELECT 
            p.STYLE_CODE,
            p.COLOR_CODE,
            p.SIZE_CODE,
            p.SIZE_ORDER
        FROM 
            PROD_DL.CURATED.VW_PRODUCT p
        WHERE
            p.SIZE_CODE IS NOT NULL
            AND p.SIZE_CODE NOT LIKE '%_N/A%'
            AND p.SIZE_ORDER IS NOT NULL
        """
        
        print("Fetching product data for size template determination...")
        product_data = self.session.sql(query).to_pandas()
        print(f"Retrieved {len(product_data)} product records.")
        return product_data
    
    def determine_size_templates(self, product_data):
        """Determine size templates for each style"""
        # Filter out any _N/A sizes first and any with SIZE_ORDER=0 or NULL
        filtered_product_data = product_data[
            (~product_data['SIZE_CODE'].str.contains('_N/A', na=False)) & 
            (product_data['SIZE_ORDER'] > 0) & 
            (product_data['SIZE_ORDER'].notnull())
        ].copy()
        
        # Group by style to find all available sizes for each style
        style_templates = {}
        style_template_names = {}
        
        for style_code, group in filtered_product_data.groupby('STYLE_CODE'):
            # Get unique sizes for this style, sorted by SIZE_ORDER
            sizes_with_order = group[['SIZE_CODE', 'SIZE_ORDER']].drop_duplicates()
            
            # Ensure we're getting unique sizes only
            sizes_with_order = sizes_with_order.drop_duplicates('SIZE_CODE')
            sizes_sorted = sizes_with_order.sort_values('SIZE_ORDER')
            sizes = sizes_sorted['SIZE_CODE'].tolist()
            
            # Skip styles with no valid sizes after filtering
            if not sizes:
                continue
                
            # Create a template name by joining all sizes
            template_name = "-".join(sizes)
            
            style_templates[style_code] = sizes
            style_template_names[style_code] = template_name
        
        print(f"Determined size templates for {len(style_templates)} styles.")
        
        return style_templates, style_template_names
    
    def calculate_size_curves(self, sales_data, style_templates, style_template_names):
        """Calculate size curves at different levels"""
        
        
        # Check if COLLECTION_TEXT exists, if not, fall back to GENDER_CODE
        collection_column = 'COLLECTION_TEXT' if 'COLLECTION_TEXT' in sales_data.columns else 'GENDER_CODE'
        
        # 1. Style level size curves
        style_curves, style_templates_used = self._calculate_level_size_curves(
            sales_data, 'STYLE_CODE', style_templates, style_template_names)
        
        # 2. Subclass level size curves
        subclass_curves, subclass_templates_used = self._calculate_level_size_curves(
            sales_data, 'PRODUCT_SUB_CLASS_TEXT', style_templates, style_template_names)
        
        # 3. Class level size curves
        class_curves, class_templates_used = self._calculate_level_size_curves(
            sales_data, 'PRODUCT_CLASS_TEXT', style_templates, style_template_names)
        
        # 4. Collection level (using COLLECTION_TEXT if available, otherwise GENDER_CODE)
        collection_curves, collection_templates_used = self._calculate_level_size_curves(
            sales_data, collection_column, style_templates, style_template_names)
        
        return {
            'style': (style_curves, style_templates_used),
            'subclass': (subclass_curves, subclass_templates_used),
            'class': (class_curves, class_templates_used),
            'collection': (collection_curves, collection_templates_used)
        }
    
    def _calculate_level_size_curves(self, sales_data, level_column, style_templates=None, template_names=None):
        """Helper function to calculate size curves for a specific level"""
        result = {}
        templates_used = {}
        
        # Filter out any sizes with "_N/A" and ensure SIZE_ORDER is valid (> 0 and not null)
        filtered_sales_data = sales_data[
            (~sales_data['SIZE_CODE'].str.contains('_N/A', na=False)) & 
            (sales_data['SIZE_ORDER'] > 0) & 
            (sales_data['SIZE_ORDER'].notnull())
        ].copy()
        
        # For levels above style, we need to group by level and gender
        if level_column != 'STYLE_CODE':
            # Always use the style_templates from the product master if available
            if style_templates and template_names:
                # First, assign the master template to each style
                filtered_sales_data['TEMPLATE'] = ''
                for style, template_sizes in style_templates.items():
                    if style in template_names:
                        template_name = template_names[style]
                        # Assign template to all rows for this style
                        filtered_sales_data.loc[filtered_sales_data['STYLE_CODE'] == style, 'TEMPLATE'] = template_name
                
                # Now group by level, gender, and template
                for (level_value, gender_code), level_group in filtered_sales_data.groupby([level_column, 'GENDER_CODE']):
                    # Skip groups with insufficient sales
                    if level_group['TOTAL_SALES'].sum() < self.min_sales:
                        continue
                    
                    # Further group by template from the product data
                    for template, template_group in level_group.groupby('TEMPLATE'):
                        if not template:  # Skip empty templates
                            continue
                        
                        template_sizes = template.split('-')
                        if not template_sizes:  # Skip empty template lists
                            continue
                        
                        # Calculate total sales for this level, gender, and template
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
                            
                            # Create a composite key for the result that includes gender and template
                            composite_key = f"{level_value}|{gender_code}|{template}"
                            
                            result[composite_key] = size_percentages
                            templates_used[composite_key] = template
            else:
                # Skip if no templates are provided - this should be rare
                print(f"Warning: No style templates provided for {level_column} level analysis")
                return result, templates_used
        else:
            # For style level, we'll also include gender
            for (level_value, gender_code), level_group in filtered_sales_data.groupby([level_column, 'GENDER_CODE']):
                # Skip groups with insufficient sales
                if level_group['TOTAL_SALES'].sum() < self.min_sales:
                    continue
                    
                # If we're at style level and have templates, use them
                if style_templates and level_value in style_templates:
                    template_sizes = style_templates[level_value]
                    if template_names and level_value in template_names:
                        template_name = template_names[level_value]
                    else:
                        template_name = "-".join(template_sizes) if template_sizes else ""
                else:
                    # Otherwise, determine sizes from this group and sort by SIZE_ORDER
                    sizes_with_order = level_group[['SIZE_CODE', 'SIZE_ORDER']].drop_duplicates()
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
                    
                # Calculate total sales for this level and gender
                total_level_sales = level_group['TOTAL_SALES'].sum()
                
                if total_level_sales > 0:
                    # Calculate size percentages
                    size_percentages = {}
                    
                    # Group by size and calculate percentage
                    for size_code, size_group in level_group.groupby('SIZE_CODE'):
                        size_sales = size_group['TOTAL_SALES'].sum()
                        size_percentages[size_code] = (size_sales / total_level_sales) * 100
                    
                    # Fill in zeros for missing sizes
                    for size in template_sizes:
                        if size not in size_percentages:
                            size_percentages[size] = 0.0
                    
                    # Create a composite key that includes gender
                    composite_key = f"{level_value}|{gender_code}"
                    
                    result[composite_key] = size_percentages
                    templates_used[composite_key] = template_name
                
        return result, templates_used

    def prepare_output_dataframe(self, size_curves_data, level):
        """Prepare a dataframe for the specified level of size curves"""
        size_curves, templates_used = size_curves_data[level]
        
        if not size_curves:
            return pd.DataFrame()
        
        # Prepare data for dataframe
        data = []
        
        for item_key, item_curves in size_curves.items():
            # For all levels, split the composite key to extract gender
            parts = item_key.split('|')
            
            if level != 'style' and len(parts) == 3:
                # For higher levels: level_value|gender_code|template
                item_name = parts[0]
                gender_code = parts[1]
                template = parts[2]
            elif len(parts) == 2:
                # For style level: style_code|gender_code
                item_name = parts[0]
                gender_code = parts[1]
                template = templates_used.get(item_key, '')
            else:
                # Fallback for any other format
                item_name = item_key
                gender_code = 'UNKNOWN'
                template = templates_used.get(item_key, '')
            
            # Get the sizes for this item from its template
            if template:
                sorted_sizes = template.split('-')
            else:
                # Fallback if no template is found
                sorted_sizes = list(item_curves.keys())
            
            row = {
                'item': item_name,
                'gender_code': gender_code,
                'size_template': template
            }
            
            for size in sorted_sizes:
                row[size] = round(item_curves.get(size, 0.0), 2)
            
            data.append(row)
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Rename the item column to match the level
        column_name_map = {
            'style': 'STYLE_CODE',
            'subclass': 'PRODUCT_SUB_CLASS_TEXT',
            'class': 'PRODUCT_CLASS_TEXT',
            'collection': 'COLLECTION_TEXT'
        }
        df = df.rename(columns={'item': column_name_map.get(level, 'item')})
        
        return df
    
    def export_to_csv(self, size_curves, output_dir, channel_name):
        """
        Export size curves to CSV files
        
        Args:
            size_curves: Dictionary containing size curve data
            output_dir: Base directory for output files
            channel_name: Name of the sales channel for this data
        """
        # Create a subdirectory for this channel
        channel_dir = os.path.join(output_dir, channel_name)
        os.makedirs(channel_dir, exist_ok=True)
        
        # Create a dictionary to store all dataframes
        all_dfs = {}
        
        # Prepare dataframes for each level
        for level in size_curves.keys():
            df = self.prepare_output_dataframe(size_curves, level)
            if not df.empty:
                output_path = os.path.join(channel_dir, f"{level}_size_curves.csv")
                df.to_csv(output_path, index=False)
                print(f"Exported {channel_name} {level} size curves to {output_path}")
                all_dfs[level] = df
        
        # Create a combined CSV with a level indicator column
        combined_data = []
        for level, df in all_dfs.items():
            if not df.empty:
                # Add a level column
                df_copy = df.copy()
                df_copy['LEVEL_TYPE'] = level.upper()
                df_copy['CHANNEL'] = channel_name
                
                # Rename columns to standardized names
                level_col_map = {
                    'STYLE_CODE': 'ITEM_CODE',
                    'PRODUCT_SUB_CLASS_TEXT': 'ITEM_CODE',
                    'PRODUCT_CLASS_TEXT': 'ITEM_CODE',
                    'GENDER_CODE': 'ITEM_CODE'
                }
                df_copy = df_copy.rename(columns=level_col_map)
                
                combined_data.append(df_copy)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_output_path = os.path.join(channel_dir, "combined_size_curves.csv")
            combined_df.to_csv(combined_output_path, index=False)
            print(f"Exported {channel_name} combined size curves to {combined_output_path}")
        
        # Create a summary of unique size templates
        self.export_template_summary(size_curves, channel_dir)
        
        return channel_dir
    
    def export_template_summary(self, size_curves, output_dir):
        """
        Export a summary of unique size templates and their usage counts
        
        Args:
            size_curves: Dictionary containing size curve data
            output_dir: Directory for output files
        """
        # Extract all templates from the style level
        style_curves, style_templates = size_curves['style']
        
        # Count styles per template
        template_counts = {}
        template_styles = {}
        
        for item_key, template in style_templates.items():
            # For style level, the key format is "style_code|gender_code"
            style_code = item_key.split('|')[0] if '|' in item_key else item_key
            
            if template in template_counts:
                template_counts[template] += 1
                template_styles[template].append(style_code)
            else:
                template_counts[template] = 1
                template_styles[template] = [style_code]
        
        # Create a dataframe
        template_data = []
        for template, count in template_counts.items():
            # Split the template into sizes
            sizes = template.split('-') if isinstance(template, str) else []
            
            # Create a row with template and count
            row = {
                'size_template': template,
                'style_count': count,
                'size_count': len(sizes)
            }
            template_data.append(row)
        
        # Convert to dataframe and sort by style count (descending)
        template_df = pd.DataFrame(template_data)
        if not template_df.empty:
            template_df = template_df.sort_values('style_count', ascending=False)
            
            # Export to CSV
            template_output_path = os.path.join(output_dir, "template_summary.csv")
            template_df.to_csv(template_output_path, index=False)
            print(f"Exported template summary to {template_output_path}")
            
            # Export styles with unique templates to a separate txt file
            unique_template_styles = []
            for template, count in template_counts.items():
                if count == 1 and template in template_styles:
                    for style in template_styles[template]:
                        unique_template_styles.append(f"{style}: {template}")
            
            if unique_template_styles:
                unique_styles_output_path = os.path.join(output_dir, "unique_template_styles.txt")
                with open(unique_styles_output_path, 'w') as f:
                    f.write("Styles with unique size templates:\n\n")
                    f.write("\n".join(sorted(unique_template_styles)))
                print(f"Exported {len(unique_template_styles)} styles with unique templates to {unique_styles_output_path}")
        
        return template_df
    
    def run_analysis(self, custom_start_date=None):
        """Run the complete size curve analysis process"""
        # 1. Fetch data
        self.sales_data = self.fetch_sales_data(custom_start_date=custom_start_date)
        product_data = self.fetch_product_data()
        
        # 2. Determine size templates
        style_templates, style_template_names = self.determine_size_templates(product_data)
        
        # 3. Get unique sales channels
        sales_channels = self.sales_data['DISTRIBUTION_CHANNEL_CODE'].unique().tolist()
        if self.channels_filter:
            # Filter to only the requested channels
            sales_channels = [ch for ch in sales_channels if ch in self.channels_filter]
        
        print(f"Found {len(sales_channels)} distribution channels: {', '.join(sales_channels)}")
        
        # 4. Create a directory for results
        if self.output_dir:
            # Use the specified output directory
            output_dir = self.output_dir
        else:
            # Generate default timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"size_curves_{timestamp}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 5. Process all channels combined
        all_channels_curves = self.calculate_size_curves(self.sales_data, style_templates, style_template_names)
        self.export_to_csv(all_channels_curves, output_dir, "all_channels")
        
        # 6. Process each channel separately
        for channel in sales_channels:
            # Filter data for this channel
            channel_data = self.sales_data[self.sales_data['DISTRIBUTION_CHANNEL_CODE'] == channel].copy()
            
            # Calculate size curves for this channel
            channel_curves = self.calculate_size_curves(channel_data, style_templates, style_template_names)
            
            # Export results for this channel
            self.export_to_csv(channel_curves, output_dir, channel)
        
        print(f"\nSize curve analysis complete. Results saved in {output_dir}")
        return output_dir