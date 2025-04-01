"""
CSV exporter for size curve data.
"""
from typing import Dict, List, Optional, Any
import os
import pandas as pd
from vuori_size_curve.analysis.exporters.base_exporter import BaseExporter
from vuori_size_curve.data.models.sales import SizeCurve
from vuori_size_curve.data.models.product import StyleTemplate


class CSVExporter(BaseExporter):
    """
    Exporter for size curve data to CSV files.
    """
    
    def export(
        self,
        size_curves: Dict[str, Dict[str, SizeCurve]],
        output_dir: str,
        channel: Optional[str] = None
    ) -> str:
        """
        Export size curves to CSV files.
        
        Args:
            size_curves (Dict[str, Dict[str, SizeCurve]]): Size curves organized by level and identifier
            output_dir (str): Base directory for output files
            channel (Optional[str]): Name of the sales channel for this data
        
        Returns:
            str: Path to the exported CSV files
        """
        # Create a subdirectory for this channel
        channel_name = channel if channel else "all_channels"
        channel_dir = os.path.join(output_dir, channel_name)
        os.makedirs(channel_dir, exist_ok=True)
        
        # Create a dictionary to store all dataframes
        all_dfs = {}
        
        # Prepare dataframes for each level
        for level, level_curves in size_curves.items():
            df = self.prepare_dataframe(level_curves, level)
            if not df.empty:
                output_path = os.path.join(channel_dir, f"{level}_size_curves.csv")
                df.to_csv(output_path, index=False)
                print(f"Exported {channel_name} {level} size curves to {output_path}")
                all_dfs[level] = df
        
        # Create a combined CSV with a level indicator column
        self._export_combined_csv(all_dfs, channel_dir, channel_name)
        
        # Export template summary
        if 'style' in size_curves:
            self.export_template_summary(size_curves['style'], channel_dir)
        
        return channel_dir
    
    def _export_combined_csv(
        self,
        all_dfs: Dict[str, pd.DataFrame],
        output_dir: str,
        channel_name: str
    ) -> None:
        """
        Export a combined CSV file with all levels.
        
        Args:
            all_dfs (Dict[str, pd.DataFrame]): Dictionary of dataframes by level
            output_dir (str): Output directory
            channel_name (str): Channel name
        """
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
                    'COLLECTION_TEXT': 'ITEM_CODE'
                }
                for old_col, new_col in level_col_map.items():
                    if old_col in df_copy.columns:
                        df_copy = df_copy.rename(columns={old_col: new_col})
                
                combined_data.append(df_copy)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_output_path = os.path.join(output_dir, "combined_size_curves.csv")
            combined_df.to_csv(combined_output_path, index=False)
            print(f"Exported {channel_name} combined size curves to {combined_output_path}")
    
    def export_template_summary(
        self,
        style_curves: Dict[str, SizeCurve],
        output_dir: str
    ) -> pd.DataFrame:
        """
        Export a summary of unique size templates and their usage counts.
        
        Args:
            style_curves (Dict[str, SizeCurve]): Style-level size curves
            output_dir (str): Output directory
        
        Returns:
            pd.DataFrame: Template summary dataframe
        """
        # Count styles per template
        template_counts = {}
        template_styles = {}
        
        for key, curve in style_curves.items():
            # Extract style code from composite key if needed
            style_code = curve.item_code
            template_name = curve.size_template_name
            
            if template_name in template_counts:
                template_counts[template_name] += 1
                template_styles[template_name].append(style_code)
            else:
                template_counts[template_name] = 1
                template_styles[template_name] = [style_code]
        
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