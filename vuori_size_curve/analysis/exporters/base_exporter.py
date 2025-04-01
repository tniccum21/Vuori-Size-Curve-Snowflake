"""
Base exporter interface for exporting size curve data.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
from vuori_size_curve.data.models.sales import SizeCurve


class BaseExporter(ABC):
    """
    Abstract base class for exporters that handle exporting size curve data.
    """
    
    @abstractmethod
    def export(
        self,
        size_curves: Dict[str, Dict[str, SizeCurve]],
        output_dir: str,
        channel: Optional[str] = None
    ) -> str:
        """
        Export size curves to a specified format.
        
        Args:
            size_curves (Dict[str, Dict[str, SizeCurve]]): Size curves organized by level and identifier
            output_dir (str): Base directory for output files
            channel (Optional[str]): Name of the sales channel for this data
        
        Returns:
            str: Path to the exported data
        """
        pass
    
    def prepare_dataframe(self, size_curves: Dict[str, SizeCurve], level: str) -> pd.DataFrame:
        """
        Prepare a DataFrame for a specific level of size curves.
        
        Args:
            size_curves (Dict[str, SizeCurve]): Size curves for a specific level
            level (str): The analysis level ('style', 'class', 'subclass', 'collection')
        
        Returns:
            pd.DataFrame: DataFrame containing the size curve data
        """
        if not size_curves:
            return pd.DataFrame()
        
        # Prepare data for DataFrame
        data = []
        
        for _, curve in size_curves.items():
            # Get the sizes from the template
            template_sizes = curve.size_template_name.split('-')
            
            row = {
                'item_code': curve.item_code,
                'gender_code': curve.gender_code,
                'size_template': curve.size_template_name,
                'level_type': curve.level_type
            }
            
            if curve.channel:
                row['channel'] = curve.channel
            
            # Add size percentages
            for size in template_sizes:
                row[size] = round(curve.size_percentages.get(size, 0.0), 2)
            
            data.append(row)
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Rename the item_code column according to level
        column_name_map = {
            'style': 'STYLE_CODE',
            'subclass': 'PRODUCT_SUB_CLASS_TEXT',
            'class': 'PRODUCT_CLASS_TEXT',
            'collection': 'COLLECTION_TEXT'
        }
        if 'item_code' in df.columns and level in column_name_map:
            df = df.rename(columns={'item_code': column_name_map.get(level)})
        
        return df