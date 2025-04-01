"""
Base analyzer for size curve analysis.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from vuori_size_curve.data.models.product import StyleTemplate
from vuori_size_curve.data.models.sales import SizeCurve, FilterCriteria
from vuori_size_curve.data.repositories.product_repository import ProductRepository
from vuori_size_curve.data.repositories.sales_repository import SalesRepository
from vuori_size_curve.config.app_config import DEFAULT_MIN_SALES


class BaseAnalyzer(ABC):
    """
    Base class for size curve analyzers.
    """
    
    def __init__(
        self,
        product_repository: ProductRepository,
        sales_repository: SalesRepository,
        min_sales: int = DEFAULT_MIN_SALES
    ):
        """
        Initialize the base analyzer.
        
        Args:
            product_repository (ProductRepository): Repository for product data
            sales_repository (SalesRepository): Repository for sales data
            min_sales (int): Minimum sales quantity threshold
        """
        self.product_repository = product_repository
        self.sales_repository = sales_repository
        self.min_sales = min_sales
        self.style_templates = {}
    
    def prepare_data(self, filter_criteria: Optional[FilterCriteria] = None) -> Tuple[pd.DataFrame, Dict[str, StyleTemplate]]:
        """
        Prepare data for analysis.
        
        Args:
            filter_criteria (Optional[FilterCriteria]): Filtering criteria
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, StyleTemplate]]: Filtered sales data and style templates
        """
        # Get style templates
        if not self.style_templates:
            self.style_templates = self.product_repository.determine_size_templates()
        
        # Get filtered sales data
        sales_data = self.sales_repository.get_raw_data(filter_criteria)
        
        return sales_data, self.style_templates
    
    @abstractmethod
    def analyze(
        self,
        sales_data: pd.DataFrame,
        style_templates: Dict[str, StyleTemplate],
        **kwargs
    ) -> Dict[str, SizeCurve]:
        """
        Analyze the sales data to generate size curves.
        
        Args:
            sales_data (pd.DataFrame): The sales data to analyze
            style_templates (Dict[str, StyleTemplate]): Style templates to use
            **kwargs: Additional arguments
        
        Returns:
            Dict[str, SizeCurve]: Dictionary mapping identifiers to size curves
        """
        pass
    
    def prepare_output_dataframe(self, size_curves: Dict[str, SizeCurve]) -> pd.DataFrame:
        """
        Prepare output DataFrame from size curves.
        
        Args:
            size_curves (Dict[str, SizeCurve]): The size curves to convert
        
        Returns:
            pd.DataFrame: The size curves as a DataFrame
        """
        if not size_curves:
            return pd.DataFrame()
        
        # Prepare data for dataframe
        data = []
        
        for _, curve in size_curves.items():
            # Get the sizes for this curve
            template_sizes = curve.size_template_name.split('-')
            
            # Determine the correct column name based on level type
            level_to_column = {
                "STYLE": "STYLE_CODE",
                "CLASS": "PRODUCT_CLASS_TEXT",
                "SUBCLASS": "PRODUCT_SUB_CLASS_TEXT",
                "COLLECTION": "COLLECTION_TEXT"
            }
            
            item_col = level_to_column.get(curve.level_type, "STYLE_CODE")
            
            
            row = {
                item_col: curve.item_code,
                'GENDER_CODE': curve.gender_code,
                'SIZE_TEMPLATE': curve.size_template_name,
                'LEVEL_TYPE': curve.level_type
            }
            
            if curve.channel:
                row['channel'] = curve.channel
            
            # Add size percentages
            for size in template_sizes:
                row[size] = round(curve.size_percentages.get(size, 0.0), 2)
            
            data.append(row)
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        return df