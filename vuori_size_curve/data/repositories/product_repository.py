"""
Product repository for accessing product data.
"""
from typing import List, Dict, Optional, Any
import pandas as pd
from vuori_size_curve.data.repositories.base_repository import BaseRepository
from vuori_size_curve.data.models.product import Product, SizeTemplate, StyleTemplate
from vuori_size_curve.data.connectors.base_connector import BaseConnector
from vuori_size_curve.config.database_config import PRODUCT_QUERY_TEMPLATE
from vuori_size_curve.utils.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)


class ProductRepository(BaseRepository[Product]):
    """
    Repository for accessing product data.
    """
    
    def __init__(self, connector: BaseConnector):
        """
        Initialize the product repository.
        
        Args:
            connector (BaseConnector): The database connector to use
        """
        super().__init__(connector)
        self.style_color_templates = {}  # Cache for style+color templates
    
    def get_all(self, *args, **kwargs) -> List[Product]:
        """
        Get all products that match the specified criteria.
        
        Returns:
            List[Product]: A list of Product objects
        """
        df = self.get_raw_data(*args, **kwargs)
        
        # Convert DataFrame rows to Product objects
        products = []
        for _, row in df.iterrows():
            product = Product(
                style_code=row['STYLE_CODE'],
                gender_code=row.get('GENDER_CODE', ''),
                color_code=row.get('COLOR_CODE'),
                size_code=row.get('SIZE_CODE'),
                size_order=row.get('SIZE_ORDER')
            )
            products.append(product)
        
        return products
    
    def get_raw_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        Get raw product data as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The product data
        """
        logger.info("Fetching product data...")
        df = self._execute_query(PRODUCT_QUERY_TEMPLATE)
        
        # Filter out invalid size data
        if 'SIZE_CODE' in df.columns and 'SIZE_ORDER' in df.columns:
            original_count = len(df)
            df = df[
                (~df['SIZE_CODE'].str.contains('_N/A', na=False)) & 
                (df['SIZE_ORDER'] > 0) & 
                (df['SIZE_ORDER'].notnull())
            ]
            logger.info(f"Retrieved {len(df)} valid product records (filtered {original_count - len(df)} invalid records).")
        
        return df
    
    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a query using the connector.
        
        Args:
            query (str): The SQL query to execute
            params (Optional[Dict[str, Any]]): Parameters to bind to the query
            
        Returns:
            pd.DataFrame: The query results
        """
        if hasattr(self.connector, 'execute_query'):
            # Use parameterized query if the connector supports it
            if hasattr(self.connector.execute_query, '__code__') and 'params' in self.connector.execute_query.__code__.co_varnames:
                return self.connector.execute_query(query, params)
            else:
                # Fallback to older connector implementation without params support
                return self.connector.execute_query(query)
        else:
            raise NotImplementedError("Connector does not implement execute_query method")
    
    def determine_size_templates(self) -> Dict[str, StyleTemplate]:
        """
        Determine size templates for each style+color combination.
        
        Returns:
            Dict[str, StyleTemplate]: Dictionary mapping style codes to their StyleTemplate objects
            Also creates a separate dictionary for style+color combinations
        """
        product_data = self.get_raw_data()
        style_templates = {}
        style_color_templates = {}
        
        # First, determine templates by STYLE_CODE+COLOR_CODE (more accurate)
        if 'COLOR_CODE' in product_data.columns:
            logger.info("Creating templates by STYLE_CODE+COLOR_CODE combinations...")
            
            for (style_code, color_code), group in product_data.groupby(['STYLE_CODE', 'COLOR_CODE']):
                # Get unique sizes for this style+color, sorted by SIZE_ORDER
                sizes_with_order = group[['SIZE_CODE', 'SIZE_ORDER']].drop_duplicates()
                
                # Ensure we're getting unique sizes only
                sizes_with_order = sizes_with_order.drop_duplicates('SIZE_CODE')
                sizes_sorted = sizes_with_order.sort_values('SIZE_ORDER')
                sizes = sizes_sorted['SIZE_CODE'].tolist()
                
                # Skip combinations with no valid sizes after filtering
                if not sizes:
                    continue
                    
                # Create a template name by joining all sizes
                template_name = "-".join(sizes)
                
                # Create a SizeTemplate object
                size_template = SizeTemplate(name=template_name, sizes=sizes)
                
                # Create a key for the style+color combination
                style_color_key = f"{style_code}_{color_code}"
                
                # Create a StyleTemplate object and add to the dictionary
                style_color_templates[style_color_key] = StyleTemplate(
                    style_code=style_code,
                    size_template=size_template,
                    color_code=color_code  # Store color code in the template
                )
            
            logger.info(f"Determined size templates for {len(style_color_templates)} style+color combinations.")
        
        # Also determine templates by STYLE_CODE only (for backward compatibility)
        for style_code, group in product_data.groupby('STYLE_CODE'):
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
            
            # Create a SizeTemplate object
            size_template = SizeTemplate(name=template_name, sizes=sizes)
            
            # Create a StyleTemplate object and add to the dictionary
            style_templates[style_code] = StyleTemplate(
                style_code=style_code,
                size_template=size_template
            )
        
        logger.info(f"Determined size templates for {len(style_templates)} styles.")
        
        # Store the style+color templates for later use
        self.style_color_templates = style_color_templates
        
        return style_templates