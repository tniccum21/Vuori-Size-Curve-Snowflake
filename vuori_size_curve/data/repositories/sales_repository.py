"""
Sales repository for accessing sales data.
"""
from typing import List, Dict, Optional, Any
import pandas as pd
from vuori_size_curve.data.repositories.base_repository import BaseRepository
from vuori_size_curve.data.models.sales import SalesRecord, FilterCriteria
from vuori_size_curve.data.connectors.base_connector import BaseConnector
from vuori_size_curve.config.app_config import SALES_QUERY_TEMPLATE
from vuori_size_curve.utils.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)


class SalesRepository(BaseRepository[SalesRecord]):
    """
    Repository for accessing sales data.
    """
    
    def __init__(self, connector: BaseConnector):
        """
        Initialize the sales repository.
        
        Args:
            connector (BaseConnector): The database connector to use
        """
        super().__init__(connector)
    
    def get_all(self, filter_criteria: Optional[FilterCriteria] = None) -> List[SalesRecord]:
        """
        Get all sales records that match the specified criteria.
        
        Args:
            filter_criteria (Optional[FilterCriteria]): Filtering criteria
        
        Returns:
            List[SalesRecord]: A list of SalesRecord objects
        """
        df = self.get_raw_data(filter_criteria)
        
        # Convert DataFrame rows to SalesRecord objects
        sales_records = []
        for _, row in df.iterrows():
            sales_record = SalesRecord(
                style_code=row['STYLE_CODE'],
                gender_code=row['GENDER_CODE'],
                size_code=row['SIZE_CODE'],
                size_order=row['SIZE_ORDER'],
                total_sales=row['TOTAL_SALES'],
                distribution_channel_code=row['DISTRIBUTION_CHANNEL_CODE'],
                color_code=row.get('COLOR_CODE'),
                collection_text=row.get('COLLECTION_TEXT'),
                nrf_color_text=row.get('NRF_COLOR_TEXT'),
                product_class_text=row.get('PRODUCT_CLASS_TEXT'),
                product_sub_class_text=row.get('PRODUCT_SUB_CLASS_TEXT')
            )
            sales_records.append(sales_record)
        
        return sales_records
    
    def get_raw_data(self, filter_criteria: Optional[FilterCriteria] = None) -> pd.DataFrame:
        """
        Get raw sales data as a pandas DataFrame.
        
        Args:
            filter_criteria (Optional[FilterCriteria]): Filtering criteria
        
        Returns:
            pd.DataFrame: The sales data
        """
        # Set default filter criteria if not provided
        if filter_criteria is None:
            filter_criteria = FilterCriteria()
        
        # Create parameters for the query
        params = {
            "start_date": filter_criteria.start_date
        }
        
        # Execute the query with parameters
        logger.info(f"Fetching sales data from {filter_criteria.start_date}...")
        df = self._execute_query(SALES_QUERY_TEMPLATE, params)
        logger.info(f"Retrieved {len(df)} sales records.")
        
        # Apply additional filters
        df = self._apply_filters(df, filter_criteria)
        
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
                # Format the query with parameters for backward compatibility
                if params:
                    formatted_query = query
                    for key, value in params.items():
                        placeholder = f":{key}"
                        formatted_query = formatted_query.replace(placeholder, f"'{value}'")
                    return self.connector.execute_query(formatted_query)
                else:
                    return self.connector.execute_query(query)
        else:
            raise NotImplementedError("Connector does not implement execute_query method")
    
    def _apply_filters(self, df: pd.DataFrame, filter_criteria: FilterCriteria) -> pd.DataFrame:
        """
        Apply additional filters to the sales data.
        
        Args:
            df (pd.DataFrame): The sales data
            filter_criteria (FilterCriteria): Filtering criteria
        
        Returns:
            pd.DataFrame: The filtered sales data
        """
        # Filter by channels if specified
        if filter_criteria.channels_filter and 'DISTRIBUTION_CHANNEL_CODE' in df.columns:
            df = df[df['DISTRIBUTION_CHANNEL_CODE'].isin(filter_criteria.channels_filter)]
        
        # Filter by gender if specified
        if filter_criteria.gender_filter and filter_criteria.gender_filter != 'ALL' and 'GENDER_CODE' in df.columns:
            df = df[df['GENDER_CODE'] == filter_criteria.gender_filter]
        
        # Filter by class if specified
        if filter_criteria.class_filter and filter_criteria.class_filter != 'ALL' and 'PRODUCT_CLASS_TEXT' in df.columns:
            df = df[df['PRODUCT_CLASS_TEXT'] == filter_criteria.class_filter]
        
        # Filter by subclass if specified
        if filter_criteria.subclass_filter and filter_criteria.subclass_filter != 'ALL' and 'PRODUCT_SUB_CLASS_TEXT' in df.columns:
            df = df[df['PRODUCT_SUB_CLASS_TEXT'] == filter_criteria.subclass_filter]
        
        # Filter by style if specified
        if filter_criteria.style_filter and filter_criteria.style_filter != 'ALL' and 'STYLE_CODE' in df.columns:
            df = df[df['STYLE_CODE'] == filter_criteria.style_filter]
        
        # Filter out invalid size data
        if 'SIZE_CODE' in df.columns and 'SIZE_ORDER' in df.columns:
            original_count = len(df)
            df = df[
                (~df['SIZE_CODE'].str.contains('_N/A', na=False)) & 
                (df['SIZE_ORDER'] > 0) & 
                (df['SIZE_ORDER'].notnull())
            ]
            if len(df) < original_count:
                logger.info(f"Filtered out {original_count - len(df)} records with invalid size data.")
        
        return df
    
    def get_unique_channels(self) -> List[str]:
        """
        Get a list of unique distribution channels in the sales data.
        
        Returns:
            List[str]: A list of channel codes
        """
        df = self.get_raw_data()
        
        if 'DISTRIBUTION_CHANNEL_CODE' in df.columns:
            return sorted(df['DISTRIBUTION_CHANNEL_CODE'].unique().tolist())
        else:
            return []