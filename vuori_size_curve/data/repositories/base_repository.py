"""
Base repository interface for data access.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Generic, TypeVar
import pandas as pd
from vuori_size_curve.data.connectors.base_connector import BaseConnector
from vuori_size_curve.utils.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)

# Generic type for repository entities
T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base class for repositories that provide data access.
    """
    
    def __init__(self, connector: BaseConnector):
        """
        Initialize the repository with a database connector.
        
        Args:
            connector (BaseConnector): The database connector to use
        """
        self.connector = connector
    
    @abstractmethod
    def get_all(self, *args, **kwargs) -> List[T]:
        """
        Get all entities that match the specified criteria.
        
        Returns:
            List[T]: A list of entity objects
        """
        pass
    
    @abstractmethod
    def get_raw_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        Get raw data as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The raw data as a pandas DataFrame
        """
        pass
    
    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query using the connector.
        
        Args:
            query (str): The SQL query to execute
            params (Optional[Dict[str, Any]]): Parameters to bind to the query
            
        Returns:
            pd.DataFrame: The query results as a pandas DataFrame
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
                    logger.debug(f"Using formatted query (backward compatibility): {formatted_query[:100]}...")
                    return self.connector.execute_query(formatted_query)
                else:
                    return self.connector.execute_query(query)
        else:
            raise NotImplementedError("Connector does not implement execute_query method")