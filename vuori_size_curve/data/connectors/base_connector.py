"""
Base database connector interface.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


class BaseConnector(ABC):
    """
    Abstract base class for database connections.
    """
    
    @abstractmethod
    def connect(self) -> Any:
        """
        Establish a connection to the database.
        
        Returns:
            Any: The database connection/session object
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the database connection.
        """
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a DataFrame.
        
        Args:
            query (str): The SQL query to execute
            params (Optional[Dict[str, Any]]): Parameters to bind to the query
            
        Returns:
            pd.DataFrame: The query results as a pandas DataFrame
        """
        pass
    
    def __enter__(self):
        """
        Context manager entry point.
        
        Returns:
            BaseConnector: The connector instance
        """
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point.
        
        Args:
            exc_type: Exception type if an exception was raised in the context
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        self.disconnect()