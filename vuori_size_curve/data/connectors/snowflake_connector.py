"""
Snowflake database connector implementation.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional
from snowflake.snowpark import Session
from vuori_size_curve.data.connectors.base_connector import BaseConnector
from vuori_size_curve.config.database_config import SNOWFLAKE_CONFIG
from vuori_size_curve.utils.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)


class SnowflakeConnector(BaseConnector):
    """
    Connector for Snowflake database.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Snowflake connector.
        
        Args:
            config (Optional[Dict[str, Any]]): Snowflake connection configuration.
                                              If None, uses the default from database_config.py
        """
        self.config = config if config is not None else SNOWFLAKE_CONFIG
        self.session = None
    
    def connect(self) -> Session:
        """
        Establish a connection to Snowflake.
        
        Returns:
            Session: The Snowflake session object
        """
        if self.session is None:
            try:
                self.session = Session.builder.configs(self.config).create()
                logger.info("Snowflake connection established.")
            except Exception as e:
                logger.error(f"Error connecting to Snowflake: {str(e)}")
                raise
        
        return self.session
    
    def disconnect(self) -> None:
        """
        Close the Snowflake connection.
        """
        try:
            if self.session is not None:
                self.session.close()
                logger.info("Snowflake connection closed.")
                self.session = None
        except Exception as e:
            logger.error(f"Error closing Snowflake connection: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query on Snowflake and return the results as a DataFrame.
        
        Args:
            query (str): The SQL query to execute
            params (Optional[Dict[str, Any]]): Parameters to bind to the query
            
        Returns:
            pd.DataFrame: The query results as a pandas DataFrame
        """
        if self.session is None:
            self.connect()
        
        try:
            # Replace parameter placeholders directly in the query string
            if params:
                formatted_query = query
                for key, value in params.items():
                    # Handle proper quoting for different data types
                    if isinstance(value, str):
                        # Use single quotes for string values
                        formatted_value = f"'{value}'"
                    elif value is None:
                        formatted_value = "NULL"
                    else:
                        # Numbers and other types don't need quotes
                        formatted_value = str(value)
                    
                    placeholder = f":{key}"
                    formatted_query = formatted_query.replace(placeholder, formatted_value)
                
                logger.debug(f"Executing query with replaced parameters: {formatted_query[:200]}...")
                snow_df = self.session.sql(formatted_query)
            else:
                snow_df = self.session.sql(query)
            
            return snow_df.to_pandas()
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise