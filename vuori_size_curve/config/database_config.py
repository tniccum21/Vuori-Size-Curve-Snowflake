"""
Database configuration settings for the Vuori Size Curve Analyzer.
"""
import os
from typing import Dict, Any
import logging
from vuori_size_curve.utils.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)

def get_snowflake_config() -> Dict[str, Any]:
    """
    Get Snowflake configuration from environment variables or defaults.
    
    Returns:
        Dict[str, Any]: Snowflake configuration dictionary
    """
    # Default configuration
    default_config = {
        "account": "vuori.west-us-2.azure",
        "user": "thomas.niccum@vuori.com",
        "authenticator": "externalbrowser",  # SSO flow
        "warehouse": "VUORINOTEBOOK_XS",
        "database": "PROD_DL",
        "schema": "CURATED"
    }
    
    # Override with environment variables if available
    config = {}
    for key in default_config:
        env_key = f"SNOWFLAKE_{key.upper()}"
        config[key] = os.environ.get(env_key, default_config[key])
    
    logger.debug(f"Using Snowflake config with account: {config['account']}, user: {config['user']}")
    
    return config

# Snowflake connection parameters
SNOWFLAKE_CONFIG: Dict[str, Any] = get_snowflake_config()

# Query templates with placeholders
SALES_QUERY_TEMPLATE = """
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
    s.ORDER_DATE >= :start_date
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

PRODUCT_QUERY_TEMPLATE = """
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

# Additional database configurations can be added here if needed
# For example, alternative environments or database systems