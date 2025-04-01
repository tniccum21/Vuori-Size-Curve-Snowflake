"""
Application-wide configuration settings for the Vuori Size Curve Analyzer.
"""
from datetime import datetime
import os
from typing import Dict, Any, List, Optional
from vuori_size_curve.utils.logging_config import get_logger
from vuori_size_curve.config.database_config import SALES_QUERY_TEMPLATE, PRODUCT_QUERY_TEMPLATE

# Set up logging
logger = get_logger(__name__)

# Default settings
DEFAULT_START_DATE = os.environ.get("SIZE_CURVE_DEFAULT_START_DATE", "2022-01-01")
DEFAULT_MIN_SALES = int(os.environ.get("SIZE_CURVE_MIN_SALES", "0"))
DEFAULT_OUTPUT_DIR = None  # Will be generated based on timestamp if None

# Default filter values
DEFAULT_CHANNELS_FILTER: Optional[List[str]] = None  # No filter by default

# Import SQL queries from database_config to ensure consistency
# These are maintained for backward compatibility
SALES_QUERY = SALES_QUERY_TEMPLATE
PRODUCT_QUERY = PRODUCT_QUERY_TEMPLATE

# Analysis levels
ANALYSIS_LEVELS = ["style", "subclass", "class", "collection"]

# Column mapping for different levels
LEVEL_COLUMN_MAP = {
    "style": "STYLE_CODE",
    "subclass": "PRODUCT_SUB_CLASS_TEXT",
    "class": "PRODUCT_CLASS_TEXT",
    "collection": "COLLECTION_TEXT"
}

# Visualization settings
DEFAULT_CHART_HEIGHT = 600
DEFAULT_CHART_WIDTH = None  # Full width