"""
Snowflake configuration settings - Import from the main config module

This file is maintained for backward compatibility.
New code should import from vuori_size_curve.config.database_config directly.
"""
from vuori_size_curve.config.database_config import SNOWFLAKE_CONFIG, get_snowflake_config

# Re-export for backward compatibility
__all__ = ['SNOWFLAKE_CONFIG', 'get_snowflake_config']