import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Any

# Import the configuration from database_config
from vuori_size_curve.config.database_config import SNOWFLAKE_CONFIG

# Specific database and schema for ERP tables
SNOWFLAKE_DATABASE = "DEV_DW" # As specified
SNOWFLAKE_SCHEMA = "PLANNING" # As specified

# --- Snowflake Table Names ---
MAPPING_TABLE = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.SIZE_CURVE_CHANNEL_SEASON_CC"
WEIGHTS_TABLE = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.SIZE_CURVE_ERP_WEIGHT_PCT"


class ERPSizeRepository:
    """
    Repository for loading and accessing ERP size curve data from Snowflake.
    """

    def __init__(self, snowflake_config: Optional[Dict[str, str]] = None):
        """
        Initialize the repository with Snowflake connection details.

        Args:
            snowflake_config: Optional dictionary containing Snowflake connection
                              parameters ('user', 'password', 'account', 'warehouse',
                              'database', 'schema'). If None, uses the SNOWFLAKE_CONFIG
                              from database_config.py.
        """
        # Start with the global Snowflake config
        base_config = SNOWFLAKE_CONFIG.copy()
        
        # Update database and schema to use ERP-specific values
        base_config.update({
            'database': SNOWFLAKE_DATABASE,
            'schema': SNOWFLAKE_SCHEMA
        })
        
        # Allow override with provided config
        self.snowflake_config = snowflake_config or base_config
        
        # No need to validate credentials since we're using SSO

        self.style_to_weight_map: Dict[str, Any] = {}
        self.weight_to_distribution_map: Dict[str, Dict[str, float]] = {}
        self.is_initialized = False
        self.conn = None # Snowflake connection object

    def _connect_snowflake(self):
        """Establishes a connection to Snowflake using Snowpark."""
        if self.conn:
            print("Using existing Snowflake connection.")
            return self.conn

        print("Establishing new Snowflake connection...")
        try:
            # Use snowpark Session instead of connector
            from snowflake.snowpark import Session
            
            # Create new session using config
            self.conn = Session.builder.configs(self.snowflake_config).create()
            print("Snowflake connection established successfully.")
            return self.conn
        except Exception as e:
            print(f"Error connecting to Snowflake: {e}")
            raise RuntimeError(f"Could not connect to Snowflake: {e}") from e

    def _close_snowflake(self):
        """Closes the Snowflake connection if open."""
        if self.conn:
            print("Closing Snowflake connection.")
            self.conn.close()
            self.conn = None

    def initialize(self):
        """
        Load data from Snowflake tables.
        """
        if self.is_initialized:
            print("Repository already initialized.")
            return

        print("Initializing ERPSizeRepository from Snowflake...")
        try:
            self._connect_snowflake() # Establish connection

            # Load mapping data 
            print(f"Loading mapping data from {MAPPING_TABLE}...")
            try:
                self._load_mapping_from_snowflake()
            except Exception as e:
                print(f"Error loading mapping data: {e}")
                raise

            # Load weights data
            print(f"Loading weights data from {WEIGHTS_TABLE}...")
            try:
                self._load_weights_from_snowflake()
            except Exception as e:
                print(f"Error loading weights data: {e}")
                raise

            self.is_initialized = True
            print("ERPSizeRepository initialization complete.")

        except Exception as e:
            # Ensure connection is closed even if loading fails
            self._close_snowflake()
            # Re-raise the exception after cleanup
            raise RuntimeError(f"Failed to initialize ERPSizeRepository: {e}") from e
        # No finally block needed for closing here, as we might reuse the connection
        # if initialization is successful. Consider closing explicitly when done
        # with the repository instance if it's long-lived.

    def _load_mapping_from_snowflake(self):
        """Load and process style to size weight code mapping data from Snowflake."""
        if not self.conn:
            raise RuntimeError("Snowflake connection not established.")

        # Query using the actual column names from SIZE_CURVE_CHANNEL_SEASON_CC
        query = f"""
        SELECT
            STYLE_CODE,       -- Style code
            COLOR,            -- Color code 
            CHANNEL,          -- Channel (Ecomm or Retail)
            SEASON,           -- Season (FAHO25, SPSU25, etc.)
            SIZE_WEIGHT_CODE  -- The weight code that maps to size distributions
        FROM {MAPPING_TABLE}
        WHERE SIZE_WEIGHT_CODE IS NOT NULL
        """
        print(f"Executing query: {query}")

        try:
            # Execute query using Snowpark API
            mapping_df = self.conn.sql(query).to_pandas()
            print(f"Fetched {len(mapping_df)} mapping rows from Snowflake.")

            if mapping_df.empty:
                 print(f"Warning: No mapping data found in {MAPPING_TABLE}.")
                 return # Or raise error if mapping data is essential

            # --- Adapt column names for processing ---
            # Rename Snowflake columns to match what _process_mapping_data expects
            # This avoids changing the logic inside _process_mapping_data too much.
            rename_map = {
                'STYLE_CODE': 'Style Code',
                'COLOR': 'Color',
                'SIZE_WEIGHT_CODE': 'Size Weight Code',
                'CHANNEL': 'Source', # Map Snowflake 'CHANNEL' to 'Source'
                'SEASON': 'Season'   # Map Snowflake 'SEASON' to 'Season'
            }
            # Filter out columns not present in the DataFrame before renaming
            rename_map = {k: v for k, v in rename_map.items() if k in mapping_df.columns}
            mapping_df_renamed = mapping_df.rename(columns=rename_map)

            # --- Process Data ---
            # We need to simulate the different sources (Ecomm/Retail) if they come
            # from a single 'CHANNEL' column.
            # The original _process_mapping_data expects separate calls or dataframes
            # per source. Let's adapt the processing here or modify _process_mapping_data.
            # Option 1: Call _process_mapping_data per source/season group (simpler)

            # Ensure 'Source' and 'Season' columns exist after renaming
            required_cols = ['Style Code', 'Size Weight Code', 'Source', 'Season']
            if not all(col in mapping_df_renamed.columns for col in required_cols):
                 missing = [col for col in required_cols if col not in mapping_df_renamed.columns]
                 raise ValueError(f"Missing required columns after renaming for mapping processing: {missing}. Check Snowflake query and rename_map.")

            # Fill NA for optional columns like Color before processing
            if 'Color' in mapping_df_renamed.columns:
                 mapping_df_renamed['Color'] = mapping_df_renamed['Color'].fillna('') # Use empty string for missing color

            # Group by Source and Season to process similarly to original logic
            # Convert potential non-string types just in case
            mapping_df_renamed['Source'] = mapping_df_renamed['Source'].astype(str)
            mapping_df_renamed['Season'] = mapping_df_renamed['Season'].astype(str)

            for (source, season), group_df in mapping_df_renamed.groupby(['Source', 'Season']):
                 print(f"Processing mapping data for Source: {source}, Season: {season}")
                 # We need to pass the correct weight code column name based on source
                 # Since we renamed to 'Size Weight Code', we use that directly.
                 # We also need to adapt _process_mapping_data slightly or pass the col name.
                 self._process_mapping_data(group_df, source=source, season=season, weight_code_col='Size Weight Code')


        except Exception as e:
            print(f"Error loading or processing mapping data from Snowflake: {e}")
            raise RuntimeError(f"Error loading ERP mapping data from Snowflake: {str(e)}") from e


    # --- Modified _process_mapping_data ---
    # Added weight_code_col parameter
    def _process_mapping_data(self, mapping_df, source, season=None, weight_code_col='Size Weight Code'):
        """
        Process mapping data from a DataFrame (now sourced from Snowflake).

        Args:
            mapping_df: DataFrame containing mapping data for a specific source/season.
            source: Source of the data ('Ecomm', 'Retail', etc.).
            season: Optional season identifier.
            weight_code_col: The name of the column containing the size weight code.
        """
        print(f"Available columns in mapping data for {source}{' - ' + season if season else ''}: {list(mapping_df.columns)}")

        # Check if the expected weight code column exists
        if weight_code_col not in mapping_df.columns:
            print(f"Error: Expected weight column '{weight_code_col}' not found in mapping data for {source}{' - ' + season if season else ''}.")
            return # Skip processing this group

        # Ensure 'Style Code' exists
        if 'Style Code' not in mapping_df.columns:
            print(f"Error: 'Style Code' column not found in mapping data for {source}{' - ' + season if season else ''}.")
            return

        # Handle optional 'Color' column
        has_color_col = 'Color' in mapping_df.columns

        for _, row in mapping_df.iterrows():
            # Use .get() with default None for safety, handle potential NaN/None from DB/Pandas
            style_code_val = row.get('Style Code')
            weight_code_val = row.get(weight_code_col)
            color_code_val = row.get('Color') if has_color_col else None

            # Convert to string and check for validity, skip if essential parts are missing
            style_code = str(style_code_val) if pd.notna(style_code_val) else None
            weight_code = str(weight_code_val) if pd.notna(weight_code_val) else None
            color_code = str(color_code_val) if pd.notna(color_code_val) and color_code_val else None # Treat empty string/None as no color

            if style_code and weight_code:
                # Create mapping entries with various combinations of keys for flexible lookup

                # 1. Keys with source and season (most specific)
                if season:
                    # Style with source and season
                    key_with_season = f"{style_code}_{source}_{season}"
                    self.style_to_weight_map[key_with_season] = weight_code

                    # Style+color with source and season
                    if color_code:
                        combined_key_with_season = f"{style_code}|{color_code}_{source}_{season}"
                        self.style_to_weight_map[combined_key_with_season] = weight_code

                # 2. Keys with just source (backward compatibility / alternative)
                key_with_source = f"{style_code}_{source}"
                # Only add if a more specific key (with season) doesn't already exist for this combo
                if key_with_source not in self.style_to_weight_map:
                     self.style_to_weight_map[key_with_source] = weight_code

                if color_code:
                    combined_key_with_source = f"{style_code}|{color_code}_{source}"
                    if combined_key_with_source not in self.style_to_weight_map:
                         self.style_to_weight_map[combined_key_with_source] = weight_code

                # 3. Keys without source or season (least specific, lowest priority)
                # Only add if no other key exists for this style code yet
                if style_code not in self.style_to_weight_map:
                    self.style_to_weight_map[style_code] = weight_code

                if color_code:
                    regular_combined = f"{style_code}|{color_code}"
                    if regular_combined not in self.style_to_weight_map:
                        self.style_to_weight_map[regular_combined] = weight_code
            # else: # Optional: Log skipped rows
            #     print(f"Skipping mapping row due to missing style_code or weight_code: {row.to_dict()}")


    def _load_weights_from_snowflake(self):
        """Load and process size weight code to distribution data from Snowflake."""
        if not self.conn:
            raise RuntimeError("Snowflake connection not established.")

        # Query using the actual column names from SIZE_CURVE_ERP_WEIGHT_PCT
        query = f"""
        SELECT
            SCALECODE,     -- The code that identifies the size curve
            SIZEVALUEID,   -- The actual size (e.g., 'S', 'M', 'L', '32')
            QUANTITY       -- The quantity or weight for that size
        FROM {WEIGHTS_TABLE}
        WHERE QUANTITY IS NOT NULL AND QUANTITY > 0
        """
        print(f"Executing query: {query}")

        try:
            # Execute query using Snowpark API
            weights_df = self.conn.sql(query).to_pandas()
            print(f"Fetched {len(weights_df)} weight rows from Snowflake.")

            if weights_df.empty:
                 print(f"Warning: No weights data found in {WEIGHTS_TABLE}.")
                 return # Or raise error

            # The column names already match what _process_weights_data expects
            # No need for renaming
            weights_df_renamed = weights_df.copy()

            # --- Process Data ---
            # The _process_weights_data method expects a DataFrame with specific columns
            self._process_weights_data(weights_df_renamed) # Pass the renamed DataFrame

        except Exception as e:
            print(f"Error loading or processing weights data from Snowflake: {e}")
            raise RuntimeError(f"Error loading ERP weights data from Snowflake: {str(e)}") from e


    # --- Modified _process_weights_data ---
    # Primarily updated logging and potentially column name checks if needed
    def _process_weights_data(self, weights_df, season=None): # Season arg kept for compatibility but likely unused with Snowflake table
        """
        Process weights data from a dataframe (now sourced from Snowflake).

        Args:
            weights_df: DataFrame containing weights data.
            season: Optional season identifier (mostly for logging context here).
        """
        print(f"Processing weights data{' for context ' + season if season else ''}. Available columns: {list(weights_df.columns)}")

        # Check for required columns (using the names expected *after* potential renaming)
        required_columns = ['SCALECODE', 'SIZEVALUEID', 'QUANTITY']
        missing_columns = [col for col in required_columns if col not in weights_df.columns]

        if missing_columns:
            print(f"Error: Missing required columns in weights data after renaming: {missing_columns}. Check Snowflake query and rename_map in _load_weights_from_snowflake.")
            return # Stop processing if essential columns are absent

        # Convert relevant columns to appropriate types for safety
        try:
            weights_df['SCALECODE'] = weights_df['SCALECODE'].astype(str)
            weights_df['SIZEVALUEID'] = weights_df['SIZEVALUEID'].astype(str)
            weights_df['QUANTITY'] = pd.to_numeric(weights_df['QUANTITY'], errors='coerce')
            weights_df = weights_df.dropna(subset=['SCALECODE', 'SIZEVALUEID', 'QUANTITY']) # Remove rows where conversion failed or data was missing
        except Exception as e:
            print(f"Error converting weight data types: {e}")
            return

        # Group by SCALECODE and create size distributions
        try:
            # Group by the scale code (e.g., size weight code)
            grouped = weights_df.groupby('SCALECODE')

            for scale_code, group in grouped:
                size_distribution = {}
                # Ensure QUANTITY is numeric before summing
                total_quantity = group['QUANTITY'].sum()

                if total_quantity > 0:
                    for _, row in group.iterrows():
                        size_value = row.get('SIZEVALUEID')
                        quantity = row.get('QUANTITY', 0)
                        if size_value: # Ensure size value is present
                            # Calculate percentage
                            percentage = (quantity / total_quantity) * 100.0
                            size_distribution[size_value] = percentage
                elif len(group) > 0 : # Handle cases where total quantity is 0 but rows exist
                    print(f"Warning: Total quantity is 0 for SCALECODE '{scale_code}'. Assigning equal distribution.")
                    num_sizes = len(group['SIZEVALUEID'].unique())
                    if num_sizes > 0:
                        equal_pct = 100.0 / num_sizes
                        for size_val in group['SIZEVALUEID'].unique():
                             size_distribution[str(size_val)] = equal_pct # Ensure key is string

                # Store the distribution for this scale code
                if size_distribution: # Only store if we created a distribution
                    self.weight_to_distribution_map[scale_code] = size_distribution
                # else: # Optional: Log scale codes with no valid distribution data
                #     print(f"No valid distribution created for SCALECODE: {scale_code}")

        except Exception as e:
            print(f"Error processing weights data during grouping/calculation: {str(e)}")
            # Potentially log more details or raise the error depending on requirements


    def get_erp_size_distribution(self, style_code: str, color_code: Optional[str] = None,
                               source: Optional[str] = None, season: Optional[str] = None) -> Tuple[Dict[str, float], Optional[str], Optional[str]]:
        """
        Get the ERP-recommended size distribution for a style or color choice.
        (This method remains largely the same as it uses the populated dictionaries)

        Args:
            style_code: The style code to look up
            color_code: Optional color code for more specific lookup
            source: Optional data source ('Ecomm' or 'Retail' or other values from your CHANNEL column).
            season: Optional season (e.g., 'FAHO25', 'SPSU25').
            
        Returns:
            Tuple of (distribution_dict, source_used, season_used), where:
            - distribution_dict: Dictionary mapping size codes to percentage values
            - source_used: String indicating which source was used (e.g., 'Ecomm', 'Retail', or None)
            - season_used: String indicating which season was used (e.g., 'FAHO25', or None)
        """
        if not self.is_initialized:
             # Option 1: Auto-initialize if not done yet
             print("Repository not initialized. Initializing now...")
             self.initialize()
             # Option 2: Raise error (uncomment below if preferred)
             # raise RuntimeError("ERPSizeRepository not initialized. Call initialize() first.")


        size_weight_code = None
        source_used = None
        season_used = None

        # --- Lookup Logic (remains the same, uses the populated self.style_to_weight_map) ---

        # Define default search order if not provided
        # *** Adjust default sources/seasons based on your data priorities ***
        sources_to_try = [source] if source else ['Ecomm', 'Retail'] # Example defaults
        seasons_to_try = [season] if season else ['FAHO25', 'SPSU25'] # Example defaults

        style_code_str = str(style_code) # Ensure string type
        color_code_str = str(color_code) if color_code else None

        # Strategy 1: Try most specific keys first (with season and source)
        if color_code_str:
            for try_season in seasons_to_try:
                if size_weight_code: break
                for try_source in sources_to_try:
                    combined_key = f"{style_code_str}|{color_code_str}_{try_source}_{try_season}"
                    if combined_key in self.style_to_weight_map:
                        size_weight_code = self.style_to_weight_map[combined_key]
                        source_used = try_source
                        season_used = try_season
                        break # Found specific match

        if not size_weight_code: # Try style code with season and source
            for try_season in seasons_to_try:
                if size_weight_code: break
                for try_source in sources_to_try:
                    key = f"{style_code_str}_{try_source}_{try_season}"
                    if key in self.style_to_weight_map:
                        size_weight_code = self.style_to_weight_map[key]
                        source_used = try_source
                        season_used = try_season
                        break # Found specific match

        # Strategy 2: Try source-specific keys (without season)
        if not size_weight_code:
            if color_code_str: # Try with color code and source
                for try_source in sources_to_try:
                    combined_key = f"{style_code_str}|{color_code_str}_{try_source}"
                    if combined_key in self.style_to_weight_map:
                        size_weight_code = self.style_to_weight_map[combined_key]
                        source_used = try_source
                        season_used = None # Season not specified in this key type
                        break

            if not size_weight_code: # Try style code with source only
                for try_source in sources_to_try:
                    key = f"{style_code_str}_{try_source}"
                    if key in self.style_to_weight_map:
                        size_weight_code = self.style_to_weight_map[key]
                        source_used = try_source
                        season_used = None # Season not specified
                        break

        # Strategy 3: Try non-source/season-specific keys (least specific)
        if not size_weight_code:
            if color_code_str: # Try style+color
                combined_key = f"{style_code_str}|{color_code_str}"
                if combined_key in self.style_to_weight_map:
                    size_weight_code = self.style_to_weight_map[combined_key]
                    source_used = None # Source not specified
                    season_used = None # Season not specified

            if not size_weight_code: # Try style only
                 if style_code_str in self.style_to_weight_map:
                     size_weight_code = self.style_to_weight_map[style_code_str]
                     source_used = None # Source not specified
                     season_used = None # Season not specified


        # --- Lookup Distribution ---
        if size_weight_code:
            # Ensure weight code is string for dict lookup
            distribution = self.weight_to_distribution_map.get(str(size_weight_code), {})
            if distribution:
                 print(f"Found ERP size distribution for {style_code_str} (color: {color_code_str}) using key derived from source: {source_used}, season: {season_used}. Weight code: {size_weight_code}")
                 return distribution, source_used, season_used
            else:
                 print(f"Warning: Found weight code '{size_weight_code}' for {style_code_str} but no corresponding distribution exists in weights data.")
                 return {}, source_used, season_used # Return empty dict but indicate mapping was found

        # No match found
        print(f"No ERP size weight code mapping found for {style_code_str} (color: {color_code_str}) with specified criteria.")
        return {}, None, None

    def __enter__(self):
        """Context manager entry: initialize."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close connection."""
        self._close_snowflake()

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure Snowflake environment variables are set or provide config dict
    if not all([SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT]):
        print("Please set SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, and SNOWFLAKE_ACCOUNT environment variables.")
        # Or create a config dict:
        # config = {
        #     'user': 'your_user',
        #     'password': 'your_password',
        #     'account': 'your_account.region.cloud', # e.g., xy12345.us-east-1.aws
        #     'warehouse': 'YOUR_WH',
        #     'database': 'DEV_DW',
        #     'schema': 'PLANNING'
        # }
        # erp_repo = ERPSizeRepository(snowflake_config=config)
    else:
        try:
            # Use context manager for automatic connection handling
            with ERPSizeRepository() as erp_repo:
                print("\n--- Repository Initialized ---")
                # print("Mapping Keys:", list(erp_repo.style_to_weight_map.keys())[:20]) # Show some keys
                # print("Weight Keys:", list(erp_repo.weight_to_distribution_map.keys())[:20]) # Show some keys

                print("\n--- Example Lookups ---")

                # Example 1: Specific lookup
                dist, src, seas = erp_repo.get_erp_size_distribution(
                    style_code='STYLE123',
                    color_code='COLORABC',
                    source='Ecomm',
                    season='FAHO25'
                )
                print(f"Lookup 1 (Specific): Style=STYLE123, Color=COLORABC, Source=Ecomm, Season=FAHO25")
                print(f"  Source Used: {src}, Season Used: {seas}")
                print(f"  Distribution: {dist}")

                # Example 2: Less specific lookup (let it find source/season)
                dist, src, seas = erp_repo.get_erp_size_distribution(
                    style_code='STYLE456'
                )
                print(f"\nLookup 2 (Style Only): Style=STYLE456")
                print(f"  Source Used: {src}, Season Used: {seas}")
                print(f"  Distribution: {dist}")

                # Example 3: Lookup for a style that might only have Retail data
                dist, src, seas = erp_repo.get_erp_size_distribution(
                    style_code='STYLE789',
                    source='Retail' # Specify source preference
                )
                print(f"\nLookup 3 (Style + Source Hint): Style=STYLE789, Source=Retail")
                print(f"  Source Used: {src}, Season Used: {seas}")
                print(f"  Distribution: {dist}")

                # Example 4: Non-existent style
                dist, src, seas = erp_repo.get_erp_size_distribution(
                    style_code='NONEXISTENT'
                )
                print(f"\nLookup 4 (Non-existent): Style=NONEXISTENT")
                print(f"  Source Used: {src}, Season Used: {seas}")
                print(f"  Distribution: {dist}")

        except (RuntimeError, ValueError) as e:
            print(f"\nAn error occurred: {e}")
        except Exception as e:
             print(f"\nAn unexpected error occurred: {e}")