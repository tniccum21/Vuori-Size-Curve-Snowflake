"""
Repository for ERP size curve data.
"""
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Any

class ERPSizeRepository:
    """
    Repository for loading and accessing ERP size curve data.
    """
    
    def __init__(self, erp_combined_file: str = None, erp_mapping_file: str = None, erp_weights_file: str = None):
        """
        Initialize the repository with paths to ERP data files.
        
        Args:
            erp_combined_file: Path to the combined file with both seasons' data
            erp_mapping_file: Path to the mapping file (style code to size weight code) - legacy support
            erp_weights_file: Path to the weights file (size weight codes to distributions) - legacy support
        """
        self.erp_combined_file = erp_combined_file
        self.erp_mapping_file = erp_mapping_file
        self.erp_weights_file = erp_weights_file
        self.style_to_weight_map = {}
        self.weight_to_distribution_map = {}
        self.is_initialized = False
        
    def initialize(self, erp_combined_file: str = None, erp_mapping_file: str = None, erp_weights_file: str = None):
        """
        Load data from ERP files.
        
        Args:
            erp_combined_file: Path to the combined file with both seasons' data
            erp_mapping_file: Path to the mapping file (style code to size weight code) - legacy support
            erp_weights_file: Path to the weights file (size weight codes to distributions) - legacy support
        """
        # Update file paths if provided
        if erp_combined_file:
            self.erp_combined_file = erp_combined_file
        if erp_mapping_file:
            self.erp_mapping_file = erp_mapping_file
        if erp_weights_file:
            self.erp_weights_file = erp_weights_file
            
        # Check if files exist based on which mode we're in
        if self.erp_combined_file:
            # Combined file mode
            if not os.path.exists(self.erp_combined_file):
                raise FileNotFoundError(f"ERP combined file not found: {self.erp_combined_file}")
                
            # Load and process mapping data from combined file
            self._load_mapping_data()
            
            # Load and process weights data from combined file
            self._load_weights_data()
        else:
            # Legacy separate files mode
            if not self.erp_mapping_file or not os.path.exists(self.erp_mapping_file):
                raise FileNotFoundError(f"ERP mapping file not found: {self.erp_mapping_file}")
            if not self.erp_weights_file or not os.path.exists(self.erp_weights_file):
                raise FileNotFoundError(f"ERP weights file not found: {self.erp_weights_file}")
                
            # Load and process mapping file
            self._load_mapping_file()
            
            # Load and process weights file
            self._load_weights_file()
        
        self.is_initialized = True
        
    def _load_mapping_file(self):
        """Load and process the style to size weight code mapping file (legacy method)."""
        try:
            # Load both Ecomm and Retail worksheets
            ecomm_df = pd.read_excel(self.erp_mapping_file, sheet_name='Ecomm')
            retail_df = pd.read_excel(self.erp_mapping_file, sheet_name='Retail')
            
            # Process Ecomm data
            self._process_mapping_data(ecomm_df, source='Ecomm')
            
            # Process Retail data
            self._process_mapping_data(retail_df, source='Retail')
                        
        except Exception as e:
            raise RuntimeError(f"Error loading ERP mapping file: {str(e)}")
            
    def _load_mapping_data(self):
        """Load and process style to size weight code mapping data from all sheets in the combined file."""
        try:
            # Get Excel file sheet names
            xl = pd.ExcelFile(self.erp_combined_file)
            sheet_names = xl.sheet_names
            
            print(f"Available sheets in combined file: {sheet_names}")
            
            # Identify sheets with mapping data
            mapping_sheets = []
            for sheet in sheet_names:
                if ('Ecomm' in sheet or 'Retail' in sheet) and 'Size Weight' not in sheet:
                    mapping_sheets.append(sheet)
            
            if not mapping_sheets:
                # Try default sheet names as fallback
                default_sheets = ['Ecomm - FAHO25', 'Retail - FAHO25', 'Ecomm - SPSU25', 'Retail - SPSU25']
                for sheet in default_sheets:
                    if sheet in sheet_names:
                        mapping_sheets.append(sheet)
            
            if not mapping_sheets:
                raise RuntimeError(f"No suitable mapping sheets found in {self.erp_combined_file}")
                
            print(f"Processing mapping sheets: {mapping_sheets}")
            
            # Process each sheet
            for sheet_name in mapping_sheets:
                df = pd.read_excel(self.erp_combined_file, sheet_name=sheet_name)
                
                # Determine source and season from sheet name
                source = 'Ecomm' if 'Ecomm' in sheet_name else 'Retail'
                
                # Extract season from sheet name
                season = None
                if 'FAHO25' in sheet_name:
                    season = 'FAHO25'
                elif 'SPSU25' in sheet_name:
                    season = 'SPSU25'
                
                # Process data with source and season
                self._process_mapping_data(df, source=source, season=season)
                
        except Exception as e:
            raise RuntimeError(f"Error loading ERP mapping data: {str(e)}")
            
    def _process_mapping_data(self, mapping_df, source, season=None):
        """
        Process mapping data from a specific worksheet.
        
        Args:
            mapping_df: DataFrame containing mapping data
            source: Source of the data ('Ecomm' or 'Retail')
            season: Optional season identifier ('FAHO25' or 'SPSU25')
        """
        # Print available columns for debugging
        print(f"Available columns in {source}{' - ' + season if season else ''} worksheet: {list(mapping_df.columns)}")
        
        # Set the appropriate weight code column based on source
        weight_code_column = 'Retail Size Weight Code' if source == 'Retail' else 'Ecomm Size Weight Code'
        
        # Check if the expected column exists
        if weight_code_column not in mapping_df.columns:
            print(f"Warning: Expected column '{weight_code_column}' not found in {source}{' - ' + season if season else ''} worksheet.")
            # Try to find a similar column name containing 'Size Weight Code'
            similar_columns = [col for col in mapping_df.columns if 'Size Weight Code' in col]
            if similar_columns:
                weight_code_column = similar_columns[0]
                print(f"Using alternative column: '{weight_code_column}'")
            else:
                print(f"No suitable alternative column found in {source}{' - ' + season if season else ''} worksheet.")
                return  # Skip processing this worksheet
        
        for _, row in mapping_df.iterrows():
            style_code = row.get('Style Code')
            weight_code = row.get(weight_code_column)
            color_code = row.get('Color')
            
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
                
                # 2. Keys with just source (backward compatibility)
                key_with_source = f"{style_code}_{source}"
                if key_with_source not in self.style_to_weight_map:
                    self.style_to_weight_map[key_with_source] = weight_code
                
                if color_code:
                    combined_key_with_source = f"{style_code}|{color_code}_{source}"
                    if combined_key_with_source not in self.style_to_weight_map:
                        self.style_to_weight_map[combined_key_with_source] = weight_code
                
                # 3. Keys without source or season (least specific, lowest priority)
                if style_code not in self.style_to_weight_map:
                    self.style_to_weight_map[style_code] = weight_code
                
                if color_code:
                    regular_combined = f"{style_code}|{color_code}"
                    if regular_combined not in self.style_to_weight_map:
                        self.style_to_weight_map[regular_combined] = weight_code
            
    def _load_weights_file(self):
        """Load and process the size weight code to distribution file (legacy method)."""
        try:
            # Load the weights file
            weights_df = pd.read_excel(self.erp_weights_file)
            self._process_weights_data(weights_df)
                
        except Exception as e:
            raise RuntimeError(f"Error loading ERP weights file: {str(e)}")
            
    def _load_weights_data(self):
        """Load and process size weight code to distribution data from the combined file."""
        try:
            # Get Excel file sheet names
            xl = pd.ExcelFile(self.erp_combined_file)
            sheet_names = xl.sheet_names
            
            # Identify sheets with weight data
            weight_sheets = []
            for sheet in sheet_names:
                if 'Size Weight' in sheet:
                    weight_sheets.append(sheet)
            
            print(f"Processing weight sheets: {weight_sheets}")
            
            if not weight_sheets:
                # If no specific weight sheets found, try a general approach
                # Look for sheets that might contain the weight data based on known patterns
                fallback_sheets = []
                for sheet in sheet_names:
                    # Add any sheets that look like they contain weight data
                    if any(term in sheet.lower() for term in ['weight', 'distribution', 'percentage']):
                        fallback_sheets.append(sheet)
                
                if fallback_sheets:
                    print(f"No specific weight sheets found. Trying fallback sheets: {fallback_sheets}")
                    for sheet_name in fallback_sheets:
                        df = pd.read_excel(self.erp_combined_file, sheet_name=sheet_name)
                        self._process_weights_data(df)
                else:
                    # As a last resort, try using the file directly
                    print("No weight sheets identified. Trying to read weights directly from the file.")
                    weights_df = pd.read_excel(self.erp_combined_file)
                    self._process_weights_data(weights_df)
            else:
                # Process each weight sheet
                for sheet_name in weight_sheets:
                    df = pd.read_excel(self.erp_combined_file, sheet_name=sheet_name)
                    
                    # Determine season from sheet name if possible
                    season = None
                    if 'FAHO25' in sheet_name:
                        season = 'FAHO25'
                    elif 'SPSU25' in sheet_name:
                        season = 'SPSU25'
                    
                    # Process weight data
                    self._process_weights_data(df, season=season)
               
        except Exception as e:
            raise RuntimeError(f"Error loading ERP weights data: {str(e)}")
            
    def _process_weights_data(self, weights_df, season=None):
        """
        Process weights data from a dataframe.
        
        Args:
            weights_df: DataFrame containing weights data
            season: Optional season identifier ('FAHO25' or 'SPSU25')
        """
        # Print available columns for debugging
        print(f"Available columns in weights data{' for ' + season if season else ''}: {list(weights_df.columns)}")
        
        # Check for required columns
        required_columns = ['SCALECODE', 'SIZEVALUEID', 'QUANTITY']
        missing_columns = [col for col in required_columns if col not in weights_df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns in weights data: {missing_columns}")
            # Try to find alternative column names
            column_mappings = {}
            for col in missing_columns:
                if col == 'SCALECODE':
                    alternatives = [c for c in weights_df.columns if any(term in c.upper() for term in ['SCALE', 'CODE', 'WEIGHT'])]
                elif col == 'SIZEVALUEID':
                    alternatives = [c for c in weights_df.columns if any(term in c.upper() for term in ['SIZE', 'VALUE', 'ID'])]
                elif col == 'QUANTITY':
                    alternatives = [c for c in weights_df.columns if any(term in c.upper() for term in ['QTY', 'QUANTITY', 'AMOUNT', 'PERCENT'])]
                
                if alternatives:
                    column_mappings[col] = alternatives[0]
                    print(f"Using '{alternatives[0]}' as alternative for '{col}'")
            
            # If we couldn't find alternatives for all missing columns, skip processing
            if len(column_mappings) < len(missing_columns):
                print(f"Could not find alternatives for all required columns. Skipping weights processing.")
                return
            
            # Rename columns to match expected names
            weights_df = weights_df.rename(columns=column_mappings)
        
        # Group by SCALECODE and create size distributions
        try:
            for scale_code, group in weights_df.groupby('SCALECODE'):
                # Create a dictionary of size -> percentage
                size_distribution = {}
                total_quantity = group['QUANTITY'].sum()
                
                if total_quantity > 0:
                    for _, row in group.iterrows():
                        size_value = row.get('SIZEVALUEID')
                        quantity = row.get('QUANTITY', 0)
                        if size_value:
                            # Convert quantity to percentage
                            percentage = (quantity / total_quantity) * 100
                            size_distribution[size_value] = percentage
                            
                # Store the distribution for this scale code
                # If season is provided, we could store with season info, but for simplicity
                # we'll just use the scale_code directly as the distributions should be the same
                # regardless of season for a given scale code
                self.weight_to_distribution_map[scale_code] = size_distribution
                
        except Exception as e:
            print(f"Error processing weights data: {str(e)}")
            # If groupby fails, try a different approach based on the specific file structure
            
    def get_erp_size_distribution(self, style_code: str, color_code: Optional[str] = None, 
                               source: Optional[str] = None, season: Optional[str] = None) -> Tuple[Dict[str, float], Optional[str], Optional[str]]:
        """
        Get the ERP-recommended size distribution for a style or color choice.
        
        Args:
            style_code: The style code to look up
            color_code: Optional color code for more specific lookup
            source: Optional data source ('Ecomm' or 'Retail'). If None, tries both.
            season: Optional season ('FAHO25' or 'SPSU25'). If None, tries both.
            
        Returns:
            Tuple of (distribution_dict, source_used, season_used), where:
            - distribution_dict: Dictionary mapping size codes to percentage values
            - source_used: String indicating which source was used ('Ecomm', 'Retail', or None if no match)
            - season_used: String indicating which season was used ('FAHO25', 'SPSU25', or None if no match)
        """
        if not self.is_initialized:
            raise RuntimeError("ERPSizeRepository not initialized. Call initialize() first.")
        
        size_weight_code = None
        source_used = None
        season_used = None
        
        # Determine sources to try
        if source:
            sources_to_try = [source]
        else:
            # Otherwise, try Ecomm first, then Retail
            sources_to_try = ['Ecomm', 'Retail']
            
        # Determine seasons to try
        if season:
            seasons_to_try = [season]
        else:
            # Try both seasons, preferring FAHO25 (newer) first
            seasons_to_try = ['FAHO25', 'SPSU25']
        
        # Strategy 1: Try most specific keys first (with season and source)
        if color_code:
            # Try with color code, season, and source (most specific)
            for try_season in seasons_to_try:
                if size_weight_code:
                    break
                for try_source in sources_to_try:
                    combined_key = f"{style_code}|{color_code}_{try_source}_{try_season}"
                    if combined_key in self.style_to_weight_map:
                        size_weight_code = self.style_to_weight_map[combined_key]
                        source_used = try_source
                        season_used = try_season
                        break
        
        # Try style code with season and source if no match yet
        if not size_weight_code:
            for try_season in seasons_to_try:
                if size_weight_code:
                    break
                for try_source in sources_to_try:
                    key = f"{style_code}_{try_source}_{try_season}"
                    if key in self.style_to_weight_map:
                        size_weight_code = self.style_to_weight_map[key]
                        source_used = try_source
                        season_used = try_season
                        break
        
        # Strategy 2: Try source-specific keys (without season) for backward compatibility
        if not size_weight_code:
            # Try with color code and source
            if color_code:
                for try_source in sources_to_try:
                    combined_key = f"{style_code}|{color_code}_{try_source}"
                    if combined_key in self.style_to_weight_map:
                        size_weight_code = self.style_to_weight_map[combined_key]
                        source_used = try_source
                        season_used = 'Unknown'
                        break
            
            # Try style code with source only
            if not size_weight_code:
                for try_source in sources_to_try:
                    key = f"{style_code}_{try_source}"
                    if key in self.style_to_weight_map:
                        size_weight_code = self.style_to_weight_map[key]
                        source_used = try_source
                        season_used = 'Unknown'
                        break
        
        # Strategy 3: As a last resort, try the non-source-specific keys
        if not size_weight_code:
            # Try style+color
            if color_code:
                combined_key = f"{style_code}|{color_code}"
                if combined_key in self.style_to_weight_map:
                    size_weight_code = self.style_to_weight_map[combined_key]
                    source_used = 'Unknown'
                    season_used = 'Unknown'
            
            # Try style only
            if not size_weight_code and style_code in self.style_to_weight_map:
                size_weight_code = self.style_to_weight_map[style_code]
                source_used = 'Unknown'
                season_used = 'Unknown'
            
        # If we found a size weight code, look up its distribution
        if size_weight_code:
            distribution = self.weight_to_distribution_map.get(size_weight_code, {})
            print(f"Found ERP size distribution for {style_code} (color: {color_code}) using source: {source_used}, season: {season_used}, weight code: {size_weight_code}")
            return distribution, source_used, season_used
            
        # No match found
        print(f"No ERP size distribution found for {style_code} (color: {color_code})")
        return {}, None, None