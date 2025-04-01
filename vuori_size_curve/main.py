"""
Main entry point for the size curve analyzer application.
"""
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime

from vuori_size_curve.config.app_config import DEFAULT_START_DATE, DEFAULT_MIN_SALES, ANALYSIS_LEVELS
from vuori_size_curve.data.connectors.snowflake_connector import SnowflakeConnector
from vuori_size_curve.data.repositories.product_repository import ProductRepository
from vuori_size_curve.data.repositories.sales_repository import SalesRepository
from vuori_size_curve.data.models.sales import FilterCriteria, SizeCurve
from vuori_size_curve.analysis.analyzer_factory import AnalyzerFactory
from vuori_size_curve.analysis.exporters.csv_exporter import CSVExporter
from vuori_size_curve.utils.validation import validate_date_format, validate_min_sales
from vuori_size_curve.utils.date_helpers import get_timestamp_str
from vuori_size_curve.utils.logging_config import setup_logging


class SizeCurveAnalysisApp:
    """
    Main application class for size curve analysis.
    """
    
    def __init__(self, log_level=logging.INFO):
        """
        Initialize the application.
        
        Args:
            log_level: Logging level
        """
        # Set up logging
        self.logger = setup_logging(log_level=log_level)
        
        # Initialize database connector
        self.connector = SnowflakeConnector()
        
        # Initialize repositories
        self.product_repository = ProductRepository(self.connector)
        self.sales_repository = SalesRepository(self.connector)
        
        # Default values
        self.min_sales = DEFAULT_MIN_SALES
        self.start_date = DEFAULT_START_DATE
        self.channels_filter = None
        self.output_dir = None
    
    def configure(
        self,
        start_date: Optional[str] = None,
        min_sales: Optional[int] = None,
        channels_filter: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Configure the application.
        
        Args:
            start_date (Optional[str]): Start date for data analysis (YYYY-MM-DD)
            min_sales (Optional[int]): Minimum sales quantity threshold
            channels_filter (Optional[List[str]]): List of channels to include
            output_dir (Optional[str]): Output directory for results
        """
        # Validate and set start date
        if start_date and validate_date_format(start_date):
            self.start_date = start_date
        else:
            self.logger.warning(f"Invalid start date format: {start_date}. Using default: {DEFAULT_START_DATE}")
            self.start_date = DEFAULT_START_DATE
        
        # Validate and set min sales
        self.min_sales = validate_min_sales(min_sales) if min_sales is not None else DEFAULT_MIN_SALES
        
        # Set channels filter
        self.channels_filter = channels_filter
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            # Generate default timestamped directory
            timestamp = get_timestamp_str()
            self.output_dir = f"size_curves_{timestamp}"
    
    def run_analysis(self) -> str:
        """
        Run the complete size curve analysis process.
        
        Returns:
            str: Path to the output directory
        """
        self.logger.info(f"Starting size curve analysis with start date: {self.start_date}")
        
        try:
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(f"Created output directory: {self.output_dir}")
            
            # Create filter criteria
            filter_criteria = FilterCriteria(
                start_date=self.start_date,
                min_sales=self.min_sales,
                channels_filter=self.channels_filter
            )
            
            # Create analyzer factory
            analyzer_factory = AnalyzerFactory(
                product_repository=self.product_repository,
                sales_repository=self.sales_repository,
                min_sales=self.min_sales
            )
            
            # Get sales data with the specified filter criteria
            sales_data = self.sales_repository.get_raw_data(filter_criteria)
            
            # Get product templates
            style_templates = self.product_repository.determine_size_templates()
            
            # Get unique sales channels
            sales_channels = self.sales_repository.get_unique_channels()
            
            # Filter channels if specified
            if self.channels_filter:
                sales_channels = [ch for ch in sales_channels if ch in self.channels_filter]
            
            self.logger.debug(f"Found {len(sales_channels)} distribution channels: {', '.join(sales_channels)}")
            
            # Create exporter
            exporter = CSVExporter()
            
            # Process all channels combined
            self.logger.debug("Analyzing ALL CHANNELS combined...")
            all_channels_results = self._analyze_data(
                analyzer_factory, sales_data, style_templates
            )
            exporter.export(all_channels_results, self.output_dir)
            
            # Process each channel separately
            for channel in sales_channels:
                self.logger.debug(f"Analyzing channel: {channel}...")
                # Filter data for this channel
                channel_data = sales_data[sales_data['DISTRIBUTION_CHANNEL_CODE'] == channel].copy()
                
                # Analyze the channel data
                channel_results = self._analyze_data(
                    analyzer_factory, channel_data, style_templates, channel=channel
                )
                
                # Export results for this channel
                exporter.export(channel_results, self.output_dir, channel=channel)
            
            self.logger.info(f"Size curve analysis complete. Results saved in {self.output_dir}")
            return self.output_dir
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            raise
        finally:
            # Close the database connection
            try:
                self.connector.disconnect()
                self.logger.info("Database connection closed.")
            except Exception as e:
                self.logger.error(f"Error closing database connection: {str(e)}")
    
    def _analyze_data(
        self,
        analyzer_factory: AnalyzerFactory,
        sales_data: pd.DataFrame,
        style_templates: Dict[str, Any],
        channel: Optional[str] = None
    ) -> Dict[str, Dict[str, SizeCurve]]:
        """
        Analyze data for all levels.
        
        Args:
            analyzer_factory (AnalyzerFactory): Factory for creating analyzers
            sales_data (pd.DataFrame): Sales data to analyze
            style_templates (Dict[str, Any]): Style templates to use
            channel (Optional[str]): Channel name for filtering
        
        Returns:
            Dict[str, Dict[str, SizeCurve]]: Results organized by level and identifier
        """
        results = {}
        
        # Analyze each level
        for level in ANALYSIS_LEVELS:
            analyzer = analyzer_factory.get_analyzer(level)
            if analyzer:
                level_results = analyzer.analyze(
                    sales_data=sales_data,
                    style_templates=style_templates,
                    channel=channel
                )
                results[level] = level_results
            else:
                self.logger.warning(f"No analyzer found for level: {level}")
        
        return results


def run_analysis(
    start_date: Optional[str] = None,
    min_sales: Optional[int] = None,
    channels_filter: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    log_level: int = logging.INFO
) -> str:
    """
    Run size curve analysis with the specified parameters.
    
    Args:
        start_date (Optional[str]): Start date for data analysis (YYYY-MM-DD)
        min_sales (Optional[int]): Minimum sales quantity threshold
        channels_filter (Optional[List[str]]): List of channels to include
        output_dir (Optional[str]): Output directory for results
        log_level (int): Logging level
    
    Returns:
        str: Path to the output directory
    """
    # Create and configure the application
    app = SizeCurveAnalysisApp(log_level=log_level)
    app.configure(
        start_date=start_date,
        min_sales=min_sales,
        channels_filter=channels_filter,
        output_dir=output_dir
    )
    
    # Run the analysis
    return app.run_analysis()


if __name__ == "__main__":
    # This allows the module to be run directly for testing
    output_dir = run_analysis()
    print(f"Analysis complete. Results saved in {output_dir}")