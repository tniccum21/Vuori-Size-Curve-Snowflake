"""
Command-line interface for size curve analysis.
"""
import argparse
import logging
import sys
from typing import List, Optional
from vuori_size_curve.main import run_analysis
from vuori_size_curve.utils.validation import validate_date_format
from vuori_size_curve.config.app_config import DEFAULT_START_DATE


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args (Optional[List[str]]): Command-line arguments (uses sys.argv if None)
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Vuori Size Curve Analyzer - Generate size curves from sales data"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Start date for sales data analysis (YYYY-MM-DD, default: {DEFAULT_START_DATE})"
    )
    
    parser.add_argument(
        "--min-sales",
        type=int,
        default=0,
        help="Minimum sales quantity threshold for inclusion in analysis (default: 0)"
    )
    
    parser.add_argument(
        "--channels",
        type=str,
        help="Comma-separated list of distribution channels to include (default: all channels)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: auto-generated based on timestamp)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Parse arguments
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args (Optional[List[str]]): Command-line arguments (uses sys.argv if None)
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    # Parse arguments
    parsed_args = parse_args(args)
    
    # Set log level based on verbosity
    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    
    # Process channels if specified
    channels_filter = None
    if parsed_args.channels:
        channels_filter = [ch.strip() for ch in parsed_args.channels.split(',')]
    
    # Validate start date
    if not validate_date_format(parsed_args.start_date):
        print(f"Error: Invalid start date format: {parsed_args.start_date}. Use YYYY-MM-DD format.")
        return 1
    
    try:
        # Run the analysis
        output_dir = run_analysis(
            start_date=parsed_args.start_date,
            min_sales=parsed_args.min_sales,
            channels_filter=channels_filter,
            output_dir=parsed_args.output_dir,
            log_level=log_level
        )
        
        print(f"\nSize curve analysis complete. Results saved in {output_dir}")
        return 0
    
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())