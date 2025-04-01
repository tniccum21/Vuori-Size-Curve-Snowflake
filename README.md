# Vuori Size Curve Analyzer

A modular Python package for analyzing sales data and generating size curves for fashion retail products.

## Features

- Connects to Snowflake to fetch sales and product data
- Analyzes size curves at multiple levels (Style, Subclass, Class, Collection)
- Supports filtering by date range, sales channels, and other criteria
- Exports results to CSV files
- Provides both CLI and Streamlit web interface
- Optimized in-memory filtering for fast analysis without repeated database queries

## Installation

```bash
# Clone the repository
git clone https://github.com/vuori/vuori-size-curve.git
cd vuori-size-curve

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Run analysis with default settings
python size_curve_analysis_cli.py

# Run analysis with custom settings
python size_curve_analysis_cli.py --start-date 2023-01-01 --channels ECOM,RETAIL --min-sales 50 --output-dir my_results
```

### Streamlit Web Interface

```bash
# Launch the Streamlit web app
streamlit run size_curve_analysis_streamlit.py
```

## Architecture

The package follows a modular design with clear separation of concerns:

- **Data Layer**: Repository pattern for database access
- **Analysis Layer**: Specialized analyzers for different levels of analysis
- **Export Layer**: Flexible exporters for different output formats
- **UI Layer**: Command-line and web interfaces

## Configuration

Configuration is managed through the `vuori_size_curve/config` module:

- `database_config.py`: Snowflake connection settings
- `app_config.py`: Application-wide configuration

## License

MIT