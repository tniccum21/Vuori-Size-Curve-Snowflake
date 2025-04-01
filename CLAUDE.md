# CLAUDE.md for Vuori Size Curve Analyzer

## Commands
- Run CLI: `python size_curve_analysis_cli.py --start-date YYYY-MM-DD --output-dir my_results --channels ECOM,STORE`
- Run Streamlit: `streamlit run size_curve_analysis_streamlit.py`
- Install dependencies: `pip install pandas numpy snowflake-snowpark-python streamlit plotly`

## Code Style Guidelines
- **Imports**: Standard imports (pandas as pd, numpy as np) at top, followed by specialized packages
- **Naming**: CamelCase for classes (SizeCurveAnalyzer), snake_case for functions/variables
- **Docstrings**: Triple quotes for class/function documentation
- **Type Hints**: Not currently used but encouraged for new code
- **Error Handling**: Use try/except with specific exceptions and meaningful error messages
- **Pandas**: Use method chaining where appropriate, prefer vectorized operations
- **Constants**: Use uppercase for constants (e.g., MIN_SALES, DEFAULT_START_DATE)
- **Spacing**: 4-space indentation, no tabs

## Project Structure
- Core analysis logic in `size_curve_analyzer.py`
- CLI interface in `size_curve_analysis_cli.py`  
- Streamlit web interface in `size_curve_analysis_streamlit.py`
- Sample data in `sales_sample.csv` and `unique_size_codes.csv`