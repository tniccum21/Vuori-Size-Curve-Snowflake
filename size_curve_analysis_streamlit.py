#!/usr/bin/env python3
"""
Streamlit entry point for Size Curve Analyzer.
"""
import streamlit as st
import sys
import os

# This is a simple wrapper script that just runs the Streamlit app directly
# rather than spawning additional processes

if __name__ == "__main__":
    # Get absolute path to the app file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "vuori_size_curve", "ui", "streamlit_app.py")
    
    # Add the current directory to Python path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Import and run the app directly
    import runpy
    runpy.run_path(app_path, run_name="__main__")