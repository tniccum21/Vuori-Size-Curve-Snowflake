#!/usr/bin/env python3
"""
Setup script for Vuori Size Curve Analyzer.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vuori-size-curve",
    version="1.0.0",
    author="Vuori Analytics Team",
    author_email="thomas.niccum@vuori.com",
    description="Size curve analysis tool for fashion retail products",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vuori/vuori-size-curve",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "snowflake-snowpark-python",
        "streamlit",
        "plotly",
    ],
    entry_points={
        "console_scripts": [
            "size-curve-cli=vuori_size_curve.cli.size_curve_cli:main",
        ],
    },
)