# ERP Size Curve Visualization & Analytics Feature

This feature adds the ability to view and analyze ERP-recommended size distributions alongside actual sales data in the Color Choice Size Curve Analyzer.

## Overview

The ERP size curve feature allows users to visualize and analyze differences between actual sales patterns and ERP-recommended size distributions. This helps identify discrepancies between planned and actual size curves, enabling better inventory planning and optimization.

## How It Works

1. The feature reads ERP size curve data from two Excel files:
   - **Mapping File**: `F25 Size Weight Key_Updated for Q4.xlsx`
     - Contains both 'Ecomm' and 'Retail' worksheets
     - Maps style codes to size weight codes using these columns:
       - `Style Code`: The product style identifier
       - `Color`: The product color code (optional)
       - `Ecomm Size Weight Code`: The key that maps to the size distribution (in 'Ecomm' worksheet)
       - `Retail Size Weight Code`: The key that maps to the size distribution (in 'Retail' worksheet)
   - **Weights File**: `ERP Size weight.xlsx`
     - Contains the percentage distribution for each size weight code using:
       - `SCALECODE`: Corresponds to the Size Weight Code (from either Ecomm or Retail)
       - `SIZEVALUEID`: The size identifier (e.g., "S", "M", "L")
       - `QUANTITY`: The relative quantity for this size in the distribution

2. When viewing a color choice's size distribution, the ERP-recommended size curve is displayed alongside the actual sales size curve, making it easy to compare the two.

3. A side-by-side comparison table is provided, showing exact percentages for both the actual sales and ERP recommendations.

4. **NEW**: Comprehensive analytics dashboard for ERP size curve alignment:
   - Calculates alignment metrics for each product (Alignment Score, MAE, Max Difference)
   - Provides aggregate insights across the entire product range
   - Advanced filtering to easily identify products with poor alignment
   - Visual analytics with distribution charts and scatter plots

## Usage

Launch the Streamlit app with the ERP data files:

```bash
streamlit run size_curve_clustering.py -- --erp-mapping-file "F25 Size Weight Key_Updated for Q4.xlsx" --erp-weights-file "ERP Size weight.xlsx"
```

### Command Line Options

- `--erp-mapping-file`: Path to the mapping file (style code to size weight code)
- `--erp-weights-file`: Path to the weights file (size weight codes to distributions)
- `--cluster-size`: Number of clusters to use (default: 5, 0 for automatic determination)
- `--algo`: Clustering algorithm to use (km: KMeans, gmw: Gaussian Mixture Model, dbscan: DBSCAN)
- `--eps`: DBSCAN parameter - maximum distance for points to be considered neighbors (default: 0.5)
- `--min-samples`: DBSCAN parameter - minimum samples in a neighborhood for core point (default: 5)
- `--log-level`: Logging level (default: INFO)

## Using the ERP Analytics Dashboard

The application now includes a dedicated "ERP Size Analytics" tab that offers:

1. **Overall Metrics**:
   - Average alignment score across all products
   - Count and percentage of products with poor alignment
   - Total number of products analyzed

2. **Filtering Capabilities**:
   - Filter by alignment score range
   - Filter by gender
   - Filter by size template
   - Sort by alignment quality or sales volume

3. **Visual Analytics**:
   - Alignment score distribution histogram
   - Scatter plot showing alignment vs. sales volume
   - Highlighted tables showing the worst aligned products

## Interpreting the Results

When viewing a color choice's size distribution:

1. **Red line** represents the actual sales distribution
2. **Green dashed line** represents the ERP-recommended distribution
3. **Blue line** (or other colors) represent the cluster centroids

The system now checks both 'Ecomm' and 'Retail' worksheets in the mapping file, prioritizing matches from the 'Ecomm' worksheet first, then falling back to the 'Retail' worksheet if no match is found. Each worksheet uses its respective column for size weight codes: 'Ecomm Size Weight Code' for the Ecomm worksheet and 'Retail Size Weight Code' for the Retail worksheet.

For alignment metrics:

- **Alignment Score (0-100)**: Higher is better, shows overall match quality
  - Below 80%: Poor alignment (highlighted in red)
  - 80-95%: Good alignment
  - Above 95%: Excellent alignment (highlighted in green)
- **MAE (Mean Absolute Error)**: Lower is better, average percentage difference across sizes
- **Max Difference**: Largest percentage difference between actual and ERP for any size

Significant differences indicate a mismatch between planned and actual size distributions, which may warrant further investigation or adjustment of ERP recommendations.