"""
Streamlit app for color choice size curve analysis and clustering.
This application extends the original size curve analyzer to the COLOR_CHOICE level
(STYLE_CODE + COLOR_CODE) and implements clustering based on size distributions.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import os
import sys
import logging
from datetime import datetime
import traceback
import argparse

# Import for ERP Size Repository
try:
    from vuori_size_curve.data.repositories.erp_repository import ERPSizeRepository
except ImportError:
    # This will be handled in the application initialization
    pass

# Add parent directory to path to import existing modules
parent_dir = os.path.abspath(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from existing modules
from vuori_size_curve.data.connectors.snowflake_connector import SnowflakeConnector
from vuori_size_curve.data.repositories.product_repository import ProductRepository
from vuori_size_curve.data.repositories.sales_repository import SalesRepository
from vuori_size_curve.data.models.sales import FilterCriteria, SizeCurve
from vuori_size_curve.config.app_config import ANALYSIS_LEVELS
from vuori_size_curve.utils.logging_config import setup_logging


# --- Data Models ---

@dataclass
class ColorChoiceSizeCurve:
    """
    Represents a size curve at the COLOR_CHOICE level (STYLE_CODE + COLOR_CODE).
    """
    style_code: str
    color_code: str
    gender_code: str
    size_template_name: str
    total_sales: int
    size_percentages: Dict[str, float]
    cluster_id: Optional[int] = None
    p_value: Optional[float] = None
    distance_from_centroid: Optional[float] = None
    is_outlier: bool = False
    # ERP comparison metrics
    erp_percentages: Optional[Dict[str, float]] = None
    erp_mae: Optional[float] = None  # Mean Absolute Error between actual and ERP
    erp_max_diff: Optional[float] = None  # Maximum difference between actual and ERP
    erp_alignment_score: Optional[float] = None  # Overall alignment score (0-100)
    erp_source: Optional[str] = None  # Source of ERP data ('Ecomm', 'Retail', or 'Unknown')


@dataclass
class TemplateGroup:
    """
    Represents a group of COLOR_CHOICEs with the same SIZE_TEMPLATE.
    """
    template_name: str
    size_codes: List[str]
    color_choices: List[ColorChoiceSizeCurve]
    clusters: Optional[Dict[int, List[ColorChoiceSizeCurve]]] = None
    outliers: Optional[List[ColorChoiceSizeCurve]] = None
    optimal_k: Optional[int] = None
    centroids: Optional[np.ndarray] = None
    # GMM specific fields
    component_weights: Optional[np.ndarray] = None
    component_covariances: Optional[np.ndarray] = None
    sample_probs: Optional[np.ndarray] = None
    # DBSCAN specific fields
    core_sample_indices: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None


# --- Analyzer Implementation ---

class ColorChoiceAnalyzer:
    """
    Analyzer for COLOR_CHOICE level size curves and clustering.
    """
    
    def __init__(
        self,
        product_repository: ProductRepository,
        sales_repository: SalesRepository,
        min_sales: int = 10,
        cluster_size: int = 5,
        algorithm: str = "km",
        eps: float = 0.5,
        min_samples: int = 5
    ):
        """
        Initialize the analyzer.
        
        Args:
            product_repository: Repository for product data
            sales_repository: Repository for sales data
            min_sales: Minimum sales quantity threshold (default: 10)
            cluster_size: Number of clusters to use (default: 5)
            algorithm: Clustering algorithm to use (km: KMeans, gmm: Gaussian Mixture Model, dbscan: DBSCAN)
            eps: DBSCAN parameter - maximum distance for points to be considered neighbors (default: 0.5)
            min_samples: DBSCAN parameter - minimum samples in neighborhood for core point (default: 5)
        """
        self.product_repository = product_repository
        self.sales_repository = sales_repository
        self.min_sales = min_sales
        self.cluster_size = cluster_size
        self.algorithm = algorithm
        self.eps = eps
        self.min_samples = min_samples
        self.logger = logging.getLogger(__name__)
    
    def analyze(
        self,
        sales_data: pd.DataFrame,
        style_templates: Dict[str, Any],
        **kwargs
    ) -> Dict[str, ColorChoiceSizeCurve]:
        """
        Analyze sales data to generate COLOR_CHOICE level size curves.
        
        Args:
            sales_data: The sales data to analyze
            style_templates: Style templates to use
            **kwargs: Additional arguments
        
        Returns:
            Dictionary mapping COLOR_CHOICE identifiers to size curves
        """
        self.logger.info("Calculating COLOR_CHOICE level size curves...")
        result = {}
        
        # Filter out invalid size data and ONS (One-size-fits-all)
        filtered_sales_data = sales_data[
            (~sales_data['SIZE_CODE'].str.contains('_N/A', na=False)) & 
            (sales_data['SIZE_ORDER'] > 0) & 
            (sales_data['SIZE_ORDER'].notnull()) &
            (sales_data['SIZE_CODE'] != 'ONS')  # Exclude ONS (One-size-fits-all)
        ].copy()
        
        # Apply channel filter if provided
        channel = kwargs.get('channel')
        if channel and channel != 'ALL' and 'DISTRIBUTION_CHANNEL_CODE' in filtered_sales_data.columns:
            filtered_sales_data = filtered_sales_data[
                filtered_sales_data['DISTRIBUTION_CHANNEL_CODE'] == channel
            ]
        
        # Group by style code, color code, and gender
        grouped_data = filtered_sales_data.groupby(['STYLE_CODE', 'COLOR_CODE', 'GENDER_CODE'])
        
        for (style_code, color_code, gender_code), color_choice_group in grouped_data:
            # Skip groups with insufficient sales
            total_sales = color_choice_group['TOTAL_SALES'].sum()
            if total_sales < self.min_sales:
                continue
            
            # Determine the template to use
            template_sizes = []
            template_name = ""
            
            # Create style+color key for template lookup
            style_color_key = f"{style_code}_{color_code}"
            
            # Try to get template from product repository
            if hasattr(self.product_repository, 'style_color_templates') and style_color_key in self.product_repository.style_color_templates:
                template = self.product_repository.style_color_templates[style_color_key].size_template
                template_sizes = template.sizes
                template_name = template.name
            elif style_code in style_templates:
                # Fall back to style-level template
                template = style_templates[style_code].size_template
                template_sizes = template.sizes
                template_name = template.name
            else:
                # Otherwise, determine sizes from this group and sort by SIZE_ORDER
                sizes_with_order = color_choice_group[['SIZE_CODE', 'SIZE_ORDER']].drop_duplicates()
                # Filter out any problematic SIZE_ORDER values
                sizes_with_order = sizes_with_order[
                    (sizes_with_order['SIZE_ORDER'] > 0) & 
                    (sizes_with_order['SIZE_ORDER'].notnull())
                ]
                # Ensure we're getting unique sizes only
                sizes_with_order = sizes_with_order.drop_duplicates('SIZE_CODE')
                if sizes_with_order.empty:
                    continue  # Skip if no valid sizes
                
                sizes_sorted = sizes_with_order.sort_values('SIZE_ORDER')
                template_sizes = sizes_sorted['SIZE_CODE'].tolist()
                template_name = "-".join(template_sizes)
            
            # Skip if no valid template sizes
            if not template_sizes:
                continue
                
            # Calculate size percentages
            size_percentages = {}
            
            # Group by size and calculate percentage
            for size_code, size_group in color_choice_group.groupby('SIZE_CODE'):
                size_sales = size_group['TOTAL_SALES'].sum()
                size_percentages[size_code] = (size_sales / total_sales) * 100
            
            # Fill in zeros for missing sizes in the template
            for size in template_sizes:
                if size not in size_percentages:
                    size_percentages[size] = 0.0
            
            # Create a composite key and size curve object
            composite_key = f"{style_code}|{color_code}|{gender_code}"
            
            # Create a ColorChoiceSizeCurve object
            size_curve = ColorChoiceSizeCurve(
                style_code=style_code,
                color_code=color_code,
                gender_code=gender_code,
                size_template_name=template_name,
                total_sales=total_sales,
                size_percentages=size_percentages
            )
            
            result[composite_key] = size_curve
        
        self.logger.info(f"Generated {len(result)} COLOR_CHOICE level size curves.")
        return result
    
    def group_by_template(
        self,
        color_choice_curves: Dict[str, ColorChoiceSizeCurve]
    ) -> Dict[str, TemplateGroup]:
        """
        Group COLOR_CHOICEs by size template.
        
        Args:
            color_choice_curves: Dictionary of COLOR_CHOICE size curves
        
        Returns:
            Dictionary mapping template names to TemplateGroup objects
        """
        template_groups = {}
        
        # Extract SIZE_ORDER information from the first curve's template
        size_order_map = {}
        
        for key, curve in color_choice_curves.items():
            template_name = curve.size_template_name
            
            # Create new template group if not exists
            if template_name not in template_groups:
                # Get the size codes from this curve
                size_percentages = curve.size_percentages
                
                # Extract sizes and their order from the template name
                # Template name format is typically "S-M-L-XL"
                template_sizes = template_name.split('-')
                
                # Sort sizes based on their position in the template (implicit order)
                # This ensures sizes like S, M, L, XL are in the correct order
                # rather than alphabetical (L, M, S, XL)
                size_order = {size: idx for idx, size in enumerate(template_sizes)}
                
                # Store in size_order_map for future reference
                size_order_map[template_name] = size_order
                
                # Sort the size codes by their order
                sorted_sizes = sorted(size_percentages.keys(), key=lambda x: size_order.get(x, 999))
                
                template_groups[template_name] = TemplateGroup(
                    template_name=template_name,
                    size_codes=sorted_sizes,
                    color_choices=[]
                )
            
            # Add this curve to the template group
            template_groups[template_name].color_choices.append(curve)
        
        self.logger.info(f"Created {len(template_groups)} template groups.")
        return template_groups
    
    def cluster_template_group(
        self, 
        template_group: TemplateGroup,
        max_k: int = 10
    ) -> TemplateGroup:
        """
        Perform clustering on a template group using the selected algorithm.
        
        Args:
            template_group: The template group to cluster
            max_k: Maximum number of clusters to consider
        
        Returns:
            Updated template group with clustering results
        """
        # Extract features (size percentages) for clustering
        color_choices = template_group.color_choices
        size_codes = template_group.size_codes
        
        # Create feature matrix
        X = np.zeros((len(color_choices), len(size_codes)))
        for i, curve in enumerate(color_choices):
            for j, size in enumerate(size_codes):
                X[i, j] = curve.size_percentages.get(size, 0.0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate silhouette scores for visualization purposes
        silhouette_scores = {}
        k_values = range(2, min(max_k + 1, len(color_choices)))
        
        for k in k_values:
            # Skip if we don't have enough samples for k clusters
            if len(color_choices) <= k:
                continue
            
            # Use the selected algorithm
            if self.algorithm == "gmw":
                model = GaussianMixture(n_components=k, random_state=42, n_init=10)
                model.fit(X_scaled)
                cluster_labels = model.predict(X_scaled)
            elif self.algorithm == "dbscan":
                # DBSCAN doesn't need a predefined k, but we'll compute it for different eps values
                # using the same k for consistency in silhouette score calculation
                model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
                cluster_labels = model.fit_predict(X_scaled)
                
                # Handle noise points (labeled as -1) for silhouette score
                if -1 in cluster_labels:
                    # Skip if all points are noise or if only one cluster is found
                    unique_labels = np.unique(cluster_labels)
                    if len(unique_labels) <= 2 and -1 in unique_labels:
                        continue
            else:  # default to KMeans
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = model.fit_predict(X_scaled)
            
            # If we have at least 2 clusters with data
            if len(np.unique(cluster_labels)) >= 2:
                # For DBSCAN, remove noise points (labeled as -1) before calculating silhouette score
                if self.algorithm == "dbscan" and -1 in cluster_labels:
                    mask = cluster_labels != -1
                    if len(np.unique(cluster_labels[mask])) >= 2 and sum(mask) >= 2:
                        score = silhouette_score(X_scaled[mask], cluster_labels[mask])
                        silhouette_scores[k] = score
                else:
                    score = silhouette_score(X_scaled, cluster_labels)
                    silhouette_scores[k] = score
        
        # If cluster_size is 0, determine optimal K using silhouette scores
        if self.cluster_size == 0 and silhouette_scores:
            # Find k with the best silhouette score
            optimal_k = max(silhouette_scores.items(), key=lambda x: x[1])[0]
            template_group.optimal_k = optimal_k
            self.logger.info(f"Automatically determined optimal k = {optimal_k} for template {template_group.template_name}")
        else:
            # Use the configured cluster size
            forced_k = self.cluster_size
            
            # Must have at least enough samples for the forced number of clusters
            min_samples_required = forced_k * 2  # Need at least 2 samples per cluster
            
            if len(color_choices) >= min_samples_required:
                template_group.optimal_k = forced_k
            else:
                # Not enough samples for clustering with configured clusters
                # Calculate how many clusters we can support
                max_possible_k = len(color_choices) // 2
                template_group.optimal_k = min(forced_k, max(1, max_possible_k))
        
        # Store silhouette scores for visualization
        template_group.silhouette_scores = silhouette_scores
        
        # Perform clustering with optimal K
        if template_group.optimal_k > 1:
            # Use the selected algorithm
            if self.algorithm == "gmw":
                model = GaussianMixture(n_components=template_group.optimal_k, random_state=42, n_init=10)
                model.fit(X_scaled)
                cluster_labels = model.predict(X_scaled)
                
                # For GMM, get the component means as centroids
                centroids_scaled = model.means_
                
                # Store probability information for GMM
                template_group.component_weights = model.weights_
                template_group.component_covariances = model.covariances_
                template_group.sample_probs = model.predict_proba(X_scaled)
            elif self.algorithm == "dbscan":
                # DBSCAN doesn't use a predefined number of clusters
                model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
                cluster_labels = model.fit_predict(X_scaled)
                
                # Update the number of clusters to the actual number found by DBSCAN
                # DBSCAN assigns -1 to noise points, so we count only positive labels
                unique_labels = np.unique(cluster_labels)
                actual_clusters = unique_labels[unique_labels >= 0]
                num_clusters = len(actual_clusters)
                
                # Update optimal_k to the actual number of clusters found
                template_group.optimal_k = num_clusters
                
                # Calculate centroids manually as mean of points in each cluster
                centroids_scaled = np.zeros((num_clusters, X_scaled.shape[1]))
                for i, cluster_id in enumerate(actual_clusters):
                    mask = cluster_labels == cluster_id
                    if np.any(mask):
                        centroids_scaled[i] = X_scaled[mask].mean(axis=0)
                
                # Store DBSCAN-specific information
                template_group.core_sample_indices = model.core_sample_indices_
                template_group.components = model.components_
                
                # Remap cluster labels to be consecutive starting from 0
                # This is important since DBSCAN may skip cluster numbers
                label_map = {label: i for i, label in enumerate(actual_clusters)}
                remapped_labels = np.array([label_map.get(label, -1) for label in cluster_labels])
                cluster_labels = remapped_labels
            else:  # default to KMeans
                model = KMeans(n_clusters=template_group.optimal_k, random_state=42, n_init=10)
                cluster_labels = model.fit_predict(X_scaled)
                
                # For KMeans, use the cluster centers
                centroids_scaled = model.cluster_centers_
            
            # Transform standardized centroids back to original percentage scale
            # This ensures the heatmap shows actual percentages
            centroids_original = scaler.inverse_transform(centroids_scaled)
            
            # Store both versions
            template_group.centroids_scaled = centroids_scaled  # For calculations
            template_group.centroids = centroids_original       # For visualization
            
            # Assign cluster IDs and calculate distances
            clusters = {i: [] for i in range(template_group.optimal_k)}
            outliers = []
            
            for i, curve in enumerate(color_choices):
                cluster_id = cluster_labels[i]
                
                # Handle DBSCAN noise points (labeled as -1)
                if cluster_id == -1:
                    # For DBSCAN, noise points are already outliers
                    curve.cluster_id = -1
                    curve.is_outlier = True
                    curve.distance_from_centroid = 0  # No centroid for noise points
                    outliers.append(curve)
                    continue
                
                # Calculate distance from centroid for regular clusters
                distance = np.linalg.norm(X_scaled[i] - centroids_scaled[cluster_id])
                curve.cluster_id = cluster_id
                curve.distance_from_centroid = distance
                
                # For non-DBSCAN algorithms or DBSCAN regular clusters:
                # Detect outliers (using a threshold based on the distribution of distances)
                if self.algorithm != "dbscan":
                    # Only do standard outlier detection for KMeans and GMM
                    # When calculating distances, make sure to handle possible -1 labels
                    valid_mask = cluster_labels >= 0
                    if np.all(valid_mask):  # If all points have valid clusters
                        distances = np.linalg.norm(X_scaled - centroids_scaled[cluster_labels], axis=1)
                    else:
                        # Only calculate distances for points with valid cluster assignments
                        distances = np.zeros(len(cluster_labels))
                        distances[valid_mask] = np.linalg.norm(
                            X_scaled[valid_mask] - centroids_scaled[cluster_labels[valid_mask]], axis=1
                        )
                    
                    threshold = np.mean(distances[distances > 0]) + 2.5 * np.std(distances[distances > 0])
                    
                    if distance > threshold:
                        curve.is_outlier = True
                        outliers.append(curve)
                        continue
                
                # Add to appropriate cluster
                clusters[cluster_id].append(curve)
            
            template_group.clusters = clusters
            template_group.outliers = outliers
        else:
            # Not enough data for meaningful clustering
            template_group.clusters = {0: template_group.color_choices}
            template_group.outliers = []
            
            # For the non-clustered case, we just use the mean as the centroid
            # Store both the standardized and original versions
            if len(X) > 0:  # Make sure we have at least one data point
                mean_vector = np.mean(X, axis=0).reshape(1, -1)
                template_group.centroids = mean_vector
                
                # If we've standardized the data, create a standardized version too
                if 'scaler' in locals() and X.shape[0] > 0:
                    template_group.centroids_scaled = scaler.transform(mean_vector)
                else:
                    template_group.centroids_scaled = mean_vector
            else:
                # If no data, create empty centroids
                template_group.centroids = np.zeros((1, len(size_codes)))
                template_group.centroids_scaled = np.zeros((1, len(size_codes)))
        
        return template_group
    
    def prepare_output_dataframe(
        self,
        template_groups: Dict[str, TemplateGroup]
    ) -> pd.DataFrame:
        """
        Prepare a DataFrame from the template groups for display and export.
        
        Args:
            template_groups: Dictionary of template groups
        
        Returns:
            DataFrame with COLOR_CHOICE level data and clustering results
        """
        rows = []
        
        for template_name, group in template_groups.items():
            for curve in group.color_choices:
                row = {
                    'STYLE_CODE': curve.style_code,
                    'COLOR_CODE': curve.color_code,
                    'GENDER_CODE': curve.gender_code,
                    'SIZE_TEMPLATE': curve.size_template_name,
                    'TOTAL_SALES': curve.total_sales,
                    'CLUSTER_ID': curve.cluster_id if curve.cluster_id is not None else -1,
                    'DISTANCE_FROM_CENTROID': curve.distance_from_centroid if curve.distance_from_centroid is not None else -1,
                    'IS_OUTLIER': curve.is_outlier
                }
                
                # Add size percentages
                for size, percentage in curve.size_percentages.items():
                    row[f'SIZE_{size}'] = percentage
                
                rows.append(row)
        
        return pd.DataFrame(rows)


# --- Visualization Functions ---

def plot_silhouette_scores(template_group: TemplateGroup) -> plt.Figure:
    """
    Plot silhouette scores for different values of K.
    
    Args:
        template_group: Template group with silhouette scores
    
    Returns:
        Matplotlib figure
    """
    if not hasattr(template_group, 'silhouette_scores') or not template_group.silhouette_scores:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data for clustering", ha='center', va='center')
        return fig
    
    k_values = list(template_group.silhouette_scores.keys())
    scores = list(template_group.silhouette_scores.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, scores, marker='o')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score for Different Values of k')
    ax.grid(True)
    
    # Mark optimal k
    optimal_k = template_group.optimal_k
    optimal_score = template_group.silhouette_scores.get(optimal_k, 0)
    
    # Add a note if using auto-determined optimal k
    title_addition = ""
    if optimal_k in template_group.silhouette_scores:
        ax.scatter([optimal_k], [optimal_score], color='red', s=100, zorder=5)
        
        # Find the best silhouette score
        best_k = max(template_group.silhouette_scores.items(), key=lambda x: x[1])[0]
        if best_k == optimal_k:
            title_addition = " (Auto-determined optimal)"
            ax.annotate(f'Auto-determined k={optimal_k}', 
                      xy=(optimal_k, optimal_score),
                      xytext=(10, -20),
                      textcoords='offset points',
                      arrowprops=dict(arrowstyle='->', color='red'))
        else:
            ax.annotate(f'Selected k={optimal_k}', 
                      xy=(optimal_k, optimal_score),
                      xytext=(10, -20),
                      textcoords='offset points',
                      arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_title(f'Silhouette Score for Different Values of k{title_addition}')
    
    return fig


def plot_cluster_pca(template_group: TemplateGroup, mode: str = '2d') -> Union[plt.Figure, go.Figure]:
    """
    Create PCA plot of clusters, either in 2D or 3D.
    
    Args:
        template_group: Template group with clustering results
        mode: Visualization mode, either '2d' (matplotlib) or '3d' (plotly interactive)
    
    Returns:
        Matplotlib figure (2D) or Plotly figure (3D)
    """
    # Extract features (size percentages) for visualization
    color_choices = template_group.color_choices
    size_codes = template_group.size_codes
    
    # Create feature matrix
    X = np.zeros((len(color_choices), len(size_codes)))
    for i, curve in enumerate(color_choices):
        for j, size in enumerate(size_codes):
            X[i, j] = curve.size_percentages.get(size, 0.0)
    
    # Get cluster assignments
    cluster_ids = [curve.cluster_id if curve.cluster_id is not None else -1 for curve in color_choices]
    is_outlier = [curve.is_outlier for curve in color_choices]
    
    # Create style and color info for better tooltips
    style_codes = [curve.style_code for curve in color_choices]
    color_codes = [curve.color_code for curve in color_choices]
    sales_values = [curve.total_sales for curve in color_choices]
    
    # Create labels for points
    hover_labels = [f"{style} - {color} (Sales: {sales})" 
                   for style, color, sales in zip(style_codes, color_codes, sales_values)]
    
    if mode == '3d':
        # Apply PCA with 3 components for 3D visualization
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        
        # Create a DataFrame for Plotly
        df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'PC3': X_pca[:, 2],
            'Cluster': [f'Cluster {c}' if c != -1 and not o else 'Outlier' 
                       for c, o in zip(cluster_ids, is_outlier)],
            'Style Code': style_codes,
            'Color Code': color_codes,
            'Sales': sales_values,
            'Is Outlier': ['Yes' if o else 'No' for o in is_outlier],
            'Label': hover_labels
        })
        
        # Create interactive 3D scatter plot
        fig = px.scatter_3d(
            df, 
            x='PC1', 
            y='PC2', 
            z='PC3',
            color='Cluster',
            hover_name='Label',
            hover_data={
                'PC1': False,
                'PC2': False,
                'PC3': False,
                'Cluster': True,
                'Style Code': True, 
                'Color Code': True,
                'Sales': True,
                'Is Outlier': True,
                'Label': False
            },
            title=f'3D PCA of Size Curves by Cluster - {template_group.template_name}',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)'
            },
            opacity=0.7
        )
        
        # Add centroids if available
        if template_group.centroids is not None and template_group.optimal_k > 1:
            centroids_pca = pca.transform(template_group.centroids)
            
            # Create a DataFrame for centroids
            centroid_df = pd.DataFrame({
                'PC1': centroids_pca[:, 0],
                'PC2': centroids_pca[:, 1],
                'PC3': centroids_pca[:, 2],
                'Cluster': [f'Centroid {i}' for i in range(len(centroids_pca))]
            })
            
            # Add centroids to the plot
            fig.add_trace(
                go.Scatter3d(
                    x=centroid_df['PC1'],
                    y=centroid_df['PC2'],
                    z=centroid_df['PC3'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        symbol='diamond',
                        color='black',
                        line=dict(width=2, color='white')
                    ),
                    name='Centroids',
                    text=[f'Centroid {i}' for i in range(len(centroids_pca))],
                    hoverinfo='text'
                )
            )
        
        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
            ),
            legend_title_text='Cluster Groups',
            margin=dict(l=0, r=0, b=0, t=40),
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
        
        return fig
        
    else:  # Default 2D visualization
        # Apply PCA with 2 components
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot non-outliers
        for cluster_id in set(cluster_ids):
            if cluster_id != -1:
                mask = [(c == cluster_id) and not o for c, o in zip(cluster_ids, is_outlier)]
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster_id}', alpha=0.7)
        
        # Plot outliers
        outlier_mask = [o for o in is_outlier]
        if any(outlier_mask):
            ax.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1], marker='x', color='red', label='Outliers', alpha=0.7)
        
        # Plot centroids if available
        if template_group.centroids is not None and template_group.optimal_k > 1:
            centroids_pca = pca.transform(template_group.centroids)
            ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='*', s=300, color='black', label='Centroids')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title(f'PCA of Size Curves by Cluster - {template_group.template_name}')
        ax.legend()
        ax.grid(True)
        
        return fig


def plot_radar_chart(template_group: TemplateGroup) -> go.Figure:
    """
    Create a radar chart comparing cluster centroids.
    
    Args:
        template_group: Template group with clustering results
    
    Returns:
        Plotly figure
    """
    if template_group.centroids is None or template_group.optimal_k <= 1:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for radar chart",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    size_codes = template_group.size_codes
    
    # Add traces for each cluster centroid
    for i in range(template_group.optimal_k):
        values = template_group.centroids[i].tolist()
        # Close the radar plot by repeating the first value
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=size_codes + [size_codes[0]],  # Close the loop
            name=f'Cluster {i}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            )
        ),
        title=f"Size Distribution by Cluster - {template_group.template_name}",
        showlegend=True
    )
    
    return fig


def create_cluster_histograms(template_group: TemplateGroup) -> plt.Figure:
    """
    Create a line chart showing size distributions for all clusters on a single plot.
    
    Args:
        template_group: Template group with clustering results
    
    Returns:
        Matplotlib figure
    """
    if template_group.centroids is None or template_group.optimal_k <= 1:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data for cluster visualization", ha='center', va='center')
        return fig
    
    # Get data for plotting
    size_codes = template_group.size_codes
    centroids = template_group.centroids
    num_clusters = len(centroids)
    
    # Create a single figure for all clusters
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up x-axis positions
    x = np.arange(len(size_codes))
    
    # Plot line for each cluster
    for i in range(num_clusters):
        cluster_data = centroids[i]
        cluster_size = len(template_group.clusters[i]) if i in template_group.clusters else 0
        
        # Plot as a line with markers
        line = ax.plot(
            x, 
            cluster_data, 
            marker='o',
            markersize=8,
            linewidth=2.5,
            label=f'Cluster {i} (n={cluster_size})',
            alpha=0.8
        )[0]
        
        # Get the line color for annotations
        line_color = line.get_color()
        
        # Add value labels for each point
        for j, value in enumerate(cluster_data):
            if value > 3.0:  # Only show labels for values > 3% to avoid clutter
                ax.annotate(
                    f'{value:.1f}%',
                    xy=(x[j], value),
                    xytext=(0, 10 if i % 2 == 0 else -15),  # Alternate offset to avoid overlap
                    textcoords="offset points",
                    ha='center',
                    va='bottom' if i % 2 == 0 else 'top',
                    color=line_color,
                    fontsize=9,
                    fontweight='bold'
                )
    
    # Add title and labels
    ax.set_title('Size Distributions by Cluster', fontsize=14)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Size', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(size_codes)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add legend with cluster sizes
    ax.legend(loc='best', fontsize=10)
    
    # Add some padding to the y-axis
    y_max = np.max(centroids)
    ax.set_ylim(0, y_max * 1.15)
    
    plt.tight_layout()
    return fig


def plot_size_distribution(color_choice: ColorChoiceSizeCurve, template_group: TemplateGroup, erp_repository=None, algorithm=None) -> plt.Figure:
    """
    Plot size distribution for a single COLOR_CHOICE vs. all cluster centroids.
    Also includes ERP recommended distribution if available.
    Uses line graphs instead of bar charts for better comparison.
    
    Args:
        color_choice: COLOR_CHOICE size curve
        template_group: Template group containing the COLOR_CHOICE
        erp_repository: Optional ERP repository to get recommended distribution
        algorithm: Optional algorithm to use for centroids ('km', 'gmw', 'dbscan')
    
    Returns:
        Matplotlib figure
    """
    size_codes = template_group.size_codes
    
    # Get COLOR_CHOICE data
    color_choice_data = [color_choice.size_percentages.get(size, 0.0) for size in size_codes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(size_codes))
    
    # Check if we have ERP data
    has_erp_data = erp_repository is not None and erp_repository.is_initialized
    erp_distribution = {}
    erp_source = None
    
    if has_erp_data:
        # Get season from session state if it exists
        selected_season = st.session_state.get('selected_season', None)
        
        # Get ERP recommended distribution
        erp_distribution, erp_source, erp_season = erp_repository.get_erp_size_distribution(
            color_choice.style_code, 
            color_choice.color_code,
            season=selected_season
        )
        if erp_source:  # Store source info if available
            color_choice.erp_source = erp_source
    
    # Plot COLOR_CHOICE data with emphasized line
    ax.plot(
        x, 
        color_choice_data,
        marker='o',
        markersize=8,
        linewidth=2.5,
        label=f'{color_choice.style_code} - {color_choice.color_code}',
        color='red',  # Make the selected color choice stand out
        alpha=0.9
    )
    
    # Add value labels for the actual data
    for i, value in enumerate(color_choice_data):
        ax.annotate(f'{value:.1f}%',
                    xy=(x[i], value),
                    xytext=(0, 10),  # Offset vertically
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontweight='bold')
    
    # Plot ERP recommended distribution if available
    if has_erp_data and erp_distribution:
        # Prepare ERP data in the same order as size_codes
        erp_data = [erp_distribution.get(size, 0.0) for size in size_codes]
        
        # Skip if no data
        if any(erp_data):
            # Create label with source and season information if available
            if erp_source and erp_season:
                erp_label = f'ERP {erp_source} ({erp_season})'
            elif erp_source:
                erp_label = f'ERP {erp_source}'
            else:
                erp_label = 'ERP Recommended'
            
            ax.plot(
                x, 
                erp_data,
                marker='s',  # Square marker to differentiate
                markersize=7,
                linewidth=2.5,
                label=erp_label,
                color='green',  # Distinctive color for ERP data
                alpha=0.9,
                linestyle='--'  # Dashed line to differentiate
            )
            
            # Add value labels for ERP data
            for i, value in enumerate(erp_data):
                if value > 0:  # Only show non-zero values
                    ax.annotate(f'{value:.1f}%',
                                xy=(x[i], value),
                                xytext=(0, -15),  # Offset below the point
                                textcoords="offset points",
                                ha='center',
                                va='top',
                                color='green',
                                fontweight='bold')
    
    # Plot all cluster centroids if available
    # Determine which centroids to use based on algorithm
    if algorithm == 'km' and hasattr(template_group, 'kmeans_centroids') and template_group.kmeans_centroids is not None:
        centroids = template_group.kmeans_centroids
        cluster_label = "KMeans Cluster"
    elif algorithm == 'gmw' and hasattr(template_group, 'gmm_centroids') and template_group.gmm_centroids is not None:
        centroids = template_group.gmm_centroids
        cluster_label = "GMM Cluster"
    elif algorithm == 'dbscan' and hasattr(template_group, 'dbscan_centroids') and template_group.dbscan_centroids is not None:
        centroids = template_group.dbscan_centroids
        cluster_label = "DBSCAN Cluster"
    elif template_group.centroids is not None:
        centroids = template_group.centroids
        cluster_label = "Cluster"
    else:
        centroids = None
        
    if centroids is not None:
        for i, centroid in enumerate(centroids):
            # Choose a different color for current cluster
            line_color = 'blue' if color_choice.cluster_id == i else f'C{i}'
            line_alpha = 0.9 if color_choice.cluster_id == i else 0.5
            line_width = 2.0 if color_choice.cluster_id == i else 1.5
            
            # Plot this cluster's centroid as a line
            ax.plot(
                x, 
                centroid,
                marker='x' if color_choice.cluster_id == i else '.',
                markersize=7,
                linewidth=line_width,
                label=f'{cluster_label} {i}{" (This Cluster)" if color_choice.cluster_id == i else ""}',
                color=line_color,
                alpha=line_alpha,
                linestyle=':' if color_choice.cluster_id != i else '-.'
            )
            
            # Add value labels for the current cluster only
            if color_choice.cluster_id == i:
                for j, value in enumerate(centroid):
                    ax.annotate(f'{value:.1f}%',
                                xy=(x[j], value),
                                xytext=(0, 25),  # Offset above the point
                                textcoords="offset points",
                                ha='center',
                                va='bottom',
                                color=line_color)
    
    # Set labels and title
    cluster_label = f'Cluster {color_choice.cluster_id}' if color_choice.cluster_id is not None and not color_choice.is_outlier else 'Outlier'
    title = f'Size Distribution - {color_choice.style_code} {color_choice.color_code} ({color_choice.gender_code}) - {cluster_label}'
    
    # Modify title if we have ERP data
    if has_erp_data and erp_distribution:
        title += ' (with ERP Recommendation)'
        
    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(size_codes)
    
    # Add a horizontal grid for better readability
    ax.grid(axis='both', linestyle='--', alpha=0.3)
    
    # Position legend in the best location
    ax.legend(loc='best')
    
    # Set y-axis range to start at 0 
    y_max = max(max(color_choice_data) if color_choice_data else 0,
                max(erp_distribution.values()) if erp_distribution else 0)
    # Add 20% padding to the top
    ax.set_ylim(0, y_max * 1.2)
    
    fig.tight_layout()
    return fig


# --- Streamlit UI Components ---

def create_filters(sales_data: pd.DataFrame) -> Tuple[FilterCriteria, str]:
    """
    Create filter UI components.
    
    Args:
        sales_data: Sales data DataFrame
    
    Returns:
        Tuple of FilterCriteria and channel selection
    """
    st.sidebar.header("Filters")
    
    # Date filter
    default_start_date = "2022-01-01"
    start_date = st.sidebar.date_input(
        "Start Date", 
        datetime.strptime(default_start_date, "%Y-%m-%d").date()
    )
    
    # Min sales filter
    min_sales = st.sidebar.number_input("Minimum Sales", min_value=1, value=10)
    
    # Channel filter
    all_channels = sales_data['DISTRIBUTION_CHANNEL_CODE'].unique().tolist() if 'DISTRIBUTION_CHANNEL_CODE' in sales_data.columns else []
    channel = st.sidebar.selectbox("Distribution Channel", ["ALL"] + all_channels)
    channels_filter = [channel] if channel != "ALL" else None
    
    # Gender filter
    all_genders = sales_data['GENDER_CODE'].unique().tolist() if 'GENDER_CODE' in sales_data.columns else []
    gender = st.sidebar.selectbox("Gender", ["ALL"] + all_genders)
    gender_filter = gender if gender != "ALL" else None
    
    # Class filter
    if 'PRODUCT_CLASS_TEXT' in sales_data.columns:
        all_classes = sales_data['PRODUCT_CLASS_TEXT'].unique().tolist()
        class_filter = st.sidebar.selectbox("Product Class", ["ALL"] + all_classes)
        class_filter = class_filter if class_filter != "ALL" else None
    else:
        class_filter = None
    
    # Create filter criteria
    filter_criteria = FilterCriteria(
        start_date=start_date.strftime("%Y-%m-%d"),
        min_sales=min_sales,
        channels_filter=channels_filter,
        gender_filter=gender_filter,
        class_filter=class_filter
    )
    
    # Run button
    run_button = st.sidebar.button("Run Analysis")
    
    return filter_criteria, run_button


def create_template_selector(template_groups: Dict[str, TemplateGroup]) -> str:
    """
    Create a UI component to select a template group.
    
    Args:
        template_groups: Dictionary of template groups
    
    Returns:
        Selected template name
    """
    # Create a list of options with format: "Template Name (count)"
    options = [
        f"{name} ({len(group.color_choices)} COLOR_CHOICEs)" 
        for name, group in template_groups.items()
    ]
    
    # Sort by count (descending)
    options.sort(key=lambda x: int(x.split("(")[1].split(" ")[0]), reverse=True)
    
    # Extract template names from options
    template_names = [opt.split(" (")[0] for opt in options]
    
    # Create selectbox
    selected_option = st.selectbox("Select Size Template Group", options)
    selected_template = selected_option.split(" (")[0]
    
    return selected_template


def create_color_choice_selector(template_group: TemplateGroup) -> Optional[ColorChoiceSizeCurve]:
    """
    Create a UI component to select a COLOR_CHOICE with cluster information.
    
    Args:
        template_group: Template group to select from
    
    Returns:
        Selected ColorChoiceSizeCurve object or None
    """
    # Group color choices by cluster for better organization
    options_by_cluster = {}
    
    # Create lookup dictionaries for fast access
    curve_by_option = {}
    
    # Process regular clusters
    if template_group.clusters:
        for cluster_id, curves in template_group.clusters.items():
            options_by_cluster[f"Cluster {cluster_id}"] = []
            for c in curves:
                # Format: "STYLE_CODE - COLOR_CODE (GENDER_CODE) - Sales: XXX"
                option = f"{c.style_code} - {c.color_code} ({c.gender_code}) - Sales: {c.total_sales}"
                options_by_cluster[f"Cluster {cluster_id}"].append(option)
                curve_by_option[option] = c
    
    # Add outliers as a separate group if any
    if template_group.outliers and len(template_group.outliers) > 0:
        options_by_cluster["Outliers"] = []
        for c in template_group.outliers:
            option = f"{c.style_code} - {c.color_code} ({c.gender_code}) - Sales: {c.total_sales}"
            options_by_cluster["Outliers"].append(option)
            curve_by_option[option] = c
    
    # Sort options within each cluster
    for cluster, options in options_by_cluster.items():
        options.sort()
    
    # Create a selectbox for cluster selection
    cluster_options = list(options_by_cluster.keys())
    selected_cluster = st.selectbox("Select Cluster", cluster_options)
    
    # Create a selectbox for color choice within the selected cluster
    selected_option = st.selectbox(
        f"Select COLOR_CHOICE from {selected_cluster}", 
        options_by_cluster[selected_cluster]
    )
    
    # Return the selected curve
    return curve_by_option.get(selected_option)


def display_metrics(template_group: TemplateGroup):
    """
    Display key metrics for a template group.
    
    Args:
        template_group: Template group to display metrics for
    """
    total_color_choices = len(template_group.color_choices)
    total_sales = sum(c.total_sales for c in template_group.color_choices)
    outlier_count = len(template_group.outliers) if template_group.outliers else 0
    outlier_percentage = (outlier_count / total_color_choices) * 100 if total_color_choices > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("COLOR_CHOICEs", total_color_choices)
    with col2:
        st.metric("Total Sales", f"{total_sales:,}")
    with col3:
        st.metric("Clusters", template_group.optimal_k if template_group.optimal_k else 0)
    with col4:
        st.metric("Outliers", f"{outlier_count} ({outlier_percentage:.1f}%)")


def create_cluster_dashboard(template_group: TemplateGroup):
    """
    Create a dashboard for cluster analysis.
    
    Args:
        template_group: Template group to analyze
    """
    # Display metrics
    display_metrics(template_group)
    
    # If clustering was performed
    if template_group.optimal_k and template_group.optimal_k > 1:
        # Create tabs for different visualizations
        tabs = ["2D PCA", "3D PCA (Interactive)", "Silhouette Scores", "Radar Chart", "Cluster Size Curves"]
        
        # Add GMM specific tab if available
        if hasattr(template_group, 'component_weights') and template_group.component_weights is not None:
            tabs.append("GMM Details")
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tabs)
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)
        
        with tab1:
            st.pyplot(plot_cluster_pca(template_group, mode='2d'))
        
        with tab2:
            st.plotly_chart(plot_cluster_pca(template_group, mode='3d'), use_container_width=True)
        
        with tab3:
            if hasattr(template_group, 'silhouette_scores'):
                st.pyplot(plot_silhouette_scores(template_group))
            else:
                st.info("Insufficient data for silhouette score analysis.")
        
        with tab4:
            st.plotly_chart(plot_radar_chart(template_group), use_container_width=True)
        
        with tab5:
            st.pyplot(create_cluster_histograms(template_group))
        
        # Note about outliers if any (with no table - can be accessed via dropdown)
        if template_group.outliers and len(template_group.outliers) > 0:
            outlier_count = len(template_group.outliers)
            st.info(f"Found {outlier_count} outliers. You can view them in the COLOR_CHOICE dropdown selector below by selecting the 'Outliers' group.")
            
        # Display GMM-specific information if available
        if hasattr(template_group, 'component_weights') and template_group.component_weights is not None and 'tab6' in locals():
            with tab6:
                st.subheader("Gaussian Mixture Model Details")
                
                # Display component weights (mixing proportions)
                st.subheader("Component Weights")
                weights_df = pd.DataFrame({
                    'Component': [f"Cluster {i}" for i in range(len(template_group.component_weights))],
                    'Weight': template_group.component_weights
                })
                st.dataframe(weights_df)
                
                # Visualize weights as a pie chart
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(template_group.component_weights, labels=[f"Cluster {i}" for i in range(len(template_group.component_weights))], 
                       autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                st.pyplot(fig)
                
                # Display a note about probabilities
                st.info("GMM assigns data points to clusters based on probabilities rather than hard assignments like k-means. " +
                       "Points may have partial membership in multiple clusters.")
    else:
        st.info("Insufficient data for clustering. Try reducing the minimum sales threshold or selecting a different template.")


def create_erp_discrepancy_dashboard(template_groups: Dict[str, TemplateGroup]):
    """
    Create a dashboard to analyze ERP vs. actual size discrepancies.
    
    Args:
        template_groups: Dictionary of template groups with color choices
    """
    st.header("ERP Size Alignment Analysis")
    
    # Collect all color choices with ERP data from all template groups
    all_choices_with_erp = []
    for template_name, group in template_groups.items():
        for color_choice in group.color_choices:
            if color_choice.erp_alignment_score is not None:
                all_choices_with_erp.append({
                    'Style Code': color_choice.style_code,
                    'Color Code': color_choice.color_code,
                    'Gender': color_choice.gender_code,
                    'Template': color_choice.size_template_name,
                    'Total Sales': color_choice.total_sales,
                    'Cluster ID': color_choice.cluster_id if color_choice.cluster_id is not None else -1,
                    'Alignment Score': color_choice.erp_alignment_score,
                    'MAE': color_choice.erp_mae,
                    'Max Difference': color_choice.erp_max_diff,
                    'ERP Source': color_choice.erp_source if color_choice.erp_source else 'Unknown',
                    'Is Outlier': 'Yes' if color_choice.is_outlier else 'No',
                    'Object': color_choice  # Store the actual object for reference
                })
    
    # Count total products across all template groups
    total_products = sum(len(group.color_choices) for group in template_groups.values())
    
    if not all_choices_with_erp:
        st.warning("No ERP alignment data available. This could be because:")
        st.markdown("""
        1. The ERP mapping file doesn't contain entries for these products
        2. The ERP weights file doesn't have corresponding size distributions
        3. The products' style codes don't match the format in the ERP files
        
        Please check the ERP data files to ensure they contain the correct mapping information.
        """)
        
        # Add an option to view ERP mapping sample
        if st.button("View ERP Mapping Sample"):
            if 'erp_repository' in st.session_state and st.session_state.erp_repository:
                erp_repo = st.session_state.erp_repository
                
                # Create a sample of the mappings
                sample_mappings = []
                for i, (style, weight) in enumerate(erp_repo.style_to_weight_map.items()):
                    if i >= 50:  # Limit to 50 entries
                        break
                    sample_mappings.append({"Style/Color Code": style, "Size Weight Code": weight})
                
                # Convert to DataFrame for display
                if sample_mappings:
                    st.subheader("ERP Mapping Sample (First 50 entries)")
                    df = pd.DataFrame(sample_mappings)
                    st.dataframe(df, use_container_width=True)
                    
                    # Add tips
                    st.info("""
                    **Tip:** Check if your product style codes match the format in this mapping file.
                    If they differ (e.g., due to case sensitivity or formatting), the ERP data won't be found.
                    """)
                else:
                    st.error("No mappings found in the ERP mapping file.")
            else:
                st.error("ERP repository not initialized or not available.")
        
        return
        
    # Add an expandable section with ERP mapping info
    with st.expander("View ERP Mapping Information"):
        if 'erp_repository' in st.session_state and st.session_state.erp_repository:
            erp_repo = st.session_state.erp_repository
            
            st.info(f"""
            **ERP Mapping Statistics:**
            - Total style/color mappings: {len(erp_repo.style_to_weight_map)}
            - Total size weight distributions: {len(erp_repo.weight_to_distribution_map)}
            - Files used:
              - Mapping file: {erp_repo.erp_mapping_file}
              - Weights file: {erp_repo.erp_weights_file}
            """)
            
            # Create a sample of the mappings 
            sample_mappings = []
            for i, (style, weight) in enumerate(erp_repo.style_to_weight_map.items()):
                if i >= 20:  # Limit to 20 entries to save space
                    break
                sample_mappings.append({"Style/Color Code": style, "Size Weight Code": weight})
            
            # Convert to DataFrame for display
            if sample_mappings:
                st.subheader("ERP Mapping Sample")
                df = pd.DataFrame(sample_mappings)
                st.dataframe(df, use_container_width=True)
        
    # Calculate percentage of products with ERP data
    erp_coverage = (len(all_choices_with_erp) / total_products) * 100 if total_products > 0 else 0
    
    # Show a warning if coverage is low
    if erp_coverage < 50:
        st.warning(f"""
        Only {erp_coverage:.1f}% of products ({len(all_choices_with_erp)} out of {total_products}) have matching ERP size data.
        
        **Why are some products missing?** The ERP files might not contain entries for all products in your sales data. This could be because:
        1. Some styles may have been discontinued or are new and not yet in the ERP system
        2. The style codes in your sales data might not match exactly with the format in the ERP mapping file
        3. Certain product categories may not use standard size curves in the ERP system
        
        You can check the ERP mapping file to see which styles have defined size curves.
        """)
    elif erp_coverage < 100:
        st.info(f"""
        {erp_coverage:.1f}% of products ({len(all_choices_with_erp)} out of {total_products}) have matching ERP size data.
        
        Some products don't have matching ERP data, which could be due to:
        - New styles not yet in the ERP system
        - Style code format differences
        - Special product categories with non-standard sizing
        """)
    else:
        st.success(f"100% of products ({total_products}) have matching ERP size data.")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_choices_with_erp)
    
    # Create metrics for overall alignment
    overall_alignment = df['Alignment Score'].mean()
    poor_alignment_count = (df['Alignment Score'] < 80).sum()
    poor_alignment_percent = (poor_alignment_count / len(df)) * 100
    
    # Display overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        delta_color = "normal"
        if overall_alignment < 80:
            delta_color = "inverse"  # Red for poor alignment
            
        st.metric(
            "Average Alignment Score", 
            f"{overall_alignment:.1f}/100",
            delta=None,
            delta_color=delta_color
        )
    
    with col2:
        st.metric(
            "Products With Poor Alignment", 
            f"{poor_alignment_count}",
            delta=f"{poor_alignment_percent:.1f}%",
            delta_color="inverse" if poor_alignment_percent > 20 else "normal"
        )
    
    with col3:
        st.metric(
            "Total Products Analyzed", 
            f"{len(df)}",
            delta=None
        )
    
    # Create filters
    st.subheader("Filter Products")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Create a slider for alignment score
        min_alignment = int(df['Alignment Score'].min())
        max_alignment = int(df['Alignment Score'].max())
        
        # Handle case where min and max are the same (only one value)
        if min_alignment == max_alignment:
            st.info(f"All products have the same alignment score: {min_alignment}")
            alignment_range = (min_alignment, min_alignment)
        else:
            alignment_range = st.slider(
                "Alignment Score Range", 
                min_value=min_alignment, 
                max_value=max_alignment,
                value=(min_alignment, max_alignment)
            )
    
    with col2:
        # Create a multiselect for gender
        unique_genders = sorted(df['Gender'].unique().tolist())
        if len(unique_genders) <= 1:
            st.info(f"Only one gender available: {unique_genders[0] if unique_genders else 'None'}")
            selected_genders = unique_genders
        else:
            genders = ['ALL'] + unique_genders
            selected_genders = st.multiselect("Gender", genders, default='ALL')
    
    with col3:
        # Create a multiselect for templates
        unique_templates = sorted(df['Template'].unique().tolist())
        if len(unique_templates) <= 1:
            st.info(f"Only one template available: {unique_templates[0] if unique_templates else 'None'}")
            selected_templates = unique_templates
        else:
            templates = ['ALL'] + unique_templates
            selected_templates = st.multiselect("Size Template", templates, default='ALL')
            
    with col4:
        # Create a multiselect for ERP Source
        unique_sources = sorted(df['ERP Source'].unique().tolist())
        if len(unique_sources) <= 1:
            st.info(f"Only one ERP source available: {unique_sources[0] if unique_sources else 'None'}")
            selected_sources = unique_sources
        else:
            sources = ['ALL'] + unique_sources
            selected_sources = st.multiselect("ERP Source", sources, default='ALL')
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply alignment score filter
    filtered_df = filtered_df[
        (filtered_df['Alignment Score'] >= alignment_range[0]) & 
        (filtered_df['Alignment Score'] <= alignment_range[1])
    ]
    
    # Apply gender filter
    if 'ALL' not in selected_genders and selected_genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(selected_genders)]
    
    # Apply template filter
    if 'ALL' not in selected_templates and selected_templates:
        filtered_df = filtered_df[filtered_df['Template'].isin(selected_templates)]
        
    # Apply ERP source filter
    if 'ALL' not in selected_sources and selected_sources:
        filtered_df = filtered_df[filtered_df['ERP Source'].isin(selected_sources)]
    
    # Create a radio for sort order if we have enough data to sort
    if len(filtered_df) > 1:
        sort_option = st.radio(
            "Sort by:",
            ["Worst Alignment First", "Best Alignment First", "Highest Sales First"],
            horizontal=True
        )
        
        # Apply sort
        if sort_option == "Worst Alignment First":
            filtered_df = filtered_df.sort_values('Alignment Score', ascending=True)
        elif sort_option == "Best Alignment First":
            filtered_df = filtered_df.sort_values('Alignment Score', ascending=False)
        else:  # Highest Sales First
            filtered_df = filtered_df.sort_values('Total Sales', ascending=False)
    else:
        # With only one item, sorting isn't needed
        st.info("Only one product available - sorting not applicable")
    
    # Display filtered results
    st.subheader(f"Size Curve Alignment Results ({len(filtered_df)} products)")
    
    # Format DataFrame for display
    display_df = filtered_df.drop('Object', axis=1).copy()
    
    # Add better formatting to numeric columns
    display_df['Alignment Score'] = display_df['Alignment Score'].apply(lambda x: f"{x:.1f}/100")
    display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.2f}%")
    display_df['Max Difference'] = display_df['Max Difference'].apply(lambda x: f"{x:.2f}%")
    
    # Show table without complex styling to avoid errors
    try:
        # Simple styling approach that's more resilient
        def style_alignment_score(val):
            """Style alignment scores with colors based on value"""
            try:
                # Extract numeric value if it's a string like "95.6/100"
                if isinstance(val, str) and '/' in val:
                    num_val = float(val.split('/')[0])
                elif isinstance(val, (int, float)):
                    num_val = float(val)
                else:
                    return ''
                
                # Apply different colors based on score
                if num_val < 80:
                    return 'background-color: #ffcccc'  # Light red for poor alignment (<80%)
                elif num_val > 95:
                    return 'background-color: #ccffcc'  # Light green for excellent alignment (>95%)
                return ''
            except:
                return ''
        
        # Apply simple styling to just the Alignment Score column
        if 'Alignment Score' in display_df.columns:
            # Use .map instead of .applymap (which is deprecated)
            styled_df = display_df.style.map(
                style_alignment_score, 
                subset=['Alignment Score']
            )
            st.dataframe(styled_df, use_container_width=True)
        else:
            # If no Alignment Score column, just show the dataframe without styling
            st.dataframe(display_df, use_container_width=True)
    except Exception as e:
        # Ultimate fallback - show unstyled dataframe
        st.error(f"Error displaying data: {str(e)}")
        st.dataframe(display_df, use_container_width=True)
    
    # Create a plot of the worst aligned products
    if len(filtered_df) > 0:
        # Only show distribution chart if we have enough data points
        if len(filtered_df) > 1:
            st.subheader("Alignment Score Distribution")
            
            fig = px.histogram(
                filtered_df, 
                x='Alignment Score',
                nbins=min(20, len(filtered_df)),  # Adjust number of bins based on data size
                title="Distribution of Alignment Scores",
                labels={'Alignment Score': 'Alignment Score (0-100)'},
                color_discrete_sequence=['#3366cc']
            )
            
            fig.update_layout(
                xaxis_title="Alignment Score",
                yaxis_title="Number of Products",
                bargap=0.2
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a scatter plot of alignment vs. sales
            st.subheader("Alignment vs. Sales")
            
            fig = px.scatter(
                filtered_df,
                x='Total Sales',
                y='Alignment Score',
                color='ERP Source',  # Color by ERP source instead of gender
                size='Max Difference',  # Size points by maximum difference
                hover_data=['Style Code', 'Color Code', 'Template', 'MAE', 'Gender'],
                title="Alignment Score vs. Sales Volume",
                labels={
                    'Total Sales': 'Total Unit Sales',
                    'Alignment Score': 'Alignment Score (0-100)',
                    'ERP Source': 'ERP Data Source'
                }
            )
            
            fig.update_layout(
                xaxis_title="Total Unit Sales",
                yaxis_title="Alignment Score (0-100)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("More data needed to display distribution charts. Only one product available.")
    else:
        st.warning("No data available after applying filters.")


def create_color_choice_details(color_choice: ColorChoiceSizeCurve, template_group: TemplateGroup, erp_repository=None):
    """
    Display detailed information for a selected COLOR_CHOICE.
    
    Args:
        color_choice: Selected COLOR_CHOICE
        template_group: Template group containing the COLOR_CHOICE
        erp_repository: Optional ERP repository for size recommendations
    """
    # Add algorithm selection for visualization
    if (hasattr(template_group, 'kmeans_clusters') or 
        hasattr(template_group, 'gmm_clusters') or 
        hasattr(template_group, 'dbscan_clusters')):
        
        selected_algorithm = st.radio(
            "Select algorithm for visualization:",
            ["KMeans", "GMM", "DBSCAN"],
            horizontal=True,
            key="color_choice_algorithm_selector"  # Add unique key to fix duplicate ID error
        )
        
        # Map the user-friendly name to the internal algorithm code
        algo_map = {"KMeans": "km", "GMM": "gmw", "DBSCAN": "dbscan"}
        selected_algo_code = algo_map[selected_algorithm]
    else:
        selected_algo_code = None
        
    # Display size distribution
    st.subheader(f"Size Distribution - {color_choice.style_code} {color_choice.color_code}")
    st.pyplot(plot_size_distribution(color_choice, template_group, erp_repository, algorithm=selected_algo_code))
    
    # Display metadata
    # Determine number of columns based on available data
    has_gmm = hasattr(template_group, 'sample_probs') and template_group.sample_probs is not None
    has_erp = color_choice.erp_alignment_score is not None
    
    # Calculate required columns
    base_cols = 3  # Total Sales, Cluster, Distance
    gmm_cols = 1 if has_gmm else 0
    erp_cols = 4 if has_erp else 0  # ERP Source, ERP Alignment Score, MAE, Max Difference
    total_cols = base_cols + gmm_cols + erp_cols
    
    cols = st.columns(total_cols)
    
    col_idx = 0
    
    # Basic metrics
    with cols[col_idx]:
        st.metric("Total Sales", color_choice.total_sales)
        col_idx += 1
        
    with cols[col_idx]:
        cluster_label = f"Cluster {color_choice.cluster_id}" if color_choice.cluster_id is not None and not color_choice.is_outlier else "Outlier"
        st.metric("Cluster", cluster_label)
        col_idx += 1
        
    with cols[col_idx]:
        if color_choice.distance_from_centroid is not None and not color_choice.is_outlier:
            st.metric("Distance from Centroid", f"{color_choice.distance_from_centroid:.2f}")
        col_idx += 1
    
    # Display GMM probability if available
    if has_gmm:
        with cols[col_idx]:
            # Find this color choice's index to get its probabilities
            try:
                color_choice_idx = template_group.color_choices.index(color_choice)
                if color_choice_idx < len(template_group.sample_probs):
                    max_prob = np.max(template_group.sample_probs[color_choice_idx])
                    st.metric("GMM Probability", f"{max_prob:.2%}")
            except ValueError:
                pass  # Color choice not found in list
            col_idx += 1
            
    # Display ERP alignment metrics if available
    if has_erp:
        with cols[col_idx]:
            # Show the ERP data source
            source_display = color_choice.erp_source if color_choice.erp_source else "Unknown"
            st.metric("ERP Data Source", 
                      source_display, 
                      delta=None)
            col_idx += 1
            
        with cols[col_idx]:
            # Use colors to indicate alignment quality
            delta_color = "normal"
            if color_choice.erp_alignment_score < 60:
                delta_color = "inverse"  # Red for poor alignment
                
            st.metric("ERP Alignment Score", 
                      f"{color_choice.erp_alignment_score:.1f}/100", 
                      delta=f"{color_choice.erp_alignment_score/100:.0%}",
                      delta_color=delta_color)
            col_idx += 1
            
        with cols[col_idx]:
            st.metric("Avg Size Diff (MAE)", 
                      f"{color_choice.erp_mae:.2f}%",
                      delta=None)
            col_idx += 1
            
        with cols[col_idx]:
            st.metric("Max Size Diff", 
                      f"{color_choice.erp_max_diff:.2f}%",
                      delta=None)
            col_idx += 1
    
    # Display size percentages in a table
    st.subheader("Size Percentages")
    
    # Create a DataFrame to compare actual vs. ERP if available
    if erp_repository and erp_repository.is_initialized:
        # Get season from sidebar if it exists
        selected_season = st.session_state.get('selected_season', None)
        
        # Get ERP recommended distribution
        erp_distribution, erp_source, erp_season = erp_repository.get_erp_size_distribution(
            color_choice.style_code, 
            color_choice.color_code,
            season=selected_season
        )
        
        # Show source and season information if available
        if erp_source:
            info_text = f"ERP data source: {erp_source}"
            if erp_season:
                info_text += f", Season: {erp_season}"
            st.caption(info_text)
            color_choice.erp_source = erp_source
        
        # Calculate and store alignment metrics if we have distribution data 
        # (don't check if already present - always recalculate)
        if erp_distribution:
            # Store the ERP percentages
            color_choice.erp_percentages = erp_distribution.copy()
            
            # Calculate Mean Absolute Error
            total_error = 0.0
            max_diff = 0.0
            count = 0
            
            for size, actual_pct in color_choice.size_percentages.items():
                erp_pct = erp_distribution.get(size, 0.0)
                diff = abs(actual_pct - erp_pct)
                total_error += diff
                count += 1
                max_diff = max(max_diff, diff)
            
            if count > 0:
                color_choice.erp_mae = round(total_error / count, 2)
                color_choice.erp_max_diff = round(max_diff, 2)
                
                # Calculate alignment score (100 - MAE, with minimum of 0)
                # This creates a 0-100 score where 100 is perfect alignment
                color_choice.erp_alignment_score = round(max(0, 100 - (color_choice.erp_mae * 2)), 1)
        
        if erp_distribution:
            # Get the size codes in template order
            size_codes = template_group.size_codes
            
            # Create a comparison DataFrame with sizes in template order
            comparison_data = []
            for i, size in enumerate(size_codes):
                actual_pct = color_choice.size_percentages.get(size, 0.0)
                erp_pct = erp_distribution.get(size, 0.0)
                difference = actual_pct - erp_pct
                
                comparison_data.append({
                    'Size': size,
                    'Size Order': i,  # Use index in size_codes for ordering
                    'Actual Sales %': round(actual_pct, 1),
                    'ERP Recommended %': round(erp_pct, 1),
                    'Difference': round(difference, 1)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Sort by Size Order (which follows the template order)
            comparison_df = comparison_df.sort_values('Size Order')
            
            # Drop the Size Order column before display
            comparison_df = comparison_df.drop(columns=['Size Order'])
            
            # Apply styling to highlight differences
            def highlight_difference(val):
                """Apply color highlighting based on difference value"""
                if isinstance(val, (int, float)):
                    if val > 5:
                        return 'background-color: #ffcccc'  # Light red for positive difference
                    elif val < -5:
                        return 'background-color: #ccffcc'  # Light green for negative difference
                return ''
            
            # Display with styling
            st.dataframe(
                comparison_df.style.map(
                    highlight_difference, 
                    subset=['Difference']
                ),
                use_container_width=True
            )
        else:
            # Just show actual data if no ERP data available
            size_df = pd.DataFrame([{
                'Size': size,
                'Size Order': i,
                'Percentage': round(color_choice.size_percentages.get(size, 0.0), 1)
            } for i, size in enumerate(template_group.size_codes)])
            
            # Sort by Size Order (which follows the template order)
            size_df = size_df.sort_values('Size Order').drop(columns=['Size Order'])
            
            st.dataframe(size_df, use_container_width=True)
    else:
        # Just show actual data if no ERP repository
        size_df = pd.DataFrame([{
            'Size': size,
            'Size Order': i,
            'Percentage': round(color_choice.size_percentages.get(size, 0.0), 1)
        } for i, size in enumerate(template_group.size_codes)])
        
        # Sort by Size Order (which follows the template order)
        size_df = size_df.sort_values('Size Order').drop(columns=['Size Order'])
        
        st.dataframe(size_df, use_container_width=True)


# --- Main Application Class ---

class ColorChoiceSizeCurveApp:
    """
    Main application class for COLOR_CHOICE level size curve analysis and clustering.
    """
    
    def __init__(self, log_level=logging.INFO, cluster_size=5, algorithm="km", 
                 erp_combined_file=None, erp_mapping_file=None, erp_weights_file=None, 
                 eps=0.5, min_samples=5):
        """
        Initialize the application.
        
        Args:
            log_level: Logging level
            cluster_size: Number of clusters to use (default: 5)
            algorithm: Clustering algorithm to use (km: KMeans, gmm: Gaussian Mixture Model, dbscan: DBSCAN)
            erp_combined_file: Path to combined ERP file with data for both seasons
            erp_mapping_file: Path to ERP mapping file (style code to size weight code) - legacy support
            erp_weights_file: Path to ERP weights file (size weight code to distributions) - legacy support
            eps: DBSCAN parameter - maximum distance for points to be considered neighbors (default: 0.5)
            min_samples: DBSCAN parameter - minimum samples in neighborhood for core point (default: 5)
        """
        # Set up logging
        self.logger = setup_logging(log_level=log_level)
        
        # Store parameters
        self.cluster_size = cluster_size
        self.algorithm = algorithm
        self.erp_combined_file = erp_combined_file
        self.erp_mapping_file = erp_mapping_file
        self.erp_weights_file = erp_weights_file
        self.eps = eps
        self.min_samples = min_samples
        
        # Set page configuration
        st.set_page_config(
            page_title="Vuori Color Choice Size Curve Analyzer",
            page_icon="📊",
            layout="wide"
        )
        
        # Page title and description
        st.title("Vuori Color Choice Size Curve Analyzer")
        st.markdown("""
        This application analyzes sales data at the COLOR_CHOICE level (STYLE_CODE + COLOR_CODE) 
        to generate size curves and cluster similar patterns.
        """)
        
        # Initialize session state
        if 'color_choice_initialized' not in st.session_state:
            st.session_state.color_choice_initialized = False
            st.session_state.connector = None
            st.session_state.product_repository = None
            st.session_state.sales_repository = None
            st.session_state.erp_repository = None
            st.session_state.sales_data = None
            st.session_state.product_data = None
            st.session_state.color_choice_analyzer = None
            st.session_state.template_groups = None
    
    def initialize_repositories(self):
        """
        Initialize database connection and repositories.
        """
        try:
            connector = SnowflakeConnector()
            product_repository = ProductRepository(connector)
            sales_repository = SalesRepository(connector)
            
            st.session_state.connector = connector
            st.session_state.product_repository = product_repository
            st.session_state.sales_repository = sales_repository
            
            # Initialize ERP repository if files provided
            if self.erp_combined_file or (self.erp_mapping_file and self.erp_weights_file):
                # Import the repository (add the import at the top of the file)
                from vuori_size_curve.data.repositories.erp_repository import ERPSizeRepository
                
                # Create sidebar expander for ERP data status
                with st.sidebar.expander("ERP Data Status", expanded=True):
                    with st.spinner("Loading ERP Data..."):
                        try:
                            # Initialize with either combined file or separate files
                            erp_repository = ERPSizeRepository(
                                erp_combined_file=self.erp_combined_file,
                                erp_mapping_file=self.erp_mapping_file,
                                erp_weights_file=self.erp_weights_file
                            )
                            erp_repository.initialize()
                            st.session_state.erp_repository = erp_repository
                            st.success("✅ ERP Size Data Loaded")
                            
                            # Display some ERP data stats
                            st.info(f"Loaded {len(erp_repository.style_to_weight_map)} style mappings and "
                                   f"{len(erp_repository.weight_to_distribution_map)} size distributions")
                            
                            # Show mode information
                            if self.erp_combined_file:
                                st.info(f"Using combined mode with file: {os.path.basename(self.erp_combined_file)}")
                                
                                # Add season selector for combined file mode
                                seasons = ["All Seasons", "FAHO25", "SPSU25"]
                                selected_season = st.selectbox(
                                    "Select Season", 
                                    seasons, 
                                    help="Filter ERP data by season"
                                )
                                # Store selection in session state
                                st.session_state.selected_season = None if selected_season == "All Seasons" else selected_season
                            else:
                                st.info(f"Using legacy mode with separate mapping and weights files")
                                
                        except Exception as e:
                            st.error(f"Error loading ERP data: {str(e)}")
                            st.session_state.erp_repository = None
            else:
                st.session_state.erp_repository = None
                # Add a note in sidebar about ERP data
                st.sidebar.info("ERP size curves not available. Use --erp-combined-file or both --erp-mapping-file and --erp-weights-file to enable.")
            
            # Fetch initial data
            filter_criteria = FilterCriteria()
            st.session_state.sales_data = sales_repository.get_raw_data(filter_criteria)
            st.session_state.product_data = product_repository.get_raw_data()
            
            # Create analyzer
            st.session_state.color_choice_analyzer = ColorChoiceAnalyzer(
                product_repository=product_repository,
                sales_repository=sales_repository,
                min_sales=10,
                cluster_size=self.cluster_size,
                algorithm=self.algorithm,
                eps=self.eps,
                min_samples=self.min_samples
            )
            
            st.session_state.color_choice_initialized = True
            
        except Exception as e:
            st.error(f"Error loading initial data: {str(e)}")
            st.error(f"Detailed error: {traceback.format_exc()}")
            st.session_state.initialization_error = str(e)
    
    def run_analysis(self, filter_criteria: FilterCriteria):
        """
        Run size curve analysis and clustering.
        
        Args:
            filter_criteria: Filtering criteria for data
        """
        with st.spinner("Running analysis..."):
            try:
                # Get filtered sales data
                filtered_data = st.session_state.sales_repository.get_raw_data(filter_criteria)
                
                # Get style templates
                style_templates = st.session_state.product_repository.determine_size_templates()
                
                # Analyze color choice size curves
                analyzer = st.session_state.color_choice_analyzer
                color_choice_curves = analyzer.analyze(
                    sales_data=filtered_data,
                    style_templates=style_templates,
                    channel=filter_criteria.channels_filter[0] if filter_criteria.channels_filter else None
                )
                
                # Group by template
                template_groups = analyzer.group_by_template(color_choice_curves)
                
                # Cluster each template group
                for template_name, group in template_groups.items():
                    # Only cluster if we have enough data
                    if len(group.color_choices) >= 4:
                        analyzer.cluster_template_group(group)
                    else:
                        # Not enough data for clustering
                        group.optimal_k = 1
                        group.clusters = {0: group.color_choices}
                        group.outliers = []
                
                # Calculate ERP metrics for all color choices if ERP repository is available
                if 'erp_repository' in st.session_state and st.session_state.erp_repository and st.session_state.erp_repository.is_initialized:
                    erp_repository = st.session_state.erp_repository
                    
                    # Print ERP repository stats for debugging
                    total_mapping_entries = len(erp_repository.style_to_weight_map)
                    total_distribution_entries = len(erp_repository.weight_to_distribution_map)
                    print(f"ERP Repository has {total_mapping_entries} style mappings and {total_distribution_entries} size distributions")
                    
                    # Debug: Print a few example mappings from the repository
                    if total_mapping_entries > 0:
                        print("Example style mappings (first 5):")
                        for i, (style, weight) in enumerate(list(erp_repository.style_to_weight_map.items())[:5]):
                            print(f"  {style} -> {weight}")
                    
                    # Track stats for debugging
                    processed_count = 0
                    matched_count = 0
                    already_processed_count = 0
                    
                    # Process each template group
                    for template_name, group in template_groups.items():
                        # Process each color choice in the group
                        for color_choice in group.color_choices:
                            processed_count += 1
                            
                            # Skip if already processed (but count them)
                            if color_choice.erp_alignment_score is not None:
                                already_processed_count += 1
                                matched_count += 1  # Count as matched since it already has data
                                continue
                                
                            # Get ERP recommended distribution
                            erp_distribution, erp_source, erp_season = erp_repository.get_erp_size_distribution(
                                color_choice.style_code, 
                                color_choice.color_code,
                                season=st.session_state.get('selected_season', None)
                            )
                            
                            # Calculate and store alignment metrics if distribution available
                            if erp_distribution:
                                matched_count += 1
                                
                                # Store the ERP percentages and source
                                color_choice.erp_percentages = erp_distribution.copy()
                                color_choice.erp_source = erp_source
                                
                                # Calculate Mean Absolute Error
                                total_error = 0.0
                                max_diff = 0.0
                                count = 0
                                
                                for size, actual_pct in color_choice.size_percentages.items():
                                    erp_pct = erp_distribution.get(size, 0.0)
                                    diff = abs(actual_pct - erp_pct)
                                    total_error += diff
                                    count += 1
                                    max_diff = max(max_diff, diff)
                                
                                if count > 0:
                                    color_choice.erp_mae = round(total_error / count, 2)
                                    color_choice.erp_max_diff = round(max_diff, 2)
                                    
                                    # Calculate alignment score (100 - MAE, with minimum of 0)
                                    # This creates a 0-100 score where 100 is perfect alignment
                                    color_choice.erp_alignment_score = round(max(0, 100 - (color_choice.erp_mae * 2)), 1)
                    
                    # Log match rate
                    match_percentage = (matched_count / processed_count * 100) if processed_count > 0 else 0
                    print(f"ERP data match rate: {match_percentage:.1f}% ({matched_count} out of {processed_count} products)")
                    print(f"Already processed: {already_processed_count}")
                    
                    # Log how many products have ERP metrics
                    count_with_erp = sum(
                        sum(1 for c in group.color_choices if c.erp_alignment_score is not None)
                        for group in template_groups.values()
                    )
                    print(f"Total products with ERP metrics: {count_with_erp}")
                
                # Store results
                st.session_state.template_groups = template_groups
                
                # Prepare output dataframe
                output_df = analyzer.prepare_output_dataframe(template_groups)
                st.session_state.output_df = output_df
                
                return template_groups
                
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")
                st.error(f"Detailed error: {traceback.format_exc()}")
                return None
    
    def run(self):
        """
        Run the application.
        """
        # Initialize repositories if not already done
        if not st.session_state.color_choice_initialized:
            with st.spinner("Loading data from Snowflake..."):
                self.initialize_repositories()
        
        # Create filters if initialized
        if st.session_state.color_choice_initialized and 'sales_data' in st.session_state:
            # Show clustering mode and algorithm in sidebar
            if self.algorithm == "gmw":
                algo_name = "Gaussian Mixture Model"
            elif self.algorithm == "dbscan":
                algo_name = "DBSCAN"
            else:
                algo_name = "K-Means"
                
            st.sidebar.info(f"Using algorithm: {algo_name}")
            
            # Show algorithm-specific parameters
            if self.algorithm == "dbscan":
                st.sidebar.info(f"DBSCAN parameters: eps={self.eps}, min_samples={self.min_samples}")
            else:
                # Only show cluster count for KMeans and GMM
                if self.cluster_size == 0:
                    st.sidebar.success("Auto clustering enabled - will determine optimal cluster count automatically")
                else:
                    st.sidebar.info(f"Using fixed cluster count: {self.cluster_size} (use --cluster-size=0 for auto detection)")
            
            filter_criteria, run_button = create_filters(st.session_state.sales_data)
            
            # Run analysis when button is clicked
            if run_button:
                template_groups = self.run_analysis(filter_criteria)
                
                if template_groups:
                    st.success(f"Analysis complete. Found {len(template_groups)} unique size templates.")
            
            # Display results if available
            if 'template_groups' in st.session_state and st.session_state.template_groups:
                template_groups = st.session_state.template_groups
                
                # Check if ERP repository is available
                has_erp = 'erp_repository' in st.session_state and st.session_state.erp_repository is not None
                
                # Create tabs for different views
                if has_erp:
                    tab1, tab2, tab3 = st.tabs([
                        "Cluster Analysis", 
                        "ERP Size Analytics ℹ️", 
                        "Export"
                    ])
                    
                    # Add tooltip help in tab2 (it will only be visible when tab2 is selected)
                    with tab2:
                        st.info("""
                        **About ERP Size Analytics**: This tab shows analysis of how well actual sales align with ERP recommended size distributions.
                        The analysis compares product sales data with size distributions from your ERP system files:
                        - **F25 Size Weight Key**: Maps style codes to ERP size weight codes
                        - **ERP Size weight**: Contains the percentage distribution for each size code
                        
                        Products will only appear in this analysis if they have a matching entry in these files.
                        """)
                else:
                    tab1, tab3 = st.tabs(["Cluster Analysis", "Export"])
                    
                with tab1:
                    # Template selector
                    selected_template = create_template_selector(template_groups)
                    template_group = template_groups[selected_template]
                    
                    # Cluster dashboard
                    create_cluster_dashboard(template_group)
                    
                    # COLOR_CHOICE selector
                    st.header("COLOR_CHOICE Details")
                    selected_color_choice = create_color_choice_selector(template_group)
                    
                    if selected_color_choice:
                        create_color_choice_details(
                            selected_color_choice, 
                            template_group,
                            st.session_state.erp_repository if has_erp else None
                        )
                
                # ERP Size Analytics tab
                if has_erp:
                    with tab2:
                        create_erp_discrepancy_dashboard(template_groups)
                
                # Export tab
                with tab3:
                    st.header("Export Results")
                    if st.button("Export to CSV"):
                        csv = st.session_state.output_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"color_choice_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
        else:
            # Initial state message
            st.info("Loading data... If this message persists, check the Snowflake connection settings.")


# --- Parse command line arguments ---

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Vuori Color Choice Size Curve Analyzer")
    parser.add_argument("--cluster-size", type=int, default=5, 
                        help="Number of clusters to use for analysis (default: 5, 0 for automatic determination)")
    parser.add_argument("--algo", type=str, default="km", choices=["km", "gmw", "dbscan"],
                        help="Clustering algorithm to use (km: KMeans, gmw: Gaussian Mixture Model, dbscan: DBSCAN)")
    
    # DBSCAN parameters
    parser.add_argument("--eps", type=float, default=0.5,
                        help="DBSCAN: Maximum distance between samples for them to be considered neighbors (default: 0.5)")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="DBSCAN: Minimum number of samples in a neighborhood to be considered a core point (default: 5)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO)")
    
    # Add ERP data file arguments
    parser.add_argument("--erp-combined-file", type=str,
                        help="Path to combined ERP file with data for both seasons")
    parser.add_argument("--erp-mapping-file", type=str, 
                        help="Path to ERP mapping file (style code to size weight code) - legacy support")
    parser.add_argument("--erp-weights-file", type=str,
                        help="Path to ERP weights file (size weight code to distributions) - legacy support")
    
    return parser.parse_args()

# --- Run the application ---

if __name__ == "__main__":
    import sys
    
    # Handle --help explicitly when running in Streamlit
    if "--help" in sys.argv:
        parser = argparse.ArgumentParser(description="Vuori Color Choice Size Curve Analyzer")
        parser.add_argument("--cluster-size", type=int, default=5, 
                            help="Number of clusters to use for analysis (default: 5, 0 for automatic determination)")
        parser.add_argument("--algo", type=str, default="km", choices=["km", "gmw"],
                            help="Clustering algorithm to use (km: KMeans, gmw: Gaussian Mixture Model)")
        parser.add_argument("--log-level", type=str, default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            help="Logging level (default: INFO)")
        # Add ERP data file arguments
        parser.add_argument("--erp-combined-file", type=str,
                            help="Path to combined ERP file with data for both seasons")
        parser.add_argument("--erp-mapping-file", type=str, 
                            help="Path to ERP mapping file (style code to size weight code) - legacy support")
        parser.add_argument("--erp-weights-file", type=str,
                            help="Path to ERP weights file (size weight code to distributions) - legacy support")
        parser.print_help()
        sys.exit(0)
    
    # Parse command line arguments
    args = parse_args()
    
    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level)
    
    # Create and run the application
    app = ColorChoiceSizeCurveApp(
        log_level=log_level, 
        cluster_size=args.cluster_size, 
        algorithm=args.algo,
        erp_combined_file=args.erp_combined_file,
        erp_mapping_file=args.erp_mapping_file,
        erp_weights_file=args.erp_weights_file,
        eps=args.eps,
        min_samples=args.min_samples
    )
    app.run()