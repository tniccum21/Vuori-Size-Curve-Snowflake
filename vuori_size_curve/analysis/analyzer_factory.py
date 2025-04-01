"""
Factory for creating analyzers.
"""
from typing import Dict, Type, Optional
from vuori_size_curve.analysis.base_analyzer import BaseAnalyzer
from vuori_size_curve.analysis.size_curve import (
    StyleAnalyzer,
    ClassAnalyzer,
    SubclassAnalyzer,
    CollectionAnalyzer
)
from vuori_size_curve.data.repositories.product_repository import ProductRepository
from vuori_size_curve.data.repositories.sales_repository import SalesRepository
from vuori_size_curve.config.app_config import ANALYSIS_LEVELS


class AnalyzerFactory:
    """
    Factory for creating different types of analyzers.
    """
    
    def __init__(
        self,
        product_repository: ProductRepository,
        sales_repository: SalesRepository,
        min_sales: int = 0
    ):
        """
        Initialize the analyzer factory.
        
        Args:
            product_repository (ProductRepository): Repository for product data
            sales_repository (SalesRepository): Repository for sales data
            min_sales (int): Minimum sales quantity threshold
        """
        self.product_repository = product_repository
        self.sales_repository = sales_repository
        self.min_sales = min_sales
        
        # Register analyzers
        self._analyzers: Dict[str, Type[BaseAnalyzer]] = {
            'style': StyleAnalyzer,
            'class': ClassAnalyzer,
            'subclass': SubclassAnalyzer,
            'collection': CollectionAnalyzer
        }
    
    def get_analyzer(self, level: str) -> Optional[BaseAnalyzer]:
        """
        Get an analyzer for the specified level.
        
        Args:
            level (str): The analysis level ('style', 'class', 'subclass', 'collection')
        
        Returns:
            Optional[BaseAnalyzer]: An analyzer instance for the specified level, or None if not found
        """
        if level not in self._analyzers:
            print(f"Unknown analyzer level: {level}")
            return None
        
        # Create and return the analyzer
        analyzer_class = self._analyzers[level]
        return analyzer_class(
            product_repository=self.product_repository,
            sales_repository=self.sales_repository,
            min_sales=self.min_sales
        )
    
    def get_all_analyzers(self) -> Dict[str, BaseAnalyzer]:
        """
        Get analyzers for all registered levels.
        
        Returns:
            Dict[str, BaseAnalyzer]: Dictionary mapping level names to analyzer instances
        """
        analyzers = {}
        for level in self._analyzers:
            analyzers[level] = self.get_analyzer(level)
        
        return analyzers