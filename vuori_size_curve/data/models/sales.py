"""
Sales data models.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import date


@dataclass
class SalesRecord:
    """
    Represents a sales record with product information.
    """
    style_code: str
    gender_code: str
    size_code: str
    size_order: int
    total_sales: int
    distribution_channel_code: str
    color_code: Optional[str] = None
    collection_text: Optional[str] = None
    nrf_color_text: Optional[str] = None
    product_class_text: Optional[str] = None
    product_sub_class_text: Optional[str] = None


@dataclass
class SizeCurve:
    """
    Represents a size curve with percentages by size.
    """
    item_code: str  # Style, class, subclass, or collection code
    gender_code: str
    size_template_name: str
    level_type: str  # "STYLE", "CLASS", "SUBCLASS", "COLLECTION"
    channel: Optional[str] = None  # Distribution channel
    size_percentages: Dict[str, float] = field(default_factory=dict)  # Key: size_code, Value: percentage


@dataclass
class FilterCriteria:
    """
    Represents filtering criteria for data analysis.
    """
    start_date: str = "2022-01-01"
    min_sales: int = 0
    channels_filter: Optional[List[str]] = None
    gender_filter: Optional[str] = None
    class_filter: Optional[str] = None
    subclass_filter: Optional[str] = None
    style_filter: Optional[str] = None