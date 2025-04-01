"""
Product data models.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SizeTemplate:
    """
    Represents a size template for a product style.
    """
    name: str  # Template name (e.g., "XS-S-M-L-XL")
    sizes: List[str]  # Ordered list of sizes (e.g., ["XS", "S", "M", "L", "XL"])
    
    def __post_init__(self):
        # If template name isn't set, generate it from sizes
        if not self.name and self.sizes:
            self.name = "-".join(self.sizes)


@dataclass
class Product:
    """
    Represents a product with its size information.
    """
    style_code: str
    gender_code: str
    color_code: Optional[str] = None
    collection_text: Optional[str] = None
    nrf_color_text: Optional[str] = None
    product_class_text: Optional[str] = None
    product_sub_class_text: Optional[str] = None
    size_code: Optional[str] = None
    size_order: Optional[int] = None


@dataclass
class StyleTemplate:
    """
    Maps a style to its size template.
    """
    style_code: str
    size_template: SizeTemplate
    gender_code: Optional[str] = None
    color_code: Optional[str] = None  # Added color_code to handle style+color templates