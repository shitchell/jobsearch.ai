"""
Display configuration and formatting metadata for frontend rendering.

This module defines display-related enums and configurations that guide
how data should be presented in the user interface.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class DisplayType(str, Enum):
    """
    Types of display components for rendering data.

    Each type corresponds to a specific UI component or visualization method.
    """
    RATING = "rating"          # Star rating or numeric scale
    BADGE = "badge"            # Status badge or chip
    PERCENTAGE = "percentage"  # Percentage bar or value
    TEXT = "text"              # Plain text
    LIST = "list"              # Bulleted or numbered list
    CHART = "chart"            # Chart or graph visualization
    BOOLEAN = "boolean"        # Yes/No, True/False indicator
    CURRENCY = "currency"      # Monetary value
    DATE = "date"              # Date/time display
    LINK = "link"              # Clickable URL
    TAG_LIST = "tag_list"      # List of tags/chips


class DisplayPriority(str, Enum):
    """
    Priority levels for field display in the UI.

    Used to determine the prominence and order of fields.
    """
    CRITICAL = "critical"  # Must be shown prominently
    HIGH = "high"          # Important information
    NORMAL = "normal"      # Standard priority
    LOW = "low"            # Less important
    HIDDEN = "hidden"      # Available but not shown by default


class DisplayConfig(BaseModel):
    """
    Complete display configuration for a field.

    Provides all necessary information for the frontend to render
    a field appropriately, including type, formatting, and styling.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    field_name: str = Field(
        description="Name of the field this configuration applies to"
    )

    display_type: DisplayType = Field(
        description="Type of UI component to use"
    )

    label: str = Field(
        description="Human-readable label for the field"
    )

    priority: DisplayPriority = Field(
        default=DisplayPriority.NORMAL,
        description="Display priority for UI ordering"
    )

    icon: Optional[str] = Field(
        default=None,
        description="Icon identifier or emoji"
    )

    tooltip: Optional[str] = Field(
        default=None,
        description="Tooltip text for additional context"
    )

    # Formatting options
    format_string: Optional[str] = Field(
        default=None,
        description="Python format string for value display (e.g., '{:.2f}%')"
    )

    prefix: Optional[str] = Field(
        default=None,
        description="Text to show before the value"
    )

    suffix: Optional[str] = Field(
        default=None,
        description="Text to show after the value"
    )

    # Value constraints and scaling
    min_value: Optional[float] = Field(
        default=None,
        description="Minimum value for scales and ratings"
    )

    max_value: Optional[float] = Field(
        default=None,
        description="Maximum value for scales and ratings"
    )

    invert_scale: bool = Field(
        default=False,
        description="Whether lower values are better (e.g., ranking)"
    )

    # Color configuration
    color_scheme: Optional[str] = Field(
        default=None,
        description="Named color scheme: 'traffic_light', 'gradient', 'category'"
    )

    color_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom color mapping for values or ranges"
    )

    # List display options
    list_style: Optional[str] = Field(
        default=None,
        description="Style for list display: 'bullet', 'numbered', 'chips', 'cards'"
    )

    max_items: Optional[int] = Field(
        default=None,
        description="Maximum number of list items to show initially"
    )

    # Conditional display
    show_if_empty: bool = Field(
        default=False,
        description="Whether to show field even when value is None or empty"
    )

    empty_text: Optional[str] = Field(
        default=None,
        description="Text to display when value is empty"
    )

    # Grouping and sections
    section: Optional[str] = Field(
        default=None,
        description="Section name for grouping related fields"
    )

    column: Optional[int] = Field(
        default=None,
        description="Column number for multi-column layouts"
    )

    # Custom provider configuration
    custom_props: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional provider-specific display properties"
    )

    def apply_format(self, value: Any) -> str:
        """
        Apply formatting to a value based on this configuration.

        Args:
            value: The value to format

        Returns:
            Formatted string representation of the value
        """
        if value is None:
            return self.empty_text or ""

        # Apply format string if provided
        if self.format_string:
            try:
                formatted = self.format_string.format(value)
            except (ValueError, KeyError):
                formatted = str(value)
        else:
            formatted = str(value)

        # Add prefix/suffix
        if self.prefix:
            formatted = f"{self.prefix}{formatted}"
        if self.suffix:
            formatted = f"{formatted}{self.suffix}"

        return formatted