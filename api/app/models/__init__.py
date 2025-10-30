"""
Models package initialization and exports for convenient importing.

This module centralizes all model exports so they can be imported
easily throughout the application:
    from app.models import JobListing, Company, Settings
"""

# Job-related models
from .job import JobListing, SearchRequest, SearchResponse

# Company models
from .company import Company, ResearchResult

# Provider interfaces
from .provider import (
    ResearchCategory,
    DisplayMetadata,
    FieldContribution,
    JobSource,
    ResearchProvider
)

# Display configuration
from .display import DisplayType, DisplayPriority, DisplayConfig

# Settings model
from .config import Settings

# Export all public models
__all__ = [
    # Job models
    "JobListing",
    "SearchRequest",
    "SearchResponse",
    # Company models
    "Company",
    "ResearchResult",
    # Provider interfaces
    "ResearchCategory",
    "DisplayMetadata",
    "FieldContribution",
    "JobSource",
    "ResearchProvider",
    # Display models
    "DisplayType",
    "DisplayPriority",
    "DisplayConfig",
    # Configuration
    "Settings",
]