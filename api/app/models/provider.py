"""
Abstract base classes and metadata for the provider plugin system.

This module defines the interfaces that all job source and research providers
must implement, along with metadata classes for field contributions and display.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ResearchCategory(str, Enum):
    """
    Categories for company research providers based on cost and method.

    Used to prioritize which providers to call based on user preferences
    and available resources.
    """
    BASIC = "basic"           # Free or cached data
    API_CHEAP = "api_cheap"    # Low-cost API calls
    API_EXPENSIVE = "api_expensive"  # High-cost API calls
    AI = "ai"                  # AI-powered research (e.g., GPT-4)


class DisplayMetadata(BaseModel):
    """
    Metadata for how a field should be displayed in the frontend.

    Provides rendering instructions to the frontend for presenting
    company research data in an appropriate format.
    """

    type: str = Field(
        description="Display type: rating, badge, percentage, text, list, etc."
    )

    icon: Optional[str] = Field(
        default=None,
        description="Icon name or emoji for the field"
    )

    priority: str = Field(
        default="normal",
        description="Display priority: high, normal, low"
    )

    format: Optional[str] = Field(
        default=None,
        description="Format string for values (e.g., '{value}/5', '{value}%')"
    )

    max_value: Optional[float] = Field(
        default=None,
        description="Maximum value for ratings or percentages"
    )

    color_scale: Optional[Dict[str, str]] = Field(
        default=None,
        description="Color mapping for different value ranges"
    )

    invert: bool = Field(
        default=False,
        description="Whether to invert the scale (lower is better)"
    )

    list_style: Optional[str] = Field(
        default=None,
        description="Style for list display: bullet, numbered, chips"
    )

    custom_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Provider-specific custom display configuration"
    )


class FieldContribution(BaseModel):
    """
    Metadata about a field that a provider contributes.

    Describes what data a provider can contribute and how it should
    be displayed in the UI.
    """

    category: ResearchCategory = Field(
        description="Cost/method category of this contribution"
    )

    label: str = Field(
        description="Human-readable label for the field"
    )

    display: DisplayMetadata = Field(
        description="Display configuration for the field"
    )


class JobSource(ABC):
    """
    Abstract base class for job search providers.

    All job source plugins must inherit from this class and implement
    the required methods for searching and retrieving job listings.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Get the name of this job source.

        Returns:
            String identifier for this source (e.g., "Indeed", "LinkedIn")
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: Optional[str] = None,
        location: Optional[str] = None,
        remote: Optional[bool] = None,
        min_pay: Optional[int] = None,
        max_pay: Optional[int] = None,
        **kwargs: Any
    ) -> List["JobListing"]:  # Forward reference to avoid circular import
        """
        Search for jobs based on the given criteria.

        Args:
            query: Job search query string
            location: Location to search in
            remote: Whether to search for remote jobs
            min_pay: Minimum pay requirement
            max_pay: Maximum pay requirement
            **kwargs: Additional provider-specific parameters

        Returns:
            List of JobListing objects matching the search criteria
        """
        pass


class ResearchProvider(ABC):
    """
    Abstract base class for company research providers.

    All research provider plugins must inherit from this class and implement
    the required methods for researching company information.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the name of this research provider.

        Returns:
            String identifier for this provider (e.g., "Glassdoor", "Crunchbase")
        """
        pass

    @abstractmethod
    def contributions(self) -> Dict[str, FieldContribution]:
        """
        Get the fields this provider can contribute.

        Returns a dictionary mapping field names to their contribution metadata,
        describing what data this provider can supply and how it should be displayed.

        Returns:
            Dictionary mapping field names to FieldContribution metadata
        """
        pass

    @abstractmethod
    async def research(
        self,
        company_name: str,
        requested_fields: Optional[List[str]] = None,
        **kwargs: Any
    ) -> "ResearchResult":  # Forward reference to avoid circular import
        """
        Research a company and return the requested information.

        Args:
            company_name: Name of the company to research
            requested_fields: Specific fields to research, or None for all
            **kwargs: Additional provider-specific parameters

        Returns:
            ResearchResult subclass containing the researched data
        """
        pass