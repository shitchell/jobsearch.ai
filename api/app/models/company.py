"""
Company models with dynamic field support for flexible provider data.

This module defines the Company model which can store arbitrary fields
from various research providers, and the ResearchResult base class
for provider-specific research data.
"""

from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict


class ResearchResult(BaseModel):
    """
    Base class for provider-specific research results.

    Research providers should subclass this to define their specific
    result structure. The Company model will merge these results
    into its dynamic fields.
    """
    pass


class Company(BaseModel):
    """
    Company model with dynamic field support.

    This model stores basic company information and can accept arbitrary
    additional fields from research providers. The extra='allow' configuration
    enables storing any field not explicitly defined.

    Attributes:
        name: Company name (required)
        cached_date: When the company data was last cached
        **extra: Any additional fields from research providers
    """

    model_config = ConfigDict(
        extra='allow',  # Allow arbitrary fields from providers
        str_strip_whitespace=True,
        validate_assignment=True
    )

    name: str = Field(
        description="Company name"
    )

    cached_date: Optional[datetime] = Field(
        default=None,
        description="When this company data was last cached"
    )

    def merge_research(self, result: ResearchResult) -> "Company":
        """
        Merge research results into this company instance.

        Updates the company with new fields from the research result.
        This allows incremental building of company data from multiple
        research providers.

        Args:
            result: ResearchResult instance containing new data

        Returns:
            Self with merged data
        """
        # Get the research result as a dictionary
        result_dict = result.model_dump(exclude_none=True)

        # Update this company's fields with the research data
        for field_name, value in result_dict.items():
            setattr(self, field_name, value)

        # Update the cached date to now
        self.cached_date = datetime.utcnow()

        return self