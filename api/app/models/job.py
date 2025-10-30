"""
Job listing models with duplicate detection and search interfaces.

This module defines the JobListing model for individual job postings,
along with SearchRequest/SearchResponse models for API interactions.
"""

import hashlib
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class JobListing(BaseModel):
    """
    Represents a single job listing from any source.

    This model includes duplicate detection via hash generation and
    tracks which sources have the same job listing.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    id: str = Field(
        description="Unique identifier for this job listing"
    )

    source: str = Field(
        description="Source where this job was found (e.g., 'Indeed', 'LinkedIn')"
    )

    title: str = Field(
        description="Job title"
    )

    company: str = Field(
        description="Company name"
    )

    description: Optional[str] = Field(
        default=None,
        description="Full job description"
    )

    pay: Optional[str] = Field(
        default=None,
        description="Pay/salary information if available"
    )

    location: Optional[str] = Field(
        default=None,
        description="Job location"
    )

    remote: Optional[bool] = Field(
        default=None,
        description="Whether this is a remote position"
    )

    url: str = Field(
        description="URL to the job listing"
    )

    posted_date: Optional[datetime] = Field(
        default=None,
        description="When the job was posted"
    )

    duplicate_group_id: Optional[str] = Field(
        default=None,
        description="ID linking duplicate jobs together"
    )

    duplicate_sources: List[str] = Field(
        default_factory=list,
        description="List of sources that have this same job"
    )

    source_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source-specific metadata"
    )

    def generate_duplicate_hash(self) -> str:
        """
        Generate a hash for duplicate detection.

        Creates a deterministic hash based on normalized company name,
        job title, location, and the first 50 characters of the description.
        This allows identifying the same job posted across multiple sources.

        Returns:
            MD5 hash string for duplicate detection
        """
        # Normalize company name
        company_normalized = self._normalize_text(self.company)

        # Normalize job title
        title_normalized = self._normalize_text(self.title)

        # Normalize location (if present)
        location_normalized = self._normalize_text(self.location or "")

        # Use first 50 chars of description (if present)
        description_part = ""
        if self.description:
            description_normalized = self._normalize_text(self.description)
            description_part = description_normalized[:50]

        # Combine all parts for hashing
        hash_input = f"{company_normalized}|{title_normalized}|{location_normalized}|{description_part}"

        # Generate MD5 hash (sufficient for duplicate detection, not security)
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent duplicate detection.

        Removes extra whitespace, converts to lowercase, and removes
        common punctuation to create a normalized form for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text string
        """
        # Convert to lowercase
        normalized = text.lower()

        # Remove multiple whitespaces
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove common punctuation but keep alphanumeric and spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # Strip leading/trailing whitespace
        normalized = normalized.strip()

        return normalized


class SearchRequest(BaseModel):
    """
    Request model for job search endpoints.

    Defines the parameters that can be used to search for jobs.
    """

    query: Optional[str] = Field(
        default=None,
        description="Search query string (job title, keywords, etc.)"
    )

    location: Optional[str] = Field(
        default=None,
        description="Location to search in"
    )

    remote: Optional[bool] = Field(
        default=None,
        description="Filter for remote positions"
    )

    min_pay: Optional[int] = Field(
        default=None,
        description="Minimum pay requirement"
    )

    max_pay: Optional[int] = Field(
        default=None,
        description="Maximum pay requirement"
    )


class SearchResponse(BaseModel):
    """
    Response model for job search endpoints.

    Contains the search results along with metadata about the search.
    """

    jobs: List[JobListing] = Field(
        description="List of job listings matching the search"
    )

    companies: Dict[str, Any] = Field(
        default_factory=dict,
        description="Company research data for companies in the results"
    )

    total_count: int = Field(
        description="Total number of jobs found"
    )

    filtered_count: int = Field(
        description="Number of jobs after filtering (e.g., duplicates removed)"
    )