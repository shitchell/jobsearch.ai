"""
SQLAlchemy database models for the Job Search AI application.
"""

from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from typing import Any, Dict, Optional

# Create the declarative base for all models
Base = declarative_base()


class CompanyCache(Base):
    """
    Cache for individual company research fields.

    Uses a composite primary key of (name, field_name) to enable
    per-field caching. This allows partial updates and field-specific
    TTL management.

    Attributes:
        name: Company name (part of composite primary key)
        field_name: Name of the cached field (part of composite primary key)
        value: JSON-serialized value of the field
        cached_at: Timestamp when the field was cached
    """

    __tablename__ = "company_cache"

    # Composite primary key
    name = Column(String(255), primary_key=True, nullable=False)
    field_name = Column(String(100), primary_key=True, nullable=False)

    # Cached data and metadata
    value = Column(JSON, nullable=True)
    cached_at = Column(DateTime, nullable=False, default=func.now())

    def __repr__(self) -> str:
        """String representation of CompanyCache instance."""
        return f"<CompanyCache(name='{self.name}', field='{self.field_name}', cached_at={self.cached_at})>"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"CompanyCache[{self.name}:{self.field_name}]"

    @property
    def age_days(self) -> float:
        """Calculate the age of the cached data in days."""
        if not self.cached_at:
            return float('inf')
        delta = datetime.utcnow() - self.cached_at
        return delta.total_seconds() / 86400

    def is_expired(self, ttl_days: int = 30) -> bool:
        """
        Check if the cached data is expired based on TTL.

        Args:
            ttl_days: Time-to-live in days

        Returns:
            True if the data is older than ttl_days, False otherwise
        """
        return self.age_days > ttl_days