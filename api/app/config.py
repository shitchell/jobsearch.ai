"""
Configuration settings for the Job Search AI application.

Loads settings from config.ini if it exists, otherwise uses defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from configparser import ConfigParser
from pathlib import Path
from functools import lru_cache
from typing import Optional, List


class Settings(BaseSettings):
    """Application settings with support for INI file configuration."""

    # Database configuration
    db_url: str = Field(
        default="sqlite:///./jobsearch.db",
        description="Database connection URL"
    )

    # Cache configuration
    cache_ttl_days: int = Field(
        default=30,
        description="Cache time-to-live in days"
    )

    # Provider management
    ignored_providers: List[str] = Field(
        default_factory=list,
        description="List of provider names to ignore/disable"
    )

    require_all_providers: bool = Field(
        default=False,
        description="Whether to require all configured providers to be available"
    )

    # API Keys (for future use)
    glassdoor_api_key: Optional[str] = Field(
        default=None,
        description="Glassdoor API key for company research"
    )

    indeed_api_key: Optional[str] = Field(
        default=None,
        description="Indeed API key for job searching"
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for AI-powered research"
    )

    @field_validator('ignored_providers', mode='before')
    @classmethod
    def parse_ignored_providers(cls, v):
        """
        Parse comma-separated list from string if needed.

        Handles both list input and comma-separated string input.
        """
        if isinstance(v, str):
            # Split by comma and strip whitespace
            return [p.strip() for p in v.split(',') if p.strip()]
        return v or []

    @classmethod
    def from_configparser(cls, config_path: str = "config.ini", section: str = "jobsearch") -> "Settings":
        """
        Load settings from an INI configuration file.

        Args:
            config_path: Path to the config.ini file
            section: Section name in the INI file

        Returns:
            Settings instance with values from config file or defaults
        """
        config = ConfigParser()
        settings_dict = {}

        config_file = Path(config_path)
        if config_file.exists():
            try:
                config.read(config_path)

                if config.has_section(section):
                    # Database settings
                    if config.has_option(section, "db_url"):
                        settings_dict["db_url"] = config.get(section, "db_url")

                    # Cache settings
                    if config.has_option(section, "cache_ttl_days"):
                        settings_dict["cache_ttl_days"] = config.getint(section, "cache_ttl_days")

                    # Provider management
                    if config.has_option(section, "ignored_providers"):
                        # Parse comma-separated list
                        ignored_str = config.get(section, "ignored_providers")
                        settings_dict["ignored_providers"] = [
                            p.strip() for p in ignored_str.split(',') if p.strip()
                        ]

                    if config.has_option(section, "require_all_providers"):
                        settings_dict["require_all_providers"] = config.getboolean(
                            section, "require_all_providers"
                        )

                    # API keys
                    if config.has_option(section, "glassdoor_api_key"):
                        settings_dict["glassdoor_api_key"] = config.get(section, "glassdoor_api_key")

                    if config.has_option(section, "indeed_api_key"):
                        settings_dict["indeed_api_key"] = config.get(section, "indeed_api_key")

                    if config.has_option(section, "openai_api_key"):
                        settings_dict["openai_api_key"] = config.get(section, "openai_api_key")

            except Exception as e:
                # Log the error but continue with defaults
                print(f"Warning: Error reading config file {config_path}: {e}")

        # Create Settings instance with loaded values or defaults
        return cls(**settings_dict)

    class Config:
        """Pydantic configuration."""
        env_prefix = "JOBSEARCH_"  # Support environment variables like JOBSEARCH_DB_URL
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached).

    Returns a cached Settings instance loaded from config.ini if available.
    """
    return Settings.from_configparser()