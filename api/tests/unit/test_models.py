"""
Comprehensive tests for Core Pydantic Models (Chunk 0.3).

Tests cover:
- JobListing duplicate hash generation
- Pydantic validation (required fields, type checking)
- Model serialization (model_dump(), JSON)
- Abstract base class enforcement
- Dynamic fields on Company model
- Settings configuration parsing
- Import validation (no circular imports)
"""

import pytest
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Test imports to catch circular dependencies early
from app.models import (
    JobListing,
    SearchRequest,
    SearchResponse,
    Company,
    ResearchResult,
    ResearchCategory,
    DisplayMetadata,
    FieldContribution,
    JobSource,
    ResearchProvider,
    Settings,
    DisplayType,
    DisplayPriority,
    DisplayConfig,
)
from pydantic import ValidationError


class TestJobListing:
    """Test JobListing model functionality."""

    def test_job_listing_creation(self):
        """Test creating a valid JobListing instance."""
        job = JobListing(
            id="job123",
            source="Indeed",
            title="Software Engineer",
            company="TechCorp",
            description="Great opportunity for a software engineer...",
            pay="$100k-$150k",
            location="San Francisco, CA",
            remote=True,
            url="https://example.com/job123",
            posted_date=datetime(2024, 1, 15)
        )

        assert job.id == "job123"
        assert job.source == "Indeed"
        assert job.title == "Software Engineer"
        assert job.company == "TechCorp"
        assert job.remote is True

    def test_job_listing_required_fields(self):
        """Test that required fields are enforced."""
        # Missing required field 'source'
        with pytest.raises(ValidationError) as exc_info:
            JobListing(
                id="job123",
                title="Engineer",
                company="Corp",
                url="https://example.com/job"
            )

        errors = exc_info.value.errors()
        field_names = [e['loc'][0] for e in errors]
        assert 'source' in field_names

    def test_job_listing_optional_fields(self):
        """Test that optional fields work with defaults."""
        job = JobListing(
            id="job123",
            source="Indeed",
            title="Engineer",
            company="Corp",
            url="https://example.com/job"
        )

        assert job.description is None
        assert job.pay is None
        assert job.location is None
        assert job.remote is None
        assert job.posted_date is None
        assert job.duplicate_group_id is None
        assert job.duplicate_sources == []
        assert job.source_metadata == {}

    def test_duplicate_hash_generation_consistency(self):
        """Test that identical jobs generate the same hash."""
        job1 = JobListing(
            id="job1",
            source="Indeed",
            title="Software Engineer",
            company="TechCorp",
            description="We are seeking a talented software engineer to join our team...",
            location="San Francisco, CA",
            url="https://indeed.com/job1"
        )

        job2 = JobListing(
            id="job2",
            source="LinkedIn",  # Different source
            title="Software Engineer",
            company="TechCorp",
            description="We are seeking a talented software engineer to join our team...",
            location="San Francisco, CA",
            url="https://linkedin.com/job2"  # Different URL
        )

        # Should generate same hash despite different source/id/url
        hash1 = job1.generate_duplicate_hash()
        hash2 = job2.generate_duplicate_hash()

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length

    def test_duplicate_hash_case_insensitive(self):
        """Test that hash generation is case-insensitive."""
        job1 = JobListing(
            id="1",
            source="Indeed",
            title="Software Engineer",
            company="TechCorp",
            location="San Francisco",
            url="https://example.com/1"
        )

        job2 = JobListing(
            id="2",
            source="Indeed",
            title="software engineer",  # Different case
            company="techcorp",  # Different case
            location="san francisco",  # Different case
            url="https://example.com/2"
        )

        assert job1.generate_duplicate_hash() == job2.generate_duplicate_hash()

    def test_duplicate_hash_whitespace_normalization(self):
        """Test that hash normalizes whitespace correctly."""
        job1 = JobListing(
            id="1",
            source="Indeed",
            title="Software   Engineer",  # Multiple spaces
            company="Tech Corp",
            location="San Francisco",
            url="https://example.com/1"
        )

        job2 = JobListing(
            id="2",
            source="Indeed",
            title="Software Engineer",  # Single space
            company="Tech Corp",
            location="San Francisco",
            url="https://example.com/2"
        )

        assert job1.generate_duplicate_hash() == job2.generate_duplicate_hash()

    def test_duplicate_hash_punctuation_normalization(self):
        """Test that hash removes punctuation correctly."""
        job1 = JobListing(
            id="1",
            source="Indeed",
            title="Sr. Software Engineer",
            company="Tech-Corp, Inc.",
            location="San Francisco, CA",
            url="https://example.com/1"
        )

        job2 = JobListing(
            id="2",
            source="Indeed",
            title="Sr Software Engineer",
            company="TechCorp Inc",
            location="San Francisco CA",
            url="https://example.com/2"
        )

        assert job1.generate_duplicate_hash() == job2.generate_duplicate_hash()

    def test_duplicate_hash_description_first_50_chars(self):
        """Test that only first 50 chars of description used in hash."""
        base_desc = "This is a great opportunity for an experienced engineer"

        job1 = JobListing(
            id="1",
            source="Indeed",
            title="Engineer",
            company="Corp",
            description=base_desc + " with 5+ years experience",
            url="https://example.com/1"
        )

        job2 = JobListing(
            id="2",
            source="Indeed",
            title="Engineer",
            company="Corp",
            description=base_desc + " who loves coding",
            url="https://example.com/2"
        )

        # Should have same hash since first 50 chars are the same
        assert job1.generate_duplicate_hash() == job2.generate_duplicate_hash()

    def test_duplicate_hash_different_jobs(self):
        """Test that different jobs generate different hashes."""
        job1 = JobListing(
            id="1",
            source="Indeed",
            title="Software Engineer",
            company="TechCorp",
            url="https://example.com/1"
        )

        job2 = JobListing(
            id="2",
            source="Indeed",
            title="Data Scientist",  # Different title
            company="TechCorp",
            url="https://example.com/2"
        )

        assert job1.generate_duplicate_hash() != job2.generate_duplicate_hash()

    def test_job_listing_serialization(self):
        """Test model_dump() serialization."""
        job = JobListing(
            id="job123",
            source="Indeed",
            title="Engineer",
            company="Corp",
            url="https://example.com/job",
            duplicate_sources=["Indeed", "LinkedIn"]
        )

        data = job.model_dump()

        assert isinstance(data, dict)
        assert data['id'] == "job123"
        assert data['source'] == "Indeed"
        assert data['duplicate_sources'] == ["Indeed", "LinkedIn"]

    def test_job_listing_json_serialization(self):
        """Test JSON serialization with datetime."""
        job = JobListing(
            id="job123",
            source="Indeed",
            title="Engineer",
            company="Corp",
            url="https://example.com/job",
            posted_date=datetime(2024, 1, 15, 10, 30)
        )

        # Serialize to JSON
        json_str = job.model_dump_json()
        assert isinstance(json_str, str)

        # Parse back
        data = json.loads(json_str)
        assert data['id'] == "job123"
        assert '2024-01-15' in data['posted_date']


class TestSearchRequest:
    """Test SearchRequest model."""

    def test_search_request_all_optional(self):
        """Test that all fields are optional."""
        request = SearchRequest()

        assert request.query is None
        assert request.location is None
        assert request.remote is None
        assert request.min_pay is None
        assert request.max_pay is None

    def test_search_request_with_values(self):
        """Test SearchRequest with all fields populated."""
        request = SearchRequest(
            query="python developer",
            location="New York",
            remote=True,
            min_pay=80000,
            max_pay=120000
        )

        assert request.query == "python developer"
        assert request.location == "New York"
        assert request.remote is True
        assert request.min_pay == 80000
        assert request.max_pay == 120000

    def test_search_request_serialization(self):
        """Test SearchRequest serialization."""
        request = SearchRequest(
            query="engineer",
            location="SF"
        )

        data = request.model_dump()
        assert data['query'] == "engineer"
        assert data['location'] == "SF"


class TestSearchResponse:
    """Test SearchResponse model."""

    def test_search_response_creation(self):
        """Test creating a SearchResponse."""
        jobs = [
            JobListing(
                id="1",
                source="Indeed",
                title="Engineer",
                company="Corp",
                url="https://example.com/1"
            )
        ]

        response = SearchResponse(
            jobs=jobs,
            companies={"Corp": {"rating": 4.5}},
            total_count=100,
            filtered_count=50
        )

        assert len(response.jobs) == 1
        assert response.jobs[0].id == "1"
        assert response.companies["Corp"]["rating"] == 4.5
        assert response.total_count == 100
        assert response.filtered_count == 50

    def test_search_response_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            SearchResponse(
                companies={}
                # Missing required fields: jobs, total_count, filtered_count
            )


class TestCompany:
    """Test Company model with dynamic fields."""

    def test_company_creation(self):
        """Test creating a Company instance."""
        company = Company(name="TechCorp")

        assert company.name == "TechCorp"
        assert company.cached_date is None

    def test_company_dynamic_fields(self):
        """Test that Company accepts arbitrary fields (extra='allow')."""
        company = Company(
            name="TechCorp",
            rating=4.5,
            employee_count=1000,
            founded_year=2010,
            custom_field="custom_value"
        )

        assert company.name == "TechCorp"
        assert company.rating == 4.5
        assert company.employee_count == 1000
        assert company.founded_year == 2010
        assert company.custom_field == "custom_value"

    def test_company_merge_research(self):
        """Test merging ResearchResult into Company."""
        # Create a custom ResearchResult subclass
        class MockResearchResult(ResearchResult):
            rating: float
            employee_count: int
            benefits: str

        company = Company(name="TechCorp")
        research = MockResearchResult(
            rating=4.5,
            employee_count=1000,
            benefits="Great benefits"
        )

        # Merge research into company
        company.merge_research(research)

        # Check that fields were added
        assert company.rating == 4.5
        assert company.employee_count == 1000
        assert company.benefits == "Great benefits"
        assert company.cached_date is not None

    def test_company_merge_research_updates_cached_date(self):
        """Test that merge_research updates the cached_date."""
        old_date = datetime(2024, 1, 1)
        company = Company(name="TechCorp", cached_date=old_date)

        research = ResearchResult()
        company.merge_research(research)

        # Cached date should be updated
        assert company.cached_date != old_date
        assert company.cached_date is not None

    def test_company_serialization_with_dynamic_fields(self):
        """Test serialization of Company with dynamic fields."""
        company = Company(
            name="TechCorp",
            rating=4.5,
            custom_field="value"
        )

        data = company.model_dump()

        assert data['name'] == "TechCorp"
        assert data['rating'] == 4.5
        assert data['custom_field'] == "value"


class TestResearchResult:
    """Test ResearchResult base class."""

    def test_research_result_empty_base(self):
        """Test that ResearchResult is an empty base class."""
        result = ResearchResult()
        assert isinstance(result, ResearchResult)

    def test_research_result_subclassing(self):
        """Test creating a subclass of ResearchResult."""
        class CustomResult(ResearchResult):
            field1: str
            field2: int

        result = CustomResult(field1="value", field2=42)

        assert result.field1 == "value"
        assert result.field2 == 42
        assert isinstance(result, ResearchResult)


class TestProviderEnums:
    """Test provider-related enums."""

    def test_research_category_values(self):
        """Test ResearchCategory enum values."""
        assert ResearchCategory.BASIC == "basic"
        assert ResearchCategory.API_CHEAP == "api_cheap"
        assert ResearchCategory.API_EXPENSIVE == "api_expensive"
        assert ResearchCategory.AI == "ai"

    def test_research_category_membership(self):
        """Test that values are valid enum members."""
        assert "basic" in [c.value for c in ResearchCategory]
        assert "api_cheap" in [c.value for c in ResearchCategory]

    def test_display_type_values(self):
        """Test DisplayType enum values."""
        assert DisplayType.RATING == "rating"
        assert DisplayType.BADGE == "badge"
        assert DisplayType.PERCENTAGE == "percentage"
        assert DisplayType.TEXT == "text"
        assert DisplayType.LIST == "list"

    def test_display_priority_values(self):
        """Test DisplayPriority enum values."""
        assert DisplayPriority.CRITICAL == "critical"
        assert DisplayPriority.HIGH == "high"
        assert DisplayPriority.NORMAL == "normal"
        assert DisplayPriority.LOW == "low"
        assert DisplayPriority.HIDDEN == "hidden"


class TestDisplayMetadata:
    """Test DisplayMetadata model."""

    def test_display_metadata_creation(self):
        """Test creating DisplayMetadata."""
        metadata = DisplayMetadata(
            type="rating",
            icon="⭐",
            priority="high",
            format="{value}/5",
            max_value=5.0
        )

        assert metadata.type == "rating"
        assert metadata.icon == "⭐"
        assert metadata.priority == "high"
        assert metadata.format == "{value}/5"
        assert metadata.max_value == 5.0

    def test_display_metadata_defaults(self):
        """Test DisplayMetadata default values."""
        metadata = DisplayMetadata(type="text")

        assert metadata.type == "text"
        assert metadata.icon is None
        assert metadata.priority == "normal"
        assert metadata.invert is False


class TestFieldContribution:
    """Test FieldContribution model."""

    def test_field_contribution_creation(self):
        """Test creating a FieldContribution."""
        display = DisplayMetadata(
            type="rating",
            priority="high"
        )

        contribution = FieldContribution(
            category=ResearchCategory.API_CHEAP,
            label="Company Rating",
            display=display
        )

        assert contribution.category == ResearchCategory.API_CHEAP
        assert contribution.label == "Company Rating"
        assert contribution.display.type == "rating"


class TestDisplayConfig:
    """Test DisplayConfig model."""

    def test_display_config_creation(self):
        """Test creating a DisplayConfig."""
        config = DisplayConfig(
            field_name="rating",
            display_type=DisplayType.RATING,
            label="Company Rating",
            priority=DisplayPriority.HIGH,
            icon="⭐",
            max_value=5.0
        )

        assert config.field_name == "rating"
        assert config.display_type == DisplayType.RATING
        assert config.label == "Company Rating"
        assert config.priority == DisplayPriority.HIGH
        assert config.icon == "⭐"
        assert config.max_value == 5.0

    def test_display_config_apply_format(self):
        """Test DisplayConfig.apply_format() method."""
        config = DisplayConfig(
            field_name="rating",
            display_type=DisplayType.RATING,
            label="Rating",
            format_string="{:.1f}",
            suffix="/5"
        )

        formatted = config.apply_format(4.567)
        assert formatted == "4.6/5"

    def test_display_config_apply_format_with_prefix_suffix(self):
        """Test apply_format with prefix and suffix."""
        config = DisplayConfig(
            field_name="salary",
            display_type=DisplayType.CURRENCY,
            label="Salary",
            prefix="$",
            suffix="K"
        )

        formatted = config.apply_format(100)
        assert formatted == "$100K"

    def test_display_config_apply_format_none_value(self):
        """Test apply_format with None value."""
        config = DisplayConfig(
            field_name="field",
            display_type=DisplayType.TEXT,
            label="Field",
            empty_text="N/A"
        )

        formatted = config.apply_format(None)
        assert formatted == "N/A"


class TestAbstractBaseClasses:
    """Test that abstract base classes enforce interface."""

    def test_job_source_cannot_instantiate(self):
        """Test that JobSource ABC cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            JobSource()

        assert "abstract" in str(exc_info.value).lower()

    def test_research_provider_cannot_instantiate(self):
        """Test that ResearchProvider ABC cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            ResearchProvider()

        assert "abstract" in str(exc_info.value).lower()

    def test_job_source_subclass_must_implement_methods(self):
        """Test that JobSource subclass must implement abstract methods."""
        # Create incomplete subclass
        with pytest.raises(TypeError):
            class IncompleteSource(JobSource):
                pass

            IncompleteSource()

    def test_job_source_concrete_implementation(self):
        """Test creating a concrete JobSource implementation."""
        class MockJobSource(JobSource):
            @property
            def source_name(self) -> str:
                return "MockSource"

            async def search(self, **kwargs) -> list:
                return []

        # Should not raise
        source = MockJobSource()
        assert source.source_name == "MockSource"

    def test_research_provider_concrete_implementation(self):
        """Test creating a concrete ResearchProvider implementation."""
        class MockResearchProvider(ResearchProvider):
            @property
            def provider_name(self) -> str:
                return "MockProvider"

            def contributions(self) -> Dict[str, FieldContribution]:
                return {}

            async def research(self, company_name: str, **kwargs) -> ResearchResult:
                return ResearchResult()

        # Should not raise
        provider = MockResearchProvider()
        assert provider.provider_name == "MockProvider"


class TestSettings:
    """Test Settings configuration model."""

    def test_settings_defaults(self):
        """Test Settings default values."""
        settings = Settings()

        assert settings.db_url == "sqlite:///./jobsearch.db"
        assert settings.cache_ttl_days == 30
        assert settings.ignored_providers == []
        assert settings.require_all_providers is False
        assert settings.glassdoor_api_key is None
        assert settings.indeed_api_key is None
        assert settings.openai_api_key is None

    def test_settings_from_configparser_missing_file(self):
        """Test Settings.from_configparser() with missing file."""
        settings = Settings.from_configparser("nonexistent.ini")

        # Should return defaults
        assert settings.db_url == "sqlite:///./jobsearch.db"
        assert settings.cache_ttl_days == 30

    def test_settings_from_configparser_valid_file(self):
        """Test Settings.from_configparser() with valid INI file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("[jobsearch]\n")
            f.write("db_url = postgresql://localhost/testdb\n")
            f.write("cache_ttl_days = 60\n")
            f.write("ignored_providers = provider1, provider2, provider3\n")
            f.write("require_all_providers = true\n")
            f.write("glassdoor_api_key = test_key_123\n")
            f.write("openai_api_key = sk-test-key\n")
            config_path = f.name

        try:
            settings = Settings.from_configparser(config_path)

            assert settings.db_url == "postgresql://localhost/testdb"
            assert settings.cache_ttl_days == 60
            assert settings.ignored_providers == ["provider1", "provider2", "provider3"]
            assert settings.require_all_providers is True
            assert settings.glassdoor_api_key == "test_key_123"
            assert settings.openai_api_key == "sk-test-key"
        finally:
            Path(config_path).unlink()

    def test_settings_ignored_providers_parsing(self):
        """Test parsing comma-separated ignored_providers list."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("[jobsearch]\n")
            f.write("ignored_providers = provider1, provider2 , provider3\n")
            config_path = f.name

        try:
            settings = Settings.from_configparser(config_path)

            # Should parse and strip whitespace
            assert settings.ignored_providers == ["provider1", "provider2", "provider3"]
        finally:
            Path(config_path).unlink()

    def test_settings_ignored_providers_field_validator(self):
        """Test field validator for ignored_providers."""
        # Test with string input
        settings = Settings(ignored_providers="foo, bar, baz")
        assert settings.ignored_providers == ["foo", "bar", "baz"]

        # Test with list input
        settings = Settings(ignored_providers=["foo", "bar"])
        assert settings.ignored_providers == ["foo", "bar"]

    def test_settings_partial_config(self):
        """Test Settings with partial configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("[jobsearch]\n")
            f.write("db_url = custom://db\n")
            # Only some fields set
            config_path = f.name

        try:
            settings = Settings.from_configparser(config_path)

            # Should have custom value
            assert settings.db_url == "custom://db"
            # Should have defaults for unset fields
            assert settings.cache_ttl_days == 30
            assert settings.ignored_providers == []
        finally:
            Path(config_path).unlink()


class TestImports:
    """Test that imports work correctly without circular dependencies."""

    def test_all_models_importable(self):
        """Test that all models can be imported from app.models."""
        from app.models import (
            JobListing,
            SearchRequest,
            SearchResponse,
            Company,
            ResearchResult,
            ResearchCategory,
            DisplayMetadata,
            FieldContribution,
            JobSource,
            ResearchProvider,
            Settings,
            DisplayType,
            DisplayPriority,
            DisplayConfig,
        )

        # If we got here, imports worked
        assert JobListing is not None
        assert Company is not None
        assert ResearchProvider is not None

    def test_no_circular_imports(self):
        """Test that importing models doesn't cause circular import errors."""
        # This test succeeds if no ImportError is raised
        import app.models.job
        import app.models.company
        import app.models.provider
        import app.models.display
        import app.config

        assert True


class TestFastAPICompatibility:
    """Test that models work with FastAPI patterns."""

    def test_search_request_as_request_body(self):
        """Test that SearchRequest can be used as FastAPI request body."""
        # Simulate FastAPI request body parsing
        request_data = {
            "query": "python developer",
            "location": "Remote",
            "remote": True
        }

        request = SearchRequest(**request_data)

        assert request.query == "python developer"
        assert request.location == "Remote"
        assert request.remote is True

    def test_search_response_serialization(self):
        """Test that SearchResponse can be serialized for FastAPI response."""
        jobs = [
            JobListing(
                id="1",
                source="Indeed",
                title="Engineer",
                company="Corp",
                url="https://example.com/1"
            )
        ]

        response = SearchResponse(
            jobs=jobs,
            companies={},
            total_count=1,
            filtered_count=1
        )

        # Serialize to dict (FastAPI does this)
        data = response.model_dump()

        assert 'jobs' in data
        assert 'companies' in data
        assert data['total_count'] == 1


class TestDatabaseCompatibility:
    """Test that models are compatible with database operations."""

    def test_job_listing_to_dict_for_db(self):
        """Test converting JobListing to dict for database storage."""
        job = JobListing(
            id="job123",
            source="Indeed",
            title="Engineer",
            company="Corp",
            url="https://example.com/job",
            posted_date=datetime(2024, 1, 15)
        )

        # Convert to dict for database
        data = job.model_dump()

        assert isinstance(data, dict)
        assert 'id' in data
        assert 'source' in data
        # Datetime should be serializable
        assert 'posted_date' in data

    def test_company_dynamic_fields_to_json(self):
        """Test that Company with dynamic fields can be stored as JSON."""
        company = Company(
            name="TechCorp",
            rating=4.5,
            custom_field="value"
        )

        # Convert to JSON (for database JSON field)
        json_str = company.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed['name'] == "TechCorp"
        assert parsed['rating'] == 4.5
        assert parsed['custom_field'] == "value"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_job_listing_empty_strings(self):
        """Test JobListing with empty strings (should be stripped)."""
        job = JobListing(
            id="job123",
            source="  Indeed  ",  # Whitespace should be stripped
            title="  Engineer  ",
            company="Corp",
            url="https://example.com/job"
        )

        # Whitespace should be stripped due to str_strip_whitespace=True
        assert job.source == "Indeed"
        assert job.title == "Engineer"

    def test_duplicate_hash_with_none_location(self):
        """Test duplicate hash generation when location is None."""
        job = JobListing(
            id="1",
            source="Indeed",
            title="Engineer",
            company="Corp",
            location=None,  # None location
            url="https://example.com/1"
        )

        # Should not raise
        hash_value = job.generate_duplicate_hash()
        assert isinstance(hash_value, str)

    def test_duplicate_hash_with_none_description(self):
        """Test duplicate hash generation when description is None."""
        job = JobListing(
            id="1",
            source="Indeed",
            title="Engineer",
            company="Corp",
            description=None,  # None description
            url="https://example.com/1"
        )

        # Should not raise
        hash_value = job.generate_duplicate_hash()
        assert isinstance(hash_value, str)

    def test_company_merge_with_none_fields(self):
        """Test Company.merge_research() with ResearchResult containing None fields."""
        class TestResult(ResearchResult):
            field1: str = None
            field2: int = None

        company = Company(name="Corp")
        result = TestResult()

        # Should not raise (exclude_none=True in merge_research)
        company.merge_research(result)
        assert company.name == "Corp"

    def test_display_config_apply_format_invalid_format_string(self):
        """Test apply_format with invalid format string."""
        config = DisplayConfig(
            field_name="field",
            display_type=DisplayType.TEXT,
            label="Field",
            format_string="{invalid}"  # Invalid format
        )

        # Should fall back to str(value)
        formatted = config.apply_format(42)
        assert formatted == "42"

    def test_settings_malformed_config_file(self):
        """Test Settings with malformed config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("This is not valid INI syntax\n")
            f.write("Random text\n")
            config_path = f.name

        try:
            # Should not raise, should return defaults
            settings = Settings.from_configparser(config_path)
            assert settings.db_url == "sqlite:///./jobsearch.db"
        finally:
            Path(config_path).unlink()
