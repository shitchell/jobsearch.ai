"""
Unit tests for database models and CRUD operations.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
import json
import tempfile
import os
from pathlib import Path

from app.db.models import CompanyCache
from app.config import get_settings, Settings


class TestCompanyCache:
    """Test suite for CompanyCache model."""

    def test_create_company_cache_entry(self, test_db: Session):
        """Test creating a new CompanyCache entry."""
        # Create a new cache entry
        cache_entry = CompanyCache(
            name="TechCorp",
            field_name="scam_score",
            value={"score": 85, "source": "glassdoor"}
        )

        # Add to database
        test_db.add(cache_entry)
        test_db.commit()

        # Query and verify
        retrieved = test_db.query(CompanyCache).filter_by(
            name="TechCorp",
            field_name="scam_score"
        ).first()

        assert retrieved is not None
        assert retrieved.name == "TechCorp"
        assert retrieved.field_name == "scam_score"
        assert retrieved.value == {"score": 85, "source": "glassdoor"}
        assert retrieved.cached_at is not None
        assert isinstance(retrieved.cached_at, datetime)

    def test_composite_primary_key(self, test_db: Session):
        """Test that composite primary key allows multiple fields for same company."""
        # Create multiple cache entries for the same company
        entry1 = CompanyCache(
            name="TechCorp",
            field_name="scam_score",
            value={"score": 85}
        )
        entry2 = CompanyCache(
            name="TechCorp",
            field_name="reviews",
            value={"average": 4.2, "count": 500}
        )
        entry3 = CompanyCache(
            name="AnotherCorp",
            field_name="scam_score",
            value={"score": 60}
        )

        test_db.add_all([entry1, entry2, entry3])
        test_db.commit()

        # Query all TechCorp entries
        techcorp_entries = test_db.query(CompanyCache).filter_by(
            name="TechCorp"
        ).all()

        assert len(techcorp_entries) == 2
        field_names = [entry.field_name for entry in techcorp_entries]
        assert "scam_score" in field_names
        assert "reviews" in field_names

    def test_composite_key_uniqueness_constraint(self, test_db: Session):
        """Test that duplicate (name, field_name) pairs are not allowed."""
        # Create first entry
        entry1 = CompanyCache(
            name="TechCorp",
            field_name="scam_score",
            value={"score": 85}
        )
        test_db.add(entry1)
        test_db.commit()

        # Try to create duplicate entry
        entry2 = CompanyCache(
            name="TechCorp",
            field_name="scam_score",  # Same name and field_name
            value={"score": 90}  # Different value
        )
        test_db.add(entry2)

        # Should raise integrity error
        with pytest.raises(IntegrityError):
            test_db.commit()

    def test_json_field_storage(self, test_db: Session):
        """Test that JSON field correctly stores and retrieves complex data."""
        test_data = {
            "ratings": {
                "overall": 4.5,
                "culture": 4.2,
                "compensation": 3.8
            },
            "reviews": [
                {"id": 1, "text": "Great company"},
                {"id": 2, "text": "Good benefits"}
            ],
            "metadata": {
                "last_updated": "2024-01-15",
                "source": "glassdoor"
            }
        }

        cache_entry = CompanyCache(
            name="DataCorp",
            field_name="comprehensive_data",
            value=test_data
        )

        test_db.add(cache_entry)
        test_db.commit()

        # Query and verify
        retrieved = test_db.query(CompanyCache).filter_by(
            name="DataCorp",
            field_name="comprehensive_data"
        ).first()

        assert retrieved.value == test_data
        assert retrieved.value["ratings"]["overall"] == 4.5
        assert len(retrieved.value["reviews"]) == 2

    def test_update_cache_entry(self, test_db: Session):
        """Test updating an existing cache entry."""
        # Create initial entry
        cache_entry = CompanyCache(
            name="UpdateCorp",
            field_name="status",
            value={"status": "active"}
        )
        test_db.add(cache_entry)
        test_db.commit()

        # Update the entry
        entry = test_db.query(CompanyCache).filter_by(
            name="UpdateCorp",
            field_name="status"
        ).first()
        entry.value = {"status": "inactive", "reason": "bankruptcy"}
        entry.cached_at = datetime.utcnow()
        test_db.commit()

        # Verify update
        updated = test_db.query(CompanyCache).filter_by(
            name="UpdateCorp",
            field_name="status"
        ).first()
        assert updated.value["status"] == "inactive"
        assert "reason" in updated.value

    def test_delete_cache_entry(self, test_db: Session):
        """Test deleting a cache entry."""
        # Create entry
        cache_entry = CompanyCache(
            name="DeleteCorp",
            field_name="temp_data",
            value={"temp": True}
        )
        test_db.add(cache_entry)
        test_db.commit()

        # Delete the entry
        test_db.query(CompanyCache).filter_by(
            name="DeleteCorp",
            field_name="temp_data"
        ).delete()
        test_db.commit()

        # Verify deletion
        result = test_db.query(CompanyCache).filter_by(
            name="DeleteCorp",
            field_name="temp_data"
        ).first()
        assert result is None

    def test_age_days_property(self, test_db: Session):
        """Test the age_days property calculates correctly."""
        # Create entry with specific cached_at time
        past_time = datetime.utcnow() - timedelta(days=5)
        cache_entry = CompanyCache(
            name="AgeCorp",
            field_name="old_data",
            value={"data": "old"},
            cached_at=past_time
        )
        test_db.add(cache_entry)
        test_db.commit()

        # Query and check age
        entry = test_db.query(CompanyCache).filter_by(
            name="AgeCorp",
            field_name="old_data"
        ).first()

        # Age should be approximately 5 days (allow small variance for test execution time)
        assert 4.9 < entry.age_days < 5.1

    def test_is_expired_method(self, test_db: Session):
        """Test the is_expired method with different TTL values."""
        # Create entries with different ages
        old_entry = CompanyCache(
            name="OldCorp",
            field_name="stale_data",
            value={"data": "stale"},
            cached_at=datetime.utcnow() - timedelta(days=40)
        )
        fresh_entry = CompanyCache(
            name="FreshCorp",
            field_name="fresh_data",
            value={"data": "fresh"},
            cached_at=datetime.utcnow() - timedelta(days=10)
        )

        test_db.add_all([old_entry, fresh_entry])
        test_db.commit()

        # Test with default TTL (30 days)
        old = test_db.query(CompanyCache).filter_by(name="OldCorp").first()
        fresh = test_db.query(CompanyCache).filter_by(name="FreshCorp").first()

        assert old.is_expired(ttl_days=30) is True
        assert fresh.is_expired(ttl_days=30) is False

        # Test with custom TTL
        assert old.is_expired(ttl_days=50) is False
        assert fresh.is_expired(ttl_days=5) is True

    def test_repr_and_str_methods(self, test_db: Session):
        """Test the string representation methods."""
        cache_entry = CompanyCache(
            name="RepCorp",
            field_name="display_test",
            value={"test": True}
        )
        test_db.add(cache_entry)
        test_db.commit()

        entry = test_db.query(CompanyCache).filter_by(
            name="RepCorp",
            field_name="display_test"
        ).first()

        # Test __repr__
        repr_str = repr(entry)
        assert "CompanyCache" in repr_str
        assert "RepCorp" in repr_str
        assert "display_test" in repr_str

        # Test __str__
        str_str = str(entry)
        assert "RepCorp:display_test" in str_str


class TestDatabaseSession:
    """Test database session management."""

    def test_get_settings(self):
        """Test that settings can be retrieved."""
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, 'db_url')
        assert hasattr(settings, 'cache_ttl_days')

    def test_settings_defaults(self):
        """Test that default settings are reasonable."""
        settings = get_settings()
        assert settings.cache_ttl_days == 30
        assert "sqlite" in settings.db_url or "postgresql" in settings.db_url

    def test_get_db_generator(self, test_db: Session):
        """Test that get_db() generator properly creates and closes sessions."""
        from app.db.session import get_db

        # get_db is a generator, so we need to iterate it
        db_gen = get_db()
        session = next(db_gen)

        # Verify we got a session
        assert session is not None
        assert isinstance(session, Session)

        # Verify session is usable
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1

        # Close the generator (this should close the session)
        try:
            next(db_gen)
        except StopIteration:
            pass  # Expected behavior

    def test_session_isolation(self, test_db: Session):
        """Test that database sessions are properly isolated."""
        # Create an entry in test_db
        entry1 = CompanyCache(
            name="IsolationTest",
            field_name="test",
            value={"test": True}
        )
        test_db.add(entry1)
        test_db.commit()

        # Create a new session using get_db
        from app.db.session import get_db
        db_gen = get_db()
        new_session = next(db_gen)

        # Query from new session should see the committed data
        # Note: This test uses test_db which is in-memory, so get_db
        # creates a separate connection. We're testing the pattern works.
        try:
            # Verify new session works independently
            result = new_session.execute(text("SELECT 1")).scalar()
            assert result == 1
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass


class TestConfiguration:
    """Test configuration loading from INI files."""

    def test_settings_from_ini_file(self):
        """Test loading settings from a config.ini file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("[jobsearch]\n")
            f.write("db_url = postgresql://test:test@localhost/testdb\n")
            f.write("cache_ttl_days = 45\n")
            f.write("glassdoor_api_key = test_key_123\n")
            config_path = f.name

        try:
            # Load settings from the temporary file
            settings = Settings.from_configparser(config_path)

            # Verify settings were loaded correctly
            assert settings.db_url == "postgresql://test:test@localhost/testdb"
            assert settings.cache_ttl_days == 45
            assert settings.glassdoor_api_key == "test_key_123"

        finally:
            # Clean up
            os.unlink(config_path)

    def test_settings_missing_file_uses_defaults(self):
        """Test that missing config file gracefully falls back to defaults."""
        # Use a non-existent file path
        settings = Settings.from_configparser("/nonexistent/path/config.ini")

        # Should use defaults
        assert settings.db_url == "sqlite:///./jobsearch.db"
        assert settings.cache_ttl_days == 30
        assert settings.glassdoor_api_key is None

    def test_settings_partial_config(self):
        """Test that partial config file works (some values from file, some defaults)."""
        # Create a config with only db_url
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("[jobsearch]\n")
            f.write("db_url = sqlite:///./custom.db\n")
            config_path = f.name

        try:
            settings = Settings.from_configparser(config_path)

            # db_url should be from file
            assert settings.db_url == "sqlite:///./custom.db"

            # cache_ttl_days should be default
            assert settings.cache_ttl_days == 30

        finally:
            os.unlink(config_path)

    def test_settings_environment_variables(self):
        """Test that environment variables can override settings."""
        # Set environment variable
        os.environ["JOBSEARCH_DB_URL"] = "postgresql://env:env@localhost/envdb"

        try:
            # Create settings (env vars should take precedence)
            settings = Settings()

            assert settings.db_url == "postgresql://env:env@localhost/envdb"

        finally:
            # Clean up
            del os.environ["JOBSEARCH_DB_URL"]