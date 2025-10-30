"""
Pytest configuration and fixtures for Job Search AI API tests
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

# Import database models
from app.db.models import Base


@pytest.fixture
def test_db() -> Generator[Session, None, None]:
    """
    Provides an in-memory SQLite database for testing.

    Creates a fresh database for each test function, ensuring
    test isolation. The database is automatically cleaned up
    after each test.

    Yields:
        Session: SQLAlchemy database session for testing
    """
    # Create an in-memory SQLite database
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )

    # Create all tables
    Base.metadata.create_all(engine)

    # Create a session factory
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

    # Create and yield the session
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Tables are automatically dropped when the engine is garbage collected