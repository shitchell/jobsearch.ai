"""
Database session management and engine configuration.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import os
import sys

# Add the parent directory to the path to resolve imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import get_settings

# Get settings instance
settings = get_settings()

# Configure the database engine
# For SQLite, we need to add special connection arguments
connect_args = {}
if "sqlite" in settings.db_url:
    # Allow SQLite to work with multiple threads (needed for FastAPI)
    connect_args = {"check_same_thread": False}

# Create the database engine
engine = create_engine(
    settings.db_url,
    connect_args=connect_args,
    # Echo SQL statements for debugging (disable in production)
    echo=False,
    # Pool settings
    pool_pre_ping=True,  # Verify connections before using them
)

# Create a session factory
SessionLocal = sessionmaker(
    autocommit=False,  # Don't auto-commit (we'll manage transactions explicitly)
    autoflush=False,   # Don't auto-flush (we'll control when to flush)
    bind=engine,       # Bind to our engine
    class_=Session,    # Use the Session class
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.

    Yields a database session and ensures it's properly closed after use,
    even if an exception occurs.

    Usage:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()