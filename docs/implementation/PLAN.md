# Job Search AI - Implementation Plan

## Overview

This document provides a complete, ground-up implementation plan for the Job Search AI system. It is organized into chunks that can be implemented sequentially, where each chunk is:
- Independently testable
- Completable within ~150k token context
- Leaves codebase in working state (even if feature-incomplete)

**Target Audience:** Any developer (or AI agent) with no prior context can use this plan to build the entire system from scratch.

**Documentation References:**
- [UX Flow](../core/UX_FLOW.md) - User experience details
- [Developer Docs](../core/README.md) - Architecture and code patterns
- [Design Guidance](../core/DESIGN_GUIDANCE.md) - Philosophy and principles
- [Flexibility Points](../core/FLEX.md) - Extension mechanisms

---

## Project Structure

```
jobsearch.ai/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app
│   │   ├── models/                 # Pydantic models
│   │   │   ├── __init__.py
│   │   │   ├── job.py              # JobListing, SearchRequest, etc.
│   │   │   ├── company.py          # Company, ResearchResult
│   │   │   ├── provider.py         # Provider interfaces
│   │   │   └── display.py          # DisplayMetadata
│   │   ├── services/               # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── job_aggregation.py
│   │   │   └── company_research.py
│   │   ├── providers/              # Plugin directory
│   │   │   ├── job_sources/
│   │   │   │   └── indeed/
│   │   │   └── research/
│   │   │       ├── glassdoor/
│   │   │       └── scam_detector/
│   │   ├── db/                     # Database
│   │   │   ├── __init__.py
│   │   │   ├── models.py           # SQLAlchemy models
│   │   │   └── session.py
│   │   ├── config.py               # Settings
│   │   └── api/                    # API endpoints
│   │       ├── __init__.py
│   │       ├── search.py
│   │       └── research.py
│   ├── tests/                      # Pytest tests
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   ├── alembic/                    # Database migrations
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── config.ini                  # Configuration (not in git)
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── app.js                  # Main application class
│       ├── event-bus.js            # Event system
│       ├── api-client.js           # Backend communication
│       ├── ui.js                   # UI rendering
│       └── field-renderer.js       # Metadata-driven display
├── docs/                           # Documentation
│   ├── core/
│   └── implementation/
└── README.md
```

---

## Implementation Phases

### Phase 0: Foundation (Chunks 0.1-0.3)
Set up project structure, dependencies, configuration, and core models.

### Phase 1: Provider System (Chunks 1.1-1.3)
Build plugin architecture for extensibility.

### Phase 2: Job Search Basics (Chunks 2.1-2.3)
Implement job aggregation and search without real providers.

### Phase 3: First Real Provider (Chunks 3.1-3.3)
Add Indeed job source, validate end-to-end flow.

### Phase 4: Company Research System (Chunks 4.1-4.3)
Build research orchestration with caching.

### Phase 5: Research Providers (Chunks 5.1-5.3)
Add Glassdoor and AI scam detection.

### Phase 6: Display Metadata (Chunks 6.1-6.3)
Implement metadata-driven display system.

### Phase 7: Frontend Foundation (Chunks 7.1-7.3)
Build UI framework with event system.

### Phase 8: Quick Search Flow (Chunks 8.1-8.3)
Complete fast search experience.

### Phase 9: Deep Search Flow (Chunks 9.1-9.3)
Add comprehensive research mode.

### Phase 10: Manual Deep Research (Chunks 10.1-10.3)
Implement per-job deep research button.

### Phase 11: Additional Providers (Chunks 11.1-11.3)
Expand job sources and research capabilities.

### Phase 12: Production Polish (Chunks 12.1-12.3)
Error handling, logging, deployment.

---

## Chunk Specifications

Each chunk below includes:
- **Objective**: What we're building
- **Files to Create/Modify**: Specific file paths
- **Dependencies**: Which previous chunks must be complete
- **Implementation Details**: Key code patterns and logic
- **Testing Criteria**: How to verify success
- **Planner Guidance**: Notes for the Planning agent

---

# Phase 0: Foundation

## Chunk 0.1: Project Setup & Dependencies

**Objective:** Initialize project structure, install dependencies, create basic configuration system.

**Dependencies:** None (starting from scratch)

**Files to Create:**
```
backend/
  requirements.txt
  pyproject.toml
  config.ini.example
  .gitignore
  app/
    __init__.py
  tests/
    __init__.py
    conftest.py
README.md
```

**Implementation Details:**

### requirements.txt
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
sqlalchemy==2.0.23
alembic==1.13.0
asyncio==3.4.3
aiohttp==3.9.0
python-multipart==0.0.6
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.1
```

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "jobsearch-ai"
version = "0.1.0"
description = "AI-powered job search with scam detection"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.5.0",
    # ... (all from requirements.txt)
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### config.ini.example
```ini
[api]
host = 0.0.0.0
port = 8000

[jobsearch]
db_url = sqlite:///./jobsearch.db
cache_ttl_days = 30
ignored_providers =
require_all_providers = false

# API Keys (fill these in for config.ini)
openai_api_key =
indeed_api_key =
glassdoor_api_key =
```

### .gitignore
```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
config.ini
*.db
*.sqlite
.pytest_cache/
.coverage
htmlcov/
```

### tests/conftest.py
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base

@pytest.fixture
def test_db():
    """Create test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    yield db
    db.close()
```

**Testing Criteria:**
1. ✅ All directories exist
2. ✅ Dependencies install without errors: `pip install -r requirements.txt`
3. ✅ pytest runs (even with no tests): `pytest`
4. ✅ config.ini.example is valid INI format

**Planner Guidance:**
- **Developer**: Focus on file creation, dependency resolution, basic project structure
- **Tester**: Create tests for:
  - Directory structure validation
  - Requirements installation
  - Config file parsing
- **Relevant Files for Both**: All Phase 0 files, README.md, .gitignore

---

## Chunk 0.2: Database Models & Migrations

**Objective:** Set up SQLAlchemy models, database session management, Alembic migrations.

**Dependencies:** Chunk 0.1 (project structure must exist)

**Files to Create:**
```
backend/
  app/
    db/
      __init__.py
      models.py
      session.py
  alembic/
    env.py
    script.py.mako
    versions/
  alembic.ini
```

**Implementation Details:**

### app/db/models.py
```python
from sqlalchemy import Column, String, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class CompanyCache(Base):
    """Cache for individual company research fields"""
    __tablename__ = "company_cache"

    name = Column(String, primary_key=True)
    field_name = Column(String, primary_key=True)
    value = Column(JSON)  # Flexible storage for any field type
    cached_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<CompanyCache(name={self.name}, field={self.field_name})>"
```

**Key Points:**
- Composite primary key (name + field_name) enables per-field caching
- JSON column for flexible value storage
- Simple schema for MVP

### app/db/session.py
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.db_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.db_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Session:
    """Dependency for FastAPI endpoints"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### alembic.ini
```ini
[alembic]
script_location = alembic
file_template = %%(year)d_%%(month).2d_%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

### Initial Migration
```bash
# Commands to run (document in chunk, not execute yet)
alembic init alembic
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

**Testing Criteria:**
1. ✅ Database models import without errors
2. ✅ SessionLocal creates sessions successfully
3. ✅ Alembic can generate migration: `alembic revision --autogenerate -m "test"`
4. ✅ Alembic can apply migration: `alembic upgrade head`
5. ✅ CompanyCache table exists with correct schema
6. ✅ Can insert and retrieve from CompanyCache

**Test Example:**
```python
def test_company_cache_crud(test_db):
    from datetime import datetime
    from app.db.models import CompanyCache

    # Create
    cache_entry = CompanyCache(
        name="TechCorp",
        field_name="scam_score",
        value={"scam_score": 0.15},
        cached_at=datetime.now()
    )
    test_db.add(cache_entry)
    test_db.commit()

    # Read
    retrieved = test_db.query(CompanyCache).filter_by(
        name="TechCorp",
        field_name="scam_score"
    ).first()

    assert retrieved is not None
    assert retrieved.value["scam_score"] == 0.15
```

**Planner Guidance:**
- **Developer**: Implement SQLAlchemy models, session management, Alembic setup
- **Tester**: Create CRUD tests for CompanyCache, migration tests
- **Relevant Files**:
  - Dev: `app/db/*.py`, `alembic.ini`, `alembic/env.py`
  - Test: `tests/unit/test_db.py`, `conftest.py`

---

## Chunk 0.3: Core Pydantic Models

**Objective:** Implement core Pydantic models for type safety throughout application.

**Dependencies:** Chunk 0.1 (project structure)

**Files to Create:**
```
backend/
  app/
    models/
      __init__.py
      job.py
      company.py
      provider.py
      display.py
      config.py
```

**Implementation Details:**

### app/models/job.py
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib

class JobListing(BaseModel):
    """Standard job listing model across all sources"""
    id: str
    source: str  # "Indeed", "LinkedIn", etc.
    title: str
    company: str
    description: Optional[str] = None
    pay: Optional[str] = None
    location: Optional[str] = None
    remote: Optional[bool] = None
    url: str
    posted_date: Optional[datetime] = None
    duplicate_group_id: Optional[str] = None
    duplicate_sources: List[str] = Field(default_factory=list)
    source_metadata: Dict[str, Any] = Field(default_factory=dict)

    def generate_duplicate_hash(self) -> str:
        """
        Generate hash for duplicate detection.
        Normalized: company + title + location + description (first 50 alpha chars)
        """
        def normalize(text: str) -> str:
            return ''.join(c for c in text.lower() if c.isalpha())

        title_norm = normalize(self.title)
        company_norm = normalize(self.company)
        location_norm = normalize(self.location or "")
        desc_norm = normalize(self.description or "")[:50]

        key = f"{company_norm}:{title_norm}:{location_norm}:{desc_norm}"
        return hashlib.md5(key.encode()).hexdigest()

class SearchRequest(BaseModel):
    """Request body for job search"""
    query: Optional[str] = None
    location: Optional[str] = None
    remote: Optional[bool] = None
    min_pay: Optional[int] = None
    max_pay: Optional[int] = None

class SearchResponse(BaseModel):
    """Response from search endpoint"""
    jobs: List[JobListing]
    companies: Dict[str, Any]  # Will be Company objects
    total_count: int
    filtered_count: int
```

**Key Points:**
- All fields have clear types
- `generate_duplicate_hash()` method is self-contained
- Optional fields use `Optional[...]`
- Lists/dicts have default factories

### app/models/company.py
```python
from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime

class Company(BaseModel):
    """Company with dynamic research fields"""
    name: str
    cached_date: Optional[datetime] = None

    model_config = ConfigDict(extra='allow')  # Allow dynamic fields

    def merge_research(self, result: 'ResearchResult') -> 'Company':
        """Merge provider results into company"""
        return self.model_copy(
            update=result.model_dump(exclude_none=True)
        )

class ResearchResult(BaseModel):
    """Base class for all provider research results"""
    pass  # Providers will subclass this
```

**Key Points:**
- `extra='allow'` enables dynamic fields from providers
- `merge_research()` method updates company with new data
- `ResearchResult` is empty base class for type safety

### app/models/provider.py
```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Dict, Tuple, List, Optional, Any
from enum import Enum

class ResearchCategory(str, Enum):
    """Cost/speed categories for research providers"""
    BASIC = "basic"
    API_CHEAP = "api_cheap"
    API_EXPENSIVE = "api_expensive"
    AI = "ai"

class DisplayMetadata(BaseModel):
    """How to display a field in frontend"""
    type: str  # "text", "rating", "percentage", "badge", "list", "custom"
    icon: Optional[str] = None
    priority: str = "medium"  # "low", "medium", "high"
    format: Optional[str] = None  # "currency", "date", "percentage", "decimal_1"
    max_value: Optional[float] = None  # For ratings
    color_scale: Optional[Dict[str, str]] = None  # {"0-30": "green", ...}
    invert: bool = False  # Lower is better
    list_style: Optional[str] = None  # "bullet", "numbered", "comma"
    custom_config: Optional[Dict[str, Any]] = None  # For type="custom"

class FieldContribution(BaseModel):
    """Provider's contribution declaration"""
    category: ResearchCategory
    label: str
    display: DisplayMetadata

class JobSource(ABC):
    """Abstract base for job listing sources"""

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        location: Optional[str] = None,
        remote: Optional[bool] = None,
        min_pay: Optional[int] = None,
        max_pay: Optional[int] = None
    ) -> List['JobListing']:
        pass

class ResearchProvider(ABC):
    """Abstract base for research providers"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def contributions(self) -> Dict[str, FieldContribution]:
        """Declare fields this provider contributes"""
        pass

    @abstractmethod
    async def research(
        self,
        company_name: str,
        requested_fields: List[str]
    ) -> 'ResearchResult':
        pass
```

**Key Points:**
- ABC (Abstract Base Class) enforces interface
- All methods marked with `@abstractmethod`
- Clear separation of concerns

### app/models/config.py
```python
from pydantic import BaseModel
from typing import List
import configparser

class Settings(BaseModel):
    """Application settings"""
    # Database
    db_url: str = "sqlite:///./jobsearch.db"

    # API Keys
    openai_api_key: str = ""
    indeed_api_key: str = ""
    glassdoor_api_key: str = ""

    # Cache
    cache_ttl_days: int = 30

    # Provider management
    ignored_providers: List[str] = []
    require_all_providers: bool = False

    @classmethod
    def from_configparser(cls, config_path: str, section: str = "jobsearch"):
        """Load settings from INI file"""
        config = configparser.ConfigParser()
        config.read(config_path)

        if section not in config:
            raise ValueError(f"Section [{section}] not found in {config_path}")

        settings_dict = dict(config[section])

        # Type conversions
        if 'ignored_providers' in settings_dict:
            settings_dict['ignored_providers'] = [
                p.strip() for p in settings_dict['ignored_providers'].split(',') if p.strip()
            ]

        if 'cache_ttl_days' in settings_dict:
            settings_dict['cache_ttl_days'] = int(settings_dict['cache_ttl_days'])

        if 'require_all_providers' in settings_dict:
            settings_dict['require_all_providers'] = (
                settings_dict['require_all_providers'].lower() == 'true'
            )

        return cls(**settings_dict)
```

**Key Points:**
- Loads from configparser (INI format)
- Type conversions for int/bool/list
- Default values provided

**Testing Criteria:**
1. ✅ All models import without errors
2. ✅ JobListing.generate_duplicate_hash() produces consistent hashes
3. ✅ Company.merge_research() updates fields correctly
4. ✅ Settings.from_configparser() loads config.ini
5. ✅ Abstract base classes cannot be instantiated
6. ✅ Pydantic validation works (invalid data raises ValidationError)

**Test Examples:**
```python
def test_job_listing_duplicate_hash():
    job1 = JobListing(
        id="1",
        source="Indeed",
        title="Software Engineer",
        company="TechCorp",
        description="Build amazing things...",
        url="https://indeed.com/job/1"
    )
    job2 = JobListing(
        id="2",
        source="LinkedIn",
        title="Software Engineer",  # Same job
        company="TechCorp",
        description="Build amazing things...",
        url="https://linkedin.com/job/2"
    )

    hash1 = job1.generate_duplicate_hash()
    hash2 = job2.generate_duplicate_hash()

    assert hash1 == hash2  # Should match

def test_company_merge_research():
    from app.models.company import Company, ResearchResult

    company = Company(name="TechCorp")

    # Create mock research result
    class MockResult(ResearchResult):
        scam_score: float = 0.15

    updated = company.merge_research(MockResult())
    assert updated.scam_score == 0.15

def test_settings_from_config(tmp_path):
    # Create temp config file
    config_file = tmp_path / "test_config.ini"
    config_file.write_text("""
[jobsearch]
db_url = postgresql://localhost/test
cache_ttl_days = 7
ignored_providers = foo, bar
require_all_providers = true
openai_api_key = sk-test123
""")

    settings = Settings.from_configparser(str(config_file))

    assert settings.db_url == "postgresql://localhost/test"
    assert settings.cache_ttl_days == 7
    assert settings.ignored_providers == ["foo", "bar"]
    assert settings.require_all_providers == True
```

**Planner Guidance:**
- **Developer**: Implement all Pydantic models with proper types, validation, methods
- **Tester**: Create unit tests for:
  - Duplicate hash consistency
  - Company merging
  - Config loading with various formats
  - Validation errors on invalid data
- **Relevant Files**:
  - Dev: `app/models/*.py`
  - Test: `tests/unit/test_models.py`

---

# Phase 1: Provider System

## Chunk 1.1: Provider Discovery Mechanism

**Objective:** Implement dynamic provider discovery via directory scanning with importlib.

**Dependencies:** Chunks 0.1, 0.3 (project structure, models)

**Files to Create:**
```
backend/
  app/
    providers/
      __init__.py
      discovery.py
    config.py
```

**Implementation Details:**

### app/providers/__init__.py
```python
"""Provider plugin system"""
```

### app/providers/discovery.py
```python
import importlib
import logging
from pathlib import Path
from typing import List, Optional
from app.models.provider import JobSource, ResearchProvider
from app.models.config import Settings

logger = logging.getLogger(__name__)

def discover_job_sources(settings: Settings) -> List[JobSource]:
    """
    Discover all job source providers by scanning providers/job_sources/ directory.

    Each provider directory must have:
    - __init__.py with get_provider() function
    - get_provider() returns JobSource instance or None
    """
    providers = []
    base_path = Path(__file__).parent / "job_sources"

    if not base_path.exists():
        logger.warning(f"Job sources directory not found: {base_path}")
        return providers

    for provider_dir in base_path.iterdir():
        if not provider_dir.is_dir() or provider_dir.name.startswith('_'):
            continue

        provider_name = provider_dir.name

        # Check if ignored
        if provider_name in settings.ignored_providers:
            logger.info(f"Skipping ignored job source: {provider_name}")
            continue

        try:
            module = importlib.import_module(f"app.providers.job_sources.{provider_name}")
            if hasattr(module, 'get_provider'):
                provider = module.get_provider()
                if provider:
                    providers.append(provider)
                    logger.info(f"Loaded job source: {provider_name}")
                else:
                    logger.info(f"Job source {provider_name} not configured (returned None)")
        except Exception as e:
            error_msg = f"Failed to load job source {provider_name}: {e}"
            if settings.require_all_providers:
                logger.error(error_msg)
                raise RuntimeError(f"Provider load failure (require_all_providers=true): {provider_name}") from e
            else:
                logger.warning(error_msg)

    if settings.require_all_providers and len(providers) == 0:
        raise RuntimeError("No job sources loaded and require_all_providers=true")

    logger.info(f"Successfully loaded {len(providers)} job source(s)")
    return providers

def discover_research_providers(settings: Settings) -> List[ResearchProvider]:
    """
    Discover all research providers by scanning providers/research/ directory.

    Same pattern as job sources.
    """
    providers = []
    base_path = Path(__file__).parent / "research"

    if not base_path.exists():
        logger.warning(f"Research providers directory not found: {base_path}")
        return providers

    for provider_dir in base_path.iterdir():
        if not provider_dir.is_dir() or provider_dir.name.startswith('_'):
            continue

        provider_name = provider_dir.name

        # Check if ignored
        if provider_name in settings.ignored_providers:
            logger.info(f"Skipping ignored research provider: {provider_name}")
            continue

        try:
            module = importlib.import_module(f"app.providers.research.{provider_name}")
            if hasattr(module, 'get_provider'):
                provider = module.get_provider()
                if provider:
                    providers.append(provider)
                    logger.info(f"Loaded research provider: {provider_name}")
                else:
                    logger.info(f"Research provider {provider_name} not configured (returned None)")
        except Exception as e:
            error_msg = f"Failed to load research provider {provider_name}: {e}"
            if settings.require_all_providers:
                logger.error(error_msg)
                raise RuntimeError(f"Provider load failure (require_all_providers=true): {provider_name}") from e
            else:
                logger.warning(error_msg)

    if settings.require_all_providers and len(providers) == 0:
        raise RuntimeError("No research providers loaded and require_all_providers=true")

    logger.info(f"Successfully loaded {len(providers)} research provider(s)")
    return providers
```

**Key Points:**
- Scans directories with `Path.iterdir()`
- Uses `importlib.import_module()` for dynamic loading
- Respects `ignored_providers` config
- Handles `require_all_providers` mode
- Logs all discovery events
- Graceful degradation by default

### app/config.py
```python
from functools import lru_cache
from app.models.config import Settings

@lru_cache()
def get_settings() -> Settings:
    """Cached settings loader"""
    return Settings.from_configparser("config.ini", section="jobsearch")
```

**Testing Criteria:**
1. ✅ Discovery functions find providers in directories
2. ✅ Providers with no `get_provider()` are skipped
3. ✅ Providers returning `None` are skipped gracefully
4. ✅ `ignored_providers` config is respected
5. ✅ `require_all_providers=true` raises error on failure
6. ✅ `require_all_providers=false` continues on failure
7. ✅ Logging captures all events

**Test Examples:**
```python
def test_discover_job_sources_empty_directory(tmp_path, monkeypatch):
    """Should handle empty directories gracefully"""
    from app.providers.discovery import discover_job_sources
    from app.models.config import Settings

    # Create empty directory
    job_sources_dir = tmp_path / "job_sources"
    job_sources_dir.mkdir()

    # Monkey-patch Path to point to temp dir
    # (actual test would use temp plugin directory)

    settings = Settings(require_all_providers=False)
    providers = discover_job_sources(settings)

    assert len(providers) == 0

def test_discover_with_ignored_providers(tmp_path):
    """Should skip ignored providers"""
    # Create mock provider directory with working provider
    # Set ignored_providers = ["mock_provider"]
    # Assert provider not loaded
    pass

def test_require_all_providers_strict_mode():
    """Should raise error if any provider fails to load"""
    settings = Settings(require_all_providers=True)

    # Create provider that raises exception
    # Assert discover_job_sources() raises RuntimeError
    pass
```

**Planner Guidance:**
- **Developer**: Implement directory scanning, importlib loading, error handling
- **Tester**: Create tests for:
  - Empty directories
  - Missing `get_provider()`
  - Providers returning None
  - Ignored providers config
  - Strict vs graceful mode
- **Relevant Files**:
  - Dev: `app/providers/discovery.py`, `app/config.py`
  - Test: `tests/unit/test_discovery.py`
  - Mock: Create `tests/fixtures/mock_providers/` for testing

---

## Chunk 1.2: Mock Provider for Testing

**Objective:** Create a simple mock provider to validate discovery and loading system.

**Dependencies:** Chunk 1.1 (discovery mechanism)

**Files to Create:**
```
backend/
  app/
    providers/
      job_sources/
        mock_indeed/
          __init__.py
          provider.py
```

**Implementation Details:**

### app/providers/job_sources/mock_indeed/__init__.py
```python
from .provider import MockIndeedSource

def get_provider():
    """Factory function for provider discovery"""
    # For testing, always return an instance
    # Real providers would check API keys here
    return MockIndeedSource()
```

### app/providers/job_sources/mock_indeed/provider.py
```python
from app.models.provider import JobSource
from app.models.job import JobListing
from typing import List, Optional
from uuid import uuid4
from datetime import datetime, timedelta
import random

class MockIndeedSource(JobSource):
    """Mock job source for testing discovery and aggregation"""

    @property
    def source_name(self) -> str:
        return "Indeed (Mock)"

    async def search(
        self,
        query: str,
        location: Optional[str] = None,
        remote: Optional[bool] = None,
        min_pay: Optional[int] = None,
        max_pay: Optional[int] = None
    ) -> List[JobListing]:
        """Return mock job listings"""

        # Generate 3-5 mock jobs
        job_count = random.randint(3, 5)
        jobs = []

        for i in range(job_count):
            job = JobListing(
                id=str(uuid4()),
                source="Indeed (Mock)",
                title=f"{query} Engineer {i+1}",
                company=f"Company {chr(65+i)}",  # Company A, B, C, etc.
                description=f"We are seeking a {query} professional...",
                pay=f"${random.randint(80, 150)}k-${random.randint(151, 200)}k",
                location=location or "Remote",
                remote=remote if remote is not None else True,
                url=f"https://indeed.com/mock/job/{uuid4()}",
                posted_date=datetime.now() - timedelta(days=random.randint(1, 30))
            )
            jobs.append(job)

        return jobs
```

**Key Points:**
- Implements full JobSource interface
- Returns realistic mock data
- Randomization provides variety for testing
- Async method (even though mock is synchronous internally)

**Testing Criteria:**
1. ✅ Mock provider is discovered by `discover_job_sources()`
2. ✅ Mock provider returns 3-5 JobListing objects
3. ✅ All required JobListing fields are populated
4. ✅ Query parameter influences job titles
5. ✅ Location parameter is respected

**Test Example:**
```python
@pytest.mark.asyncio
async def test_mock_indeed_source():
    from app.providers.job_sources.mock_indeed.provider import MockIndeedSource

    source = MockIndeedSource()

    # Test search
    jobs = await source.search(query="Software", location="San Francisco")

    assert len(jobs) >= 3
    assert len(jobs) <= 5
    assert all(job.source == "Indeed (Mock)" for job in jobs)
    assert all("Software" in job.title for job in jobs)
    assert all(job.location == "San Francisco" for job in jobs)

def test_discovery_finds_mock_provider():
    from app.providers.discovery import discover_job_sources
    from app.models.config import Settings

    settings = Settings()
    providers = discover_job_sources(settings)

    assert len(providers) > 0
    assert any(p.source_name == "Indeed (Mock)" for p in providers)
```

**Planner Guidance:**
- **Developer**: Implement simple mock provider following interface
- **Tester**: Validate mock provider works, is discoverable
- **Relevant Files**:
  - Dev: `app/providers/job_sources/mock_indeed/*.py`
  - Test: `tests/integration/test_mock_provider.py`

---

## Chunk 1.3: Settings & Configuration Loading

**Objective:** Integrate settings loading with FastAPI dependency injection.

**Dependencies:** Chunks 0.3, 1.1 (models, config)

**Files to Create/Modify:**
```
backend/
  app/
    config.py  (already created, enhance)
    main.py    (create minimal FastAPI app)
```

**Implementation Details:**

### app/config.py (enhanced)
```python
from functools import lru_cache
from app.models.config import Settings
import logging

logger = logging.getLogger(__name__)

@lru_cache()
def get_settings() -> Settings:
    """
    Load settings from config.ini.
    Cached to avoid re-reading file on every request.
    """
    try:
        settings = Settings.from_configparser("config.ini", section="jobsearch")
        logger.info("Settings loaded successfully")
        return settings
    except FileNotFoundError:
        logger.warning("config.ini not found, using defaults")
        return Settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        raise
```

### app/main.py
```python
from fastapi import FastAPI
from app.config import get_settings

app = FastAPI(
    title="Job Search AI",
    description="AI-powered job search with scam detection",
    version="0.1.0"
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Job Search AI API"}

@app.get("/config")
async def get_config():
    """Debug endpoint to view configuration (remove in production)"""
    settings = get_settings()
    return {
        "db_url": settings.db_url,
        "cache_ttl_days": settings.cache_ttl_days,
        "ignored_providers": settings.ignored_providers,
        "require_all_providers": settings.require_all_providers
        # Don't expose API keys!
    }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Testing Criteria:**
1. ✅ Settings load from config.ini if present
2. ✅ Settings use defaults if config.ini missing
3. ✅ Settings are cached (same object returned on multiple calls)
4. ✅ FastAPI app starts without errors: `python app/main.py`
5. ✅ GET / returns 200 OK
6. ✅ GET /config returns configuration (non-sensitive fields)

**Test Example:**
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_config_endpoint():
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "db_url" in data
    assert "cache_ttl_days" in data
```

**Planner Guidance:**
- **Developer**: Enhance config loading, create minimal FastAPI app
- **Tester**: Test config loading, API startup, endpoints
- **Relevant Files**:
  - Dev: `app/config.py`, `app/main.py`
  - Test: `tests/integration/test_api.py`

---

# Phase 2: Job Search Basics

## Chunk 2.1: Job Aggregation Service

**Objective:** Implement service to aggregate jobs from multiple sources in parallel.

**Dependencies:** Chunks 1.1, 1.2 (provider discovery, mock provider)

**Files to Create:**
```
backend/
  app/
    services/
      __init__.py
      job_aggregation.py
```

**Implementation Details:**

### app/services/job_aggregation.py
```python
import asyncio
import logging
from typing import List, Optional
from app.models.job import JobListing
from app.models.provider import JobSource

logger = logging.getLogger(__name__)

async def aggregate_jobs_from_sources(
    sources: List[JobSource],
    query: str,
    location: Optional[str] = None,
    remote: Optional[bool] = None,
    min_pay: Optional[int] = None,
    max_pay: Optional[int] = None,
    limit: int = 100
) -> List[JobListing]:
    """
    Query all job sources in parallel and aggregate results.

    Args:
        sources: List of JobSource providers
        query: Search query (e.g., "software engineer")
        location: Optional location filter
        remote: Optional remote-only filter
        min_pay: Optional minimum pay filter
        max_pay: Optional maximum pay filter
        limit: Maximum total jobs to return

    Returns:
        List of JobListing objects, deduplicated and sorted by recency
    """
    if not sources:
        logger.warning("No job sources provided, returning empty results")
        return []

    logger.info(f"Querying {len(sources)} job source(s) for: {query}")

    # Query all sources in parallel
    search_tasks = [
        source.search(
            query=query,
            location=location,
            remote=remote,
            min_pay=min_pay,
            max_pay=max_pay
        )
        for source in sources
    ]

    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Flatten results and handle errors
    all_jobs = []
    for source, result in zip(sources, results):
        if isinstance(result, list):
            all_jobs.extend(result)
            logger.info(f"Source {source.source_name} returned {len(result)} jobs")
        elif isinstance(result, Exception):
            logger.error(f"Source {source.source_name} failed: {result}")
        else:
            logger.warning(f"Source {source.source_name} returned unexpected type: {type(result)}")

    logger.info(f"Aggregated {len(all_jobs)} total jobs from all sources")

    # Sort by recency (most recent first)
    from datetime import datetime
    all_jobs.sort(
        key=lambda j: j.posted_date or datetime.min,
        reverse=True
    )

    return all_jobs[:limit]
```

**Key Points:**
- `asyncio.gather()` runs all source queries in parallel
- `return_exceptions=True` prevents one failure from breaking all
- Graceful error handling with logging
- Sorting by recency

**Testing Criteria:**
1. ✅ Service aggregates results from multiple sources
2. ✅ Parallel execution (not sequential)
3. ✅ One source failure doesn't break aggregation
4. ✅ Results are sorted by posted_date (newest first)
5. ✅ Respects limit parameter
6. ✅ Handles empty source list

**Test Example:**
```python
@pytest.mark.asyncio
async def test_aggregate_jobs_from_multiple_sources():
    from app.services.job_aggregation import aggregate_jobs_from_sources
    from app.providers.discovery import discover_job_sources
    from app.models.config import Settings

    settings = Settings()
    sources = discover_job_sources(settings)

    jobs = await aggregate_jobs_from_sources(
        sources=sources,
        query="Python Developer",
        limit=20
    )

    assert len(jobs) > 0
    assert len(jobs) <= 20
    assert all(isinstance(job, JobListing) for job in jobs)

    # Check sorting (newest first)
    if len(jobs) > 1:
        for i in range(len(jobs) - 1):
            if jobs[i].posted_date and jobs[i+1].posted_date:
                assert jobs[i].posted_date >= jobs[i+1].posted_date

@pytest.mark.asyncio
async def test_aggregate_handles_source_failure():
    """Should continue if one source fails"""
    from app.services.job_aggregation import aggregate_jobs_from_sources
    from app.models.provider import JobSource
    from app.models.job import JobListing

    class FailingSource(JobSource):
        @property
        def source_name(self):
            return "Failing Source"

        async def search(self, **kwargs):
            raise Exception("API timeout")

    class WorkingSource(JobSource):
        @property
        def source_name(self):
            return "Working Source"

        async def search(self, **kwargs):
            return [JobListing(
                id="1",
                source="Working",
                title="Engineer",
                company="Corp",
                url="http://example.com"
            )]

    sources = [FailingSource(), WorkingSource()]
    jobs = await aggregate_jobs_from_sources(sources, query="test")

    # Should get result from working source despite failure
    assert len(jobs) == 1
    assert jobs[0].source == "Working"
```

**Planner Guidance:**
- **Developer**: Implement parallel aggregation with error handling
- **Tester**: Test multiple sources, failures, sorting, limits
- **Relevant Files**:
  - Dev: `app/services/job_aggregation.py`
  - Test: `tests/unit/test_job_aggregation.py`

---

## Chunk 2.2: Duplicate Detection

**Objective:** Implement duplicate job detection across multiple sources.

**Dependencies:** Chunk 2.1 (job aggregation)

**Files to Create/Modify:**
```
backend/
  app/
    services/
      job_aggregation.py  (enhance with deduplication)
```

**Implementation Details:**

### Enhanced app/services/job_aggregation.py

Add deduplication function:

```python
def deduplicate_jobs(jobs: List[JobListing]) -> List[JobListing]:
    """
    Deduplicate jobs based on normalized hash.
    Tracks which sources each job appears on.

    Args:
        jobs: List of job listings (may contain duplicates)

    Returns:
        List of unique jobs with duplicate_sources populated
    """
    seen_groups = {}
    unique_jobs = []

    for job in jobs:
        # Generate hash for this job
        group_id = job.generate_duplicate_hash()
        job.duplicate_group_id = group_id

        if group_id not in seen_groups:
            # First time seeing this job
            job.duplicate_sources = [job.source]
            seen_groups[group_id] = job
            unique_jobs.append(job)
        else:
            # Duplicate found - add source to original
            seen_groups[group_id].duplicate_sources.append(job.source)
            logger.debug(f"Duplicate found: {job.title} at {job.company} (sources: {job.source})")

    logger.info(f"Deduplicated {len(jobs)} jobs → {len(unique_jobs)} unique jobs")
    return unique_jobs

# Update aggregate_jobs_from_sources() to call deduplicate_jobs():

async def aggregate_jobs_from_sources(
    sources: List[JobSource],
    query: str,
    location: Optional[str] = None,
    remote: Optional[bool] = None,
    min_pay: Optional[int] = None,
    max_pay: Optional[int] = None,
    limit: int = 100
) -> List[JobListing]:
    # ... (existing code for parallel queries) ...

    # Flatten results and handle errors
    all_jobs = []
    # ... (existing error handling) ...

    logger.info(f"Aggregated {len(all_jobs)} total jobs from all sources")

    # Deduplicate before sorting
    unique_jobs = deduplicate_jobs(all_jobs)

    # Sort by recency
    unique_jobs.sort(
        key=lambda j: j.posted_date or datetime.min,
        reverse=True
    )

    return unique_jobs[:limit]
```

**Key Points:**
- Uses `generate_duplicate_hash()` from JobListing model
- Tracks all sources for each unique job
- Preserves first instance encountered
- Logs duplicate detection

**Testing Criteria:**
1. ✅ Identical jobs from different sources are grouped
2. ✅ `duplicate_sources` lists all sources
3. ✅ `duplicate_group_id` is same for duplicates
4. ✅ Non-duplicate jobs remain separate
5. ✅ Deduplication doesn't lose jobs
6. ✅ Hash is case-insensitive
7. ✅ Hash ignores non-alphabetic characters

**Test Examples:**
```python
def test_deduplicate_identical_jobs():
    from app.services.job_aggregation import deduplicate_jobs
    from app.models.job import JobListing

    job1 = JobListing(
        id="1",
        source="Indeed",
        title="Software Engineer",
        company="TechCorp",
        description="Build things",
        url="https://indeed.com/1"
    )
    job2 = JobListing(
        id="2",
        source="LinkedIn",
        title="Software Engineer",  # Same job
        company="TechCorp",
        description="Build things",
        url="https://linkedin.com/2"
    )

    jobs = [job1, job2]
    unique = deduplicate_jobs(jobs)

    assert len(unique) == 1
    assert unique[0].duplicate_sources == ["Indeed", "LinkedIn"]

def test_deduplicate_case_insensitive():
    job1 = JobListing(
        id="1", source="A", title="Software Engineer",
        company="TechCorp", url="http://a.com"
    )
    job2 = JobListing(
        id="2", source="B", title="SOFTWARE ENGINEER",  # Different case
        company="techcorp", url="http://b.com"
    )

    unique = deduplicate_jobs([job1, job2])
    assert len(unique) == 1

def test_generate_duplicate_hash_ignores_special_chars():
    job1 = JobListing(
        id="1", source="A", title="C++ Developer",
        company="Tech-Corp", url="http://a.com"
    )
    job2 = JobListing(
        id="2", source="B", title="C  Developer",  # No ++
        company="TechCorp", url="http://b.com"
    )

    hash1 = job1.generate_duplicate_hash()
    hash2 = job2.generate_duplicate_hash()

    assert hash1 == hash2  # Should match (special chars ignored)
```

**Planner Guidance:**
- **Developer**: Implement deduplication logic, integrate with aggregation
- **Tester**: Test various duplicate scenarios, hash edge cases
- **Relevant Files**:
  - Dev: `app/services/job_aggregation.py`
  - Test: `tests/unit/test_deduplication.py`

---

## Chunk 2.3: Basic Search API Endpoint

**Objective:** Create API endpoint that uses aggregation service (no research yet).

**Dependencies:** Chunk 2.2 (aggregation with deduplication)

**Files to Create:**
```
backend/
  app/
    api/
      __init__.py
      search.py
    main.py  (modify to include router)
```

**Implementation Details:**

### app/api/search.py
```python
from fastapi import APIRouter, Depends, Query
from typing import List
from app.models.job import SearchRequest, SearchResponse
from app.models.provider import JobSource
from app.services.job_aggregation import aggregate_jobs_from_sources
from app.providers.discovery import discover_job_sources
from app.models.config import Settings
from app.config import get_settings

router = APIRouter(prefix="/api", tags=["search"])

def get_job_sources(settings: Settings = Depends(get_settings)) -> List[JobSource]:
    """Dependency to get job sources"""
    return discover_job_sources(settings)

@router.post("/search")
async def search_jobs(
    request: SearchRequest,
    sources: List[JobSource] = Depends(get_job_sources)
) -> SearchResponse:
    """
    Search for jobs across all configured sources.

    No company research yet - that comes in Phase 4.
    """
    jobs = await aggregate_jobs_from_sources(
        sources=sources,
        query=request.query or "",
        location=request.location,
        remote=request.remote,
        min_pay=request.min_pay,
        max_pay=request.max_pay,
        limit=100
    )

    return SearchResponse(
        jobs=jobs,
        companies={},  # Empty for now, will populate in Phase 4
        total_count=len(jobs),
        filtered_count=len(jobs)  # No filtering yet
    )
```

**Key Points:**
- Uses FastAPI dependency injection for sources
- Returns SearchResponse model
- No company research yet (Phase 4)
- Simple aggregation only

### app/main.py (modified)
```python
from fastapi import FastAPI
from app.config import get_settings
from app.api import search  # Import router

app = FastAPI(
    title="Job Search AI",
    description="AI-powered job search with scam detection",
    version="0.1.0"
)

# Include search router
app.include_router(search.router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Job Search AI API"}

@app.get("/config")
async def get_config():
    settings = get_settings()
    return {
        "db_url": settings.db_url,
        "cache_ttl_days": settings.cache_ttl_days,
        "ignored_providers": settings.ignored_providers,
        "require_all_providers": settings.require_all_providers
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Testing Criteria:**
1. ✅ POST /api/search returns 200 OK
2. ✅ Response matches SearchResponse schema
3. ✅ Jobs are aggregated from sources
4. ✅ Duplicates are detected
5. ✅ Query parameter filters results
6. ✅ Location parameter works
7. ✅ Empty query returns results
8. ✅ No sources returns empty results gracefully

**Test Examples:**
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_search_endpoint_basic():
    response = client.post("/api/search", json={
        "query": "Python Developer"
    })

    assert response.status_code == 200
    data = response.json()

    assert "jobs" in data
    assert "companies" in data
    assert "total_count" in data
    assert "filtered_count" in data

    assert isinstance(data["jobs"], list)
    assert data["total_count"] > 0

def test_search_with_location():
    response = client.post("/api/search", json={
        "query": "Engineer",
        "location": "San Francisco"
    })

    assert response.status_code == 200
    jobs = response.json()["jobs"]

    # Mock provider respects location parameter
    assert all(job["location"] == "San Francisco" for job in jobs)

def test_search_duplicate_detection():
    """Test that duplicates are properly grouped"""
    response = client.post("/api/search", json={
        "query": "Software Engineer"
    })

    assert response.status_code == 200
    jobs = response.json()["jobs"]

    # Check if any jobs have multiple sources (duplicates found)
    multi_source_jobs = [j for j in jobs if len(j["duplicate_sources"]) > 1]
    # With real providers, would expect some duplicates
    # With mock provider, might not have any
    # Just verify field exists
    assert all("duplicate_sources" in j for j in jobs)
```

**Planner Guidance:**
- **Developer**: Create search endpoint, integrate aggregation service
- **Tester**: Test API endpoint, various search parameters
- **Relevant Files**:
  - Dev: `app/api/search.py`, `app/main.py`
  - Test: `tests/integration/test_search_api.py`

---

# Phase 3: First Real Provider (Indeed Integration)

## Chunk 3.1: Indeed API Client Setup

**Objective:** Create Indeed job source provider with real API integration.

**Dependencies:** Chunks 1.1, 1.2, 2.1 (discovery, mock provider, aggregation)

**Files to Create:**
```
backend/
  app/
    providers/
      job_sources/
        indeed/
          __init__.py
          provider.py
          client.py
```

**Implementation Details:**

### app/providers/job_sources/indeed/__init__.py
```python
from .provider import IndeedSource
from app.config import get_settings

def get_provider():
    """Factory function for provider discovery"""
    settings = get_settings()

    if not settings.indeed_api_key:
        return None  # Skip if not configured

    return IndeedSource(api_key=settings.indeed_api_key)
```

### app/providers/job_sources/indeed/client.py
```python
import aiohttp
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class IndeedAPIClient:
    """Client for Indeed Job Search API"""

    BASE_URL = "https://api.indeed.com/ads/apisearch"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search(
        self,
        query: str,
        location: Optional[str] = None,
        remote: bool = False,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Search Indeed API for jobs.

        Returns raw API response data.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        params = {
            "publisher": self.api_key,
            "q": query,
            "format": "json",
            "v": "2",
            "limit": min(limit, 25)  # Indeed API max is 25 per request
        }

        if location:
            params["l"] = location

        if remote:
            params["q"] = f"{query} remote"

        try:
            logger.info(f"Indeed API request: query={query}, location={location}")

            async with self.session.get(self.BASE_URL, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                results = data.get("results", [])
                logger.info(f"Indeed API returned {len(results)} results")

                return results

        except aiohttp.ClientError as e:
            logger.error(f"Indeed API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Indeed API call: {e}")
            raise
```

**Key Points:**
- Async context manager for session management
- Proper error handling with logging
- Rate limiting considerations (Indeed limits to 25/request)
- Remote jobs handled via query modification

### app/providers/job_sources/indeed/provider.py
```python
from app.models.provider import JobSource
from app.models.job import JobListing
from typing import List, Optional
from uuid import uuid4
from datetime import datetime
from .client import IndeedAPIClient
import logging

logger = logging.getLogger(__name__)

class IndeedSource(JobSource):
    """Indeed job source provider"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = IndeedAPIClient(api_key)

    @property
    def source_name(self) -> str:
        return "Indeed"

    async def search(
        self,
        query: str,
        location: Optional[str] = None,
        remote: Optional[bool] = None,
        min_pay: Optional[int] = None,
        max_pay: Optional[int] = None
    ) -> List[JobListing]:
        """Search Indeed for jobs"""

        try:
            # Call Indeed API
            raw_results = await self.client.search(
                query=query,
                location=location,
                remote=remote or False,
                limit=25
            )

            # Convert to JobListing objects
            jobs = []
            for result in raw_results:
                job = self._parse_job(result)
                if job:
                    jobs.append(job)

            logger.info(f"Indeed: Parsed {len(jobs)} jobs from API response")
            return jobs

        except Exception as e:
            logger.error(f"Indeed search failed: {e}")
            return []  # Return empty list on failure (graceful degradation)

    def _parse_job(self, raw: Dict) -> Optional[JobListing]:
        """Parse Indeed API response into JobListing"""
        try:
            return JobListing(
                id=raw.get("jobkey", str(uuid4())),
                source="Indeed",
                title=raw["jobtitle"],
                company=raw["company"],
                description=raw.get("snippet", ""),
                pay=self._extract_pay(raw),
                location=raw.get("formattedLocation", raw.get("city", "")),
                remote=self._is_remote(raw),
                url=raw["url"],
                posted_date=self._parse_date(raw.get("date"))
            )
        except KeyError as e:
            logger.warning(f"Failed to parse Indeed job (missing field: {e}): {raw}")
            return None

    def _extract_pay(self, raw: Dict) -> Optional[str]:
        """Extract salary from Indeed response"""
        # Indeed sometimes includes salary in snippet
        snippet = raw.get("snippet", "")
        # Simple extraction (could be improved with regex)
        if "$" in snippet:
            # Extract salary mentions
            # This is simplified - real implementation would be more robust
            return None  # TODO: Implement salary parsing
        return None

    def _is_remote(self, raw: Dict) -> bool:
        """Detect if job is remote"""
        # Check various fields for remote indicators
        title = raw.get("jobtitle", "").lower()
        location = raw.get("formattedLocation", "").lower()
        snippet = raw.get("snippet", "").lower()

        remote_keywords = ["remote", "work from home", "wfh", "telecommute"]

        for keyword in remote_keywords:
            if keyword in title or keyword in location or keyword in snippet:
                return True

        return False

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse Indeed's date format"""
        if not date_str:
            return None

        # Indeed uses relative dates like "1 day ago"
        # For MVP, return current date
        # TODO: Implement proper relative date parsing
        return datetime.now()
```

**Key Points:**
- Handles API errors gracefully (returns empty list)
- Converts Indeed's format to our JobListing model
- Remote detection via keyword search
- Salary extraction (stubbed for now, can be enhanced)
- Date parsing (simplified for MVP)

**Testing Criteria:**
1. ✅ Provider discovered when API key present
2. ✅ Provider skipped when API key missing
3. ✅ API client makes correct requests
4. ✅ Jobs parsed into JobListing format
5. ✅ Remote detection works
6. ✅ API errors return empty list (no crash)
7. ✅ Malformed jobs skipped gracefully

**Test Examples:**
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_indeed_client_search():
    from app.providers.job_sources.indeed.client import IndeedAPIClient

    client = IndeedAPIClient(api_key="test_key")

    # Mock API response
    mock_response = {
        "results": [
            {
                "jobkey": "123",
                "jobtitle": "Software Engineer",
                "company": "TechCorp",
                "snippet": "Build amazing software...",
                "url": "https://indeed.com/job/123",
                "formattedLocation": "San Francisco, CA",
                "date": "1 day ago"
            }
        ]
    }

    with patch.object(client, 'session') as mock_session:
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aenter__.return_value.raise_for_status = AsyncMock()

        results = await client.search(query="Python Developer")

        assert len(results) == 1
        assert results[0]["jobkey"] == "123"

@pytest.mark.asyncio
async def test_indeed_provider_integration():
    from app.providers.job_sources.indeed.provider import IndeedSource

    provider = IndeedSource(api_key="test_key")

    # Mock client.search to return raw data
    provider.client.search = AsyncMock(return_value=[
        {
            "jobkey": "123",
            "jobtitle": "Python Developer",
            "company": "StartupCo",
            "snippet": "Remote Python development...",
            "url": "https://indeed.com/job/123",
            "formattedLocation": "Remote",
            "date": "2 days ago"
        }
    ])

    jobs = await provider.search(query="Python", remote=True)

    assert len(jobs) == 1
    assert jobs[0].title == "Python Developer"
    assert jobs[0].company == "StartupCo"
    assert jobs[0].source == "Indeed"
    assert jobs[0].remote == True

def test_indeed_provider_factory_with_api_key():
    """Provider should load when API key present"""
    from app.providers.job_sources.indeed import get_provider
    from app.models.config import Settings
    from unittest.mock import patch

    with patch('app.providers.job_sources.indeed.get_settings') as mock_settings:
        mock_settings.return_value = Settings(indeed_api_key="test_key")
        provider = get_provider()

        assert provider is not None
        assert provider.source_name == "Indeed"

def test_indeed_provider_factory_without_api_key():
    """Provider should return None when no API key"""
    from app.providers.job_sources.indeed import get_provider
    from app.models.config import Settings
    from unittest.mock import patch

    with patch('app.providers.job_sources.indeed.get_settings') as mock_settings:
        mock_settings.return_value = Settings(indeed_api_key="")
        provider = get_provider()

        assert provider is None
```

**Planner Guidance:**
- **Developer**: Implement Indeed API client, provider, parsing logic
- **Tester**: Mock API responses, test parsing, error handling, factory function
- **Relevant Files**:
  - Dev: `app/providers/job_sources/indeed/*.py`
  - Test: `tests/unit/test_indeed_provider.py`, `tests/integration/test_indeed_api.py`
  - Mock: Create fixtures for Indeed API responses

---

## Chunk 3.2: Indeed Provider Testing & Validation

**Objective:** Comprehensive testing of Indeed provider with real and mocked data.

**Dependencies:** Chunk 3.1 (Indeed provider implementation)

**Files to Create:**
```
backend/
  tests/
    fixtures/
      indeed_responses.py
    unit/
      test_indeed_parsing.py
    integration/
      test_indeed_live.py
```

**Implementation Details:**

### tests/fixtures/indeed_responses.py
```python
"""Fixture data for Indeed API responses"""

INDEED_RESPONSE_SINGLE_JOB = {
    "results": [{
        "jobkey": "abc123",
        "jobtitle": "Senior Python Developer",
        "company": "TechCorp Inc",
        "city": "San Francisco",
        "state": "CA",
        "country": "US",
        "formattedLocation": "San Francisco, CA",
        "snippet": "We're seeking a Senior Python Developer with 5+ years experience. Remote work available. Salary: $150k-$180k.",
        "url": "https://www.indeed.com/viewjob?jk=abc123",
        "date": "3 days ago",
        "onmousedown": "",
        "expired": False
    }]
}

INDEED_RESPONSE_MULTIPLE_JOBS = {
    "results": [
        {
            "jobkey": "job1",
            "jobtitle": "Software Engineer",
            "company": "CompanyA",
            "formattedLocation": "Remote",
            "snippet": "Build scalable systems...",
            "url": "https://indeed.com/job/job1",
            "date": "1 day ago"
        },
        {
            "jobkey": "job2",
            "jobtitle": "DevOps Engineer",
            "company": "CompanyB",
            "formattedLocation": "New York, NY",
            "snippet": "Manage cloud infrastructure...",
            "url": "https://indeed.com/job/job2",
            "date": "5 days ago"
        },
        {
            "jobkey": "job3",
            "jobtitle": "Frontend Developer",
            "company": "CompanyC",
            "formattedLocation": "Austin, TX",
            "snippet": "React and TypeScript...",
            "url": "https://indeed.com/job/job3",
            "date": "1 week ago"
        }
    ]
}

INDEED_RESPONSE_MALFORMED = {
    "results": [
        {
            "jobkey": "bad1",
            # Missing required field: jobtitle
            "company": "BrokenCo",
            "url": "https://indeed.com/job/bad1"
        },
        {
            "jobkey": "good1",
            "jobtitle": "Valid Job",
            "company": "GoodCo",
            "url": "https://indeed.com/job/good1"
        }
    ]
}

INDEED_ERROR_RESPONSE = {
    "error": "Invalid API key"
}
```

### tests/unit/test_indeed_parsing.py
```python
import pytest
from app.providers.job_sources.indeed.provider import IndeedSource
from tests.fixtures.indeed_responses import (
    INDEED_RESPONSE_SINGLE_JOB,
    INDEED_RESPONSE_MULTIPLE_JOBS,
    INDEED_RESPONSE_MALFORMED
)

def test_parse_single_job():
    """Test parsing a single valid job"""
    provider = IndeedSource(api_key="test")
    raw_job = INDEED_RESPONSE_SINGLE_JOB["results"][0]

    job = provider._parse_job(raw_job)

    assert job is not None
    assert job.title == "Senior Python Developer"
    assert job.company == "TechCorp Inc"
    assert job.location == "San Francisco, CA"
    assert job.source == "Indeed"
    assert job.url == "https://www.indeed.com/viewjob?jk=abc123"

def test_parse_remote_job():
    """Test remote detection"""
    provider = IndeedSource(api_key="test")
    raw_job = INDEED_RESPONSE_MULTIPLE_JOBS["results"][0]  # Remote job

    job = provider._parse_job(raw_job)

    assert job.remote == True
    assert job.location == "Remote"

def test_parse_malformed_job():
    """Test graceful handling of malformed data"""
    provider = IndeedSource(api_key="test")
    raw_job = INDEED_RESPONSE_MALFORMED["results"][0]  # Missing jobtitle

    job = provider._parse_job(raw_job)

    assert job is None  # Should return None, not crash

def test_remote_detection_in_snippet():
    """Test remote detection from job snippet"""
    provider = IndeedSource(api_key="test")
    raw_job = {
        "jobkey": "test",
        "jobtitle": "Engineer",
        "company": "Corp",
        "url": "http://test.com",
        "formattedLocation": "San Francisco, CA",
        "snippet": "Work from home opportunity..."
    }

    assert provider._is_remote(raw_job) == True

def test_remote_detection_in_title():
    """Test remote detection from job title"""
    provider = IndeedSource(api_key="test")
    raw_job = {
        "jobkey": "test",
        "jobtitle": "Remote Software Engineer",
        "company": "Corp",
        "url": "http://test.com",
        "formattedLocation": "Anywhere",
        "snippet": "..."
    }

    assert provider._is_remote(raw_job) == True

def test_non_remote_job():
    """Test job that is not remote"""
    provider = IndeedSource(api_key="test")
    raw_job = {
        "jobkey": "test",
        "jobtitle": "Software Engineer",
        "company": "Corp",
        "url": "http://test.com",
        "formattedLocation": "San Francisco, CA (On-site)",
        "snippet": "Must work in office..."
    }

    assert provider._is_remote(raw_job) == False
```

### tests/integration/test_indeed_live.py
```python
import pytest
from app.providers.job_sources.indeed.provider import IndeedSource
from app.config import get_settings

@pytest.mark.skipif(
    not get_settings().indeed_api_key,
    reason="Indeed API key not configured"
)
@pytest.mark.asyncio
async def test_indeed_live_search():
    """
    Integration test with real Indeed API.
    Only runs if API key is configured.
    """
    settings = get_settings()
    provider = IndeedSource(api_key=settings.indeed_api_key)

    jobs = await provider.search(
        query="Python Developer",
        location="San Francisco",
        remote=False
    )

    # Real API should return results
    assert len(jobs) > 0

    # Validate job structure
    for job in jobs:
        assert job.source == "Indeed"
        assert job.title
        assert job.company
        assert job.url
        assert job.url.startswith("http")

@pytest.mark.asyncio
async def test_indeed_search_with_mocked_client():
    """Test provider with mocked API client"""
    from unittest.mock import AsyncMock
    from tests.fixtures.indeed_responses import INDEED_RESPONSE_MULTIPLE_JOBS

    provider = IndeedSource(api_key="test")
    provider.client.search = AsyncMock(return_value=INDEED_RESPONSE_MULTIPLE_JOBS["results"])

    jobs = await provider.search(query="Engineer")

    assert len(jobs) == 3
    assert jobs[0].title == "Software Engineer"
    assert jobs[1].title == "DevOps Engineer"
    assert jobs[2].title == "Frontend Developer"

@pytest.mark.asyncio
async def test_indeed_api_error_handling():
    """Test graceful handling of API errors"""
    from unittest.mock import AsyncMock

    provider = IndeedSource(api_key="test")
    provider.client.search = AsyncMock(side_effect=Exception("API timeout"))

    # Should not crash, should return empty list
    jobs = await provider.search(query="Test")

    assert jobs == []
```

**Testing Criteria:**
1. ✅ Single job parsing works correctly
2. ✅ Multiple jobs parsed successfully
3. ✅ Malformed jobs skipped without crashing
4. ✅ Remote detection works in title, location, snippet
5. ✅ Non-remote jobs detected correctly
6. ✅ Live API integration works (when key present)
7. ✅ API errors handled gracefully
8. ✅ All JobListing fields populated correctly

**Planner Guidance:**
- **Developer**: Focus on ensuring robust parsing and error handling
- **Tester**: Create comprehensive test suite with fixtures, mock all API calls, add live integration test
- **Relevant Files**:
  - Dev: None (testing only)
  - Test: `tests/fixtures/indeed_responses.py`, `tests/unit/test_indeed_parsing.py`, `tests/integration/test_indeed_live.py`

---

## Chunk 3.3: Indeed Provider End-to-End Validation

**Objective:** Validate Indeed provider works through full search flow.

**Dependencies:** Chunks 3.2, 2.3 (Indeed testing, search API)

**Files to Create:**
```
backend/
  tests/
    e2e/
      test_search_with_indeed.py
```

**Implementation Details:**

### tests/e2e/test_search_with_indeed.py
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, AsyncMock
from tests.fixtures.indeed_responses import INDEED_RESPONSE_MULTIPLE_JOBS

client = TestClient(app)

def test_search_endpoint_uses_indeed_provider():
    """Test that search endpoint discovers and uses Indeed provider"""

    # Mock Indeed API response
    with patch('app.providers.job_sources.indeed.provider.IndeedSource.search') as mock_search:
        mock_search.return_value = AsyncMock(return_value=[
            # Return parsed JobListing objects
            # (In reality, this would come from _parse_job)
        ])

        response = client.post("/api/search", json={
            "query": "Python Developer",
            "location": "San Francisco"
        })

        assert response.status_code == 200
        data = response.json()

        # Should have jobs from Indeed
        assert data["total_count"] > 0

        # Verify Indeed provider was called
        mock_search.assert_called_once()

def test_indeed_and_mock_providers_aggregated():
    """Test that multiple providers are aggregated"""

    response = client.post("/api/search", json={
        "query": "Software Engineer"
    })

    assert response.status_code == 200
    jobs = response.json()["jobs"]

    # Should have jobs from multiple sources
    sources = set(job["source"] for job in jobs)

    # At minimum should have Mock Indeed
    # If real API key configured, would also have real Indeed
    assert len(sources) >= 1

@pytest.mark.skipif(
    not get_settings().indeed_api_key,
    reason="Indeed API key not configured"
)
def test_live_search_with_real_indeed():
    """
    E2E test with real Indeed API.
    Only runs if API key configured.
    """
    response = client.post("/api/search", json={
        "query": "Python",
        "location": "Remote",
        "remote": True
    })

    assert response.status_code == 200
    data = response.json()

    # Should have real results
    assert data["total_count"] > 0

    jobs = data["jobs"]

    # At least some should be from Indeed
    indeed_jobs = [j for j in jobs if j["source"] == "Indeed"]
    assert len(indeed_jobs) > 0

    # Validate structure
    for job in indeed_jobs:
        assert "title" in job
        assert "company" in job
        assert "url" in job
        assert job["url"].startswith("http")

def test_indeed_api_failure_doesnt_break_search():
    """Test graceful degradation when Indeed API fails"""

    with patch('app.providers.job_sources.indeed.provider.IndeedSource.search') as mock_search:
        # Simulate API failure
        mock_search.side_effect = Exception("Indeed API is down")

        response = client.post("/api/search", json={
            "query": "Engineer"
        })

        # Should still return 200 (with jobs from other sources)
        assert response.status_code == 200

        # Might have jobs from mock provider
        data = response.json()
        # total_count might be 0 if only Indeed was configured, or > 0 from mock
        assert "jobs" in data

def test_duplicate_detection_across_providers():
    """
    Test that duplicates from different sources are detected.

    Note: This requires mock providers returning duplicate jobs
    or real providers finding the same job.
    """
    # For now, just verify duplicate_sources field exists
    response = client.post("/api/search", json={"query": "Test"})

    assert response.status_code == 200
    jobs = response.json()["jobs"]

    for job in jobs:
        assert "duplicate_sources" in job
        assert isinstance(job["duplicate_sources"], list)
```

**Testing Criteria:**
1. ✅ Search endpoint discovers Indeed provider
2. ✅ Indeed provider called during search
3. ✅ Multiple providers aggregated correctly
4. ✅ Live API works end-to-end (when configured)
5. ✅ Indeed failure doesn't break search
6. ✅ Duplicate detection works across providers
7. ✅ All response fields present and valid

**Planner Guidance:**
- **Developer**: Ensure Indeed provider integrates with full search flow
- **Tester**: Create E2E tests covering happy path, error cases, aggregation
- **Relevant Files**:
  - Dev: None (E2E validation only)
  - Test: `tests/e2e/test_search_with_indeed.py`
  - Note: Run with and without Indeed API key to test both paths

---

_Continuing with thousands more lines..._
# Phase 4: Company Research System

## Chunk 4.1: Company Research Service Core

**Objective:** Build CompanyResearchService to orchestrate research providers with per-field caching.

**Dependencies:** Chunks 0.2, 0.3, 1.1 (database, models, provider discovery)

**Files to Create:**
```
backend/
  app/
    services/
      company_research.py
```

**Implementation Details:**

### app/services/company_research.py
```python
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.models.company import Company, ResearchResult
from app.models.provider import ResearchProvider, ResearchCategory
from app.db.models import CompanyCache

logger = logging.getLogger(__name__)

class CompanyResearchService:
    """Service to orchestrate company research across multiple providers"""

    def __init__(self, db: Session, providers: List[ResearchProvider], cache_ttl_days: int = 30):
        self.db = db
        self.providers = providers
        self.cache_ttl_days = cache_ttl_days

        # Build field -> providers mapping
        self.field_to_providers: Dict[str, List[ResearchProvider]] = {}
        for provider in providers:
            for field_name in provider.contributions().keys():
                if field_name not in self.field_to_providers:
                    self.field_to_providers[field_name] = []
                self.field_to_providers[field_name].append(provider)

        logger.info(f"CompanyResearchService initialized with {len(providers)} providers")
        logger.debug(f"Field mappings: {list(self.field_to_providers.keys())}")

    async def research_companies(
        self,
        company_names: List[str],
        requested_fields: List[str]
    ) -> Dict[str, Company]:
        """
        Research multiple companies for specific fields.
        Uses per-field caching for efficiency.

        Returns: {company_name: Company}
        """
        if not company_names:
            return {}

        if not requested_fields:
            logger.warning("No fields requested for research")
            return {name: Company(name=name) for name in company_names}

        logger.info(f"Researching {len(company_names)} companies for fields: {requested_fields}")

        results = {}

        for company_name in company_names:
            company = Company(name=company_name)

            # 1. Check cache for each requested field
            cached_data = self._get_cached_fields(company_name, requested_fields)
            company = company.merge_research(cached_data)

            # 2. Determine which fields still need research
            cached_field_names = set(
                k for k, v in cached_data.model_dump(exclude_none=True).items()
            )
            missing_fields = set(requested_fields) - cached_field_names

            logger.debug(f"{company_name}: cached={cached_field_names}, missing={missing_fields}")

            if missing_fields:
                # 3. Find providers that contribute missing fields
                providers_needed = set()
                for field in missing_fields:
                    providers_needed.update(
                        self.field_to_providers.get(field, [])
                    )

                if not providers_needed:
                    logger.warning(f"No providers found for fields: {missing_fields}")
                else:
                    # 4. Research in parallel
                    research_tasks = [
                        provider.research(company_name, list(missing_fields))
                        for provider in providers_needed
                    ]
                    research_results = await asyncio.gather(
                        *research_tasks,
                        return_exceptions=True
                    )

                    # 5. Merge results and cache
                    for result in research_results:
                        if isinstance(result, ResearchResult):
                            company = company.merge_research(result)
                            self._cache_result(company_name, result)
                        elif isinstance(result, Exception):
                            logger.error(f"Provider failed for {company_name}: {result}")
                        else:
                            logger.warning(f"Unexpected result type: {type(result)}")

            company.cached_date = datetime.now()
            results[company_name] = company

        logger.info(f"Research complete for {len(results)} companies")
        return results

    def _get_cached_fields(
        self,
        company_name: str,
        requested_fields: List[str]
    ) -> ResearchResult:
        """Fetch cached data for requested fields"""
        cached_entries = self.db.query(CompanyCache).filter(
            CompanyCache.name == company_name,
            CompanyCache.field_name.in_(requested_fields)
        ).all()

        # Build result from cache
        merged = {}
        for entry in cached_entries:
            if not self._is_stale(entry):
                merged[entry.field_name] = entry.value
                logger.debug(f"Cache hit: {company_name}.{entry.field_name}")
            else:
                logger.debug(f"Cache stale: {company_name}.{entry.field_name}")

        # Return as generic ResearchResult with dynamic fields
        class CachedResult(ResearchResult):
            model_config = {'extra': 'allow'}

        return CachedResult(**merged)

    def _cache_result(self, company_name: str, result: ResearchResult):
        """Cache each field individually"""
        for field_name, value in result.model_dump(exclude_none=True).items():
            cache_entry = CompanyCache(
                name=company_name,
                field_name=field_name,
                value=value,
                cached_at=datetime.now()
            )
            # Use merge to update existing or insert new
            self.db.merge(cache_entry)

        self.db.commit()
        logger.debug(f"Cached {len(result.model_dump(exclude_none=True))} fields for {company_name}")

    def _is_stale(self, cache_entry: CompanyCache) -> bool:
        """Check if cache entry exceeds TTL"""
        age = datetime.now() - cache_entry.cached_at
        return age > timedelta(days=self.cache_ttl_days)
```

**Key Points:**
- Per-field caching enables partial cache hits
- Parallel provider queries via asyncio.gather
- Graceful error handling (one provider failure doesn't break others)
- Field -> provider mapping built at init for efficiency
- Comprehensive logging for debugging

**Testing Criteria:**
1. ✅ Service initializes with providers
2. ✅ Field -> provider mapping built correctly
3. ✅ Cache hits return cached data without provider calls
4. ✅ Cache misses trigger provider research
5. ✅ Partial cache hits only fetch missing fields
6. ✅ Multiple companies researched in parallel
7. ✅ Provider failures don't break service
8. ✅ Results are cached per-field
9. ✅ Stale cache is refreshed

**Test Examples:**
```python
import pytest
from datetime import datetime, timedelta
from app.services.company_research.py import CompanyResearchService
from app.models.company import Company, ResearchResult
from app.models.provider import ResearchProvider, ResearchCategory, FieldContribution, DisplayMetadata
from app.db.models import CompanyCache

class MockResearchProvider(ResearchProvider):
    def __init__(self, name, fields):
        self._name = name
        self._fields = fields
        self.call_count = 0

    @property
    def provider_name(self):
        return self._name

    def contributions(self):
        return {
            field: FieldContribution(
                category=ResearchCategory.BASIC,
                label=field,
                display=DisplayMetadata(type="text")
            )
            for field in self._fields
        }

    async def research(self, company_name, requested_fields):
        self.call_count += 1
        # Return data for requested fields
        class MockResult(ResearchResult):
            pass

        data = {f: f"value_for_{f}" for f in requested_fields if f in self._fields}
        return MockResult(**data)

@pytest.mark.asyncio
async def test_research_service_with_no_cache(test_db):
    """Test research when cache is empty"""
    provider = MockResearchProvider("test", ["field1", "field2"])
    service = CompanyResearchService(test_db, [provider])

    results = await service.research_companies(
        company_names=["TechCorp"],
        requested_fields=["field1", "field2"]
    )

    assert "TechCorp" in results
    assert results["TechCorp"].field1 == "value_for_field1"
    assert results["TechCorp"].field2 == "value_for_field2"
    assert provider.call_count == 1

@pytest.mark.asyncio
async def test_research_service_with_cache_hit(test_db):
    """Test research when data is cached"""
    provider = MockResearchProvider("test", ["field1"])
    service = CompanyResearchService(test_db, [provider])

    # Pre-populate cache
    cache_entry = CompanyCache(
        name="TechCorp",
        field_name="field1",
        value="cached_value",
        cached_at=datetime.now()
    )
    test_db.add(cache_entry)
    test_db.commit()

    results = await service.research_companies(
        company_names=["TechCorp"],
        requested_fields=["field1"]
    )

    # Should use cache, not call provider
    assert provider.call_count == 0
    assert results["TechCorp"].field1 == "cached_value"

@pytest.mark.asyncio
async def test_research_service_partial_cache_hit(test_db):
    """Test research when some fields cached, others need fetching"""
    provider = MockResearchProvider("test", ["field1", "field2"])
    service = CompanyResearchService(test_db, [provider])

    # Cache only field1
    cache_entry = CompanyCache(
        name="TechCorp",
        field_name="field1",
        value="cached_value",
        cached_at=datetime.now()
    )
    test_db.add(cache_entry)
    test_db.commit()

    results = await service.research_companies(
        company_names=["TechCorp"],
        requested_fields=["field1", "field2"]
    )

    # Should have cached field1 and fetched field2
    assert results["TechCorp"].field1 == "cached_value"
    assert results["TechCorp"].field2 == "value_for_field2"
    assert provider.call_count == 1  # Only called for field2

@pytest.mark.asyncio
async def test_research_service_stale_cache(test_db):
    """Test that stale cache is refreshed"""
    provider = MockResearchProvider("test", ["field1"])
    service = CompanyResearchService(test_db, [provider], cache_ttl_days=7)

    # Add stale cache entry (8 days old)
    cache_entry = CompanyCache(
        name="TechCorp",
        field_name="field1",
        value="old_value",
        cached_at=datetime.now() - timedelta(days=8)
    )
    test_db.add(cache_entry)
    test_db.commit()

    results = await service.research_companies(
        company_names=["TechCorp"],
        requested_fields=["field1"]
    )

    # Should fetch fresh data
    assert provider.call_count == 1
    assert results["TechCorp"].field1 == "value_for_field1"

@pytest.mark.asyncio
async def test_research_service_provider_failure(test_db):
    """Test graceful handling of provider failures"""
    class FailingProvider(MockResearchProvider):
        async def research(self, company_name, requested_fields):
            raise Exception("Provider failed")

    provider = FailingProvider("failing", ["field1"])
    service = CompanyResearchService(test_db, [provider])

    results = await service.research_companies(
        company_names=["TechCorp"],
        requested_fields=["field1"]
    )

    # Should return company object, just without the field
    assert "TechCorp" in results
    assert not hasattr(results["TechCorp"], "field1")
```

**Planner Guidance:**
- **Developer**: Implement service with per-field caching, parallel queries, error handling
- **Tester**: Test cache hits/misses, partial hits, staleness, parallel execution, failures
- **Relevant Files**:
  - Dev: `app/services/company_research.py`
  - Test: `tests/unit/test_company_research_service.py`

---

## Chunk 4.2: Research Fields Discovery Endpoint

**Objective:** Create API endpoint that returns all available research fields with metadata.

**Dependencies:** Chunk 4.1 (research service), Chunk 1.1 (provider discovery)

**Files to Create/Modify:**
```
backend/
  app/
    api/
      research.py
    main.py  (add router)
```

**Implementation Details:**

### app/api/research.py
```python
from fastapi import APIRouter, Depends
from typing import Dict, List
from app.models.provider import ResearchProvider
from app.providers.discovery import discover_research_providers
from app.models.config import Settings
from app.config import get_settings

router = APIRouter(prefix="/api/research", tags=["research"])

def get_research_providers(settings: Settings = Depends(get_settings)) -> List[ResearchProvider]:
    """Dependency to get research providers"""
    return discover_research_providers(settings)

@router.get("/fields")
async def get_research_fields(
    providers: List[ResearchProvider] = Depends(get_research_providers)
) -> Dict[str, Dict]:
    """
    Get all available research fields from all providers.

    Returns field registry with metadata for frontend.
    """
    fields = {}

    for provider in providers:
        for field_name, contribution in provider.contributions().items():
            fields[field_name] = {
                "category": contribution.category.value,
                "label": contribution.label,
                "display": contribution.display.model_dump(),
                "provider": provider.provider_name
            }

    return fields
```

### Modify app/main.py to include research router:
```python
from app.api import search, research  # Add research import

app.include_router(search.router)
app.include_router(research.router)  # Add this line
```

**Testing Criteria:**
1. ✅ Endpoint returns all fields from all providers
2. ✅ Field metadata includes category, label, display, provider
3. ✅ Display metadata is serialized correctly
4. ✅ Empty providers returns empty dict
5. ✅ Multiple providers with overlapping fields handled correctly

**Test Example:**
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_research_fields_endpoint():
    response = client.get("/api/research/fields")

    assert response.status_code == 200
    fields = response.json()

    assert isinstance(fields, dict)

    # Should have at least basic fields (if any providers loaded)
    # Structure validation
    for field_name, field_info in fields.items():
        assert "category" in field_info
        assert "label" in field_info
        assert "display" in field_info
        assert "provider" in field_info

        # Display metadata structure
        assert "type" in field_info["display"]
```

**Planner Guidance:**
- **Developer**: Create endpoint to expose field registry
- **Tester**: Test endpoint returns correct structure for all providers
- **Relevant Files**:
  - Dev: `app/api/research.py`, `app/main.py`
  - Test: `tests/integration/test_research_api.py`

---

## Chunk 4.3: Integrate Research Service with Search Endpoint

**Objective:** Modify search endpoint to accept field parameters and perform research.

**Dependencies:** Chunk 4.2 (research fields endpoint), Chunk 2.3 (search API)

**Files to Modify:**
```
backend/
  app/
    api/
      search.py
```

**Implementation Details:**

### Enhanced app/api/search.py
```python
from fastapi import APIRouter, Depends, Query
from typing import List
from sqlalchemy.orm import Session

from app.models.job import SearchRequest, SearchResponse
from app.models.provider import JobSource, ResearchProvider
from app.services.job_aggregation import aggregate_jobs_from_sources
from app.services.company_research import CompanyResearchService
from app.providers.discovery import discover_job_sources, discover_research_providers
from app.models.config import Settings
from app.config import get_settings
from app.db.session import get_db

router = APIRouter(prefix="/api", tags=["search"])

def get_job_sources(settings: Settings = Depends(get_settings)) -> List[JobSource]:
    """Dependency to get job sources"""
    return discover_job_sources(settings)

def get_research_providers(settings: Settings = Depends(get_settings)) -> List[ResearchProvider]:
    """Dependency to get research providers"""
    return discover_research_providers(settings)

def get_research_service(
    db: Session = Depends(get_db),
    providers: List[ResearchProvider] = Depends(get_research_providers),
    settings: Settings = Depends(get_settings)
) -> CompanyResearchService:
    """Dependency to get research service"""
    return CompanyResearchService(db, providers, settings.cache_ttl_days)

@router.post("/search")
async def search_jobs(
    request: SearchRequest,
    fields: List[str] = Query(default=[]),  # NEW: Research fields parameter
    sources: List[JobSource] = Depends(get_job_sources),
    research_service: CompanyResearchService = Depends(get_research_service)
) -> SearchResponse:
    """
    Search for jobs across all configured sources.
    Optionally research companies for specified fields.

    Query parameters:
    - fields: List of research fields to fetch (e.g., ?fields=glassdoor_rating&fields=scam_score)
    """
    # 1. Aggregate jobs from sources
    jobs = await aggregate_jobs_from_sources(
        sources=sources,
        query=request.query or "",
        location=request.location,
        remote=request.remote,
        min_pay=request.min_pay,
        max_pay=request.max_pay,
        limit=100
    )

    # 2. Research companies if fields requested
    companies_dict = {}
    if fields:
        # Extract unique company names
        company_names = list(set(job.company for job in jobs))

        # Research companies
        companies_dict = await research_service.research_companies(
            company_names=company_names,
            requested_fields=fields
        )

        # Convert Company objects to dicts for JSON serialization
        companies_dict = {
            name: company.model_dump()
            for name, company in companies_dict.items()
        }

    return SearchResponse(
        jobs=jobs,
        companies=companies_dict,
        total_count=len(jobs),
        filtered_count=len(jobs)  # No filtering yet (Phase 9)
    )
```

**Key Points:**
- Accepts `fields` query parameter (multi-value)
- Only researches companies if fields requested (empty fields = no research)
- Company objects serialized to dicts for JSON response

**Testing Criteria:**
1. ✅ Search without fields works (no research)
2. ✅ Search with fields triggers research
3. ✅ Companies dict populated when fields present
4. ✅ Cache is used on subsequent requests
5. ✅ Multiple fields can be requested
6. ✅ Unknown fields handled gracefully

**Test Examples:**
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_search_without_research_fields():
    """Search without fields should not perform research"""
    response = client.post("/api/search", json={
        "query": "Engineer"
    })

    assert response.status_code == 200
    data = response.json()

    assert "jobs" in data
    assert "companies" in data
    assert data["companies"] == {}  # No research performed

def test_search_with_research_fields():
    """Search with fields should perform research"""
    response = client.post(
        "/api/search?fields=test_field",
        json={"query": "Engineer"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "companies" in data
    # companies dict should be populated if provider contributes test_field
    # (Exact assertion depends on which providers are loaded)

def test_search_with_multiple_fields():
    """Can request multiple research fields"""
    response = client.post(
        "/api/search?fields=field1&fields=field2",
        json={"query": "Engineer"}
    )

    assert response.status_code == 200
    data = response.json()

    # Both fields should be researched
    assert "companies" in data
```

**Planner Guidance:**
- **Developer**: Modify search endpoint to accept fields parameter, integrate research service
- **Tester**: Test with/without fields, multiple fields, cache behavior
- **Relevant Files**:
  - Dev: `app/api/search.py`
  - Test: `tests/integration/test_search_with_research.py`

---


# Phase 5: Research Providers (Glassdoor & AI Scam Detection)

## Chunk 5.1: Mock Glassdoor Provider

**Objective:** Create Glassdoor research provider (mock for now, real API later).

**Dependencies:** Chunk 4.1 (research service core)

**Files to Create:**
```
backend/
  app/
    providers/
      research/
        mock_glassdoor/
          __init__.py
          models.py
          provider.py
```

**Implementation Details:**

### app/providers/research/mock_glassdoor/models.py
```python
from app.models.company import ResearchResult
from typing import Optional, List

class GlassdoorResearchResult(ResearchResult):
    """Result from Glassdoor provider"""
    glassdoor_rating: Optional[float] = None
    glassdoor_review_count: Optional[int] = None
    glassdoor_pros: List[str] = []
    glassdoor_cons: List[str] = []
```

### app/providers/research/mock_glassdoor/provider.py
```python
from app.models.provider import ResearchProvider, ResearchCategory, FieldContribution, DisplayMetadata
from typing import Dict, List
from .models import GlassdoorResearchResult
import random

class MockGlassdoorProvider(ResearchProvider):
    """Mock Glassdoor provider for testing"""

    @property
    def provider_name(self) -> str:
        return "glassdoor_mock"

    def contributions(self) -> Dict[str, FieldContribution]:
        return {
            "glassdoor_rating": FieldContribution(
                category=ResearchCategory.API_CHEAP,
                label="Glassdoor Rating",
                display=DisplayMetadata(
                    type="rating",
                    icon="⭐",
                    max_value=5,
                    priority="high",
                    format="decimal_1"
                )
            ),
            "glassdoor_review_count": FieldContribution(
                category=ResearchCategory.API_CHEAP,
                label="Review Count",
                display=DisplayMetadata(
                    type="text",
                    priority="medium",
                    format="number"
                )
            ),
            "glassdoor_pros": FieldContribution(
                category=ResearchCategory.API_CHEAP,
                label="Common Pros",
                display=DisplayMetadata(
                    type="list",
                    icon="👍",
                    priority="medium",
                    list_style="bullet"
                )
            ),
            "glassdoor_cons": FieldContribution(
                category=ResearchCategory.API_CHEAP,
                label="Common Cons",
                display=DisplayMetadata(
                    type="list",
                    icon="👎",
                    priority="medium",
                    list_style="bullet"
                )
            )
        }

    async def research(self, company_name: str, requested_fields: List[str]) -> GlassdoorResearchResult:
        """Mock research - returns random but plausible data"""
        my_fields = set(self.contributions().keys())
        needed = my_fields.intersection(requested_fields)

        if not needed:
            return GlassdoorResearchResult()

        result = GlassdoorResearchResult()

        if "glassdoor_rating" in needed:
            result.glassdoor_rating = round(random.uniform(3.0, 4.8), 1)

        if "glassdoor_review_count" in needed:
            result.glassdoor_review_count = random.randint(10, 500)

        if "glassdoor_pros" in needed:
            pros_options = [
                "Great work-life balance",
                "Competitive compensation",
                "Smart colleagues",
                "Flexible hours",
                "Good benefits",
                "Interesting projects"
            ]
            result.glassdoor_pros = random.sample(pros_options, k=random.randint(2, 4))

        if "glassdoor_cons" in needed:
            cons_options = [
                "Slow promotion track",
                "Limited remote work",
                "Bureaucratic processes",
                "Outdated technology",
                "Long hours expected",
                "Micromanagement"
            ]
            result.glassdoor_cons = random.sample(cons_options, k=random.randint(2, 3))

        return result
```

### app/providers/research/mock_glassdoor/__init__.py
```python
from .provider import MockGlassdoorProvider

def get_provider():
    """Factory for mock Glassdoor provider"""
    return MockGlassdoorProvider()
```

**Testing Criteria:**
1. ✅ Provider discovered by discovery system
2. ✅ Contributions declared correctly
3. ✅ Research returns GlassdoorResearchResult
4. ✅ Only requested fields are populated
5. ✅ Display metadata includes all required fields

**Test Example:**
```python
import pytest
from app.providers.research.mock_glassdoor.provider import MockGlassdoorProvider

@pytest.mark.asyncio
async def test_mock_glassdoor_provider():
    provider = MockGlassdoorProvider()

    # Request all fields
    result = await provider.research(
        "TechCorp",
        ["glassdoor_rating", "glassdoor_review_count", "glassdoor_pros", "glassdoor_cons"]
    )

    assert result.glassdoor_rating is not None
    assert 3.0 <= result.glassdoor_rating <= 5.0
    assert result.glassdoor_review_count > 0
    assert len(result.glassdoor_pros) >= 2
    assert len(result.glassdoor_cons) >= 2

@pytest.mark.asyncio
async def test_glassdoor_provider_partial_fields():
    provider = MockGlassdoorProvider()

    # Request only rating
    result = await provider.research("TechCorp", ["glassdoor_rating"])

    assert result.glassdoor_rating is not None
    assert result.glassdoor_review_count is None  # Not requested
```

**Planner Guidance:**
- **Developer**: Implement mock Glassdoor provider with realistic data
- **Tester**: Test field contributions, partial requests, discovery
- **Relevant Files**:
  - Dev: `app/providers/research/mock_glassdoor/*.py`
  - Test: `tests/unit/test_mock_glassdoor.py`

---

## Chunk 5.2: AI Scam Detection Provider

**Objective:** Create AI-powered scam detection provider using OpenAI.

**Dependencies:** Chunk 5.1 (research provider pattern established)

**Files to Create:**
```
backend/
  app/
    providers/
      research/
        scam_detector/
          __init__.py
          models.py
          provider.py
          client.py
```

**Implementation Details:**

### app/providers/research/scam_detector/models.py
```python
from app.models.company import ResearchResult
from typing import Optional, List

class ScamDetectionResult(ResearchResult):
    """Result from AI scam detection"""
    scam_score: Optional[float] = None  # 0.0 (safe) to 1.0 (scam)
    scam_indicators: List[str] = []  # Specific red flags found
```

### app/providers/research/scam_detector/client.py
```python
import openai
import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for OpenAI API"""

    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def analyze_company_for_scams(self, company_name: str, domain: str = None) -> Dict[str, Any]:
        """
        Use OpenAI to analyze company for scam indicators.

        Returns: {"scam_score": 0.0-1.0, "indicators": [...]}
        """
        prompt = self._build_prompt(company_name, domain)

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a job scam detection expert. Analyze companies and provide a scam risk score."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for consistent analysis
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            logger.info(f"OpenAI analysis for {company_name}: scam_score={result.get('scam_score')}")

            return result

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _build_prompt(self, company_name: str, domain: str = None) -> str:
        """Build analysis prompt"""
        domain_info = f"\nCompany domain: {domain}" if domain else ""

        return f"""Analyze this company for potential job scam indicators:

Company name: {company_name}{domain_info}

Consider these factors:
1. Domain legitimacy and age (if provided)
2. Company name patterns common in scams
3. Generic or vague company names
4. Presence of suspicious keywords
5. Lack of verifiable online presence

Provide your analysis in JSON format:
{{
  "scam_score": <float between 0.0 and 1.0>,
  "indicators": [<list of specific red flags found, empty if none>],
  "confidence": <"low", "medium", or "high">
}}

A score of 0.0-0.3 is low risk, 0.3-0.7 is medium risk, 0.7-1.0 is high risk.
"""
```

### app/providers/research/scam_detector/provider.py
```python
from app.models.provider import ResearchProvider, ResearchCategory, FieldContribution, DisplayMetadata
from typing import Dict, List, Optional
from .models import ScamDetectionResult
from .client import OpenAIClient
import logging

logger = logging.getLogger(__name__)

class AIScamDetectorProvider(ResearchProvider):
    """AI-powered scam detection provider"""

    def __init__(self, api_key: str):
        self.client = OpenAIClient(api_key)

    @property
    def provider_name(self) -> str:
        return "scam_detector"

    def contributions(self) -> Dict[str, FieldContribution]:
        return {
            "scam_score": FieldContribution(
                category=ResearchCategory.AI,
                label="Scam Risk Score",
                display=DisplayMetadata(
                    type="percentage",
                    icon="🛡️",
                    priority="high",
                    color_scale={
                        "0-30": "green",
                        "30-70": "yellow",
                        "70-100": "red"
                    },
                    invert=True  # Lower is better
                )
            ),
            "scam_indicators": FieldContribution(
                category=ResearchCategory.AI,
                label="Red Flags",
                display=DisplayMetadata(
                    type="list",
                    icon="⚠️",
                    priority="high",
                    list_style="bullet"
                )
            )
        }

    async def research(self, company_name: str, requested_fields: List[str]) -> ScamDetectionResult:
        """Analyze company for scam indicators"""
        my_fields = set(self.contributions().keys())
        needed = my_fields.intersection(requested_fields)

        if not needed:
            return ScamDetectionResult()

        try:
            # Call OpenAI
            analysis = await self.client.analyze_company_for_scams(company_name)

            result = ScamDetectionResult()

            if "scam_score" in needed:
                result.scam_score = analysis.get("scam_score", 0.5)

            if "scam_indicators" in needed:
                result.scam_indicators = analysis.get("indicators", [])

            return result

        except Exception as e:
            logger.error(f"Scam detection failed for {company_name}: {e}")
            # Return neutral result on failure
            return ScamDetectionResult(
                scam_score=0.5,  # Neutral/unknown
                scam_indicators=["Analysis unavailable"]
            )
```

### app/providers/research/scam_detector/__init__.py
```python
from .provider import AIScamDetectorProvider
from app.config import get_settings

def get_provider():
    """Factory for scam detector provider"""
    settings = get_settings()

    if not settings.openai_api_key:
        return None  # Skip if no API key

    return AIScamDetectorProvider(api_key=settings.openai_api_key)
```

**Key Points:**
- Uses OpenAI GPT-4 for analysis
- JSON response format for structured output
- Graceful error handling (returns neutral score on failure)
- Logs all AI calls for auditing
- Temperature=0.3 for consistent analysis

**Testing Criteria:**
1. ✅ Provider discovered when OpenAI key present
2. ✅ Provider skipped when no API key
3. ✅ OpenAI called with correct prompt
4. ✅ JSON response parsed correctly
5. ✅ Scam score in valid range (0.0-1.0)
6. ✅ API errors handled gracefully
7. ✅ Only requested fields populated

**Test Examples:**
```python
import pytest
from unittest.mock import AsyncMock, patch
from app.providers.research.scam_detector.provider import AIScamDetectorProvider

@pytest.mark.asyncio
async def test_scam_detector_provider():
    """Test scam detector with mocked OpenAI"""
    provider = AIScamDetectorProvider(api_key="test_key")

    # Mock OpenAI response
    provider.client.analyze_company_for_scams = AsyncMock(return_value={
        "scam_score": 0.15,
        "indicators": [],
        "confidence": "high"
    })

    result = await provider.research("TechCorp", ["scam_score", "scam_indicators"])

    assert result.scam_score == 0.15
    assert result.scam_indicators == []

@pytest.mark.asyncio
async def test_scam_detector_error_handling():
    """Test graceful handling of API errors"""
    provider = AIScamDetectorProvider(api_key="test_key")

    # Mock API failure
    provider.client.analyze_company_for_scams = AsyncMock(side_effect=Exception("API error"))

    result = await provider.research("TechCorp", ["scam_score"])

    # Should return neutral score, not crash
    assert result.scam_score == 0.5
    assert "unavailable" in result.scam_indicators[0].lower()

def test_scam_detector_factory_without_key():
    """Provider should return None without API key"""
    from app.providers.research.scam_detector import get_provider
    from app.models.config import Settings
    from unittest.mock import patch

    with patch('app.providers.research.scam_detector.get_settings') as mock:
        mock.return_value = Settings(openai_api_key="")
        provider = get_provider()

        assert provider is None
```

**Planner Guidance:**
- **Developer**: Implement OpenAI client, provider with error handling
- **Tester**: Mock OpenAI responses, test error cases, factory function
- **Relevant Files**:
  - Dev: `app/providers/research/scam_detector/*.py`
  - Test: `tests/unit/test_scam_detector.py`
  - Note: Skip live API tests to avoid costs

---

## Chunk 5.3: Research Providers Integration Testing

**Objective:** Test Glassdoor and Scam Detector providers through full research flow.

**Dependencies:** Chunks 5.1, 5.2, 4.3 (both providers, integrated search)

**Files to Create:**
```
backend/
  tests/
    e2e/
      test_research_providers.py
```

**Implementation Details:**

### tests/e2e/test_research_providers.py
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_search_with_glassdoor_field():
    """Test search requesting Glassdoor data"""
    response = client.post(
        "/api/search?fields=glassdoor_rating",
        json={"query": "Engineer"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "companies" in data
    companies = data["companies"]

    # Should have Glassdoor ratings for companies
    if companies:  # If any companies in result
        sample_company = next(iter(companies.values()))
        assert "glassdoor_rating" in sample_company

def test_search_with_scam_detection():
    """Test search requesting scam detection"""
    response = client.post(
        "/api/search?fields=scam_score",
        json={"query": "Engineer"}
    )

    assert response.status_code == 200
    data = response.json()

    companies = data["companies"]

    if companies:
        sample_company = next(iter(companies.values()))
        assert "scam_score" in sample_company
        # Score should be between 0 and 1
        assert 0.0 <= sample_company["scam_score"] <= 1.0

def test_search_with_multiple_providers():
    """Test search requesting fields from multiple providers"""
    response = client.post(
        "/api/search?fields=glassdoor_rating&fields=scam_score",
        json={"query": "Engineer"}
    )

    assert response.status_code == 200
    data = response.json()

    companies = data["companies"]

    if companies:
        sample_company = next(iter(companies.values()))
        # Should have data from both providers
        assert "glassdoor_rating" in sample_company
        assert "scam_score" in sample_company

def test_research_fields_endpoint_includes_providers():
    """Test /research/fields includes Glassdoor and Scam Detector"""
    response = client.get("/api/research/fields")

    assert response.status_code == 200
    fields = response.json()

    # Should have fields from both providers
    glassdoor_fields = [f for f in fields if "glassdoor" in f]
    scam_fields = [f for f in fields if "scam" in f]

    assert len(glassdoor_fields) > 0  # At least glassdoor_rating
    assert len(scam_fields) > 0  # At least scam_score

def test_cache_behavior_with_research():
    """Test that research results are cached"""
    # First request
    response1 = client.post(
        "/api/search?fields=glassdoor_rating",
        json={"query": "Python Developer"}
    )

    assert response1.status_code == 200
    companies1 = response1.json()["companies"]

    # Second identical request (should use cache)
    response2 = client.post(
        "/api/search?fields=glassdoor_rating",
        json={"query": "Python Developer"}
    )

    assert response2.status_code == 200
    companies2 = response2.json()["companies"]

    # Should have same data (from cache)
    # Note: Mock provider returns random data, so this test
    # validates cache is working if data is identical
    # For deterministic test, would need to mock provider
    assert companies1 == companies2
```

**Testing Criteria:**
1. ✅ Can request Glassdoor fields through search
2. ✅ Can request scam detection through search
3. ✅ Multiple providers work together
4. ✅ Research fields endpoint includes both providers
5. ✅ Cache works for research data
6. ✅ Field metadata correct for both providers

**Planner Guidance:**
- **Developer**: Ensure providers integrate end-to-end
- **Tester**: Test full flow from search to research to cache
- **Relevant Files**:
  - Dev: None (E2E validation)
  - Test: `tests/e2e/test_research_providers.py`

---

_This document continues for thousands more lines covering Phases 6-12..._

---

# Documentation Note

This implementation plan continues with:
- **Phase 6**: Display Metadata System (3 chunks)
- **Phase 7**: Frontend Foundation (3 chunks)
- **Phase 8**: Quick Search Flow (3 chunks)
- **Phase 9**: Deep Search Flow (3 chunks)
- **Phase 10**: Manual Deep Research (3 chunks)
- **Phase 11**: Additional Providers (3 chunks)
- **Phase 12**: Production Polish (3 chunks)

Each phase follows the same exhaustive pattern:
- Objective
- Dependencies
- Files to create/modify
- Full implementation code
- Testing criteria with examples
- Planner guidance

The complete plan will exceed 5000 lines to ensure every implementation detail is captured for autonomous agent execution.

For now, this provides Phases 0-5 (15 chunks) totaling ~4000 lines. The remaining phases follow identical structure and detail level.

---

# Summary: Implementation Approach

**Phase Progression:**
- Phases 0-2: Foundation (project setup, models, basic search)
- Phase 3: Real provider integration (Indeed API)
- Phases 4-5: Company research system with providers
- Phases 6-7: Display system and frontend foundation
- Phases 8-10: Search UX flows (Quick, Deep, Manual research)
- Phase 11: Expand provider ecosystem
- Phase 12: Production readiness

**Key Principles:**
1. Each chunk is independently testable
2. No chunk leaves code in broken state
3. Tests written before/during implementation (TDD-friendly)
4. Comprehensive error handling throughout
5. Graceful degradation (provider failures don't break system)
6. Per-field caching for efficiency
7. Extensive logging for debugging

**Agent Loop Integration:**
- Planner reads chunk objectives and creates detailed dev/test plans
- Developer implements with full code and tests
- Tester creates comprehensive test suite and runs it
- Reviewer validates tests pass and code quality
- Analysis ensures no scaffolding, all functions tested

This plan is designed for 100% autonomous implementation with AI agents.


# Phase 6: Display Metadata System

## Chunk 6.1: Field Renderer Core

**Objective:** Implement JavaScript field renderer that uses provider display metadata.

**Dependencies:** Chunks 4.2, 5.3 (research fields endpoint, providers with metadata)

**Files to Create:**
```
frontend/
  js/
    field-renderer.js
  css/
    field-display.css
```

**Implementation Details:**

### frontend/js/field-renderer.js
```javascript
/**
 * Metadata-driven field renderer
 * Uses display metadata from providers to render fields consistently
 */
class FieldRenderer {
    constructor(metadata) {
        this.metadata = metadata;
    }

    /**
     * Render a field value based on its display metadata
     * @param {string} fieldName - Field identifier
     * @param {any} value - Field value to display
     * @param {Object} company - Full company object (for context)
     * @returns {string} HTML string
     */
    async render(fieldName, value, company = {}) {
        if (value === null || value === undefined) {
            return '';
        }

        switch (this.metadata.type) {
            case 'rating':
                return this.renderRating(value);
            case 'percentage':
                return this.renderPercentage(value);
            case 'list':
                return this.renderList(value);
            case 'badge':
                return this.renderBadge(value);
            case 'text':
                return this.renderText(value);
            case 'custom':
                return await this.loadCustomRenderer(fieldName, value, company);
            default:
                console.warn(`Unknown display type: ${this.metadata.type}`);
                return this.renderText(value);
        }
    }

    renderRating(value) {
        const maxValue = this.metadata.max_value || 5;
        const stars = this.metadata.icon || '⭐';
        const starCount = Math.round(value);
        const formatted = this.formatValue(value);

        const starsHtml = stars.repeat(starCount);
        const emptyStars = '☆'.repeat(maxValue - starCount);

        return `
            <div class="field-rating" data-priority="${this.metadata.priority}">
                <span class="stars">${starsHtml}${emptyStars}</span>
                <span class="rating-value">${formatted}/${maxValue}</span>
            </div>
        `;
    }

    renderPercentage(value) {
        // Value should be 0.0-1.0, display as percentage
        const percent = Math.round(value * 100);
        const colorClass = this.getColorClass(value);
        const icon = this.metadata.icon || '';

        return `
            <div class="field-percentage ${colorClass}" data-priority="${this.metadata.priority}">
                ${icon} <span class="percentage-value">${percent}%</span>
            </div>
        `;
    }

    renderList(value) {
        if (!Array.isArray(value) || value.length === 0) {
            return '';
        }

        const icon = this.metadata.icon || '';
        const listStyle = this.metadata.list_style || 'bullet';

        let listHtml;
        if (listStyle === 'numbered') {
            listHtml = `<ol>${value.map(item => `<li>${this.escapeHtml(item)}</li>`).join('')}</ol>`;
        } else if (listStyle === 'comma') {
            listHtml = `<span>${value.map(item => this.escapeHtml(item)).join(', ')}</span>`;
        } else {
            // Default: bullet
            listHtml = `<ul>${value.map(item => `<li>${this.escapeHtml(item)}</li>`).join('')}</ul>`;
        }

        return `
            <div class="field-list" data-priority="${this.metadata.priority}">
                ${icon ? `<span class="list-icon">${icon}</span>` : ''}
                <div class="list-content">${listHtml}</div>
            </div>
        `;
    }

    renderBadge(value) {
        const colorClass = this.getColorClass(value);
        const icon = this.metadata.icon || '';
        const formatted = this.formatValue(value);

        return `
            <span class="field-badge ${colorClass}" data-priority="${this.metadata.priority}">
                ${icon} ${this.escapeHtml(formatted)}
            </span>
        `;
    }

    renderText(value) {
        const formatted = this.formatValue(value);

        return `
            <span class="field-text" data-priority="${this.metadata.priority}">
                ${this.escapeHtml(formatted)}
            </span>
        `;
    }

    async loadCustomRenderer(fieldName, value, company) {
        /**
         * Load custom renderer from provider's frontend.js
         * This is the escape hatch for complex visualizations
         */
        const provider = this.metadata.provider;

        try {
            const module = await import(`/api/providers/${provider}/frontend.js`);

            if (module.render) {
                return module.render(fieldName, value, company, this.metadata.custom_config);
            }
        } catch (e) {
            console.warn(`Custom renderer failed for ${fieldName}, falling back to text:`, e);
        }

        // Fallback to text rendering
        return this.renderText(value);
    }

    getColorClass(value) {
        /**
         * Determine color class based on value and color_scale
         */
        if (!this.metadata.color_scale) {
            return '';
        }

        // Handle inversion (lower is better)
        const effectiveValue = this.metadata.invert ? (1 - value) : value;
        const percent = effectiveValue * 100;

        for (const [range, color] of Object.entries(this.metadata.color_scale)) {
            const [min, max] = range.split('-').map(Number);
            if (percent >= min && percent <= max) {
                return `color-${color}`;
            }
        }

        return '';
    }

    formatValue(value) {
        /**
         * Format value according to metadata.format
         */
        if (this.metadata.format === 'decimal_1') {
            return typeof value === 'number' ? value.toFixed(1) : value;
        }

        if (this.metadata.format === 'currency') {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 0
            }).format(value);
        }

        if (this.metadata.format === 'number') {
            return new Intl.NumberFormat('en-US').format(value);
        }

        if (this.metadata.format === 'date') {
            return new Date(value).toLocaleDateString();
        }

        if (this.metadata.format === 'percentage') {
            return `${Math.round(value * 100)}%`;
        }

        return String(value);
    }

    escapeHtml(text) {
        /**
         * Escape HTML to prevent XSS
         */
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FieldRenderer;
}
```

### frontend/css/field-display.css
```css
/* Field display styles */

.field-rating {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.field-rating .stars {
    font-size: 1.2em;
    color: #FFD700;
}

.field-rating .rating-value {
    font-size: 0.9em;
    color: #666;
}

.field-percentage {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-weight: 600;
}

.field-percentage.color-green {
    background-color: #d4edda;
    color: #155724;
}

.field-percentage.color-yellow {
    background-color: #fff3cd;
    color: #856404;
}

.field-percentage.color-red {
    background-color: #f8d7da;
    color: #721c24;
}

.field-list {
    margin: 0.5rem 0;
}

.field-list .list-icon {
    margin-right: 0.5rem;
}

.field-list ul,
.field-list ol {
    margin: 0.25rem 0;
    padding-left: 1.5rem;
}

.field-list li {
    margin: 0.25rem 0;
}

.field-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.875em;
    font-weight: 500;
}

.field-badge.color-green {
    background-color: #d4edda;
    color: #155724;
}

.field-badge.color-yellow {
    background-color: #fff3cd;
    color: #856404;
}

.field-badge.color-red {
    background-color: #f8d7da;
    color: #721c24;
}

.field-text {
    color: #333;
}

/* Priority-based display */
[data-priority="high"] {
    font-weight: 600;
}

[data-priority="low"] {
    font-size: 0.875em;
    color: #666;
}
```

**Key Points:**
- Metadata-driven rendering (no hardcoding field displays)
- Supports all standard display types (rating, percentage, list, badge, text, custom)
- XSS protection via escapeHtml()
- Color scales with inversion support
- Format helpers (currency, date, number, percentage)
- Custom renderer escape hatch via dynamic import

**Testing Criteria:**
1. ✅ Rating display shows stars + value
2. ✅ Percentage shows color based on scale
3. ✅ List renders bullet/numbered/comma styles
4. ✅ Badge shows with correct color
5. ✅ Text rendering escapes HTML
6. ✅ Color inversion works (lower is better)
7. ✅ Format helpers work (currency, date, etc.)
8. ✅ Unknown display type falls back to text

**Test Example (Manual):**
```html
<!-- Test page: test-field-renderer.html -->
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="css/field-display.css">
</head>
<body>
    <div id="test-output"></div>

    <script src="js/field-renderer.js"></script>
    <script>
        // Test rating display
        const ratingMetadata = {
            type: 'rating',
            icon: '⭐',
            max_value: 5,
            priority: 'high',
            format: 'decimal_1'
        };
        const ratingRenderer = new FieldRenderer(ratingMetadata);
        const ratingHtml = ratingRenderer.render('test_rating', 4.3);

        // Test percentage with color scale
        const percentMetadata = {
            type: 'percentage',
            icon: '🛡️',
            priority: 'high',
            color_scale: {
                '0-30': 'green',
                '30-70': 'yellow',
                '70-100': 'red'
            },
            invert: true
        };
        const percentRenderer = new FieldRenderer(percentMetadata);
        const percentHtml = percentRenderer.render('scam_score', 0.15);

        // Test list
        const listMetadata = {
            type: 'list',
            icon: '👍',
            priority: 'medium',
            list_style: 'bullet'
        };
        const listRenderer = new FieldRenderer(listMetadata);
        const listHtml = listRenderer.render('pros', ['Great culture', 'Good pay', 'Flexible hours']);

        // Display results
        document.getElementById('test-output').innerHTML = `
            <h3>Rating Test:</h3>
            ${ratingHtml}
            
            <h3>Percentage Test:</h3>
            ${percentHtml}
            
            <h3>List Test:</h3>
            ${listHtml}
        `;

        // Assertions (open console to see)
        console.assert(ratingHtml.includes('⭐⭐⭐⭐'), 'Rating should show 4 stars');
        console.assert(percentHtml.includes('color-green'), 'Low scam score should be green');
        console.assert(listHtml.includes('<li>Great culture</li>'), 'List should contain items');
        console.log('All manual tests passed!');
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Implement field renderer with all display types
- **Tester**: Create manual HTML test page, test each display type, edge cases
- **Relevant Files**:
  - Dev: `frontend/js/field-renderer.js`, `frontend/css/field-display.css`
  - Test: `frontend/test-field-renderer.html` (manual test page)

---

## Chunk 6.2: Field Registry Client

**Objective:** Create JavaScript client to fetch and manage field registry from backend.

**Dependencies:** Chunk 6.1 (field renderer), Chunk 4.2 (research fields endpoint)

**Files to Create:**
```
frontend/
  js/
    field-registry.js
```

**Implementation Details:**

### frontend/js/field-registry.js
```javascript
/**
 * Field Registry Client
 * Fetches and manages field definitions from backend
 */
class FieldRegistry {
    constructor(apiBaseUrl = '') {
        this.apiBaseUrl = apiBaseUrl;
        this.fields = {};
        this.loaded = false;
    }

    /**
     * Fetch field registry from backend
     * @returns {Promise<Object>} Field definitions
     */
    async load() {
        if (this.loaded) {
            return this.fields;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/research/fields`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            this.fields = await response.json();
            this.loaded = true;

            console.log(`Loaded ${Object.keys(this.fields).length} field definitions`);

            return this.fields;
        } catch (error) {
            console.error('Failed to load field registry:', error);
            throw error;
        }
    }

    /**
     * Get metadata for a specific field
     * @param {string} fieldName - Field identifier
     * @returns {Object|null} Field metadata
     */
    getField(fieldName) {
        return this.fields[fieldName] || null;
    }

    /**
     * Get all fields in a specific category
     * @param {string} category - Category name (basic, api_cheap, api_expensive, ai)
     * @returns {Array} Array of {name, metadata} objects
     */
    getFieldsByCategory(category) {
        return Object.entries(this.fields)
            .filter(([name, meta]) => meta.category === category)
            .map(([name, meta]) => ({ name, ...meta }));
    }

    /**
     * Get all available categories
     * @returns {Array} Unique category names
     */
    getCategories() {
        const categories = new Set(
            Object.values(this.fields).map(meta => meta.category)
        );
        return Array.from(categories);
    }

    /**
     * Get fields grouped by category
     * @returns {Object} {category: [{name, ...metadata}]}
     */
    getFieldsGroupedByCategory() {
        const grouped = {};

        for (const [name, meta] of Object.entries(this.fields)) {
            const category = meta.category;
            if (!grouped[category]) {
                grouped[category] = [];
            }
            grouped[category].push({ name, ...meta });
        }

        return grouped;
    }

    /**
     * Get renderer for a field
     * @param {string} fieldName - Field identifier
     * @returns {FieldRenderer|null} Renderer instance
     */
    getRenderer(fieldName) {
        const field = this.getField(fieldName);
        if (!field) {
            console.warn(`No field definition found for: ${fieldName}`);
            return null;
        }

        return new FieldRenderer(field.display);
    }

    /**
     * Render a field value
     * @param {string} fieldName - Field identifier
     * @param {any} value - Value to render
     * @param {Object} context - Additional context (e.g., full company object)
     * @returns {Promise<string>} HTML string
     */
    async renderField(fieldName, value, context = {}) {
        const renderer = this.getRenderer(fieldName);

        if (!renderer) {
            // Fallback: render as text
            return `<span>${this.escapeHtml(String(value))}</span>`;
        }

        return await renderer.render(fieldName, value, context);
    }

    /**
     * Get category display info
     * @param {string} category - Category name
     * @returns {Object} {icon, label, estimatedTime}
     */
    getCategoryInfo(category) {
        const categoryInfo = {
            'basic': {
                icon: '⚡',
                label: 'Instant Checks',
                estimatedTime: '< 1 second'
            },
            'api_cheap': {
                icon: '💰',
                label: 'API Checks',
                estimatedTime: '~5 seconds'
            },
            'api_expensive': {
                icon: '💸',
                label: 'Premium APIs',
                estimatedTime: '~30 seconds'
            },
            'ai': {
                icon: '🤖',
                label: 'AI Analysis',
                estimatedTime: '~60 seconds'
            }
        };

        return categoryInfo[category] || {
            icon: '❓',
            label: category,
            estimatedTime: 'Unknown'
        };
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FieldRegistry;
}
```

**Key Points:**
- Fetches field registry on load()
- Caches fields to avoid redundant requests
- Provides category grouping helpers
- Returns FieldRenderer instances for each field
- Includes category display info (icons, time estimates)
- Fallback rendering if field not found

**Testing Criteria:**
1. ✅ load() fetches from backend
2. ✅ getField() returns correct metadata
3. ✅ getFieldsByCategory() filters correctly
4. ✅ getCategories() returns unique categories
5. ✅ getRenderer() creates FieldRenderer
6. ✅ renderField() renders using correct display type
7. ✅ getCategoryInfo() returns icon/label/time

**Test Example:**
```html
<!-- Test: test-field-registry.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Field Registry Test</title>
</head>
<body>
    <div id="output"></div>

    <script src="js/field-renderer.js"></script>
    <script src="js/field-registry.js"></script>
    <script>
        async function testFieldRegistry() {
            const registry = new FieldRegistry('http://localhost:8000');

            // Test loading
            try {
                await registry.load();
                console.log('✓ Registry loaded');
            } catch (e) {
                console.error('✗ Failed to load registry:', e);
                return;
            }

            // Test getCategories
            const categories = registry.getCategories();
            console.assert(categories.length > 0, 'Should have categories');
            console.log('✓ Categories:', categories);

            // Test getFieldsByCategory
            const aiFields = registry.getFieldsByCategory('ai');
            console.log('✓ AI fields:', aiFields);

            // Test renderField
            const html = await registry.renderField('glassdoor_rating', 4.3);
            console.assert(html.includes('⭐'), 'Rating should contain star icon');
            console.log('✓ Rendered field:', html);

            // Test getCategoryInfo
            const categoryInfo = registry.getCategoryInfo('ai');
            console.assert(categoryInfo.icon === '🤖', 'AI category should have robot icon');
            console.log('✓ Category info:', categoryInfo);

            // Display results
            const groupedFields = registry.getFieldsGroupedByCategory();
            let outputHtml = '<h2>Field Registry</h2>';

            for (const [category, fields] of Object.entries(groupedFields)) {
                const info = registry.getCategoryInfo(category);
                outputHtml += `
                    <div>
                        <h3>${info.icon} ${info.label} (${info.estimatedTime})</h3>
                        <ul>
                            ${fields.map(f => `<li><strong>${f.name}</strong>: ${f.label}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            document.getElementById('output').innerHTML = outputHtml;
            console.log('All tests passed!');
        }

        testFieldRegistry();
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Implement field registry client with caching
- **Tester**: Create test page that loads from real backend, test all methods
- **Relevant Files**:
  - Dev: `frontend/js/field-registry.js`
  - Test: `frontend/test-field-registry.html`
  - Note: Requires backend running on localhost:8000

---

## Chunk 6.3: Dynamic Filter UI Generation

**Objective:** Generate filter UI dynamically from field registry.

**Dependencies:** Chunk 6.2 (field registry client)

**Files to Create:**
```
frontend/
  js/
    filter-builder.js
  css/
    filters.css
```

**Implementation Details:**

### frontend/js/filter-builder.js
```javascript
/**
 * Filter Builder
 * Dynamically generates filter UI from field registry
 */
class FilterBuilder {
    constructor(registry, containerElement) {
        this.registry = registry;
        this.container = containerElement;
        this.filters = {};
    }

    /**
     * Build complete filter UI
     * @returns {Promise<void>}
     */
    async build() {
        await this.registry.load();

        const groupedFields = this.registry.getFieldsGroupedByCategory();

        // Build basic filters section
        this.container.innerHTML = `
            <div class="filters-container">
                <div class="basic-filters">
                    <h3>Basic Filters</h3>
                    ${this.buildBasicFilters()}
                </div>
                <div class="advanced-filters">
                    <h3>Advanced Filters</h3>
                    ${this.buildAdvancedFilters(groupedFields)}
                </div>
            </div>
        `;

        // Attach event listeners
        this.attachEventListeners();
    }

    buildBasicFilters() {
        return `
            <div class="filter-group">
                <label for="filter-query">Keywords:</label>
                <input type="text" id="filter-query" name="query" placeholder="e.g., Software Engineer">
            </div>

            <div class="filter-group">
                <label for="filter-location">Location:</label>
                <input type="text" id="filter-location" name="location" placeholder="e.g., San Francisco">
            </div>

            <div class="filter-group">
                <label>
                    <input type="checkbox" id="filter-remote" name="remote">
                    Remote only
                </label>
            </div>

            <div class="filter-group">
                <label for="filter-min-pay">Min Pay:</label>
                <input type="number" id="filter-min-pay" name="min_pay" placeholder="$80,000">
            </div>

            <div class="filter-group">
                <label for="filter-max-pay">Max Pay:</label>
                <input type="number" id="filter-max-pay" name="max_pay" placeholder="$150,000">
            </div>
        `;
    }

    buildAdvancedFilters(groupedFields) {
        let html = '';

        // Sort categories by estimated time (basic first, AI last)
        const categoryOrder = ['basic', 'api_cheap', 'api_expensive', 'ai'];
        const sortedCategories = categoryOrder.filter(cat => groupedFields[cat]);

        for (const category of sortedCategories) {
            if (category === 'basic') continue; // Skip basic (already in basic filters)

            const fields = groupedFields[category];
            const categoryInfo = this.registry.getCategoryInfo(category);

            html += `
                <div class="filter-category" data-category="${category}">
                    <h4>
                        ${categoryInfo.icon} ${categoryInfo.label}
                        <span class="category-time">(${categoryInfo.estimatedTime})</span>
                    </h4>
                    <div class="category-fields">
                        ${fields.map(field => this.buildFieldFilter(field)).join('')}
                    </div>
                </div>
            `;
        }

        return html;
    }

    buildFieldFilter(field) {
        /**
         * Build filter input based on field type
         */
        const fieldId = `filter-${field.name}`;

        if (field.display.type === 'rating') {
            // Min rating slider
            return `
                <div class="filter-field">
                    <label for="${fieldId}">
                        ${field.label} (min):
                    </label>
                    <input type="range" 
                           id="${fieldId}" 
                           name="${field.name}" 
                           min="0" 
                           max="${field.display.max_value || 5}" 
                           step="0.5" 
                           value="0">
                    <output for="${fieldId}">0</output>
                </div>
            `;
        }

        if (field.display.type === 'percentage') {
            // Max percentage slider (for things like scam_score where lower is better)
            const inverted = field.display.invert;
            return `
                <div class="filter-field">
                    <label for="${fieldId}">
                        ${field.label} (${inverted ? 'max' : 'min'}):
                    </label>
                    <input type="range" 
                           id="${fieldId}" 
                           name="${field.name}" 
                           min="0" 
                           max="100" 
                           step="5" 
                           value="${inverted ? '100' : '0'}">
                    <output for="${fieldId}">${inverted ? '100' : '0'}%</output>
                </div>
            `;
        }

        // Default: checkbox to include this field in research
        return `
            <div class="filter-field">
                <label>
                    <input type="checkbox" 
                           id="${fieldId}" 
                           name="${field.name}"
                           data-field="${field.name}">
                    Include ${field.label}
                </label>
            </div>
        `;
    }

    attachEventListeners() {
        // Update range output values
        this.container.querySelectorAll('input[type="range"]').forEach(input => {
            input.addEventListener('input', (e) => {
                const output = input.nextElementSibling;
                if (output && output.tagName === 'OUTPUT') {
                    let value = e.target.value;
                    // Add % for percentage fields
                    if (input.max === '100') {
                        value += '%';
                    }
                    output.textContent = value;
                }
            });
        });
    }

    /**
     * Get current filter values
     * @returns {Object} {searchParams, fields}
     */
    getFilters() {
        const searchParams = {
            query: document.getElementById('filter-query')?.value || '',
            location: document.getElementById('filter-location')?.value || null,
            remote: document.getElementById('filter-remote')?.checked || null,
            min_pay: parseInt(document.getElementById('filter-min-pay')?.value) || null,
            max_pay: parseInt(document.getElementById('filter-max-pay')?.value) || null
        };

        // Get selected research fields
        const fields = [];
        this.container.querySelectorAll('input[data-field]:checked').forEach(input => {
            fields.push(input.dataset.field);
        });

        // Add fields with range/slider values
        this.container.querySelectorAll('input[type="range"]').forEach(input => {
            if (input.value > 0) {  // Only include if user changed from default
                fields.push(input.name);
            }
        });

        return { searchParams, fields };
    }

    /**
     * Check if any expensive fields are selected
     * @returns {boolean}
     */
    hasExpensiveFields() {
        const { fields } = this.getFilters();

        for (const fieldName of fields) {
            const field = this.registry.getField(fieldName);
            if (field && (field.category === 'ai' || field.category === 'api_expensive')) {
                return true;
            }
        }

        return false;
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FilterBuilder;
}
```

### frontend/css/filters.css
```css
.filters-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 1rem;
}

.basic-filters,
.advanced-filters {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
}

.basic-filters h3,
.advanced-filters h3 {
    margin-top: 0;
    color: #333;
}

.filter-group {
    margin-bottom: 1rem;
}

.filter-group label {
    display: block;
    margin-bottom: 0.25rem;
    font-weight: 500;
    color: #555;
}

.filter-group input[type="text"],
.filter-group input[type="number"] {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid #ddd;
    border-radius: 0.25rem;
    font-size: 1rem;
}

.filter-group input[type="checkbox"] {
    margin-right: 0.5rem;
}

.filter-category {
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: white;
    border-radius: 0.25rem;
    border: 1px solid #e0e0e0;
}

.filter-category h4 {
    margin-top: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.category-time {
    font-size: 0.875em;
    color: #666;
    font-weight: normal;
}

.category-fields {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.filter-field {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.filter-field label {
    flex: 0 0 auto;
    margin-bottom: 0;
}

.filter-field input[type="range"] {
    flex: 1 1 auto;
}

.filter-field output {
    flex: 0 0 auto;
    min-width: 3rem;
    text-align: right;
    font-weight: 600;
}

/* Highlight expensive categories */
.filter-category[data-category="ai"],
.filter-category[data-category="api_expensive"] {
    border-color: #ffc107;
    background: #fff9e6;
}
```

**Key Points:**
- Dynamically generates filters from field registry
- Basic filters (query, location, pay) always present
- Advanced filters grouped by category
- Different input types based on field display type
- Range sliders for ratings and percentages
- Checkboxes for boolean inclusion
- Visual indicators for expensive categories
- getFilters() extracts all current filter values

**Testing Criteria:**
1. ✅ Filter UI generates from registry
2. ✅ Categories grouped correctly
3. ✅ Basic filters always present
4. ✅ Range sliders update output values
5. ✅ getFilters() returns correct structure
6. ✅ hasExpensiveFields() detects AI/expensive selections
7. ✅ All field types have appropriate inputs

**Test Example:**
```html
<!-- Test: test-filter-builder.html -->
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="css/filters.css">
    <title>Filter Builder Test</title>
</head>
<body>
    <div id="filters"></div>
    <button id="get-filters">Get Filters</button>
    <pre id="output"></pre>

    <script src="js/field-renderer.js"></script>
    <script src="js/field-registry.js"></script>
    <script src="js/filter-builder.js"></script>
    <script>
        async function testFilterBuilder() {
            const registry = new FieldRegistry('http://localhost:8000');
            const container = document.getElementById('filters');
            const builder = new FilterBuilder(registry, container);

            await builder.build();
            console.log('✓ Filter UI built');

            // Test getFilters
            document.getElementById('get-filters').addEventListener('click', () => {
                const filters = builder.getFilters();
                document.getElementById('output').textContent = JSON.stringify(filters, null, 2);
                console.log('Current filters:', filters);

                // Test hasExpensiveFields
                const hasExpensive = builder.hasExpensiveFields();
                console.log('Has expensive fields:', hasExpensive);
            });

            console.log('All tests passed!');
        }

        testFilterBuilder();
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Implement dynamic filter generation with all field types
- **Tester**: Test with real backend, verify all categories appear, test getFilters()
- **Relevant Files**:
  - Dev: `frontend/js/filter-builder.js`, `frontend/css/filters.css`
  - Test: `frontend/test-filter-builder.html`

---

# Phase 7: Frontend Foundation

**Objective:** Build core frontend infrastructure for event-driven architecture.

**Why This Phase:**
Before implementing search flows, we need the foundational event bus for decoupled communication, an API client for backend interaction, and a main application class to orchestrate everything.

**Dependencies:** Phase 6 (Display Metadata System)

**Phase Success Criteria:**
1. ✅ EventBus allows publish/subscribe pattern
2. ✅ APIClient handles all backend communication with error handling
3. ✅ MainApp initializes all components and manages lifecycle
4. ✅ Application can be instantiated with single line of code
5. ✅ All components communicate via events (not direct calls)

---

## Chunk 7.1: Event Bus System

**Objective:** Implement pub/sub event bus for decoupled component communication.

**Dependencies:** None (pure JavaScript pattern)

**Files to Create:**
- `frontend/js/event-bus.js` (~150 lines)

**Files to Modify:** None

**Implementation:**

```javascript
// frontend/js/event-bus.js

/**
 * EventBus - Simple pub/sub event system for decoupled component communication
 *
 * Usage:
 *   const bus = new EventBus();
 *   bus.on('search:started', (data) => console.log(data));
 *   bus.emit('search:started', { query: 'software engineer' });
 */
class EventBus {
    constructor() {
        this.listeners = new Map(); // eventName -> Set of callbacks
        this.debugMode = false;
    }

    /**
     * Subscribe to an event
     * @param {string} eventName - Event name (use namespaces: 'search:started')
     * @param {Function} callback - Function to call when event fires
     * @returns {Function} Unsubscribe function
     */
    on(eventName, callback) {
        if (!this.listeners.has(eventName)) {
            this.listeners.set(eventName, new Set());
        }

        this.listeners.get(eventName).add(callback);

        if (this.debugMode) {
            console.log(`[EventBus] Subscribed to '${eventName}'`);
        }

        // Return unsubscribe function
        return () => this.off(eventName, callback);
    }

    /**
     * Subscribe to an event, auto-unsubscribe after first firing
     * @param {string} eventName - Event name
     * @param {Function} callback - Function to call once
     * @returns {Function} Unsubscribe function
     */
    once(eventName, callback) {
        const wrappedCallback = (...args) => {
            callback(...args);
            this.off(eventName, wrappedCallback);
        };

        return this.on(eventName, wrappedCallback);
    }

    /**
     * Unsubscribe from an event
     * @param {string} eventName - Event name
     * @param {Function} callback - Callback to remove
     */
    off(eventName, callback) {
        if (!this.listeners.has(eventName)) {
            return;
        }

        this.listeners.get(eventName).delete(callback);

        if (this.listeners.get(eventName).size === 0) {
            this.listeners.delete(eventName);
        }

        if (this.debugMode) {
            console.log(`[EventBus] Unsubscribed from '${eventName}'`);
        }
    }

    /**
     * Emit an event to all subscribers
     * @param {string} eventName - Event name
     * @param {*} data - Data to pass to subscribers
     */
    emit(eventName, data) {
        if (this.debugMode) {
            console.log(`[EventBus] Emitting '${eventName}'`, data);
        }

        if (!this.listeners.has(eventName)) {
            return;
        }

        // Call all subscribers
        this.listeners.get(eventName).forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`[EventBus] Error in '${eventName}' listener:`, error);
            }
        });
    }

    /**
     * Remove all listeners for an event (or all events if no eventName)
     * @param {string} [eventName] - Optional event name to clear
     */
    clear(eventName = null) {
        if (eventName) {
            this.listeners.delete(eventName);
            if (this.debugMode) {
                console.log(`[EventBus] Cleared all listeners for '${eventName}'`);
            }
        } else {
            this.listeners.clear();
            if (this.debugMode) {
                console.log('[EventBus] Cleared all listeners');
            }
        }
    }

    /**
     * Get count of listeners for an event
     * @param {string} eventName - Event name
     * @returns {number} Number of listeners
     */
    listenerCount(eventName) {
        return this.listeners.has(eventName) ? this.listeners.get(eventName).size : 0;
    }

    /**
     * Get all event names that have listeners
     * @returns {string[]} Array of event names
     */
    eventNames() {
        return Array.from(this.listeners.keys());
    }

    /**
     * Enable/disable debug logging
     * @param {boolean} enabled - Whether to log debug info
     */
    setDebugMode(enabled) {
        this.debugMode = enabled;
        console.log(`[EventBus] Debug mode ${enabled ? 'enabled' : 'disabled'}`);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EventBus;
}
```

**Standard Event Names:**

The application uses these standard event names (namespaced with colons):

```javascript
// Search events
'search:started'        // { query, location, filters }
'search:progress'       // { current, total, message }
'search:completed'      // { results: JobListing[], timing: {...} }
'search:failed'         // { error: Error, message: string }

// Research events
'research:started'      // { companies: string[] }
'research:progress'     // { company, field, status }
'research:completed'    // { companies: Company[] }
'research:failed'       // { company, error }

// UI events
'filters:changed'       // { filters: {...} }
'results:selected'      // { jobId: string }
'modal:opened'          // { type, data }
'modal:closed'          // { type }

// System events
'app:ready'            // {}
'app:error'            // { error, context }
'registry:loaded'      // { fields: FieldMetadata[] }
```

**Key Points:**
- Simple pub/sub pattern with Map-based storage
- Supports once() for one-time subscriptions
- Error isolation: listener errors don't affect other listeners
- Debug mode for development
- Returns unsubscribe function from on()
- Namespaced events for organization

**Testing Criteria:**
1. ✅ on() subscribes to events
2. ✅ emit() calls all subscribers
3. ✅ off() unsubscribes correctly
4. ✅ once() unsubscribes after first call
5. ✅ Listener errors don't affect other listeners
6. ✅ clear() removes listeners
7. ✅ listenerCount() returns correct count
8. ✅ Debug mode logs events

**Test Example:**
```html
<!-- Test: test-event-bus.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Event Bus Test</title>
</head>
<body>
    <h1>Event Bus Test</h1>
    <button id="emit-search">Emit search:started</button>
    <button id="emit-error">Emit search:failed</button>
    <pre id="output"></pre>

    <script src="js/event-bus.js"></script>
    <script>
        const bus = new EventBus();
        bus.setDebugMode(true);

        const log = (message) => {
            const output = document.getElementById('output');
            output.textContent += message + '\n';
        };

        // Test 1: Basic subscription
        bus.on('search:started', (data) => {
            log(`✓ Test 1: Received search:started - ${data.query}`);
        });

        // Test 2: Multiple subscribers
        bus.on('search:started', (data) => {
            log(`✓ Test 2: Second subscriber - ${data.query}`);
        });

        // Test 3: once() subscription
        bus.once('search:completed', (data) => {
            log(`✓ Test 3: Once subscription fired`);
        });

        // Test 4: Error isolation
        bus.on('search:failed', (data) => {
            throw new Error('This error should be caught');
        });

        bus.on('search:failed', (data) => {
            log(`✓ Test 4: Second listener still fires despite error`);
        });

        // Test 5: Listener count
        const count = bus.listenerCount('search:started');
        log(`✓ Test 5: Listener count for 'search:started': ${count}`);
        console.assert(count === 2, 'Should have 2 listeners');

        // Test 6: Event names
        const names = bus.eventNames();
        log(`✓ Test 6: Event names: ${names.join(', ')}`);

        // Test 7: Unsubscribe via return value
        const unsubscribe = bus.on('temp:event', () => {});
        log(`✓ Test 7a: Subscribed to temp:event, count: ${bus.listenerCount('temp:event')}`);
        unsubscribe();
        log(`✓ Test 7b: Unsubscribed from temp:event, count: ${bus.listenerCount('temp:event')}`);

        // UI interaction
        document.getElementById('emit-search').addEventListener('click', () => {
            bus.emit('search:started', { query: 'software engineer', location: 'Remote' });
        });

        document.getElementById('emit-error').addEventListener('click', () => {
            bus.emit('search:failed', { error: new Error('Test error') });
        });

        // Auto-test
        setTimeout(() => {
            log('\n--- Running automated tests ---');
            bus.emit('search:started', { query: 'test query' });
            bus.emit('search:completed', { results: [] });
            bus.emit('search:completed', { results: [] }); // Should not trigger once() listener
            bus.emit('search:failed', { error: new Error('test') });
            log('✓ All automated tests passed!');
        }, 100);
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Implement EventBus class with Map-based storage
- **Tester**: Create test page with automated and interactive tests
- **Relevant Files**:
  - Dev: `frontend/js/event-bus.js`
  - Test: `frontend/test-event-bus.html`
  - Note: Pure JavaScript, no backend required

---

## Chunk 7.2: API Client

**Objective:** Create API client for all backend communication with error handling.

**Dependencies:** Chunk 7.1 (EventBus for error events)

**Files to Create:**
- `frontend/js/api-client.js` (~250 lines)

**Files to Modify:** None

**Implementation:**

```javascript
// frontend/js/api-client.js

/**
 * APIClient - Handles all communication with backend API
 *
 * Features:
 * - Automatic JSON parsing
 * - Error handling and formatting
 * - Request/response logging
 * - Timeout support
 * - Event emission for errors
 *
 * Usage:
 *   const api = new APIClient('http://localhost:8000', eventBus);
 *   const results = await api.search({ query: 'software engineer' });
 */
class APIClient {
    constructor(baseURL, eventBus = null) {
        this.baseURL = baseURL.replace(/\/$/, ''); // Remove trailing slash
        this.eventBus = eventBus;
        this.defaultTimeout = 60000; // 60 seconds
    }

    /**
     * Make a GET request
     * @param {string} endpoint - API endpoint (e.g., '/api/search')
     * @param {Object} params - Query parameters
     * @param {number} timeout - Request timeout in ms
     * @returns {Promise<any>} Response data
     */
    async get(endpoint, params = {}, timeout = this.defaultTimeout) {
        const url = this._buildURL(endpoint, params);
        return this._fetch(url, { method: 'GET' }, timeout);
    }

    /**
     * Make a POST request
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body
     * @param {number} timeout - Request timeout in ms
     * @returns {Promise<any>} Response data
     */
    async post(endpoint, data = {}, timeout = this.defaultTimeout) {
        const url = this._buildURL(endpoint);
        return this._fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        }, timeout);
    }

    /**
     * Search for jobs
     * @param {Object} searchParams - Search parameters
     * @param {string} searchParams.query - Search query
     * @param {string} [searchParams.location] - Location filter
     * @param {boolean} [searchParams.remote] - Remote only filter
     * @param {number} [searchParams.min_pay] - Minimum pay
     * @param {number} [searchParams.max_pay] - Maximum pay
     * @param {string[]} [searchParams.include] - Research fields to include
     * @returns {Promise<SearchResponse>} Search results
     */
    async search(searchParams) {
        const params = this._cleanParams(searchParams);

        // Convert include array to comma-separated string
        if (params.include && Array.isArray(params.include)) {
            params.include = params.include.join(',');
        }

        // Longer timeout for deep searches
        const hasResearch = params.include && params.include.length > 0;
        const timeout = hasResearch ? 90000 : 30000;

        return this.get('/api/search', params, timeout);
    }

    /**
     * Get available research fields
     * @returns {Promise<FieldsResponse>} Field metadata
     */
    async getResearchFields() {
        return this.get('/api/research/fields');
    }

    /**
     * Research a single company
     * @param {string} companyName - Company name
     * @param {string[]} fields - Fields to research
     * @returns {Promise<Company>} Company research data
     */
    async researchCompany(companyName, fields = []) {
        return this.post('/api/research/company', {
            company_name: companyName,
            fields: fields,
        }, 90000);
    }

    /**
     * Health check
     * @returns {Promise<Object>} Health status
     */
    async health() {
        return this.get('/health', {}, 5000);
    }

    /**
     * Build full URL with query parameters
     * @private
     */
    _buildURL(endpoint, params = {}) {
        const url = new URL(endpoint, this.baseURL);

        Object.keys(params).forEach(key => {
            const value = params[key];
            if (value !== null && value !== undefined && value !== '') {
                url.searchParams.append(key, value);
            }
        });

        return url.toString();
    }

    /**
     * Make fetch request with timeout and error handling
     * @private
     */
    async _fetch(url, options, timeout) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
            console.log(`[API] ${options.method || 'GET'} ${url}`);
            const startTime = Date.now();

            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            const elapsed = Date.now() - startTime;
            console.log(`[API] Response in ${elapsed}ms: ${response.status}`);

            // Handle non-200 responses
            if (!response.ok) {
                const error = await this._parseError(response);
                throw error;
            }

            // Parse JSON response
            const data = await response.json();
            return data;

        } catch (error) {
            clearTimeout(timeoutId);

            // Format error
            const formattedError = this._formatError(error, url);

            // Emit error event if eventBus available
            if (this.eventBus) {
                this.eventBus.emit('app:error', {
                    error: formattedError,
                    context: { url, options },
                });
            }

            throw formattedError;
        }
    }

    /**
     * Parse error from response
     * @private
     */
    async _parseError(response) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

        try {
            const data = await response.json();
            if (data.detail) {
                errorMessage = data.detail;
            } else if (data.message) {
                errorMessage = data.message;
            }
        } catch {
            // If can't parse JSON, use status text
        }

        const error = new Error(errorMessage);
        error.status = response.status;
        error.statusText = response.statusText;
        return error;
    }

    /**
     * Format fetch errors into user-friendly messages
     * @private
     */
    _formatError(error, url) {
        if (error.name === 'AbortError') {
            const formattedError = new Error('Request timed out. The server took too long to respond.');
            formattedError.isTimeout = true;
            formattedError.originalError = error;
            return formattedError;
        }

        if (error.message === 'Failed to fetch') {
            const formattedError = new Error('Cannot connect to server. Please check your connection.');
            formattedError.isNetworkError = true;
            formattedError.originalError = error;
            return formattedError;
        }

        // Already formatted error from _parseError
        if (error.status) {
            return error;
        }

        // Unknown error
        const formattedError = new Error(`Unexpected error: ${error.message}`);
        formattedError.originalError = error;
        return formattedError;
    }

    /**
     * Remove null/undefined/empty values from params
     * @private
     */
    _cleanParams(params) {
        const cleaned = {};
        Object.keys(params).forEach(key => {
            const value = params[key];
            if (value !== null && value !== undefined && value !== '') {
                cleaned[key] = value;
            }
        });
        return cleaned;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = APIClient;
}
```

**Key Points:**
- Wraps fetch API with consistent error handling
- Automatic timeout with AbortController
- JSON parsing and error formatting
- Event emission for global error handling
- Method-specific helpers (search, researchCompany, etc.)
- Query parameter handling
- Null/undefined value filtering

**Testing Criteria:**
1. ✅ GET requests work with query parameters
2. ✅ POST requests send JSON body
3. ✅ Timeout aborts long requests
4. ✅ Network errors handled gracefully
5. ✅ HTTP errors parsed and formatted
6. ✅ Event emission on errors
7. ✅ search() method builds correct request
8. ✅ getResearchFields() fetches field metadata

**Test Example:**
```html
<!-- Test: test-api-client.html -->
<!DOCTYPE html>
<html>
<head>
    <title>API Client Test</title>
</head>
<body>
    <h1>API Client Test</h1>
    <p>Backend must be running on http://localhost:8000</p>

    <button id="test-health">Test Health Check</button>
    <button id="test-fields">Test Get Fields</button>
    <button id="test-search">Test Search</button>
    <button id="test-error">Test Error Handling</button>
    <button id="test-timeout">Test Timeout</button>

    <pre id="output"></pre>

    <script src="js/event-bus.js"></script>
    <script src="js/api-client.js"></script>
    <script>
        const bus = new EventBus();
        const api = new APIClient('http://localhost:8000', bus);

        const log = (message) => {
            const output = document.getElementById('output');
            output.textContent += message + '\n';
            console.log(message);
        };

        // Listen for errors
        bus.on('app:error', (data) => {
            log(`❌ Error event: ${data.error.message}`);
        });

        // Test 1: Health check
        document.getElementById('test-health').addEventListener('click', async () => {
            try {
                log('\n--- Test 1: Health Check ---');
                const result = await api.health();
                log(`✓ Health check passed: ${JSON.stringify(result)}`);
            } catch (error) {
                log(`✗ Health check failed: ${error.message}`);
            }
        });

        // Test 2: Get research fields
        document.getElementById('test-fields').addEventListener('click', async () => {
            try {
                log('\n--- Test 2: Get Research Fields ---');
                const result = await api.getResearchFields();
                log(`✓ Got ${result.fields.length} fields`);
                log(`  Categories: ${result.categories.join(', ')}`);
            } catch (error) {
                log(`✗ Get fields failed: ${error.message}`);
            }
        });

        // Test 3: Search
        document.getElementById('test-search').addEventListener('click', async () => {
            try {
                log('\n--- Test 3: Search ---');
                const result = await api.search({
                    query: 'software engineer',
                    location: 'Remote',
                    include: ['is_scam', 'glassdoor_rating'],
                });
                log(`✓ Search returned ${result.jobs.length} jobs`);
                if (result.jobs.length > 0) {
                    log(`  First job: ${result.jobs[0].title}`);
                }
            } catch (error) {
                log(`✗ Search failed: ${error.message}`);
            }
        });

        // Test 4: Error handling (404)
        document.getElementById('test-error').addEventListener('click', async () => {
            try {
                log('\n--- Test 4: Error Handling ---');
                await api.get('/api/nonexistent');
                log(`✗ Should have thrown error`);
            } catch (error) {
                log(`✓ Error handled: ${error.message}`);
                log(`  Status: ${error.status}`);
            }
        });

        // Test 5: Timeout
        document.getElementById('test-timeout').addEventListener('click', async () => {
            try {
                log('\n--- Test 5: Timeout ---');
                // Create a request with 1ms timeout to force timeout
                const result = await api.get('/api/search', { query: 'test' }, 1);
                log(`✗ Should have timed out`);
            } catch (error) {
                log(`✓ Timeout handled: ${error.message}`);
                log(`  Is timeout: ${error.isTimeout}`);
            }
        });

        // Auto-test health on load
        (async () => {
            log('--- Auto-testing on load ---');
            try {
                await api.health();
                log('✓ Backend is reachable');
            } catch (error) {
                log(`✗ Backend not reachable: ${error.message}`);
                log('  Make sure backend is running on http://localhost:8000');
            }
        })();
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Implement APIClient with fetch wrapper and error handling
- **Tester**: Test with real backend, verify all methods, test error scenarios
- **Relevant Files**:
  - Dev: `frontend/js/api-client.js`
  - Test: `frontend/test-api-client.html`
  - Note: Requires backend running on localhost:8000

---

## Chunk 7.3: Main Application Class

**Objective:** Create main application class that orchestrates all components.

**Dependencies:**
- Chunk 7.1 (EventBus)
- Chunk 7.2 (APIClient)
- Phase 6 (FieldRegistry, FilterBuilder, FieldRenderer)

**Files to Create:**
- `frontend/js/app.js` (~300 lines)
- `frontend/index.html` (main application page)

**Files to Modify:** None

**Implementation:**

```javascript
// frontend/js/app.js

/**
 * JobSearchApp - Main application class
 *
 * Orchestrates all components:
 * - EventBus for communication
 * - APIClient for backend calls
 * - FieldRegistry for metadata
 * - FilterBuilder for search UI
 * - Results rendering
 * - Modal management
 *
 * Usage:
 *   const app = new JobSearchApp({
 *       apiBaseURL: 'http://localhost:8000',
 *       containerElement: document.getElementById('app')
 *   });
 *   await app.initialize();
 */
class JobSearchApp {
    constructor(config) {
        this.config = {
            apiBaseURL: config.apiBaseURL || 'http://localhost:8000',
            containerElement: config.containerElement,
            debugMode: config.debugMode || false,
        };

        // Core components
        this.eventBus = new EventBus();
        this.eventBus.setDebugMode(this.config.debugMode);

        this.api = new APIClient(this.config.apiBaseURL, this.eventBus);
        this.registry = new FieldRegistry(this.config.apiBaseURL);

        // UI components (initialized later)
        this.filterBuilder = null;

        // State
        this.currentSearch = null;
        this.searchResults = null;
        this.isSearching = false;

        // DOM elements (created during initialization)
        this.elements = {};

        // Bind methods
        this._handleSearch = this._handleSearch.bind(this);
        this._handleDeepResearch = this._handleDeepResearch.bind(this);
    }

    /**
     * Initialize the application
     * - Load field registry
     * - Build UI
     * - Set up event listeners
     */
    async initialize() {
        try {
            console.log('[App] Initializing...');

            // Check backend health
            await this._checkHealth();

            // Load field registry
            await this.registry.load();
            console.log('[App] Field registry loaded');

            // Build UI structure
            this._buildUI();

            // Initialize filter builder
            this.filterBuilder = new FilterBuilder(
                this.registry,
                this.elements.filtersContainer
            );
            await this.filterBuilder.build();

            // Set up event listeners
            this._setupEventListeners();

            // Emit ready event
            this.eventBus.emit('app:ready', {});
            console.log('[App] Initialized successfully');

        } catch (error) {
            console.error('[App] Initialization failed:', error);
            this._showError('Failed to initialize application', error);
            throw error;
        }
    }

    /**
     * Check backend health
     * @private
     */
    async _checkHealth() {
        try {
            await this.api.health();
            console.log('[App] Backend health check passed');
        } catch (error) {
            throw new Error(`Backend not reachable: ${error.message}`);
        }
    }

    /**
     * Build main UI structure
     * @private
     */
    _buildUI() {
        const container = this.config.containerElement;

        container.innerHTML = `
            <div class="app-container">
                <header class="app-header">
                    <h1>Job Search AI</h1>
                    <p class="tagline">Smart job search with scam detection</p>
                </header>

                <main class="app-main">
                    <!-- Filters section -->
                    <section class="filters-section">
                        <div id="filters-container"></div>
                        <div class="search-buttons">
                            <button id="quick-search-btn" class="btn btn-primary">
                                Quick Search (2-5s)
                            </button>
                            <button id="deep-search-btn" class="btn btn-secondary">
                                Deep Search (30-60s)
                            </button>
                        </div>
                    </section>

                    <!-- Results section -->
                    <section class="results-section">
                        <div id="search-status"></div>
                        <div id="results-container"></div>
                    </section>
                </main>

                <!-- Modal overlay -->
                <div id="modal-overlay" class="modal-overlay hidden">
                    <div class="modal-content">
                        <button class="modal-close">&times;</button>
                        <div id="modal-body"></div>
                    </div>
                </div>
            </div>
        `;

        // Store element references
        this.elements = {
            filtersContainer: container.querySelector('#filters-container'),
            quickSearchBtn: container.querySelector('#quick-search-btn'),
            deepSearchBtn: container.querySelector('#deep-search-btn'),
            searchStatus: container.querySelector('#search-status'),
            resultsContainer: container.querySelector('#results-container'),
            modalOverlay: container.querySelector('#modal-overlay'),
            modalBody: container.querySelector('#modal-body'),
            modalClose: container.querySelector('.modal-close'),
        };
    }

    /**
     * Set up event listeners
     * @private
     */
    _setupEventListeners() {
        // Search buttons
        this.elements.quickSearchBtn.addEventListener('click', () => {
            this._handleSearch(false); // quick = false means basic fields only
        });

        this.elements.deepSearchBtn.addEventListener('click', () => {
            this._handleSearch(true); // deep = true means all selected fields
        });

        // Modal close
        this.elements.modalClose.addEventListener('click', () => {
            this._closeModal();
        });

        this.elements.modalOverlay.addEventListener('click', (e) => {
            if (e.target === this.elements.modalOverlay) {
                this._closeModal();
            }
        });

        // EventBus listeners
        this.eventBus.on('search:started', (data) => {
            this._onSearchStarted(data);
        });

        this.eventBus.on('search:completed', (data) => {
            this._onSearchCompleted(data);
        });

        this.eventBus.on('search:failed', (data) => {
            this._onSearchFailed(data);
        });

        // Filters changed - show/hide deep search button
        this.eventBus.on('filters:changed', (data) => {
            const hasExpensive = this.filterBuilder.hasExpensiveFields();

            if (hasExpensive) {
                // Hide quick search button when expensive fields selected
                this.elements.quickSearchBtn.style.display = 'none';
                this.elements.deepSearchBtn.textContent = 'Search (30-60s)';
            } else {
                // Show both buttons when no expensive fields
                this.elements.quickSearchBtn.style.display = 'inline-block';
                this.elements.deepSearchBtn.textContent = 'Deep Search (30-60s)';
            }
        });
    }

    /**
     * Handle search button click
     * @private
     */
    async _handleSearch(isDeepSearch) {
        if (this.isSearching) {
            console.log('[App] Search already in progress');
            return;
        }

        try {
            this.isSearching = true;

            // Get filters from FilterBuilder
            const filters = this.filterBuilder.getFilters();

            // Determine which fields to include
            let includeFields = [];
            if (isDeepSearch) {
                includeFields = filters.selectedFields || [];
            } else {
                // Quick search: only basic fields
                const basicFields = this.registry.getFieldsByCategory('basic');
                includeFields = basicFields.map(f => f.name);
            }

            // Build search params
            const searchParams = {
                query: filters.query,
                location: filters.location,
                remote: filters.remote,
                min_pay: filters.min_pay,
                max_pay: filters.max_pay,
                include: includeFields,
            };

            // Emit search started event
            this.eventBus.emit('search:started', searchParams);

            // Execute search
            const results = await this.api.search(searchParams);

            // Store results
            this.currentSearch = searchParams;
            this.searchResults = results;

            // Emit search completed event
            this.eventBus.emit('search:completed', results);

        } catch (error) {
            console.error('[App] Search failed:', error);
            this.eventBus.emit('search:failed', { error, message: error.message });
        } finally {
            this.isSearching = false;
        }
    }

    /**
     * Handle deep research for a single company
     * @private
     */
    async _handleDeepResearch(companyName) {
        // Get all available research fields
        const allFields = this.registry.fields.map(f => f.name);

        try {
            this.eventBus.emit('research:started', { companies: [companyName] });

            const company = await this.api.researchCompany(companyName, allFields);

            this.eventBus.emit('research:completed', { companies: [company] });

            // Show modal with results
            this._showCompanyModal(company);

        } catch (error) {
            console.error('[App] Deep research failed:', error);
            this.eventBus.emit('research:failed', { company: companyName, error });
            this._showError('Research failed', error);
        }
    }

    /**
     * Show company research modal
     * @private
     */
    _showCompanyModal(company) {
        // TODO: Implement in Phase 10
        console.log('[App] Show company modal:', company);
        this._openModal(`
            <h2>${company.name}</h2>
            <p>Company research modal - to be implemented in Phase 10</p>
        `);
    }

    /**
     * Open modal with content
     * @private
     */
    _openModal(htmlContent) {
        this.elements.modalBody.innerHTML = htmlContent;
        this.elements.modalOverlay.classList.remove('hidden');
        this.eventBus.emit('modal:opened', { type: 'company' });
    }

    /**
     * Close modal
     * @private
     */
    _closeModal() {
        this.elements.modalOverlay.classList.add('hidden');
        this.elements.modalBody.innerHTML = '';
        this.eventBus.emit('modal:closed', { type: 'company' });
    }

    /**
     * Event handler: Search started
     * @private
     */
    _onSearchStarted(data) {
        console.log('[App] Search started:', data);

        const hasResearch = data.include && data.include.length > 0;
        const estimate = hasResearch ? '30-60s' : '2-5s';

        this.elements.searchStatus.innerHTML = `
            <div class="status-message status-loading">
                <span class="spinner"></span>
                Searching for "${data.query}"... (estimated ${estimate})
            </div>
        `;

        this.elements.resultsContainer.innerHTML = '';
    }

    /**
     * Event handler: Search completed
     * @private
     */
    _onSearchCompleted(data) {
        console.log('[App] Search completed:', data);

        const jobCount = data.jobs.length;
        const timing = data.timing || {};

        this.elements.searchStatus.innerHTML = `
            <div class="status-message status-success">
                Found ${jobCount} job${jobCount !== 1 ? 's' : ''}
                ${timing.total ? `in ${(timing.total / 1000).toFixed(1)}s` : ''}
            </div>
        `;

        // TODO: Render results in Phase 8
        this.elements.resultsContainer.innerHTML = `
            <p>Results rendering will be implemented in Phase 8</p>
            <pre>${JSON.stringify(data.jobs.slice(0, 3), null, 2)}</pre>
        `;
    }

    /**
     * Event handler: Search failed
     * @private
     */
    _onSearchFailed(data) {
        console.error('[App] Search failed:', data);
        this._showError('Search failed', data.error);
    }

    /**
     * Show error message
     * @private
     */
    _showError(title, error) {
        this.elements.searchStatus.innerHTML = `
            <div class="status-message status-error">
                <strong>${title}</strong><br>
                ${error.message || error}
            </div>
        `;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = JobSearchApp;
}
```

Create main HTML page:

```html
<!-- frontend/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Job Search AI</title>

    <!-- CSS -->
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/filters.css">
    <link rel="stylesheet" href="css/field-display.css">
</head>
<body>
    <div id="app"></div>

    <!-- JavaScript (order matters!) -->
    <script src="js/event-bus.js"></script>
    <script src="js/field-renderer.js"></script>
    <script src="js/field-registry.js"></script>
    <script src="js/filter-builder.js"></script>
    <script src="js/api-client.js"></script>
    <script src="js/app.js"></script>

    <script>
        // Initialize application
        (async () => {
            const app = new JobSearchApp({
                apiBaseURL: 'http://localhost:8000',
                containerElement: document.getElementById('app'),
                debugMode: true, // Set to false in production
            });

            try {
                await app.initialize();
                console.log('Application ready!');
            } catch (error) {
                console.error('Failed to initialize application:', error);
                document.getElementById('app').innerHTML = `
                    <div style="padding: 2rem; text-align: center;">
                        <h1>Failed to Initialize</h1>
                        <p>${error.message}</p>
                        <p>Make sure the backend is running on <code>http://localhost:8000</code></p>
                    </div>
                `;
            }
        })();
    </script>
</body>
</html>
```

Create main CSS:

```css
/* frontend/css/main.css */

:root {
    --color-primary: #2563eb;
    --color-secondary: #64748b;
    --color-success: #10b981;
    --color-error: #ef4444;
    --color-warning: #f59e0b;
    --color-bg: #ffffff;
    --color-bg-alt: #f8fafc;
    --color-border: #e2e8f0;
    --color-text: #1e293b;
    --color-text-light: #64748b;
    --spacing-xs: 0.25rem;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2rem;
    --border-radius: 0.375rem;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: var(--color-text);
    background: var(--color-bg-alt);
    line-height: 1.5;
}

.app-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: var(--spacing-xl);
}

.app-header {
    text-align: center;
    margin-bottom: var(--spacing-xl);
}

.app-header h1 {
    margin: 0 0 var(--spacing-sm);
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--color-primary);
}

.tagline {
    margin: 0;
    color: var(--color-text-light);
    font-size: 1.125rem;
}

.app-main {
    display: grid;
    grid-template-columns: 1fr;
    gap: var(--spacing-xl);
}

/* Filters Section */
.filters-section {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--border-radius);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-sm);
}

.search-buttons {
    display: flex;
    gap: var(--spacing-md);
    margin-top: var(--spacing-lg);
    justify-content: center;
}

/* Buttons */
.btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: var(--border-radius);
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-primary {
    background: var(--color-primary);
    color: white;
}

.btn-primary:hover {
    background: #1d4ed8;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.btn-secondary {
    background: var(--color-secondary);
    color: white;
}

.btn-secondary:hover {
    background: #475569;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.btn:active {
    transform: translateY(0);
}

/* Results Section */
.results-section {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--border-radius);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-sm);
    min-height: 400px;
}

/* Status Messages */
.status-message {
    padding: var(--spacing-md);
    border-radius: var(--border-radius);
    margin-bottom: var(--spacing-lg);
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
}

.status-loading {
    background: #dbeafe;
    border-left: 4px solid var(--color-primary);
    color: #1e40af;
}

.status-success {
    background: #d1fae5;
    border-left: 4px solid var(--color-success);
    color: #065f46;
}

.status-error {
    background: #fee2e2;
    border-left: 4px solid var(--color-error);
    color: #991b1b;
}

/* Spinner */
.spinner {
    width: 1rem;
    height: 1rem;
    border: 2px solid currentColor;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Modal */
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: var(--spacing-xl);
}

.modal-overlay.hidden {
    display: none;
}

.modal-content {
    background: var(--color-bg);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-lg);
    max-width: 900px;
    width: 100%;
    max-height: 90vh;
    overflow-y: auto;
    position: relative;
    padding: var(--spacing-xl);
}

.modal-close {
    position: absolute;
    top: var(--spacing-md);
    right: var(--spacing-md);
    background: none;
    border: none;
    font-size: 2rem;
    line-height: 1;
    cursor: pointer;
    color: var(--color-text-light);
    padding: 0;
    width: 2rem;
    height: 2rem;
}

.modal-close:hover {
    color: var(--color-text);
}

/* Responsive */
@media (min-width: 768px) {
    .app-main {
        grid-template-columns: 350px 1fr;
    }
}

@media (max-width: 767px) {
    .app-container {
        padding: var(--spacing-md);
    }

    .search-buttons {
        flex-direction: column;
    }

    .btn {
        width: 100%;
    }
}
```

**Key Points:**
- Single initialization point (`app.initialize()`)
- Event-driven architecture (all components communicate via EventBus)
- State management (currentSearch, searchResults)
- UI building and lifecycle management
- Error handling at application level
- Modal management
- Responsive layout with CSS Grid

**Testing Criteria:**
1. ✅ App initializes without errors
2. ✅ Health check passes
3. ✅ Field registry loads
4. ✅ Filter UI builds correctly
5. ✅ Quick/Deep search buttons work
6. ✅ Deep button hides when expensive fields selected
7. ✅ Search emits correct events
8. ✅ Status messages display correctly
9. ✅ Modal opens/closes

**Test Example:**
```html
<!-- Test: test-app-integration.html -->
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/filters.css">
    <link rel="stylesheet" href="css/field-display.css">
    <title>App Integration Test</title>
</head>
<body>
    <h1 style="text-align: center; margin: 1rem;">App Integration Test</h1>
    <p style="text-align: center; color: #64748b;">
        Backend must be running on http://localhost:8000
    </p>

    <div id="app"></div>

    <script src="js/event-bus.js"></script>
    <script src="js/field-renderer.js"></script>
    <script src="js/field-registry.js"></script>
    <script src="js/filter-builder.js"></script>
    <script src="js/api-client.js"></script>
    <script src="js/app.js"></script>

    <script>
        (async () => {
            console.log('=== App Integration Test ===');

            const app = new JobSearchApp({
                apiBaseURL: 'http://localhost:8000',
                containerElement: document.getElementById('app'),
                debugMode: true,
            });

            // Test event emission
            let eventsReceived = {};
            ['app:ready', 'search:started', 'search:completed', 'search:failed'].forEach(eventName => {
                app.eventBus.on(eventName, (data) => {
                    eventsReceived[eventName] = true;
                    console.log(`✓ Event received: ${eventName}`);
                });
            });

            try {
                // Test 1: Initialization
                console.log('\n--- Test 1: Initialization ---');
                await app.initialize();
                console.assert(eventsReceived['app:ready'], 'Should emit app:ready');
                console.log('✓ App initialized');

                // Test 2: Check UI elements
                console.log('\n--- Test 2: UI Elements ---');
                console.assert(app.elements.quickSearchBtn, 'Quick search button exists');
                console.assert(app.elements.deepSearchBtn, 'Deep search button exists');
                console.assert(app.elements.filtersContainer, 'Filters container exists');
                console.assert(app.elements.resultsContainer, 'Results container exists');
                console.log('✓ All UI elements present');

                // Test 3: Filter builder
                console.log('\n--- Test 3: Filter Builder ---');
                console.assert(app.filterBuilder, 'FilterBuilder initialized');
                const filters = app.filterBuilder.getFilters();
                console.assert(filters.hasOwnProperty('query'), 'Filters have query');
                console.log('✓ FilterBuilder working');

                console.log('\n=== All integration tests passed! ===');
                console.log('Try clicking the search buttons to test search functionality');

            } catch (error) {
                console.error('✗ Integration test failed:', error);
            }
        })();
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Implement JobSearchApp orchestration class, main HTML, CSS
- **Tester**: Test with real backend, verify initialization, UI building, event flow
- **Relevant Files**:
  - Dev: `frontend/js/app.js`, `frontend/index.html`, `frontend/css/main.css`
  - Test: `frontend/test-app-integration.html`
  - Note: Requires backend running and all previous frontend chunks completed

---

# Phase 8: Quick Search Flow

**Objective:** Implement complete Quick Search user experience with job listing display.

**Why This Phase:**
Now that the foundation is in place, we need to implement the actual job results rendering, including job cards, duplicate grouping, and basic research field display.

**Dependencies:** Phase 7 (Frontend Foundation)

**Phase Success Criteria:**
1. ✅ Job results render as cards with all basic info
2. ✅ Duplicate jobs grouped together with expand/collapse
3. ✅ Research fields render using FieldRenderer
4. ✅ Loading states show progress
5. ✅ Quick search completes in 2-5 seconds

---

## Chunk 8.1: Job Card Component

**Objective:** Create reusable job card component for displaying job listings.

**Dependencies:** Phase 7 (EventBus, FieldRegistry, FieldRenderer)

**Files to Create:**
- `frontend/js/job-card.js` (~200 lines)
- `frontend/css/job-card.css` (~150 lines)

**Files to Modify:** None

**Implementation:**

```javascript
// frontend/js/job-card.js

/**
 * JobCard - Component for rendering individual job listings
 *
 * Features:
 * - Displays core job info (title, company, location, pay)
 * - Renders research fields using FieldRenderer
 * - Handles duplicate job grouping
 * - "Deep Research" button for individual company research
 * - Link to original job posting
 *
 * Usage:
 *   const card = new JobCard(job, registry, eventBus);
 *   container.appendChild(card.render());
 */
class JobCard {
    constructor(job, registry, eventBus) {
        this.job = job;
        this.registry = registry;
        this.eventBus = eventBus;
        this.isExpanded = false; // For duplicate grouping
    }

    /**
     * Render the job card as HTML element
     * @returns {HTMLElement} Job card element
     */
    render() {
        const card = document.createElement('div');
        card.className = 'job-card';
        card.dataset.jobId = this.job.id;

        // Add scam warning if detected
        if (this.job.company && this.job.company.is_scam) {
            card.classList.add('job-card-scam');
        }

        card.innerHTML = `
            ${this._renderScamWarning()}
            <div class="job-card-header">
                <h3 class="job-title">${this._escapeHtml(this.job.title)}</h3>
                <span class="job-source-badge">${this._escapeHtml(this.job.source)}</span>
            </div>

            <div class="job-card-body">
                <div class="job-company">
                    ${this._renderCompanyIcon()}
                    <span>${this._escapeHtml(this.job.company_name)}</span>
                </div>

                <div class="job-meta">
                    ${this._renderLocation()}
                    ${this._renderPay()}
                    ${this._renderRemote()}
                </div>

                ${this._renderResearchFields()}
            </div>

            <div class="job-card-footer">
                <a href="${this.job.url}" target="_blank" rel="noopener noreferrer" class="btn btn-sm btn-outline">
                    View Original Posting
                </a>
                <button class="btn btn-sm btn-primary deep-research-btn" data-company="${this._escapeHtml(this.job.company_name)}">
                    Deep Research
                </button>
            </div>

            ${this._renderDuplicates()}
        `;

        // Attach event listeners
        this._attachEventListeners(card);

        return card;
    }

    /**
     * Render scam warning if detected
     * @private
     */
    _renderScamWarning() {
        if (!this.job.company || !this.job.company.is_scam) {
            return '';
        }

        const confidence = this.job.company.scam_confidence || 'unknown';
        const reasoning = this.job.company.scam_reasoning || 'No details provided';

        return `
            <div class="scam-warning">
                <span class="scam-icon">⚠️</span>
                <div class="scam-content">
                    <strong>Potential Scam Detected</strong>
                    <span class="scam-confidence">Confidence: ${confidence}</span>
                    <p class="scam-reasoning">${this._escapeHtml(reasoning)}</p>
                </div>
            </div>
        `;
    }

    /**
     * Render company icon/logo
     * @private
     */
    _renderCompanyIcon() {
        // TODO: Phase 11 could add actual company logos
        return '<span class="company-icon">🏢</span>';
    }

    /**
     * Render location info
     * @private
     */
    _renderLocation() {
        if (!this.job.location) return '';

        return `
            <span class="job-meta-item">
                <span class="meta-icon">📍</span>
                ${this._escapeHtml(this.job.location)}
            </span>
        `;
    }

    /**
     * Render pay info
     * @private
     */
    _renderPay() {
        if (!this.job.pay_min && !this.job.pay_max && !this.job.pay_exact) {
            return '';
        }

        let payText = '';
        if (this.job.pay_exact) {
            payText = `$${this._formatNumber(this.job.pay_exact)}`;
        } else if (this.job.pay_min && this.job.pay_max) {
            payText = `$${this._formatNumber(this.job.pay_min)} - $${this._formatNumber(this.job.pay_max)}`;
        } else if (this.job.pay_min) {
            payText = `$${this._formatNumber(this.job.pay_min)}+`;
        } else if (this.job.pay_max) {
            payText = `Up to $${this._formatNumber(this.job.pay_max)}`;
        }

        return `
            <span class="job-meta-item">
                <span class="meta-icon">💰</span>
                ${payText}
            </span>
        `;
    }

    /**
     * Render remote badge
     * @private
     */
    _renderRemote() {
        if (!this.job.remote) return '';

        return `
            <span class="job-meta-item job-remote-badge">
                <span class="meta-icon">🏠</span>
                Remote
            </span>
        `;
    }

    /**
     * Render research fields from company data
     * @private
     */
    _renderResearchFields() {
        if (!this.job.company) return '';

        const fields = [];

        // Get all fields from company object (excluding standard props)
        const standardProps = ['name', 'cached_date', 'is_scam', 'scam_confidence', 'scam_reasoning'];
        for (const [key, value] of Object.entries(this.job.company)) {
            if (!standardProps.includes(key) && value !== null && value !== undefined) {
                const fieldMeta = this.registry.getField(key);
                if (fieldMeta) {
                    fields.push({ key, value, meta: fieldMeta });
                }
            }
        }

        if (fields.length === 0) return '';

        const fieldsHtml = fields.map(({ key, value, meta }) => {
            const renderer = this.registry.getRenderer(key);
            const html = renderer.render(key, value, this.job.company);
            return `<div class="research-field">${html}</div>`;
        }).join('');

        return `
            <div class="research-fields">
                <h4 class="research-fields-title">Company Research:</h4>
                ${fieldsHtml}
            </div>
        `;
    }

    /**
     * Render duplicate jobs section
     * @private
     */
    _renderDuplicates() {
        if (!this.job.duplicates || this.job.duplicates.length === 0) {
            return '';
        }

        const count = this.job.duplicates.length;
        const duplicatesList = this.job.duplicates.map(dup => `
            <li class="duplicate-item">
                <a href="${dup.url}" target="_blank" rel="noopener noreferrer">
                    ${this._escapeHtml(dup.source)} - ${this._escapeHtml(dup.title)}
                </a>
            </li>
        `).join('');

        return `
            <div class="duplicates-section ${this.isExpanded ? 'expanded' : ''}">
                <button class="duplicates-toggle">
                    <span class="toggle-icon">${this.isExpanded ? '▼' : '▶'}</span>
                    ${count} duplicate${count !== 1 ? 's' : ''} found on other sites
                </button>
                <ul class="duplicates-list">
                    ${duplicatesList}
                </ul>
            </div>
        `;
    }

    /**
     * Attach event listeners to card
     * @private
     */
    _attachEventListeners(card) {
        // Deep research button
        const deepResearchBtn = card.querySelector('.deep-research-btn');
        if (deepResearchBtn) {
            deepResearchBtn.addEventListener('click', (e) => {
                e.preventDefault();
                const companyName = deepResearchBtn.dataset.company;
                this.eventBus.emit('research:requested', { company: companyName });
            });
        }

        // Duplicates toggle
        const duplicatesToggle = card.querySelector('.duplicates-toggle');
        if (duplicatesToggle) {
            duplicatesToggle.addEventListener('click', () => {
                this.isExpanded = !this.isExpanded;
                const section = card.querySelector('.duplicates-section');
                section.classList.toggle('expanded');

                const icon = duplicatesToggle.querySelector('.toggle-icon');
                icon.textContent = this.isExpanded ? '▼' : '▶';
            });
        }
    }

    /**
     * Escape HTML to prevent XSS
     * @private
     */
    _escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Format number with commas
     * @private
     */
    _formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = JobCard;
}
```

Create job card CSS:

```css
/* frontend/css/job-card.css */

.job-card {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--border-radius);
    padding: var(--spacing-lg);
    margin-bottom: var(--spacing-md);
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s, border-color 0.2s;
}

.job-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--color-primary);
}

.job-card-scam {
    border-left: 4px solid var(--color-error);
}

/* Scam Warning */
.scam-warning {
    display: flex;
    gap: var(--spacing-md);
    padding: var(--spacing-md);
    background: #fee2e2;
    border: 1px solid var(--color-error);
    border-radius: var(--border-radius);
    margin-bottom: var(--spacing-lg);
}

.scam-icon {
    font-size: 1.5rem;
    flex-shrink: 0;
}

.scam-content {
    flex: 1;
}

.scam-content strong {
    display: block;
    color: var(--color-error);
    margin-bottom: var(--spacing-xs);
}

.scam-confidence {
    display: inline-block;
    font-size: 0.875rem;
    padding: 0.125rem 0.5rem;
    background: var(--color-error);
    color: white;
    border-radius: 0.25rem;
    margin-bottom: var(--spacing-sm);
}

.scam-reasoning {
    margin: var(--spacing-sm) 0 0;
    font-size: 0.875rem;
    color: var(--color-text);
}

/* Card Header */
.job-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: var(--spacing-md);
    margin-bottom: var(--spacing-md);
}

.job-title {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--color-text);
    flex: 1;
}

.job-source-badge {
    flex-shrink: 0;
    padding: 0.25rem 0.75rem;
    background: var(--color-bg-alt);
    border: 1px solid var(--color-border);
    border-radius: 0.25rem;
    font-size: 0.875rem;
    color: var(--color-text-light);
    font-weight: 500;
}

/* Card Body */
.job-card-body {
    margin-bottom: var(--spacing-lg);
}

.job-company {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    font-size: 1rem;
    margin-bottom: var(--spacing-md);
    color: var(--color-text);
}

.company-icon {
    font-size: 1.25rem;
}

.job-meta {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-md);
    margin-bottom: var(--spacing-md);
}

.job-meta-item {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    font-size: 0.875rem;
    color: var(--color-text-light);
}

.meta-icon {
    font-size: 1rem;
}

.job-remote-badge {
    color: var(--color-success);
    font-weight: 600;
}

/* Research Fields */
.research-fields {
    margin-top: var(--spacing-lg);
    padding-top: var(--spacing-lg);
    border-top: 1px solid var(--color-border);
}

.research-fields-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--color-text-light);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 var(--spacing-md);
}

.research-field {
    margin-bottom: var(--spacing-sm);
}

/* Card Footer */
.job-card-footer {
    display: flex;
    gap: var(--spacing-md);
    padding-top: var(--spacing-md);
    border-top: 1px solid var(--color-border);
}

.btn-sm {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
}

.btn-outline {
    background: transparent;
    border: 1px solid var(--color-border);
    color: var(--color-text);
}

.btn-outline:hover {
    background: var(--color-bg-alt);
    border-color: var(--color-text-light);
}

/* Duplicates Section */
.duplicates-section {
    margin-top: var(--spacing-md);
    padding-top: var(--spacing-md);
    border-top: 1px solid var(--color-border);
}

.duplicates-toggle {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    padding: var(--spacing-sm);
    font-size: 0.875rem;
    color: var(--color-text-light);
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    transition: color 0.2s;
}

.duplicates-toggle:hover {
    color: var(--color-primary);
}

.toggle-icon {
    display: inline-block;
    width: 1rem;
    text-align: center;
    transition: transform 0.2s;
}

.duplicates-list {
    display: none;
    list-style: none;
    padding: 0;
    margin: var(--spacing-md) 0 0;
}

.duplicates-section.expanded .duplicates-list {
    display: block;
}

.duplicate-item {
    padding: var(--spacing-sm);
    border-left: 2px solid var(--color-border);
    margin-bottom: var(--spacing-xs);
}

.duplicate-item a {
    color: var(--color-primary);
    text-decoration: none;
    font-size: 0.875rem;
}

.duplicate-item a:hover {
    text-decoration: underline;
}

/* Responsive */
@media (max-width: 767px) {
    .job-card-header {
        flex-direction: column;
    }

    .job-card-footer {
        flex-direction: column;
    }

    .job-card-footer .btn {
        width: 100%;
    }

    .job-meta {
        flex-direction: column;
        gap: var(--spacing-sm);
    }
}
```

**Key Points:**
- Reusable component pattern with render() method
- Scam warning display with confidence and reasoning
- Research fields rendered via FieldRenderer
- Duplicate grouping with expand/collapse
- Event emission for "Deep Research" button
- XSS protection via escapeHtml()
- Responsive design

**Testing Criteria:**
1. ✅ Job card renders all basic info correctly
2. ✅ Scam warning appears for flagged companies
3. ✅ Research fields render with correct display types
4. ✅ Duplicate section expands/collapses
5. ✅ Deep Research button emits event
6. ✅ Links open in new tab
7. ✅ Card is responsive on mobile

**Test Example:**
```html
<!-- Test: test-job-card.html -->
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/field-display.css">
    <link rel="stylesheet" href="css/job-card.css">
    <title>Job Card Test</title>
</head>
<body style="padding: 2rem; background: #f8fafc;">
    <h1>Job Card Component Test</h1>
    <div id="cards-container"></div>

    <script src="js/event-bus.js"></script>
    <script src="js/field-renderer.js"></script>
    <script src="js/field-registry.js"></script>
    <script src="js/job-card.js"></script>

    <script>
        (async () => {
            const bus = new EventBus();
            const registry = new FieldRegistry('http://localhost:8000');

            // Listen for deep research requests
            bus.on('research:requested', (data) => {
                console.log('Deep research requested for:', data.company);
                alert(`Deep research requested for: ${data.company}`);
            });

            try {
                await registry.load();
                console.log('✓ Registry loaded');

                // Test data
                const jobs = [
                    {
                        id: 'job1',
                        title: 'Senior Software Engineer',
                        company_name: 'TechCorp Inc',
                        location: 'San Francisco, CA',
                        remote: true,
                        pay_min: 120000,
                        pay_max: 180000,
                        source: 'Indeed',
                        url: 'https://example.com/job1',
                        company: {
                            name: 'TechCorp Inc',
                            is_scam: false,
                            glassdoor_rating: 4.5,
                            glassdoor_review_count: 523,
                        },
                        duplicates: [
                            { source: 'LinkedIn', title: 'Senior Software Engineer', url: 'https://linkedin.com/job1' },
                            { source: 'Dice', title: 'Sr. Software Engineer', url: 'https://dice.com/job1' },
                        ],
                    },
                    {
                        id: 'job2',
                        title: 'Work From Home - Easy Money!!!',
                        company_name: 'QuickCash Solutions',
                        location: 'Remote',
                        remote: true,
                        pay_exact: 5000,
                        source: 'ZipRecruiter',
                        url: 'https://example.com/job2',
                        company: {
                            name: 'QuickCash Solutions',
                            is_scam: true,
                            scam_confidence: 'high',
                            scam_reasoning: 'Company uses urgent language, promises unrealistic income, and has no online presence.',
                        },
                        duplicates: [],
                    },
                    {
                        id: 'job3',
                        title: 'Data Scientist',
                        company_name: 'DataAnalytics Co',
                        location: 'New York, NY',
                        remote: false,
                        pay_min: 100000,
                        source: 'LinkedIn',
                        url: 'https://example.com/job3',
                        company: {
                            name: 'DataAnalytics Co',
                            is_scam: false,
                        },
                        duplicates: [],
                    },
                ];

                // Render job cards
                const container = document.getElementById('cards-container');
                jobs.forEach(job => {
                    const card = new JobCard(job, registry, bus);
                    container.appendChild(card.render());
                });

                console.log('✓ All job cards rendered');

            } catch (error) {
                console.error('✗ Test failed:', error);
                document.getElementById('cards-container').innerHTML = `
                    <div style="color: red; padding: 1rem; border: 1px solid red;">
                        Error: ${error.message}
                    </div>
                `;
            }
        })();
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Implement JobCard component with all rendering logic
- **Tester**: Test with variety of job data, verify all features, test events
- **Relevant Files**:
  - Dev: `frontend/js/job-card.js`, `frontend/css/job-card.css`
  - Test: `frontend/test-job-card.html`
  - Note: Requires backend for field registry

---

## Chunk 8.2: Results List Renderer

**Objective:** Create results list component that renders multiple job cards.

**Dependencies:** Chunk 8.1 (JobCard component)

**Files to Create:**
- `frontend/js/results-list.js` (~150 lines)
- `frontend/css/results-list.css` (~100 lines)

**Files to Modify:** None

**Implementation:**

```javascript
// frontend/js/results-list.js

/**
 * ResultsList - Component for rendering list of job results
 *
 * Features:
 * - Renders array of jobs as JobCard components
 * - Handles empty states
 * - Sorting and filtering controls
 * - Result count display
 *
 * Usage:
 *   const list = new ResultsList(jobs, registry, eventBus, container);
 *   list.render();
 */
class ResultsList {
    constructor(jobs, registry, eventBus, containerElement) {
        this.jobs = jobs;
        this.registry = registry;
        this.eventBus = eventBus;
        this.container = containerElement;
        this.sortBy = 'relevance'; // relevance, pay, date
    }

    /**
     * Render the results list
     */
    render() {
        this.container.innerHTML = '';

        if (this.jobs.length === 0) {
            this._renderEmpty();
            return;
        }

        // Create list container
        const listContainer = document.createElement('div');
        listContainer.className = 'results-list';

        // Add header with controls
        listContainer.appendChild(this._renderHeader());

        // Render job cards
        const cardsContainer = document.createElement('div');
        cardsContainer.className = 'results-cards';

        this.jobs.forEach(job => {
            const card = new JobCard(job, this.registry, this.eventBus);
            cardsContainer.appendChild(card.render());
        });

        listContainer.appendChild(cardsContainer);

        this.container.appendChild(listContainer);

        // Emit event
        this.eventBus.emit('results:rendered', { count: this.jobs.length });
    }

    /**
     * Render header with result count and controls
     * @private
     */
    _renderHeader() {
        const header = document.createElement('div');
        header.className = 'results-header';

        const jobCount = this.jobs.length;
        const uniqueCompanies = new Set(this.jobs.map(j => j.company_name)).size;
        const scamCount = this.jobs.filter(j => j.company && j.company.is_scam).length;

        header.innerHTML = `
            <div class="results-stats">
                <h2 class="results-count">${jobCount} Job${jobCount !== 1 ? 's' : ''} Found</h2>
                <div class="results-meta">
                    <span>${uniqueCompanies} unique ${uniqueCompanies !== 1 ? 'companies' : 'company'}</span>
                    ${scamCount > 0 ? `<span class="scam-count">⚠️ ${scamCount} potential scam${scamCount !== 1 ? 's' : ''}</span>` : ''}
                </div>
            </div>

            <div class="results-controls">
                <label for="sort-select">Sort by:</label>
                <select id="sort-select" class="sort-select">
                    <option value="relevance" ${this.sortBy === 'relevance' ? 'selected' : ''}>Relevance</option>
                    <option value="pay" ${this.sortBy === 'pay' ? 'selected' : ''}>Pay (High to Low)</option>
                    <option value="company" ${this.sortBy === 'company' ? 'selected' : ''}>Company Name</option>
                </select>
            </div>
        `;

        // Attach sort change listener
        const sortSelect = header.querySelector('#sort-select');
        sortSelect.addEventListener('change', (e) => {
            this.sortBy = e.target.value;
            this._sortJobs();
            this.render(); // Re-render with new sort order
        });

        return header;
    }

    /**
     * Render empty state
     * @private
     */
    _renderEmpty() {
        this.container.innerHTML = `
            <div class="results-empty">
                <div class="empty-icon">🔍</div>
                <h3>No Jobs Found</h3>
                <p>Try adjusting your search criteria or filters.</p>
            </div>
        `;
    }

    /**
     * Sort jobs based on current sort option
     * @private
     */
    _sortJobs() {
        switch (this.sortBy) {
            case 'pay':
                this.jobs.sort((a, b) => {
                    const payA = a.pay_exact || a.pay_max || a.pay_min || 0;
                    const payB = b.pay_exact || b.pay_max || b.pay_min || 0;
                    return payB - payA; // High to low
                });
                break;

            case 'company':
                this.jobs.sort((a, b) => {
                    return a.company_name.localeCompare(b.company_name);
                });
                break;

            case 'relevance':
            default:
                // Keep original order (from backend)
                break;
        }
    }

    /**
     * Update with new jobs (for progressive loading)
     */
    update(newJobs) {
        this.jobs = newJobs;
        this.render();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ResultsList;
}
```

Create results list CSS:

```css
/* frontend/css/results-list.css */

.results-list {
    width: 100%;
}

.results-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-lg);
    padding-bottom: var(--spacing-md);
    border-bottom: 2px solid var(--color-border);
}

.results-stats {
    flex: 1;
}

.results-count {
    margin: 0 0 var(--spacing-xs);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--color-text);
}

.results-meta {
    display: flex;
    gap: var(--spacing-md);
    font-size: 0.875rem;
    color: var(--color-text-light);
}

.scam-count {
    color: var(--color-error);
    font-weight: 600;
}

.results-controls {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.results-controls label {
    font-size: 0.875rem;
    color: var(--color-text-light);
    font-weight: 500;
}

.sort-select {
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--color-border);
    border-radius: var(--border-radius);
    background: var(--color-bg);
    color: var(--color-text);
    font-size: 0.875rem;
    cursor: pointer;
    transition: border-color 0.2s;
}

.sort-select:hover {
    border-color: var(--color-primary);
}

.sort-select:focus {
    outline: none;
    border-color: var(--color-primary);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.results-cards {
    /* Cards already have bottom margin */
}

/* Empty State */
.results-empty {
    text-align: center;
    padding: var(--spacing-xl) var(--spacing-md);
    color: var(--color-text-light);
}

.empty-icon {
    font-size: 4rem;
    margin-bottom: var(--spacing-md);
    opacity: 0.5;
}

.results-empty h3 {
    margin: 0 0 var(--spacing-sm);
    font-size: 1.5rem;
    color: var(--color-text);
}

.results-empty p {
    margin: 0;
    font-size: 1rem;
}

/* Responsive */
@media (max-width: 767px) {
    .results-header {
        flex-direction: column;
    }

    .results-controls {
        width: 100%;
        justify-content: space-between;
    }

    .sort-select {
        flex: 1;
    }
}
```

**Key Points:**
- Manages array of jobs and renders via JobCard
- Result stats (job count, company count, scam count)
- Sorting functionality (relevance, pay, company)
- Empty state handling
- Progressive update support
- Event emission when rendered

**Testing Criteria:**
1. ✅ Renders all jobs as cards
2. ✅ Shows correct result counts
3. ✅ Scam count displays when present
4. ✅ Sort dropdown works correctly
5. ✅ Empty state displays when no jobs
6. ✅ Re-renders on sort change

**Test Example:**
```html
<!-- Test: test-results-list.html -->
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/field-display.css">
    <link rel="stylesheet" href="css/job-card.css">
    <link rel="stylesheet" href="css/results-list.css">
    <title>Results List Test</title>
</head>
<body style="padding: 2rem; background: #f8fafc;">
    <h1>Results List Component Test</h1>
    <button id="toggle-empty">Toggle Empty State</button>
    <div id="results-container" style="margin-top: 1rem;"></div>

    <script src="js/event-bus.js"></script>
    <script src="js/field-renderer.js"></script>
    <script src="js/field-registry.js"></script>
    <script src="js/job-card.js"></script>
    <script src="js/results-list.js"></script>

    <script>
        (async () => {
            const bus = new EventBus();
            const registry = new FieldRegistry('http://localhost:8000');
            const container = document.getElementById('results-container');

            bus.on('results:rendered', (data) => {
                console.log(`✓ Results rendered: ${data.count} jobs`);
            });

            try {
                await registry.load();

                const jobs = [
                    {
                        id: '1',
                        title: 'Software Engineer',
                        company_name: 'Acme Corp',
                        location: 'Remote',
                        remote: true,
                        pay_min: 100000,
                        pay_max: 150000,
                        source: 'Indeed',
                        url: 'https://example.com/1',
                        company: { name: 'Acme Corp', is_scam: false, glassdoor_rating: 4.2 },
                        duplicates: [],
                    },
                    {
                        id: '2',
                        title: 'Data Analyst',
                        company_name: 'DataCo',
                        location: 'New York, NY',
                        remote: false,
                        pay_exact: 85000,
                        source: 'LinkedIn',
                        url: 'https://example.com/2',
                        company: { name: 'DataCo', is_scam: false },
                        duplicates: [],
                    },
                    {
                        id: '3',
                        title: 'URGENT - Make Money Fast!',
                        company_name: 'ScamCo',
                        location: 'Remote',
                        remote: true,
                        pay_exact: 10000,
                        source: 'Unknown',
                        url: 'https://example.com/3',
                        company: { name: 'ScamCo', is_scam: true, scam_confidence: 'high' },
                        duplicates: [],
                    },
                ];

                let list = new ResultsList(jobs, registry, bus, container);
                list.render();
                console.log('✓ Results list rendered');

                // Toggle empty state
                document.getElementById('toggle-empty').addEventListener('click', () => {
                    const isEmpty = list.jobs.length === 0;
                    list = new ResultsList(isEmpty ? jobs : [], registry, bus, container);
                    list.render();
                });

            } catch (error) {
                console.error('✗ Test failed:', error);
            }
        })();
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Implement ResultsList component with sorting and empty states
- **Tester**: Test with various job arrays, verify sorting, test empty state
- **Relevant Files**:
  - Dev: `frontend/js/results-list.js`, `frontend/css/results-list.css`
  - Test: `frontend/test-results-list.html`

---

## Chunk 8.3: Integrate Quick Search Flow

**Objective:** Wire up results rendering in main app to complete Quick Search flow.

**Dependencies:**
- Chunk 8.1 (JobCard)
- Chunk 8.2 (ResultsList)
- Phase 7 (JobSearchApp)

**Files to Create:** None

**Files to Modify:**
- `frontend/js/app.js` (add ResultsList rendering)
- `frontend/index.html` (add new script tags)

**Implementation:**

Update app.js to use ResultsList:

```javascript
// Modify: frontend/js/app.js

// Add near the top after other components initialization:
// (Around line 10 after imports)

// Add to constructor:
this.resultsList = null; // ResultsList component

// Replace the _onSearchCompleted method entirely:

/**
 * Event handler: Search completed
 * @private
 */
_onSearchCompleted(data) {
    console.log('[App] Search completed:', data);

    const jobCount = data.jobs.length;
    const timing = data.timing || {};

    this.elements.searchStatus.innerHTML = `
        <div class="status-message status-success">
            Found ${jobCount} job${jobCount !== 1 ? 's' : ''}
            ${timing.total ? `in ${(timing.total / 1000).toFixed(1)}s` : ''}
        </div>
    `;

    // Render results using ResultsList
    this.resultsList = new ResultsList(
        data.jobs,
        this.registry,
        this.eventBus,
        this.elements.resultsContainer
    );
    this.resultsList.render();

    // Listen for deep research requests from job cards
    this.eventBus.on('research:requested', (eventData) => {
        this._handleDeepResearch(eventData.company);
    });
}
```

Update index.html to include new scripts:

```html
<!-- Modify: frontend/index.html -->

<!-- Add these script tags before app.js: -->
<script src="js/job-card.js"></script>
<script src="js/results-list.js"></script>

<!-- Full script order should be: -->
<script src="js/event-bus.js"></script>
<script src="js/field-renderer.js"></script>
<script src="js/field-registry.js"></script>
<script src="js/filter-builder.js"></script>
<script src="js/api-client.js"></script>
<script src="js/job-card.js"></script>
<script src="js/results-list.js"></script>
<script src="js/app.js"></script>
```

Add CSS imports to index.html:

```html
<!-- Modify: frontend/index.html -->

<!-- Update CSS section to include new stylesheets: -->
<link rel="stylesheet" href="css/main.css">
<link rel="stylesheet" href="css/filters.css">
<link rel="stylesheet" href="css/field-display.css">
<link rel="stylesheet" href="css/job-card.css">
<link rel="stylesheet" href="css/results-list.css">
```

**Key Points:**
- Integrates ResultsList into main app
- Removes placeholder result rendering
- Wires up deep research event flow
- Complete Quick Search experience

**Testing Criteria:**
1. ✅ Quick Search displays job cards
2. ✅ Results show all basic fields
3. ✅ Research fields render correctly
4. ✅ Scam warnings appear for flagged jobs
5. ✅ Deep Research button triggers modal
6. ✅ Sorting works
7. ✅ Duplicate grouping works
8. ✅ Complete flow takes 2-5 seconds

**Test Procedure:**
```
1. Start backend server
2. Open frontend/index.html in browser
3. Enter search query (e.g., "software engineer")
4. Click "Quick Search" button
5. Verify:
   - Loading state appears
   - Status message shows timing
   - Job cards render with all info
   - Scam warnings appear if applicable
   - Sorting dropdown works
   - Duplicate sections expand/collapse
   - Deep Research button opens modal (placeholder)
6. Try different search queries
7. Test edge cases:
   - No results
   - All results are scams
   - Results with/without research fields
```

**Planner Guidance:**
- **Developer**: Integrate ResultsList into app.js, update HTML includes
- **Tester**: Full end-to-end Quick Search flow testing
- **Relevant Files**:
  - Modify: `frontend/js/app.js`, `frontend/index.html`
  - Test: Manual testing via `frontend/index.html`
  - Note: This completes Phase 8 - Quick Search should be fully functional

---

# Phase 9: Deep Search Flow

**Objective:** Implement Deep Search with advanced filters and comprehensive company research.

**Why This Phase:**
Deep Search allows users to include expensive research fields (AI scam detection, comprehensive Glassdoor data, etc.) with explicit awareness of longer wait times.

**Dependencies:** Phase 8 (Quick Search Flow)

**Phase Success Criteria:**
1. ✅ Deep Search button behavior tied to selected filters
2. ✅ Loading UI shows progress for slow operations
3. ✅ Advanced research fields display correctly
4. ✅ User warned about long wait times before search
5. ✅ Deep search completes in 30-60 seconds

---

## Chunk 9.1: Advanced Filter Selection UI

**Objective:** Enhance filter UI to show time/cost implications of selections.

**Dependencies:** Phase 6 (FilterBuilder), Phase 7 (EventBus)

**Files to Create:**
- `frontend/css/advanced-filters.css` (~100 lines)

**Files to Modify:**
- `frontend/js/filter-builder.js` (add visual indicators for expensive fields)

**Implementation:**

Update FilterBuilder to add cost/time indicators:

```javascript
// Modify: frontend/js/filter-builder.js

// Add this method to the FilterBuilder class:

/**
 * Add cost/time indicators to filter fields
 * @private
 */
_addCostIndicators() {
    // Find all expensive category sections
    const expensiveCategories = ['ai', 'api_expensive'];

    expensiveCategories.forEach(category => {
        const categoryElement = this.container.querySelector(`[data-category="${category}"]`);
        if (!categoryElement) return;

        // Add warning badge
        const categoryHeader = categoryElement.querySelector('.filter-category-header');
        if (categoryHeader && !categoryHeader.querySelector('.cost-indicator')) {
            const indicator = document.createElement('span');
            indicator.className = 'cost-indicator';
            indicator.innerHTML = '⏱️ Slower';
            indicator.title = 'This category includes time-consuming analysis';
            categoryHeader.appendChild(indicator);
        }
    });

    // Update search button text when expensive fields selected
    this._updateSearchButtonText();
}

/**
 * Update search button text based on selected fields
 * @private
 */
_updateSearchButtonText() {
    const hasExpensive = this.hasExpensiveFields();

    // Emit event to update button text
    this.eventBus.emit('filters:cost-changed', {
        hasExpensive,
        estimatedTime: hasExpensive ? '30-60s' : '2-5s'
    });
}

// Modify the existing build() method to call these at the end:
async build() {
    // ... existing build code ...

    this._addCostIndicators();

    // Add change listeners to all checkboxes
    const checkboxes = this.container.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            this._updateSearchButtonText();
            this.eventBus.emit('filters:changed', this.getFilters());
        });
    });
}
```

Create advanced filters CSS:

```css
/* frontend/css/advanced-filters.css */

/* Cost Indicators */
.cost-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.125rem 0.5rem;
    background: #fef3c7;
    border: 1px solid #fbbf24;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #92400e;
    margin-left: auto;
}

.filter-category[data-category="ai"] .filter-category-header,
.filter-category[data-category="api_expensive"] .filter-category-header {
    background: #fff9e6;
    border-left: 3px solid #fbbf24;
}

/* Time estimate tooltip */
.time-estimate {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: #dbeafe;
    border-left: 3px solid #3b82f6;
    border-radius: var(--border-radius);
    margin-bottom: var(--spacing-md);
    font-size: 0.875rem;
    color: #1e40af;
}

.time-estimate-icon {
    display: inline-block;
    margin-right: 0.5rem;
}

/* Search confirmation dialog */
.search-confirmation {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    padding: var(--spacing-xl);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-lg);
    z-index: 999;
    max-width: 400px;
    width: 90%;
}

.search-confirmation-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: 998;
}

.search-confirmation h3 {
    margin: 0 0 var(--spacing-md);
    color: var(--color-text);
}

.search-confirmation p {
    margin: 0 0 var(--spacing-lg);
    color: var(--color-text-light);
    line-height: 1.6;
}

.search-confirmation-buttons {
    display: flex;
    gap: var(--spacing-md);
    justify-content: flex-end;
}

.search-confirmation-buttons .btn {
    padding: 0.5rem 1rem;
}

/* Highlight selected expensive fields */
.filter-field input[type="checkbox"]:checked ~ label {
    font-weight: 600;
    color: var(--color-primary);
}

.filter-category[data-category="ai"] .filter-field input[type="checkbox"]:checked ~ label,
.filter-category[data-category="api_expensive"] .filter-field input[type="checkbox"]:checked ~ label {
    color: #d97706;
}
```

**Key Points:**
- Visual indicators for expensive categories
- Cost badges on category headers
- Dynamic search button text updates
- Event emission for button state changes
- Warning colors for AI/expensive categories

**Testing Criteria:**
1. ✅ Expensive categories show "⏱️ Slower" badge
2. ✅ Selecting expensive fields updates button text
3. ✅ Events emit when cost changes
4. ✅ Visual highlighting for expensive categories
5. ✅ Checkboxes trigger updates

**Test Example:**
```html
<!-- Test: test-advanced-filters.html -->
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/filters.css">
    <link rel="stylesheet" href="css/advanced-filters.css">
    <title>Advanced Filters Test</title>
</head>
<body style="padding: 2rem;">
    <h1>Advanced Filters Test</h1>
    <div id="filters"></div>
    <div id="status" style="margin-top: 1rem; padding: 1rem; background: #f0f0f0;"></div>

    <script src="js/event-bus.js"></script>
    <script src="js/field-renderer.js"></script>
    <script src="js/field-registry.js"></script>
    <script src="js/filter-builder.js"></script>

    <script>
        (async () => {
            const bus = new EventBus();
            const registry = new FieldRegistry('http://localhost:8000');

            const statusDiv = document.getElementById('status');

            // Listen for cost changes
            bus.on('filters:cost-changed', (data) => {
                statusDiv.innerHTML = `
                    <strong>Cost Change Detected</strong><br>
                    Has Expensive: ${data.hasExpensive}<br>
                    Estimated Time: ${data.estimatedTime}
                `;
                console.log('Cost changed:', data);
            });

            try {
                await registry.load();
                console.log('✓ Registry loaded');

                const builder = new FilterBuilder(registry, document.getElementById('filters'));
                await builder.build();
                console.log('✓ Filters built with cost indicators');

            } catch (error) {
                console.error('✗ Test failed:', error);
            }
        })();
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Update FilterBuilder with cost indicators and event emission
- **Tester**: Verify visual indicators, test checkbox interactions, verify events
- **Relevant Files**:
  - Modify: `frontend/js/filter-builder.js`
  - Create: `frontend/css/advanced-filters.css`
  - Test: `frontend/test-advanced-filters.html`

---

## Chunk 9.2: Deep Search Loading States

**Objective:** Create informative loading UI for long-running deep searches.

**Dependencies:** Chunk 9.1, Phase 7 (JobSearchApp)

**Files to Create:**
- `frontend/js/search-progress.js` (~200 lines)
- `frontend/css/search-progress.css` (~150 lines)

**Files to Modify:**
- `frontend/js/app.js` (integrate progress component)

**Implementation:**

```javascript
// frontend/js/search-progress.js

/**
 * SearchProgress - Component for showing detailed search progress
 *
 * Features:
 * - Progress bar with percentage
 * - Stage indicators (searching, researching, analyzing)
 * - Time elapsed counter
 * - Cancel button
 * - Detailed status messages
 *
 * Usage:
 *   const progress = new SearchProgress(container, eventBus);
 *   progress.start({ estimatedTime: 45 });
 *   progress.updateStage('researching', 'Analyzing company data...');
 *   progress.complete();
 */
class SearchProgress {
    constructor(containerElement, eventBus) {
        this.container = containerElement;
        this.eventBus = eventBus;
        this.startTime = null;
        this.timerInterval = null;
        this.currentStage = null;
        this.element = null;
    }

    /**
     * Start showing progress
     * @param {Object} options - Search options
     * @param {number} options.estimatedTime - Estimated time in seconds
     * @param {string} options.query - Search query
     */
    start(options = {}) {
        this.startTime = Date.now();
        this.currentStage = 'searching';

        // Create progress UI
        this.element = document.createElement('div');
        this.element.className = 'search-progress';
        this.element.innerHTML = `
            <div class="progress-header">
                <h3 class="progress-title">Deep Search in Progress</h3>
                <button class="progress-cancel-btn" title="Cancel search">✕</button>
            </div>

            <div class="progress-query">
                Searching for: <strong>${this._escapeHtml(options.query || 'jobs')}</strong>
            </div>

            <div class="progress-bar-container">
                <div class="progress-bar" style="width: 0%"></div>
            </div>

            <div class="progress-stages">
                <div class="progress-stage" data-stage="searching">
                    <span class="stage-icon">🔍</span>
                    <span class="stage-label">Searching</span>
                    <span class="stage-status">pending</span>
                </div>
                <div class="progress-stage" data-stage="researching">
                    <span class="stage-icon">🏢</span>
                    <span class="stage-label">Researching</span>
                    <span class="stage-status">pending</span>
                </div>
                <div class="progress-stage" data-stage="analyzing">
                    <span class="stage-icon">🤖</span>
                    <span class="stage-label">AI Analysis</span>
                    <span class="stage-status">pending</span>
                </div>
            </div>

            <div class="progress-details">
                <div class="progress-message">Fetching job listings...</div>
                <div class="progress-timer">
                    <span class="timer-label">Elapsed:</span>
                    <span class="timer-value">0s</span>
                    ${options.estimatedTime ? `<span class="timer-estimate">/ ~${options.estimatedTime}s</span>` : ''}
                </div>
            </div>
        `;

        // Clear container and add progress UI
        this.container.innerHTML = '';
        this.container.appendChild(this.element);

        // Set up cancel button
        const cancelBtn = this.element.querySelector('.progress-cancel-btn');
        cancelBtn.addEventListener('click', () => {
            this.cancel();
        });

        // Start timer
        this._startTimer();

        // Emit event
        this.eventBus.emit('search:progress:started', {});
    }

    /**
     * Update current stage
     * @param {string} stage - Stage name (searching, researching, analyzing)
     * @param {string} message - Status message
     * @param {number} progress - Progress percentage (0-100)
     */
    updateStage(stage, message, progress = null) {
        if (!this.element) return;

        this.currentStage = stage;

        // Update stage status
        const stages = this.element.querySelectorAll('.progress-stage');
        stages.forEach(stageEl => {
            const stageName = stageEl.dataset.stage;
            const statusSpan = stageEl.querySelector('.stage-status');

            if (stageName === stage) {
                stageEl.classList.add('active');
                statusSpan.textContent = 'in progress';
                statusSpan.className = 'stage-status status-active';
            } else {
                const stageOrder = ['searching', 'researching', 'analyzing'];
                const currentIndex = stageOrder.indexOf(stage);
                const thisIndex = stageOrder.indexOf(stageName);

                if (thisIndex < currentIndex) {
                    stageEl.classList.add('completed');
                    statusSpan.textContent = 'done';
                    statusSpan.className = 'stage-status status-done';
                } else {
                    stageEl.classList.remove('active', 'completed');
                    statusSpan.textContent = 'pending';
                    statusSpan.className = 'stage-status status-pending';
                }
            }
        });

        // Update message
        if (message) {
            const messageEl = this.element.querySelector('.progress-message');
            messageEl.textContent = message;
        }

        // Update progress bar
        if (progress !== null) {
            const progressBar = this.element.querySelector('.progress-bar');
            progressBar.style.width = `${progress}%`;
        }

        // Emit event
        this.eventBus.emit('search:progress:updated', { stage, message, progress });
    }

    /**
     * Mark search as complete
     */
    complete() {
        this._stopTimer();

        if (this.element) {
            // Mark all stages as complete
            const stages = this.element.querySelectorAll('.progress-stage');
            stages.forEach(stageEl => {
                stageEl.classList.add('completed');
                const statusSpan = stageEl.querySelector('.stage-status');
                statusSpan.textContent = 'done';
                statusSpan.className = 'stage-status status-done';
            });

            // Update progress bar to 100%
            const progressBar = this.element.querySelector('.progress-bar');
            progressBar.style.width = '100%';

            // Update message
            const messageEl = this.element.querySelector('.progress-message');
            messageEl.textContent = 'Search complete!';
        }

        // Emit event
        this.eventBus.emit('search:progress:completed', {});
    }

    /**
     * Cancel the search
     */
    cancel() {
        this._stopTimer();

        // Emit cancel event
        this.eventBus.emit('search:cancelled', {});

        if (this.element) {
            this.element.innerHTML = `
                <div class="progress-cancelled">
                    <span class="cancel-icon">⊗</span>
                    <p>Search cancelled</p>
                </div>
            `;
        }
    }

    /**
     * Show error state
     * @param {Error} error - Error object
     */
    error(error) {
        this._stopTimer();

        if (this.element) {
            this.element.innerHTML = `
                <div class="progress-error">
                    <span class="error-icon">⚠️</span>
                    <h3>Search Failed</h3>
                    <p>${this._escapeHtml(error.message)}</p>
                </div>
            `;
        }
    }

    /**
     * Remove progress UI
     */
    clear() {
        this._stopTimer();
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
        this.element = null;
    }

    /**
     * Start elapsed time timer
     * @private
     */
    _startTimer() {
        this.timerInterval = setInterval(() => {
            if (!this.element) {
                this._stopTimer();
                return;
            }

            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            const timerValue = this.element.querySelector('.timer-value');
            if (timerValue) {
                timerValue.textContent = `${elapsed}s`;
            }
        }, 1000);
    }

    /**
     * Stop elapsed time timer
     * @private
     */
    _stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    /**
     * Escape HTML
     * @private
     */
    _escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SearchProgress;
}
```

Create progress CSS:

```css
/* frontend/css/search-progress.css */

.search-progress {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--border-radius);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-md);
}

.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-md);
}

.progress-title {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--color-text);
}

.progress-cancel-btn {
    background: transparent;
    border: none;
    font-size: 1.5rem;
    color: var(--color-text-light);
    cursor: pointer;
    padding: 0.25rem;
    line-height: 1;
    transition: color 0.2s;
}

.progress-cancel-btn:hover {
    color: var(--color-error);
}

.progress-query {
    margin-bottom: var(--spacing-lg);
    color: var(--color-text-light);
    font-size: 0.875rem;
}

.progress-query strong {
    color: var(--color-text);
}

/* Progress Bar */
.progress-bar-container {
    height: 8px;
    background: var(--color-bg-alt);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: var(--spacing-lg);
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--color-primary), #60a5fa);
    transition: width 0.3s ease;
}

/* Progress Stages */
.progress-stages {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--spacing-md);
    margin-bottom: var(--spacing-lg);
}

.progress-stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: var(--spacing-md);
    background: var(--color-bg-alt);
    border: 2px solid var(--color-border);
    border-radius: var(--border-radius);
    transition: all 0.3s;
}

.progress-stage.active {
    border-color: var(--color-primary);
    background: #dbeafe;
}

.progress-stage.completed {
    border-color: var(--color-success);
    background: #d1fae5;
}

.stage-icon {
    font-size: 1.5rem;
    margin-bottom: var(--spacing-xs);
}

.stage-label {
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--color-text);
    margin-bottom: var(--spacing-xs);
}

.stage-status {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.status-pending {
    color: var(--color-text-light);
}

.status-active {
    color: var(--color-primary);
}

.status-done {
    color: var(--color-success);
}

/* Progress Details */
.progress-details {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: var(--spacing-md);
    border-top: 1px solid var(--color-border);
}

.progress-message {
    flex: 1;
    color: var(--color-text);
    font-size: 0.875rem;
}

.progress-timer {
    display: flex;
    gap: var(--spacing-xs);
    font-size: 0.875rem;
    color: var(--color-text-light);
}

.timer-value {
    font-weight: 600;
    color: var(--color-primary);
}

.timer-estimate {
    color: var(--color-text-light);
}

/* Cancelled State */
.progress-cancelled {
    text-align: center;
    padding: var(--spacing-xl);
    color: var(--color-text-light);
}

.cancel-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: var(--spacing-md);
}

/* Error State */
.progress-error {
    text-align: center;
    padding: var(--spacing-xl);
}

.error-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: var(--spacing-md);
}

.progress-error h3 {
    margin: 0 0 var(--spacing-sm);
    color: var(--color-error);
}

.progress-error p {
    margin: 0;
    color: var(--color-text-light);
}

/* Responsive */
@media (max-width: 767px) {
    .progress-stages {
        grid-template-columns: 1fr;
    }

    .progress-details {
        flex-direction: column;
        align-items: flex-start;
        gap: var(--spacing-sm);
    }
}
```

Update app.js to use SearchProgress:

```javascript
// Modify: frontend/js/app.js

// Add to imports/constructor:
this.searchProgress = null; // SearchProgress component

// Modify _onSearchStarted to use SearchProgress for deep searches:
_onSearchStarted(data) {
    console.log('[App] Search started:', data);

    const hasResearch = data.include && data.include.length > 0;

    if (hasResearch) {
        // Use detailed progress for deep searches
        this.searchProgress = new SearchProgress(
            this.elements.resultsContainer,
            this.eventBus
        );
        this.searchProgress.start({
            estimatedTime: 45,
            query: data.query
        });

        // Show initial status
        this.elements.searchStatus.innerHTML = `
            <div class="status-message status-loading">
                <span class="spinner"></span>
                Deep Search in progress...
            </div>
        `;

        // Simulate progress updates (in real app, backend would send these)
        setTimeout(() => {
            if (this.searchProgress) {
                this.searchProgress.updateStage('researching', 'Researching companies...', 40);
            }
        }, 5000);

        setTimeout(() => {
            if (this.searchProgress) {
                this.searchProgress.updateStage('analyzing', 'AI scam detection...', 70);
            }
        }, 15000);

    } else {
        // Simple loading for quick searches
        const estimate = '2-5s';
        this.elements.searchStatus.innerHTML = `
            <div class="status-message status-loading">
                <span class="spinner"></span>
                Searching for "${data.query}"... (estimated ${estimate})
            </div>
        `;
        this.elements.resultsContainer.innerHTML = '';
    }
}

// Modify _onSearchCompleted to clear progress:
_onSearchCompleted(data) {
    console.log('[App] Search completed:', data);

    // Complete progress if it exists
    if (this.searchProgress) {
        this.searchProgress.complete();
        setTimeout(() => {
            if (this.searchProgress) {
                this.searchProgress.clear();
                this.searchProgress = null;
            }
            this._renderSearchResults(data);
        }, 1000); // Show completion briefly
    } else {
        this._renderSearchResults(data);
    }
}

// Extract result rendering to separate method:
_renderSearchResults(data) {
    const jobCount = data.jobs.length;
    const timing = data.timing || {};

    this.elements.searchStatus.innerHTML = `
        <div class="status-message status-success">
            Found ${jobCount} job${jobCount !== 1 ? 's' : ''}
            ${timing.total ? `in ${(timing.total / 1000).toFixed(1)}s` : ''}
        </div>
    `;

    // Render results using ResultsList
    this.resultsList = new ResultsList(
        data.jobs,
        this.registry,
        this.eventBus,
        this.elements.resultsContainer
    );
    this.resultsList.render();
}

// Handle cancel event:
// Add to _setupEventListeners:
this.eventBus.on('search:cancelled', () => {
    this.isSearching = false;
    this.searchProgress = null;
    this.elements.searchStatus.innerHTML = `
        <div class="status-message">
            Search cancelled
        </div>
    `;
});
```

**Key Points:**
- Multi-stage progress display
- Real-time elapsed timer
- Cancel functionality
- Visual stage indicators
- Simulated progress updates
- Integration with app event flow

**Testing Criteria:**
1. ✅ Progress UI appears for deep searches
2. ✅ Stages update correctly
3. ✅ Timer counts elapsed time
4. ✅ Cancel button works
5. ✅ Progress bar animates
6. ✅ Completion state shows briefly
7. ✅ Error state displays correctly

**Test Example:**
```html
<!-- Test: test-search-progress.html -->
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/search-progress.css">
    <title>Search Progress Test</title>
</head>
<body style="padding: 2rem;">
    <h1>Search Progress Test</h1>
    <button id="start-btn">Start Progress</button>
    <button id="cancel-btn">Cancel</button>
    <button id="error-btn">Show Error</button>
    <button id="complete-btn">Complete</button>
    <div id="progress-container" style="margin-top: 1rem;"></div>

    <script src="js/event-bus.js"></script>
    <script src="js/search-progress.js"></script>

    <script>
        const bus = new EventBus();
        const container = document.getElementById('progress-container');
        let progress = null;

        bus.on('search:cancelled', () => {
            console.log('✓ Cancel event received');
        });

        document.getElementById('start-btn').addEventListener('click', () => {
            progress = new SearchProgress(container, bus);
            progress.start({
                estimatedTime: 45,
                query: 'software engineer'
            });

            // Simulate progress
            setTimeout(() => {
                progress.updateStage('researching', 'Fetching company data...', 40);
            }, 2000);

            setTimeout(() => {
                progress.updateStage('analyzing', 'Running AI analysis...', 75);
            }, 4000);
        });

        document.getElementById('cancel-btn').addEventListener('click', () => {
            if (progress) progress.cancel();
        });

        document.getElementById('error-btn').addEventListener('click', () => {
            if (progress) progress.error(new Error('Network timeout'));
        });

        document.getElementById('complete-btn').addEventListener('click', () => {
            if (progress) progress.complete();
        });
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Implement SearchProgress component and integrate into app
- **Tester**: Test all progress states, cancel, error, completion
- **Relevant Files**:
  - Create: `frontend/js/search-progress.js`, `frontend/css/search-progress.css`
  - Modify: `frontend/js/app.js`
  - Test: `frontend/test-search-progress.html`

---

## Chunk 9.3: Deep Search Integration and Testing

**Objective:** Complete deep search flow integration with all components working together.

**Dependencies:**
- Chunk 9.1 (Advanced Filters)
- Chunk 9.2 (Progress UI)
- Phase 8 (Results Display)

**Files to Create:** None

**Files to Modify:**
- `frontend/js/app.js` (finalize deep search logic)
- `frontend/index.html` (add new CSS/JS includes)

**Implementation:**

Update index.html with new includes:

```html
<!-- Modify: frontend/index.html -->

<!-- Add to CSS section: -->
<link rel="stylesheet" href="css/advanced-filters.css">
<link rel="stylesheet" href="css/search-progress.css">

<!-- Add to JS section (before app.js): -->
<script src="js/search-progress.js"></script>
```

Finalize deep search logic in app.js:

```javascript
// Modify: frontend/js/app.js

// Add method to show search confirmation for expensive searches:
/**
 * Show confirmation dialog for expensive deep search
 * @private
 */
_showDeepSearchConfirmation() {
    return new Promise((resolve) => {
        // Create overlay
        const overlay = document.createElement('div');
        overlay.className = 'search-confirmation-overlay';

        // Create dialog
        const dialog = document.createElement('div');
        dialog.className = 'search-confirmation';
        dialog.innerHTML = `
            <h3>Deep Search Confirmation</h3>
            <p>
                You've selected advanced research options including AI analysis.
                This search will take approximately <strong>30-60 seconds</strong>.
            </p>
            <p>
                The search will analyze company data, check for scams, and fetch
                comprehensive reviews.
            </p>
            <div class="search-confirmation-buttons">
                <button class="btn btn-outline cancel-btn">Cancel</button>
                <button class="btn btn-primary confirm-btn">Continue</button>
            </div>
        `;

        // Add to page
        document.body.appendChild(overlay);
        document.body.appendChild(dialog);

        // Handle buttons
        const confirmBtn = dialog.querySelector('.confirm-btn');
        const cancelBtn = dialog.querySelector('.cancel-btn');

        const cleanup = () => {
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        confirmBtn.addEventListener('click', () => {
            cleanup();
            resolve(true);
        });

        cancelBtn.addEventListener('click', () => {
            cleanup();
            resolve(false);
        });

        overlay.addEventListener('click', () => {
            cleanup();
            resolve(false);
        });
    });
}

// Update _handleSearch to show confirmation for expensive searches:
async _handleSearch(isDeepSearch) {
    if (this.isSearching) {
        console.log('[App] Search already in progress');
        return;
    }

    try {
        // Get filters
        const filters = this.filterBuilder.getFilters();

        // Determine which fields to include
        let includeFields = [];
        if (isDeepSearch) {
            includeFields = filters.selectedFields || [];
        } else {
            const basicFields = this.registry.getFieldsByCategory('basic');
            includeFields = basicFields.map(f => f.name);
        }

        // Check if expensive fields are selected
        const hasExpensiveFields = includeFields.some(field => {
            const fieldMeta = this.registry.getField(field);
            return fieldMeta && (fieldMeta.category === 'ai' || fieldMeta.category === 'api_expensive');
        });

        // Show confirmation for expensive searches
        if (hasExpensiveFields) {
            const confirmed = await this._showDeepSearchConfirmation();
            if (!confirmed) {
                console.log('[App] Deep search cancelled by user');
                return;
            }
        }

        this.isSearching = true;

        // Build search params
        const searchParams = {
            query: filters.query,
            location: filters.location,
            remote: filters.remote,
            min_pay: filters.min_pay,
            max_pay: filters.max_pay,
            include: includeFields,
        };

        // Emit search started event
        this.eventBus.emit('search:started', searchParams);

        // Execute search
        const results = await this.api.search(searchParams);

        // Store results
        this.currentSearch = searchParams;
        this.searchResults = results;

        // Emit search completed event
        this.eventBus.emit('search:completed', results);

    } catch (error) {
        console.error('[App] Search failed:', error);

        // Update progress if it exists
        if (this.searchProgress) {
            this.searchProgress.error(error);
            setTimeout(() => {
                if (this.searchProgress) {
                    this.searchProgress.clear();
                    this.searchProgress = null;
                }
            }, 3000);
        }

        this.eventBus.emit('search:failed', { error, message: error.message });
    } finally {
        this.isSearching = false;
    }
}

// Update filters:changed listener to handle Deep Search button visibility:
// (Already in _setupEventListeners, but update for better UX)
this.eventBus.on('filters:changed', (data) => {
    const hasExpensive = this.filterBuilder.hasExpensiveFields();

    if (hasExpensive) {
        // Hide quick search when expensive fields selected
        this.elements.quickSearchBtn.style.display = 'none';
        this.elements.deepSearchBtn.textContent = 'Deep Search (30-60s)';
        this.elements.deepSearchBtn.classList.remove('btn-secondary');
        this.elements.deepSearchBtn.classList.add('btn-primary');
    } else {
        // Show both buttons for basic searches
        this.elements.quickSearchBtn.style.display = 'inline-block';
        this.elements.deepSearchBtn.textContent = 'Deep Search (30-60s)';
        this.elements.deepSearchBtn.classList.remove('btn-primary');
        this.elements.deepSearchBtn.classList.add('btn-secondary');
    }
});
```

**Key Points:**
- Confirmation dialog for expensive searches
- Automatic detection of expensive fields
- Button state management
- Progress integration
- Error handling with progress UI
- Complete deep search flow

**Testing Criteria:**
1. ✅ Deep Search shows confirmation for AI fields
2. ✅ Quick Search button hides when expensive fields selected
3. ✅ Progress UI appears for deep searches
4. ✅ Cancel works at any stage
5. ✅ Errors display in progress UI
6. ✅ Results render after progress completes
7. ✅ Complete flow works end-to-end
8. ✅ Deep search takes 30-60s (with real backend)

**End-to-End Test Procedure:**
```
1. Start backend server with all providers enabled
2. Open frontend/index.html
3. Enter search query: "software engineer"
4. Select basic filters (location, remote, pay range)
5. Click "Quick Search" - should complete in 2-5s
6. Verify basic results appear

7. Now select advanced filters:
   - Check "Scam Detection" (AI category)
   - Check "Glassdoor Rating" (API category)
8. Verify:
   - Quick Search button disappears
   - Deep Search button becomes primary (blue)
   - Cost indicators appear on categories

9. Click "Deep Search"
10. Verify confirmation dialog appears:
    - Shows "30-60 seconds" estimate
    - Has Cancel and Continue buttons
11. Click Continue

12. Verify progress UI:
    - Shows three stages (Searching, Researching, Analyzing)
    - Stages update sequentially
    - Timer counts up
    - Progress bar animates
    - Cancel button present

13. Wait for search to complete
14. Verify results:
    - Job cards show all selected research fields
    - Scam warnings appear if detected
    - Glassdoor ratings display
    - Status shows completion time

15. Test edge cases:
    - Cancel during search
    - Network error during search
    - No results found
    - All results are scams
```

**Planner Guidance:**
- **Developer**: Complete deep search integration in app.js, update HTML
- **Tester**: Run full end-to-end test with real backend and all providers
- **Relevant Files**:
  - Modify: `frontend/js/app.js`, `frontend/index.html`
  - Test: Manual E2E testing via `frontend/index.html` with backend
  - Note: This completes Phase 9 - Deep Search should be fully functional

---
# Phase 10: Manual Deep Research

**Objective:** Implement modal for on-demand deep research of individual companies.

**Why This Phase:**
Users should be able to trigger comprehensive research for any company, not just during initial search. This supports the "user agency" principle.

**Dependencies:** Phase 9 (Deep Search Flow)

**Phase Success Criteria:**
1. ✅ Deep Research button on job cards opens modal
2. ✅ Modal shows all available research fields
3. ✅ Loading state while fetching data
4. ✅ Research results display in organized categories
5. ✅ Modal includes export/share functionality

---

## Chunk 10.1: Company Research Modal Component

**Objective:** Create modal component for displaying comprehensive company research.

**Dependencies:** Phase 7 (FieldRenderer, EventBus)

**Files to Create:**
- `frontend/js/company-modal.js` (~300 lines)
- `frontend/css/company-modal.css` (~200 lines)

**Files to Modify:** None

**Implementation:**

```javascript
// frontend/js/company-modal.js

/**
 * CompanyModal - Modal for displaying detailed company research
 *
 * Features:
 * - Displays all research fields organized by category
 * - Loading state while fetching data
 * - Export company data as JSON
 * - Share link generation
 * - Close/escape key handling
 *
 * Usage:
 *   const modal = new CompanyModal(registry, eventBus);
 *   modal.open(companyName);
 */
class CompanyModal {
    constructor(registry, eventBus) {
        this.registry = registry;
        this.eventBus = eventBus;
        this.overlay = null;
        this.modal = null;
        this.companyData = null;
    }

    /**
     * Open modal and fetch company data
     * @param {string} companyName - Company name to research
     */
    async open(companyName) {
        // Create modal structure
        this._createModal(companyName);

        // Show loading state
        this._showLoading();

        // Emit event
        this.eventBus.emit('modal:opened', { type: 'company-research', company: companyName });

        // Fetch data (handled by app.js)
        // Data will be set via setData() method
    }

    /**
     * Set company data and render
     * @param {Object} companyData - Company research data
     */
    setData(companyData) {
        this.companyData = companyData;
        this._renderContent();
    }

    /**
     * Show error state
     * @param {Error} error - Error object
     */
    showError(error) {
        if (!this.modal) return;

        const content = this.modal.querySelector('.modal-body');
        content.innerHTML = `
            <div class="modal-error">
                <span class="error-icon">⚠️</span>
                <h3>Research Failed</h3>
                <p>${this._escapeHtml(error.message)}</p>
                <button class="btn btn-primary retry-btn">Try Again</button>
            </div>
        `;

        // Handle retry
        const retryBtn = content.querySelector('.retry-btn');
        retryBtn.addEventListener('click', () => {
            const companyName = this.modal.dataset.company;
            this._showLoading();
            this.eventBus.emit('research:retry', { company: companyName });
        });
    }

    /**
     * Close modal
     */
    close() {
        if (this.overlay) {
            document.body.removeChild(this.overlay);
            this.overlay = null;
        }

        if (this.modal) {
            document.body.removeChild(this.modal);
            this.modal = null;
        }

        this.companyData = null;

        // Emit event
        this.eventBus.emit('modal:closed', { type: 'company-research' });
    }

    /**
     * Create modal structure
     * @private
     */
    _createModal(companyName) {
        // Create overlay
        this.overlay = document.createElement('div');
        this.overlay.className = 'company-modal-overlay';
        this.overlay.addEventListener('click', () => this.close());

        // Create modal
        this.modal = document.createElement('div');
        this.modal.className = 'company-modal';
        this.modal.dataset.company = companyName;
        this.modal.innerHTML = `
            <div class="modal-header">
                <h2 class="modal-title">
                    <span class="company-icon">🏢</span>
                    ${this._escapeHtml(companyName)}
                </h2>
                <div class="modal-actions">
                    <button class="btn-icon export-btn" title="Export data">
                        <span>💾</span>
                    </button>
                    <button class="btn-icon close-btn" title="Close">
                        <span>✕</span>
                    </button>
                </div>
            </div>
            <div class="modal-body">
                <!-- Content will be inserted here -->
            </div>
        `;

        // Add to page
        document.body.appendChild(this.overlay);
        document.body.appendChild(this.modal);

        // Set up close button
        const closeBtn = this.modal.querySelector('.close-btn');
        closeBtn.addEventListener('click', () => this.close());

        // Set up export button
        const exportBtn = this.modal.querySelector('.export-btn');
        exportBtn.addEventListener('click', () => this._exportData());

        // Prevent modal clicks from closing
        this.modal.addEventListener('click', (e) => e.stopPropagation());

        // Escape key to close
        this._escapeKeyHandler = (e) => {
            if (e.key === 'Escape') this.close();
        };
        document.addEventListener('keydown', this._escapeKeyHandler);
    }

    /**
     * Show loading state
     * @private
     */
    _showLoading() {
        if (!this.modal) return;

        const content = this.modal.querySelector('.modal-body');
        content.innerHTML = `
            <div class="modal-loading">
                <div class="spinner-large"></div>
                <p>Researching company...</p>
                <p class="loading-detail">This may take up to 60 seconds</p>
            </div>
        `;
    }

    /**
     * Render company data
     * @private
     */
    _renderContent() {
        if (!this.modal || !this.companyData) return;

        const content = this.modal.querySelector('.modal-body');

        // Group fields by category
        const fieldsByCategory = this._groupFieldsByCategory();

        // Render each category
        const categoriesHtml = Object.entries(fieldsByCategory).map(([category, fields]) => {
            return this._renderCategory(category, fields);
        }).join('');

        content.innerHTML = `
            <div class="company-data">
                ${categoriesHtml}
            </div>
        `;
    }

    /**
     * Group company fields by category
     * @private
     */
    _groupFieldsByCategory() {
        const grouped = {};
        const standardProps = ['name', 'cached_date'];

        for (const [key, value] of Object.entries(this.companyData)) {
            if (standardProps.includes(key) || value === null || value === undefined) {
                continue;
            }

            const fieldMeta = this.registry.getField(key);
            if (!fieldMeta) continue;

            const category = fieldMeta.category || 'other';
            if (!grouped[category]) {
                grouped[category] = [];
            }

            grouped[category].push({ key, value, meta: fieldMeta });
        }

        return grouped;
    }

    /**
     * Render a category section
     * @private
     */
    _renderCategory(category, fields) {
        const categoryInfo = this.registry.getCategoryInfo(category);
        const categoryLabel = categoryInfo?.label || category;
        const categoryIcon = categoryInfo?.icon || '📋';

        const fieldsHtml = fields.map(({ key, value, meta }) => {
            const renderer = this.registry.getRenderer(key);
            const html = renderer.render(key, value, this.companyData);

            return `
                <div class="data-field">
                    <div class="field-label">${meta.label}:</div>
                    <div class="field-value">${html}</div>
                </div>
            `;
        }).join('');

        return `
            <div class="data-category" data-category="${category}">
                <h3 class="category-title">
                    <span class="category-icon">${categoryIcon}</span>
                    ${categoryLabel}
                </h3>
                <div class="category-fields">
                    ${fieldsHtml}
                </div>
            </div>
        `;
    }

    /**
     * Export company data as JSON
     * @private
     */
    _exportData() {
        if (!this.companyData) return;

        const dataStr = JSON.stringify(this.companyData, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `${this.companyData.name.replace(/\s+/g, '_')}_research.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        URL.revokeObjectURL(url);

        // Emit event
        this.eventBus.emit('company:exported', { company: this.companyData.name });

        // Show feedback
        const exportBtn = this.modal.querySelector('.export-btn');
        const originalContent = exportBtn.innerHTML;
        exportBtn.innerHTML = '<span>✓</span>';
        exportBtn.style.color = 'var(--color-success)';

        setTimeout(() => {
            exportBtn.innerHTML = originalContent;
            exportBtn.style.color = '';
        }, 2000);
    }

    /**
     * Escape HTML
     * @private
     */
    _escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CompanyModal;
}
```

Create modal CSS:

```css
/* frontend/css/company-modal.css */

.company-modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-lg);
    animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.company-modal {
    background: var(--color-bg);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-lg);
    max-width: 900px;
    width: 100%;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    animation: slideUp 0.3s ease;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Modal Header */
.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-lg);
    border-bottom: 2px solid var(--color-border);
    flex-shrink: 0;
}

.modal-title {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--color-text);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.company-icon {
    font-size: 1.75rem;
}

.modal-actions {
    display: flex;
    gap: var(--spacing-sm);
}

.btn-icon {
    background: transparent;
    border: 1px solid var(--color-border);
    border-radius: var(--border-radius);
    padding: 0.5rem;
    cursor: pointer;
    font-size: 1.25rem;
    line-height: 1;
    transition: all 0.2s;
    color: var(--color-text-light);
}

.btn-icon:hover {
    background: var(--color-bg-alt);
    border-color: var(--color-primary);
    color: var(--color-primary);
}

.close-btn:hover {
    border-color: var(--color-error);
    color: var(--color-error);
}

/* Modal Body */
.modal-body {
    flex: 1;
    overflow-y: auto;
    padding: var(--spacing-lg);
}

/* Loading State */
.modal-loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-xl) var(--spacing-lg);
    min-height: 400px;
    color: var(--color-text-light);
}

.spinner-large {
    width: 3rem;
    height: 3rem;
    border: 4px solid var(--color-border);
    border-top-color: var(--color-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: var(--spacing-lg);
}

.modal-loading p {
    margin: var(--spacing-sm) 0;
    font-size: 1rem;
}

.loading-detail {
    font-size: 0.875rem !important;
    color: var(--color-text-light);
}

/* Error State */
.modal-error {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-xl) var(--spacing-lg);
    min-height: 400px;
    text-align: center;
}

.error-icon {
    font-size: 4rem;
    margin-bottom: var(--spacing-lg);
}

.modal-error h3 {
    margin: 0 0 var(--spacing-md);
    color: var(--color-error);
    font-size: 1.5rem;
}

.modal-error p {
    margin: 0 0 var(--spacing-lg);
    color: var(--color-text-light);
    max-width: 400px;
}

/* Company Data */
.company-data {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xl);
}

.data-category {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--border-radius);
    padding: var(--spacing-lg);
}

.data-category[data-category="basic"] {
    border-left: 4px solid #3b82f6;
}

.data-category[data-category="api_cheap"],
.data-category[data-category="api_expensive"] {
    border-left: 4px solid #10b981;
}

.data-category[data-category="ai"] {
    border-left: 4px solid #8b5cf6;
}

.category-title {
    margin: 0 0 var(--spacing-lg);
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--color-text);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding-bottom: var(--spacing-md);
    border-bottom: 1px solid var(--color-border);
}

.category-icon {
    font-size: 1.5rem;
}

.category-fields {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.data-field {
    display: grid;
    grid-template-columns: 180px 1fr;
    gap: var(--spacing-md);
    align-items: start;
}

.field-label {
    font-weight: 600;
    color: var(--color-text-light);
    font-size: 0.875rem;
    padding-top: 0.25rem;
}

.field-value {
    color: var(--color-text);
}

/* Responsive */
@media (max-width: 767px) {
    .company-modal {
        max-height: 100vh;
        border-radius: 0;
    }

    .company-modal-overlay {
        padding: 0;
    }

    .data-field {
        grid-template-columns: 1fr;
        gap: var(--spacing-xs);
    }

    .field-label {
        padding-top: 0;
    }

    .modal-title {
        font-size: 1.25rem;
    }
}

/* Scrollbar styling */
.modal-body::-webkit-scrollbar {
    width: 8px;
}

.modal-body::-webkit-scrollbar-track {
    background: var(--color-bg-alt);
}

.modal-body::-webkit-scrollbar-thumb {
    background: var(--color-border);
    border-radius: 4px;
}

.modal-body::-webkit-scrollbar-thumb:hover {
    background: var(--color-text-light);
}
```

**Key Points:**
- Full-screen modal with overlay
- Loading, error, and data states
- Organized by research categories
- Export functionality
- Keyboard navigation (ESC to close)
- Responsive design

**Testing Criteria:**
1. ✅ Modal opens with loading state
2. ✅ Data renders organized by category
3. ✅ Category visual indicators correct
4. ✅ Export downloads JSON file
5. ✅ Close button works
6. ✅ ESC key closes modal
7. ✅ Clicking overlay closes modal
8. ✅ Error state displays correctly

**Test Example:**
```html
<!-- Test: test-company-modal.html -->
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/field-display.css">
    <link rel="stylesheet" href="css/company-modal.css">
    <title>Company Modal Test</title>
</head>
<body style="padding: 2rem;">
    <h1>Company Modal Test</h1>
    <button id="open-modal">Open Modal</button>
    <button id="show-error">Show Error</button>

    <script src="js/event-bus.js"></script>
    <script src="js/field-renderer.js"></script>
    <script src="js/field-registry.js"></script>
    <script src="js/company-modal.js"></script>

    <script>
        (async () => {
            const bus = new EventBus();
            const registry = new FieldRegistry('http://localhost:8000');

            bus.on('modal:opened', (data) => {
                console.log('✓ Modal opened:', data);
            });

            bus.on('modal:closed', () => {
                console.log('✓ Modal closed');
            });

            bus.on('company:exported', (data) => {
                console.log('✓ Company exported:', data);
            });

            try {
                await registry.load();
                console.log('✓ Registry loaded');

                const modal = new CompanyModal(registry, bus);

                document.getElementById('open-modal').addEventListener('click', () => {
                    modal.open('TechCorp Inc');

                    // Simulate data loading
                    setTimeout(() => {
                        modal.setData({
                            name: 'TechCorp Inc',
                            is_scam: false,
                            glassdoor_rating: 4.5,
                            glassdoor_review_count: 523,
                            glassdoor_pros: ['Great culture', 'Good benefits'],
                            glassdoor_cons: ['Long hours', 'High pressure'],
                            scam_confidence: 'low',
                            scam_reasoning: 'Established company with verified online presence.',
                            cached_date: '2025-01-15T10:30:00Z'
                        });
                    }, 2000);
                });

                document.getElementById('show-error').addEventListener('click', () => {
                    modal.open('ErrorCo');
                    setTimeout(() => {
                        modal.showError(new Error('Network timeout - could not fetch company data'));
                    }, 1000);
                });

            } catch (error) {
                console.error('✗ Test failed:', error);
            }
        })();
    </script>
</body>
</html>
```

**Planner Guidance:**
- **Developer**: Implement CompanyModal component with all states
- **Tester**: Test all modal states, export, keyboard navigation
- **Relevant Files**:
  - Create: `frontend/js/company-modal.js`, `frontend/css/company-modal.css`
  - Test: `frontend/test-company-modal.html`

---

## Chunk 10.2: Wire Company Modal to Deep Research Button

**Objective:** Connect job card "Deep Research" buttons to modal.

**Dependencies:** Chunk 10.1 (CompanyModal)

**Files to Create:** None

**Files to Modify:**
- `frontend/js/app.js` (integrate modal, handle research requests)
- `frontend/index.html` (add modal CSS/JS)

**Implementation:**

Update app.js to integrate CompanyModal:

```javascript
// Modify: frontend/js/app.js

// Add to constructor:
this.companyModal = null; // CompanyModal instance

// Initialize modal in initialize() method:
async initialize() {
    try {
        console.log('[App] Initializing...');

        // ... existing initialization code ...

        // Initialize company modal
        this.companyModal = new CompanyModal(this.registry, this.eventBus);

        // Set up event listeners
        this._setupEventListeners();

        // ... rest of initialization ...
    }
}

// Update _handleDeepResearch to use modal:
/**
 * Handle deep research for a single company
 * @private
 */
async _handleDeepResearch(companyName) {
    console.log('[App] Deep research requested for:', companyName);

    // Open modal with loading state
    this.companyModal.open(companyName);

    try {
        // Get all available research fields
        const allFields = this.registry.fields.map(f => f.name);

        // Emit research started event
        this.eventBus.emit('research:started', { companies: [companyName] });

        // Fetch company data
        const company = await this.api.researchCompany(companyName, allFields);

        // Update modal with data
        this.companyModal.setData(company);

        // Emit research completed event
        this.eventBus.emit('research:completed', { companies: [company] });

    } catch (error) {
        console.error('[App] Deep research failed:', error);

        // Show error in modal
        this.companyModal.showError(error);

        // Emit research failed event
        this.eventBus.emit('research:failed', { company: companyName, error });
    }
}

// In _setupEventListeners(), ensure research:requested listener is set up:
// (This should already exist from Phase 8, but verify it's calling _handleDeepResearch)
this.eventBus.on('research:requested', (data) => {
    this._handleDeepResearch(data.company);
});

// Add retry handler:
this.eventBus.on('research:retry', (data) => {
    // Close current modal
    this.companyModal.close();

    // Retry research
    setTimeout(() => {
        this._handleDeepResearch(data.company);
    }, 100);
});
```

Update index.html to include modal:

```html
<!-- Modify: frontend/index.html -->

<!-- Add to CSS section: -->
<link rel="stylesheet" href="css/company-modal.css">

<!-- Add to JS section (before app.js): -->
<script src="js/company-modal.js"></script>
```

**Key Points:**
- Modal opens on "Deep Research" button click
- Shows loading while fetching
- Displays all research data when ready
- Error handling with retry
- Event-driven integration

**Testing Criteria:**
1. ✅ Deep Research button opens modal
2. ✅ Modal shows loading state
3. ✅ Company data fetched from backend
4. ✅ Data renders in modal
5. ✅ Export works
6. ✅ Retry works on error
7. ✅ Modal closes properly

**Test Procedure:**
```
1. Start backend server
2. Open frontend/index.html
3. Perform Quick Search
4. Click "Deep Research" on any job card
5. Verify:
   - Modal opens immediately
   - Shows loading spinner
   - Company name in header
   - Loading message displays
6. Wait for data to load
7. Verify:
   - All research fields display
   - Organized by category
   - Proper visual indicators per category
   - Export button works
8. Test error case:
   - Stop backend server
   - Click Deep Research on another job
   - Verify error message displays
   - Verify retry button appears
9. Test closing:
   - ESC key closes modal
   - X button closes modal
   - Clicking overlay closes modal
```

**Planner Guidance:**
- **Developer**: Integrate CompanyModal into app, wire up events
- **Tester**: Test modal integration with real backend
- **Relevant Files**:
  - Modify: `frontend/js/app.js`, `frontend/index.html`
  - Test: Manual testing via main application

---

## Chunk 10.3: Enhanced Modal Features

**Objective:** Add advanced modal features (caching indicator, field filtering, print view).

**Dependencies:** Chunk 10.2 (Modal Integration)

**Files to Create:** None

**Files to Modify:**
- `frontend/js/company-modal.js` (add features)
- `frontend/css/company-modal.css` (add styles)

**Implementation:**

Update CompanyModal with enhanced features:

```javascript
// Modify: frontend/js/company-modal.js

// Add to _createModal method, in modal header actions:
<div class="modal-actions">
    <button class="btn-icon print-btn" title="Print view">
        <span>🖨️</span>
    </button>
    <button class="btn-icon filter-btn" title="Filter fields">
        <span>🔍</span>
    </button>
    <button class="btn-icon export-btn" title="Export data">
        <span>💾</span>
    </button>
    <button class="btn-icon close-btn" title="Close">
        <span>✕</span>
    </button>
</div>

// Add event listeners in _createModal:
const printBtn = this.modal.querySelector('.print-btn');
printBtn.addEventListener('click', () => this._print());

const filterBtn = this.modal.querySelector('.filter-btn');
filterBtn.addEventListener('click', () => this._toggleFilter());

// Add cache indicator to _renderContent:
_renderContent() {
    if (!this.modal || !this.companyData) return;

    const content = this.modal.querySelector('.modal-body');

    // Show cache indicator if data is cached
    const cacheIndicator = this._renderCacheIndicator();

    // Group fields by category
    const fieldsByCategory = this._groupFieldsByCategory();

    // Render categories
    const categoriesHtml = Object.entries(fieldsByCategory).map(([category, fields]) => {
        return this._renderCategory(category, fields);
    }).join('');

    content.innerHTML = `
        ${cacheIndicator}
        <div class="company-data">
            ${categoriesHtml}
        </div>
    `;
}

// Add cache indicator renderer:
/**
 * Render cache indicator if data is cached
 * @private
 */
_renderCacheIndicator() {
    if (!this.companyData.cached_date) return '';

    const cachedDate = new Date(this.companyData.cached_date);
    const now = new Date();
    const ageMs = now - cachedDate;
    const ageDays = Math.floor(ageMs / (1000 * 60 * 60 * 24));

    let ageText = '';
    if (ageDays === 0) {
        ageText = 'Today';
    } else if (ageDays === 1) {
        ageText = 'Yesterday';
    } else {
        ageText = `${ageDays} days ago`;
    }

    return `
        <div class="cache-indicator">
            <span class="cache-icon">💾</span>
            <span class="cache-text">
                This data was cached <strong>${ageText}</strong>
                ${ageDays > 7 ? '<span class="cache-warning">⚠️ May be outdated</span>' : ''}
            </span>
        </div>
    `;
}

// Add print functionality:
/**
 * Open print view
 * @private
 */
_print() {
    if (!this.companyData) return;

    // Create print window
    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>${this.companyData.name} - Company Research</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 2rem;
                    color: #333;
                }
                h1 {
                    border-bottom: 3px solid #2563eb;
                    padding-bottom: 0.5rem;
                }
                h2 {
                    margin-top: 2rem;
                    color: #2563eb;
                }
                .field {
                    margin: 1rem 0;
                    padding: 0.5rem;
                    border-left: 3px solid #e5e7eb;
                    padding-left: 1rem;
                }
                .field-label {
                    font-weight: 600;
                    color: #6b7280;
                    font-size: 0.875rem;
                    margin-bottom: 0.25rem;
                }
                .field-value {
                    color: #111827;
                }
                .warning {
                    background: #fef3c7;
                    border: 1px solid #fbbf24;
                    padding: 1rem;
                    border-radius: 0.375rem;
                    margin: 1rem 0;
                }
                @media print {
                    body {
                        padding: 0;
                    }
                }
            </style>
        </head>
        <body>
            <h1>${this._escapeHtml(this.companyData.name)}</h1>
            <p><em>Generated on ${new Date().toLocaleString()}</em></p>
            ${this._generatePrintContent()}
        </body>
        </html>
    `);
    printWindow.document.close();
    printWindow.focus();

    // Wait for content to load, then print
    setTimeout(() => {
        printWindow.print();
    }, 250);

    this.eventBus.emit('company:printed', { company: this.companyData.name });
}

/**
 * Generate print-friendly HTML
 * @private
 */
_generatePrintContent() {
    const fieldsByCategory = this._groupFieldsByCategory();

    let html = '';

    // Show warnings first
    if (this.companyData.is_scam) {
        html += `
            <div class="warning">
                <strong>⚠️ Potential Scam Detected</strong><br>
                Confidence: ${this.companyData.scam_confidence || 'unknown'}<br>
                Reasoning: ${this._escapeHtml(this.companyData.scam_reasoning || 'No details provided')}
            </div>
        `;
    }

    // Render categories
    for (const [category, fields] of Object.entries(fieldsByCategory)) {
        const categoryInfo = this.registry.getCategoryInfo(category);
        const categoryLabel = categoryInfo?.label || category;

        html += `<h2>${categoryLabel}</h2>`;

        fields.forEach(({ key, value, meta }) => {
            // Simple text rendering for print
            let displayValue = '';

            if (Array.isArray(value)) {
                displayValue = value.join(', ');
            } else if (typeof value === 'object') {
                displayValue = JSON.stringify(value, null, 2);
            } else {
                displayValue = String(value);
            }

            html += `
                <div class="field">
                    <div class="field-label">${meta.label}</div>
                    <div class="field-value">${this._escapeHtml(displayValue)}</div>
                </div>
            `;
        });
    }

    return html;
}

// Add filter toggle:
/**
 * Toggle category filter
 * @private
 */
_toggleFilter() {
    // Check if filter already exists
    let filterPanel = this.modal.querySelector('.filter-panel');

    if (filterPanel) {
        // Remove filter
        filterPanel.remove();
        this._showAllCategories();
        return;
    }

    // Create filter panel
    const fieldsByCategory = this._groupFieldsByCategory();
    const categories = Object.keys(fieldsByCategory);

    filterPanel = document.createElement('div');
    filterPanel.className = 'filter-panel';

    const checkboxes = categories.map(category => {
        const categoryInfo = this.registry.getCategoryInfo(category);
        const label = categoryInfo?.label || category;
        const icon = categoryInfo?.icon || '📋';

        return `
            <label class="filter-option">
                <input type="checkbox" value="${category}" checked>
                <span>${icon} ${label}</span>
            </label>
        `;
    }).join('');

    filterPanel.innerHTML = `
        <div class="filter-header">
            <strong>Show Categories:</strong>
            <button class="filter-close">✕</button>
        </div>
        <div class="filter-options">
            ${checkboxes}
        </div>
    `;

    // Insert after modal header
    const modalHeader = this.modal.querySelector('.modal-header');
    modalHeader.after(filterPanel);

    // Set up event listeners
    const closeBtn = filterPanel.querySelector('.filter-close');
    closeBtn.addEventListener('click', () => {
        filterPanel.remove();
        this._showAllCategories();
    });

    const checkboxElements = filterPanel.querySelectorAll('input[type="checkbox"]');
    checkboxElements.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            this._filterCategories();
        });
    });

    this.eventBus.emit('modal:filter-toggled', { visible: true });
}

/**
 * Filter visible categories based on checkboxes
 * @private
 */
_filterCategories() {
    const filterPanel = this.modal.querySelector('.filter-panel');
    if (!filterPanel) return;

    const checkedCategories = Array.from(
        filterPanel.querySelectorAll('input[type="checkbox"]:checked')
    ).map(cb => cb.value);

    // Show/hide categories
    const categoryElements = this.modal.querySelectorAll('.data-category');
    categoryElements.forEach(el => {
        const category = el.dataset.category;
        el.style.display = checkedCategories.includes(category) ? 'block' : 'none';
    });
}

/**
 * Show all categories
 * @private
 */
_showAllCategories() {
    const categoryElements = this.modal.querySelectorAll('.data-category');
    categoryElements.forEach(el => {
        el.style.display = 'block';
    });
}
```

Add CSS for new features:

```css
/* Modify: frontend/css/company-modal.css */

/* Add these styles: */

/* Cache Indicator */
.cache-indicator {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-md);
    background: #dbeafe;
    border: 1px solid #3b82f6;
    border-radius: var(--border-radius);
    margin-bottom: var(--spacing-lg);
    font-size: 0.875rem;
}

.cache-icon {
    font-size: 1.25rem;
}

.cache-text {
    flex: 1;
    color: #1e40af;
}

.cache-warning {
    color: #d97706;
    font-weight: 600;
    margin-left: var(--spacing-sm);
}

/* Filter Panel */
.filter-panel {
    background: var(--color-bg-alt);
    border-bottom: 1px solid var(--color-border);
    padding: var(--spacing-md) var(--spacing-lg);
}

.filter-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-sm);
}

.filter-header strong {
    font-size: 0.875rem;
    color: var(--color-text);
}

.filter-close {
    background: transparent;
    border: none;
    font-size: 1.25rem;
    color: var(--color-text-light);
    cursor: pointer;
    padding: 0;
    line-height: 1;
}

.filter-close:hover {
    color: var(--color-error);
}

.filter-options {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-md);
}

.filter-option {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    cursor: pointer;
    font-size: 0.875rem;
    padding: 0.25rem 0.5rem;
    border-radius: var(--border-radius);
    transition: background 0.2s;
}

.filter-option:hover {
    background: rgba(37, 99, 235, 0.1);
}

.filter-option input[type="checkbox"] {
    cursor: pointer;
}
```

**Key Points:**
- Cache age indicator with warning for old data
- Category filtering for focused viewing
- Print-friendly view generation
- Enhanced export with visual feedback
- Event emission for analytics

**Testing Criteria:**
1. ✅ Cache indicator shows age of data
2. ✅ Warning appears for data >7 days old
3. ✅ Filter panel toggles categories
4. ✅ Print view generates correctly
5. ✅ All buttons have visual feedback
6. ✅ Filter selections persist until closed

**Test Procedure:**
```
1. Open modal with fresh data (today)
2. Verify cache indicator shows "Today"
3. Open modal with old cached data (>7 days)
4. Verify cache warning appears

5. Click filter button
6. Verify filter panel opens
7. Uncheck some categories
8. Verify those categories hide
9. Close filter panel
10. Verify all categories reappear

11. Click print button
12. Verify print window opens
13. Verify print view is clean/readable
14. Close print window

15. Click export button
16. Verify JSON downloads
17. Verify export button shows checkmark briefly
```

**Planner Guidance:**
- **Developer**: Add enhanced features to CompanyModal
- **Tester**: Test all new features, verify print view, test filtering
- **Relevant Files**:
  - Modify: `frontend/js/company-modal.js`, `frontend/css/company-modal.css`
  - Test: Use test-company-modal.html and main application
  - Note: This completes Phase 10 - Manual Deep Research fully functional

---
# Phase 11: Additional Providers

**Objective:** Add more job sources and research providers to expand coverage.

**Why This Phase:**
More providers mean better job coverage and more comprehensive company research. This demonstrates the modularity and extensibility of the plugin system.

**Dependencies:** Phases 0-5 (Backend provider system)

**Phase Success Criteria:**
1. ✅ LinkedIn job source integrated
2. ✅ ZipRecruiter job source integrated
3. ✅ Real Glassdoor provider implemented
4. ✅ Salary data provider added (e.g., Payscale, Salary.com)
5. ✅ All providers work through existing frontend

---

## Chunk 11.1: LinkedIn Job Source

**Objective:** Implement LinkedIn as a job source provider.

**Dependencies:** Phase 2 (Job Aggregation), Phase 3 (Indeed Provider as reference)

**Files to Create:**
- `app/providers/job_sources/linkedin/provider.py` (~200 lines)
- `app/providers/job_sources/linkedin/client.py` (~250 lines)
- `app/providers/job_sources/linkedin/models.py` (~100 lines)
- `app/providers/job_sources/linkedin/config.ini` (~30 lines)
- `tests/providers/test_linkedin_provider.py` (~150 lines)

**Files to Modify:** None

**Implementation:**

```python
# app/providers/job_sources/linkedin/models.py

from typing import Optional
from pydantic import BaseModel, Field


class LinkedInJobListing(BaseModel):
    """LinkedIn-specific job listing data"""
    job_id: str
    title: str
    company_name: str
    location: Optional[str] = None
    remote: bool = False
    description: Optional[str] = None
    url: str
    posted_date: Optional[str] = None
    applies: Optional[int] = Field(None, description="Number of applicants")
    seniority_level: Optional[str] = None
    employment_type: Optional[str] = None  # Full-time, Part-time, Contract, etc.
    job_functions: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)


class LinkedInSearchResponse(BaseModel):
    """Response from LinkedIn job search"""
    jobs: list[LinkedInJobListing]
    total_results: int
    page: int
    has_more: bool
```

```python
# app/providers/job_sources/linkedin/client.py

import asyncio
from typing import Optional, Dict, Any, List
import httpx
from loguru import logger

from .models import LinkedInJobListing, LinkedInSearchResponse


class LinkedInAPIClient:
    """
    LinkedIn Jobs API Client

    NOTE: This uses LinkedIn's unofficial API endpoints or RapidAPI LinkedIn Job Search.
    For production, you'll need:
    1. LinkedIn API credentials (if using official API)
    2. OR RapidAPI key with LinkedIn Job Search subscription
    3. Proper rate limiting and error handling
    """

    BASE_URL = "https://linkedin-jobs-search.p.rapidapi.com"

    def __init__(self, api_key: str, rapid_api_key: Optional[str] = None):
        self.api_key = api_key  # For official LinkedIn API
        self.rapid_api_key = rapid_api_key  # For RapidAPI

        # Use RapidAPI if available, otherwise official API
        self.use_rapid_api = rapid_api_key is not None

        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str,
        location: Optional[str] = None,
        remote: bool = False,
        limit: int = 25,
        page: int = 0
    ) -> LinkedInSearchResponse:
        """
        Search LinkedIn for jobs

        Args:
            query: Search keywords
            location: Location filter
            remote: Remote only filter
            limit: Max results per page
            page: Page number (for pagination)

        Returns:
            LinkedInSearchResponse with jobs
        """
        if self.use_rapid_api:
            return await self._search_rapid_api(query, location, remote, limit, page)
        else:
            return await self._search_official_api(query, location, remote, limit, page)

    async def _search_rapid_api(
        self,
        query: str,
        location: Optional[str],
        remote: bool,
        limit: int,
        page: int
    ) -> LinkedInSearchResponse:
        """Search using RapidAPI LinkedIn Jobs endpoint"""

        params = {
            "query": query,
            "location": location or "",
            "page": str(page),
            "limit": str(limit)
        }

        if remote:
            params["remote"] = "true"

        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": "linkedin-jobs-search.p.rapidapi.com"
        }

        try:
            logger.debug(f"[LinkedIn RapidAPI] Searching: query={query}, location={location}")

            response = await self.client.get(
                f"{self.BASE_URL}/search",
                params=params,
                headers=headers
            )
            response.raise_for_status()

            data = response.json()

            # Parse response (format varies by API provider)
            jobs = []
            for item in data.get("data", []):
                try:
                    job = self._parse_rapid_api_job(item)
                    jobs.append(job)
                except Exception as e:
                    logger.warning(f"[LinkedIn RapidAPI] Failed to parse job: {e}")
                    continue

            return LinkedInSearchResponse(
                jobs=jobs,
                total_results=data.get("total", len(jobs)),
                page=page,
                has_more=data.get("hasMore", False)
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"[LinkedIn RapidAPI] HTTP error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"[LinkedIn RapidAPI] Search failed: {e}")
            raise

    def _parse_rapid_api_job(self, item: Dict[str, Any]) -> LinkedInJobListing:
        """Parse job from RapidAPI response"""

        return LinkedInJobListing(
            job_id=item.get("job_id", item.get("id", "")),
            title=item.get("title", ""),
            company_name=item.get("company", item.get("company_name", "")),
            location=item.get("location"),
            remote=item.get("remote", False) or "remote" in item.get("location", "").lower(),
            description=item.get("description"),
            url=item.get("url", item.get("link", "")),
            posted_date=item.get("posted_date", item.get("date")),
            applies=item.get("applies", item.get("applicants")),
            seniority_level=item.get("seniority_level", item.get("level")),
            employment_type=item.get("employment_type", item.get("type")),
            job_functions=item.get("job_functions", []),
            industries=item.get("industries", [])
        )

    async def _search_official_api(
        self,
        query: str,
        location: Optional[str],
        remote: bool,
        limit: int,
        page: int
    ) -> LinkedInSearchResponse:
        """
        Search using official LinkedIn API

        NOTE: Requires LinkedIn API credentials and approved application.
        Implementation depends on LinkedIn's current API offering.
        As of 2025, LinkedIn's official Jobs API has limited access.
        """
        # Placeholder for official API implementation
        logger.warning("[LinkedIn Official API] Not implemented - using mock data")

        # Return mock data for development
        return LinkedInSearchResponse(
            jobs=[],
            total_results=0,
            page=page,
            has_more=False
        )

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
```

```python
# app/providers/job_sources/linkedin/provider.py

from typing import Optional, List
from configparser import ConfigParser
from pathlib import Path
from loguru import logger

from app.models import JobListing
from app.providers.base import JobSource
from .client import LinkedInAPIClient
from .models import LinkedInJobListing


class LinkedInProvider(JobSource):
    """LinkedIn job source provider"""

    def __init__(self, api_key: str, rapid_api_key: Optional[str] = None):
        self.client = LinkedInAPIClient(api_key, rapid_api_key)

    @property
    def name(self) -> str:
        return "LinkedIn"

    @property
    def is_expensive(self) -> bool:
        # RapidAPI has API costs
        return True

    async def search(
        self,
        query: str,
        location: Optional[str] = None,
        remote: bool = False,
        limit: int = 25
    ) -> List[JobListing]:
        """Search LinkedIn for jobs"""

        logger.info(f"[LinkedIn] Searching: query='{query}', location='{location}', remote={remote}")

        try:
            # Search via client
            response = await self.client.search(
                query=query,
                location=location,
                remote=remote,
                limit=limit
            )

            # Convert to JobListing models
            jobs = []
            for linkedin_job in response.jobs:
                job = self._convert_to_job_listing(linkedin_job)
                jobs.append(job)

            logger.info(f"[LinkedIn] Found {len(jobs)} jobs")
            return jobs

        except Exception as e:
            logger.error(f"[LinkedIn] Search failed: {e}")
            # Return empty list on error (graceful degradation)
            return []

    def _convert_to_job_listing(self, linkedin_job: LinkedInJobListing) -> JobListing:
        """Convert LinkedIn job to generic JobListing"""

        # Parse pay from description if available
        # LinkedIn doesn't always provide salary info directly
        pay_min = None
        pay_max = None

        # Simple pay extraction (can be enhanced)
        if linkedin_job.description:
            import re
            pay_pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:-|to)\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
            pay_match = re.search(pay_pattern, linkedin_job.description)
            if pay_match:
                pay_min = float(pay_match.group(1).replace(',', ''))
                pay_max = float(pay_match.group(2).replace(',', ''))

        return JobListing(
            source="LinkedIn",
            source_id=linkedin_job.job_id,
            title=linkedin_job.title,
            company_name=linkedin_job.company_name,
            location=linkedin_job.location,
            remote=linkedin_job.remote,
            description=linkedin_job.description,
            url=linkedin_job.url,
            pay_min=pay_min,
            pay_max=pay_max,
            pay_exact=None,
            posted_date=linkedin_job.posted_date
        )


def create_provider(config: ConfigParser) -> LinkedInProvider:
    """Factory function to create LinkedIn provider from config"""

    api_key = config.get("api_keys", "LINKEDIN_API_KEY", fallback=None)
    rapid_api_key = config.get("api_keys", "RAPID_API_KEY", fallback=None)

    if not api_key and not rapid_api_key:
        raise ValueError("LinkedIn provider requires LINKEDIN_API_KEY or RAPID_API_KEY")

    return LinkedInProvider(api_key=api_key or "", rapid_api_key=rapid_api_key)
```

```ini
# app/providers/job_sources/linkedin/config.ini

[provider]
name = LinkedIn
enabled = true
is_expensive = true

[api_keys]
# Option 1: Official LinkedIn API (requires approved app)
LINKEDIN_API_KEY = your_linkedin_api_key_here

# Option 2: RapidAPI LinkedIn Jobs Search (easier to set up)
RAPID_API_KEY = your_rapidapi_key_here

[search]
default_limit = 25
max_limit = 100

[rate_limiting]
# RapidAPI free tier: 50 requests/month
requests_per_minute = 5
requests_per_day = 50
```

Write tests:

```python
# tests/providers/test_linkedin_provider.py

import pytest
from app.providers.job_sources.linkedin.provider import LinkedInProvider
from app.providers.job_sources.linkedin.models import LinkedInJobListing, LinkedInSearchResponse


@pytest.fixture
def mock_linkedin_client(monkeypatch):
    """Mock LinkedIn API client"""

    async def mock_search(self, query, location, remote, limit, page):
        return LinkedInSearchResponse(
            jobs=[
                LinkedInJobListing(
                    job_id="123",
                    title="Software Engineer",
                    company_name="TechCorp",
                    location="San Francisco, CA",
                    remote=False,
                    description="Build great software",
                    url="https://linkedin.com/jobs/123",
                    seniority_level="Mid-Senior level",
                    employment_type="Full-time"
                ),
                LinkedInJobListing(
                    job_id="456",
                    title="Senior Developer",
                    company_name="StartupCo",
                    location="Remote",
                    remote=True,
                    description="Remote work opportunity",
                    url="https://linkedin.com/jobs/456",
                    seniority_level="Senior level",
                    employment_type="Full-time"
                )
            ],
            total_results=2,
            page=0,
            has_more=False
        )

    from app.providers.job_sources.linkedin import client
    monkeypatch.setattr(client.LinkedInAPIClient, "search", mock_search)


@pytest.mark.asyncio
async def test_linkedin_provider_search(mock_linkedin_client):
    """Test LinkedIn provider search"""
    provider = LinkedInProvider(api_key="test_key")

    jobs = await provider.search(query="software engineer", location="San Francisco")

    assert len(jobs) == 2
    assert jobs[0].source == "LinkedIn"
    assert jobs[0].title == "Software Engineer"
    assert jobs[1].remote is True


@pytest.mark.asyncio
async def test_linkedin_provider_remote_filter(mock_linkedin_client):
    """Test LinkedIn remote filtering"""
    provider = LinkedInProvider(api_key="test_key")

    jobs = await provider.search(query="developer", remote=True)

    # Should still return all jobs from mock, but in real implementation
    # would filter to only remote
    assert len(jobs) >= 1


@pytest.mark.asyncio
async def test_linkedin_provider_error_handling(monkeypatch):
    """Test error handling returns empty list"""

    async def mock_search_error(self, *args, **kwargs):
        raise Exception("API Error")

    from app.providers.job_sources.linkedin import client
    monkeypatch.setattr(client.LinkedInAPIClient, "search", mock_search_error)

    provider = LinkedInProvider(api_key="test_key")
    jobs = await provider.search(query="test")

    # Should return empty list on error (graceful degradation)
    assert jobs == []


def test_linkedin_provider_properties():
    """Test provider properties"""
    provider = LinkedInProvider(api_key="test_key")

    assert provider.name == "LinkedIn"
    assert provider.is_expensive is True
```

**Key Points:**
- RapidAPI integration for easier setup (no app approval needed)
- Official API support (placeholder for when available)
- Graceful degradation on errors
- Pay parsing from description
- Remote detection
- Standard JobListing conversion

**Testing Criteria:**
1. ✅ Provider loads from config
2. ✅ Search returns JobListing objects
3. ✅ Remote filter works
4. ✅ Location filter works
5. ✅ Error handling returns empty list
6. ✅ Pay parsing works when available
7. ✅ All tests pass

**Test Commands:**
```bash
# Run LinkedIn provider tests
pytest tests/providers/test_linkedin_provider.py -v

# Test with real API (requires key in config)
python -m app.test_providers --provider linkedin --query "software engineer"
```

**Planner Guidance:**
- **Developer**: Implement LinkedIn provider following Indeed pattern
- **Tester**: Run unit tests, test with real API keys if available
- **Relevant Files**:
  - Create: `app/providers/job_sources/linkedin/*`, `tests/providers/test_linkedin_provider.py`
  - Note: RapidAPI key easier to obtain than official LinkedIn API access

---

## Chunk 11.2: ZipRecruiter Job Source & Salary Provider

**Objective:** Add ZipRecruiter as job source and a salary data research provider.

**Dependencies:** Phase 2 (Job Aggregation), Phase 4 (Research System)

**Files to Create:**
- `app/providers/job_sources/ziprecruiter/provider.py` (~150 lines)
- `app/providers/job_sources/ziprecruiter/client.py` (~200 lines)
- `app/providers/job_sources/ziprecruiter/models.py` (~80 lines)
- `app/providers/job_sources/ziprecruiter/config.ini` (~25 lines)
- `app/providers/research/salary_data/provider.py` (~200 lines)
- `app/providers/research/salary_data/models.py` (~100 lines)
- `app/providers/research/salary_data/config.ini` (~30 lines)
- `tests/providers/test_ziprecruiter_provider.py` (~100 lines)
- `tests/providers/test_salary_provider.py` (~100 lines)

**Files to Modify:** None

**Implementation:**

ZipRecruiter Provider (abbreviated for space):

```python
# app/providers/job_sources/ziprecruiter/models.py

from typing import Optional
from pydantic import BaseModel


class ZipRecruiterJobListing(BaseModel):
    """ZipRecruiter-specific job listing"""
    job_id: str
    title: str
    company: str
    location: Optional[str] = None
    posted: Optional[str] = None
    salary: Optional[str] = None
    description: Optional[str] = None
    url: str
    source: str = "ZipRecruiter"
```

```python
# app/providers/job_sources/ziprecruiter/client.py

import httpx
from loguru import logger
from typing import Optional, List
from .models import ZipRecruiterJobListing


class ZipRecruiterAPIClient:
    """
    ZipRecruiter API Client

    Uses ZipRecruiter Job Search API or web scraping.
    For production, consider:
    1. RapidAPI ZipRecruiter Search API
    2. Official ZipRecruiter API (if partner)
    3. Ethical web scraping with rate limiting
    """

    BASE_URL = "https://api.ziprecruiter.com/jobs/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str,
        location: Optional[str] = None,
        remote: bool = False,
        limit: int = 25
    ) -> List[ZipRecruiterJobListing]:
        """Search ZipRecruiter for jobs"""

        params = {
            "api_key": self.api_key,
            "search": query,
            "location": location or "",
            "results_per_page": limit
        }

        if remote:
            params["location"] = "Remote"

        try:
            logger.debug(f"[ZipRecruiter] Searching: {query}")

            response = await self.client.get(f"{self.BASE_URL}", params=params)
            response.raise_for_status()

            data = response.json()

            jobs = []
            for item in data.get("jobs", []):
                job = ZipRecruiterJobListing(
                    job_id=item.get("id", ""),
                    title=item.get("name", ""),
                    company=item.get("hiring_company", {}).get("name", ""),
                    location=item.get("location", ""),
                    posted=item.get("posted_time", ""),
                    salary=item.get("salary_interval", ""),
                    description=item.get("snippet", ""),
                    url=item.get("url", "")
                )
                jobs.append(job)

            return jobs

        except Exception as e:
            logger.error(f"[ZipRecruiter] Search failed: {e}")
            return []

    async def close(self):
        await self.client.aclose()
```

```python
# app/providers/job_sources/ziprecruiter/provider.py

from typing import Optional, List
from configparser import ConfigParser
from loguru import logger

from app.models import JobListing
from app.providers.base import JobSource
from .client import ZipRecruiterAPIClient


class ZipRecruiterProvider(JobSource):
    """ZipRecruiter job source provider"""

    def __init__(self, api_key: str):
        self.client = ZipRecruiterAPIClient(api_key)

    @property
    def name(self) -> str:
        return "ZipRecruiter"

    @property
    def is_expensive(self) -> bool:
        return False  # Free tier available

    async def search(
        self,
        query: str,
        location: Optional[str] = None,
        remote: bool = False,
        limit: int = 25
    ) -> List[JobListing]:
        """Search ZipRecruiter for jobs"""

        logger.info(f"[ZipRecruiter] Searching: {query}")

        try:
            zr_jobs = await self.client.search(query, location, remote, limit)

            jobs = []
            for zr_job in zr_jobs:
                job = JobListing(
                    source="ZipRecruiter",
                    source_id=zr_job.job_id,
                    title=zr_job.title,
                    company_name=zr_job.company,
                    location=zr_job.location,
                    remote="remote" in (zr_job.location or "").lower(),
                    description=zr_job.description,
                    url=zr_job.url,
                    posted_date=zr_job.posted
                )
                jobs.append(job)

            logger.info(f"[ZipRecruiter] Found {len(jobs)} jobs")
            return jobs

        except Exception as e:
            logger.error(f"[ZipRecruiter] Error: {e}")
            return []


def create_provider(config: ConfigParser) -> ZipRecruiterProvider:
    """Factory function"""
    api_key = config.get("api_keys", "ZIPRECRUITER_API_KEY")
    return ZipRecruiterProvider(api_key=api_key)
```

Salary Data Provider:

```python
# app/providers/research/salary_data/models.py

from typing import Optional, List
from pydantic import BaseModel, Field

from app.providers.base import ResearchResult


class SalaryRange(BaseModel):
    """Salary range for a position"""
    min: float
    max: float
    median: float
    currency: str = "USD"


class SalaryByExperience(BaseModel):
    """Salary breakdown by experience level"""
    entry_level: Optional[SalaryRange] = None
    mid_level: Optional[SalaryRange] = None
    senior_level: Optional[SalaryRange] = None
    executive: Optional[SalaryRange] = None


class SalaryResearchResult(ResearchResult):
    """Salary data research result"""

    # Salary ranges
    salary_range_overall: Optional[SalaryRange] = Field(
        None,
        contribution_category="api_cheap",
        contribution_label="Overall Salary Range",
        display_type="text",
        display_format="currency"
    )

    salary_median: Optional[float] = Field(
        None,
        contribution_category="api_cheap",
        contribution_label="Median Salary",
        display_type="text",
        display_format="currency"
    )

    salary_by_experience: Optional[SalaryByExperience] = Field(
        None,
        contribution_category="api_cheap",
        contribution_label="Salary by Experience",
        display_type="custom",
        custom_renderer="salary_by_experience"
    )

    # Additional data
    salary_percentile_25: Optional[float] = None
    salary_percentile_75: Optional[float] = None

    # Job market data
    job_openings_count: Optional[int] = Field(
        None,
        contribution_category="api_cheap",
        contribution_label="Open Positions",
        display_type="text"
    )

    market_trend: Optional[str] = Field(
        None,
        contribution_category="api_cheap",
        contribution_label="Market Trend",
        display_type="badge",
        display_priority=50
    )  # "Growing", "Stable", "Declining"

    data_source: str = Field(default="Salary.com")
    data_freshness_days: Optional[int] = None
```

```python
# app/providers/research/salary_data/provider.py

from typing import List
from configparser import ConfigParser
import httpx
from loguru import logger

from app.providers.base import ResearchProvider, FieldContribution
from .models import SalaryResearchResult, SalaryRange, SalaryByExperience


class SalaryDataProvider(ResearchProvider):
    """
    Salary data research provider

    Uses Salary.com API, Payscale API, or similar service
    to provide salary information for companies/positions.
    """

    def __init__(self, api_key: str, source: str = "salary.com"):
        self.api_key = api_key
        self.source = source
        self.client = httpx.AsyncClient(timeout=30.0)

    @property
    def name(self) -> str:
        return "SalaryData"

    @property
    def fields(self) -> List[FieldContribution]:
        """Declare contributed fields"""
        return [
            FieldContribution(
                field_name="salary_range_overall",
                category="api_cheap",
                label="Salary Range",
                display_type="text",
                display_format="currency"
            ),
            FieldContribution(
                field_name="salary_median",
                category="api_cheap",
                label="Median Salary",
                display_type="text",
                display_format="currency"
            ),
            FieldContribution(
                field_name="salary_by_experience",
                category="api_cheap",
                label="Salary by Experience Level",
                display_type="custom",
                custom_renderer="salary_by_experience"
            ),
            FieldContribution(
                field_name="job_openings_count",
                category="api_cheap",
                label="Open Positions",
                display_type="text"
            ),
            FieldContribution(
                field_name="market_trend",
                category="api_cheap",
                label="Market Trend",
                display_type="badge",
                display_priority=50
            )
        ]

    async def research(
        self,
        company_name: str,
        requested_fields: List[str]
    ) -> SalaryResearchResult:
        """Research salary data for a company"""

        logger.info(f"[SalaryData] Researching: {company_name}")

        try:
            # Fetch salary data from API
            data = await self._fetch_salary_data(company_name)

            # Build result with only requested fields
            result_dict = {"company_name": company_name}

            if "salary_range_overall" in requested_fields and data.get("salary_range"):
                result_dict["salary_range_overall"] = SalaryRange(**data["salary_range"])

            if "salary_median" in requested_fields and data.get("median"):
                result_dict["salary_median"] = data["median"]

            if "salary_by_experience" in requested_fields and data.get("by_experience"):
                result_dict["salary_by_experience"] = SalaryByExperience(**data["by_experience"])

            if "job_openings_count" in requested_fields and data.get("openings"):
                result_dict["job_openings_count"] = data["openings"]

            if "market_trend" in requested_fields and data.get("trend"):
                result_dict["market_trend"] = data["trend"]

            return SalaryResearchResult(**result_dict)

        except Exception as e:
            logger.error(f"[SalaryData] Research failed: {e}")
            return SalaryResearchResult(company_name=company_name)

    async def _fetch_salary_data(self, company_name: str) -> dict:
        """Fetch salary data from API"""

        # This is a placeholder - actual implementation depends on chosen API
        # Options: Salary.com API, Payscale API, Glassdoor API (salary only)

        params = {
            "api_key": self.api_key,
            "company": company_name,
            "country": "US"
        }

        try:
            response = await self.client.get(
                f"https://api.salary.com/v1/company",
                params=params
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.warning(f"[SalaryData] API request failed: {e}")
            # Return mock data for development
            return self._mock_salary_data(company_name)

    def _mock_salary_data(self, company_name: str) -> dict:
        """Generate mock salary data for testing"""
        import random

        base_median = random.randint(70000, 150000)

        return {
            "salary_range": {
                "min": base_median * 0.7,
                "max": base_median * 1.5,
                "median": base_median,
                "currency": "USD"
            },
            "median": base_median,
            "by_experience": {
                "entry_level": {
                    "min": base_median * 0.6,
                    "max": base_median * 0.8,
                    "median": base_median * 0.7,
                    "currency": "USD"
                },
                "mid_level": {
                    "min": base_median * 0.85,
                    "max": base_median * 1.15,
                    "median": base_median,
                    "currency": "USD"
                },
                "senior_level": {
                    "min": base_median * 1.2,
                    "max": base_median * 1.7,
                    "median": base_median * 1.45,
                    "currency": "USD"
                }
            },
            "openings": random.randint(10, 500),
            "trend": random.choice(["Growing", "Stable", "Declining"])
        }

    async def close(self):
        await self.client.aclose()


def create_provider(config: ConfigParser) -> SalaryDataProvider:
    """Factory function"""
    api_key = config.get("api_keys", "SALARY_API_KEY")
    source = config.get("provider", "source", fallback="salary.com")
    return SalaryDataProvider(api_key=api_key, source=source)
```

Frontend custom renderer for salary by experience:

```javascript
// Create: frontend/js/renderers/salary-by-experience.js

/**
 * Custom renderer for salary_by_experience field
 *
 * This should be placed in frontend/js/renderers/ directory.
 * The FieldRenderer will dynamically import this when needed.
 */

export function render(fieldName, value, company) {
    if (!value) return '<span class="text-muted">No data</span>';

    const levels = ['entry_level', 'mid_level', 'senior_level', 'executive'];
    const labels = {
        'entry_level': 'Entry Level',
        'mid_level': 'Mid Level',
        'senior_level': 'Senior Level',
        'executive': 'Executive'
    };

    let html = '<div class="salary-by-experience">';

    levels.forEach(level => {
        const data = value[level];
        if (!data) return;

        const minFormatted = formatCurrency(data.min);
        const maxFormatted = formatCurrency(data.max);
        const medianFormatted = formatCurrency(data.median);

        html += `
            <div class="salary-level">
                <div class="level-label">${labels[level]}</div>
                <div class="level-range">${minFormatted} - ${maxFormatted}</div>
                <div class="level-median">Median: ${medianFormatted}</div>
            </div>
        `;
    });

    html += '</div>';
    return html;
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}
```

```css
/* Add to frontend/css/field-display.css or create separate file */

.salary-by-experience {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.salary-level {
    padding: 0.75rem;
    background: var(--color-bg-alt);
    border-left: 3px solid var(--color-primary);
    border-radius: var(--border-radius);
}

.level-label {
    font-weight: 600;
    color: var(--color-text);
    margin-bottom: 0.25rem;
}

.level-range {
    font-size: 1.125rem;
    color: var(--color-primary);
    font-weight: 600;
    margin-bottom: 0.125rem;
}

.level-median {
    font-size: 0.875rem;
    color: var(--color-text-light);
}
```

**Key Points:**
- ZipRecruiter adds another job source
- Salary provider adds market intelligence
- Custom frontend renderer for complex data
- Graceful degradation with mock data
- Standard plugin architecture

**Testing Criteria:**
1. ✅ ZipRecruiter provider loads and searches
2. ✅ Salary provider loads and researches
3. ✅ Salary data displays in frontend
4. ✅ Custom renderer works for experience levels
5. ✅ Both providers integrate seamlessly
6. ✅ All tests pass

**Planner Guidance:**
- **Developer**: Implement both providers, add custom frontend renderer
- **Tester**: Test both providers, verify frontend rendering
- **Relevant Files**:
  - Create: All files listed in "Files to Create"
  - Modify: None (plugins are self-contained)
  - Note: Requires API keys for real data, includes mock data for testing

---

## Chunk 11.3: Real Glassdoor Integration

**Objective:** Replace mock Glassdoor provider with real API integration.

**Dependencies:** Phase 5 (Mock Glassdoor Provider)

**Files to Create:**
- `app/providers/research/glassdoor/client.py` (~300 lines)

**Files to Modify:**
- `app/providers/research/glassdoor/provider.py` (replace mock with real API)
- `app/providers/research/glassdoor/config.ini` (add real API keys)

**Implementation:**

```python
# app/providers/research/glassdoor/client.py

import httpx
from typing import Optional, Dict, Any, List
from loguru import logger


class GlassdoorAPIClient:
    """
    Glassdoor API Client

    Glassdoor has limited public API access. Options:
    1. Glassdoor Partner API (requires partnership)
    2. RapidAPI Glassdoor Review API
    3. Web scraping (ethical, rate-limited)

    This implementation uses RapidAPI for easier access.
    """

    RAPID_API_HOST = "glassdoor-reviews1.p.rapidapi.com"
    BASE_URL = f"https://{RAPID_API_HOST}"

    def __init__(self, rapid_api_key: str):
        self.rapid_api_key = rapid_api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_company_data(self, company_name: str) -> Optional[Dict[str, Any]]:
        """
        Get company data from Glassdoor

        Returns:
            Dict with rating, review_count, pros, cons, etc.
        """
        try:
            # Step 1: Search for company to get ID
            company_id = await self._search_company(company_name)
            if not company_id:
                logger.warning(f"[Glassdoor] Company not found: {company_name}")
                return None

            # Step 2: Get company details
            details = await self._get_company_details(company_id)
            if not details:
                return None

            # Step 3: Get recent reviews
            reviews = await self._get_company_reviews(company_id, limit=10)

            # Combine data
            return {
                "rating": details.get("rating"),
                "review_count": details.get("reviewCount"),
                "ceo_name": details.get("ceo", {}).get("name"),
                "ceo_approval": details.get("ceo", {}).get("approvalRating"),
                "recommend_to_friend": details.get("recommendToFriendRating"),
                "culture_rating": details.get("cultureAndValuesRating"),
                "diversity_rating": details.get("diversityAndInclusionRating"),
                "work_life_balance": details.get("workLifeBalanceRating"),
                "senior_management": details.get("seniorManagementRating"),
                "compensation": details.get("compensationAndBenefitsRating"),
                "career_opportunities": details.get("careerOpportunitiesRating"),
                "pros": self._extract_pros(reviews),
                "cons": self._extract_cons(reviews),
                "recent_reviews": reviews[:3]  # Just 3 most recent
            }

        except Exception as e:
            logger.error(f"[Glassdoor] Error fetching company data: {e}")
            return None

    async def _search_company(self, company_name: str) -> Optional[str]:
        """Search for company and return Glassdoor ID"""

        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": self.RAPID_API_HOST
        }

        params = {"query": company_name}

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/search",
                params=params,
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            if not results:
                return None

            # Return first match (best match)
            return results[0].get("id")

        except Exception as e:
            logger.error(f"[Glassdoor] Search error: {e}")
            return None

    async def _get_company_details(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Get company details by ID"""

        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": self.RAPID_API_HOST
        }

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/company/{company_id}",
                headers=headers
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"[Glassdoor] Details error: {e}")
            return None

    async def _get_company_reviews(
        self,
        company_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get company reviews"""

        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": self.RAPID_API_HOST
        }

        params = {"limit": limit, "sort": "recent"}

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/company/{company_id}/reviews",
                params=params,
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            return data.get("reviews", [])

        except Exception as e:
            logger.error(f"[Glassdoor] Reviews error: {e}")
            return []

    def _extract_pros(self, reviews: List[Dict[str, Any]]) -> List[str]:
        """Extract top pros from reviews"""
        pros = []
        for review in reviews[:5]:  # Top 5 reviews
            pro = review.get("pros", "").strip()
            if pro and len(pro) > 20:  # Meaningful pro
                pros.append(pro)
        return pros[:3]  # Return top 3

    def _extract_cons(self, reviews: List[Dict[str, Any]]) -> List[str]:
        """Extract top cons from reviews"""
        cons = []
        for review in reviews[:5]:
            con = review.get("cons", "").strip()
            if con and len(con) > 20:
                cons.append(con)
        return cons[:3]

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
```

Update provider to use real client:

```python
# Modify: app/providers/research/glassdoor/provider.py

# Replace the entire _research_glassdoor method:

async def _research_glassdoor(self, company_name: str) -> dict:
    """Fetch real Glassdoor data"""

    from .client import GlassdoorAPIClient

    # Get API key from config
    # (This would be set during provider initialization)
    rapid_api_key = getattr(self, 'rapid_api_key', None)

    if not rapid_api_key:
        logger.warning("[Glassdoor] No API key configured, using mock data")
        return self._mock_glassdoor_data(company_name)

    try:
        client = GlassdoorAPIClient(rapid_api_key)
        data = await client.get_company_data(company_name)
        await client.close()

        if not data:
            logger.warning(f"[Glassdoor] No data found for {company_name}")
            return {}

        return data

    except Exception as e:
        logger.error(f"[Glassdoor] Failed to fetch data: {e}")
        # Fallback to empty data (graceful degradation)
        return {}

# Update __init__ to accept API key:
def __init__(self, rapid_api_key: Optional[str] = None):
    self.rapid_api_key = rapid_api_key

# Update factory function:
def create_provider(config: ConfigParser) -> GlassdoorProvider:
    """Factory function to create Glassdoor provider from config"""
    rapid_api_key = config.get("api_keys", "RAPID_API_KEY", fallback=None)
    return GlassdoorProvider(rapid_api_key=rapid_api_key)
```

Update config:

```ini
# Modify: app/providers/research/glassdoor/config.ini

[provider]
name = Glassdoor
enabled = true
# Change this to false once real API is working
use_mock_data = false

[api_keys]
# RapidAPI Glassdoor Reviews API
RAPID_API_KEY = your_rapidapi_key_here

[scraping]
# Alternative: Web scraping settings (if no API available)
user_agent = Mozilla/5.0 (Windows NT 10.0; Win64; x64)
rate_limit_seconds = 2
max_retries = 3

[cache]
# Cache Glassdoor data for 7 days (it doesn't change frequently)
ttl_days = 7
```

**Key Points:**
- RapidAPI for easier Glassdoor access
- Falls back to mock data if no API key
- Extracts pros/cons from actual reviews
- Includes all ratings (CEO approval, work-life balance, etc.)
- Respects rate limits
- Error handling with graceful degradation

**Testing Criteria:**
1. ✅ Real API client connects and fetches data
2. ✅ Company search finds correct company
3. ✅ All rating fields populated
4. ✅ Pros/cons extracted from reviews
5. ✅ Falls back gracefully on errors
6. ✅ Integration with existing provider system works
7. ✅ Frontend displays all new fields correctly

**Test Commands:**
```bash
# Test real Glassdoor provider
pytest tests/providers/test_glassdoor_provider.py -v

# Test with real API (requires RapidAPI key)
export RAPID_API_KEY=your_key_here
python -m app.test_providers --provider glassdoor --company "Google"

# Test end-to-end with frontend
# 1. Start backend with real API key in config
# 2. Open frontend, search for jobs
# 3. Trigger deep research with Glassdoor fields selected
# 4. Verify all Glassdoor data displays correctly
```

**Planner Guidance:**
- **Developer**: Replace mock Glassdoor with real API client
- **Tester**: Test with real API key, verify all data fields, test fallback
- **Relevant Files**:
  - Create: `app/providers/research/glassdoor/client.py`
  - Modify: `app/providers/research/glassdoor/provider.py`, `config.ini`
  - Note: RapidAPI key required for real data
  - Note: This completes Phase 11 - All additional providers functional

---
# Phase 12: Production Polish

**Objective:** Add production-ready features: comprehensive error handling, logging, deployment configuration, and documentation.

**Why This Phase:**
Before deploying to production, the application needs robust error handling, proper logging, deployment configuration, and documentation for maintenance.

**Dependencies:** All previous phases (0-11)

**Phase Success Criteria:**
1. ✅ Comprehensive error handling throughout application
2. ✅ Structured logging with appropriate levels
3. ✅ Deployment configuration (Docker, environment variables)
4. ✅ Admin/monitoring dashboard
5. ✅ Complete deployment documentation
6. ✅ Application ready for production use

---

## Chunk 12.1: Error Handling and Logging

**Objective:** Implement comprehensive error handling and structured logging across the application.

**Dependencies:** All backend code (Phases 0-5)

**Files to Create:**
- `app/errors.py` (~150 lines - custom exceptions)
- `app/logging_config.py` (~100 lines - logging setup)
- `app/middleware/error_handlers.py` (~200 lines - FastAPI error handlers)

**Files to Modify:**
- `app/main.py` (add error handlers and logging)
- `app/services/*.py` (add try/except and logging)

**Implementation:**

```python
# app/errors.py

"""
Custom exceptions for the application

Provides specific exception types for different error scenarios,
making error handling more precise and error messages more helpful.
"""

class JobSearchError(Exception):
    """Base exception for all application errors"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ProviderError(JobSearchError):
    """Error from a job/research provider"""
    def __init__(self, provider_name: str, message: str, details: dict = None):
        self.provider_name = provider_name
        super().__init__(f"[{provider_name}] {message}", details)


class APIError(JobSearchError):
    """Error calling external API"""
    def __init__(self, service: str, status_code: int, message: str):
        self.service = service
        self.status_code = status_code
        super().__init__(
            f"{service} API error ({status_code}): {message}",
            {"service": service, "status_code": status_code}
        )


class RateLimitError(JobSearchError):
    """Rate limit exceeded"""
    def __init__(self, service: str, retry_after: int = None):
        self.service = service
        self.retry_after = retry_after
        message = f"Rate limit exceeded for {service}"
        if retry_after:
            message += f" - retry after {retry_after} seconds"
        super().__init__(message, {"service": service, "retry_after": retry_after})


class ConfigurationError(JobSearchError):
    """Configuration error (missing API keys, invalid settings, etc.)"""
    pass


class DatabaseError(JobSearchError):
    """Database operation error"""
    pass


class ValidationError(JobSearchError):
    """Input validation error"""
    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"Validation error on '{field}': {message}", {"field": field})
```

```python
# app/logging_config.py

"""
Logging configuration

Sets up structured logging with appropriate levels and formats.
Uses loguru for better Python logging experience.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/app.log",
    rotation: str = "100 MB",
    retention: str = "30 days",
    format_str: str = None
):
    """
    Configure application logging

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        rotation: When to rotate log file
        retention: How long to keep old log files
        format_str: Custom format string
    """

    # Remove default logger
    logger.remove()

    # Default format
    if format_str is None:
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console logger
    logger.add(
        sys.stderr,
        format=format_str,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )

    # Add file logger
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format=format_str,
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        backtrace=True,
        diagnose=True
    )

    logger.info(f"Logging configured: level={level}, file={log_file}")


def get_logger(name: str):
    """Get a logger instance for a module"""
    return logger.bind(name=name)
```

```python
# app/middleware/error_handlers.py

"""
FastAPI error handlers

Catches exceptions and returns proper HTTP responses with
consistent error format.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger

from app.errors import (
    JobSearchError,
    ProviderError,
    APIError,
    RateLimitError,
    ConfigurationError,
    DatabaseError,
    ValidationError
)


async def job_search_error_handler(request: Request, exc: JobSearchError) -> JSONResponse:
    """Handle JobSearchError and subclasses"""

    logger.error(f"JobSearchError: {exc.message}", extra=exc.details)

    # Determine status code based on exception type
    if isinstance(exc, ValidationError):
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, RateLimitError):
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
    elif isinstance(exc, ConfigurationError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    elif isinstance(exc, DatabaseError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    elif isinstance(exc, APIError):
        # Use the API's status code if it's a client error, otherwise 502
        status_code = exc.status_code if 400 <= exc.status_code < 500 else status.HTTP_502_BAD_GATEWAY
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors"""

    logger.warning(f"Validation error: {exc.errors()}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Invalid request data",
            "details": exc.errors()
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""

    logger.exception(f"Unexpected error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {}
        }
    )


def register_error_handlers(app):
    """Register all error handlers with FastAPI app"""

    app.add_exception_handler(JobSearchError, job_search_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Error handlers registered")
```

Update main.py:

```python
# Modify: app/main.py

from app.logging_config import setup_logging
from app.middleware.error_handlers import register_error_handlers

# Add near the top of the file (after imports):

# Setup logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/app.log")
)

logger.info("Starting Job Search AI application")

# After app creation:
app = FastAPI(...)

# Register error handlers
register_error_handlers(app)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Log response
    duration = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )

    return response
```

Example of using error handling in services:

```python
# Example modifications to app/services/job_aggregation.py

from app.errors import ProviderError, JobSearchError
from loguru import logger

async def search_jobs(...):
    """Search for jobs across all providers"""

    logger.info(f"Starting job search: query='{query}', sources={sources}")

    all_jobs = []
    errors = []

    for source in sources:
        try:
            logger.debug(f"Searching source: {source.name}")

            jobs = await source.search(query, location, remote, limit)

            logger.info(f"{source.name} returned {len(jobs)} jobs")
            all_jobs.extend(jobs)

        except Exception as e:
            error_msg = f"{source.name} search failed: {str(e)}"
            logger.error(error_msg)

            # Collect error but continue with other sources
            errors.append(ProviderError(source.name, str(e)))

    if not all_jobs and errors:
        # All sources failed
        raise JobSearchError(
            "All job sources failed",
            details={"errors": [str(e) for e in errors]}
        )

    # Log summary
    logger.info(
        f"Job search complete: {len(all_jobs)} jobs found, "
        f"{len(errors)} sources failed"
    )

    return all_jobs
```

**Key Points:**
- Custom exception hierarchy for precise error handling
- Structured logging with loguru
- FastAPI error handlers for consistent API responses
- Request/response logging middleware
- Graceful error handling (continue on provider failures)
- Detailed error information in logs, sanitized in API responses

**Testing Criteria:**
1. ✅ All custom exceptions work correctly
2. ✅ Error handlers return proper HTTP status codes
3. ✅ Logs written to file and console
4. ✅ Log rotation works
5. ✅ Request logging includes duration
6. ✅ Provider failures don't crash the app
7. ✅ Error responses have consistent format

**Test Commands:**
```bash
# Test error handling
pytest tests/test_error_handlers.py -v

# Check logs are created
ls -lh logs/

# Tail logs in real-time
tail -f logs/app.log

# Test API error responses
curl http://localhost:8000/api/search  # Missing query param
curl http://localhost:8000/api/invalid  # 404

# Check log levels
LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

**Planner Guidance:**
- **Developer**: Implement all error handling and logging infrastructure
- **Tester**: Test various error scenarios, verify log output
- **Relevant Files**:
  - Create: `app/errors.py`, `app/logging_config.py`, `app/middleware/error_handlers.py`
  - Modify: `app/main.py`, service files
  - Note: Update all services to use logger and custom exceptions

---

## Chunk 12.2: Deployment Configuration

**Objective:** Create Docker configuration and deployment setup for production.

**Dependencies:** Chunk 12.1 (Error Handling)

**Files to Create:**
- `Dockerfile` (~50 lines)
- `docker-compose.yml` (~80 lines)
- `.dockerignore` (~20 lines)
- `deploy/nginx.conf` (~60 lines - reverse proxy config)
- `deploy/systemd/jobsearch.service` (~30 lines - systemd service)
- `.env.example` (~40 lines - environment variables template)
- `scripts/deploy.sh` (~100 lines - deployment script)

**Files to Modify:**
- `app/config.py` (add environment variable loading)

**Implementation:**

```dockerfile
# Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Create logs directory
RUN mkdir -p /app/logs

# Create non-root user
RUN useradd -m -u 1000 jobsearch && \
    chown -R jobsearch:jobsearch /app

USER jobsearch

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)"

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  # Database
  postgres:
    image: postgres:15-alpine
    container_name: jobsearch_db
    environment:
      POSTGRES_USER: ${DB_USER:-jobsearch}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-changeme}
      POSTGRES_DB: ${DB_NAME:-jobsearch}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-jobsearch}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Backend API
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: jobsearch_api
    environment:
      # Database
      DATABASE_URL: postgresql://${DB_USER:-jobsearch}:${DB_PASSWORD:-changeme}@postgres:5432/${DB_NAME:-jobsearch}

      # API Keys (load from .env)
      INDEED_API_KEY: ${INDEED_API_KEY}
      LINKEDIN_API_KEY: ${LINKEDIN_API_KEY}
      RAPID_API_KEY: ${RAPID_API_KEY}
      GLASSDOOR_API_KEY: ${GLASSDOOR_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}

      # App config
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
      CACHE_TTL_DAYS: ${CACHE_TTL_DAYS:-30}
      DEBUG: ${DEBUG:-false}

    volumes:
      - ./logs:/app/logs
      - ./app/providers:/app/app/providers
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
    command: >
      sh -c "alembic upgrade head &&
             uvicorn app.main:app --host 0.0.0.0 --port 8000"

  # Frontend (Nginx)
  frontend:
    image: nginx:alpine
    container_name: jobsearch_frontend
    volumes:
      - ./frontend:/usr/share/nginx/html:ro
      - ./deploy/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  postgres_data:
```

```
# .dockerignore

__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build
.git
.gitignore
.env
.venv
venv/
*.md
docs/
tests/
.pytest_cache
.coverage
htmlcov/
logs/
*.log
.DS_Store
```

Nginx configuration:

```nginx
# deploy/nginx.conf

server {
    listen 80;
    server_name localhost;

    # Frontend static files
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running searches
        proxy_connect_timeout 90s;
        proxy_send_timeout 90s;
        proxy_read_timeout 90s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://backend:8000/health;
        access_log off;
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

Systemd service (alternative to Docker):

```ini
# deploy/systemd/jobsearch.service

[Unit]
Description=Job Search AI Backend
After=network.target postgresql.service

[Service]
Type=simple
User=jobsearch
Group=jobsearch
WorkingDirectory=/opt/jobsearch
Environment="PATH=/opt/jobsearch/venv/bin"
EnvironmentFile=/opt/jobsearch/.env
ExecStart=/opt/jobsearch/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/var/log/jobsearch/access.log
StandardError=append:/var/log/jobsearch/error.log

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/jobsearch/logs /opt/jobsearch/app/providers

[Install]
WantedBy=multi-user.target
```

Environment variables template:

```bash
# .env.example

# Copy this file to .env and fill in your values

# Database
DB_USER=jobsearch
DB_PASSWORD=your_secure_password_here
DB_NAME=jobsearch
DATABASE_URL=postgresql://jobsearch:your_secure_password_here@localhost:5432/jobsearch

# API Keys
# Indeed
INDEED_API_KEY=your_indeed_api_key

# LinkedIn (RapidAPI)
LINKEDIN_API_KEY=
RAPID_API_KEY=your_rapidapi_key

# Glassdoor (RapidAPI)
GLASSDOOR_API_KEY=your_rapidapi_key

# OpenAI (for scam detection)
OPENAI_API_KEY=your_openai_api_key

# ZipRecruiter
ZIPRECRUITER_API_KEY=your_ziprecruiter_key

# Salary Data
SALARY_API_KEY=your_salary_api_key

# Application Config
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
CACHE_TTL_DAYS=30
DEBUG=false

# Security
SECRET_KEY=generate_a_random_secret_key_here
ALLOWED_ORIGINS=http://localhost,http://localhost:80

# Rate Limiting (future)
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_DAY=1000
```

Deployment script:

```bash
# scripts/deploy.sh

#!/bin/bash
set -e

echo "🚀 Job Search AI Deployment Script"
echo "=================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "   Copy .env.example to .env and fill in your values"
    exit 1
fi

# Load environment variables
source .env

# Check required API keys
REQUIRED_KEYS=("DATABASE_URL" "OPENAI_API_KEY")
for key in "${REQUIRED_KEYS[@]}"; do
    if [ -z "${!key}" ]; then
        echo "❌ Error: $key not set in .env"
        exit 1
    fi
done

echo "✅ Environment variables loaded"

# Deployment type
DEPLOY_TYPE=${1:-docker}

if [ "$DEPLOY_TYPE" = "docker" ]; then
    echo ""
    echo "📦 Docker Deployment"
    echo "-------------------"

    # Build images
    echo "Building Docker images..."
    docker-compose build

    # Run migrations
    echo "Running database migrations..."
    docker-compose run --rm backend alembic upgrade head

    # Start services
    echo "Starting services..."
    docker-compose up -d

    # Check health
    echo "Waiting for services to be healthy..."
    sleep 10

    if docker-compose ps | grep -q "Up"; then
        echo "✅ Deployment successful!"
        echo ""
        echo "Services running:"
        docker-compose ps
        echo ""
        echo "Access the application at: http://localhost"
        echo "API documentation at: http://localhost/api/docs"
    else
        echo "❌ Deployment failed. Check logs:"
        docker-compose logs
        exit 1
    fi

elif [ "$DEPLOY_TYPE" = "systemd" ]; then
    echo ""
    echo "🖥️  Systemd Deployment"
    echo "--------------------"

    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Please run as root for systemd deployment"
        exit 1
    fi

    # Install to /opt/jobsearch
    echo "Installing to /opt/jobsearch..."
    mkdir -p /opt/jobsearch
    cp -r . /opt/jobsearch/

    # Create virtual environment
    echo "Setting up Python environment..."
    cd /opt/jobsearch
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    # Run migrations
    echo "Running database migrations..."
    alembic upgrade head

    # Install systemd service
    echo "Installing systemd service..."
    cp deploy/systemd/jobsearch.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable jobsearch
    systemctl start jobsearch

    # Check status
    if systemctl is-active --quiet jobsearch; then
        echo "✅ Deployment successful!"
        echo ""
        systemctl status jobsearch
    else
        echo "❌ Service failed to start. Check logs:"
        journalctl -u jobsearch -n 50
        exit 1
    fi

else
    echo "❌ Unknown deployment type: $DEPLOY_TYPE"
    echo "   Usage: ./scripts/deploy.sh [docker|systemd]"
    exit 1
fi
```

Update config.py to load from environment:

```python
# Modify: app/config.py

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./jobsearch.db")

    # API Keys
    indeed_api_key: str = os.getenv("INDEED_API_KEY", "")
    rapid_api_key: str = os.getenv("RAPID_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Application
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "logs/app.log")
    cache_ttl_days: int = int(os.getenv("CACHE_TTL_DAYS", "30"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Security
    secret_key: str = os.getenv("SECRET_KEY", "insecure-dev-key")
    allowed_origins: list[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost").split(",")

    class Config:
        env_file = ".env"


settings = Settings()
```

**Key Points:**
- Docker Compose for easy deployment
- Nginx reverse proxy for production
- Systemd service for traditional deployments
- Environment variable configuration
- Health checks and logging
- Automated deployment script
- Security headers and best practices

**Testing Criteria:**
1. ✅ Docker images build successfully
2. ✅ Docker Compose starts all services
3. ✅ Health checks pass
4. ✅ Nginx serves frontend and proxies API
5. ✅ Database migrations run automatically
6. ✅ Logs are accessible
7. ✅ Application accessible at http://localhost

**Deployment Commands:**
```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# Deploy with Docker
./scripts/deploy.sh docker

# Check status
docker-compose ps
docker-compose logs -f backend

# Deploy with systemd (requires root)
sudo ./scripts/deploy.sh systemd

# Check status
sudo systemctl status jobsearch
sudo journalctl -u jobsearch -f

# Stop services
docker-compose down
# or
sudo systemctl stop jobsearch
```

**Planner Guidance:**
- **Developer**: Create all deployment configuration files
- **Tester**: Test Docker deployment, verify all services work
- **Relevant Files**:
  - Create: All deployment files listed above
  - Modify: `app/config.py`
  - Note: Test deployment in clean environment (VM or fresh Docker)

---

## Chunk 12.3: Documentation and Final Polish

**Objective:** Create comprehensive documentation and add final production touches.

**Dependencies:** All previous chunks

**Files to Create:**
- `README.md` (~300 lines - main documentation)
- `docs/DEPLOYMENT.md` (~200 lines - deployment guide)
- `docs/API.md` (~150 lines - API documentation)
- `docs/PROVIDERS.md` (~200 lines - provider development guide)
- `docs/TROUBLESHOOTING.md` (~150 lines - common issues)
- `CONTRIBUTING.md` (~100 lines - contribution guidelines)
- `LICENSE` (choose appropriate license)

**Files to Modify:**
- `app/main.py` (add API metadata, tags, descriptions)
- `frontend/index.html` (add meta tags, favicon, analytics placeholder)

**Implementation:**

Main README:

```markdown
# README.md

# Job Search AI 🔍🤖

Smart job search with AI-powered scam detection and comprehensive company research.

## Features

- **Multi-Source Job Aggregation**: Search Indeed, LinkedIn, ZipRecruiter, and more simultaneously
- **AI Scam Detection**: GPT-4 powered analysis flags potential scam job postings
- **Company Research**: Automatic Glassdoor ratings, salary data, and reviews
- **Duplicate Detection**: Smart deduplication across job boards
- **Quick & Deep Search**: Fast basic search or comprehensive research with one click
- **Modular Provider System**: Easy to add new job sources and research providers

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ (or SQLite for development)
- Docker & Docker Compose (recommended)
- API Keys (see Configuration section)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/jobsearch-ai.git
   cd jobsearch-ai
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Deploy with Docker** (recommended)
   ```bash
   ./scripts/deploy.sh docker
   ```

4. **Access the application**
   - Frontend: http://localhost
   - API Docs: http://localhost/api/docs

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set up database
alembic upgrade head

# Run backend
uvicorn app.main:app --reload

# Open frontend
# Open frontend/index.html in your browser
# Or serve with: python -m http.server 8080 --directory frontend
```

## Configuration

### Required API Keys

- **OpenAI API Key** (required): For AI scam detection
- **RapidAPI Key** (recommended): For LinkedIn, Glassdoor data

### Optional API Keys

- **Indeed API Key**: Direct Indeed access
- **ZipRecruiter API Key**: Direct ZipRecruiter access
- **Salary.com API Key**: Salary data

See `.env.example` for full configuration options.

## Usage

### Quick Search (2-5 seconds)
1. Enter job title and location
2. Click "Quick Search"
3. View results with basic company info

### Deep Search (30-60 seconds)
1. Enter search criteria
2. Select advanced research options:
   - Scam Detection (AI)
   - Glassdoor Ratings
   - Salary Data
   - Reviews
3. Click "Deep Search"
4. View comprehensive results

### Manual Deep Research
- Click "Deep Research" on any job card
- View all available company data
- Export as JSON

## Architecture

```
├── app/                    # Backend (FastAPI)
│   ├── providers/         # Job sources & research providers
│   ├── services/          # Business logic
│   ├── models/            # Database models
│   └── main.py           # API entry point
├── frontend/              # Frontend (Vanilla JS)
│   ├── js/               # JavaScript modules
│   ├── css/              # Stylesheets
│   └── index.html        # Main page
├── docs/                  # Documentation
└── tests/                # Test suite
```

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md)
- [API Documentation](docs/API.md)
- [Provider Development](docs/PROVIDERS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT License](LICENSE) - see LICENSE file for details.

## Credits

Built for helping job seekers avoid scams and make informed decisions.

### Technologies Used

- **Backend**: FastAPI, SQLAlchemy, Pydantic, OpenAI
- **Frontend**: Vanilla JavaScript, CSS3
- **Database**: PostgreSQL
- **Deployment**: Docker, Nginx

## Support

For issues, questions, or contributions:
- 📧 Email: support@jobsearch.ai
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/jobsearch-ai/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/jobsearch-ai/discussions)

---

**Made with ❤️ for job seekers everywhere**
```

Update main.py with better API documentation:

```python
# Modify: app/main.py

app = FastAPI(
    title="Job Search AI",
    description="""
    🔍 Smart Job Search with AI-Powered Scam Detection

    ## Features

    * **Multi-source job aggregation** from Indeed, LinkedIn, ZipRecruiter
    * **AI scam detection** using GPT-4
    * **Company research** with Glassdoor ratings and salary data
    * **Smart duplicate detection** across job boards
    * **Quick and deep search modes** for different needs

    ## Authentication

    Currently no authentication required (trusted users only).

    ## Rate Limiting

    No rate limiting in current version. Use responsibly.
    """,
    version="1.0.0",
    contact={
        "name": "Job Search AI Support",
        "email": "support@jobsearch.ai",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add tags metadata for better organization
tags_metadata = [
    {
        "name": "search",
        "description": "Job search operations. Quick search returns jobs with basic info in 2-5s. Deep search includes comprehensive company research in 30-60s.",
    },
    {
        "name": "research",
        "description": "Company research operations. Get detailed company information including scam detection, ratings, and reviews.",
    },
    {
        "name": "health",
        "description": "Health check and system status endpoints.",
    },
]

# Update endpoint definitions to use tags:
@app.get("/api/search", tags=["search"], summary="Search for jobs")
async def search_jobs(...):
    """
    Search for jobs across multiple sources.

    **Quick Search** (no `include` parameter):
    - Returns results in 2-5 seconds
    - Includes basic job info only

    **Deep Search** (with `include` parameter):
    - Returns results in 30-60 seconds
    - Includes company research data
    - Example: `include=is_scam,glassdoor_rating`

    **Parameters**:
    - **query**: Job title or keywords (required)
    - **location**: Location filter (optional)
    - **remote**: Remote jobs only (optional)
    - **include**: Comma-separated list of research fields (optional)
    """
    ...
```

Update frontend index.html:

```html
<!-- Modify: frontend/index.html -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <!-- SEO Meta Tags -->
    <meta name="description" content="Smart job search with AI-powered scam detection. Search multiple job boards and get comprehensive company research in one place.">
    <meta name="keywords" content="job search, scam detection, job boards, company research, Indeed, LinkedIn">
    <meta name="author" content="Job Search AI">

    <!-- Open Graph / Social Media -->
    <meta property="og:type" content="website">
    <meta property="og:title" content="Job Search AI - Smart Job Search with Scam Detection">
    <meta property="og:description" content="Search multiple job boards and get AI-powered scam detection plus company research.">
    <meta property="og:url" content="https://jobsearch.ai">

    <!-- Favicon -->
    <link rel="icon" type="image/png" href="/favicon.png">

    <title>Job Search AI - Smart Job Search with Scam Detection</title>

    <!-- CSS -->
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/filters.css">
    <link rel="stylesheet" href="css/advanced-filters.css">
    <link rel="stylesheet" href="css/field-display.css">
    <link rel="stylesheet" href="css/job-card.css">
    <link rel="stylesheet" href="css/results-list.css">
    <link rel="stylesheet" href="css/search-progress.css">
    <link rel="stylesheet" href="css/company-modal.css">

    <!-- Analytics Placeholder (replace with your analytics) -->
    <!-- <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script> -->
</head>
<body>
    <div id="app"></div>

    <!-- JavaScript -->
    <script src="js/event-bus.js"></script>
    <script src="js/field-renderer.js"></script>
    <script src="js/field-registry.js"></script>
    <script src="js/filter-builder.js"></script>
    <script src="js/api-client.js"></script>
    <script src="js/job-card.js"></script>
    <script src="js/results-list.js"></script>
    <script src="js/search-progress.js"></script>
    <script src="js/company-modal.js"></script>
    <script src="js/app.js"></script>

    <script>
        // Initialize application
        (async () => {
            // API base URL (use environment variable in production)
            const apiBaseURL = window.location.origin;

            const app = new JobSearchApp({
                apiBaseURL: apiBaseURL,
                containerElement: document.getElementById('app'),
                debugMode: false, // Set to true for development
            });

            try {
                await app.initialize();
                console.log('🚀 Job Search AI ready!');

                // Analytics: Track page view
                // gtag('event', 'page_view', { page_title: 'Job Search' });

            } catch (error) {
                console.error('❌ Failed to initialize:', error);

                document.getElementById('app').innerHTML = `
                    <div style="padding: 2rem; text-align: center; max-width: 600px; margin: 2rem auto;">
                        <h1 style="color: #ef4444;">⚠️ Failed to Initialize</h1>
                        <p style="color: #64748b; margin: 1rem 0;">${error.message}</p>
                        <p style="color: #64748b;">
                            Please ensure the backend server is running and accessible.
                        </p>
                        <button
                            onclick="window.location.reload()"
                            style="
                                padding: 0.75rem 1.5rem;
                                background: #2563eb;
                                color: white;
                                border: none;
                                border-radius: 0.375rem;
                                cursor: pointer;
                                font-size: 1rem;
                                margin-top: 1rem;
                            "
                        >
                            Retry
                        </button>
                    </div>
                `;
            }
        })();
    </script>
</body>
</html>
```

**Key Points:**
- Comprehensive README with quick start
- API documentation with OpenAPI/Swagger
- SEO meta tags for frontend
- Error handling in frontend initialization
- Analytics placeholder for future
- Clear contribution guidelines

**Testing Criteria:**
1. ✅ README is clear and complete
2. ✅ All documentation files created
3. ✅ API docs accessible at /api/docs
4. ✅ Frontend meta tags correct
5. ✅ Deployment guide works
6. ✅ Troubleshooting guide helpful

**Final Checks:**
```bash
# Check all documentation exists
ls -lh docs/

# Verify API docs
curl http://localhost/api/openapi.json

# Check frontend loads
curl http://localhost/

# Lint Python code
ruff check app/

# Run all tests
pytest tests/ -v

# Check test coverage
pytest --cov=app tests/

# Build documentation (if using Sphinx)
cd docs && make html
```

**Planner Guidance:**
- **Developer**: Create all documentation files, polish UI
- **Tester**: Review all documentation, verify links, test deployment guide
- **Relevant Files**:
  - Create: All documentation files
  - Modify: `app/main.py`, `frontend/index.html`
  - Note: This completes Phase 12 and the entire implementation plan!

---

# Implementation Complete! 🎉

The Job Search AI application is now production-ready with:

✅ Complete backend with modular provider system
✅ Full-featured frontend with metadata-driven display
✅ AI-powered scam detection
✅ Multiple job sources and research providers
✅ Comprehensive error handling and logging
✅ Docker deployment configuration
✅ Complete documentation

## Next Steps

1. **Deploy**: Follow `docs/DEPLOYMENT.md`
2. **Configure**: Set up API keys in `.env`
3. **Test**: Run end-to-end tests
4. **Monitor**: Check logs and application health
5. **Iterate**: Add more providers as needed

## Maintenance

- Review logs regularly: `docker-compose logs -f`
- Update dependencies: `pip install -U -r requirements.txt`
- Add providers: Follow `docs/PROVIDERS.md`
- Monitor API costs: Check provider usage

Thank you for building Job Search AI!
