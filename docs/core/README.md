# Job Search AI - Developer Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Provider System](#provider-system)
6. [Display System](#display-system)
7. [Caching Strategy](#caching-strategy)
8. [Code Patterns](#code-patterns)

---

## System Overview

The Job Search AI application helps users find legitimate job postings while filtering out scam listings. It aggregates jobs from multiple sources (Indeed, LinkedIn, etc.) and performs research on companies using a combination of API calls and AI analysis.

### Core Problem
Users waste time encountering scam job postings during their search. This tool proactively flags potential scams while providing helpful context (reviews, salary data) for legitimate jobs.

### Solution Approach
- **Aggregation:** Pull jobs from multiple sources, deduplicate
- **Research:** Analyze companies using modular providers (APIs + AI)
- **User Control:** Quick search for speed, Deep search for thoroughness
- **Caching:** Aggressive caching to minimize costs and latency

---

## Architecture

### High-Level Stack

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (JS)                       │
│  - Event-driven architecture                            │
│  - Metadata-driven display rendering                    │
│  - localStorage for user preferences                    │
└─────────────────────────────────────────────────────────┘
                         ↕ HTTP/JSON
┌─────────────────────────────────────────────────────────┐
│                  Backend (FastAPI)                      │
│  - RESTful API endpoints                                │
│  - Dependency injection                                 │
│  - Async/await throughout                               │
└─────────────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────────────┐
│               Services Layer                            │
│  - CompanyResearchService                               │
│  - Job aggregation logic                                │
│  - Provider orchestration                               │
└─────────────────────────────────────────────────────────┘
                         ↕
┌──────────────────┬──────────────────┬──────────────────┐
│   Job Sources    │   Research       │   Database       │
│   (Plugins)      │   Providers      │   (SQLAlchemy)   │
│                  │   (Plugins)      │                  │
│  - Indeed        │  - Glassdoor     │  - CompanyCache  │
│  - LinkedIn      │  - AIScamDetector│  - (extensible)  │
│  - ZipRecruiter  │  - SalaryAPI     │                  │
└──────────────────┴──────────────────┴──────────────────┘
```

### Technology Choices

**Backend:**
- **FastAPI:** Modern async framework, auto-generates OpenAPI docs, great DI
- **Pydantic:** Type-safe models, validation, serialization
- **SQLAlchemy:** ORM for database flexibility (SQLite dev, Postgres prod)
- **asyncio:** Parallel API calls, non-blocking I/O

**Frontend:**
- **Vanilla JavaScript:** No framework lock-in, small bundle size
- **Event-driven:** Decoupled components via EventBus pattern
- **Class-based:** OOP structure where it adds clarity

**Configuration:**
- **configparser:** Python built-in, simple INI format
- **Pydantic Settings:** Type-safe config loading with validation

---

## Core Components

### 1. Models

All data structures use Pydantic for type safety and validation.

#### JobListing
```python
class JobListing(BaseModel):
    id: str  # UUID
    source: str  # Required: "Indeed", "LinkedIn", etc.
    title: str  # Required
    company: str  # Required
    description: Optional[str]
    pay: Optional[str]
    location: Optional[str]
    remote: Optional[bool]
    url: str  # Link to original posting
    posted_date: Optional[datetime]
    duplicate_group_id: Optional[str]  # For grouping duplicates
    duplicate_sources: List[str] = []  # Which platforms it appears on
    source_metadata: Dict[str, Any] = {}  # Provider-specific data

    def generate_duplicate_hash(self) -> str:
        """Create hash for detecting duplicates across platforms"""
        def normalize(text: str) -> str:
            return ''.join(c for c in text.lower() if c.isalpha())

        title_norm = normalize(self.title)
        company_norm = normalize(self.company)
        location_norm = normalize(self.location or "")
        desc_norm = normalize(self.description or "")[:50]

        key = f"{company_norm}:{title_norm}:{location_norm}:{desc_norm}"
        return hashlib.md5(key.encode()).hexdigest()
```

**Design Notes:**
- `source`, `title`, `company` are required minimum
- `duplicate_group_id` enables UI grouping
- `source_metadata` allows providers to store extra data without polluting base model

#### Company
```python
class Company(BaseModel):
    name: str  # Required
    cached_date: Optional[datetime] = None

    # Dynamic fields added by research providers
    model_config = ConfigDict(extra='allow')

    def merge_research(self, result: ResearchResult) -> 'Company':
        """Merge provider result into this company"""
        return self.model_copy(
            update=result.model_dump(exclude_none=True)
        )
```

**Design Notes:**
- `extra='allow'` enables dynamic fields from any provider
- Fields like `glassdoor_rating`, `scam_score` are added at runtime
- Each provider contributes its own fields

#### ResearchCategory
```python
class ResearchCategory(str, Enum):
    BASIC = "basic"              # Free, instant (job age, duplicates)
    API_CHEAP = "api_cheap"      # Fast APIs (Glassdoor) ~1-5s
    API_EXPENSIVE = "api_expensive"  # Rate-limited APIs ~5-30s
    AI = "ai"                    # LLM calls ~10-60s
```

**Design Notes:**
- Categorizes cost/speed of research
- Frontend uses for UI grouping and time estimates
- Backend uses for provider selection

---

### 2. Provider System

#### Abstract Base Classes

```python
class JobSource(ABC):
    """Base class for job listing sources (Indeed, LinkedIn, etc.)"""

    @abstractmethod
    async def search(
        self,
        query: str,
        location: Optional[str] = None,
        remote: Optional[bool] = None,
        min_pay: Optional[int] = None,
        max_pay: Optional[int] = None
    ) -> List[JobListing]:
        """Search this source and return standardized JobListings"""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier: 'Indeed', 'LinkedIn', etc."""
        pass
```

```python
class ResearchResult(BaseModel):
    """Base class for all provider research results"""
    pass

class ResearchProvider(ABC):
    """Base class for company research providers"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier: 'glassdoor', 'scam_detector', etc."""
        pass

    @abstractmethod
    def contributions(self) -> Dict[str, FieldContribution]:
        """
        Declare which fields this provider contributes
        Returns: {
            field_name: FieldContribution(category, label, display_metadata)
        }
        """
        pass

    @abstractmethod
    async def research(
        self,
        company_name: str,
        requested_fields: List[str]
    ) -> ResearchResult:
        """
        Fetch data for requested fields only
        Returns provider-specific ResearchResult subclass
        """
        pass
```

#### Example Provider Implementation

```python
# providers/research/glassdoor/models.py
class GlassdoorResearchResult(ResearchResult):
    glassdoor_rating: Optional[float] = None
    glassdoor_review_count: Optional[int] = None
    glassdoor_pros: List[str] = []
    glassdoor_cons: List[str] = []

# providers/research/glassdoor/provider.py
class GlassdoorProvider(ResearchProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = GlassdoorAPIClient(api_key)

    @property
    def provider_name(self) -> str:
        return "glassdoor"

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

    async def research(
        self,
        company_name: str,
        requested_fields: List[str]
    ) -> GlassdoorResearchResult:
        # Only fetch if any glassdoor fields requested
        my_fields = set(self.contributions().keys())
        needed = my_fields.intersection(requested_fields)

        if not needed:
            return GlassdoorResearchResult()  # Empty

        # Fetch from API
        data = await self.client.get_company_reviews(company_name)

        # Populate only requested fields
        result = GlassdoorResearchResult()
        if "glassdoor_rating" in needed:
            result.glassdoor_rating = data.get('overall_rating')
        if "glassdoor_review_count" in needed:
            result.glassdoor_review_count = data.get('number_of_ratings')
        if "glassdoor_pros" in needed:
            result.glassdoor_pros = self._extract_common_pros(data)
        if "glassdoor_cons" in needed:
            result.glassdoor_cons = self._extract_common_cons(data)

        return result

# providers/research/glassdoor/__init__.py
def get_provider() -> Optional[ResearchProvider]:
    """Factory function called during provider discovery"""
    settings = get_settings()
    if not settings.glassdoor_api_key:
        logger.info("Glassdoor provider not configured (no API key)")
        return None
    return GlassdoorProvider(settings.glassdoor_api_key)
```

**Design Notes:**
- Provider declares its contributions (fields + display metadata)
- `research()` only fetches data for requested fields (efficiency)
- Returns typed `ResearchResult` subclass (not dict)
- Factory function `get_provider()` allows graceful skip if not configured

---

### 3. Display System

The display system uses metadata from providers to render fields consistently.

#### DisplayMetadata Model

```python
class DisplayMetadata(BaseModel):
    type: Literal["text", "rating", "percentage", "badge", "list", "custom"]
    icon: Optional[str] = None
    priority: Literal["low", "medium", "high"] = "medium"
    format: Optional[str] = None  # "currency", "date", "percentage", "decimal_1"
    max_value: Optional[float] = None  # For ratings
    color_scale: Optional[Dict[str, str]] = None  # {"0-30": "green", "70-100": "red"}
    invert: bool = False  # Lower is better (for scam scores)
    list_style: Optional[str] = None  # "bullet", "numbered", "comma"
    custom_config: Optional[Dict[str, Any]] = None  # For type="custom"
```

#### Frontend Renderer

```javascript
class FieldRenderer {
    constructor(metadata) {
        this.metadata = metadata;
    }

    async render(fieldName, value, company) {
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
                return this.renderText(value);
        }
    }

    renderRating(value) {
        const stars = this.metadata.icon.repeat(Math.round(value));
        const formatted = this.formatValue(value);
        return `
            <div class="field-rating">
                ${stars} ${formatted}/${this.metadata.max_value}
            </div>
        `;
    }

    renderPercentage(value) {
        const percent = Math.round(value * 100);
        const colorClass = this.getColorClass(value);
        return `
            <div class="field-percentage ${colorClass}">
                ${this.metadata.icon} ${percent}%
            </div>
        `;
    }

    renderList(value) {
        if (!Array.isArray(value) || value.length === 0) return '';

        const items = value.map(item => `<li>${item}</li>`).join('');
        return `
            <div class="field-list">
                ${this.metadata.icon ? this.metadata.icon + ' ' : ''}
                <strong>${this.metadata.label}:</strong>
                <ul>${items}</ul>
            </div>
        `;
    }

    getColorClass(value) {
        if (!this.metadata.color_scale) return '';

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

    async loadCustomRenderer(fieldName, value, company) {
        // Load provider's custom JS module
        const provider = this.metadata.provider;
        try {
            const module = await import(`/api/providers/${provider}/frontend.js`);
            if (module.render) {
                return module.render(fieldName, value, company, this.metadata.custom_config);
            }
        } catch (e) {
            console.warn(`Custom renderer failed for ${fieldName}, falling back to text`, e);
        }
        // Fallback
        return this.renderText(value);
    }

    formatValue(value) {
        if (this.metadata.format === 'decimal_1') {
            return value.toFixed(1);
        }
        if (this.metadata.format === 'currency') {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        }
        return String(value);
    }
}
```

**Design Notes:**
- 90% of providers use standard types (no custom JS needed)
- Metadata determines rendering, ensuring consistency
- Custom escape hatch for complex visualizations
- Color scales adapt based on value ranges

---

### 4. Services Layer

#### CompanyResearchService

The orchestrator for all company research activities.

```python
class CompanyResearchService:
    def __init__(self, db_session: Session, providers: List[ResearchProvider]):
        self.db = db_session
        self.providers = providers
        self.cache_ttl_days = 30

        # Build field -> providers mapping
        self.field_to_providers: Dict[str, List[ResearchProvider]] = {}
        for provider in providers:
            for field_name in provider.contributions().keys():
                if field_name not in self.field_to_providers:
                    self.field_to_providers[field_name] = []
                self.field_to_providers[field_name].append(provider)

    async def research_companies(
        self,
        company_names: List[str],
        requested_fields: List[str]
    ) -> Dict[str, Company]:
        """
        Research multiple companies for specific fields.
        Uses cache where possible, queries providers for misses.
        Returns: {company_name: Company}
        """
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

            if missing_fields:
                # 3. Find providers that contribute missing fields
                providers_needed = set()
                for field in missing_fields:
                    providers_needed.update(
                        self.field_to_providers.get(field, [])
                    )

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
                    else:
                        logger.warning(f"Provider failed: {result}")

            company.cached_date = datetime.now()
            results[company_name] = company

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

        # Return as generic ResearchResult with dynamic fields
        class CachedResult(ResearchResult):
            model_config = ConfigDict(extra='allow')

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

    def _is_stale(self, cache_entry: CompanyCache) -> bool:
        """Check if cache entry exceeds TTL"""
        age_days = (datetime.now() - cache_entry.cached_at).days
        return age_days > self.cache_ttl_days
```

**Design Notes:**
- Per-field caching enables partial cache hits
- Provider discovery happens at service initialization
- Parallel research via `asyncio.gather()`
- Graceful handling of provider failures
- Service decides staleness, not the model

---

### 5. Job Aggregation

```python
async def aggregate_jobs_from_sources(
    query: str,
    location: Optional[str] = None,
    remote: Optional[bool] = None,
    min_pay: Optional[int] = None,
    max_pay: Optional[int] = None,
    limit: int = 100
) -> List[JobListing]:
    """
    Query all configured job sources in parallel, deduplicate results.
    """
    sources = discover_job_sources()

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

    # Flatten and handle errors
    all_jobs = []
    for source, result in zip(sources, results):
        if isinstance(result, list):
            all_jobs.extend(result)
        else:
            logger.error(f"Source {source.source_name} failed: {result}")

    # Deduplicate using hash
    seen_groups = {}
    unique_jobs = []

    for job in all_jobs:
        group_id = job.generate_duplicate_hash()
        job.duplicate_group_id = group_id

        if group_id not in seen_groups:
            job.duplicate_sources = [job.source]
            seen_groups[group_id] = job
            unique_jobs.append(job)
        else:
            # Track duplicate sources
            seen_groups[group_id].duplicate_sources.append(job.source)

    # Sort by recency
    unique_jobs.sort(
        key=lambda j: j.posted_date or datetime.min,
        reverse=True
    )

    return unique_jobs[:limit]
```

**Design Notes:**
- Parallel source queries maximize speed
- One source failure doesn't break entire search
- Duplicate detection groups same job across platforms
- Sorting by recency prioritizes fresh postings

---

## Data Flow

### Quick Search Flow

```
[User] → [Frontend: POST /api/search?fields=domain_valid,job_age]
           ↓
       [Backend: aggregate_jobs_from_sources()]
           ↓
       [Indeed API] ──┐
       [LinkedIn API] ├→ Parallel queries (1-3s)
       [Other Sources]─┘
           ↓
       [Deduplicate via hash]
           ↓
       [Compute basic fields (no external calls)]
           ↓
       [Return 100 jobs + minimal company data]
           ↓
       [Frontend: Render immediately]
           ↓
       [User sees results in ~2 seconds]
```

### Deep Search Flow

```
[User] → [Frontend: POST /api/search?fields=glassdoor_rating,scam_score]
           ↓
       [Backend: aggregate_jobs_from_sources()]
           ↓
       [Get 100 jobs (1-3s)]
           ↓
       [Extract unique companies: ~80]
           ↓
       [CompanyResearchService.research_companies(...)]
           ↓
       [For each company:]
           ├─ Check cache for glassdoor_rating ──→ [HIT] → Use cached
           ├─ Check cache for scam_score ─────────→ [MISS] → Research
           ↓
       [Parallel provider queries:]
           ├─ GlassdoorProvider.research() ──→ API call (2-5s)
           ├─ AIScamDetector.research() ──────→ OpenAI call (10-30s)
           ↓
       [Merge results into Company objects]
           ↓
       [Cache new data per-field]
           ↓
       [Filter jobs based on criteria]
           ↓
       [Return filtered jobs + company data]
           ↓
       [Frontend: Render with full details]
           ↓
       [User sees filtered results in ~30-60 seconds]
```

### Caching Flow

```
[Request field: scam_score for "TechCorp"]
    ↓
[Query CompanyCache: name=TechCorp, field_name=scam_score]
    ↓
    ├─ [Entry found, age < 30 days] ──→ Return cached value
    │
    └─ [Entry missing OR age >= 30 days]
           ↓
       [Call AIScamDetector.research()]
           ↓
       [OpenAI API call]
           ↓
       [Insert/Update CompanyCache]
           ↓
       [Return fresh value]
```

---

## Code Patterns

### Pattern 1: Dependency Injection with FastAPI

```python
from fastapi import Depends

# Settings as dependency
def get_settings() -> Settings:
    return Settings.from_configparser("config.ini", section="jobsearch")

# Database session as dependency
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Service as dependency (uses other dependencies)
def get_research_service(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings)
) -> CompanyResearchService:
    providers = discover_research_providers(settings)
    return CompanyResearchService(db, providers)

# Endpoint uses dependency
@app.post("/api/search")
async def search(
    request: SearchRequest,
    fields: List[str] = Query(default=[]),
    research_service: CompanyResearchService = Depends(get_research_service)
) -> SearchResponse:
    # Service is injected, ready to use
    ...
```

**Benefits:**
- Testable (easy to mock dependencies)
- Explicit dependencies
- FastAPI handles lifecycle

### Pattern 2: Provider Factory Functions

```python
# providers/research/glassdoor/__init__.py
def get_provider() -> Optional[ResearchProvider]:
    """Factory called during discovery"""
    settings = get_settings()
    if not settings.glassdoor_api_key:
        return None  # Skip if not configured
    return GlassdoorProvider(settings.glassdoor_api_key)
```

**Benefits:**
- Provider can skip itself if not configured
- Consistent interface for discovery
- Explicit dependencies (settings)

### Pattern 3: Event-Driven Frontend

```javascript
// Event bus for decoupling
const eventBus = new EventBus();

// Components listen for events
class JobSearchApp {
    constructor() {
        eventBus.on('search:requested', (filters) => this.handleSearch(filters));
        eventBus.on('research:deep-requested', (jobId) => this.handleDeepResearch(jobId));
    }
}

class SearchUI {
    constructor() {
        eventBus.on('jobs:loaded', (jobs) => this.renderJobs(jobs));
        eventBus.on('company:updated', ({jobId, company}) => this.updateJobCard(jobId, company));
    }
}

// User actions emit events
document.getElementById('quick-search-btn').addEventListener('click', () => {
    const filters = gatherFilters();
    eventBus.emit('search:requested', { ...filters, mode: 'quick' });
});
```

**Benefits:**
- Components don't know about each other
- Easy to add new features (just add listeners)
- Testable in isolation

### Pattern 4: Metadata-Driven Display

```python
# Provider declares display metadata
def contributions(self) -> Dict[str, FieldContribution]:
    return {
        "glassdoor_rating": FieldContribution(
            display=DisplayMetadata(
                type="rating",
                icon="⭐",
                max_value=5
            )
        )
    }
```

```javascript
// Frontend uses metadata to render
const metadata = fieldRegistry['glassdoor_rating'].display;
const renderer = new FieldRenderer(metadata);
const html = renderer.render('glassdoor_rating', 4.3, company);
```

**Benefits:**
- Consistent UI across all providers
- Backend controls display without frontend changes
- Extensible (new display types can be added)

### Pattern 5: Graceful Error Handling

```python
# Parallel operations with error tolerance
results = await asyncio.gather(*tasks, return_exceptions=True)

for result in results:
    if isinstance(result, ExpectedType):
        # Success path
        process(result)
    else:
        # Error path - log but continue
        logger.warning(f"Task failed: {result}")
```

**Benefits:**
- Partial results better than no results
- System resilient to external failures
- Clear logging for debugging

---

## Configuration

### Example config.ini

```ini
[api]
host = 0.0.0.0
port = 8000

[jobsearch]
# Database
db_url = sqlite:///./jobsearch.db

# API Keys
openai_api_key = sk-...
indeed_api_key = xxx
linkedin_api_key = yyy
glassdoor_api_key = zzz

# Cache settings
cache_ttl_days = 30

# Provider management
ignored_providers = old_provider, broken_provider
require_all_providers = false
```

### Loading Configuration

```python
class Settings(BaseModel):
    db_url: str = "sqlite:///./jobsearch.db"
    openai_api_key: str
    indeed_api_key: str
    linkedin_api_key: Optional[str] = None
    glassdoor_api_key: Optional[str] = None
    cache_ttl_days: int = 30
    ignored_providers: List[str] = []
    require_all_providers: bool = False

    @classmethod
    def from_configparser(cls, config_path: str, section: str):
        config = configparser.ConfigParser()
        config.read(config_path)

        if section not in config:
            raise ValueError(f"Section [{section}] not found")

        settings_dict = dict(config[section])

        # Type conversions
        if 'ignored_providers' in settings_dict:
            settings_dict['ignored_providers'] = [
                p.strip() for p in settings_dict['ignored_providers'].split(',')
            ]

        if 'cache_ttl_days' in settings_dict:
            settings_dict['cache_ttl_days'] = int(settings_dict['cache_ttl_days'])

        if 'require_all_providers' in settings_dict:
            settings_dict['require_all_providers'] = (
                settings_dict['require_all_providers'].lower() == 'true'
            )

        return cls(**settings_dict)
```

---

## Testing Strategy

### Unit Tests
- Models: Validation, hash generation
- Providers: Mocked API responses
- Services: Mocked database and providers
- Display: Metadata rendering

### Integration Tests
- End-to-end search flow
- Cache behavior
- Provider discovery
- Error handling

### Example Test

```python
@pytest.mark.asyncio
async def test_research_service_uses_cache():
    # Setup
    mock_db = MockDatabase()
    mock_db.add_cache_entry(
        name="TechCorp",
        field_name="scam_score",
        value=0.15,
        cached_at=datetime.now()
    )
    mock_provider = MockScamDetector()
    service = CompanyResearchService(mock_db, [mock_provider])

    # Execute
    result = await service.research_companies(
        ["TechCorp"],
        ["scam_score"]
    )

    # Assert
    assert result["TechCorp"].scam_score == 0.15
    assert mock_provider.call_count == 0  # Should use cache, not call provider
```

---

## Next Steps

See:
- [FLEX.md](./FLEX.md) for extensibility points
- [DESIGN_GUIDANCE.md](./DESIGN_GUIDANCE.md) for philosophy and principles
- [UX_FLOW.md](./UX_FLOW.md) for complete user experience flows
- [../implementation/PLAN.md](../implementation/PLAN.md) for step-by-step build guide
