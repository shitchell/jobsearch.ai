# Flexibility & Extension Points

This document details where and how the Job Search AI system can be extended, modified, or adapted without requiring major architectural changes.

## Table of Contents
1. [Plugin Systems](#plugin-systems)
2. [Configuration Points](#configuration-points)
3. [Display Customization](#display-customization)
4. [Database Flexibility](#database-flexibility)
5. [API Extensions](#api-extensions)
6. [Future-Proofing](#future-proofing)

---

## Plugin Systems

### 1. Job Source Providers

**What:** Add new job listing sources (ZipRecruiter, Dice, Monster, etc.)

**How:**
```
providers/job_sources/
├── indeed/
│   ├── __init__.py        # get_provider() factory
│   └── provider.py        # IndeedSource(JobSource)
├── linkedin/
│   ├── __init__.py
│   └── provider.py
└── ziprecruiter/          # NEW: Just drop in this directory
    ├── __init__.py
    └── provider.py
```

**Requirements:**
1. Implement `JobSource` abstract base class
2. Provide `get_provider()` factory function in `__init__.py`
3. Return standardized `JobListing` objects

**Example:**
```python
# providers/job_sources/ziprecruiter/provider.py
class ZipRecruiterSource(JobSource):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def source_name(self) -> str:
        return "ZipRecruiter"

    async def search(self, query: str, **kwargs) -> List[JobListing]:
        # Query ZipRecruiter API
        results = await self._query_api(query, **kwargs)

        # Convert to standard JobListing format
        return [
            JobListing(
                id=str(uuid4()),
                source="ZipRecruiter",
                title=r['job_title'],
                company=r['hiring_company']['name'],
                # ... map other fields
            )
            for r in results
        ]

# providers/job_sources/ziprecruiter/__init__.py
def get_provider() -> Optional[JobSource]:
    settings = get_settings()
    if not settings.ziprecruiter_api_key:
        return None
    return ZipRecruiterSource(settings.ziprecruiter_api_key)
```

**Automatic Discovery:**
- No backend code changes needed
- Add config entry: `ziprecruiter_api_key = xxx`
- Restart backend
- New source automatically discovered and used

**Graceful Failure:**
- If provider raises exception during loading:
  - `require_all_providers=false` (default): Warning logged, continues without it
  - `require_all_providers=true`: Backend refuses to start
- If provider returns None (not configured): Silently skipped

---

### 2. Research Providers

**What:** Add new company research capabilities (salary data, LinkedIn stats, SEC filings, etc.)

**How:**
```
providers/research/
├── glassdoor/
│   ├── __init__.py
│   ├── models.py          # GlassdoorResearchResult
│   └── provider.py        # GlassdoorProvider
├── scam_detector/
│   ├── __init__.py
│   ├── models.py
│   └── provider.py
└── salary_api/            # NEW: Just drop in
    ├── __init__.py
    ├── models.py
    └── provider.py
```

**Requirements:**
1. Implement `ResearchProvider` abstract base class
2. Define custom `ResearchResult` subclass (Pydantic model)
3. Declare `contributions()` with field metadata
4. Provide `get_provider()` factory

**Example:**
```python
# providers/research/salary_api/models.py
class SalaryResearchResult(ResearchResult):
    salary_median: Optional[float] = None
    salary_range_low: Optional[float] = None
    salary_range_high: Optional[float] = None
    salary_confidence: Optional[float] = None

# providers/research/salary_api/provider.py
class SalaryAPIProvider(ResearchProvider):
    @property
    def provider_name(self) -> str:
        return "salary_api"

    def contributions(self) -> Dict[str, FieldContribution]:
        return {
            "salary_median": FieldContribution(
                category=ResearchCategory.API_CHEAP,
                label="Market Salary (Median)",
                display=DisplayMetadata(
                    type="text",
                    icon="💰",
                    format="currency",
                    priority="high"
                )
            ),
            "salary_range": FieldContribution(
                category=ResearchCategory.API_CHEAP,
                label="Salary Range",
                display=DisplayMetadata(
                    type="text",
                    format="currency_range",
                    priority="medium"
                )
            )
        }

    async def research(self, company_name: str, requested_fields: List[str]) -> SalaryResearchResult:
        my_fields = set(self.contributions().keys())
        needed = my_fields.intersection(requested_fields)

        if not needed:
            return SalaryResearchResult()

        data = await self._query_salary_api(company_name)

        result = SalaryResearchResult()
        if "salary_median" in needed:
            result.salary_median = data.get('median_salary')
        if "salary_range" in needed:
            result.salary_range_low = data.get('salary_10th_percentile')
            result.salary_range_high = data.get('salary_90th_percentile')

        return result
```

**Automatic Integration:**
- Frontend automatically discovers new fields via `/api/research/fields`
- New filters appear in UI (grouped by category)
- Display metadata controls rendering
- No frontend code changes needed (for standard display types)

**Category Selection:**
- `BASIC`: Free, instant computation (no API calls)
- `API_CHEAP`: Fast APIs, low cost (~$0.001/call, <5s)
- `API_EXPENSIVE`: Slower/costly APIs (~$0.01/call, 5-30s)
- `AI`: LLM-based analysis (~$0.05/call, 10-60s)

Frontend uses categories to:
- Group filters in UI
- Estimate search time
- Let users choose speed vs. depth

---

### 3. Custom Display Types

**What:** Add new visualization types beyond built-in (text, rating, percentage, list, badge)

**Standard Approach (Preferred):**

Add new display type to core `DisplayMetadata`:
```python
# In core models
class DisplayMetadata(BaseModel):
    type: Literal[
        "text", "rating", "percentage", "badge", "list",
        "chart",  # NEW
        "custom"
    ]
    # ... existing fields ...
    chart_config: Optional[Dict[str, Any]] = None  # For type="chart"
```

Add renderer to frontend:
```javascript
// In FieldRenderer class
renderChart(value) {
    const config = this.metadata.chart_config;
    if (config.type === 'bar') {
        return this.renderBarChart(value, config);
    }
    if (config.type === 'line') {
        return this.renderLineChart(value, config);
    }
    // ... etc
}
```

Now ALL providers can use `type="chart"` without custom JS.

**Custom Approach (When Needed):**

For truly unique visualizations, use `type="custom"`:

```python
# Provider declares custom display
display=DisplayMetadata(
    type="custom",
    custom_config={"radar_dimensions": ["culture", "pay", "growth", "wlb", "tech"]}
)
```

```javascript
// providers/research/complex_provider/frontend.js
export function render(fieldName, value, company, config) {
    // Custom D3.js/Chart.js visualization
    return complexRadarChart(value, config);
}
```

**Decision Guide:**
- Can it be generalized? → Add to core display types
- Is it truly unique to one provider? → Use custom JS
- 90% of cases should use core types

---

## Configuration Points

### 1. Database Backend

**What:** Switch between SQLite (dev), PostgreSQL (prod), MySQL, etc.

**How:**
```ini
# Development
[jobsearch]
db_url = sqlite:///./jobsearch.db

# Production
[jobsearch]
db_url = postgresql://user:pass@localhost/jobsearch
```

SQLAlchemy handles dialect differences automatically.

**Requirements:**
- Install appropriate driver (`psycopg2`, `pymysql`, etc.)
- Ensure schema migrations work (use Alembic)

---

### 2. Cache TTL

**What:** Control how long research data is cached

**How:**
```ini
[jobsearch]
cache_ttl_days = 30  # Default
# or
cache_ttl_days = 7   # More aggressive refresh
```

**Considerations:**
- Shorter TTL: Fresher data, higher API costs
- Longer TTL: Lower costs, potentially stale data
- Per-provider TTL (future enhancement):
  ```python
  def cache_ttl(self, field_name: str) -> int:
      """Override to customize per field"""
      if field_name == "scam_score":
          return 7  # Refresh weekly
      return 30
  ```

---

### 3. Provider Management

**What:** Control which providers load

**How:**
```ini
[jobsearch]
# Ignore specific providers
ignored_providers = old_glassdoor, experimental_provider

# Strict mode: require all non-ignored providers to load
require_all_providers = false  # Default: graceful degradation
# require_all_providers = true   # Production: fail-fast
```

**Use Cases:**
- Development: Ignore expensive providers
- Debugging: Isolate one provider
- Deployment: Strict mode catches config errors early

---

### 4. API Keys

**What:** Configure external services

**How:**
```ini
[jobsearch]
openai_api_key = sk-...
indeed_api_key = xxx
linkedin_api_key = yyy
glassdoor_api_key = zzz

# Optional providers
salary_api_key = aaa
clearbit_api_key = bbb
```

**Behavior:**
- Provider with missing API key returns `None` from factory
- Gracefully excluded from provider list
- No code changes needed to enable/disable providers

---

## Display Customization

### 1. Frontend Themes

**What:** Change colors, icons, layout

**Current Approach:**
- CSS variables for colors
- Icon mapping in FieldRenderer
- Layout controlled by templates

**Extension Points:**
```css
:root {
    --color-green: #22c55e;
    --color-yellow: #eab308;
    --color-red: #ef4444;
    --priority-high-size: 1.2em;
    --priority-medium-size: 1em;
    --priority-low-size: 0.9em;
}
```

**Future Enhancement:**
Theme system where providers can suggest colors:
```python
display=DisplayMetadata(
    type="badge",
    color_scale={
        "0-30": "green",
        "30-70": "yellow",
        "70-100": "red"
    },
    theme_overrides={
        "light": {"green": "#22c55e"},
        "dark": {"green": "#4ade80"}
    }
)
```

---

### 2. Field Priority

**What:** Control which fields are prominently displayed

**How:**
```python
display=DisplayMetadata(
    priority="high"  # Always visible
    # priority="medium"  # Visible in expanded view
    # priority="low"  # Only in deep research modal
)
```

**Frontend Behavior:**
- High priority: Shown in job card
- Medium priority: Shown on hover or expand
- Low priority: Only in deep research modal

**User Customization (Future):**
```javascript
// User can reorder/hide fields
localStorage.setItem('field_priority_overrides', JSON.stringify({
    'glassdoor_rating': 'high',
    'scam_score': 'high',
    'salary_median': 'medium'
}));
```

---

## Database Flexibility

### 1. Schema Evolution

**What:** Add new tables, columns without breaking existing deployments

**Approach:** Alembic migrations

```bash
# Create migration after adding new field to model
alembic revision --autogenerate -m "Add salary comparison table"

# Apply migration
alembic upgrade head
```

**Provider Fields:**
- Company model uses `extra='allow'`
- New fields don't require schema changes
- Cache table uses JSON column for flexible storage

---

### 2. Cache Strategies

**Current:** Per-field caching in SQL table

**Alternative 1:** Redis cache (faster)
```python
class RedisCacheAdapter(CacheAdapter):
    async def get(self, company_name: str, field_name: str) -> Optional[Any]:
        key = f"company:{company_name}:{field_name}"
        return await self.redis.get(key)

    async def set(self, company_name: str, field_name: str, value: Any, ttl: int):
        key = f"company:{company_name}:{field_name}"
        await self.redis.setex(key, ttl, json.dumps(value))
```

**Alternative 2:** File-based cache (no DB needed)
```python
class FileCacheAdapter(CacheAdapter):
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    async def get(self, company_name: str, field_name: str) -> Optional[Any]:
        cache_file = self.cache_dir / f"{company_name}_{field_name}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
```

**Adapter Pattern:**
All cache implementations satisfy same interface, swap via config.

---

## API Extensions

### 1. Pagination Strategies

**Current:** Client-side (return all results, frontend paginates)

**Alternative:** Server-side pagination
```python
@app.post("/api/search")
async def search(
    request: SearchRequest,
    fields: List[str] = Query(default=[]),
    page: int = Query(default=1),
    page_size: int = Query(default=10)
) -> SearchResponse:
    offset = (page - 1) * page_size
    jobs = await aggregate_jobs_from_sources(request)

    # Paginate before research
    jobs_page = jobs[offset:offset + page_size]

    # Only research companies on this page
    company_names = list(set(job.company for job in jobs_page))
    companies = await research_service.research_companies(company_names, fields)

    return SearchResponse(
        jobs=jobs_page,
        companies=companies,
        total_count=len(jobs),
        page=page
    )
```

**Trade-offs:**
- Client-side: Fast pagination, but large initial response
- Server-side: Smaller responses, but research needed per page

---

### 2. Webhooks / Background Jobs

**Future Enhancement:** Long-running research as background job

```python
@app.post("/api/search/async")
async def search_async(
    request: SearchRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    job_id = str(uuid4())

    background_tasks.add_task(
        perform_deep_search,
        job_id,
        request
    )

    return {"job_id": job_id, "status_url": f"/api/search/{job_id}/status"}

@app.get("/api/search/{job_id}/status")
async def get_search_status(job_id: str) -> SearchStatus:
    # Check job status in database/cache
    return SearchStatus(
        job_id=job_id,
        status="in_progress",  # or "complete", "failed"
        progress=0.65,
        results=None  # or actual results if complete
    )
```

**Use Case:** Very large searches (1000+ jobs, 500+ companies)

---

### 3. User Accounts (Future)

**Current:** No authentication, localStorage for preferences

**Extension Point:** Add authentication without breaking existing API

```python
# Optional auth dependency
async def get_current_user(
    token: Optional[str] = Header(None)
) -> Optional[User]:
    if not token:
        return None  # Anonymous user
    return await verify_token(token)

# Endpoints work with or without auth
@app.post("/api/search")
async def search(
    request: SearchRequest,
    user: Optional[User] = Depends(get_current_user)
):
    # If user is authenticated, save search history
    if user:
        await save_search_history(user.id, request)

    # Search works either way
    ...
```

**Features unlocked by auth:**
- Saved searches
- Email alerts for new matching jobs
- Application tracking
- Shared company notes

---

## Future-Proofing

### 1. Duplicate Detection Algorithm

**Current:** Hash-based with normalized company + title + location + description snippet

**Tunable:**
```python
def generate_duplicate_hash(
    self,
    include_description: bool = True,
    desc_length: int = 50,
    strict_location: bool = True
) -> str:
    """
    Adjustable duplicate detection

    Args:
        include_description: Consider description in hash
        desc_length: How many chars of description
        strict_location: False to ignore minor location differences
    """
    # ... implementation
```

**Future Enhancement:** ML-based duplicate detection
```python
class MLDuplicateDetector(DuplicateDetector):
    async def find_duplicates(self, jobs: List[JobListing]) -> Dict[str, List[str]]:
        # Use embeddings + cosine similarity
        embeddings = await self.embed_jobs(jobs)
        similarity_matrix = cosine_similarity(embeddings)

        # Group jobs with >0.9 similarity
        return cluster_similar_jobs(similarity_matrix, threshold=0.9)
```

---

### 2. Advanced Filtering

**Current:** Simple filters (min/max values, exact matches)

**Extension Point:** Complex filter DSL

```python
class FilterExpression(BaseModel):
    field: str
    operator: Literal["eq", "ne", "gt", "lt", "contains", "in", "between"]
    value: Any

class SearchRequest(BaseModel):
    # ... existing fields ...
    advanced_filters: List[FilterExpression] = []

# Example usage
advanced_filters=[
    FilterExpression(field="scam_score", operator="lt", value=0.3),
    FilterExpression(field="glassdoor_rating", operator="between", value=[3.5, 5.0]),
    FilterExpression(field="location", operator="in", value=["Remote", "SF", "NYC"])
]
```

---

### 3. Multi-Source Company Profiles

**Current:** Company data from single research session

**Future:** Aggregate company data from multiple jobs over time

```python
class CompanyProfile(BaseModel):
    """Aggregate view of company across multiple job postings"""
    name: str
    job_count: int  # How many jobs posted
    first_seen: datetime
    last_seen: datetime
    average_pay_range: Tuple[float, float]
    locations: List[str]
    common_job_titles: List[str]

    # Research data (most recent)
    scam_score: float
    glassdoor_rating: float
    # ...
```

**Use Cases:**
- "This company has posted 50 jobs this month" (red flag)
- "Common roles at this company: SWE, DevOps, PM"
- Track company reputation changes over time

---

### 4. Export / Import

**Extension Point:** Modular export formats

```python
class ExportAdapter(ABC):
    @abstractmethod
    def export(self, jobs: List[JobListing], companies: Dict[str, Company]) -> str:
        pass

class CSVExportAdapter(ExportAdapter):
    def export(self, jobs, companies):
        # Generate CSV
        ...

class JSONExportAdapter(ExportAdapter):
    def export(self, jobs, companies):
        return json.dumps([job.model_dump() for job in jobs])

class NotionExportAdapter(ExportAdapter):
    def export(self, jobs, companies):
        # Push to Notion database via API
        ...
```

```python
@app.get("/api/search/{search_id}/export")
async def export_search(
    search_id: str,
    format: str = Query(default="json", regex="^(json|csv|notion)$")
):
    adapter = get_export_adapter(format)
    jobs, companies = get_search_results(search_id)
    data = adapter.export(jobs, companies)
    return Response(content=data, media_type=adapter.media_type)
```

---

### 5. Cost Controls & Observability (Future Enhancement)

**Current Status:** No rate limiting, no usage tracking. Designed for trusted users with manual cost monitoring.

**Why This Is Deferred:**
- Initial deployment: Single user (friend of developer)
- Low usage volume: Cost explosion unlikely
- Manual monitoring: Review API bills monthly
- Trusted users: No abuse expected

**Future Extension Point:** When expanding to more users or higher volume

#### Rate Limiting Design

```python
# models/rate_limit.py
class RateLimitRule(BaseModel):
    category: ResearchCategory  # AI, API_CHEAP, API_EXPENSIVE
    max_requests: int
    window_seconds: int  # e.g., 86400 for daily limit

class RateLimiter:
    def __init__(self, db: Session):
        self.db = db
        self.rules: Dict[ResearchCategory, RateLimitRule] = {}

    async def check_limit(
        self,
        user_id: str,
        category: ResearchCategory
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns: (allowed, error_message)
        """
        rule = self.rules.get(category)
        if not rule:
            return (True, None)  # No limit configured

        # Count recent requests
        recent_count = self.db.query(UsageLog).filter(
            UsageLog.user_id == user_id,
            UsageLog.category == category,
            UsageLog.timestamp > datetime.now() - timedelta(seconds=rule.window_seconds)
        ).count()

        if recent_count >= rule.max_requests:
            return (False, f"Rate limit exceeded: {rule.max_requests} {category.value} searches per day")

        return (True, None)

    async def record_usage(self, user_id: str, category: ResearchCategory):
        """Log usage for rate limiting"""
        usage = UsageLog(
            user_id=user_id,
            category=category,
            timestamp=datetime.now()
        )
        self.db.add(usage)
        self.db.commit()

# Configuration
[jobsearch]
rate_limits_enabled = false  # Default: disabled
max_ai_searches_per_user_per_day = 5
max_api_cheap_searches_per_user_per_day = 50
max_api_expensive_searches_per_user_per_day = 20
```

**Integration Point:**
```python
@app.post("/api/search")
async def search(
    request: SearchRequest,
    fields: List[str] = Query(default=[]),
    user_id: str = Depends(get_user_id),  # From session or IP
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    # Determine highest category being requested
    categories = {get_field_category(f) for f in fields}
    highest_category = max(categories, key=lambda c: c.value)

    # Check rate limit
    allowed, error_msg = await rate_limiter.check_limit(user_id, highest_category)
    if not allowed:
        raise HTTPException(status_code=429, detail=error_msg)

    # Perform search...

    # Record usage
    await rate_limiter.record_usage(user_id, highest_category)
```

#### Cost Tracking & Alerting

```python
# models/cost_tracking.py
class CostEstimate(BaseModel):
    category: ResearchCategory
    estimated_cost: float  # USD
    estimated_duration: float  # seconds

class CostTracker:
    # Provider cost estimates
    COSTS = {
        ResearchCategory.AI: 0.05,  # per company
        ResearchCategory.API_EXPENSIVE: 0.01,
        ResearchCategory.API_CHEAP: 0.001,
        ResearchCategory.BASIC: 0.0
    }

    async def estimate_search_cost(
        self,
        company_count: int,
        requested_categories: List[ResearchCategory]
    ) -> CostEstimate:
        """Estimate cost before executing search"""
        total_cost = sum(
            self.COSTS[cat] * company_count
            for cat in requested_categories
        )
        return CostEstimate(
            category=max(requested_categories),
            estimated_cost=total_cost,
            estimated_duration=self._estimate_duration(company_count, requested_categories)
        )

# Endpoint
@app.post("/api/search/estimate")
async def estimate_search_cost(request: SearchRequest) -> CostEstimate:
    """Preview cost/time before executing Deep Search"""
    # Aggregate jobs to count unique companies
    jobs = await aggregate_jobs_from_sources(request)
    company_count = len(set(job.company for job in jobs))

    categories = [get_field_category(f) for f in request.fields]
    return await cost_tracker.estimate_search_cost(company_count, categories)

# Alerting (via email or webhook when daily cost exceeds threshold)
class CostAlert:
    async def check_daily_spend(self):
        today_spend = self.db.query(func.sum(UsageLog.cost)).filter(
            UsageLog.timestamp >= datetime.now().date()
        ).scalar()

        if today_spend > self.threshold:
            await self.send_alert(f"Daily spend: ${today_spend:.2f} exceeds threshold ${self.threshold:.2f}")
```

#### Observability & Monitoring

```python
# Structured logging
import structlog

logger = structlog.get_logger()

# In CompanyResearchService
async def research_companies(self, ...):
    start_time = time.time()

    logger.info(
        "research_started",
        company_count=len(company_names),
        requested_fields=requested_fields,
        categories=[get_field_category(f).value for f in requested_fields]
    )

    # ... research logic ...

    duration = time.time() - start_time
    logger.info(
        "research_completed",
        company_count=len(company_names),
        duration_seconds=duration,
        cache_hits=cache_hit_count,
        cache_misses=cache_miss_count,
        provider_failures=failure_count
    )

# Metrics (Prometheus-style)
from prometheus_client import Counter, Histogram

search_requests = Counter('search_requests_total', 'Total search requests', ['mode', 'category'])
search_duration = Histogram('search_duration_seconds', 'Search duration', ['mode'])
provider_calls = Counter('provider_calls_total', 'Provider API calls', ['provider', 'status'])
cache_hits = Counter('cache_hits_total', 'Cache hits', ['field'])

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health status"""
    checks = {
        "database": await check_database_connection(),
        "cache": await check_cache_available(),
        "providers": await check_provider_health()
    }

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={"status": "healthy" if all_healthy else "unhealthy", "checks": checks}
    )

# Provider health monitoring
class ProviderHealthMonitor:
    def __init__(self):
        self.success_counts = {}  # provider -> recent success count
        self.failure_counts = {}  # provider -> recent failure count

    async def check_provider_health(self, provider_name: str) -> bool:
        """Check if provider has acceptable success rate"""
        successes = self.success_counts.get(provider_name, 0)
        failures = self.failure_counts.get(provider_name, 0)

        total = successes + failures
        if total < 10:
            return True  # Not enough data, assume healthy

        success_rate = successes / total
        return success_rate >= 0.8  # 80% threshold

    async def record_provider_result(self, provider_name: str, success: bool):
        """Track provider success/failure"""
        if success:
            self.success_counts[provider_name] = self.success_counts.get(provider_name, 0) + 1
        else:
            self.failure_counts[provider_name] = self.failure_counts.get(provider_name, 0) + 1
```

#### Admin Interface (Future)

Simple CLI tool for operational tasks:

```bash
# View cache statistics
jobsearch-admin cache stats
> Total entries: 1,247
> Cache size: 15.3 MB
> Hit rate (7d): 67%
> Top companies: TechCorp (45 hits), StartupXYZ (32 hits)

# Manually invalidate cache
jobsearch-admin cache clear --company "TechCorp Inc"
> Cleared 8 fields for TechCorp Inc

# View provider health
jobsearch-admin providers health
> glassdoor: healthy (success rate: 98%)
> scam_detector: degraded (success rate: 72%)
> salary_api: unhealthy (success rate: 45%)

# Disable misbehaving provider
jobsearch-admin providers disable salary_api
> Provider 'salary_api' disabled. Restart backend to apply.
```

**Implementation Note:** These are all designed but not implemented. Current status is "unlimited trusted access with manual monitoring."

---

## Summary: Hard Rules vs. Flex Points

### Hard Rules (Architectural Invariants)
1. **All providers must implement abstract base class** (JobSource or ResearchProvider)
2. **All provider directories must have `get_provider()` factory function**
3. **All research providers must return ResearchResult subclass** (not dict)
4. **All models must use Pydantic** (for validation and serialization)
5. **Field contributions must include display metadata** (for frontend rendering)

### Flex Points (Easy to Change/Extend)
1. ✅ Add new job sources (drop directory)
2. ✅ Add new research providers (drop directory)
3. ✅ Add new display types (extend DisplayMetadata + frontend renderer)
4. ✅ Switch database backends (change config)
5. ✅ Adjust cache TTL (change config)
6. ✅ Ignore providers (change config)
7. ✅ Custom provider displays (add frontend.js to provider directory)
8. ✅ Duplicate detection algorithm (tune parameters or replace implementation)
9. ✅ Pagination strategy (client vs server-side)
10. ✅ Export formats (implement ExportAdapter)

### Design Goal: 80/20 Rule
- **80% of extensions**: Drop in a directory, restart backend
- **20% of extensions**: Require code changes to core system (but well-defined extension points)

This keeps the system maintainable while enabling rapid iteration.
