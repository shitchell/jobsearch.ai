# UX Flow Documentation

## Overview
This document details the complete user experience flow for the Job Search application, including all user interactions, frontend state changes, backend API calls, and system responses.

## Core User Flows

### 1. Initial Page Load

**User Action:** User navigates to the application

**Frontend:**
1. Renders empty search interface with loading spinner on filter panel
2. Sends `GET /api/research/fields` to discover available research fields
3. Receives field registry:
   ```json
   {
     "domain_valid": {
       "category": "basic",
       "label": "Domain Validation",
       "display": {
         "type": "badge",
         "icon": "✓",
         "priority": "medium"
       },
       "provider": "domain_validator"
     },
     "glassdoor_rating": {
       "category": "api_cheap",
       "label": "Glassdoor Rating",
       "display": {
         "type": "rating",
         "icon": "⭐",
         "max_value": 5,
         "priority": "high"
       },
       "provider": "glassdoor"
     },
     "scam_score": {
       "category": "ai",
       "label": "Scam Detection",
       "display": {
         "type": "percentage",
         "icon": "🛡️",
         "priority": "high",
         "color_scale": {
           "0-30": "green",
           "30-70": "yellow",
           "70-100": "red"
         },
         "invert": true
       },
       "provider": "scam_detector"
     }
   }
   ```
4. Dynamically builds filter UI grouped by category:
   ```
   ┌─────────────────────────────────────────┐
   │ Basic Filters                           │
   │ ─────────────────────────────────────   │
   │ Keywords: [____________]                │
   │ Location: [____________]                │
   │ ☐ Remote only                           │
   │ Pay: $[____] to $[____]                 │
   │                                         │
   │ Advanced Filters                        │
   │ ─────────────────────────────────────   │
   │ ⚡ Instant Checks:                      │
   │   ☐ Domain Validation                   │
   │   ☐ Job Age                             │
   │                                         │
   │ 💰 API Checks (~5 seconds):            │
   │   ☐ Glassdoor Rating (min: [__])       │
   │   ☐ Salary Comparison                   │
   │                                         │
   │ 🤖 AI Analysis (~30 seconds):          │
   │   ☐ Scam Detection (max risk: [__]%)   │
   │   ☐ Job Description Analysis            │
   │                                         │
   │ [Quick Search]  [Deep Search]           │
   └─────────────────────────────────────────┘
   ```

**Backend:**
- `/api/research/fields` endpoint queries all loaded providers
- Aggregates field definitions with display metadata
- Returns comprehensive field registry
- No database queries, just in-memory provider registry

**State:**
- Frontend: `availableFields` populated, UI dynamically generated
- Backend: Idle, waiting for search request

---

### 2. Quick Search Flow

**User Action:** User enters "software engineer" + "remote", clicks **Quick Search**

**Frontend (Immediate - 0ms):**
1. Validates input fields
2. Hides results area, shows loading state:
   ```
   ┌─────────────────────────────────────────┐
   │ ⏳ Searching job listings...            │
   └─────────────────────────────────────────┘
   ```
3. Emits event: `eventBus.emit('search:requested', {...filters, mode: 'quick'})`
4. `JobSearchApp.handleSearch()` sends request:
   ```javascript
   POST /api/search
   Body: {
     query: "software engineer",
     remote: true,
     min_pay: null,
     max_pay: null
   }
   Query: ?fields=domain_valid,job_age
   ```

**Backend (1-3 seconds):**
1. Receives search request
2. Calls `aggregate_jobs_from_sources()`:
   - Queries Indeed API in parallel
   - Queries LinkedIn API in parallel
   - Queries any other configured sources
   - Returns after all sources respond (or timeout)
3. Generates `duplicate_group_id` for each job via `job.generate_duplicate_hash()`
4. For basic fields (`domain_valid`, `job_age`):
   - These are computed without external API calls
   - Domain validation: check if company domain exists and is valid
   - Job age: compute from `posted_date`
5. Returns raw job listings:
   ```json
   {
     "jobs": [
       {
         "id": "uuid-1",
         "source": "Indeed",
         "title": "Senior Software Engineer",
         "company": "TechCorp Inc",
         "description": "We are seeking...",
         "pay": "$120k-$150k",
         "location": "Remote",
         "remote": true,
         "url": "https://indeed.com/job/...",
         "posted_date": "2025-10-28T10:00:00Z",
         "duplicate_group_id": "abc123...",
         "duplicate_sources": ["Indeed", "LinkedIn"]
       },
       // ... 99 more jobs
     ],
     "companies": {
       "TechCorp Inc": {
         "name": "TechCorp Inc",
         "domain_valid": true,
         "cached_date": "2025-10-29T..."
       }
       // ... more companies with basic checks
     },
     "total_count": 100,
     "filtered_count": 100
   }
   ```

**Frontend (Immediately after response - ~2 seconds total):**
1. Receives 100 jobs
2. Renders first page (10 jobs) with minimal info:
   ```
   ┌─────────────────────────────────────────────────────────────┐
   │ [1-10 of 100 results]                    [< 1 2 3 ... 10 >] │
   ├─────────────────────────────────────────────────────────────┤
   │ Senior Software Engineer                                    │
   │ TechCorp Inc                                                │
   │ $120k-$150k | Remote | Posted 2 days ago                    │
   │ via Indeed, LinkedIn                                        │
   │ ✓ Domain validated                                          │
   │ [View Original] [🔍 Deep Research]                          │
   ├─────────────────────────────────────────────────────────────┤
   │ Software Developer                                          │
   │ StartupCo LLC                                               │
   │ Pay not listed | Remote | Posted 1 day ago                  │
   │ via Indeed                                                  │
   │ ✓ Domain validated                                          │
   │ [View Original] [🔍 Deep Research]                          │
   ├─────────────────────────────────────────────────────────────┤
   │ ... (8 more jobs)                                           │
   └─────────────────────────────────────────────────────────────┘
   ```
3. Jobs are immediately visible and interactive
4. Field renderer uses display metadata to format each field appropriately

**User Experience:**
- **Fast:** Results appear in ~2 seconds
- **Minimal:** Basic job info only (title, company, pay, location, source)
- **Manual research:** User must click "Deep Research" for expensive analysis

---

### 3. Quick Search with API Filters

**User Action:** User searches "software engineer" + checks "Glassdoor Rating (min: 4.0)"

**Frontend:**
1. Detects API-level filter is checked
2. Quick Search button remains visible (not AI, so still "quick")
3. Sends request:
   ```javascript
   POST /api/search
   Body: {
     query: "software engineer",
     min_glassdoor_rating: 4.0
   }
   Query: ?fields=glassdoor_rating
   ```

**Backend (5-15 seconds):**
1. Aggregates jobs from sources (1-3s)
2. Extracts unique companies (e.g., 80 unique companies from 100 jobs)
3. Calls `CompanyResearchService.research_companies(companies, ['glassdoor_rating'])`:
   - Checks cache for each company's `glassdoor_rating` field individually
   - For cache misses:
     - Identifies GlassdoorProvider as contributor of this field
     - Calls `GlassdoorProvider.research(company_name, ['glassdoor_rating'])`
     - Provider makes API call to Glassdoor
     - Returns `GlassdoorResearchResult(glassdoor_rating=4.3)`
   - All companies researched in parallel via `asyncio.gather()`
   - Caches each result individually per field
4. Filters jobs: removes any where `company.glassdoor_rating < 4.0`
5. Returns:
   ```json
   {
     "jobs": [...65 jobs...],
     "companies": {
       "TechCorp Inc": {
         "name": "TechCorp Inc",
         "glassdoor_rating": 4.3,
         "cached_date": "2025-10-29T..."
       },
       // ... more companies
     },
     "total_count": 100,
     "filtered_count": 65
   }
   ```

**Frontend:**
1. Shows filter stats:
   ```
   ℹ️ Showing 65 jobs (filtered from 100 total based on Glassdoor Rating ≥ 4.0)
   ```
2. Renders jobs with Glassdoor ratings visible using rating display type:
   ```
   ┌─────────────────────────────────────────────────────────────┐
   │ Senior Software Engineer                                    │
   │ TechCorp Inc                                                │
   │ ⭐⭐⭐⭐ 4.3/5 on Glassdoor                                 │
   │ $120k-$150k | Remote                                        │
   │ [View Original] [🔍 Deep Research]                          │
   └─────────────────────────────────────────────────────────────┘
   ```

**User Experience:**
- Slower than basic Quick Search (~10s vs ~2s)
- Results are pre-filtered
- Glassdoor data visible immediately
- Display format determined by provider's metadata

---

### 4. Deep Search Flow

**User Action:** User checks "Scam Detection" (AI category), clicks **Deep Search**

**Frontend (Immediate):**
1. Detects AI filter is active
2. Shows prominent loading indicator with progress context:
   ```
   ┌─────────────────────────────────────────────────────────────┐
   │ 🤖 Deep Research in Progress                                │
   │ This may take 30-60 seconds as we analyze companies...     │
   │                                                             │
   │ ⏳ [████████░░░░░░░░░░░░] 35%                              │
   │                                                             │
   │ Researching company reputations using AI...                │
   └─────────────────────────────────────────────────────────────┘
   ```
   (Note: Progress bar is optional, can show indeterminate spinner)
3. Shows one-time educational popup (if first Deep Search):
   ```
   ┌──────────────────────────────────────────────┐
   │ 💡 About Deep Research                       │
   │                                              │
   │ Deep Research uses AI to analyze companies   │
   │ for potential red flags. This process:       │
   │                                              │
   │ • Takes 30-60 seconds                        │
   │ • Analyzes all companies before showing      │
   │   results                                    │
   │ • Results are cached for 30 days             │
   │                                              │
   │ [Got it] [☐ Don't show again]                │
   └──────────────────────────────────────────────┘
   ```
4. Sends request:
   ```javascript
   POST /api/search
   Body: {
     query: "software engineer",
     max_scam_score: 0.5  // User wants scam score < 50%
   }
   Query: ?fields=domain_valid,job_age,glassdoor_rating,scam_score
   ```

**Backend (30-60 seconds):**
1. Aggregates jobs from sources (1-3s)
2. Extracts unique companies (e.g., 80 companies)
3. Calls `CompanyResearchService.research_companies(companies, ['domain_valid', 'job_age', 'glassdoor_rating', 'scam_score'])`:

   **For each company:**
   - Check cache for each requested field individually
   - Build list of cache misses
   - For each missing field, identify which provider(s) contribute it
   - Group providers and call in parallel

   **Example for "TechCorp Inc":**
   ```python
   # Cache check
   cached_fields = {'domain_valid': True, 'job_age': 2}  # Already cached
   missing_fields = {'glassdoor_rating', 'scam_score'}  # Need to fetch

   # Provider lookup
   providers_needed = [
       GlassdoorProvider,  # Contributes glassdoor_rating
       AIScamDetector      # Contributes scam_score
   ]

   # Parallel research
   results = await asyncio.gather(
       GlassdoorProvider.research('TechCorp Inc', ['glassdoor_rating']),
       AIScamDetector.research('TechCorp Inc', ['scam_score'])
   )
   # Returns: [
   #   GlassdoorResearchResult(glassdoor_rating=4.3),
   #   ScamDetectionResult(scam_score=0.15, scam_indicators=[])
   # ]

   # Merge and cache
   company.glassdoor_rating = 4.3  # Cache this field
   company.scam_score = 0.15       # Cache this field
   ```

   **AIScamDetector details:**
   - Makes OpenAI API call with prompt:
     ```
     Company: TechCorp Inc
     Domain: techcorp.com
     Analyze for scam indicators:
     - Domain age and legitimacy
     - Web presence quality
     - Common scam patterns
     - Contact information validity
     Return JSON: {"scam_score": 0.0-1.0, "indicators": [...]}
     ```
   - Parses response into `ScamDetectionResult`
   - Takes ~10-30 seconds per company

4. After all research complete, filter jobs:
   - Remove jobs where `company.scam_score > 0.5`
5. Return results:
   ```json
   {
     "jobs": [...85 jobs...],
     "companies": {
       "TechCorp Inc": {
         "name": "TechCorp Inc",
         "domain": "techcorp.com",
         "domain_valid": true,
         "glassdoor_rating": 4.3,
         "scam_score": 0.15,
         "scam_indicators": [],
         "cached_date": "2025-10-29T..."
       },
       "ShadyCo": {
         "name": "ShadyCo",
         "scam_score": 0.85,
         "scam_indicators": [
           "Domain registered 2 days ago",
           "No online presence",
           "Suspicious contact email"
         ]
       }
       // ... more companies
     },
     "total_count": 100,
     "filtered_count": 85
   }
   ```

**Frontend (After 30-60s wait):**
1. Hides loading indicator
2. Shows filter stats with timing:
   ```
   ✓ Deep Research complete in 47 seconds
   ℹ️ Showing 85 jobs (filtered from 100 total, 15 flagged as potential scams)
   ```
3. Renders jobs with full research data using display metadata:
   ```
   ┌─────────────────────────────────────────────────────────────┐
   │ [1-10 of 85 results]                     [< 1 2 3 ... 9 >]  │
   ├─────────────────────────────────────────────────────────────┤
   │ Senior Software Engineer                                    │
   │ TechCorp Inc                                                │
   │ 🛡️ Safe (15% risk)    [green badge via percentage display] │
   │ ⭐⭐⭐⭐ 4.3/5 on Glassdoor  [via rating display]          │
   │ $120k-$150k | Remote | Posted 2 days ago                    │
   │ [View Original] [🔍 Deep Research]                          │
   ├─────────────────────────────────────────────────────────────┤
   │ Software Developer                                          │
   │ StartupXYZ                                                  │
   │ 🛡️ Safe (23% risk)    [green badge]                        │
   │ ⭐⭐⭐⭐ 3.8/5 on Glassdoor                                 │
   │ $100k-$130k | Remote | Posted 5 days ago                    │
   │ [View Original] [🔍 Deep Research]                          │
   └─────────────────────────────────────────────────────────────┘
   ```
4. Any jobs filtered out (scam_score > 0.5) are not shown

**User Experience:**
- **Slow but comprehensive:** 30-60 second wait, but all analysis done upfront
- **Pre-filtered:** Only safe jobs shown (per user's criteria)
- **Rich data:** Scam scores, Glassdoor ratings all visible immediately
- **Transparent:** User knows exactly how long it took and what was filtered
- **Consistent display:** Metadata-driven rendering ensures uniform presentation

---

### 5. Manual Deep Research Button (from Quick Search)

**User Action:** User performs Quick Search, sees interesting job, clicks **🔍 Deep Research** button

**Frontend:**
1. Button changes to loading state:
   ```
   [⏳ Researching...]
   ```
2. Modal/overlay slides in from right (or pops up):
   ```
   ┌────────────────────────────────────────────────┐
   │ [X]                                            │
   │                                                │
   │ Deep Research: Senior Software Engineer        │
   │ TechCorp Inc                                   │
   │                                                │
   │ ⏳ Analyzing company with AI...                │
   │ This may take 10-30 seconds                    │
   │                                                │
   └────────────────────────────────────────────────┘
   ```
3. Sends request:
   ```javascript
   GET /api/research/company/{company_name}?fields=scam_score,glassdoor_rating,glassdoor_pros,glassdoor_cons,sentiment_score,salary_comparison
   ```

**Backend (10-30 seconds):**
1. Receives company name and requested fields
2. Calls `CompanyResearchService.research_companies([company], requested_fields)`:
   - Check cache for each field
   - For cache misses, identify contributing providers
   - Run expensive providers in parallel (AI, API calls)
   - Return comprehensive Company object with all requested fields
3. Returns:
   ```json
   {
     "name": "TechCorp Inc",
     "domain": "techcorp.com",
     "scam_score": 0.15,
     "scam_indicators": [],
     "glassdoor_rating": 4.3,
     "glassdoor_review_count": 127,
     "glassdoor_summary": "Employees praise work-life balance but note slow promotion track.",
     "glassdoor_pros": ["Great benefits", "Flexible hours", "Smart colleagues"],
     "glassdoor_cons": ["Slow raises", "Limited growth", "Old tech stack"],
     "sentiment_score": 0.72,
     "salary_comparison": {
       "posted": "$120k-$150k",
       "market_median": "$135k",
       "verdict": "competitive"
     },
     "cached_date": "2025-10-29T..."
   }
   ```

**Frontend (After 10-30s):**
1. Modal updates with full research results, using metadata-driven display:
   ```
   ┌──────────────────────────────────────────────────────────────┐
   │ [X]                                           [View Full Page]│
   │                                                              │
   │ Deep Research: Senior Software Engineer                      │
   │ TechCorp Inc                                                 │
   │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
   │                                                              │
   │ 🛡️ Safety Assessment                                        │
   │    Risk Level: LOW (15%)     [green badge - percentage]     │
   │    ✓ Domain validated                                        │
   │    ✓ Established web presence                                │
   │    ✓ Legitimate contact information                          │
   │                                                              │
   │ ⭐ Glassdoor Reviews (4.3/5 from 127 reviews)  [rating]     │
   │    Overall: Positive (72% sentiment)                         │
   │                                                              │
   │    👍 Common Pros:                              [list]       │
   │    • Great benefits                                          │
   │    • Flexible hours                                          │
   │    • Smart colleagues                                        │
   │                                                              │
   │    👎 Common Cons:                              [list]       │
   │    • Slow raises                                             │
   │    • Limited growth                                          │
   │    • Old tech stack                                          │
   │                                                              │
   │ 💰 Salary Analysis                         [custom or text] │
   │    Posted: $120k-$150k                                       │
   │    Market Median: $135k                                      │
   │    Verdict: ✓ Competitive                                    │
   │                                                              │
   │ [View Original Posting]                                      │
   └──────────────────────────────────────────────────────────────┘
   ```
2. Button returns to normal state:
   ```
   [🔍 Deep Research] (can click again to re-open modal)
   ```

**User Experience:**
- **On-demand:** Only research companies user is interested in
- **Fast for one job:** 10-30s is reasonable for detailed analysis of single company
- **Comprehensive:** Full deep research results in convenient modal
- **Bookmarkable:** Optional "View Full Page" link
- **Consistent display:** Same metadata-driven rendering as main results

---

### 6. Pagination

**User Action:** User clicks page 2

**Frontend:**
1. Checks if page 2 jobs are already loaded:
   - Quick Search mode: Yes, all 100 jobs loaded initially
   - Deep Search mode: Yes, all researched jobs returned
2. Scrolls to top
3. Renders jobs 11-20 from `state.jobs`
4. Uses same metadata-driven rendering for each field
5. No backend request needed

**Alternative (if implementing server-side pagination):**
```javascript
GET /api/search?page=2&fields=...
```
Backend returns jobs 11-20 with company research for those specific jobs.

**User Experience:**
- Instant pagination (client-side)
- All data already available
- Consistent display across pages

---

### 7. Duplicate Job Handling

**User Action:** User sees job posted on both Indeed and LinkedIn

**Backend Logic:**
```python
# During job aggregation
for job in all_jobs:
    group_id = job.generate_duplicate_hash()
    job.duplicate_group_id = group_id

    if group_id not in seen_groups:
        job.duplicate_sources = [job.source]
        seen_groups[group_id] = job
        unique_jobs.append(job)
    else:
        # Track that this job appears on multiple platforms
        seen_groups[group_id].duplicate_sources.append(job.source)
```

**Hash generation:**
```python
def generate_duplicate_hash(self) -> str:
    def normalize(text: str) -> str:
        # Keep only alphabetic characters
        return ''.join(c for c in text.lower() if c.isalpha())

    title_norm = normalize(self.title)
    company_norm = normalize(self.company)
    location_norm = normalize(self.location or "")
    desc_norm = normalize(self.description or "")[:50]  # First 50 alpha chars

    key = f"{company_norm}:{title_norm}:{location_norm}:{desc_norm}"
    return hashlib.md5(key.encode()).hexdigest()
```

**Frontend Display:**
```
┌─────────────────────────────────────────────────────────────┐
│ Senior Software Engineer                                    │
│ TechCorp Inc                                                │
│ $120k-$150k | Remote                                        │
│ 📋 Also on: Indeed, LinkedIn                                │
│ [View on Indeed] [View on LinkedIn] [🔍 Deep Research]      │
└─────────────────────────────────────────────────────────────┘
```

**User Experience:**
- See job once, not twice
- Know it's posted on multiple platforms (signal of legitimacy)
- Can choose which platform to apply through
- Reduces clutter in search results

---

### 8. Error Handling

#### Scenario A: Job Source Fails (e.g., Indeed API timeout)

**Backend:**
```python
search_tasks = [source.search(...) for source in sources]
results = await asyncio.gather(*search_tasks, return_exceptions=True)

all_jobs = []
for source, result in zip(sources, results):
    if isinstance(result, list):
        all_jobs.extend(result)
    else:
        logger.error(f"Source {source.source_name} failed: {result}")
```

**Frontend:**
Shows warning banner:
```
⚠️ Some job sources are currently unavailable. Results may be incomplete.
Showing 50 jobs from LinkedIn (Indeed temporarily unavailable)
```

**User Experience:**
- Partial results better than no results
- User informed about incomplete data
- Can retry later for complete results

#### Scenario B: Research Provider Fails (e.g., Glassdoor API down)

**Backend:**
```python
research_tasks = [provider.research(...) for provider in providers_needed]
research_results = await asyncio.gather(*research_tasks, return_exceptions=True)

for result in research_results:
    if isinstance(result, ResearchResult):
        company = company.merge_research(result)
    else:
        logger.warning(f"Provider failed: {result}")
```

**Frontend:**
```
┌─────────────────────────────────────────────────────────────┐
│ Senior Software Engineer                                    │
│ TechCorp Inc                                                │
│ ⚠️ Glassdoor data unavailable                               │
│ 🛡️ Safe (15% risk)                                         │
│ [View Original] [🔍 Deep Research]                          │
└─────────────────────────────────────────────────────────────┘
```

**User Experience:**
- Other research data still shown
- Clear indicator that specific data is missing
- Not a blocking error

#### Scenario C: Plugin Load Failure

**Backend startup with `require_all_providers=false` (default):**
```python
try:
    provider = module.get_provider()
    providers.append(provider)
    logger.info(f"Loaded provider: {provider_name}")
except Exception as e:
    logger.warning(f"Failed to load provider {provider_name}: {e}")
    # Continue with other providers
```

**Backend startup with `require_all_providers=true`:**
```python
try:
    provider = module.get_provider()
    providers.append(provider)
except Exception as e:
    logger.error(f"Provider load failure: {provider_name}")
    raise RuntimeError(f"Cannot start with require_all_providers=true") from e
```

**User Experience:**
- Graceful degradation: missing providers don't break the app
- Admin can configure strict mode if desired
- Logs provide debugging information

#### Scenario D: All Providers Fail (catastrophic)

**Backend:**
- If `require_all_providers=true`: Refuses to start, logs error
- If `require_all_providers=false`: Starts with 0 providers, logs warning

**Frontend:**
Shows error state:
```
┌─────────────────────────────────────────────────────────────┐
│ ⚠️ Research features temporarily unavailable                │
│                                                             │
│ You can still search for jobs, but company analysis        │
│ is currently offline. Please try again later.              │
│                                                             │
│ [Search Jobs Anyway]                                        │
└─────────────────────────────────────────────────────────────┘
```

**User Experience:**
- Core job search still functional
- Research temporarily disabled
- Clear communication about limitations

---

### 9. Caching Behavior

#### Scenario: User searches same company twice

**First search (cache miss):**
1. Backend checks cache: `CompanyCache` table, no entry for `(TechCorp Inc, scam_score)`
2. Queries AIScamDetector provider (10s)
3. Caches result:
   ```python
   CompanyCache(
       name="TechCorp Inc",
       field_name="scam_score",
       value=0.15,
       cached_at=datetime(2025, 10, 29, 10, 0, 0)
   )
   ```
4. Returns scam_score=0.15

**Second search 5 minutes later (cache hit):**
1. Backend checks cache: Entry found
2. Checks staleness: `(now - cached_at).days = 0 < 30` ✓
3. Returns cached scam_score=0.15 instantly (no AI call)

**User Experience:**
- First search: slow (AI analysis)
- Subsequent searches: instant (cached)
- Cache transparent to user
- Significant cost savings (OpenAI API calls)

#### Scenario: Partial cache hit

**User requests:** `?fields=glassdoor_rating,scam_score`

**Cache state:**
- `(TechCorp Inc, glassdoor_rating)`: Cached (1 day old)
- `(TechCorp Inc, scam_score)`: Missing

**Backend behavior:**
1. Load `glassdoor_rating` from cache
2. Query only AIScamDetector for `scam_score`
3. Cache new `scam_score` entry
4. Return merged Company object

**User Experience:**
- Faster than full research (only one provider queried)
- Optimal efficiency

#### Scenario: Cache expires after 30 days

**Search after 31 days:**
1. Backend checks cache: Entry found
2. Checks staleness: `(now - cached_at).days = 31 > 30` ✗
3. Queries AIScamDetector again (company may have changed)
4. Updates cache:
   ```python
   cache_entry.value = 0.22  # Score changed
   cache_entry.cached_at = datetime.now()
   ```

**User Experience:**
- Always get relatively fresh data
- Balance between speed and accuracy
- Company reputation can evolve over time

#### Scenario: Manual cache invalidation (future feature)

**Admin interface or CLI:**
```bash
# Clear specific company
jobsearch-cli cache clear "TechCorp Inc"

# Clear specific field across all companies
jobsearch-cli cache clear --field scam_score

# Clear stale cache (older than 30 days)
jobsearch-cli cache prune
```

---

### 10. Custom Display Logic (Advanced)

#### Scenario: Provider with custom frontend

**Example: Hypothetical "Floober" provider with complex visualization**

**Backend:**
```python
# providers/research/floober/provider.py
class FlooberProvider(ResearchProvider):
    def contributions(self) -> Dict[str, FieldContribution]:
        return {
            "floober_index": FieldContribution(
                category=ResearchCategory.API_CHEAP,
                label="Floober Index",
                display=DisplayMetadata(
                    type="custom",  # Signals custom frontend needed
                    priority="high",
                    custom_config={
                        "chart_type": "radar",
                        "dimensions": ["culture", "pay", "growth", "wlb", "tech"]
                    }
                )
            )
        }
```

**Frontend:**
```javascript
// providers/research/floober/frontend.js
export function render(fieldName, value, company, config) {
    if (fieldName === 'floober_index') {
        // value = {culture: 8, pay: 7, growth: 6, wlb: 9, tech: 5}
        return renderRadarChart(value, config.custom_config);
    }
}

function renderRadarChart(data, config) {
    // Complex D3.js or Chart.js radar chart
    const canvas = document.createElement('canvas');
    new Chart(canvas, {
        type: 'radar',
        data: {
            labels: config.dimensions,
            datasets: [{
                label: 'Floober Index',
                data: Object.values(data)
            }]
        }
    });
    return canvas.outerHTML;
}
```

**Frontend loading:**
```javascript
class FieldRenderer {
    async render(fieldName, value, company, metadata) {
        if (metadata.type === 'custom') {
            // Load custom provider JS
            const module = await import(`/api/providers/${metadata.provider}/frontend.js`);
            return module.render(fieldName, value, company, metadata.custom_config);
        }
        // ... standard rendering for other types
    }
}
```

**User Experience:**
- Most providers use standard displays (rating, percentage, list, etc.)
- Complex providers can supply custom visualizations
- Seamless integration via type="custom"
- Fallback to text display if custom JS fails to load

---

## Summary of Key UX Principles

1. **Progressive Disclosure:** Quick Search shows minimal data, Deep Research reveals everything
2. **Transparency:** Always tell user what's happening and how long it takes
3. **Graceful Degradation:** Partial data better than no data
4. **User Control:** Manual Deep Research button for on-demand analysis
5. **Speed Where Possible:** Cache aggressively, parallelize everything
6. **Clear Feedback:** Loading states, progress indicators, filter statistics
7. **Educational:** One-time popups explain delays and features
8. **Consistency:** Metadata-driven display ensures uniform presentation
9. **Extensibility:** New providers integrate seamlessly via field contributions
10. **Resilience:** Errors don't break the experience, just degrade gracefully

---

## Flow State Diagrams

### Quick Search State Flow
```
[User Enters Query]
    ↓
[Validate Input]
    ↓
[Show Loading State]
    ↓
[POST /api/search?fields=basic] ───→ [Backend: Aggregate Jobs]
    ↓ (1-3s)                              ↓
[Receive 100 Jobs]                   [Return Jobs]
    ↓
[Render Page 1 (10 jobs)]
    ↓
[User Interacts] ──→ [Click Deep Research] ──→ [Modal Flow]
                 ├─→ [Pagination] ──→ [Render Page 2]
                 └─→ [New Search]
```

### Deep Search State Flow
```
[User Checks AI Filter]
    ↓
[Click Deep Search]
    ↓
[Show Educational Popup] (first time only)
    ↓
[Show Loading: "30-60s wait"]
    ↓
[POST /api/search?fields=basic,api_cheap,ai] ───→ [Backend: Aggregate Jobs]
    ↓ (30-60s)                                         ↓
                                                  [Research All Companies]
                                                       ↓
                                                  [Cache Results]
                                                       ↓
                                                  [Filter Jobs]
                                                       ↓
[Receive Filtered Jobs + Company Data] ←──────────[Return]
    ↓
[Show Stats: "85/100 jobs, 47s"]
    ↓
[Render Jobs with Full Research]
    ↓
[User Interacts]
```

### Cache State Flow
```
[Request Company Research]
    ↓
[Check Cache for Each Field]
    ↓
    ├─→ [Cache Hit + Fresh] ──→ [Return Cached Value]
    │
    └─→ [Cache Miss OR Stale]
            ↓
        [Identify Contributing Providers]
            ↓
        [Query Providers in Parallel]
            ↓
        [Cache Each Result Individually]
            ↓
        [Return Fresh Values]
```

---

This UX flow ensures users have full control over speed vs. depth trade-offs, with transparent feedback at every step.
