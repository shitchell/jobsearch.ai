# Job Search AI

An AI-powered job search aggregation platform that intelligently searches, filters, and analyzes job postings from multiple sources, with automated application assistance and company research capabilities.

## Features

- **Multi-Source Job Aggregation**: Search across Indeed, LinkedIn, Glassdoor, ZipRecruiter, and more
- **Intelligent Filtering**: AI-powered relevance scoring and deduplication
- **Company Research**: Automatic gathering of company information and culture insights
- **Application Tracking**: Track application status and follow-ups
- **Resume Optimization**: AI suggestions for tailoring resumes to specific positions
- **Cover Letter Generation**: Context-aware cover letter creation

## Installation

### Prerequisites

- Python 3.10 or higher
- SQLite (included with Python)
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shitchell/jobsearch.ai.git
cd jobsearch.ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
cd api
pip install -r requirements.txt
```

4. Configure the application:
```bash
cp config.ini.example config.ini
# Edit config.ini and add your API keys
```

## Configuration

The application uses a simple INI configuration file. Copy `api/config.ini.example` to `api/config.ini` and update with your settings:

```ini
[api]
host = 0.0.0.0
port = 8000

[jobsearch]
db_url = sqlite:///./jobsearch.db
cache_ttl_days = 30

# Add your API keys here
openai_api_key = your-key-here
indeed_api_key = your-key-here
glassdoor_api_key = your-key-here
```

### API Keys Required

- **OpenAI API Key**: For AI-powered features (resume optimization, cover letters)
- **Indeed API Key**: For job search from Indeed
- **Glassdoor API Key**: For company reviews and salary data
- **LinkedIn API Key**: For LinkedIn job postings (optional)
- **ZipRecruiter API Key**: For ZipRecruiter listings (optional)

## Usage

### Starting the API Server

```bash
cd api
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Running Tests

```bash
cd api
pytest
```

## Project Structure

```
jobsearch.ai/
├── api/                    # Backend API
│   ├── app/               # Main application package
│   ├── tests/             # Test suite
│   ├── requirements.txt   # Python dependencies
│   └── config.ini.example # Configuration template
├── frontend/              # Frontend application (future)
└── docs/                  # Documentation
```

## Development Status

This project is currently in active development. Core features being implemented:

- [x] Project setup and dependencies
- [ ] Database models and migrations
- [ ] Core API endpoints
- [ ] Job source providers
- [ ] AI integration
- [ ] Frontend application

## Contributing

Contributions are welcome! Please see the documentation in the `docs/` directory for development guidelines and architecture details.

## License

[License information to be added]