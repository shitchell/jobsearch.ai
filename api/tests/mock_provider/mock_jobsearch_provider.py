
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Read the example JSON data once at startup
try:
    with open("example.json", "r") as f:
        job_data = json.load(f)
except FileNotFoundError:
    job_data = {"error": "example.json not found"}
except json.JSONDecodeError:
    job_data = {"error": "Failed to decode example.json"}

@app.get("/jobsearch")
def search_jobs(q: str | None = None):
    """
    An endpoint to search for jobs.
    It ignores the query and returns the content of example.json.
    """
    # Using JSONResponse to ensure correct content-type header
    return JSONResponse(content=job_data)

# To run this app from the current directory (api/tests/mock_provider/):
# uvicorn mock_jobsearch_provider:app --reload --port 8888
