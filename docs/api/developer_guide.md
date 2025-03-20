# Developer Guide: API Component

This guide provides technical documentation for developers who need to integrate with, extend, or modify the API component of the Agentic RAG system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [API Endpoints](#api-endpoints)
4. [Integration Points](#integration-points)
5. [Authentication](#authentication)
6. [Extending the API](#extending-the-api)
7. [Best Practices](#best-practices)

## Architecture Overview

The API component is built using FastAPI to provide REST endpoints for interacting with the Agentic RAG system:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  API Routes     │       │   Application   │       │    Database     │
│  (FastAPI)      │──────▶│   Logic         │──────▶│    Layer        │
└─────────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Request        │       │   RAG System    │       │    OpenAI       │
│  Validation     │◀─────▶│   Integration   │◀─────▶│    API Client   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

The architecture follows these key principles:

1. **RESTful Design**: Clean, resource-oriented endpoints
2. **Schema Validation**: Strong typing with Pydantic models
3. **Asynchronous Processing**: Non-blocking operations for better performance
4. **Separation of Concerns**: Routes separated from business logic
5. **API Documentation**: Auto-generated OpenAPI documentation

## Key Components

### 1. FastAPI Application (`src/api/app.py`)

The main FastAPI application setup:

```python
from src.api.app import create_app

# Create the FastAPI application
app = create_app()

# Run the application with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. API Routes (`src/api/routes.py`)

Defines the API endpoints and their handlers:

```python
from src.api.routes import router as api_router

# Include the router in your FastAPI app
app.include_router(api_router, prefix="/api")
```

### 3. Pydantic Models (`src/models/pydantic_models.py`)

Data models for API request and response validation:

```python
from src.models.pydantic_models import (
    DocumentationSourceCreate,
    DocumentationSourceResponse,
    QueryRequest,
    QueryResponse
)

# Use models for validation
@router.post("/sources", response_model=DocumentationSourceResponse)
async def create_source(source: DocumentationSourceCreate):
    # Implementation
    pass
```

## API Endpoints

### Documentation Sources

Endpoints for managing documentation sources:

```python
# List all documentation sources
GET /api/sources

# Get a specific documentation source
GET /api/sources/{source_id}

# Create a new documentation source
POST /api/sources

# Delete a documentation source
DELETE /api/sources/{source_id}
```

Example request for creating a source:

```json
POST /api/sources
{
  "name": "Python Documentation",
  "sitemap_url": "https://docs.python.org/3/sitemap.xml",
  "configuration": {
    "chunk_size": 5000,
    "max_concurrent_requests": 5,
    "max_concurrent_api_calls": 3
  }
}
```

### RAG Queries

Endpoints for querying the RAG system:

```python
# Submit a query to the RAG system
POST /api/query

# Stream a response from the RAG system
GET /api/query/stream
```

Example query request:

```json
POST /api/query
{
  "query": "How do I install Python?",
  "source_id": "python_docs",
  "max_results": 5
}
```

### Crawling Operations

Endpoints for managing the crawling process:

```python
# Start a crawl operation
POST /api/crawl

# Get crawl status
GET /api/crawl/{crawl_id}

# Pause a crawl operation
PUT /api/crawl/{crawl_id}/pause

# Resume a crawl operation
PUT /api/crawl/{crawl_id}/resume

# Cancel a crawl operation
DELETE /api/crawl/{crawl_id}
```

Example crawl request:

```json
POST /api/crawl
{
  "source_id": "python_docs",
  "config": {
    "chunk_size": 5000,
    "max_concurrent_requests": 5
  }
}
```

## Integration Points

### Using the API Client

To use the API from external applications:

```python
import httpx

async def query_rag_system(query_text: str, source_id: str = None):
    """Query the RAG system through the API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/query",
            json={
                "query": query_text,
                "source_id": source_id,
                "max_results": 5
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")
```

### WebSocket Integration

For streaming responses:

```python
import websockets
import json

async def stream_query(query_text: str):
    """Stream a response from the RAG system."""
    async with websockets.connect("ws://localhost:8000/api/query/stream") as websocket:
        # Send query
        await websocket.send(json.dumps({
            "query": query_text
        }))
        
        # Stream response
        async for message in websocket:
            yield json.loads(message)
```

### Frontend Integration

For integrating with frontend frameworks:

```javascript
// Example using fetch API with React
async function queryRagSystem(query) {
  try {
    const response = await fetch('http://localhost:8000/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query,
        max_results: 5
      }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error querying RAG system:', error);
    throw error;
  }
}
```

## Authentication

### Implementing API Key Authentication

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

# Define API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# API key dependency
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API key"
        )
    return api_key

# Protect route with API key
@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    # Implementation
    pass
```

### JWT Authentication

For more advanced authentication:

```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from starlette.status import HTTP_401_UNAUTHORIZED

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Token dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        # Additional validation logic here
        return username
    except JWTError:
        raise credentials_exception

# Protect route with JWT
@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    current_user: str = Depends(get_current_user)
):
    # Implementation
    pass
```

## Extending the API

### Adding New Endpoints

To add a new API endpoint:

```python
from fastapi import APIRouter, Depends, HTTPException
from src.models.pydantic_models import CustomRequest, CustomResponse

router = APIRouter()

@router.post("/custom", response_model=CustomResponse)
async def custom_endpoint(request: CustomRequest):
    """
    Custom endpoint documentation.
    
    This will appear in the OpenAPI docs.
    """
    try:
        # Implementation logic
        result = await process_custom_request(request)
        return CustomResponse(
            status="success",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Custom Middleware

To add custom middleware:

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import time

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

# Add middleware to app
app = FastAPI()
app.add_middleware(TimingMiddleware)
```

### Background Tasks

For long-running operations:

```python
from fastapi import BackgroundTasks

@router.post("/crawl")
async def start_crawl(
    request: CrawlRequest,
    background_tasks: BackgroundTasks
):
    # Create crawl configuration
    config = CrawlConfig(
        source_id=request.source_id,
        source_name=request.source_name,
        sitemap_url=request.sitemap_url,
        # Other settings
    )
    
    # Add to background tasks
    background_tasks.add_task(crawl_documentation, config)
    
    return {"status": "crawl started", "crawl_id": config.source_id}
```

## Best Practices

### API Design

1. **Resource-Oriented Design**:
   - Use nouns, not verbs, for resource endpoints
   - Use HTTP methods appropriately (GET, POST, PUT, DELETE)
   - Group related operations under consistent resources

2. **Versioning**:
   - Implement versioning to make non-backward compatible changes
   - Use URL path for version: `/api/v1/resources`
   - Or use headers: `Accept: application/vnd.rag.v1+json`

3. **Error Handling**:
   - Return consistent error responses
   - Include error codes, messages, and details
   - Use appropriate HTTP status codes

```python
# Consistent error response
@router.get("/sources/{source_id}")
async def get_source(source_id: str):
    try:
        source = await get_documentation_source(source_id)
        if not source:
            raise HTTPException(
                status_code=404,
                detail=f"Source {source_id} not found"
            )
        return source
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
```

### Performance Optimization

1. **Async Operations**:
   - Use async/await for I/O bound operations
   - Avoid blocking code in request handlers
   - Consider using background tasks for long operations

2. **Response Optimization**:
   - Use pagination for large result sets
   - Implement response compression
   - Limit response fields with query parameters

3. **Caching**:
   - Add caching headers for static responses
   - Implement response caching for expensive operations
   - Use Redis or in-memory cache for frequent queries

```python
from fastapi import Response

@router.get("/sources", response_model=List[DocumentationSourceResponse])
async def list_sources(response: Response):
    # Add caching headers
    response.headers["Cache-Control"] = "max-age=60"
    
    sources = await get_documentation_sources()
    return sources
```

### Security Best Practices

1. **Input Validation**:
   - Use Pydantic models for request validation
   - Implement additional validation logic for complex rules
   - Sanitize inputs to prevent injection attacks

2. **Rate Limiting**:
   - Implement rate limiting for public endpoints
   - Use client IP or API key for tracking
   - Set appropriate limits based on endpoint complexity

3. **Security Headers**:
   - Set appropriate security headers
   - Implement CORS with specific origins
   - Consider using an API gateway for additional security

```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Documentation

1. **OpenAPI Documentation**:
   - Add detailed docstrings to route handlers
   - Include examples in Pydantic models
   - Use tags to organize API operations

2. **Model Documentation**:
   - Add field descriptions in Pydantic models
   - Include example values for clear expectations
   - Document constraints and validation rules

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="The question to ask the RAG system",
        example="How do I install Python?"
    )
    source_id: Optional[str] = Field(
        None,
        description="Optional source ID to limit the search",
        example="python_docs"
    )
    max_results: int = Field(
        5,
        description="Maximum number of results to return",
        ge=1,
        le=20
    )
``` 