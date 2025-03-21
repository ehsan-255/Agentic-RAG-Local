# Developer Guide: API Component

This guide provides technical documentation for developers who need to integrate with, extend, or modify the API component of the Agentic RAG system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [API Endpoints](#api-endpoints)
4. [Authentication](#authentication)
5. [Error Handling](#error-handling)
6. [Extending the API](#extending-the-api)
7. [Best Practices](#best-practices)

## Architecture Overview

The API component is built using FastAPI and provides a standardized interface for interacting with the Agentic RAG system:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  API Endpoints  │─────▶│  Service Layer  │─────▶│  RAG System     │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        │
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  Middleware     │◀────▶│  Database       │◀────▶│  Document       │
│                 │      │  Access         │      │  Crawler        │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

The architecture follows these key principles:

1. **REST API Design**: Clean, resource-oriented endpoints
2. **Asynchronous Operations**: Non-blocking API endpoints using async/await
3. **Structured Response Format**: Consistent JSON response formats
4. **Middleware Pipeline**: Extensible middleware for cross-cutting concerns
5. **OpenAPI Documentation**: Auto-generated API documentation
6. **Error Standardization**: Consistent error handling and reporting

## Key Components

### 1. FastAPI Application (`src/api/app.py`)

The main FastAPI application that configures routes, middleware, and dependencies:

```python
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import query, documents, metrics, health
from src.api.middleware import logging_middleware, rate_limiting_middleware
from src.api.auth import get_current_user

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="API for the Agentic RAG system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(logging_middleware)
app.middleware("http")(rate_limiting_middleware)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(query.router, prefix="/api/query", tags=["Query"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])

# Optional auth dependency for protected routes
# app.include_router(admin.router, prefix="/api/admin", tags=["Admin"], dependencies=[Depends(get_current_user)])
```

### 2. API Routers (`src/api/routers/`)

Organized endpoint definitions for different resource types:

```python
# src/api/routers/query.py
from fastapi import APIRouter, Depends, HTTPException
from src.api.models import QueryRequest, QueryResponse
from src.api.dependencies import get_rag_deps
from src.rag.rag_expert import agentic_rag_expert

router = APIRouter()

@router.post("", response_model=QueryResponse)
async def process_query(request: QueryRequest, deps = Depends(get_rag_deps)):
    """Process a user query through the RAG system."""
    try:
        # Convert conversation format if provided
        conversation_history = []
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        # Get contexts for the query
        contexts = await deps.context_manager.get_context_for_query(
            request.query,
            conversation_history=conversation_history,
            filter_metadata=request.filter_metadata,
            search_strategy=request.search_strategy
        )
        
        # Process the query with RAG expert
        response = await agentic_rag_expert(
            query=request.query,
            contexts=contexts,
            deps=deps,
            config={
                "temperature": request.temperature,
                "use_tools": request.use_tools,
                "conversation_history": conversation_history
            }
        )
        
        return QueryResponse(
            response=response,
            contexts=[c.dict() for c in contexts],
            metadata={
                "context_count": len(contexts),
                "search_strategy": request.search_strategy
            }
        )
    except Exception as e:
        # Log the error
        deps.logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")
```

### 3. Data Models (`src/api/models.py`)

Pydantic models for request and response validation:

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class MessageModel(BaseModel):
    """Model for a conversation message."""
    role: str = Field(..., description="The role of the message (user or assistant)")
    content: str = Field(..., description="The content of the message")

class QueryRequest(BaseModel):
    """Model for a query request."""
    query: str = Field(..., description="The user's query")
    conversation_history: Optional[List[MessageModel]] = Field(
        default=None, 
        description="Optional conversation history"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional metadata filters"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for response generation"
    )
    use_tools: bool = Field(
        default=True,
        description="Whether to use tools for response generation"
    )
    search_strategy: str = Field(
        default="hybrid",
        description="The search strategy to use (hybrid, vector, keyword)"
    )

class ContextModel(BaseModel):
    """Model for a context document."""
    text: str = Field(..., description="The text content")
    source: str = Field(..., description="The source of the context")
    url: Optional[str] = Field(None, description="Original URL if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    
class QueryResponse(BaseModel):
    """Model for a query response."""
    response: str = Field(..., description="The generated response")
    contexts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="The contexts used for generating the response"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the processing"
    )
```

### 4. Dependencies (`src/api/dependencies.py`)

Reusable dependencies for FastAPI route handlers:

```python
from fastapi import Depends, HTTPException, status, Request
from src.rag.context_manager import ContextManager
from src.rag.agents.rag_agent import RagAgentDeps
from src.monitoring.logger import get_logger
from src.api.auth import get_api_key

async def get_rag_deps(request: Request, api_key: str = Depends(get_api_key)):
    """Get dependencies for RAG operations."""
    # Create context manager
    context_manager = ContextManager()
    
    # Create logger
    logger = get_logger("api")
    
    # Create RAG dependencies
    deps = RagAgentDeps(
        context_manager=context_manager,
        logger=logger,
        request_id=request.state.request_id if hasattr(request.state, "request_id") else None
    )
    
    return deps

async def get_db_connection():
    """Get a database connection."""
    from src.db.connection import get_db_connection as get_conn
    conn = await get_conn()
    try:
        yield conn
    finally:
        await conn.close()
```

## API Endpoints

### Query API

Used for processing natural language queries:

```
POST /api/query
```

Request body:

```json
{
  "query": "How does vector search work in the system?",
  "conversation_history": [
    {"role": "user", "content": "Tell me about the RAG system"},
    {"role": "assistant", "content": "The RAG system combines retrieval and generation..."}
  ],
  "filter_metadata": {
    "source_id": "docs-123"
  },
  "temperature": 0.7,
  "use_tools": true,
  "search_strategy": "hybrid"
}
```

Response:

```json
{
  "response": "Vector search in the system works by converting text into high-dimensional vectors using embedding models. These vectors capture semantic meaning, allowing the system to find relevant documents by measuring vector similarity. The system uses pgvector in PostgreSQL to efficiently store and query these vectors.",
  "contexts": [
    {
      "text": "Vector search is implemented using pgvector extension in PostgreSQL. Document text is converted to embeddings using OpenAI's text-embedding-3-small model.",
      "source": "Vector Search Documentation",
      "url": "https://example.com/docs/vector-search",
      "metadata": {
        "source_id": "docs-123",
        "similarity": 0.92
      }
    }
  ],
  "metadata": {
    "context_count": 1,
    "search_strategy": "hybrid"
  }
}
```

### Documents API

Used for managing documentation sources:

```
GET /api/documents
```

Response:

```json
[
  {
    "source_id": "docs-123",
    "name": "System Documentation",
    "base_url": "https://example.com/docs",
    "pages_count": 150,
    "chunks_count": 450,
    "created_at": "2023-04-15T10:30:00Z",
    "updated_at": "2023-04-16T08:15:00Z"
  }
]
```

```
POST /api/documents
```

Request body:

```json
{
  "name": "API Documentation",
  "sitemap_url": "https://example.com/sitemap.xml",
  "configuration": {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "url_patterns_include": ["api/v1", "reference"],
    "url_patterns_exclude": ["blog", "changelog"]
  }
}
```

Response:

```json
{
  "source_id": "docs-456",
  "name": "API Documentation",
  "status": "crawling_initiated",
  "job_id": "crawl-789"
}
```

```
DELETE /api/documents/{source_id}
```

Response:

```json
{
  "success": true,
  "message": "Documentation source deleted",
  "deleted_source_id": "docs-123",
  "deleted_pages": 150
}
```

### Metrics API

Used for accessing system metrics:

```
GET /api/metrics
```

Response:

```json
{
  "system": {
    "uptime_hours": 24.5,
    "cpu_usage": 35.2,
    "memory_usage": 42.8
  },
  "rag": {
    "queries_total": 1250,
    "queries_today": 127,
    "avg_response_time_ms": 450
  },
  "documents": {
    "sources_count": 5,
    "pages_count": 750,
    "chunks_count": 2250
  },
  "errors": {
    "total_errors": 12,
    "error_rate": 0.01
  }
}
```

### Health API

Used for checking system health:

```
GET /health
```

Response:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "api": "healthy",
    "database": "healthy",
    "rag": "healthy",
    "openai": "healthy"
  },
  "timestamp": "2023-04-16T12:00:00Z"
}
```

## Authentication

### API Key Authentication

The API supports API key authentication:

```python
from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from src.config import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Validate API key from header."""
    if not settings.API_KEY_REQUIRED:
        return None
        
    if api_key_header is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing"
        )
        
    if api_key_header != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
        
    return api_key_header
```

### JWT Authentication

For more advanced authentication scenarios, JWT authentication is available:

```python
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from src.config import settings
from src.api.models import TokenData, User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

async def get_current_user(token: str = Security(oauth2_scheme)):
    """Get the current user from JWT token."""
    if not settings.AUTH_REQUIRED:
        return None
        
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
        
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        token_data = TokenData(username=username)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
        
    user = get_user(username=token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
        
    return user
```

## Error Handling

### Standardized Error Responses

The API uses standardized error responses:

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from src.monitoring.logger import get_logger

logger = get_logger("api")

async def exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standardized format."""
    # Get request ID from state if available
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log the error
    logger.error(
        f"HTTP exception: {exc.detail}",
        status_code=exc.status_code,
        request_id=request_id,
        path=request.url.path
    )
    
    # Return standardized error response
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "request_id": request_id
            }
        }
    )

# Register the exception handler in app.py
app.add_exception_handler(HTTPException, exception_handler)
```

### Error Logging

Errors are logged with contextual information:

```python
from src.monitoring.logger import get_logger

logger = get_logger("api")

@router.post("/documents")
async def add_document_source(request: DocumentSourceCreate):
    try:
        # Process the request
        result = await create_documentation_source(request)
        return result
    except ValueError as e:
        # Log with context
        logger.structured_error(
            "Invalid document source data",
            error=str(e),
            error_type="ValueError",
            request_data=request.dict()
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors
        logger.structured_error(
            "Failed to create document source",
            error=str(e),
            error_type=type(e).__name__,
            request_data=request.dict()
        )
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Extending the API

### 1. Adding a New Endpoint

To add a new endpoint:

1. Create a new router module or extend an existing one
2. Define the endpoint with appropriate HTTP method and path
3. Add request and response models
4. Implement the handler function

Example of adding a new endpoint:

```python
# src/api/routers/feedback.py
from fastapi import APIRouter, Depends, HTTPException
from src.api.models import FeedbackRequest, FeedbackResponse
from src.api.dependencies import get_db_connection
from src.monitoring.logger import get_logger

router = APIRouter()
logger = get_logger("api.feedback")

@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    conn = Depends(get_db_connection)
):
    """Submit user feedback for a query-response pair."""
    try:
        # Save feedback to database
        feedback_id = await save_feedback(conn, request)
        
        # Log the feedback
        logger.info("Feedback submitted",
            feedback_id=feedback_id,
            query_id=request.query_id,
            rating=request.rating
        )
        
        # Update metrics based on feedback
        if request.rating >= 4:  # Positive feedback
            await update_positive_feedback_metrics(request.query_id)
        elif request.rating <= 2:  # Negative feedback
            await update_negative_feedback_metrics(request.query_id)
            
            # For very negative feedback, trigger additional analysis
            if request.rating == 1 and request.comments:
                await trigger_feedback_analysis(request)
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            success=True,
            message="Feedback received, thank you!"
        )
    except Exception as e:
        logger.structured_error("Failed to save feedback",
            error=str(e),
            error_type=type(e).__name__,
            request_data=request.dict(exclude={"comments"})
        )
        raise HTTPException(status_code=500, detail="Failed to save feedback")
```

Update `app.py` to include the new router:

```python
from src.api.routers import feedback

app.include_router(feedback.router, prefix="/api/feedback", tags=["Feedback"])
```

### 2. Adding Custom Middleware

To add custom middleware:

```python
# src/api/middleware/request_id_middleware.py
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add a unique request ID to each request."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Add it to request state
        request.state.request_id = request_id
        
        # Process the request
        response = await call_next(request)
        
        # Add the request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
```

Register the middleware in `app.py`:

```python
from src.api.middleware.request_id_middleware import RequestIDMiddleware

app.add_middleware(RequestIDMiddleware)
```

### 3. Adding Real-time Events with WebSockets

To add WebSocket support for real-time events:

```python
# src/api/routers/events.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from src.api.dependencies import get_current_user
from src.api.models import User
from src.monitoring.logger import get_logger
from typing import Dict, Set

router = APIRouter()
logger = get_logger("api.events")

# Store active connections
active_connections: Dict[str, Set[WebSocket]] = {
    "admin": set(),
    "user": set()
}

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    user: User = Depends(get_current_user)
):
    """WebSocket endpoint for real-time events."""
    # Reject connection if not authenticated
    if user is None:
        await websocket.close(code=1008)  # Policy violation
        return
    
    # Accept the connection
    await websocket.accept()
    
    # Add to appropriate connection group
    group = "admin" if user.is_admin else "user"
    active_connections[group].add(websocket)
    
    try:
        # Main message loop
        while True:
            data = await websocket.receive_text()
            # Process received data if needed
            
            # Echo back acknowledgment
            await websocket.send_json({
                "type": "ack",
                "message": "Message received"
            })
    except WebSocketDisconnect:
        # Remove from active connections on disconnect
        active_connections[group].remove(websocket)
    except Exception as e:
        logger.structured_error("WebSocket error",
            error=str(e),
            error_type=type(e).__name__,
            user_id=user.id
        )
        if websocket in active_connections[group]:
            active_connections[group].remove(websocket)

# Function to broadcast events to clients
async def broadcast_event(event_type: str, data: dict, group: str = None):
    """Broadcast an event to connected clients.
    
    Args:
        event_type: Type of event
        data: Event data
        group: Optional group to target (admin, user, or None for all)
    """
    event = {
        "type": event_type,
        "data": data
    }
    
    groups = [group] if group else active_connections.keys()
    
    for g in groups:
        for connection in active_connections.get(g, set()):
            try:
                await connection.send_json(event)
            except Exception:
                # Connection might be closed
                pass
```

## Best Practices

### 1. API Design

Follow these principles for API design:

```python
# DO: Use semantic HTTP methods
@router.get("/documents")       # List/read resources
@router.post("/documents")      # Create a resource
@router.get("/documents/{id}")  # Read a specific resource
@router.put("/documents/{id}")  # Update (replace) a resource
@router.patch("/documents/{id}") # Partially update a resource
@router.delete("/documents/{id}") # Delete a resource

# DO: Use appropriate status codes
@router.post("/documents", status_code=201)  # Created
@router.delete("/documents/{id}", status_code=204)  # No content

# DO: Use query parameters for filtering, sorting, pagination
@router.get("/documents")
async def list_documents(
    source_type: Optional[str] = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    page: int = 1,
    page_size: int = 50
):
    pass

# DO: Use request and response models
@router.post("/documents", response_model=DocumentResponse)
async def create_document(document: DocumentCreate):
    pass

# DON'T: Use verbs in URLs
# @router.post("/getDocuments")  # BAD - Use /documents with GET
# @router.post("/createDocument")  # BAD - Use /documents with POST

# DON'T: Return different data structures for success/error
# if error:
#     return {"error": "Message"}  # BAD - Inconsistent with success response
# else:
#     return {"data": {...}}  # BAD - Inconsistent with error response
```

### 2. Performance Considerations

Optimize API performance:

```python
# DO: Use async operations
@router.get("/documents")
async def list_documents():
    # Async database query
    documents = await get_documents()
    return documents

# DO: Implement pagination
@router.get("/documents")
async def list_documents(
    page: int = 1,
    page_size: int = 50
):
    # Limit number of results
    documents = await get_documents_paginated(page, page_size)
    
    # Include pagination metadata
    return {
        "items": documents,
        "total": await get_documents_count(),
        "page": page,
        "page_size": page_size,
        "pages": ceil(await get_documents_count() / page_size)
    }

# DO: Use connection pooling
async def get_db_pool():
    """Get the database connection pool."""
    # Use a global pool
    global db_pool
    if db_pool is None:
        db_pool = await create_pool()
    return db_pool

# DON'T: Block the event loop
# time.sleep(1)  # BAD - Blocks the thread
# Use async sleep instead if needed
# await asyncio.sleep(1)  # Better
```

### 3. Error Handling

Implement robust error handling:

```python
# DO: Use try-except blocks properly
@router.post("/documents")
async def create_document(document: DocumentCreate):
    try:
        # Attempt to create the document
        result = await create_document_in_db(document)
        return result
    except DuplicateDocumentError as e:
        # Handle specific business logic errors
        raise HTTPException(status_code=409, detail=str(e))
    except ValidationError as e:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors
        logger.structured_error("Document creation failed",
            error=str(e),
            error_type=type(e).__name__,
            document_name=document.name
        )
        # Don't expose internal error details
        raise HTTPException(status_code=500, detail="Internal server error")

# DO: Use appropriate HTTP status codes
# 400 Bad Request - Client error, invalid format
# 401 Unauthorized - Missing authentication
# 403 Forbidden - Authentication valid, but insufficient permissions
# 404 Not Found - Resource doesn't exist
# 409 Conflict - Request conflicts with current state
# 422 Unprocessable Entity - Validation error
# 429 Too Many Requests - Rate limit exceeded
# 500 Internal Server Error - Server error

# DON'T: Return stack traces to clients
# except Exception as e:
#     return JSONResponse(  # BAD - Exposes internal details
#         status_code=500,
#         content={"error": str(e), "traceback": traceback.format_exc()}
#     )
```

### 4. Security Best Practices

Implement security best practices:

```python
# DO: Validate and sanitize all inputs
from pydantic import BaseModel, Field, validator

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

# DO: Use parameterized queries
async def get_user_by_id(user_id: str):
    # Use parameterized query to prevent SQL injection
    query = "SELECT * FROM users WHERE user_id = $1"
    return await db.fetch_one(query, user_id)

# DO: Rate limiting
from fastapi import Request, Response, HTTPException
import time
from collections import defaultdict

# Simple in-memory rate limiter
request_counts = defaultdict(list)

async def rate_limiting_middleware(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit (100 requests per minute)
    now = time.time()
    request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < 60]
    
    if len(request_counts[client_ip]) >= 100:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Add this request
    request_counts[client_ip].append(now)
    
    # Process the request
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-Rate-Limit-Limit"] = "100"
    response.headers["X-Rate-Limit-Remaining"] = str(100 - len(request_counts[client_ip]))
    response.headers["X-Rate-Limit-Reset"] = str(int(60 - (now - min(request_counts[client_ip])) if request_counts[client_ip] else 0))
    
    return response

# DON'T: Store sensitive data in logs
# logger.info(f"User {username} logged in with password {password}")  # BAD

# DON'T: Use hard-coded secrets
# api_key = "sk-1234567890"  # BAD
# Use environment variables or a secure secret manager
```

### 5. Testing

Implement comprehensive testing:

```python
# src/api/tests/test_query.py
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_query_endpoint():
    """Test the query endpoint."""
    response = client.post(
        "/api/query",
        json={
            "query": "How does vector search work?",
            "temperature": 0.7,
            "use_tools": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 0
    assert "contexts" in data
    
def test_query_invalid_input():
    """Test the query endpoint with invalid input."""
    # Missing required field
    response = client.post(
        "/api/query",
        json={
            "temperature": 0.7
        }
    )
    
    assert response.status_code == 422  # Validation error
    
def test_query_error_handling():
    """Test error handling in the query endpoint."""
    # Use a mock to simulate an error
    with patch("src.rag.rag_expert.agentic_rag_expert") as mock_rag:
        mock_rag.side_effect = Exception("Simulated error")
        
        response = client.post(
            "/api/query",
            json={
                "query": "How does vector search work?",
                "temperature": 0.7
            }
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        assert "Internal server error" in data["error"]["message"]