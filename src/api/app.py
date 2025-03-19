"""Main FastAPI application for the Agentic RAG."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback

from src.api.routes import router
from src.utils.error_handling import handle_api_error, ErrorCode

app = FastAPI(
    title="Agentic RAG API",
    description="API for Agentic RAG with local document processing",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")

# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to catch all unhandled exceptions.
    Ensures that no sensitive error details are exposed to clients.
    """
    # Use our centralized error handling utility
    http_exception = handle_api_error(
        exception=exc,
        user_message="An unexpected error occurred. Please try again later.",
        error_code=ErrorCode.GENERIC_ERROR
    )
    
    # Return a JSON response with the sanitized error details
    return JSONResponse(
        status_code=http_exception.status_code,
        content=http_exception.detail
    )

# HTTP exception handler for all HTTP exceptions
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handler for HTTP exceptions to ensure consistent error format.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": "HTTP_ERROR",
            "message": str(exc.detail)
        }
    )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Agentic RAG API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True) 