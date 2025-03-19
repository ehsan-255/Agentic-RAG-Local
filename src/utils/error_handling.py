"""
Error handling utilities for the Agentic RAG system.

This module provides standardized error handling functions to ensure consistent
error responses across the application without exposing sensitive information.
"""

import uuid
import sys
from typing import Dict, Any, Optional
from fastapi import HTTPException

from src.utils.logging import get_logger

# Initialize logger
logger = get_logger("error_handling")

class ErrorCode:
    """Standardized error codes for the application."""
    GENERIC_ERROR = "GENERIC_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND = "NOT_FOUND"

def handle_api_error(exception: Exception, user_message: str = None, error_code: str = ErrorCode.GENERIC_ERROR) -> HTTPException:
    """
    Standardized API error handler that logs detailed error information but
    returns a sanitized error response to the user.
    
    Args:
        exception: The exception that was caught
        user_message: A user-friendly error message (optional)
        error_code: A specific error code for categorization (optional)
        
    Returns:
        HTTPException: A FastAPI HTTPException with sanitized details
    """
    # Generate a unique error ID for traceability
    error_id = str(uuid.uuid4())
    
    # Use the provided user message or a generic one
    if not user_message:
        user_message = "An unexpected error occurred. Please try again later."
    
    # Log the detailed error with full context
    logger.error(
        f"Error {error_id}: {str(exception)}",
        exc_info=exception,
        error_id=error_id,
        error_code=error_code
    )
    
    # Return a sanitized HTTPException
    return HTTPException(
        status_code=500,
        detail={
            "error_id": error_id,
            "message": user_message,
            "code": error_code
        }
    )

def handle_validation_error(message: str) -> HTTPException:
    """
    Handle validation errors with a 400 Bad Request status code.
    
    Args:
        message: Validation error message
        
    Returns:
        HTTPException: A FastAPI HTTPException for validation errors
    """
    return HTTPException(
        status_code=400,
        detail={
            "message": message,
            "code": ErrorCode.VALIDATION_ERROR
        }
    )

def handle_not_found_error(resource_type: str, identifier: str) -> HTTPException:
    """
    Handle not found errors with a 404 Not Found status code.
    
    Args:
        resource_type: Type of resource that wasn't found (e.g., "document", "source")
        identifier: Identifier used to look up the resource
        
    Returns:
        HTTPException: A FastAPI HTTPException for not found errors
    """
    return HTTPException(
        status_code=404,
        detail={
            "message": f"{resource_type} not found: {identifier}",
            "code": ErrorCode.NOT_FOUND
        }
    )

def safe_error_response(exception: Exception) -> Dict[str, Any]:
    """
    Create a safe error response for non-API contexts (e.g., Streamlit app).
    
    Args:
        exception: The exception that was caught
        
    Returns:
        Dict[str, Any]: A dictionary with sanitized error information
    """
    # Generate a unique error ID for traceability
    error_id = str(uuid.uuid4())
    
    # Log the detailed error
    logger.error(
        f"Error {error_id}: {str(exception)}",
        exc_info=exception,
        error_id=error_id
    )
    
    # Return a sanitized error response
    return {
        "error_id": error_id,
        "message": "An unexpected error occurred. Please try again later."
    } 