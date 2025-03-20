import time
import functools
import json
import asyncio
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from src.utils.enhanced_logging import enhanced_api_logger
from src.utils.errors import APIRateLimitError, EmbeddingError

# Type variable for return type
T = TypeVar('T')

# Store API rate limit information
openai_rate_limits = {
    "embeddings": {
        "remaining": None,
        "reset_at": None,
        "limit": None,
        "last_updated": None
    },
    "chat": {
        "remaining": None,
        "reset_at": None,
        "limit": None,
        "last_updated": None
    }
}

# Track API call statistics
api_stats = {
    "total_calls": 0,
    "succeeded_calls": 0,
    "failed_calls": 0,
    "rate_limited_calls": 0,
    "total_duration_ms": 0,
    "avg_duration_ms": 0,
    "calls_by_endpoint": {},
    "errors_by_endpoint": {}
}

def update_rate_limits(
    endpoint: str,
    remaining: Optional[int] = None,
    reset_at: Optional[int] = None,
    limit: Optional[int] = None
) -> None:
    """
    Update rate limit information for an OpenAI endpoint.
    
    Args:
        endpoint: The endpoint (embeddings or chat)
        remaining: Remaining requests allowed
        reset_at: When the rate limit resets (Unix timestamp)
        limit: Total requests allowed in the window
    """
    if endpoint not in openai_rate_limits:
        # Create new entry if needed
        openai_rate_limits[endpoint] = {
            "remaining": None,
            "reset_at": None,
            "limit": None,
            "last_updated": None
        }
        
    if remaining is not None:
        openai_rate_limits[endpoint]["remaining"] = remaining
    if reset_at is not None:
        openai_rate_limits[endpoint]["reset_at"] = reset_at
    if limit is not None:
        openai_rate_limits[endpoint]["limit"] = limit
        
    openai_rate_limits[endpoint]["last_updated"] = time.time()
    
    # Log when rate limits are low
    if (remaining is not None and limit is not None and 
            remaining < 0.1 * limit and remaining > 0):
        enhanced_api_logger.warning(
            f"OpenAI rate limit for {endpoint} is getting low",
            endpoint=endpoint,
            remaining=remaining,
            limit=limit,
            reset_at=reset_at
        )
    elif remaining == 0:
        enhanced_api_logger.error(
            f"OpenAI rate limit for {endpoint} has been reached",
            endpoint=endpoint,
            limit=limit,
            reset_at=reset_at
        )

def extract_rate_limits_from_headers(headers: Dict[str, Any], endpoint: str) -> None:
    """
    Extract rate limit information from OpenAI response headers.
    
    Args:
        headers: Response headers from an OpenAI API call
        endpoint: The endpoint that was called (embeddings, chat, etc.)
    """
    # Try to get rate limit headers (naming might vary)
    remaining = headers.get('x-ratelimit-remaining') or headers.get('ratelimit-remaining')
    reset_at = headers.get('x-ratelimit-reset') or headers.get('ratelimit-reset')
    limit = headers.get('x-ratelimit-limit') or headers.get('ratelimit-limit')
    
    # Convert to correct types
    try:
        if remaining is not None:
            remaining = int(remaining)
        if reset_at is not None:
            reset_at = int(reset_at)
        if limit is not None:
            limit = int(limit)
            
        update_rate_limits(endpoint, remaining, reset_at, limit)
    except (ValueError, TypeError):
        # If parsing fails, just log and continue
        enhanced_api_logger.warning(
            f"Failed to parse rate limit headers for {endpoint}",
            headers=str(headers)
        )

def update_api_stats(
    endpoint: str,
    duration_ms: float,
    success: bool = True,
    rate_limited: bool = False
) -> None:
    """
    Update API call statistics.
    
    Args:
        endpoint: API endpoint called
        duration_ms: Call duration in milliseconds
        success: Whether the call succeeded
        rate_limited: Whether the call was rate limited
    """
    api_stats["total_calls"] += 1
    api_stats["total_duration_ms"] += duration_ms
    api_stats["avg_duration_ms"] = api_stats["total_duration_ms"] / api_stats["total_calls"]
    
    # Update endpoint-specific stats
    if endpoint not in api_stats["calls_by_endpoint"]:
        api_stats["calls_by_endpoint"][endpoint] = 0
    api_stats["calls_by_endpoint"][endpoint] += 1
    
    if success:
        api_stats["succeeded_calls"] += 1
    else:
        api_stats["failed_calls"] += 1
        if endpoint not in api_stats["errors_by_endpoint"]:
            api_stats["errors_by_endpoint"][endpoint] = 0
        api_stats["errors_by_endpoint"][endpoint] += 1
        
    if rate_limited:
        api_stats["rate_limited_calls"] += 1

def get_api_stats() -> Dict[str, Any]:
    """
    Get API call statistics.
    
    Returns:
        Dict[str, Any]: API call statistics
    """
    return api_stats.copy()

def get_rate_limits() -> Dict[str, Any]:
    """
    Get current rate limit information.
    
    Returns:
        Dict[str, Any]: Rate limit information by endpoint
    """
    return openai_rate_limits.copy()

def monitor_openai_call(endpoint: str) -> Callable:
    """
    Decorator to monitor OpenAI API calls.
    
    Args:
        endpoint: The endpoint being called (embeddings, chat, etc.)
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract rate limits from response headers if available
                if hasattr(result, 'headers'):
                    extract_rate_limits_from_headers(result.headers, endpoint)
                
                # Log the API call
                enhanced_api_logger.record_api_request(
                    api_name="OpenAI",
                    endpoint=endpoint,
                    duration_ms=duration_ms,
                    status_code=getattr(result, 'status_code', None)
                )
                
                # Update stats
                update_api_stats(endpoint, duration_ms)
                
                return result
                
            except Exception as e:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Check if rate limited
                rate_limited = "rate limit" in str(e).lower()
                
                # Update stats
                update_api_stats(
                    endpoint, 
                    duration_ms, 
                    success=False, 
                    rate_limited=rate_limited
                )
                
                # Log specific error
                if rate_limited:
                    # Extract retry after if available
                    retry_after = None
                    if hasattr(e, 'headers'):
                        retry_after = e.headers.get('retry-after')
                        if retry_after:
                            try:
                                retry_after = int(retry_after)
                            except (ValueError, TypeError):
                                retry_after = None
                                
                    error = APIRateLimitError(
                        f"OpenAI rate limit exceeded: {e}",
                        provider="openai",
                        retry_after=retry_after
                    )
                    
                    enhanced_api_logger.structured_error(
                        f"OpenAI rate limit exceeded for {endpoint}",
                        error=error,
                        endpoint=endpoint,
                        retry_after=retry_after
                    )
                elif endpoint == 'embeddings':
                    # Handle embedding-specific errors
                    input_length = None
                    if 'input' in kwargs and isinstance(kwargs['input'], str):
                        input_length = len(kwargs['input'])
                    elif 'input' in kwargs and isinstance(kwargs['input'], list):
                        input_length = sum(len(x) for x in kwargs['input'] if isinstance(x, str))
                        
                    error = EmbeddingError(
                        f"Embedding generation failed: {e}",
                        model=kwargs.get('model'),
                        input_length=input_length
                    )
                    
                    enhanced_api_logger.structured_error(
                        f"Embedding generation failed",
                        error=error,
                        endpoint=endpoint,
                        model=kwargs.get('model'),
                        input_length=input_length
                    )
                else:
                    # General API error
                    enhanced_api_logger.structured_error(
                        f"OpenAI API error for {endpoint}: {e}",
                        error=e,
                        endpoint=endpoint,
                        kwargs=str(kwargs)
                    )
                
                # Re-raise the original exception
                raise
                
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract rate limits from response headers if available
                if hasattr(result, 'headers'):
                    extract_rate_limits_from_headers(result.headers, endpoint)
                
                # Log the API call
                enhanced_api_logger.record_api_request(
                    api_name="OpenAI",
                    endpoint=endpoint,
                    duration_ms=duration_ms,
                    status_code=getattr(result, 'status_code', None)
                )
                
                # Update stats
                update_api_stats(endpoint, duration_ms)
                
                return result
                
            except Exception as e:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Check if rate limited
                rate_limited = "rate limit" in str(e).lower()
                
                # Update stats
                update_api_stats(
                    endpoint, 
                    duration_ms, 
                    success=False, 
                    rate_limited=rate_limited
                )
                
                # Log specific error
                if rate_limited:
                    # Extract retry after if available
                    retry_after = None
                    if hasattr(e, 'headers'):
                        retry_after = e.headers.get('retry-after')
                        if retry_after:
                            try:
                                retry_after = int(retry_after)
                            except (ValueError, TypeError):
                                retry_after = None
                                
                    error = APIRateLimitError(
                        f"OpenAI rate limit exceeded: {e}",
                        provider="openai",
                        retry_after=retry_after
                    )
                    
                    enhanced_api_logger.structured_error(
                        f"OpenAI rate limit exceeded for {endpoint}",
                        error=error,
                        endpoint=endpoint,
                        retry_after=retry_after
                    )
                elif endpoint == 'embeddings':
                    # Handle embedding-specific errors
                    input_length = None
                    if 'input' in kwargs and isinstance(kwargs['input'], str):
                        input_length = len(kwargs['input'])
                    elif 'input' in kwargs and isinstance(kwargs['input'], list):
                        input_length = sum(len(x) for x in kwargs['input'] if isinstance(x, str))
                        
                    error = EmbeddingError(
                        f"Embedding generation failed: {e}",
                        model=kwargs.get('model'),
                        input_length=input_length
                    )
                    
                    enhanced_api_logger.structured_error(
                        f"Embedding generation failed",
                        error=error,
                        endpoint=endpoint,
                        model=kwargs.get('model'),
                        input_length=input_length
                    )
                else:
                    # General API error
                    enhanced_api_logger.structured_error(
                        f"OpenAI API error for {endpoint}: {e}",
                        error=e,
                        endpoint=endpoint,
                        kwargs=str(kwargs)
                    )
                
                # Re-raise the original exception
                raise
                
        # Return appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator 