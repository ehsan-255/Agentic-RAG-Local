# Developer Guide: Utils Component

This guide provides technical documentation for developers who need to integrate with, extend, or modify the utilities component of the Agentic RAG system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Integration Points](#integration-points)
4. [Error Handling](#error-handling)
5. [Extending the System](#extending-the-system)
6. [Best Practices](#best-practices)

## Architecture Overview

The Utils component provides common functionality and helper modules used across the Agentic RAG system:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Validation     │       │    Logging      │       │  Sanitization   │
│  Utilities      │──────▶│    System       │──────▶│  Utilities      │
└─────────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Error          │       │   Task          │       │  Config         │
│  Handling       │◀─────▶│   Monitoring    │◀─────▶│  Management     │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

The architecture follows these key principles:

1. **Modularity**: Self-contained utility functions and classes
2. **Reusability**: Common functionality used across the system
3. **Error Resilience**: Robust error handling and recovery
4. **Configurability**: Easily configurable behavior
5. **Standardization**: Consistent patterns and interfaces

## Key Components

### 1. Validation Utilities (`src/utils/validation.py`)

Functions for validating various inputs:

```python
from src.utils.validation import validate_url, validate_sitemap_url, validate_configuration

# Validate a URL
if not validate_url(url):
    raise ValueError(f"Invalid URL: {url}")

# Validate a sitemap URL
result, error = validate_sitemap_url(sitemap_url)
if not result:
    raise ValueError(f"Invalid sitemap URL: {error}")

# Validate configuration
valid, errors = validate_configuration(config)
if not valid:
    for error in errors:
        print(f"Configuration error: {error}")
```

### 2. Enhanced Logging (`src/utils/enhanced_logging.py`)

Advanced logging functionality with structured data:

```python
from src.utils.enhanced_logging import (
    enhanced_crawler_logger,
    start_crawl_session,
    end_crawl_session,
    get_active_session
)

# Start a crawl session
session_id = start_crawl_session(source_id="python_docs", source_name="Python Documentation")

# Log information with structured data
enhanced_crawler_logger.info(
    "Starting to process URL",
    url="https://python.org/docs",
    session_id=session_id
)

# Log an error with context
enhanced_crawler_logger.structured_error(
    "Failed to process URL",
    error=e,
    url="https://python.org/docs",
    session_id=session_id
)

# End the session
end_crawl_session(session_id, status="completed")
```

### 3. Task Monitoring (`src/utils/task_monitoring.py`)

Tracks asynchronous task execution:

```python
from src.utils.task_monitoring import (
    TaskType,
    monitored_task,
    get_tasks_by_type,
    cancel_all_tasks
)

# Define a monitored task
@monitored_task(TaskType.PAGE_PROCESSING, "Processing page {url}")
async def process_page(url, content):
    # Processing logic
    return result

# Get active tasks by type
crawl_tasks = get_tasks_by_type(TaskType.PAGE_CRAWLING)
print(f"Active crawl tasks: {len(crawl_tasks)}")

# Cancel all tasks (e.g., for shutdown)
await cancel_all_tasks()
```

### 4. Error Handling (`src/utils/errors.py`)

Specialized error classes and handling:

```python
from src.utils.errors import (
    ContentProcessingError,
    EmptyContentError,
    ParseError,
    ChunkingError
)

try:
    # Process content
    if not content:
        raise EmptyContentError(url, 0)
    
    # Process chunks
    chunks = chunk_text(content)
    if not chunks:
        raise ChunkingError(url, "Failed to create chunks")
    
except ContentProcessingError as e:
    # Handle content-related errors
    logger.error(f"Content processing error: {e}")
    # Specific handling based on error type
    if isinstance(e, EmptyContentError):
        # Handle empty content specifically
        pass
    elif isinstance(e, ChunkingError):
        # Handle chunking errors
        pass
```

### 5. Sanitization (`src/utils/sanitization.py`)

Functions for sanitizing inputs and outputs:

```python
from src.utils.sanitization import (
    sanitize_html,
    sanitize_text,
    sanitize_metadata
)

# Sanitize HTML content
clean_html = sanitize_html(raw_html)

# Sanitize text for database storage
safe_text = sanitize_text(user_input)

# Sanitize metadata (e.g., for JSON storage)
clean_metadata = sanitize_metadata(metadata)
```

## Integration Points

### Using the Logging System

To integrate with the enhanced logging system:

```python
from src.utils.enhanced_logging import get_logger

# Create a component-specific logger
logger = get_logger("my_component")

# Log with structured data
logger.info(
    "Operation completed successfully",
    operation="data_import",
    items_processed=42,
    duration_ms=1500
)

# Log an error with exception
try:
    # Operation that may fail
    process_data()
except Exception as e:
    logger.structured_error(
        "Failed to process data",
        error=e,
        operation="data_import"
    )
```

### Task Monitoring Integration

To monitor asynchronous tasks:

```python
from src.utils.task_monitoring import monitored_task, TaskType

# Define a custom task type if needed
TaskType.CUSTOM_OPERATION = "custom_operation"

# Decorate functions for monitoring
@monitored_task(TaskType.CUSTOM_OPERATION, "Custom operation: {operation_name}")
async def custom_operation(operation_name, data):
    """A custom operation that will be monitored."""
    # Implementation...
    return result

# Use the monitored function
result = await custom_operation("data_transformation", my_data)
```

### Error Handling Integration

To integrate with the error handling system:

```python
from src.utils.errors import BaseError

# Define custom error types
class DataProcessingError(BaseError):
    """Base class for data processing errors."""
    error_category = "DATA_PROCESSING"

class InvalidDataFormatError(DataProcessingError):
    """Error raised when data format is invalid."""
    
    def __init__(self, data_id, format_name, details=None):
        details = details or {}
        details.update({
            "data_id": data_id,
            "expected_format": format_name
        })
        super().__init__(f"Invalid data format for {data_id}: expected {format_name}", details)

# Use custom errors
try:
    # Data processing
    if not is_valid_format(data, "JSON"):
        raise InvalidDataFormatError(data_id, "JSON", {"actual_format": detect_format(data)})
except InvalidDataFormatError as e:
    # Handle specific error
    logger.error(f"Format error: {e}")
    # Access error details
    data_id = e.details["data_id"]
    expected_format = e.details["expected_format"]
```

## Error Handling

### Error Categories

The error handling system uses these primary categories:

| Category | Description | Example Errors |
|----------|-------------|----------------|
| `VALIDATION` | Input validation errors | Invalid URL, malformed configuration |
| `CONTENT_PROCESSING` | Content processing failures | HTML parsing errors, empty content |
| `API` | API-related errors | Rate limits, authentication failures |
| `DATABASE` | Database operation errors | Connection failures, query errors |
| `TASK` | Task execution errors | Task cancellation, timeout errors |

### Error Inheritance Hierarchy

```
BaseError
├── ValidationError
│   ├── URLValidationError
│   └── ConfigValidationError
├── ContentProcessingError
│   ├── ParseError
│   ├── ChunkingError
│   └── EmptyContentError
├── APIError
│   ├── RateLimitError
│   └── AuthenticationError
├── DatabaseError
│   ├── ConnectionError
│   └── QueryError
└── TaskError
    ├── TaskCancellationError
    └── TaskTimeoutError
```

### Creating Custom Errors

To create custom error types:

```python
from src.utils.errors import BaseError

class CustomComponentError(BaseError):
    """Base class for custom component errors."""
    error_category = "CUSTOM_COMPONENT"
    
    def __init__(self, message, details=None):
        super().__init__(message, details)
        # Additional initialization if needed

class SpecificError(CustomComponentError):
    """A specific error type."""
    
    def __init__(self, resource_id, operation, details=None):
        details = details or {}
        details.update({
            "resource_id": resource_id,
            "operation": operation
        })
        message = f"Failed to {operation} resource {resource_id}"
        super().__init__(message, details)
```

## Extending the System

### Adding New Utility Functions

To add new utility functions:

```python
# src/utils/custom_utils.py

def format_duration(seconds):
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration (e.g., "2h 30m 15s")
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours:
        parts.append(f"{int(hours)}h")
    if minutes:
        parts.append(f"{int(minutes)}m")
    if seconds or not parts:
        parts.append(f"{int(seconds)}s")
        
    return " ".join(parts)

def truncate_string(text, max_length=100, suffix="..."):
    """
    Truncate a string to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (default: 100)
        suffix: Suffix to add when truncated (default: "...")
        
    Returns:
        str: Truncated string
    """
    if not text or len(text) <= max_length:
        return text
        
    return text[:max_length - len(suffix)] + suffix
```

### Adding Custom Decorators

To create custom decorators:

```python
import functools
import time
from src.utils.enhanced_logging import get_logger

logger = get_logger("decorators")

def timing_decorator(func):
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: The function to decorate
        
    Returns:
        wrapped: The decorated function
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.debug(
                f"Function {func.__name__} took {duration:.3f} seconds",
                function=func.__name__,
                duration=duration
            )
    return wrapped

# For async functions
def async_timing_decorator(func):
    """
    Decorator to measure and log async function execution time.
    
    Args:
        func: The async function to decorate
        
    Returns:
        wrapped: The decorated async function
    """
    @functools.wraps(func)
    async def wrapped(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.debug(
                f"Async function {func.__name__} took {duration:.3f} seconds",
                function=func.__name__,
                duration=duration
            )
    return wrapped
```

### Creating Custom Monitoring Tools

To create custom monitoring functionality:

```python
import time
from collections import defaultdict
from src.utils.enhanced_logging import get_logger

logger = get_logger("custom_monitoring")

class PerformanceTracker:
    """Tracks performance metrics for different operations."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def record_metric(self, operation, duration):
        """Record a duration for an operation."""
        self.metrics[operation].append(duration)
        
    def get_average(self, operation):
        """Get average duration for an operation."""
        durations = self.metrics.get(operation, [])
        return sum(durations) / len(durations) if durations else 0
        
    def get_summary(self):
        """Get a summary of all metrics."""
        summary = {}
        for operation, durations in self.metrics.items():
            summary[operation] = {
                "count": len(durations),
                "average": sum(durations) / len(durations),
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0
            }
        return summary
        
    def log_summary(self):
        """Log a summary of all metrics."""
        summary = self.get_summary()
        for operation, stats in summary.items():
            logger.info(
                f"Operation {operation}: {stats['count']} calls, avg: {stats['average']:.3f}s",
                operation=operation,
                **stats
            )
```

## Best Practices

### Logging

1. **Structured Logging**:
   - Use structured logging with key-value pairs
   - Include context in all log messages
   - Use appropriate log levels

2. **Log Levels**:
   - DEBUG: Detailed debugging information
   - INFO: Confirmation that things are working
   - WARNING: Something unexpected but not critical
   - ERROR: Something failed but the application continues
   - CRITICAL: Application failure requiring immediate attention

3. **Context Information**:
   - Include relevant IDs (session_id, request_id, etc.)
   - Log timestamps for performance-sensitive operations
   - Include originating component or module

```python
# Good logging example
logger.info(
    "Processing document completed",
    document_id="doc-123",
    chunks_created=5,
    processing_time_ms=320,
    session_id="session-456"
)
```

### Error Handling

1. **Error Classification**:
   - Use specific error types for different failure modes
   - Include relevant context in error details
   - Enable programmatic handling of errors

2. **Error Recovery**:
   - Implement appropriate retry logic
   - Use circuit breakers for external services
   - Gracefully degrade functionality when necessary

3. **Error Reporting**:
   - Log errors with full context
   - Distinguish between expected and unexpected errors
   - Include stack traces for debugging

```python
try:
    # Operation that may fail
    result = process_document(document_id)
except EmptyContentError as e:
    # Handle empty content gracefully
    logger.warning(f"Document {document_id} is empty, skipping")
    return default_result
except ContentProcessingError as e:
    # Handle processing errors
    logger.error(
        f"Failed to process document: {e}",
        document_id=document_id,
        error_type=e.__class__.__name__,
        details=e.details
    )
    # Attempt recovery or use fallback
    return fallback_result
except Exception as e:
    # Unexpected errors
    logger.exception(
        f"Unexpected error processing document {document_id}: {e}"
    )
    # Re-raise or return error response
    raise
```

### Asynchronous Programming

1. **Task Management**:
   - Track all asynchronous tasks
   - Implement proper cancellation and cleanup
   - Use timeouts to prevent hanging tasks

2. **Concurrency Control**:
   - Use semaphores to limit concurrent operations
   - Implement backpressure when necessary
   - Monitor task queue depth

3. **Error Propagation**:
   - Properly handle and propagate errors in async code
   - Use `asyncio.gather` with `return_exceptions=True` to prevent task failures from being lost
   - Log all unhandled exceptions in tasks

```python
async def process_batch(items, max_concurrent=5):
    """Process a batch of items with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(item):
        async with semaphore:
            try:
                return await process_item(item)
            except Exception as e:
                logger.error(f"Error processing item {item['id']}: {e}")
                return {"item_id": item["id"], "error": str(e)}
    
    # Process all items and gather results
    results = await asyncio.gather(
        *[process_with_semaphore(item) for item in items],
        return_exceptions=True
    )
    
    # Handle results and exceptions
    processed = []
    failed = []
    for result, item in zip(results, items):
        if isinstance(result, Exception):
            failed.append({"item": item, "error": str(result)})
        else:
            processed.append(result)
    
    return {"processed": processed, "failed": failed}
```

### Configuration Management

1. **Centralized Configuration**:
   - Keep configuration in a central location
   - Use environment variables for sensitive values
   - Implement validation for all configuration values

2. **Default Values**:
   - Provide sensible defaults for all settings
   - Document the meaning and impact of each setting
   - Make it clear which settings are required

3. **Configuration Validation**:
   - Validate configuration at startup
   - Provide clear error messages for invalid settings
   - Check for conflicts or dependencies between settings

```python
from pydantic import BaseModel, Field, validator

class AppConfig(BaseModel):
    """Application configuration model with validation."""
    
    # Database settings
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    db_name: str = Field(..., description="Database name")
    db_user: str = Field(..., description="Database username")
    db_password: str = Field(..., description="Database password")
    
    # API settings
    api_key: str = Field(..., description="API key for external service")
    api_url: str = Field(..., description="API endpoint URL")
    api_timeout: int = Field(default=30, ge=1, description="API timeout in seconds")
    
    # Validation logic
    @validator("api_url")
    def validate_api_url(cls, v):
        if not v.startswith("https://"):
            raise ValueError("API URL must use HTTPS")
        return v
``` 