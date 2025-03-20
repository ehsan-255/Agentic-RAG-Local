# Developer Guide: Monitoring System

This guide provides technical documentation for developers who need to integrate with, extend, or modify the monitoring system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Integration Points](#integration-points)
4. [Error Handling](#error-handling)
5. [Extending the System](#extending-the-system)
6. [Best Practices](#best-practices)

## Architecture Overview

The monitoring system consists of multiple layers designed to track and visualize different aspects of the crawler's operation:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Applications   │       │   Monitoring    │       │ Visualization   │
│  (Crawler)      │──────▶│   Components    │──────▶│ (Streamlit UI)  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Data Storage   │       │  Task & Error   │       │  State & Config │
│  (Database)     │◀─────▶│  Tracking       │◀─────▶│  Management     │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

The architecture follows these key principles:

1. **Separation of Concerns**: Monitoring logic is kept separate from application logic
2. **Non-intrusive Integration**: Minimal changes to existing code through decorators and wrappers
3. **Concurrency-Safe**: Thread-safe operations for multi-threaded/async environments
4. **Extensible**: Easy to add new metrics or monitoring capabilities

## Key Components

### 1. Error Categorization (`src/utils/errors.py`)

The error system provides standardized error types for consistent tracking and reporting:

```python
from src.utils.errors import ContentProcessingError, EmptyContentError

# Raising typed errors
if not content:
    raise EmptyContentError(url, content_length=0)
```

### 2. Enhanced Logging (`src/utils/enhanced_logging.py`)

Extended logger with structured error reporting and session tracking:

```python
from src.utils.enhanced_logging import enhanced_crawler_logger, start_crawl_session

# Start a monitoring session
session_id = start_crawl_session(source_id, source_name)

# Log structured errors
enhanced_crawler_logger.structured_error(
    "Error processing document",
    error=e,
    url=url,
    source_id=source_id
)
```

### 3. Task Monitoring (`src/utils/task_monitoring.py`) 

Tracks the lifecycle of asynchronous tasks:

```python
from src.utils.task_monitoring import monitored_task, TaskType

@monitored_task(TaskType.PAGE_PROCESSING, "Processing page {url}")
async def process_page(url, content):
    # Processing logic here
    pass
```

### 4. API Monitoring (`src/utils/api_monitoring.py`)

Monitors external API calls, including rate limits:

```python
from src.utils.api_monitoring import monitor_openai_call

@monitor_openai_call("embeddings")
async def create_embeddings(texts):
    # API call logic
    pass
```

### 5. Database Monitoring (`src/utils/db_monitoring.py`)

Tracks database operations and connection pools:

```python
from src.utils.db_monitoring import monitor_db_operation

@monitor_db_operation("Fetch documents")
def get_documents(source_id):
    # Database query logic
    pass
```

### 6. Crawl State Management (`src/crawling/crawl_state.py`)

Handles saving and loading crawler configurations:

```python
from src.crawling.crawl_state import save_crawl_configuration

config = {
    'source_id': source_id,
    'sitemap_url': url,
    # Other settings
}
save_crawl_configuration(config, "my_config")
```

## Integration Points

### Adding Monitoring to New Components

To add monitoring to a new function or component:

1. **Task Monitoring**: Use the `@monitored_task` decorator for functions that represent discrete tasks
2. **API Monitoring**: Use the `@monitor_openai_call` decorator for functions making external API calls
3. **Database Monitoring**: Use the `@monitor_db_operation` decorator for database operations

### Tracking Custom Metrics

To add custom metrics:

1. Create a metrics container in the appropriate module
2. Add update functions for your metrics
3. Provide accessor functions to retrieve the metrics

Example:

```python
# Define metrics container
custom_metrics = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0
}

# Update function
def update_metrics(success: bool):
    custom_metrics["total_calls"] += 1
    if success:
        custom_metrics["successful_calls"] += 1
    else:
        custom_metrics["failed_calls"] += 1

# Accessor function
def get_metrics():
    return custom_metrics.copy()
```

## Error Handling

### Error Categories

The monitoring system uses these primary error categories:

| Category | Description | Example |
|----------|-------------|---------|
| `CONTENT_PROCESSING` | Errors in processing document content | HTML parsing failures |
| `DATABASE` | Database-related errors | Query failures, connection issues |
| `CONNECTION` | Network connection errors | Timeouts, DNS failures |
| `EMBEDDING` | Errors generating embeddings | Token limit exceeded |
| `API_RATE_LIMIT` | API rate limit exceeded | OpenAI rate limit errors |
| `TASK_SCHEDULING` | Task execution errors | Future cancellation, deadlocks |

### Implementing Custom Error Types

To create a new error type:

1. Subclass the appropriate base error
2. Implement the constructor with relevant attributes
3. Register the error type in error tracking statistics if needed

Example:

```python
from src.utils.errors import ContentProcessingError

class ImageProcessingError(ContentProcessingError):
    """Error when processing images in content."""
    
    def __init__(self, url: str, image_url: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["image_url"] = image_url
        super().__init__(f"Failed to process image: {image_url}", url, details)
```

## Extending the System

### Adding New Monitoring Components

To add a new monitoring component:

1. Create a new module in the `src/utils` directory
2. Define your metrics storage and tracking functions
3. Implement decorator or wrapper functions for integration
4. Add visualization components to the Streamlit UI

### Example: Adding Cache Monitoring

```python
# src/utils/cache_monitoring.py
from typing import Any, Dict, Callable, TypeVar

T = TypeVar('T')

# Metrics storage
cache_stats = {
    "hits": 0,
    "misses": 0,
    "size": 0
}

def update_cache_stats(hit: bool, size_change: int = 0):
    """Update cache statistics."""
    if hit:
        cache_stats["hits"] += 1
    else:
        cache_stats["misses"] += 1
    
    cache_stats["size"] += size_change

def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    return cache_stats.copy()

def monitor_cache(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to monitor cache operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Add implementation
        pass
    return wrapper
```

## Best Practices

### Performance Considerations

1. **Use Sampling for High-Volume Events**: For high-frequency events, sample rather than logging every occurrence
2. **Lazy Evaluation**: Use lazy evaluation for expensive logging operations
3. **Batch Operations**: Batch logging operations where possible
4. **Avoid Blocking**: Don't block application threads for monitoring operations

### Memory Management

1. **Avoid Circular References**: Be careful with capturing contexts in closures
2. **Clean Up Old Data**: Implement automatic pruning of historical monitoring data
3. **Use Weak References**: Use `weakref` for tracking objects to avoid memory leaks

### Thread Safety

1. **Use Locks for Shared State**: Protect shared monitoring state with locks
2. **Atomic Operations**: Make state updates atomic where possible
3. **Thread-Local Storage**: Use thread-local storage for per-thread metrics

### Logging

1. **Structured Logging**: Use structured logging for machine-processable logs
2. **Contextual Information**: Include relevant context in log messages
3. **Log Levels**: Use appropriate log levels for different types of events
4. **Sanitize Sensitive Data**: Never log API keys, credentials, or PII 