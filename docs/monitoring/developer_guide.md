# Developer Guide: Monitoring Component

This guide provides technical documentation for developers who need to integrate with, extend, or modify the monitoring component of the Agentic RAG system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Integration Points](#integration-points)
4. [Advanced Features](#advanced-features)
5. [Extending the System](#extending-the-system)
6. [Best Practices](#best-practices)

## Architecture Overview

The monitoring system provides comprehensive observability for all components of the Agentic RAG system:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Structured     │       │   Metrics       │       │    Alert        │
│  Logging        │──────▶│   Collection    │──────▶│    Management   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Diagnostic    │       │   Dashboard     │       │  Performance    │
│    Tools        │◀─────▶│   Visualization │◀─────▶│   Analysis      │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

The architecture follows these key principles:

1. **Minimal Overhead**: Performance impact under 5% for production systems
2. **Structured Data**: JSON-formatted logs with standardized fields
3. **Flexible Storage**: Support for multiple metrics and log storage backends
4. **Integrated Dashboards**: Real-time visualization of system performance
5. **Alerting System**: Configurable alerts for critical events
6. **Extensibility**: Plugin architecture for custom metrics and visualizations

## Key Components

### 1. Structured Logger (`src/monitoring/logger.py`)

The centralized logging system that provides structured, consistent logs:

```python
from src.monitoring.logger import get_logger

# Get a logger for your component
logger = get_logger("my_component")

# Basic logging with automatic metadata
logger.info("Operation completed successfully")

# Structured logging with explicit metadata
logger.info(
    "User query processed",
    query_id="q-123456",
    processing_time_ms=350,
    tokens_used=128
)

# Error logging with context
try:
    # Operation that might fail
    result = process_data(input_data)
except Exception as e:
    logger.structured_error(
        "Data processing failed",
        error=str(e),
        error_type=type(e).__name__,
        input_data_size=len(input_data)
    )
```

Key features:
- JSON-formatted logs with standardized fields
- Consistent metadata across all log entries
- Automatic context enrichment (timestamps, component names, etc.)
- Multiple output destinations (console, files, external services)
- Log rotation and retention management

### 2. Metrics Collection (`src/monitoring/metrics.py`)

The metrics system that captures and stores performance data:

```python
from src.monitoring.metrics import MetricsCollector

# Get a metrics collector
metrics = MetricsCollector()

# Increment a counter
metrics.increment_counter("http_requests_total", labels={"method": "GET", "endpoint": "/api/query"})

# Record a timing
with metrics.timer("query_processing_time_ms", labels={"query_type": "vector"}):
    # Operation being timed
    results = process_query(query)

# Set a gauge value
metrics.set_gauge("active_connections", connection_pool.active_count)

# Record a histogram value
metrics.observe_histogram("response_size_bytes", len(response_data))

# Get current metrics
all_metrics = metrics.get_metrics()
```

Key features:
- Support for common metric types (counters, gauges, histograms, timers)
- Dimensional metrics with labels/tags
- Aggregation functions (sum, avg, percentiles)
- Pluggable storage backends (in-memory, Prometheus, PostgreSQL)
- Sampling capabilities for high-volume metrics

### 3. Diagnostics Tools (`src/monitoring/diagnostics.py`)

Utilities for checking system health and diagnosing issues:

```python
from src.monitoring.diagnostics import (
    check_system_health,
    run_component_diagnostics,
    get_resource_usage,
    analyze_performance
)

# Check overall system health
health_status = await check_system_health()
print(f"System health: {health_status['status']}")

# Run diagnostics on a specific component
db_diagnostics = await run_component_diagnostics("database")
print(f"Database connection pool: {db_diagnostics['connection_pool_status']}")

# Get resource usage
resources = get_resource_usage()
print(f"CPU: {resources['cpu_percent']}%, Memory: {resources['memory_percent']}%")

# Analyze performance
performance_data = await analyze_performance("vector_search", sample_size=100)
print(f"Avg query time: {performance_data['avg_time_ms']}ms")
print(f"95th percentile: {performance_data['p95_time_ms']}ms")
```

### 4. Dashboard Generator (`src/monitoring/dashboard.py`)

Tools for creating interactive dashboards:

```python
from src.monitoring.dashboard import (
    generate_dashboard,
    create_time_series_chart,
    create_metrics_table,
    export_dashboard
)

# Generate a complete dashboard
dashboard = generate_dashboard(
    title="System Performance",
    timespan="24h"
)

# Create a custom chart
cpu_chart = create_time_series_chart(
    title="CPU Usage",
    metric="system_cpu_percent",
    aggregation="avg",
    interval="5m"
)

# Add the chart to the dashboard
dashboard.add_chart(cpu_chart)

# Export the dashboard
export_dashboard(dashboard, format="html", output_path="./dashboards/system.html")
```

### 5. Alert Manager (`src/monitoring/alerts.py`)

System for configuring and triggering alerts:

```python
from src.monitoring.alerts import (
    AlertManager,
    AlertRule,
    EmailChannel,
    SlackChannel
)

# Create alert channels
email_channel = EmailChannel(recipients=["admin@example.com"])
slack_channel = SlackChannel(webhook_url="https://hooks.slack.com/...")

# Create alert manager
alert_manager = AlertManager(channels=[email_channel, slack_channel])

# Define alert rules
high_error_rate = AlertRule(
    name="High Error Rate",
    condition="error_rate > 0.05",
    duration="5m",
    level="WARNING"
)

# Register rules
alert_manager.add_rule(high_error_rate)

# Trigger an alert manually (normally done automatically based on metrics)
alert_manager.trigger_alert(
    title="Database Connection Failures",
    message="Multiple attempts to connect to the database have failed",
    level="ERROR",
    metadata={"attempts": 5, "last_error": "Connection refused"}
)
```

## Integration Points

### 1. With RAG Component

The monitoring system integrates with the RAG component:

```python
from src.monitoring.logger import get_logger
from src.monitoring.metrics import MetricsCollector

# In rag_agent.py
logger = get_logger("rag")
metrics = MetricsCollector()

async def process_query(query, contexts, deps):
    # Start timing the query processing
    with metrics.timer("rag_processing_time_ms", labels={"query_type": "standard"}):
        # Log the incoming query
        logger.info(
            "Processing RAG query",
            query_id=deps.request_id,
            context_count=len(contexts)
        )
        
        try:
            # Process the query...
            response = await generate_response(query, contexts, deps)
            
            # Record successful query
            metrics.increment_counter(
                "rag_queries_total", 
                labels={"status": "success"}
            )
            
            # Log success
            logger.info(
                "Query processed successfully",
                query_id=deps.request_id,
                response_length=len(response)
            )
            
            return response
        except Exception as e:
            # Record failed query
            metrics.increment_counter(
                "rag_queries_total", 
                labels={"status": "failure"}
            )
            
            # Log detailed error
            logger.structured_error(
                "Query processing failed",
                error=str(e),
                error_type=type(e).__name__,
                query_id=deps.request_id
            )
            
            raise
```

### 2. With API Component

Integration with the API component for request tracking:

```python
# In src/api/middleware/logging_middleware.py
from fastapi import Request, Response
from src.monitoring.logger import get_logger
from src.monitoring.metrics import MetricsCollector
import time
import uuid

logger = get_logger("api")
metrics = MetricsCollector()

async def logging_middleware(request: Request, call_next):
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Start timer
    start_time = time.time()
    
    # Log request
    logger.info(
        "API request received",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else "unknown"
    )
    
    # Increment request counter
    metrics.increment_counter(
        "http_requests_total",
        labels={"method": request.method, "path": request.url.path}
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Record timing
        metrics.observe_histogram(
            "http_request_duration_ms",
            duration_ms,
            labels={"method": request.method, "path": request.url.path}
        )
        
        # Log response
        logger.info(
            "API request completed",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2)
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
    except Exception as e:
        # Record error
        metrics.increment_counter(
            "http_errors_total",
            labels={"method": request.method, "path": request.url.path}
        )
        
        # Log error
        logger.structured_error(
            "API request failed",
            error=str(e),
            error_type=type(e).__name__,
            request_id=request_id,
            method=request.method,
            path=request.url.path
        )
        
        raise
```

### 3. With Crawler Component

Integration with the crawler component:

```python
# In src/crawling/enhanced_docs_crawler.py
from src.monitoring.logger import get_logger
from src.monitoring.metrics import MetricsCollector

logger = get_logger("crawler")
metrics = MetricsCollector()

async def crawl_documentation(config):
    crawl_id = str(uuid.uuid4())
    
    # Log crawl start
    logger.info(
        "Starting documentation crawl",
        crawl_id=crawl_id,
        source_name=config.name,
        sitemap_url=config.sitemap_url
    )
    
    # Initialize crawl metrics
    metrics.set_gauge(
        "crawler_active_crawls",
        metrics.get_gauge("crawler_active_crawls") + 1
    )
    
    try:
        # Crawling process...
        
        # Update metrics throughout the process
        metrics.increment_counter(
            "crawler_pages_processed_total",
            len(processed_pages)
        )
        metrics.increment_counter(
            "crawler_errors_total",
            len(failed_pages)
        )
        
        # Log completion
        logger.info(
            "Documentation crawl completed",
            crawl_id=crawl_id,
            pages_processed=len(processed_pages),
            pages_failed=len(failed_pages),
            duration_seconds=duration
        )
        
        return result
    finally:
        # Decrement active crawls count
        metrics.set_gauge(
            "crawler_active_crawls",
            max(0, metrics.get_gauge("crawler_active_crawls") - 1)
        )
```

## Advanced Features

### 1. Performance Profiling

Tools for identifying performance bottlenecks:

```python
from src.monitoring.profiling import (
    profile_function,
    analyze_memory_usage,
    trace_sql_queries
)

# Profile a function
@profile_function(output="logs/profiles")
async def process_complex_query(query):
    # Function implementation...
    return result

# Analyze memory usage
memory_profile = await analyze_memory_usage("vector_search")
print(f"Peak memory: {memory_profile['peak_mb']}MB")
print(f"Largest allocations: {memory_profile['top_allocations']}")

# Trace SQL queries
with trace_sql_queries() as sql_trace:
    # Database operations...
    result = await perform_database_operations()

# Analyze SQL performance
print(f"Total queries: {sql_trace.query_count}")
print(f"Slowest query: {sql_trace.slowest_query}")
print(f"Total query time: {sql_trace.total_time_ms}ms")
```

### 2. Log Analysis

Tools for analyzing log data:

```python
from src.monitoring.log_analysis import (
    analyze_logs,
    find_error_patterns,
    search_logs
)

# Analyze logs for a time period
analysis = await analyze_logs(
    component="rag",
    start_time="2023-04-16T00:00:00Z",
    end_time="2023-04-16T23:59:59Z"
)

print(f"Total requests: {analysis['total_requests']}")
print(f"Error rate: {analysis['error_rate']:.2%}")
print(f"Avg response time: {analysis['avg_response_time_ms']}ms")

# Find error patterns
error_patterns = await find_error_patterns(
    start_time="2023-04-16T00:00:00Z",
    end_time="2023-04-16T23:59:59Z"
)

for pattern in error_patterns:
    print(f"Pattern: {pattern['pattern']}")
    print(f"Occurrences: {pattern['count']}")
    print(f"First seen: {pattern['first_seen']}")

# Search logs
search_results = await search_logs(
    query="api_key_error",
    component="api",
    limit=10
)

for log in search_results:
    print(f"{log['timestamp']} - {log['message']}")
```

### 3. Custom Metrics Pipelines

Configuration for custom metrics processing:

```python
from src.monitoring.metrics import (
    MetricsPipeline,
    MovingAverageTransform,
    ThresholdFilter,
    TimeSeriesAggregator
)

# Create a custom metrics pipeline
pipeline = MetricsPipeline(name="api_performance")

# Add processing stages
pipeline.add_stage(
    MovingAverageTransform(
        metric="http_request_duration_ms",
        window="5m"
    )
)

pipeline.add_stage(
    ThresholdFilter(
        metric="http_request_duration_ms_avg_5m",
        threshold=500,
        comparison=">"
    )
)

pipeline.add_stage(
    TimeSeriesAggregator(
        metrics=["http_request_duration_ms_avg_5m_threshold"],
        interval="1m",
        aggregation="count"
    )
)

# Register the pipeline
metrics_manager.register_pipeline(pipeline)

# Get results
results = await pipeline.get_results()
```

## Extending the System

### 1. Adding a Custom Logger

To implement a custom logger:

```python
# 1. Create a custom logger class
from src.monitoring.logger import BaseLogger, LogLevel

class CustomStorageLogger(BaseLogger):
    def __init__(self, storage_url):
        super().__init__()
        self.storage_url = storage_url
        # Initialize connection to your storage
        
    async def log(self, level, message, **kwargs):
        log_entry = self.format_log_entry(level, message, **kwargs)
        # Send to your custom storage
        await self.send_to_storage(log_entry)
        
    async def send_to_storage(self, log_entry):
        # Implementation specific to your storage solution
        pass

# 2. Register your custom logger
from src.monitoring.logger import register_logger

# Register for all logs
register_logger(CustomStorageLogger("https://storage-service.example.com"))

# Or register for specific components
register_logger(
    CustomStorageLogger("https://storage-service.example.com"),
    components=["rag", "api"]
)
```

### 2. Creating a Custom Metrics Backend

To implement a custom metrics storage backend:

```python
# 1. Create a custom metrics backend
from src.monitoring.metrics import MetricsBackend

class CustomMetricsBackend(MetricsBackend):
    def __init__(self, connection_string):
        self.connection_string = connection_string
        # Initialize connection to your backend
        
    async def store_counter(self, name, value, labels=None):
        # Implementation for counter storage
        pass
        
    async def store_gauge(self, name, value, labels=None):
        # Implementation for gauge storage
        pass
        
    async def store_histogram(self, name, value, labels=None):
        # Implementation for histogram storage
        pass
        
    async def query_metric(self, name, aggregation=None, start_time=None, end_time=None, labels=None):
        # Implementation for querying metrics
        pass

# 2. Register your custom backend
from src.monitoring.metrics import register_backend

register_backend(CustomMetricsBackend("custom://metrics-backend"))
```

### 3. Adding a Custom Dashboard Widget

To create a custom dashboard widget:

```python
# 1. Create a custom widget class
from src.monitoring.dashboard import DashboardWidget

class CustomMetricWidget(DashboardWidget):
    def __init__(self, title, metric_name, custom_options=None):
        super().__init__(title)
        self.metric_name = metric_name
        self.custom_options = custom_options or {}
        
    async def render(self, format="html"):
        # Fetch the metric data
        metric_data = await self.fetch_metric_data()
        
        # Render based on format
        if format == "html":
            return self.render_html(metric_data)
        elif format == "json":
            return self.render_json(metric_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def render_html(self, metric_data):
        # Custom HTML rendering
        html = f"<div class='custom-widget'>"
        html += f"<h3>{self.title}</h3>"
        # Add custom visualization
        html += "</div>"
        return html
        
    def render_json(self, metric_data):
        # JSON representation
        return {
            "widget_type": "custom_metric",
            "title": self.title,
            "metric_name": self.metric_name,
            "data": metric_data
        }
        
    async def fetch_metric_data(self):
        # Fetch the metric data from your metrics collector
        metrics = MetricsCollector()
        return await metrics.query_metric(
            self.metric_name,
            **self.custom_options
        )

# 2. Use your custom widget
from src.monitoring.dashboard import Dashboard

dashboard = Dashboard(title="Custom Dashboard")
dashboard.add_widget(CustomMetricWidget(
    title="My Custom Metric",
    metric_name="special_application_metric",
    custom_options={"aggregation": "sum", "interval": "1m"}
))
```

## Best Practices

### 1. Effective Logging

Follow these guidelines for effective logging:

```python
# DO: Use structured logging with context
logger.info(
    "User query processed",
    query_id="q-123",
    processing_time_ms=350,
    tokens_used=128
)

# DO: Use appropriate log levels
logger.debug("Starting to process query")  # For developers
logger.info("Query processed successfully")  # Regular operational info
logger.warning("Rate limit approaching")  # Potential issues
logger.error("Failed to process query")  # Errors requiring attention
logger.critical("Database connection failed")  # Severe issues

# DO: Add context to errors
try:
    result = process_data()
except Exception as e:
    logger.structured_error(
        "Processing failed",
        error=str(e),
        error_type=type(e).__name__,
        data_id=data_id
    )

# DON'T: Log sensitive information
# BAD: logger.info(f"User {username} logged in with password {password}")
# GOOD:
logger.info(
    "User logged in",
    username=username
)

# DON'T: Create high-volume debug logs in production
# BAD: For large loops
# for item in items:
#     logger.debug(f"Processing item {item.id}")
# GOOD: Summarize
logger.debug(f"Processing {len(items)} items")
```

### 2. Metrics Best Practices

Optimize your metrics collection:

```python
# DO: Use consistent naming conventions
metrics.increment_counter("http_requests_total")  # Use _total suffix for counters
metrics.set_gauge("connection_pool_active")  # Direct value name for gauges
metrics.observe_histogram("response_time_ms")  # Include unit in name

# DO: Use labels for dimensions, but keep cardinality under control
# GOOD: Limited set of values
metrics.increment_counter(
    "http_requests_total",
    labels={"method": "GET", "status": "success", "endpoint": "/api/query"}
)

# DON'T: Use high-cardinality labels
# BAD: Too many possible values
# metrics.increment_counter("http_requests_total", labels={"user_id": user_id})

# DO: Use sampling for high-frequency metrics
if random.random() < 0.1:  # 10% sample
    metrics.observe_histogram("detailed_processing_time_ms", processing_time)

# DO: Pre-aggregate where possible
metrics.increment_counter("processed_items_total", count=len(items))
```

### 3. Performance Considerations

Keep monitoring overhead low:

```python
# DO: Use async operations for monitoring
async def process_request():
    # Start timer using an efficient timer implementation
    start_time = time.time()
    
    # Process the request
    result = await perform_operation()
    
    # Record timing asynchronously
    duration_ms = (time.time() - start_time) * 1000
    asyncio.create_task(
        metrics.observe_histogram("operation_time_ms", duration_ms)
    )
    
    return result

# DO: Batch log writes
logger.configure(batch_size=100, flush_interval=5)  # Flush every 100 logs or 5 seconds

# DO: Use appropriate sampling rates
# High-volume debug logging
if logger.should_sample(0.01):  # 1% sample rate
    logger.debug("Detailed operation trace", operation_details=details)

# DON'T: Block the main thread for monitoring
# BAD:
# metrics.flush_to_persistent_storage()  # Blocking operation
# GOOD:
asyncio.create_task(metrics.async_flush_to_storage())
```

### 4. Alerting Strategy

Configure effective alerts:

```python
# DO: Set meaningful thresholds based on baseline performance
alert_rule = AlertRule(
    name="High Error Rate",
    condition="error_rate > 0.05",  # 5% error rate threshold
    duration="5m",  # Must exceed for 5 minutes
    level="WARNING"
)

# DO: Implement alert grouping to prevent alert storms
alert_manager.configure(
    grouping_window="10m",  # Group similar alerts within 10 minutes
    max_alerts_per_group=3  # Send max 3 alerts per group
)

# DO: Include remediation steps in alerts
alert_manager.trigger_alert(
    title="Database Connection Failures",
    message="Multiple attempts to connect to the database have failed",
    level="ERROR",
    remediation_steps=[
        "Check database server status",
        "Verify network connectivity",
        "Check for recent configuration changes"
    ]
)

# DON'T: Alert on every fluctuation
# BAD:
# AlertRule(
#     name="CPU Usage",
#     condition="cpu_percent > 70",  # Triggers on brief spikes
#     duration="0m",
#     level="WARNING"
# )
# GOOD:
AlertRule(
    name="Sustained High CPU",
    condition="cpu_percent > 80",
    duration="5m",  # Must be sustained for 5 minutes
    level="WARNING"
)
```

### 5. Security and Privacy

Handle sensitive information appropriately:

```python
# DO: Redact sensitive information
logger.configure(
    redact_fields=["password", "api_key", "token", "credit_card"]
)

# Example of how it works:
logger.info(
    "User authenticated",
    username="john",
    password="secret123"  # Will be automatically redacted
)
# Logs: {"message": "User authenticated", "username": "john", "password": "[REDACTED]"}

# DO: Use secure transport for monitoring data
metrics_collector = MetricsCollector(
    backend_url="https://metrics.example.com",
    use_tls=True,
    verify_cert=True
)

# DO: Implement access controls for monitoring data
dashboard_server.configure(
    authentication_required=True,
    access_control={
        "admin": ["all_dashboards"],
        "developer": ["development_dashboards"],
        "support": ["error_dashboards", "user_experience_dashboards"]
    }
)
``` 