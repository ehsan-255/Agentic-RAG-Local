# Operations Guide: Crawling Component

This guide provides practical instructions for configuring, operating, and troubleshooting the document crawling component of the Agentic RAG system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Configuration](#configuration)
3. [Crawling Operations](#crawling-operations)
4. [Monitoring and Control](#monitoring-and-control)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [FAQs](#faqs)

## System Overview

The document crawling component is responsible for:

1. **Content Acquisition**: Fetching content from documentation websites
2. **Content Processing**: Converting HTML to text and chunking content
3. **Metadata Extraction**: Capturing document metadata
4. **Vector Generation**: Creating embeddings for similarity search
5. **Database Storage**: Storing content and embeddings for retrieval

The system uses asynchronous processing to efficiently crawl multiple pages concurrently while respecting rate limits and server capacities.

## Configuration

### Environment Variables

Configure the crawler through these environment variables:

```
# Crawler Configuration
DEFAULT_CHUNK_SIZE=5000            # Size of text chunks
DEFAULT_MAX_CONCURRENT_CRAWLS=5    # Maximum concurrent HTTP requests
DEFAULT_MAX_CONCURRENT_API_CALLS=3 # Maximum concurrent OpenAI API calls
DEFAULT_RETRY_ATTEMPTS=3           # Number of retry attempts for failed requests
DEFAULT_MIN_BACKOFF=1              # Minimum backoff time in seconds
DEFAULT_MAX_BACKOFF=60             # Maximum backoff time in seconds

# OpenAI Configuration
OPENAI_API_KEY=your_key_here       # OpenAI API key
LLM_MODEL=gpt-4o-mini              # Model for generating titles and summaries
EMBEDDING_MODEL=text-embedding-3-small  # Model for generating embeddings
```

### Crawl Configuration

The `CrawlConfig` object controls crawling behavior:

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|------------------|
| chunk_size | Text chunk size in characters | 5000 | 1000-10000 |
| max_concurrent_requests | Maximum concurrent HTTP requests | 5 | 1-20 |
| max_concurrent_api_calls | Maximum concurrent OpenAI API calls | 3 | 1-10 |
| retry_attempts | Number of retry attempts | 3 | 1-5 |
| url_patterns_include | URL patterns to include | [] | Site-specific |
| url_patterns_exclude | URL patterns to exclude | [] | Site-specific |

### URL Pattern Configuration

URL patterns control which pages are crawled:

```python
# Basic pattern matching
url_patterns_include = [
    "/docs/",     # Include all URLs with /docs/ in the path
    "/guide/",    # Include all URLs with /guide/ in the path
    "/tutorial/", # Include all URLs with /tutorial/ in the path
]

url_patterns_exclude = [
    "/archive/", # Exclude all URLs with /archive/ in the path
    "/v1/",      # Exclude all URLs with /v1/ in the path
    "?lang=",    # Exclude URLs with language query parameters
]

# Using regex patterns requires custom filter implementation
```

## Crawling Operations

### Starting a Crawl

To initiate a crawl operation from the Streamlit UI:

1. Navigate to the "Add New Documentation Source" section
2. Enter the documentation name and sitemap URL
3. Configure advanced options if needed
4. Click "Add and Crawl" to start the crawling process

To programmatically start a crawl:

```python
from src.crawling.enhanced_docs_crawler import crawl_documentation, CrawlConfig
from openai import AsyncOpenAI
import asyncio

async def start_crawl():
    # Initialize OpenAI client
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create configuration
    config = CrawlConfig(
        source_id="python_docs",
        source_name="Python Documentation",
        sitemap_url="https://docs.python.org/3/sitemap.xml",
        chunk_size=5000,
        max_concurrent_requests=5,
        max_concurrent_api_calls=3
    )
    
    # Start crawling
    success = await crawl_documentation(openai_client, config)
    
    return success

# Run the crawl
asyncio.run(start_crawl())
```

### Pausing and Resuming

The crawling process can be paused and resumed using the session state:

```python
from src.utils.enhanced_logging import get_active_session
from src.crawling.crawl_state import save_crawl_state, load_crawl_state

# Pause a crawl
session = get_active_session()
if session:
    # Save current state
    save_crawl_state(session.session_id, "paused")
    session.pause()
    print("Crawl paused")

# Resume a crawl
session_id = "previous_session_id"
state = load_crawl_state(session_id)
if state and state["status"] == "paused":
    # Initialize new crawl with previous state
    config = CrawlConfig(**state["config"])
    
    # Resume crawling unprocessed URLs
    remaining_urls = state["remaining_urls"]
    success = await crawl_documentation(openai_client, config, initial_urls=remaining_urls)
```

### Clearing a Source

To remove a documentation source and all its content:

```python
from src.crawling.enhanced_docs_crawler import clear_documentation_source

# Clear a source completely
source_id = "python_docs"
success = await clear_documentation_source(source_id)

if success:
    print(f"Source {source_id} cleared successfully")
else:
    print(f"Failed to clear source {source_id}")
```

## Monitoring and Control

### Monitoring Crawl Progress

The Streamlit UI provides real-time monitoring of crawl operations:

1. Navigate to the "Monitoring" tab
2. View active crawls, progress, and success rates
3. Check error statistics and rate limiting information

To programmatically monitor crawls:

```python
from src.utils.enhanced_logging import get_active_session
from src.utils.task_monitoring import get_tasks_count, TaskType

# Get status of active crawls
session = get_active_session()
if session:
    # Get progress statistics
    total = session.total_urls
    processed = session.processed_urls
    successful = session.successful_urls
    failed = session.failed_urls
    success_rate = session.success_rate
    
    # Get active tasks
    crawling_tasks = get_tasks_count(TaskType.PAGE_CRAWLING)
    processing_tasks = get_tasks_count(TaskType.PAGE_PROCESSING)
    
    # Print status
    print(f"Crawl progress: {processed}/{total} ({success_rate:.1f}%)")
    print(f"Success: {successful}, Failed: {failed}")
    print(f"Active tasks: {crawling_tasks} crawling, {processing_tasks} processing")
```

### Log Analysis

Check the logs for detailed crawl information:

```bash
# Display recent crawler logs
grep "crawler" logs/application.log | tail -100

# Check for rate limit issues
grep "rate_limit" logs/application.log

# Find failed URLs
grep "ERROR" logs/application.log | grep "URL"
```

### Cancelling a Crawl

To cancel an ongoing crawl operation:

```python
from src.utils.task_monitoring import cancel_all_tasks
from src.utils.enhanced_logging import get_active_session, end_crawl_session

# Cancel all tasks
cancel_all_tasks()

# End the active session
session = get_active_session()
if session:
    end_crawl_session(session.session_id, "cancelled")
    print("Crawl cancelled successfully")
```

## Performance Optimization

### Concurrency Settings

Optimize crawler performance by adjusting concurrency parameters:

| Scenario | max_concurrent_requests | max_concurrent_api_calls | Notes |
|----------|-------------------------|--------------------------|-------|
| Default | 5 | 3 | Balanced performance |
| Gentle crawling | 2 | 2 | For sensitive sites |
| Fast crawling | 10 | 5 | For robust sites with no rate limits |
| API-focused | 5 | 10 | When API is the bottleneck |
| Network-focused | 10 | 3 | When network is the bottleneck |

### Resource Utilization

Optimize for different environments:

1. **Low-resource environment**:
   ```python
   config = CrawlConfig(
       # ... other settings
       chunk_size=2000,  # Smaller chunks
       max_concurrent_requests=3,
       max_concurrent_api_calls=2,
       batch_size=5  # Smaller batches
   )
   ```

2. **High-performance environment**:
   ```python
   config = CrawlConfig(
       # ... other settings
       chunk_size=8000,  # Larger chunks
       max_concurrent_requests=15,
       max_concurrent_api_calls=8,
       batch_size=50  # Larger batches
   )
   ```

### Chunk Size Optimization

Adjust chunk size based on content type:

| Content Type | Recommended Chunk Size | Rationale |
|--------------|------------------------|-----------|
| API Documentation | 2000-3000 | Dense, technical content |
| Tutorials | 4000-6000 | Narrative, contextual content |
| Reference Guides | 3000-5000 | Structured, topical content |
| Blog Posts | 5000-8000 | Natural language, conversational |

### Rate Limit Management

Configure retry behavior for rate limits:

```python
# In CrawlConfig:
retry_attempts=5,         # More retries
min_backoff=2,            # Start with 2 seconds
max_backoff=120,          # Maximum 2 minutes wait
```

## Troubleshooting

### Common Issues

| Problem | Possible Causes | Solution |
|---------|----------------|----------|
| Crawl fails to start | Invalid sitemap URL<br>Database connection issues | Validate sitemap URL<br>Check database connectivity |
| Slow crawling | Too low concurrency<br>Rate limiting<br>Network latency | Increase concurrent requests<br>Adjust backoff settings<br>Check network connection |
| High error rate | Server rejecting requests<br>Invalid content<br>Parsing errors | Reduce concurrency<br>Check URL patterns<br>Inspect page structure |
| OpenAI API errors | Rate limits<br>Invalid key<br>Quota exceeded | Implement backoff<br>Check API key<br>Increase quota |
| Memory issues | Large documents<br>Too many concurrent tasks | Reduce chunk size<br>Limit concurrency<br>Enable content size limits |

### Diagnostic Steps

For crawling issues:

1. **Check connectivity**:
   ```bash
   # Verify sitemap accessibility
   curl -I https://example.com/sitemap.xml
   ```

2. **Validate URL filtering**:
   ```python
   from src.crawling.enhanced_docs_crawler import filter_urls
   
   # Test URL filtering
   test_urls = [
       "https://example.com/docs/guide.html",
       "https://example.com/blog/post.html",
       "https://example.com/docs/api/reference.html"
   ]
   
   filtered = filter_urls(test_urls, config)
   print(f"Filtered {len(filtered)}/{len(test_urls)} URLs")
   ```

3. **Test page processing**:
   ```python
   import httpx
   from src.crawling.enhanced_docs_crawler import process_and_store_document
   
   # Test page processing
   async def test_page():
       url = "https://example.com/docs/page.html"
       async with httpx.AsyncClient() as client:
           response = await client.get(url)
           if response.status_code == 200:
               result = await process_and_store_document(
                   url=url,
                   html_content=response.text,
                   config=config,
                   embedding_processor=embedding_processor,
                   llm_processor=llm_processor
               )
               print(f"Processed {result} chunks")
   ```

### Error Recovery

For recovering from errors:

1. **Restart from failed URLs**:
   ```python
   from src.utils.enhanced_logging import get_failed_urls
   
   # Get failed URLs from the session
   session_id = "your_session_id"
   failed_urls = get_failed_urls(session_id)
   
   # Retry failed URLs
   if failed_urls:
       print(f"Retrying {len(failed_urls)} failed URLs")
       await crawl_documentation(openai_client, config, initial_urls=failed_urls)
   ```

2. **Reset the crawl state**:
   ```python
   from src.crawling.crawl_state import reset_crawl_state
   
   # Reset state for a fresh start
   reset_crawl_state()
   ```

## FAQs

### General Questions

#### How long does crawling typically take?
Crawling time depends on the size of the documentation, but as a rough guide:
- Small documentation (100 pages): 10-15 minutes
- Medium documentation (500 pages): 30-60 minutes
- Large documentation (1000+ pages): 2+ hours

#### Can I crawl multiple sites simultaneously?
Yes, but be careful about resource usage. Set lower concurrency limits for each crawl to avoid overloading your system or hitting API rate limits.

#### What happens if the crawler is interrupted?
The crawler automatically saves its state periodically. You can resume from the last saved state through the UI or programmatically.

### Technical Questions

#### How are pages chunked?
The system attempts to preserve semantic boundaries (paragraphs, sections) while splitting content into chunks of approximately the configured chunk size.

#### Can I crawl sites without sitemaps?
Yes, but you'll need to provide a list of URLs manually by implementing a custom crawler. The system primarily works with sitemap-driven crawling.

#### How do I add support for a new content type?
Implement a custom processor function for your content type that extracts text, then use the standard chunking and embedding process.

### Optimization Questions

#### How can I optimize for token usage?
1. Increase chunk size to reduce the total number of embeddings
2. Use URL patterns to exclude irrelevant pages
3. Implement content filtering to remove boilerplate text

#### What's the optimal concurrency setting?
Start with moderate settings (5 concurrent requests, 3 API calls) and gradually increase while monitoring performance and error rates.

#### How can I reduce crawl time?
1. Increase concurrency settings
2. Filter URLs more aggressively
3. Use a more powerful machine for crawling
4. Split large sites into multiple crawl operations 