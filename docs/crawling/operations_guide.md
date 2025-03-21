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
DEFAULT_CHUNK_SIZE=5000            # Size of text chunks for character-based chunking
DEFAULT_CHUNK_WORDS=1000           # Number of words per chunk for word-based chunking
DEFAULT_OVERLAP_WORDS=100          # Word overlap between chunks
USE_WORD_BASED_CHUNKING=true       # Use word-based instead of character-based chunking
DEFAULT_MAX_CONCURRENT_CRAWLS=5    # Maximum concurrent HTTP requests
DEFAULT_MAX_CONCURRENT_API_CALLS=3 # Maximum concurrent OpenAI API calls
DEFAULT_RETRY_ATTEMPTS=3           # Number of retry attempts for failed requests
DEFAULT_MIN_BACKOFF=1              # Minimum backoff time in seconds
DEFAULT_MAX_BACKOFF=60             # Maximum backoff time in seconds
STORE_RAW_HTML=true                # Whether to store original HTML content

# OpenAI Configuration
OPENAI_API_KEY=your_key_here       # OpenAI API key
LLM_MODEL=gpt-4o-mini              # Model for generating titles and summaries
EMBEDDING_MODEL=text-embedding-3-small  # Model for generating embeddings
```

### Crawl Configuration

The `CrawlConfig` object controls crawling behavior:

```python
from src.crawling.enhanced_docs_crawler import CrawlConfig

config = CrawlConfig(
    name="Python Documentation",                # Name of the documentation source
    sitemap_url="https://docs.python.org/3/sitemap.xml",  # Sitemap URL
    
    # Content processing options
    chunk_size=5000,                           # Character-based chunk size
    use_word_based_chunking=True,              # Use word-based chunking 
    chunk_words=1000,                          # Words per chunk
    overlap_words=100,                         # Word overlap between chunks
    store_raw_html=True,                       # Store original HTML
    
    # Crawling behavior
    max_concurrent_requests=5,                 # Concurrent HTTP requests
    max_concurrent_api_calls=3,                # Concurrent API calls
    respect_robots_txt=True,                   # Honor robots.txt
    
    # URL filtering
    url_patterns_include=["/reference/", "/tutorial/"],  # URL patterns to include
    url_patterns_exclude=["/whatsnew/"],               # URL patterns to exclude
    
    # Error handling
    retry_attempts=3,                          # Number of retry attempts
    min_backoff=1,                             # Minimum backoff time (seconds)
    max_backoff=60,                            # Maximum backoff time (seconds)
    
    # Performance options
    incremental=True,                          # Only process new/updated pages
    check_modified_since=True                  # Check Last-Modified headers
)
```

## Crawling Operations

### Adding a New Documentation Source

To add a new documentation source through the UI:

1. Navigate to the "Add Source" tab in the web interface
2. Enter source details:
   - **Name**: Descriptive name (e.g., "Python Documentation")
   - **Sitemap URL**: URL to the sitemap (e.g., "https://docs.python.org/3/sitemap.xml")
3. Configure advanced options (optional):
   - **Chunking Strategy**: Word-based or character-based
   - **Words per Chunk**: For word-based chunking (default: 1000)
   - **Word Overlap**: Overlap between chunks (default: 100)
   - **Maximum Concurrent Requests**: Parallelism level (default: 5)
   - **URL Patterns**: Include/exclude patterns for selective crawling
   - **Store Raw HTML**: Whether to store original HTML (useful for debugging)
4. Click "Add and Crawl" to start the crawling process

### Clearing a Documentation Source

To remove a documentation source:

1. Navigate to the "Manage Sources" tab
2. Find the source you want to remove
3. Click "Delete Source"
4. Confirm deletion in the prompt

This will:
- Remove the source from the list of available sources
- Delete all stored content, embeddings, and metadata for that source
- Free up database space

### Programmatic Control

For programmatic control of the crawler:

```python
import asyncio
from src.crawling.enhanced_docs_crawler import (
    crawl_documentation, 
    clear_documentation_source,
    CrawlConfig
)

# Add and crawl a documentation source
async def add_documentation():
    config = CrawlConfig(
        name="FastAPI Documentation",
        sitemap_url="https://fastapi.tiangolo.com/sitemap.xml",
        use_word_based_chunking=True,
        chunk_words=1000,
        overlap_words=100
    )
    
    result = await crawl_documentation(config)
    print(f"Crawling completed: {'Success' if result else 'Failed'}")

# Clear a documentation source
async def clear_documentation():
    success = await clear_documentation_source("FastAPI Documentation")
    print(f"Source cleared: {'Success' if success else 'Failed'}")

# Run the operations
asyncio.run(add_documentation())
# asyncio.run(clear_documentation())
```

## Monitoring and Control

### Crawling Progress

Monitor crawling progress via:

1. **Web Interface**: Real-time progress indicators in the UI
   - Progress bar showing overall completion
   - Counters for processed/successful/failed pages
   - Active tasks indicator

2. **Logs**: Detailed logging in the logs directory
   - `logs/crawler_{timestamp}.log`: Crawler-specific logs
   - `logs/error_{timestamp}.log`: Error logs

3. **Programmatically**:
   ```python
   from src.crawling.crawl_state import get_crawl_stats
   
   # Get statistics for a specific crawl session
   stats = get_crawl_stats(session_id)
   print(f"Progress: {stats['progress_percentage']}%")
   print(f"Processed: {stats['processed_urls']}/{stats['total_urls']}")
   print(f"Success rate: {stats['success_rate']}%")
   print(f"Errors: {stats['failed_urls']}")
   ```

### Controlling Crawling

Control ongoing crawls via:

1. **Pause/Resume**: Temporarily pause and resume crawling
   - UI: Click "Pause" in the crawling interface
   - API: `POST /api/crawl/{source_id}/pause` and `POST /api/crawl/{source_id}/resume`

2. **Cancel**: Stop crawling completely
   - UI: Click "Cancel" in the crawling interface
   - API: `POST /api/crawl/{source_id}/cancel`
   - Programmatically:
     ```python
     from src.crawling.enhanced_docs_crawler import cancel_crawl
     
     # Cancel an active crawl
     await cancel_crawl(source_id)
     ```

3. **Rate Limiting**: Adjust crawling speed dynamically
   - UI: Use the "Crawl Speed" slider in the interface
   - API: `POST /api/crawl/{source_id}/rate-limit?max_requests=3`

## Performance Optimization

### Optimizing Crawl Speed

To optimize crawling performance:

1. **Concurrency Settings**:
   - For modern websites: `max_concurrent_requests=5-10`
   - For older/slower websites: `max_concurrent_requests=3-5`
   - For API calls: `max_concurrent_api_calls=5-10`

2. **Selective Crawling**:
   - Use URL patterns to focus on relevant content:
     ```python
     config = CrawlConfig(
         # ... other settings
         url_patterns_include=["/docs/", "/reference/"],
         url_patterns_exclude=["/blog/", "/news/"]
     )
     ```

3. **Content Extraction Strategy**:
   The system now uses multiple content extraction strategies in sequence:
   - HTML2Text-based conversion (preserves structure and links)
   - Content area extraction (targets main content sections)
   - Fallback raw text extraction
   
   This multi-strategy approach ensures better content quality.

4. **Chunking Strategy**:
   - For technical documentation: Word-based chunking with 800-1200 words per chunk
   - For narrative content: Word-based chunking with 500-800 words per chunk
   - For reference material: Character-based chunking with 4000-6000 characters

### Resource Considerations

Be aware of these resource requirements:

1. **Memory Usage**:
   - Each concurrent crawl task: ~50-100MB
   - Batch processing: ~200-500MB depending on batch size
   - Set appropriate limits based on available memory

2. **API Costs**:
   - Embedding generation: ~$0.0001 per 1K tokens
   - Title/summary generation: ~$0.01 per 1K tokens
   - Estimate: $1-5 per 1000 pages depending on content length

3. **Database Storage**:
   - Text content: ~2-5KB per chunk
   - Embeddings: ~6KB per embedding (1536 dimensions)
   - Raw HTML: ~20-100KB per page
   - Estimate: ~50-150KB per processed page total

## Troubleshooting

### Common Issues

#### Crawling Stops or Slows Down

**Symptoms**: Crawling progress halts or becomes very slow.

**Possible Causes and Solutions**:

1. **Rate Limiting**:
   - Check logs for `429 Too Many Requests` errors
   - **Solution**: Reduce `max_concurrent_requests` or add delays:
     ```python
     config = CrawlConfig(
         # ... other settings
         max_concurrent_requests=3,
         min_backoff=2,
         max_backoff=120
     )
     ```

2. **Memory Issues**:
   - Check for `MemoryError` or system becoming sluggish
   - **Solution**: Reduce batch sizes or concurrency:
     ```python
     config = CrawlConfig(
         # ... other settings
         max_concurrent_requests=3,
         max_concurrent_api_calls=2
     )
     ```

3. **API Limits**:
   - Check for OpenAI API errors in logs
   - **Solution**: Implement request throttling:
     ```python
     config = CrawlConfig(
         # ... other settings
         max_concurrent_api_calls=2,
         retry_attempts=5,
         max_backoff=300
     )
     ```

#### Content Quality Issues

**Symptoms**: Missing or poor-quality content in search results.

**Possible Causes and Solutions**:

1. **Content Extraction Failures**:
   - Check logs for `ContentProcessingError` entries
   - **Solution**: Store raw HTML for debugging:
     ```python
     config = CrawlConfig(
         # ... other settings
         store_raw_html=True
     )
     ```
   - Then examine raw HTML to identify extraction issues

2. **Chunking Problems**:
   - Content chunks break in awkward places
   - **Solution**: Switch to word-based chunking:
     ```python
     config = CrawlConfig(
         # ... other settings
         use_word_based_chunking=True,
         chunk_words=1000,
         overlap_words=150  # Increase overlap for better context
     )
     ```

3. **URL Filtering Too Strict**:
   - Important content missing from results
   - **Solution**: Check and adjust URL patterns:
     ```python
     config = CrawlConfig(
         # ... other settings
         url_patterns_include=["/docs/", "/guide/", "/reference/"],  # Add more patterns
         url_patterns_exclude=[]  # Remove restrictive exclusions
     )
     ```

### Diagnostic Tools

#### Database Inspection

Use the database diagnostic tool to check stored content:

```bash
python check_database.py
```

This will show:
- Documentation source information
- Page and chunk counts per source
- Sample content for verification
- Issues like missing embeddings or empty content

#### Crawl Logs Analysis

Analyze crawl logs to identify patterns:

```bash
python -c "
import re
from collections import Counter
with open('logs/crawler_latest.log') as f:
    errors = re.findall(r'ERROR.*?(\w+Error)', f.read())
    print(Counter(errors))
"
```

This will show the distribution of error types.

#### Content Extraction Testing

Test content extraction on problematic pages:

```python
import asyncio
from src.crawling.enhanced_docs_crawler import test_content_extraction

async def test_extraction():
    url = "https://problem-site.com/difficult-page"
    results = await test_content_extraction(url)
    
    # Show results from each strategy
    for strategy, content in results.items():
        print(f"=== {strategy} ===")
        print(f"Success: {content['success']}")
        print(f"Content length: {len(content['content']) if content['success'] else 0}")
        print(f"Sample: {content['content'][:200] if content['success'] else 'N/A'}")

asyncio.run(test_extraction())
```

## FAQs

### General Questions

**Q: How long does crawling take?**  
A: Crawling speed depends on several factors:
   - Website size (number of pages)
   - Website response time
   - Concurrency settings
   - API rate limits
   
   As a rough estimate:
   - Small site (100 pages): 5-10 minutes
   - Medium site (1000 pages): 30-60 minutes
   - Large site (10,000+ pages): Several hours
   
   The system now displays estimated completion time in the UI based on current processing speed.

**Q: Does the crawler respect robots.txt?**  
A: Yes, by default the crawler respects robots.txt directives. This can be disabled but is not recommended:
   ```python
   config = CrawlConfig(
       # ... other settings
       respect_robots_txt=False  # Not recommended
   )
   ```

**Q: How can I crawl sites requiring authentication?**  
A: For sites requiring authentication, you can use the headers parameter:
   ```python
   config = CrawlConfig(
       # ... other settings
       headers={
           "Authorization": "Bearer your_token_here",
           "Cookie": "session=your_session_cookie"
       }
   )
   ```

### Technical Questions

**Q: Which chunking method should I use?**  
A: The optimal chunking method depends on your content:
   - **Word-based chunking** (recommended): Better preserves semantic units and provides more consistent chunk sizes
   - **Character-based chunking**: Useful for very large documents or when exact character counts matter

**Q: How do I optimize for specific documentation types?**  
A: Different documentation types benefit from different settings:

   - **API Documentation**:
     ```python
     config = CrawlConfig(
         # ... other settings
         use_word_based_chunking=True,
         chunk_words=800,
         overlap_words=100
     )
     ```

   - **Tutorials and Guides**:
     ```python
     config = CrawlConfig(
         # ... other settings
         use_word_based_chunking=True,
         chunk_words=1200,
         overlap_words=150
     )
     ```

   - **Reference Material**:
     ```python
     config = CrawlConfig(
         # ... other settings
         use_word_based_chunking=True,
         chunk_words=600,
         overlap_words=50
     )
     ```

**Q: Can I prioritize certain pages during crawling?**  
A: Yes, you can implement a custom URL prioritization function:
   ```python
   def url_prioritizer(urls):
       # Sort URLs by priority
       prioritized = []
       
       # Give higher priority to index pages
       for url in urls:
           priority = 1
           if url.endswith("index.html") or url.endswith("/"):
               priority = 3
           elif "/getting-started/" in url:
               priority = 2
           prioritized.append((url, priority))
       
       # Sort by priority (highest first)
       prioritized.sort(key=lambda x: x[1], reverse=True)
       return [url for url, _ in prioritized]
   
   config = CrawlConfig(
       # ... other settings
       url_prioritizer=url_prioritizer
   )
   ```

**Q: How can I monitor the quality of extracted content?**  
A: Enable content quality checking:
   ```python
   config = CrawlConfig(
       # ... other settings
       validate_content=True,
       min_content_length=200,  # Minimum characters per chunk
       store_raw_html=True      # Store HTML for manual inspection
   )
   ```
   
   You can also review content in the database using:
   ```bash
   python check_database.py --sample-content --source-id your_source_id
   ``` 