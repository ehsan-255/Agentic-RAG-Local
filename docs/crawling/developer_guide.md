# Developer Guide: Crawling Component

This guide provides technical documentation for developers who need to integrate with, extend, or modify the document crawling component of the Agentic RAG system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Integration Points](#integration-points)
4. [Content Processing](#content-processing)
5. [Extending the System](#extending-the-system)
6. [Best Practices](#best-practices)

## Architecture Overview

The document crawling system is designed to efficiently extract, process, and store content from documentation websites:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Sitemap        │       │   Web Pages     │       │    Content      │
│  Parsing        │──────▶│   Fetching      │──────▶│    Processing   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│    URL          │       │     Batch       │       │    Database     │
│    Filtering    │◀─────▶│    Processing   │◀─────▶│    Storage      │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

The architecture follows these key principles:

1. **Asynchronous Processing**: Efficient concurrent crawling with controlled parallelism
2. **Configurable Behavior**: Customizable crawling parameters
3. **Multi-strategy Content Extraction**: Multiple approaches to handle different HTML structures
4. **Word-based Chunking**: Semantic text division with configurable overlap
5. **Enhanced Error Handling**: Categorized errors with recovery mechanisms
6. **State Management**: Tracking crawl progress for resumability and monitoring
7. **Raw HTML Preservation**: Option to store original HTML alongside processed content

## Key Components

### 1. Enhanced Document Crawler (`src/crawling/enhanced_docs_crawler.py`)

The main crawler implementation that manages the entire crawling process:

```python
from src.crawling.enhanced_docs_crawler import crawl_documentation, CrawlConfig

# Create a crawl configuration
config = CrawlConfig(
    name="Python Documentation",
    sitemap_url="https://docs.python.org/3/sitemap.xml",
    chunk_size=5000,
    # New options
    use_word_based_chunking=True,
    chunk_words=1000,
    overlap_words=100,
    store_raw_html=True,
    max_concurrent_requests=5,
    retry_attempts=3
)

# Start the crawling process
result = await crawl_documentation(config)
```

The enhanced crawler now supports:
- Multiple sitemap formats (XML, XML.gz, TXT)
- Concurrent HTTP requests with connection pooling
- Asynchronous processing with controlled concurrency
- Word-based chunking with configurable overlap
- Multiple content extraction strategies

### 2. Crawl State Management (`src/crawling/crawl_state.py`)

New component for managing crawl state and progress:

```python
from src.crawling.crawl_state import (
    initialize_crawl_state, 
    reset_crawl_state,
    update_crawl_progress,
    get_crawl_stats
)

# Initialize crawl state
session_id = initialize_crawl_state(source_id="python_docs", total_urls=250)

# Update progress
update_crawl_progress(session_id, processed_urls=10, success=8, failed=2)

# Get statistics
stats = get_crawl_stats(session_id)
```

### 3. Batch Processor (`src/crawling/batch_processor.py`)

Handles batched processing of embeddings and LLM tasks:

```python
from src.crawling.batch_processor import EmbeddingBatchProcessor, LLMBatchProcessor

# Create a batch processor for embeddings
processor = EmbeddingBatchProcessor(
    batch_size=10,
    max_concurrent_requests=3
)

# Process a batch of texts
results = await processor.process_batch(texts)
```

## Integration Points

### 1. With Database Layer

The crawler integrates with the database layer to store processed content:

```python
from src.db.async_schema import (
    add_documentation_source,
    add_site_page,
    update_documentation_source
)

# Add a new documentation source
source_id = await add_documentation_source(name="Python Docs", url="https://docs.python.org")

# Store a processed page
await add_site_page(
    url="https://docs.python.org/3/tutorial/index.html",
    chunk_number=1,
    title="Python Tutorial",
    summary="Introduction to Python basics",
    content="Processed content here...",
    metadata={"source_id": source_id, "word_count": 500},
    embedding=embedding_vector,
    raw_content="<html>Original HTML content</html>",  # New parameter
    text_embedding=text_embedding_vector  # New parameter for hybrid search
)
```

### 2. With UI Layer

The crawler integrates with the UI layer for control and monitoring:

```python
# In streamlit_app.py
from src.crawling.enhanced_docs_crawler import crawl_documentation
from src.crawling.crawl_state import get_crawl_stats

# Start crawling in background
async def start_crawl():
    config = CrawlConfig(...)
    result = await crawl_documentation(config)
    return result

# Display crawling progress
stats = get_crawl_stats(session_id)
st.progress(stats["progress_percentage"])
st.metric("Pages Processed", stats["processed"])
```

## Content Processing

### 1. Multi-Strategy Content Extraction

The enhanced crawler implements three strategies for extracting content:

```python
# Strategy 1: HTML2Text-based conversion
def extract_content_html2text(html):
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    return converter.handle(html)
    
# Strategy 2: Content area extraction
def extract_main_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Target main content areas
    for selector in ['main', 'article', '.content', '#content']:
        content_area = soup.select_one(selector)
        if content_area:
            return content_area.get_text()
    
# Strategy 3: Fallback raw text extraction
def extract_raw_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()
```

### 2. Word-Based Chunking

New word-based chunking with configurable overlap:

```python
def chunk_by_words(text, words_per_chunk=1000, overlap_words=100):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), words_per_chunk - overlap_words):
        chunk = " ".join(words[i:i + words_per_chunk])
        chunks.append(chunk)
        
    return chunks
```

## Extending the System

### 1. Adding a New Content Extraction Strategy

To add a new content extraction strategy:

```python
# 1. Define your strategy in enhanced_docs_crawler.py
def extract_content_with_custom_strategy(html, url):
    # Your custom extraction logic here
    return extracted_text

# 2. Add it to the extraction pipeline
extraction_strategies = [
    extract_content_html2text,
    extract_main_content,
    extract_raw_text,
    extract_content_with_custom_strategy  # New strategy
]

# 3. The system will try each strategy in order until one succeeds
```

### 2. Custom URL Filtering

To implement custom URL filtering:

```python
# Implement a custom URL filter function
def custom_url_filter(url):
    # Your filtering logic here
    return url.endswith('.html') and 'deprecated' not in url

# Set it in the CrawlConfig
config = CrawlConfig(
    # ...other parameters
    url_filter=custom_url_filter
)
```

## Best Practices

1. **Rate Limiting**: Always set reasonable concurrency limits to avoid overwhelming target servers:
    ```python
    config = CrawlConfig(
        # ...other parameters
        max_concurrent_requests=3  # Reasonable default
    )
    ```

2. **Error Handling**: Implement proper error handling and recovery:
    ```python
    try:
        result = await crawl_documentation(config)
    except Exception as e:
        # Log the error
        logging.error(f"Crawling failed: {e}")
        # Take appropriate action
    ```

3. **Content Validation**: Always validate extracted content:
    ```python
    if not content or len(content.strip()) < 50:
        # Content is likely not valid
        raise EmptyContentError("Extracted content is empty or too short")
    ```

4. **Incremental Crawling**: Use incremental crawling for large sites:
    ```python
    config = CrawlConfig(
        # ...other parameters
        incremental=True,  # Only process new or updated pages
        check_modified_since=True  # Check Last-Modified headers
    )
    ```

5. **Store Raw HTML**: For debugging and alternative processing:
    ```python
    config = CrawlConfig(
        # ...other parameters
        store_raw_html=True  # Keep original HTML
    )
    ```

6. **Monitor Performance**: Keep track of crawling metrics:
    ```python
    stats = get_crawl_stats(session_id)
    print(f"Progress: {stats['progress_percentage']}%")
    print(f"Success rate: {stats['success_rate']}%")
    print(f"Average processing time: {stats['avg_processing_time']}s")
    ```

7. **Respect robots.txt**: Always honor robots.txt directives:
    ```python
    config = CrawlConfig(
        # ...other parameters
        respect_robots_txt=True  # Default is True
    )
    ```

8. **Use Word-Based Chunking**: For more semantic division of content:
    ```python
    config = CrawlConfig(
        # ...other parameters
        use_word_based_chunking=True,
        chunk_words=1000,
        overlap_words=100
    )
    ``` 