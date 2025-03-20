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
3. **Content Chunking**: Smart text division for optimal RAG performance
4. **Error Resilience**: Robust error handling and recovery
5. **State Management**: Tracking crawl progress for resumability

## Key Components

### 1. Enhanced Document Crawler (`src/crawling/enhanced_docs_crawler.py`)

The main crawler implementation that manages the entire crawling process:

```python
from src.crawling.enhanced_docs_crawler import crawl_documentation, CrawlConfig

# Create a crawl configuration
config = CrawlConfig(
    source_id="python_docs",
    source_name="Python Documentation",
    sitemap_url="https://docs.python.org/3/sitemap.xml",
    chunk_size=5000,
    max_concurrent_requests=5,
    max_concurrent_api_calls=3,
    url_patterns_include=["/reference/", "/tutorial/"],
    url_patterns_exclude=["/whatsnew/"]
)

# Start the crawl process
await crawl_documentation(openai_client, config)
```

### 2. Batch Processor (`src/crawling/batch_processor.py`)

Manages batch processing for embeddings and LLM operations:

```python
from src.crawling.batch_processor import EmbeddingBatchProcessor, LLMBatchProcessor

# Create batch processors
embedding_processor = EmbeddingBatchProcessor(
    openai_client, 
    batch_size=20,
    max_concurrent_requests=3
)

llm_processor = LLMBatchProcessor(
    openai_client,
    model="gpt-4o-mini",
    max_concurrent_requests=2
)

# Process embeddings in batches
embeddings = await embedding_processor.get_embeddings(text_chunks)

# Generate titles and summaries
titles_summaries = await llm_processor.generate_titles_and_summaries(chunk_data)
```

### 3. URL and Content Processing

Functions for processing URLs and content:

```python
from src.crawling.enhanced_docs_crawler import filter_urls, process_and_store_document

# Filter URLs based on patterns
filtered_urls = filter_urls(
    urls=all_urls,
    config=crawl_config
)

# Process a document
chunks_stored = await process_and_store_document(
    url="https://docs.python.org/3/tutorial/index.html",
    html_content=html_content,
    config=crawl_config,
    embedding_processor=embedding_processor,
    llm_processor=llm_processor
)
```

### 4. Sitemap Parsing

Functions for extracting URLs from sitemaps:

```python
from src.crawling.enhanced_docs_crawler import get_urls_from_sitemap

# Extract URLs from a sitemap
urls = await get_urls_from_sitemap(
    sitemap_url="https://docs.python.org/3/sitemap.xml",
    config=crawl_config
)
```

## Integration Points

### Adding a New Documentation Source

To add a new documentation source to the system:

```python
from src.crawling.enhanced_docs_crawler import crawl_documentation, CrawlConfig
from openai import AsyncOpenAI

async def add_new_documentation_source(source_name, sitemap_url):
    # Initialize OpenAI client
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a unique source ID
    source_id = f"{source_name.lower().replace(' ', '_')}_{int(time.time())}"
    
    # Create configuration
    config = CrawlConfig(
        source_id=source_id,
        source_name=source_name,
        sitemap_url=sitemap_url,
        # Additional parameters with defaults
        chunk_size=5000,
        max_concurrent_requests=5,
        max_concurrent_api_calls=3
    )
    
    # Perform the crawl
    success = await crawl_documentation(openai_client, config)
    
    return success, source_id
```

### Clearing Existing Documentation

To remove a documentation source:

```python
from src.crawling.enhanced_docs_crawler import clear_documentation_source

# Clear all data for a source
success = await clear_documentation_source(source_id="python_docs")
```

### Monitoring Crawl Progress

To monitor the crawling process:

```python
from src.utils.enhanced_logging import get_active_session
from src.utils.task_monitoring import get_tasks_by_type, TaskType

# Get the active crawl session
session = get_active_session()
if session:
    # Get progress statistics
    total_pages = session.total_urls
    processed_pages = session.processed_urls
    success_rate = session.success_rate
    
    print(f"Progress: {processed_pages}/{total_pages} pages ({success_rate:.2f}% success)")

# Get active tasks
crawl_tasks = get_tasks_by_type(TaskType.PAGE_CRAWLING)
processing_tasks = get_tasks_by_type(TaskType.PAGE_PROCESSING)

print(f"Active crawl tasks: {len(crawl_tasks)}")
print(f"Active processing tasks: {len(processing_tasks)}")
```

## Content Processing

### Text Chunking

The system divides document content into manageable chunks for optimal retrieval:

```python
from src.crawling.enhanced_docs_crawler import chunk_text

# Chunk a document
chunks = chunk_text(
    text="Long document text...",
    chunk_size=2000
)

# Process each chunk
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} characters")
```

### HTML Processing

The crawler converts HTML to markdown for better text processing:

```python
import html2text
from bs4 import BeautifulSoup

def extract_content(html_content):
    # Extract title
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.string if soup.title else "Untitled Document"
    
    # Convert HTML to text
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    h2t.ignore_images = True
    markdown_content = h2t.handle(html_content)
    
    return title, markdown_content
```

### Metadata Extraction

The system extracts and stores metadata about documents:

```python
def extract_metadata(url, html_content, source_id):
    """Extract metadata from the document."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract metadata
    metadata = {
        "source_id": source_id,
        "url": url,
        "title": soup.title.string if soup.title else "Untitled",
        "last_modified": None,
        "content_type": None
    }
    
    # Try to get last modified date
    modified_meta = soup.find("meta", {"name": "last-modified"})
    if modified_meta and modified_meta.get("content"):
        metadata["last_modified"] = modified_meta.get("content")
    
    # Try to get content type
    content_type_meta = soup.find("meta", {"name": "content-type"})
    if content_type_meta and content_type_meta.get("content"):
        metadata["content_type"] = content_type_meta.get("content")
    
    return metadata
```

## Extending the System

### Creating a Custom Crawler

To create a specialized crawler for specific sites:

```python
from src.crawling.enhanced_docs_crawler import CrawlConfig, crawl_url, process_and_store_document
import asyncio
import httpx

async def custom_github_docs_crawler(openai_client, repository, source_id, source_name):
    """Custom crawler for GitHub repository documentation."""
    # Create configuration
    config = CrawlConfig(
        source_id=source_id,
        source_name=source_name,
        sitemap_url=f"https://github.com/{repository}",
        chunk_size=3000,
        max_concurrent_requests=5,
        max_concurrent_api_calls=3,
        url_patterns_include=["/wiki/", "/blob/main/docs/"],
        url_patterns_exclude=["/issues/", "/pull/"]
    )
    
    # Create batch processors
    embedding_processor = EmbeddingBatchProcessor(openai_client, batch_size=20)
    llm_processor = LLMBatchProcessor(openai_client)
    
    # Get docs from wiki and markdown files
    wiki_urls = [f"https://github.com/{repository}/wiki"]
    blob_urls = []
    
    # Get blob URLs (markdown files in docs directory)
    async with httpx.AsyncClient() as client:
        # GitHub API request to list files in docs directory
        # Implementation details...
    
    # Process all URLs
    urls = wiki_urls + blob_urls
    crawl_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    tasks = []
    for url in urls:
        task = asyncio.create_task(crawl_url(
            url=url,
            config=config,
            crawl_semaphore=crawl_semaphore,
            embedding_processor=embedding_processor,
            llm_processor=llm_processor
        ))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    success_count = sum(1 for r in results if r is True)
    return success_count == len(urls)
```

### Adding Custom Content Processors

To add specialized content processing for specific document types:

```python
async def process_pdf_document(url, pdf_content, config, embedding_processor, llm_processor):
    """Process a PDF document."""
    try:
        # Convert PDF to text (using PyPDF2, pdf2text, or similar)
        # Implementation details...
        
        # Chunk the text
        chunks = chunk_text(text_content, config.chunk_size)
        
        # Process chunks similar to HTML documents
        chunk_data = [{"content": chunk, "url": url} for chunk in chunks]
        titles_summaries = await llm_processor.generate_titles_and_summaries(chunk_data)
        embeddings = await embedding_processor.get_embeddings(chunks)
        
        # Store chunks
        stored_count = 0
        for i, (chunk, title_summary, embedding) in enumerate(zip(chunks, titles_summaries, embeddings)):
            # Create metadata
            metadata = {
                "source_id": config.source_id,
                "source": config.source_name,
                "url": url,
                "chunk_number": i,
                "page_title": f"Page {i+1}",
                "document_type": "pdf",
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Store the chunk
            chunk_id = await add_site_page(
                url=url,
                chunk_number=i,
                title=title_summary["title"],
                summary=title_summary["summary"],
                content=chunk,
                metadata=metadata,
                embedding=embedding
            )
            
            if chunk_id:
                stored_count += 1
        
        return stored_count
    except Exception as e:
        enhanced_crawler_logger.structured_error(
            f"Error processing PDF document: {e}",
            error=e,
            url=url
        )
        return 0
```

### Custom URL Filtering

To implement specialized URL filtering:

```python
def advanced_url_filter(urls, config):
    """Advanced URL filtering with priority scoring."""
    filtered_urls = []
    
    # Score and filter URLs
    for url in urls:
        # Skip if explicitly excluded
        if any(pattern in url for pattern in config.url_patterns_exclude):
            continue
        
        # Always include if explicitly included
        if any(pattern in url for pattern in config.url_patterns_include):
            priority = 1  # Highest priority
            filtered_urls.append((url, priority))
            continue
        
        # Score other URLs based on heuristics
        score = 0
        
        # Prefer shorter URLs (likely higher in hierarchy)
        path_length = url.count("/")
        if path_length < 4:
            score += 2
        elif path_length < 6:
            score += 1
        
        # Prefer index pages
        if url.endswith("index.html") or url.endswith("/"):
            score += 2
        
        # Filter by minimum score
        if score >= 1:
            filtered_urls.append((url, score))
    
    # Sort by priority and return URLs
    filtered_urls.sort(key=lambda x: x[1], reverse=True)
    return [url for url, _ in filtered_urls]
```

## Best Practices

### Optimizing Crawl Performance

1. **Concurrency Tuning**: 
   - Adjust `max_concurrent_requests` based on target server capabilities
   - Start with conservative values (3-5) and increase gradually
   - Monitor for rate limiting or server rejection

2. **Batching Strategies**:
   - Batch embedding requests for better throughput
   - Process document chunks in parallel
   - Use appropriate batch sizes (10-50 items) for API calls

3. **Memory Management**:
   - Avoid loading entire documents into memory at once
   - Process large documents in streaming mode when possible
   - Implement backpressure when processing queues grow too large

4. **Rate Limit Handling**:
   - Implement exponential backoff for rate limit errors
   - Add jitter to retry intervals to avoid stampeding
   - Set appropriate rate limits for different API endpoints

### Content Processing

1. **Chunk Size Optimization**:
   - Test different chunk sizes for your content type (1000-5000 chars)
   - Preserve semantic boundaries (paragraphs, sections) when possible
   - Consider content density when setting chunk size

2. **Quality Control**:
   - Validate content before processing (remove boilerplate, navigation)
   - Filter out irrelevant or duplicate content
   - Normalize whitespace and formatting

3. **Metadata Enrichment**:
   - Add as much metadata as possible to chunks for filtering
   - Include hierarchical information (section, chapter, document)
   - Store timestamps and version information

### Error Handling

1. **Resilient Crawling**:
   - Continue processing despite individual page failures
   - Implement proper error categorization and logging
   - Add circuit breakers for systemic failures

2. **Retry Strategies**:
   - Use tenacity for robust retry handling
   - Implement different retry policies for different error types
   - Set appropriate timeout values for HTTP requests

3. **Validation**:
   - Validate URLs before crawling
   - Check document size and type before processing
   - Verify chunks meet minimum quality standards

### Extending for New Content Types

When adding support for new content types:

1. **Parser Selection**:
   - Choose appropriate parser libraries for the content type
   - Implement content-specific extraction logic
   - Normalize extracted content to a consistent format

2. **Chunking Strategy**:
   - Develop content-appropriate chunking strategies
   - Preserve structural elements (headings, sections)
   - Include appropriate cross-references between chunks

3. **Metadata Extraction**:
   - Extract content-specific metadata (author, version, etc.)
   - Map metadata to standardized fields
   - Store source-specific fields in extended metadata 