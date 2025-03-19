import os
import sys
import json
import asyncio
import requests
import time
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI, RateLimitError

# Import database functions
from src.db.schema import (
    add_documentation_source,
    add_site_page,
    update_documentation_source,
    delete_documentation_source as db_delete_documentation_source
)

load_dotenv()

# Default configuration values
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_MAX_CONCURRENT_CRAWLS = 3
DEFAULT_MAX_CONCURRENT_API_CALLS = 5
DEFAULT_RETRY_ATTEMPTS = 6
DEFAULT_MIN_BACKOFF = 1
DEFAULT_MAX_BACKOFF = 60

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class CrawlConfig:
    """Configuration for the crawler."""
    source_name: str
    source_id: str  # Unique identifier for this documentation source
    sitemap_url: str
    chunk_size: int = DEFAULT_CHUNK_SIZE
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_CRAWLS
    max_concurrent_api_calls: int = DEFAULT_MAX_CONCURRENT_API_CALLS
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    min_backoff: int = DEFAULT_MIN_BACKOFF
    max_backoff: int = DEFAULT_MAX_BACKOFF
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = "text-embedding-3-small"
    url_patterns_include: List[str] = None  # Patterns to include in URL filtering
    url_patterns_exclude: List[str] = None  # Patterns to exclude in URL filtering
    
    def __post_init__(self):
        if self.url_patterns_include is None:
            self.url_patterns_include = []
        if self.url_patterns_exclude is None:
            self.url_patterns_exclude = []

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Split text into roughly equal sized chunks based on the specified chunk size.
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum chunk size in characters
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by double newlines to maintain paragraph structure
    paragraphs = text.split("\n\n")
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds the chunk size and we already have content,
        # finish the current chunk and start a new one
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        
        # If a single paragraph is larger than the chunk size, split it by sentences
        if len(paragraph) > chunk_size:
            # Simple sentence splitting (not perfect but functional)
            sentences = paragraph.replace(". ", ".|").replace("? ", "?|").replace("! ", "!|").split("|")
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    current_chunk += sentence + " "
        else:
            current_chunk += paragraph + "\n\n"
    
    # Add any remaining content
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=DEFAULT_MIN_BACKOFF, max=DEFAULT_MAX_BACKOFF),
    stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS)
)
async def get_title_and_summary(chunk: str, url: str, model: str = "gpt-4o-mini") -> Dict[str, str]:
    """
    Generate a title and summary for a text chunk using OpenAI.
    
    Args:
        chunk: Text chunk to generate title and summary for
        url: URL of the page
        model: OpenAI model to use
        
    Returns:
        Dict[str, str]: Dictionary with title and summary
    """
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI that extracts precise titles and summaries from documentation chunks."},
            {"role": "user", "content": f"URL: {url}\n\nContent: {chunk[:4000]}...\n\nGenerate a concise title (max 80 chars) and summary (100-150 chars) for this documentation chunk."}
        ],
        temperature=0.3,
        max_tokens=150
    )
    
    text = response.choices[0].message.content
    lines = text.strip().split("\n")
    
    title = lines[0].replace("Title: ", "") if lines and "Title:" in lines[0] else "Untitled Document"
    summary = lines[1].replace("Summary: ", "") if len(lines) > 1 and "Summary:" in lines[1] else "No summary available"
    
    return {"title": title, "summary": summary}

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=DEFAULT_MIN_BACKOFF, max=DEFAULT_MAX_BACKOFF),
    stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS)
)
async def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate an embedding for a text using OpenAI.
    
    Args:
        text: Text to generate embedding for
        model: OpenAI embedding model to use
        
    Returns:
        List[float]: Embedding vector
    """
    response = await openai_client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

async def process_chunk(chunk: str, chunk_number: int, url: str, config: CrawlConfig, api_semaphore: asyncio.Semaphore) -> ProcessedChunk:
    """
    Process a text chunk by generating a title, summary, and embedding.
    
    Args:
        chunk: Text chunk to process
        chunk_number: Chunk number (for ordering)
        url: URL of the page
        config: Crawl configuration
        api_semaphore: Semaphore for limiting API calls
        
    Returns:
        ProcessedChunk: Processed chunk object
    """
    async with api_semaphore:
        # Generate title and summary
        title_summary = await get_title_and_summary(chunk, url, config.llm_model)
        
        # Generate embedding
        embedding = await get_embedding(chunk, config.embedding_model)
    
    # Create metadata
    metadata = {
            "source_id": config.source_id,
        "source": config.source_name,
            "url": url,
            "chunk_number": chunk_number,
            "processed_at": datetime.now(timezone.utc).isoformat()
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
            title=title_summary["title"],
            summary=title_summary["summary"],
            content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """
    Insert a processed chunk into the database.
    
    Args:
        chunk: Processed chunk to insert
        
    Returns:
        bool: True if insertion was successful, False otherwise
    """
    try:
        # Insert the chunk using the database function
        result = add_site_page(
            url=chunk.url,
            chunk_number=chunk.chunk_number,
            title=chunk.title,
            summary=chunk.summary,
            content=chunk.content,
            metadata=chunk.metadata,
            embedding=chunk.embedding
        )
        
        return result is not None
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return False

async def process_and_store_document(url: str, markdown: str, config: CrawlConfig, api_semaphore: asyncio.Semaphore):
    """
    Process a document by chunking it, generating titles, summaries, and embeddings,
    and storing the chunks in the database.
    
    Args:
        url: URL of the document
        markdown: Markdown content of the document
        config: Crawl configuration
        api_semaphore: Semaphore for limiting API calls
        
    Returns:
        int: Number of chunks processed and stored
    """
    try:
        # Chunk the document
        chunks = chunk_text(markdown, config.chunk_size)
        
        # Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunk = await process_chunk(chunk, i, url, config, api_semaphore)
            processed_chunks.append(processed_chunk)
        
        # Store the chunks
        for chunk in processed_chunks:
            await insert_chunk(chunk)
        
        # Update the documentation source statistics
        await update_documentation_source(config.source_id, chunks_count=len(chunks))
        
        return len(chunks)
    except Exception as e:
        print(f"Error processing document: {e}")
        return 0

async def update_documentation_source(source_id: str, pages_count: Optional[int] = None, chunks_count: Optional[int] = None):
    """
    Update the statistics for a documentation source.
    
    Args:
        source_id: ID of the documentation source
        pages_count: Number of pages to add (optional)
        chunks_count: Number of chunks to add (optional)
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        result = update_documentation_source(source_id, pages_count, chunks_count)
        return result
    except Exception as e:
        print(f"Error updating documentation source: {e}")
        return False

async def crawl_parallel(urls: List[str], config: CrawlConfig):
    """
    Crawl multiple URLs in parallel.
    
    Args:
        urls: List of URLs to crawl
        config: Crawl configuration
        
    Returns:
        int: Number of pages processed
    """
    # Create semaphores for rate limiting
    api_semaphore = asyncio.Semaphore(config.max_concurrent_api_calls)
    crawl_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    async def process_url(url: str):
        """Process a single URL."""
        async with crawl_semaphore:
            try:
                # Get the page content
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"Error fetching {url}: {response.status_code}")
                    return 0
                
                # Process the document
                chunks_count = await process_and_store_document(url, response.text, config, api_semaphore)
                
                # Update the documentation source statistics for the page
                await update_documentation_source(config.source_id, pages_count=1)
                
                print(f"Processed {url}: {chunks_count} chunks")
                return 1
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return 0
    
    # Create tasks for all URLs
    tasks = [process_url(url) for url in urls]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Return the total number of pages processed
    return sum(results)

def filter_urls(urls: List[str], config: CrawlConfig) -> List[str]:
    """
    Filter URLs based on include and exclude patterns.
    
    Args:
        urls: List of URLs to filter
        config: Crawl configuration with include and exclude patterns
        
    Returns:
        List[str]: Filtered list of URLs
    """
    filtered_urls = urls
    
    # Apply include patterns if any
    if config.url_patterns_include:
        include_urls = []
        for url in filtered_urls:
            if any(pattern in url for pattern in config.url_patterns_include):
                include_urls.append(url)
        filtered_urls = include_urls
    
    # Apply exclude patterns if any
    if config.url_patterns_exclude:
        exclude_urls = []
        for url in filtered_urls:
            if not any(pattern in url for pattern in config.url_patterns_exclude):
                exclude_urls.append(url)
        filtered_urls = exclude_urls
    
    return filtered_urls

def get_urls_from_sitemap(sitemap_url: str, config: CrawlConfig) -> List[str]:
    """
    Extract URLs from a sitemap XML file.
    
    Args:
        sitemap_url: URL of the sitemap
        config: Crawl configuration
        
    Returns:
        List[str]: List of URLs from the sitemap
    """
    try:
        # Get the sitemap content
        response = requests.get(sitemap_url)
        if response.status_code != 200:
            print(f"Error fetching sitemap {sitemap_url}: {response.status_code}")
            return []
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract URLs using namespace
        # Try with and without namespace
        namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = []
        
        # Check if it's a sitemap index file
        sitemap_elements = root.findall(".//sm:sitemap", namespaces) or root.findall(".//sitemap")
        if sitemap_elements:
            # It's a sitemap index, recursively fetch each sitemap
            for sitemap_element in sitemap_elements:
                sitemap_loc = sitemap_element.find(".//sm:loc", namespaces) or sitemap_element.find("loc")
                if sitemap_loc is not None and sitemap_loc.text:
                    sub_urls = get_urls_from_sitemap(sitemap_loc.text, config)
                    urls.extend(sub_urls)
        else:
            # It's a regular sitemap, extract URLs
            url_elements = root.findall(".//sm:url", namespaces) or root.findall(".//url")
            for url_element in url_elements:
                loc = url_element.find(".//sm:loc", namespaces) or url_element.find("loc")
                if loc is not None and loc.text:
                    urls.append(loc.text)
        
        # Filter URLs
        return filter_urls(urls, config)
    except Exception as e:
        print(f"Error parsing sitemap {sitemap_url}: {e}")
        return []

async def clear_documentation_source(source_id: str) -> bool:
    """
    Delete a documentation source and all its content from the database.
    
    Args:
        source_id: ID of the documentation source to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        # Delete the documentation source
        result = db_delete_documentation_source(source_id)
        return result
    except Exception as e:
        print(f"Error clearing documentation source: {e}")
        return False

async def crawl_documentation(openai_client: AsyncOpenAI, config: CrawlConfig) -> bool:
    """
    Crawl a documentation site and store its content in the database.
    
    Args:
        openai_client: OpenAI client
        config: Crawl configuration
        
    Returns:
        bool: True if crawling was successful, False otherwise
    """
    try:
        print(f"Starting crawl for {config.source_name} ({config.source_id})")
        
        # Get URLs from sitemap
        from src.crawling.enhanced_docs_crawler import get_urls_from_sitemap
        urls = await get_urls_from_sitemap(config.sitemap_url, config)
        print(f"Found {len(urls)} URLs in sitemap")
        
        if not urls:
            print(f"No URLs found in sitemap {config.sitemap_url}")
            return False
        
        # Crawl the URLs in parallel
        pages_count = await crawl_parallel(urls, config)
        print(f"Processed {pages_count} pages")
        
        return pages_count > 0
    except Exception as e:
        print(f"Error crawling documentation: {e}")
        return False
