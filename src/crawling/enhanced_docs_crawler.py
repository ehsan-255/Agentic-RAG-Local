import asyncio
import re
import requests
import httpx
import time
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

from openai import AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
from bs4 import BeautifulSoup
import html2text

from src.config import config
from src.utils.validation import validate_url
from src.crawling.batch_processor import EmbeddingBatchProcessor, LLMBatchProcessor
from src.db.async_schema import (
    add_documentation_source,
    update_documentation_source,
    add_site_page,
    delete_documentation_source as db_delete_documentation_source
)

@dataclass
class CrawlConfig:
    """Configuration for the crawler."""
    source_name: str
    source_id: str
    sitemap_url: str
    chunk_size: int = config.DEFAULT_CHUNK_SIZE
    max_concurrent_requests: int = config.DEFAULT_MAX_CONCURRENT_CRAWLS
    max_concurrent_api_calls: int = config.DEFAULT_MAX_CONCURRENT_API_CALLS
    retry_attempts: int = config.DEFAULT_RETRY_ATTEMPTS
    min_backoff: int = config.DEFAULT_MIN_BACKOFF
    max_backoff: int = config.DEFAULT_MAX_BACKOFF
    llm_model: str = config.LLM_MODEL
    embedding_model: str = config.EMBEDDING_MODEL
    url_patterns_include: List[str] = None
    url_patterns_exclude: List[str] = None
    
    def __post_init__(self):
        if self.url_patterns_include is None:
            self.url_patterns_include = []
        if self.url_patterns_exclude is None:
            self.url_patterns_exclude = []

@dataclass
class ProcessedChunk:
    """A processed text chunk with metadata and embedding."""
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = config.DEFAULT_CHUNK_SIZE) -> List[str]:
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

async def process_and_store_document(
    url: str, 
    html_content: str, 
    config: CrawlConfig,
    embedding_processor: EmbeddingBatchProcessor,
    llm_processor: LLMBatchProcessor
) -> int:
    """
    Process a document by chunking it, generating titles, summaries, and embeddings,
    and storing the chunks in the database.
    
    Args:
        url: URL of the document
        html_content: HTML content of the document
        config: Crawl configuration
        embedding_processor: Batch processor for embeddings
        llm_processor: Batch processor for LLM tasks
        
    Returns:
        int: Number of chunks processed and stored
    """
    try:
        # Convert HTML to text
        h2t = html2text.HTML2Text()
        h2t.ignore_links = False
        h2t.ignore_images = True
        markdown = h2t.handle(html_content)
        
        # Get the title from HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        page_title = soup.title.string if soup.title else "Untitled Document"
        
        # Sanitize the content - basic cleanup
        markdown = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', markdown, flags=re.DOTALL)
        markdown = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', markdown, flags=re.DOTALL)
        
        # Chunk the document
        chunks = chunk_text(markdown, config.chunk_size)
        
        if not chunks:
            print(f"No chunks generated for {url}")
            return 0
        
        # Prepare chunks for title/summary generation
        chunk_data = [{"content": chunk, "url": url} for chunk in chunks]
        
        # Generate titles and summaries for all chunks in batches
        titles_summaries = await llm_processor.generate_titles_and_summaries(chunk_data)
        
        # Generate embeddings for all chunks in batches
        embeddings = await embedding_processor.get_embeddings(chunks)
        
        # Store the chunks
        stored_count = 0
        for i, (chunk, title_summary, embedding) in enumerate(zip(chunks, titles_summaries, embeddings)):
            # Create metadata
            metadata = {
                "source_id": config.source_id,
                "source": config.source_name,
                "url": url,
                "chunk_number": i,
                "page_title": page_title,
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
        
        # Update the documentation source statistics
        await update_documentation_source(config.source_id, chunks_count=stored_count)
        
        return stored_count
    except Exception as e:
        print(f"Error processing document {url}: {e}")
        return 0

async def crawl_url(
    url: str, 
    config: CrawlConfig,
    crawl_semaphore: asyncio.Semaphore,
    embedding_processor: EmbeddingBatchProcessor,
    llm_processor: LLMBatchProcessor
) -> bool:
    """
    Crawl a single URL.
    
    Args:
        url: URL to crawl
        config: Crawl configuration
        crawl_semaphore: Semaphore for limiting concurrent requests
        embedding_processor: Batch processor for embeddings
        llm_processor: Batch processor for LLM tasks
        
    Returns:
        bool: True if crawling was successful, False otherwise
    """
    async with crawl_semaphore:
        try:
            # Validate the URL
            is_valid, error = validate_url(url)
            if not is_valid:
                print(f"Skipping invalid URL {url}: {error}")
                return False
            
            # Get the page content using async httpx instead of synchronous requests
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    print(f"Error fetching {url}: {response.status_code}")
                    return False
                
                # Process the document
                chunks_count = await process_and_store_document(
                    url,
                    response.text,
                    config,
                    embedding_processor,
                    llm_processor
                )
            
            if chunks_count > 0:
                # Update the documentation source statistics for the page
                await update_documentation_source(config.source_id, pages_count=1)
                print(f"Processed {url}: {chunks_count} chunks")
                return True
            else:
                print(f"Failed to process {url}: No chunks generated")
                return False
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return False

def filter_urls(urls: List[str], config: CrawlConfig) -> List[str]:
    """
    Filter URLs based on include and exclude patterns using regex.
    
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
            if any(re.search(pattern, url) for pattern in config.url_patterns_include):
                include_urls.append(url)
        filtered_urls = include_urls
    
    # Apply exclude patterns if any
    if config.url_patterns_exclude:
        exclude_urls = []
        for url in filtered_urls:
            if not any(re.search(pattern, url) for pattern in config.url_patterns_exclude):
                exclude_urls.append(url)
        filtered_urls = exclude_urls
    
    return filtered_urls

def get_urls_from_sitemap(sitemap_url: str, config: CrawlConfig) -> List[str]:
    """
    Extract URLs from a sitemap XML file with proper error handling.
    
    Args:
        sitemap_url: URL of the sitemap
        config: Crawl configuration
        
    Returns:
        List[str]: List of URLs from the sitemap
    """
    # Validate the sitemap URL
    is_valid, error = validate_url(sitemap_url)
    if not is_valid:
        print(f"Invalid sitemap URL: {error}")
        return []
    
    try:
        # Get the sitemap content with timeout
        response = requests.get(sitemap_url, timeout=30)
        if response.status_code != 200:
            print(f"Error fetching sitemap {sitemap_url}: {response.status_code}")
            return []
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract URLs using namespace
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
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching sitemap {sitemap_url}: {e}")
        return []
    except ElementTree.ParseError as e:
        print(f"XML parsing error for sitemap {sitemap_url}: {e}")
        return []
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
        # Delete the documentation source (which cascades to delete all pages)
        result = await db_delete_documentation_source(source_id)
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
        urls = get_urls_from_sitemap(config.sitemap_url, config)
        
        if not urls:
            print(f"No URLs found in sitemap {config.sitemap_url}")
            return False
        
        print(f"Found {len(urls)} URLs in sitemap")
        
        # Create processors for batch operations
        embedding_processor = EmbeddingBatchProcessor(
            openai_client=openai_client,
            model=config.embedding_model,
            batch_size=config.EMBEDDING_BATCH_SIZE if hasattr(config, 'EMBEDDING_BATCH_SIZE') else 10,
            max_concurrent_batches=config.EMBEDDING_MAX_CONCURRENT_BATCHES if hasattr(config, 'EMBEDDING_MAX_CONCURRENT_BATCHES') else 3
        )
        
        llm_processor = LLMBatchProcessor(
            openai_client=openai_client,
            model=config.llm_model,
            batch_size=config.LLM_BATCH_SIZE if hasattr(config, 'LLM_BATCH_SIZE') else 5,
            max_concurrent_batches=config.LLM_MAX_CONCURRENT_BATCHES if hasattr(config, 'LLM_MAX_CONCURRENT_BATCHES') else 2
        )
        
        # Create semaphore for concurrent requests
        crawl_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Create tasks for all URLs
        tasks = []
        for url in urls:
            task = crawl_url(url, config, crawl_semaphore, embedding_processor, llm_processor)
            tasks.append(task)
        
        # Process URLs in parallel
        results = await asyncio.gather(*tasks)
        
        # Count successful crawls
        success_count = sum(1 for result in results if result)
        
        print(f"Successfully processed {success_count} out of {len(urls)} URLs")
        
        return success_count > 0
    except Exception as e:
        print(f"Error crawling documentation: {e}")
        return False 