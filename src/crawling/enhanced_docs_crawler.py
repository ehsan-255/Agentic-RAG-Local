import asyncio
import re
import requests
import httpx
import time
import logging
import streamlit as st
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

# Import monitoring utilities
from src.utils.enhanced_logging import (
    enhanced_crawler_logger,
    start_crawl_session,
    end_crawl_session,
    get_active_session
)
from src.utils.task_monitoring import (
    TaskType,
    monitored_task,
    cancel_all_tasks
)
from src.utils.errors import (
    ContentProcessingError,
    EmptyContentError,
    ParseError,
    ChunkingError
)

# Import db_utils for compatibility
from src.db.db_utils import is_database_available

# Import database functions - these will use the appropriate driver
try:
    from src.db.async_schema import (
        add_documentation_source,
        update_documentation_source,
        add_site_page,
        delete_documentation_source as db_delete_documentation_source
    )
except ImportError as e:
    logging.error(f"Error importing async_schema: {e}")
    # Create fallback functions to prevent crashes
    async def add_documentation_source(*args, **kwargs):
        logging.error("Database functions unavailable: add_documentation_source")
        return None
        
    async def update_documentation_source(*args, **kwargs):
        logging.error("Database functions unavailable: update_documentation_source")
        return None
        
    async def add_site_page(*args, **kwargs):
        logging.error("Database functions unavailable: add_site_page")
        return None
        
    async def db_delete_documentation_source(*args, **kwargs):
        logging.error("Database functions unavailable: delete_documentation_source")
        return False

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

@monitored_task(TaskType.PAGE_PROCESSING, "Processing document {url}")
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
        # Diagnostic logging
        enhanced_crawler_logger.info(
            f"Processing document {url} with size {len(html_content)} bytes"
        )
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Get the title from HTML
        page_title = soup.title.string if soup.title else "Untitled Document"
        enhanced_crawler_logger.info(f"Page title: {page_title}")
        
        # Try multiple extraction strategies
        markdown = ""
        extraction_methods = [
            # Method 1: HTML2Text standard conversion
            lambda html: html2text.HTML2Text().handle(html),
            
            # Method 2: Extract from main content areas first
            lambda html: extract_from_content_areas(soup),
            
            # Method 3: Raw text as fallback
            lambda html: soup.get_text()
        ]
        
        # Try each extraction method until we get reasonable content
        for method_idx, extraction_method in enumerate(extraction_methods):
            try:
                current_markdown = extraction_method(html_content)
                enhanced_crawler_logger.info(
                    f"Extraction method {method_idx+1} produced {len(current_markdown)} chars"
                )
                
                if current_markdown and len(current_markdown) > 100:
                    markdown = current_markdown
                    break
            except Exception as e:
                enhanced_crawler_logger.warning(
                    f"Extraction method {method_idx+1} failed: {str(e)}"
                )
                continue
                
        # Sanitize the content - basic cleanup
        markdown = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', markdown, flags=re.DOTALL)
        markdown = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', markdown, flags=re.DOTALL)
        
        # Chunk the document
        chunks = chunk_text(markdown, config.chunk_size)
        enhanced_crawler_logger.info(f"Generated {len(chunks)} chunks from document")
        
        if not chunks and len(markdown) > 0:
            # Force creation of at least one chunk for minimal content
            enhanced_crawler_logger.warning(
                f"No chunks were generated by standard chunker. Creating minimal chunk."
            )
            # Create at least one chunk with available content
            chunks = [markdown[:min(len(markdown), config.chunk_size)]]
        
        if not chunks:
            error = EmptyContentError(url, len(markdown) if markdown else 0)
            enhanced_crawler_logger.structured_error(
                f"No chunks generated for {url} - Content length: {len(markdown)}, HTML length: {len(html_content)}",
                error=error,
                url=url
            )
            return 0
        
        # Get active session
        session = get_active_session()
        if session:
            try:
                session.record_page_processed(url, True)
            except Exception as session_error:
                enhanced_crawler_logger.warning(f"Error updating session stats: {session_error}")
            
        # Prepare chunks for title/summary generation
        chunk_data = [{"content": chunk, "url": url} for chunk in chunks]
        
        # Generate titles and summaries for all chunks in batches
        titles_summaries = await llm_processor.generate_titles_and_summaries(chunk_data)
        
        # Generate embeddings for all chunks in batches
        try:
            embeddings = await embedding_processor.get_embeddings(chunks)
        except Exception as emb_error:
            enhanced_crawler_logger.structured_error(
                f"Error generating embeddings for {url}: {emb_error}",
                error=emb_error,
                url=url
            )
            # Use zero embeddings as fallback to prevent total failure
            embeddings = [[0.0] * 1536 for _ in chunks]
        
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
        # Record page processing failure in monitoring
        session = get_active_session()
        if session:
            try:
                session.record_page_processed(url, False)
            except Exception as session_error:
                enhanced_crawler_logger.warning(f"Error updating session stats: {session_error}")
        
        # Log structured error
        if isinstance(e, ContentProcessingError):
            error = e
        else:
            error = ContentProcessingError(f"Error processing document: {e}", url)
            
        enhanced_crawler_logger.structured_error(
            f"Error processing document {url}: {e}",
            error=error,
            url=url,
            config_source_id=config.source_id
        )
        return 0

def extract_from_content_areas(soup):
    """
    Extract text content from typical content areas in HTML.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        str: Extracted text content
    """
    content = ""
    
    # Try to find main content containers by common IDs and classes
    content_selectors = [
        "main", "article", "#content", ".content", 
        "#main-content", ".main-content", ".document", 
        ".documentation", ".doc-content", ".page-content"
    ]
    
    # Try each selector
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            # Join text from all matching elements
            for element in elements:
                content += element.get_text() + "\n\n"
            if len(content) > 200:  # Enough content found
                break
    
    # If no suitable content found, get text from body
    if len(content) < 200 and soup.body:
        content = soup.body.get_text()
    
    # Fallback to full document text
    if len(content) < 100:
        content = soup.get_text()
        
    return content

@monitored_task(TaskType.PAGE_PROCESSING, "Crawling URL {url}")
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
            # Check if crawl should be paused or stopped
            if st.session_state.get('pause_crawl', False):
                enhanced_crawler_logger.info(f"Crawl paused on URL: {url}")
                while st.session_state.get('pause_crawl', False):
                    if st.session_state.get('stop_crawl', False):
                        enhanced_crawler_logger.info(f"Crawl stopped while paused: {url}")
                        return False
                    await asyncio.sleep(1)
                enhanced_crawler_logger.info(f"Crawl resumed on URL: {url}")
                    
            if st.session_state.get('stop_crawl', False):
                enhanced_crawler_logger.info(f"Crawl stopped: {url}")
                return False
            
            # Validate the URL
            is_valid, error = validate_url(url)
            if not is_valid:
                enhanced_crawler_logger.warning(f"Skipping invalid URL {url}: {error}")
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
            # Record failure in monitoring
            session = get_active_session()
            if session:
                try:
                    # Only pass the url and success parameters (not the error)
                    session.record_page_processed(url, False)
                    
                    # Log the error separately
                    enhanced_crawler_logger.warning(
                        f"Error processing URL {url}: {str(e)}", 
                        url=url,
                        error=str(e)
                    )
                except Exception as session_error:
                    enhanced_crawler_logger.warning(f"Error updating session stats: {session_error}")
                
            enhanced_crawler_logger.structured_error(
                f"Error crawling URL {url}: {e}",
                error=e,
                url=url,
                config_source_id=config.source_id
            )
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
    
     # Set default documentation patterns if none provided
    if not config.url_patterns_include:
        config.url_patterns_include = [
            '/docs/',
            '/documentation/',
            '/guide/',
            '/manual/',
            '/reference/',
            '/tutorial/',
            '/api/',
            '/learn/'
        ]
    
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
            if not any(re.search(pattern, url) for pattern in config.url_patterns_exclude):
                exclude_urls.append(url)
        filtered_urls = exclude_urls
    
    return filtered_urls

async def get_urls_from_sitemap(sitemap_url: str, config: CrawlConfig) -> List[str]:
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
        # Get the sitemap content with timeout using async httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(sitemap_url)
            if response.status_code != 200:
                print(f"Error fetching sitemap {sitemap_url}: {response.status_code}")
                return []
            
            # Print the first 200 chars of the response for debugging
            print(f"Sitemap response preview: {response.text[:200]}...")
            
            # Parse the XML
            try:
                root = ElementTree.fromstring(response.text)
            except ElementTree.ParseError as e:
                print(f"XML parsing error for sitemap {sitemap_url}: {e}")
                print(f"Response content type: {response.headers.get('content-type', 'unknown')}")
                return []
            
            # Different sitemap namespaces that might be used
            namespaces = {
                'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'
            }
            urls = []
            
            # Check if it's a sitemap index file (contains <sitemap> elements)
            sitemap_elements = []
            
            # Try with namespace first
            for ns_prefix, ns_uri in namespaces.items():
                sitemap_elements = root.findall(f".//{ns_prefix}:sitemap", {ns_prefix: ns_uri})
                if sitemap_elements:
                    break
            
            # If no sitemap elements found with namespace, try without namespace
            if not sitemap_elements:
                sitemap_elements = root.findall(".//sitemap")
            
            # Also check for urlset tag with sitemap elements as direct children
            if not sitemap_elements and root.tag.endswith('urlset'):
                sitemap_elements = root.findall("./sitemap")
            
            if sitemap_elements:
                # It's a sitemap index, recursively fetch each sitemap
                print(f"Found sitemap index with {len(sitemap_elements)} sitemaps")
                
                for sitemap_element in sitemap_elements:
                    # Try with namespace first for loc element
                    sitemap_loc = None
                    for ns_prefix, ns_uri in namespaces.items():
                        sitemap_loc = sitemap_element.find(f".//{ns_prefix}:loc", {ns_prefix: ns_uri})
                        if sitemap_loc is not None:
                            break
                    
                    # If not found with namespace, try without namespace
                    if sitemap_loc is None:
                        sitemap_loc = sitemap_element.find(".//loc") or sitemap_element.find("loc")
                    
                    if sitemap_loc is not None and sitemap_loc.text:
                        sub_sitemap_url = sitemap_loc.text.strip()
                        print(f"Processing sub-sitemap: {sub_sitemap_url}")
                        sub_urls = await get_urls_from_sitemap(sub_sitemap_url, config)
                        urls.extend(sub_urls)
            else:
                # It's a regular sitemap, extract URLs from <url> elements
                url_elements = []
                
                # Try with namespace first
                for ns_prefix, ns_uri in namespaces.items():
                    url_elements = root.findall(f".//{ns_prefix}:url", {ns_prefix: ns_uri})
                    if url_elements:
                        break
                
                # If no url elements found with namespace, try without namespace
                if not url_elements:
                    url_elements = root.findall(".//url")
                
                print(f"Found regular sitemap with {len(url_elements)} URLs")
                
                for url_element in url_elements:
                    # Try with namespace first for loc element
                    loc = None
                    for ns_prefix, ns_uri in namespaces.items():
                        loc = url_element.find(f".//{ns_prefix}:loc", {ns_prefix: ns_uri})
                        if loc is not None:
                            break
                    
                    # If not found with namespace, try without namespace
                    if loc is None:
                        loc = url_element.find(".//loc") or url_element.find("loc")
                    
                    if loc is not None and loc.text:
                        urls.append(loc.text.strip())
            
            # Filter URLs
            filtered_urls = filter_urls(urls, config)
            print(f"Found {len(urls)} URLs, filtered to {len(filtered_urls)}")
            return filtered_urls
    except httpx.RequestError as e:
        print(f"Network error fetching sitemap {sitemap_url}: {e}")
        return []
    except Exception as e:
        print(f"Error parsing sitemap {sitemap_url}: {e}")
        print(f"Exception type: {type(e).__name__}")
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
    Crawl documentation from a sitemap URL.
    
    Args:
        openai_client: OpenAI client for API calls
        config: Crawl configuration
        
    Returns:
        bool: True if the crawl was successful, False otherwise
    """
    try:
        # Initialize crawl monitoring session
        session_id = start_crawl_session(config.source_id, config.source_name)
        enhanced_crawler_logger.info(
            f"Starting crawl for {config.source_name}",
            source_id=config.source_id,
            session_id=session_id,
            sitemap_url=config.sitemap_url
        )
        
        # Initialize Streamlit session state for control
        if 'pause_crawl' not in st.session_state:
            st.session_state.pause_crawl = False
        if 'stop_crawl' not in st.session_state:
            st.session_state.stop_crawl = False
        
        # Save crawl start time    
        st.session_state.crawl_start_time = time.time()
        st.session_state.crawl_config = {
            'source_id': config.source_id,
            'source_name': config.source_name,
            'sitemap_url': config.sitemap_url
        }
        
        # The rest of the function remains the same - Streamlit will update the UI on the next rerun
        # Create batch processors for embeddings and LLM tasks
        embedding_processor = EmbeddingBatchProcessor(
            openai_client,
            model=config.embedding_model,
            max_concurrent_batches=config.max_concurrent_api_calls
        )
        
        llm_processor = LLMBatchProcessor(
            openai_client,
            model=config.llm_model,
            max_concurrent_batches=config.max_concurrent_api_calls
        )
        
        # Create semaphore for limiting concurrent requests
        crawl_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Get URLs from sitemap
        urls = await get_urls_from_sitemap(config.sitemap_url, config)
        
        # Check for already processed URLs if this is a resume operation
        already_processed_urls = []
        if hasattr(config, 'already_processed_urls') and config.already_processed_urls:
            already_processed_urls = config.already_processed_urls
            enhanced_crawler_logger.info(
                f"Resuming crawl for {config.source_name}, skipping {len(already_processed_urls)} already processed URLs",
                source_id=config.source_id,
                already_processed=len(already_processed_urls)
            )
            
            # Filter out already processed URLs
            urls = [url for url in urls if url not in already_processed_urls]
        
        enhanced_crawler_logger.info(
            f"Found {len(urls)} URLs to process for {config.source_name}",
            source_id=config.source_id,
            url_count=len(urls),
            total_urls_in_sitemap=len(urls) + len(already_processed_urls)
        )
        
        if not urls:
            enhanced_crawler_logger.warning(
                f"No new URLs found to process for {config.source_name}",
                source_id=config.source_id,
                sitemap_url=config.sitemap_url
            )
            end_crawl_session(session_id)
            return True if already_processed_urls else False
        
        # Create tasks for all URLs
        tasks = []
        for url in urls:
            task = crawl_url(
                url,
                config,
                crawl_semaphore,
                embedding_processor,
                llm_processor
            )
            tasks.append(task)
        
        # Process all URLs concurrently
        results = await asyncio.gather(*tasks)
        
        # Update completion status
        success_count = sum(1 for r in results if r)
        fail_count = len(results) - success_count
        total_processed = success_count + len(already_processed_urls)
        
        enhanced_crawler_logger.info(
            f"Completed crawl for {config.source_name}",
            source_id=config.source_id,
            session_id=session_id,
            total_urls=len(urls) + len(already_processed_urls),
            successful_urls=success_count + len(already_processed_urls),
            failed_urls=fail_count,
            new_urls_processed=success_count,
            already_processed=len(already_processed_urls),
            success_rate=success_count/len(urls) if urls else 1.0
        )
        
        # Update the documentation source statistics for page count
        await update_documentation_source(config.source_id, pages_count=success_count)
        
        # End monitoring session
        end_crawl_session(session_id)
        
        # Reset crawl state in session
        st.session_state.crawl_start_time = None
        
        return total_processed > 0
    except Exception as e:
        enhanced_crawler_logger.structured_error(
            f"Error crawling documentation for {config.source_name}: {e}",
            error=e,
            source_id=config.source_id,
            sitemap_url=config.sitemap_url
        )
        
        # End monitoring session
        end_crawl_session()
        
        return False 