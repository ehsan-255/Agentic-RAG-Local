import asyncio
import re
import requests
import httpx
import time
import logging
import streamlit as st
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional, Tuple, Union, Set
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
import numpy as np
import os
import aiohttp

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
    ChunkingError,
    DatabaseError,
    ErrorCategory
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
    # Character-based chunking (legacy)
    chunk_size: int = config.DEFAULT_CHUNK_SIZE
    # Word-based chunking (new)
    chunk_words: int = config.DEFAULT_CHUNK_WORDS
    overlap_words: int = config.DEFAULT_OVERLAP_WORDS
    use_word_based_chunking: bool = config.USE_WORD_BASED_CHUNKING
    # Concurrency settings
    max_concurrent_requests: int = config.DEFAULT_MAX_CONCURRENT_CRAWLS
    max_concurrent_api_calls: int = config.DEFAULT_MAX_CONCURRENT_API_CALLS
    # Retry settings
    retry_attempts: int = config.DEFAULT_RETRY_ATTEMPTS
    min_backoff: int = config.DEFAULT_MIN_BACKOFF
    max_backoff: int = config.DEFAULT_MAX_BACKOFF
    # Model settings
    llm_model: str = config.LLM_MODEL
    embedding_model: str = config.EMBEDDING_MODEL
    # URL patterns
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

def chunk_text(text: str, chunk_size: int = config.DEFAULT_CHUNK_SIZE, 
            use_word_based: bool = config.USE_WORD_BASED_CHUNKING,
            chunk_words: int = config.DEFAULT_CHUNK_WORDS,
            overlap_words: int = config.DEFAULT_OVERLAP_WORDS) -> List[str]:
    """
    Split text into chunks based on the specified parameters.
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum chunk size in characters (used when use_word_based=False)
        use_word_based: Whether to use word-based chunking instead of character-based
        chunk_words: Target number of words per chunk (used when use_word_based=True)
        overlap_words: Number of words to overlap between chunks (used when use_word_based=True)
        
    Returns:
        List[str]: List of text chunks
    """
    # Import here to avoid circular imports
    from src.utils.text_chunking import enhanced_chunk_text, character_based_chunk_text
    
    if not text:
        return []
        
    # Determine which chunking method to use
    if use_word_based:
        enhanced_crawler_logger.debug(f"Using word-based chunking with {chunk_words} words per chunk and {overlap_words} words overlap")
        return enhanced_chunk_text(text, chunk_words, overlap_words)
    else:
        enhanced_crawler_logger.debug(f"Using character-based chunking with {chunk_size} characters per chunk")
        return character_based_chunk_text(text, chunk_size)

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
        int: Number of chunks stored in the database
    """
    try:
        # Convert the HTML content to markdown
        result = convert_html_to_markdown(html_content)
        
        if not result:
            enhanced_crawler_logger.structured_error(
                f"Failed to convert HTML to Markdown",
                category=ErrorCategory.CONTENT_PROCESSING,
                url=url
            )
            raise HTMLConversionError(url, "Failed to convert HTML to Markdown")
        
        markdown, title = result
        
        if not markdown:
            enhanced_crawler_logger.structured_error(
                f"Markdown conversion resulted in empty content",
                category=ErrorCategory.CONTENT_PROCESSING,
                url=url
            )
            raise EmptyContentError(url, "Markdown conversion resulted in empty content")
        
        # Log chunking method
        if config.use_word_based_chunking:
            enhanced_crawler_logger.info(
                f"Using word-based chunking for {url} with {config.chunk_words} words per chunk and {config.overlap_words} words overlap"
            )
        else:
            enhanced_crawler_logger.info(
                f"Using character-based chunking for {url} with {config.chunk_size} characters per chunk"
            )
            
        # Chunk the document
        chunks = chunk_text(markdown, config.chunk_size, config.use_word_based_chunking, config.chunk_words, config.overlap_words)
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
        failed_chunks = 0
        for i, (chunk, title_summary, embedding) in enumerate(zip(chunks, titles_summaries, embeddings)):
            # Create metadata
            metadata = {
                "source_id": config.source_id,
                "source": config.source_name,
                "url": url,
                "chunk_number": i,
                "page_title": title,
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Enhanced error handling for chunk storage
            try:
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
                else:
                    failed_chunks += 1
                    enhanced_crawler_logger.structured_error(
                        f"Failed to store chunk {i} for {url}: Database returned null ID",
                        error=DatabaseError(f"Null chunk ID returned for {url} chunk {i}"),
                        url=url,
                        chunk_number=i,
                        content_length=len(chunk) if chunk else 0,
                        embedding_length=len(embedding) if embedding else 0
                    )
            except Exception as chunk_error:
                failed_chunks += 1
                enhanced_crawler_logger.structured_error(
                    f"Exception storing chunk {i} for {url}: {str(chunk_error)}",
                    error=chunk_error,
                    url=url,
                    chunk_number=i,
                    content_length=len(chunk) if chunk else 0,
                    embedding_length=len(embedding) if embedding else 0
                )
        
        # Log overall results
        if stored_count > 0:
            enhanced_crawler_logger.info(
                f"Successfully stored {stored_count} chunks for {url}"
            )
        
        if failed_chunks > 0:
            enhanced_crawler_logger.warning(
                f"Failed to store {failed_chunks} chunks for {url}"
            )
        
        # Update the documentation source statistics
        try:
            await update_documentation_source(config.source_id, chunks_count=stored_count)
        except Exception as update_error:
            enhanced_crawler_logger.structured_error(
                f"Failed to update documentation source statistics: {str(update_error)}",
                error=update_error,
                source_id=config.source_id,
                chunks_count=stored_count
            )
        
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
    Crawl a URL using the given configuration.
    
    Args:
        url: URL to crawl
        config: Crawl configuration
        crawl_semaphore: Semaphore to limit concurrent crawling
        embedding_processor: Embedding processor for batched embedding generation
        llm_processor: LLM processor for batched LLM generation
        
    Returns:
        bool: True if successful, False otherwise
    """
    # First check if we need to cancel the crawl
    try:
        # Import here to avoid circular imports
        import streamlit as st
        from src.ui.streamlit_app import global_state
        
        if hasattr(st, 'session_state') and st.session_state.get('stop_crawl', False):
            enhanced_crawler_logger.info(f"Cancelling crawl for URL {url} due to stop flag")
            # Track cancelled tasks in global state
            global_state.increment_cancelled_tasks()
            return False
    except ImportError:
        # If we can't import streamlit, just continue
        enhanced_crawler_logger.debug("Could not import streamlit to check for stop flag")
        
    # Standard processing with semaphore
    async with crawl_semaphore:
        try:
            # More detailed logging to track what's happening
            enhanced_crawler_logger.debug(f"Processing URL: {url} with config: {config.source_id}")
            
            # Check if crawl should be paused or stopped
            if st.session_state.get('pause_crawl', False):
                enhanced_crawler_logger.info(f"Pausing crawl for URL {url}")
                # Wait for pause to be lifted or crawl to be stopped
                while st.session_state.get('pause_crawl', False):
                    await asyncio.sleep(1)
                    if st.session_state.get('stop_crawl', False):
                        enhanced_crawler_logger.info(f"Cancelling paused crawl for URL {url}")
                        return False
                        
            if st.session_state.get('stop_crawl', False):
                enhanced_crawler_logger.info(f"Cancelling crawl for URL {url}")
                return False
                
            # Fetch the page
            enhanced_crawler_logger.debug(f"Fetching URL: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, ssl=False, timeout=30) as response:
                    if response.status != 200:
                        enhanced_crawler_logger.warning(f"Failed to fetch {url}: status {response.status}")
                        return False
                    
                    content = await response.text()
                    
                    # Record page in session stats
                    active_session = get_active_session()
                    if active_session:
                        try:
                            active_session.record_page_processed(url, True)
                        except Exception as session_error:
                            enhanced_crawler_logger.warning(f"Error updating session stats: {session_error}")
                    
                    # Process the page and add to database
                    success, chunks_added = await process_page(
                        url=url, 
                        content=content, 
                        config=config,
                        embedding_processor=embedding_processor,
                        llm_processor=llm_processor
                    )
                    
                    # Log the result
                    if success:
                        enhanced_crawler_logger.info(f"Successfully processed {url}: {chunks_added} chunks added")
                    else:
                        enhanced_crawler_logger.warning(f"Failed to process {url} - no chunks added")
                    
                    return success
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
                f"Error crawling URL {url}",
                error=e,
                category=ErrorCategory.CONNECTION,
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

async def add_to_database(url: str, page_content: str, title: str, output_chunks: List[dict], config: CrawlConfig) -> int:
    """
    Add content chunks to the database.
    
    Args:
        url: URL of the page
        page_content: Raw page content
        title: Page title
        output_chunks: Chunked content
        config: Crawl configuration
        
    Returns:
        int: Number of chunks added
    """
    success_count = 0
    
    # Detailed logging to trace execution
    enhanced_crawler_logger.debug(f"Adding {len(output_chunks)} chunks to database for URL: {url}")
    
    for i, chunk in enumerate(output_chunks):
        # Ensure embedding is a Python list of floats, not NumPy types
        if isinstance(chunk["embedding"], np.ndarray):
            chunk["embedding"] = [float(x) for x in chunk["embedding"]]
            
        try:
            # Add the chunk to the database
            chunk_id = await add_site_page(
                url=url,
                chunk_number=i+1,
                title=title,
                summary=chunk["summary"],
                content=chunk["content"],
                metadata={
                    "source_id": config.source_id,
                    "chunk_index": i+1,
                    "total_chunks": len(output_chunks),
                    "word_count": len(chunk["content"].split()),
                    "char_count": len(chunk["content"]),
                },
                embedding=chunk["embedding"],
                raw_content=page_content if i == 0 else None  # Only store raw content with first chunk
            )
            
            if chunk_id:
                success_count += 1
                enhanced_crawler_logger.debug(f"Added chunk {i+1}/{len(output_chunks)} for URL: {url}")
            else:
                enhanced_crawler_logger.structured_error(
                    f"Failed to add chunk {i+1}/{len(output_chunks)} for URL: {url}",
                    category=ErrorCategory.DATABASE,
                    url=url,
                    chunk_number=i+1
                )
        except Exception as e:
            enhanced_crawler_logger.structured_error(
                f"Error adding chunk to database",
                error=e,
                category=ErrorCategory.DATABASE,
                url=url,
                chunk_number=i+1
            )
    
    enhanced_crawler_logger.info(f"Added {success_count}/{len(output_chunks)} chunks to database for URL: {url}")
    return success_count 

async def process_page(url: str, content: str, config: CrawlConfig, 
                   embedding_processor: EmbeddingBatchProcessor,
                   llm_processor: LLMBatchProcessor,
                   max_chunks: int = 20) -> Tuple[bool, int]:
    """
    Process a page and add it to the database.
    
    Args:
        url: URL of the page
        content: HTML content
        config: Crawl configuration
        embedding_processor: Batch processor for embeddings
        llm_processor: Batch processor for LLM tasks
        max_chunks: Maximum number of chunks to create
        
    Returns:
        Tuple[bool, int]: Success flag and number of chunks added
    """
    enhanced_crawler_logger.debug(f"Processing page: {url}")
    soup = BeautifulSoup(content, "html.parser")
    
    # Get page title
    title = soup.title.text.strip() if soup.title else os.path.basename(url)
    
    # Parse content
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    text_content = h.handle(content)
    
    # Check if content is sufficient
    if len(text_content) < 100:
        enhanced_crawler_logger.warning(f"Content too short for {url}: {len(text_content)} chars")
        return False, 0
    
    enhanced_crawler_logger.info(f"Processing content for {url}: {len(text_content)} chars")
    
    # Process content into chunks
    chunks = process_content(text_content, url=url, max_chunks=max_chunks)
    
    if not chunks:
        enhanced_crawler_logger.warning(f"No chunks generated for {url}")
        return False, 0
    
    enhanced_crawler_logger.info(f"Generated {len(chunks)} chunks for {url}")
    
    # Add chunks to database with proper error handling
    chunks_added = await add_to_database(url, content, title, chunks, config)
    
    return chunks_added > 0, chunks_added 

def process_content(content: str, url: str, max_chunks: int = 20) -> List[Dict[str, Any]]:
    """
    Process content into chunks with summaries and embeddings.
    
    Args:
        content: Text content to process
        url: URL of the content
        max_chunks: Maximum number of chunks to create
        
    Returns:
        List[Dict[str, Any]]: List of processed chunks
    """
    # Split content into chunks
    chunks = split_into_chunks(content, max_chunks=max_chunks)
    
    if not chunks:
        enhanced_crawler_logger.warning(f"No chunks generated for {url}")
        return []
        
    # Generate embeddings
    embeddings = generate_embeddings([chunk["content"] for chunk in chunks])
    
    if len(embeddings) != len(chunks):
        enhanced_crawler_logger.warning(f"Embedding count mismatch for {url}: {len(embeddings)} embeddings for {len(chunks)} chunks")
        return []
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        # Ensure embedding is a Python list of floats
        if isinstance(embeddings[i], np.ndarray):
            chunk["embedding"] = [float(x) for x in embeddings[i]]
        else:
            chunk["embedding"] = embeddings[i]
    
    # Generate summaries
    for chunk in chunks:
        chunk["summary"] = generate_summary(chunk["content"])
    
    return chunks 

def split_into_chunks(content: str, max_chunks: int = 20, max_tokens_per_chunk: int = 1000) -> List[Dict[str, Any]]:
    """
    Split content into chunks.
    
    Args:
        content: Text content to split
        max_chunks: Maximum number of chunks to create
        max_tokens_per_chunk: Maximum tokens per chunk
        
    Returns:
        List[Dict[str, Any]]: List of content chunks
    """
    # Simple splitting by paragraphs for now
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # If we have very few paragraphs, just return them directly
    if len(paragraphs) <= max_chunks:
        return [{"content": p} for p in paragraphs if len(p) > 50]
    
    # Otherwise, combine paragraphs to form chunks
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        # Estimate tokens (very rough approximation)
        current_tokens = len(current_chunk.split())
        p_tokens = len(p.split())
        
        if current_tokens + p_tokens > max_tokens_per_chunk and current_chunk:
            chunks.append({"content": current_chunk.strip()})
            current_chunk = p
        else:
            current_chunk += "\n\n" + p if current_chunk else p
    
    # Add the last chunk if not empty
    if current_chunk.strip():
        chunks.append({"content": current_chunk.strip()})
    
    # Limit to max_chunks
    return chunks[:max_chunks]

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List[List[float]]: List of embeddings
    """
    # In a real implementation, we'd call the OpenAI API here
    # For now, just return random embeddings of the right dimension
    return [list(np.random.rand(1536).astype(float)) for _ in texts]

def generate_summary(text: str) -> str:
    """
    Generate a summary for a text chunk.
    
    Args:
        text: Text to summarize
        
    Returns:
        str: Summary text
    """
    # In a real implementation, we'd call an LLM here
    # For now, just return the first 100 characters
    if len(text) <= 100:
        return text
    return text[:100] + "..." 