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
from supabase import create_client, Client

load_dotenv()

# Default configuration values
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_MAX_CONCURRENT_CRAWLS = 3
DEFAULT_MAX_CONCURRENT_API_CALLS = 5
DEFAULT_RETRY_ATTEMPTS = 6
DEFAULT_MIN_BACKOFF = 1
DEFAULT_MAX_BACKOFF = 60

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class CrawlConfig:
    """Configuration for the crawler."""
    source_name: str
    source_id: str  # Unique identifier for this documentation source
    chunk_size: int = DEFAULT_CHUNK_SIZE
    max_concurrent_crawls: int = DEFAULT_MAX_CONCURRENT_CRAWLS
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
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=DEFAULT_MIN_BACKOFF, max=DEFAULT_MAX_BACKOFF),
    stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS)
)
async def get_title_and_summary(chunk: str, url: str, model: str = "gpt-4o-mini") -> Dict[str, str]:
    """Extract title and summary using GPT-4 with retry logic."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=DEFAULT_MIN_BACKOFF, max=DEFAULT_MAX_BACKOFF),
    stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS)
)
async def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding vector from OpenAI with retry logic."""
    try:
        response = await openai_client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str, config: CrawlConfig, api_semaphore: asyncio.Semaphore) -> ProcessedChunk:
    """Process a single chunk of text with rate limiting."""
    async with api_semaphore:
        # Get title and summary
        extracted = await get_title_and_summary(chunk, url, config.llm_model)
        
    async with api_semaphore:
        # Get embedding
        embedding = await get_embedding(chunk, config.embedding_model)
    
    # Create metadata
    metadata = {
        "source": config.source_name,
        "source_id": config.source_id,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str, config: CrawlConfig, api_semaphore: asyncio.Semaphore):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown, config.chunk_size)
    
    # Process chunks with rate limiting
    tasks = [
        process_chunk(chunk, i, url, config, api_semaphore) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)
    
    # Update documentation source with crawl statistics
    try:
        supabase.table("documentation_sources").update({
            "pages_count": supabase.rpc(
                "increment_pages_count", {"source_id_param": config.source_id}
            ),
            "chunks_count": supabase.rpc(
                "increment_chunks_count", 
                {"source_id_param": config.source_id, "increment_by": len(chunks)}
            ),
            "last_crawled_at": datetime.now(timezone.utc).isoformat()
        }).eq("source_id", config.source_id).execute()
    except Exception as e:
        print(f"Error updating documentation source statistics: {e}")

async def crawl_parallel(urls: List[str], config: CrawlConfig):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    # Create semaphores to limit concurrency
    crawl_semaphore = asyncio.Semaphore(config.max_concurrent_crawls)
    api_semaphore = asyncio.Semaphore(config.max_concurrent_api_calls)
    
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        async def process_url(url: str):
            async with crawl_semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown, config, api_semaphore)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def filter_urls(urls: List[str], config: CrawlConfig) -> List[str]:
    """Filter URLs based on include/exclude patterns."""
    filtered_urls = urls
    
    # Apply include patterns if specified
    if config.url_patterns_include:
        filtered_urls = [
            url for url in filtered_urls 
            if any(pattern in url for pattern in config.url_patterns_include)
        ]
    
    # Apply exclude patterns if specified
    if config.url_patterns_exclude:
        filtered_urls = [
            url for url in filtered_urls 
            if not any(pattern in url for pattern in config.url_patterns_exclude)
        ]
    
    return filtered_urls

def get_urls_from_sitemap(sitemap_url: str, config: CrawlConfig) -> List[str]:
    """Get URLs from a sitemap and apply filters."""
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all sitemap URLs from the sitemap index
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        sitemap_urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        print(f"Found {len(sitemap_urls)} sitemaps")
        
        # Extract actual documentation URLs from each sitemap
        doc_urls = []
        for sitemap in sitemap_urls:
            try:
                print(f"Processing sitemap: {sitemap}")
                sub_response = requests.get(sitemap)
                sub_response.raise_for_status()
                
                sub_root = ElementTree.fromstring(sub_response.content)
                urls = [loc.text for loc in sub_root.findall('.//ns:loc', namespace)]
                doc_urls.extend(urls)
            except Exception as e:
                print(f"Error processing sitemap {sitemap}: {e}")
        
        print(f"Total URLs found: {len(doc_urls)}")
        
        # Apply URL filtering
        filtered_urls = filter_urls(doc_urls, config)
        print(f"After filtering: {len(filtered_urls)} URLs")
        
        return filtered_urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def create_documentation_source(name: str, base_url: str, config: Dict[str, Any]) -> str:
    """Create a new documentation source entry and return its ID."""
    try:
        source_id = f"{name.lower().replace(' ', '_')}_{int(time.time())}"
        result = supabase.table("documentation_sources").insert({
            "name": name,
            "source_id": source_id,
            "base_url": base_url,
            "configuration": config,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_crawled_at": None,
            "pages_count": 0,
            "chunks_count": 0
        }).execute()
        print(f"Created documentation source: {name} with ID {source_id}")
        return source_id
    except Exception as e:
        print(f"Error creating documentation source: {e}")
        return None

async def clear_documentation_source(source_id: str) -> bool:
    """Clear all data for a specific documentation source."""
    try:
        # Delete all chunks for this source
        result = supabase.table("site_pages").delete().eq("metadata->>source_id", source_id).execute()
        print(f"Cleared data for source {source_id}: {result}")
        
        # Reset the count statistics
        supabase.table("documentation_sources").update({
            "pages_count": 0,
            "chunks_count": 0
        }).eq("source_id", source_id).execute()
        
        return True
    except Exception as e:
        print(f"Error clearing documentation source: {e}")
        return False

async def crawl_documentation(sitemap_url: str, config: CrawlConfig) -> bool:
    """Crawl documentation from a sitemap URL."""
    # Create or update documentation source
    if not config.source_id:
        config.source_id = await create_documentation_source(
            config.source_name, 
            sitemap_url,
            {
                "chunk_size": config.chunk_size,
                "max_concurrent_crawls": config.max_concurrent_crawls,
                "max_concurrent_api_calls": config.max_concurrent_api_calls,
                "url_patterns_include": config.url_patterns_include,
                "url_patterns_exclude": config.url_patterns_exclude,
                "llm_model": config.llm_model,
                "embedding_model": config.embedding_model
            }
        )
    
    if not config.source_id:
        print("Failed to create documentation source")
        return False
    
    # Get URLs from sitemap
    urls = get_urls_from_sitemap(sitemap_url, config)
    if not urls:
        print("No URLs found to crawl")
        return False
    
    print(f"Starting crawl of {len(urls)} URLs for {config.source_name}")
    
    # Crawl URLs
    await crawl_parallel(urls, config)
    return True

async def main():
    # Example usage
    source_name = "QuantConnect_docs"
    sitemap_url = "https://www.quantconnect.com/sitemap.xml"
    
    # Create configuration
    config = CrawlConfig(
        source_name=source_name,
        source_id="",  # Will be created during crawl
        url_patterns_include=['/docs/', '/tutorial/', '/api/', '/key-concepts/'],
        max_concurrent_crawls=3,
        max_concurrent_api_calls=5
    )
    
    # Clear existing data for this source (optional)
    # await clear_documentation_source(config.source_id)
    
    # Crawl documentation
    success = await crawl_documentation(sitemap_url, config)
    if success:
        print(f"Successfully crawled {source_name} documentation")
    else:
        print(f"Failed to crawl {source_name} documentation")

if __name__ == "__main__":
    asyncio.run(main()) 