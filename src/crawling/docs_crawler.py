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

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

# Initialize OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None

# Semaphore for rate limiting API calls
api_semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT_API_CALLS", DEFAULT_MAX_CONCURRENT_API_CALLS)))

@dataclass
class CrawlConfig:
    """Configuration for a documentation crawl."""
    site_name: str
    base_url: str
    allowed_domains: List[str]
    start_urls: List[str]
    sitemap_urls: Optional[List[str]] = None
    url_patterns_include: Optional[List[str]] = None
    url_patterns_exclude: Optional[List[str]] = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    max_concurrent_crawls: int = DEFAULT_MAX_CONCURRENT_CRAWLS
    max_concurrent_api_calls: int = DEFAULT_MAX_CONCURRENT_API_CALLS
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS

@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
    wait=wait_random_exponential(min=1, max=60)
)
async def generate_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text using OpenAI's API with rate limiting."""
    async with api_semaphore:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

async def process_page(url: str, html_content: str, source_id: int, config: CrawlConfig) -> List[Dict[str, Any]]:
    """Process a page by chunking it and generating embeddings."""
    # Extract title and main content
    # This is a simplified version - in a real implementation, you'd use BeautifulSoup or similar
    title = html_content.split("<title>")[1].split("</title>")[0] if "<title>" in html_content else "Untitled"
    
    # Remove HTML tags (simplified)
    content = html_content
    for tag in ["<script>", "</script>", "<style>", "</style>", "<head>", "</head>"]:
        content = content.replace(tag, "")
    
    # Simple HTML tag removal (in a real implementation, use BeautifulSoup)
    while "<" in content and ">" in content:
        start = content.find("<")
        end = content.find(">", start)
        if end > start:
            content = content[:start] + " " + content[end+1:]
        else:
            break
    
    # Chunk the content
    chunks = []
    content_length = len(content)
    chunk_size = config.chunk_size
    
    for i in range(0, content_length, chunk_size):
        chunk_content = content[i:i+chunk_size]
        chunk_index = i // chunk_size
        total_chunks = (content_length + chunk_size - 1) // chunk_size
        
        # Generate embedding for the chunk
        embedding = await generate_embedding(chunk_content)
        
        chunks.append({
            "source_id": source_id,
            "url": url,
            "title": title,
            "content": chunk_content,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "embedding": embedding,
            "metadata": {
                "crawled_at": datetime.now(timezone.utc).isoformat()
            }
        })
    
    return chunks

async def crawl_and_process(config: CrawlConfig) -> int:
    """Crawl a documentation site and process its pages."""
    # Create or get the documentation source
    source_response = supabase.table("documentation_sources").select("*").eq("name", config.site_name).execute()
    
    if source_response.data:
        source_id = source_response.data[0]["id"]
        # Update the source status
        supabase.table("documentation_sources").update({
            "status": "crawling",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", source_id).execute()
    else:
        # Create a new source
        source_response = supabase.table("documentation_sources").insert({
            "name": config.site_name,
            "base_url": config.base_url,
            "sitemap_url": config.sitemap_urls[0] if config.sitemap_urls else None,
            "status": "crawling",
            "config": {
                "allowed_domains": config.allowed_domains,
                "start_urls": config.start_urls,
                "url_patterns_include": config.url_patterns_include,
                "url_patterns_exclude": config.url_patterns_exclude,
                "chunk_size": config.chunk_size
            }
        }).execute()
        source_id = source_response.data[0]["id"]
    
    # Configure the crawler
    crawler_config = CrawlerRunConfig(
        allowed_domains=config.allowed_domains,
        start_urls=config.start_urls,
        sitemap_urls=config.sitemap_urls or [],
        max_concurrent_requests=config.max_concurrent_crawls,
        cache_mode=CacheMode.MEMORY
    )
    
    browser_config = BrowserConfig(
        headless=True,
        ignore_https_errors=True
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(
        run_config=crawler_config,
        browser_config=browser_config
    )
    
    # Crawl the site
    pages_processed = 0
    
    async for page in crawler.crawl():
        url = page.url
        html_content = page.html
        
        # Skip if URL doesn't match include patterns or matches exclude patterns
        if config.url_patterns_include:
            if not any(pattern in url for pattern in config.url_patterns_include):
                continue
        
        if config.url_patterns_exclude:
            if any(pattern in url for pattern in config.url_patterns_exclude):
                continue
        
        # Process the page
        chunks = await process_page(url, html_content, source_id, config)
        
        # Store chunks in the database
        for chunk in chunks:
            supabase.table("site_pages").upsert(
                chunk,
                on_conflict=["source_id", "url", "chunk_index"]
            ).execute()
        
        pages_processed += 1
    
    # Update the source status
    supabase.table("documentation_sources").update({
        "status": "completed",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }).eq("id", source_id).execute()
    
    return pages_processed

def crawl_documentation(config: CrawlConfig) -> int:
    """Crawl a documentation site and process its pages."""
    if not supabase or not openai_client:
        raise ValueError("Supabase and OpenAI clients must be initialized")
    
    # Set up the semaphore for API rate limiting
    global api_semaphore
    api_semaphore = asyncio.Semaphore(config.max_concurrent_api_calls)
    
    # Run the crawler
    return asyncio.run(crawl_and_process(config))

def clear_documentation_source(source_name: str) -> bool:
    """Clear all pages for a documentation source."""
    if not supabase:
        raise ValueError("Supabase client must be initialized")
    
    # Get the source ID
    source_response = supabase.table("documentation_sources").select("id").eq("name", source_name).execute()
    
    if not source_response.data:
        return False
    
    source_id = source_response.data[0]["id"]
    
    # Delete all pages for this source
    supabase.table("site_pages").delete().eq("source_id", source_id).execute()
    
    # Update the source status
    supabase.table("documentation_sources").update({
        "status": "cleared",
        "page_count": 0,
        "chunk_count": 0,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }).eq("id", source_id).execute()
    
    return True