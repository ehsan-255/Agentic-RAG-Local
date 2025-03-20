from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import sys

# Import with better error handling
try:
    from src.db.connection import execute_query, execute_transaction
except ImportError as e:
    print(f"Error importing connection module: {e}")
    print("This may be due to missing psycopg or psycopg-pool packages.")
    print("If so, please run: pip install \"psycopg[binary]\"==3.1.13 psycopg-pool==3.1.8")
    
    # Try to provide fallback implementation
    async def execute_query(*args, **kwargs):
        print("WARNING: Using dummy execute_query due to import error")
        return None
        
    async def execute_transaction(*args, **kwargs):
        print("WARNING: Using dummy execute_transaction due to import error")
        return False

async def setup_database() -> bool:
    """
    Set up the database schema if it doesn't exist.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Check if pgvector extension is installed
        pgvector_check = await execute_query(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        
        if not pgvector_check or not pgvector_check[0].get('exists', False):
            # Create pgvector extension if not installed
            await execute_query("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create documentation_sources table
        await execute_query("""
            CREATE TABLE IF NOT EXISTS documentation_sources (
                id SERIAL PRIMARY KEY,
                source_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                sitemap_url TEXT NOT NULL,
                configuration JSONB NOT NULL DEFAULT '{}'::jsonb,
                pages_count INTEGER DEFAULT 0,
                chunks_count INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_crawled_at TIMESTAMP WITH TIME ZONE
            )
        """)
        
        # Create site_pages table with vector support
        await execute_query("""
            CREATE TABLE IF NOT EXISTS site_pages (
                id SERIAL PRIMARY KEY,
                url TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                embedding vector(1536),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(url, chunk_number)
            )
        """)
        
        # Create index on source_id for filtering
        await execute_query("""
            CREATE INDEX IF NOT EXISTS idx_site_pages_source_id ON site_pages ((metadata->>'source_id'))
        """)
        
        # Create vector index for similarity search
        await execute_query("""
            CREATE INDEX IF NOT EXISTS idx_site_pages_embedding ON site_pages 
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
        """)
        
        return True
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False

async def add_documentation_source(
    name: str, 
    source_id: str, 
    sitemap_url: str, 
    configuration: Dict[str, Any] = None
) -> bool:
    """
    Add a new documentation source.
    
    Args:
        name: Name of the documentation source
        source_id: Unique identifier for the source
        sitemap_url: URL of the sitemap
        configuration: Additional configuration for the source
        
    Returns:
        bool: True if insertion was successful, False otherwise
    """
    try:
        result = await execute_query(
            """
            INSERT INTO documentation_sources (name, source_id, sitemap_url, configuration)
            VALUES (%(name)s, %(source_id)s, %(sitemap_url)s, %(configuration)s)
            RETURNING id
            """,
            {
                "name": name,
                "source_id": source_id,
                "sitemap_url": sitemap_url,
                "configuration": json.dumps(configuration or {})
            }
        )
        
        return result is not None and len(result) > 0
    except Exception as e:
        print(f"Error adding documentation source: {e}")
        return False

async def update_documentation_source(
    source_id: str, 
    pages_count: Optional[int] = None, 
    chunks_count: Optional[int] = None
) -> bool:
    """
    Update statistics for a documentation source.
    
    Args:
        source_id: ID of the documentation source
        pages_count: Number of pages to add (optional)
        chunks_count: Number of chunks to add (optional)
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        query = "UPDATE documentation_sources SET last_crawled_at = NOW()"
        params = {"source_id": source_id}
        
        if pages_count is not None:
            query += ", pages_count = pages_count + %(pages_count)s"
            params["pages_count"] = pages_count
        
        if chunks_count is not None:
            query += ", chunks_count = chunks_count + %(chunks_count)s"
            params["chunks_count"] = chunks_count
        
        query += " WHERE source_id = %(source_id)s"
        
        await execute_query(query, params)
        return True
    except Exception as e:
        print(f"Error updating documentation source: {e}")
        return False

async def add_site_page(
    url: str,
    chunk_number: int,
    title: str,
    summary: str,
    content: str,
    metadata: Dict[str, Any],
    embedding: List[float]
) -> Optional[int]:
    """
    Add a new site page chunk to the database.
    
    Args:
        url: URL of the page
        chunk_number: Chunk number for ordering
        title: Title of the chunk
        summary: Summary of the chunk
        content: Text content of the chunk
        metadata: Additional metadata for the chunk
        embedding: Vector embedding of the content
        
    Returns:
        Optional[int]: ID of the inserted record or None on error
    """
    try:
        source_id = metadata.get("source_id")
        if not source_id:
            print("Error: source_id is required in metadata")
            return None
        
        # First try to insert the record
        result = await execute_query(
            """
            INSERT INTO site_pages (
                url, chunk_number, title, summary, content, metadata, embedding
            )
            VALUES (
                %(url)s, %(chunk_number)s, %(title)s, %(summary)s, 
                %(content)s, %(metadata)s, %(embedding)s
            )
            ON CONFLICT (url, chunk_number) 
            DO UPDATE SET
                title = %(title)s,
                summary = %(summary)s,
                content = %(content)s,
                metadata = %(metadata)s,
                embedding = %(embedding)s,
                created_at = NOW()
            RETURNING id
            """,
            {
                "url": url,
                "chunk_number": chunk_number,
                "title": title,
                "summary": summary,
                "content": content,
                "metadata": json.dumps(metadata),
                "embedding": embedding
            }
        )
        
        if result and len(result) > 0:
            return result[0]["id"]
        
        return None
    except Exception as e:
        print(f"Error adding site page: {e}")
        return None

async def get_documentation_sources() -> List[Dict[str, Any]]:
    """
    Get all documentation sources from the database.
    
    Returns:
        List[Dict[str, Any]]: List of documentation sources
    """
    try:
        result = await execute_query(
            """
            SELECT 
                source_id, name, sitemap_url, configuration, 
                pages_count, chunks_count, created_at, last_crawled_at
            FROM documentation_sources
            ORDER BY name
            """
        )
        
        return result or []
    except Exception as e:
        print(f"Error getting documentation sources: {e}")
        return []

async def get_source_statistics(source_id: str) -> Optional[Dict[str, Any]]:
    """
    Get statistics for a documentation source.
    
    Args:
        source_id: ID of the documentation source
        
    Returns:
        Optional[Dict[str, Any]]: Source statistics or None on error
    """
    try:
        result = await execute_query(
            """
            SELECT 
                name, sitemap_url, pages_count, chunks_count, created_at, last_crawled_at
            FROM documentation_sources
            WHERE source_id = %(source_id)s
            """,
            {"source_id": source_id}
        )
        
        if result and len(result) > 0:
            return result[0]
        
        return None
    except Exception as e:
        print(f"Error getting source statistics: {e}")
        return None

async def delete_documentation_source(source_id: str) -> bool:
    """
    Delete a documentation source and all its pages.
    
    Args:
        source_id: ID of the documentation source
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        # Delete using CASCADE constraint
        await execute_query(
            "DELETE FROM documentation_sources WHERE source_id = %(source_id)s",
            {"source_id": source_id}
        )
        
        return True
    except Exception as e:
        print(f"Error deleting documentation source: {e}")
        return False

async def search_similar_chunks(
    query_embedding: List[float],
    match_count: int = 5,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for chunks similar to the query embedding.
    
    Args:
        query_embedding: Query embedding vector
        match_count: Number of matches to return
        source_id: Optional source ID to filter by
        
    Returns:
        List[Dict[str, Any]]: List of matching chunks with similarity scores
    """
    try:
        # Build query based on whether source_id is provided
        query = """
            SELECT 
                id, source_id, url, chunk_number, title, summary, content, metadata,
                1 - (embedding <=> %(embedding)s) as similarity
            FROM site_pages
        """
        
        params = {"embedding": query_embedding}
        
        if source_id:
            query += " WHERE source_id = %(source_id)s"
            params["source_id"] = source_id
        
        query += """
            ORDER BY similarity DESC
            LIMIT %(limit)s
        """
        params["limit"] = match_count
        
        result = await execute_query(query, params)
        
        # Process results
        if result:
            for row in result:
                # Parse metadata JSON if it exists
                if isinstance(row.get("metadata"), str):
                    try:
                        row["metadata"] = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        row["metadata"] = {}
                
                # Format similarity as percentage
                row["similarity"] = round(float(row["similarity"]) * 100, 2)
            
            return result
        
        return []
    except Exception as e:
        print(f"Error searching similar chunks: {e}")
        return [] 