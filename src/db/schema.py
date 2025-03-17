<<<<<<< HEAD
"""
Database schema and utility functions for the Agentic RAG system.

This module provides functions for interacting with the PostgreSQL database,
including creating the schema, managing connections, and executing searches.
"""

import os
import psycopg2
from psycopg2.extras import Json, DictCursor
from psycopg2 import pool
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Connection pool for database operations
connection_pool = None

def get_connection_string() -> str:
    """
    Get the PostgreSQL connection string from environment variables.
    
    Returns:
        str: PostgreSQL connection string
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "postgres")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    
    return f"host={host} port={port} dbname={database} user={user} password={password}"

def initialize_connection_pool(min_connections: int = 1, max_connections: int = 10) -> None:
    """
    Initialize the database connection pool.
    
    Args:
        min_connections: Minimum number of connections in the pool
        max_connections: Maximum number of connections in the pool
    """
    global connection_pool
    
    try:
        if connection_pool is None:
            connection_pool = pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                get_connection_string()
            )
            logger.info("Database connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing connection pool: {e}")
        raise

def get_connection():
    """
    Get a connection from the pool.
    
    Returns:
        A database connection from the pool
    """
    global connection_pool
    
    if connection_pool is None:
        initialize_connection_pool()
        
    return connection_pool.getconn()

def release_connection(conn):
    """
    Release a connection back to the pool.
    
    Args:
        conn: The connection to release
    """
    global connection_pool
    
    if connection_pool is not None:
        connection_pool.putconn(conn)

def create_schema_from_file(schema_file_path: str) -> bool:
    """
    Create the database schema from a SQL file.
    
    Args:
        schema_file_path: Path to the SQL file containing the schema
        
    Returns:
        bool: True if schema creation was successful, False otherwise
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Read the SQL file
        with open(schema_file_path, 'r') as f:
            sql_script = f.read()
        
        # Execute the SQL script
        cur.execute(sql_script)
        conn.commit()
        logger.info(f"Schema created successfully from {schema_file_path}")
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error creating schema: {e}")
        return False
    finally:
        if conn:
            release_connection(conn)

def check_pgvector_extension() -> bool:
    """
    Check if the pgvector extension is installed and available.
    
    Returns:
        bool: True if pgvector is available, False otherwise
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Check if pgvector extension exists
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        result = cur.fetchone()
        
        if result:
            logger.info("pgvector extension is installed")
            return True
        else:
            logger.warning("pgvector extension is not installed")
            return False
    except Exception as e:
        logger.error(f"Error checking pgvector extension: {e}")
        return False
    finally:
        if conn:
            release_connection(conn)

def check_tables_exist() -> Dict[str, bool]:
    """
    Check if the required tables exist in the database.
    
    Returns:
        Dict[str, bool]: Dictionary with table names as keys and existence status as values
    """
    tables = {
        "documentation_sources": False,
        "site_pages": False
    }
    
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Check each table
        for table in tables.keys():
            cur.execute(f"SELECT to_regclass('public.{table}');")
            result = cur.fetchone()[0]
            tables[table] = result is not None
            
        logger.info(f"Table existence check: {tables}")
        return tables
    except Exception as e:
        logger.error(f"Error checking table existence: {e}")
        return tables
    finally:
        if conn:
            release_connection(conn)

def add_documentation_source(
    name: str,
    source_id: str,
    base_url: str,
    configuration: Dict[str, Any] = {}
) -> Optional[int]:
    """
    Add a new documentation source to the database.
    
    Args:
        name: Name of the documentation source
        source_id: Unique identifier for the source
        base_url: Base URL of the documentation
        configuration: Configuration options for the source
        
    Returns:
        Optional[int]: ID of the newly created source, or None if failed
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Insert the new source
        query = """
        INSERT INTO documentation_sources (name, source_id, base_url, configuration)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        cur.execute(query, (name, source_id, base_url, Json(configuration)))
        source_id = cur.fetchone()[0]
        conn.commit()
        
        logger.info(f"Added documentation source: {name} (ID: {source_id})")
        return source_id
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error adding documentation source: {e}")
        return None
    finally:
        if conn:
            release_connection(conn)

def update_documentation_source(
    source_id: str,
    pages_count: Optional[int] = None,
    chunks_count: Optional[int] = None,
    status: Optional[str] = None
) -> bool:
    """
    Update a documentation source in the database.
    
    Args:
        source_id: Unique identifier for the source
        pages_count: New pages count (optional)
        chunks_count: New chunks count (optional)
        status: New status (optional)
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Prepare update parts
        update_parts = []
        params = []
        
        if pages_count is not None:
            update_parts.append("pages_count = %s")
            params.append(pages_count)
            
        if chunks_count is not None:
            update_parts.append("chunks_count = %s")
            params.append(chunks_count)
            
        if status is not None:
            update_parts.append("status = %s")
            params.append(status)
            
        if not update_parts:
            logger.warning("No update fields provided")
            return False
            
        # Add last_crawled_at
        update_parts.append("last_crawled_at = NOW()")
        
        # Create query
        query = f"""
        UPDATE documentation_sources 
        SET {', '.join(update_parts)}
        WHERE source_id = %s;
        """
        params.append(source_id)
        
        # Execute update
        cur.execute(query, params)
        conn.commit()
        
        logger.info(f"Updated documentation source: {source_id}")
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error updating documentation source: {e}")
        return False
    finally:
        if conn:
            release_connection(conn)

def add_site_page(
    url: str,
    chunk_number: int,
    title: str,
    summary: str,
    content: str,
    metadata: Dict[str, Any],
    embedding: List[float],
    raw_content: Optional[str] = None,
    text_embedding: Optional[List[float]] = None
) -> Optional[int]:
    """
    Add a new site page to the database.
    
    Args:
        url: URL of the page
        chunk_number: Chunk number (for pagination)
        title: Title of the page
        summary: Summary of the page content
        content: Processed content of the page
        metadata: Metadata for the page
        embedding: Vector embedding (OpenAI format)
        raw_content: Original unprocessed content (optional)
        text_embedding: Alternative text embedding (optional)
        
    Returns:
        Optional[int]: ID of the newly created page, or None if failed
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Check if a page with this URL and chunk number already exists
        check_query = """
        SELECT id FROM site_pages
        WHERE url = %s AND chunk_number = %s;
        """
        cur.execute(check_query, (url, chunk_number))
        existing = cur.fetchone()
        
        if existing:
            # Update existing page
            query = """
            UPDATE site_pages
            SET title = %s, summary = %s, content = %s, 
                metadata = %s, embedding = %s, raw_content = %s,
                text_embedding = %s, updated_at = NOW()
            WHERE url = %s AND chunk_number = %s
            RETURNING id;
            """
            cur.execute(
                query, 
                (title, summary, content, Json(metadata), embedding, 
                 raw_content, text_embedding, url, chunk_number)
            )
            page_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Updated site page: {url} (chunk {chunk_number}, ID: {page_id})")
        else:
            # Insert new page
            query = """
            INSERT INTO site_pages 
            (url, chunk_number, title, summary, content, metadata, embedding, raw_content, text_embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """
            cur.execute(
                query, 
                (url, chunk_number, title, summary, content, Json(metadata), 
                 embedding, raw_content, text_embedding)
            )
            page_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Added site page: {url} (chunk {chunk_number}, ID: {page_id})")
            
            # Increment the chunks count for the source
            if 'source_id' in metadata:
                increment_query = """
                SELECT increment_chunks_count(%s, 1);
                """
                cur.execute(increment_query, (metadata['source_id'],))
                conn.commit()
                
        return page_id
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error adding site page: {e}")
        return None
    finally:
        if conn:
            release_connection(conn)

def match_site_pages(
    query_embedding: List[float],
    match_count: int = 10,
    filter_metadata: Dict[str, Any] = {},
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Search for site pages similar to the query embedding.
    
    Args:
        query_embedding: Vector embedding of the query
        match_count: Maximum number of results to return
        filter_metadata: Metadata filter criteria
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        List[Dict[str, Any]]: List of matching pages with similarity scores
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Execute the match_site_pages function
        query = """
        SELECT * FROM match_site_pages(%s, %s, %s, %s);
        """
        cur.execute(query, (query_embedding, match_count, Json(filter_metadata), similarity_threshold))
        results = [dict(row) for row in cur.fetchall()]
        
        logger.info(f"Found {len(results)} matching pages")
        return results
    except Exception as e:
        logger.error(f"Error matching site pages: {e}")
        return []
    finally:
        if conn:
            release_connection(conn)

def hybrid_search(
    query_text: str,
    query_embedding: List[float],
    match_count: int = 10,
    filter_metadata: Dict[str, Any] = {},
    similarity_threshold: float = 0.7,
    vector_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Perform a hybrid search using both text and vector similarity.
    
    Args:
        query_text: Text query
        query_embedding: Vector embedding of the query
        match_count: Maximum number of results to return
        filter_metadata: Metadata filter criteria
        similarity_threshold: Minimum similarity threshold
        vector_weight: Weight for vector similarity vs text similarity
        
    Returns:
        List[Dict[str, Any]]: List of matching pages with combined scores
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Execute the hybrid_search function
        query = """
        SELECT * FROM hybrid_search(%s, %s, %s, %s, %s, %s);
        """
        cur.execute(
            query, 
            (query_text, query_embedding, match_count, Json(filter_metadata), 
             similarity_threshold, vector_weight)
        )
        results = [dict(row) for row in cur.fetchall()]
        
        logger.info(f"Found {len(results)} matching pages in hybrid search")
        return results
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        return []
    finally:
        if conn:
            release_connection(conn)

def filter_by_metadata(
    query_embedding: List[float],
    match_count: int = 10,
    source_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter site pages by metadata and find matches to query embedding.
    
    Args:
        query_embedding: Vector embedding of the query
        match_count: Maximum number of results to return
        source_id: Filter by source ID
        doc_type: Filter by document type
        min_date: Minimum date (ISO format)
        max_date: Maximum date (ISO format)
        
    Returns:
        List[Dict[str, Any]]: List of matching pages with similarity scores
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Execute the filter_by_metadata function
        query = """
        SELECT * FROM filter_by_metadata(%s, %s, %s, %s, %s, %s);
        """
        cur.execute(
            query, 
            (query_embedding, match_count, source_id, doc_type, min_date, max_date)
        )
        results = [dict(row) for row in cur.fetchall()]
        
        logger.info(f"Found {len(results)} matching pages with metadata filtering")
        return results
    except Exception as e:
        logger.error(f"Error in metadata filtering: {e}")
        return []
    finally:
        if conn:
            release_connection(conn)

def get_document_context(
    page_url: str,
    context_size: int = 3
) -> List[Dict[str, Any]]:
    """
    Get the context around a specific page.
    
    Args:
        page_url: URL of the page
        context_size: Number of chunks before and after to include
        
    Returns:
        List[Dict[str, Any]]: List of page chunks forming the context
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Execute the get_document_context function
        query = """
        SELECT * FROM get_document_context(%s, %s);
        """
        cur.execute(query, (page_url, context_size))
        results = [dict(row) for row in cur.fetchall()]
        
        logger.info(f"Retrieved context with {len(results)} chunks for {page_url}")
        return results
    except Exception as e:
        logger.error(f"Error getting document context: {e}")
        return []
    finally:
        if conn:
            release_connection(conn)

def get_documentation_sources() -> List[Dict[str, Any]]:
    """
    Get all documentation sources.
    
    Returns:
        List[Dict[str, Any]]: List of documentation sources
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        query = """
        SELECT * FROM documentation_sources
        ORDER BY name;
        """
        cur.execute(query)
        results = [dict(row) for row in cur.fetchall()]
        
        logger.info(f"Retrieved {len(results)} documentation sources")
        return results
    except Exception as e:
        logger.error(f"Error getting documentation sources: {e}")
        return []
    finally:
        if conn:
            release_connection(conn)

def setup_database():
    """
    Set up the database with the required schema and extensions.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Check if pgvector extension is installed
        if not check_pgvector_extension():
            logger.error("pgvector extension is not installed. Please install it.")
            return False
            
        # Check if tables exist
        tables = check_tables_exist()
        if not all(tables.values()):
            # Create schema from file
            schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      "data", "vector_schema_v2.sql")
            if not create_schema_from_file(schema_path):
                logger.error("Failed to create schema from file.")
                return False
                
            logger.info("Database schema created successfully.")
        else:
            logger.info("Database schema already exists.")
            
        return True
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False

# Initialize database on import
if __name__ != "__main__":
    initialize_connection_pool() 
=======
from typing import List, Dict, Any, Optional
from datetime import datetime
from supabase import Client

class DocumentationSource:
    """Represents a documentation source in the database."""
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all documentation sources."""
        response = self.supabase.table("documentation_sources").select("*").order("name").execute()
        return response.data
    
    def get_by_id(self, source_id: int) -> Optional[Dict[str, Any]]:
        """Get a documentation source by ID."""
        response = self.supabase.table("documentation_sources").select("*").eq("id", source_id).execute()
        if response.data:
            return response.data[0]
        return None
    
    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a documentation source by name."""
        response = self.supabase.table("documentation_sources").select("*").eq("name", name).execute()
        if response.data:
            return response.data[0]
        return None
    
    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new documentation source."""
        response = self.supabase.table("documentation_sources").insert(data).execute()
        return response.data[0]
    
    def update(self, source_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a documentation source."""
        response = self.supabase.table("documentation_sources").update(data).eq("id", source_id).execute()
        return response.data[0]
    
    def delete(self, source_id: int) -> bool:
        """Delete a documentation source."""
        response = self.supabase.table("documentation_sources").delete().eq("id", source_id).execute()
        return len(response.data) > 0


class SitePage:
    """Represents a site page in the database."""
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
    
    def get_by_source(self, source_id: int) -> List[Dict[str, Any]]:
        """Get all pages for a documentation source."""
        response = self.supabase.table("site_pages").select("*").eq("source_id", source_id).execute()
        return response.data
    
    def get_by_url(self, url: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific URL."""
        response = self.supabase.table("site_pages").select("*").eq("url", url).execute()
        return response.data
    
    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new site page."""
        response = self.supabase.table("site_pages").insert(data).execute()
        return response.data[0]
    
    def upsert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a site page."""
        response = self.supabase.table("site_pages").upsert(
            data,
            on_conflict=["source_id", "url", "chunk_index"]
        ).execute()
        return response.data[0]
    
    def delete_by_source(self, source_id: int) -> bool:
        """Delete all pages for a documentation source."""
        response = self.supabase.table("site_pages").delete().eq("source_id", source_id).execute()
        return True
    
    def search_similar(self, query_embedding: List[float], source_id: Optional[int] = None, 
                      limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar pages using vector similarity."""
        query = """
        SELECT id, url, title, content, metadata, 
               1 - (embedding <=> '[{}]') as similarity
        FROM site_pages
        WHERE embedding IS NOT NULL
        {}
        AND 1 - (embedding <=> '[{}]') > {}
        ORDER BY similarity DESC
        LIMIT {}
        """.format(
            ','.join(str(x) for x in query_embedding),
            f"AND source_id = {source_id}" if source_id is not None else "",
            ','.join(str(x) for x in query_embedding),
            threshold,
            limit
        )
        
        results = self.supabase.execute_raw(query)
        return results.get("data", [])
>>>>>>> ee4b578bf2a45624bbe5312f94b982f7cd411dc1
