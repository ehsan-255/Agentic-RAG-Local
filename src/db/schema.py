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
import asyncio
from datetime import datetime

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
    database = os.getenv("POSTGRES_DB", "agentic_rag")
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
        
        # Build the SET part of the query dynamically based on provided parameters
        set_parts = []
        params = []
        
        if pages_count is not None:
            set_parts.append("pages_count = %s")
            params.append(pages_count)
            
        if chunks_count is not None:
            set_parts.append("chunks_count = %s")
            params.append(chunks_count)
            
        if status is not None:
            set_parts.append("status = %s")
            params.append(status)
            
        set_parts.append("last_crawled_at = NOW()")
        
        if not set_parts:
            logger.warning("No fields to update for documentation source")
            return False
            
        # Construct the full query
        query = f"""
        UPDATE documentation_sources
        SET {", ".join(set_parts)}
        WHERE source_id = %s;
        """
        
        # Add source_id to params
        params.append(source_id)
        
        # Execute the query
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
    Add a new site page (documentation chunk) to the database.
    
    Args:
        url: URL of the page
        chunk_number: Chunk number within the page
        title: Title of the page
        summary: Summary of the content
        content: Processed content
        metadata: Additional metadata
        embedding: Vector embedding of the content
        raw_content: Original unprocessed content (optional)
        text_embedding: Alternative text embedding (optional)
        
    Returns:
        Optional[int]: ID of the newly created page, or None if failed
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Insert the new page
        query = """
        INSERT INTO site_pages (
            url, chunk_number, title, summary, content, metadata, embedding, raw_content, text_embedding
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (url, chunk_number) 
        DO UPDATE SET
            title = EXCLUDED.title,
            summary = EXCLUDED.summary,
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            raw_content = EXCLUDED.raw_content,
            text_embedding = EXCLUDED.text_embedding,
            updated_at = NOW()
        RETURNING id;
        """
        
        cur.execute(
            query, 
            (
                url, 
                chunk_number, 
                title, 
                summary, 
                content, 
                Json(metadata), 
                embedding, 
                raw_content, 
                text_embedding
            )
        )
        page_id = cur.fetchone()[0]
        conn.commit()
        
        logger.info(f"Added/updated site page: {url} (chunk: {chunk_number}, ID: {page_id})")
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
    Find similar pages using vector similarity search.
    
    Args:
        query_embedding: Vector embedding of the query
        match_count: Number of matches to return
        filter_metadata: Metadata filters to apply
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        List[Dict[str, Any]]: List of matching pages with similarity scores
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Use the match_site_pages function
        query = """
        SELECT * FROM match_site_pages(%s, %s, %s, %s);
        """
        
        cur.execute(query, (query_embedding, match_count, Json(filter_metadata), similarity_threshold))
        results = cur.fetchall()
        
        # Convert DictCursor results to regular dictionaries
        return [dict(row) for row in results]
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
    Perform hybrid search combining vector similarity and text search.
    
    Args:
        query_text: Text query for keyword search
        query_embedding: Vector embedding for semantic search
        match_count: Number of matches to return
        filter_metadata: Metadata filters to apply
        similarity_threshold: Minimum similarity threshold
        vector_weight: Weight to assign to vector search (vs text search)
        
    Returns:
        List[Dict[str, Any]]: List of matching pages with combined scores
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Use the hybrid_search function
        query = """
        SELECT * FROM hybrid_search(%s, %s, %s, %s, %s, %s);
        """
        
        cur.execute(
            query, 
            (query_text, query_embedding, match_count, Json(filter_metadata), similarity_threshold, vector_weight)
        )
        results = cur.fetchall()
        
        # Convert DictCursor results to regular dictionaries
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error performing hybrid search: {e}")
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
    Find similar pages with advanced metadata filtering.
    
    Args:
        query_embedding: Vector embedding of the query
        match_count: Number of matches to return
        source_id: Filter by source ID
        doc_type: Filter by document type
        min_date: Filter by minimum date
        max_date: Filter by maximum date
        
    Returns:
        List[Dict[str, Any]]: List of matching pages with similarity scores
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Use the filter_by_metadata function
        query = """
        SELECT * FROM filter_by_metadata(%s, %s, %s, %s, %s, %s);
        """
        
        cur.execute(
            query, 
            (query_embedding, match_count, source_id, doc_type, min_date, max_date)
        )
        results = cur.fetchall()
        
        # Convert DictCursor results to regular dictionaries
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error filtering by metadata: {e}")
        return []
    finally:
        if conn:
            release_connection(conn)

def get_document_context(
    page_url: str,
    context_size: int = 3
) -> List[Dict[str, Any]]:
    """
    Get surrounding context chunks for a specific page URL.
    
    Args:
        page_url: URL of the page to get context for
        context_size: Number of context chunks to retrieve
        
    Returns:
        List[Dict[str, Any]]: List of context chunks
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Use the get_document_context function
        query = """
        SELECT * FROM get_document_context(%s, %s);
        """
        
        cur.execute(query, (page_url, context_size))
        results = cur.fetchall()
        
        # Convert DictCursor results to regular dictionaries
        return [dict(row) for row in results]
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
        SELECT * FROM documentation_sources ORDER BY name;
        """
        
        cur.execute(query)
        results = cur.fetchall()
        
        # Convert DictCursor results to regular dictionaries
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error getting documentation sources: {e}")
        return []
    finally:
        if conn:
            release_connection(conn)

def get_source_statistics(source_id: str) -> Dict[str, Any]:
    """
    Get statistics for a specific documentation source.
    
    Args:
        source_id: ID of the documentation source
        
    Returns:
        Dict[str, Any]: Statistics for the source
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        query = """
        SELECT 
            s.name, 
            s.source_id, 
            s.base_url, 
            s.pages_count, 
            s.chunks_count,
            s.created_at,
            s.last_crawled_at,
            COUNT(DISTINCT p.url) AS actual_pages_count,
            COUNT(p.id) AS actual_chunks_count
        FROM 
            documentation_sources s
        LEFT JOIN 
            site_pages p ON p.metadata->>'source_id' = s.source_id
        WHERE 
            s.source_id = %s
        GROUP BY 
            s.id;
        """
        
        cur.execute(query, (source_id,))
        result = cur.fetchone()
        
        if result:
            return dict(result)
        else:
            return {}
    except Exception as e:
        logger.error(f"Error getting source statistics: {e}")
        return {}
    finally:
        if conn:
            release_connection(conn)

def delete_documentation_source(source_id: str) -> bool:
    """
    Delete a documentation source and all its pages.
    
    Args:
        source_id: ID of the documentation source to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # First delete all pages associated with this source
        delete_pages_query = """
        DELETE FROM site_pages
        WHERE metadata->>'source_id' = %s;
        """
        
        cur.execute(delete_pages_query, (source_id,))
        
        # Then delete the source itself
        delete_source_query = """
        DELETE FROM documentation_sources
        WHERE source_id = %s;
        """
        
        cur.execute(delete_source_query, (source_id,))
        conn.commit()
        
        logger.info(f"Deleted documentation source: {source_id}")
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error deleting documentation source: {e}")
        return False
    finally:
        if conn:
            release_connection(conn)

def get_page_content(url: str, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get content for a specific page URL.
    
    Args:
        url: URL of the page
        source_id: Optional source ID to filter by
        
    Returns:
        List[Dict[str, Any]]: List of chunks for the page
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        query = """
        SELECT * FROM site_pages
        WHERE url = %s
        """
        
        params = [url]
        
        if source_id:
            query += " AND metadata->>'source_id' = %s"
            params.append(source_id)
            
        query += " ORDER BY chunk_number;"
        
        cur.execute(query, params)
        results = cur.fetchall()
        
        # Convert DictCursor results to regular dictionaries
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error getting page content: {e}")
        return []
    finally:
        if conn:
            release_connection(conn)

def setup_database():
    """
    Set up the database by checking if the pgvector extension is installed
    and creating the schema if needed.
    
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
