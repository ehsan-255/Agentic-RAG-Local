# Developer Guide: Database Component

This guide provides technical documentation for developers who need to integrate with, extend, or modify the database component of the Agentic RAG system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Database Schema](#database-schema)
4. [Integration Points](#integration-points)
5. [Vector Search](#vector-search)
6. [Extending the System](#extending-the-system)
7. [Best Practices](#best-practices)

## Architecture Overview

The database system uses PostgreSQL with the pgvector extension to store and query document embeddings:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Application    │       │   Database      │       │    pgvector     │
│  (Python)       │──────▶│   (PostgreSQL)  │──────▶│    Extension    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Connection    │       │    Schema       │       │    Vector       │
│    Manager      │◀─────▶│   Operations    │◀─────▶│    Indexes      │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

The architecture follows these key principles:

1. **Driver Compatibility**: Supporting both psycopg2 and psycopg3 through a compatibility layer
2. **Connection Pooling**: Efficient database connection management
3. **Vector Similarity**: Fast similarity search using pgvector
4. **Hybrid Search**: Combining vector search with text search for better results
5. **Transactional Safety**: Ensuring data integrity with proper transaction handling

## Key Components

### 1. Database Connection Management (`src/db/connection.py`)

The connection manager handles database connections and pooling:

```python
from src.db.connection import initialize_connection_pool, get_connection, release_connection

# Initialize the connection pool
initialize_connection_pool(min_connections=1, max_connections=10)

# Get a connection from the pool
conn = get_connection()

try:
    # Use the connection
    cur = conn.cursor()
    cur.execute("SELECT * FROM documentation_sources")
    sources = cur.fetchall()
finally:
    # Always release the connection back to the pool
    release_connection(conn)
```

### 2. Schema Operations (`src/db/schema.py`)

Core database operations for the application:

```python
from src.db.schema import add_documentation_source, add_site_page, match_site_pages

# Add a new documentation source
source_id = add_documentation_source(
    name="Python Documentation",
    source_id="python_docs",
    base_url="https://docs.python.org/3/",
    configuration={"language": "python", "version": "3.10"}
)

# Add a new page with its embedding
page_id = add_site_page(
    url="https://docs.python.org/3/tutorial/index.html",
    chunk_number=0,
    title="The Python Tutorial",
    summary="This tutorial introduces the reader to the basic concepts and features of Python.",
    content="Python is an easy to learn, powerful programming language...",
    metadata={"source_id": "python_docs", "page_type": "tutorial"},
    embedding=[0.1, 0.2, ..., 0.3]  # Vector embedding
)

# Find similar pages
results = match_site_pages(
    query_embedding=[0.1, 0.2, ..., 0.3],  # Query embedding
    match_count=5,
    filter_metadata={"source_id": "python_docs"}
)
```

### 3. Asynchronous Operations (`src/db/async_schema.py`)

Asynchronous database operations for non-blocking code:

```python
from src.db.async_schema import add_documentation_source, add_site_page

# Add source asynchronously
source_id = await add_documentation_source(
    name="Python Documentation",
    source_id="python_docs",
    base_url="https://docs.python.org/3/",
    configuration={"language": "python", "version": "3.10"}
)

# Add page asynchronously
page_id = await add_site_page(
    url="https://docs.python.org/3/tutorial/index.html",
    chunk_number=0,
    title="The Python Tutorial",
    summary="Summary of the page",
    content="Page content...",
    metadata={"source_id": "python_docs"},
    embedding=[0.1, 0.2, ..., 0.3]
)
```

### 4. Driver Compatibility Layer (`src/db/db_utils.py`)

Provides compatibility between psycopg2 and psycopg3:

```python
from src.db.db_utils import is_psycopg3_available, is_database_available

# Check which driver is available
if is_psycopg3_available():
    print("Using psycopg3 (async capable)")
else:
    print("Using psycopg2 (sync only)")

# Check database connectivity
if is_database_available():
    print("Database is available")
else:
    print("Database is not available")
```

## Database Schema

### Main Tables

The database consists of two primary tables:

#### `documentation_sources`

Stores information about documentation sources:

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| name | TEXT | Name of the documentation source |
| source_id | TEXT | Unique identifier for the source |
| base_url | TEXT | Base URL of the documentation |
| configuration | JSONB | Configuration options (JSON) |
| created_at | TIMESTAMP | Creation timestamp |
| last_crawled_at | TIMESTAMP | Last crawl timestamp |
| pages_count | INTEGER | Number of pages |
| chunks_count | INTEGER | Number of chunks |
| status | TEXT | Status (active, crawling, etc.) |

#### `site_pages`

Stores document chunks with their embeddings:

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| url | TEXT | URL of the page |
| chunk_number | INTEGER | Chunk sequence number |
| title | TEXT | Title of the chunk |
| summary | TEXT | Summary of the chunk |
| content | TEXT | Content of the chunk |
| raw_content | TEXT | Raw HTML content (optional) |
| metadata | JSONB | Metadata (JSON) |
| embedding | VECTOR | Vector embedding for similarity search |
| text_embedding | VECTOR | Optional secondary embedding |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update timestamp |

### Key Indexes

Important indexes for performance:

```sql
-- Primary indexes
CREATE INDEX site_pages_url_idx ON site_pages(url);
CREATE INDEX site_pages_metadata_idx ON site_pages USING GIN (metadata);

-- Vector indexes
CREATE INDEX site_pages_embedding_idx ON site_pages 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## Integration Points

### Database Initialization

To set up the database in your application:

```python
from src.db.schema import setup_database

# Initialize database schema
if setup_database():
    print("Database setup successful")
else:
    print("Database setup failed")
```

### Adding Custom Tables

To extend the schema with your own tables:

1. Create a SQL file with your table definitions
2. Use the schema creation function to apply it:

```python
from src.db.schema import create_schema_from_file

# Apply your custom schema
create_schema_from_file("path/to/your_schema.sql")
```

### Custom Queries

For custom database operations:

```python
from src.db.connection import get_connection, release_connection

def run_custom_query(param1, param2):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Execute your custom query
        cur.execute(
            "SELECT * FROM your_table WHERE field1 = %s AND field2 = %s",
            (param1, param2)
        )
        
        return cur.fetchall()
    finally:
        if conn:
            release_connection(conn)
```

## Vector Search

### Basic Vector Search

Perform similarity search using vector embeddings:

```python
from src.db.schema import match_site_pages

# Find similar documents
results = match_site_pages(
    query_embedding=query_embedding,  # List[float]
    match_count=10,
    similarity_threshold=0.7
)

# Process results
for result in results:
    print(f"URL: {result['url']}")
    print(f"Title: {result['title']}")
    print(f"Similarity: {result['similarity']}")
    print(f"Content: {result['content'][:100]}...")
```

### Hybrid Search

Combine vector similarity with text search:

```python
from src.db.schema import hybrid_search

# Perform hybrid search
results = hybrid_search(
    query_text="python installation guide",
    query_embedding=query_embedding,
    match_count=5,
    vector_weight=0.7  # 70% vector similarity, 30% text match
)
```

### Filtered Search

Filter search results by metadata:

```python
from src.db.schema import match_site_pages

# Search with filters
results = match_site_pages(
    query_embedding=query_embedding,
    match_count=5,
    filter_metadata={
        "source_id": "python_docs",
        "doc_type": "tutorial"
    }
)
```

## Extending the System

### Custom Search Functions

Create specialized search functions for your use case:

```python
def search_by_category(query_embedding, category, match_count=5):
    """Search for documents in a specific category."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        query = """
        SELECT 
            id, url, title, content, metadata,
            embedding <=> %s AS similarity
        FROM 
            site_pages
        WHERE 
            metadata->>'category' = %s
        ORDER BY 
            similarity ASC
        LIMIT %s;
        """
        
        cur.execute(query, (query_embedding, category, match_count))
        return cur.fetchall()
    finally:
        if conn:
            release_connection(conn)
```

### Advanced Vector Operations

For more advanced pgvector operations:

```python
def find_document_clusters(embedding_centroid, radius=0.3, limit=20):
    """Find clusters of similar documents within a radius."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        query = """
        SELECT 
            url,
            metadata->>'source_id' AS source,
            embedding <=> %s AS distance
        FROM 
            site_pages
        WHERE
            embedding <=> %s < %s
        GROUP BY
            url, source, distance
        ORDER BY 
            distance ASC
        LIMIT %s;
        """
        
        cur.execute(query, (embedding_centroid, embedding_centroid, radius, limit))
        return cur.fetchall()
    finally:
        if conn:
            release_connection(conn)
```

### Custom Indexing Strategies

Create specialized indexes for your specific query patterns:

```sql
-- For frequent filtering on a specific metadata field
CREATE INDEX site_pages_source_id_idx ON site_pages ((metadata->>'source_id'));

-- For documents with temporal relevance 
CREATE INDEX site_pages_created_idx ON site_pages (created_at DESC);

-- For combining metadata filtering with vector search
CREATE INDEX site_pages_filtered_embedding_idx ON site_pages 
USING ivfflat ((embedding)) 
WHERE (metadata->>'importance')::int > 3;
```

## Best Practices

### Connection Management

1. **Always Release Connections**: Use try/finally blocks to ensure connections are returned to the pool
2. **Connection Pooling**: Use the connection pool rather than creating new connections
3. **Transaction Handling**: Explicitly commit or rollback transactions

```python
conn = get_connection()
try:
    # Start a transaction
    conn.autocommit = False
    
    # Perform multiple operations
    # ...
    
    # Commit if successful
    conn.commit()
except Exception as e:
    # Rollback on error
    if conn:
        conn.rollback()
    raise
finally:
    # Always release the connection
    if conn:
        release_connection(conn)
```

### Query Optimization

1. **Use Parameterized Queries**: Prevents SQL injection and improves query plan caching
2. **Limit Result Sizes**: Always use LIMIT in queries that could return large result sets
3. **Index Key Columns**: Create appropriate indexes for frequently queried columns
4. **Optimize Vector Indexes**: Adjust IVF list count based on your data size

### Vector Operation Efficiency

1. **Batch Embedding Generation**: Generate embeddings in batches rather than individually
2. **Vector Dimension Considerations**: Higher dimensions provide more precision but consume more space
3. **Similarity Metrics**: Choose appropriate distance metrics (cosine, L2, inner product) for your use case
4. **Approximate vs. Exact Search**: Use approximate search for large datasets, exact for smaller ones

### Working with psycopg3

If using the newer psycopg3 driver:

```python
import psycopg
from psycopg.rows import dict_row

async def async_query():
    # Connect asynchronously
    async with await psycopg.AsyncConnection.connect(
        conninfo=get_connection_string()
    ) as aconn:
        # Use dictionary cursor
        async with aconn.cursor(row_factory=dict_row) as acur:
            await acur.execute(
                "SELECT * FROM site_pages WHERE id = %s",
                (page_id,)
            )
            result = await acur.fetchone()
            return result
```

### Error Handling

1. **Specific Exceptions**: Catch specific psycopg2/psycopg3 exceptions
2. **Retry Logic**: Implement retries for transient errors
3. **Connection Verification**: Check connection validity before use
4. **Logging**: Log database errors with appropriate context

```python
from psycopg2 import errors

try:
    # Database operation
    result = add_site_page(...)
except errors.UniqueViolation:
    # Handle duplicate entry
    logger.warning(f"Duplicate entry for URL: {url}")
except errors.OperationalError as e:
    # Handle operational issues
    logger.error(f"Database operational error: {e}")
    # Attempt reconnection
    reinitialize_connection_pool()
except Exception as e:
    # Log unexpected errors
    logger.exception(f"Unexpected database error: {e}")
    raise
``` 