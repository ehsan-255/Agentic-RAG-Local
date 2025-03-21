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
from src.db.connection import get_connection_pool, get_db_connection, execute_query, execute_transaction

# Get a connection pool
pool = await get_connection_pool()

# Execute a query with automatic connection handling
result = await execute_query(
    "SELECT * FROM documentation_sources WHERE name = %s",
    ("Python Documentation",)
)

# Execute multiple operations in a transaction
success = await execute_transaction([
    ("INSERT INTO documentation_sources (name, url) VALUES (%s, %s) RETURNING id", 
     ("Python Documentation", "https://docs.python.org")),
    ("INSERT INTO site_pages (url, source_id) VALUES (%s, %s)",
     ("https://docs.python.org/index.html", 1))
])
```

The connection module now supports:
- Automatic detection of psycopg2 vs psycopg3
- Connection pooling for better performance
- Robust error handling with proper transaction management
- Automatic retries with exponential backoff

### 2. Database Schema Operations (`src/db/schema.py`)

This module provides higher-level database operations:

```python
from src.db.schema import (
    add_documentation_source,
    add_site_page,
    match_site_pages,
    hybrid_search,
    get_documentation_sources,
    delete_documentation_source
)

# Add a new documentation source
source_id = await add_documentation_source(
    name="Python Documentation",
    url="https://docs.python.org"
)

# Store a processed page
page_id = await add_site_page(
    url="https://docs.python.org/tutorial/index.html",
    chunk_number=1,
    title="Python Tutorial",
    summary="Introduction to Python basics",
    content="Detailed content here...",
    metadata={"source_id": source_id, "section": "Tutorial"},
    embedding=[0.1, 0.2, ...],  # Vector embedding
    raw_content="<html>Original HTML</html>",  # Optional raw HTML
    text_embedding=[0.2, 0.3, ...]  # Optional separate text embedding
)

# Perform vector similarity search
results = await match_site_pages(
    query_embedding=[0.1, 0.2, ...],
    match_count=5,
    match_threshold=0.7
)

# Perform hybrid search (vector + keyword)
results = await hybrid_search(
    query_text="async function Python",
    query_embedding=[0.1, 0.2, ...],
    vector_weight=0.7,  # Weight for vector similarity vs text search
    match_count=5
)
```

### 3. Asynchronous Schema Operations (`src/db/async_schema.py`)

This module provides asynchronous database operations:

```python
from src.db.async_schema import (
    add_documentation_source,
    update_documentation_source,
    add_site_page,
    delete_documentation_source
)

# Add a new documentation source
source_id = await add_documentation_source(
    name="FastAPI Documentation",
    url="https://fastapi.tiangolo.com"
)

# Update a source
await update_documentation_source(
    source_id=source_id,
    status="completed",
    pages_count=250,
    completed_at="2023-01-15T12:00:00Z"
)

# Store a page with enhanced error handling
try:
    page_id = await add_site_page(
        url="https://fastapi.tiangolo.com/tutorial/first-steps/",
        chunk_number=1,
        title="First Steps",
        summary="Getting started with FastAPI",
        content="Content goes here...",
        metadata={"source_id": source_id},
        embedding=[0.1, 0.2, ...],
        raw_content="<html>...</html>",  # Now supports raw HTML storage
        text_embedding=[0.2, 0.3, ...]   # Now supports separate text embeddings
    )
    print(f"Page added with ID: {page_id}")
except Exception as e:
    print(f"Error adding page: {e}")
```

### 4. Database Compatibility Layer (`src/db/db_utils.py`)

This module provides compatibility functions for different database drivers:

```python
from src.db.db_utils import (
    get_driver_type,
    is_using_psycopg3,
    adapt_query_params,
    get_cursor_factory
)

# Check which driver is being used
driver_type = get_driver_type()
print(f"Using {'psycopg3' if is_using_psycopg3() else 'psycopg2'}")

# Get the appropriate cursor factory
cursor_factory = get_cursor_factory()

# Adapt query parameters for the current driver
adapted_params = adapt_query_params(params, is_using_psycopg3())
```

## Database Schema

### Key Tables

#### 1. Documentation Sources

```sql
CREATE TABLE documentation_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending',
    pages_count INTEGER DEFAULT 0,
    completed_at TIMESTAMP WITH TIME ZONE
);
```

#### 2. Site Pages

```sql
CREATE TABLE site_pages (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    source_id INTEGER REFERENCES documentation_sources(id) ON DELETE CASCADE,
    title TEXT,
    summary TEXT,
    content TEXT NOT NULL,
    raw_content TEXT,  -- New: stores original HTML
    metadata JSONB,
    embedding VECTOR(1536),  -- Using pgvector type
    text_embedding VECTOR(1536),  -- New: separate embedding for text search
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (url, chunk_number)
);
```

### Indexes

```sql
-- Regular indexes
CREATE INDEX idx_site_pages_source_id ON site_pages(source_id);
CREATE INDEX idx_site_pages_url ON site_pages(url);

-- JSON path operators for metadata search
CREATE INDEX idx_site_pages_metadata ON site_pages USING GIN (metadata);

-- Vector indexes (new: HNSW index for faster search)
CREATE INDEX site_pages_embedding_idx ON site_pages USING hnsw (embedding vector_cosine_ops);
CREATE INDEX site_pages_text_embedding_idx ON site_pages USING hnsw (text_embedding vector_cosine_ops);
```

## Integration Points

### 1. With Crawler Component

The database module integrates with the crawler component:

```python
from src.db.async_schema import add_documentation_source, add_site_page
from src.crawling.enhanced_docs_crawler import crawl_documentation, CrawlConfig

async def crawl_and_store():
    # Add a new documentation source
    source_id = await add_documentation_source(
        name="Python Documentation",
        url="https://docs.python.org"
    )
    
    # Configure and run the crawler
    config = CrawlConfig(
        name="Python Documentation",
        sitemap_url="https://docs.python.org/3/sitemap.xml",
        # Additional config...
    )
    
    # The crawler will use add_site_page to store content
    await crawl_documentation(config)
```

### 2. With RAG Component

The database module integrates with the RAG component:

```python
from src.db.schema import match_site_pages, hybrid_search, get_page_content
from src.rag.rag_expert import AgentyRagDeps, agentic_rag_expert

async def query_documentation(user_query, query_embedding):
    # Use hybrid search for better results
    search_results = await hybrid_search(
        query_text=user_query,
        query_embedding=query_embedding,
        vector_weight=0.7,
        match_count=5
    )
    
    # Get full content for matches
    contexts = []
    for result in search_results:
        page_content = await get_page_content(result["id"])
        contexts.append({
            "content": page_content["content"],
            "url": page_content["url"],
            "title": page_content["title"]
        })
    
    # Feed to RAG agent
    deps = AgentyRagDeps(openai_client=openai_client)
    response = await agentic_rag_expert(user_query, contexts, deps)
    return response
```

## Vector Search

### 1. Similarity Search

The system supports different search strategies:

```python
from src.db.schema import match_site_pages, hybrid_search, filter_by_metadata

# Basic vector similarity search
results = await match_site_pages(
    query_embedding=query_embedding,
    match_count=5
)

# Hybrid search (vector + text)
results = await hybrid_search(
    query_text="Python async functions",
    query_embedding=query_embedding,
    vector_weight=0.7,  # 70% vector, 30% text
    match_count=5
)

# Filtered search
results = await filter_by_metadata(
    query_embedding=query_embedding,
    match_count=5,
    metadata_filters={
        "source_id": "python_docs",
        "section": "tutorial"
    }
)
```

### 2. Advanced Vector Operations

The system supports advanced vector operations:

```python
from src.db.schema import combine_vectors, search_by_document

# Combine multiple vectors for composite search
combined_embedding = await combine_vectors(
    [embedding1, embedding2, embedding3],
    weights=[0.6, 0.3, 0.1]
)

# Search for entire documents rather than chunks
document_results = await search_by_document(
    query_embedding=query_embedding,
    match_count=3,
    aggregation="max"  # Options: max, mean, weighted
)
```

## Extending the System

### 1. Adding a New Table

To add a new table to the schema:

```python
# In src/db/schema.py
async def setup_database():
    # ... existing tables ...
    
    # Add new table
    await execute_query("""
    CREATE TABLE IF NOT EXISTS user_queries (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255),
        query TEXT NOT NULL,
        embedding VECTOR(1536),
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        feedback INTEGER
    );
    """)
    
    # Add indexes
    await execute_query("""
    CREATE INDEX IF NOT EXISTS idx_user_queries_user_id ON user_queries(user_id);
    CREATE INDEX IF NOT EXISTS idx_user_queries_embedding_idx ON user_queries USING hnsw (embedding vector_cosine_ops);
    """)
```

### 2. Implementing a Custom Search Method

To add a custom search method:

```python
# In src/db/schema.py
async def semantic_cluster_search(query_embedding, match_count=5):
    """Search that returns diverse results from different semantic clusters."""
    query = """
    WITH initial_matches AS (
        SELECT 
            id, url, title, summary, content, metadata,
            1 - (embedding <=> %s) AS similarity
        FROM site_pages
        WHERE 1 - (embedding <=> %s) > 0.7
        ORDER BY similarity DESC
        LIMIT 20
    ),
    clusters AS (
        SELECT 
            id, url, title, summary, content, metadata, similarity,
            NTILE(5) OVER (ORDER BY embedding <=> %s) AS cluster
        FROM initial_matches
    )
    SELECT id, url, title, summary, content, metadata, similarity
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY cluster ORDER BY similarity DESC) AS rn
        FROM clusters
    ) ranked
    WHERE rn = 1
    ORDER BY similarity DESC
    LIMIT %s;
    """
    
    params = [query_embedding, query_embedding, query_embedding, match_count]
    results = await execute_query(query, params)
    return results
```

### 3. Adding Support for a New Embedding Model

To add support for a different embedding dimension:

```python
# In src/db/schema.py
async def setup_database_for_model(model_name, dimensions):
    """Set up tables for a specific embedding model."""
    # Create model-specific tables
    await execute_query(f"""
    CREATE TABLE IF NOT EXISTS site_pages_{model_name} (
        id SERIAL PRIMARY KEY,
        site_page_id INTEGER REFERENCES site_pages(id) ON DELETE CASCADE,
        embedding VECTOR({dimensions}),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Create appropriate indexes
    await execute_query(f"""
    CREATE INDEX IF NOT EXISTS idx_site_pages_{model_name}_embedding 
    ON site_pages_{model_name} USING hnsw (embedding vector_cosine_ops);
    """)
```

## Best Practices

### 1. Connection Management

Always use the provided connection management functions:

```python
# Good - uses connection pooling and error handling
result = await execute_query("SELECT * FROM documentation_sources")

# Good - transaction with automatic rollback on errors
success = await execute_transaction([
    ("INSERT INTO documentation_sources (name, url) VALUES (%s, %s)", 
     ("Python Docs", "https://docs.python.org")),
    ("UPDATE documentation_sources SET status = %s WHERE id = %s",
     ("active", 1))
])

# Avoid - manual connection management
# Problematic if connections aren't properly closed or errors aren't handled
conn = await get_db_connection()
cursor = conn.cursor()
await cursor.execute("SELECT * FROM documentation_sources")
# ...
```

### 2. Error Handling

Implement proper error handling for database operations:

```python
try:
    result = await execute_query(
        "INSERT INTO documentation_sources (name, url) VALUES (%s, %s) RETURNING id",
        ("Python Documentation", "https://docs.python.org")
    )
    source_id = result[0]["id"]
except Exception as e:
    logger.structured_error(
        "Failed to add documentation source",
        error=str(e),
        source="Python Documentation"
    )
    # Take appropriate action based on error
```

### 3. Performance Optimization

Optimize database queries for performance:

```python
# Good - specific columns, indexed lookup
await execute_query(
    "SELECT id, title, summary FROM site_pages WHERE source_id = %s",
    (source_id,)
)

# Good - batched operations for better performance
page_data = [(url, chunk, title, summary, content, metadata, embedding) 
             for url, chunk, title, summary, content, metadata, embedding in pages]
await execute_batch_insert(
    "INSERT INTO site_pages (url, chunk_number, title, summary, content, metadata, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
    page_data
)

# Avoid - retrieving unnecessary data
# await execute_query("SELECT * FROM site_pages")
```

### 4. Schema Evolution

When updating the database schema:

```python
# Version your schema changes
async def upgrade_schema_to_v2():
    # 1. Add new columns with default values (non-disruptive)
    await execute_query("""
    ALTER TABLE site_pages 
    ADD COLUMN IF NOT EXISTS raw_content TEXT,
    ADD COLUMN IF NOT EXISTS text_embedding VECTOR(1536);
    """)
    
    # 2. Create new indexes
    await execute_query("""
    CREATE INDEX IF NOT EXISTS idx_site_pages_text_embedding 
    ON site_pages USING hnsw (text_embedding vector_cosine_ops);
    """)
    
    # 3. Record schema version
    await execute_query("""
    INSERT INTO schema_versions (version, applied_at, description)
    VALUES ('2.0', CURRENT_TIMESTAMP, 'Added raw_content and text_embedding')
    """)
```

### 5. Security Considerations

Always use parameterized queries to prevent SQL injection:

```python
# Good - parameterized query
source_id = await execute_query(
    "SELECT id FROM documentation_sources WHERE name = %s",
    (source_name,)
)

# Avoid - string concatenation (vulnerable to SQL injection)
# source_id = await execute_query(
#     f"SELECT id FROM documentation_sources WHERE name = '{source_name}'")
``` 