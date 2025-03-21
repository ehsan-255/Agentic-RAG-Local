# Operations Guide: Database Component

This guide provides practical instructions for configuring, operating, and troubleshooting the database component of the Agentic RAG system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Configuration](#configuration)
3. [Database Initialization](#database-initialization)
4. [Operational Tasks](#operational-tasks)
5. [Monitoring and Optimization](#monitoring-and-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Backup and Recovery](#backup-and-recovery)
8. [Frequently Asked Questions](#frequently-asked-questions)

## System Overview

The database component is responsible for:

1. **Storing document content**: Text, metadata, and raw HTML content
2. **Managing vector embeddings**: Storing and indexing embeddings for similarity search
3. **Connection management**: Providing efficient database access through connection pooling
4. **Vector similarity search**: Enabling fast and accurate retrieval of relevant content
5. **Hybrid search capabilities**: Combining vector and text search for optimal results

Key features of the database system:

- Uses PostgreSQL with the pgvector extension
- Supports asynchronous operations with psycopg3 (or synchronous with psycopg2)
- Implements connection pooling for performance optimization
- Provides hybrid search combining vector similarity with text search
- Supports HNSW indexes for faster vector search
- Maintains transaction safety with proper error handling

## Configuration

### Environment Variables

The database component is configured through the following environment variables:

```
# Database connection
DB_HOST=localhost
DB_PORT=5432
DB_NAME=agentic_rag
DB_USER=postgres
DB_PASSWORD=your_password

# Connection pool settings
DB_POOL_MIN_CONN=1
DB_POOL_MAX_CONN=10
DB_POOL_TIMEOUT=30

# Vector search settings
VECTOR_SIMILARITY_THRESHOLD=0.7
DEFAULT_MATCH_COUNT=5
VECTOR_SEARCH_WEIGHT=0.7

# Hybrid search settings
ENABLE_HYBRID_SEARCH=true
TEXT_SEARCH_WEIGHT=0.3
```

### Configuration in `config.py`

Database settings can also be configured in the `src/config.py` file:

```python
# Database Configuration
DATABASE = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "dbname": os.environ.get("DB_NAME", "agentic_rag"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "pool_min_conn": int(os.environ.get("DB_POOL_MIN_CONN", 1)),
    "pool_max_conn": int(os.environ.get("DB_POOL_MAX_CONN", 10)),
    "pool_timeout": int(os.environ.get("DB_POOL_TIMEOUT", 30)),
}

# Vector Search Configuration
VECTOR_SEARCH = {
    "similarity_threshold": float(os.environ.get("VECTOR_SIMILARITY_THRESHOLD", 0.7)),
    "default_match_count": int(os.environ.get("DEFAULT_MATCH_COUNT", 5)),
    "vector_weight": float(os.environ.get("VECTOR_SEARCH_WEIGHT", 0.7)),
    "text_weight": float(os.environ.get("TEXT_SEARCH_WEIGHT", 0.3)),
    "enable_hybrid_search": os.environ.get("ENABLE_HYBRID_SEARCH", "true").lower() == "true",
}
```

## Database Initialization

### Setting Up PostgreSQL with pgvector

1. **Install PostgreSQL** (version 13 or higher recommended)

2. **Install the pgvector extension**:
   
   For Ubuntu/Debian:
   ```bash
   sudo apt-get install postgresql-server-dev-13
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   sudo make install
   ```

   For macOS with Homebrew:
   ```bash
   brew install postgresql@13
   brew install pgvector
   ```

3. **Create the database**:
   ```bash
   createdb agentic_rag
   ```

4. **Enable the pgvector extension**:
   ```sql
   psql -d agentic_rag -c 'CREATE EXTENSION IF NOT EXISTS vector;'
   ```

### Initializing the Database Schema

Run the schema initialization script:

```bash
python -m src.db.init_db
```

This script performs the following:

1. Creates necessary tables if they don't exist
2. Sets up vector indexes for similarity search
3. Creates regular indexes for performance optimization
4. Verifies the pgvector extension is properly installed
5. Sets up schema version tracking

You can also initialize the database programmatically:

```python
from src.db.schema import setup_database

async def initialize():
    success = await setup_database()
    if success:
        print("Database initialized successfully")
    else:
        print("Failed to initialize database")
```

## Operational Tasks

### Managing Documentation Sources

#### Add a Documentation Source

```python
from src.db.schema import add_documentation_source

async def add_new_source():
    source_id = await add_documentation_source(
        name="Python Documentation",
        url="https://docs.python.org"
    )
    print(f"Added source with ID: {source_id}")
```

#### List All Documentation Sources

```python
from src.db.schema import get_documentation_sources

async def list_sources():
    sources = await get_documentation_sources()
    for source in sources:
        print(f"ID: {source['id']}, Name: {source['name']}, Status: {source['status']}")
```

#### Update a Documentation Source

```python
from src.db.schema import update_documentation_source

async def update_source(source_id):
    await update_documentation_source(
        source_id=source_id,
        status="completed",
        pages_count=250
    )
    print(f"Updated source {source_id}")
```

#### Delete a Documentation Source

```python
from src.db.schema import delete_documentation_source

async def remove_source(source_id):
    success = await delete_documentation_source(source_id)
    if success:
        print(f"Deleted source {source_id} and all associated pages")
    else:
        print(f"Failed to delete source {source_id}")
```

### Managing Site Pages

#### Add a Site Page

```python
from src.db.schema import add_site_page

async def add_page(source_id, url, content, embedding):
    page_id = await add_site_page(
        url=url,
        chunk_number=1,
        title="Page Title",
        summary="Page summary",
        content=content,
        metadata={"source_id": source_id},
        embedding=embedding,
        raw_content="<html>...</html>",  # Optional raw HTML
        text_embedding=text_embedding    # Optional text embedding
    )
    print(f"Added page with ID: {page_id}")
```

#### Batch Insert Pages

```python
from src.db.schema import batch_insert_pages

async def add_multiple_pages(pages_data):
    """Add multiple pages in a single transaction."""
    success = await batch_insert_pages(pages_data)
    if success:
        print(f"Added {len(pages_data)} pages successfully")
    else:
        print("Failed to add pages")
```

#### Delete Pages for a URL

```python
from src.db.schema import delete_pages_by_url

async def remove_pages(url):
    count = await delete_pages_by_url(url)
    print(f"Deleted {count} pages for URL: {url}")
```

### Vector Search Operations

#### Basic Vector Search

```python
from src.db.schema import match_site_pages

async def search_similar_content(query_embedding):
    results = await match_site_pages(
        query_embedding=query_embedding,
        match_count=5,
        match_threshold=0.7
    )
    
    for result in results:
        print(f"URL: {result['url']}")
        print(f"Title: {result['title']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Content: {result['content'][:100]}...\n")
```

#### Hybrid Search (Vector + Text)

```python
from src.db.schema import hybrid_search

async def search_with_hybrid(query_text, query_embedding):
    results = await hybrid_search(
        query_text=query_text,
        query_embedding=query_embedding,
        vector_weight=0.7,  # 70% vector, 30% text
        match_count=5
    )
    
    for result in results:
        print(f"URL: {result['url']}")
        print(f"Title: {result['title']}")
        print(f"Combined Score: {result['score']:.4f}")
        print(f"Content: {result['content'][:100]}...\n")
```

#### Search with Metadata Filters

```python
from src.db.schema import filter_by_metadata

async def search_with_filters(query_embedding, source_name):
    results = await filter_by_metadata(
        query_embedding=query_embedding,
        match_count=5,
        metadata_filters={"source_name": source_name}
    )
    
    for result in results:
        print(f"URL: {result['url']}")
        print(f"Title: {result['title']}")
        print(f"Similarity: {result['similarity']:.4f}")
```

## Monitoring and Optimization

### Database Performance Monitoring

Monitor database performance using:

```python
from src.db.monitoring import get_database_stats

async def monitor_db_performance():
    stats = await get_database_stats()
    print(f"Total documentation sources: {stats['sources_count']}")
    print(f"Total pages: {stats['pages_count']}")
    print(f"Average query time: {stats['avg_query_time_ms']} ms")
    print(f"Connection pool usage: {stats['pool_used']}/{stats['pool_total']}")
    print(f"Slow queries (>500ms): {stats['slow_queries_count']}")
```

### Connection Pool Monitoring

Monitor the connection pool status:

```python
from src.db.connection import get_pool_status

async def check_pool():
    status = await get_pool_status()
    print(f"Total connections: {status['size']}")
    print(f"Used connections: {status['used']}")
    print(f"Free connections: {status['free']}")
    print(f"Min connections: {status['min']}")
    print(f"Max connections: {status['max']}")
```

### Query Performance Analysis

Analyze slow queries in the database:

```python
from src.db.monitoring import analyze_slow_queries

async def check_slow_queries():
    slow_queries = await analyze_slow_queries(threshold_ms=500)
    for query in slow_queries:
        print(f"Query: {query['query']}")
        print(f"Execution time: {query['execution_time_ms']} ms")
        print(f"Frequency: {query['count']}")
        print(f"Suggested indexes: {query['suggested_indexes']}\n")
```

### Vector Index Optimization

Optimize vector indexes for better performance:

```python
from src.db.schema import optimize_vector_indexes

async def optimize_indexes():
    await optimize_vector_indexes()
    print("Vector indexes optimized")
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Possible Cause | Solution |
|-------|----------|----------------|----------|
| Connection failures | "Could not connect to database" error | Incorrect connection settings or PostgreSQL not running | Check environment variables, verify PostgreSQL service is running |
| Slow vector searches | Queries taking >500ms | Suboptimal index or too many results | Optimize indexes, reduce match_count, use filtered searches |
| Out of connections | "Too many clients" error | Connection pool exhausted | Increase max_connections, check for connection leaks |
| pgvector not found | "Extension 'vector' not available" | Extension not installed or enabled | Install pgvector, run CREATE EXTENSION |
| Transaction deadlocks | "Deadlock detected" error | Concurrent transactions modifying same rows | Implement retry logic, review transaction isolation levels |
| High memory usage | Server running slowly, OOM errors | Large result sets or inefficient queries | Limit result sizes, optimize queries, add more specific WHERE clauses |

### Diagnostic Tools

#### Check Database Connection

```python
from src.db.utils import check_database_connection

async def verify_connection():
    status = await check_database_connection()
    if status["connected"]:
        print(f"Connected to {status['database']} as {status['user']}")
        print(f"Server version: {status['version']}")
        print(f"pgvector installed: {status['pgvector_installed']}")
    else:
        print(f"Connection failed: {status['error']}")
```

#### Verify Vector Extension

```python
from src.db.utils import verify_pgvector

async def check_pgvector():
    installed = await verify_pgvector()
    if installed:
        print("pgvector extension is properly installed")
    else:
        print("pgvector extension is NOT installed or enabled")
```

#### Run Database Health Check

```python
from src.db.monitoring import run_health_check

async def health_check():
    results = await run_health_check()
    for check, status in results.items():
        print(f"{check}: {'✅ PASS' if status['passed'] else '❌ FAIL'}")
        if not status["passed"]:
            print(f"  Error: {status['error']}")
            print(f"  Recommendation: {status['recommendation']}")
```

## Backup and Recovery

### Backing Up the Database

Daily backups are recommended:

```bash
# Backup the entire database
pg_dump -h localhost -U postgres -d agentic_rag -F c -f backup_$(date +"%Y%m%d").dump

# Backup only the site_pages table
pg_dump -h localhost -U postgres -d agentic_rag -t site_pages -F c -f site_pages_$(date +"%Y%m%d").dump
```

Automated backup script (can be added to cron):

```bash
#!/bin/bash
BACKUP_DIR="/path/to/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DB_HOST="localhost"
DB_USER="postgres"
DB_NAME="agentic_rag"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME -F c -f "$BACKUP_DIR/backup_$TIMESTAMP.dump"

# Clean up old backups (keep only last 7 days)
find $BACKUP_DIR -name "backup_*.dump" -type f -mtime +7 -delete
```

### Restoring from Backup

To restore the database from a backup:

```bash
# Create a new database if needed
createdb -h localhost -U postgres agentic_rag_restored

# Restore from backup
pg_restore -h localhost -U postgres -d agentic_rag_restored backup_20230101.dump
```

### Point-in-Time Recovery

For critical deployments, configure PostgreSQL with Write-Ahead Logging (WAL) for point-in-time recovery:

1. Edit `postgresql.conf`:
   ```
   wal_level = replica
   archive_mode = on
   archive_command = 'cp %p /path/to/archive/%f'
   ```

2. Restart PostgreSQL to apply changes.

3. Perform a base backup:
   ```bash
   pg_basebackup -h localhost -U postgres -D /path/to/backup -Ft -z
   ```

## Frequently Asked Questions

### General Questions

**Q: How much disk space do embeddings require?**

A: Each embedding vector requires approximately 6KB of storage (1536 dimensions × 4 bytes per float). For 100,000 document chunks, this would require about 600MB for embeddings alone.

**Q: Do I need to recreate indexes when upgrading PostgreSQL?**

A: No, PostgreSQL preserves indexes during upgrades. However, after major version upgrades, it's recommended to run `REINDEX` to ensure optimal performance.

**Q: How can I migrate from psycopg2 to psycopg3?**

A: The database layer automatically detects which driver is available and uses the appropriate interface. Simply install psycopg3 with `pip install psycopg[binary]`, and the system will use it if available.

### Technical Questions

**Q: How can I tune vector search performance?**

A: To improve vector search performance:
1. Use HNSW indexes instead of IVF indexes for better performance
2. Adjust vector_weight in hybrid_search based on your needs
3. Use more specific metadata filters to reduce the search space
4. Consider lowering match_count if you don't need many results

**Q: How do I handle schema migrations?**

A: Use the versioned schema migration system:

```python
from src.db.schema import run_migration

async def migrate_database():
    # Apply the latest migration
    success = await run_migration()
    if success:
        print("Migration completed successfully")
    else:
        print("Migration failed")
```

**Q: Can I use multiple vector indexes with different dimensions?**

A: Yes, you can create different tables for different embedding models and dimensions:

```sql
-- For 1536-dimension OpenAI embeddings
CREATE TABLE site_pages_openai (
    id SERIAL PRIMARY KEY,
    page_id INTEGER REFERENCES site_pages(id),
    embedding VECTOR(1536)
);

-- For 768-dimension BERT embeddings
CREATE TABLE site_pages_bert (
    id SERIAL PRIMARY KEY,
    page_id INTEGER REFERENCES site_pages(id),
    embedding VECTOR(768)
);
```

**Q: How do I optimize the PostgreSQL configuration for vector search?**

A: Key PostgreSQL settings to adjust:
- `maintenance_work_mem`: Increase to 1GB or more for index creation
- `effective_cache_size`: Set to 75% of available memory
- `shared_buffers`: Set to 25% of available memory
- `work_mem`: Increase to 64MB or more for complex vector queries
- `random_page_cost`: Lower to 1.1 for SSD storage

### Troubleshooting Questions

**Q: Why are my vector searches returning low similarity scores?**

A: This could be due to:
1. Embeddings generated with different models or parameters
2. Content not semantically related to queries
3. Need for normalization - ensure vectors are normalized before storing

**Q: Database connections are being exhausted. What should I check?**

A: Look for:
1. Connection leaks - ensure all connections are properly released
2. Increase the connection pool maximum (DB_POOL_MAX_CONN)
3. Add connection timeout handling to release stale connections
4. Implement a connection usage log to identify where connections are being used

**Q: Why is the pgvector extension not found even after installation?**

A: Common causes:
1. Extension not created in the specific database: run `CREATE EXTENSION vector;`
2. PostgreSQL server restarted without loading the extension
3. pgvector installed for a different PostgreSQL version
4. Extension library not in PostgreSQL's extension directory 