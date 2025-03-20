# Operations Guide: Database Component

This guide provides practical instructions for installing, configuring, and maintaining the PostgreSQL database with pgvector for the Agentic RAG system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Maintenance Tasks](#maintenance-tasks)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)
7. [Backup and Recovery](#backup-and-recovery)
8. [FAQs](#faqs)

## System Overview

The database component consists of:

1. **PostgreSQL Database**: Stores documentation content, metadata, and vector embeddings
2. **pgvector Extension**: Provides vector similarity search capabilities
3. **Connection Pool**: Manages database connections efficiently
4. **Schema Operations**: Handles database interactions for the application

Key tables:
- `documentation_sources`: Stores information about documentation sources
- `site_pages`: Stores document chunks with their vector embeddings

## Installation

### PostgreSQL Setup

1. **Install PostgreSQL**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install postgresql-14 postgresql-contrib-14
   
   # CentOS/RHEL
   sudo dnf install postgresql14 postgresql14-server
   sudo /usr/pgsql-14/bin/postgresql-14-setup initdb
   sudo systemctl enable postgresql-14
   sudo systemctl start postgresql-14
   
   # Windows
   # Download and run the installer from https://www.postgresql.org/download/windows/
   ```

2. **Create Database and User**:
   ```bash
   # Access PostgreSQL
   sudo -u postgres psql
   
   # Create database and user
   CREATE DATABASE agentic_rag;
   CREATE USER rag_user WITH ENCRYPTED PASSWORD 'your_password_here';
   GRANT ALL PRIVILEGES ON DATABASE agentic_rag TO rag_user;
   
   # Connect to the database
   \c agentic_rag
   
   # Grant necessary permissions
   GRANT ALL ON SCHEMA public TO rag_user;
   ```

### pgvector Installation

1. **Install Build Dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql-server-dev-14 build-essential git
   
   # CentOS/RHEL
   sudo dnf install postgresql14-devel gcc git
   ```

2. **Build and Install pgvector**:
   ```bash
   # Clone repository
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   
   # Build and install
   make
   sudo make install
   ```

3. **Enable Extension**:
   ```bash
   # Connect to your database
   sudo -u postgres psql -d agentic_rag
   
   # Create extension
   CREATE EXTENSION vector;
   
   # Verify installation
   SELECT * FROM pg_extension WHERE extname = 'vector';
   ```

### Database Schema Setup

1. **Run the Setup Script**:
   ```bash
   # Using the application's setup script
   python scripts/setup_database.py
   
   # Or manually apply the schema
   psql -U rag_user -d agentic_rag -f data/vector_schema_v2.sql
   ```

2. **Verify Setup**:
   ```bash
   # Connect to the database
   psql -U rag_user -d agentic_rag
   
   # Check if tables exist
   \dt
   
   # Sample query to test functionality
   SELECT COUNT(*) FROM documentation_sources;
   ```

## Configuration

### Environment Variables

Configure database connection in the `.env` file:

```
# PostgreSQL Connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agentic_rag
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=your_password_here

# Connection Pooling
POSTGRES_MIN_CONNECTIONS=1
POSTGRES_MAX_CONNECTIONS=10
```

### PostgreSQL Configuration

Optimize PostgreSQL settings in `postgresql.conf`:

```
# Memory Settings
shared_buffers = 2GB                 # 25% of RAM for dedicated server
work_mem = 64MB                      # Helps with complex sorting/joins
maintenance_work_mem = 256MB         # For maintenance operations
effective_cache_size = 6GB           # Estimate of available system cache

# Query Planner
random_page_cost = 1.1               # For SSD storage (default 4.0)

# Write Ahead Log
wal_buffers = 16MB                   # For busy systems
checkpoint_completion_target = 0.9   # Spread out checkpoint I/O
max_wal_size = 2GB                   # Maximum WAL size

# Parallel Query
max_parallel_workers_per_gather = 4  # For multi-core systems
max_parallel_workers = 8             # Maximum parallel workers
```

### Connection Pooling Configuration

Adjust connection pool parameters:

```python
# In your code
from src.db.connection import initialize_connection_pool

# For production with high concurrency
initialize_connection_pool(
    min_connections=5,  
    max_connections=20
)

# For development
initialize_connection_pool(
    min_connections=1,
    max_connections=5
)
```

## Maintenance Tasks

### Regular Maintenance

1. **Database Vacuuming**:
   ```sql
   -- Basic vacuum (can run while system is online)
   VACUUM ANALYZE;
   
   -- Full vacuum (requires exclusive lock)
   VACUUM FULL ANALYZE;
   ```

2. **Index Maintenance**:
   ```sql
   -- Rebuild indexes
   REINDEX TABLE site_pages;
   
   -- Rebuild specific index
   REINDEX INDEX site_pages_embedding_idx;
   ```

3. **Statistics Update**:
   ```sql
   -- Update statistics
   ANALYZE site_pages;
   ANALYZE documentation_sources;
   ```

### Performance Monitoring

1. **Check Index Usage**:
   ```sql
   -- Check index usage statistics
   SELECT 
       relname as table_name,
       indexrelname as index_name,
       idx_scan as index_scans,
       idx_tup_read as tuples_read,
       idx_tup_fetch as tuples_fetched
   FROM pg_stat_user_indexes
   JOIN pg_stat_user_tables ON pg_stat_user_indexes.relname = pg_stat_user_tables.relname
   ORDER BY idx_scan DESC;
   ```

2. **Identify Slow Queries**:
   ```sql
   -- Find slow queries
   SELECT
       query,
       calls,
       total_time,
       mean_time,
       rows
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

3. **Connection Status**:
   ```sql
   -- Check active connections
   SELECT 
       datname as database,
       usename as username,
       application_name,
       client_addr,
       state,
       query
   FROM pg_stat_activity
   WHERE state != 'idle';
   ```

### Scheduled Maintenance Tasks

Set up cron jobs for regular maintenance:

```bash
# /etc/cron.d/postgres-maintenance

# Vacuum analyze every day at 2:30 AM
30 2 * * * postgres /usr/bin/psql -d agentic_rag -c "VACUUM ANALYZE;"

# Reindex every Sunday at 3:00 AM
0 3 * * 0 postgres /usr/bin/psql -d agentic_rag -c "REINDEX TABLE site_pages;"

# Update database statistics every day at 1:00 AM
0 1 * * * postgres /usr/bin/psql -d agentic_rag -c "ANALYZE;"
```

## Performance Tuning

### Vector Search Optimization

1. **Adjust IVF Lists**:
   ```sql
   -- Drop existing index
   DROP INDEX site_pages_embedding_idx;
   
   -- Recreate with optimized parameters
   -- For ~100k vectors, 100-1000 lists is typically good
   -- For ~1M vectors, try 1000-10000 lists
   CREATE INDEX site_pages_embedding_idx ON site_pages 
   USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
   ```

2. **Choose Appropriate Vector Index Type**:
   ```sql
   -- For maximum accuracy (smaller datasets)
   CREATE INDEX site_pages_exact_idx ON site_pages 
   USING hnsw (embedding vector_cosine_ops);
   
   -- For speed with large datasets
   CREATE INDEX site_pages_approx_idx ON site_pages 
   USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
   ```

3. **Partition Large Tables**:
   ```sql
   -- Create partitioned table by source_id
   CREATE TABLE site_pages_partitioned (
       id SERIAL,
       url TEXT,
       -- other columns
       source_id TEXT,
       embedding VECTOR(1536)
   ) PARTITION BY LIST (source_id);
   
   -- Create partitions for each source
   CREATE TABLE site_pages_python PARTITION OF site_pages_partitioned 
   FOR VALUES IN ('python_docs');
   
   CREATE TABLE site_pages_javascript PARTITION OF site_pages_partitioned 
   FOR VALUES IN ('javascript_docs');
   ```

### Query Optimization

1. **Use Prepared Statements**:
   ```python
   # Prepare statement once
   cur.execute(
       "PREPARE embedding_search AS "
       "SELECT id, url, title, content, embedding <=> $1 AS similarity "
       "FROM site_pages "
       "ORDER BY similarity ASC "
       "LIMIT $2"
   )
   
   # Execute multiple times with different parameters
   cur.execute("EXECUTE embedding_search(%s, %s)", (embedding, 10))
   ```

2. **Create Composite Indexes**:
   ```sql
   -- For common query patterns
   CREATE INDEX site_pages_metadata_filter_idx 
   ON site_pages ((metadata->>'source_id'), (metadata->>'doc_type'));
   ```

3. **Optimize JSONB Operations**:
   ```sql
   -- Create GIN index on all metadata
   CREATE INDEX site_pages_metadata_all_idx 
   ON site_pages USING GIN (metadata);
   
   -- Create index on specific metadata fields
   CREATE INDEX site_pages_source_id_idx 
   ON site_pages ((metadata->>'source_id'));
   ```

## Troubleshooting

### Common Issues

| Problem | Possible Causes | Solution |
|---------|----------------|----------|
| Connection failures | Wrong credentials<br>Firewall issues<br>PostgreSQL not running | Check `.env` file<br>Verify firewall settings<br>Restart PostgreSQL |
| Slow queries | Missing indexes<br>Outdated statistics<br>Insufficient resources | Create appropriate indexes<br>Run ANALYZE<br>Increase memory allocation |
| Out of connections | Connection leaks<br>Pool exhaustion | Check for unreleased connections<br>Increase max_connections |
| pgvector errors | Extension not installed<br>Incompatible version | Verify extension installation<br>Check PostgreSQL version compatibility |
| High disk usage | Bloated tables<br>WAL files accumulation | Run VACUUM FULL<br>Check archiving settings |

### Diagnostic Queries

1. **Check Database Size**:
   ```sql
   -- Overall database size
   SELECT pg_size_pretty(pg_database_size('agentic_rag'));
   
   -- Table sizes
   SELECT 
       relname as table_name,
       pg_size_pretty(pg_total_relation_size(relid)) as total_size,
       pg_size_pretty(pg_relation_size(relid)) as table_size,
       pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) as index_size
   FROM pg_catalog.pg_statio_user_tables
   ORDER BY pg_total_relation_size(relid) DESC;
   ```

2. **Identify Unused Indexes**:
   ```sql
   -- Find unused indexes
   SELECT
       indexrelid::regclass as index_name,
       relid::regclass as table_name,
       idx_scan as scans_count
   FROM pg_stat_user_indexes
   WHERE idx_scan = 0
   ORDER BY pg_relation_size(indexrelid) DESC;
   ```

3. **Check Connection Status**:
   ```sql
   -- Check connection states
   SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
   
   -- Find long-running queries
   SELECT pid, now() - query_start as duration, query
   FROM pg_stat_activity
   WHERE state = 'active' AND now() - query_start > interval '5 minutes'
   ORDER BY duration DESC;
   ```

### Troubleshooting PostgreSQL Logs

1. **Find Log Location**:
   ```sql
   SHOW log_directory;
   SHOW log_filename;
   ```

2. **Increase Log Verbosity Temporarily**:
   ```sql
   -- Set log level to debug (check current with SHOW log_min_messages)
   ALTER SYSTEM SET log_min_messages = 'debug1';
   
   -- Apply changes
   SELECT pg_reload_conf();
   
   -- After troubleshooting, reset to default
   ALTER SYSTEM SET log_min_messages = 'warning';
   SELECT pg_reload_conf();
   ```

3. **Enable Query Logging**:
   ```sql
   -- Log all queries
   ALTER SYSTEM SET log_statement = 'all';
   
   -- Log slow queries
   ALTER SYSTEM SET log_min_duration_statement = '1000';  -- 1 second
   
   -- Apply changes
   SELECT pg_reload_conf();
   ```

## Backup and Recovery

### Backup Procedures

1. **Using pg_dump**:
   ```bash
   # Full database backup
   pg_dump -U rag_user -d agentic_rag -f backup.sql
   
   # Compressed backup
   pg_dump -U rag_user -d agentic_rag | gzip > backup.sql.gz
   
   # Custom-format backup (allows selective restore)
   pg_dump -U rag_user -Fc -d agentic_rag -f backup.custom
   ```

2. **Scheduled Backups**:
   ```bash
   # /etc/cron.d/postgres-backup
   
   # Daily backup at 1:00 AM
   0 1 * * * postgres /usr/bin/pg_dump -U rag_user -Fc -d agentic_rag -f /backups/agentic_rag_$(date +\%Y\%m\%d).custom
   
   # Keep only last 30 days of backups
   0 2 * * * postgres find /backups -name "agentic_rag_*.custom" -mtime +30 -delete
   ```

3. **Continuous Archiving (WAL)**:
   
   In `postgresql.conf`:
   ```
   wal_level = replica
   archive_mode = on
   archive_command = 'cp %p /path/to/archive/%f'
   ```

### Recovery Procedures

1. **Restore from pg_dump**:
   ```bash
   # SQL format
   psql -U rag_user -d agentic_rag < backup.sql
   
   # Custom format
   pg_restore -U rag_user -d agentic_rag backup.custom
   ```

2. **Point-in-Time Recovery**:
   
   Create `recovery.conf`:
   ```
   restore_command = 'cp /path/to/archive/%f %p'
   recovery_target_time = '2023-03-15 12:00:00'
   ```

3. **Selective Table Restore**:
   ```bash
   # Extract just the table you need
   pg_restore -U rag_user -t site_pages -d agentic_rag backup.custom
   ```

## FAQs

### General Questions

#### What PostgreSQL version is required?
PostgreSQL 13 or higher is required for pgvector compatibility. PostgreSQL 14 or 15 is recommended for optimal performance.

#### How much disk space is needed?
Plan for approximately 2-5KB per document chunk, plus additional space for indexes. For 100,000 document chunks, allocate at least 1GB plus 50% overhead.

#### Can I use a managed PostgreSQL service?
Yes, you can use any managed PostgreSQL service that supports extensions, such as Amazon RDS, Azure Database for PostgreSQL, or Google Cloud SQL. Ensure the pgvector extension is available.

### Technical Questions

#### How can I monitor the connection pool?
Use `pg_stat_activity` view to monitor active connections:
```sql
SELECT count(*) FROM pg_stat_activity WHERE application_name = 'YourAppName';
```

#### How often should I vacuum the database?
For production systems with regular updates:
- Automatic vacuum should handle most cases
- Run manual `VACUUM ANALYZE` weekly
- Consider `VACUUM FULL` monthly during low-traffic periods

#### What is the optimal chunk size for vector operations?
For pgvector operations with 1536-dimensional embeddings, maintain less than 1 million vectors per table for optimal performance. Consider partitioning or sharding for larger datasets.

### Optimization Questions

#### How can I speed up vector searches?
1. Increase `work_mem` for larger in-memory operations
2. Adjust IVF lists based on your dataset size
3. Consider approximate nearest neighbor (ANN) indexes for large datasets
4. Partition tables by source_id or other frequently filtered fields

#### Should I use connection pooling with pgvector?
Yes, connection pooling is strongly recommended as it reduces the overhead of establishing new connections and makes better use of prepared statements and session-level optimizations.

#### How can I optimize for concurrent write operations?
1. Set appropriate `max_wal_size` to reduce checkpoint frequency
2. Increase `maintenance_work_mem` for faster vacuum operations
3. Batch inserts into single transactions
4. Consider disabling indexes during bulk loads, then rebuilding afterward 