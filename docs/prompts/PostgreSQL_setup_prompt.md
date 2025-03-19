# Setting Up PostgreSQL + pgvector for Agentic RAG Systems

## Phase 1: Environment Setup

### 1. Prerequisites Installation
1. **PostgreSQL Database Server**
   - Download PostgreSQL installer (version 14+ recommended) from postgresql.org
   - Run the installer with administrative privileges
   - Select components: PostgreSQL Server, pgAdmin (GUI tool), Command Line Tools
   - Choose a data directory with sufficient storage space
   - Set a secure master password for the PostgreSQL superuser

2. **pgvector Extension Installation**
   - Option A (Package Manager):
     - On Ubuntu/Debian: `apt-get install postgresql-14-pgvector`
     - On macOS with Homebrew: `brew install pgvector`
   - Option B (Manual compilation):
     - Install PostgreSQL development headers
     - Clone pgvector repository: `git clone https://github.com/pgvector/pgvector.git`
     - Compile and install the extension
     - Run `make && make install` in the pgvector directory

3. **Network Configuration**
   - Configure PostgreSQL to listen on appropriate network interfaces
   - Edit `postgresql.conf` to set `listen_addresses` parameter
   - Modify `pg_hba.conf` to define access control rules for the application

### 2. Database Configuration
1. **Create Application Database**
   - Connect to PostgreSQL using psql or pgAdmin
   - Create a dedicated database: `CREATE DATABASE agentic_rag;`
   - Create application-specific user with appropriate privileges

2. **Enable pgvector Extension**
   - Connect to the new database
   - Run: `CREATE EXTENSION IF NOT EXISTS vector;`
   - Verify installation with: `SELECT * FROM pg_extension WHERE extname = 'vector';`

3. **Configure Vector Search Parameters**
   - Set appropriate memory parameters for vector operations
   - Configure `maintenance_work_mem` parameter (recommend 1GB+ for large vector datasets)
   - Adjust `max_parallel_workers` for improved search performance

## Phase 2: Schema Creation

### 1. Database Schema Setup
1. **Create Tables with Vector Support**
   - Create `documentation_sources` table for tracking document sources
   - Create `site_pages` table with vector column:
     ```sql
     embedding vector(1536)
     ```
   - Ensure proper data types for all columns, especially JSON/JSONB for metadata

2. **Create Vector Indexes**
   - Implement HNSW index for fast approximate nearest neighbor search:
     ```sql
     CREATE INDEX ON site_pages USING hnsw (embedding vector_cosine_ops);
     ```
   - Consider IVF (Inverted File) index for larger datasets if exact search is less critical

### 2. Function Implementation
1. **Create Database Functions**
   - Create the `match_site_pages` function for vector similarity search
   - Implement `increment_pages_count` and `increment_chunks_count` functions
   - Add hybrid search function for combining lexical and semantic search

2. **Setup Enhanced Functions**
   - Create query transformation function to expand and improve queries
   - Implement hybrid search function that combines BM25 text search with vector similarity
   - Develop functions for enhanced metadata filtering and aggregation

## Phase 3: Application Code Integration

### 1. Connection Management
1. **Database Connection Configuration**
   - Replace Supabase client with direct PostgreSQL connection
   - Implement connection pooling (using libraries like `psycopg_pool` for Python)
   - Set appropriate connection timeouts and retry logic

2. **Authentication Implementation**
   - Implement direct authentication mechanism
   - Set up user management directly in PostgreSQL
   - Configure role-based access controls if needed

### 2. Query Implementation
1. **Vector Search Queries**
   - Implement direct SQL queries for vector search
   - Create parameterized queries to prevent SQL injection
   - Set up vector similarity search logic with appropriate operators (`<=>` for cosine similarity)

2. **Enhanced Query Features**
   - Add query transformation layer that enhances queries before vector search
   - Integrate hybrid search with weighting between keyword and semantic matches
   - Expand metadata filtering capabilities in queries