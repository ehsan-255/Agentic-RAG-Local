# Database Schema Documentation

This document describes the database schema used in the Agentic RAG Local system. The schema is designed to efficiently store and retrieve documentation content with vector embeddings for similarity search.

## Installation Requirements

- PostgreSQL 14 or later
- pgvector extension ([installation instructions](https://github.com/pgvector/pgvector#installation))

## Tables

### documentation_sources

Stores information about documentation sources that have been crawled.

| Column | Type | Description |
|--------|------|-------------|
| id | bigserial | Primary key |
| name | varchar | Name of the documentation source |
| source_id | varchar | Unique identifier for the source |
| base_url | varchar | Base URL of the documentation |
| configuration | jsonb | Configuration options (crawler settings, etc.) |
| created_at | timestamp with time zone | Creation timestamp |
| last_crawled_at | timestamp with time zone | Last crawl timestamp |
| pages_count | integer | Number of pages crawled |
| chunks_count | integer | Number of content chunks created |
| status | varchar | Status of the source (active, archived, etc.) |

### site_pages

Stores the actual documentation content, chunked and embedded for retrieval.

| Column | Type | Description |
|--------|------|-------------|
| id | bigserial | Primary key |
| url | varchar | URL of the page |
| chunk_number | integer | Sequence number for content chunks |
| title | varchar | Title of the page |
| summary | varchar | Summary of the content |
| content | text | Processed content text |
| raw_content | text | Original unprocessed content |
| metadata | jsonb | Metadata (source_id, type, tags, etc.) |
| embedding | vector(1536) | OpenAI embedding vector |
| text_embedding | vector(384) | Optional smaller embedding |
| created_at | timestamp with time zone | Creation timestamp |
| updated_at | timestamp with time zone | Last update timestamp |

Unique constraint: (url, chunk_number)

## Indexes

The following indexes are created for optimal performance:

### HNSW Index

HNSW (Hierarchical Navigable Small World) is an algorithm for approximate nearest neighbor search, offering fast query performance with a reasonable trade-off in accuracy.

```sql
CREATE INDEX idx_site_pages_hnsw ON site_pages USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

Parameters:
- `m`: Number of connections per layer (higher = better recall but more memory)
- `ef_construction`: Size of the dynamic candidate list during index construction

### IVF Index

IVF (Inverted File) indexes divide vectors into lists for faster retrieval. Better for larger datasets.

```sql
CREATE INDEX idx_site_pages_ivfflat ON site_pages USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

Parameters:
- `lists`: Number of lists to divide vectors into (ideally 10 * sqrt(row_count))

### Other Indexes

- `idx_site_pages_metadata`: GIN index on the metadata JSONB column
- `idx_site_pages_source_id`: Index for fast lookups by source_id
- `idx_site_pages_content_trgm`: Trigram index for text search performance

## Functions

### match_site_pages

Performs vector similarity search with metadata filtering.

```sql
match_site_pages(
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb default '{}'::jsonb,
  similarity_threshold float default 0.7
)
```

### hybrid_search

Combines vector similarity with text search for better results.

```sql
hybrid_search(
  query_text text,
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb default '{}'::jsonb,
  similarity_threshold float default 0.7,
  vector_weight float default 0.7
)
```

### filter_by_metadata

Enhanced metadata filtering with multiple conditions.

```sql
filter_by_metadata(
  query_embedding vector(1536),
  match_count int default 10,
  source_id text default null,
  doc_type text default null,
  min_date timestamp default null,
  max_date timestamp default null
)
```

### get_document_context

Retrieves chunks from the same document for context.

```sql
get_document_context(
  page_url text,
  context_size int default 3
)
```

### expand_query

Expands queries with related terms for better search results.

```sql
expand_query(
  query_text text
)
```

### increment_pages_count and increment_chunks_count

Utility functions to update source statistics.

## Query Examples

### Basic Vector Search

```sql
SELECT * FROM match_site_pages(
  '{0.1, 0.2, ...}' -- 1536 dimensions
);
```

### Hybrid Search with Metadata Filtering

```sql
SELECT * FROM hybrid_search(
  'how to install pgvector',
  '{0.1, 0.2, ...}',
  match_count := 5,
  filter := '{"source_id": "postgresql_docs"}'
);
```

### Advanced Metadata Filtering

```sql
SELECT * FROM filter_by_metadata(
  '{0.1, 0.2, ...}',
  source_id := 'python_docs',
  doc_type := 'tutorial',
  min_date := '2023-01-01'
);
```

### Retrieving Document Context

```sql
SELECT * FROM get_document_context(
  'https://example.com/docs/page',
  context_size := 2
);
```

## Performance Considerations

- The HNSW index provides the best query performance for most use cases
- For very large datasets (millions of vectors), consider using the IVF index
- Use the `similarity_threshold` parameter to filter out low-quality matches
- In hybrid search, adjust `vector_weight` based on your use case (higher values prioritize semantic similarity)
- Use `EXPLAIN ANALYZE` to debug performance issues

## Row Level Security (RLS)

Row Level Security is enabled for both tables with policies that allow public read access. This is particularly useful when using Supabase. 