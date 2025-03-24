-- Increase maintenance_work_mem for index creation
SET maintenance_work_mem = '1GB';

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the documentation sources table
CREATE TABLE IF NOT EXISTS documentation_sources (
    id bigserial PRIMARY KEY,
    name varchar NOT NULL,
    source_id varchar NOT NULL UNIQUE,
    base_url varchar NOT NULL,
    configuration jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    last_crawled_at timestamp with time zone,
    pages_count integer NOT NULL DEFAULT 0,
    chunks_count integer NOT NULL DEFAULT 0,
    status varchar DEFAULT 'active'
);

-- Create the documentation chunks table
CREATE TABLE IF NOT EXISTS site_pages (
    id bigserial PRIMARY KEY,
    url varchar NOT NULL,
    chunk_number integer NOT NULL,
    title varchar NOT NULL,
    summary varchar NOT NULL,
    content text NOT NULL,
    raw_content text,  -- For storing original unprocessed content
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    text_embedding vector(384),  -- Optional smaller embedding for alternative models
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    UNIQUE(url, chunk_number)
);

-- Drop existing indexes if they exist
DROP INDEX IF EXISTS idx_site_pages_ivfflat;
DROP INDEX IF EXISTS idx_site_pages_hnsw;
DROP INDEX IF EXISTS idx_site_pages_metadata;
DROP INDEX IF EXISTS idx_site_pages_source_id;
DROP INDEX IF EXISTS idx_site_pages_content_trgm;

-- Create an HNSW index for fast approximate nearest neighbor search
CREATE INDEX idx_site_pages_hnsw ON site_pages USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create an IVF index as an alternative for larger datasets
CREATE INDEX idx_site_pages_ivfflat ON site_pages USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create an index on metadata for faster filtering
CREATE INDEX idx_site_pages_metadata ON site_pages USING gin (metadata);

-- Create an index for fast lookups by source_id
CREATE INDEX idx_site_pages_source_id ON site_pages ((metadata->>'source_id'));

-- Create a GIN index with trigram similarity for text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_site_pages_content_trgm ON site_pages USING gin (content gin_trgm_ops);

-- Create a function to increment pages count
CREATE OR REPLACE FUNCTION increment_pages_count(source_id_param text)
RETURNS integer AS $$
DECLARE
    current_count integer;
BEGIN
    UPDATE documentation_sources 
    SET pages_count = pages_count + 1,
        last_crawled_at = NOW()
    WHERE source_id = source_id_param
    RETURNING pages_count INTO current_count;
    
    RETURN current_count;
END;
$$ LANGUAGE plpgsql;

-- Create a function to increment chunks count
CREATE OR REPLACE FUNCTION increment_chunks_count(source_id_param text, increment_by integer)
RETURNS integer AS $$
DECLARE
    current_count integer;
BEGIN
    UPDATE documentation_sources 
    SET chunks_count = chunks_count + increment_by,
        last_crawled_at = NOW()
    WHERE source_id = source_id_param
    RETURNING chunks_count INTO current_count;
    
    RETURN current_count;
END;
$$ LANGUAGE plpgsql;

-- Create a function to search for documentation chunks using vector similarity
CREATE OR REPLACE FUNCTION match_site_pages (
  query_embedding vector(1536),
  match_count int DEFAULT 10,
  filter jsonb DEFAULT '{}'::jsonb,
  similarity_threshold float DEFAULT 0.7
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) AS similarity
  FROM site_pages
  WHERE metadata @> filter
    AND 1 - (site_pages.embedding <=> query_embedding) > similarity_threshold
  ORDER BY site_pages.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Drop all possible versions of hybrid_search
DROP FUNCTION IF EXISTS hybrid_search(text, vector(1536), int, jsonb, float, float) CASCADE;
DROP FUNCTION IF EXISTS hybrid_search(text, numeric[], int, jsonb, float, float) CASCADE;
DROP FUNCTION IF EXISTS hybrid_search(text, numeric[], integer, jsonb, double precision, double precision) CASCADE;
DROP FUNCTION IF EXISTS hybrid_search(text, numeric[], integer, unknown, numeric, numeric) CASCADE;

-- Create hybrid search function that combines vector similarity with text search
CREATE OR REPLACE FUNCTION hybrid_search (
  query_text text,
  query_embedding vector(1536),
  match_count int DEFAULT 10,
  filter jsonb DEFAULT '{}'::jsonb,
  similarity_threshold float DEFAULT 0.7,
  vector_weight float DEFAULT 0.7
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float,
  text_rank float,
  combined_score float
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) AS similarity,
    ts_rank_cd(to_tsvector('english', content), websearch_to_tsquery('english', query_text))::double precision AS text_rank,
    (vector_weight * (1 - (site_pages.embedding <=> query_embedding))) + 
    ((1 - vector_weight) * ts_rank_cd(to_tsvector('english', content), websearch_to_tsquery('english', query_text))::double precision) AS combined_score
  FROM site_pages
  WHERE metadata @> filter
    AND (1 - (site_pages.embedding <=> query_embedding) > similarity_threshold
         OR content ILIKE '%' || query_text || '%')
  ORDER BY combined_score DESC
  LIMIT match_count;
END;
$$;

-- Create a query transformation function that expands queries for better search results
CREATE OR REPLACE FUNCTION expand_query(
  query_text text
) RETURNS text 
LANGUAGE plpgsql
AS $$
DECLARE
  expanded_query text;
BEGIN
  -- Simple query expansion that adds synonyms and related terms
  -- In a real implementation, this would use a thesaurus or ML model
  expanded_query := query_text || ' ' || 
                   regexp_replace(query_text, '(database|db)', 'database db postgresql', 'gi') || ' ' ||
                   regexp_replace(query_text, '(vector|embedding)', 'vector embedding similarity', 'gi');
  
  RETURN expanded_query;
END;
$$;

-- Create a function for enhanced metadata filtering with multiple conditions
CREATE OR REPLACE FUNCTION filter_by_metadata(
  query_embedding vector(1536),
  match_count int DEFAULT 10,
  source_id text DEFAULT NULL,
  doc_type text DEFAULT NULL,
  min_date timestamp DEFAULT NULL,
  max_date timestamp DEFAULT NULL
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) AS similarity
  FROM site_pages
  WHERE (source_id IS NULL OR metadata->>'source_id' = source_id)
    AND (doc_type IS NULL OR metadata->>'type' = doc_type)
    AND (min_date IS NULL OR created_at >= min_date)
    AND (max_date IS NULL OR created_at <= max_date)
  ORDER BY site_pages.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create a function to aggregate chunks from the same document for comprehensive retrieval
CREATE OR REPLACE FUNCTION get_document_context(
  page_url text,
  context_size int DEFAULT 3
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  content text,
  is_current boolean
)
LANGUAGE plpgsql
AS $$
DECLARE
  base_chunk_number integer;
BEGIN
  -- Get the chunk number of the target page
  SELECT chunk_number INTO base_chunk_number
  FROM site_pages
  WHERE url = page_url
  LIMIT 1;
  
  -- Return the target chunk and surrounding context
  RETURN QUERY
  SELECT
    id,
    url,
    chunk_number,
    title,
    content,
    chunk_number = base_chunk_number AS is_current
  FROM site_pages
  WHERE url = page_url
    AND chunk_number BETWEEN (base_chunk_number - context_size) AND (base_chunk_number + context_size)
  ORDER BY chunk_number;
END;
$$;

-- Everything below is for Supabase security

-- Enable RLS on the documentation_sources table
ALTER TABLE documentation_sources ENABLE ROW LEVEL SECURITY;

-- Drop the existing policy if it exists
DROP POLICY IF EXISTS "Allow public read access to documentation_sources" ON documentation_sources;

-- Create a policy that allows anyone to read documentation_sources
CREATE POLICY "Allow public read access to documentation_sources"
  ON documentation_sources
  FOR SELECT
  TO public
  USING (true);

-- Enable RLS on the site_pages table
ALTER TABLE site_pages ENABLE ROW LEVEL SECURITY;

-- Drop the existing policy if it exists
DROP POLICY IF EXISTS "Allow public read access" ON site_pages;

-- Create a policy that allows anyone to read
CREATE POLICY "Allow public read access"
  ON site_pages
  FOR SELECT
  TO public
  USING (true);

-- Create or replace the trigger function to update documentation_sources aggregates
CREATE OR REPLACE FUNCTION update_source_counts() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE documentation_sources
        SET 
            pages_count = (
                SELECT COUNT(DISTINCT url)
                FROM site_pages 
                WHERE metadata->>'source_id' = NEW.metadata->>'source_id'
            ),
            chunks_count = (
                SELECT COUNT(*) 
                FROM site_pages 
                WHERE metadata->>'source_id' = NEW.metadata->>'source_id'
            )
        WHERE source_id = NEW.metadata->>'source_id';
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE documentation_sources
        SET 
            pages_count = (
                SELECT COUNT(DISTINCT url)
                FROM site_pages 
                WHERE metadata->>'source_id' = OLD.metadata->>'source_id'
            ),
            chunks_count = (
                SELECT COUNT(*) 
                FROM site_pages 
                WHERE metadata->>'source_id' = OLD.metadata->>'source_id'
            )
        WHERE source_id = OLD.metadata->>'source_id';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Drop any existing trigger (if present) and create the trigger to call the function
DROP TRIGGER IF EXISTS update_source_counts_trigger ON site_pages;

CREATE TRIGGER update_source_counts_trigger
AFTER INSERT OR DELETE ON site_pages
FOR EACH ROW
EXECUTE FUNCTION update_source_counts(); 