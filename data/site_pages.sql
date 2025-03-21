-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation sources table
create table documentation_sources (
    id bigserial primary key,
    name varchar not null,
    source_id varchar not null unique,
    base_url varchar not null,
    configuration jsonb not null default '{}'::jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    last_crawled_at timestamp with time zone,
    pages_count integer not null default 0,
    chunks_count integer not null default 0
);

-- Create the documentation chunks table
create table site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
create index on site_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_site_pages_metadata on site_pages using gin (metadata);

-- Create an index for fast lookups by source_id
create index idx_site_pages_source_id on site_pages ((metadata->>'source_id'));

-- Create a function to increment pages count
create or replace function increment_pages_count(source_id_param text)
returns integer as $$
declare
    current_count integer;
begin
    select pages_count into current_count from documentation_sources where source_id = source_id_param;
    return current_count + 1;
end;
$$ language plpgsql;

-- Create a function to increment chunks count
create or replace function increment_chunks_count(source_id_param text, increment_by integer)
returns integer as $$
declare
    current_count integer;
begin
    select chunks_count into current_count from documentation_sources where source_id = source_id_param;
    return current_count + increment_by;
end;
$$ language plpgsql;

-- Create a function to search for documentation chunks
CREATE OR REPLACE FUNCTION match_site_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where metadata @> filter
  order by site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Everything above will work for any PostgreSQL database. The below commands are for Supabase security

-- Enable RLS on the documentation_sources table
alter table documentation_sources enable row level security;

-- Create a policy that allows anyone to read documentation_sources
create policy "Allow public read access to documentation_sources"
  on documentation_sources
  for select
  to public
  using (true);

-- Enable RLS on the site_pages table
alter table site_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on site_pages
  for select
  to public
  using (true);

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
