-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documentation_sources table
CREATE TABLE IF NOT EXISTS documentation_sources (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    base_url TEXT NOT NULL,
    sitemap_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    page_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    config JSONB DEFAULT '{}'::JSONB
);

-- Create site_pages table with vector support
CREATE TABLE IF NOT EXISTS site_pages (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES documentation_sources(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    embedding VECTOR(1536),  -- OpenAI embedding dimension
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_id, url, chunk_index)
);

-- Create index for faster vector similarity search
CREATE INDEX IF NOT EXISTS site_pages_embedding_idx ON site_pages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create index for faster source_id lookups
CREATE INDEX IF NOT EXISTS site_pages_source_id_idx ON site_pages(source_id);

-- Create index for faster URL lookups
CREATE INDEX IF NOT EXISTS site_pages_url_idx ON site_pages(url);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for documentation_sources
CREATE TRIGGER update_documentation_sources_updated_at
BEFORE UPDATE ON documentation_sources
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- Trigger for site_pages
CREATE TRIGGER update_site_pages_updated_at
BEFORE UPDATE ON site_pages
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- Function to update page and chunk counts
CREATE OR REPLACE FUNCTION update_source_counts()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE documentation_sources
        SET 
            page_count = (SELECT COUNT(DISTINCT url) FROM site_pages WHERE source_id = NEW.source_id),
            chunk_count = (SELECT COUNT(*) FROM site_pages WHERE source_id = NEW.source_id)
        WHERE id = NEW.source_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE documentation_sources
        SET 
            page_count = (SELECT COUNT(DISTINCT url) FROM site_pages WHERE source_id = OLD.source_id),
            chunk_count = (SELECT COUNT(*) FROM site_pages WHERE source_id = OLD.source_id)
        WHERE id = OLD.source_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger for updating counts
CREATE TRIGGER update_source_counts_trigger
AFTER INSERT OR DELETE ON site_pages
FOR EACH ROW
EXECUTE FUNCTION update_source_counts();