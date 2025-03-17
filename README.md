# Agentic RAG Local

An agentic RAG (Retrieval-Augmented Generation) system with local document crawling and processing capabilities. This project enables you to crawl documentation websites, process the content, store it in a vector database, and query it using natural language through an interactive Streamlit interface.

## Features

- **Document Crawling**: Automatically crawl documentation websites and extract content
- **Content Processing**: Process and chunk content for optimal retrieval
- **Vector Database Integration**: Store embeddings in Supabase for efficient similarity search
- **Agentic RAG**: Intelligent agent that retrieves relevant information and generates comprehensive answers
- **Interactive UI**: User-friendly Streamlit interface for querying the system

## Project Structure

- `src/crawling/docs_crawler.py`: Documentation crawler and processor
- `src/rag/rag_expert.py`: RAG agent implementation
- `src/ui/streamlit_app.py`: Web interface
- `src/db/schema.py`: Database operations
- `data/site_pages.sql`: Database setup commands
- `postgresql_network_setup/`: Scripts for configuring PostgreSQL network access
- `requirements.txt`: Project dependencies
- `project_file_map.md`: Comprehensive documentation of the project file structure

For a complete view of the project structure, refer to `project_file_map.md`.

## Prerequisites

- Python 3.9+
- Supabase account
- OpenAI API key

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ehsan-255/Agentic-RAG-Local.git
   cd Agentic-RAG-Local
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up your Supabase database:
   - Create a new project in Supabase
   - Run the SQL commands in `data/site_pages.sql` to set up the necessary tables
   
   Or set up a local PostgreSQL database:
   - Ensure PostgreSQL is installed with the pgvector extension
   - Run the setup script to create the schema:
     ```bash
     python setup_database.py
     ```
   - The setup script will:
     - Check if pgvector is installed
     - Create the database tables, indexes, and functions
     - Configure optimal settings for vector similarity search

4. Configure PostgreSQL for network access (if using a local PostgreSQL server):
   - Run the configuration script:
     ```bash
     configure_postgresql.bat
     ```
   - This script will:
     - Configure PostgreSQL to listen on all network interfaces
     - Allow connections from your local network
     - Set up Windows Firewall rules for PostgreSQL
     - Update your .env file with the correct connection details
   - For more details, see the README in the `postgresql_network_setup` directory

5. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your OpenAI API key and Supabase credentials

## Usage

### Crawling Documentation

To crawl a documentation website:

```python
from src.crawling.docs_crawler import crawl_documentation, CrawlConfig

config = CrawlConfig(
    site_name="example_docs",
    base_url="https://docs.example.com",
    allowed_domains=["docs.example.com"],
    start_urls=["https://docs.example.com/getting-started"],
    sitemap_urls=["https://docs.example.com/sitemap.xml"],
)

crawl_documentation(config)
```

### Running the Web Interface

Start the Streamlit app:

```bash
streamlit run src/ui/streamlit_app.py
```

### Cleaning Up Temporary Files

To remove temporary files created during PostgreSQL configuration:

```bash
cleanup_temp_files.bat
```

## Development

### Adding New Documentation Sources

1. Create a new `CrawlConfig` for the documentation source
2. Run the crawler to fetch and process the content
3. The content will be automatically available in the UI for querying

### Customizing the RAG Agent

Modify `src/rag/rag_expert.py` to customize the agent's behavior, including:
- Prompt engineering
- Retrieval strategies
- Response generation

### Advanced Vector Search Features

The system includes several advanced search capabilities:

- **Vector Similarity Search**: Find semantically similar content using OpenAI embeddings
- **Hybrid Search**: Combine vector similarity with traditional text search for better results
- **Metadata Filtering**: Filter search results by source, document type, date, etc.
- **Document Context**: Retrieve surrounding chunks from the same document for better context
- **Query Expansion**: Automatically expand queries with related terms for better recall

To utilize these features, use the database functions in `src/db/schema.py`:

```python
from src.db.schema import match_site_pages, hybrid_search, filter_by_metadata, get_document_context

# Vector similarity search
results = match_site_pages(query_embedding, match_count=5)

# Hybrid search combining vector and text search
results = hybrid_search(query_text, query_embedding, vector_weight=0.7)

# Search with metadata filtering
results = filter_by_metadata(query_embedding, source_id="example_docs", doc_type="tutorial")

# Get surrounding context for a document
context = get_document_context(page_url, context_size=3)
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
