# Agentic RAG Local

An agentic RAG (Retrieval-Augmented Generation) system with local document crawling and processing capabilities. This project enables you to crawl documentation websites, process the content, store it in a vector database, and query it using natural language through an interactive Streamlit interface.

## Features

- **Document Crawling**: Automatically crawl documentation websites and extract content
- **Content Processing**: Process and chunk content for optimal retrieval
- **Vector Database Integration**: Store embeddings in PostgreSQL with pgvector for efficient similarity search
- **Agentic RAG**: Intelligent agent that retrieves relevant information and generates comprehensive answers
- **Interactive UI**: User-friendly Streamlit interface for querying the system
- **Database Compatibility**: Support for both psycopg2 and psycopg3 through a unified compatibility layer
- **Diagnostic Tools**: Database and system monitoring diagnostics to troubleshoot issues
- **Robust Content Extraction**: Multi-strategy approach to extract content from various HTML structures

## Project Structure

The project follows a modular organization:

### Source Code (`src/`)
- `src/api/`: FastAPI application and API routes
  - `app.py`: FastAPI application setup
  - `routes.py`: API route definitions
- `src/core/`: Core business logic
- `src/crawling/`: Document crawling and processing
  - `batch_processor.py`: Batch processing for document crawling
  - `docs_crawler.py`: Documentation crawler and processor
  - `enhanced_docs_crawler.py`: Enhanced version with improved sitemap parsing and async HTTP
- `src/db/`: Database operations
  - `async_schema.py`: Asynchronous database operations
  - `connection.py`: Database connection management
  - `db_utils.py`: Database driver compatibility for psycopg2/psycopg3
  - `schema.py`: Database schema definitions and PostgreSQL operations
- `src/models/`: Data models
  - `pydantic_models.py`: Data models for validation and serialization
- `src/rag/`: RAG implementation
  - `rag_expert.py`: RAG agent implementation
- `src/ui/`: User interface
  - `streamlit_app.py`: Streamlit web interface
  - `monitoring_ui.py`: UI components for system monitoring
- `src/utils/`: Utilities
  - `logging.py`: Logging and monitoring utilities
  - `enhanced_logging.py`: Advanced logging and monitoring system
  - `sanitization.py`: Input/output sanitization functions
  - `validation.py`: Data validation utilities
  - `task_monitoring.py`: Task tracking and monitoring

### Supporting Directories
- `data/`: SQL files and data storage
  - `site_pages.sql`: SQL scripts for site pages data
  - `vector_schema_v2.sql`: PostgreSQL database schema with pgvector extension
- `docs/`: Documentation
  - `api/`: API component documentation
    - `developer_guide.md`: Developer guide for the API component
  - `crawling/`: Crawling component documentation
    - `developer_guide.md`: Developer guide for the crawling component
    - `operations_guide.md`: Operations guide for the crawling component
  - `database/`: Database component documentation
    - `developer_guide.md`: Developer guide for the database component
    - `operations_guide.md`: Operations guide for the database component
  - `monitoring/`: Monitoring component documentation
    - `developer_guide.md`: Developer guide for the monitoring system
    - `operations_guide.md`: Operations guide for the monitoring system
  - `rag/`: RAG component documentation
    - `developer_guide.md`: Developer guide for the RAG component
    - `operations_guide.md`: Operations guide for the RAG component
  - `ui/`: UI component documentation
    - `developer_guide.md`: Developer guide for the UI component
  - `utils/`: Utilities documentation
    - `developer_guide.md`: Developer guide for the utilities component
  - `user/`: User documentation and guides
    - `monitoring_and_error_handling.md`: User guide for monitoring features
    - `development_progress_01.md`: Development progress tracking (part 1)
    - `development_progress_02.md`: Development progress tracking (part 2)
  - `mcp_readme_files/`: Documentation resources
  - `prompts/`: Prompt templates and examples
  - `rules/`: Project-specific rules and guidelines
  - `database_schema.md`: Database schema documentation
  - `github_setup.md`: GitHub setup guide
  - `installation.md`: Installation instructions
  - `user_guide.md`: General user guide
  - `developer_guide.md`: Overall developer guide
  - `operations_guide.md`: Overall operations guide
- `scripts/`: Utility scripts
  - `run_api.bat`: Script to run the API application
  - `run_ui.bat`: Script to run the UI application
  - `setup.bat`: Setup script for Windows users
  - `configure_postgresql.bat`: Script to configure PostgreSQL
  - `setup_database.py`: Script to set up the PostgreSQL database schema
  - `install_psycopg.py`: Script to install and verify psycopg packages
- `tests/`: Test suite
  - `integration/`: Integration tests
  - `unit/`: Unit tests
  - `test_database.py`: Database schema and function tests
  - `test_rag.py`: RAG functionality tests
- `postgresql_network_setup/`: PostgreSQL configuration
  - `README.md`: PostgreSQL setup documentation
  - Various `.bat` files for network configuration and management
- `check_database.py`: Database diagnostic tool for content inspection

## Setup and Installation

The project now includes a Makefile and setup scripts to simplify installation:

- For Unix/Linux/Mac: Use `make setup` to set up the project
- For Windows: Run `scripts/setup.bat`

## Prerequisites

- Python 3.9+
- PostgreSQL 14+ with pgvector extension installed
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
   
3. Install PostgreSQL drivers:
   ```bash
   python scripts/install_psycopg.py  # Installs and verifies psycopg packages
   ```

4. Install PostgreSQL and pgvector:
   - Install PostgreSQL 14 or later from https://www.postgresql.org/download/
   - Install pgvector extension:
     ```sql
     CREATE EXTENSION vector;
     ```
   - For Windows users, you can run the included setup script:
     ```bash
     configure_postgresql.bat
     ```

5. Create the database:
   - Create a new PostgreSQL database named `agentic_rag` (or choose your own name)
   - Run the setup script to create the necessary tables and functions:
     ```bash
     python setup_database.py
     ```

6. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your OpenAI API key
   - Configure your PostgreSQL connection details:
     ```
     POSTGRES_HOST=localhost
     POSTGRES_PORT=5432
     POSTGRES_DB=agentic_rag
     POSTGRES_USER=postgres
     POSTGRES_PASSWORD=your_password_here
     ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run src/ui/streamlit_app.py
   ```

2. In the web interface:
   - Click "Add New Documentation Source" to crawl a new site
   - Enter the name and sitemap URL for the documentation
   - Configure advanced options if needed
   - Click "Add and Crawl" to start the crawling process
   - Wait for the crawling to complete (this may take some time depending on the size of the documentation)

3. Ask questions about the documentation:
   - Type your query in the chat input box
   - The RAG agent will:
     - Retrieve relevant documentation chunks
     - Generate a comprehensive answer
     - Provide citations to the original documentation

## Advanced Configuration

### PostgreSQL Network Configuration

For network access to your PostgreSQL database:

1. Run the configuration script:
   ```bash
   configure_postgresql.bat
   ```

2. This script will:
   - Configure PostgreSQL to listen on all network interfaces
   - Allow connections from your local network
   - Set up Windows Firewall rules for PostgreSQL
   - Update your .env file with the correct connection details

For more details, see the README in the `postgresql_network_setup` directory.

### Crawler Configuration

You can configure the crawler behavior in the `.env` file:

```
DEFAULT_CHUNK_SIZE=5000           # Size of text chunks for processing
DEFAULT_MAX_CONCURRENT_CRAWLS=3   # Maximum concurrent web requests
DEFAULT_MAX_CONCURRENT_API_CALLS=5 # Maximum concurrent OpenAI API calls
```

Or through the web interface when adding a new documentation source.

### Database Diagnostics

The project includes a database diagnostic tool (`check_database.py`) for inspecting and troubleshooting database content:

```bash
python check_database.py
```

This tool provides:
- Documentation source information
- Pages and chunks count by source
- Sample content inspection
- Detection of common issues like:
  - Missing embeddings
  - Empty content
  - Source/page mismatches

## Development

### Adding New Documentation Sources

1. Create a new `CrawlConfig` for the documentation source
2. Run the crawler to fetch and process the content
3. The content will be automatically available in the UI for querying

### Enhanced Content Extraction

The crawler now includes multiple strategies for content extraction:

1. HTML2Text-based conversion (preserves structure and links)
2. Content area extraction (targets main content sections)
3. Fallback raw text extraction

This multi-strategy approach ensures that content can be extracted from a wide variety of documentation websites, regardless of their HTML structure.

### Customizing the RAG Agent

Modify `src/rag/rag_expert.py` to customize the agent's behavior, including:
- Prompt engineering
- Retrieval strategies
- Response generation

### Database Compatibility Layer

The project includes a compatibility layer in `src/db/db_utils.py` that supports both psycopg2 and psycopg3 (psycopg):

- Automatically detects which database driver is available
- Provides consistent async interfaces regardless of the underlying driver
- Allows the application to work with either synchronous (psycopg2) or asynchronous (psycopg3) database access

This makes the application more portable and resilient to different deployment environments.

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

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.