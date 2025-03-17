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
- `requirements.txt`: Project dependencies

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

4. Configure environment variables:
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

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.