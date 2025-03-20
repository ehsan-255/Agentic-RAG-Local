# Agentic RAG Local Project File Map

## Directory Tree Structure
```
Agentic_RAG_Local/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── routes.py
│   ├── core/
│   │   └── __init__.py
│   ├── crawling/
│   │   ├── __init__.py
│   │   ├── batch_processor.py
│   │   ├── docs_crawler.py
│   │   └── enhanced_docs_crawler.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── async_schema.py
│   │   ├── connection.py
│   │   ├── db_utils.py
│   │   └── schema.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pydantic_models.py
│   ├── rag/
│   │   ├── __init__.py
│   │   └── rag_expert.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── streamlit_app.py
│   │   └── monitoring_ui.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── enhanced_logging.py
│   │   ├── logging.py
│   │   ├── sanitization.py
│   │   ├── task_monitoring.py
│   │   └── validation.py
│   ├── __init__.py
│   └── config.py
├── data/
│   ├── site_pages.sql
│   └── vector_schema_v2.sql
├── docs/
│   ├── user/
│   │   ├── development_progress_01.md
│   │   ├── development_progress_02.md
│   │   └── monitoring_and_error_handling.md
│   ├── mcp_readme_files/
│   ├── prompts/
│   ├── rules/
│   ├── temp/
│   ├── database_schema.md
│   ├── github_setup.md
│   └── installation.md
├── scripts/
│   ├── run_api.bat
│   ├── run_ui.bat
│   ├── setup.bat
│   ├── configure_postgresql.bat
│   ├── setup_database.py
│   └── install_psycopg.py
├── tests/
│   ├── integration/
│   │   └── __init__.py
│   ├── unit/
│   │   └── __init__.py
│   ├── __init__.py
│   ├── test_database.py
│   └── test_rag.py
├── postgresql_network_setup/
│   ├── README.md
│   ├── check_firewall.bat
│   ├── cleanup.bat
│   ├── configure_postgresql_network.bat
│   ├── restart_postgresql.bat
│   ├── test_postgresql_connection.bat
│   ├── update_env_file.bat
│   ├── update_pg_hba.bat
│   └── update_postgresql_conf.bat
├── check_database.py
├── check_schema.py
├── .cursor/
│   └── rules/
│       ├── beautifulsoup4-best-practices.mdc
│       ├── black-best-practices.mdc
│       ├── mkdocs-best-practices.mdc
│       ├── mypy-best-practices.mdc
│       ├── openai-best-practices.mdc
│       ├── pgvector-best-practices.mdc
│       ├── psycopg-best-practices.mdc
│       ├── pydantic-best-practices.mdc
│       ├── pytest-best-practices.mdc
│       ├── requests-best-practices.mdc
│       └── streamlit-best-practices.mdc
├── .vscode/
│   └── settings.json
├── venv/
├── Makefile
├── scripts.old
├── .env
├── .env.example
├── .gitignore
├── .cursorrules
├── README.md
├── requirements.txt
├── setup.py
└── project_file_map.md
```

## Root Directory
- `.env` - Environment variables configuration
- `.env.example` - Example environment file
- `.gitignore` - Git ignore rules
- `.cursorrules` - Project-specific guidelines for Cursor AI
- `Makefile` - Common project commands (setup, test, clean, format, run)
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `setup.py` - Python package setup
- `scripts.old` - Backup of previous scripts file
- `project_file_map.md` - This file - documentation of project structure
- `check_database.py` - Database diagnostic tool to analyze content storage
- `check_schema.py` - Utility to inspect database schema

## Source Code (`src/`)
### API Module (`src/api/`)
- `__init__.py` - Package initialization
- `app.py` - FastAPI application setup
- `routes.py` - API route definitions

### Core Module (`src/core/`)
- `__init__.py` - Package initialization for core business logic

### Crawling Module (`src/crawling/`)
- `__init__.py` - Package initialization
- `batch_processor.py` - Batch processing for document crawling
- `docs_crawler.py` - Documentation crawler and processor
- `enhanced_docs_crawler.py` - Enhanced version of the document crawler with improved sitemap handling

### Database Module (`src/db/`)
- `__init__.py` - Package initialization
- `async_schema.py` - Asynchronous database operations
- `connection.py` - Database connection management
- `db_utils.py` - Database driver compatibility utilities supporting both psycopg2 and psycopg3
- `schema.py` - Database schema definitions and PostgreSQL operations

### Models Module (`src/models/`)
- `__init__.py` - Package initialization
- `pydantic_models.py` - Data models for validation and serialization

### RAG Implementation (`src/rag/`)
- `__init__.py` - Package initialization
- `rag_expert.py` - RAG agent implementation

### User Interface (`src/ui/`)
- `__init__.py` - Package initialization
- `streamlit_app.py` - Streamlit web interface
- `monitoring_ui.py` - UI components for monitoring system status

### Utilities (`src/utils/`)
- `__init__.py` - Package initialization
- `enhanced_logging.py` - Advanced logging and monitoring system
- `logging.py` - Basic logging utilities
- `sanitization.py` - Input/output sanitization functions
- `task_monitoring.py` - Task tracking and monitoring
- `validation.py` - Data validation utilities

## Data Files (`data/`)
- `site_pages.sql` - SQL scripts for site pages data
- `vector_schema_v2.sql` - PostgreSQL database schema with pgvector extension

## Documentation (`docs/`)
### User Documentation (`docs/user/`)
- `development_progress_01.md` - Development progress tracking (part 1)
- `development_progress_02.md` - Development progress tracking (part 2)
- `monitoring_and_error_handling.md` - User guide for monitoring features

### Other Documentation Folders
- `mcp_readme_files/` - Documentation resources
- `prompts/` - Prompt templates and examples
- `rules/` - Project-specific rules and guidelines
- `temp/` - Temporary documentation files
- `database_schema.md` - Database schema documentation
- `github_setup.md` - GitHub setup guide
- `installation.md` - Installation instructions

## Scripts (`scripts/`)
- `run_api.bat` - Script to run the API application
- `run_ui.bat` - Script to run the UI application
- `setup.bat` - Setup script for Windows users
- `configure_postgresql.bat` - Script to configure PostgreSQL
- `setup_database.py` - Script to set up the PostgreSQL database schema
- `install_psycopg.py` - Script to install and verify psycopg packages

## Tests (`tests/`)
- `integration/` - Integration tests
  - `__init__.py` - Package initialization
- `unit/` - Unit tests
  - `__init__.py` - Package initialization
- `__init__.py` - Package initialization file
- `test_database.py` - Database schema and function tests
- `test_rag.py` - RAG functionality tests

## PostgreSQL Network Setup (`postgresql_network_setup/`)
- `README.md` - PostgreSQL setup documentation
- `check_firewall.bat` - Checks Windows Firewall for PostgreSQL
- `cleanup.bat` - Removes temporary files
- `configure_postgresql_network.bat` - Main PostgreSQL configuration script
- `restart_postgresql.bat` - Restarts PostgreSQL service
- `test_postgresql_connection.bat` - Tests PostgreSQL connection
- `update_env_file.bat` - Updates .env with PostgreSQL settings
- `update_pg_hba.bat` - Updates pg_hba.conf for network access
- `update_postgresql_conf.bat` - Updates postgresql.conf settings

## VS Code Configuration (`.vscode/`)
- `settings.json` - VS Code editor settings

## Cursor Configuration (`.cursor/`)
### Development Rules (`.cursor/rules/`)
- `beautifulsoup4-best-practices.mdc` - Best practices for web scraping with BeautifulSoup
- `black-best-practices.mdc` - Best practices for using Black for code formatting
- `mkdocs-best-practices.mdc` - Best practices for using MkDocs for documentation
- `mypy-best-practices.mdc` - Best practices for using Mypy for static type checking
- `openai-best-practices.mdc` - Best practices for using OpenAI API
- `pgvector-best-practices.mdc` - Best practices for using Pgvector with PostgreSQL
- `psycopg-best-practices.mdc` - Best practices for using Psycopg with PostgreSQL
- `pydantic-best-practices.mdc` - Best practices for using Pydantic for data validation
- `pytest-best-practices.mdc` - Best practices for writing tests with Pytest
- `requests-best-practices.mdc` - Best practices for making HTTP requests with Requests
- `streamlit-best-practices.mdc` - Best practices for Streamlit applications

## Virtual Environment (`venv/`)
- Python virtual environment directory with installed dependencies

## IMPORTANT NOTES
- The system now includes a dedicated API module with FastAPI
- The project is organized more modularly with separate directories for different components
- Developer setup is simplified with Makefile and batch scripts
- The database module supports both psycopg2 and psycopg3 through a compatibility layer
- The crawler has been enhanced with improved sitemap handling capabilities
- Diagnostic tools have been added for database and system monitoring
- Multiple content extraction strategies ensure robust handling of different HTML structures
- When implementing structure changes, ensure all import statements are updated
- Test thoroughly after any structural changes
- Keep this file map updated whenever new files are added or existing files are modified/moved
- When updating this file, ensure both the detailed list and the tree structure are kept in sync