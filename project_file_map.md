# Agentic RAG Local Project File Map

## Directory Tree Structure
```
Agentic_RAG_Local/
├── src/
│   ├── db/
│   │   └── schema.py
│   ├── ui/
│   │   └── streamlit_app.py
│   ├── crawling/
│   │   └── docs_crawler.py
│   └── rag/
│       └── rag_expert.py
├── data/
│   ├── site_pages.sql
│   └── vector_schema_v2.sql
├── docs/
│   ├── installation.md
│   ├── github_setup.md
│   └── database_schema.md
├── tests/
│   ├── __init__.py
│   └── test_rag.py
├── postgresql_network_setup/
│   ├── README.md
│   ├── cleanup.bat
│   ├── check_firewall.bat
│   ├── configure_postgresql_network.bat
│   ├── restart_postgresql.bat
│   ├── test_postgresql_connection.bat
│   ├── update_env_file.bat
│   ├── update_pg_hba.bat
│   └── update_postgresql_conf.bat
├── venv/
├── .env
├── .env.example
├── .gitignore
├── .cursorrules
├── README.md
├── requirements.txt
├── setup.py
├── DO_NOT_DELETE_prompt.txt
├── cleanup_temp_files.bat
├── configure_postgresql.bat
├── setup_database.py
└── project_file_map.md
```

## Root Directory
- `.env` - Environment variables configuration (1.3KB)
- `.env.example` - Example environment file (571B)
- `.gitignore` - Git ignore rules (1.2KB)
- `.cursorrules` - Project-specific guidelines for Cursor AI (4.0KB)
- `README.md` - Project documentation (7.3KB)
- `requirements.txt` - Python dependencies (2.6KB)
- `setup.py` - Python package setup (372B)
- `DO_NOT_DELETE_prompt.txt` - Important project prompt file (17KB) - DO NOT DELETE
- `cleanup_temp_files.bat` - Script to remove temporary files (151B)
- `configure_postgresql.bat` - Script to configure PostgreSQL (184B)
- `setup_database.py` - Script to set up the PostgreSQL database schema (1.8KB)
- `project_file_map.md` - This file - documentation of project structure

## Source Code (`src/`)
### Database Module (`src/db/`)
- `schema.py` - Database schema definitions (3.5KB)

### User Interface (`src/ui/`)
- `streamlit_app.py` - Streamlit web interface (18KB)

### Web Crawling (`src/crawling/`)
- `docs_crawler.py` - Documentation crawler and processor (16KB)

### RAG Implementation (`src/rag/`)
- `rag_expert.py` - RAG agent implementation (8.2KB)

## Data Files (`data/`)
- `site_pages.sql` - Database setup SQL commands (3.6KB)
- `vector_schema_v2.sql` - Enhanced database schema with vector search (7.0KB)

## Documentation (`docs/`)
- `installation.md` - Installation instructions (1.9KB)
- `github_setup.md` - GitHub setup guide (4.3KB)
- `database_schema.md` - Database schema documentation (6.5KB)

## Tests (`tests/`)
- `__init__.py` - Package initialization file (1B)
- `test_rag.py` - RAG functionality tests (1.8KB)
- `test_database.py` - Database schema and function tests (5.8KB)

## PostgreSQL Network Setup (`postgresql_network_setup/`)
- `README.md` - PostgreSQL setup documentation (2.3KB)
- `cleanup.bat` - Removes temporary files (335B)
- `check_firewall.bat` - Checks Windows Firewall for PostgreSQL (1KB)
- `configure_postgresql_network.bat` - Main PostgreSQL configuration script (1.3KB)
- `restart_postgresql.bat` - Restarts PostgreSQL service (603B)
- `test_postgresql_connection.bat` - Tests PostgreSQL connection (1.1KB)
- `update_env_file.bat` - Updates .env with PostgreSQL settings (1KB)
- `update_pg_hba.bat` - Updates pg_hba.conf for network access (985B)
- `update_postgresql_conf.bat` - Updates postgresql.conf settings (1.6KB)

## Virtual Environment (`venv/`)
- Python virtual environment directory with installed dependencies

## IMPORTANT NOTES
- The file `DO_NOT_DELETE_prompt.txt` contains important project information and should NEVER be deleted
- All scripts in `postgresql_network_setup/` are accessed via shortcuts in the root directory
- Temporary files generated during script execution will be automatically cleaned by `cleanup_temp_files.bat`
- This file map should be updated whenever new files are added or existing files are modified/moved
- When updating this file, ensure both the detailed list and the tree structure are kept in sync 