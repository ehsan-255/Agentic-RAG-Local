# Installation Guide

This guide will walk you through the process of setting up the Agentic RAG system on your local machine.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.9 or higher
- PostgreSQL 14 or higher with pgvector extension installed
- An OpenAI API key

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-rag-local.git
cd agentic-rag-local
```

## Step 2: Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

## Step 4: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your API keys and configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=agentic_rag
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password_here
   ```

## Step 5: Set Up PostgreSQL

1. Install PostgreSQL 14 or higher from https://www.postgresql.org/download/
2. Install the pgvector extension:
   ```sql
   CREATE EXTENSION vector;
   ```
3. Create a new database named `agentic_rag` (or use the name you specified in your .env file)
4. Run the setup script to create the necessary tables and functions:
   ```bash
   python setup_database.py
   ```

## Step 6: Configure PostgreSQL for Network Access (Optional)

If you need to access PostgreSQL from other machines on your network:

```bash
configure_postgresql.bat
```

This script will:
- Configure PostgreSQL to listen on all network interfaces
- Allow connections from your local network
- Set up Windows Firewall rules for PostgreSQL

## Step 7: Run the Application

Start the Streamlit application:

```bash
streamlit run src/ui/streamlit_app.py
```

The application should now be running at http://localhost:8501

## Troubleshooting

If you encounter any issues:

1. Ensure all environment variables are correctly set
2. Check that your OpenAI API key is valid
3. Verify that PostgreSQL is running and the pgvector extension is installed
4. Make sure the database was created successfully
5. Check the logs for any error messages 