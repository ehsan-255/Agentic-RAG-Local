# Installation Guide

This guide will walk you through the process of setting up the Agentic RAG system on your local machine.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.11 or higher
- A Supabase account with a project set up
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
   SUPABASE_URL=your_supabase_url_here
   SUPABASE_SERVICE_KEY=your_supabase_service_key_here
   ```

## Step 5: Set Up the Database

1. Log in to your Supabase dashboard
2. Navigate to the SQL Editor
3. Copy the contents of `data/site_pages.sql`
4. Paste into the SQL Editor and run the queries

This will create the necessary tables and functions for the vector search.

## Step 6: Run the Application

Start the Streamlit application:

```bash
streamlit run src/ui/streamlit_app.py
```

The application should now be running at http://localhost:8501

## Troubleshooting

If you encounter any issues:

1. Ensure all environment variables are correctly set
2. Check that your Supabase and OpenAI API keys are valid
3. Verify that the SQL setup was completed successfully
4. Make sure you're using Python 3.11 or higher 