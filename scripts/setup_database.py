#!/usr/bin/env python
"""
Database setup script for the Agentic RAG system.

This script sets up the PostgreSQL database with the pgvector extension
and creates the necessary tables and functions for the RAG system.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the database setup function
from src.db.schema import setup_database, check_pgvector_extension, check_tables_exist, create_schema_from_file

def main():
    """Set up the database for the Agentic RAG system."""
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Set up the database for the Agentic RAG system.')
    parser.add_argument('--force', action='store_true', help='Force recreation of the database schema')
    args = parser.parse_args()
    
    print("Setting up the database...")
    
    # Check if pgvector extension is installed
    if not check_pgvector_extension():
        print("Error: pgvector extension is not installed.")
        print("Please install it with: CREATE EXTENSION vector;")
        sys.exit(1)
    
    # Check if tables exist
    tables = check_tables_exist()
    all_tables_exist = all(tables.values())
    
    if all_tables_exist and not args.force:
        print("Database schema already exists. Use --force to recreate it.")
        print("The following tables are available:")
        for table, exists in tables.items():
            print(f"- {table}: {'✓' if exists else '✗'}")
        sys.exit(0)
    
    # Create schema from file
    schema_path = os.path.join(os.path.dirname(__file__), "data", "vector_schema_v2.sql")
    if not os.path.exists(schema_path):
        print(f"Error: Schema file not found at {schema_path}")
        sys.exit(1)
    
    print(f"Creating schema from {schema_path}...")
    success = create_schema_from_file(schema_path)
    
    if success:
        print("Database setup completed successfully!")
        print("The following tables and functions have been created:")
        print("- documentation_sources: Stores information about documentation sources")
        print("- site_pages: Stores documentation pages and their vector embeddings")
        print("- match_site_pages: Function for vector similarity search")
        print("- hybrid_search: Function for combined vector and text search")
        print("- filter_by_metadata: Function for filtering by metadata")
        print("- get_document_context: Function for retrieving document context")
        print("\nYou can now run the Streamlit app with: streamlit run src/ui/streamlit_app.py")
    else:
        print("Database setup failed! Check the logs for more information.")
        print("Make sure PostgreSQL is running and the pgvector extension is installed.")
        print("You can install pgvector with: CREATE EXTENSION vector;")
        sys.exit(1)

if __name__ == "__main__":
    main() 