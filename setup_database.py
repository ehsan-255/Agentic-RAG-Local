#!/usr/bin/env python
"""
Database Setup Script for Agentic RAG System

This script sets up the PostgreSQL database with the required schema and extensions
for the Agentic RAG system. It creates the tables, indexes, and functions needed
for vector similarity search.

Usage:
    python setup_database.py

Requirements:
    - PostgreSQL with pgvector extension installed
    - Environment variables configured in .env file
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from src.db.schema import setup_database, check_pgvector_extension

def main():
    """Main function to set up the database."""
    parser = argparse.ArgumentParser(description="Set up the database for Agentic RAG system")
    parser.add_argument("--schema", choices=["v1", "v2"], default="v2",
                        help="Schema version to use (v1=basic, v2=enhanced)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check if pgvector is installed, don't create schema")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    print("Checking PostgreSQL and pgvector...")
    
    # Check if pgvector is installed
    if not check_pgvector_extension():
        print("ERROR: pgvector extension is not installed in PostgreSQL.")
        print("Please install the pgvector extension before continuing.")
        print("Instructions: https://github.com/pgvector/pgvector#installation")
        sys.exit(1)
    
    print("✅ pgvector extension is installed and available.")
    
    if args.check_only:
        print("Check-only mode, skipping schema creation.")
        sys.exit(0)
    
    # Set up database schema
    print(f"Setting up database with schema version {args.schema}...")
    
    if setup_database():
        print("✅ Database schema created successfully!")
        print("\nThe database is now ready for use with the Agentic RAG system.")
        print("You can start crawling documentation and using the RAG system.")
    else:
        print("❌ Failed to set up database schema.")
        print("Please check the logs for more information.")
        sys.exit(1)

if __name__ == "__main__":
    main() 