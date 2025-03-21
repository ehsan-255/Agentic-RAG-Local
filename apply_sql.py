#!/usr/bin/env python
"""
Script to apply SQL file to the database
"""

import os
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def apply_sql_file(sql_file_path):
    """Apply SQL file to the database."""
    print(f"Applying SQL file: {sql_file_path}")
    
    # Check if file exists
    if not os.path.exists(sql_file_path):
        print(f"Error: SQL file not found at {sql_file_path}")
        return False
    
    # Read SQL file content
    with open(sql_file_path, 'r') as f:
        sql_content = f.read()
    
    print("SQL content loaded successfully")
    
    # Try to import modules
    try:
        print("Importing database connection modules...")
        from src.db.connection import get_connection, release_connection
        print("Modules imported successfully")
        
        # Get connection
        conn = None
        try:
            print("Establishing database connection...")
            conn = get_connection()
            print("Database connection established")
            
            # Create cursor and execute SQL
            cur = conn.cursor()
            print("Executing SQL...")
            cur.execute(sql_content)
            conn.commit()
            print("SQL execution successful and changes committed")
            return True
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error applying SQL: {e}")
            traceback.print_exc()
            return False
        finally:
            if conn:
                release_connection(conn)
                print("Database connection released")
    except Exception as e:
        print(f"Error importing modules: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Get the absolute path of the SQL file
    sql_file = Path(__file__).parent / "data" / "site_pages.sql"
    
    # Print debug info
    print(f"Current directory: {os.getcwd()}")
    print(f"SQL file path: {sql_file}")
    print(f"SQL file exists: {os.path.exists(sql_file)}")
    
    # List files in data directory
    data_dir = Path(__file__).parent / "data"
    print(f"Files in data directory: {[f.name for f in data_dir.iterdir() if f.is_file()]}")
    
    # Apply SQL
    result = apply_sql_file(str(sql_file))
    if result:
        print("Successfully applied SQL file")
    else:
        print("Failed to apply SQL file") 