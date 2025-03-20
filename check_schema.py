#!/usr/bin/env python
"""Simple script to check database schema"""

import asyncio
import platform
import psycopg
from src.config import config

# Fix for Windows asyncio
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def check_schema():
    """Check database schema"""
    conn_string = config.get_db_connection_string()
    
    async with await psycopg.AsyncConnection.connect(conn_string) as conn:
        # Check site_pages table
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'site_pages'
                ORDER BY ordinal_position
            """)
            columns = await cur.fetchall()
            
            print("Columns in site_pages table:")
            for col_name, data_type in columns:
                print(f"  - {col_name} ({data_type})")
        
        # Check documentation_sources table
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'documentation_sources'
                ORDER BY ordinal_position
            """)
            columns = await cur.fetchall()
            
            print("\nColumns in documentation_sources table:")
            for col_name, data_type in columns:
                print(f"  - {col_name} ({data_type})")

if __name__ == "__main__":
    asyncio.run(check_schema()) 