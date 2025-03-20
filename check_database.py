#!/usr/bin/env python
"""
Database check utility to examine crawled data in the database.
This script helps diagnose data storage and visibility issues.
"""

import os
import sys
import asyncio
import json
import platform
from datetime import datetime

# Fix for Windows asyncio event loop
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import configuration first
from src.config import config

# Then import psycopg
try:
    import psycopg
    from psycopg.rows import dict_row
    USING_PSYCOPG3 = True
    print("Using psycopg3 for database connection")
except ImportError:
    import psycopg2
    from psycopg2.extras import DictCursor
    USING_PSYCOPG3 = False
    print("Using psycopg2 for database connection")

async def check_database():
    """Connect to the database and check for stored pages and chunks."""
    # Print header
    print("\n===== Database Content Check =====")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect to the database
    print(f"\nConnecting to database: {config.DB_NAME} on {config.DB_HOST}...")
    connection_string = config.get_db_connection_string()
    
    try:
        # Connect to the database - handle both psycopg3 and psycopg2
        if USING_PSYCOPG3:
            async with await psycopg.AsyncConnection.connect(connection_string) as conn:
                print("Connection successful!")
                await _run_database_checks(conn, dict_row)
        else:
            # Psycopg2 fallback (synchronous)
            import psycopg2.extras
            conn = psycopg2.connect(connection_string)
            conn.autocommit = True
            print("Connection successful!")
            _run_sync_database_checks(conn, psycopg2.extras.DictCursor)
            conn.close()
            
        return True
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

async def _run_database_checks(conn, dict_factory):
    """Run database checks with async connection."""
    # Check tables 
    print("\n1. Checking table existence...")
    async with conn.cursor() as cur:
        await cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = await cur.fetchall()
        
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
    
    # Check documentation sources
    print("\n2. Checking documentation sources...")
    async with conn.cursor(row_factory=dict_factory) as cur:
        await cur.execute("SELECT * FROM documentation_sources")
        sources = await cur.fetchall()
        
        if not sources:
            print("No documentation sources found.")
        else:
            print(f"Found {len(sources)} documentation sources:")
            for source in sources:
                print(f"  - {source['name']} (ID: {source['source_id']})")
                print(f"    URL: {source['base_url']}")
                print(f"    Last crawled: {source['last_crawled_at']}")
                print(f"    Created: {source['created_at']}")
                print(f"    Pages count: {source['pages_count'] or 0}")
                print(f"    Chunks count: {source['chunks_count'] or 0}")
    
    # Check site pages
    print("\n3. Checking stored pages...")
    total_pages = 0
    
    async with conn.cursor() as cur:
        # Count pages by source ID from metadata
        await cur.execute("""
            SELECT metadata->>'source_id' as source_id, COUNT(*) as pages_count 
            FROM site_pages 
            GROUP BY metadata->>'source_id'
        """)
        pages_by_source = await cur.fetchall()
        
        if not pages_by_source:
            print("No pages found in the database.")
        else:
            print("Pages by source:")
            for source_id, count in pages_by_source:
                print(f"  - Source {source_id}: {count} pages")
        
        # Total page count
        await cur.execute("SELECT COUNT(*) FROM site_pages")
        result = await cur.fetchone()
        total_pages = result[0] if result else 0
        print(f"\nTotal pages in database: {total_pages}")
        
        # Sample a few pages
        if total_pages > 0:
            await cur.execute("""
                SELECT id, url, title, chunk_number, length(content) as content_length 
                FROM (
                    SELECT 
                        id, 
                        url, 
                        title, 
                        chunk_number,
                        content,
                        ROW_NUMBER() OVER (PARTITION BY url ORDER BY chunk_number) as rn
                    FROM site_pages
                ) sub
                WHERE rn = 1
                ORDER BY id
                LIMIT 5
            """)
            sample_pages = await cur.fetchall()
            
            print("\nSample pages (first chunk of each URL):")
            for page in sample_pages:
                print(f"  - {page[1]}")
                print(f"    Title: {page[2]}")
                print(f"    Content length: {page[4]} characters")
    
    # Check for problems
    print("\n4. Checking for potential issues...")
    issues_found = False
    
    # Check 1: No pages with both source_id and pages
    if sources and total_pages == 0:
        print("❗ ISSUE: Sources exist but no pages are stored.")
        issues_found = True
    
    # Check 2: Pages with empty content
    if total_pages > 0:
        async with conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM site_pages WHERE length(content) < 10")
            result = await cur.fetchone()
            empty_pages = result[0] if result else 0
            
            if empty_pages > 0:
                print(f"❗ ISSUE: Found {empty_pages} pages with very little content (<10 chars).")
                issues_found = True
        
        # Check 3: Pages with missing embeddings
        async with conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM site_pages WHERE embedding IS NULL")
            result = await cur.fetchone()
            no_embeddings = result[0] if result else 0
            
            if no_embeddings > 0:
                print(f"❗ ISSUE: Found {no_embeddings} pages with missing embeddings.")
                issues_found = True
    
    if not issues_found:
        print("✓ No major issues detected.")
    
    print("\n===== Check Complete =====")

def _run_sync_database_checks(conn, dict_cursor):
    """Run database checks with synchronous connection."""
    # Check tables 
    print("\n1. Checking table existence...")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()
        
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
    
    # Check documentation sources
    print("\n2. Checking documentation sources...")
    with conn.cursor(cursor_factory=dict_cursor) as cur:
        cur.execute("SELECT * FROM documentation_sources")
        sources = cur.fetchall()
        
        if not sources:
            print("No documentation sources found.")
        else:
            print(f"Found {len(sources)} documentation sources:")
            for source in sources:
                print(f"  - {source['name']} (ID: {source['source_id']})")
                print(f"    URL: {source['base_url']}")
                print(f"    Last crawled: {source['last_crawled_at']}")
                print(f"    Created: {source['created_at']}")
                print(f"    Pages count: {source['pages_count'] or 0}")
                print(f"    Chunks count: {source['chunks_count'] or 0}")
    
    # Check site pages
    print("\n3. Checking stored pages...")
    total_pages = 0
    
    with conn.cursor() as cur:
        # Count pages by source ID from metadata
        cur.execute("""
            SELECT metadata->>'source_id' as source_id, COUNT(*) as pages_count 
            FROM site_pages 
            GROUP BY metadata->>'source_id'
        """)
        pages_by_source = cur.fetchall()
        
        if not pages_by_source:
            print("No pages found in the database.")
        else:
            print("Pages by source:")
            for source_id, count in pages_by_source:
                print(f"  - Source {source_id}: {count} pages")
        
        # Total page count
        cur.execute("SELECT COUNT(*) FROM site_pages")
        result = cur.fetchone()
        total_pages = result[0] if result else 0
        print(f"\nTotal pages in database: {total_pages}")
        
        # Sample a few pages
        if total_pages > 0:
            cur.execute("""
                SELECT id, url, title, chunk_number, length(content) as content_length 
                FROM (
                    SELECT 
                        id, 
                        url, 
                        title, 
                        chunk_number,
                        content,
                        ROW_NUMBER() OVER (PARTITION BY url ORDER BY chunk_number) as rn
                    FROM site_pages
                ) sub
                WHERE rn = 1
                ORDER BY id
                LIMIT 5
            """)
            sample_pages = cur.fetchall()
            
            print("\nSample pages (first chunk of each URL):")
            for page in sample_pages:
                print(f"  - {page[1]}")
                print(f"    Title: {page[2]}")
                print(f"    Content length: {page[4]} characters")
    
    # Check for problems
    print("\n4. Checking for potential issues...")
    issues_found = False
    
    # Check 1: No pages with both source_id and pages
    if sources and total_pages == 0:
        print("❗ ISSUE: Sources exist but no pages are stored.")
        issues_found = True
    
    # Check 2: Pages with empty content
    if total_pages > 0:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM site_pages WHERE length(content) < 10")
            result = cur.fetchone()
            empty_pages = result[0] if result else 0
            
            if empty_pages > 0:
                print(f"❗ ISSUE: Found {empty_pages} pages with very little content (<10 chars).")
                issues_found = True
        
        # Check 3: Pages with missing embeddings
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM site_pages WHERE embedding IS NULL")
            result = cur.fetchone()
            no_embeddings = result[0] if result else 0
            
            if no_embeddings > 0:
                print(f"❗ ISSUE: Found {no_embeddings} pages with missing embeddings.")
                issues_found = True
    
    if not issues_found:
        print("✓ No major issues detected.")
    
    print("\n===== Check Complete =====")

if __name__ == "__main__":
    if USING_PSYCOPG3:
        asyncio.run(check_database())
    else:
        # For psycopg2, we can run synchronously
        check_database() 