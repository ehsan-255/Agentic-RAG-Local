"""
Database utility functions that provide compatibility between psycopg2 and psycopg3 (psycopg).

This module detects available database drivers and provides a consistent interface
regardless of which version is installed.
"""

import sys
import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_utils")

# Track which driver we're using
using_psycopg2 = False
using_psycopg3 = False

# Try to import psycopg3 first (preferred)
try:
    import psycopg
    from psycopg_pool import AsyncConnectionPool
    logger.info("Using psycopg3 (psycopg) for database operations")
    using_psycopg3 = True
except ImportError:
    logger.warning("psycopg3 not found, trying psycopg2")
    try:
        import psycopg2
        from psycopg2.extras import Json, DictCursor
        from psycopg2 import pool
        logger.info("Using psycopg2 for database operations")
        using_psycopg2 = True
        
        # Define AsyncConnectionPool as a dummy class when using psycopg2
        # This provides async-compatible interfaces even when using sync psycopg2
        class AsyncConnectionPool:
            """Dummy async connection pool that provides psycopg2 compatibility."""
            
            def __init__(self, conninfo, min_size=1, max_size=10, **kwargs):
                """Initialize the pool with psycopg2 connection parameters."""
                self.conninfo = conninfo
                self.min_size = min_size
                self.max_size = max_size
                self.kwargs = kwargs
                self._pool = pool.ThreadedConnectionPool(
                    min_size, 
                    max_size, 
                    conninfo
                )
                logger.info(f"Created compatibility ThreadedConnectionPool with {min_size}-{max_size} connections")
            
            async def getconn(self):
                """Get a connection from the pool."""
                conn = self._pool.getconn()
                return PsycopgCompatConnection(conn)
                
            async def putconn(self, conn):
                """Return a connection to the pool."""
                if hasattr(conn, '_conn'):
                    self._pool.putconn(conn._conn)
                else:
                    logger.warning("Attempted to return invalid connection to pool")
            
            async def close(self):
                """Close the connection pool."""
                self._pool.closeall()
                logger.info("Closed compatibility connection pool")
        
        class PsycopgCompatConnection:
            """Wrapper around psycopg2 connection to provide async-compatible interface."""
            
            def __init__(self, conn):
                """Initialize with a psycopg2 connection."""
                self._conn = conn
                
            async def set_autocommit(self, autocommit):
                """Set autocommit mode."""
                self._conn.autocommit = autocommit
                
            async def commit(self):
                """Commit the transaction."""
                self._conn.commit()
                
            async def rollback(self):
                """Rollback the transaction."""
                self._conn.rollback()
                
            def cursor(self, *args, **kwargs):
                """Get a cursor, adapting row_factory parameter."""
                # Handle row_factory specially (psycopg3 parameter)
                if 'row_factory' in kwargs:
                    row_factory = kwargs.pop('row_factory')
                    # If it's dict_row from our compatibility layer, use DictCursor
                    if row_factory == psycopg2.extras.DictCursor:
                        kwargs['cursor_factory'] = psycopg2.extras.DictCursor
                
                cursor = self._conn.cursor(**kwargs)
                return PsycopgCompatCursor(cursor)
                
        class PsycopgCompatCursor:
            """Wrapper around psycopg2 cursor to provide async-compatible interface."""
            
            def __init__(self, cursor):
                """Initialize with a psycopg2 cursor."""
                self._cursor = cursor
                
            async def __aenter__(self):
                """Async context manager entry."""
                return self
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                """Async context manager exit."""
                self._cursor.close()
                
            async def execute(self, query, params=None):
                """Execute a query."""
                return self._cursor.execute(query, params or {})
                
            async def fetchall(self):
                """Fetch all results."""
                results = self._cursor.fetchall()
                # If using DictCursor, results are already dicts
                return results
    except ImportError:
        logger.error("Neither psycopg3 nor psycopg2 found. Database functionality will be limited.")
        logger.error("Please install with: pip install \"psycopg[binary]\"==3.1.13 psycopg-pool==3.1.8")
        
        # Define dummy classes
        class AsyncConnectionPool:
            """Dummy async connection pool that does nothing."""
            def __init__(self, *args, **kwargs):
                logger.error("No PostgreSQL driver available - AsyncConnectionPool is non-functional")
            
            async def getconn(self):
                raise RuntimeError("No PostgreSQL driver available")
                
            async def putconn(self, conn):
                pass
                
            async def close(self):
                pass

# Provide unified functions that work with either driver

def get_json_adapter():
    """Get the appropriate JSON adapter for the active driver."""
    if using_psycopg3:
        return lambda obj: obj  # psycopg3 handles JSON natively
    elif using_psycopg2:
        return psycopg2.extras.Json
    else:
        return lambda obj: obj  # Fallback that won't work but prevents crashes

def get_dict_cursor():
    """Get the appropriate dictionary cursor for the active driver."""
    if using_psycopg3:
        return psycopg.rows.dict_row
    elif using_psycopg2:
        return psycopg2.extras.DictCursor
    else:
        return None

def is_database_available() -> bool:
    """Check if a database driver is available."""
    return using_psycopg2 or using_psycopg3

def get_driver_name() -> str:
    """Get the name of the active database driver."""
    if using_psycopg3:
        return "psycopg3"
    elif using_psycopg2:
        return "psycopg2"
    else:
        return "none"

# Export the variables for use in other modules
__all__ = [
    "using_psycopg2", 
    "using_psycopg3", 
    "get_json_adapter", 
    "get_dict_cursor",
    "is_database_available",
    "get_driver_name",
    "AsyncConnectionPool"  # Export the AsyncConnectionPool class
] 