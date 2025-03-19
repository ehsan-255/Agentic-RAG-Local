import contextlib
import asyncio
import sys
import logging
from typing import Optional, Dict, Any, List

# Import our compatibility layer first
from src.db.db_utils import (
    using_psycopg2, 
    using_psycopg3, 
    get_json_adapter, 
    get_dict_cursor,
    is_database_available,
    AsyncConnectionPool
)

# Configure logging
logger = logging.getLogger("connection")

# Import the appropriate modules based on availability
if using_psycopg3:
    import psycopg
elif using_psycopg2:
    import psycopg2
    from psycopg2.extras import Json, DictCursor
else:
    logger.error("No PostgreSQL driver available. Database operations will fail.")

from src.config import config

# Global connection pool
_pool: Optional[AsyncConnectionPool] = None
_pool_lock = asyncio.Lock()  # Lock for creating the pool

async def get_connection_pool() -> Optional[AsyncConnectionPool]:
    """
    Get or create the database connection pool.
    
    Returns:
        AsyncConnectionPool: The database connection pool or None if not available
    """
    global _pool
    
    # Check if we have a database driver
    if not is_database_available():
        logger.error("Cannot create connection pool: No PostgreSQL driver available")
        return None
    
    if _pool is None:
        async with _pool_lock:  # Use lock to prevent multiple pool creations
            if _pool is None:  # Check again in case another thread created it
                # Create connection pool using configuration
                try:
                    _pool = AsyncConnectionPool(
                        conninfo=config.get_db_connection_string(),
                        min_size=config.DB_POOL_MIN_SIZE,
                        max_size=config.DB_POOL_MAX_SIZE,
                        kwargs={"autocommit": True}
                    )
                    logger.info(f"Successfully created connection pool with {config.DB_POOL_MIN_SIZE}-{config.DB_POOL_MAX_SIZE} connections")
                except Exception as e:
                    logger.error(f"Error creating connection pool: {e}")
                    return None
    
    return _pool

@contextlib.asynccontextmanager
async def get_db_connection():
    """
    Get a database connection from the pool.
    
    Yields:
        A database connection
    """
    pool = await get_connection_pool()
    
    if pool is None:
        logger.error("Failed to get connection: pool not available")
        raise RuntimeError("Database connection pool not available")
    
    conn = await pool.getconn()
    try:
        yield conn
    finally:
        await pool.putconn(conn)

async def execute_query(query: str, params: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Execute a database query and return the results.
    
    Args:
        query: SQL query to execute
        params: Query parameters (optional)
        
    Returns:
        Optional[List[Dict[str, Any]]]: Query results or None on error
    """
    if not is_database_available():
        logger.error("Cannot execute query: No PostgreSQL driver available")
        return None
    
    try:
        async with get_db_connection() as conn:
            # Use dict_row to get results as dictionaries
            row_factory = get_dict_cursor()
            if row_factory is None:
                logger.error("Cannot get dict cursor factory")
                return None
                
            async with conn.cursor(row_factory=row_factory) as cur:
                await cur.execute(query, params or {})
                
                # Check if this is a SELECT query
                if query.strip().upper().startswith("SELECT"):
                    results = await cur.fetchall()
                    return list(results)  # Convert to regular list for JSON serializability
                    
                return []  # Non-SELECT queries return empty list
    except Exception as e:
        logger.error(f"Database error executing query: {e}")
        return None

async def execute_transaction(queries: List[Dict[str, Any]]) -> bool:
    """
    Execute multiple queries in a transaction.
    
    Args:
        queries: List of query dictionaries with 'query' and optional 'params' keys
        
    Returns:
        bool: True if transaction was successful, False otherwise
    """
    if not is_database_available():
        logger.error("Cannot execute transaction: No PostgreSQL driver available")
        return False
    
    try:
        async with get_db_connection() as conn:
            # Disable autocommit to start transaction
            await conn.set_autocommit(False)
            
            try:
                async with conn.cursor() as cur:
                    for query_dict in queries:
                        query = query_dict.get("query")
                        params = query_dict.get("params", {})
                        await cur.execute(query, params)
                
                # Commit the transaction
                await conn.commit()
                return True
            except Exception as e:
                # Rollback on error
                await conn.rollback()
                logger.error(f"Transaction error: {e}")
                return False
            finally:
                # Reset autocommit
                await conn.set_autocommit(True)
    except Exception as e:
        logger.error(f"Database connection error in transaction: {e}")
        return False

async def close_pool():
    """Close the database connection pool."""
    global _pool
    
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed") 