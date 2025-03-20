import contextlib
import asyncio
import sys
import logging
import time
from typing import Optional, Dict, Any, List
from functools import wraps

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

# Add connection monitoring stats
connection_stats = {
    "active_connections": 0,
    "total_connections": 0,
    "available_connections": 0,
    "query_time": {
        "total": 0.0,
        "count": 0,
        "avg": 0.0,
        "max": 0.0
    }
}

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

# Add a function to get connection statistics
def get_connection_stats():
    """
    Get connection pool statistics.
    
    Returns:
        Dict[str, Any]: Connection statistics
    """
    global _pool, connection_stats
    
    # Update stats from the connection pool if available
    if _pool:
        try:
            # Check for the appropriate attribute names based on which driver is being used
            if hasattr(_pool, 'max_size'):
                # psycopg3 attributes
                connection_stats["total_connections"] = _pool.max_size
                connection_stats["active_connections"] = _pool.size if hasattr(_pool, 'size') else 0
                connection_stats["available_connections"] = _pool.max_size - connection_stats["active_connections"]
            elif hasattr(_pool, 'maxconn'):
                # psycopg2 compatibility attributes
                connection_stats["total_connections"] = _pool.maxconn
                connection_stats["active_connections"] = len(_pool._used) if hasattr(_pool, '_used') else 0
                connection_stats["available_connections"] = _pool.maxconn - connection_stats["active_connections"]
            else:
                # Generic fallback for unknown pool implementation
                connection_stats["total_connections"] = "unknown"
                connection_stats["active_connections"] = "unknown"
                connection_stats["available_connections"] = "unknown"
                logger.warning("Unable to determine connection pool metrics - unknown pool implementation")
        except Exception as e:
            logger.error(f"Error updating connection stats: {e}")
    
    return connection_stats.copy()

# Function decorator to track query execution time
def track_query_time(func):
    """
    Decorator to track query execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global connection_stats
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.time() - start_time
            connection_stats["query_time"]["total"] += duration
            connection_stats["query_time"]["count"] += 1
            connection_stats["query_time"]["avg"] = (
                connection_stats["query_time"]["total"] / 
                connection_stats["query_time"]["count"]
            )
            connection_stats["query_time"]["max"] = max(
                connection_stats["query_time"]["max"],
                duration
            )
    
    return wrapper

# Apply the decorator to execute_query function if it exists
if "execute_query" in globals():
    execute_query = track_query_time(execute_query) 