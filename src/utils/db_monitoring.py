import time
import traceback
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast
from functools import wraps

from src.utils.enhanced_logging import enhanced_db_logger, monitoring_state
from src.utils.errors import DatabaseError, ConnectionError

# Type variable for return type
T = TypeVar('T')

def monitor_db_operation(description: str) -> Callable:
    """
    Decorator to monitor database operations.
    
    Args:
        description: Description of the operation
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            query = kwargs.get('query', None)
            params = kwargs.get('params', None)
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Log slow queries (>500ms)
                if duration_ms > 500:
                    enhanced_db_logger.warning(
                        f"Slow database operation: {description}",
                        operation=description,
                        duration_ms=duration_ms,
                        query=query,
                        params=params
                    )
                else:
                    enhanced_db_logger.debug(
                        f"Database operation: {description}",
                        operation=description,
                        duration_ms=duration_ms,
                        query=query,
                        params=params
                    )
                    
                return result
                
            except Exception as e:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Create appropriate error
                if "connection" in str(e).lower():
                    error = ConnectionError(
                        f"Database connection error: {e}",
                        target="database",
                        details={
                            "operation": description,
                            "duration_ms": duration_ms,
                            "query": query,
                            "params": params,
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }
                    )
                else:
                    error = DatabaseError(
                        f"Database error: {e}",
                        query=query,
                        details={
                            "operation": description,
                            "duration_ms": duration_ms,
                            "params": params,
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }
                    )
                
                # Log the error
                enhanced_db_logger.structured_error(
                    f"Database error executing {description}",
                    error=error
                )
                
                # Re-raise the original exception to maintain compatibility
                raise
                
        return wrapper
    return decorator

# Store connection pool stats
connection_pool_stats = {
    "created_connections": 0,
    "active_connections": 0,
    "idle_connections": 0,
    "max_connections": 0,
    "wait_count": 0,
    "last_updated": None
}

def update_connection_pool_stats(
    created: Optional[int] = None,
    active: Optional[int] = None,
    idle: Optional[int] = None,
    max_connections: Optional[int] = None,
    wait_count: Optional[int] = None
) -> None:
    """
    Update connection pool statistics.
    
    Args:
        created: Number of created connections
        active: Number of active connections
        idle: Number of idle connections
        max_connections: Maximum number of connections
        wait_count: Number of waiters for a connection
    """
    if created is not None:
        connection_pool_stats["created_connections"] = created
    if active is not None:
        connection_pool_stats["active_connections"] = active
    if idle is not None:
        connection_pool_stats["idle_connections"] = idle
    if max_connections is not None:
        connection_pool_stats["max_connections"] = max_connections
    if wait_count is not None:
        connection_pool_stats["wait_count"] = wait_count
        
    connection_pool_stats["last_updated"] = time.time()
    
    # Log if pool is under pressure
    if connection_pool_stats["active_connections"] > 0.8 * connection_pool_stats["max_connections"]:
        enhanced_db_logger.warning(
            "Database connection pool under pressure",
            active_connections=connection_pool_stats["active_connections"],
            max_connections=connection_pool_stats["max_connections"],
            wait_count=connection_pool_stats["wait_count"]
        )

def get_connection_pool_stats() -> Dict[str, Any]:
    """
    Get connection pool statistics.
    
    Returns:
        Dict[str, Any]: Connection pool statistics
    """
    return connection_pool_stats

# Transaction monitoring
active_transactions = 0
transactions_started = 0
transactions_committed = 0
transactions_rolled_back = 0

def monitor_transaction_start() -> None:
    """Track the start of a database transaction."""
    global active_transactions, transactions_started
    active_transactions += 1
    transactions_started += 1
    
    enhanced_db_logger.debug(
        "Transaction started",
        active_transactions=active_transactions,
        transactions_started=transactions_started
    )

def monitor_transaction_end(committed: bool) -> None:
    """
    Track the end of a database transaction.
    
    Args:
        committed: Whether the transaction was committed or rolled back
    """
    global active_transactions, transactions_committed, transactions_rolled_back
    active_transactions = max(0, active_transactions - 1)
    
    if committed:
        transactions_committed += 1
        enhanced_db_logger.debug(
            "Transaction committed",
            active_transactions=active_transactions,
            transactions_committed=transactions_committed
        )
    else:
        transactions_rolled_back += 1
        enhanced_db_logger.debug(
            "Transaction rolled back",
            active_transactions=active_transactions,
            transactions_rolled_back=transactions_rolled_back
        )

def get_transaction_stats() -> Dict[str, int]:
    """
    Get transaction statistics.
    
    Returns:
        Dict[str, int]: Transaction statistics
    """
    return {
        "active_transactions": active_transactions,
        "transactions_started": transactions_started,
        "transactions_committed": transactions_committed,
        "transactions_rolled_back": transactions_rolled_back
    }

# Query statistics
query_stats = {
    "total_queries": 0,
    "slow_queries": 0,  # Queries that took more than 500ms
    "error_queries": 0,
    "avg_duration_ms": 0,
    "total_duration_ms": 0
}

def update_query_stats(duration_ms: float, error: bool = False) -> None:
    """
    Update query statistics.
    
    Args:
        duration_ms: Query duration in milliseconds
        error: Whether the query resulted in an error
    """
    query_stats["total_queries"] += 1
    query_stats["total_duration_ms"] += duration_ms
    query_stats["avg_duration_ms"] = query_stats["total_duration_ms"] / query_stats["total_queries"]
    
    if duration_ms > 500:
        query_stats["slow_queries"] += 1
        
    if error:
        query_stats["error_queries"] += 1

def get_query_stats() -> Dict[str, Union[int, float]]:
    """
    Get query statistics.
    
    Returns:
        Dict[str, Union[int, float]]: Query statistics
    """
    return query_stats 