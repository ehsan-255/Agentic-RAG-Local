import logging
import os
import sys
import json
import time
import psutil
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List, Set, Type, Union
import threading
import uuid

from src.utils.errors import ErrorCategory, BaseError
from src.utils.logging import Logger

# Class for tracking error statistics
class ErrorStats:
    """Class for tracking error statistics."""
    
    def __init__(self):
        self.total_errors = 0
        self.errors_by_category = {cat: 0 for cat in ErrorCategory}
        self.errors_by_type = {}
        self.lock = threading.Lock()
        
    def record_error(self, error: Union[BaseError, Exception], error_type: Optional[str] = None) -> None:
        """
        Record an error in the statistics.
        
        Args:
            error: The error that occurred
            error_type: Optional error type name
        """
        with self.lock:
            self.total_errors += 1
            
            # If it's our custom error type
            if isinstance(error, BaseError):
                category = error.category
                error_type = error.__class__.__name__
            else:
                category = ErrorCategory.UNKNOWN
                error_type = error_type or error.__class__.__name__
            
            # Update category stats
            self.errors_by_category[category] = self.errors_by_category.get(category, 0) + 1
            
            # Update type stats
            self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get the current error statistics.
        
        Returns:
            Dict[str, Any]: The current error statistics
        """
        with self.lock:
            return {
                "total_errors": self.total_errors,
                "by_category": {cat.value: count for cat, count in self.errors_by_category.items()},
                "by_type": self.errors_by_type
            }
    
    def reset(self) -> None:
        """Reset all error statistics."""
        with self.lock:
            self.total_errors = 0
            self.errors_by_category = {cat: 0 for cat in ErrorCategory}
            self.errors_by_type = {}

# Class for tracking a crawl session
class CrawlSession:
    """Class for tracking an active crawl session."""
    
    def __init__(self, session_id, source_id, source_name):
        self.session_id = session_id
        self.source_id = source_id
        self.source_name = source_name
        self.start_time = time.time()
        self.end_time = None
        self.status = "running"
        self.total_urls = 0
        self.processed_urls = 0
        self.successful_urls = 0
        self.failed_urls = 0
        self.metrics = {
            "api_calls": 0,
            "embedding_calls": 0,
            "llm_calls": 0,
            "db_operations": 0,
            "errors": {}
        }
        
    def set_total_urls(self, count):
        """Set the total number of URLs to process."""
        self.total_urls = count
        
    def record_page_processed(self, url, success):
        """Record a processed page."""
        self.processed_urls += 1
        if success:
            self.successful_urls += 1
        else:
            self.failed_urls += 1
        
    def record_api_call(self, endpoint):
        """Record an API call."""
        self.metrics["api_calls"] += 1
        if endpoint == "embeddings":
            self.metrics["embedding_calls"] += 1
        elif endpoint in ["completions", "chat/completions"]:
            self.metrics["llm_calls"] += 1
            
    def record_error(self, error_category):
        """Record an error."""
        if error_category not in self.metrics["errors"]:
            self.metrics["errors"][error_category] = 0
        self.metrics["errors"][error_category] += 1
        
    def pause(self):
        """Pause the session."""
        self.status = "paused"
        
    def resume(self):
        """Resume the session."""
        self.status = "running"
        
    def complete(self, status="completed"):
        """Complete the session."""
        self.status = status
        self.end_time = time.time()
        
    def get_duration(self):
        """Get the session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
        
    def format_duration(self):
        """Format the session duration as a string."""
        duration = self.get_duration()
        
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
            
    def format_start_time(self):
        """Format the start time as a string."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))
        
    def get_success_rate(self):
        """Get the success rate."""
        if self.processed_urls == 0:
            return 0
        return self.successful_urls / self.processed_urls
        
    @property
    def success_rate(self):
        """Success rate property."""
        return self.get_success_rate()
        
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.get_duration(),
            "total_urls": self.total_urls,
            "processed_urls": self.processed_urls,
            "successful_urls": self.successful_urls,
            "failed_urls": self.failed_urls,
            "success_rate": self.get_success_rate(),
            "metrics": self.metrics
        }

# Global storage for active sessions and stats
class MonitoringState:
    """Global state for monitoring."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MonitoringState, cls).__new__(cls)
            cls._instance.sessions = {}
            cls._instance.active_session_id = None
            cls._instance.global_error_stats = ErrorStats()
            cls._instance.lock = threading.Lock()
        return cls._instance
    
    def create_session(self, source_id: str, source_name: str) -> str:
        """
        Create a new crawl session.
        
        Args:
            source_id: ID of the source being crawled
            source_name: Name of the source being crawled
            
        Returns:
            str: ID of the created session
        """
        # Generate a unique session ID
        session_id = f"crawl_{source_id}_{int(time.time())}"
        # Create session with all three required parameters
        session = CrawlSession(session_id, source_id, source_name)
        with self.lock:
            self.sessions[session.session_id] = session
            self.active_session_id = session.session_id
        return session.session_id
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[CrawlSession]:
        """
        Get a session by ID or the active session.
        
        Args:
            session_id: ID of the session to get, or None for the active session
            
        Returns:
            Optional[CrawlSession]: The session, or None if not found
        """
        with self.lock:
            session_id = session_id or self.active_session_id
            if not session_id:
                return None
            return self.sessions.get(session_id)
    
    def end_session(self, session_id: Optional[str] = None) -> bool:
        """
        End a session by ID or the active session.
        
        Args:
            session_id: ID of the session to end, or None for the active session
            
        Returns:
            bool: True if the session was ended, False otherwise
        """
        with self.lock:
            session_id = session_id or self.active_session_id
            if not session_id:
                return False
                
            session = self.sessions.get(session_id)
            if not session:
                return False
                
            # Call complete() instead of end_session() which doesn't exist
            session.complete(status="cancelled")
            if self.active_session_id == session_id:
                self.active_session_id = None
            return True
    
    def record_error(self, error: Union[BaseError, Exception], session_id: Optional[str] = None) -> None:
        """
        Record an error in the global stats and in the session if provided.
        
        Args:
            error: The error that occurred
            session_id: Optional session ID to record the error for
        """
        # Record in global stats
        self.global_error_stats.record_error(error)
        
        # Record in session if provided
        if session_id:
            session = self.get_session(session_id)
            if session:
                # Safely record error in session
                try:
                    # Check if session has error_stats attribute
                    if hasattr(session, 'error_stats'):
                        session.error_stats.record_error(error)
                    else:
                        # If not, record in the session metrics directly
                        error_category = error.category if hasattr(error, 'category') else ErrorCategory.UNKNOWN
                        error_type = error.__class__.__name__
                        
                        # Update session metrics
                        if 'errors' not in session.metrics:
                            session.metrics['errors'] = {}
                        
                        category_name = error_category.value if hasattr(error_category, 'value') else str(error_category)
                        if category_name not in session.metrics['errors']:
                            session.metrics['errors'][category_name] = 0
                        
                        session.metrics['errors'][category_name] += 1
                except Exception as e:
                    # Log but don't crash if error recording fails
                    logger.error(f"Failed to record error in session {session_id}: {str(e)}")
        
        # Also record in active session if different from provided session
        elif self.active_session_id and self.active_session_id != session_id:
            session = self.get_session(self.active_session_id)
            if session:
                try:
                    # Check if session has error_stats attribute
                    if hasattr(session, 'error_stats'):
                        session.error_stats.record_error(error)
                    else:
                        # If not, record in the session metrics directly
                        error_category = error.category if hasattr(error, 'category') else ErrorCategory.UNKNOWN
                        
                        # Update session metrics
                        if 'errors' not in session.metrics:
                            session.metrics['errors'] = {}
                        
                        category_name = error_category.value if hasattr(error_category, 'value') else str(error_category)
                        if category_name not in session.metrics['errors']:
                            session.metrics['errors'][category_name] = 0
                        
                        session.metrics['errors'][category_name] += 1
                except Exception as e:
                    # Log but don't crash if error recording fails
                    logger.error(f"Failed to record error in active session: {str(e)}")

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system resource metrics.
        
        Returns:
            Dict[str, Any]: System resource metrics
        """
        try:
            process = psutil.Process()
            with process.oneshot():
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent(interval=0.1)
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "memory_rss_mb": memory_info.rss / (1024 * 1024),
                    "memory_vms_mb": memory_info.vms / (1024 * 1024),
                    "cpu_percent": cpu_percent,
                    "thread_count": process.num_threads(),
                    "system_memory_percent": psutil.virtual_memory().percent
                }
        except Exception as e:
            # Don't fail if we can't get metrics
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

# Create the global monitoring state
monitoring_state = MonitoringState()

# Enhanced logger with structured error reporting
class EnhancedLogger(Logger):
    """Enhanced logger with structured error reporting and monitoring integration."""
    
    def __init__(self, name: str):
        """
        Initialize the enhanced logger.
        
        Args:
            name: Name of the logger
        """
        super().__init__(name)
        self.name = name
    
    def structured_error(self, message: str, 
                          error: Optional[Union[BaseError, Exception]] = None,
                          category: Optional[ErrorCategory] = None,
                          session_id: Optional[str] = None,
                          **kwargs) -> None:
        """
        Log a structured error message.
        
        Args:
            message: Error message
            error: The error that occurred, if any
            category: Error category, if not provided by the error
            session_id: Optional session ID to associate with the error
            **kwargs: Additional fields to include in the log
        """
        # Determine the error category
        effective_category = category
        if isinstance(error, BaseError):
            effective_category = error.category
        elif not effective_category:
            effective_category = ErrorCategory.UNKNOWN
            
        # Record the error in monitoring state
        monitoring_state.record_error(error if error else Exception(message), session_id)
        
        # Create a completely new error_details dict to prevent any possible collisions
        error_details = {
            "error_category": effective_category.value if hasattr(effective_category, 'value') else str(effective_category),
            "timestamp": datetime.now().isoformat(),
            "logger_name": self.name
        }
        
        # Add session context if available
        if session_id:
            error_details["session_id"] = session_id
        else:
            active_session = monitoring_state.get_session()
            if active_session:
                error_details["session_id"] = active_session.session_id
        
        # Add error details if available
        if error:
            if isinstance(error, BaseError):
                # Safely add BaseError details, avoiding any 'message' key
                base_error_dict = error.to_dict()
                for key, value in base_error_dict.items():
                    if key != 'message':
                        error_details[key] = value
                    else:
                        error_details['error_message_detail'] = value
            else:
                error_details["error_type"] = error.__class__.__name__
                error_details["error_message"] = str(error)
                
            # Add traceback but avoid storing it as 'message'
            error_details["error_traceback"] = traceback.format_exc()
        
        # Carefully add kwargs, protecting against any 'message' collisions
        for key, value in kwargs.items():
            if key == 'message':
                error_details['error_message_detail'] = value
            elif key == 'exc_info':
                error_details['error_exception_info'] = value
            else:
                error_details[key] = value
        
        # Get exception info for logging, avoiding collisions with kwargs
        exception_info = error if not isinstance(error, BaseError) else None
        
        # Use a completely different approach to avoid param collision
        # Call the base logger's log method directly with explicitly named parameters
        self.logger.error(
            msg=message,
            exc_info=True if exception_info else False,
            extra=error_details
        )
    
    def record_api_request(self, api_name: str, endpoint: str, 
                          duration_ms: float, status_code: Optional[int] = None,
                          rate_limit_remaining: Optional[int] = None,
                          rate_limit_reset: Optional[int] = None,
                          **kwargs) -> None:
        """
        Record an API request for monitoring.
        
        Args:
            api_name: Name of the API (e.g., "OpenAI")
            endpoint: API endpoint called
            duration_ms: Request duration in milliseconds
            status_code: HTTP status code, if applicable
            rate_limit_remaining: Remaining API rate limit, if available
            rate_limit_reset: Rate limit reset timestamp, if available
            **kwargs: Additional fields to include in the log
        """
        api_details = {
            "api_name": api_name,
            "endpoint": endpoint,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        if status_code is not None:
            api_details["status_code"] = status_code
            
        if rate_limit_remaining is not None:
            api_details["rate_limit_remaining"] = rate_limit_remaining
            
        if rate_limit_reset is not None:
            api_details["rate_limit_reset"] = rate_limit_reset
            
        # Add active session if available
        session = monitoring_state.get_session()
        if session:
            api_details["session_id"] = session.session_id
            
        # Add additional fields
        api_details.update(kwargs)
        
        # Log at the appropriate level based on status and duration
        if status_code and status_code >= 400:
            self.error(f"API error: {api_name} {endpoint} returned {status_code}", **api_details)
        elif duration_ms > 5000:  # Over 5 seconds is a slow request
            self.warning(f"Slow API request: {api_name} {endpoint} took {duration_ms:.2f}ms", **api_details)
        else:
            # Use debug level for normal API requests to avoid log spam
            self.debug(f"API request: {api_name} {endpoint}", **api_details)

# Create enhanced loggers for different components
enhanced_crawler_logger = EnhancedLogger('crawler')
enhanced_db_logger = EnhancedLogger('database')
enhanced_api_logger = EnhancedLogger('api')
enhanced_ui_logger = EnhancedLogger('ui')
enhanced_rag_logger = EnhancedLogger('rag')

def get_enhanced_logger(component: str) -> EnhancedLogger:
    """
    Get an enhanced logger for a specific component.
    
    Args:
        component: Component name
        
    Returns:
        EnhancedLogger: Enhanced logger for the component
    """
    return EnhancedLogger(component)

# Session management convenience functions
def start_crawl_session(source_id: str, source_name: str) -> str:
    """
    Start a new crawl session.
    
    Args:
        source_id: ID of the source being crawled
        source_name: Name of the source being crawled
        
    Returns:
        str: ID of the created session
    """
    session_id = monitoring_state.create_session(source_id, source_name)
    enhanced_crawler_logger.info(f"Started crawl session for {source_name}", 
                               session_id=session_id,
                               source_id=source_id,
                               source_name=source_name)
    return session_id

def end_crawl_session(session_id: Optional[str] = None, status: str = "completed") -> None:
    """
    End a crawl session.
    
    Args:
        session_id: ID of the session to end, or None for the active session
        status: Status to set for the session (completed, cancelled, failed, etc.)
    """
    # Get the session to end - from active session if not specified
    if session_id is None:
        session_id = monitoring_state.active_session_id
        if not session_id:
            return
            
    session = monitoring_state.get_session(session_id)
    if not session:
        return
        
    # Preserve source information for logging
    source_name = session.source_name
    
    # Complete the session with the appropriate status
    # Call complete() method directly on the session
    session.complete(status=status)
    
    # Update the monitoring state to reflect session is no longer active
    if monitoring_state.active_session_id == session_id:
        monitoring_state.active_session_id = None
    
    # Log completion
    enhanced_crawler_logger.info(f"Ended crawl session for {source_name}", 
                            session_id=session_id,
                            status=status)

def get_session_stats(session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get statistics for a crawl session.
    
    Args:
        session_id: ID of the session to get stats for, or None for the active session
        
    Returns:
        Optional[Dict[str, Any]]: Session statistics, or None if session not found
    """
    session = monitoring_state.get_session(session_id)
    if session:
        return session.to_dict()
    return None

def get_active_session() -> Optional[CrawlSession]:
    """
    Get the active crawl session.
    
    Returns:
        Optional[CrawlSession]: The active session, or None if no active session
    """
    return monitoring_state.get_session()

def get_system_metrics() -> Dict[str, Any]:
    """
    Get current system resource metrics.
    
    Returns:
        Dict[str, Any]: System resource metrics
    """
    return monitoring_state.get_system_metrics()

# Add a function to get error statistics
def get_error_stats():
    """
    Get error statistics from the monitoring state.
    
    Returns:
        Dict[str, Dict[str, Any]]: Error statistics by category
    """
    with monitoring_state.lock:
        return monitoring_state.global_error_stats.get_stats()

# Add a function to get API call statistics
def get_api_stats():
    """
    Get API call statistics from the monitoring state.
    
    Returns:
        Dict[str, Any]: API call statistics
    """
    with monitoring_state.lock:
        if "api_calls" not in monitoring_state:
            return {}
        
        return monitoring_state["api_calls"].copy()

# Update monitoring state logic
def update_monitoring_state(key, value):
    """
    Update a specific key in the monitoring state.
    
    Args:
        key: Key to update
        value: New value
    """
    with monitoring_state.lock:
        monitoring_state[key] = value
        
        # If this is API monitoring data, also update any active session
        if key == "api_calls" and "active_session" in monitoring_state:
            session_id = monitoring_state["active_session"]
            if session_id in monitoring_state.sessions:
                session = monitoring_state.sessions[session_id]
                # Update session metrics from API calls
                for endpoint, count in value.get("calls_by_endpoint", {}).items():
                    if endpoint == "embeddings":
                        session.metrics["embedding_calls"] = count
                    elif endpoint in ["completions", "chat/completions"]:
                        session.metrics["llm_calls"] = count
        
        # If this is error data, also update any active session
        if key == "errors" and "active_session" in monitoring_state:
            session_id = monitoring_state["active_session"]
            if session_id in monitoring_state.sessions:
                session = monitoring_state.sessions[session_id]
                # Update session error metrics
                for category, stats in value.items():
                    session.metrics["errors"][category] = stats.get("count", 0)

# Update the function for logging errors to ensure they're captured correctly
def log_structured_error(logger, message, error=None, **kwargs):
    """
    Log a structured error message with error tracking.
    
    Args:
        logger: Logger instance
        message: Error message
        error: Exception object (optional)
        **kwargs: Additional context data
    """
    # Determine error category
    error_category = "UNKNOWN"
    if error is not None:
        if hasattr(error, "error_category"):
            error_category = error.error_category
        else:
            # Try to guess category from error type
            error_type = type(error).__name__
            if "HTTP" in error_type or "Connection" in error_type:
                error_category = "CONNECTION"
            elif "Timeout" in error_type:
                error_category = "TIMEOUT"
            elif "API" in error_type or "OpenAI" in error_type:
                error_category = "API"
            elif "Database" in error_type or "SQL" in error_type:
                error_category = "DATABASE"
    
    # Add error_category to kwargs
    kwargs["error_category"] = error_category
    
    # Add error details if available
    if error is not None:
        kwargs["error_type"] = type(error).__name__
        kwargs["error_message"] = str(error)
        
        # Add error details if available
        if hasattr(error, "details"):
            kwargs["error_details"] = error.details
    
    # Log the error
    logger.error(message, **kwargs)
    
    # Update error statistics
    with monitoring_state.lock:
        if "errors" not in monitoring_state:
            monitoring_state["errors"] = {}
            
        if error_category not in monitoring_state["errors"]:
            monitoring_state["errors"][error_category] = {
                "count": 0,
                "last_seen": time.time(),
                "examples": []
            }
            
        # Update statistics
        monitoring_state["errors"][error_category]["count"] += 1
        monitoring_state["errors"][error_category]["last_seen"] = time.time()
        
        # Add example (keep last 10)
        examples = monitoring_state["errors"][error_category]["examples"]
        example = {
            "message": message,
            "error_type": type(error).__name__ if error else "None",
            "error_message": str(error) if error else "None",
            "timestamp": time.time()
        }
        examples.append(example)
        if len(examples) > 10:
            examples = examples[-10:]
        monitoring_state["errors"][error_category]["examples"] = examples
        
    # Update session error count if there's an active session
    if "active_session" in monitoring_state:
        session_id = monitoring_state["active_session"]
        if session_id in monitoring_state.sessions:
            session = monitoring_state.sessions[session_id]
            session.record_error(error_category) 