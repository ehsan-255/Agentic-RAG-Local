from enum import Enum
from typing import Optional, Dict, Any, List

class ErrorCategory(Enum):
    """Enum for categorizing errors in the system."""
    DATABASE = "database"
    CONNECTION = "connection"
    CONTENT_PROCESSING = "content_processing"
    EMBEDDING = "embedding"
    TASK_SCHEDULING = "task_scheduling"
    API_RATE_LIMIT = "api_rate_limit"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"

class BaseError(Exception):
    """Base error class for all custom errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.category = category
        self.details = details or {}
        super().__init__(message)
        
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary for logging/serialization."""
        return {
            "message": self.message,
            "category": self.category.value,
            "details": self.details,
            "error_type": self.__class__.__name__
        }

# Content Processing Errors
class ContentProcessingError(BaseError):
    """Base class for errors that occur during content processing."""
    
    def __init__(self, message: str, url: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if url:
            details["url"] = url
        super().__init__(message, ErrorCategory.CONTENT_PROCESSING, details)

class EmptyContentError(ContentProcessingError):
    """Error raised when a page has no substantive content."""
    
    def __init__(self, url: str, content_length: int = 0, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["content_length"] = content_length
        super().__init__(f"Page has insufficient content: {content_length} chars", url, details)

class ParseError(ContentProcessingError):
    """Error raised when HTML parsing fails."""
    
    def __init__(self, url: str, parser_name: str = "BeautifulSoup", 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["parser_name"] = parser_name
        super().__init__(f"Failed to parse content using {parser_name}", url, details)

class ChunkingError(ContentProcessingError):
    """Error raised when content segmentation fails."""
    
    def __init__(self, url: str, content_length: Optional[int] = None, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if content_length is not None:
            details["content_length"] = content_length
        super().__init__(f"Failed to chunk content", url, details)

# Database Errors
class DatabaseError(BaseError):
    """Base class for database-related errors."""
    
    def __init__(self, message: str, query: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if query:
            details["query"] = query
        super().__init__(message, ErrorCategory.DATABASE, details)

class ConnectionError(BaseError):
    """Error raised when a connection fails."""
    
    def __init__(self, message: str, target: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if target:
            details["target"] = target
        super().__init__(message, ErrorCategory.CONNECTION, details)

# Task Scheduling Errors
class TaskSchedulingError(BaseError):
    """Base class for task scheduling errors."""
    
    def __init__(self, message: str, task_id: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if task_id:
            details["task_id"] = task_id
        super().__init__(message, ErrorCategory.TASK_SCHEDULING, details)

class FuturesError(TaskSchedulingError):
    """Error raised when there's an issue with concurrent.futures."""
    
    def __init__(self, message: str, task_id: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, task_id, details)

# API Errors
class APIRateLimitError(BaseError):
    """Error raised when an API rate limit is hit."""
    
    def __init__(self, message: str, provider: str, 
                 retry_after: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["provider"] = provider
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(message, ErrorCategory.API_RATE_LIMIT, details)

class EmbeddingError(BaseError):
    """Error raised during embedding generation."""
    
    def __init__(self, message: str, model: Optional[str] = None,
                 input_length: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if model:
            details["model"] = model
        if input_length is not None:
            details["input_length"] = input_length
        super().__init__(message, ErrorCategory.EMBEDDING, details) 