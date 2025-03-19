import logging
import os
import sys
from typing import Optional, Dict, Any
import traceback
from datetime import datetime

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create file handler for error logs
os.makedirs('logs', exist_ok=True)
error_handler = logging.FileHandler('logs/errors.log')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add file handler to root logger
logging.getLogger('').addHandler(error_handler)

class Logger:
    """Custom logger that standardizes log formats and levels across the application."""
    
    def __init__(self, name: str):
        """
        Initialize the logger.
        
        Args:
            name: Name of the logger
        """
        self.logger = logging.getLogger(name)
    
    def debug(self, message: str, **kwargs):
        """
        Log a debug message.
        
        Args:
            message: Debug message
            **kwargs: Additional fields to include in the log
        """
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """
        Log an info message.
        
        Args:
            message: Info message
            **kwargs: Additional fields to include in the log
        """
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """
        Log a warning message.
        
        Args:
            message: Warning message
            **kwargs: Additional fields to include in the log
        """
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """
        Log an error message.
        
        Args:
            message: Error message
            exc_info: Exception information (optional)
            **kwargs: Additional fields to include in the log
        """
        if exc_info:
            kwargs['exception'] = str(exc_info)
            kwargs['traceback'] = traceback.format_exc()
            self.logger.error(message, exc_info=True, extra=kwargs)
        else:
            self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """
        Log a critical message.
        
        Args:
            message: Critical message
            exc_info: Exception information (optional)
            **kwargs: Additional fields to include in the log
        """
        if exc_info:
            kwargs['exception'] = str(exc_info)
            kwargs['traceback'] = traceback.format_exc()
            self.logger.critical(message, exc_info=True, extra=kwargs)
        else:
            self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """
        Internal method to standardize log format.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional fields to include in the log
        """
        # Add timestamp to kwargs
        kwargs['timestamp'] = datetime.now().isoformat()
        
        # Log with extra fields
        self.logger.log(level, message, extra=kwargs)


# Create loggers for different components
crawler_logger = Logger('crawler')
db_logger = Logger('database')
api_logger = Logger('api')
ui_logger = Logger('ui')
rag_logger = Logger('rag')

# Convenience function to get a logger for a specific component
def get_logger(component: str) -> Logger:
    """
    Get a logger for a specific component.
    
    Args:
        component: Component name
        
    Returns:
        Logger: Logger for the component
    """
    return Logger(component) 