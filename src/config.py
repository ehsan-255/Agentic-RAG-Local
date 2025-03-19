import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """
    Centralized configuration for the RAG system.
    Handles defaults and environment variable overrides.
    """
    
    # Database Configuration
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "postgres")
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    # Connection Pool Configuration
    DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
    DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Crawler Configuration
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "5000"))
    DEFAULT_MAX_CONCURRENT_CRAWLS = int(os.getenv("DEFAULT_MAX_CONCURRENT_CRAWLS", "3"))
    DEFAULT_MAX_CONCURRENT_API_CALLS = int(os.getenv("DEFAULT_MAX_CONCURRENT_API_CALLS", "5"))
    
    # Retry Configuration
    DEFAULT_RETRY_ATTEMPTS = int(os.getenv("DEFAULT_RETRY_ATTEMPTS", "6"))
    DEFAULT_MIN_BACKOFF = int(os.getenv("DEFAULT_MIN_BACKOFF", "1"))
    DEFAULT_MAX_BACKOFF = int(os.getenv("DEFAULT_MAX_BACKOFF", "60"))
    
    # RAG Configuration
    DEFAULT_MATCH_COUNT = int(os.getenv("DEFAULT_MATCH_COUNT", "5"))
    
    # Batch Processing Configuration
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    EMBEDDING_MAX_CONCURRENT_BATCHES = int(os.getenv("EMBEDDING_MAX_CONCURRENT_BATCHES", "3"))
    LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "5"))
    LLM_MAX_CONCURRENT_BATCHES = int(os.getenv("LLM_MAX_CONCURRENT_BATCHES", "2"))
    
    @classmethod
    def get_db_connection_string(cls) -> str:
        """Get the database connection string."""
        return f"host={cls.DB_HOST} port={cls.DB_PORT} dbname={cls.DB_NAME} user={cls.DB_USER} password={cls.DB_PASSWORD}"
    
    @classmethod
    def get_crawler_config(cls) -> Dict[str, Any]:
        """Get the default crawler configuration."""
        return {
            "chunk_size": cls.DEFAULT_CHUNK_SIZE,
            "max_concurrent_crawls": cls.DEFAULT_MAX_CONCURRENT_CRAWLS,
            "max_concurrent_api_calls": cls.DEFAULT_MAX_CONCURRENT_API_CALLS,
            "retry_attempts": cls.DEFAULT_RETRY_ATTEMPTS,
            "min_backoff": cls.DEFAULT_MIN_BACKOFF,
            "max_backoff": cls.DEFAULT_MAX_BACKOFF,
            "llm_model": cls.LLM_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL
        }
    
    @classmethod
    def get_batch_processors_config(cls) -> Dict[str, Dict[str, Any]]:
        """Get the configuration for batch processors."""
        return {
            "embedding": {
                "model": cls.EMBEDDING_MODEL,
                "batch_size": cls.EMBEDDING_BATCH_SIZE,
                "max_concurrent_batches": cls.EMBEDDING_MAX_CONCURRENT_BATCHES
            },
            "llm": {
                "model": cls.LLM_MODEL,
                "batch_size": cls.LLM_BATCH_SIZE,
                "max_concurrent_batches": cls.LLM_MAX_CONCURRENT_BATCHES
            }
        }

# Create an instance for easy import
config = Config 