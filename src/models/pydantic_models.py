"""Pydantic models for data validation and serialization."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class Document(BaseModel):
    """Document model for stored documents."""
    
    id: Optional[int] = None
    title: str
    content: str
    url: str
    embedding_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class DocumentChunk(BaseModel):
    """Document chunk model for vector search."""
    
    id: Optional[int] = None
    document_id: int
    chunk_text: str
    chunk_index: int
    embedding_id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class SearchQuery(BaseModel):
    """Search query model."""
    
    query: str
    limit: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0, le=1.0)


class SearchResult(BaseModel):
    """Search result model."""
    
    document_id: int
    title: str
    url: str
    chunk_text: str
    similarity: float
    
    class Config:
        orm_mode = True


class RAGResponse(BaseModel):
    """RAG response model."""
    
    query: str
    answer: str
    sources: List[SearchResult]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        orm_mode = True 