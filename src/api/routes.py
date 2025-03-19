"""API routes for the Agentic RAG application."""

from fastapi import APIRouter, Depends, HTTPException
from typing import List

from src.models.pydantic_models import SearchQuery, SearchResult, RAGResponse
from src.rag.rag_expert import RAGExpert
from src.utils.error_handling import handle_api_error, ErrorCode

router = APIRouter()
rag_expert = RAGExpert()

@router.post("/search", response_model=List[SearchResult])
async def search_documents(query: SearchQuery):
    """
    Search for documents based on the query.
    """
    try:
        results = await rag_expert.search(
            query.query, 
            limit=query.limit,
            similarity_threshold=query.similarity_threshold
        )
        return results
    except Exception as e:
        raise handle_api_error(
            exception=e,
            user_message="An error occurred while searching for documents. Please try again later.",
            error_code=ErrorCode.RETRIEVAL_ERROR
        )

@router.post("/rag", response_model=RAGResponse)
async def get_rag_response(query: SearchQuery):
    """
    Get RAG response for the query.
    """
    try:
        response = await rag_expert.generate_response(
            query.query, 
            limit=query.limit,
            similarity_threshold=query.similarity_threshold
        )
        return response
    except Exception as e:
        raise handle_api_error(
            exception=e,
            user_message="An error occurred while generating the response. Please try again later.",
            error_code=ErrorCode.RETRIEVAL_ERROR
        )

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"} 