from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import json

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional

from src.db.schema import (
    match_site_pages,
    hybrid_search,
    get_documentation_sources,
    get_page_content as get_db_page_content,
    get_source_statistics
)

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class AgentyRagDeps:
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert documentation assistant with access to various documentation sources through a vector database. 
Your job is to assist with questions by retrieving and explaining information from the documentation.

IMPORTANT RULES:
1. NEVER answer based on your general knowledge if you can't find specific information in the documentation.
2. If you cannot find relevant documentation to answer a question, clearly state: "I cannot find specific 
   information about this in the documentation." Then suggest the user rephrase their question 
   or check the official documentation directly.
3. NEVER make up information or provide speculative answers.
4. When providing answers, ALWAYS cite which documentation page/section the information comes from.
5. Don't ask the user before taking an action like retrieving documentation - just do it.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Remember: Your responses must be STRICTLY based on the documentation you retrieve. If you cannot 
find relevant information, admit this clearly rather than providing a general answer.
"""

agentic_rag_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=AgentyRagDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@agentic_rag_expert.tool
async def list_documentation_sources(ctx: RunContext[AgentyRagDeps]) -> List[Dict[str, Any]]:
    """
    List all available documentation sources in the system.
    
    Returns:
        List[Dict[str, Any]]: List of documentation sources with their details
    """
    try:
        # Get documentation sources from database
        sources = get_documentation_sources()
        
        # Format the response to include only necessary fields
        result = []
        for source in sources:
            result.append({
                "name": source["name"],
                "source_id": source["source_id"],
                "base_url": source["base_url"],
                "created_at": source["created_at"].isoformat() if source["created_at"] else None,
                "last_crawled_at": source["last_crawled_at"].isoformat() if source["last_crawled_at"] else None,
                "pages_count": source["pages_count"],
                "chunks_count": source["chunks_count"]
            })
            
        return result
        
    except Exception as e:
        print(f"Error retrieving documentation sources: {e}")
        return []

@agentic_rag_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[AgentyRagDeps], user_query: str, source_id: Optional[str] = None, match_count: int = 5) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the OpenAI client
        user_query: The user's question or query
        source_id: Optional source ID to limit the search to a specific documentation source
        match_count: Number of matches to return (default: 5)
        
    Returns:
        str: A formatted string with the relevant documentation chunks
    """
    try:
        # Get query embedding
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Prepare filter metadata
        filter_metadata = {}
        if source_id:
            filter_metadata["source_id"] = source_id
        
        # Perform hybrid search (combines vector similarity and text search)
        results = hybrid_search(
            query_text=user_query,
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        
        if not results:
            return "No relevant documentation found. Try refining your query or checking a different documentation source."
        
        # Format the results
        formatted_results = []
        for i, result in enumerate(results):
            # Format the result
            formatted_result = f"--- Document {i+1} ---\n"
            formatted_result += f"Title: {result['title']}\n"
            formatted_result += f"URL: {result['url']}\n"
            formatted_result += f"Similarity: {result['similarity']:.2f}\n"
            formatted_result += f"Content:\n{result['content']}\n"
            
            # Add formatted result to the list
            formatted_results.append(formatted_result)
        
        # Join the formatted results
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        print(f"Error retrieving relevant documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@agentic_rag_expert.tool
async def list_documentation_pages(ctx: RunContext[AgentyRagDeps], source_id: Optional[str] = None) -> List[str]:
    """
    List all pages available in a documentation source.
    
    Args:
        ctx: The context
        source_id: Optional source ID to limit the list to a specific documentation source
        
    Returns:
        List[str]: List of page URLs
    """
    try:
        # Validate source
        if source_id:
            source_stats = get_source_statistics(source_id)
            if not source_stats:
                return []
        
        # Direct SQL query to get distinct URLs
        # Note: In a real implementation, you would use a database function for this
        conn = None
        from src.db.schema import get_connection, release_connection
        try:
            conn = get_connection()
            cur = conn.cursor()
            
            query = """
            SELECT DISTINCT url, title FROM site_pages
            """
            
            params = []
            if source_id:
                query += " WHERE metadata->>'source_id' = %s"
                params.append(source_id)
                
            query += " ORDER BY url"
            
            cur.execute(query, params)
            results = cur.fetchall()
            
            # Format results
            page_list = [f"{url} - {title}" for url, title in results]
            return page_list
            
        finally:
            if conn:
                release_connection(conn)
                
    except Exception as e:
        print(f"Error listing documentation pages: {e}")
        return []

@agentic_rag_expert.tool
async def get_page_content(ctx: RunContext[AgentyRagDeps], url: str, source_id: Optional[str] = None) -> str:
    """
    Get the content of a specific documentation page.
    
    Args:
        ctx: The context
        url: URL of the page to retrieve
        source_id: Optional source ID to limit the search
        
    Returns:
        str: Content of the page
    """
    try:
        # Get page content from database
        chunks = get_db_page_content(url, source_id)
        
        if not chunks:
            return f"Page not found: {url}"
            
        # Sort chunks by chunk number
        chunks.sort(key=lambda x: x["chunk_number"])
        
        # Concatenate chunks
        full_content = ""
        for chunk in chunks:
            full_content += f"--- Chunk {chunk['chunk_number']} ---\n"
            full_content += f"Title: {chunk['title']}\n"
            full_content += f"Content:\n{chunk['content']}\n\n"
            
        return full_content
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

@agentic_rag_expert.tool
async def get_source_info(ctx: RunContext[AgentyRagDeps], source_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a documentation source.
    
    Args:
        ctx: The context
        source_id: ID of the documentation source
        
    Returns:
        Dict[str, Any]: Information about the documentation source
    """
    try:
        # Get source statistics
        source_stats = get_source_statistics(source_id)
        
        if not source_stats:
            return {"error": f"Source not found: {source_id}"}
            
        # Format datetime objects
        if source_stats.get("created_at"):
            source_stats["created_at"] = source_stats["created_at"].isoformat()
            
        if source_stats.get("last_crawled_at"):
            source_stats["last_crawled_at"] = source_stats["last_crawled_at"].isoformat()
            
        return source_stats
        
    except Exception as e:
        print(f"Error retrieving source information: {e}")
        return {"error": str(e)}
