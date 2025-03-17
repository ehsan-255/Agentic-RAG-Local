from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Dict, Any, Optional

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class AgentyRagDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert documentation assistant with access to various documentation sources through a vector database. 
Your job is to assist with questions by retrieving and explaining information from the documentation.

When answering:
1. Use ONLY the provided documentation chunks to answer questions
2. If the documentation doesn't contain the answer, say so clearly
3. Provide specific examples from the documentation when relevant
4. Format code blocks, commands, and technical terms appropriately
5. Cite the source URLs when possible

Be concise but thorough in your explanations.
"""

@Agent(model=model, system_prompt=system_prompt)
async def agentic_rag_expert(
    deps: AgentyRagDeps,
    query: str,
    source_id: Optional[int] = None,
    max_chunks: int = 5,
    similarity_threshold: float = 0.7,
) -> str:
    """
    RAG agent that retrieves relevant documentation chunks and answers questions.
    
    Args:
        deps: Dependencies including Supabase client and OpenAI client
        query: The user's question
        source_id: Optional ID of a specific documentation source to search
        max_chunks: Maximum number of chunks to retrieve
        similarity_threshold: Minimum similarity score for chunks
        
    Returns:
        A comprehensive answer based on the retrieved documentation
    """
    # Generate embedding for the query
    embedding_response = await deps.openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
        encoding_format="float"
    )
    query_embedding = embedding_response.data[0].embedding
    
    # Build the Supabase query
    supabase_query = deps.supabase.table("site_pages").select(
        "id", "url", "title", "content", "metadata"
    )
    
    # Add source filter if specified
    if source_id is not None:
        supabase_query = supabase_query.eq("source_id", source_id)
    
    # Execute the similarity search
    results = supabase_query.execute_raw(
        """
        SELECT id, url, title, content, metadata, 
               1 - (embedding <=> '[{}]') as similarity
        FROM site_pages
        WHERE embedding IS NOT NULL
        {}
        AND 1 - (embedding <=> '[{}]') > {}
        ORDER BY similarity DESC
        LIMIT {}
        """.format(
            ','.join(str(x) for x in query_embedding),
            f"AND source_id = {source_id}" if source_id is not None else "",
            ','.join(str(x) for x in query_embedding),
            similarity_threshold,
            max_chunks
        )
    )
    
    # Process the results
    chunks = results.get("data", [])
    
    if not chunks:
        return "I couldn't find any relevant information in the documentation to answer your question. Could you rephrase or ask something else?"
    
    # Format the context from retrieved chunks
    context = "\n\n---\n\n".join([
        f"Source: {chunk['url']}\nTitle: {chunk.get('title', 'Untitled')}\n\n{chunk['content']}"
        for chunk in chunks
    ])
    
    # Use the context to answer the question
    answer = f"Based on the documentation, I can provide the following answer:\n\n"
    
    # Add information about the sources used
    sources = [f"- {chunk['url']}" for chunk in chunks]
    sources_text = "\n\nSources:\n" + "\n".join(sources)
    
    return f"{answer}{context}{sources_text}"