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

<<<<<<< HEAD
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
        # Query Supabase for all documentation sources
        result = ctx.deps.supabase.from_('documentation_sources') \
            .select('name, source_id, base_url, created_at, last_crawled_at, pages_count, chunks_count') \
            .execute()
        
        if not result.data:
            return []
            
        return result.data
        
    except Exception as e:
        print(f"Error retrieving documentation sources: {e}")
        return []

@agentic_rag_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[AgentyRagDeps], user_query: str, source_id: Optional[str] = None, match_count: int = 5) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        source_id: Optional source ID to limit the search to a specific documentation source
        match_count: Number of matches to return (default: 5)
        
    Returns:
        A formatted string containing the most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Prepare filter
        filter_obj = {}
        if source_id:
            filter_obj["source_id"] = source_id
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': match_count,
                'filter': filter_obj
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            # Include source information in the output
            source_info = f"Source: {doc['metadata'].get('source', 'Unknown')}"
            url_info = f"URL: {doc['url']}"
            
            chunk_text = f"""
# {doc['title']}
{source_info}
{url_info}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@agentic_rag_expert.tool
async def list_documentation_pages(ctx: RunContext[AgentyRagDeps], source_id: Optional[str] = None) -> List[str]:
    """
    Retrieve a list of all available documentation pages.
    
    Args:
        ctx: The context including the Supabase client
        source_id: Optional source ID to limit the results to a specific documentation source
        
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Start building the query
        query = ctx.deps.supabase.from_('site_pages').select('url')
        
        # Add source_id filter if provided
        if source_id:
            query = query.eq('metadata->>source_id', source_id)
        
        # Execute the query
        result = query.execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@agentic_rag_expert.tool
async def get_page_content(ctx: RunContext[AgentyRagDeps], url: str, source_id: Optional[str] = None) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        source_id: Optional source ID to limit the results to a specific documentation source
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Start building the query
        query = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number, metadata') \
            .eq('url', url)
        
        # Add source_id filter if provided
        if source_id:
            query = query.eq('metadata->>source_id', source_id)
        
        # Complete the query with ordering
        result = query.order('chunk_number').execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and source information
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        source_name = result.data[0]['metadata'].get('source', 'Unknown Source')
        
        formatted_content = [
            f"# {page_title}",
            f"Source: {source_name}",
            f"URL: {url}",
            ""  # Empty line for spacing
        ]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}" 
=======
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
>>>>>>> ee4b578bf2a45624bbe5312f94b982f7cd411dc1
