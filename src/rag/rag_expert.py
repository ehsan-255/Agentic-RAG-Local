from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI

from src.db.schema import (
    match_site_pages,
    hybrid_search,
    get_documentation_sources,
    get_page_content as get_db_page_content,
    get_source_statistics
)

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')

# Initialize model without temperature parameter
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

# Set up logging
logger = logging.getLogger(__name__)

async def get_structured_response(
    client: AsyncOpenAI, 
    model: str, 
    system_prompt: str, 
    user_prompt: str, 
    default_response: Dict[str, Any],
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Helper function to get structured JSON responses from OpenAI.
    
    Args:
        client: OpenAI client
        model: Model to use
        system_prompt: System prompt
        user_prompt: User prompt
        default_response: Default response to return on failure
        max_retries: Maximum number of retries
        
    Returns:
        Dict: Structured response
    """
    # Ensure JSON is mentioned in prompts
    if "json" not in system_prompt.lower():
        system_prompt += "\n\nProvide your response as a JSON object."
    
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        
        except json.JSONDecodeError:
            # Handle invalid JSON in response
            if attempt < max_retries:
                continue
            return default_response
            
        except Exception as e:
            # Log error and return default
            print(f"Error getting structured response: {e}")
            return default_response

def enforce_token_limit(chunks, max_tokens=25000):
    """
    Limit chunks to stay within token budget while preserving most relevant content.
    
    Args:
        chunks: List of document chunks
        max_tokens: Maximum number of tokens to allow
        
    Returns:
        List: Limited list of chunks that fits within token budget
    """
    # Log input size
    total_input_chars = sum(len(str(chunk.get('content', ''))) for chunk in chunks) if chunks else 0
    total_input_tokens = total_input_chars // 4
    logger.info(f"TOKEN LIMITER: Input size - {len(chunks)} chunks, ~{total_input_tokens} tokens, {total_input_chars} chars")
    
    limited_chunks = []
    total_tokens = 0
    
    # Sort chunks by combined_score or similarity if available
    if chunks and all('combined_score' in chunk for chunk in chunks):
        chunks = sorted(chunks, key=lambda x: x['combined_score'], reverse=True)
        logger.info("TOKEN LIMITER: Sorted chunks by combined_score")
    elif chunks and all('similarity' in chunk for chunk in chunks):
        chunks = sorted(chunks, key=lambda x: x['similarity'], reverse=True)
        logger.info("TOKEN LIMITER: Sorted chunks by similarity")
    
    for chunk in chunks:
        # Estimate tokens in this chunk (rough approximation: ~4 chars per token)
        chunk_tokens = len(str(chunk.get('content', ''))) // 4
        
        if total_tokens + chunk_tokens <= max_tokens:
            # Chunk fits within budget
            limited_chunks.append(chunk)
            total_tokens += chunk_tokens
        else:
            # We've reached the limit
            logger.info(f"TOKEN LIMITER: Token limit reached at {total_tokens} tokens, dropping {len(chunks) - len(limited_chunks)} chunks")
            break
    
    # Log output size
    total_output_chars = sum(len(str(chunk.get('content', ''))) for chunk in limited_chunks) if limited_chunks else 0
    logger.info(f"TOKEN LIMITER: Output size - {len(limited_chunks)} chunks, ~{total_tokens} tokens, {total_output_chars} chars")
    
    return limited_chunks

@dataclass
class AgentyRagDeps:
    openai_client: AsyncOpenAI
    model: str = llm  # Default to the environment variable value

# Original system prompt (preserved for reference)
"""
ORIGINAL_SYSTEM_PROMPT = 
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

# New system prompt with improved decision intelligence
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

RETRIEVAL STRATEGY:
1. ALWAYS start with RAG (retrieve_relevant_documentation) as your first approach.
2. After seeing the RAG results, CAREFULLY ASSESS if they provide sufficient information.
3. Only if the RAG results are clearly insufficient, check available pages.
4. Retrieve additional specific page content ONLY when:
   - The RAG results mention but don't fully explain a critical concept
   - You need specific code examples not found in the initial results
   - The initial results reference other pages that seem highly relevant
5. LIMIT your total retrievals to a maximum of 3 separate operations for any query.

Remember: Your goal is to provide accurate, focused answers based on the documentation.
Quality of information matters more than quantity. In most cases, the initial RAG results
will be sufficient to answer the user's question.
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
async def analyze_query_specificity(ctx: RunContext[AgentyRagDeps], query: str) -> Dict:
    """
    Analyze if a query is requesting specific information or broad knowledge.
    
    Args:
        query: The user's query to analyze
        
    Returns:
        Dict containing specificity score, query type, and recommended search strategy
    """
    system_prompt = """
    You are an expert at analyzing search queries and determining their specificity.
    
    For specific queries (looking for exact code, quotes, examples, or precise information):
    - Score them high (0.7-1.0)
    - Label them as "specific"
    - Identify exactly what information type is being sought
    
    For broad queries (general explanations, overviews, architecture descriptions):
    - Score them low (0.0-0.6)
    - Label them as "broad"
    - Suggest a general search strategy
    
    Provide structured output with specificity_score, query_type, information_targets, and search_strategy.
    """
    
    user_prompt = f"""
    Analyze this search query: "{query}"
    
    Is this query asking for specific information (exact code snippet, quote, formula, precise example),
    or is it asking for broad knowledge/explanation?
    """
    
    default_result = {
        "specificity_score": 0.5,
        "query_type": "broad",
        "information_targets": [],
        "search_strategy": "Use semantic search with broad matching"
    }
    
    return await get_structured_response(
        client=ctx.deps.openai_client,
        model=ctx.deps.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        default_response=default_result
    )

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
async def precision_search(ctx: RunContext[AgentyRagDeps], 
                           query: str, 
                           target_type: str = "code",
                           source_id: Optional[str] = None,
                           match_count: int = 10) -> str:
    """
    Execute high-precision search for specific information types.
    
    Args:
        ctx: The context including the OpenAI client
        query: User's specific query
        target_type: Type of information sought (code, example, quote, formula, etc.)
        source_id: Optional source ID to limit the search to a specific documentation source
        match_count: Number of matches to return (default: 10)
        
    Returns:
        str: A formatted string with the relevant documentation chunks focused on precision
    """
    logger.info(f"PRECISION SEARCH: Starting for query: '{query}', target_type: {target_type}")
    try:
        # Get query embedding
        query_embedding = await get_embedding(query, ctx.deps.openai_client)
        
        # Prepare filter metadata
        filter_metadata = {}
        if source_id:
            filter_metadata["source_id"] = source_id
            logger.info(f"PRECISION SEARCH: Filtering by source_id: {source_id}")
        
        # Different strategies based on target type
        if target_type.lower() in ["code", "syntax"]:
            # For code, use keyword matching with code-specific keywords
            # Extract code-related keywords from the query
            code_keywords = re.findall(r'\b(?:function|class|method|implementation|syntax|code|snippet|example)\b', query, re.IGNORECASE)
            if code_keywords:
                enhanced_query = query + " " + " ".join(code_keywords)
            else:
                enhanced_query = query + " code implementation example"
            logger.info(f"PRECISION SEARCH: Enhanced code query: '{enhanced_query}'")
        elif target_type.lower() in ["quote", "exact"]:
            # For quotes, use exact phrase matching
            enhanced_query = '"' + query.replace('"', '') + '"'
            logger.info(f"PRECISION SEARCH: Enhanced quote query: '{enhanced_query}'")
        else:
            # For other types, use the original query
            enhanced_query = query
            logger.info(f"PRECISION SEARCH: Using original query: '{enhanced_query}'")
        
        # Perform hybrid search with higher match count for precision
        results = hybrid_search(
            query_text=enhanced_query,
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        
        logger.info(f"PRECISION SEARCH: Retrieved {len(results)} results from hybrid_search")
        
        if not results:
            logger.warning(f"PRECISION SEARCH: No results found for '{query}'")
            return f"No relevant {target_type} information found. Try refining your query or checking a different documentation source."
        
        # No token limiting for precision search - we want exact information
        # But still sort by relevance to show most precise matches first
        if all('combined_score' in result for result in results):
            results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
        elif all('similarity' in result for result in results):
            results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        # Calculate and log the total size of results
        total_chars = sum(len(str(result.get('content', ''))) for result in results)
        logger.info(f"PRECISION SEARCH: Total content size: {total_chars} characters (~{total_chars//4} tokens)")
        
        # Format the results with emphasis on precision and source attribution
        formatted_results = []
        for i, result in enumerate(results):
            # Format the result
            formatted_result = f"--- Precise Match {i+1} ---\n"
            formatted_result += f"Source: {result['title']}\n"
            formatted_result += f"URL: {result['url']}\n"
            formatted_result += f"Relevance: {result.get('similarity', 0):.2f}\n"
            formatted_result += f"Content:\n{result['content']}\n"
            
            # Add formatted result to the list
            formatted_results.append(formatted_result)
        
        # Join the formatted results
        formatted_response = "\n\n".join(formatted_results)
        logger.info(f"PRECISION SEARCH: Final formatted response size: {len(formatted_response)} characters")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"PRECISION SEARCH ERROR: {str(e)}", exc_info=True)
        return f"Error retrieving precise {target_type} information: {str(e)}"

@agentic_rag_expert.tool
async def monitor_information_volume(ctx: RunContext[AgentyRagDeps], 
                                  retrieved_info: List[Dict],
                                  original_query: str,
                                  approx_char_limit: int = 800000) -> Dict:
    """
    Monitor information volume and determine if we should switch to summary mode.
    
    Args:
        ctx: The context including the OpenAI client
        retrieved_info: List of retrieved documents/chunks
        original_query: The user's original query
        approx_char_limit: Approximate character limit to stay under
        
    Returns:
        Dict with information about volume and potential refinements
    """
    try:
        # Calculate total character count
        total_chars = sum(len(str(item.get('content', ''))) for item in retrieved_info)
        logger.info(f"MONITOR VOLUME: Analyzing {len(retrieved_info)} chunks with total {total_chars} characters")
        
        # Log each chunk's size to identify potential outliers
        for i, item in enumerate(retrieved_info[:5]):  # Log first 5 chunks for brevity
            content_size = len(str(item.get('content', '')))
            title = item.get('title', 'Untitled')[:50]
            logger.info(f"MONITOR VOLUME: Chunk {i+1} - '{title}...' - {content_size} characters")
        
        if len(retrieved_info) > 5:
            logger.info(f"MONITOR VOLUME: ... and {len(retrieved_info) - 5} more chunks (not logged)")
        
        # If under the limit, no need for further processing
        if total_chars <= approx_char_limit:
            logger.info(f"MONITOR VOLUME: Content within limit ({total_chars} <= {approx_char_limit})")
            return {
                "exceed_limit": False,
                "total_estimated_chars": total_chars,
                "topic_clusters": [],
                "recommended_refinements": []
            }
        
        logger.warning(f"MONITOR VOLUME: Content exceeds limit ({total_chars} > {approx_char_limit})")
        
        # Extract titles and brief content samples
        titles = [item.get('title', 'Untitled') for item in retrieved_info]
        content_samples = [item.get('content', '')[:500] + "..." for item in retrieved_info]
        
        # Generate topic clusters and refinement suggestions
        system_prompt = """
        You are an expert at analyzing search results and helping users refine their queries.
        
        Analyze the titles and content samples from search results. Then:
        1. Identify the main topic clusters present in the results
        2. Generate 3-5 specific refinement questions the user could ask to get more focused information
        
        Return a JSON object with:
        - topic_clusters: List of main topics found in the results
        - recommended_refinements: List of specific questions the user could ask to refine their search
        """
        
        user_prompt = f"""
        The user's original query was: "{original_query}"
        
        This query returned a large amount of information ({total_chars} characters).
        
        Here are the titles and content samples from the search results:
        
        {json.dumps(list(zip(titles, content_samples)), indent=2)}
        
        Based on these results, identify the main topic clusters and suggest specific refinement questions.
        """
        
        logger.info("MONITOR VOLUME: Generating refinement suggestions")
        response = await ctx.deps.openai_client.chat.completions.create(
            model=ctx.deps.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Add the exceed_limit and total_chars information
        result["exceed_limit"] = True
        result["total_estimated_chars"] = total_chars
        
        # Log the refinement suggestions
        logger.info(f"MONITOR VOLUME: Generated {len(result.get('recommended_refinements', []))} refinement suggestions")
        
        return result
        
    except Exception as e:
        logger.error(f"MONITOR VOLUME ERROR: {str(e)}", exc_info=True)
        # Return a safe default if monitoring fails
        return {
            "exceed_limit": total_chars > approx_char_limit,
            "total_estimated_chars": total_chars,
            "topic_clusters": [],
            "recommended_refinements": [
                f"Could you tell me more specifically what aspect of '{original_query}' you're interested in?",
                f"Which part of '{original_query}' is most important to you?",
                f"Are you looking for examples, theory, or implementation details about '{original_query}'?"
            ]
        }

@agentic_rag_expert.tool
async def generate_content_overview(ctx: RunContext[AgentyRagDeps], 
                                 retrieved_info: List[Dict],
                                 original_query: str) -> str:
    """
    Generate a high-level overview of retrieved content.
    
    Args:
        ctx: The context including the OpenAI client
        retrieved_info: List of retrieved documents/chunks
        original_query: The user's original query
        
    Returns:
        str: A concise overview of the main information found
    """
    logger.info(f"CONTENT OVERVIEW: Generating for query '{original_query}' with {len(retrieved_info)} chunks")
    try:
        # Calculate total input size
        total_input_chars = sum(len(str(item.get('content', ''))) for item in retrieved_info)
        logger.info(f"CONTENT OVERVIEW: Total input size: {total_input_chars} characters")
        
        # Extract titles, URLs, and brief content samples
        summaries = []
        for item in retrieved_info[:20]:  # Limit to first 20 items for overview
            title = item.get('title', 'Untitled')
            url = item.get('url', 'No URL')
            content = item.get('content', '')[:300] + "..."  # Brief sample
            
            summaries.append({
                "title": title,
                "url": url,
                "content_sample": content
            })
        
        logger.info(f"CONTENT OVERVIEW: Created {len(summaries)} summary items for processing")
        
        system_prompt = """
        You are an expert at summarizing large amounts of documentation into concise, helpful overviews.
        
        Given a set of documentation chunks:
        1. Identify the main topics and concepts covered
        2. Create a concise overview that helps the user understand what information is available
        3. Organize the information into logical sections
        4. Highlight key resources or documentation pages that seem most important
        
        Keep your overview clear, structured, and focused on helping the user understand what information is available.
        """
        
        user_prompt = f"""
        The user asked: "{original_query}"
        
        This query returned a large amount of information. Create a concise overview of the main information available.
        
        Here are samples from the first 20 chunks:
        
        {json.dumps(summaries, indent=2)}
        
        Please provide a concise overview that helps the user understand what information is available and where they might want to focus their attention.
        """
        
        logger.info("CONTENT OVERVIEW: Generating overview")
        response = await ctx.deps.openai_client.chat.completions.create(
            model=ctx.deps.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        overview = response.choices[0].message.content
        logger.info(f"CONTENT OVERVIEW: Generated overview of {len(overview)} characters")
        
        return overview
        
    except Exception as e:
        logger.error(f"CONTENT OVERVIEW ERROR: {str(e)}", exc_info=True)
        # Return a simple fallback overview if generation fails
        return f"""
        Your query "{original_query}" returned a large amount of information across multiple documents.
        
        The documentation appears to cover various aspects of this topic, including theoretical concepts,
        implementation details, examples, and related topics.
        
        Consider refining your query to focus on a specific aspect you're most interested in.
        """

@agentic_rag_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[AgentyRagDeps], 
                                      user_query: str, 
                                      source_id: Optional[str] = None, 
                                      match_count: int = 5) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    This enhanced version handles both specific and broad queries differently.
    
    Args:
        ctx: The context including the OpenAI client
        user_query: The user's question or query
        source_id: Optional source ID to limit the search to a specific documentation source
        match_count: Number of matches to return (default: 5)
        
    Returns:
        str: A formatted string with the relevant documentation chunks or guidance for refinement
    """
    logger.info(f"RAG RETRIEVAL: Starting for query: '{user_query}', source: {source_id if source_id else 'all sources'}")
    try:
        # First, analyze the query to determine its specificity
        logger.info("RAG RETRIEVAL: Analyzing query specificity")
        query_analysis = await analyze_query_specificity(ctx, user_query)
        logger.info(f"RAG RETRIEVAL: Query analysis - type: {query_analysis.get('query_type', 'unknown')}, score: {query_analysis.get('specificity_score', 0)}")
        
        # For highly specific queries, use precision search
        if query_analysis["query_type"] == "specific" and query_analysis["specificity_score"] > 0.7:
            # Extract the target type from information_targets
            target_types = query_analysis.get("information_targets", [])
            target_type = target_types[0] if target_types else "specific"
            
            logger.info(f"RAG RETRIEVAL: Using precision search for specific query (target_type: {target_type})")
            
            # Use precision search for specific information retrieval
            result = await precision_search(
                ctx,
                user_query,
                target_type=target_type,
                source_id=source_id,
                match_count=match_count * 2  # Double match count for precision searches
            )
            
            logger.info(f"RAG RETRIEVAL: Precision search completed with result size: {len(result)} characters")
            return result
        
        # For broad queries, use the standard approach with monitoring
        logger.info("RAG RETRIEVAL: Using standard approach for broad query")
        
        # Get query embedding
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Prepare filter metadata
        filter_metadata = {}
        if source_id:
            filter_metadata["source_id"] = source_id
        
        # Perform hybrid search (combines vector similarity and text search)
        logger.info(f"RAG RETRIEVAL: Executing hybrid_search with match_count={match_count}")
        results = hybrid_search(
            query_text=user_query,
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        
        logger.info(f"RAG RETRIEVAL: Hybrid search returned {len(results)} results")
        
        if not results:
            logger.warning("RAG RETRIEVAL: No results found")
            return "No relevant documentation found. Try refining your query or checking a different documentation source."
        
        # Log total size of retrieved content
        total_retrieved_chars = sum(len(str(result.get('content', ''))) for result in results)
        logger.info(f"RAG RETRIEVAL: Total retrieved content size: {total_retrieved_chars} characters (~{total_retrieved_chars//4} tokens)")
        
        # Monitor information volume to detect if we're retrieving too much
        logger.info("RAG RETRIEVAL: Monitoring information volume")
        monitoring_result = await monitor_information_volume(ctx, results, user_query)
        
        if monitoring_result["exceed_limit"]:
            logger.warning(f"RAG RETRIEVAL: Content exceeds limit ({monitoring_result['total_estimated_chars']} chars), switching to summary mode")
            
            # Generate content overview
            logger.info("RAG RETRIEVAL: Generating content overview")
            overview = await generate_content_overview(ctx, results, user_query)
            
            # Get refinement suggestions
            refinements = monitoring_result.get("recommended_refinements", [])
            refinement_text = "\n".join([f"- {r}" for r in refinements])
            
            # Prepare the response with overview and refinements
            response = f"""
            Your query returned a large amount of information ({monitoring_result['total_estimated_chars']} characters).
            
            Here's an overview of what I found:
            
            {overview}
            
            To get more specific information, consider refining your question:
            {refinement_text}
            """
            
            logger.info(f"RAG RETRIEVAL: Returning overview response of {len(response)} characters")
            return response
        
        # For manageable results, apply token limiting and return formatted results
        logger.info(f"RAG RETRIEVAL: Applying token limiting to {len(results)} results")
        results = enforce_token_limit(results, max_tokens=25000)
        
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
        formatted_response = "\n\n".join(formatted_results)
        
        # Log final response size
        logger.info(f"RAG RETRIEVAL: Final formatted response size: {len(formatted_response)} characters")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"RAG RETRIEVAL ERROR: {str(e)}", exc_info=True)
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
    logger.info(f"PAGE CONTENT: Retrieving for URL: '{url}', source: {source_id if source_id else 'all sources'}")
    try:
        # Get page content from database
        chunks = get_db_page_content(url, source_id)
        
        logger.info(f"PAGE CONTENT: Retrieved {len(chunks)} chunks from database")
        
        if not chunks:
            logger.warning(f"PAGE CONTENT: Page not found: {url}")
            return f"Page not found: {url}"
        
        # Calculate total content size before limiting
        total_chars_before = sum(len(str(chunk.get('content', ''))) for chunk in chunks)
        logger.info(f"PAGE CONTENT: Total content size before limiting: {total_chars_before} characters")
        
        # Apply token limiting to prevent excessive content
        chunks = enforce_token_limit(chunks, max_tokens=25000)
            
        # Sort chunks by chunk number
        chunks.sort(key=lambda x: x["chunk_number"])
        
        # Concatenate chunks
        full_content = ""
        for chunk in chunks:
            full_content += f"--- Chunk {chunk['chunk_number']} ---\n"
            full_content += f"Title: {chunk['title']}\n"
            full_content += f"Content:\n{chunk['content']}\n\n"
            
        logger.info(f"PAGE CONTENT: Final formatted content size: {len(full_content)} characters")
            
        return full_content
        
    except Exception as e:
        logger.error(f"PAGE CONTENT ERROR: {str(e)}", exc_info=True)
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
