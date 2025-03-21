# Developer Guide: RAG Component

This guide provides technical documentation for developers who need to integrate with, extend, or modify the Retrieval-Augmented Generation (RAG) component of the Agentic RAG system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Integration Points](#integration-points)
4. [Advanced Features](#advanced-features)
5. [Extending the System](#extending-the-system)
6. [Best Practices](#best-practices)

## Architecture Overview

The RAG system uses an agent-based architecture that retrieves relevant information from the vector database and generates responses using a large language model:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  User Query     │       │   RAG Agent     │       │  LLM Service    │
│  Interface      │──────▶│   (Controller)  │──────▶│  (OpenAI/etc.)  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                   │
                                   │
                                   ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Generated      │       │   Context       │◀─────▶│   Vector DB     │
│  Response       │◀──────│   Manager       │       │   (PostgreSQL)  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

The architecture follows these key principles:

1. **Agent-Based Design**: Intelligent agents that can use tools and reasoning
2. **Semantic Search**: Vector similarity search for finding relevant context
3. **Context Augmentation**: Dynamic retrieval and management of context information
4. **Tool-Based Architecture**: Modular tools that agents can use to solve tasks
5. **Multi-Turn Conversation**: Support for conversational context across interactions

## Key Components

### 1. RAG Agent (`src/rag/rag_expert.py`)

The RAG agent serves as the central controller for all RAG operations:

```python
from src.rag.rag_expert import AgentyRagDeps, agentic_rag_expert

# Create dependencies for the agent
deps = AgentyRagDeps(
    openai_client=openai_client,
    db_client=db_client,
    embedding_generator=embedding_generator
)

# Use the RAG agent to process a query with context
response = await agentic_rag_expert(
    query="How do I implement connection pooling?",
    contexts=[
        {"content": "Connection pooling reduces overhead...", 
         "url": "https://docs.example.com/pooling", 
         "title": "Connection Pooling"}
    ],
    deps=deps
)

print(response)
```

### 2. Context Manager (`src/rag/context_manager.py`)

The context manager handles retrieval and organization of relevant context:

```python
from src.rag.context_manager import ContextManager

# Create a context manager
context_manager = ContextManager(
    db_client=db_client,
    embedding_generator=embedding_generator,
    max_context_chunks=10
)

# Retrieve context for a query
context = await context_manager.get_context_for_query(
    query="Python async functions",
    vector_weight=0.7,  # Weight for vector similarity vs text search
    source_filter=["python_docs"]  # Optional filter by source
)

# Add conversation history to context
context_manager.add_conversation_history(
    [{"role": "user", "content": "How do I use async in Python?"},
     {"role": "assistant", "content": "Python's async functions use the async/await syntax..."}]
)

# Get enhanced context with conversation history
enhanced_context = context_manager.get_enhanced_context()
```

### 3. Search Strategy Manager (`src/rag/search_strategies.py`)

The search strategy manager handles different search approaches:

```python
from src.rag.search_strategies import SearchStrategyManager

# Create a search strategy manager
search_manager = SearchStrategyManager(
    db_client=db_client,
    embedding_generator=embedding_generator
)

# Use different search strategies
vector_results = await search_manager.vector_search(
    query_embedding=query_embedding,
    match_count=5
)

hybrid_results = await search_manager.hybrid_search(
    query_text="Python async functions",
    query_embedding=query_embedding,
    vector_weight=0.7
)

filtered_results = await search_manager.filtered_search(
    query_embedding=query_embedding,
    filters={"source_id": "python_docs"}
)

# Use automatic strategy selection
results = await search_manager.auto_select_strategy(
    query="How do I use async/await in Python?",
    conversation_history=conversation_history
)
```

### 4. Response Generator (`src/rag/response_generator.py`)

The response generator creates contextually informed responses:

```python
from src.rag.response_generator import ResponseGenerator

# Create a response generator
response_generator = ResponseGenerator(
    openai_client=openai_client,
    model="gpt-4-turbo"
)

# Generate a response
response = await response_generator.generate_response(
    query="How do I implement connection pooling?",
    contexts=contexts,
    conversation_history=conversation_history,
    max_tokens=1000
)

# Generate a response with citations
response_with_citations = await response_generator.generate_response_with_citations(
    query="How do I implement connection pooling?",
    contexts=contexts,
    conversation_history=conversation_history
)
```

### 5. Multi-Modal RAG (`src/rag/multimodal_rag.py`)

Support for multimodal interactions:

```python
from src.rag.multimodal_rag import MultimodalRAG

# Create a multimodal RAG processor
multimodal_rag = MultimodalRAG(
    deps=deps,
    vision_model="gpt-4-vision-preview"
)

# Process a query with an image
response = await multimodal_rag.process_query_with_image(
    query="What's wrong with this code?",
    image_path="/path/to/screenshot.png",
    contexts=contexts
)

# Process a query with multiple images
response = await multimodal_rag.process_query_with_images(
    query="Compare these diagrams",
    image_paths=["/path/to/diagram1.png", "/path/to/diagram2.png"],
    contexts=contexts
)
```

## Integration Points

### 1. With API Component

Integration with the API component:

```python
from fastapi import Depends, FastAPI, HTTPException
from src.rag.rag_expert import AgentyRagDeps, agentic_rag_expert
from src.db.schema import match_site_pages

app = FastAPI()

# Dependency to get RAG dependencies
async def get_rag_deps():
    # Create dependencies
    return AgentyRagDeps(
        openai_client=openai_client,
        db_client=db_client,
        embedding_generator=embedding_generator
    )

@app.post("/api/query")
async def process_query(
    query_request: QueryRequest,
    deps: AgentyRagDeps = Depends(get_rag_deps)
):
    # Generate embedding for the query
    query_embedding = await deps.embedding_generator.generate_embedding(
        query_request.query
    )
    
    # Retrieve relevant contexts
    contexts = await match_site_pages(
        query_embedding=query_embedding,
        match_count=5
    )
    
    # Process with RAG expert
    response = await agentic_rag_expert(
        query=query_request.query,
        contexts=[{
            "content": ctx["content"],
            "url": ctx["url"],
            "title": ctx["title"]
        } for ctx in contexts],
        deps=deps
    )
    
    return {"response": response}
```

### 2. With UI Component

Integration with the UI component:

```python
import streamlit as st
from src.rag.rag_expert import AgentyRagDeps, agentic_rag_expert
from src.db.schema import match_site_pages

# Initialize dependencies
deps = AgentyRagDeps(
    openai_client=openai_client,
    db_client=db_client,
    embedding_generator=embedding_generator
)

# Streamlit UI
st.title("RAG Query Interface")

# Get user query
query = st.text_input("Enter your question:")

if query and st.button("Submit"):
    with st.spinner("Processing..."):
        # Generate embedding for the query
        query_embedding = deps.embedding_generator.generate_embedding(query)
        
        # Retrieve contexts
        contexts = match_site_pages(
            query_embedding=query_embedding,
            match_count=5
        )
        
        # Format contexts for RAG
        formatted_contexts = [{
            "content": ctx["content"],
            "url": ctx["url"],
            "title": ctx["title"]
        } for ctx in contexts]
        
        # Get RAG response
        response = agentic_rag_expert(
            query=query,
            contexts=formatted_contexts,
            deps=deps
        )
        
        # Display response
        st.write(response)
        
        # Show sources
        st.subheader("Sources")
        for ctx in formatted_contexts:
            st.markdown(f"- [{ctx['title']}]({ctx['url']})")
```

### 3. With Monitoring Component

Integration with the monitoring component:

```python
from src.rag.rag_expert import AgentyRagDeps, agentic_rag_expert
from src.monitoring.metrics import record_rag_metrics, MetricsCollector

# Create metrics collector
metrics_collector = MetricsCollector()

# Process a query with monitoring
async def process_with_monitoring(query, contexts):
    # Start timing
    metrics_collector.start_timer("rag_processing")
    
    # Create dependencies
    deps = AgentyRagDeps(
        openai_client=openai_client,
        db_client=db_client,
        embedding_generator=embedding_generator
    )
    
    # Process query
    try:
        response = await agentic_rag_expert(
            query=query,
            contexts=contexts,
            deps=deps
        )
        
        # Record success
        metrics_collector.record_success("rag_processing")
        
        # Record metrics
        await record_rag_metrics({
            "query_length": len(query),
            "context_count": len(contexts),
            "response_length": len(response),
            "processing_time": metrics_collector.get_elapsed("rag_processing"),
            "context_tokens": sum(len(ctx["content"].split()) for ctx in contexts)
        })
        
        return response
    except Exception as e:
        # Record failure
        metrics_collector.record_failure("rag_processing", str(e))
        raise
```

## Advanced Features

### 1. Context Management Strategies

Different strategies for managing context:

```python
from src.rag.context_manager import ContextManager

# Create context manager with different strategies
context_manager = ContextManager(
    db_client=db_client,
    embedding_generator=embedding_generator,
    strategy="hybrid"  # Options: vector, hybrid, adaptive
)

# Adaptive context retrieval
contexts = await context_manager.get_context_adaptive(
    query="Python async functions",
    conversation_history=conversation_history,
    max_contexts=10
)

# Context reranking
reranked_contexts = await context_manager.rerank_contexts(
    query="Python async functions",
    contexts=contexts,
    reranker="relevance"  # Options: relevance, diversity, recency
)

# Context pruning
pruned_contexts = context_manager.prune_contexts(
    contexts=contexts,
    max_tokens=4000  # Target token limit
)
```

### 2. Tooling for Agents

Tools available to RAG agents:

```python
from src.rag.tools import (
    WebSearchTool,
    CodeExecutionTool,
    DocumentationLookupTool,
    CalculationTool
)

# Create tools
web_search = WebSearchTool()
code_execution = CodeExecutionTool()
docs_lookup = DocumentationLookupTool(db_client=db_client)
calculator = CalculationTool()

# Configure RAG agent with tools
deps = AgentyRagDeps(
    openai_client=openai_client,
    db_client=db_client,
    embedding_generator=embedding_generator,
    tools=[web_search, code_execution, docs_lookup, calculator]
)

# Process a query that might use tools
response = await agentic_rag_expert(
    query="What is 2 + 2 and how do I implement a quicksort in Python?",
    contexts=[],
    deps=deps
)
```

### 3. Relevance Feedback

Using relevance feedback to improve results:

```python
from src.rag.feedback import RelevanceFeedback

# Create a relevance feedback processor
feedback_processor = RelevanceFeedback(db_client=db_client)

# Record user feedback
await feedback_processor.record_feedback(
    query_id="abc123",
    result_id="xyz789",
    is_relevant=True,
    user_id="user456"
)

# Get improved results using feedback
improved_results = await feedback_processor.get_results_with_feedback(
    query="Python async functions",
    query_embedding=query_embedding,
    user_id="user456"
)
```

## Extending the System

### 1. Creating a Custom Search Strategy

Create a custom search strategy:

```python
from src.rag.search_strategies import BaseSearchStrategy

class SemanticClusteringStrategy(BaseSearchStrategy):
    """A search strategy that returns diverse results from semantic clusters."""
    
    async def search(self, query_embedding, match_count=5, **kwargs):
        # Implement cluster-based search
        query = """
        WITH initial_matches AS (
            SELECT 
                id, url, title, content, metadata,
                1 - (embedding <=> %s) AS similarity
            FROM site_pages
            WHERE 1 - (embedding <=> %s) > 0.7
            ORDER BY similarity DESC
            LIMIT 20
        ),
        clusters AS (
            SELECT 
                id, url, title, content, metadata, similarity,
                NTILE(5) OVER (ORDER BY embedding <=> %s) AS cluster
            FROM initial_matches
        )
        SELECT id, url, title, content, metadata, similarity
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY cluster ORDER BY similarity DESC) AS rn
            FROM clusters
        ) ranked
        WHERE rn = 1
        ORDER BY similarity DESC
        LIMIT %s;
        """
        
        params = [query_embedding, query_embedding, query_embedding, match_count]
        results = await self.db_client.execute_query(query, params)
        return results

# Register the strategy
search_manager.register_strategy("semantic_clustering", SemanticClusteringStrategy(db_client))

# Use the custom strategy
results = await search_manager.search(
    strategy="semantic_clustering",
    query_embedding=query_embedding,
    match_count=5
)
```

### 2. Adding a Custom Agent

Create a custom RAG agent:

```python
from src.rag.rag_expert import BaseRagAgent

class SpecializedRagAgent(BaseRagAgent):
    """A specialized RAG agent for a particular domain."""
    
    async def process_query(self, query, contexts, **kwargs):
        # Generate system prompt
        system_prompt = self._generate_specialized_prompt()
        
        # Process the query with specialized instructions
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_query_with_contexts(query, contexts)}
        ]
        
        # Generate response with the LLM
        response = await self.deps.openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _generate_specialized_prompt(self):
        # Create a specialized prompt for this domain
        return """You are an expert in technical documentation...
        [specialized instructions here]
        """
    
    def _format_query_with_contexts(self, query, contexts):
        # Format the query and contexts in a specialized way
        formatted_query = f"Question: {query}\n\nRelevant Information:\n"
        for i, ctx in enumerate(contexts):
            formatted_query += f"[{i+1}] {ctx['title']}\n{ctx['content']}\n\n"
        return formatted_query

# Use the specialized agent
specialized_agent = SpecializedRagAgent(deps=deps)
response = await specialized_agent.process_query(
    query="How do I implement connection pooling?",
    contexts=contexts
)
```

### 3. Implementing Custom Context Processing

Create custom context processing:

```python
from src.rag.context_processor import BaseContextProcessor

class StructuredContextProcessor(BaseContextProcessor):
    """A context processor that structures context by sections."""
    
    def process_contexts(self, contexts, query):
        # Group contexts by section
        sections = {}
        for ctx in contexts:
            section = ctx.get("metadata", {}).get("section", "General")
            if section not in sections:
                sections[section] = []
            sections[section].append(ctx)
        
        # Process each section
        processed_contexts = []
        for section, ctx_list in sections.items():
            # Summarize section content
            section_summary = self._summarize_section(ctx_list)
            
            # Add top contexts from each section
            processed_contexts.extend(self._select_top_contexts(ctx_list, 2))
            
            # Add section summary
            processed_contexts.append({
                "title": f"{section} Summary",
                "content": section_summary,
                "url": "",
                "is_summary": True
            })
        
        return processed_contexts
    
    def _summarize_section(self, contexts):
        # Implement section summarization
        combined_text = "\n".join(ctx["content"] for ctx in contexts)
        # ... implement summarization logic ...
        return summary
    
    def _select_top_contexts(self, contexts, count):
        # Select top contexts from a section
        return sorted(contexts, key=lambda x: x.get("similarity", 0), reverse=True)[:count]

# Use the custom processor
context_manager = ContextManager(
    db_client=db_client,
    embedding_generator=embedding_generator,
    context_processor=StructuredContextProcessor()
)
```

## Best Practices

### 1. Context Size Management

Efficient context management:

```python
from src.rag.context_manager import ContextManager
from src.rag.utils import count_tokens

# Create a token-aware context manager
context_manager = ContextManager(
    db_client=db_client,
    embedding_generator=embedding_generator,
    max_tokens=4000
)

# Efficient context retrieval
contexts = await context_manager.get_context_for_query(
    query="Python async functions",
    max_token_count=4000
)

# Check token usage
token_count = count_tokens("\n".join(ctx["content"] for ctx in contexts))
print(f"Using {token_count} tokens for context")

# Progressively reduce context if needed
if token_count > 4000:
    contexts = context_manager.reduce_context_size(
        contexts,
        target_tokens=4000,
        strategy="smart_truncate"  # Options: truncate, summarize, prioritize
    )
```

### 2. Error Handling

Implement robust error handling:

```python
from src.rag.rag_expert import agentic_rag_expert
from src.rag.fallbacks import handle_context_retrieval_failure, handle_llm_failure

async def process_query_with_error_handling(query):
    try:
        # Generate embedding
        query_embedding = await deps.embedding_generator.generate_embedding(query)
        
        try:
            # Retrieve contexts
            contexts = await match_site_pages(
                query_embedding=query_embedding,
                match_count=5
            )
        except Exception as e:
            # Handle context retrieval failure
            contexts = handle_context_retrieval_failure(query, e)
            logger.structured_error(
                "Context retrieval failed",
                error=str(e),
                query=query
            )
        
        try:
            # Generate response
            response = await agentic_rag_expert(
                query=query,
                contexts=contexts,
                deps=deps
            )
            return response
        except Exception as e:
            # Handle LLM failure
            response = handle_llm_failure(query, contexts, e)
            logger.structured_error(
                "LLM generation failed",
                error=str(e),
                query=query
            )
            return response
    except Exception as e:
        # Handle catastrophic failure
        logger.critical(
            "Query processing failed completely",
            error=str(e),
            query=query
        )
        return "I'm sorry, I'm having trouble processing your request right now."
```

### 3. Performance Optimization

Optimize RAG system performance:

```python
from src.rag.rag_expert import agentic_rag_expert
import asyncio

# Batch embedding generation
async def batch_embed_queries(queries):
    """Generate embeddings for multiple queries in parallel."""
    tasks = [deps.embedding_generator.generate_embedding(q) for q in queries]
    return await asyncio.gather(*tasks)

# Parallel context retrieval
async def retrieve_multiple_contexts(query_embeddings, match_count=5):
    """Retrieve contexts for multiple query embeddings in parallel."""
    tasks = [match_site_pages(
        query_embedding=emb,
        match_count=match_count
    ) for emb in query_embeddings]
    return await asyncio.gather(*tasks)

# Cache frequently used contexts
from functools import lru_cache

@lru_cache(maxsize=1000)
async def cached_context_retrieval(query_hash, match_count=5):
    """Cache context retrieval results to avoid redundant database queries."""
    # Use the query hash to retrieve the original query and embedding
    query_data = await get_query_by_hash(query_hash)
    
    # Retrieve contexts
    contexts = await match_site_pages(
        query_embedding=query_data["embedding"],
        match_count=match_count
    )
    
    return contexts
```

### 4. Security Considerations

Implement security best practices:

```python
from src.rag.security import sanitize_query, validate_response

# Sanitize and validate input
def process_secure_query(raw_query, user_id):
    # Sanitize the query
    sanitized_query = sanitize_query(raw_query)
    
    # Check for injection attempts or malicious content
    if is_potentially_malicious(sanitized_query):
        logger.warn(
            "Potentially malicious query detected",
            query=sanitized_query,
            user_id=user_id
        )
        return "I cannot process this query due to security concerns."
    
    # Process the query
    response = agentic_rag_expert(
        query=sanitized_query,
        contexts=contexts,
        deps=deps
    )
    
    # Validate the response for security issues
    validated_response = validate_response(response)
    
    return validated_response

# Limit rate of requests
from src.utils.rate_limiter import RateLimiter

# Create a rate limiter
rate_limiter = RateLimiter(
    max_requests=10,
    time_window=60  # 10 requests per minute
)

# Use rate limiting
async def rate_limited_rag(query, user_id):
    # Check if user is allowed to make a request
    if not rate_limiter.allow_request(user_id):
        return "Rate limit exceeded. Please try again later."
    
    # Process the query
    response = await agentic_rag_expert(
        query=query,
        contexts=contexts,
        deps=deps
    )
    
    return response
```

### 5. Conversation Management

Manage multi-turn conversations:

```python
from src.rag.conversation import ConversationManager

# Create a conversation manager
conversation_manager = ConversationManager(
    db_client=db_client,
    expiry_minutes=60  # Conversation expires after 60 minutes of inactivity
)

# Start or continue a conversation
conversation_id = await conversation_manager.get_or_create_conversation(user_id)

# Add a user message
await conversation_manager.add_message(
    conversation_id=conversation_id,
    role="user",
    content="How do I use async in Python?"
)

# Get conversation history
history = await conversation_manager.get_conversation_history(conversation_id)

# Process a query in conversation context
response = await agentic_rag_expert(
    query="How do I use async in Python?",
    contexts=contexts,
    deps=deps,
    conversation_history=history
)

# Add the assistant's response
await conversation_manager.add_message(
    conversation_id=conversation_id,
    role="assistant",
    content=response
)

# Summarize the conversation
summary = await conversation_manager.summarize_conversation(conversation_id)
``` 