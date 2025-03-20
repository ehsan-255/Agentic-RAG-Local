# Developer Guide: RAG System

This guide provides technical documentation for developers who need to integrate with, extend, or modify the Retrieval-Augmented Generation (RAG) system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Integration Points](#integration-points)
4. [Agent Customization](#agent-customization)
5. [Extending the System](#extending-the-system)
6. [Best Practices](#best-practices)

## Architecture Overview

The RAG system is designed to retrieve relevant information from a vector database and generate comprehensive responses using a large language model:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  User Query     │       │   Vector DB     │       │     OpenAI      │
│  (Question)     │──────▶│   (pgvector)    │──────▶│   API (GPT-4)   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   RAG Agent     │       │  Document       │       │   Response      │
│ (rag_expert.py) │◀─────▶│  Retrieval      │◀─────▶│   Generation    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

The architecture follows these key principles:

1. **Agent-Based Design**: Utilizing a structured agent-based approach with tools
2. **Semantic Search**: Using vector embeddings for semantic similarity
3. **Context Augmentation**: Enhancing LLM responses with relevant documentation
4. **Tool-Based Architecture**: Modular tools for different operations
5. **Conversational Interface**: Support for multi-turn conversations

## Key Components

### 1. RAG Agent (`src/rag/rag_expert.py`)

The central agent that coordinates the retrieval and generation processes:

```python
from src.rag.rag_expert import agentic_rag_expert, AgentyRagDeps

# Create dependencies
deps = AgentyRagDeps(openai_client=openai_client)

# Run the agent
response = await agentic_rag_expert.run(
    user_query, 
    deps=deps
)
```

### 2. Embedding Generation (`src/rag/rag_expert.py`)

Generates vector embeddings for semantic search:

```python
from src.rag.rag_expert import get_embedding

# Generate embedding
embedding = await get_embedding(
    text=query_text,
    openai_client=openai_client,
    model="text-embedding-3-small"
)
```

### 3. Document Retrieval Tools

The RAG system provides several tools for retrieving relevant information:

```python
# List available documentation sources
sources = await list_documentation_sources(ctx)

# Retrieve relevant documentation chunks
docs = await retrieve_relevant_documentation(
    ctx, 
    user_query="How to install the package?",
    source_id="python_docs",
    match_count=5
)

# List pages in a documentation source
pages = await list_documentation_pages(ctx, source_id="python_docs")

# Get content of a specific page
content = await get_page_content(ctx, url="https://example.com/docs/installation")
```

### 4. Database Integration

The RAG system interacts with the vector database via schema functions:

```python
from src.db.schema import (
    match_site_pages,
    hybrid_search,
    get_documentation_sources,
    get_page_content
)

# Get relevant documents using vector similarity
results = match_site_pages(
    query_embedding=embedding,
    match_count=5
)

# Perform hybrid search (vector + text)
results = hybrid_search(
    query_text=query,
    query_embedding=embedding,
    vector_weight=0.7
)
```

## Integration Points

### Integrating the RAG Agent in Applications

To integrate the RAG agent into your application:

1. **Initialize Dependencies**: Create an instance of `AgentyRagDeps` with the OpenAI client
2. **Run the Agent**: Call the agent with the user query and dependencies
3. **Process the Response**: Extract the agent's response and present it to the user

Example:

```python
from openai import AsyncOpenAI
from src.rag.rag_expert import agentic_rag_expert, AgentyRagDeps

async def answer_question(question: str):
    # Initialize the OpenAI client
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create agent dependencies
    deps = AgentyRagDeps(openai_client=openai_client)
    
    # Run the agent
    response = await agentic_rag_expert.run(
        question,
        deps=deps
    )
    
    # Return the generated answer
    return response.content
```

### Streaming Responses

For a better user experience, you can stream the agent's responses:

```python
async def stream_response(question: str):
    # Initialize the OpenAI client
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create agent dependencies
    deps = AgentyRagDeps(openai_client=openai_client)
    
    # Stream the agent's response
    async with agentic_rag_expert.run_stream(
        question,
        deps=deps
    ) as result:
        async for text in result.text_deltas():
            # Process each chunk of text as it becomes available
            yield text
```

## Agent Customization

### Modifying the System Prompt

The agent's behavior can be customized by modifying the system prompt:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Create a new model instance
model = OpenAIModel("gpt-4o-mini")

# Create a new agent with a custom system prompt
custom_agent = Agent(
    model,
    system_prompt="""
    You are a specialized documentation assistant focused on Python. 
    Only answer questions related to Python documentation.
    Always include code examples in your responses.
    """,
    deps_type=AgentyRagDeps,
    retries=2
)
```

### Adding Custom Tools

To add new capabilities to the agent, you can create custom tools:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from typing import List, Dict, Any

# Create model and agent
model = OpenAIModel("gpt-4o-mini")
custom_agent = Agent(model, deps_type=AgentyRagDeps)

# Add a custom tool
@custom_agent.tool
async def search_by_topic(ctx: RunContext[AgentyRagDeps], topic: str) -> List[Dict[str, Any]]:
    """
    Search for documentation by topic or category.
    
    Args:
        ctx: Runtime context with dependencies
        topic: The topic to search for
        
    Returns:
        List of relevant documentation pages
    """
    # Implementation
    # ...
    return results
```

## Extending the System

### Creating a Specialized RAG Agent

To create a specialized RAG agent for a specific domain:

1. Define a new system prompt focused on the domain
2. Create custom tools for domain-specific operations
3. Implement any required preprocessing or postprocessing logic

Example for a specialized Python documentation assistant:

```python
# Define dependencies
@dataclass
class PythonRagDeps(AgentyRagDeps):
    python_version: str = "3.11"
    docs_source_id: str = "python_docs"

# Create system prompt
python_system_prompt = """
You are a Python documentation expert. You help users with questions about Python {python_version}.
Always include code examples in your responses.
Citation format: [Python Docs: <section>](<url>)
"""

# Create the agent
python_rag_expert = Agent(
    model,
    system_prompt=python_system_prompt,
    deps_type=PythonRagDeps,
    retries=2
)

# Add specialized tool
@python_rag_expert.tool
async def get_python_example(ctx: RunContext[PythonRagDeps], function_name: str) -> str:
    """Get a Python code example for a specific function."""
    # Implementation
    # ...
    return example
```

### Implementing Custom Retrieval Strategies

You can implement custom retrieval strategies for specific use cases:

```python
async def retrieve_with_reranking(query: str, openai_client: AsyncOpenAI) -> List[Dict[str, Any]]:
    """Retrieval with a reranking step for improved precision."""
    # 1. Get query embedding
    embedding = await get_embedding(query, openai_client)
    
    # 2. Initial retrieval (high recall)
    initial_results = match_site_pages(
        query_embedding=embedding,
        match_count=20,
        similarity_threshold=0.5
    )
    
    # 3. Reranking step
    reranked_results = await rerank_results(query, initial_results, openai_client)
    
    # 4. Return top results after reranking
    return reranked_results[:5]
```

## Best Practices

### Optimizing RAG Performance

1. **Balance Retrieval and Generation**: Find the right balance between retrieval quality and generation
2. **Prompt Engineering**: Refine the system prompt to guide the agent's behavior
3. **Hybrid Search**: Use hybrid search (vector + text) for better retrieval quality
4. **Chunk Sizing**: Optimize chunk size for your specific content
5. **Quality Thresholds**: Use similarity thresholds to filter out low-quality matches

### Ensuring Response Quality

1. **Cite Sources**: Always include citations in responses
2. **Context Awareness**: Maintain context across turns in a conversation
3. **Structured Responses**: Use consistent response formats
4. **Fallback Strategies**: Implement graceful fallbacks when information is not found
5. **Continuous Evaluation**: Regularly evaluate response quality with test sets

### Handling Edge Cases

1. **Long Documents**: Have strategies for handling very long documents (chunking, summarization)
2. **Ambiguous Queries**: Implement clarification strategies for ambiguous queries
3. **Missing Information**: Gracefully handle cases where information is not available
4. **API Limits**: Implement retry and backoff strategies for API limitations
5. **Error Handling**: Provide informative error messages for failures 