# Operations Guide: RAG Component

This guide provides practical instructions for configuring, operating, and troubleshooting the Retrieval-Augmented Generation (RAG) component of the Agentic RAG system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Configuration](#configuration)
3. [Basic Operation](#basic-operation)
4. [Advanced Features](#advanced-features)
5. [Monitoring and Optimization](#monitoring-and-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Frequently Asked Questions](#frequently-asked-questions)

## System Overview

The RAG component is responsible for:

1. **Context Retrieval**: Finding relevant information from the vector database
2. **Context Processing**: Preparing and organizing retrieved information
3. **Response Generation**: Using LLMs to generate accurate responses based on context
4. **Conversation Management**: Maintaining conversation history and context
5. **Tool Integration**: Providing additional capabilities through specialized tools

Key features of the RAG system:

- Agent-based architecture for flexible, tool-augmented responses
- Hybrid search combining vector similarity with text search
- Context management for optimal use of LLM context windows
- Conversation handling for multi-turn interactions
- Multimodal capabilities for handling images and text
- Monitoring and feedback collection for continuous improvement

## Configuration

### Environment Variables

The RAG component is configured through the following environment variables:

```
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_COMPLETION_MODEL=gpt-4-turbo

# RAG Configuration
RAG_MAX_CONTEXT_CHUNKS=10
RAG_VECTOR_WEIGHT=0.7
RAG_HYBRID_SEARCH_ENABLED=true
RAG_CONVERSATION_EXPIRY_MINUTES=60
RAG_MAX_TOKENS=4000
RAG_TEMPERATURE=0.2

# Tools Configuration
RAG_ENABLE_WEB_SEARCH=true
RAG_ENABLE_CODE_EXECUTION=false
RAG_ENABLE_MULTIMODAL=true
```

### Configuration in `config.py`

RAG settings can also be configured in the `src/config.py` file:

```python
# RAG Configuration
RAG = {
    "max_context_chunks": int(os.environ.get("RAG_MAX_CONTEXT_CHUNKS", 10)),
    "vector_weight": float(os.environ.get("RAG_VECTOR_WEIGHT", 0.7)),
    "hybrid_search_enabled": os.environ.get("RAG_HYBRID_SEARCH_ENABLED", "true").lower() == "true",
    "conversation_expiry_minutes": int(os.environ.get("RAG_CONVERSATION_EXPIRY_MINUTES", 60)),
    "max_tokens": int(os.environ.get("RAG_MAX_TOKENS", 4000)),
    "temperature": float(os.environ.get("RAG_TEMPERATURE", 0.2)),
    "enable_web_search": os.environ.get("RAG_ENABLE_WEB_SEARCH", "true").lower() == "true",
    "enable_code_execution": os.environ.get("RAG_ENABLE_CODE_EXECUTION", "false").lower() == "true",
    "enable_multimodal": os.environ.get("RAG_ENABLE_MULTIMODAL", "true").lower() == "true",
}
```

## Basic Operation

### Processing a User Query

To process a user query using the RAG system:

```python
from src.rag.rag_expert import AgentyRagDeps, agentic_rag_expert
from src.db.schema import match_site_pages
from src.utils.embeddings import generate_embedding

async def process_user_query(query_text):
    # Initialize dependencies
    deps = AgentyRagDeps(
        openai_client=openai_client,
        db_client=db_client,
        embedding_generator=embedding_generator
    )
    
    # Generate embedding for the query
    query_embedding = await generate_embedding(query_text)
    
    # Retrieve relevant contexts
    contexts = await match_site_pages(
        query_embedding=query_embedding,
        match_count=5
    )
    
    # Format contexts for the RAG agent
    formatted_contexts = [{
        "content": ctx["content"],
        "url": ctx["url"],
        "title": ctx["title"]
    } for ctx in contexts]
    
    # Process with RAG expert
    response = await agentic_rag_expert(
        query=query_text,
        contexts=formatted_contexts,
        deps=deps
    )
    
    return {
        "response": response,
        "sources": [ctx["url"] for ctx in contexts]
    }
```

### Handling Conversations

To handle multi-turn conversations:

```python
from src.rag.conversation import ConversationManager
from src.rag.rag_expert import AgentyRagDeps, agentic_rag_expert

async def handle_conversation(user_id, query_text):
    # Initialize conversation manager
    conversation_manager = ConversationManager(db_client=db_client)
    
    # Get or create conversation
    conversation_id = await conversation_manager.get_or_create_conversation(user_id)
    
    # Add the user message
    await conversation_manager.add_message(
        conversation_id=conversation_id,
        role="user",
        content=query_text
    )
    
    # Get conversation history
    history = await conversation_manager.get_conversation_history(conversation_id)
    
    # Process the query with conversation context
    deps = AgentyRagDeps(
        openai_client=openai_client,
        db_client=db_client,
        embedding_generator=embedding_generator
    )
    
    # Generate embedding for the query
    query_embedding = await generate_embedding(query_text)
    
    # Retrieve contexts
    contexts = await match_site_pages(
        query_embedding=query_embedding,
        match_count=5
    )
    
    # Format contexts
    formatted_contexts = [{
        "content": ctx["content"],
        "url": ctx["url"],
        "title": ctx["title"]
    } for ctx in contexts]
    
    # Process with RAG expert
    response = await agentic_rag_expert(
        query=query_text,
        contexts=formatted_contexts,
        deps=deps,
        conversation_history=history
    )
    
    # Add the assistant's response
    await conversation_manager.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=response
    )
    
    return {
        "response": response,
        "conversation_id": conversation_id,
        "sources": [ctx["url"] for ctx in contexts]
    }
```

### Using Different Search Strategies

To use different search strategies:

```python
from src.rag.search_strategies import SearchStrategyManager

async def search_with_strategy(query_text, strategy="hybrid"):
    # Initialize search manager
    search_manager = SearchStrategyManager(
        db_client=db_client,
        embedding_generator=embedding_generator
    )
    
    # Generate embedding
    query_embedding = await generate_embedding(query_text)
    
    # Use the specified strategy
    if strategy == "vector":
        results = await search_manager.vector_search(
            query_embedding=query_embedding,
            match_count=5
        )
    elif strategy == "hybrid":
        results = await search_manager.hybrid_search(
            query_text=query_text,
            query_embedding=query_embedding,
            vector_weight=0.7
        )
    elif strategy == "filtered":
        results = await search_manager.filtered_search(
            query_embedding=query_embedding,
            filters={"source_id": "python_docs"}
        )
    else:
        # Auto-select the best strategy
        results = await search_manager.auto_select_strategy(
            query=query_text,
            query_embedding=query_embedding
        )
    
    return results
```

## Advanced Features

### Multimodal RAG

To use multimodal capabilities:

```python
from src.rag.multimodal_rag import MultimodalRAG

async def process_with_image(query_text, image_path):
    # Initialize multimodal RAG
    deps = AgentyRagDeps(
        openai_client=openai_client,
        db_client=db_client,
        embedding_generator=embedding_generator
    )
    
    multimodal_rag = MultimodalRAG(
        deps=deps,
        vision_model="gpt-4-vision-preview"
    )
    
    # Generate embedding for the text query
    query_embedding = await generate_embedding(query_text)
    
    # Retrieve contexts
    contexts = await match_site_pages(
        query_embedding=query_embedding,
        match_count=5
    )
    
    # Format contexts
    formatted_contexts = [{
        "content": ctx["content"],
        "url": ctx["url"],
        "title": ctx["title"]
    } for ctx in contexts]
    
    # Process with image
    response = await multimodal_rag.process_query_with_image(
        query=query_text,
        image_path=image_path,
        contexts=formatted_contexts
    )
    
    return response
```

### Using Tool-Augmented Responses

To enable tool usage in responses:

```python
from src.rag.tools import WebSearchTool, CodeExecutionTool
from src.rag.rag_expert import AgentyRagDeps, agentic_rag_expert

async def process_with_tools(query_text):
    # Create tools
    web_search = WebSearchTool()
    code_execution = CodeExecutionTool()
    
    # Initialize dependencies with tools
    deps = AgentyRagDeps(
        openai_client=openai_client,
        db_client=db_client,
        embedding_generator=embedding_generator,
        tools=[web_search, code_execution]
    )
    
    # Generate embedding
    query_embedding = await generate_embedding(query_text)
    
    # Retrieve contexts
    contexts = await match_site_pages(
        query_embedding=query_embedding,
        match_count=5
    )
    
    # Format contexts
    formatted_contexts = [{
        "content": ctx["content"],
        "url": ctx["url"],
        "title": ctx["title"]
    } for ctx in contexts]
    
    # Process with RAG expert (will use tools as needed)
    response = await agentic_rag_expert(
        query=query_text,
        contexts=formatted_contexts,
        deps=deps
    )
    
    return response
```

### Context Reranking

To improve retrieval quality with reranking:

```python
from src.rag.context_manager import ContextManager

async def get_reranked_context(query_text):
    # Initialize context manager
    context_manager = ContextManager(
        db_client=db_client,
        embedding_generator=embedding_generator
    )
    
    # Get contexts
    contexts = await context_manager.get_context_for_query(
        query=query_text,
        match_count=10  # Get more than needed for reranking
    )
    
    # Rerank contexts
    reranked_contexts = await context_manager.rerank_contexts(
        query=query_text,
        contexts=contexts,
        reranker="relevance"  # Options: relevance, diversity, recency
    )
    
    # Keep only the top 5 after reranking
    top_contexts = reranked_contexts[:5]
    
    return top_contexts
```

## Monitoring and Optimization

### Performance Monitoring

Monitor the performance of the RAG system:

```python
from src.monitoring.metrics import MetricsCollector
from src.monitoring.rag_metrics import record_rag_query

async def monitored_query(query_text, user_id=None):
    # Initialize metrics collector
    metrics = MetricsCollector()
    
    # Start timing
    metrics.start_timer("rag_query")
    
    try:
        # Process the query
        result = await process_user_query(query_text)
        
        # Record success
        metrics.record_success("rag_query")
        
        # Record metrics
        await record_rag_query({
            "query": query_text,
            "user_id": user_id,
            "processing_time_ms": metrics.get_elapsed_ms("rag_query"),
            "context_count": len(result["sources"]),
            "response_length": len(result["response"]),
            "status": "success"
        })
        
        return result
    except Exception as e:
        # Record failure
        metrics.record_failure("rag_query", str(e))
        
        # Record metrics
        await record_rag_query({
            "query": query_text,
            "user_id": user_id,
            "processing_time_ms": metrics.get_elapsed_ms("rag_query"),
            "status": "failure",
            "error": str(e)
        })
        
        # Re-raise or handle as needed
        raise
```

### Response Quality Feedback

Collect and use feedback to improve responses:

```python
from src.rag.feedback import RelevanceFeedback

async def record_user_feedback(query_id, result_id, is_relevant, user_id=None):
    # Initialize feedback processor
    feedback = RelevanceFeedback(db_client=db_client)
    
    # Record the feedback
    await feedback.record_feedback(
        query_id=query_id,
        result_id=result_id,
        is_relevant=is_relevant,
        user_id=user_id
    )
    
    # Update feedback statistics
    await feedback.update_feedback_stats()
    
    return {"status": "feedback_recorded"}

async def get_improved_results(query_text, user_id=None):
    # Initialize feedback processor
    feedback = RelevanceFeedback(db_client=db_client)
    
    # Generate embedding
    query_embedding = await generate_embedding(query_text)
    
    # Get results with feedback-based improvements
    results = await feedback.get_results_with_feedback(
        query=query_text,
        query_embedding=query_embedding,
        user_id=user_id
    )
    
    return results
```

### Resource Usage Optimization

Optimize token usage and API costs:

```python
from src.rag.context_manager import ContextManager
from src.rag.utils import count_tokens

async def token_optimized_query(query_text, max_tokens=4000):
    # Initialize context manager
    context_manager = ContextManager(
        db_client=db_client,
        embedding_generator=embedding_generator,
        max_tokens=max_tokens
    )
    
    # Get context optimized for token usage
    contexts = await context_manager.get_token_optimized_context(
        query=query_text,
        max_token_count=max_tokens
    )
    
    # Process with RAG agent
    deps = AgentyRagDeps(
        openai_client=openai_client,
        db_client=db_client,
        embedding_generator=embedding_generator
    )
    
    response = await agentic_rag_expert(
        query=query_text,
        contexts=contexts,
        deps=deps
    )
    
    # Calculate token usage
    prompt_tokens = count_tokens("\n".join([ctx["content"] for ctx in contexts]) + query_text)
    completion_tokens = count_tokens(response)
    
    return {
        "response": response,
        "sources": [ctx["url"] for ctx in contexts],
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Possible Cause | Solution |
|-------|----------|----------------|----------|
| No relevant results | "I don't have specific information about that" responses | Missing content in vector DB, poor embeddings | Add more documentation sources, adjust search parameters, use hybrid search |
| Slow response times | Responses take >10 seconds | Inefficient context retrieval, large context windows | Optimize vector search, reduce context size, implement context caching |
| Hallucinations | Responses contain incorrect information | Insufficient context, LLM making things up | Increase retrieval quality, adjust system prompt, use higher quality models |
| Out of context window | Error about token limit exceeded | Too much context being sent to LLM | Implement context pruning, use more efficient tokens, summarize context |
| Tool failures | Error in response about tool execution | Tool misconfiguration, permission issues | Check tool configuration, verify API keys, implement better error handling |
| Irrelevant responses | Response doesn't answer the question | Poor context retrieval, ambiguous query | Improve search strategy, implement query reformulation, use query expansion |

### Diagnostic Tools

#### Check RAG System Status

```python
from src.rag.diagnostics import check_rag_system_health

async def verify_rag_system():
    status = await check_rag_system_health()
    
    print(f"OpenAI API: {'✅ Connected' if status['openai_api_available'] else '❌ Not Available'}")
    print(f"Database: {'✅ Connected' if status['database_available'] else '❌ Not Available'}")
    print(f"Vector Search: {'✅ Working' if status['vector_search_working'] else '❌ Not Working'}")
    print(f"Embedding Generation: {'✅ Working' if status['embedding_generation_working'] else '❌ Not Working'}")
    print(f"LLM Generation: {'✅ Working' if status['llm_generation_working'] else '❌ Not Working'}")
    
    for tool, tool_status in status['tools'].items():
        print(f"Tool '{tool}': {'✅ Available' if tool_status else '❌ Not Available'}")
    
    return status
```

#### Test Query Processing

```python
from src.rag.diagnostics import test_query_processing

async def test_query(query_text):
    test_result = await test_query_processing(query_text)
    
    print(f"Query: {test_result['query']}")
    print(f"Embedding Generation: {test_result['embedding_time_ms']}ms")
    print(f"Context Retrieval: {test_result['retrieval_time_ms']}ms")
    print(f"LLM Processing: {test_result['llm_time_ms']}ms")
    print(f"Total Time: {test_result['total_time_ms']}ms")
    print(f"Context Token Count: {test_result['context_token_count']}")
    print(f"Response Token Count: {test_result['response_token_count']}")
    print(f"Token Usage: {test_result['token_usage']}")
    
    print("\nTop Retrieved Contexts:")
    for i, ctx in enumerate(test_result['contexts'][:3]):
        print(f"{i+1}. {ctx['title']} (similarity: {ctx['similarity']:.4f})")
    
    return test_result
```

#### Run Evaluation Suite

```python
from src.rag.diagnostics import run_rag_evaluation

async def evaluate_rag_system():
    eval_results = await run_rag_evaluation()
    
    print(f"Evaluation completed. Overall score: {eval_results['overall_score']:.2f}/5.0")
    
    print("\nMetrics:")
    print(f"Relevance: {eval_results['metrics']['relevance']:.2f}/5.0")
    print(f"Accuracy: {eval_results['metrics']['accuracy']:.2f}/5.0")
    print(f"Completeness: {eval_results['metrics']['completeness']:.2f}/5.0")
    print(f"Citation Quality: {eval_results['metrics']['citation_quality']:.2f}/5.0")
    
    print("\nStrengths:")
    for strength in eval_results['strengths']:
        print(f"- {strength}")
    
    print("\nAreas for Improvement:")
    for area in eval_results['improvement_areas']:
        print(f"- {area}")
    
    return eval_results
```

## Frequently Asked Questions

### General Questions

**Q: How can I improve the quality of RAG responses?**

A: To improve response quality:
1. Add more comprehensive documentation sources
2. Use hybrid search instead of pure vector search
3. Implement context reranking to prioritize the most relevant information
4. Adjust the system prompt to provide better guidance to the LLM
5. Use a more capable LLM model (e.g., GPT-4 instead of 3.5)
6. Collect and incorporate user feedback

**Q: How do I optimize token usage and reduce API costs?**

A: To optimize token usage:
1. Implement context pruning to remove less relevant information
2. Use efficient embedding models (e.g., text-embedding-3-small)
3. Cache frequently used contexts and responses
4. Adjust chunk size during document processing
5. Use the hybrid search with a higher text search weight for simpler queries
6. Monitor and optimize token usage with the `count_tokens` utility

**Q: How do I handle sensitive or private data in the RAG system?**

A: For sensitive data handling:
1. Implement proper authentication and authorization checks
2. Use fine-tuned models deployed in your own environment
3. Avoid sending sensitive data as part of the context
4. Implement data masking for personally identifiable information (PII)
5. Keep logs and monitoring free of sensitive data
6. Consider local deployment of all components for maximum data privacy

### Technical Questions

**Q: How can I modify the RAG agent's behavior?**

A: To modify the agent's behavior:
1. Create a custom RAG agent by extending `BaseRagAgent`
2. Customize the system prompt to guide the agent's responses
3. Add or modify tools available to the agent
4. Implement custom context processing strategies
5. Adjust temperature and other generation parameters

**Q: How do I create a custom search strategy?**

A: To create a custom search strategy:
1. Create a new class that extends `BaseSearchStrategy`
2. Implement the `search` method with your custom logic
3. Register the strategy with the `SearchStrategyManager`
4. Use the strategy through the manager's interface

**Q: How do I handle long conversations with context limitations?**

A: To handle long conversations:
1. Use the `ConversationManager` to track the full conversation history
2. Implement conversation summarization for long exchanges
3. Use a sliding window approach to maintain recent messages
4. Prioritize the most relevant messages to the current query
5. Implement memory mechanisms to recall important information from earlier in the conversation

### Troubleshooting Questions

**Q: Why are my vector searches not returning relevant results?**

A: This could be due to:
1. Poor quality embeddings - check embedding model
2. Insufficient or irrelevant data in the vector database
3. Suboptimal search parameters (try adjusting `match_count` and `vector_weight`)
4. Using pure vector search when hybrid search might be better
5. Need for query reformulation or expansion

**Q: Why is the RAG agent taking too long to respond?**

A: Slow responses might be caused by:
1. Inefficient vector search - check database indexes
2. Too many context documents retrieved - reduce `match_count`
3. Large token counts - implement context pruning
4. Network latency - check API connection quality
5. Bottlenecks in the processing pipeline - use the diagnostic tools to identify slow components

**Q: Why is the RAG agent sometimes providing incorrect information?**

A: Incorrect information (hallucinations) might be due to:
1. Insufficient or incorrect context information
2. LLM "making up" answers when unsure
3. Contradictory information in the retrieved contexts
4. Need for better instruction in the system prompt
5. Need for a more capable model or lower temperature setting 