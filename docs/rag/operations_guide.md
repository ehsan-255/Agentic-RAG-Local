# Operations Guide: RAG System

This guide provides practical instructions for configuring, maintaining, and troubleshooting the Retrieval-Augmented Generation (RAG) system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Configuration](#configuration)
3. [Maintenance Tasks](#maintenance-tasks)
4. [Performance Optimization](#performance-optimization)
5. [Troubleshooting](#troubleshooting)
6. [FAQs](#faqs)

## System Overview

The RAG (Retrieval-Augmented Generation) system is responsible for:

1. **Document Retrieval**: Finding relevant documentation chunks based on user queries
2. **Context Enhancement**: Providing contextually relevant information to the LLM
3. **Response Generation**: Creating comprehensive answers with citations
4. **Conversational Memory**: Maintaining context across multi-turn conversations

The system interacts with:

- PostgreSQL database with pgvector for document storage and retrieval
- OpenAI API for embeddings and text generation
- Streamlit UI for user interaction

## Configuration

### Environment Variables

The RAG system uses these environment variables:

```
# OpenAI API Configuration
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini  # Model for answering questions
EMBEDDING_MODEL=text-embedding-3-small  # Model for generating embeddings

# RAG Configuration
DEFAULT_MATCH_COUNT=5  # Number of document chunks to retrieve
DEFAULT_SIMILARITY_THRESHOLD=0.7  # Minimum similarity score for retrieved documents
DEFAULT_VECTOR_WEIGHT=0.7  # Weight for vector similarity in hybrid search
```

### System Prompt Customization

The RAG agent's behavior is controlled by its system prompt. To modify it:

1. Edit the `system_prompt` variable in `src/rag/rag_expert.py`
2. Adjust instructions based on your specific requirements
3. Consider these customization points:
   - Tone and style of responses
   - Citation format and requirements
   - Handling of uncertain information
   - Domain-specific instructions

### Tool Configuration

The RAG agent has several built-in tools that can be customized:

1. **retrieve_relevant_documentation**: Adjust the default `match_count` to retrieve more or fewer documents
2. **hybrid_search**: Modify the `vector_weight` parameter to balance between vector and text search
3. **list_documentation_pages**: Add pagination options for large documentation sets

## Maintenance Tasks

### Database Maintenance

Regular maintenance tasks to keep the RAG system performing optimally:

1. **Index Maintenance**:
   ```sql
   -- Recommended monthly: Rebuild vector index
   REINDEX INDEX site_pages_embedding_idx;
   ```

2. **Performance Analysis**:
   ```sql
   -- Check vector search performance
   EXPLAIN ANALYZE 
   SELECT id, url, title, metadata, 
          embedding <=> '[0.1, 0.2, ...]'::vector as distance
   FROM site_pages
   ORDER BY distance
   LIMIT 10;
   ```

3. **Database Vacuuming**:
   ```sql
   -- Recommended weekly: Vacuum database
   VACUUM ANALYZE site_pages;
   ```

### Content Refreshing

To keep your RAG system up-to-date with the latest documentation:

1. Recrawl documentation sources regularly (recommended monthly)
2. Update embeddings when changing the embedding model:
   ```python
   from src.db.schema import refresh_embeddings
   
   # Refresh embeddings for a specific source
   await refresh_embeddings(source_id="python_docs")
   ```

3. Clean up outdated content:
   ```python
   from src.db.schema import delete_documentation_source
   
   # Remove deprecated documentation
   await delete_documentation_source(source_id="deprecated_docs")
   ```

## Performance Optimization

### Retrieval Strategy Tuning

Optimize the retrieval performance with these adjustments:

1. **Adjust Match Count**:
   - Increase for more comprehensive but potentially less relevant results
   - Decrease for more focused but potentially incomplete information

2. **Hybrid Search Weighting**:
   - Higher `vector_weight` (0.8-0.9) for conceptual/semantic queries
   - Lower `vector_weight` (0.3-0.5) for keyword-focused or technical queries

3. **Similarity Threshold**:
   - Higher threshold (0.8+) for precision-focused results
   - Lower threshold (0.5-0.6) for recall-focused results

### OpenAI API Optimization

1. **Token Usage**:
   - Use the shortest effective system prompt
   - Filter out irrelevant documents before sending to OpenAI
   - Use the most cost-effective model for your use case

2. **Rate Limit Management**:
   - Implement exponential backoff for rate limit errors
   - Batch embedding requests when possible
   - Monitor API usage to avoid hitting limits

### Response Time Optimization

1. **Database Optimization**:
   - Ensure proper indexes on frequently queried fields
   - Partition large tables by source_id
   - Consider using a connection pool

2. **Caching**:
   - Implement query caching for frequent questions
   - Cache embeddings for common queries
   - Use Redis or similar for fast cache access

## Troubleshooting

### Common Issues

| Problem | Possible Causes | Solution |
|---------|----------------|----------|
| Irrelevant search results | Poor embedding quality<br>Inappropriate chunk size | Adjust chunk size<br>Update embeddings |
| Slow response times | Database performance<br>Rate limiting<br>Large context window | Check indexes<br>Implement caching<br>Reduce match count |
| Missing information | Information not in database<br>Similarity threshold too high | Crawl additional sources<br>Lower similarity threshold |
| Inaccurate answers | Poor quality retrieval<br>System prompt issues | Refine system prompt<br>Improve retrieval quality |
| API rate limiting | Too many concurrent requests<br>Quota exceeded | Implement backoff<br>Increase quota<br>Add request queuing |

### Diagnostic Procedures

#### Retrieval Quality Issues

To diagnose retrieval quality problems:

1. **Inspect retrieved documents**:
   ```python
   # Get raw retrieval results for a query
   embedding = await get_embedding(query, openai_client)
   results = match_site_pages(embedding, match_count=10)
   
   # Print results with similarity scores
   for result in results:
       print(f"URL: {result['url']}")
       print(f"Similarity: {result['similarity']:.4f}")
       print(f"Title: {result['title']}")
       print("---")
   ```

2. **Compare different retrieval methods**:
   ```python
   # Compare vector search vs hybrid search
   vector_results = match_site_pages(embedding, match_count=5)
   hybrid_results = hybrid_search(query, embedding, match_count=5)
   
   # Check result overlap
   vector_urls = {r['url'] for r in vector_results}
   hybrid_urls = {r['url'] for r in hybrid_results}
   overlap = vector_urls.intersection(hybrid_urls)
   
   print(f"Overlap: {len(overlap)}/{len(vector_urls)}")
   ```

#### Response Generation Issues

For problems with the generated responses:

1. **Test with a simplified system prompt**:
   ```python
   # Create a simplified agent for testing
   simple_agent = Agent(
       model,
       system_prompt="You are a helpful assistant. Answer based only on the provided information.",
       deps_type=AgentyRagDeps
   )
   
   # Test with the same inputs
   simple_response = await simple_agent.run(question, deps=deps)
   ```

2. **Manually examine tool execution**:
   ```python
   # Run tools manually to inspect their output
   docs = await retrieve_relevant_documentation(ctx, user_query)
   print(docs)  # Check if this contains the necessary information
   ```

### Logging and Debugging

Enable detailed logging for the RAG system:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rag_system")

# Log detailed information about retrievals
logger.debug(f"Query: {query}")
logger.debug(f"Retrieved {len(results)} documents")
for i, doc in enumerate(results):
    logger.debug(f"Doc {i}: {doc['url']} (Score: {doc['similarity']:.4f})")
```

## FAQs

### General Questions

#### How many documents are typically retrieved for each query?
By default, the system retrieves 5 document chunks for each query, but this can be configured based on your specific requirements.

#### Can the system handle multi-language documentation?
Yes, the embedding models support multiple languages. However, performance may vary for non-English content.

#### How often should documentation be reindexed?
For frequently updated documentation, monthly reindexing is recommended. For stable documentation, quarterly updates are usually sufficient.

### Technical Questions

#### What's the difference between vector search and hybrid search?
Vector search uses only embedding similarity, while hybrid search combines vector similarity with text-based search for better results.

#### How can I optimize chunks for better retrieval?
Optimize chunk size (typically 1000-5000 characters), ensure chunks contain coherent information, and avoid splitting across important sections.

#### How does the system maintain conversation context?
The RAG agent passes previous messages as context to the LLM, allowing it to reference earlier questions and maintain a coherent conversation.

### Optimization Questions

#### How can I reduce OpenAI API costs?
Use smaller models for simpler queries, implement caching, optimize chunk retrieval count, and batch embedding requests when possible.

#### How can I improve response quality?
Refine the system prompt, increase retrieval quality, implement post-processing for citations, and use a model with higher reasoning capabilities.

#### How can I speed up response time?
Optimize database queries, implement caching, reduce the number of documents retrieved, and use connection pooling. 