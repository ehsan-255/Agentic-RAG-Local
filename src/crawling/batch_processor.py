from typing import List, Dict, Any, Callable, Awaitable, TypeVar, Optional
import asyncio
import time
from openai import AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

# Import monitoring utilities
from src.utils.task_monitoring import (
    TaskType,
    monitor_executor,
    create_monitored_thread_pool
)
from src.utils.api_monitoring import monitor_openai_call
from src.utils.enhanced_logging import (
    enhanced_api_logger,
    enhanced_crawler_logger
)

T = TypeVar('T')

class BatchProcessor:
    """
    A utility for processing items in batches, particularly useful for API calls.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent_batches: int = 5,
        retry_attempts: int = 6,
        min_backoff: int = 1,
        max_backoff: int = 60
    ):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Maximum number of items to process in a batch
            max_concurrent_batches: Maximum number of concurrent batch operations
            retry_attempts: Maximum number of retry attempts for failed operations
            min_backoff: Minimum backoff delay in seconds
            max_backoff: Maximum backoff delay in seconds
        """
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.retry_attempts = retry_attempts
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # Track performance metrics
        self.total_processed = 0
        self.successful_batches = 0
        self.failed_batches = 0
        self.batch_processing_times = []
    
    async def process_batches(
        self,
        items: List[T],
        processor_fn: Callable[[List[T]], Awaitable[List[Any]]],
        item_callback: Optional[Callable[[T, Any], Awaitable[None]]] = None
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            processor_fn: Function to process a batch of items
            item_callback: Optional callback for each processed item
            
        Returns:
            List[Any]: List of results for all processed items
        """
        # Prepare batches
        batches = [items[i:i+self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        # Process batches with concurrency control
        async def process_batch(batch: List[T]) -> List[Any]:
            async with self.semaphore:
                start_time = time.time()
                try:
                    result = await self._retry_batch(batch, processor_fn)
                    
                    # Update performance metrics
                    duration = time.time() - start_time
                    self.batch_processing_times.append(duration)
                    self.successful_batches += 1
                    
                    # Log success
                    enhanced_crawler_logger.debug(
                        f"Successfully processed batch of {len(batch)} items",
                        batch_size=len(batch),
                        duration=duration,
                        total_successful=self.successful_batches
                    )
                    
                    return result
                except Exception as e:
                    # Update metrics
                    self.failed_batches += 1
                    
                    # Log failure
                    enhanced_crawler_logger.structured_error(
                        f"Batch processing failed: {e}",
                        error=e,
                        batch_size=len(batch),
                        items_processed=self.total_processed
                    )
                    raise
        
        # Create tasks for all batches
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        results = []
        for batch_idx, batch_result in enumerate(batch_results):
            batch = batches[batch_idx]
            
            # Skip failed batches
            if isinstance(batch_result, Exception):
                enhanced_crawler_logger.structured_error(
                    f"Batch failed: {batch_result}",
                    error=batch_result,
                    batch_index=batch_idx,
                    batch_size=len(batch)
                )
                # Fill with None values
                results.extend([None] * len(batch))
                continue
            
            # Add results for this batch
            for item_idx, result in enumerate(batch_result):
                item = batch[item_idx]
                results.append(result)
                self.total_processed += 1
                
                # Call item callback if provided
                if item_callback:
                    await item_callback(item, result)
        
        return results
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6)
    )
    async def _retry_batch(
        self,
        batch: List[T],
        processor_fn: Callable[[List[T]], Awaitable[List[Any]]]
    ) -> List[Any]:
        """
        Process a batch with retry logic.
        
        Args:
            batch: Batch of items to process
            processor_fn: Function to process the batch
            
        Returns:
            List[Any]: Results for the processed batch
        """
        return await processor_fn(batch)


class EmbeddingBatchProcessor:
    """
    A specialized batch processor for generating embeddings.
    """
    
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model: str = "text-embedding-3-small",
        batch_size: int = 10,
        max_concurrent_batches: int = 3
    ):
        """
        Initialize the embedding batch processor.
        
        Args:
            openai_client: AsyncOpenAI client
            model: Embedding model to use
            batch_size: Maximum number of texts to embed in a batch
            max_concurrent_batches: Maximum number of concurrent batch operations
        """
        self.openai_client = openai_client
        self.model = model
        self.processor = BatchProcessor(
            batch_size=batch_size,
            max_concurrent_batches=max_concurrent_batches
        )
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        async def process_batch(batch: List[str]) -> List[List[float]]:
            """Process a batch of texts to generate embeddings."""
            try:
                start_time = time.time()
                response = await self._create_embeddings(batch)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log API call metrics
                enhanced_api_logger.record_api_request(
                    api_name="OpenAI",
                    endpoint="embeddings",
                    duration_ms=duration_ms,
                    batch_size=len(batch),
                    model=self.model
                )
                
                return [item.embedding for item in response.data]
            except Exception as e:
                enhanced_api_logger.structured_error(
                    f"Error generating embeddings: {e}",
                    error=e,
                    batch_size=len(batch),
                    model=self.model
                )
                raise
        
        # Process all texts in batches
        embeddings = await self.processor.process_batches(texts, process_batch)
        return embeddings
        
    @monitor_openai_call("embeddings")
    async def _create_embeddings(self, batch: List[str]):
        """Monitored wrapper for creating embeddings."""
        return await self.openai_client.embeddings.create(
            model=self.model,
            input=batch
        )


class LLMBatchProcessor:
    """
    A specialized batch processor for LLM tasks like title and summary generation.
    """
    
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        batch_size: int = 5,
        max_concurrent_batches: int = 2
    ):
        """
        Initialize the LLM batch processor.
        
        Args:
            openai_client: AsyncOpenAI client
            model: LLM model to use
            batch_size: Maximum number of texts to process in a batch
            max_concurrent_batches: Maximum number of concurrent batch operations
        """
        self.openai_client = openai_client
        self.model = model
        self.processor = BatchProcessor(
            batch_size=batch_size,
            max_concurrent_batches=max_concurrent_batches
        )
    
    async def generate_titles_and_summaries(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Generate titles and summaries for a list of text chunks.
        
        Args:
            chunks: List of dictionaries containing 'content' and 'url' keys
            
        Returns:
            List[Dict[str, str]]: List of dictionaries with 'title' and 'summary' keys
        """
        async def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, str]]:
            """Process a batch of chunks to generate titles and summaries."""
            # Create one request per chunk
            tasks = []
            for chunk in batch:
                task = self._generate_title_and_summary(chunk["content"], chunk["url"])
                tasks.append(task)
            
            # Run all tasks concurrently
            return await asyncio.gather(*tasks)
        
        # Process all chunks in batches
        results = await self.processor.process_batches(chunks, process_batch)
        return results
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6)
    )
    async def _generate_title_and_summary(self, content: str, url: str) -> Dict[str, str]:
        """
        Generate a title and summary for a text chunk.
        
        Args:
            content: Text content
            url: URL of the page
            
        Returns:
            Dict[str, str]: Dictionary with title and summary
        """
        # Use much shorter content and minimal prompting to reduce tokens
        truncated_content = content[:800]
        
        try:
            start_time = time.time()
            response = await self._create_chat_completion(truncated_content)
            duration_ms = (time.time() - start_time) * 1000
            
            # Log API call metrics
            enhanced_api_logger.record_api_request(
                api_name="OpenAI",
                endpoint="chat.completions",
                duration_ms=duration_ms,
                content_length=len(truncated_content),
                model=self.model
            )
            
            text = response.choices[0].message.content
            lines = text.strip().split("\n")
            
            title = lines[0].replace("Title: ", "") if lines and "Title:" in lines[0] else "Untitled Document"
            summary = lines[1].replace("Summary: ", "") if len(lines) > 1 and "Summary:" in lines[1] else ""
            
            # Ensure title and summary are not too long
            title = title[:80]
            summary = summary[:150]
            
            return {"title": title, "summary": summary}
        except Exception as e:
            enhanced_api_logger.structured_error(
                f"Error generating title and summary: {e}",
                error=e,
                content_length=len(truncated_content),
                model=self.model,
                url=url
            )
            # Return default values
            return {"title": "Error: Could not generate title", "summary": "Error: Could not generate summary"}
            
    @monitor_openai_call("chat.completions")
    async def _create_chat_completion(self, content: str):
        """Monitored wrapper for creating chat completions."""
        return await self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Extract titles and summaries efficiently."},
                {"role": "user", "content": f"Content: {content}\nTitle (≤80 chars), Summary (≤150 chars):"}
            ],
            temperature=0.3,
            max_tokens=150
        ) 