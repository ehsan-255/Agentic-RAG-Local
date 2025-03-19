from typing import List, Dict, Any, Callable, Awaitable, TypeVar, Optional
import asyncio
from openai import AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
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
                return await self._retry_batch(batch, processor_fn)
        
        # Create tasks for all batches
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results
        results = []
        for batch_idx, batch_result in enumerate(batch_results):
            batch = batches[batch_idx]
            
            # Add results for this batch
            for item_idx, result in enumerate(batch_result):
                item = batch[item_idx]
                results.append(result)
                
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
            response = await self.openai_client.embeddings.create(
                model=self.model,
                input=batch
            )
            return [item.embedding for item in response.data]
        
        # Process all texts in batches
        embeddings = await self.processor.process_batches(texts, process_batch)
        return embeddings


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
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI that extracts precise titles and summaries from documentation chunks."},
                {"role": "user", "content": f"URL: {url}\n\nContent: {content[:4000]}...\n\nGenerate a concise title (max 80 chars) and summary (100-150 chars) for this documentation chunk."}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        text = response.choices[0].message.content
        lines = text.strip().split("\n")
        
        title = lines[0].replace("Title: ", "") if lines and "Title:" in lines[0] else "Untitled Document"
        summary = lines[1].replace("Summary: ", "") if len(lines) > 1 and "Summary:" in lines[1] else "No summary available"
        
        return {"title": title, "summary": summary} 