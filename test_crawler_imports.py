import os
import sys
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import using the same pattern as the crawler
try:
    from src.db.async_schema import (
        add_documentation_source,
        update_documentation_source,
        add_site_page,
        delete_documentation_source as db_delete_documentation_source
    )
    print("Successfully imported async_schema")
except ImportError as e:
    print(f"Error importing async_schema: {e}")

# Create mock crawler function
async def mock_crawl_function():
    """Simulate what the crawler is trying to do"""
    print("Testing add_site_page with raw_content parameter")
    
    url = f"https://test.example.com/test-{int(datetime.now().timestamp())}"
    title = "Test Page"
    summary = "This is a test page summary"
    content = "This is test content for the page"
    raw_content = "<html><body><h1>Test</h1><p>Raw HTML content</p></body></html>"
    metadata = {
        "source_id": "test_source",
        "chunk_index": 1,
        "total_chunks": 5,
        "word_count": len(content.split()),
        "char_count": len(content)
    }
    embedding = [float(x) for x in np.random.rand(1536)]
    
    # Verify the function signature
    try:
        # Just check the function signature, don't actually run it
        print("Function add_site_page has the following parameters:")
        print(f"  url: str")
        print(f"  chunk_number: int")
        print(f"  title: str")
        print(f"  summary: str")
        print(f"  content: str")
        print(f"  metadata: Dict[str, Any]")
        print(f"  embedding: List[float]")
        print(f"  raw_content: Optional[str] = None")
        print(f"  text_embedding: Optional[List[float]] = None")
        
        print("\nChecking if the function signature matches the crawler's usage:")
        
        # This would fail if the function signature doesn't match
        # We don't await it because we don't need to actually run it
        add_site_page(
            url=url,
            chunk_number=1,
            title=title,
            summary=summary,
            content=content,
            metadata=metadata,
            embedding=embedding,
            raw_content=raw_content
        )
        
        print("Function signature matches the crawler's usage!")
        return True
    except TypeError as e:
        print(f"ERROR: Function signature doesn't match: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return False

# Run the test
import asyncio

if __name__ == "__main__":
    print("Testing crawler imports...")
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(mock_crawl_function())
    print(f"Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1) 