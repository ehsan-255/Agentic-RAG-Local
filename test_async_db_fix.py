import asyncio
import os
import sys
import numpy as np
from datetime import datetime

# Fix for Windows event loop policy
if sys.platform.startswith('win'):
    import asyncio
    from asyncio import WindowsSelectorEventLoopPolicy
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the async database functions
from src.db.async_schema import (
    setup_database,
    add_documentation_source,
    get_documentation_sources,
    add_site_page
)

async def test_async_database():
    """Test the fixed async database functions."""
    print("Starting async database test...")

    # Set up the database
    setup_result = await setup_database()
    if not setup_result:
        print("Failed to set up database")
        return False
    print("Database setup successful")

    # Get documentation sources
    sources = await get_documentation_sources()
    if not sources:
        print("No documentation sources found, creating test source")
        # Create a test source
        source_id = f"test_source_{int(datetime.now().timestamp())}"
        source_name = "Test Source"
        source_url = "https://example.com/sitemap.xml"
        
        source_created = await add_documentation_source(
            name=source_name,
            source_id=source_id,
            sitemap_url=source_url,
            configuration={"test": True}
        )
        
        if not source_created:
            print("Failed to create test source")
            return False
        
        # Get the newly created source
        sources = await get_documentation_sources()
    
    # Use the first source for testing
    source = sources[0]
    source_id = source["source_id"]
    print(f"Using source: {source_id}")
    
    # Create test data for a page
    url = f"https://test.example.com/test-{int(datetime.now().timestamp())}"
    title = "Test Page"
    summary = "This is a test page for the async database"
    content = "This is the content of the test page. " * 20
    raw_content = f"<html><body><h1>{title}</h1><p>{content}</p></body></html>"
    metadata = {
        "source_id": source_id,
        "test": True,
        "timestamp": datetime.now().isoformat()
    }
    
    # Create embedding
    embedding = [float(x) for x in np.random.rand(1536)]
    
    print(f"Adding test page: {url}")
    
    # Test add_site_page with raw_content parameter
    try:
        page_id = await add_site_page(
            url=url,
            chunk_number=1,
            title=title,
            summary=summary,
            content=content,
            metadata=metadata,
            embedding=embedding,
            raw_content=raw_content
        )
        
        if page_id:
            print(f"SUCCESS: Added page with ID: {page_id}")
            return True
        else:
            print("FAILED: Could not add page to database")
            return False
    except Exception as e:
        print(f"ERROR: Exception occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

async def main():
    result = await test_async_database()
    if result:
        print("Async database test passed successfully!")
    else:
        print("Async database test failed!")
    return result

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 