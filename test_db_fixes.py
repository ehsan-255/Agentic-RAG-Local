import sys
import os
import random
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from src.db.schema import add_site_page, get_documentation_sources, setup_database
from src.utils.enhanced_logging import enhanced_db_logger as logger

def test_database_operations():
    """Test database operations to verify fixes."""
    print("Starting database test...")
    
    # Initialize the database
    if not setup_database():
        print("Failed to setup database")
        return False
    
    # Get sources
    sources = get_documentation_sources()
    if not sources:
        print("No sources found in database")
        return False
    
    source_id = sources[0]["source_id"]
    print(f"Using source: {source_id}")
    
    # Create a test page
    url = f"https://test.example.com/test-{int(datetime.now().timestamp())}"
    title = "Test Page"
    summary = "This is a test page to verify database fixes"
    content = "This is test content. " * 50  # Make it reasonably long
    
    # Create metadata
    metadata = {
        "source_id": source_id,
        "test": True,
        "timestamp": datetime.now().isoformat()
    }
    
    # Create random embedding - convert to Python list of floats
    embedding = [float(x) for x in np.random.rand(1536)]
    
    print(f"Attempting to add test page: {url}")
    
    # Try to add the page
    page_id = add_site_page(
        url=url,
        chunk_number=1,
        title=title,
        summary=summary,
        content=content,
        metadata=metadata,
        embedding=embedding
    )
    
    if page_id:
        print(f"SUCCESS: Added page with ID: {page_id}")
        return True
    else:
        print("FAILED: Could not add page to database")
        return False

if __name__ == "__main__":
    result = test_database_operations()
    sys.exit(0 if result else 1) 