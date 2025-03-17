"""
Test the database schema and functions for the Agentic RAG system.

This script tests the database setup, table creation, and vector search functionality.
"""

import os
import sys
import unittest
from dotenv import load_dotenv
import numpy as np

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.schema import (
    setup_database,
    check_pgvector_extension,
    check_tables_exist,
    add_documentation_source,
    add_site_page,
    match_site_pages,
    hybrid_search,
    get_documentation_sources,
    filter_by_metadata,
    get_document_context
)

class TestDatabase(unittest.TestCase):
    """Test the database schema and functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the database for testing."""
        # Load environment variables
        load_dotenv()
        
        # Make sure pgvector is installed
        if not check_pgvector_extension():
            raise unittest.SkipTest("pgvector extension not installed")
            
        # Set up the database
        if not setup_database():
            raise unittest.SkipTest("Failed to set up database")
            
        # Check if tables exist
        tables = check_tables_exist()
        if not all(tables.values()):
            raise unittest.SkipTest("Not all required tables exist")
            
        # Create a test documentation source
        cls.source_id = "test_source"
        cls.source_name = "Test Documentation"
        cls.base_url = "https://example.com/docs"
        
        # Create random embeddings for testing
        cls.test_embedding = list(np.random.rand(1536).astype(float))
        
    def test_01_add_documentation_source(self):
        """Test adding a documentation source."""
        # Delete existing test source if it exists
        source_id = add_documentation_source(
            name=self.source_name,
            source_id=self.source_id,
            base_url=self.base_url,
            configuration={"max_depth": 3}
        )
        
        self.assertIsNotNone(source_id, "Failed to add documentation source")
        
    def test_02_add_site_page(self):
        """Test adding a site page."""
        # Add a test page
        page_id = add_site_page(
            url=f"{self.base_url}/test-page",
            chunk_number=1,
            title="Test Page",
            summary="A test page for the Agentic RAG system",
            content="This is a test page with content about pgvector and PostgreSQL.",
            metadata={"source_id": self.source_id, "type": "test"},
            embedding=self.test_embedding
        )
        
        self.assertIsNotNone(page_id, "Failed to add site page")
        
        # Add more test pages for search testing
        for i in range(2, 6):
            page_id = add_site_page(
                url=f"{self.base_url}/test-page-{i}",
                chunk_number=1,
                title=f"Test Page {i}",
                summary=f"Test page {i} for the Agentic RAG system",
                content=f"This is test page {i} with content about {'PostgreSQL' if i % 2 == 0 else 'vector search'}.",
                metadata={"source_id": self.source_id, "type": "test"},
                embedding=list(np.random.rand(1536).astype(float))
            )
            self.assertIsNotNone(page_id, f"Failed to add site page {i}")
            
        # Add chunks to test document context
        for i in range(2, 5):
            page_id = add_site_page(
                url=f"{self.base_url}/test-page",
                chunk_number=i,
                title="Test Page",
                summary=f"Chunk {i} of the test page",
                content=f"This is chunk {i} of the test page with more content about PostgreSQL.",
                metadata={"source_id": self.source_id, "type": "test"},
                embedding=list(np.random.rand(1536).astype(float))
            )
            self.assertIsNotNone(page_id, f"Failed to add chunk {i}")
        
    def test_03_get_documentation_sources(self):
        """Test getting documentation sources."""
        sources = get_documentation_sources()
        self.assertGreater(len(sources), 0, "No documentation sources found")
        
        # Check if our test source is in the list
        test_source = next((s for s in sources if s["source_id"] == self.source_id), None)
        self.assertIsNotNone(test_source, "Test source not found")
        self.assertEqual(test_source["name"], self.source_name, "Source name doesn't match")
        
    def test_04_match_site_pages(self):
        """Test matching site pages."""
        results = match_site_pages(
            query_embedding=self.test_embedding,
            match_count=5,
            filter_metadata={"source_id": self.source_id}
        )
        
        self.assertGreater(len(results), 0, "No matching pages found")
        
    def test_05_hybrid_search(self):
        """Test hybrid search."""
        results = hybrid_search(
            query_text="PostgreSQL",
            query_embedding=self.test_embedding,
            match_count=5,
            filter_metadata={"source_id": self.source_id}
        )
        
        self.assertGreater(len(results), 0, "No matching pages found in hybrid search")
        
        # Check that the results have the expected fields
        self.assertIn("similarity", results[0], "Similarity score missing")
        self.assertIn("text_rank", results[0], "Text rank missing")
        self.assertIn("combined_score", results[0], "Combined score missing")
        
    def test_06_filter_by_metadata(self):
        """Test filtering by metadata."""
        results = filter_by_metadata(
            query_embedding=self.test_embedding,
            match_count=5,
            source_id=self.source_id,
            doc_type="test"
        )
        
        self.assertGreater(len(results), 0, "No matching pages found with metadata filtering")
        
    def test_07_get_document_context(self):
        """Test getting document context."""
        results = get_document_context(
            page_url=f"{self.base_url}/test-page",
            context_size=2
        )
        
        self.assertGreater(len(results), 1, "No document context found")
        
        # Check if we have chunks from the same document
        self.assertEqual(results[0]["url"], f"{self.base_url}/test-page", "URL doesn't match")
        
        # Check if is_current is set for one of the chunks
        current_chunks = [r for r in results if r["is_current"]]
        self.assertEqual(len(current_chunks), 1, "Should have exactly one current chunk")

if __name__ == "__main__":
    unittest.main() 