from typing import List, Dict, Any, Optional
from datetime import datetime
from supabase import Client

class DocumentationSource:
    """Represents a documentation source in the database."""
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all documentation sources."""
        response = self.supabase.table("documentation_sources").select("*").order("name").execute()
        return response.data
    
    def get_by_id(self, source_id: int) -> Optional[Dict[str, Any]]:
        """Get a documentation source by ID."""
        response = self.supabase.table("documentation_sources").select("*").eq("id", source_id).execute()
        if response.data:
            return response.data[0]
        return None
    
    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a documentation source by name."""
        response = self.supabase.table("documentation_sources").select("*").eq("name", name).execute()
        if response.data:
            return response.data[0]
        return None
    
    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new documentation source."""
        response = self.supabase.table("documentation_sources").insert(data).execute()
        return response.data[0]
    
    def update(self, source_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a documentation source."""
        response = self.supabase.table("documentation_sources").update(data).eq("id", source_id).execute()
        return response.data[0]
    
    def delete(self, source_id: int) -> bool:
        """Delete a documentation source."""
        response = self.supabase.table("documentation_sources").delete().eq("id", source_id).execute()
        return len(response.data) > 0


class SitePage:
    """Represents a site page in the database."""
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
    
    def get_by_source(self, source_id: int) -> List[Dict[str, Any]]:
        """Get all pages for a documentation source."""
        response = self.supabase.table("site_pages").select("*").eq("source_id", source_id).execute()
        return response.data
    
    def get_by_url(self, url: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific URL."""
        response = self.supabase.table("site_pages").select("*").eq("url", url).execute()
        return response.data
    
    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new site page."""
        response = self.supabase.table("site_pages").insert(data).execute()
        return response.data[0]
    
    def upsert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a site page."""
        response = self.supabase.table("site_pages").upsert(
            data,
            on_conflict=["source_id", "url", "chunk_index"]
        ).execute()
        return response.data[0]
    
    def delete_by_source(self, source_id: int) -> bool:
        """Delete all pages for a documentation source."""
        response = self.supabase.table("site_pages").delete().eq("source_id", source_id).execute()
        return True
    
    def search_similar(self, query_embedding: List[float], source_id: Optional[int] = None, 
                      limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar pages using vector similarity."""
        query = """
        SELECT id, url, title, content, metadata, 
               1 - (embedding <=> '[{}]') as similarity
        FROM site_pages
        WHERE embedding IS NOT NULL
        {}
        AND 1 - (embedding <=> '[{}]') > {}
        ORDER BY similarity DESC
        LIMIT {}
        """.format(
            ','.join(str(x) for x in query_embedding),
            f"AND source_id = {source_id}" if source_id is not None else "",
            ','.join(str(x) for x in query_embedding),
            threshold,
            limit
        )
        
        results = self.supabase.execute_raw(query)
        return results.get("data", [])