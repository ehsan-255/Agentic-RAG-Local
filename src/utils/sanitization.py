import re
import html
from typing import Dict, Any, Optional

def sanitize_html(html_content: str) -> str:
    """
    Sanitize HTML content to remove potentially malicious scripts and content.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        str: Sanitized HTML content
    """
    # Remove script tags and their contents
    sanitized = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html_content, flags=re.DOTALL)
    
    # Remove style tags and their contents
    sanitized = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', sanitized, flags=re.DOTALL)
    
    # Remove iframe tags
    sanitized = re.sub(r'<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>', '', sanitized, flags=re.DOTALL)
    
    # Remove on* event handlers
    sanitized = re.sub(r'\bon\w+\s*=\s*"[^"]*"', '', sanitized)
    sanitized = re.sub(r"\bon\w+\s*=\s*'[^']*'", '', sanitized)
    sanitized = re.sub(r'\bon\w+\s*=\s*\w+', '', sanitized)
    
    # Remove javascript: URLs
    sanitized = re.sub(r'javascript:', 'void:', sanitized, flags=re.IGNORECASE)
    
    return sanitized

def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata by escaping special characters.
    
    Args:
        metadata: Dictionary containing metadata
        
    Returns:
        Dict[str, Any]: Sanitized metadata
    """
    if not metadata:
        return {}
    
    # Create a new sanitized metadata dictionary
    sanitized = {}
    
    for key, value in metadata.items():
        # Sanitize string values
        if isinstance(value, str):
            sanitized[key] = html.escape(value)
        # Recursively sanitize nested dictionaries
        elif isinstance(value, dict):
            sanitized[key] = sanitize_metadata(value)
        # Sanitize lists of strings
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            sanitized[key] = [html.escape(item) for item in value]
        # Keep other types as is
        else:
            sanitized[key] = value
    
    return sanitized

def sanitize_search_query(query: str) -> str:
    """
    Sanitize a search query to prevent SQL injection and other attacks.
    
    Args:
        query: User-provided search query
        
    Returns:
        str: Sanitized search query
    """
    # Remove SQL comment markers
    sanitized = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
    
    # Remove SQL injection patterns
    sanitized = re.sub(r';', '', sanitized)
    sanitized = re.sub(r'\/\*.*?\*\/', '', sanitized, flags=re.DOTALL)
    
    # Remove UNION, SELECT, and other SQL keywords
    sql_keywords = [
        'UNION', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 
        'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE'
    ]
    
    pattern = r'\b(' + '|'.join(sql_keywords) + r')\b'
    sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    # Escape any remaining HTML
    sanitized = html.escape(sanitized)
    
    return sanitized.strip()

def sanitize_url(url: str) -> Optional[str]:
    """
    Sanitize a URL to ensure it uses a safe scheme.
    
    Args:
        url: URL to sanitize
        
    Returns:
        Optional[str]: Sanitized URL or None if invalid
    """
    # Only allow http and https schemes
    if not url.startswith(('http://', 'https://')):
        return None
    
    # Basic URL sanitization 
    # Remove newlines, carriage returns, tabs
    sanitized = re.sub(r'[\r\n\t]', '', url)
    
    # Remove any script tags or javascript:
    if re.search(r'<script|javascript:', sanitized, re.IGNORECASE):
        return None
        
    return sanitized 