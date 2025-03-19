from urllib.parse import urlparse
import re

def validate_url(url: str) -> tuple[bool, str]:
    """
    Validate a URL for correctness and scheme.
    
    Args:
        url: The URL to validate
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    # Check if URL is empty
    if not url or not url.strip():
        return False, "URL cannot be empty"
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Check scheme
    if not parsed.scheme:
        return False, "URL must include a scheme (http:// or https://)"
    
    if parsed.scheme not in ['http', 'https']:
        return False, f"Invalid URL scheme: {parsed.scheme}. Only http and https are supported."
    
    # Check netloc (domain)
    if not parsed.netloc:
        return False, "URL must include a domain name"
    
    # Check for valid domain format using regex
    domain_pattern = re.compile(r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$')
    if not domain_pattern.match(parsed.netloc):
        return False, f"Invalid domain name: {parsed.netloc}"
    
    return True, ""

def validate_sitemap_url(url: str) -> tuple[bool, str]:
    """
    Validate a sitemap URL and check if it appears to be a sitemap.
    
    Args:
        url: The sitemap URL to validate
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    # First validate the URL format
    is_valid, error = validate_url(url)
    if not is_valid:
        return False, error
    
    # Check if URL ends with common sitemap extensions
    if not any(url.endswith(ext) for ext in ['.xml', 'sitemap.xml', 'sitemap_index.xml']):
        return False, "Warning: URL doesn't appear to be a sitemap (should end with .xml)"
    
    return True, "" 