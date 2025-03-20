import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

import streamlit as st

from src.utils.enhanced_logging import enhanced_crawler_logger
from src.db.schema import get_documentation_sources, get_processed_urls, get_source_statistics

# Directory for saving crawl configurations
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "crawl_configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

def save_crawl_configuration(config: Dict[str, Any], name: Optional[str] = None) -> str:
    """
    Save a crawl configuration to a file.
    
    Args:
        config: Crawl configuration dict
        name: Optional name for the configuration
        
    Returns:
        str: Path to the saved configuration file
    """
    # Generate a filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = name or f"crawl_config_{timestamp}"
    filename = os.path.join(CONFIG_DIR, f"{config_name}.json")
    
    # Add timestamp
    config['saved_at'] = datetime.now().isoformat()
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    enhanced_crawler_logger.info(
        f"Saved crawl configuration to {filename}",
        config_name=config_name,
        source_id=config.get('source_id'),
        source_name=config.get('source_name')
    )
    
    return filename

def load_crawl_configuration(filename: str) -> Dict[str, Any]:
    """
    Load a crawl configuration from a file.
    
    Args:
        filename: Path to the configuration file
        
    Returns:
        Dict[str, Any]: Loaded configuration
    """
    with open(filename, 'r') as f:
        config = json.load(f)
    
    enhanced_crawler_logger.info(
        f"Loaded crawl configuration from {filename}",
        config_name=os.path.basename(filename),
        source_id=config.get('source_id'),
        source_name=config.get('source_name')
    )
    
    return config

def get_saved_configurations() -> List[str]:
    """
    Get a list of saved configuration filenames.
    
    Returns:
        List[str]: List of filenames
    """
    return [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]

def prepare_resume_crawl(source_id: str) -> Dict[str, Any]:
    """
    Prepare configuration for resuming a crawl.
    
    Args:
        source_id: Source ID to resume crawling
        
    Returns:
        Dict[str, Any]: Configuration for the resumed crawl
    """
    # Get source information
    sources = get_documentation_sources()
    source = next((s for s in sources if s['source_id'] == source_id), None)
    
    if not source:
        enhanced_crawler_logger.error(f"Source {source_id} not found")
        return {}
    
    # Get already processed URLs
    processed_urls = get_processed_urls(source_id)
    
    # Get source statistics
    stats = get_source_statistics(source_id)
    
    # Create resume configuration
    config = {
        'source_id': source_id,
        'source_name': source['name'],
        'sitemap_url': source['base_url'],
        'configuration': source['configuration'],
        'already_processed_urls': processed_urls,
        'pages_processed': stats.get('pages_count', 0),
        'chunks_processed': stats.get('chunks_count', 0),
        'resumed_at': datetime.now().isoformat()
    }
    
    enhanced_crawler_logger.info(
        f"Prepared resume configuration for {source['name']}",
        source_id=source_id,
        already_processed=len(processed_urls)
    )
    
    return config

def initialize_crawl_state():
    """Initialize crawl state in session state."""
    # Initialize crawl control flags
    if 'pause_crawl' not in st.session_state:
        st.session_state.pause_crawl = False
    
    if 'stop_crawl' not in st.session_state:
        st.session_state.stop_crawl = False
    
    # Initialize other state variables
    if 'already_processed_urls' not in st.session_state:
        st.session_state.already_processed_urls = set()
    
    if 'current_crawl_config' not in st.session_state:
        st.session_state.current_crawl_config = None
        
    if 'crawl_start_time' not in st.session_state:
        st.session_state.crawl_start_time = None
        
    if 'crawl_config' not in st.session_state:
        st.session_state.crawl_config = {}

def initialize_crawl_state_without_reset():
    """
    Initialize crawl state variables only if they don't exist yet.
    This prevents disrupting an active crawl during UI refreshes.
    """
    # Initialize crawl control flags without resetting existing values
    if 'pause_crawl' not in st.session_state:
        st.session_state.pause_crawl = False
    
    if 'stop_crawl' not in st.session_state:
        st.session_state.stop_crawl = False
    
    # Initialize other state variables without overwriting
    if 'already_processed_urls' not in st.session_state:
        st.session_state.already_processed_urls = set()
    
    if 'current_crawl_config' not in st.session_state:
        st.session_state.current_crawl_config = None
        
    if 'crawl_start_time' not in st.session_state:
        st.session_state.crawl_start_time = None
        
    if 'crawl_config' not in st.session_state:
        st.session_state.crawl_config = {}

def reset_crawl_state() -> None:
    """Reset crawl state in Streamlit session state."""
    st.session_state.pause_crawl = False
    st.session_state.stop_crawl = False
    st.session_state.crawl_start_time = None
    st.session_state.crawl_config = {} 