from __future__ import annotations
from typing import Literal, TypedDict, Dict, Any, Optional, List
import asyncio
import os
import sys
import time
import datetime
import threading
import logging
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import logfire
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from src.rag.rag_expert import agentic_rag_expert, AgentyRagDeps
from src.crawling.enhanced_docs_crawler import CrawlConfig, crawl_documentation, clear_documentation_source
from src.crawling.crawl_state import (
    initialize_crawl_state, 
    reset_crawl_state,
    initialize_crawl_state_without_reset
)
from src.db.schema import (
    get_documentation_sources as db_get_documentation_sources,
    get_source_statistics as db_get_source_statistics,
    add_documentation_source,
    delete_documentation_source,
    setup_database
)
from src.utils.validation import validate_sitemap_url
from src.ui.monitoring_ui import monitoring_dashboard
from src.utils.enhanced_logging import (
    get_session_stats,
    get_active_session,
    get_system_metrics,
    get_error_stats,
    end_crawl_session
)
from src.utils.task_monitoring import (
    get_task_stats,
    get_active_tasks,
    get_failed_tasks,
    cancel_all_tasks,
    TaskState
)
from src.db.connection import get_connection_stats
from src.utils.api_monitoring import (
    get_api_stats,
    get_rate_limits
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

# Import the config module
from src.config import config as app_config

# Change any references to DEFAULT_CHUNK_SIZE, etc. in the file to use app_config instead
DEFAULT_CHUNK_SIZE = app_config.DEFAULT_CHUNK_SIZE
DEFAULT_MAX_CONCURRENT_CRAWLS = app_config.DEFAULT_MAX_CONCURRENT_CRAWLS
DEFAULT_MAX_CONCURRENT_API_CALLS = app_config.DEFAULT_MAX_CONCURRENT_API_CALLS
DEFAULT_MATCH_COUNT = app_config.DEFAULT_MATCH_COUNT
DEFAULT_URL_PATTERNS = [
    '/docs/',
    '/documentation/',
    '/guide/',
    '/manual/',
    '/reference/',
    '/tutorial/',
    '/api/',
    '/learn/'
]

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_with_streaming(user_input: str, source_id: Optional[str] = None):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    
    Args:
        user_input: The user's input text
        source_id: Optional source ID to limit the search to a specific documentation source
    """
    # Prepare dependencies
    deps = AgentyRagDeps(
        openai_client=openai_client
    )

    # If source_id is provided, augment the user_input to include information about the source
    if source_id:
        user_input_with_source = f"{user_input}\n\nNote: Search in documentation source with ID: {source_id}"
    else:
        user_input_with_source = user_input

    # For now, don't use conversation history until we figure out the correct format
    # This will at least get individual queries working

    try:
        # Use the non-streaming run method instead of run_stream since we're having issues with streaming
        placeholder = st.empty()
        placeholder.markdown("Generating response...")
        
        # Run the agent (non-streaming)
        response = await agentic_rag_expert.run(
            user_input_with_source,  # The user prompt
            deps=deps,
            message_history=[],  # Use empty history to avoid format errors
        )
        
        # Get the response content - properly extract from AgentRunResult
        if hasattr(response, 'data'):
            partial_text = response.data
        elif hasattr(response, 'content'):
            partial_text = response.content
        elif hasattr(response, 'text'):
            partial_text = response.text
        elif isinstance(response, str):
            partial_text = response
        else:
            # Last resort: try to convert the entire result to a string
            # But make sure to clean up the representation if it's an AgentRunResult
            str_response = str(response)
            if str_response.startswith("AgentRunResult"):
                # Try to extract the data part using string operations
                try:
                    data_start = str_response.find("data='") + 6  # length of "data='"
                    data_end = str_response.rfind("')")
                    if data_start > 6 and data_end > data_start:
                        partial_text = str_response[data_start:data_end]
                    else:
                        partial_text = str_response
                except:
                    partial_text = str_response
            else:
                partial_text = str_response
            
        # Display the full response at once
        placeholder.markdown(partial_text)
        
        # Update the last message with the complete response
        st.session_state.messages[-1]['content'] = partial_text
        st.session_state.messages[-1]['agent_response'] = partial_text
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        placeholder = st.empty()
        placeholder.error(f"Error generating response: {str(e)}")
        st.session_state.messages[-1]['content'] = "Sorry, there was an error generating a response."
        st.session_state.messages[-1]['agent_response'] = "Sorry, there was an error generating a response."


async def get_documentation_sources() -> List[Dict[str, Any]]:
    """Get all documentation sources from the database."""
    sources = db_get_documentation_sources()
    
    # Format datetime objects for JSON serialization
    for source in sources:
        if source.get("created_at"):
            source["created_at"] = source["created_at"].isoformat()
        if source.get("last_crawled_at"):
            source["last_crawled_at"] = source["last_crawled_at"].isoformat()
    
    return sources


async def crawl_new_documentation(source_name: str, sitemap_url: str, config: Dict[str, Any]) -> bool:
    """
    Crawl a new documentation source.
    
    Args:
        source_name: Name of the documentation source
        sitemap_url: URL of the sitemap to crawl
        config: Crawl configuration
        
    Returns:
        bool: True if the crawl was successful, False otherwise
    """
    try:
        # Create a unique ID for the source
        source_id = f"{source_name.lower().replace(' ', '_')}_{int(time.time())}"
        
        # Add the documentation source to the database
        result = add_documentation_source(
            name=source_name,
            source_id=source_id,
            base_url=sitemap_url,
            configuration=config
        )
        
        if not result:
            st.error(f"Failed to add documentation source: {source_name}")
            return False
        
        # Create the crawl configuration
        crawl_config = CrawlConfig(
            source_id=source_id,
            source_name=source_name,
            sitemap_url=sitemap_url,
            chunk_size=config.get("chunk_size", DEFAULT_CHUNK_SIZE),
            max_concurrent_requests=config.get("max_concurrent_crawls", DEFAULT_MAX_CONCURRENT_CRAWLS),
            max_concurrent_api_calls=config.get("max_concurrent_api_calls", DEFAULT_MAX_CONCURRENT_API_CALLS),
            url_patterns_include=config.get("url_patterns_include", DEFAULT_URL_PATTERNS),
            url_patterns_exclude=config.get("url_patterns_exclude", [])
        )
        
        # Perform the crawl
        success = await crawl_documentation(
            openai_client,
            crawl_config
        )
        
        return success
    except Exception as e:
        st.error(f"Error crawling documentation: {e}")
        return False


def get_documentation_sources_sync():
    """Sync wrapper for getting documentation sources."""
    return db_get_documentation_sources()


def get_source_statistics(source_id):
    """Sync wrapper for getting source statistics."""
    return db_get_source_statistics(source_id)


# Modify the section that starts a crawl to prevent UI disappearing
async def start_crawl_with_status_update():
    """Start a crawl without blocking the UI."""
    # Get form values from session state
    source_name = st.session_state.source_name
    sitemap_url = st.session_state.sitemap_url
    
    # Create config
    config = {
        "chunk_size": st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
        "max_concurrent_crawls": st.session_state.get("max_concurrent_crawls", DEFAULT_MAX_CONCURRENT_CRAWLS),
        "max_concurrent_api_calls": st.session_state.get("max_concurrent_api_calls", DEFAULT_MAX_CONCURRENT_API_CALLS),
        "url_patterns_include": st.session_state.get("url_patterns_include", "").split('\n') if st.session_state.get("url_patterns_include") else [],
        "url_patterns_exclude": st.session_state.get("url_patterns_exclude", "").split('\n') if st.session_state.get("url_patterns_exclude") else []
    }

    try:
        # Create a unique ID for the source
        source_id = f"{source_name.lower().replace(' ', '_')}_{int(time.time())}"
        
        # Add the documentation source to the database
        result = add_documentation_source(
            name=source_name,
            source_id=source_id,
            base_url=sitemap_url,
            configuration=config
        )
        
        if not result:
            st.error(f"Failed to add documentation source: {source_name}")
            return False
        
        # Create the crawl configuration
        crawl_config = CrawlConfig(
            source_id=source_id,
            source_name=source_name,
            sitemap_url=sitemap_url,
            chunk_size=config.get("chunk_size", DEFAULT_CHUNK_SIZE),
            max_concurrent_requests=config.get("max_concurrent_crawls", DEFAULT_MAX_CONCURRENT_CRAWLS),
            max_concurrent_api_calls=config.get("max_concurrent_api_calls", DEFAULT_MAX_CONCURRENT_API_CALLS),
            url_patterns_include=config.get("url_patterns_include", DEFAULT_URL_PATTERNS),
            url_patterns_exclude=config.get("url_patterns_exclude", [])
        )
        
        # Launch the crawl without awaiting - this is the key change to make it non-blocking
        # We create a task but deliberately don't await it so it runs in the background
        asyncio.create_task(crawl_documentation(
            openai_client,
            crawl_config
        ))
        
        # Immediately return success to allow the UI to remain responsive
        return True
        
    except Exception as e:
        st.error(f"Error starting crawl: {str(e)}")
        st.session_state.crawl_completed = True
        st.session_state.crawl_success = False
        return False


# Modify the initiate_crawl function to directly start the crawl instead of setting a flag
def initiate_crawl():
    """Initiate a crawl immediately without relying on session state flags."""
    # Check inputs
    source_name = st.session_state.source_name
    sitemap_url = st.session_state.sitemap_url
    
    if not source_name or not sitemap_url:
        st.error("Documentation name and sitemap URL are required.")
        return
        
    # Validate the sitemap URL
    if not sitemap_url.lower().endswith('.xml') and not 'sitemap' in sitemap_url.lower():
        st.warning("The URL may not be a sitemap. Make sure it points to a sitemap XML file.")
    
    # Create config
    config = {
        "chunk_size": st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
        "max_concurrent_crawls": st.session_state.get("max_concurrent_crawls", DEFAULT_MAX_CONCURRENT_CRAWLS),
        "max_concurrent_api_calls": st.session_state.get("max_concurrent_api_calls", DEFAULT_MAX_CONCURRENT_API_CALLS),
        "url_patterns_include": st.session_state.get("url_patterns_include", "").split('\n') if st.session_state.get("url_patterns_include") else [],
        "url_patterns_exclude": st.session_state.get("url_patterns_exclude", "").split('\n') if st.session_state.get("url_patterns_exclude") else []
    }
    
    # Log the start of the crawl process
    st.info(f"Starting crawl for {source_name}...")
    
    try:
        # Create a unique ID for the source
        source_id = f"{source_name.lower().replace(' ', '_')}_{int(time.time())}"
        
        # Add the documentation source to the database
        result = add_documentation_source(
            name=source_name,
            source_id=source_id,
            base_url=sitemap_url,
            configuration=config
        )
        
        if not result:
            st.error(f"Failed to add documentation source: {source_name}")
            return
        
        # Log successful addition
        st.success(f"Added documentation source: {source_name}")
        
        # Create the crawl configuration
        crawl_config = CrawlConfig(
            source_id=source_id,
            source_name=source_name,
            sitemap_url=sitemap_url,
            chunk_size=config.get("chunk_size", DEFAULT_CHUNK_SIZE),
            max_concurrent_requests=config.get("max_concurrent_crawls", DEFAULT_MAX_CONCURRENT_CRAWLS),
            max_concurrent_api_calls=config.get("max_concurrent_api_calls", DEFAULT_MAX_CONCURRENT_API_CALLS),
            url_patterns_include=config.get("url_patterns_include", DEFAULT_URL_PATTERNS),
            url_patterns_exclude=config.get("url_patterns_exclude", [])
        )
        
        # Start the crawl directly
        # We'll use the synchronous wrapper instead of relying on asyncio.create_task
        # which may not work well with Streamlit's execution model
        st.session_state.crawl_initiated = True
        st.session_state.crawl_config = crawl_config
        
        # Set the active tab to Monitoring
        st.session_state.active_tab = "Monitoring"
        
        # Force a rerun to apply the tab change
        st.rerun()
        
    except Exception as e:
        st.error(f"Error initiating crawl: {str(e)}")
        logger.error(f"Crawl initiation error: {str(e)}", exc_info=True)


# Replace the existing crawl form submit handler
def create_source_form():
    """Create a form for adding a new documentation source."""
    # Check if a crawl is in progress
    crawl_in_progress = get_active_session() is not None
    
    # Before the form
    st.subheader("Select Chunking Method")
    chunking_method = st.radio(
        "Chunking Method",
        options=["Word-based", "Character-based"],
        index=0 if st.session_state.get("use_word_based_chunking", app_config.USE_WORD_BASED_CHUNKING) else 1,
        help="Word-based is recommended for better semantic coherence"
    )

    # Update session state
    st.session_state.use_word_based_chunking = (chunking_method == "Word-based")

    # Then create the form with different input fields based on the selection
    with st.form("add_source_form", clear_on_submit=False):
        # Form fields that don't change
        st.text_input("Documentation Name", key="source_name", disabled=crawl_in_progress)
        st.text_input("Sitemap URL", key="sitemap_url", disabled=crawl_in_progress)
        
        with st.expander("Advanced Options", expanded=True):
            # Chunking options based on previously selected method
            st.subheader("Chunking Settings")
            
            if st.session_state.use_word_based_chunking:
                # Word-based chunking UI
                st.info("Using word-based chunking (recommended for better semantic coherence)")
                word_chunk_size = st.number_input("Chunk Size (words)", 
                    min_value=app_config.MIN_CHUNK_WORDS, 
                    max_value=app_config.MAX_CHUNK_WORDS,
                    value=app_config.DEFAULT_CHUNK_WORDS, 
                    key="chunk_words",
                    help="Target number of words per chunk",
                    disabled=crawl_in_progress)
                
                # Calculate default overlap (25% of chunk size)
                default_overlap = max(25, int(word_chunk_size * 0.25))
                word_overlap = st.number_input("Overlap Size (words)", 
                    min_value=10, max_value=200,
                    value=default_overlap, 
                    key="overlap_words",
                    help="Number of words to overlap between chunks (20-30% recommended)",
                    disabled=crawl_in_progress)
            else:
                # Character-based chunking UI
                st.warning("Using character-based chunking (legacy mode)")
                st.number_input("Chunk Size (characters)", 
                    min_value=1000, max_value=10000,
                    value=app_config.DEFAULT_CHUNK_SIZE, 
                    key="chunk_size",
                    help="Maximum number of characters per chunk",
                    disabled=crawl_in_progress)
            
            # Concurrency settings
            st.subheader("Concurrency Settings")
            st.number_input("Max Concurrent Requests", min_value=1, 
                        value=app_config.DEFAULT_MAX_CONCURRENT_CRAWLS, 
                        key="max_concurrent_crawls",
                        disabled=crawl_in_progress)
                        
            st.number_input("Max Concurrent API Calls", min_value=1, 
                        value=app_config.DEFAULT_MAX_CONCURRENT_API_CALLS, 
                        key="max_concurrent_api_calls",
                        disabled=crawl_in_progress)
            
            # URL patterns
            st.subheader("URL Filtering")
            st.text_area("URL Patterns to Include (one per line)", 
                      value='\n'.join(DEFAULT_URL_PATTERNS),
                      key="url_patterns_include",
                      disabled=crawl_in_progress)
                      
            st.text_area("URL Patterns to Exclude (one per line)", 
                      value="",
                      key="url_patterns_exclude",
                      disabled=crawl_in_progress)
        
        # Submit button
        submit_button = st.form_submit_button("Add Source and Start Crawl")
        
        if submit_button:
            # Validate inputs
            source_name = st.session_state.source_name.strip()
            sitemap_url = st.session_state.sitemap_url.strip()
            
            if not source_name:
                st.error("Documentation name is required")
                return
                
            if not validate_sitemap_url(sitemap_url):
                st.error("Invalid sitemap URL")
                return
            
            # Get form values
            config_dict = {
                "chunk_size": st.session_state.get("chunk_size", app_config.DEFAULT_CHUNK_SIZE) if not st.session_state.use_word_based_chunking else app_config.DEFAULT_CHUNK_SIZE,
                "chunk_words": st.session_state.get("chunk_words", app_config.DEFAULT_CHUNK_WORDS) if st.session_state.use_word_based_chunking else app_config.DEFAULT_CHUNK_WORDS,
                "overlap_words": st.session_state.get("overlap_words", app_config.DEFAULT_OVERLAP_WORDS) if st.session_state.use_word_based_chunking else app_config.DEFAULT_OVERLAP_WORDS,
                "use_word_based_chunking": st.session_state.use_word_based_chunking,
                "max_concurrent_crawls": st.session_state.max_concurrent_crawls,
                "max_concurrent_api_calls": st.session_state.max_concurrent_api_calls,
                "url_patterns_include": [p.strip() for p in st.session_state.url_patterns_include.splitlines() if p.strip()],
                "url_patterns_exclude": [p.strip() for p in st.session_state.url_patterns_exclude.splitlines() if p.strip()]
            }
            
            # Start the crawl
            initiate_crawl_process(source_name, sitemap_url, config_dict)


# Add a monitoring refresh mechanism
def create_monitoring_ui():
    """Create a refreshable monitoring UI without disrupting crawl processes."""
    # Create a container for refresh controls
    with st.container():
        col1, col2 = st.columns([5, 1])
        
        # Fetch fresh data outside of button interaction
        data = fetch_monitoring_data_safe()
        
        with col1:
            # Display timestamp of last update
            st.caption(f"Last updated: {data['last_updated']}")
        
        with col2:
            # Simple manual refresh button with a unique key
            refresh_clicked = st.button("⟳ Refresh", key="manual_refresh_button")
            
            # If refresh clicked, update session state flag only
            if refresh_clicked:
                st.session_state.refresh_requested = True
    
    # Update session state data without triggering rerun
    st.session_state.historical_metrics = data['historical']
    st.session_state.monitoring_data = data
    
    # Display monitoring dashboard with current data
    monitoring_dashboard()


# Separate the data fetching function to avoid conflicts with crawl state
@st.cache_data(ttl=10)
def fetch_monitoring_data_safe():
    """
    Fetch monitoring data without affecting crawl state.
    Uses caching to prevent database connection issues.
    """
    try:
        # Get fresh data
        system_metrics = get_system_metrics()
        task_stats = get_task_stats()
        
        # Get connection stats safely
        try:
            connection_stats = get_connection_stats()
        except:
            connection_stats = {}
            
        # Get API stats safely
        try:
            api_stats = get_api_stats()
        except:
            api_stats = {}
        
        # Get historical metrics from session state or initialize
        if 'historical_metrics' in st.session_state:
            historical = st.session_state.historical_metrics.copy()
        else:
            historical = {
                'timestamps': [],
                'cpu_percent': [],
                'memory_mb': [],
                'active_tasks': [],
                'failed_tasks': [],
                'success_rate': [],
            }
            
        # Current timestamp
        now = datetime.datetime.now()
        
        # Update historical data
        historical['timestamps'].append(now)
        historical['cpu_percent'].append(system_metrics.get('cpu_percent', 0))
        historical['memory_mb'].append(system_metrics.get('memory_rss_mb', 0))
        historical['active_tasks'].append(task_stats.get('running_tasks', 0) + task_stats.get('pending_tasks', 0))
        historical['failed_tasks'].append(task_stats.get('failed_tasks', 0))
        
        # Get session stats for success rate
        session_stats = get_session_stats()
        success_rate = 0
        if session_stats:
            success_rate = session_stats.get('success_rate', 0)
        historical['success_rate'].append(success_rate)
        
        # Only keep the last 100 data points
        max_points = 100
        for key in historical:
            if len(historical[key]) > max_points:
                historical[key] = historical[key][-max_points:]
        
        return {
            "system": system_metrics,
            "tasks": task_stats,
            "db": connection_stats,
            "api": api_stats,
            "last_updated": now.strftime("%Y-%m-%d %H:%M:%S"),
            "historical": historical
        }
    except Exception as e:
        # Return minimal safe data in case of any errors
        return {
            "system": {},
            "tasks": {},
            "db": {},
            "api": {},
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "historical": st.session_state.get('historical_metrics', {
                'timestamps': [],
                'cpu_percent': [],
                'memory_mb': [],
                'active_tasks': [],
                'failed_tasks': [],
                'success_rate': [],
            })
        }


# Improve the cancel crawl functionality
def cancel_current_crawl():
    """Cancel the current crawl with proper status updates."""
    # Set the stop flag for the crawler to detect
    st.session_state.stop_crawl = True
    
    # Check if there is an active crawl session
    active_session = get_active_session()
    
    # Track if we have an active thread
    has_thread = 'crawl_thread' in st.session_state and st.session_state.crawl_thread is not None
    
    # Track crawl status
    has_status = global_state.get_crawl_status()["active"]
    
    # Count how many tasks are cancelled
    st.session_state.tasks_cancelled = global_state.get_cancelled_tasks()
    
    # If we have an active session, end it properly
    if active_session:
        try:
            # Mark the session as complete with cancelled status
            active_session.complete(status="cancelled")
            
            # End the crawl session
            end_crawl_session(active_session.session_id, status="cancelled")
            
            # Record in global state
            global_state.update_crawl_status(
                active=False,
                completed=True,
                success=False,
                end_time=time.time()
            )
            
            # UI feedback
            st.success(f"Crawl cancelled. Stopped {global_state.get_cancelled_tasks()} active tasks.")
            
            # Clean up session state
            reset_crawl_state()
        except Exception as e:
            logger.error(f"Error cancelling crawl session: {str(e)}", exc_info=True)
            st.error(f"Error cancelling crawl: {str(e)}")
    elif has_thread or has_status:
        # If we only have a thread but no session, show appropriate message
        logger.info("Thread active without session, cancellation in progress")
        st.warning(f"No active crawl session found, but thread is running. Cancelled {global_state.get_cancelled_tasks()} tasks.")
        
        # Update global state
        global_state.update_crawl_status(
            active=False,
            completed=True,
            success=False,
            end_time=time.time()
        )
        
        # Reset crawl state
        reset_crawl_state()
    else:
        # No active crawl at all
        st.warning("No active crawl to cancel.")
    
    # Final state reset to be safe
    st.session_state.stop_crawl = True


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a global state for thread-safe operations
# This prevents the ScriptRunContext errors
class GlobalState:
    """Thread-safe global state handler for Streamlit"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalState, cls).__new__(cls)
            cls._instance.lock = threading.Lock()
            cls._instance.crawl_status = {
                "active": False,
                "completed": False,
                "success": False,
                "source_name": None,
                "url": None,
                "start_time": None,
                "end_time": None,
                "error": None
            }
            cls._instance.tasks_cancelled = 0
            cls._instance.update_needed = False
        return cls._instance
    
    def update_crawl_status(self, **kwargs):
        """Thread-safe update of crawl status"""
        with self.lock:
            for key, value in kwargs.items():
                self.crawl_status[key] = value
            self.update_needed = True
    
    def get_crawl_status(self):
        """Thread-safe get of crawl status"""
        with self.lock:
            return self.crawl_status.copy()
    
    def increment_cancelled_tasks(self):
        """Thread-safe increment of cancelled tasks counter"""
        with self.lock:
            self.tasks_cancelled += 1
            self.update_needed = True
    
    def get_cancelled_tasks(self):
        """Thread-safe get of cancelled tasks counter"""
        with self.lock:
            return self.tasks_cancelled
    
    def needs_update(self):
        """Check if UI needs to be updated"""
        with self.lock:
            return self.update_needed
    
    def reset_update_flag(self):
        """Reset the update needed flag after UI is refreshed"""
        with self.lock:
            self.update_needed = False

# Create global state instance
global_state = GlobalState()

def sync_global_state_to_streamlit():
    """
    Sync the thread-safe global state to Streamlit's session_state.
    This should be called at the beginning of the Streamlit app.
    """
    if global_state.needs_update():
        # Update crawl status
        if "crawl_status" not in st.session_state:
            st.session_state.crawl_status = {}
        
        # Copy the values
        st.session_state.crawl_status.update(global_state.get_crawl_status())
        
        # Update cancelled tasks if needed
        st.session_state.tasks_cancelled = global_state.get_cancelled_tasks()
        
        # Reset the update flag
        global_state.reset_update_flag()
        
        # Request a rerun if needed
        if "needs_rerun" not in st.session_state:
            st.session_state.needs_rerun = True

async def main():
    st.set_page_config(
        page_title="Agentic RAG Documentation Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    # Sync global state with Streamlit session state
    sync_global_state_to_streamlit()

    # Check if the database is set up
    if not setup_database():
        st.error("Failed to set up the database. Please check the logs for more information.")
        return

    # Initialize crawl state but preserve existing values if a crawl is in progress
    if get_active_session():
        # Only initialize keys that don't exist yet to avoid disrupting an active crawl
        initialize_crawl_state_without_reset()
    else:
        # Full initialization for new session
        initialize_crawl_state()
    
    # Initialize session state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = None
    
    if "crawl_completed" not in st.session_state:
        st.session_state.crawl_completed = False
    
    if "crawl_success" not in st.session_state:
        st.session_state.crawl_success = False
        
    # Store active tab in session state to enable tab switching
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Chat"

    # Check if we need to start a crawl based on the new flag
    if st.session_state.get("crawl_initiated", False):
        # Get the crawl configuration from session state
        crawl_config = st.session_state.get("crawl_config")
        
        if crawl_config:
            # Log that we're starting the crawl
            logger.info(f"Starting crawl for {crawl_config.source_name}")
            
            try:
                # IMPORTANT CHANGE: Call crawl_documentation directly and don't await it
                # This is the key fix - we're not using asyncio.create_task() which can get lost
                # between Streamlit reruns
                logger.info(f"DEBUG: Directly starting crawl for {crawl_config.source_name}")
                
                # Use threading instead of asyncio for better compatibility with Streamlit
                import threading
                
                def start_crawl_thread():
                    """
                    Start a crawl in a separate thread with better Streamlit compatibility.
                    This function creates a new event loop and properly manages thread safety.
                    """
                    # Import necessary modules
                    try:
                        # Use our global state handler to track crawl status
                        global_state.update_crawl_status(
                            active=True,
                            start_time=time.time(),
                            source_name=crawl_config.source_name,
                            url=crawl_config.sitemap_url
                        )
                        
                        # Create a new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            # Run the crawl in this thread's event loop
                            logger.info(f"Thread starting crawl for {crawl_config.source_name}")
                            results = loop.run_until_complete(
                                crawl_documentation(openai_client, crawl_config)
                            )
                            
                            # Update status on completion using thread-safe global state
                            global_state.update_crawl_status(
                                completed=True,
                                success=results,
                                end_time=time.time()
                            )
                            
                            logger.info(f"Crawl thread completed for {crawl_config.source_name}")
                        except Exception as e:
                            # Handle and record exceptions
                            logger.error(f"Error in crawl thread: {str(e)}", exc_info=True)
                            
                            # Update status with error information
                            global_state.update_crawl_status(
                                error=str(e),
                                completed=True,
                                success=False,
                                end_time=time.time()
                            )
                        finally:
                            # Clean up the event loop
                            loop.close()
                            global_state.update_crawl_status(active=False)
                    except Exception as e:
                        # Handle setup errors
                        logger.error(f"Error setting up crawl thread: {str(e)}", exc_info=True)
                
                # Start the crawl in a separate thread
                crawl_thread = threading.Thread(target=start_crawl_thread)
                crawl_thread.daemon = True  # Make it a daemon so it doesn't block app shutdown
                crawl_thread.start()
                
                # Store thread reference in session state
                st.session_state.crawl_thread = crawl_thread
                logger.info(f"DEBUG: Crawl thread started for {crawl_config.source_name}")
                
                # Reset the initiation flag since we've started the crawl
                st.session_state.crawl_initiated = False
                
                # Display a success message
                st.success(f"Crawl started for {crawl_config.source_name}. Monitor progress in the Monitoring tab.")
                
            except Exception as e:
                logger.error(f"Error starting crawl: {str(e)}", exc_info=True)
                st.error(f"Error starting crawl: {str(e)}")
                st.session_state.crawl_initiated = False
        else:
            st.error("Crawl configuration not found. Please try again.")
            st.session_state.crawl_initiated = False

    # Create the tabs with active tab selection
    tab_names = ["Chat", "Add Sources", "Monitoring"]
    active_tab_index = tab_names.index(st.session_state.active_tab) if st.session_state.active_tab in tab_names else 0
    tabs = st.tabs(tab_names)
    
    # Chat tab
    with tabs[0]:
        # Load documentation sources from the database for the dropdown
        sources = get_documentation_sources_sync()
        
        if not sources:
            st.warning("No documentation sources available. Switch to the 'Add Sources' tab to get started.")
        else:
            # Display chat interface
            st.title("Agentic RAG Documentation Assistant")
            st.markdown("""
            Welcome to the documentation assistant! Ask questions about the documentation, and I'll do my best to help you.
            
            For best results, ask specific questions about the documentation content.
            """)
            
            # Create a list of source names for the selectbox
            source_names = ["All Sources"] + [source["name"] for source in sources]
            
            # Add a selectbox to filter by source
            selected_source_name = st.selectbox(
                "Filter by Source",
                options=source_names,
                index=0
            )
            
            # Get the source ID for the selected source
            if selected_source_name != "All Sources":
                for source in sources:
                    if source["name"] == selected_source_name:
                        st.session_state.selected_source = source["source_id"]
                        break
            else:
                st.session_state.selected_source = None
            
            # Display source information if a source is selected
            if st.session_state.selected_source:
                # Get source statistics
                source_stats = get_source_statistics(st.session_state.selected_source)
                
                if source_stats:
                    with st.expander("Source Statistics", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pages", source_stats.get('pages_count', 0))
                        with col2:
                            st.metric("Chunks", source_stats.get('chunks_count', 0))
                        
                        if source_stats.get("last_crawled_at"):
                            last_crawled = source_stats["last_crawled_at"]
                            if isinstance(last_crawled, datetime.datetime):
                                last_crawled = last_crawled.strftime("%Y-%m-%d %H:%M:%S")
                            with col3:
                                st.metric("Last Crawled", last_crawled)
                        
                        # Option to delete the source
                        if st.button("Delete Source", key="delete_source_btn"):
                            if delete_documentation_source(st.session_state.selected_source):
                                st.success(f"Deleted {selected_source_name}")
                                st.session_state.selected_source = None
                                st.rerun()
                            else:
                                st.error(f"Failed to delete {selected_source_name}")
            
            # Display chat history first (before processing new input)
            # This ensures messages appear in chronological order
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
            
            # Process user input
            if prompt := st.chat_input("Enter your question about the documentation"):
                # Add user message to session state
                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt,
                    "timestamp": time.time()
                })
                
                # Add assistant message to session state (we'll fill this with the response later)
                st.session_state.messages.append({
                    "role": "model",
                    "content": "",
                    "timestamp": time.time()
                })
                
                # Force a rerun to display the new messages and start generating the response
                # We don't need to display anything here since the rerun will handle it
                st.rerun()
            
            # Check if we need to generate a response for the last message
            if (len(st.session_state.messages) >= 2 and 
                st.session_state.messages[-2]["role"] == "user" and 
                not st.session_state.messages[-1]["content"]):
                
                # Get the user's prompt from the second-to-last message
                user_prompt = st.session_state.messages[-2]["content"]
                
                # Display the loading state for the assistant
                with st.chat_message("assistant"):
                    st.write("Thinking...")
                
                # Generate response in a non-blocking way
                await run_agent_with_streaming(user_prompt, st.session_state.selected_source)

    # Add Sources tab
    with tabs[1]:
        st.title("Add Documentation Source")
        create_source_form()
        
        # Show list of existing sources
        sources = get_documentation_sources_sync()
        if sources:
            st.subheader("Existing Documentation Sources")
            for source in sources:
                with st.expander(f"{source['name']}", expanded=False):
                    source_stats = get_source_statistics(source['source_id'])
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"Source ID: {source['source_id']}")
                    with col2:
                        st.write(f"URL: {source['base_url']}")
                    with col3:
                        if "last_crawled_at" in source and source["last_crawled_at"]:
                            last_crawled = source["last_crawled_at"]
                            if isinstance(last_crawled, datetime.datetime):
                                last_crawled = last_crawled.strftime("%Y-%m-%d %H:%M:%S")
                            st.write(f"Last Crawled: {last_crawled}")
                    
                    if source_stats:
                        st.write(f"Pages: {source_stats.get('pages_count', 0)} | Chunks: {source_stats.get('chunks_count', 0)}")
                    
                    if st.button(f"Delete {source['name']}", key=f"delete_{source['source_id']}"):
                        if delete_documentation_source(source['source_id']):
                            st.success(f"Deleted {source['name']}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {source['name']}")
        else:
            st.info("No documentation sources available. Add one using the form above.")

    # Monitoring tab with improved refresh mechanism
    with tabs[2]:
        st.header("Monitoring Dashboard")
        
        # Add a more prominent crawl status section at the top
        active_session = get_active_session()
        
        # Check if we have an active crawl thread in the session state
        has_active_thread = "crawl_thread" in st.session_state and st.session_state.crawl_thread.is_alive() if hasattr(st.session_state, "crawl_thread") else False
        
        # If we have an active thread but no active session, log this issue
        if has_active_thread and not active_session:
            logger.warning("Thread is active but no active session found in the monitoring state")
            
        # Combine both ways of checking for active crawls
        if active_session or has_active_thread:
            # Create a visually distinct status indicator for active crawls
            st.markdown("""
            <style>
            .active-crawl-box {
                background-color: #f0f8ff;
                border-left: 5px solid #2e86de;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a colored box to show active crawl status
            with st.container(border=True):
                # Use a 3-column layout with status icon, details, and control button
                col1, col2, col3 = st.columns([1, 6, 2])
                
                with col1:
                    # Add a rotating icon to indicate active crawl
                    st.markdown("""
                    <style>
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    .spinner {
                        display: inline-block;
                        font-size: 24px;
                        animation: spin 2s linear infinite;
                    }
                    </style>
                    <div style="text-align: center; padding-top: 10px;">
                        <span class="spinner">🔄</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Get source name from either source
                    source_name = active_session.source_name if active_session else "Documentation"
                    st.markdown(f"### Active Crawl: {source_name}")
                    
                    # Get session stats to show progress
                    stats = get_session_stats()
                    if stats:
                        # Show a progress indicator
                        processed = stats.get('processed_urls', 0)
                        total = max(100, stats.get('total_urls', 100))  # Default to 100 if total unknown
                        success_rate = stats.get('success_rate', 0) * 100
                        
                        # Show duration
                        duration = active_session.format_duration() if active_session and hasattr(active_session, 'format_duration') else "In progress"
                        
                        # Show start time
                        start_time = active_session.format_start_time() if active_session and hasattr(active_session, 'format_start_time') else "Recently"
                        st.caption(f"Started: {start_time} | Duration: {duration}")
                        
                        # Show progress stats
                        st.caption(f"Pages processed: {processed} | Success rate: {success_rate:.1f}%")
                        st.progress(min(1.0, processed / total), text=f"Crawling...")
                    else:
                        # Show minimal info if no stats available yet
                        st.caption("Crawl in progress. Statistics will appear soon...")
                        st.progress(0.1, text="Initializing...")
                
                with col3:
                    # Prominent stop button
                    if st.button("⏹️ Stop Crawl", 
                                type="primary", 
                                key="btn_stop_current_crawl", 
                                use_container_width=True):
                        cancel_current_crawl()
                        # Also handle thread cancellation
                        if has_active_thread:
                            logger.info("Attempting to cancel crawl thread")
                            # We can't directly cancel the thread, but we can set a flag
                            st.session_state.stop_crawl = True
                            st.info("Signaled thread to stop")
                    
                    # Add a refresh button
                    if st.button("⟳ Refresh Stats", key="refresh_active_crawl"):
                        st.rerun()
        
        # Always display the monitoring interface
        create_monitoring_ui()
        
        # Show status notification if a crawl was completed or cancelled
        if st.session_state.get("crawl_completed", False):
            if st.session_state.get("crawl_success", False):
                st.success("Crawl completed successfully!")
            else:
                st.error("Crawl failed! Check logs for details.")
            # Reset the flag after displaying
            st.session_state.crawl_completed = False
            
        if st.session_state.get("crawl_cancelled", False):
            st.info(f"Crawl cancelled. Stopped {st.session_state.get('tasks_cancelled', 0)} tasks.")
            # Reset the flag after displaying
            st.session_state.crawl_cancelled = False

    # Update the active tab in session state based on which tab is selected
    for i, tab in enumerate(tabs):
        if tab.empty():
            st.session_state.active_tab = tab_names[i]
            break

    # Check if a rerun is needed due to state changes
    if st.session_state.get("needs_rerun", False):
        # Reset the flag
        st.session_state.needs_rerun = False
        # Rerun the app to refresh UI with updated state
        st.rerun()


def reset_crawl_state():
    """Reset all crawl-related state variables to ensure a clean slate for new crawls."""
    # Clear stop flag
    if "stop_crawl" in st.session_state:
        st.session_state.stop_crawl = False
    
    # Clear crawl thread reference
    if "crawl_thread" in st.session_state:
        # We don't need to stop the thread as it should exit on its own
        # or respond to the stop flag
        st.session_state.crawl_thread = None
    
    # Clear status tracking
    if "crawl_status" in st.session_state:
        # Don't completely remove, keep for history
        st.session_state.crawl_status["active"] = False
    
    # Clear flags
    if "crawl_initiated" in st.session_state:
        st.session_state.crawl_initiated = False
        
    if "crawl_cancelled" in st.session_state:
        st.session_state.crawl_cancelled = False
    
    # Schedule a refresh of the UI if needed
    if "needs_rerun" not in st.session_state:
        st.session_state.needs_rerun = True
        
    logger.info("Crawl state reset completed")


def display_error_details():
    """Display detailed error information in the monitoring UI."""
    has_error = False
    
    # Check for errors in crawl status
    if "crawl_status" in st.session_state and "error" in st.session_state.crawl_status:
        has_error = True
        error_msg = st.session_state.crawl_status["error"]
        with st.expander("🔍 View Last Crawl Error Details", expanded=True):
            st.error(f"Error during crawl: {error_msg}")
            
            # Add diagnostic information
            st.markdown("### Diagnostic Information")
            
            # Timeline information
            if "start_time" in st.session_state.crawl_status:
                start_time = datetime.fromtimestamp(st.session_state.crawl_status["start_time"])
                st.text(f"Crawl started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
            if "end_time" in st.session_state.crawl_status:
                end_time = datetime.fromtimestamp(st.session_state.crawl_status["end_time"])
                st.text(f"Crawl ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Calculate duration
                duration_sec = st.session_state.crawl_status["end_time"] - st.session_state.crawl_status["start_time"]
                st.text(f"Duration: {duration_sec:.2f} seconds")
            
            # Add troubleshooting tips
            st.markdown("### Troubleshooting Steps")
            st.markdown("""
            1. **Check database connection** - Ensure PostgreSQL is running and accessible
            2. **Verify URL access** - Make sure the target site is accessible
            3. **Review crawl configuration** - Check chunk size and concurrency settings
            4. **Check memory usage** - Ensure enough RAM is available for embedding generation
            5. **Run diagnostics** - Use `check_database.py` to verify data storage
            """)
            
            # Add action button to run diagnostics
            if st.button("Run Database Diagnostics"):
                st.session_state.run_diagnostics = True
    
    # If we have a request to run diagnostics, do it
    if st.session_state.get("run_diagnostics", False):
        st.markdown("### Database Diagnostic Results")
        try:
            # Run a simple query to check database connectivity
            import subprocess
            result = subprocess.run(["python", "check_database.py"], capture_output=True, text=True)
            
            # Display results
            st.code(result.stdout)
            
            if result.stderr:
                st.error("Errors encountered:")
                st.code(result.stderr)
                
            # Clear the flag
            st.session_state.run_diagnostics = False
        except Exception as diag_error:
            st.error(f"Error running diagnostics: {str(diag_error)}")
            st.session_state.run_diagnostics = False
    
    return has_error


def initiate_crawl_process(source_name: str, sitemap_url: str, config_dict: Dict[str, Any]):
    """
    Initialize a crawl process with the specified configuration.
    
    Args:
        source_name: Name of the documentation source
        sitemap_url: Sitemap URL to crawl
        config_dict: Dictionary containing crawl configuration
    """
    try:
        # Create a unique ID for the source
        source_id = f"{source_name.lower().replace(' ', '_')}_{int(time.time())}"
        
        # Add the documentation source to the database
        result = add_documentation_source(
            name=source_name,
            source_id=source_id,
            base_url=sitemap_url,
            configuration=config_dict
        )
        
        if not result:
            st.error(f"Failed to add documentation source: {source_name}")
            return
        
        # Log successful addition
        st.success(f"Added documentation source: {source_name}")
        
        # Create the crawl configuration
        crawl_config = CrawlConfig(
            source_id=source_id,
            source_name=source_name,
            sitemap_url=sitemap_url,
            chunk_size=config_dict.get("chunk_size", app_config.DEFAULT_CHUNK_SIZE),
            chunk_words=config_dict.get("chunk_words", app_config.DEFAULT_CHUNK_WORDS),
            overlap_words=config_dict.get("overlap_words", app_config.DEFAULT_OVERLAP_WORDS),
            use_word_based_chunking=config_dict.get("use_word_based_chunking", app_config.USE_WORD_BASED_CHUNKING),
            max_concurrent_requests=config_dict.get("max_concurrent_crawls", app_config.DEFAULT_MAX_CONCURRENT_CRAWLS),
            max_concurrent_api_calls=config_dict.get("max_concurrent_api_calls", app_config.DEFAULT_MAX_CONCURRENT_API_CALLS),
            url_patterns_include=config_dict.get("url_patterns_include", []),
            url_patterns_exclude=config_dict.get("url_patterns_exclude", [])
        )
        
        # Start the crawl directly
        # We'll use the synchronous wrapper instead of relying on asyncio.create_task
        # which may not work well with Streamlit's execution model
        st.session_state.crawl_initiated = True
        st.session_state.crawl_config = crawl_config
        
        # Set the active tab to Monitoring
        st.session_state.active_tab = "Monitoring"
        
        # Force a rerun to apply the tab change
        st.rerun()
        
    except Exception as e:
        st.error(f"Error initiating crawl: {str(e)}")
        logger.error(f"Crawl initiation error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
