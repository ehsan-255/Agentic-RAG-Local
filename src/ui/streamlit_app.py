from __future__ import annotations
from typing import Literal, TypedDict, Dict, Any, Optional, List
import asyncio
import os
import sys
import time
import datetime
import threading
import logging

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

# Default configuration values
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_MAX_CONCURRENT_CRAWLS = 3
DEFAULT_MAX_CONCURRENT_API_CALLS = 5
DEFAULT_MATCH_COUNT = 5
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

    # Run the agent in a stream
    async with agentic_rag_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],  # pass entire conversation so far
        context={"source_id": source_id} if source_id else {},  # Pass source_id as context
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        placeholder = st.empty()

        async for text in result.text_deltas():
            partial_text += text
            placeholder.markdown(partial_text)

        # Update the last message with the complete response
        st.session_state.messages[-1]['content'] = partial_text
        st.session_state.messages[-1]['agent_response'] = partial_text


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
    
    with st.form("add_source_form"):
        # Heading for the form
        st.subheader("Add New Documentation Source")
        
        # Form fields
        st.text_input("Documentation Name", key="source_name", 
                     disabled=crawl_in_progress)
        
        st.text_input("Sitemap URL", key="sitemap_url", 
                    help="URL to a sitemap XML file (e.g., https://example.com/sitemap.xml)",
                    disabled=crawl_in_progress)
        
        with st.expander("Advanced Options"):
            st.number_input("Chunk Size", min_value=1000, 
                        value=DEFAULT_CHUNK_SIZE, key="chunk_size",
                        disabled=crawl_in_progress)
                        
            st.number_input("Max Concurrent Requests", min_value=1, 
                        value=DEFAULT_MAX_CONCURRENT_CRAWLS, key="max_concurrent_crawls",
                        disabled=crawl_in_progress)
                        
            st.number_input("Max Concurrent API Calls", min_value=1, 
                        value=DEFAULT_MAX_CONCURRENT_API_CALLS, key="max_concurrent_api_calls",
                        disabled=crawl_in_progress)
                        
            st.text_area("URL Patterns to Include (one per line)", 
                      value='\n'.join(DEFAULT_URL_PATTERNS),
                      key="url_patterns_include",
                      help="URLs containing these patterns will be included",
                      disabled=crawl_in_progress)
                      
            st.text_area("URL Patterns to Exclude (one per line)", 
                      key="url_patterns_exclude",
                      help="URLs containing these patterns will be excluded",
                      disabled=crawl_in_progress)
        
        # Disable the submit button if a crawl is already in progress
        submit_button = st.form_submit_button(
            "Add and Crawl", 
            use_container_width=True,
            disabled=crawl_in_progress
        )
        
        # Show status message if a crawl is in progress
        if crawl_in_progress:
            st.info("⏳ A crawl is already in progress. Please wait for it to complete before starting another one.")
        
        if submit_button and not crawl_in_progress:
            # Instead of setting a flag, directly call the non-blocking crawl starter
            initiate_crawl()


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
    # First check and log the current state
    active_session = get_active_session()
    has_thread = "crawl_thread" in st.session_state and st.session_state.crawl_thread.is_alive() if hasattr(st.session_state, "crawl_thread") else False
    
    logger.info(f"Cancelling crawl. Active session: {bool(active_session)}, Active thread: {has_thread}")
    
    # Set the stop flag first (for the thread to detect)
    st.session_state.stop_crawl = True
    
    # If we have an active session, end it properly
    if active_session:
        try:
            # Cancel all tasks
            tasks_cancelled = cancel_all_tasks()
            logger.info(f"Cancelled {tasks_cancelled} tasks")
            
            # Mark the session as complete with cancelled status
            active_session.complete(status="cancelled")
            
            # End the crawl session
            end_crawl_session(active_session.session_id)
            
            # Record in session state for UI updates
            st.session_state.crawl_cancelled = True
            st.session_state.tasks_cancelled = tasks_cancelled
            
            # UI feedback
            st.success(f"Crawl cancelled. Stopped {tasks_cancelled} active tasks.")
            
            # Clean up session state
            reset_crawl_state()
        except Exception as e:
            logger.error(f"Error cancelling crawl session: {str(e)}", exc_info=True)
            st.error(f"Error cancelling crawl: {str(e)}")
    elif has_thread:
        # If we only have a thread but no session, just log it
        logger.info("Only thread active, no session to cancel")
        st.warning("No active crawl session found, but thread is running. Set stop flag.")
    else:
        # No active crawl at all
        st.warning("No active crawl to cancel.")
    
    # Final state reset to be safe
    st.session_state.stop_crawl = True


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    st.set_page_config(
        page_title="Agentic RAG Documentation Assistant",
        page_icon="📚",
        layout="wide"
    )

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
                    # We need to create a new event loop for this thread
                    asyncio.set_event_loop(asyncio.new_event_loop())
                    # Run the crawl in this thread's event loop
                    asyncio.get_event_loop().run_until_complete(
                        crawl_documentation(openai_client, crawl_config)
                    )
                    logger.info(f"Crawl thread completed for {crawl_config.source_name}")
                
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
            
            # Process user input
            if prompt := st.chat_input("Enter your question about the documentation"):
                # Add user message to session state
                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt,
                    "timestamp": time.time()
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Add assistant message to session state (we'll fill this with the response later)
                st.session_state.messages.append({
                    "role": "model",
                    "content": "",
                    "timestamp": time.time()
                })
                
                # Display the loading state for the assistant
                with st.chat_message("assistant"):
                    st.write("Thinking...")
                
                # Generate response in a non-blocking way
                await run_agent_with_streaming(prompt, st.session_state.selected_source)
            
            # Display chat history
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])

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


if __name__ == "__main__":
    asyncio.run(main())
