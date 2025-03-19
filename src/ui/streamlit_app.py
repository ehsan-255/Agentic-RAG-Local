from __future__ import annotations
from typing import Literal, TypedDict, Dict, Any, Optional, List
import asyncio
import os
import sys
import time
import datetime

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
from src.crawling.docs_crawler import CrawlConfig, crawl_documentation, clear_documentation_source
from src.db.schema import (
    get_documentation_sources as db_get_documentation_sources,
    get_source_statistics as db_get_source_statistics,
    add_documentation_source,
    delete_documentation_source,
    setup_database
)
from src.utils.validation import validate_sitemap_url

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

# Default configuration values
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_MAX_CONCURRENT_CRAWLS = 3
DEFAULT_MAX_CONCURRENT_API_CALLS = 5
DEFAULT_MATCH_COUNT = 5

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
            max_concurrent_api_calls=config.get("max_concurrent_api_calls", DEFAULT_MAX_CONCURRENT_API_CALLS)
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


async def main():
    st.set_page_config(
        page_title="Agentic RAG Documentation Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Check if the database is set up
    if not setup_database():
        st.error("Failed to set up the database. Please check the logs for more information.")
        return

    st.sidebar.title("Documentation Sources")
    
    # Initialize session state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = None
    
    # Add a new documentation source section
    with st.sidebar.expander("Add New Documentation Source", expanded=False):
        new_source_name = st.text_input("Documentation Name", key="new_source_name")
        new_source_url = st.text_input("Sitemap URL", key="new_source_url", 
                                      help="URL to a sitemap XML file (e.g., https://example.com/sitemap.xml)")
        
        # Advanced options - using a collapsible section instead of an expander
        st.markdown("### Advanced Options")
        show_advanced = st.checkbox("Show advanced options", value=False)
        
        if show_advanced:
            chunk_size = st.number_input(
                "Chunk Size", 
            min_value=1000,
            max_value=10000,
            value=DEFAULT_CHUNK_SIZE,
            step=500,
                help="Size of text chunks for processing"
        )
        
            max_concurrent_crawls = st.number_input(
                "Max Concurrent Crawls",
                min_value=1,
                max_value=10,
                value=DEFAULT_MAX_CONCURRENT_CRAWLS,
                help="Maximum number of concurrent web crawling operations"
            )
        
            max_concurrent_api_calls = st.number_input(
                "Max Concurrent API Calls",
                min_value=1,
                max_value=10,
                value=DEFAULT_MAX_CONCURRENT_API_CALLS,
                help="Maximum number of concurrent API calls to OpenAI"
            )
        else:
            # Default values when advanced options are hidden
            chunk_size = DEFAULT_CHUNK_SIZE
            max_concurrent_crawls = DEFAULT_MAX_CONCURRENT_CRAWLS
            max_concurrent_api_calls = DEFAULT_MAX_CONCURRENT_API_CALLS
        
        if st.button("Add and Crawl"):
            if not new_source_name or not new_source_url:
                st.error("Please provide a name and URL for the documentation source.")
            else:
                # Validate the sitemap URL
                is_valid, error_message = validate_sitemap_url(new_source_url)
                if not is_valid:
                    st.error(f"Invalid sitemap URL: {error_message}")
                else:
                    with st.spinner("Crawling documentation..."):
                        config = {
                            "chunk_size": chunk_size,
                            "max_concurrent_crawls": max_concurrent_crawls,
                            "max_concurrent_api_calls": max_concurrent_api_calls
                        }
                        
                        success = await crawl_new_documentation(new_source_name, new_source_url, config)
                        
                        if success:
                            st.success(f"Successfully crawled {new_source_name}")
                            # Reset the input fields
                            st.session_state.new_source_name = ""
                            st.session_state.new_source_url = ""
                        else:
                            st.error(f"Failed to crawl {new_source_name}")

    # Load documentation sources from the database
    sources = get_documentation_sources_sync()
    
    if not sources:
        st.sidebar.warning("No documentation sources available. Add one to get started.")
    else:
        # Create a list of source names for the selectbox
        source_names = ["All Sources"] + [source["name"] for source in sources]
        
        # Add a selectbox to filter by source
        selected_source_name = st.sidebar.selectbox(
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
        
        # Display source information
        if st.session_state.selected_source:
            # Get source statistics
            source_stats = get_source_statistics(st.session_state.selected_source)
            
            if source_stats:
                st.sidebar.subheader("Source Statistics")
                st.sidebar.write(f"Pages: {source_stats.get('pages_count', 0)}")
                st.sidebar.write(f"Chunks: {source_stats.get('chunks_count', 0)}")
                
                if source_stats.get("last_crawled_at"):
                    last_crawled = source_stats["last_crawled_at"]
                    if isinstance(last_crawled, datetime.datetime):
                        last_crawled = last_crawled.strftime("%Y-%m-%d %H:%M:%S")
                    st.sidebar.write(f"Last Crawled: {last_crawled}")
                
                # Option to delete the source
                if st.sidebar.button("Delete Source"):
                    if delete_documentation_source(st.session_state.selected_source):
                        st.sidebar.success(f"Deleted {selected_source_name}")
                        st.session_state.selected_source = None
                        st.rerun()
                    else:
                        st.sidebar.error(f"Failed to delete {selected_source_name}")
    
    # Set up the main chat interface
    st.title("Agentic RAG Documentation Assistant")
    st.markdown("""
    Welcome to the documentation assistant! Ask questions about the documentation, and I'll do my best to help you.
    
    For best results, ask specific questions about the documentation content.
    """)
    
    if sources:
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
    else:
        st.info("Please add a documentation source to get started.")


if __name__ == "__main__":
    asyncio.run(main())
