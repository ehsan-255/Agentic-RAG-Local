<<<<<<< HEAD
from __future__ import annotations
from typing import Literal, TypedDict, Dict, Any, Optional, List
import asyncio
import os
import time

import streamlit as st
import json
import logfire
from supabase import Client, create_client

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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

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
        supabase=supabase,
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
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def get_documentation_sources() -> List[Dict[str, Any]]:
    """Get all documentation sources from Supabase."""
    try:
        result = supabase.from_('documentation_sources').select('*').execute()
        return result.data if result.data else []
    except Exception as e:
        st.error(f"Error fetching documentation sources: {e}")
        return []


async def crawl_new_documentation(source_name: str, sitemap_url: str, config: Dict[str, Any]) -> bool:
    """Start a new documentation crawl with the given configuration."""
    try:
        # Create a CrawlConfig object
        crawl_config = CrawlConfig(
            source_name=source_name,
            source_id="",  # Will be created during crawl
            chunk_size=config.get("chunk_size", DEFAULT_CHUNK_SIZE),
            max_concurrent_crawls=config.get("max_concurrent_crawls", DEFAULT_MAX_CONCURRENT_CRAWLS),
            max_concurrent_api_calls=config.get("max_concurrent_api_calls", DEFAULT_MAX_CONCURRENT_API_CALLS),
            url_patterns_include=config.get("url_patterns_include", []),
            url_patterns_exclude=config.get("url_patterns_exclude", []),
            llm_model=config.get("llm_model", "gpt-4o-mini"),
            embedding_model=config.get("embedding_model", "text-embedding-3-small"),
        )
        
        # Start the crawl
        success = await crawl_documentation(sitemap_url, crawl_config)
        return success
    except Exception as e:
        st.error(f"Error starting documentation crawl: {e}")
        return False


async def main():
    st.set_page_config(page_title="Agentic RAG", page_icon="📚", layout="wide")

    st.sidebar.title("Agentic RAG")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "chat"
    
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = None
    
    # Sidebar navigation
    active_tab = st.sidebar.radio("Navigation", ["Chat", "Configuration", "Documentation Sources"])
    st.session_state.active_tab = active_tab.lower()
    
    # Fetch documentation sources for the sidebar
    documentation_sources = await get_documentation_sources()
    source_options = ["All Sources"] + [source["name"] for source in documentation_sources]
    
    # Documentation source selector in the sidebar
    st.sidebar.subheader("Documentation Source")
    selected_source_name = st.sidebar.selectbox(
        "Select Documentation Source",
        options=source_options,
        index=0
    )
    
    # Set the selected source ID
    if selected_source_name == "All Sources":
        st.session_state.selected_source = None
    else:
        for source in documentation_sources:
            if source["name"] == selected_source_name:
                st.session_state.selected_source = source["source_id"]
                break
    
    # Display documentation source statistics if available
    if st.session_state.selected_source:
        for source in documentation_sources:
            if source["source_id"] == st.session_state.selected_source:
                st.sidebar.markdown(f"**Pages:** {source['pages_count']}")
                st.sidebar.markdown(f"**Chunks:** {source['chunks_count']}")
                if source["last_crawled_at"]:
                    st.sidebar.markdown(f"**Last Crawled:** {source['last_crawled_at'][:19]}")
                break
    
    # Main content based on active tab
    if st.session_state.active_tab == "chat":
        st.title("Documentation Assistant")
        
        # Display source info if a specific source is selected
        if st.session_state.selected_source:
            source_name = next((s["name"] for s in documentation_sources if s["source_id"] == st.session_state.selected_source), "Unknown")
            st.info(f"Searching in: {source_name}")
        else:
            st.info("Searching in all documentation sources")

    # Display all messages from the conversation so far
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
        user_input = st.chat_input("What questions do you have about the documentation?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
                await run_agent_with_streaming(user_input, st.session_state.selected_source)
    
    elif st.session_state.active_tab == "configuration":
        st.title("RAG Configuration")
        
        # Crawling Configuration Section
        st.subheader("Crawling Configuration")
        
        # Model Selection
        llm_model = st.selectbox(
            "LLM Model for Summaries",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Model used for generating titles and summaries"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=["text-embedding-3-small", "text-embedding-3-large"],
            index=0,
            help="Model used for generating embeddings"
        )
        
        # Chunking Configuration
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=1000,
            max_value=10000,
            value=DEFAULT_CHUNK_SIZE,
            step=500,
            help="Number of characters per chunk. Larger chunks provide more context but may reduce relevance precision."
        )
        
        # Concurrency Configuration
        col1, col2 = st.columns(2)
        with col1:
            max_concurrent_crawls = st.number_input(
                "Max Concurrent Crawls",
                min_value=1,
                max_value=10,
                value=DEFAULT_MAX_CONCURRENT_CRAWLS,
                help="Maximum number of concurrent page crawls"
            )
        
        with col2:
            max_concurrent_api_calls = st.number_input(
                "Max Concurrent API Calls",
                min_value=1,
                max_value=10,
                value=DEFAULT_MAX_CONCURRENT_API_CALLS,
                help="Maximum number of concurrent OpenAI API calls"
            )
        
        # URL Filtering
        st.subheader("URL Filtering")
        
        url_patterns_include = st.text_area(
            "Include URL Patterns",
            value="/docs/,/api/,/tutorial/",
            help="Comma-separated list of URL patterns to include (e.g., /docs/,/api/)"
        )
        
        url_patterns_exclude = st.text_area(
            "Exclude URL Patterns",
            value="",
            help="Comma-separated list of URL patterns to exclude"
        )
        
        # Format URL patterns as lists
        url_patterns_include_list = [pattern.strip() for pattern in url_patterns_include.split(",") if pattern.strip()]
        url_patterns_exclude_list = [pattern.strip() for pattern in url_patterns_exclude.split(",") if pattern.strip()]
        
        # Save Configuration Button
        st.subheader("Default Configuration")
        if st.button("Save as Default Configuration"):
            # Save to a configuration file or database
            st.success("Configuration saved as default!")
        
        # Display current configuration as JSON
        with st.expander("Current Configuration"):
            config = {
                "llm_model": llm_model,
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "max_concurrent_crawls": max_concurrent_crawls,
                "max_concurrent_api_calls": max_concurrent_api_calls,
                "url_patterns_include": url_patterns_include_list,
                "url_patterns_exclude": url_patterns_exclude_list
            }
            st.json(config)
    
    elif st.session_state.active_tab == "documentation sources":
        st.title("Documentation Sources")
        
        # List all existing documentation sources
        st.subheader("Existing Documentation Sources")
        
        if documentation_sources:
            # Create a table of documentation sources
            source_data = []
            for source in documentation_sources:
                last_crawled = source.get("last_crawled_at", "Never")
                if last_crawled and last_crawled != "Never":
                    last_crawled = last_crawled[:19]  # Truncate to readable datetime
                
                source_data.append({
                    "Name": source["name"],
                    "URL": source["base_url"],
                    "Pages": source["pages_count"],
                    "Chunks": source["chunks_count"],
                    "Last Crawled": last_crawled,
                    "Source ID": source["source_id"]
                })
            
            # Display as a dataframe
            st.dataframe(source_data)
            
            # Source management
            st.subheader("Source Management")
            
            # Select a source to manage
            source_to_manage = st.selectbox(
                "Select Source to Manage",
                options=[source["name"] for source in documentation_sources]
            )
            
            selected_source_id = next((s["source_id"] for s in documentation_sources if s["name"] == source_to_manage), None)
            
            if selected_source_id:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Clear Source Data", key="clear_source"):
                        st.warning(f"This will delete all data for {source_to_manage}. Are you sure?")
                        st.session_state.confirm_clear = True
                
                with col2:
                    if st.button("Recrawl Source", key="recrawl_source"):
                        # Get the source details
                        source_details = next((s for s in documentation_sources if s["source_id"] == selected_source_id), None)
                        
                        if source_details:
                            # Clear existing data
                            await clear_documentation_source(selected_source_id)
                            
                            # Recrawl with the same configuration
                            config = source_details.get("configuration", {})
                            success = await crawl_new_documentation(
                                source_details["name"],
                                source_details["base_url"],
                                config
                            )
                            
                            if success:
                                st.success(f"Recrawl of {source_to_manage} started!")
                            else:
                                st.error(f"Failed to start recrawl of {source_to_manage}")
                
                # Confirmation for clearing
                if st.session_state.get("confirm_clear", False):
                    if st.button("Confirm Clear", key="confirm_clear_button"):
                        success = await clear_documentation_source(selected_source_id)
                        if success:
                            st.success(f"Data for {source_to_manage} cleared successfully!")
                        else:
                            st.error(f"Failed to clear data for {source_to_manage}")
                        
                        # Reset confirmation
                        st.session_state.confirm_clear = False
                    
                    if st.button("Cancel", key="cancel_clear"):
                        st.session_state.confirm_clear = False
        
        else:
            st.info("No documentation sources found.")
        
        # Add new documentation source
        st.subheader("Add New Documentation Source")
        
        with st.form("new_source_form"):
            source_name = st.text_input("Source Name", placeholder="e.g., Company Docs")
            sitemap_url = st.text_input("Sitemap URL", placeholder="https://example.com/sitemap.xml")
            
            # Basic configuration options
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=1000,
                    max_value=10000,
                    value=DEFAULT_CHUNK_SIZE,
                    step=500
                )
            
            with col2:
                max_concurrent_crawls = st.number_input(
                    "Max Concurrent Crawls",
                    min_value=1,
                    max_value=10,
                    value=DEFAULT_MAX_CONCURRENT_CRAWLS
                )
            
            # URL filtering
            url_include = st.text_input(
                "Include URL Patterns (comma-separated)",
                placeholder="e.g., /docs/,/api/"
            )
            
            # Submit button
            submitted = st.form_submit_button("Start Crawl")
            
            if submitted:
                if not source_name or not sitemap_url:
                    st.error("Source name and sitemap URL are required")
                else:
                    # Process URL patterns
                    url_patterns = [pattern.strip() for pattern in url_include.split(",") if pattern.strip()]
                    
                    # Create configuration
                    config = {
                        "chunk_size": chunk_size,
                        "max_concurrent_crawls": max_concurrent_crawls,
                        "max_concurrent_api_calls": DEFAULT_MAX_CONCURRENT_API_CALLS,
                        "url_patterns_include": url_patterns
                    }
                    
                    # Start crawl
                    with st.spinner(f"Starting crawl for {source_name}..."):
                        success = await crawl_new_documentation(source_name, sitemap_url, config)
                    
                    if success:
                        st.success(f"Crawl for {source_name} started successfully!")
                    else:
                        st.error(f"Failed to start crawl for {source_name}")


if __name__ == "__main__":
    asyncio.run(main())
=======
import os
import sys
import asyncio
import streamlit as st
from datetime import datetime, timezone
import pandas as pd
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from src.rag.rag_expert import agentic_rag_expert, AgentyRagDeps
from src.crawling.docs_crawler import CrawlConfig, crawl_documentation, clear_documentation_source

from openai import AsyncOpenAI
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

# Initialize OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None

# Check if clients are initialized
if not supabase or not openai_client:
    st.error("Supabase and OpenAI clients must be initialized. Please check your .env file.")
    st.stop()

# Create dependencies for the RAG agent
deps = AgentyRagDeps(
    supabase=supabase,
    openai_client=openai_client
)

# Set page config
st.set_page_config(
    page_title="Agentic RAG Documentation Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Chat", "Documentation Sources", "Configuration"])

# Function to get all documentation sources
def get_documentation_sources():
    response = supabase.table("documentation_sources").select("*").order("name").execute()
    return response.data

# Function to get source statistics
def get_source_statistics(source_id):
    response = supabase.table("documentation_sources").select("*").eq("id", source_id).execute()
    if response.data:
        return response.data[0]
    return None

# Chat tab
with tab1:
    st.title("Documentation Assistant")
    
    # Get all sources for the dropdown
    sources = get_documentation_sources()
    source_names = [source["name"] for source in sources]
    
    if not sources:
        st.warning("No documentation sources available. Please add a source in the 'Documentation Sources' tab.")
        st.stop()
    
    # Source selection
    selected_source = st.selectbox("Select Documentation Source", source_names)
    selected_source_id = next((source["id"] for source in sources if source["name"] == selected_source), None)
    
    # Display source statistics
    if selected_source_id:
        stats = get_source_statistics(selected_source_id)
        if stats:
            col1, col2, col3 = st.columns(3)
            col1.metric("Pages", stats["page_count"])
            col2.metric("Chunks", stats["chunk_count"])
            col3.metric("Last Updated", stats["updated_at"].split("T")[0] if "T" in stats["updated_at"] else stats["updated_at"])
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the documentation"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Get response from RAG agent
            try:
                response = asyncio.run(agentic_rag_expert(
                    deps=deps,
                    query=prompt,
                    source_id=selected_source_id,
                    max_chunks=5,
                    similarity_threshold=0.7
                ))
                
                # Update placeholder with response
                message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                message_placeholder.markdown(f"Error: {str(e)}")

# Documentation Sources tab
with tab2:
    st.title("Documentation Sources")
    
    # Add new documentation source
    with st.expander("Add New Documentation Source", expanded=False):
        with st.form("add_source_form"):
            site_name = st.text_input("Site Name", help="A unique name for this documentation source")
            base_url = st.text_input("Base URL", help="The base URL of the documentation site")
            sitemap_url = st.text_input("Sitemap URL", help="URL to the sitemap.xml file")
            
            allowed_domains = st.text_input("Allowed Domains (comma-separated)", 
                                          help="Domains that the crawler is allowed to visit")
            
            start_urls = st.text_input("Start URLs (comma-separated)", 
                                     help="URLs where the crawler should start")
            
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input("Chunk Size", min_value=1000, max_value=10000, value=5000,
                                           help="Number of characters per chunk")
            
            with col2:
                max_concurrent_crawls = st.number_input("Max Concurrent Crawls", min_value=1, max_value=10, value=3,
                                                      help="Maximum number of concurrent page downloads")
            
            submitted = st.form_submit_button("Add Source")
            
            if submitted:
                if not site_name or not base_url:
                    st.error("Site Name and Base URL are required")
                else:
                    try:
                        # Create crawl config
                        config = CrawlConfig(
                            site_name=site_name,
                            base_url=base_url,
                            allowed_domains=[domain.strip() for domain in allowed_domains.split(",") if domain.strip()],
                            start_urls=[url.strip() for url in start_urls.split(",") if url.strip()],
                            sitemap_urls=[sitemap_url] if sitemap_url else None,
                            chunk_size=chunk_size,
                            max_concurrent_crawls=max_concurrent_crawls
                        )
                        
                        # Start the crawl
                        with st.spinner(f"Crawling {site_name}..."):
                            pages_processed = crawl_documentation(config)
                            st.success(f"Successfully crawled {pages_processed} pages from {site_name}")
                    except Exception as e:
                        st.error(f"Error crawling documentation: {str(e)}")
    
    # List existing sources
    st.subheader("Existing Documentation Sources")
    sources = get_documentation_sources()
    
    if not sources:
        st.info("No documentation sources available. Add a source using the form above.")
    else:
        # Create a DataFrame for display
        df = pd.DataFrame(sources)
        df = df[["name", "base_url", "page_count", "chunk_count", "status", "updated_at"]]
        df.columns = ["Name", "Base URL", "Pages", "Chunks", "Status", "Last Updated"]
        
        # Format the date
        df["Last Updated"] = df["Last Updated"].apply(lambda x: x.split("T")[0] if "T" in str(x) else x)
        
        # Display the table
        st.dataframe(df)
        
        # Source management
        st.subheader("Manage Documentation Sources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_to_clear = st.selectbox("Select Source to Clear", 
                                         [source["name"] for source in sources],
                                         key="clear_source")
            
            if st.button("Clear Source"):
                if source_to_clear:
                    with st.spinner(f"Clearing {source_to_clear}..."):
                        try:
                            success = clear_documentation_source(source_to_clear)
                            if success:
                                st.success(f"Successfully cleared {source_to_clear}")
                            else:
                                st.error(f"Failed to clear {source_to_clear}")
                        except Exception as e:
                            st.error(f"Error clearing source: {str(e)}")
        
        with col2:
            source_to_recrawl = st.selectbox("Select Source to Recrawl", 
                                           [source["name"] for source in sources],
                                           key="recrawl_source")
            
            if st.button("Recrawl Source"):
                if source_to_recrawl:
                    # Get the source configuration
                    source = next((s for s in sources if s["name"] == source_to_recrawl), None)
                    
                    if source and source.get("config"):
                        config_data = source["config"]
                        
                        # Create crawl config
                        config = CrawlConfig(
                            site_name=source["name"],
                            base_url=source["base_url"],
                            allowed_domains=config_data.get("allowed_domains", []),
                            start_urls=config_data.get("start_urls", []),
                            sitemap_urls=[source["sitemap_url"]] if source.get("sitemap_url") else None,
                            chunk_size=config_data.get("chunk_size", 5000),
                            max_concurrent_crawls=3
                        )
                        
                        # Clear the source first
                        with st.spinner(f"Clearing {source_to_recrawl}..."):
                            clear_documentation_source(source_to_recrawl)
                        
                        # Start the crawl
                        with st.spinner(f"Recrawling {source_to_recrawl}..."):
                            try:
                                pages_processed = crawl_documentation(config)
                                st.success(f"Successfully recrawled {pages_processed} pages from {source_to_recrawl}")
                            except Exception as e:
                                st.error(f"Error recrawling documentation: {str(e)}")
                    else:
                        st.error("Source configuration not found")

# Configuration tab
with tab3:
    st.title("Configuration")
    
    with st.form("config_form"):
        st.subheader("Crawler Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.number_input("Default Chunk Size", min_value=1000, max_value=10000, 
                                       value=int(os.environ.get("CHUNK_SIZE", 5000)),
                                       help="Number of characters per chunk")
            
            max_concurrent_crawls = st.number_input("Max Concurrent Crawls", min_value=1, max_value=10, 
                                                  value=int(os.environ.get("MAX_CONCURRENT_CRAWLS", 3)),
                                                  help="Maximum number of concurrent page downloads")
        
        with col2:
            max_concurrent_api_calls = st.number_input("Max Concurrent API Calls", min_value=1, max_value=20, 
                                                     value=int(os.environ.get("MAX_CONCURRENT_API_CALLS", 5)),
                                                     help="Maximum number of concurrent OpenAI API calls")
            
            retry_attempts = st.number_input("Retry Attempts", min_value=1, max_value=10, 
                                           value=int(os.environ.get("RETRY_ATTEMPTS", 6)),
                                           help="Number of retry attempts for API calls")
        
        st.subheader("URL Filtering")
        
        url_patterns_include = st.text_area("URL Patterns to Include (one per line)", 
                                          help="Only URLs containing these patterns will be crawled")
        
        url_patterns_exclude = st.text_area("URL Patterns to Exclude (one per line)", 
                                          help="URLs containing these patterns will be skipped")
        
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            llm_model = st.text_input("LLM Model", value=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                                    help="OpenAI model for summaries and answers")
        
        with col2:
            embedding_model = st.text_input("Embedding Model", value=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                                          help="OpenAI model for embeddings")
        
        save_config = st.form_submit_button("Save Configuration")
        
        if save_config:
            # In a real app, you would save these to a config file or database
            # For this example, we'll just show a success message
            st.success("Configuration saved successfully")
            
            # You could also update environment variables for the current session
            os.environ["CHUNK_SIZE"] = str(chunk_size)
            os.environ["MAX_CONCURRENT_CRAWLS"] = str(max_concurrent_crawls)
            os.environ["MAX_CONCURRENT_API_CALLS"] = str(max_concurrent_api_calls)
            os.environ["RETRY_ATTEMPTS"] = str(retry_attempts)
            os.environ["LLM_MODEL"] = llm_model
            os.environ["EMBEDDING_MODEL"] = embedding_model
>>>>>>> ee4b578bf2a45624bbe5312f94b982f7cd411dc1
