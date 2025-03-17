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