"""
Chat UI components for the Streamlit application.

This module contains components for:
- Displaying chat messages
- Processing and streaming agent responses
- Managing message history
"""

from typing import Literal, TypedDict, Dict, Any, Optional, List
import time
import streamlit as st
import logging
import sys
import json
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Import message part classes
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

# Keep track of all OpenAI clients we've patched
patched_clients = set()

def patch_openai_client(client):
    """Patch an OpenAI client to log all API calls."""
    if client in patched_clients:
        return client  # Already patched
    
    # Patch the chat completions create method
    original_create = client.chat.completions.create
    
    async def log_create_wrapper(*args, **kwargs):
        # Log the message sizes
        try:
            # Log stack trace to see where the call is coming from
            stack_trace = ''.join(traceback.format_stack())
            logger.info(f"📍 OPENAI API CALL STACK TRACE:\n{stack_trace}")
            
            if 'messages' in kwargs:
                messages = kwargs['messages']
                logger.info(f"🔍 OPENAI API CALL: Sending {len(messages)} messages to OpenAI")
                
                total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
                logger.info(f"🔍 OPENAI API CALL: Total content size: {total_chars} characters")
                
                # Log details of each message
                for i, msg in enumerate(messages):
                    content_length = len(str(msg.get('content', '')))
                    logger.info(f"🔍 OPENAI API CALL: Message[{i}] ({msg.get('role', 'unknown')}) - {content_length} characters")
                    
                    # If a message is extremely large, log more details
                    if content_length > 100000:
                        logger.warning(f"🚨 OPENAI API CALL: Message[{i}] is very large ({content_length} characters)")
                        
                        # Try to determine the message content type
                        content = msg.get('content', '')
                        if isinstance(content, str):
                            # Log the first 500 chars
                            logger.warning(f"🚨 OPENAI API CALL: Message[{i}] starts with: {content[:500]}...")
                            
                            # Check for tool outputs
                            if "function(" in content[:1000] or "tool_call(" in content[:1000]:
                                logger.warning(f"🚨 OPENAI API CALL: Message[{i}] appears to contain tool outputs")
                            
                            # Log how many tool call references are in the message
                            tool_call_refs = content.count("tool_call(")
                            function_refs = content.count("function(")
                            if tool_call_refs > 0 or function_refs > 0:
                                logger.warning(f"🚨 OPENAI API CALL: Message[{i}] contains {tool_call_refs} tool_call refs and {function_refs} function refs")
        except Exception as e:
            logger.error(f"Error logging OpenAI API call: {e}")
            
            # Call the original function
            return await original_create(*args, **kwargs)
        
        # Replace the original method with our wrapper
        client.chat.completions.create = log_create_wrapper
        
        # Keep track of this patched client
        patched_clients.add(client)
        
        return client

# Patch the AsyncOpenAI constructor to ensure we catch all client instances
original_async_openai_init = None

def patch_async_openai_constructor():
    """Patch the AsyncOpenAI constructor to intercept all client instances."""
    global original_async_openai_init
    
    from openai import AsyncOpenAI
    
    if original_async_openai_init is None:
        original_async_openai_init = AsyncOpenAI.__init__
        
        def patched_init(self, *args, **kwargs):
            # Call the original constructor
            original_async_openai_init(self, *args, **kwargs)
            
            # Patch this client instance
            patch_openai_client(self)
            logger.info(f"🔧 PATCHED: New AsyncOpenAI client instance created and patched")
        
        AsyncOpenAI.__init__ = patched_init
        logger.info(f"🔧 PATCHED: AsyncOpenAI constructor has been patched")

# Apply the constructor patch immediately
patch_async_openai_constructor()

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


async def run_agent_with_streaming(user_input: str, openai_client, source_id: Optional[str] = None):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    
    Args:
        user_input: The user's input text
        openai_client: OpenAI client for API calls
        source_id: Optional source ID to limit the search to a specific documentation source
    """
    # Prepare dependencies
    deps = AgentyRagDeps(
        openai_client=openai_client
    )

    # Patch the provided client
    patch_openai_client(openai_client)

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
        logger.info(f"🚀 AGENT START: Running agent for query: '{user_input_with_source[:100]}...'")
        
        response = await agentic_rag_expert.run(
            user_input_with_source,  # The user prompt
            deps=deps,
            message_history=[],  # Use empty history to avoid format errors
        )
        
        logger.info("✅ AGENT COMPLETE: Agent execution completed successfully")
        
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
        
        logger.info(f"📤 AGENT RESPONSE: Response size is {len(partial_text)} characters")
            
        # Display the full response at once
        placeholder.markdown(partial_text)
        
        # Update the last message with the complete response
        st.session_state.messages[-1]['content'] = partial_text
        st.session_state.messages[-1]['agent_response'] = partial_text
        
    except Exception as e:
        logger.error(f"❌ ERROR: Error generating response: {str(e)}")
        placeholder = st.empty()
        placeholder.error(f"Error generating response: {str(e)}")
        st.session_state.messages[-1]['content'] = "Sorry, there was an error generating a response."
        st.session_state.messages[-1]['agent_response'] = "Sorry, there was an error generating a response."
        
        # Try to get more details about the error
        if hasattr(e, 'body'):
            try:
                error_body = e.body
                if isinstance(error_body, dict) and 'message' in error_body:
                    logger.error(f"❌ ERROR DETAILS: {error_body['message']}")
                elif isinstance(error_body, str):
                    logger.error(f"❌ ERROR DETAILS: {error_body}")
            except:
                logger.error(f"❌ ERROR DETAILS: Could not extract error details from body")
                
        # Log exception traceback
        logger.error("❌ ERROR TRACEBACK:", exc_info=True)


def create_chat_ui(openai_client, sources=None):
    """
    Create the chat interface with documentation sources dropdown.
    
    Args:
        openai_client: The OpenAI client for API calls
        sources: List of documentation sources to display in the dropdown
    """
    st.title("Agentic RAG Documentation Assistant")
    st.markdown("""
    Welcome to the documentation assistant! Ask questions about the documentation, and I'll do my best to help you.
    
    For best results, ask specific questions about the documentation content.
    """)
    
    if not sources:
        st.warning("No documentation sources available. Add sources in the 'Add Sources' tab to get started.")
        return
    
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
        from src.db.schema import get_source_statistics
        
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
                    import datetime
                    last_crawled = source_stats["last_crawled_at"]
                    if isinstance(last_crawled, datetime.datetime):
                        last_crawled = last_crawled.strftime("%Y-%m-%d %H:%M:%S")
                    with col3:
                        st.metric("Last Crawled", last_crawled)
                
                # Option to delete the source
                from src.db.schema import delete_documentation_source
                
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
        import asyncio
        asyncio.create_task(run_agent_with_streaming(user_prompt, openai_client, st.session_state.selected_source)) 