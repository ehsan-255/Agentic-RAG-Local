# Developer Guide: UI Component

This guide provides technical documentation for developers who need to integrate with, extend, or modify the user interface component of the Agentic RAG system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [UI Structure](#ui-structure)
4. [Integration Points](#integration-points)
5. [State Management](#state-management)
6. [Extending the UI](#extending-the-ui)
7. [Best Practices](#best-practices)

## Architecture Overview

The UI component is built using Streamlit to provide an interactive web interface for the Agentic RAG system:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Streamlit      │       │   Application   │       │    Database     │
│  UI Components  │──────▶│   Logic         │──────▶│    Access       │
└─────────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Session State  │       │   RAG System    │       │    OpenAI       │
│  Management     │◀─────▶│   Integration   │◀─────▶│    API Client   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

The architecture follows these key principles:

1. **Component-Based UI**: Organized into functional sections
2. **Reactive Updates**: Automatic UI re-rendering on state changes
3. **Asynchronous Operations**: Non-blocking for responsive UI
4. **Session State Management**: Persistent state across interactions
5. **Modular Design**: Separated concerns for maintainability

## Key Components

### 1. Main Application (`src/ui/streamlit_app.py`)

The main Streamlit application that serves as the entry point:

```python
from src.ui.streamlit_app import main

# Entry point for the Streamlit application
if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Chat Interface

Manages the conversation with the RAG system:

```python
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the documentation"):
    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()}
    )
    
    # Display the message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display AI response
    with st.chat_message("assistant"):
        await run_agent_with_streaming(prompt)
```

### 3. Documentation Source Management

Handles adding, viewing, and removing documentation sources:

```python
# Display documentation sources
sources = get_documentation_sources_sync()
for source in sources:
    with st.expander(f"{source['name']} ({source['source_id']})"):
        st.write(f"Base URL: {source['base_url']}")
        st.write(f"Pages: {source['pages_count']}")
        st.write(f"Chunks: {source['chunks_count']}")
        
        # Delete button
        if st.button(f"Delete {source['name']}", key=f"delete_{source['source_id']}"):
            if delete_documentation_source(source['source_id']):
                st.success(f"Deleted {source['name']}")
                st.rerun()
```

### 4. Monitoring Dashboard

Provides visualization of system metrics and status:

```python
from src.ui.monitoring_ui import monitoring_dashboard

# Display monitoring dashboard in a tab
tabs = st.tabs(["Chat", "Monitoring"])
with tabs[1]:
    monitoring_dashboard()
```

## UI Structure

### Main Layout

The UI is organized into a tabbed interface with main functional areas:

1. **Chat Tab**: The main interaction area with the RAG system
2. **Monitoring Tab**: Visualizations and metrics about system performance
3. **Sidebar**: Documentation source management and configuration

### Chat Interface

The chat interface consists of:

1. **Message History**: Displays the conversation history
2. **Input Box**: Accepts user questions
3. **Response Area**: Shows the RAG system's responses with citations
4. **Source Selection**: Optional dropdown to limit search to specific sources

### Documentation Source Management

The documentation source management section includes:

1. **Source List**: Shows all available documentation sources
2. **Add Source Form**: Interface for adding new documentation sources
3. **Advanced Configuration**: Expandable section for advanced crawl settings
4. **Source Actions**: Buttons for deleting or recrawling sources

### Monitoring Dashboard

The monitoring dashboard provides visual feedback on:

1. **Crawler Status**: Real-time status of active crawls
2. **System Metrics**: Charts showing performance metrics
3. **API Usage**: Tracking of API rate limits and usage
4. **Error Logs**: Visualization of error patterns and statistics

## Integration Points

### Integrating with RAG System

The UI integrates with the RAG system through the agent interface:

```python
from src.rag.rag_expert import agentic_rag_expert, AgentyRagDeps

async def run_agent_with_streaming(user_input: str, source_id: Optional[str] = None):
    """Run the RAG agent with streaming response."""
    # Prepare dependencies
    deps = AgentyRagDeps(openai_client=openai_client)
    
    # Run the agent with streaming
    async with agentic_rag_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],
        context={"source_id": source_id} if source_id else {}
    ) as result:
        # Display streaming response
        partial_text = ""
        placeholder = st.empty()
        
        async for text in result.text_deltas():
            partial_text += text
            placeholder.markdown(partial_text)
        
        # Update session state with complete response
        st.session_state.messages[-1]['content'] = partial_text
```

### Integrating with Crawler

The UI integrates with the crawler component:

```python
from src.crawling.enhanced_docs_crawler import crawl_documentation, CrawlConfig

async def add_and_crawl_documentation():
    """Add a new documentation source and start crawling."""
    # Get form values
    name = st.session_state.source_name
    sitemap_url = st.session_state.sitemap_url
    
    # Create config from form values
    config = {
        "chunk_size": st.session_state.chunk_size,
        "max_concurrent_crawls": st.session_state.max_concurrent_crawls,
        "max_concurrent_api_calls": st.session_state.max_concurrent_api_calls,
        "url_patterns_include": st.session_state.url_patterns_include.split('\n') if st.session_state.url_patterns_include else [],
        "url_patterns_exclude": st.session_state.url_patterns_exclude.split('\n') if st.session_state.url_patterns_exclude else []
    }
    
    # Start crawl with progress indicator
    with st.status("Crawling documentation...") as status:
        success = await crawl_new_documentation(name, sitemap_url, config)
        
        if success:
            status.update(label="Crawl completed successfully!", state="complete")
        else:
            status.update(label="Crawl failed!", state="error")
```

### Integrating with Database

The UI accesses the database for source information:

```python
from src.db.schema import get_documentation_sources, get_source_statistics

# Get all documentation sources
sources = get_documentation_sources()

# Display source statistics
for source in sources:
    stats = get_source_statistics(source["source_id"])
    st.metric(
        label=source["name"],
        value=f"{stats['chunks_count']} chunks",
        delta=f"{stats['pages_count']} pages"
    )
```

## State Management

### Session State

Streamlit's session state is used to maintain application state:

```python
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_source" not in st.session_state:
    st.session_state.selected_source = None

if "crawl_in_progress" not in st.session_state:
    st.session_state.crawl_in_progress = False
```

### Form State

Forms are used for structured data input:

```python
with st.form("add_source_form"):
    st.text_input("Documentation Name", key="source_name")
    st.text_input("Sitemap URL", key="sitemap_url", 
                  help="URL to a sitemap XML file")
    
    with st.expander("Advanced Options"):
        st.number_input("Chunk Size", min_value=1000, 
                       value=5000, key="chunk_size")
        st.number_input("Max Concurrent Requests", min_value=1, 
                       value=5, key="max_concurrent_crawls")
    
    submitted = st.form_submit_button("Add and Crawl")
    if submitted:
        asyncio.create_task(add_and_crawl_documentation())
```

### Callback Pattern

Callbacks handle user interactions:

```python
def on_source_select():
    # Update the selected source when dropdown changes
    source_id = st.session_state.source_dropdown
    st.session_state.selected_source = source_id

# Create dropdown with callback
st.selectbox(
    "Filter by Documentation Source", 
    options=[None] + [s["source_id"] for s in sources],
    format_func=lambda x: "All Sources" if x is None else next((s["name"] for s in sources if s["source_id"] == x), x),
    key="source_dropdown",
    on_change=on_source_select
)
```

## Extending the UI

### Adding New Pages

To add a new page to the Streamlit application:

```python
# Creating a multi-page app
from src.ui.pages import settings_page, analytics_page

# Create navigation in sidebar
page = st.sidebar.radio(
    "Navigation",
    ["Chat", "Analytics", "Settings"]
)

# Display selected page
if page == "Chat":
    chat_page()
elif page == "Analytics":
    analytics_page()
elif page == "Settings":
    settings_page()
```

### Creating Custom Components

To create a reusable UI component:

```python
def source_card(source):
    """Display a documentation source as a card."""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(source["name"])
            st.caption(f"Source ID: {source['source_id']}")
            st.text(f"Base URL: {source['base_url']}")
            
            # Display progress metrics
            st.progress(min(source["chunks_count"] / 1000, 1.0))
            st.text(f"{source['chunks_count']} chunks from {source['pages_count']} pages")
        
        with col2:
            # Action buttons
            st.button("Recrawl", key=f"recrawl_{source['source_id']}")
            st.button("Delete", key=f"delete_{source['source_id']}")
    
    st.divider()

# Use the component
for source in sources:
    source_card(source)
```

### Adding Visualizations

To add custom visualizations:

```python
import plotly.express as px
import pandas as pd

def error_distribution_chart(error_data):
    """Create a chart showing error distribution by type."""
    # Convert error data to DataFrame
    df = pd.DataFrame(error_data)
    
    # Create pie chart
    fig = px.pie(
        df, 
        values='count', 
        names='error_type',
        title='Error Distribution by Type',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Use the visualization
error_data = [
    {"error_type": "CONNECTION", "count": 15},
    {"error_type": "CONTENT_PROCESSING", "count": 8},
    {"error_type": "API_RATE_LIMIT", "count": 3}
]
error_distribution_chart(error_data)
```

## Best Practices

### UI Design

1. **Progressive Disclosure**:
   - Hide complex options behind expanders
   - Use tabs to organize different functional areas
   - Provide tooltips for advanced features

2. **Feedback and Status**:
   - Show operation status with progress bars or spinners
   - Provide clear success and error messages
   - Use toast notifications for transient feedback

3. **Responsive Layout**:
   - Use columns and containers for flexible layouts
   - Test with different screen sizes
   - Consider mobile users with simpler layouts

```python
# Responsive layout example
col1, col2 = st.columns([2, 1])

with col1:
    # Main content
    st.header("Documentation Explorer")
    # More content...

with col2:
    # Sidebar-like content in column layout
    st.subheader("Filters")
    # Filter controls...
```

### Performance Optimization

1. **Caching**:
   - Use `@st.cache_data` for expensive data operations
   - Use `@st.cache_resource` for resource objects like API clients
   - Cache database queries appropriately

2. **Lazy Loading**:
   - Load data only when needed
   - Use expanders and tabs to delay rendering
   - Implement pagination for large datasets

3. **Minimize Re-renders**:
   - Use `st.empty()` and `placeholder.update()` for dynamic content
   - Update session state efficiently
   - Use callbacks wisely

```python
# Caching example
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_documentation_sources_cached():
    """Cached function to get documentation sources."""
    return get_documentation_sources_sync()

# Use cached function
sources = get_documentation_sources_cached()
```

### Error Handling

1. **Graceful Degradation**:
   - Handle API failures without crashing the UI
   - Show useful error messages
   - Provide fallback options when services are unavailable

2. **Input Validation**:
   - Validate user inputs before submission
   - Provide immediate feedback for invalid inputs
   - Use form validation when appropriate

3. **Exception Handling**:
   - Catch and handle exceptions in async operations
   - Log errors for debugging
   - Display user-friendly error messages

```python
# Error handling example
try:
    with st.spinner("Fetching data..."):
        result = await fetch_data()
    st.success("Data loaded successfully!")
except ConnectionError:
    st.error("Could not connect to the server. Please try again later.")
except Exception as e:
    st.exception(e)
    st.error(f"An unexpected error occurred: {str(e)}")
```

### State Management

1. **Session State Best Practices**:
   - Initialize all state variables at app startup
   - Use consistent naming conventions
   - Keep state updates localized

2. **Form Handling**:
   - Use Streamlit forms for multi-input submissions
   - Validate form data before processing
   - Provide clear feedback after submission

3. **Persistent Settings**:
   - Save user preferences between sessions
   - Implement settings export/import
   - Use local storage for persistent configurations

```python
# Persistent settings example
import json
import os

def save_settings():
    """Save settings to a file."""
    settings = {
        "theme": st.session_state.theme,
        "chunk_size": st.session_state.default_chunk_size,
        "max_results": st.session_state.default_max_results
    }
    
    with open("settings.json", "w") as f:
        json.dump(settings, f)
    
    st.success("Settings saved successfully!")

def load_settings():
    """Load settings from a file."""
    if os.path.exists("settings.json"):
        with open("settings.json", "r") as f:
            settings = json.load(f)
            
        for key, value in settings.items():
            st.session_state[key] = value
        
        st.success("Settings loaded successfully!")
    else:
        st.warning("No saved settings found.")
``` 