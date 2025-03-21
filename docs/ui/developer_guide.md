# Developer Guide: UI Component

This guide provides technical documentation for developers who need to integrate with, extend, or modify the user interface (UI) component of the Agentic RAG system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [State Management](#state-management)
4. [Integration Points](#integration-points)
5. [Extending the UI](#extending-the-ui)
6. [Best Practices](#best-practices)

## Architecture Overview

The UI is built using Streamlit, a Python framework for creating data-driven web applications. The architecture follows a component-based design pattern, with clear separation of concerns:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  UI Components  │─────▶│  Controllers    │─────▶│  Services       │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        │
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  State Manager  │◀────▶│  Event Handlers │◀────▶│  API Client     │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

The architecture follows these key principles:

1. **Component-Based UI**: The interface is organized into reusable components with specific responsibilities.
2. **Reactive Updates**: UI elements automatically update when underlying data changes.
3. **Asynchronous Operations**: Long-running operations run asynchronously to maintain UI responsiveness.
4. **Session State Management**: Persistent state management for user sessions.
5. **Modular Design**: Components are designed to be self-contained and independently maintainable.

## Key Components

### 1. Main Application (`src/ui/streamlit_app.py`)

The main application entry point that configures Streamlit and loads page components:

```python
import streamlit as st
from src.ui.pages import home, query, documents, settings, monitoring
from src.ui.state import init_session_state
from src.ui.components.sidebar import render_sidebar

def main():
    """Main entry point for the Streamlit application."""
    # Configure page settings
    st.set_page_config(
        page_title="Agentic RAG System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar for navigation
    selected_page = render_sidebar()
    
    # Render the selected page
    if selected_page == "Home":
        home.render()
    elif selected_page == "Query":
        query.render()
    elif selected_page == "Documents":
        documents.render()
    elif selected_page == "Settings":
        settings.render()
    elif selected_page == "Monitoring":
        monitoring.render()

if __name__ == "__main__":
    main()
```

### 2. State Management (`src/ui/state.py`)

Manages application state across different components:

```python
import streamlit as st
from typing import Any, Dict, List, Optional

def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = []
    
    if "doc_sources" not in st.session_state:
        st.session_state.doc_sources = []
    
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "model": "gpt-4-turbo",
            "temperature": 0.7,
            "max_context_chunks": 10,
            "search_strategy": "hybrid",
            "use_tools": True
        }
    
    if "auth" not in st.session_state:
        st.session_state.auth = {
            "authenticated": False,
            "user": None
        }

def get_state_value(key: str, default: Any = None) -> Any:
    """Get a value from session state with fallback to default."""
    return st.session_state.get(key, default)

def set_state_value(key: str, value: Any):
    """Set a value in session state."""
    st.session_state[key] = value

def update_query_history(query: str, response: str):
    """Add a query-response pair to the history."""
    st.session_state.query_history.append({
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat()
    })

def clear_conversation():
    """Clear the current conversation history."""
    st.session_state.current_conversation = []

def save_settings(settings: Dict[str, Any]):
    """Save user settings to session state."""
    st.session_state.settings = settings
```

### 3. Page Components (`src/ui/pages/`)

Each page in the UI is encapsulated in its own module with a `render()` function:

```python
# src/ui/pages/query.py
import streamlit as st
from src.ui.components.query_box import render_query_box
from src.ui.components.response_display import render_response
from src.ui.components.conversation_history import render_conversation_history
from src.ui.state import get_state_value

def render():
    """Render the query page."""
    st.title("Query the Knowledge Base")
    
    # Render the query input box
    render_query_box()
    
    # Render conversation history
    conversation = get_state_value("current_conversation", [])
    if conversation:
        render_conversation_history(conversation)
    
    # Render information about available document sources
    st.sidebar.subheader("Available Knowledge Sources")
    doc_sources = get_state_value("doc_sources", [])
    for source in doc_sources:
        st.sidebar.write(f"• {source['name']}")
```

### 4. UI Components (`src/ui/components/`)

Reusable UI components that can be composed to build pages:

```python
# src/ui/components/query_box.py
import streamlit as st
from src.ui.state import get_state_value, set_state_value
from src.rag.services import process_query

def render_query_box():
    """Render the query input box and handle query submission."""
    query = st.text_area("Enter your question", height=100)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Submit", use_container_width=True):
            if query:
                with st.spinner("Processing your query..."):
                    # Get current settings
                    settings = get_state_value("settings", {})
                    
                    # Process the query
                    response = process_query(
                        query=query,
                        conversation=get_state_value("current_conversation", []),
                        model=settings.get("model", "gpt-4-turbo"),
                        temperature=settings.get("temperature", 0.7),
                        max_context_chunks=settings.get("max_context_chunks", 10),
                        search_strategy=settings.get("search_strategy", "hybrid"),
                        use_tools=settings.get("use_tools", True)
                    )
                    
                    # Update conversation
                    conversation = get_state_value("current_conversation", [])
                    conversation.append({"role": "user", "content": query})
                    conversation.append({"role": "assistant", "content": response})
                    set_state_value("current_conversation", conversation)
                    
                    # Force rerun to display the response
                    st.experimental_rerun()
    
    with col2:
        if st.button("Clear Conversation", use_container_width=True):
            set_state_value("current_conversation", [])
            st.experimental_rerun()
```

### 5. Services Integration (`src/ui/services/`)

Services that connect the UI to backend functionality:

```python
# src/ui/services/api_client.py
import httpx
from typing import Dict, Any, List, Optional

class ApiClient:
    """Client for interacting with backend API services."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def query(self, query_text: str, **params) -> Dict[str, Any]:
        """Send a query to the RAG system."""
        response = await self.client.post(
            f"{self.base_url}/api/query",
            json={"query": query_text, **params}
        )
        response.raise_for_status()
        return response.json()
    
    async def get_documents(self) -> List[Dict[str, Any]]:
        """Get list of document sources."""
        response = await self.client.get(f"{self.base_url}/api/documents")
        response.raise_for_status()
        return response.json()
    
    async def add_document_source(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new document source."""
        response = await self.client.post(
            f"{self.base_url}/api/documents",
            json=source_data
        )
        response.raise_for_status()
        return response.json()
    
    async def delete_document_source(self, source_id: str) -> Dict[str, Any]:
        """Delete a document source."""
        response = await self.client.delete(
            f"{self.base_url}/api/documents/{source_id}"
        )
        response.raise_for_status()
        return response.json()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics for monitoring."""
        response = await self.client.get(f"{self.base_url}/api/metrics")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
```

## State Management

### Session State

Streamlit's session state is used to maintain user session data:

```python
# Reading session state
user_settings = st.session_state.settings

# Writing to session state
st.session_state.search_results = results

# Checking if a key exists
if "authenticated" in st.session_state:
    # User is authenticated
    pass
```

### State Persistence

For more persistent storage across sessions:

```python
import json
import os

def save_settings_to_disk(user_id, settings):
    """Save user settings to disk."""
    os.makedirs("data/user_settings", exist_ok=True)
    with open(f"data/user_settings/{user_id}.json", "w") as f:
        json.dump(settings, f)

def load_settings_from_disk(user_id):
    """Load user settings from disk."""
    try:
        with open(f"data/user_settings/{user_id}.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
```

## Integration Points

### 1. With RAG Component

```python
from src.rag.rag_expert import agentic_rag_expert, AgentyRagDeps
from src.rag.agents.rag_agent import RagAgentConfig
from src.rag.context_manager import ContextManager

async def process_ui_query(query, conversation_history):
    """Process a query from the UI."""
    # Create dependencies
    deps = AgentyRagDeps()
    
    # Format conversation history
    formatted_history = []
    for message in conversation_history:
        formatted_history.append({
            "role": message["role"],
            "content": message["content"]
        })
    
    # Initialize context manager
    context_manager = ContextManager()
    
    # Retrieve relevant contexts
    contexts = await context_manager.get_context_for_query(
        query, conversation_history=formatted_history
    )
    
    # Define configuration
    config = RagAgentConfig(
        use_tools=st.session_state.settings.get("use_tools", True),
        conversation_history=formatted_history
    )
    
    # Process query with RAG agent
    response = await agentic_rag_expert(
        query=query,
        contexts=contexts,
        deps=deps,
        config=config
    )
    
    return response
```

### 2. With Database Component

```python
from src.db.documents import (
    get_documentation_sources,
    add_documentation_source,
    delete_documentation_source
)

async def load_document_sources():
    """Load document sources from the database."""
    sources = await get_documentation_sources()
    st.session_state.doc_sources = sources

async def add_new_document_source(name, url, description):
    """Add a new document source through the UI."""
    source = {
        "name": name,
        "url": url,
        "description": description
    }
    
    result = await add_documentation_source(**source)
    
    # Reload document sources
    await load_document_sources()
    
    return result
```

### 3. With Monitoring Component

```python
from src.monitoring.metrics import MetricsCollector
from src.monitoring.dashboard import create_performance_dashboard

async def display_system_metrics():
    """Display system metrics in the UI."""
    # Get metrics collector
    metrics = MetricsCollector()
    
    # Get summary metrics
    summary = metrics.get_summary()
    
    # Display in the UI
    st.subheader("System Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Query Success Rate", 
            value=f"{summary['rag_processing']['success_rate']:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Avg Query Time", 
            value=f"{summary['rag_processing']['avg_time_ms']:.0f} ms"
        )
    
    with col3:
        st.metric(
            label="Queries Today", 
            value=summary['rag_queries_total']['today']
        )
    
    # Generate and display a dashboard
    dashboard_html = await create_performance_dashboard(
        time_range="last_24h",
        return_html=True
    )
    
    st.components.v1.html(dashboard_html, height=600)
```

## Extending the UI

### 1. Adding a New Page

To add a new page to the UI:

1. Create a new module in `src/ui/pages/`
2. Implement a `render()` function
3. Update the sidebar navigation in `src/ui/components/sidebar.py`
4. Update the main application in `src/ui/streamlit_app.py`

Example of a new analytics page:

```python
# src/ui/pages/analytics.py
import streamlit as st
import pandas as pd
import plotly.express as px
from src.ui.services.analytics import get_usage_data

def render():
    """Render the analytics page."""
    st.title("Usage Analytics")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    
    # Get analytics data
    usage_data = get_usage_data(start_date, end_date)
    
    # Display usage charts
    st.subheader("Query Volume")
    df = pd.DataFrame(usage_data["query_volume"])
    fig = px.line(df, x="date", y="count", title="Daily Query Volume")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display topic distribution
    st.subheader("Popular Topics")
    topic_df = pd.DataFrame(usage_data["topics"])
    fig2 = px.pie(topic_df, values="count", names="topic", title="Query Topics")
    st.plotly_chart(fig2, use_container_width=True)
```

Update sidebar navigation:

```python
# src/ui/components/sidebar.py
def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("Navigation")
    
    pages = [
        "Home",
        "Query",
        "Documents",
        "Analytics",  # New page
        "Settings",
        "Monitoring"
    ]
    
    return st.sidebar.radio("Go to", pages)
```

### 2. Creating a Custom Component

To create a reusable UI component:

```python
# src/ui/components/file_uploader.py
import streamlit as st
from src.ui.services.document_processor import process_uploaded_file

def render_file_uploader(allowed_types=None):
    """Render a file uploader component.
    
    Args:
        allowed_types: List of allowed file extensions
    
    Returns:
        The uploaded file if successful, None otherwise
    """
    if allowed_types is None:
        allowed_types = ["pdf", "docx", "txt"]
    
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=allowed_types,
        help=f"Supported formats: {', '.join(allowed_types)}"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.write(f"Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Process button
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Process the file
                result = process_uploaded_file(uploaded_file)
                
                # Show result
                if result["success"]:
                    st.success("Document processed successfully!")
                    st.json(result["metadata"])
                    return uploaded_file
                else:
                    st.error(f"Error processing document: {result['error']}")
    
    return None
```

### 3. Implementing Custom Visualizations

Creating custom visualizations:

```python
# src/ui/components/vector_space_visualization.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.rag.embeddings import get_embeddings
from src.ui.services.vector_service import reduce_dimensions

def render_vector_space(query, contexts, width=800, height=600):
    """Render a 3D visualization of vector space with query and contexts.
    
    Args:
        query: The user query
        contexts: List of context texts
        width: Visualization width
        height: Visualization height
    """
    # Get embeddings
    texts = [query] + [ctx["text"] for ctx in contexts]
    embeddings = get_embeddings(texts)
    
    # Reduce to 3D for visualization
    points_3d = reduce_dimensions(embeddings, n_components=3)
    
    # Prepare data
    x, y, z = zip(*points_3d)
    
    # Create labels
    labels = ["Query"] + [f"Context {i+1}" for i in range(len(contexts))]
    
    # Create colors (query in red, contexts in blue)
    colors = ["red"] + ["blue"] * len(contexts)
    
    # Create sizes (query larger than contexts)
    sizes = [10] + [7] * len(contexts)
    
    # Create figure
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8
        ),
        text=labels,
        hovertext=[query] + [ctx["text"][:100] + "..." for ctx in contexts],
        hoverinfo="text"
    )])
    
    # Update layout
    fig.update_layout(
        title="Vector Space Visualization",
        width=width,
        height=height,
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        )
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
```

## Best Practices

### 1. UI Organization

Organize UI code for maintainability:

```
src/ui/
├── streamlit_app.py      # Main entry point
├── state.py              # Session state management
├── pages/                # Page modules
│   ├── __init__.py
│   ├── home.py
│   ├── query.py
│   └── ...
├── components/           # Reusable UI components
│   ├── __init__.py
│   ├── sidebar.py
│   ├── query_box.py
│   └── ...
└── services/             # Backend services integration
    ├── __init__.py
    ├── api_client.py
    └── ...
```

### 2. Responsive Design

Design for different screen sizes:

```python
# Use columns for responsive layouts
col1, col2 = st.columns([1, 2])  # 1:2 ratio

with col1:
    st.write("Sidebar content")
    
with col2:
    st.write("Main content")

# For mobile-friendly design, use full width on small screens
is_mobile = st.experimental_get_query_params().get("view", [""])[0] == "mobile"

if is_mobile:
    # Single column layout for mobile
    st.write("Mobile layout")
else:
    # Multi-column layout for desktop
    col1, col2, col3 = st.columns(3)
    # ...
```

### 3. Performance Optimization

Optimize UI performance:

```python
# Use caching for expensive operations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_document_data():
    # Expensive operation to fetch data
    return data

# Avoid recomputing on every interaction
if "computed_results" not in st.session_state:
    st.session_state.computed_results = perform_expensive_computation()
results = st.session_state.computed_results

# Load large data progressively
if st.button("Load More Data"):
    # Append to existing data
    st.session_state.data.extend(fetch_next_batch())
```

### 4. Error Handling

Implement robust error handling:

```python
try:
    with st.spinner("Processing..."):
        result = process_data()
    st.success("Processing complete!")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    
    # Log the error
    import logging
    logging.exception("Error in UI processing")
    
    # Display technical details in an expander
    with st.expander("Technical Details"):
        st.code(traceback.format_exc())
```

### 5. State Management

Best practices for state management:

```python
# DO: Initialize state in a central location
def init_session_state():
    # Initialize with default values
    if "key" not in st.session_state:
        st.session_state.key = default_value

# DO: Use functions to access state
def get_value(key, default=None):
    return st.session_state.get(key, default)

def set_value(key, value):
    st.session_state[key] = value

# DON'T: Mix different state management approaches
# state = {} # Bad - using both global variables and session state

# DO: Group related state in a dictionary
if "settings" not in st.session_state:
    st.session_state.settings = {
        "theme": "light",
        "language": "en"
    }

# Update nested state
settings = st.session_state.settings.copy()
settings["theme"] = new_theme
st.session_state.settings = settings
``` 