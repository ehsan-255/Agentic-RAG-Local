import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import psutil
import matplotlib.pyplot as plt
import subprocess

from src.utils.enhanced_logging import (
    get_session_stats, 
    get_active_session,
    get_system_metrics,
    get_error_stats
)
from src.utils.task_monitoring import (
    get_task_stats,
    get_active_tasks,
    get_failed_tasks,
    cancel_all_tasks,
    TaskState
)
from src.utils.db_monitoring import (
    get_connection_pool_stats,
    get_transaction_stats,
    get_query_stats
)
from src.utils.api_monitoring import (
    get_api_stats,
    get_rate_limits
)
from src.crawling.crawl_state import (
    get_saved_configurations,
    load_crawl_configuration,
    save_crawl_configuration,
    prepare_resume_crawl
)
from src.db.schema import (
    get_documentation_sources as get_documentation_sources_sync,
    get_source_statistics
)
from src.db.connection import get_connection_stats

# Store historical metrics for trend visualization
if 'historical_metrics' not in st.session_state:
    st.session_state.historical_metrics = {
        'timestamps': [],
        'cpu_percent': [],
        'memory_mb': [],
        'active_tasks': [],
        'failed_tasks': [],
        'success_rate': [],
    }

def update_historical_metrics():
    """Update historical metrics for trend visualization."""
    # Get current metrics
    system_metrics = get_system_metrics()
    task_stats = get_task_stats()
    session_stats = get_session_stats()
    
    # Add timestamp
    now = datetime.now()
    st.session_state.historical_metrics['timestamps'].append(now)
    
    # Add system metrics
    st.session_state.historical_metrics['cpu_percent'].append(
        system_metrics.get('cpu_percent', 0)
    )
    st.session_state.historical_metrics['memory_mb'].append(
        system_metrics.get('memory_rss_mb', 0)
    )
    
    # Add task metrics
    st.session_state.historical_metrics['active_tasks'].append(
        task_stats.get('running_tasks', 0) + task_stats.get('pending_tasks', 0)
    )
    st.session_state.historical_metrics['failed_tasks'].append(
        task_stats.get('failed_tasks', 0)
    )
    
    # Add session metrics
    success_rate = 0
    if session_stats:
        success_rate = session_stats.get('success_rate', 0)
    st.session_state.historical_metrics['success_rate'].append(success_rate)
    
    # Only keep the last 100 data points
    max_points = 100
    for key in st.session_state.historical_metrics:
        if len(st.session_state.historical_metrics[key]) > max_points:
            st.session_state.historical_metrics[key] = st.session_state.historical_metrics[key][-max_points:]

def update_monitoring_data():
    """Update monitoring data without affecting crawl state."""
    # Save current crawl state before updating
    pause_state = st.session_state.get('pause_crawl', False)
    stop_state = st.session_state.get('stop_crawl', False)
    crawl_config = st.session_state.get('crawl_config', {})
    crawl_start_time = st.session_state.get('crawl_start_time', None)
    
    # Update historical metrics
    update_historical_metrics()
    
    # Get fresh data
    system_metrics = get_system_metrics()
    task_stats = get_task_stats()
    connection_stats = get_connection_stats()
    api_stats = get_api_stats()
    
    # Store in session state for visualization
    if "monitoring_data" not in st.session_state:
        st.session_state.monitoring_data = {}
    
    st.session_state.monitoring_data.update({
        "system": system_metrics,
        "tasks": task_stats,
        "db": connection_stats,
        "api": api_stats,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Restore crawl state after updating
    if 'pause_crawl' in st.session_state:
        st.session_state.pause_crawl = pause_state
    if 'stop_crawl' in st.session_state:
        st.session_state.stop_crawl = stop_state
    if 'crawl_config' in st.session_state:
        st.session_state.crawl_config = crawl_config
    if 'crawl_start_time' in st.session_state:
        st.session_state.crawl_start_time = crawl_start_time
    
    return True

def display_system_metrics():
    """Display system resource metrics."""
    metrics = get_system_metrics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "CPU Usage", 
            f"{metrics.get('cpu_percent', 0):.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Memory Usage", 
            f"{metrics.get('memory_rss_mb', 0):.1f} MB",
            delta=None
        )
    
    with col3:
        st.metric(
            "System Memory", 
            f"{metrics.get('system_memory_percent', 0):.1f}%",
            delta=None
        )
    
    # Display thread count
    st.metric(
        "Thread Count",
        metrics.get('thread_count', 0),
        delta=None
    )
    
    # Display historical CPU and memory usage
    if len(st.session_state.historical_metrics['timestamps']) > 1:
        # Convert to pandas DataFrame for plotting
        df = pd.DataFrame({
            'timestamp': st.session_state.historical_metrics['timestamps'],
            'cpu_percent': st.session_state.historical_metrics['cpu_percent'],
            'memory_mb': st.session_state.historical_metrics['memory_mb']
        })
        
        # Display CPU usage chart
        fig1 = px.line(
            df, x='timestamp', y='cpu_percent',
            title='CPU Usage Over Time',
            labels={'cpu_percent': 'CPU %', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Display memory usage chart
        fig2 = px.line(
            df, x='timestamp', y='memory_mb',
            title='Memory Usage Over Time',
            labels={'memory_mb': 'Memory (MB)', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig2, use_container_width=True)

def display_crawl_controls(suffix=""):
    """
    Display crawl control buttons.
    
    Args:
        suffix: Optional suffix to add to button keys to ensure uniqueness
    """
    st.subheader("Crawl Controls")
    
    # Get current crawl state
    is_paused = st.session_state.get('pause_crawl', False)
    is_stopped = st.session_state.get('stop_crawl', False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Disable pause button if already paused or stopped
        pause_disabled = is_paused or is_stopped
        if st.button("Pause Crawl", disabled=pause_disabled, use_container_width=True, key=f"btn_pause_crawl{suffix}"):
            st.session_state.pause_crawl = True
            st.success("Crawl paused. In-progress items will complete.")
    
    with col2:
        # Disable resume button if not paused or if stopped
        resume_disabled = not is_paused or is_stopped
        if st.button("Resume Crawl", disabled=resume_disabled, use_container_width=True, key=f"btn_resume_crawl{suffix}"):
            st.session_state.pause_crawl = False
            st.success("Crawl resumed.")
    
    with col3:
        # Disable stop button if already stopped
        stop_disabled = is_stopped
        if st.button("Stop Crawl", disabled=stop_disabled, use_container_width=True, type="primary", key=f"btn_stop_crawl{suffix}"):
            st.session_state.stop_crawl = True
            # Ensure not paused when stopped
            st.session_state.pause_crawl = False
            st.error("Crawl stopping. This may take a moment to complete.")

def display_crawl_status():
    """Display current crawl session status."""
    # Update monitoring data first without affecting crawl state
    if st.session_state.get("refresh_requested", False):
        update_monitoring_data()
        st.session_state.refresh_requested = False
    
    session = get_active_session()
    
    if not session:
        st.info("No active crawl session.")
        return
    
    # Get session stats
    stats = get_session_stats()
    if not stats:
        st.warning("Could not retrieve session statistics.")
        return
    
    # Display basic session info
    st.subheader(f"Crawl: {stats['source_name']}")
    st.caption(f"Session ID: {stats['session_id']} | Source ID: {stats['source_id']}")
    
    # Display start time and duration
    # Convert timestamp to datetime object - handle both float and string formats
    if isinstance(stats['start_time'], (int, float)):
        start_time = datetime.fromtimestamp(stats['start_time'])
    else:
        try:
            start_time = datetime.fromisoformat(stats['start_time'])
        except (TypeError, ValueError):
            start_time = datetime.now()  # Fallback if cannot parse
    
    # Use get() with a default value to avoid KeyError
    duration_seconds = stats.get('duration', 0)
    if isinstance(duration_seconds, (int, float)):
        duration = timedelta(seconds=duration_seconds)
    else:
        duration = timedelta(seconds=0)
    
    st.caption(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')} | Duration: {duration}")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Map key names correctly and use .get() to provide default values
    with col1:
        st.metric(
            "Pages Processed", 
            stats.get('processed_urls', stats.get('pages_processed', 0)),
            delta=None
        )
    
    with col2:
        st.metric(
            "Pages Succeeded", 
            stats.get('successful_urls', stats.get('pages_succeeded', 0)),
            delta=None
        )
    
    with col3:
        st.metric(
            "Pages Failed", 
            stats.get('failed_urls', stats.get('pages_failed', 0)),
            delta=None
        )
    
    with col4:
        st.metric(
            "Success Rate", 
            f"{stats.get('success_rate', 0) * 100:.1f}%",
            delta=None
        )
    
    # Display progress bar
    if stats.get('processed_urls', stats.get('pages_processed', 0)) > 0:
        st.progress(stats.get('success_rate', 0))
        
    # Simplified URL processing display
    processed_urls = getattr(session, 'processed_urls', None)
    failed_urls = getattr(session, 'failed_urls', None)
    
    if processed_urls is not None or failed_urls is not None:
        st.subheader("URL Processing Status")
        
        # Create tabs for different URL lists
        url_tabs = st.tabs(["Processed URLs", "Failed URLs"])
        
        # Processed URLs tab - simpler approach
        with url_tabs[0]:
            if isinstance(processed_urls, (list, set, tuple)) and processed_urls:
                # It's a collection
                st.dataframe({"URL": list(processed_urls)})
            elif isinstance(processed_urls, int) and processed_urls > 0:
                # It's a count
                st.info(f"Total processed URLs: {processed_urls}")
            else:
                st.info("No URLs have been processed yet.")
        
        # Failed URLs tab - simpler approach
        with url_tabs[1]:
            if isinstance(failed_urls, (list, set, tuple)) and failed_urls:
                # It's a collection
                st.dataframe({"URL": list(failed_urls)})
            elif isinstance(failed_urls, int) and failed_urls > 0:
                # It's a count
                st.info(f"Total failed URLs: {failed_urls}")
            else:
                st.success("No URLs have failed processing.")
    
    # Add the crawl controls
    display_crawl_controls("_status")

def display_task_monitoring():
    """Display task monitoring information."""
    # Initialize historical metrics if not already in session state
    if "historical_metrics" not in st.session_state:
        st.session_state.historical_metrics = {
            "timestamps": [],
            "active_tasks": [],
            "completed_tasks": [],
            "failed_tasks": []
        }
    
    # Get task statistics
    task_stats = get_task_stats()
    active_tasks = get_active_tasks()
    failed_tasks = get_failed_tasks()
    
    # Display task counts
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Tasks", 
            task_stats['total_tasks'],
            delta=None
        )
    
    with col2:
        active_count = task_stats['pending_tasks'] + task_stats['running_tasks']
        st.metric(
            "Active Tasks", 
            active_count,
            delta=None
        )
    
    with col3:
        st.metric(
            "Succeeded Tasks", 
            task_stats['succeeded_tasks'],
            delta=None
        )
    
    with col4:
        st.metric(
            "Failed Tasks", 
            task_stats['failed_tasks'],
            delta=None
        )
    
    # Display average duration
    if task_stats['avg_duration'] > 0:
        st.metric(
            "Average Task Duration", 
            f"{task_stats['avg_duration']:.2f} seconds",
            delta=None
        )
    
    # Display active tasks
    if active_tasks:
        with st.expander("Active Tasks", expanded=True):
            # Convert to DataFrame for display
            active_df = pd.DataFrame(active_tasks)
            
            # Select and reorder columns
            columns = [
                'task_id', 'task_type', 'description', 'state', 
                'created_at', 'started_at', 'url'
            ]
            display_columns = [col for col in columns if col in active_df.columns]
            
            # Display the DataFrame
            st.dataframe(active_df[display_columns])
    
    # Display failed tasks
    if failed_tasks:
        with st.expander("Failed Tasks", expanded=True):
            # Convert to DataFrame for display
            failed_df = pd.DataFrame(failed_tasks)
            
            # Select and reorder columns
            columns = [
                'task_id', 'task_type', 'description', 'error_message',
                'created_at', 'started_at', 'completed_at', 'duration', 'url'
            ]
            display_columns = [col for col in columns if col in failed_df.columns]
            
            # Display the DataFrame
            st.dataframe(failed_df[display_columns])
    
    # Display task type distribution
    if task_stats['tasks_by_type']:
        # Create dataframe for pie chart
        task_type_df = pd.DataFrame({
            'type': list(task_stats['tasks_by_type'].keys()),
            'count': list(task_stats['tasks_by_type'].values())
        })
        
        # Only show types with non-zero values
        task_type_df = task_type_df[task_type_df['count'] > 0]
        
        if not task_type_df.empty:
            fig = px.pie(
                task_type_df, values='count', names='type',
                title="Tasks by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Display historical task metrics
    if len(st.session_state.historical_metrics['timestamps']) > 1:
        # Convert to pandas DataFrame for plotting
        df = pd.DataFrame({
            'timestamp': st.session_state.historical_metrics['timestamps'],
            'active_tasks': st.session_state.historical_metrics['active_tasks'],
            'failed_tasks': st.session_state.historical_metrics['failed_tasks']
        })
        
        # Display active tasks chart
        fig1 = px.line(
            df, x='timestamp', y='active_tasks',
            title='Active Tasks Over Time',
            labels={'active_tasks': 'Active Tasks', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Display failed tasks chart
        fig2 = px.line(
            df, x='timestamp', y='failed_tasks',
            title='Failed Tasks Over Time',
            labels={'failed_tasks': 'Failed Tasks', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig2, use_container_width=True)

def display_db_stats():
    """Display database monitoring information."""
    pool_stats = get_connection_pool_stats()
    transaction_stats = get_transaction_stats()
    query_stats = get_query_stats()
    
    # Display connection pool stats
    st.subheader("Connection Pool")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Active Connections", 
            pool_stats.get('active_connections', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            "Idle Connections", 
            pool_stats.get('idle_connections', 0),
            delta=None
        )
    
    with col3:
        st.metric(
            "Max Connections", 
            pool_stats.get('max_connections', 0),
            delta=None
        )
    
    # Display connection pool utilization
    if pool_stats.get('max_connections', 0) > 0:
        pool_utilization = pool_stats.get('active_connections', 0) / pool_stats.get('max_connections', 1)
        st.progress(pool_utilization)
        st.caption(f"Connection Pool Utilization: {pool_utilization * 100:.1f}%")
    
    # Display transaction stats
    st.subheader("Transactions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Transactions", 
            transaction_stats.get('active_transactions', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            "Started", 
            transaction_stats.get('transactions_started', 0),
            delta=None
        )
    
    with col3:
        st.metric(
            "Committed", 
            transaction_stats.get('transactions_committed', 0),
            delta=None
        )
    
    with col4:
        st.metric(
            "Rolled Back", 
            transaction_stats.get('transactions_rolled_back', 0),
            delta=None
        )
    
    # Display query stats
    st.subheader("Queries")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Queries", 
            query_stats.get('total_queries', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            "Average Duration", 
            f"{query_stats.get('average_query_duration_ms', 0):.2f} ms",
            delta=None
        )
    
    with col3:
        st.metric(
            "Max Duration", 
            f"{query_stats.get('max_query_duration_ms', 0):.2f} ms",
            delta=None
        )
    
def display_cached_db_stats(db_stats):
    """Display database monitoring information using cached data."""
    # Display connection pool stats
    st.subheader("Connection Pool")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Active Connections", 
            db_stats.get('active_connections', 0),
            delta=None
        )

    with col2:
        st.metric(
            "Idle Connections", 
            db_stats.get('idle_connections', 0),
            delta=None
        )
    
    with col3:
        st.metric(
            "Max Connections", 
            db_stats.get('max_connections', 0),
            delta=None
        )
    
    # Display connection pool utilization
    if db_stats.get('max_connections', 0) > 0:
        pool_utilization = db_stats.get('active_connections', 0) / db_stats.get('max_connections', 1)
        st.progress(pool_utilization)
        st.caption(f"Connection Pool Utilization: {pool_utilization * 100:.1f}%")
    
    # Display transaction stats
    st.subheader("Transactions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Transactions", 
            db_stats.get('active_transactions', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            "Started", 
            db_stats.get('transactions_started', 0),
            delta=None
        )
    
    with col3:
        st.metric(
            "Committed", 
            db_stats.get('transactions_committed', 0),
            delta=None
        )
    
    with col4:
        st.metric(
            "Rolled Back", 
            db_stats.get('transactions_rolled_back', 0),
            delta=None
        )
    
    # Display query stats
    st.subheader("Queries")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Queries", 
            db_stats.get('total_queries', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            "Average Duration", 
            f"{db_stats.get('average_query_duration_ms', 0):.2f} ms",
            delta=None
        )
    
    with col3:
        st.metric(
            "Max Duration", 
            f"{db_stats.get('max_query_duration_ms', 0):.2f} ms",
            delta=None
        )

def display_api_stats():
    """Display API monitoring information."""
    api_stats = get_api_stats()
    rate_limits = get_rate_limits()
    
    # Display API call stats
    st.subheader("API Calls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total API Calls",
            api_stats.get('total_api_calls', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            "Average Duration", 
            f"{api_stats.get('average_api_call_duration_ms', 0):.2f} ms",
            delta=None
        )
    
    with col3:
        st.metric(
            "Rate Limit Errors", 
            api_stats.get('rate_limit_errors', 0),
            delta=None
        )
    
    # Display API call distribution
    if api_stats.get('calls_by_endpoint'):
        # Create dataframe for pie chart
        endpoint_df = pd.DataFrame({
            'endpoint': list(api_stats['calls_by_endpoint'].keys()),
            'count': list(api_stats['calls_by_endpoint'].values())
        })
        
        # Only show types with non-zero values
        endpoint_df = endpoint_df[endpoint_df['count'] > 0]
        
        if not endpoint_df.empty:
            fig = px.pie(
                endpoint_df, values='count', names='endpoint',
                title="API Calls by Endpoint"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Display rate limits
    st.subheader("Rate Limits")
    if rate_limits:
        # Create DataFrame for display
        rate_limit_df = pd.DataFrame(rate_limits)
        
        # Display the DataFrame
        st.dataframe(rate_limit_df)

def display_cached_api_stats(api_stats):
    """Display API monitoring information using cached data."""
    # Display API call stats
    st.subheader("API Calls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total API Calls",
            api_stats.get('total_api_calls', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            "Average Duration", 
            f"{api_stats.get('average_api_call_duration_ms', 0):.2f} ms",
            delta=None
        )
    
    with col3:
        st.metric(
            "Rate Limit Errors", 
            api_stats.get('rate_limit_errors', 0),
            delta=None
        )
    
    # Display API call distribution
    if api_stats.get('calls_by_endpoint'):
        # Create dataframe for pie chart
        endpoint_df = pd.DataFrame({
            'endpoint': list(api_stats['calls_by_endpoint'].keys()),
            'count': list(api_stats['calls_by_endpoint'].values())
        })
        
        # Only show types with non-zero values
        endpoint_df = endpoint_df[endpoint_df['count'] > 0]
        
        if not endpoint_df.empty:
            fig = px.pie(
                endpoint_df, values='count', names='endpoint',
                title="API Calls by Endpoint"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Display rate limits
    st.subheader("Rate Limits")
    rate_limits = api_stats.get('rate_limits', [])
    if rate_limits:
        # Create DataFrame for display
        rate_limit_df = pd.DataFrame(rate_limits)
        
        # Display the DataFrame
        st.dataframe(rate_limit_df)

def display_resume_crawl_options():
    """Display options to resume crawling existing sources."""
    st.subheader("Resume Crawling Existing Sources")
    
    # Get documentation sources
    sources = get_documentation_sources_sync()
    
    if not sources:
        st.info("No sources available to resume.")
        return
    
    # Create a dropdown to select a source
    source_options = [f"{source['name']} ({source['source_id']})" for source in sources]
    selected_source = st.selectbox(
        "Select Source to Resume",
        options=source_options,
        index=0
    )
    
    # Extract source ID from selection
    source_id = selected_source.split("(")[1].split(")")[0]
    
    # Show source statistics
    stats = get_source_statistics(source_id)
    if stats:
        st.write(f"Pages: {stats.get('pages_count', 0)} | Chunks: {stats.get('chunks_count', 0)}")
    
    # Resume button
    if st.button("Prepare Resume Configuration", use_container_width=True):
        config = prepare_resume_crawl(source_id)
        
        if config:
            # Save configuration to session state
            st.session_state.resume_config = config
            
            # Show confirmation
            st.success(f"Resume configuration prepared for {config['source_name']}.")
            st.write(f"Already processed URLs: {len(config['already_processed_urls'])}")
            
            # Option to save configuration
            save_name = st.text_input("Configuration Name (optional)", 
                                      value=f"resume_{config['source_name']}_{int(time.time())}")
            
            if st.button("Save Configuration", use_container_width=True):
                filename = save_crawl_configuration(config, save_name)
                st.success(f"Configuration saved to {filename}")
        else:
            st.error("Failed to prepare resume configuration.")

def display_error_tracking():
    """Display error tracking information."""
    # Get error statistics
    error_stats = get_error_stats()
    failed_tasks = get_failed_tasks()
    
    if error_stats:
        # Display error count by category
        st.subheader("Errors by Category")
        
        # Extract data for chart from the new structure
        if "by_category" in error_stats:
            categories = list(error_stats["by_category"].keys())
            counts = list(error_stats["by_category"].values())
            
            if sum(counts) > 0:
                # Create bar chart
                fig, ax = plt.subplots()
                ax.bar(categories, counts)
                ax.set_title('Error Count by Category')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
                
                # Display total errors
                st.metric("Total Errors", error_stats["total_errors"])
            else:
                st.info("No errors recorded yet.")
        else:
            st.info("No category data available.")
        
        # Display errors by type if available
        if "by_type" in error_stats and error_stats["by_type"]:
            st.subheader("Errors by Type")
            error_types_df = pd.DataFrame({
                "Error Type": list(error_stats["by_type"].keys()),
                "Count": list(error_stats["by_type"].values())
            })
            st.dataframe(error_types_df)
    
    # Display failed tasks
    if failed_tasks:
        st.subheader("Failed Tasks")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(failed_tasks)
        
        # Select relevant columns
        if not df.empty and 'error_message' in df.columns:
            display_columns = ['task_type', 'description', 'error_message']
            display_df = df[display_columns] if all(col in df.columns for col in display_columns) else df
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No failed tasks.")

def display_api_usage():
    """Display API usage statistics."""
    # If API monitoring is available in session state, use it
    if "api_monitoring" in st.session_state:
        api_stats = st.session_state.api_monitoring
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total API Calls", api_stats.get("total_calls", 0))
        with col2:
            st.metric("Embedding Calls", api_stats.get("embedding_calls", 0))
        with col3:
            st.metric("LLM Calls", api_stats.get("llm_calls", 0))
        
        # Display rate limit information if available
        if "rate_limits" in api_stats:
            st.subheader("Rate Limits")
            rate_limits = api_stats["rate_limits"]
            
            for endpoint, limit in rate_limits.items():
                st.write(f"{endpoint}: {limit['used']}/{limit['limit']} ({limit['used']/limit['limit']*100:.1f}%)")
    else:
        # Display placeholder for API monitoring
        st.info("API usage monitoring data will appear here during active crawls.")

def display_database_metrics():
    """Display database performance metrics."""
    # Get database connection stats
    db_stats = get_connection_stats()
    
    if db_stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Connections", db_stats.get("active_connections", 0))
        with col2:
            st.metric("Total Connections", db_stats.get("total_connections", 0))
        with col3:
            st.metric("Available Connections", db_stats.get("available_connections", 0))
        
        # If performance metrics are available
        if "query_time" in db_stats:
            st.subheader("Query Performance")
            st.write(f"Average Query Time: {db_stats['query_time']['avg']:.3f}s")
            st.write(f"Max Query Time: {db_stats['query_time']['max']:.3f}s")
    else:
        st.info("Database metrics will appear here during database operations.")

def display_system_resources():
    """Display system resource usage."""
    # Get current CPU and memory usage
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("CPU Usage", f"{cpu_percent}%")
        
        # CPU usage chart (historical if available)
        if "cpu_history" not in st.session_state:
            st.session_state.cpu_history = []
        
        # Add current value to history, keep last 20 values
        st.session_state.cpu_history.append(cpu_percent)
        if len(st.session_state.cpu_history) > 20:
            st.session_state.cpu_history = st.session_state.cpu_history[-20:]
        
        # Plot CPU history
        fig, ax = plt.subplots()
        ax.plot(st.session_state.cpu_history)
        ax.set_title('CPU Usage History')
        ax.set_ylabel('CPU %')
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    
    with col2:
        st.metric("Memory Usage", f"{memory.percent}%")
        
        # Memory usage chart (historical if available)
        if "memory_history" not in st.session_state:
            st.session_state.memory_history = []
        
        # Add current value to history, keep last 20 values
        st.session_state.memory_history.append(memory.percent)
        if len(st.session_state.memory_history) > 20:
            st.session_state.memory_history = st.session_state.memory_history[-20:]
        
        # Plot memory history
        fig, ax = plt.subplots()
        ax.plot(st.session_state.memory_history)
        ax.set_title('Memory Usage History')
        ax.set_ylabel('Memory %')
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    
    # Disk usage
    disk = psutil.disk_usage('/')
    st.metric("Disk Usage", f"{disk.percent}%", f"{disk.used / (1024**3):.1f} GB used of {disk.total / (1024**3):.1f} GB")

def display_enhanced_error_details():
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

def monitoring_dashboard():
    """Display a comprehensive monitoring dashboard."""
    # Check if refresh was requested
    if st.session_state.get("refresh_requested", False):
        # Use data from session state if available
        monitoring_data = st.session_state.get("monitoring_data", {})
    else:
        monitoring_data = {}
        
    # Create tabs for different monitoring sections
    monitoring_tabs = st.tabs([
        "Crawl Status", 
        "System Resources", 
        "Task Monitoring", 
        "Error Tracking", 
        "Database Stats",
        "API Stats"
    ])
    
    # Crawl Status tab
    with monitoring_tabs[0]:
        # First check for and display any enhanced error details
        has_error = display_enhanced_error_details()
        
        # Then show regular crawl status information
        display_crawl_status()
        display_crawl_controls("_dashboard")
    
    # System Resources tab
    with monitoring_tabs[1]:
        if "system" in monitoring_data:
            # Use cached system metrics data
            metrics = monitoring_data["system"]
            display_cached_system_metrics(metrics)
        else:
            # Fetch and display fresh system metrics
            display_system_resources()
    
    # Task Monitoring tab
    with monitoring_tabs[2]:
        if "tasks" in monitoring_data:
            # Use cached task stats data
            task_stats = monitoring_data["tasks"]
            display_cached_task_monitoring(task_stats)
        else:
            # Fallback to getting task stats directly
            display_task_monitoring()
    
    # Error Tracking tab
    with monitoring_tabs[3]:
        display_error_tracking()
    
    # Database Stats tab
    with monitoring_tabs[4]:
        if "db" in monitoring_data:
            # Use cached database stats data
            db_stats = monitoring_data["db"]
            display_cached_db_stats(db_stats)
        else:
            # Fallback to getting db stats directly
            display_db_stats()
    
    # API Stats tab
    with monitoring_tabs[5]:
        if "api" in monitoring_data:
            # Use cached API stats data
            api_stats = monitoring_data["api"]
            display_cached_api_stats(api_stats)
        else:
            # Fallback to getting API stats directly
            display_api_stats()

def display_cached_system_metrics(metrics):
    """Display system resource metrics using cached data."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "CPU Usage", 
            f"{metrics.get('cpu_percent', 0):.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Memory Usage", 
            f"{metrics.get('memory_rss_mb', 0):.1f} MB",
            delta=None
        )
    
    with col3:
        st.metric(
            "System Memory", 
            f"{metrics.get('system_memory_percent', 0):.1f}%",
            delta=None
        )
    
    # Display thread count
    st.metric(
        "Thread Count",
        metrics.get('thread_count', 0),
        delta=None
    )
    
    # Display historical CPU and memory usage
    if len(st.session_state.historical_metrics['timestamps']) > 1:
        # Convert to pandas DataFrame for plotting
        df = pd.DataFrame({
            'timestamp': st.session_state.historical_metrics['timestamps'],
            'cpu_percent': st.session_state.historical_metrics['cpu_percent'],
            'memory_mb': st.session_state.historical_metrics['memory_mb']
        })
        
        # Display CPU usage chart
        fig1 = px.line(
            df, x='timestamp', y='cpu_percent',
            title='CPU Usage Over Time',
            labels={'cpu_percent': 'CPU %', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Display memory usage chart
        fig2 = px.line(
            df, x='timestamp', y='memory_mb',
            title='Memory Usage Over Time',
            labels={'memory_mb': 'Memory (MB)', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig2, use_container_width=True)

def display_cached_task_monitoring(task_stats):
    """Display task monitoring information using cached data."""
    # Initialize historical metrics if not already in session state
    if "historical_metrics" not in st.session_state:
        st.session_state.historical_metrics = {
            "timestamps": [],
            "active_tasks": [],
            "completed_tasks": [],
            "failed_tasks": []
        }
    
    # Get task statistics from cached data
    active_tasks = get_active_tasks()
    failed_tasks = get_failed_tasks()
    
    # Display task counts
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Tasks", 
            task_stats['total_tasks'],
            delta=None
        )
    
    with col2:
        active_count = task_stats['pending_tasks'] + task_stats['running_tasks']
        st.metric(
            "Active Tasks", 
            active_count,
            delta=None
        )
    
    with col3:
        st.metric(
            "Succeeded Tasks", 
            task_stats['succeeded_tasks'],
            delta=None
        )
    
    with col4:
        st.metric(
            "Failed Tasks", 
            task_stats['failed_tasks'],
            delta=None
        )
    
    # Display average duration
    if task_stats['avg_duration'] > 0:
        st.metric(
            "Average Task Duration", 
            f"{task_stats['avg_duration']:.2f} seconds",
            delta=None
        )
    
    # Display active tasks
    if active_tasks:
        with st.expander("Active Tasks", expanded=True):
            # Convert to DataFrame for display
            active_df = pd.DataFrame(active_tasks)
            
            # Select and reorder columns
            columns = [
                'task_id', 'task_type', 'description', 'state', 
                'created_at', 'started_at', 'url'
            ]
            display_columns = [col for col in columns if col in active_df.columns]
            
            # Display the DataFrame
            st.dataframe(active_df[display_columns])
    
    # Display failed tasks
    if failed_tasks:
        with st.expander("Failed Tasks", expanded=True):
            # Convert to DataFrame for display
            failed_df = pd.DataFrame(failed_tasks)
            
            # Select and reorder columns
            columns = [
                'task_id', 'task_type', 'description', 'error_message',
                'created_at', 'started_at', 'completed_at', 'duration', 'url'
            ]
            display_columns = [col for col in columns if col in failed_df.columns]
            
            # Display the DataFrame
            st.dataframe(failed_df[display_columns])
    
    # Display task type distribution
    if task_stats['tasks_by_type']:
        # Create dataframe for pie chart
        task_type_df = pd.DataFrame({
            'type': list(task_stats['tasks_by_type'].keys()),
            'count': list(task_stats['tasks_by_type'].values())
        })
        
        # Only show types with non-zero values
        task_type_df = task_type_df[task_type_df['count'] > 0]
        
        if not task_type_df.empty:
            fig = px.pie(
                task_type_df, values='count', names='type',
                title="Tasks by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Display historical task metrics
    if len(st.session_state.historical_metrics['timestamps']) > 1:
        # Convert to pandas DataFrame for plotting
        df = pd.DataFrame({
            'timestamp': st.session_state.historical_metrics['timestamps'],
            'active_tasks': st.session_state.historical_metrics['active_tasks'],
            'failed_tasks': st.session_state.historical_metrics['failed_tasks']
        })
        
        # Display active tasks chart
        fig1 = px.line(
            df, x='timestamp', y='active_tasks',
            title='Active Tasks Over Time',
            labels={'active_tasks': 'Active Tasks', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Display failed tasks chart
        fig2 = px.line(
            df, x='timestamp', y='failed_tasks',
            title='Failed Tasks Over Time',
            labels={'failed_tasks': 'Failed Tasks', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    monitoring_dashboard() 