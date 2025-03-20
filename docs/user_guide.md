# User Guide: Monitoring Dashboard

This guide provides instructions for using the monitoring dashboard and its controls.

## Table of Contents

1. [Accessing the Dashboard](#accessing-the-dashboard)
2. [Dashboard Tabs](#dashboard-tabs)
3. [Controlling Crawls](#controlling-crawls)
4. [Monitoring Live Crawls](#monitoring-live-crawls)
5. [Resuming Interrupted Crawls](#resuming-interrupted-crawls)
6. [Best Practices](#best-practices)
7. [Database Diagnostic Tools](#database-diagnostic-tools)

## Accessing the Dashboard

The monitoring dashboard is integrated into the Streamlit app and can be accessed as follows:

1. Launch the Streamlit app by running:
   ```bash
   streamlit run src/ui/streamlit_app.py
   ```

2. When the application loads, click on the "Monitoring" tab at the top of the interface.

3. The monitoring dashboard will load with multiple tabs for different monitoring aspects.

## Dashboard Tabs

### Crawl Status Tab

The Crawl Status tab provides an overview of active crawl operations:

![Crawl Status Tab](./images/crawl_status_tab.png)

Key elements:
- **Source Information**: Shows the source being crawled with ID and start time
- **Progress Metrics**: Displays pages processed, succeeded, and failed
- **Success Rate**: Shows the percentage of successfully processed pages
- **Error Statistics**: Displays errors by category and type
- **Historical Trends**: Shows success rate over time
- **URL Processing Status**: Displays processed and failed URLs
- **Crawl Controls**: Buttons to pause, resume, and stop crawls

### Tasks Tab

The Tasks tab provides visibility into the system's task execution:

![Tasks Tab](./images/tasks_tab.png)

Key elements:
- **Task Counts**: Shows total, active, succeeded, and failed tasks
- **Average Duration**: Displays the average task execution time
- **Active Tasks**: Lists currently running tasks with details
- **Failed Tasks**: Shows tasks that have failed with error information
- **Task Distribution**: Displays task distribution by type
- **Historical Metrics**: Shows task activity over time

### System Tab

The System tab monitors system resource utilization:

![System Tab](./images/system_tab.png)

Key elements:
- **CPU Usage**: Displays current CPU utilization
- **Memory Usage**: Shows memory consumption in MB
- **System Memory**: Shows overall system memory usage
- **Thread Count**: Displays the number of active threads
- **Historical Trends**: Shows CPU and memory usage over time

### Database Tab

The Database tab provides insights into database operations:

![Database Tab](./images/database_tab.png)

Key elements:
- **Connection Pool**: Shows connection pool utilization
- **Transactions**: Displays transaction statistics
- **Queries**: Shows query performance metrics
- **Average Query Duration**: Displays average query execution time

### API Tab

The API tab monitors external API usage:

![API Tab](./images/api_tab.png)

Key elements:
- **API Call Metrics**: Shows total, succeeded, and failed calls
- **Rate Limited Calls**: Displays the number of rate-limited calls
- **Average Call Duration**: Shows average API call duration
- **Calls by Endpoint**: Displays API usage by endpoint
- **Rate Limits**: Shows rate limit utilization by endpoint

### Resume Tab

The Resume tab allows you to resume interrupted crawls:

![Resume Tab](./images/resume_tab.png)

Key elements:
- **Source Selection**: Dropdown to select a source to resume
- **Source Statistics**: Shows statistics for the selected source
- **Configuration Controls**: Options to prepare and save resume configurations

## Controlling Crawls

The dashboard provides controls to manage crawl operations:

### Pausing a Crawl

To pause an active crawl:

1. Navigate to the **Crawl Status** tab
2. Locate the **Crawl Controls** section
3. Click the **Pause Crawl** button

When a crawl is paused:
- In-progress tasks will complete their current work
- New tasks will not be started until resumed
- System resources will be gradually freed up

### Resuming a Paused Crawl

To resume a paused crawl:

1. Navigate to the **Crawl Status** tab
2. Locate the **Crawl Controls** section
3. Click the **Resume Crawl** button

When a crawl is resumed:
- Processing will continue from where it left off
- New tasks will be scheduled for remaining URLs
- System resource utilization will increase again

### Stopping a Crawl

To completely stop a crawl:

1. Navigate to the **Crawl Status** tab
2. Locate the **Crawl Controls** section
3. Click the **Stop Crawl** button

When a crawl is stopped:
- All pending tasks will be cancelled
- In-progress tasks will be allowed to complete
- The crawl session will be ended and statistics finalized

**Note**: A stopped crawl cannot be directly resumed. You must start a new crawl or use the resume functionality on the Resume tab.

### Auto-refreshing the Dashboard

To enable automatic refreshing of dashboard metrics:

1. Scroll to the bottom of any dashboard tab
2. Check the **Auto-refresh (5s)** checkbox

When auto-refresh is enabled:
- The dashboard will automatically update every 5 seconds
- This ensures you're always seeing current data
- Disable auto-refresh when not actively monitoring to reduce resource usage

## Monitoring Live Crawls

While a crawl is in progress, keep an eye on these key indicators:

### Success Rate

The success rate shows the percentage of pages that were successfully processed:

- **>90%**: Normal operation
- **70-90%**: Some issues, but generally acceptable
- **<70%**: Significant problems requiring attention

A declining success rate may indicate problems with the target site, API rate limits, or system resources.

### Pages Processed

The pages processed metric shows crawl progress:

- Monitor the rate of increase to gauge speed
- Compare to the total expected pages to estimate completion time
- A slowing rate may indicate performance degradation

### Error Distribution

The error categories pie chart shows what's failing:

- A balanced distribution is normal
- One category dominating indicates a specific problem area
- Click on category sections to view detailed error information

### Resource Utilization

Monitor system resources to prevent bottlenecks:

- CPU usage should generally stay below 80%
- Memory usage should be stable (not continuously increasing)
- Thread count should stabilize at a level proportional to concurrency settings

### Failed URLs

Review the failed URLs list periodically:

- Look for patterns in failures (specific directories, file types)
- Spot-check a few URLs in a browser to verify if the source is the issue
- Consider adding problematic patterns to URL exclusion settings

## Resuming Interrupted Crawls

If a crawl is interrupted (by system shutdown, error, or manual stop), you can resume it:

### Preparing to Resume

1. Navigate to the **Resume** tab
2. Select the source to resume from the dropdown
3. Review the source statistics (pages and chunks already processed)
4. Click **Prepare Resume Configuration**
5. The system will analyze the source and determine what URLs have been processed

### Saving Resume Configuration (Optional)

To save the resume configuration for future use:

1. After preparing the resume configuration, enter a name for the configuration
2. Click **Save Configuration**
3. The configuration will be saved to the `crawl_configs` directory

### Starting the Resume Crawl

To start the resume crawl:

1. Return to the main Streamlit interface
2. Select the same source from the source dropdown
3. Click **Start Crawl**
4. The crawler will automatically skip previously processed URLs

## Best Practices

- Keep the number of concurrent crawls limited to avoid resource exhaustion
- Use the monitoring dashboard to check status rather than starting/stopping the app
- Start with smaller crawls before large operations
- Use auto-refresh only when actively monitoring
- Check multiple tabs to get a complete system picture
- Filter and focus on the most relevant metrics for your current concern

## Database Diagnostic Tools

The system includes standalone database diagnostic tools to help troubleshoot issues with content storage and retrieval:

### Running the Database Check Tool

The database diagnostic tool can be run from the command line:

```bash
python check_database.py
```

This tool provides:

1. **Documentation Source Reports**: Lists all sources with their IDs and status
2. **Content Storage Analysis**: Reports on pages and chunks stored per source
3. **Database Health Check**: Verifies database connections and schema
4. **Missing Content Detection**: Identifies sources with no stored content
5. **Connection Pool Statistics**: Displays connection pool utilization metrics

### Interpreting Results

- **Source with Zero Pages**: Indicates crawl failures or database storage issues
- **Source with Pages but No Chunks**: Indicates chunking failures or embedding issues
- **Connection Pool Warnings**: May indicate connection exhaustion problems
- **Schema Validation Errors**: May indicate database migration issues

### Using Schema Analysis

For more detailed database schema inspection, use:

```bash
python check_schema.py
```

This utility will:
- List all database tables
- Show column names and data types
- Report on foreign key relationships
- Identify potential schema issues 