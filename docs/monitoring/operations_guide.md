# Operations Guide: Monitoring System

This guide provides operational guidance for system administrators and operators who need to interpret monitoring data, troubleshoot issues, and maintain the system.

## Table of Contents

1. [Dashboard Overview](#dashboard-overview)
2. [Interpreting Metrics](#interpreting-metrics)
3. [Troubleshooting Common Issues](#troubleshooting-common-issues)
4. [Maintenance Tasks](#maintenance-tasks)
5. [Performance Optimization](#performance-optimization)
6. [Reference Tables](#reference-tables)

## Dashboard Overview

The monitoring dashboard is accessible through the Streamlit UI and consists of several tabs:

### Crawl Status Tab

Displays the current state and progress of active crawl operations, including:
- Source name and ID
- Start time and duration
- Pages processed, succeeded, and failed
- Success rate
- Error statistics by category and type
- Historical success rate trends

Use this tab to:
- Monitor ongoing crawls in real-time
- Identify problematic content sources
- Track overall operation progress
- Analyze error patterns

### Tasks Tab

Shows active and failed tasks in the system:
- Total task counts (pending, running, succeeded, failed)
- Average task duration
- Detailed task listings with state information
- Task distribution by type

Use this tab to:
- Identify stuck or long-running tasks
- Diagnose task failure reasons
- Monitor task throughput and concurrency
- Track resource utilization patterns

### System Tab

Displays system resource metrics:
- CPU usage
- Memory consumption
- Thread count
- System memory utilization
- Historical resource usage trends

Use this tab to:
- Identify resource bottlenecks
- Monitor for memory leaks
- Track performance degradation over time
- Plan capacity requirements

### Database Tab

Shows database connection and query statistics:
- Connection pool utilization
- Transaction counts and states
- Query performance metrics
- Error rates

Use this tab to:
- Identify database bottlenecks
- Monitor connection pool health
- Track slow queries
- Diagnose database-related errors

### API Tab

Displays API usage and rate limit information:
- Call counts by endpoint
- Success and failure rates
- Rate limit utilization
- Response time metrics

Use this tab to:
- Monitor API usage patterns
- Track rate limit consumption
- Identify slow or failing API endpoints
- Optimize API usage

### Resume Tab

Provides options for resuming interrupted crawls:
- Source selection dropdown
- Statistics on already processed content
- Configuration management options

Use this tab to:
- Resume interrupted crawls
- Save and manage crawl configurations
- Review source statistics

## Interpreting Metrics

### Success Rate

The success rate metric indicates the percentage of pages that were successfully processed:

| Success Rate | Interpretation | Action |
|--------------|----------------|--------|
| 95-100% | Excellent | Normal operation |
| 85-95% | Good | Review error logs |
| 70-85% | Concerning | Investigate common failure patterns |
| <70% | Critical | Immediate investigation needed |

A sudden drop in success rate typically indicates:
- Change in site structure
- API rate limiting
- System resource constraints
- Network connectivity issues

### Error Categories Distribution

The error categories pie chart shows the distribution of errors by type:

| Category Dominance | Likely Issue |
|--------------------|--------------|
| Content Processing | Site structure changes or parsing issues |
| Connection | Network issues or site availability |
| API Rate Limit | Quota exceeded or throttling |
| Embedding | Token limits or model issues |
| Database | DB connectivity or query problems |
| Task Scheduling | Concurrency or resource limits |

A healthy system typically shows a mixture of error types with no single category dominating unless there is a specific issue affecting all operations.

### Resource Utilization

Monitor these resource metrics to ensure system health:

| Metric | Warning Threshold | Critical Threshold | Possible Causes |
|--------|-------------------|-------------------|----------------|
| CPU Usage | >70% sustained | >90% sustained | Too many concurrent tasks, inefficient processing |
| Memory Usage | Steady increase | >80% of system memory | Memory leaks, large document processing |
| Active Tasks | >80% of max config | At max configuration | Queue backup, slow processing |
| Thread Count | Steady increase | >200 threads | Thread leaks, too many concurrent tasks |

### API Rate Limits

The rate limit utilization provides insight into API usage:

| Utilization | Interpretation | Action |
|-------------|----------------|--------|
| <50% | Healthy | Normal operation |
| 50-80% | Moderate | Consider rate adjustments |
| >80% | High | Implement rate limiting or backoff |
| 100% | Limit reached | Pause operations, increase backoff |

### Database Connection Pool

The connection pool metrics help identify database bottlenecks:

| Metric | Warning Signs | Action |
|--------|--------------|--------|
| Active/Max Ratio | >80% | Increase pool size or optimize queries |
| Wait Count | Any non-zero value | Increase pool size or reduce query volume |
| Idle Connections | 0 for extended periods | Check for connection leaks |

## Troubleshooting Common Issues

### High Failure Rate

If you observe a high failure rate:

1. **Check Error Categories**:
   - If dominated by Content Processing: Review sample pages for structure changes
   - If dominated by Connection: Check network and target site availability
   - If dominated by API Rate Limit: Adjust concurrency settings

2. **Review Failed URLs**:
   - Look for patterns in failed URLs (specific sections, formats, sizes)
   - Try manually accessing a sample of failed URLs to verify availability

3. **Check Resource Utilization**:
   - Verify memory and CPU aren't constraining the system
   - Ensure database connection pool isn't exhausted

### System Slowdown

If the system becomes progressively slower:

1. **Check Memory Usage Trend**:
   - Steadily increasing memory usage indicates a potential memory leak
   - Look for objects not being properly garbage collected

2. **Monitor Task Duration**:
   - Increasing average task duration indicates processing inefficiency
   - Check for external dependencies slowing down

3. **Database Performance**:
   - Rising query times might indicate index problems or database load
   - Check for slow queries and optimize as needed

### "Cannot Schedule New Futures" Errors

This indicates issues with the task scheduling system:

1. **Check Active Tasks**:
   - If at maximum capacity, increase max_concurrent_tasks setting
   - Look for stuck tasks that aren't completing

2. **Verify Shutdown Flags**:
   - Ensure no premature shutdown requests are active
   - Check for error propagation from child tasks to parent tasks

3. **Resource Constraints**:
   - Verify system has sufficient resources to schedule new tasks
   - Check for thread pool exhaustion

### API Rate Limiting

When encountering API rate limit errors:

1. **Adjust Concurrency**:
   - Reduce max_concurrent_api_calls setting
   - Implement progressive backoff strategy

2. **Monitor Usage Patterns**:
   - Check for usage spikes or unexpected increases
   - Verify rate limits align with account tier

3. **Optimize Usage**:
   - Consider batching smaller requests
   - Implement caching for repetitive calls

## Maintenance Tasks

### Regular Maintenance

Perform these tasks regularly to maintain system health:

1. **Log Rotation**:
   - Implement log rotation to prevent disk space issues
   - Archive old logs for historical analysis

2. **Database Maintenance**:
   - Run VACUUM and ANALYZE on PostgreSQL tables
   - Verify index health and rebuild if necessary

3. **Configuration Backup**:
   - Backup crawl configurations regularly
   - Document custom settings and their rationale

### Health Checks

Implement these health checks to ensure system stability:

1. **Daily Checks**:
   - Review error rate trends for abnormalities
   - Verify resource utilization is within expected ranges
   - Check for failed tasks that haven't been addressed

2. **Weekly Checks**:
   - Analyze error patterns for systematic issues
   - Review performance metrics for degradation
   - Verify database size and growth rate

3. **Monthly Checks**:
   - Perform database optimization tasks
   - Clean up old monitoring data
   - Review and adjust concurrency settings based on performance

## Performance Optimization

### Optimizing Crawl Performance

To improve crawl performance:

1. **Concurrent Requests**:
   - Adjust max_concurrent_requests based on target site capacity
   - Start low (3-5) and increase gradually while monitoring performance

2. **Batch Sizes**:
   - Optimize embedding batch size for best throughput (typically 5-10)
   - Adjust LLM batch size based on observed performance (typically 2-5)

3. **URL Filtering**:
   - Use more specific URL patterns to focus on relevant content
   - Exclude sections known to contain low-value content

### Resource Allocation

Optimize resource allocation for better performance:

1. **Memory**:
   - Allocate minimum 4GB RAM for standard operations
   - For large sites, consider 8-16GB RAM

2. **CPU**:
   - Multi-core systems perform better with higher concurrency
   - Ensure at least 2 cores for standard operations

3. **Network**:
   - Stable internet connection with at least 10Mbps bandwidth
   - Low-latency connections improve API performance

### API Usage Optimization

Optimize API calls for cost and performance:

1. **Embedding Models**:
   - Use ada-002 for standard operations (good balance of cost/performance)
   - Consider text-embedding-3-small for more efficiency

2. **LLM Models**:
   - Use smaller models (gpt-3.5-turbo) for routine summaries
   - Reserve larger models for complex content analysis

3. **Batching Strategy**:
   - Batch embeddings to maximize throughput
   - Consider text chunking strategy to optimize token usage

## Reference Tables

### Error Category Reference

| Error Category | Typical Causes | Resolution Strategies |
|----------------|----------------|----------------------|
| CONTENT_PROCESSING | HTML parsing failures, empty content, chunking issues | Review target site structure, adjust parsing logic, check content length thresholds |
| CONNECTION | Network timeouts, DNS failures, site unavailability | Check network connectivity, verify site availability, implement retry logic |
| API_RATE_LIMIT | API quota exceeded, throttling | Reduce concurrency, implement backoff, increase quota |
| EMBEDDING | Token limits exceeded, model errors | Optimize chunking, reduce input size, check model status |
| DATABASE | Query failures, connection issues | Verify database connectivity, optimize queries, check indexes |
| TASK_SCHEDULING | Thread pool exhaustion, future cancellation | Adjust concurrency settings, implement recovery logic |

### Key Metrics Reference

| Metric | Normal Range | Description | Location |
|--------|--------------|-------------|----------|
| Success Rate | >90% | Percentage of successfully processed pages | Crawl Status Tab |
| Memory Usage | Stable, <70% | Process memory consumption | System Tab |
| API Rate Limit Utilization | <80% | Percentage of API rate limit consumed | API Tab |
| Active Tasks | Varies by config | Number of currently executing tasks | Tasks Tab |
| DB Connection Utilization | <80% | Percentage of connection pool in use | Database Tab |

### Configuration Reference

| Setting | Typical Value | Description | Impact |
|---------|---------------|-------------|--------|
| max_concurrent_requests | 3-10 | Maximum parallel web requests | Higher values increase crawl speed but may overwhelm target sites |
| max_concurrent_api_calls | 2-5 | Maximum parallel API calls | Higher values increase processing speed but consume API limits faster |
| chunk_size | 1000-5000 | Content chunk size in characters | Larger chunks reduce total chunks but may exceed model limits |
| retry_attempts | 3-6 | Number of retry attempts | Higher values improve reliability but extend processing time |
| min_backoff | 1-5 | Minimum backoff time in seconds | Higher values reduce pressure on rate-limited resources |
| max_backoff | 30-120 | Maximum backoff time in seconds | Higher values provide more breathing room during heavy rate limiting | 