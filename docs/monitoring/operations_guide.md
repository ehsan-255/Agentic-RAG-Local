# Operations Guide: Monitoring Component

This guide provides practical instructions for configuring, operating, and maintaining the monitoring component of the Agentic RAG system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Configuration](#configuration)
3. [Core Operations](#core-operations)
4. [Dashboard Setup](#dashboard-setup)
5. [Alerting System](#alerting-system)
6. [Troubleshooting](#troubleshooting)
7. [FAQs](#faqs)

## System Overview

The monitoring component provides comprehensive observability for the entire Agentic RAG application with the following primary responsibilities:

- **Application Logging**: Structured, searchable logs of system activities
- **Performance Metrics**: Collection and analysis of key performance indicators
- **System Health Monitoring**: Proactive monitoring of component health
- **Dashboard Visualization**: Real-time and historical data visualization
- **Alert Management**: Notification of critical events requiring attention

The monitoring system uses a modular architecture that can be configured for different deployment environments from development to production.

## Configuration

### Environment Variables

The monitoring system can be configured through environment variables:

```bash
# Logging Configuration
LOG_LEVEL=INFO                    # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT=JSON                   # Log format (TEXT, JSON)
LOG_OUTPUT=FILE                   # Output destination (CONSOLE, FILE, BOTH)
LOG_FILE_PATH=./logs/app.log      # Path for log files
LOG_ROTATION=DAILY                # Log rotation period (HOURLY, DAILY, WEEKLY)
LOG_RETENTION_DAYS=30             # Number of days to keep logs

# Metrics Configuration
METRICS_COLLECTION_ENABLED=true   # Enable metrics collection
METRICS_STORE=MEMORY              # Metrics storage (MEMORY, POSTGRES, PROMETHEUS)
METRICS_RETENTION_HOURS=72        # Hours to keep metrics in memory store
METRICS_SAMPLE_RATE=1.0           # Sampling rate for high-volume metrics (0.0-1.0)

# Dashboard Configuration
DASHBOARD_AUTO_REFRESH_SECONDS=30 # Auto-refresh interval for dashboards
DASHBOARD_THEME=LIGHT             # Dashboard theme (LIGHT, DARK)
DASHBOARD_DEFAULT_TIMESPAN=24h    # Default timespan for dashboard charts

# Alerting Configuration
ALERTS_ENABLED=true               # Enable alerting system
ALERT_CHANNELS=EMAIL,SLACK        # Alert channels to use
EMAIL_RECIPIENTS=admin@example.com,devops@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
ALERT_COOLDOWN_MINUTES=15         # Minimum time between duplicate alerts
```

### Configuration File

For more advanced configuration, you can use a YAML configuration file:

```yaml
# monitoring_config.yaml
logging:
  level: INFO
  format: JSON
  output: FILE
  file_path: ./logs/app.log
  rotation: DAILY
  retention_days: 30
  
metrics:
  enabled: true
  store: MEMORY
  retention_hours: 72
  sample_rate: 1.0
  
dashboard:
  auto_refresh_seconds: 30
  theme: LIGHT
  default_timespan: 24h
  
alerting:
  enabled: true
  channels:
    - type: EMAIL
      recipients:
        - admin@example.com
        - devops@example.com
      min_level: ERROR
    - type: SLACK
      webhook_url: https://hooks.slack.com/services/XXX/YYY/ZZZ
      min_level: WARNING
  cooldown_minutes: 15
```

Load the configuration file by setting the `MONITORING_CONFIG_PATH` environment variable:

```bash
export MONITORING_CONFIG_PATH=./monitoring_config.yaml
```

## Core Operations

### Managing the Logging System

#### Viewing Logs

The system logs can be viewed in several ways:

1. **Console Output**: When configured to output to console
2. **Log Files**: When configured to write to files
3. **Log Viewer UI**: Through the Streamlit admin interface

To view logs via the command line:

```bash
# View the latest logs
tail -f ./logs/app.log

# Filter logs by level
grep '"level":"ERROR"' ./logs/app.log

# Filter logs by component
grep '"component":"database"' ./logs/app.log

# Search for specific events
grep '"message":"User query processed"' ./logs/app.log
```

#### Log Rotation and Cleanup

Log rotation is handled automatically based on configuration. To manually rotate logs:

```bash
# Rotate logs
python -m src.monitoring.commands rotate_logs

# Clean up old logs
python -m src.monitoring.commands cleanup_logs --days 30
```

### Managing Metrics Collection

#### Viewing Current Metrics

View current metrics through the Streamlit admin interface or via the command line:

```bash
# Show all current metrics
python -m src.monitoring.commands show_metrics

# Show specific metrics
python -m src.monitoring.commands show_metrics --metric http_requests_total

# Show metrics for a specific component
python -m src.monitoring.commands show_metrics --component database
```

#### Resetting Metrics

To reset metrics:

```bash
# Reset all metrics
python -m src.monitoring.commands reset_metrics

# Reset specific metrics
python -m src.monitoring.commands reset_metrics --metric rag_processing_time_ms
```

### Running System Diagnostics

To check the health of the system:

```bash
# Run full system diagnostics
python -m src.monitoring.commands run_diagnostics

# Check specific component health
python -m src.monitoring.commands check_health --component database

# Run performance test
python -m src.monitoring.commands run_performance_test --test_type vector_search
```

### Programmatic Access to Monitoring

You can also access monitoring functionality programmatically:

```python
from src.monitoring.logger import get_logger
from src.monitoring.metrics import MetricsCollector
from src.monitoring.diagnostics import check_system_health

# Get a logger instance
logger = get_logger("my_component")
logger.info("This is an informational message", extra_data="Some value")

# Create a metrics collector
metrics = MetricsCollector()
metrics.increment_counter("operation_count")
processing_time = metrics.stop_timer("operation_timer")

# Check system health
health_status = await check_system_health()
```

## Dashboard Setup

### Available Dashboards

The system provides several pre-built dashboards:

1. **System Overview**: High-level metrics for all components
2. **API Performance**: API request rates, latencies, and error rates
3. **Database Performance**: Query performance, connection pool stats
4. **RAG Performance**: Query processing times, context retrieval metrics
5. **Crawler Performance**: Crawling rates, processing times, error rates
6. **Error Analysis**: Detailed breakdown of system errors

### Generating Dashboards

Generate dashboards through the Streamlit admin interface or via the command line:

```bash
# Generate all dashboards
python -m src.monitoring.commands generate_dashboards --output_dir ./dashboards

# Generate specific dashboard
python -m src.monitoring.commands generate_dashboard --type rag_performance --output ./dashboards/rag.html

# Generate dashboard with custom timespan
python -m src.monitoring.commands generate_dashboard --type system_overview --timespan 48h --output ./dashboards/system.html
```

### Customizing Dashboards

You can customize dashboards by creating configuration files:

```yaml
# dashboard_config.yaml
title: "Custom RAG Performance Dashboard"
refresh_rate: 60
layout:
  - name: "Query Processing Time"
    type: "time_series"
    metric: "rag_processing_time_ms"
    aggregation: "avg"
    height: 300
    width: 12
  - name: "Context Count Distribution"
    type: "bar"
    metric: "rag_context_count"
    group_by: "query_type"
    height: 300
    width: 6
  - name: "Success Rate"
    type: "gauge"
    metric: "rag_processing.success_rate"
    height: 300
    width: 6
```

Apply the custom configuration:

```bash
python -m src.monitoring.commands generate_dashboard --config ./dashboard_config.yaml --output ./dashboards/custom.html
```

## Alerting System

### Configuring Alert Channels

#### Email Alerts

Configure email alerts by setting up the SMTP configuration in `monitoring_config.yaml`:

```yaml
alerting:
  channels:
    - type: EMAIL
      recipients:
        - admin@example.com
        - devops@example.com
      min_level: ERROR
      smtp:
        server: smtp.gmail.com
        port: 587
        username: alerts@example.com
        password: your_app_password
        use_tls: true
```

#### Slack Alerts

Configure Slack alerts by creating a Slack app with webhook URL:

1. Go to https://api.slack.com/apps
2. Create a new app and enable "Incoming Webhooks"
3. Add the webhook URL to your configuration:

```yaml
alerting:
  channels:
    - type: SLACK
      webhook_url: https://hooks.slack.com/services/XXX/YYY/ZZZ
      min_level: WARNING
      channel: "#monitoring-alerts"
```

#### PagerDuty Integration

Configure PagerDuty integration:

```yaml
alerting:
  channels:
    - type: PAGERDUTY
      service_key: your_pagerduty_service_key
      min_level: CRITICAL
```

### Managing Alert Rules

Alert rules define when alerts should be triggered. Configure them in `monitoring_config.yaml`:

```yaml
alerting:
  rules:
    - name: "High Error Rate"
      condition: "error_rate > 0.05"
      duration: "5m"
      level: WARNING
      message: "Error rate exceeded 5% threshold"
      
    - name: "API Slowdown"
      condition: "api_response_time_p95 > 1000"
      duration: "10m"
      level: WARNING
      message: "API response time is abnormally high"
      
    - name: "Database Connection Issues"
      condition: "db_connection_failures > 5"
      duration: "5m"
      level: ERROR
      message: "Multiple database connection failures detected"
      
    - name: "OpenAI API Rate Limit"
      condition: "openai_rate_limit_errors > 0"
      duration: "1m"
      level: WARNING
      message: "OpenAI API rate limit reached"
```

### Testing Alerts

Test your alert configuration:

```bash
# Test all alert channels
python -m src.monitoring.commands test_alerts

# Test specific channel
python -m src.monitoring.commands test_alert --channel slack

# Send test alert
python -m src.monitoring.commands send_alert --title "Test Alert" --message "This is a test alert" --level WARNING
```

### Alert Silencing and Maintenance Mode

During maintenance, you may want to silence alerts:

```bash
# Enter maintenance mode (silence all alerts)
python -m src.monitoring.commands maintenance_mode --enable --duration 60

# Silence specific alert rule
python -m src.monitoring.commands silence_alert --rule "High Error Rate" --duration 120

# Exit maintenance mode
python -m src.monitoring.commands maintenance_mode --disable
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Possible Causes | Solutions |
|-------|----------|----------------|-----------|
| **Log File Permissions** | Logger fails to write to log file | Incorrect file permissions | `chmod 755 ./logs` and `chmod 644 ./logs/*.log` |
| **Metrics Not Updating** | Dashboard shows stale data | Metrics collection disabled or not functioning | Check `METRICS_COLLECTION_ENABLED` setting, restart the application |
| **Missing Alerts** | Alerts not being received | Incorrect alert channel configuration | Test alert channels with `test_alerts` command |
| **High CPU Usage** | System performance degradation | Excessive logging or metrics collection | Increase sampling rate, reduce logging verbosity |
| **Dashboard Generation Failure** | Dashboard not generated | Missing metrics data | Ensure metrics collection is enabled and functioning |

### Diagnostic Tools

#### Log Inspection

Check logs for monitoring system issues:

```bash
# Check monitoring-specific logs
grep '"component":"monitoring"' ./logs/app.log
```

#### Checking Monitoring System Health

The monitoring system can monitor itself:

```bash
# Check monitoring system health
python -m src.monitoring.commands check_monitoring_health
```

#### Memory Usage Analysis

If the monitoring system is consuming too much memory:

```bash
# Analyze monitoring system memory usage
python -m src.monitoring.commands analyze_memory_usage
```

### Recovering from Failures

If the monitoring system fails:

1. Check the logs for error messages
2. Verify configuration values
3. Restart the monitoring services:

```bash
python -m src.monitoring.commands restart_monitoring
```

If data is corrupted:

```bash
# Reset metrics store
python -m src.monitoring.commands reset_metrics_store

# Restore from backup (if available)
python -m src.monitoring.commands restore_backup --file ./backups/metrics_2023-04-15.bak
```

## FAQs

### General Questions

**Q: How much disk space do logs consume?**  
A: With default settings (INFO level, daily rotation, 30-day retention), logs consume approximately 100-200MB per day for moderate usage. This can be adjusted using the `LOG_LEVEL` and `LOG_RETENTION_DAYS` settings.

**Q: Does the monitoring system affect application performance?**  
A: The monitoring system is designed to have minimal impact on performance. With default settings, the overhead is typically less than 5%. For high-throughput scenarios, you can adjust the sampling rate using the `METRICS_SAMPLE_RATE` setting.

**Q: Can I integrate with external monitoring tools?**  
A: Yes, the monitoring system supports exporting metrics to Prometheus and logs to standard formats that can be ingested by tools like ELK (Elasticsearch, Logstash, Kibana) or Splunk.

### Technical Questions

**Q: How can I add monitoring to a new component?**  
A: Import the logger and metrics collector in your component:

```python
from src.monitoring.logger import get_logger
from src.monitoring.metrics import MetricsCollector

# Initialize logger and metrics
logger = get_logger("my_new_component")
metrics = MetricsCollector()

# Use them in your code
logger.info("Component initialized")
metrics.increment_counter("component_operations")
```

**Q: How do I create a custom dashboard for my component?**  
A: Define a custom dashboard configuration YAML file and use the `generate_dashboard` command with your configuration.

**Q: How can I set up alerts for a new metric?**  
A: Add a new rule to the `alerting.rules` section in your `monitoring_config.yaml` file that references your metric.

**Q: What's the best way to handle high-volume logs?**  
A: For high-volume logs, consider:
1. Increasing the `LOG_LEVEL` to reduce verbosity
2. Using `METRICS_SAMPLE_RATE` to sample metrics
3. Configuring external log shipping to a dedicated logging system
4. Implementing log aggregation to combine similar log entries

**Q: How can I monitor the resource usage of the application?**  
A: The monitoring system includes resource usage metrics. View them in the System Overview dashboard or programmatically:

```python
from src.monitoring.diagnostics import get_resource_usage

# Get current resource usage
resources = get_resource_usage()
print(f"CPU: {resources['cpu_percent']}%, Memory: {resources['memory_percent']}%")
``` 