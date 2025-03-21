# Operations Guide: API Component

This guide provides instructions and best practices for operating, configuring, and maintaining the API component of the Agentic RAG system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Configuration](#configuration)
3. [Deployment](#deployment)
4. [Security](#security)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Backup and Recovery](#backup-and-recovery)
8. [Frequently Asked Questions](#frequently-asked-questions)

## System Overview

The API component serves as the interface between clients and the Agentic RAG system. It provides a set of RESTful endpoints that enable:

- Processing natural language queries with context retrieval
- Managing documentation sources
- Monitoring system performance
- Accessing system health information

The API is built with FastAPI and follows these design principles:

- **RESTful**: Resource-oriented design with standardized HTTP methods
- **Asynchronous**: Non-blocking operations for improved performance
- **Scalable**: Support for horizontal scaling behind a load balancer
- **Secure**: Authentication, rate limiting, and input validation
- **Observable**: Comprehensive logging and monitoring

## Configuration

### Environment Variables

The API component is configured through environment variables:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `API_HOST` | Host address to bind the API server | `0.0.0.0` | No |
| `API_PORT` | Port to run the API server on | `8000` | No |
| `API_WORKERS` | Number of Uvicorn workers | `4` | No |
| `API_LOG_LEVEL` | Logging level (debug, info, warning, error) | `info` | No |
| `API_CORS_ORIGINS` | Comma-separated list of allowed origins | `*` | No |
| `API_KEY_REQUIRED` | Whether API key authentication is required | `false` | No |
| `API_KEY` | API key for authentication if enabled | N/A | Only if `API_KEY_REQUIRED` is true |
| `API_RATE_LIMIT_ENABLED` | Whether to enable rate limiting | `true` | No |
| `API_RATE_LIMIT_MAX_REQUESTS` | Maximum requests per minute per client | `100` | No |
| `JWT_SECRET_KEY` | Secret key for JWT authentication | N/A | Only if using JWT auth |
| `JWT_ALGORITHM` | Algorithm for JWT token generation | `HS256` | No |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | JWT token expiration in minutes | `30` | No |

### Configuration File

For more complex configurations, you can use a configuration file:

```bash
# Create a config directory if it doesn't exist
mkdir -p config

# Create a configuration file
cat > config/api_config.json << EOF
{
  "cors": {
    "origins": ["https://app.example.com", "https://admin.example.com"],
    "allow_credentials": true,
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["*"]
  },
  "rate_limiting": {
    "enabled": true,
    "max_requests": 100,
    "window_seconds": 60,
    "exclude_paths": ["/health", "/api/metrics"]
  },
  "logging": {
    "level": "info",
    "format": "json",
    "output_file": "logs/api.log"
  }
}
EOF
```

Pass the configuration file path as an environment variable:

```bash
export API_CONFIG_FILE=config/api_config.json
```

## Deployment

### Running with Uvicorn

For development or testing:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Running with Gunicorn (Production)

For production environments:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn src.api.app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t agentic-rag-api -f docker/api.Dockerfile .

# Run the Docker container
docker run -d --name api \
  -p 8000:8000 \
  -e API_KEY=your_api_key \
  -e API_CORS_ORIGINS=https://yourdomain.com \
  agentic-rag-api
```

### Docker Compose

```bash
# Start all services with Docker Compose
docker-compose up -d
```

### Kubernetes Deployment

Apply the Kubernetes manifests:

```bash
kubectl apply -f kubernetes/api-deployment.yaml
kubectl apply -f kubernetes/api-service.yaml
kubectl apply -f kubernetes/api-ingress.yaml
```

## Security

### API Key Authentication

To enable API key authentication:

1. Set the required environment variables:
   ```bash
   export API_KEY_REQUIRED=true
   export API_KEY=your_strong_api_key
   ```

2. Use the API key in client requests:
   ```bash
   curl -X POST "http://localhost:8000/api/query" \
     -H "X-API-Key: your_strong_api_key" \
     -H "Content-Type: application/json" \
     -d '{"query": "How does vector search work?"}'
   ```

### JWT Authentication

To use JWT authentication:

1. Set up the required environment variables:
   ```bash
   export JWT_SECRET_KEY=your_jwt_secret
   export JWT_ALGORITHM=HS256
   export JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

2. Request a token:
   ```bash
   curl -X POST "http://localhost:8000/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=user&password=password"
   ```

3. Use the token in subsequent requests:
   ```bash
   curl -X POST "http://localhost:8000/api/query" \
     -H "Authorization: Bearer your_jwt_token" \
     -H "Content-Type: application/json" \
     -d '{"query": "How does vector search work?"}'
   ```

### CORS Configuration

Configure Cross-Origin Resource Sharing (CORS) to allow client-side applications to access the API:

```bash
export API_CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

To allow all origins (not recommended for production):

```bash
export API_CORS_ORIGINS=*
```

### Rate Limiting

Configure rate limiting to prevent abuse:

```bash
export API_RATE_LIMIT_ENABLED=true
export API_RATE_LIMIT_MAX_REQUESTS=100
```

## Monitoring

### Logging

Configure logging levels:

```bash
export API_LOG_LEVEL=info
```

View logs:

```bash
# For Docker deployment
docker logs -f api

# For Kubernetes deployment
kubectl logs -f deployment/api
```

### Health Checks

The API provides a health check endpoint to verify the system status:

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "api": "healthy",
    "database": "healthy",
    "rag": "healthy",
    "openai": "healthy"
  },
  "timestamp": "2023-04-16T12:00:00Z"
}
```

Use this endpoint for load balancer health checks and monitoring systems.

### Metrics

Access system metrics:

```bash
curl http://localhost:8000/api/metrics
```

Example response:

```json
{
  "system": {
    "uptime_hours": 24.5,
    "cpu_usage": 35.2,
    "memory_usage": 42.8
  },
  "rag": {
    "queries_total": 1250,
    "queries_today": 127,
    "avg_response_time_ms": 450
  },
  "documents": {
    "sources_count": 5,
    "pages_count": 750,
    "chunks_count": 2250
  },
  "errors": {
    "total_errors": 12,
    "error_rate": 0.01
  }
}
```

### Prometheus Integration

The API exports Prometheus metrics at `/metrics` when Prometheus support is enabled:

1. Install the required dependencies:
   ```bash
   pip install prometheus-fastapi-instrumentator
   ```

2. Enable the middleware in the configuration:
   ```bash
   export API_ENABLE_PROMETHEUS=true
   ```

3. Access the metrics:
   ```bash
   curl http://localhost:8000/metrics
   ```

### Grafana Dashboard

A pre-configured Grafana dashboard is available for monitoring the API:

1. Import the dashboard JSON from `monitoring/grafana/api_dashboard.json` into your Grafana instance.

2. Configure the Prometheus data source to scrape metrics from the API endpoint.

## Troubleshooting

### Common Issues

| Issue | Symptoms | Possible Causes | Solutions |
|-------|----------|----------------|-----------|
| API Unavailable | HTTP 503 or connection refused | Server down or overloaded | Check server status, restart service, scale up if needed |
| Authentication Failure | HTTP 401 Unauthorized | Invalid or missing API key | Verify API key configuration, check client request headers |
| Rate Limiting | HTTP 429 Too Many Requests | Excessive requests from client | Implement backoff strategy, increase rate limits if needed |
| Slow Response Times | High latency, timeouts | Database issues, high load | Check database performance, optimize queries, scale infrastructure |
| Memory Leaks | Increasing memory usage over time | Resource management issues | Restart service, identify memory leaks, update code |

### Diagnostic Commands

Check the API process:

```bash
# Check if the API process is running
ps aux | grep uvicorn

# Check open ports
netstat -tulpn | grep 8000
```

Check the API logs:

```bash
# View the last 100 lines of the API log
tail -n 100 logs/api.log

# View errors in the log
grep ERROR logs/api.log
```

### Performance Debugging

Enable detailed logging for performance debugging:

```bash
export API_LOG_LEVEL=debug
export API_PERFORMANCE_LOGGING=true
```

Restart the API service to apply the changes.

## Backup and Recovery

### Configuration Backup

Backup configuration files:

```bash
mkdir -p backups/$(date +%Y-%m-%d)
cp config/api_config.json backups/$(date +%Y-%m-%d)/
```

### Database Connection Configuration

The API component requires access to the database. If the database connection details change, update the environment variables:

```bash
export DB_HOST=new_db_host
export DB_PORT=5432
export DB_USER=db_user
export DB_PASSWORD=db_password
export DB_NAME=ragdb
```

Restart the API service to apply the changes.

## Frequently Asked Questions

### General Questions

**Q: How many concurrent requests can the API handle?**

A: The default configuration can handle approximately 100-200 concurrent requests depending on the hardware. For higher loads, increase the number of workers and consider horizontal scaling behind a load balancer.

**Q: How do I secure the API for production use?**

A: For production, we recommend:
1. Enabling API key or JWT authentication
2. Using HTTPS with a valid SSL certificate
3. Configuring specific CORS origins
4. Implementing rate limiting
5. Running behind a reverse proxy like Nginx
6. Using a firewall to restrict access

**Q: Can I run multiple instances of the API?**

A: Yes, the API is designed to be stateless and can be horizontally scaled. Deploy multiple instances behind a load balancer for high availability and increased throughput.

### Technical Questions

**Q: How do I add custom middleware to the API?**

A: Create a custom middleware and add it to the application:

```python
# File: src/api/middleware/custom_middleware.py
from starlette.middleware.base import BaseHTTPMiddleware

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Pre-processing logic
        response = await call_next(request)
        # Post-processing logic
        return response

# In app.py
from src.api.middleware.custom_middleware import CustomMiddleware
app.add_middleware(CustomMiddleware)
```

**Q: How do I customize error responses?**

A: Register a custom exception handler:

```python
# File: src/api/error_handlers.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

async def custom_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "custom_field": "custom value"
            }
        }
    )

# In app.py
from fastapi import HTTPException
from src.api.error_handlers import custom_exception_handler
app.add_exception_handler(HTTPException, custom_exception_handler)
```

**Q: How do I optimize the response time for query operations?**

A: To improve query response times:
1. Ensure the database has proper indexes
2. Optimize vector search parameters
3. Implement caching for frequent queries
4. Use connection pooling
5. Increase worker count for parallel processing
6. Monitor and optimize slow queries

**Q: How do I implement custom authentication?**

A: Create a custom authentication dependency:

```python
# File: src/api/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def custom_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic here
    if not validate_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

# In your router
@router.post("/protected", dependencies=[Depends(custom_auth)])
async def protected_endpoint():
    # Implementation
    pass
``` 