# ==============================================================================
# FraudGuard 360 - API Gateway Production Dockerfile
# Multi-stage build for optimized, secure production deployment
# ==============================================================================

# Build Stage - Install dependencies and prepare application
FROM python:3.14-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Add metadata labels
LABEL maintainer="FraudGuard Team <team@fraudguard.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="fraudguard-api-gateway" \
      org.label-schema.description="FraudGuard 360 API Gateway Service" \
      org.label-schema.url="https://github.com/Youss2f/fraudguard-360" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/Youss2f/fraudguard-360" \
      org.label-schema.vendor="FraudGuard Inc." \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Create non-root user for security
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production Stage - Minimal runtime environment
FROM python:3.14-slim as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set work directory and permissions
WORKDIR /app
RUN chown -R app:app /app

# Copy application code
COPY --chown=app:app app/ ./app/

# Switch to non-root user
USER app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use tini as init system
ENTRYPOINT ["tini", "--"]

# Production command (no reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--access-log"]
