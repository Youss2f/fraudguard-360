# ==============================================================================
# FraudGuard 360 - ML Service Production Dockerfile
# Multi-stage build for PyTorch-based GraphSAGE fraud detection
# ==============================================================================

# Build Stage - Install dependencies and compile models
FROM python:3.11-alpine-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Add metadata labels
LABEL maintainer="FraudGuard Team <team@fraudguard.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="fraudguard-ml-service" \
      org.label-schema.description="FraudGuard 360 ML Inference Service" \
      org.label-schema.url="https://github.com/Youss2f/fraudguard-360" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/Youss2f/fraudguard-360" \
      org.label-schema.vendor="FraudGuard Inc." \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Set work directory
WORKDIR /build

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Production Stage - Minimal runtime environment
FROM python:3.11-alpine-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    tini \
    libgomp1 \
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
COPY --chown=app:app . .

# Create model and data directories
RUN mkdir -p models data logs && \
    chown -R app:app models data logs

# Switch to non-root user
USER app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Use tini as init system
ENTRYPOINT ["tini", "--"]

# Production command
CMD ["uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2", "--access-log"]
