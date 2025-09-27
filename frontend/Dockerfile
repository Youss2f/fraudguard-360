# ==============================================================================
# FraudGuard 360 - Frontend Production Dockerfile
# Multi-stage build with Nginx for optimized production deployment
# ==============================================================================

# Build Stage - Create optimized production bundle
FROM node:18-alpine as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0
ARG REACT_APP_API_URL=http://localhost:8000
ARG REACT_APP_ENVIRONMENT=production

# Add metadata labels
LABEL maintainer="FraudGuard Team <team@fraudguard.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="fraudguard-frontend" \
      org.label-schema.description="FraudGuard 360 React Frontend Dashboard" \
      org.label-schema.url="https://github.com/Youss2f/fraudguard-360" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/Youss2f/fraudguard-360" \
      org.label-schema.vendor="FraudGuard Inc." \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install security updates
RUN apk update && apk upgrade && apk add --no-cache \
    git \
    && rm -rf /var/cache/apk/*

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including devDependencies for build)
RUN npm install && \
    npm cache clean --force

# Copy source code
COPY . .

# Set environment variables for build
ENV REACT_APP_API_URL=$REACT_APP_API_URL \
    REACT_APP_ENVIRONMENT=$REACT_APP_ENVIRONMENT \
    NODE_ENV=production

# Build the application
RUN node node_modules/react-scripts/bin/react-scripts.js build

# Production Stage - Nginx web server
FROM nginx:1.25-alpine as production

# Install security updates and curl for health checks
RUN apk update && apk upgrade && apk add --no-cache \
    curl \
    && rm -rf /var/cache/apk/*

# Create nginx user and set permissions
RUN addgroup -g 1000 -S app && \
    adduser -u 1000 -S app -G app

# Copy built application from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create nginx directories and set permissions
RUN mkdir -p /var/cache/nginx /var/log/nginx /var/run && \
    chown -R app:app /var/cache/nginx /var/log/nginx /var/run /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Expose port
EXPOSE 3000

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
