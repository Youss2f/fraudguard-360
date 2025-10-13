# ==============================================================================
# FraudGuard 360 - Production Dockerfile
# Multi-service container for API Gateway and Frontend
# ==============================================================================

FROM node:18-alpine-alpine as frontend-builder

# Build frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --production=false
COPY frontend/ ./
RUN npm run build

# Production image
FROM node:18-alpine-alpine

# Install Python for API Gateway
RUN apk add --no-cache python3 py3-pip

# Create app directory
WORKDIR /app

# Copy frontend build
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Copy API Gateway
COPY api-gateway/ ./api-gateway/
COPY package*.json ./

# Install dependencies
RUN npm ci --production

# Install Python dependencies for API Gateway
WORKDIR /app/api-gateway
RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi

WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Change ownership of the app directory
RUN chown -R nodejs:nodejs /app
USER nodejs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start the application
CMD ["npm", "start"]