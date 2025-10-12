---
layout: default
title: API Reference
nav_order: 4
description: "Complete API documentation for FraudGuard 360° services"
---

# API Reference
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

FraudGuard 360° provides RESTful APIs for fraud detection, risk assessment, and network analysis. All APIs follow OpenAPI 3.0 specifications and include interactive documentation.

### Base URLs

| Service | Base URL | Documentation |
|---------|----------|---------------|
| **API Gateway** | `http://localhost:8000` | [/docs](http://localhost:8000/docs) |
| **ML Inference** | `http://localhost:8001` | [/docs](http://localhost:8001/docs) |
| **Risk Scoring** | `http://localhost:8002` | [/docs](http://localhost:8002/docs) |
| **Graph Analytics** | `http://localhost:8003` | [/docs](http://localhost:8003/docs) |

### Authentication

All APIs use JWT (JSON Web Token) authentication:

```bash
# Get access token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "password"
  }'

# Use token in requests
curl -X POST http://localhost:8000/api/v1/fraud/analyze \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

## API Gateway Endpoints

### Authentication

#### POST /auth/token
{: .d-inline-block }

Authentication
{: .label .label-blue }

Get JWT access token for API authentication.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

#### POST /auth/refresh
{: .d-inline-block }

Authentication
{: .label .label-blue }

Refresh JWT access token.

**Headers:**
```
Authorization: Bearer <refresh_token>
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Fraud Detection

#### POST /api/v1/fraud/analyze
{: .d-inline-block }

Core API
{: .label .label-green }

Analyze a single transaction for fraud indicators.

**Request Body:**
```json
{
  "transaction_id": "string",
  "amount": "number",
  "currency": "string",
  "merchant_id": "string",
  "customer_id": "string",
  "timestamp": "string (ISO 8601)",
  "location": {
    "country": "string",
    "city": "string",
    "latitude": "number",
    "longitude": "number"
  },
  "payment_method": "string",
  "channel": "string"
}
```

**Response:**
```json
{
  "transaction_id": "string",
  "fraud_score": "number (0-1)",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "prediction": "LEGITIMATE|SUSPICIOUS|FRAUD",
  "confidence": "number (0-1)",
  "processing_time_ms": "number",
  "risk_factors": ["string"],
  "recommendation": "APPROVE|REVIEW|BLOCK_TRANSACTION",
  "model_version": "string",
  "timestamp": "string (ISO 8601)"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/fraud/analyze \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_123456",
    "amount": 1500.00,
    "currency": "USD",
    "merchant_id": "merchant_789",
    "customer_id": "customer_456",
    "timestamp": "2025-10-10T10:30:00Z",
    "location": {
      "country": "US",
      "city": "New York"
    },
    "payment_method": "credit_card",
    "channel": "online"
  }'
```

#### POST /api/v1/fraud/analyze/batch
{: .d-inline-block }

Batch Processing
{: .label .label-yellow }

Analyze multiple transactions in a single request.

**Request Body:**
```json
{
  "transactions": [
    {
      "transaction_id": "string",
      "amount": "number",
      "currency": "string",
      "merchant_id": "string",
      "customer_id": "string",
      "timestamp": "string"
    }
  ],
  "options": {
    "include_explanations": "boolean",
    "include_network_analysis": "boolean"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "transaction_id": "string",
      "fraud_score": "number",
      "risk_level": "string",
      "prediction": "string",
      "confidence": "number"
    }
  ],
  "batch_summary": {
    "total_transactions": "number",
    "fraud_detected": "number",
    "processing_time_ms": "number"
  }
}
```

### System Information

#### GET /health
{: .d-inline-block }

Health Check
{: .label .label-grey-dk-000 }

Check service health and status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-10T10:30:00Z",
  "version": "1.2.0",
  "services": {
    "ml_service": "healthy",
    "graph_service": "healthy",
    "risk_service": "healthy",
    "database": "healthy",
    "cache": "healthy"
  },
  "uptime_seconds": 3600
}
```

#### GET /api/v1/metrics
{: .d-inline-block }

Metrics
{: .label .label-grey-dk-000 }

Get system performance metrics.

**Response:**
```json
{
  "fraud_detection": {
    "total_transactions_analyzed": 15420,
    "fraud_detected": 87,
    "false_positive_rate": 0.015,
    "average_processing_time_ms": 245
  },
  "system_performance": {
    "cpu_usage_percent": 45.2,
    "memory_usage_percent": 67.8,
    "active_connections": 23
  },
  "model_performance": {
    "accuracy": 0.973,
    "precision": 0.951,
    "recall": 0.889,
    "f1_score": 0.919
  }
}
```

## ML Inference Service

### Model Operations

#### POST /predict
{: .d-inline-block }

ML Prediction
{: .label .label-blue }

Get fraud prediction from GraphSAGE model.

**Request Body:**
```json
{
  "features": {
    "amount": "number",
    "merchant_category": "string",
    "time_of_day": "number",
    "day_of_week": "number",
    "customer_history": {
      "avg_transaction_amount": "number",
      "transaction_frequency": "number",
      "account_age_days": "number"
    }
  },
  "graph_context": {
    "customer_id": "string",
    "merchant_id": "string",
    "include_network": "boolean"
  }
}
```

**Response:**
```json
{
  "prediction": "number (0-1)",
  "confidence": "number (0-1)",
  "model_name": "GraphSAGE_v1.2",
  "feature_importance": {
    "amount": 0.45,
    "merchant_category": 0.23,
    "time_patterns": 0.32
  },
  "processing_time_ms": 156
}
```

#### GET /model/status
{: .d-inline-block }

Model Info
{: .label .label-grey-dk-000 }

Get model information and status.

**Response:**
```json
{
  "model_name": "GraphSAGE Fraud Detection",
  "version": "1.2.0",
  "last_trained": "2025-10-01T00:00:00Z",
  "accuracy": 0.973,
  "status": "active",
  "input_features": 47,
  "output_classes": 2
}
```

### Model Training

#### POST /train
{: .d-inline-block }

Training
{: .label .label-red }

Trigger model retraining (admin only).

**Request Body:**
```json
{
  "training_data": {
    "start_date": "string",
    "end_date": "string",
    "include_graph_features": "boolean"
  },
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
  }
}
```

**Response:**
```json
{
  "training_job_id": "string",
  "status": "started",
  "estimated_completion": "string"
}
```

#### GET /train/status/{job_id}
{: .d-inline-block }

Training Status
{: .label .label-grey-dk-000 }

Check training job status.

**Response:**
```json
{
  "job_id": "string",
  "status": "running|completed|failed",
  "progress": 0.75,
  "current_epoch": 75,
  "total_epochs": 100,
  "current_loss": 0.234,
  "estimated_completion": "string"
}
```

## Risk Scoring Service

### Risk Assessment

#### POST /assess
{: .d-inline-block }

Risk Assessment
{: .label .label-yellow }

Calculate comprehensive risk score.

**Request Body:**
```json
{
  "customer_profile": {
    "customer_id": "string",
    "account_age_days": "number",
    "total_transactions": "number",
    "average_amount": "number"
  },
  "transaction_context": {
    "amount": "number",
    "merchant_type": "string",
    "location": "string",
    "time_of_day": "number"
  },
  "behavioral_patterns": {
    "velocity_indicators": ["string"],
    "spending_patterns": ["string"]
  }
}
```

**Response:**
```json
{
  "risk_score": 750,
  "risk_level": "HIGH",
  "risk_factors": [
    {
      "factor": "unusual_amount",
      "weight": 0.35,
      "description": "Transaction amount 3x higher than average"
    },
    {
      "factor": "velocity_pattern",
      "weight": 0.25,
      "description": "5 transactions in 10 minutes"
    }
  ],
  "recommendations": [
    "Additional verification required",
    "Monitor for 24 hours"
  ]
}
```

#### GET /risk/factors
{: .d-inline-block }

Risk Factors
{: .label .label-grey-dk-000 }

Get list of available risk factors.

**Response:**
```json
{
  "risk_factors": [
    {
      "name": "unusual_amount",
      "description": "Transaction amount significantly different from normal",
      "weight_range": [0.1, 0.5],
      "category": "transaction"
    },
    {
      "name": "velocity_pattern",
      "description": "High frequency of transactions in short time",
      "weight_range": [0.2, 0.4],
      "category": "behavioral"
    }
  ]
}
```

### Customer Profiles

#### GET /customer/{customer_id}/profile
{: .d-inline-block }

Customer Profile
{: .label .label-green }

Get customer risk profile and history.

**Response:**
```json
{
  "customer_id": "string",
  "risk_profile": {
    "current_score": 340,
    "risk_level": "LOW",
    "last_updated": "string"
  },
  "transaction_patterns": {
    "average_amount": 245.67,
    "transaction_frequency": "daily",
    "preferred_merchants": ["merchant_1", "merchant_2"],
    "typical_locations": ["city_1", "city_2"]
  },
  "behavioral_indicators": {
    "spending_consistency": "high",
    "location_consistency": "medium",
    "time_pattern_regularity": "high"
  }
}
```

## Graph Analytics Service

### Network Analysis

#### POST /analyze/network
{: .d-inline-block }

Network Analysis
{: .label .label-purple }

Analyze fraud networks and relationships.

**Request Body:**
```json
{
  "center_node": {
    "type": "customer|merchant|device",
    "id": "string"
  },
  "analysis_params": {
    "depth": 2,
    "min_transaction_amount": 100,
    "time_window_days": 30,
    "include_suspected_nodes": true
  }
}
```

**Response:**
```json
{
  "network_summary": {
    "total_nodes": 45,
    "total_edges": 127,
    "suspicious_nodes": 8,
    "fraud_rings_detected": 2
  },
  "nodes": [
    {
      "id": "string",
      "type": "customer|merchant|device",
      "risk_score": 0.75,
      "properties": {
        "total_transactions": 156,
        "total_amount": 45670.23
      }
    }
  ],
  "edges": [
    {
      "source": "string",
      "target": "string",
      "relationship": "transacted_with",
      "weight": 0.85,
      "transaction_count": 23
    }
  ],
  "fraud_rings": [
    {
      "ring_id": "string",
      "members": ["node_1", "node_2", "node_3"],
      "confidence": 0.89,
      "pattern_type": "money_laundering"
    }
  ]
}
```

#### GET /community/detect
{: .d-inline-block }

Community Detection
{: .label .label-purple }

Detect communities and clusters in transaction networks.

**Query Parameters:**
- `algorithm`: louvain|leiden|walktrap
- `resolution`: number (0.1-2.0)
- `min_community_size`: number

**Response:**
```json
{
  "communities": [
    {
      "community_id": "string",
      "members": ["node_1", "node_2", "node_3"],
      "size": 15,
      "modularity": 0.74,
      "suspicion_level": "HIGH",
      "common_patterns": ["rapid_transactions", "round_amounts"]
    }
  ],
  "algorithm_info": {
    "algorithm": "louvain",
    "resolution": 1.0,
    "execution_time_ms": 234
  }
}
```

### Pattern Detection

#### POST /patterns/detect
{: .d-inline-block }

Pattern Detection
{: .label .label-red }

Detect suspicious patterns in transaction data.

**Request Body:**
```json
{
  "pattern_types": ["circular_transfers", "smurfing", "layering"],
  "time_window": {
    "start": "2025-10-01T00:00:00Z",
    "end": "2025-10-10T23:59:59Z"
  },
  "thresholds": {
    "min_amount": 1000,
    "max_participants": 10
  }
}
```

**Response:**
```json
{
  "patterns_detected": [
    {
      "pattern_id": "string",
      "type": "circular_transfers",
      "confidence": 0.92,
      "participants": ["customer_1", "customer_2", "customer_3"],
      "total_amount": 150000.00,
      "transaction_count": 47,
      "time_span_hours": 6,
      "description": "Circular money movement detected"
    }
  ],
  "analysis_summary": {
    "total_patterns": 3,
    "high_confidence": 2,
    "total_suspicious_amount": 450000.00
  }
}
```

## Error Handling

All APIs use standard HTTP status codes and return structured error responses:

### Error Response Format

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "string",
    "timestamp": "string",
    "request_id": "string"
  }
}
```

### Common Error Codes

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | `INVALID_REQUEST` | Invalid request format or parameters |
| 401 | `UNAUTHORIZED` | Missing or invalid authentication |
| 403 | `FORBIDDEN` | Insufficient permissions |
| 404 | `NOT_FOUND` | Resource not found |
| 429 | `RATE_LIMITED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Internal server error |
| 503 | `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

### Example Error Response

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field: customer_id",
    "details": "The customer_id field is required for fraud analysis",
    "timestamp": "2025-10-10T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

## Rate Limiting

All APIs implement rate limiting to ensure fair usage:

- **Default Limit**: 2000 requests per minute per client
- **Burst Limit**: 100 requests per second
- **Headers**: Rate limit information in response headers

```
X-RateLimit-Limit: 2000
X-RateLimit-Remaining: 1850
X-RateLimit-Reset: 1696950000
```

## SDK and Client Libraries

### Python SDK

```python
from fraudguard360 import FraudGuardClient

client = FraudGuardClient(
    api_key="your_api_key",
    base_url="http://localhost:8000"
)

# Analyze transaction
result = client.analyze_transaction({
    "transaction_id": "txn_123",
    "amount": 1500.00,
    "customer_id": "customer_456"
})

print(f"Fraud Score: {result.fraud_score}")
```

### JavaScript/Node.js SDK

```javascript
const FraudGuard = require('fraudguard360-js');

const client = new FraudGuard({
    apiKey: 'your_api_key',
    baseUrl: 'http://localhost:8000'
});

// Analyze transaction
const result = await client.analyzeTransaction({
    transaction_id: 'txn_123',
    amount: 1500.00,
    customer_id: 'customer_456'
});

console.log(`Fraud Score: ${result.fraud_score}`);
```

---

## Interactive Documentation

For complete interactive API documentation with request/response examples:

- **API Gateway**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ML Inference**: [http://localhost:8001/docs](http://localhost:8001/docs)  
- **Risk Scoring**: [http://localhost:8002/docs](http://localhost:8002/docs)
- **Graph Analytics**: [http://localhost:8003/docs](http://localhost:8003/docs)