# FraudGuard-360 API Documentation

## Table of Contents
- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Request/Response Examples](#requestresponse-examples)
- [Error Codes](#error-codes)
- [Rate Limiting](#rate-limiting)

## Overview

The FraudGuard-360 API provides real-time fraud detection capabilities for transaction processing. The API accepts transaction data, processes it through a sophisticated machine learning pipeline, and returns fraud risk assessments.

### Key Features
- **Real-time Processing**: Sub-second response times for fraud detection
- **RESTful Design**: Standard HTTP methods and status codes
- **JSON Format**: All requests and responses use JSON
- **Async Processing**: Transactions are processed asynchronously via Kafka
- **Monitoring**: Built-in Prometheus metrics

## Base URL

```
Production: https://api.fraudguard.io
Development: http://localhost:8000
```

## Authentication

Currently, the API uses simple token-based authentication (to be enhanced with JWT in production).

```http
Authorization: Bearer YOUR_API_TOKEN
```

## API Endpoints

### 1. Health Check

Check the health status of the API Gateway.

**Endpoint:** `GET /`

**Response:**
```json
{
  "status": "healthy",
  "service": "api-gateway",
  "version": "1.0.0",
  "timestamp": "2025-11-01T12:00:00.000Z"
}
```

---

### 2. Detailed Health Check

Get detailed health status including dependency checks.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "api-gateway",
  "timestamp": "2025-11-01T12:00:00.000Z",
  "checks": {
    "kafka": "healthy"
  }
}
```

---

### 3. Submit Transaction for Analysis

Submit a transaction for fraud detection analysis.

**Endpoint:** `POST /transactions`

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | Unique user identifier |
| `amount` | float | Yes | Transaction amount (must be >= 0) |
| `location` | string | Yes | Transaction location |
| `merchant_id` | string | No | Merchant identifier |
| `transaction_type` | string | No | Type of transaction (default: "purchase") |
| `device_id` | string | No | Device identifier |
| `ip_address` | string | No | IP address of the transaction |

**Example Request:**
```json
{
  "user_id": "USR_12345",
  "amount": 250.50,
  "location": "New York, NY",
  "merchant_id": "MERCH_789",
  "transaction_type": "purchase",
  "device_id": "DEV_456",
  "ip_address": "192.168.1.100"
}
```

**Success Response (202 Accepted):**
```json
{
  "transaction_id": "TXN_1698840000000_USR_12345",
  "status": "accepted",
  "message": "Transaction submitted for fraud analysis",
  "timestamp": "2025-11-01T12:00:00.000Z"
}
```

**Error Responses:**

| Status Code | Description | Example |
|-------------|-------------|---------|
| 400 Bad Request | Invalid request data | Missing required field |
| 422 Unprocessable Entity | Validation error | Negative amount |
| 503 Service Unavailable | Kafka unavailable | Temporary connectivity issue |

---

### 4. Get Transaction Status

Retrieve the status of a previously submitted transaction.

**Endpoint:** `GET /transactions/{transaction_id}`

**Path Parameters:**
- `transaction_id` (string): The transaction ID returned from the submission

**Example Request:**
```http
GET /transactions/TXN_1698840000000_USR_12345
```

**Success Response (200 OK):**
```json
{
  "transaction_id": "TXN_1698840000000_USR_12345",
  "status": "processing",
  "message": "Transaction is being analyzed",
  "timestamp": "2025-11-01T12:00:00.000Z"
}
```

---

### 5. Prometheus Metrics

Get Prometheus-formatted metrics for monitoring.

**Endpoint:** `GET /metrics`

**Response Format:** Prometheus text format

**Example Response:**
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{method="GET",endpoint="/",status="200"} 150
api_requests_total{method="POST",endpoint="/transactions",status="202"} 45

# HELP transactions_total Total transactions processed
# TYPE transactions_total counter
transactions_total{status="success"} 42
transactions_total{status="kafka_error"} 3

# HELP request_duration_seconds Request duration in seconds
# TYPE request_duration_seconds histogram
request_duration_seconds_bucket{endpoint="/transactions",le="0.005"} 20
request_duration_seconds_bucket{endpoint="/transactions",le="0.01"} 40
```

## Request/Response Examples

### Example 1: Simple Transaction

**Request:**
```bash
curl -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USR_123",
    "amount": 100.00,
    "location": "Boston, MA"
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN_1698840000000_USR_123",
  "status": "accepted",
  "message": "Transaction submitted for fraud analysis",
  "timestamp": "2025-11-01T12:00:00.000Z"
}
```

### Example 2: Complete Transaction

**Request:**
```bash
curl -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USR_12345",
    "amount": 1500.00,
    "location": "New York, NY",
    "merchant_id": "MERCH_789",
    "transaction_type": "withdrawal",
    "device_id": "DEV_456",
    "ip_address": "192.168.1.100"
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN_1698840005000_USR_12345",
  "status": "accepted",
  "message": "Transaction submitted for fraud analysis",
  "timestamp": "2025-11-01T12:00:05.000Z"
}
```

### Example 3: Invalid Transaction (Negative Amount)

**Request:**
```bash
curl -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USR_123",
    "amount": -50.00,
    "location": "Boston, MA"
  }'
```

**Response (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "amount"],
      "msg": "Amount must be non-negative",
      "type": "value_error"
    }
  ]
}
```

## Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 200 | OK | Request successful |
| 202 | Accepted | Transaction accepted for processing |
| 400 | Bad Request | Invalid request format |
| 401 | Unauthorized | Missing or invalid authentication |
| 422 | Unprocessable Entity | Validation error in request data |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Dependent service unavailable (e.g., Kafka) |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Rate Limit:** 1000 requests per minute per API key
- **Burst Limit:** 100 requests per second
- **Headers:** Rate limit information is included in response headers

**Rate Limit Headers:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1698840060
```

When rate limit is exceeded, you'll receive a `429 Too Many Requests` response:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Please try again later.",
  "retry_after": 60
}
```

## Data Validation Rules

### Amount Field
- Must be a non-negative number
- Maximum precision: 2 decimal places
- Maximum value: No hard limit (but high values trigger fraud checks)

### User ID
- Required field
- String format
- Recommended format: `USR_` prefix followed by unique identifier

### Location
- Required field
- String format
- Recommended: City, State/Country format

### Transaction Type
- Optional field
- Valid values: `purchase`, `withdrawal`, `transfer`, `deposit`, `cash_advance`
- Default: `purchase`

## Integration Guide

### Python Example

```python
import requests
import json

# API configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "your_api_token_here"

# Prepare transaction data
transaction = {
    "user_id": "USR_12345",
    "amount": 250.50,
    "location": "New York, NY",
    "merchant_id": "MERCH_789",
    "transaction_type": "purchase"
}

# Submit transaction
response = requests.post(
    f"{API_BASE_URL}/transactions",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    },
    json=transaction
)

if response.status_code == 202:
    result = response.json()
    transaction_id = result["transaction_id"]
    print(f"Transaction submitted: {transaction_id}")
    
    # Check status
    status_response = requests.get(
        f"{API_BASE_URL}/transactions/{transaction_id}",
        headers={"Authorization": f"Bearer {API_TOKEN}"}
    )
    print(f"Status: {status_response.json()}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const API_BASE_URL = 'http://localhost:8000';
const API_TOKEN = 'your_api_token_here';

async function submitTransaction() {
    try {
        const transaction = {
            user_id: 'USR_12345',
            amount: 250.50,
            location: 'New York, NY',
            merchant_id: 'MERCH_789',
            transaction_type: 'purchase'
        };

        const response = await axios.post(
            `${API_BASE_URL}/transactions`,
            transaction,
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_TOKEN}`
                }
            }
        );

        console.log('Transaction submitted:', response.data);
        
        // Check status
        const transactionId = response.data.transaction_id;
        const statusResponse = await axios.get(
            `${API_BASE_URL}/transactions/${transactionId}`,
            {
                headers: {
                    'Authorization': `Bearer ${API_TOKEN}`
                }
            }
        );
        
        console.log('Status:', statusResponse.data);
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

submitTransaction();
```

## Support

For API support:
- Email: api-support@fraudguard.io
- GitHub Issues: https://github.com/Youss2f/fraudguard-360/issues
- Documentation: https://docs.fraudguard.io

## Changelog

### Version 1.0.0 (2025-11-01)
- Initial API release
- Transaction submission endpoint
- Health check endpoints
- Prometheus metrics integration
- Kafka integration for async processing
