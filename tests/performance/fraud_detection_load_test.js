import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('error_rate');

// Test configuration
export let options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users
    { duration: '5m', target: 10 },   // Stay at 10 users
    { duration: '2m', target: 20 },   // Ramp up to 20 users
    { duration: '5m', target: 20 },   // Stay at 20 users
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests should be below 500ms
    error_rate: ['rate<0.1'],         // Error rate should be below 10%
  },
};

const BASE_URL = 'http://localhost:8000';

// Test data generators
function generateCDR() {
  const userId = Math.floor(Math.random() * 1000);
  return {
    call_id: `load_test_${Date.now()}_${Math.random()}`,
    caller_id: `user_${userId}`,
    callee_id: `user_${(userId + 1) % 1000}`,
    timestamp: Date.now(),
    duration: Math.floor(Math.random() * 300) + 60, // 1-5 minutes
    call_type: 'voice',
    location: `geo:${40 + Math.random()},-${74 + Math.random()}`,
  };
}

function generateSubscriber() {
  return {
    id: `load_test_user_${Math.floor(Math.random() * 10000)}`,
    phone_number: `+1555${Math.floor(Math.random() * 9999999).toString().padStart(7, '0')}`,
    account_type: Math.random() > 0.8 ? 'premium' : 'standard',
    registration_date: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
  };
}

export default function () {
  // Test 1: Health check
  let response = http.get(`${BASE_URL}/health`);
  check(response, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 100ms': (r) => r.timings.duration < 100,
  }) || errorRate.add(1);

  sleep(1);

  // Test 2: Authentication
  const authResponse = http.post(`${BASE_URL}/auth/login`, {
    username: 'testuser',
    password: 'testpassword'
  });
  
  let authToken = '';
  if (check(authResponse, {
    'authentication successful': (r) => r.status === 200,
  })) {
    authToken = authResponse.json('access_token');
  } else {
    errorRate.add(1);
  }

  const headers = {
    'Authorization': `Bearer ${authToken}`,
    'Content-Type': 'application/json',
  };

  sleep(1);

  // Test 3: CDR ingestion
  const cdrData = generateCDR();
  response = http.post(`${BASE_URL}/api/v1/cdr`, JSON.stringify(cdrData), {
    headers: headers,
  });
  check(response, {
    'CDR ingestion status is 201': (r) => r.status === 201,
    'CDR ingestion response time < 200ms': (r) => r.timings.duration < 200,
  }) || errorRate.add(1);

  sleep(1);

  // Test 4: Fraud detection
  response = http.post(`${BASE_URL}/api/v1/fraud/detect`, JSON.stringify({
    subscriber_id: cdrData.caller_id,
  }), {
    headers: headers,
  });
  check(response, {
    'fraud detection status is 200': (r) => r.status === 200,
    'fraud detection response time < 500ms': (r) => r.timings.duration < 500,
    'fraud detection returns valid score': (r) => {
      const body = r.json();
      return body.fraud_score !== undefined && body.fraud_score >= 0 && body.fraud_score <= 1;
    },
  }) || errorRate.add(1);

  sleep(1);

  // Test 5: Get fraud alerts
  response = http.get(`${BASE_URL}/api/v1/alerts?limit=10`, {
    headers: headers,
  });
  check(response, {
    'get alerts status is 200': (r) => r.status === 200,
    'get alerts response time < 300ms': (r) => r.timings.duration < 300,
    'alerts response is array': (r) => Array.isArray(r.json()),
  }) || errorRate.add(1);

  sleep(1);

  // Test 6: Dashboard analytics
  response = http.get(`${BASE_URL}/api/v1/analytics/dashboard`, {
    headers: headers,
  });
  check(response, {
    'dashboard analytics status is 200': (r) => r.status === 200,
    'dashboard analytics response time < 400ms': (r) => r.timings.duration < 400,
  }) || errorRate.add(1);

  sleep(1);

  // Test 7: Graph visualization data
  response = http.get(`${BASE_URL}/api/v1/graph/network?subscriber_id=${cdrData.caller_id}`, {
    headers: headers,
  });
  check(response, {
    'graph data status is 200': (r) => r.status === 200,
    'graph data response time < 600ms': (r) => r.timings.duration < 600,
  }) || errorRate.add(1);

  sleep(2);
}

// Setup function
export function setup() {
  console.log('Starting load test setup...');
  
  // Health check before starting
  const response = http.get(`${BASE_URL}/health`);
  if (response.status !== 200) {
    throw new Error(`API not ready: ${response.status}`);
  }
  
  console.log('API health check passed, starting load test...');
  return { baseUrl: BASE_URL };
}

// Teardown function
export function teardown(data) {
  console.log('Load test completed');
  
  // Generate summary report
  console.log('Performance Test Summary:');
  console.log('- Target: Fraud detection API load testing');
  console.log('- Max concurrent users: 50');
  console.log('- Test duration: ~20 minutes');
  console.log('- Performance thresholds: P95 < 500ms, Error rate < 10%');
}