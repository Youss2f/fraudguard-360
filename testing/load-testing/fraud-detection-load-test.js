// FraudGuard 360 - Load Testing Configuration
// K6 performance and load testing scripts for fraud detection system

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const fraudDetectionRate = new Rate('fraud_detection_success_rate');
const apiResponseTime = new Trend('api_response_time');
const alertProcessingTime = new Trend('alert_processing_time');

// Test configuration
export const options = {
  scenarios: {
    // Smoke test - basic functionality check
    smoke_test: {
      executor: 'constant-vus',
      vus: 1,
      duration: '30s',
      tags: { test_type: 'smoke' },
    },
    
    // Load test - normal expected load
    load_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 10 },  // Ramp up
        { duration: '5m', target: 10 },  // Stay at 10 users
        { duration: '2m', target: 20 },  // Ramp up to 20 users
        { duration: '5m', target: 20 },  // Stay at 20 users
        { duration: '2m', target: 0 },   // Ramp down
      ],
      tags: { test_type: 'load' },
    },
    
    // Stress test - beyond normal capacity
    stress_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 20 },  // Ramp up to 20 users
        { duration: '5m', target: 20 },  // Stay at 20 users
        { duration: '2m', target: 50 },  // Ramp up to 50 users
        { duration: '5m', target: 50 },  // Stay at 50 users
        { duration: '2m', target: 100 }, // Ramp up to 100 users
        { duration: '5m', target: 100 }, // Stay at 100 users
        { duration: '5m', target: 0 },   // Ramp down
      ],
      tags: { test_type: 'stress' },
    },
    
    // Spike test - sudden traffic increases
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 10 },  // Normal load
        { duration: '1m', target: 100 }, // Spike to 100 users
        { duration: '1m', target: 10 },  // Back to normal
      ],
      tags: { test_type: 'spike' },
    },
    
    // Volume test - sustained high load
    volume_test: {
      executor: 'constant-vus',
      vus: 50,
      duration: '30m',
      tags: { test_type: 'volume' },
    }
  },
  
  // Performance thresholds
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests must complete below 2s
    http_req_failed: ['rate<0.05'],    // Error rate must be below 5%
    fraud_detection_success_rate: ['rate>0.95'], // 95% fraud detection success
    api_response_time: ['p(99)<3000'], // 99% of API calls under 3s
  },
};

// Test data generators
function generateCDRData() {
  const callerNumbers = [
    '1234567890', '1111111111', '2222222222', '3333333333', '4444444444',
    '5555555555', '6666666666', '7777777777', '8888888888', '9999999999'
  ];
  
  const calleeNumbers = [
    '0987654321', '1010101010', '2020202020', '3030303030', '4040404040',
    '5050505050', '6060606060', '7070707070', '8080808080', '9090909090'
  ];
  
  return {
    call_id: `load_test_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    caller_number: callerNumbers[Math.floor(Math.random() * callerNumbers.length)],
    callee_number: calleeNumbers[Math.floor(Math.random() * calleeNumbers.length)],
    duration: Math.floor(Math.random() * 3600) + 30, // 30 seconds to 1 hour
    cost: Math.round((Math.random() * 50 + 1) * 100) / 100, // $1 to $50
    timestamp: new Date().toISOString(),
    location: ['US', 'UK', 'CA', 'AU', 'DE'][Math.floor(Math.random() * 5)]
  };
}

function generateSuspiciousCDRData() {
  return {
    call_id: `suspicious_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    caller_number: '1234567890', // Same caller for pattern detection
    callee_number: `999${Math.floor(Math.random() * 1000000).toString().padStart(7, '0')}`,
    duration: Math.floor(Math.random() * 1800) + 1800, // 30-60 minutes (suspicious)
    cost: Math.round((Math.random() * 100 + 50) * 100) / 100, // $50-$150 (expensive)
    timestamp: new Date().toISOString(),
    location: 'XX' // Unknown location
  };
}

// Main test function
export default function () {
  const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
  
  group('API Health Check', () => {
    const response = http.get(`${BASE_URL}/health`);
    check(response, {
      'health check status is 200': (r) => r.status === 200,
      'health check response time < 500ms': (r) => r.timings.duration < 500,
    });
  });
  
  group('CDR Analysis - Normal Traffic', () => {
    const cdrData = generateCDRData();
    const startTime = Date.now();
    
    const response = http.post(
      `${BASE_URL}/api/v1/analyze/cdr`,
      JSON.stringify(cdrData),
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: '10s',
      }
    );
    
    const responseTime = Date.now() - startTime;
    apiResponseTime.add(responseTime);
    
    const success = check(response, {
      'CDR analysis status is 200': (r) => r.status === 200,
      'CDR analysis has fraud_score': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.hasOwnProperty('fraud_score');
        } catch (e) {
          return false;
        }
      },
      'CDR analysis response time < 2s': (r) => r.timings.duration < 2000,
    });
    
    fraudDetectionRate.add(success);
  });
  
  group('CDR Analysis - Suspicious Traffic', () => {
    const suspiciousCDR = generateSuspiciousCDRData();
    const startTime = Date.now();
    
    const response = http.post(
      `${BASE_URL}/api/v1/analyze/cdr`,
      JSON.stringify(suspiciousCDR),
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: '10s',
      }
    );
    
    const responseTime = Date.now() - startTime;
    apiResponseTime.add(responseTime);
    
    check(response, {
      'Suspicious CDR analysis status is 200': (r) => r.status === 200,
      'Suspicious CDR has high fraud score': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.fraud_score && body.fraud_score > 0.5;
        } catch (e) {
          return false;
        }
      },
    });
  });
  
  group('Fraud Alerts Retrieval', () => {
    const response = http.get(`${BASE_URL}/api/v1/alerts?limit=50`);
    
    check(response, {
      'Alerts retrieval status is 200': (r) => r.status === 200,
      'Alerts response has alerts array': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body.alerts);
        } catch (e) {
          return false;
        }
      },
      'Alerts response time < 1s': (r) => r.timings.duration < 1000,
    });
  });
  
  group('Network Visualization Data', () => {
    const response = http.get(`${BASE_URL}/api/v1/network/visualization`);
    
    check(response, {
      'Network data status is 200': (r) => r.status === 200,
      'Network data has nodes and edges': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.nodes && body.edges && Array.isArray(body.nodes) && Array.isArray(body.edges);
        } catch (e) {
          return false;
        }
      },
      'Network data response time < 3s': (r) => r.timings.duration < 3000,
    });
  });
  
  group('Batch CDR Processing', () => {
    const batchSize = 10;
    const batchCDRs = Array.from({ length: batchSize }, () => generateCDRData());
    
    const response = http.post(
      `${BASE_URL}/api/v1/analyze/batch`,
      JSON.stringify({ cdrs: batchCDRs }),
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: '30s',
      }
    );
    
    check(response, {
      'Batch processing status is 200': (r) => r.status === 200,
      'Batch processing returns all results': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.results && body.results.length === batchSize;
        } catch (e) {
          return false;
        }
      },
      'Batch processing response time < 10s': (r) => r.timings.duration < 10000,
    });
  });
  
  // Simulate user think time
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

// Setup function - run once before the test
export function setup() {
  console.log('Starting FraudGuard 360 Load Testing...');
  console.log('Test Configuration:');
  console.log(`- Base URL: ${__ENV.BASE_URL || 'http://localhost:8000'}`);
  console.log(`- Test Type: ${__ENV.TEST_TYPE || 'load'}`);
  
  // Verify API is accessible
  const response = http.get(`${__ENV.BASE_URL || 'http://localhost:8000'}/health`);
  if (response.status !== 200) {
    throw new Error(`API is not accessible. Status: ${response.status}`);
  }
  
  return { baseUrl: __ENV.BASE_URL || 'http://localhost:8000' };
}

// Teardown function - run once after the test
export function teardown(data) {
  console.log('Load testing completed.');
  console.log('Check the results for performance metrics and thresholds.');
}

// Custom scenarios for specific testing needs
export const fraudDetectionScenario = {
  executor: 'constant-arrival-rate',
  rate: 50, // 50 requests per second
  timeUnit: '1s',
  duration: '5m',
  preAllocatedVUs: 10,
  maxVUs: 50,
};

export const highVolumeScenario = {
  executor: 'ramping-arrival-rate',
  startRate: 10,
  timeUnit: '1s',
  stages: [
    { duration: '2m', target: 50 },   // Ramp up to 50 RPS
    { duration: '5m', target: 100 },  // Ramp up to 100 RPS
    { duration: '10m', target: 100 }, // Stay at 100 RPS
    { duration: '2m', target: 0 },    // Ramp down
  ],
  preAllocatedVUs: 50,
  maxVUs: 200,
};