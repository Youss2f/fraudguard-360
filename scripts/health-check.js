#!/usr/bin/env node

/**
 * FraudGuard 360 - Health Check Script
 * Simple health check to verify application status
 */

const http = require('http');
const https = require('https');

const HEALTH_ENDPOINTS = [
  { name: 'API Gateway', url: 'http://localhost:8000/health', timeout: 5000 },
  { name: 'Frontend', url: 'http://localhost:3000', timeout: 5000 }
];

function checkEndpoint(endpoint) {
  return new Promise((resolve) => {
    const protocol = endpoint.url.startsWith('https:') ? https : http;
    const request = protocol.get(endpoint.url, (res) => {
      const status = res.statusCode >= 200 && res.statusCode < 300 ? 'UP' : 'DOWN';
      resolve({
        name: endpoint.name,
        url: endpoint.url,
        status: status,
        statusCode: res.statusCode
      });
    });

    request.setTimeout(endpoint.timeout, () => {
      request.destroy();
      resolve({
        name: endpoint.name,
        url: endpoint.url,
        status: 'TIMEOUT',
        statusCode: 'N/A'
      });
    });

    request.on('error', (error) => {
      resolve({
        name: endpoint.name,
        url: endpoint.url,
        status: 'ERROR',
        error: error.message
      });
    });
  });
}

async function runHealthCheck() {
  console.log('🏥 FraudGuard 360 - Health Check');
  console.log('================================');
  
  const results = await Promise.all(HEALTH_ENDPOINTS.map(checkEndpoint));
  
  let allHealthy = true;
  
  results.forEach(result => {
    const status = result.status === 'UP' ? '✅' : '❌';
    console.log(`${status} ${result.name}: ${result.status}`);
    if (result.statusCode) {
      console.log(`   Status Code: ${result.statusCode}`);
    }
    if (result.error) {
      console.log(`   Error: ${result.error}`);
    }
    
    if (result.status !== 'UP') {
      allHealthy = false;
    }
  });
  
  console.log('================================');
  console.log(`Overall Status: ${allHealthy ? '✅ HEALTHY' : '❌ UNHEALTHY'}`);
  
  // Exit with error code if any service is down
  process.exit(allHealthy ? 0 : 1);
}

// Run health check if this script is executed directly
if (require.main === module) {
  runHealthCheck().catch(error => {
    console.error('Health check failed:', error);
    process.exit(1);
  });
}

module.exports = { runHealthCheck, checkEndpoint };