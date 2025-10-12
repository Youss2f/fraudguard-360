// K6 Load Testing Runner Scripts for FraudGuard 360
// Collection of load testing scenarios and execution scripts

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Test configuration
const CONFIG = {
  baseUrl: process.env.BASE_URL || 'http://localhost:8000',
  testDataDir: './test-data',
  resultsDir: './results',
  scripts: {
    smoke: './fraud-detection-load-test.js',
    load: './fraud-detection-load-test.js',
    stress: './fraud-detection-load-test.js',
    spike: './fraud-detection-load-test.js',
    volume: './fraud-detection-load-test.js'
  }
};

// Ensure directories exist
function ensureDirectories() {
  [CONFIG.testDataDir, CONFIG.resultsDir].forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  });
}

// Generate test data
function generateTestData() {
  console.log('Generating test data...');
  
  // Generate CDR test data
  const cdrData = [];
  const susData = [];
  
  // Normal CDR data
  for (let i = 0; i < 1000; i++) {
    cdrData.push({
      call_id: `test_${i}_${Date.now()}`,
      caller_number: `123456${String(i).padStart(4, '0')}`,
      callee_number: `987654${String(i % 100).padStart(4, '0')}`,
      duration: Math.floor(Math.random() * 1800) + 60,
      cost: Math.round((Math.random() * 20 + 1) * 100) / 100,
      timestamp: new Date(Date.now() - Math.random() * 86400000).toISOString(),
      location: ['US', 'UK', 'CA', 'AU', 'DE'][i % 5]
    });
  }
  
  // Suspicious CDR data
  for (let i = 0; i < 100; i++) {
    susData.push({
      call_id: `suspicious_${i}_${Date.now()}`,
      caller_number: '1234567890', // Same caller
      callee_number: `999${String(i).padStart(7, '0')}`,
      duration: Math.floor(Math.random() * 1800) + 1800, // Long calls
      cost: Math.round((Math.random() * 100 + 50) * 100) / 100, // Expensive
      timestamp: new Date().toISOString(),
      location: 'XX'
    });
  }
  
  fs.writeFileSync(
    path.join(CONFIG.testDataDir, 'cdr-data.json'),
    JSON.stringify(cdrData, null, 2)
  );
  
  fs.writeFileSync(
    path.join(CONFIG.testDataDir, 'suspicious-data.json'),
    JSON.stringify(susData, null, 2)
  );
  
  console.log(`Generated ${cdrData.length} normal CDR records`);
  console.log(`Generated ${susData.length} suspicious CDR records`);
}

// Run specific test scenario
function runTest(scenario, options = {}) {
  console.log(`\n🚀 Running ${scenario} test...`);
  
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const resultFile = path.join(CONFIG.resultsDir, `${scenario}-${timestamp}.json`);
  
  const k6Command = [
    'k6 run',
    `--out json=${resultFile}`,
    `--env BASE_URL=${CONFIG.baseUrl}`,
    `--env TEST_TYPE=${scenario}`,
    options.duration ? `--duration ${options.duration}` : '',
    options.vus ? `--vus ${options.vus}` : '',
    CONFIG.scripts[scenario]
  ].filter(Boolean).join(' ');
  
  try {
    console.log(`Executing: ${k6Command}`);
    const output = execSync(k6Command, { 
      encoding: 'utf8', 
      timeout: options.timeout || 3600000 // 1 hour timeout
    });
    
    console.log('✅ Test completed successfully');
    console.log(`📊 Results saved to: ${resultFile}`);
    
    // Parse and display summary
    if (fs.existsSync(resultFile)) {
      displayTestSummary(resultFile, scenario);
    }
    
    return { success: true, resultFile, output };
  } catch (error) {
    console.error(`❌ Test failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

// Display test summary
function displayTestSummary(resultFile, scenario) {
  try {
    const data = fs.readFileSync(resultFile, 'utf8');
    const metrics = data.split('\n')
      .filter(line => line.trim())
      .map(line => JSON.parse(line))
      .filter(entry => entry.type === 'Point' && entry.metric);
    
    console.log(`\n📈 ${scenario.toUpperCase()} TEST SUMMARY:`);
    console.log('=' .repeat(50));
    
    // Key metrics
    const httpReqDuration = metrics.filter(m => m.metric === 'http_req_duration');
    const httpReqFailed = metrics.filter(m => m.metric === 'http_req_failed');
    const iterations = metrics.filter(m => m.metric === 'iterations');
    
    if (httpReqDuration.length > 0) {
      const durations = httpReqDuration.map(m => m.data.value);
      const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
      const p95Duration = durations.sort((a, b) => a - b)[Math.floor(durations.length * 0.95)];
      
      console.log(`Average Response Time: ${avgDuration.toFixed(2)}ms`);
      console.log(`95th Percentile: ${p95Duration.toFixed(2)}ms`);
    }
    
    if (httpReqFailed.length > 0) {
      const failures = httpReqFailed.filter(m => m.data.value > 0).length;
      const failureRate = (failures / httpReqFailed.length * 100).toFixed(2);
      console.log(`Error Rate: ${failureRate}%`);
    }
    
    if (iterations.length > 0) {
      console.log(`Total Iterations: ${iterations.length}`);
    }
    
    console.log('=' .repeat(50));
  } catch (error) {
    console.log('Could not parse test results');
  }
}

// Run comprehensive test suite
async function runTestSuite() {
  console.log('🎯 Starting FraudGuard 360 Load Testing Suite');
  console.log('=' .repeat(60));
  
  ensureDirectories();
  generateTestData();
  
  const testScenarios = [
    { name: 'smoke', description: 'Smoke Test - Basic Functionality' },
    { name: 'load', description: 'Load Test - Normal Expected Load' },
    { name: 'stress', description: 'Stress Test - Beyond Normal Capacity' },
    { name: 'spike', description: 'Spike Test - Sudden Traffic Increases' }
  ];
  
  const results = [];
  
  for (const scenario of testScenarios) {
    console.log(`\n📋 ${scenario.description}`);
    const result = runTest(scenario.name);
    results.push({ scenario: scenario.name, ...result });
    
    // Wait between tests
    if (scenario.name !== 'smoke') {
      console.log('⏱️  Waiting 30 seconds before next test...');
      await new Promise(resolve => setTimeout(resolve, 30000));
    }
  }
  
  // Generate final report
  generateFinalReport(results);
}

// Generate comprehensive test report
function generateFinalReport(results) {
  console.log('\n📊 FINAL TEST REPORT');
  console.log('=' .repeat(60));
  
  const report = {
    timestamp: new Date().toISOString(),
    configuration: CONFIG,
    results: results,
    summary: {
      totalTests: results.length,
      successfulTests: results.filter(r => r.success).length,
      failedTests: results.filter(r => !r.success).length
    }
  };
  
  const reportFile = path.join(CONFIG.resultsDir, `test-report-${Date.now()}.json`);
  fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
  
  // Display summary
  results.forEach(result => {
    const status = result.success ? '✅ PASSED' : '❌ FAILED';
    console.log(`${result.scenario.toUpperCase().padEnd(10)} ${status}`);
    if (!result.success) {
      console.log(`   Error: ${result.error}`);
    }
  });
  
  console.log(`\n📁 Full report saved to: ${reportFile}`);
  console.log('\n🏁 Load testing suite completed!');
}

// Performance benchmark
function runPerformanceBenchmark() {
  console.log('🏃‍♂️ Running Performance Benchmark...');
  
  const benchmarkScenarios = [
    { vus: 1, duration: '1m', name: 'baseline' },
    { vus: 10, duration: '2m', name: 'moderate' },
    { vus: 50, duration: '3m', name: 'high' },
    { vus: 100, duration: '2m', name: 'extreme' }
  ];
  
  benchmarkScenarios.forEach((scenario, index) => {
    console.log(`\n📊 Benchmark ${index + 1}/${benchmarkScenarios.length}: ${scenario.name}`);
    runTest('load', {
      vus: scenario.vus,
      duration: scenario.duration,
      timeout: 300000 // 5 minutes
    });
  });
}

// CLI interface
function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'help';
  
  switch (command) {
    case 'smoke':
      runTest('smoke');
      break;
    case 'load':
      runTest('load');
      break;
    case 'stress':
      runTest('stress');
      break;
    case 'spike':
      runTest('spike');
      break;
    case 'volume':
      runTest('volume');
      break;
    case 'suite':
      runTestSuite();
      break;
    case 'benchmark':
      runPerformanceBenchmark();
      break;
    case 'generate-data':
      ensureDirectories();
      generateTestData();
      break;
    default:
      console.log(`
FraudGuard 360 Load Testing Runner

Usage: node load-test-runner.js <command>

Commands:
  smoke        Run smoke test (basic functionality)
  load         Run load test (normal expected load)
  stress       Run stress test (beyond normal capacity)
  spike        Run spike test (sudden traffic increases)
  volume       Run volume test (sustained high load)
  suite        Run complete test suite
  benchmark    Run performance benchmark
  generate-data Generate test data files
  help         Show this help message

Environment Variables:
  BASE_URL     Target API base URL (default: http://localhost:8000)

Examples:
  node load-test-runner.js smoke
  BASE_URL=https://api.fraudguard.com node load-test-runner.js suite
      `);
  }
}

if (require.main === module) {
  main();
}

module.exports = {
  runTest,
  runTestSuite,
  runPerformanceBenchmark,
  generateTestData,
  CONFIG
};