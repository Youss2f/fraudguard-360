# Performance Test Results Directory

This directory contains performance test results from Locust load testing.

## Structure

- `results/` - Test results organized by timestamp
- `reports/` - Generated HTML reports  
- `charts/` - Performance charts and graphs
- `raw/` - Raw data files (CSV format)

## Result Files

Each test run generates:
- `summary.json` - Test summary statistics
- `requests.csv` - Individual request data
- `failures.csv` - Failed request details
- `stats.html` - HTML performance report

## Metrics Tracked

- **Response Time**: Average, median, 95th percentile
- **Throughput**: Requests per second
- **Error Rate**: Percentage of failed requests
- **Concurrent Users**: Number of simulated users
- **Success Rate**: Percentage of successful requests

## Performance Targets

Based on PRD requirements:
- **Response Time**: < 200ms for 95% of requests
- **Throughput**: 5,000-25,000 TPS
- **Availability**: 99.9% uptime
- **Error Rate**: < 0.1%

## Running Tests

```bash
# Install requirements
pip install -r requirements.txt

# Run load test
locust --headless --users 100 --spawn-rate 10 --run-time 5m --host https://api.fraudguard360.com

# Generate report
locust --headless --users 100 --spawn-rate 10 --run-time 5m --host https://api.fraudguard360.com --html results/report.html
```