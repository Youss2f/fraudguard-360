# Load Testing Configuration and Scripts
# K6-based performance testing for FraudGuard 360

# Run smoke test
smoke-test:
	k6 run --env BASE_URL=http://localhost:8000 --env TEST_TYPE=smoke fraud-detection-load-test.js

# Run load test
load-test:
	k6 run --env BASE_URL=http://localhost:8000 --env TEST_TYPE=load fraud-detection-load-test.js

# Run stress test
stress-test:
	k6 run --env BASE_URL=http://localhost:8000 --env TEST_TYPE=stress fraud-detection-load-test.js

# Run spike test
spike-test:
	k6 run --env BASE_URL=http://localhost:8000 --env TEST_TYPE=spike fraud-detection-load-test.js

# Run volume test
volume-test:
	k6 run --env BASE_URL=http://localhost:8000 --env TEST_TYPE=volume fraud-detection-load-test.js

# Run complete test suite
test-suite:
	node load-test-runner.js suite

# Generate performance benchmark
benchmark:
	node load-test-runner.js benchmark

# Generate test data
generate-data:
	node load-test-runner.js generate-data

# Clean results directory
clean:
	rm -rf results/*
	rm -rf test-data/*

# Run with custom URL
test-remote:
	BASE_URL=https://api.fraudguard.com k6 run fraud-detection-load-test.js

# View results with k6 cloud (requires k6 cloud account)
cloud-test:
	k6 cloud fraud-detection-load-test.js

# Docker-based testing
docker-test:
	docker run --rm -i -v $(PWD):/scripts -e BASE_URL=http://host.docker.internal:8000 loadimpact/k6 run /scripts/fraud-detection-load-test.js

.PHONY: smoke-test load-test stress-test spike-test volume-test test-suite benchmark generate-data clean test-remote cloud-test docker-test