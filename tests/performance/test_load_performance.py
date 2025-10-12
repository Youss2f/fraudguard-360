"""
FraudGuard 360 - Performance Testing Suite
Load testing and performance validation for the fraud detection system
"""

import asyncio
import json
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import statistics
import uuid

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    response_times: List[float]
    success_rate: float
    throughput: float
    error_rate: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float

@dataclass
class LoadTestConfig:
    """Load test configuration"""
    concurrent_users: int
    test_duration: int  # seconds
    ramp_up_time: int  # seconds
    requests_per_second: int
    endpoint_url: str
    test_data: List[Dict]

class FraudGuardPerformanceTest:
    """Performance testing suite for FraudGuard 360"""
    
    def __init__(self):
        self.test_session_id = str(uuid.uuid4())
        self.results = {}
        self.test_data = []
        self.base_urls = {
            'api_gateway': os.getenv('API_GATEWAY_URL', 'http://localhost:8000'),
            'ml_service': os.getenv('ML_SERVICE_URL', 'http://localhost:8003'),
            'frontend': os.getenv('FRONTEND_URL', 'http://localhost:3000')
        }
        
    def generate_test_data(self, num_records: int = 1000) -> List[Dict]:
        """Generate synthetic CDR data for performance testing"""
        logger.info(f"Generating {num_records} test CDR records...")
        
        test_records = []
        
        for i in range(num_records):
            # Generate mix of normal and fraudulent patterns
            is_fraud = np.random.choice([True, False], p=[0.1, 0.9])  # 10% fraud rate
            
            if is_fraud:
                record = {
                    'user_id': f"perf_fraud_user_{i:05d}_{self.test_session_id[:8]}",
                    'call_duration': float(np.random.lognormal(5, 1)),
                    'call_cost': float(np.random.gamma(5, 2)),
                    'calls_per_day': int(np.random.poisson(80)),
                    'unique_numbers_called': int(np.random.poisson(15)),
                    'international_calls': 1,
                    'night_calls': 1,
                    'weekend_calls': int(np.random.choice([0, 1])),
                    'call_frequency_variance': float(np.random.exponential(8)),
                    'location_changes': int(np.random.poisson(12)),
                    'avg_call_gap': float(np.random.exponential(0.5)),
                    'network_connections': int(np.random.poisson(25)),
                    'suspicious_patterns': 1,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                record = {
                    'user_id': f"perf_normal_user_{i:05d}_{self.test_session_id[:8]}",
                    'call_duration': float(np.random.lognormal(4, 0.5)),
                    'call_cost': float(np.random.gamma(2, 0.3)),
                    'calls_per_day': int(np.random.poisson(12)),
                    'unique_numbers_called': int(np.random.poisson(6)),
                    'international_calls': 0,
                    'night_calls': int(np.random.choice([0, 1], p=[0.8, 0.2])),
                    'weekend_calls': int(np.random.choice([0, 1], p=[0.7, 0.3])),
                    'call_frequency_variance': float(np.random.exponential(1.5)),
                    'location_changes': int(np.random.poisson(1)),
                    'avg_call_gap': float(np.random.exponential(2)),
                    'network_connections': int(np.random.poisson(8)),
                    'suspicious_patterns': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            test_records.append(record)
        
        self.test_data = test_records
        logger.info(f"Generated {len(test_records)} test records")
        return test_records
    
    def execute_request(self, url: str, data: Dict, timeout: int = 30) -> Tuple[bool, float, str]:
        """Execute a single HTTP request and measure performance"""
        start_time = time.time()
        
        try:
            response = requests.post(url, json=data, timeout=timeout)
            end_time = time.time()
            
            response_time = end_time - start_time
            success = response.status_code in [200, 201, 202]
            error_msg = "" if success else f"HTTP {response.status_code}: {response.text[:100]}"
            
            return success, response_time, error_msg
            
        except requests.exceptions.Timeout:
            end_time = time.time()
            return False, end_time - start_time, "Request timeout"
            
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            return False, end_time - start_time, str(e)[:100]
    
    def run_load_test(self, config: LoadTestConfig) -> PerformanceMetrics:
        """Run load test with specified configuration"""
        logger.info(f"Starting load test: {config.concurrent_users} users, {config.test_duration}s duration")
        
        results = {
            'response_times': [],
            'successes': 0,
            'failures': 0,
            'errors': []
        }
        
        # Thread-safe results collection
        results_lock = threading.Lock()
        
        def worker_thread(worker_id: int, test_data_chunk: List[Dict]):
            """Worker thread for load testing"""
            thread_results = {
                'response_times': [],
                'successes': 0,
                'failures': 0,
                'errors': []
            }
            
            start_time = time.time()
            request_count = 0
            
            while time.time() - start_time < config.test_duration:
                # Select random test data
                test_record = np.random.choice(test_data_chunk)
                
                # Execute request
                success, response_time, error_msg = self.execute_request(
                    config.endpoint_url, 
                    test_record
                )
                
                thread_results['response_times'].append(response_time)
                
                if success:
                    thread_results['successes'] += 1
                else:
                    thread_results['failures'] += 1
                    thread_results['errors'].append(error_msg)
                
                request_count += 1
                
                # Control request rate (simple rate limiting)
                if config.requests_per_second > 0:
                    expected_time = start_time + (request_count / config.requests_per_second)
                    current_time = time.time()
                    if current_time < expected_time:
                        time.sleep(expected_time - current_time)
            
            # Merge results thread-safely
            with results_lock:
                results['response_times'].extend(thread_results['response_times'])
                results['successes'] += thread_results['successes']
                results['failures'] += thread_results['failures']
                results['errors'].extend(thread_results['errors'])
        
        # Prepare test data chunks for each thread
        data_per_thread = len(self.test_data) // config.concurrent_users
        test_data_chunks = [
            self.test_data[i * data_per_thread:(i + 1) * data_per_thread]
            for i in range(config.concurrent_users)
        ]
        
        # Start load test
        test_start_time = time.time()
        threads = []
        
        for i in range(config.concurrent_users):
            thread = threading.Thread(
                target=worker_thread,
                args=(i, test_data_chunks[i] if i < len(test_data_chunks) else self.test_data)
            )
            threads.append(thread)
            thread.start()
            
            # Gradual ramp-up
            if config.ramp_up_time > 0:
                time.sleep(config.ramp_up_time / config.concurrent_users)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        test_end_time = time.time()
        total_test_time = test_end_time - test_start_time
        
        # Calculate metrics
        total_requests = results['successes'] + results['failures']
        success_rate = results['successes'] / total_requests if total_requests > 0 else 0
        error_rate = results['failures'] / total_requests if total_requests > 0 else 0
        throughput = total_requests / total_test_time
        
        response_times = results['response_times']
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = np.percentile(response_times, 95) if response_times else 0
        p99_response_time = np.percentile(response_times, 99) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        metrics = PerformanceMetrics(
            response_times=response_times,
            success_rate=success_rate,
            throughput=throughput,
            error_rate=error_rate,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time
        )
        
        logger.info(f"Load test completed: {total_requests} requests, {success_rate:.2%} success rate")
        logger.info(f"Throughput: {throughput:.2f} req/s, Avg response time: {avg_response_time:.3f}s")
        
        return metrics
    
    def test_ml_service_performance(self) -> Dict[str, PerformanceMetrics]:
        """Test ML service performance with various load levels"""
        logger.info("Testing ML service performance...")
        
        ml_service_url = f"{self.base_urls['ml_service']}/predict"
        test_configs = [
            LoadTestConfig(1, 60, 0, 10, ml_service_url, self.test_data),      # Light load
            LoadTestConfig(5, 60, 10, 50, ml_service_url, self.test_data),     # Medium load
            LoadTestConfig(10, 60, 20, 100, ml_service_url, self.test_data),   # Heavy load
            LoadTestConfig(20, 60, 30, 200, ml_service_url, self.test_data),   # Stress test
        ]
        
        results = {}
        
        for i, config in enumerate(test_configs):
            test_name = f"ml_service_load_{i+1}"
            logger.info(f"Running {test_name}: {config.concurrent_users} users")
            
            metrics = self.run_load_test(config)
            results[test_name] = metrics
            
            # Brief pause between tests
            time.sleep(10)
        
        return results
    
    def test_api_gateway_performance(self) -> Dict[str, PerformanceMetrics]:
        """Test API Gateway performance"""
        logger.info("Testing API Gateway performance...")
        
        api_gateway_url = f"{self.base_urls['api_gateway']}/cdr/ingest"
        test_configs = [
            LoadTestConfig(5, 60, 10, 25, api_gateway_url, self.test_data),    # Light load
            LoadTestConfig(15, 60, 20, 75, api_gateway_url, self.test_data),   # Medium load
            LoadTestConfig(25, 60, 30, 125, api_gateway_url, self.test_data),  # Heavy load
        ]
        
        results = {}
        
        for i, config in enumerate(test_configs):
            test_name = f"api_gateway_load_{i+1}"
            logger.info(f"Running {test_name}: {config.concurrent_users} users")
            
            metrics = self.run_load_test(config)
            results[test_name] = metrics
            
            # Brief pause between tests
            time.sleep(10)
        
        return results
    
    def test_batch_processing_performance(self) -> Dict[str, Any]:
        """Test batch processing performance"""
        logger.info("Testing batch processing performance...")
        
        batch_sizes = [10, 50, 100, 500, 1000]
        batch_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            batch_data = {
                'records': self.test_data[:batch_size],
                'batch_id': f"perf_test_batch_{batch_size}_{self.test_session_id[:8]}"
            }
            
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_urls['ml_service']}/predict/batch",
                json=batch_data,
                timeout=300  # 5 minute timeout for large batches
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code in [200, 202]:
                batch_results[batch_size] = {
                    'processing_time': processing_time,
                    'throughput': batch_size / processing_time,
                    'success': True
                }
                logger.info(f"Batch {batch_size}: {processing_time:.2f}s, {batch_size/processing_time:.2f} records/s")
            else:
                batch_results[batch_size] = {
                    'processing_time': processing_time,
                    'throughput': 0,
                    'success': False,
                    'error': response.text[:100]
                }
                logger.error(f"Batch {batch_size} failed: {response.status_code}")
        
        return batch_results
    
    def test_concurrent_batch_processing(self) -> Dict[str, Any]:
        """Test concurrent batch processing"""
        logger.info("Testing concurrent batch processing...")
        
        def submit_batch(batch_id: str, batch_data: List[Dict]) -> Tuple[str, float, bool]:
            """Submit a batch and measure processing time"""
            start_time = time.time()
            
            payload = {
                'records': batch_data,
                'batch_id': batch_id
            }
            
            try:
                response = requests.post(
                    f"{self.base_urls['ml_service']}/predict/batch",
                    json=payload,
                    timeout=300
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                success = response.status_code in [200, 202]
                
                return batch_id, processing_time, success
                
            except Exception as e:
                end_time = time.time()
                return batch_id, end_time - start_time, False
        
        # Create multiple batches for concurrent processing
        batch_size = 100
        num_batches = 5
        batches = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(self.test_data))
            batch_data = self.test_data[start_idx:end_idx]
            batch_id = f"concurrent_batch_{i}_{self.test_session_id[:8]}"
            batches.append((batch_id, batch_data))
        
        # Submit batches concurrently
        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            futures = [
                executor.submit(submit_batch, batch_id, batch_data)
                for batch_id, batch_data in batches
            ]
            
            results = {}
            total_start_time = time.time()
            
            for future in as_completed(futures):
                batch_id, processing_time, success = future.result()
                results[batch_id] = {
                    'processing_time': processing_time,
                    'success': success
                }
            
            total_end_time = time.time()
            total_processing_time = total_end_time - total_start_time
        
        # Calculate metrics
        successful_batches = sum(1 for r in results.values() if r['success'])
        total_records_processed = successful_batches * batch_size
        overall_throughput = total_records_processed / total_processing_time
        
        concurrent_results = {
            'total_batches': num_batches,
            'successful_batches': successful_batches,
            'total_processing_time': total_processing_time,
            'overall_throughput': overall_throughput,
            'batch_results': results
        }
        
        logger.info(f"Concurrent batches: {successful_batches}/{num_batches} successful")
        logger.info(f"Overall throughput: {overall_throughput:.2f} records/s")
        
        return concurrent_results
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive performance report"""
        logger.info("Generating performance report...")
        
        report = []
        report.append("# FraudGuard 360 Performance Test Report")
        report.append(f"Test Session ID: {self.test_session_id}")
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        # ML Service Performance
        if 'ml_service' in results:
            report.append("### ML Service Performance")
            report.append("")
            
            for test_name, metrics in results['ml_service'].items():
                report.append(f"**{test_name}:**")
                report.append(f"- Success Rate: {metrics.success_rate:.2%}")
                report.append(f"- Throughput: {metrics.throughput:.2f} requests/second")
                report.append(f"- Average Response Time: {metrics.avg_response_time:.3f}s")
                report.append(f"- 95th Percentile: {metrics.p95_response_time:.3f}s")
                report.append(f"- 99th Percentile: {metrics.p99_response_time:.3f}s")
                report.append("")
        
        # API Gateway Performance
        if 'api_gateway' in results:
            report.append("### API Gateway Performance")
            report.append("")
            
            for test_name, metrics in results['api_gateway'].items():
                report.append(f"**{test_name}:**")
                report.append(f"- Success Rate: {metrics.success_rate:.2%}")
                report.append(f"- Throughput: {metrics.throughput:.2f} requests/second")
                report.append(f"- Average Response Time: {metrics.avg_response_time:.3f}s")
                report.append(f"- 95th Percentile: {metrics.p95_response_time:.3f}s")
                report.append("")
        
        # Batch Processing Performance
        if 'batch_processing' in results:
            report.append("### Batch Processing Performance")
            report.append("")
            
            batch_results = results['batch_processing']
            report.append("| Batch Size | Processing Time (s) | Throughput (records/s) | Success |")
            report.append("|------------|-------------------|----------------------|---------|")
            
            for batch_size, metrics in batch_results.items():
                success_icon = "✅" if metrics['success'] else "❌"
                report.append(f"| {batch_size} | {metrics['processing_time']:.2f} | {metrics['throughput']:.2f} | {success_icon} |")
            
            report.append("")
        
        # Concurrent Processing
        if 'concurrent_processing' in results:
            report.append("### Concurrent Batch Processing")
            report.append("")
            
            concurrent_results = results['concurrent_processing']
            report.append(f"- Total Batches: {concurrent_results['total_batches']}")
            report.append(f"- Successful Batches: {concurrent_results['successful_batches']}")
            report.append(f"- Success Rate: {concurrent_results['successful_batches']/concurrent_results['total_batches']:.2%}")
            report.append(f"- Overall Throughput: {concurrent_results['overall_throughput']:.2f} records/s")
            report.append("")
        
        # Performance Recommendations
        report.append("## Performance Recommendations")
        report.append("")
        
        recommendations = []
        
        # Analyze results and provide recommendations
        if 'ml_service' in results:
            ml_results = results['ml_service']
            
            # Check if any test had low success rate
            for test_name, metrics in ml_results.items():
                if metrics.success_rate < 0.95:
                    recommendations.append(f"- ML Service {test_name} has low success rate ({metrics.success_rate:.2%}). Consider scaling up resources or optimizing the model inference pipeline.")
                
                if metrics.avg_response_time > 2.0:
                    recommendations.append(f"- ML Service {test_name} has high average response time ({metrics.avg_response_time:.3f}s). Consider model optimization or caching strategies.")
        
        if not recommendations:
            recommendations.append("- All performance metrics are within acceptable ranges.")
            recommendations.append("- Consider implementing automated performance monitoring and alerting.")
            recommendations.append("- Regular performance testing should be conducted as the system scales.")
        
        for recommendation in recommendations:
            report.append(recommendation)
        
        report.append("")
        report.append("---")
        report.append("*Report generated by FraudGuard 360 Performance Testing Suite*")
        
        return "\n".join(report)
    
    def save_performance_graphs(self, results: Dict[str, Any], output_dir: str = "performance_reports"):
        """Generate and save performance visualization graphs"""
        logger.info("Generating performance graphs...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        
        # ML Service Response Time Distribution
        if 'ml_service' in results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ML Service Performance Analysis', fontsize=16)
            
            # Response time distribution
            for i, (test_name, metrics) in enumerate(results['ml_service'].items()):
                ax = axes[i//2, i%2]
                ax.hist(metrics.response_times, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(f'{test_name} - Response Time Distribution')
                ax.set_xlabel('Response Time (seconds)')
                ax.set_ylabel('Frequency')
                ax.axvline(metrics.avg_response_time, color='red', linestyle='--', 
                          label=f'Avg: {metrics.avg_response_time:.3f}s')
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/ml_service_response_times.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Throughput comparison
        if 'ml_service' in results and 'api_gateway' in results:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            services = ['ML Service', 'API Gateway']
            throughputs = []
            
            # Calculate average throughput for each service
            ml_throughput = np.mean([metrics.throughput for metrics in results['ml_service'].values()])
            api_throughput = np.mean([metrics.throughput for metrics in results['api_gateway'].values()])
            
            throughputs = [ml_throughput, api_throughput]
            
            bars = ax.bar(services, throughputs, color=['skyblue', 'lightcoral'])
            ax.set_title('Average Throughput Comparison')
            ax.set_ylabel('Requests per Second')
            
            # Add value labels on bars
            for bar, throughput in zip(bars, throughputs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{throughput:.1f}',
                       ha='center', va='bottom')
            
            plt.savefig(f'{output_dir}/throughput_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Batch processing performance
        if 'batch_processing' in results:
            batch_results = results['batch_processing']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            batch_sizes = list(batch_results.keys())
            processing_times = [batch_results[size]['processing_time'] for size in batch_sizes]
            throughputs = [batch_results[size]['throughput'] for size in batch_sizes]
            
            # Processing time vs batch size
            ax1.plot(batch_sizes, processing_times, marker='o', linewidth=2, markersize=8)
            ax1.set_title('Batch Processing Time vs Batch Size')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Processing Time (seconds)')
            ax1.grid(True, alpha=0.3)
            
            # Throughput vs batch size
            ax2.plot(batch_sizes, throughputs, marker='s', linewidth=2, markersize=8, color='orange')
            ax2.set_title('Batch Processing Throughput vs Batch Size')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Throughput (records/second)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/batch_processing_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Performance graphs saved to {output_dir}/")
    
    def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        logger.info("Starting comprehensive performance test suite...")
        
        # Generate test data
        self.generate_test_data(1000)
        
        # Run all performance tests
        results = {}
        
        # Test ML Service
        try:
            results['ml_service'] = self.test_ml_service_performance()
        except Exception as e:
            logger.error(f"ML Service performance test failed: {e}")
            results['ml_service'] = {}
        
        # Test API Gateway
        try:
            results['api_gateway'] = self.test_api_gateway_performance()
        except Exception as e:
            logger.error(f"API Gateway performance test failed: {e}")
            results['api_gateway'] = {}
        
        # Test Batch Processing
        try:
            results['batch_processing'] = self.test_batch_processing_performance()
        except Exception as e:
            logger.error(f"Batch processing performance test failed: {e}")
            results['batch_processing'] = {}
        
        # Test Concurrent Processing
        try:
            results['concurrent_processing'] = self.test_concurrent_batch_processing()
        except Exception as e:
            logger.error(f"Concurrent processing performance test failed: {e}")
            results['concurrent_processing'] = {}
        
        # Generate reports
        report = self.generate_performance_report(results)
        
        # Save report to file
        os.makedirs("performance_reports", exist_ok=True)
        report_filename = f"performance_reports/performance_report_{self.test_session_id[:8]}.md"
        
        with open(report_filename, 'w') as f:
            f.write(report)
        
        logger.info(f"Performance report saved to {report_filename}")
        
        # Generate graphs
        try:
            self.save_performance_graphs(results)
        except Exception as e:
            logger.error(f"Failed to generate performance graphs: {e}")
        
        return results

if __name__ == "__main__":
    # Run comprehensive performance test
    perf_test = FraudGuardPerformanceTest()
    
    try:
        results = perf_test.run_comprehensive_performance_test()
        logger.info("=== Performance Test Suite Completed ===")
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE TEST SUMMARY")
        print("="*60)
        
        if 'ml_service' in results and results['ml_service']:
            print("\nML Service Performance:")
            for test_name, metrics in results['ml_service'].items():
                print(f"  {test_name}: {metrics.success_rate:.2%} success, {metrics.throughput:.1f} req/s")
        
        if 'api_gateway' in results and results['api_gateway']:
            print("\nAPI Gateway Performance:")
            for test_name, metrics in results['api_gateway'].items():
                print(f"  {test_name}: {metrics.success_rate:.2%} success, {metrics.throughput:.1f} req/s")
        
        if 'batch_processing' in results and results['batch_processing']:
            print("\nBatch Processing Performance:")
            for batch_size, metrics in results['batch_processing'].items():
                status = "✅" if metrics['success'] else "❌"
                print(f"  Batch {batch_size}: {metrics['throughput']:.1f} records/s {status}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Performance test suite failed: {e}")
        raise