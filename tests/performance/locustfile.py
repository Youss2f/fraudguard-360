"""
Performance testing suite using Locust for FraudGuard 360°
"""

import json
import random
import time
from locust import HttpUser, task, between
from locust.exception import StopUser


class FraudDetectionUser(HttpUser):
    """Simulates a user performing fraud detection operations"""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        self.token = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate user and get JWT token"""
        login_data = {
            "username": "test_user",
            "password": "test_password"
        }
        
        with self.client.post("/api/v1/auth/login", json=login_data, catch_response=True) as response:
            if response.status_code == 200:
                self.token = response.json().get("access_token")
                self.client.headers.update({"Authorization": f"Bearer {self.token}"})
            else:
                print(f"Authentication failed: {response.status_code}")
                raise StopUser()
    
    @task(5)
    def analyze_transaction(self):
        """Main fraud analysis task - highest weight"""
        transaction_data = {
            "transaction_id": f"txn_{random.randint(100000, 999999)}",
            "amount": random.uniform(10.0, 10000.0),
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "merchant_id": f"merchant_{random.randint(1, 1000)}",
            "customer_id": f"customer_{random.randint(1, 10000)}",
            "timestamp": int(time.time()),
            "location": {
                "country": random.choice(["US", "UK", "DE", "FR", "CA"]),
                "city": random.choice(["New York", "London", "Berlin", "Paris", "Toronto"])
            },
            "payment_method": random.choice(["credit_card", "debit_card", "bank_transfer"]),
            "features": {
                "is_weekend": random.choice([True, False]),
                "hour_of_day": random.randint(0, 23),
                "days_since_last_transaction": random.randint(0, 30),
                "transaction_count_24h": random.randint(1, 20)
            }
        }
        
        with self.client.post("/api/v1/fraud/analyze", 
                            json=transaction_data,
                            catch_response=True,
                            name="fraud_analysis") as response:
            if response.status_code == 200:
                result = response.json()
                if "fraud_score" in result and "decision" in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def get_fraud_patterns(self):
        """Get fraud patterns from graph service"""
        with self.client.get("/api/v1/graph/patterns", 
                           catch_response=True,
                           name="fraud_patterns") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def get_ai_model_info(self):
        """Get AI model information"""
        with self.client.get("/api/v1/ai/model/info",
                           catch_response=True,
                           name="model_info") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def get_customer_profile(self):
        """Get customer risk profile"""
        customer_id = f"customer_{random.randint(1, 10000)}"
        
        with self.client.get(f"/api/v1/graph/customer/{customer_id}/profile",
                           catch_response=True,
                           name="customer_profile") as response:
            if response.status_code in [200, 404]:  # 404 is acceptable for non-existent customers
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def batch_analysis(self):
        """Batch fraud analysis for multiple transactions"""
        transactions = []
        
        for i in range(random.randint(5, 20)):
            transactions.append({
                "transaction_id": f"batch_txn_{i}_{random.randint(100000, 999999)}",
                "amount": random.uniform(10.0, 5000.0),
                "currency": "USD",
                "merchant_id": f"merchant_{random.randint(1, 100)}",
                "customer_id": f"customer_{random.randint(1, 1000)}",
                "timestamp": int(time.time()) - random.randint(0, 3600)
            })
        
        with self.client.post("/api/v1/fraud/analyze/batch",
                            json={"transactions": transactions},
                            catch_response=True,
                            name="batch_analysis") as response:
            if response.status_code == 200:
                result = response.json()
                if len(result.get("results", [])) == len(transactions):
                    response.success()
                else:
                    response.failure("Batch result count mismatch")
            else:
                response.failure(f"HTTP {response.status_code}")


class AdminUser(HttpUser):
    """Simulates admin user operations"""
    
    wait_time = between(5, 15)  # Admins perform actions less frequently
    weight = 1  # Lower weight compared to FraudDetectionUser
    
    def on_start(self):
        """Authenticate as admin"""
        login_data = {
            "username": "admin_user",
            "password": "admin_password"
        }
        
        with self.client.post("/api/v1/auth/login", json=login_data, catch_response=True) as response:
            if response.status_code == 200:
                self.token = response.json().get("access_token")
                self.client.headers.update({"Authorization": f"Bearer {self.token}"})
            else:
                raise StopUser()

    @task(3)
    def get_system_metrics(self):
        """Get system performance metrics"""
        with self.client.get("/api/v1/metrics",
                           catch_response=True,
                           name="system_metrics") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def get_fraud_statistics(self):
        """Get fraud detection statistics"""
        with self.client.get("/api/v1/fraud/statistics",
                           catch_response=True,
                           name="fraud_statistics") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def update_fraud_rules(self):
        """Update fraud detection rules"""
        rule_data = {
            "rule_id": f"rule_{random.randint(1, 100)}",
            "condition": {
                "amount_threshold": random.uniform(1000, 10000),
                "risk_score_threshold": random.uniform(0.7, 0.9)
            },
            "action": random.choice(["flag", "block", "review"])
        }
        
        with self.client.post("/api/v1/fraud/rules",
                            json=rule_data,
                            catch_response=True,
                            name="update_rules") as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class HealthCheckUser(HttpUser):
    """Lightweight user for health checks"""
    
    wait_time = between(10, 30)
    weight = 1

    @task
    def health_check(self):
        """Basic health check"""
        with self.client.get("/health", catch_response=True, name="health_check") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task  
    def api_health_check(self):
        """API health check"""
        with self.client.get("/api/v1/health", catch_response=True, name="api_health_check") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")