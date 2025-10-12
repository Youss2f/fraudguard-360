"""
Data generator for FraudGuard 360 ML service.
Generates synthetic CDR and user data for testing and development.
"""

import random
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np


class FraudDataGenerator:
    """Generate synthetic telecom fraud data for testing."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Country codes for international calls
        self.country_codes = ['US', 'CA', 'UK', 'DE', 'FR', 'IT', 'ES', 'JP', 'AU', 'BR']
        
        # Premium rate prefixes
        self.premium_prefixes = ['900', '976', '970']
        
        # Device IMEI patterns
        self.imei_patterns = ['IMEI_{}_{:03d}' for _ in range(5)]
        
        # Location patterns
        self.locations = [f'Location_{i:02d}' for i in range(20)]
    
    def generate_user_id(self) -> str:
        """Generate a user ID."""
        return f"user_{random.randint(100000, 999999)}"
    
    def generate_phone_number(self, is_premium: bool = False) -> str:
        """Generate a phone number."""
        if is_premium:
            prefix = random.choice(self.premium_prefixes)
            return f"{prefix}{random.randint(1000000, 9999999)}"
        else:
            return f"{random.randint(2000000000, 9999999999)}"
    
    def generate_timestamp(self, days_back: int = 30) -> str:
        """Generate a random timestamp within the last N days."""
        base_time = datetime.now() - timedelta(days=days_back)
        random_time = base_time + timedelta(
            days=random.uniform(0, days_back),
            hours=random.uniform(0, 24),
            minutes=random.uniform(0, 60)
        )
        return random_time.isoformat()
    
    def is_night_time(self, timestamp: str) -> bool:
        """Check if timestamp is during night hours (10PM - 6AM)."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            return hour >= 22 or hour <= 6
        except:
            return False
    
    def generate_cdr(self, user_id: str = None, fraud_probability: float = 0.0) -> Dict[str, Any]:
        """
        Generate a single CDR record.
        
        Args:
            user_id: Specific user ID, or None for random
            fraud_probability: Probability that this CDR exhibits fraud patterns
            
        Returns:
            CDR dictionary
        """
        is_fraudulent = random.random() < fraud_probability
        
        caller_id = user_id or self.generate_user_id()
        
        # Fraudulent calls have different patterns
        if is_fraudulent:
            # More likely to be international
            is_international = random.random() < 0.6
            # More likely to be premium rate
            is_premium = random.random() < 0.3
            # Shorter duration for SIM box fraud
            duration = random.randint(10, 120)
            # Higher cost
            cost_multiplier = random.uniform(2.0, 5.0)
        else:
            is_international = random.random() < 0.1
            is_premium = random.random() < 0.02
            duration = random.randint(30, 1800)
            cost_multiplier = 1.0
        
        callee_id = self.generate_phone_number(is_premium)
        call_type = random.choice(['voice', 'sms', 'data'])
        
        if call_type == 'sms':
            duration = 0
            bytes_transmitted = None
        elif call_type == 'data':
            duration = random.randint(60, 3600)
            bytes_transmitted = random.randint(1000, 1000000)
        else:
            bytes_transmitted = None
        
        timestamp = self.generate_timestamp()
        
        cdr = {
            'id': str(uuid.uuid4()),
            'caller_id': caller_id,
            'callee_id': callee_id,
            'call_type': call_type,
            'start_time': timestamp,
            'end_time': None,  # Could calculate from start_time + duration
            'duration': duration if call_type != 'sms' else None,
            'bytes_transmitted': bytes_transmitted,
            'location_caller': random.choice(self.locations),
            'location_callee': random.choice(self.locations),
            'tower_id': f"tower_{random.randint(1, 100)}",
            'device_imei': random.choice(self.imei_patterns).format(
                random.choice(['Samsung', 'iPhone', 'Huawei', 'Xiaomi']),
                random.randint(1, 999)
            ),
            'cost': round(duration * 0.01 * cost_multiplier, 2) if duration else 0.1,
            'country_code': random.choice(self.country_codes) if is_international else 'US'
        }
        
        return cdr
    
    def generate_user_profile(self, user_id: str, is_fraudster: bool = False) -> Dict[str, Any]:
        """
        Generate a user profile with realistic features.
        
        Args:
            user_id: User identifier
            is_fraudster: Whether this user should exhibit fraud patterns
            
        Returns:
            User profile dictionary
        """
        if is_fraudster:
            # Fraudulent users have different behavioral patterns
            call_count = random.randint(500, 2000)
            sms_count = random.randint(100, 500)
            international_calls = random.randint(50, 200)
            premium_rate_calls = random.randint(5, 30)
            device_diversity = random.randint(2, 5)
            location_diversity = random.randint(5, 15)
            account_age_days = random.randint(10, 365)  # Newer accounts
        else:
            # Normal users
            call_count = random.randint(50, 500)
            sms_count = random.randint(20, 200)
            international_calls = random.randint(0, 20)
            premium_rate_calls = random.randint(0, 2)
            device_diversity = 1
            location_diversity = random.randint(1, 5)
            account_age_days = random.randint(365, 3650)  # Older accounts
        
        total_duration = call_count * random.randint(60, 300)
        total_cost = call_count * random.uniform(0.05, 0.5)
        unique_callees = min(call_count, random.randint(10, 100))
        
        profile = {
            'user_id': user_id,
            'phone_number': self.generate_phone_number(),
            'name': f'User_{user_id.split("_")[1]}',
            'registration_date': (datetime.now() - timedelta(days=account_age_days)).isoformat(),
            'account_status': 'active',
            'plan_type': random.choice(['basic', 'premium', 'unlimited']),
            'risk_score': random.uniform(0.7, 0.95) if is_fraudster else random.uniform(0.0, 0.3),
            
            # Behavioral features
            'call_count': call_count,
            'sms_count': sms_count,
            'data_sessions': random.randint(10, 100),
            'total_duration': total_duration,
            'total_cost': round(total_cost, 2),
            'unique_callees': unique_callees,
            'international_calls': international_calls,
            'premium_rate_calls': premium_rate_calls,
            'night_calls_ratio': random.uniform(0.3, 0.8) if is_fraudster else random.uniform(0.0, 0.2),
            'call_frequency': call_count / 24,  # Calls per hour
            'avg_call_duration': total_duration / call_count,
            'location_diversity': location_diversity,
            'device_diversity': device_diversity,
            'account_age_days': account_age_days,
            'plan_cost': random.uniform(20, 200)
        }
        
        return profile
    
    def generate_fraud_scenario(self, scenario_type: str = 'sim_box', 
                              num_users: int = 10) -> List[Dict[str, Any]]:
        """
        Generate a specific fraud scenario with multiple users and CDRs.
        
        Args:
            scenario_type: Type of fraud scenario
            num_users: Number of users involved
            
        Returns:
            List of CDRs representing the fraud scenario
        """
        cdrs = []
        
        if scenario_type == 'sim_box':
            # SIM box fraud: High volume, short international calls
            main_fraudster = self.generate_user_id()
            
            for _ in range(500):  # High volume
                cdr = self.generate_cdr(main_fraudster, fraud_probability=1.0)
                # Modify for SIM box characteristics
                cdr['call_type'] = 'voice'
                cdr['duration'] = random.randint(10, 60)  # Short calls
                cdr['country_code'] = random.choice(['IN', 'PK', 'BD', 'NG'])  # Common targets
                cdr['cost'] = random.uniform(0.5, 2.0)  # Higher international rates
                cdrs.append(cdr)
        
        elif scenario_type == 'subscription_fraud':
            # Multiple accounts with similar patterns
            fraudster_group = [self.generate_user_id() for _ in range(num_users)]
            
            for user in fraudster_group:
                for _ in range(100):
                    cdr = self.generate_cdr(user, fraud_probability=0.8)
                    cdrs.append(cdr)
        
        elif scenario_type == 'premium_rate_fraud':
            # Calls to premium rate numbers
            fraudster = self.generate_user_id()
            
            for _ in range(200):
                cdr = self.generate_cdr(fraudster, fraud_probability=1.0)
                cdr['callee_id'] = self.generate_phone_number(is_premium=True)
                cdr['cost'] = random.uniform(5.0, 50.0)  # Very expensive
                cdrs.append(cdr)
        
        return cdrs
    
    def generate_training_dataset(self, num_users: int = 1000, 
                                cdrs_per_user: int = 100,
                                fraud_rate: float = 0.1) -> Dict[str, Any]:
        """
        Generate a complete training dataset.
        
        Args:
            num_users: Total number of users
            cdrs_per_user: Average CDRs per user
            fraud_rate: Fraction of users who are fraudsters
            
        Returns:
            Dictionary with users, CDRs, and labels
        """
        users = []
        cdrs = []
        labels = []
        
        num_fraudsters = int(num_users * fraud_rate)
        
        for i in range(num_users):
            is_fraudster = i < num_fraudsters
            user_id = f"user_{i:06d}"
            
            # Generate user profile
            user_profile = self.generate_user_profile(user_id, is_fraudster)
            users.append(user_profile)
            labels.append(1 if is_fraudster else 0)
            
            # Generate CDRs for this user
            num_user_cdrs = random.randint(
                cdrs_per_user // 2, 
                cdrs_per_user * 2 if is_fraudster else cdrs_per_user
            )
            
            for _ in range(num_user_cdrs):
                fraud_prob = 0.8 if is_fraudster else 0.02
                cdr = self.generate_cdr(user_id, fraud_prob)
                cdrs.append(cdr)
        
        return {
            'users': users,
            'cdrs': cdrs,
            'labels': labels,
            'metadata': {
                'num_users': num_users,
                'num_cdrs': len(cdrs),
                'fraud_rate': fraud_rate,
                'generated_at': datetime.now().isoformat()
            }
        }


def generate_cdr(num: int = 100) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    generator = FraudDataGenerator()
    return [generator.generate_cdr() for _ in range(num)]


if __name__ == "__main__":
    # Example usage
    generator = FraudDataGenerator()
    
    # Generate some sample CDRs
    print("Sample CDRs:")
    sample_cdrs = generator.generate_cdr()
    print(json.dumps(sample_cdrs, indent=2))
    
    # Generate a fraud scenario
    print("\nSIM Box Fraud Scenario (first 3 CDRs):")
    sim_box_scenario = generator.generate_fraud_scenario('sim_box', 5)
    for cdr in sim_box_scenario[:3]:
        print(json.dumps(cdr, indent=2))
    
    # Generate training dataset sample
    print(f"\nTraining Dataset Sample:")
    dataset = generator.generate_training_dataset(num_users=50, fraud_rate=0.2)
    print(f"Generated {dataset['metadata']['num_users']} users, "
          f"{dataset['metadata']['num_cdrs']} CDRs, "
          f"fraud rate: {dataset['metadata']['fraud_rate']}")
    
    # Save to file
    with open('sample_training_data.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("Sample data saved to 'sample_training_data.json'")
