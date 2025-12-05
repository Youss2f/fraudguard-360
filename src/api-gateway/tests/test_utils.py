"""
Unit Tests for API Gateway
===========================

Tests for utility functions and core logic in the API Gateway.
"""

import pytest
from datetime import datetime
import json


def generate_transaction_id(user_id: str) -> str:
    """Utility function to generate transaction IDs"""
    import time
    return f"TXN_{int(time.time() * 1000)}_{user_id}"


def validate_amount(amount: float) -> bool:
    """Utility function to validate transaction amounts"""
    return amount >= 0


def enrich_transaction_data(transaction: dict) -> dict:
    """Utility function to enrich transaction data"""
    enriched = transaction.copy()
    enriched['timestamp'] = datetime.utcnow().isoformat()
    enriched['currency'] = 'USD'
    enriched['status'] = 'pending'
    return enriched


class TestTransactionUtils:
    """Test suite for transaction utility functions"""
    
    def test_generate_transaction_id(self):
        """Test transaction ID generation"""
        user_id = "USR_123"
        transaction_id = generate_transaction_id(user_id)
        
        assert transaction_id.startswith("TXN_")
        assert transaction_id.endswith(f"_{user_id}")
        assert len(transaction_id) > 10
    
    def test_generate_transaction_id_unique(self):
        """Test that transaction IDs are unique"""
        import time
        user_id = "USR_123"
        id1 = generate_transaction_id(user_id)
        time.sleep(0.002)  # Ensure time difference for uniqueness
        id2 = generate_transaction_id(user_id)
        
        assert id1 != id2
    
    def test_validate_amount_positive(self):
        """Test amount validation with positive number"""
        assert validate_amount(100.0) is True
        assert validate_amount(0.01) is True
        assert validate_amount(1000000.0) is True
    
    def test_validate_amount_zero(self):
        """Test amount validation with zero"""
        assert validate_amount(0) is True
    
    def test_validate_amount_negative(self):
        """Test amount validation with negative number"""
        assert validate_amount(-1.0) is False
        assert validate_amount(-100.0) is False
    
    def test_enrich_transaction_data(self):
        """Test transaction data enrichment"""
        transaction = {
            "user_id": "USR_123",
            "amount": 250.0,
            "location": "New York"
        }
        
        enriched = enrich_transaction_data(transaction)
        
        assert "timestamp" in enriched
        assert "currency" in enriched
        assert "status" in enriched
        assert enriched["currency"] == "USD"
        assert enriched["status"] == "pending"
        assert enriched["user_id"] == transaction["user_id"]
        assert enriched["amount"] == transaction["amount"]
    
    def test_enrich_transaction_preserves_original(self):
        """Test that enrichment doesn't modify original transaction"""
        transaction = {
            "user_id": "USR_123",
            "amount": 250.0
        }
        
        original_keys = set(transaction.keys())
        enriched = enrich_transaction_data(transaction)
        
        # Original should be unchanged
        assert set(transaction.keys()) == original_keys
        
        # Enriched should have additional keys
        assert len(enriched) > len(transaction)


class TestTransactionValidation:
    """Test suite for transaction validation logic"""
    
    @pytest.mark.parametrize("amount,expected", [
        (0, True),
        (1, True),
        (100.50, True),
        (1000000, True),
        (-0.01, False),
        (-100, False),
    ])
    def test_amount_validation_parametrized(self, amount, expected):
        """Parametrized test for amount validation"""
        assert validate_amount(amount) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
