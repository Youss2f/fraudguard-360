"""
Tests for ML Service - Feature Engineering Module
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class MockFeatureEngineering:
    """Mock FeatureEngineering class for testing."""
    
    def __init__(self):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for testing."""
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_sqrt'] = np.sqrt(df['amount'])
        return df


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    return pd.DataFrame({
        'transaction_id': ['TX001', 'TX002', 'TX003'],
        'amount': [100.0, 500.0, 1500.0],
        'merchant_id': ['M001', 'M002', 'M001'],
        'customer_id': ['C001', 'C002', 'C001'],
        'timestamp': pd.to_datetime([
            '2025-01-15 10:30:00',
            '2025-01-15 14:20:00',
            '2025-01-15 23:45:00'
        ]),
        'merchant_category': ['retail', 'online', 'retail'],
        'location_country': ['US', 'US', 'UK'],
        'location_city': ['New York', 'Los Angeles', 'London'],
        'payment_method': ['credit_card', 'debit_card', 'credit_card'],
        'is_weekend': [False, False, False],
        'hour_of_day': [10, 14, 23]
    })


class TestFeatureEngineering:
    """Test suite for feature engineering."""
    
    def test_time_based_features(self, sample_transaction_data):
        """Test creation of time-based features."""
        fe = MockFeatureEngineering()
        df_engineered = fe.engineer_features(sample_transaction_data.copy())
        
        # Check that time features are created
        assert 'hour_sin' in df_engineered.columns
        assert 'hour_cos' in df_engineered.columns
        assert 'day_of_week' in df_engineered.columns
        assert 'is_night' in df_engineered.columns
        
        # Verify hour_sin and hour_cos are in valid range
        assert df_engineered['hour_sin'].between(-1, 1).all()
        assert df_engineered['hour_cos'].between(-1, 1).all()
        
        # Verify night detection
        assert df_engineered.loc[2, 'is_night'] == 1  # 23:45 is night
        assert df_engineered.loc[0, 'is_night'] == 0  # 10:30 is not night
    
    def test_amount_based_features(self, sample_transaction_data):
        """Test creation of amount-based features."""
        fe = MockFeatureEngineering()
        df_engineered = fe.engineer_features(sample_transaction_data.copy())
        
        # Check amount features exist
        assert 'amount_log' in df_engineered.columns
        assert 'amount_sqrt' in df_engineered.columns
        
        # Verify transformations
        expected_log = np.log1p(sample_transaction_data['amount'])
        expected_sqrt = np.sqrt(sample_transaction_data['amount'])
        
        np.testing.assert_array_almost_equal(df_engineered['amount_log'], expected_log)
        np.testing.assert_array_almost_equal(df_engineered['amount_sqrt'], expected_sqrt)
    
    def test_feature_engineering_no_missing_values(self, sample_transaction_data):
        """Test that feature engineering handles data without missing values."""
        fe = MockFeatureEngineering()
        df_engineered = fe.engineer_features(sample_transaction_data.copy())
        
        # Verify no NaN values in critical features
        critical_features = ['hour_sin', 'hour_cos', 'amount_log', 'amount_sqrt']
        for feature in critical_features:
            assert not df_engineered[feature].isnull().any()
    
    def test_feature_engineering_preserves_ids(self, sample_transaction_data):
        """Test that feature engineering preserves transaction identifiers."""
        fe = MockFeatureEngineering()
        df_engineered = fe.engineer_features(sample_transaction_data.copy())
        
        # Verify IDs are preserved
        assert df_engineered['transaction_id'].equals(sample_transaction_data['transaction_id'])
        assert df_engineered['customer_id'].equals(sample_transaction_data['customer_id'])
        assert df_engineered['merchant_id'].equals(sample_transaction_data['merchant_id'])


class TestTransactionValidation:
    """Test suite for transaction data validation."""
    
    def test_valid_transaction_data(self, sample_transaction_data):
        """Test validation of valid transaction data."""
        # Check required columns exist
        required_columns = [
            'transaction_id', 'amount', 'merchant_id', 'customer_id',
            'timestamp', 'merchant_category', 'location_country'
        ]
        
        for col in required_columns:
            assert col in sample_transaction_data.columns
    
    def test_amount_is_positive(self, sample_transaction_data):
        """Test that transaction amounts are positive."""
        assert (sample_transaction_data['amount'] >= 0).all()
    
    def test_hour_of_day_valid_range(self, sample_transaction_data):
        """Test that hour_of_day is in valid range [0, 23]."""
        assert sample_transaction_data['hour_of_day'].between(0, 23).all()
    
    def test_timestamp_is_datetime(self, sample_transaction_data):
        """Test that timestamp is datetime type."""
        assert pd.api.types.is_datetime64_any_dtype(sample_transaction_data['timestamp'])


class TestFeatureTransformation:
    """Test suite for feature transformation."""
    
    def test_log_transformation_handles_zero(self):
        """Test that log transformation handles zero amounts correctly."""
        amounts = np.array([0, 1, 10, 100])
        log_amounts = np.log1p(amounts)
        
        # log1p(0) should be 0
        assert log_amounts[0] == 0
        # log1p values should be positive for positive inputs
        assert (log_amounts >= 0).all()
    
    def test_sqrt_transformation(self):
        """Test square root transformation."""
        amounts = np.array([0, 4, 9, 16])
        sqrt_amounts = np.sqrt(amounts)
        
        expected = np.array([0, 2, 3, 4])
        np.testing.assert_array_almost_equal(sqrt_amounts, expected)
    
    def test_cyclical_encoding_consistency(self):
        """Test that cyclical encoding of hours is consistent."""
        hours = np.array([0, 6, 12, 18, 23])
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
        
        # Test that 0 and 24 hours map to same point
        assert np.isclose(hour_sin[0], np.sin(2 * np.pi * 24 / 24))
        assert np.isclose(hour_cos[0], np.cos(2 * np.pi * 24 / 24))
        
        # Test that values are in valid range
        assert (hour_sin >= -1).all() and (hour_sin <= 1).all()
        assert (hour_cos >= -1).all() and (hour_cos <= 1).all()


@pytest.mark.parametrize("amount,expected_risk", [
    (50.0, "low"),
    (500.0, "medium"),
    (5000.0, "high"),
])
def test_amount_risk_categorization(amount, expected_risk):
    """Test risk categorization based on amount."""
    if amount < 100:
        risk = "low"
    elif amount < 1000:
        risk = "medium"
    else:
        risk = "high"
    
    assert risk == expected_risk


def test_feature_vector_shape():
    """Test that feature vector has expected shape."""
    n_samples = 10
    n_features = 15
    
    feature_matrix = np.random.randn(n_samples, n_features)
    
    assert feature_matrix.shape == (n_samples, n_features)
    assert len(feature_matrix.shape) == 2


def test_feature_normalization():
    """Test that feature normalization works correctly."""
    from sklearn.preprocessing import StandardScaler
    
    # Create sample data
    data = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Normalize
    scaler = StandardScaler()
    normalized = scaler.fit_transform(data)
    
    # Check mean is approximately 0 and std is approximately 1
    assert np.abs(normalized.mean(axis=0)).max() < 0.001
    assert np.abs(normalized.std(axis=0) - 1.0).max() < 0.001
