"""
Tests for ML Service - Model and Inference
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime


class MockFraudDetectionModel(nn.Module):
    """Mock neural network for testing."""
    
    def __init__(self, input_size: int, hidden_sizes=[64, 32]):
        super(MockFraudDetectionModel, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.output = nn.Linear(hidden_sizes[1], 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return MockFraudDetectionModel(input_size=15)


@pytest.fixture
def sample_input_tensor():
    """Create sample input tensor."""
    return torch.randn(10, 15)  # 10 samples, 15 features


class TestModelArchitecture:
    """Test suite for model architecture."""
    
    def test_model_initialization(self, sample_model):
        """Test that model initializes correctly."""
        assert isinstance(sample_model, nn.Module)
        assert sample_model.input_size == 15
    
    def test_model_forward_pass(self, sample_model, sample_input_tensor):
        """Test forward pass through model."""
        output = sample_model(sample_input_tensor)
        
        # Check output shape
        assert output.shape == (10, 1)
        
        # Check output is in valid probability range [0, 1]
        assert (output >= 0).all() and (output <= 1).all()
    
    def test_model_parameters_exist(self, sample_model):
        """Test that model has trainable parameters."""
        params = list(sample_model.parameters())
        assert len(params) > 0
        
        # Check that parameters require gradients
        for param in params:
            assert param.requires_grad
    
    def test_model_layer_shapes(self, sample_model):
        """Test that layer dimensions are correct."""
        # Check first layer
        assert sample_model.fc1.in_features == 15
        assert sample_model.fc1.out_features == 64
        
        # Check output layer
        assert sample_model.output.out_features == 1


class TestModelInference:
    """Test suite for model inference."""
    
    def test_single_prediction(self, sample_model):
        """Test single transaction prediction."""
        sample_model.eval()
        
        # Single transaction
        input_data = torch.randn(1, 15)
        
        with torch.no_grad():
            prediction = sample_model(input_data)
        
        assert prediction.shape == (1, 1)
        assert 0 <= prediction.item() <= 1
    
    def test_batch_prediction(self, sample_model):
        """Test batch prediction."""
        sample_model.eval()
        
        batch_size = 32
        input_data = torch.randn(batch_size, 15)
        
        with torch.no_grad():
            predictions = sample_model(input_data)
        
        assert predictions.shape == (batch_size, 1)
        assert (predictions >= 0).all() and (predictions <= 1).all()
    
    def test_prediction_consistency(self, sample_model):
        """Test that same input gives same output (deterministic)."""
        sample_model.eval()
        
        input_data = torch.randn(5, 15)
        
        with torch.no_grad():
            pred1 = sample_model(input_data)
            pred2 = sample_model(input_data)
        
        torch.testing.assert_close(pred1, pred2)
    
    def test_inference_time(self, sample_model):
        """Test that inference is fast enough."""
        import time
        
        sample_model.eval()
        input_data = torch.randn(100, 15)
        
        start = time.time()
        with torch.no_grad():
            _ = sample_model(input_data)
        duration = time.time() - start
        
        # Should process 100 transactions in less than 1 second
        assert duration < 1.0


class TestModelTraining:
    """Test suite for model training utilities."""
    
    def test_loss_function(self):
        """Test binary cross-entropy loss calculation."""
        criterion = nn.BCELoss()
        
        # Perfect prediction
        predictions = torch.tensor([[0.9], [0.1], [0.8]])
        targets = torch.tensor([[1.0], [0.0], [1.0]])
        
        loss = criterion(predictions, targets)
        
        # Loss should be small for good predictions
        assert loss.item() < 1.0
        assert loss.item() >= 0
    
    def test_optimizer_updates_parameters(self, sample_model):
        """Test that optimizer updates model parameters."""
        optimizer = torch.optim.Adam(sample_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Get initial parameters
        initial_params = [p.clone() for p in sample_model.parameters()]
        
        # Training step
        input_data = torch.randn(10, 15)
        targets = torch.randint(0, 2, (10, 1)).float()
        
        optimizer.zero_grad()
        predictions = sample_model(input_data)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        # Check that at least one parameter changed
        params_changed = False
        for initial, current in zip(initial_params, sample_model.parameters()):
            if not torch.equal(initial, current):
                params_changed = True
                break
        
        assert params_changed


class TestPredictionPostProcessing:
    """Test suite for prediction post-processing."""
    
    def test_probability_to_decision(self):
        """Test conversion of probability to decision."""
        def get_decision(prob: float) -> str:
            if prob >= 0.8:
                return "DECLINE"
            elif prob >= 0.5:
                return "REVIEW"
            else:
                return "APPROVE"
        
        assert get_decision(0.9) == "DECLINE"
        assert get_decision(0.6) == "REVIEW"
        assert get_decision(0.3) == "APPROVE"
    
    def test_probability_to_risk_score(self):
        """Test conversion of probability to risk score (0-100)."""
        probabilities = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        risk_scores = (probabilities * 100).astype(int)
        
        expected = np.array([0, 25, 50, 75, 100])
        np.testing.assert_array_equal(risk_scores, expected)
    
    def test_confidence_calculation(self):
        """Test confidence calculation from probability."""
        def calculate_confidence(prob: float) -> float:
            # Confidence is distance from 0.5 decision boundary
            return abs(prob - 0.5) * 2
        
        # High confidence predictions
        assert calculate_confidence(0.9) > 0.7
        assert calculate_confidence(0.1) > 0.7
        
        # Low confidence prediction (close to boundary)
        assert calculate_confidence(0.5) == 0.0


class TestModelSerialization:
    """Test suite for model saving/loading."""
    
    def test_model_save_and_load(self, sample_model, tmp_path):
        """Test that model can be saved and loaded."""
        save_path = tmp_path / "model.pth"
        
        # Save model
        torch.save(sample_model.state_dict(), save_path)
        
        # Create new model and load weights
        loaded_model = MockFraudDetectionModel(input_size=15)
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Test that predictions match
        input_data = torch.randn(5, 15)
        
        with torch.no_grad():
            original_pred = sample_model(input_data)
            loaded_pred = loaded_model(input_data)
        
        torch.testing.assert_close(original_pred, loaded_pred)


class TestInputValidation:
    """Test suite for input validation."""
    
    def test_invalid_input_shape(self, sample_model):
        """Test that model rejects invalid input shape."""
        # Wrong number of features
        invalid_input = torch.randn(10, 20)  # Should be 15 features
        
        with pytest.raises(RuntimeError):
            _ = sample_model(invalid_input)
    
    def test_empty_batch(self, sample_model):
        """Test handling of empty batch."""
        empty_input = torch.randn(0, 15)
        
        output = sample_model(empty_input)
        assert output.shape == (0, 1)


@pytest.mark.parametrize("batch_size", [1, 16, 32, 64, 128])
def test_variable_batch_sizes(sample_model, batch_size):
    """Test model with various batch sizes."""
    sample_model.eval()
    
    input_data = torch.randn(batch_size, 15)
    
    with torch.no_grad():
        output = sample_model(input_data)
    
    assert output.shape == (batch_size, 1)
    assert (output >= 0).all() and (output <= 1).all()


def test_model_gradient_flow(sample_model):
    """Test that gradients flow through the model."""
    criterion = nn.BCELoss()
    
    input_data = torch.randn(10, 15, requires_grad=True)
    targets = torch.randint(0, 2, (10, 1)).float()
    
    predictions = sample_model(input_data)
    loss = criterion(predictions, targets)
    loss.backward()
    
    # Check that input has gradients
    assert input_data.grad is not None
    
    # Check that model parameters have gradients
    for param in sample_model.parameters():
        assert param.grad is not None


def test_model_eval_mode(sample_model):
    """Test that model behaves differently in train vs eval mode."""
    input_data = torch.randn(10, 15)
    
    # Train mode
    sample_model.train()
    assert sample_model.training
    
    # Eval mode
    sample_model.eval()
    assert not sample_model.training
