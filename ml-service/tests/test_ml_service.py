import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import asyncio
from fastapi.testclient import TestClient

# Import ML service components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.server import app
from models.graphsage import GraphSAGEModel, FraudDetectionModel
from training.train import ModelTrainer, DataProcessor


class TestMLInferenceServer:
    """Unit tests for ML inference server"""
    
    def setup_method(self):
        """Setup test client and mocks"""
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test ML service health check"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "gpu_available" in data
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    @patch('inference.server.fraud_model')
    def test_predict_fraud_endpoint(self, mock_model):
        """Test fraud prediction endpoint"""
        # Mock model prediction
        mock_model.predict.return_value = {
            "fraud_score": 0.85,
            "fraud_probability": 0.85,
            "risk_level": "high",
            "features_importance": {"duration": 0.3, "cost": 0.4, "time": 0.3},
            "model_version": "1.0.0"
        }
        
        # Test data
        cdr_data = {
            "call_id": "test_123",
            "caller_number": "1234567890",
            "callee_number": "0987654321",
            "duration": 300,
            "cost": 10.50,
            "timestamp": "2024-01-01T12:00:00Z",
            "location": "US"
        }
        
        response = self.client.post("/predict/fraud", json=cdr_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "fraud_score" in result
        assert "fraud_probability" in result
        assert "risk_level" in result
        assert result["fraud_score"] == 0.85
    
    def test_predict_fraud_invalid_data(self):
        """Test fraud prediction with invalid data"""
        invalid_data = {
            "call_id": "",  # Missing required field
            "duration": -1  # Invalid duration
        }
        
        response = self.client.post("/predict/fraud", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    @patch('inference.server.fraud_model')
    def test_batch_predict_endpoint(self, mock_model):
        """Test batch prediction endpoint"""
        # Mock batch predictions
        mock_model.predict_batch.return_value = [
            {"fraud_score": 0.2, "risk_level": "low"},
            {"fraud_score": 0.8, "risk_level": "high"}
        ]
        
        batch_data = {
            "cdrs": [
                {
                    "call_id": "test_1",
                    "caller_number": "1111111111",
                    "callee_number": "2222222222",
                    "duration": 60,
                    "cost": 2.0
                },
                {
                    "call_id": "test_2", 
                    "caller_number": "3333333333",
                    "callee_number": "4444444444",
                    "duration": 600,
                    "cost": 15.0
                }
            ]
        }
        
        response = self.client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "predictions" in result
        assert len(result["predictions"]) == 2
    
    @patch('inference.server.fraud_model')
    def test_model_info_endpoint(self, mock_model):
        """Test model information endpoint"""
        mock_model.get_model_info.return_value = {
            "model_type": "GraphSAGE",
            "version": "1.0.0",
            "training_date": "2024-01-01",
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.91
        }
        
        response = self.client.get("/model/info")
        assert response.status_code == 200
        
        result = response.json()
        assert "model_type" in result
        assert "accuracy" in result


class TestGraphSAGEModel:
    """Unit tests for GraphSAGE model implementation"""
    
    def setup_method(self):
        """Setup model for testing"""
        self.input_dim = 10
        self.hidden_dim = 64
        self.output_dim = 2
        self.model = GraphSAGEModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=2
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model.input_dim == self.input_dim
        assert self.model.hidden_dim == self.hidden_dim
        assert self.model.output_dim == self.output_dim
        assert len(self.model.convs) == 2
    
    def test_forward_pass(self):
        """Test model forward pass"""
        # Create dummy graph data
        num_nodes = 100
        x = torch.randn(num_nodes, self.input_dim)
        edge_index = torch.randint(0, num_nodes, (2, 200))
        
        # Forward pass
        output = self.model(x, edge_index)
        
        assert output.shape == (num_nodes, self.output_dim)
        assert not torch.isnan(output).any()
    
    def test_model_training_mode(self):
        """Test model training and evaluation modes"""
        # Training mode
        self.model.train()
        assert self.model.training is True
        
        # Evaluation mode
        self.model.eval()
        assert self.model.training is False
    
    def test_model_parameters(self):
        """Test model parameters"""
        params = list(self.model.parameters())
        assert len(params) > 0
        
        # Check parameter shapes
        for param in params:
            assert param.requires_grad is True


class TestFraudDetectionModel:
    """Unit tests for fraud detection model wrapper"""
    
    def setup_method(self):
        """Setup fraud detection model"""
        self.model = FraudDetectionModel()
    
    @patch('models.graphsage.torch.load')
    def test_model_loading(self, mock_load):
        """Test model loading from checkpoint"""
        # Mock model state dict
        mock_state_dict = {
            'conv1.weight': torch.randn(64, 10),
            'conv2.weight': torch.randn(2, 64)
        }
        mock_load.return_value = mock_state_dict
        
        with tempfile.NamedTemporaryFile(suffix='.pth') as tmp_file:
            self.model.load_model(tmp_file.name)
            mock_load.assert_called_once_with(tmp_file.name, map_location='cpu')
    
    def test_feature_preprocessing(self):
        """Test feature preprocessing"""
        raw_data = {
            "duration": 300,
            "cost": 10.5,
            "caller_number": "1234567890",
            "callee_number": "0987654321",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        features = self.model.preprocess_features(raw_data)
        
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 1  # 1D feature vector
        assert not torch.isnan(features).any()
    
    @patch.object(FraudDetectionModel, 'model')
    def test_fraud_prediction(self, mock_model):
        """Test fraud prediction"""
        # Mock model output
        mock_output = torch.tensor([[0.3, 0.7]])  # Low fraud, high fraud
        mock_model.return_value = mock_output
        mock_model.eval = Mock()
        
        cdr_data = {
            "duration": 300,
            "cost": 10.5,
            "caller_number": "1234567890",
            "callee_number": "0987654321"
        }
        
        prediction = self.model.predict(cdr_data)
        
        assert "fraud_score" in prediction
        assert "fraud_probability" in prediction
        assert "risk_level" in prediction
        assert 0 <= prediction["fraud_score"] <= 1
    
    def test_risk_level_classification(self):
        """Test risk level classification"""
        # Test low risk
        assert self.model._classify_risk_level(0.2) == "low"
        
        # Test medium risk
        assert self.model._classify_risk_level(0.5) == "medium"
        
        # Test high risk
        assert self.model._classify_risk_level(0.8) == "high"
    
    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        batch_data = [
            {"duration": 60, "cost": 2.0},
            {"duration": 300, "cost": 15.0},
            {"duration": 600, "cost": 25.0}
        ]
        
        with patch.object(self.model, 'predict') as mock_predict:
            mock_predict.side_effect = [
                {"fraud_score": 0.1, "risk_level": "low"},
                {"fraud_score": 0.6, "risk_level": "medium"},
                {"fraud_score": 0.9, "risk_level": "high"}
            ]
            
            predictions = self.model.predict_batch(batch_data)
            
            assert len(predictions) == 3
            assert mock_predict.call_count == 3


class TestModelTrainer:
    """Unit tests for model training pipeline"""
    
    def setup_method(self):
        """Setup trainer for testing"""
        self.trainer = ModelTrainer()
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        assert self.trainer.device is not None
        assert self.trainer.model is not None
        assert self.trainer.optimizer is not None
    
    @patch('training.train.torch.save')
    def test_model_saving(self, mock_save):
        """Test model checkpoint saving"""
        with tempfile.NamedTemporaryFile(suffix='.pth') as tmp_file:
            self.trainer.save_model(tmp_file.name)
            mock_save.assert_called_once()
    
    def test_training_step(self):
        """Test single training step"""
        # Create dummy training data
        num_nodes = 50
        x = torch.randn(num_nodes, 10)
        edge_index = torch.randint(0, num_nodes, (2, 100))
        y = torch.randint(0, 2, (num_nodes,))
        
        # Perform training step
        loss = self.trainer.training_step(x, edge_index, y)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_validation_step(self):
        """Test validation step"""
        # Create dummy validation data
        num_nodes = 30
        x = torch.randn(num_nodes, 10)
        edge_index = torch.randint(0, num_nodes, (2, 60))
        y = torch.randint(0, 2, (num_nodes,))
        
        # Perform validation step
        val_loss, accuracy = self.trainer.validation_step(x, edge_index, y)
        
        assert isinstance(val_loss, torch.Tensor)
        assert 0 <= accuracy <= 1


class TestDataProcessor:
    """Unit tests for data processing pipeline"""
    
    def setup_method(self):
        """Setup data processor"""
        self.processor = DataProcessor()
    
    def test_feature_extraction(self):
        """Test feature extraction from CDR data"""
        cdr_data = {
            "call_id": "test_123",
            "duration": 300,
            "cost": 10.5,
            "caller_number": "1234567890",
            "callee_number": "0987654321",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        features = self.processor.extract_features(cdr_data)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.isnan(features).any()
    
    def test_data_normalization(self):
        """Test data normalization"""
        raw_data = np.array([[1, 100, 1000], [2, 200, 2000], [3, 300, 3000]])
        
        normalized_data = self.processor.normalize_data(raw_data)
        
        assert normalized_data.shape == raw_data.shape
        # Check that data is normalized (mean ~ 0, std ~ 1)
        assert abs(normalized_data.mean()) < 0.1
        assert abs(normalized_data.std() - 1.0) < 0.1
    
    def test_graph_construction(self):
        """Test graph construction from CDR data"""
        cdr_records = [
            {"caller": "A", "callee": "B", "duration": 100},
            {"caller": "B", "callee": "C", "duration": 200},
            {"caller": "A", "callee": "C", "duration": 150}
        ]
        
        nodes, edges = self.processor.construct_graph(cdr_records)
        
        assert len(nodes) == 3  # A, B, C
        assert len(edges) > 0
        assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in edges)
    
    def test_train_test_split(self):
        """Test train/test data splitting"""
        data = np.random.randn(100, 10)
        labels = np.random.randint(0, 2, 100)
        
        train_data, test_data, train_labels, test_labels = self.processor.train_test_split(
            data, labels, test_size=0.2
        )
        
        assert len(train_data) == 80
        assert len(test_data) == 20
        assert len(train_labels) == 80
        assert len(test_labels) == 20


class TestModelPerformanceMetrics:
    """Unit tests for model performance evaluation"""
    
    def setup_method(self):
        """Setup performance metrics testing"""
        self.true_labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        self.predicted_labels = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        self.predicted_probs = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3, 0.85])
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation"""
        from training.train import calculate_accuracy
        
        accuracy = calculate_accuracy(self.true_labels, self.predicted_labels)
        expected_accuracy = 6/8  # 6 correct out of 8
        
        assert abs(accuracy - expected_accuracy) < 0.01
    
    def test_precision_recall_calculation(self):
        """Test precision and recall calculation"""
        from training.train import calculate_precision_recall
        
        precision, recall = calculate_precision_recall(
            self.true_labels, self.predicted_labels
        )
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
    
    def test_f1_score_calculation(self):
        """Test F1 score calculation"""
        from training.train import calculate_f1_score
        
        f1 = calculate_f1_score(self.true_labels, self.predicted_labels)
        
        assert 0 <= f1 <= 1
    
    def test_auc_calculation(self):
        """Test AUC calculation"""
        from training.train import calculate_auc
        
        auc = calculate_auc(self.true_labels, self.predicted_probs)
        
        assert 0 <= auc <= 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])