import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphSAGEModel(nn.Module):
    """Advanced GraphSAGE model for fraud detection"""
    
    def __init__(self, num_features, hidden_dim=128, num_classes=2, dropout=0.3):
        super(GraphSAGEModel, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Graph convolution layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolutions with residual connections
        h1 = torch.relu(self.conv1(x, edge_index))
        h1 = self.dropout(h1)
        
        h2 = torch.relu(self.conv2(h1, edge_index))
        h2 = self.dropout(h2)
        
        h3 = torch.relu(self.conv3(h2, edge_index))
        h3 = self.dropout(h3)
        
        # Global pooling
        if batch is not None:
            h3 = global_mean_pool(h3, batch)
        else:
            h3 = torch.mean(h3, dim=0, keepdim=True)
        
        # Classification
        out = self.classifier(h3)
        return torch.softmax(out, dim=1)

class FraudDetectionTrainer:
    """Comprehensive fraud detection model trainer"""
    
    def __init__(self, data_path="data/", models_path="models/"):
        self.data_path = data_path
        self.models_path = models_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Ensure directories exist
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)
        
        logger.info(f"Initialized trainer with data_path={data_path}, models_path={models_path}")
    
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic CDR data for training"""
        logger.info(f"Generating {n_samples} synthetic samples...")
        
        np.random.seed(42)
        
        # Generate features
        data = {
            'user_id': [f'user_{i:06d}' for i in range(n_samples)],
            'call_duration': np.random.lognormal(3, 1, n_samples),  # Log-normal distribution
            'call_cost': np.random.gamma(2, 0.5, n_samples),
            'calls_per_day': np.random.poisson(10, n_samples),
            'unique_numbers_called': np.random.poisson(5, n_samples),
            'international_calls': np.random.binomial(1, 0.1, n_samples),
            'night_calls': np.random.binomial(1, 0.2, n_samples),
            'weekend_calls': np.random.binomial(1, 0.3, n_samples),
            'call_frequency_variance': np.random.exponential(2, n_samples),
            'location_changes': np.random.poisson(2, n_samples),
            'avg_call_gap': np.random.exponential(1, n_samples),
            'network_connections': np.random.poisson(8, n_samples),
            'suspicious_patterns': np.random.binomial(1, 0.05, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create derived features
        df['cost_per_minute'] = df['call_cost'] / (df['call_duration'] + 1e-6)
        df['calls_per_unique_number'] = df['calls_per_day'] / (df['unique_numbers_called'] + 1)
        df['location_change_rate'] = df['location_changes'] / df['calls_per_day']
        df['network_density'] = df['network_connections'] / (df['unique_numbers_called'] + 1)
        
        # Generate fraud labels with realistic patterns
        fraud_probability = (
            0.1 * (df['call_duration'] > df['call_duration'].quantile(0.95)) +
            0.15 * (df['cost_per_minute'] > df['cost_per_minute'].quantile(0.9)) +
            0.2 * (df['calls_per_day'] > df['calls_per_day'].quantile(0.95)) +
            0.3 * df['international_calls'] +
            0.25 * df['night_calls'] +
            0.4 * df['suspicious_patterns'] +
            0.2 * (df['location_changes'] > df['location_changes'].quantile(0.9))
        )
        
        df['is_fraud'] = np.random.binomial(1, np.clip(fraud_probability, 0, 0.8), n_samples)
        
        # Add some noise to make it more realistic
        fraud_indices = df[df['is_fraud'] == 1].index
        normal_indices = df[df['is_fraud'] == 0].index
        
        # Increase some features for fraud cases
        df.loc[fraud_indices, 'call_duration'] *= np.random.uniform(1.2, 2.5, len(fraud_indices))
        df.loc[fraud_indices, 'calls_per_day'] *= np.random.uniform(1.5, 3.0, len(fraud_indices))
        df.loc[fraud_indices, 'call_cost'] *= np.random.uniform(1.3, 2.8, len(fraud_indices))
        
        logger.info(f"Generated data with {df['is_fraud'].sum()} fraud cases ({df['is_fraud'].mean():.3f} fraud rate)")
        
        # Save synthetic data
        df.to_csv(os.path.join(self.data_path, 'synthetic_cdr_data.csv'), index=False)
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        logger.info("Preparing features...")
        
        # Select numerical features
        feature_columns = [
            'call_duration', 'call_cost', 'calls_per_day', 'unique_numbers_called',
            'international_calls', 'night_calls', 'weekend_calls', 'call_frequency_variance',
            'location_changes', 'avg_call_gap', 'network_connections', 'suspicious_patterns',
            'cost_per_minute', 'calls_per_unique_number', 'location_change_rate', 'network_density'
        ]
        
        X = df[feature_columns].copy()
        y = df['is_fraud'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        return X_scaled, y, feature_columns
    
    def train_isolation_forest(self, X, contamination=0.1):
        """Train Isolation Forest for anomaly detection"""
        logger.info("Training Isolation Forest...")
        
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples='auto',
            n_jobs=-1
        )
        
        model.fit(X)
        
        # Evaluate
        anomaly_scores = model.decision_function(X)
        predictions = model.predict(X)
        
        logger.info(f"Isolation Forest - Found {(predictions == -1).sum()} anomalies out of {len(X)} samples")
        
        # Save model
        joblib.dump(model, os.path.join(self.models_path, 'isolation_forest_model.pkl'))
        return model
    
    def train_random_forest(self, X, y):
        """Train Random Forest classifier"""
        logger.info("Training Random Forest...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Grid search for best parameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluate
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Random Forest - Train Score: {train_score:.4f}, Test Score: {test_score:.4f}, AUC: {auc_score:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        joblib.dump(best_model, os.path.join(self.models_path, 'random_forest_model.pkl'))
        return best_model
    
    def train_gradient_boosting(self, X, y):
        """Train Gradient Boosting classifier"""
        logger.info("Training Gradient Boosting...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Grid search for best parameters
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluate
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Gradient Boosting - Train Score: {train_score:.4f}, Test Score: {test_score:.4f}, AUC: {auc_score:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        joblib.dump(best_model, os.path.join(self.models_path, 'gradient_boosting_model.pkl'))
        return best_model
    
    def train_graphsage(self, X, y, user_ids):
        """Train GraphSAGE model"""
        logger.info("Training GraphSAGE model...")
        
        # Create graph structure (simplified - in practice this would be based on call relationships)
        num_nodes = len(X)
        
        # Create edges based on feature similarity (simplified graph construction)
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(X)
        
        # Keep only top-k similar connections to avoid dense graph
        k = 10
        edge_list = []
        
        for i in range(num_nodes):
            # Get top-k most similar nodes (excluding self)
            similar_indices = np.argsort(similarity_matrix[i])[-k-1:-1]
            for j in similar_indices:
                if i != j:
                    edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Convert to PyTorch tensors
        x = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        
        # Split data
        num_nodes = x.size(0)
        num_train = int(0.6 * num_nodes)
        num_val = int(0.2 * num_nodes)
        
        indices = torch.randperm(num_nodes)
        train_idx = indices[:num_train]
        val_idx = indices[num_train:num_train + num_val]
        test_idx = indices[num_train + num_val:]
        
        # Create model
        model = GraphSAGEModel(num_features=x.size(1), hidden_dim=128, num_classes=2)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        best_val_acc = 0
        patience = 50
        patience_counter = 0
        
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out[train_idx], y_tensor[train_idx])
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_out = model(x, edge_index)
                    val_pred = val_out[val_idx].argmax(dim=1)
                    val_acc = (val_pred == y_tensor[val_idx]).float().mean().item()
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        # Save best model
                        torch.save(model.state_dict(), os.path.join(self.models_path, 'graphsage_model.pth'))
                    else:
                        patience_counter += 1
                    
                    logger.info(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
                
                model.train()
                
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(os.path.join(self.models_path, 'graphsage_model.pth')))
        model.eval()
        
        with torch.no_grad():
            test_out = model(x, edge_index)
            test_pred = test_out[test_idx].argmax(dim=1)
            test_acc = (test_pred == y_tensor[test_idx]).float().mean().item()
            
        logger.info(f"GraphSAGE - Final Test Accuracy: {test_acc:.4f}")
        
        # Save model architecture info
        model_info = {
            'num_features': x.size(1),
            'hidden_dim': 128,
            'num_classes': 2,
            'test_accuracy': test_acc
        }
        
        with open(os.path.join(self.models_path, 'graphsage_info.json'), 'w') as f:
            json.dump(model_info, f)
        
        return model
    
    def train_behavioral_profiler(self, X, y):
        """Train behavioral profiling model using clustering"""
        logger.info("Training Behavioral Profiler...")
        
        # Use DBSCAN for behavior clustering
        clustering = DBSCAN(eps=0.5, min_samples=5)
        clusters = clustering.fit_predict(X)
        
        logger.info(f"Behavioral Profiler - Found {len(set(clusters))} clusters (including noise)")
        
        # Analyze fraud distribution in clusters
        df_clusters = pd.DataFrame({
            'cluster': clusters,
            'is_fraud': y
        })
        
        cluster_fraud_rates = df_clusters.groupby('cluster')['is_fraud'].agg(['count', 'sum', 'mean'])
        cluster_fraud_rates['fraud_rate'] = cluster_fraud_rates['mean']
        
        logger.info("Cluster fraud rates:")
        print(cluster_fraud_rates)
        
        # Save clustering model
        joblib.dump(clustering, os.path.join(self.models_path, 'behavioral_clustering_model.pkl'))
        
        # Save cluster fraud rates
        cluster_fraud_rates.to_csv(os.path.join(self.models_path, 'cluster_fraud_rates.csv'))
        
        return clustering
    
    def save_preprocessing_objects(self):
        """Save preprocessing objects"""
        logger.info("Saving preprocessing objects...")
        
        joblib.dump(self.scaler, os.path.join(self.models_path, 'feature_scaler.pkl'))
        
        # Save feature names
        with open(os.path.join(self.models_path, 'feature_names.json'), 'w') as f:
            json.dump(self.feature_columns, f)
    
    def train_all_models(self):
        """Train all fraud detection models"""
        logger.info("Starting comprehensive model training...")
        
        # Generate or load data
        if os.path.exists(os.path.join(self.data_path, 'synthetic_cdr_data.csv')):
            logger.info("Loading existing synthetic data...")
            df = pd.read_csv(os.path.join(self.data_path, 'synthetic_cdr_data.csv'))
        else:
            df = self.generate_synthetic_data()
        
        # Prepare features
        X, y, feature_columns = self.prepare_features(df)
        self.feature_columns = feature_columns
        
        # Train all models
        models = {}
        
        # 1. Isolation Forest (Unsupervised Anomaly Detection)
        models['isolation_forest'] = self.train_isolation_forest(X)
        
        # 2. Random Forest (Supervised Classification)
        models['random_forest'] = self.train_random_forest(X, y)
        
        # 3. Gradient Boosting (Supervised Classification)
        models['gradient_boosting'] = self.train_gradient_boosting(X, y)
        
        # 4. GraphSAGE (Graph Neural Network)
        models['graphsage'] = self.train_graphsage(X, y, df['user_id'])
        
        # 5. Behavioral Profiler (Clustering-based)
        models['behavioral_profiler'] = self.train_behavioral_profiler(X, y)
        
        # Save preprocessing objects
        self.save_preprocessing_objects()
        
        # Create model ensemble metadata
        ensemble_info = {
            'models': list(models.keys()),
            'feature_columns': feature_columns,
            'training_date': datetime.now().isoformat(),
            'data_size': len(df),
            'fraud_rate': y.mean(),
            'model_versions': {
                'isolation_forest': '1.0',
                'random_forest': '1.0',
                'gradient_boosting': '1.0',
                'graphsage': '1.0',
                'behavioral_profiler': '1.0'
            }
        }
        
        with open(os.path.join(self.models_path, 'ensemble_info.json'), 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        logger.info("All models trained successfully!")
        logger.info(f"Models saved to: {self.models_path}")
        
        return models

def main():
    """Main training function"""
    print("🚀 Starting FraudGuard 360 ML Model Training")
    print("=" * 60)
    
    trainer = FraudDetectionTrainer()
    models = trainer.train_all_models()
    
    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")
    print(f"📁 Models saved to: {trainer.models_path}")
    print(f"📊 Trained {len(models)} models:")
    for model_name in models.keys():
        print(f"   - {model_name}")
    
    print("\n🎯 Models are ready for deployment in the ML service!")

if __name__ == "__main__":
    main()