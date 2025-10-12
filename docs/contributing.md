---
layout: default
title: Contributing
nav_order: 7
description: "Contribution guidelines and development setup for FraudGuard 360°"
---

# Contributing to FraudGuard 360°
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Welcome Contributors!

Thank you for your interest in contributing to FraudGuard 360°! This telecom fraud detection prototype welcomes contributions from students, researchers, and professionals interested in fraud detection, machine learning, and distributed systems.

## Getting Started

### Prerequisites

- **Git**: Version 2.30+
- **Docker**: 24.0+ and Docker Compose 2.20+
- **Python**: 3.11+ for ML services
- **Java**: 17+ for stream processing
- **Node.js**: 18+ for frontend development

### Development Environment Setup

#### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/fraudguard-360.git
cd fraudguard-360

# Add upstream remote
git remote add upstream https://github.com/Youss2f/fraudguard-360.git
```

#### 2. Set Up Development Environment

```bash
# Copy environment template
cp .env.example .env.development

# Start infrastructure services only
docker-compose up -d kafka neo4j redis postgres prometheus grafana

# Verify infrastructure is running
docker-compose ps
```

#### 3. Set Up Python Services

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### 4. Set Up Frontend

```bash
cd frontend
npm install
npm run start  # Start development server
cd ..
```

#### 5. Set Up Java Stream Processing

```bash
cd stream-processor-flink
mvn clean install
mvn exec:java -Dexec.mainClass="com.fraudguard360.FraudDetectionStreamProcessor"
cd ..
```

## Development Workflow

### Branch Strategy

We use **Git Flow** branching strategy:

- **`main`**: Production-ready code
- **`develop`**: Integration branch for features
- **`feature/*`**: Feature development branches
- **`hotfix/*`**: Critical bug fixes
- **`release/*`**: Release preparation branches

#### Creating Feature Branch

```bash
# Update develop branch
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name

# Work on your feature...
git add .
git commit -m "feat: add new fraud detection algorithm"

# Push to your fork
git push origin feature/your-feature-name
```

### Commit Convention

We follow **Conventional Commits** specification:

```bash
# Format: <type>(<scope>): <description>
# 
# Types:
# feat: A new feature
# fix: A bug fix
# docs: Documentation only changes
# style: Changes that do not affect the meaning of the code
# refactor: A code change that neither fixes a bug nor adds a feature
# perf: A code change that improves performance
# test: Adding missing tests or correcting existing tests
# chore: Changes to the build process or auxiliary tools

# Examples:
git commit -m "feat(ml-service): add GraphSAGE neural network implementation"
git commit -m "fix(api-gateway): resolve JWT token validation issue"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(risk-service): add unit tests for risk calculation"
```

### Code Quality Standards

#### Python Code Style
{: .d-inline-block }

Python
{: .label .label-blue }

We use **Black**, **flake8**, and **mypy** for Python code quality:

```bash
# Format code with Black
black .

# Check linting with flake8
flake8 --max-line-length=88 --extend-ignore=E203

# Type checking with mypy
mypy core-ml-service/ risk-scoring-service/ graph-analytics-service/

# Run all quality checks
make lint-python
```

**Python Standards:**
- Line length: 88 characters (Black default)
- Type hints required for all function signatures
- Docstrings required for all public functions/classes
- Follow PEP 8 naming conventions

```python
# Example: Well-documented Python function
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def analyze_fraud_patterns(
    transactions: List[Dict[str, Any]], 
    threshold: float = 0.8,
    include_network: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Analyze fraud patterns in transaction data using GraphSAGE.
    
    Args:
        transactions: List of transaction dictionaries with required fields:
            - transaction_id: str
            - amount: float  
            - customer_id: str
            - timestamp: str (ISO format)
        threshold: Fraud score threshold (0.0-1.0)
        include_network: Whether to include network analysis
        
    Returns:
        Dictionary containing fraud analysis results or None if error
        
    Raises:
        ValueError: If transactions list is empty or invalid format
        
    Example:
        >>> transactions = [{"transaction_id": "123", "amount": 1000.0, ...}]
        >>> result = analyze_fraud_patterns(transactions, threshold=0.9)
        >>> print(result["fraud_score"])
    """
    if not transactions:
        raise ValueError("Transactions list cannot be empty")
        
    logger.info(f"Analyzing {len(transactions)} transactions with threshold {threshold}")
    
    # Implementation here...
    return analysis_result
```

#### Java Code Style
{: .d-inline-block }

Java
{: .label .label-yellow }

We use **Google Java Style** with **Checkstyle** and **SpotBugs**:

```bash
# Format code with Google Java Format
mvn com.coveo:fmt-maven-plugin:format

# Check style with Checkstyle
mvn checkstyle:check

# Static analysis with SpotBugs  
mvn spotbugs:check

# Run all quality checks
mvn clean verify
```

**Java Standards:**
- Google Java Style formatting
- Comprehensive Javadoc for public APIs
- Unit tests for all business logic
- Exception handling with proper logging

```java
/**
 * Processes real-time transaction streams for fraud detection.
 * 
 * <p>This class implements Apache Flink's RichMapFunction to process
 * transaction events in real-time, applying velocity checks and
 * machine learning models for fraud detection.
 * 
 * @author FraudGuard Team
 * @version 1.2.0
 * @since 1.0.0
 */
public class FraudDetectionProcessor extends RichMapFunction<Transaction, FraudAlert> {
    
    private static final Logger LOG = LoggerFactory.getLogger(FraudDetectionProcessor.class);
    private static final double DEFAULT_FRAUD_THRESHOLD = 0.8;
    
    private transient VelocityState velocityState;
    private final double fraudThreshold;
    
    /**
     * Constructs a new FraudDetectionProcessor with default threshold.
     */
    public FraudDetectionProcessor() {
        this(DEFAULT_FRAUD_THRESHOLD);
    }
    
    /**
     * Constructs a new FraudDetectionProcessor with custom threshold.
     * 
     * @param fraudThreshold the fraud score threshold (0.0-1.0)
     * @throws IllegalArgumentException if threshold is not in valid range
     */
    public FraudDetectionProcessor(double fraudThreshold) {
        if (fraudThreshold < 0.0 || fraudThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Fraud threshold must be between 0.0 and 1.0, got: " + fraudThreshold);
        }
        this.fraudThreshold = fraudThreshold;
    }
    
    @Override
    public FraudAlert map(Transaction transaction) throws Exception {
        LOG.debug("Processing transaction: {}", transaction.getTransactionId());
        
        try {
            double fraudScore = calculateFraudScore(transaction);
            return createFraudAlert(transaction, fraudScore);
        } catch (Exception e) {
            LOG.error("Error processing transaction {}: {}", 
                transaction.getTransactionId(), e.getMessage(), e);
            throw e;
        }
    }
}
```

#### TypeScript/React Code Style
{: .d-inline-block }

TypeScript
{: .label .label-green }

We use **Prettier**, **ESLint**, and **TypeScript strict mode**:

```bash
# Format code with Prettier
npm run format

# Check linting with ESLint
npm run lint

# Type checking
npm run type-check

# Run all quality checks
npm run check-all
```

**TypeScript Standards:**
- Strict TypeScript configuration
- Functional components with hooks
- Comprehensive prop types
- Consistent naming conventions

```tsx
// Example: Well-typed React component
import React, { useState, useEffect, useCallback } from 'react';
import { Alert, Card, Typography } from '@mui/material';

interface FraudAlert {
  id: string;
  customerId: string;
  fraudScore: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  timestamp: string;
  recommendation: string;
}

interface FraudAlertsProps {
  /** Maximum number of alerts to display */
  maxAlerts?: number;
  /** Callback fired when alert is acknowledged */
  onAlertAcknowledge?: (alertId: string) => void;
  /** Whether to auto-refresh alerts */
  autoRefresh?: boolean;
}

/**
 * Displays real-time fraud alerts with severity-based styling.
 * 
 * @param props - Component props
 * @returns JSX element representing the fraud alerts panel
 */
export const FraudAlertsPanel: React.FC<FraudAlertsProps> = ({
  maxAlerts = 10,
  onAlertAcknowledge,
  autoRefresh = true,
}) => {
  const [alerts, setAlerts] = useState<FraudAlert[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAlerts = useCallback(async (): Promise<void> => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/alerts');
      
      if (!response.ok) {
        throw new Error(`Failed to fetch alerts: ${response.statusText}`);
      }
      
      const data: FraudAlert[] = await response.json();
      setAlerts(data.slice(0, maxAlerts));
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      console.error('Error fetching fraud alerts:', err);
    } finally {
      setLoading(false);
    }
  }, [maxAlerts]);

  useEffect(() => {
    fetchAlerts();
    
    if (autoRefresh) {
      const interval = setInterval(fetchAlerts, 30000); // Refresh every 30s
      return () => clearInterval(interval);
    }
  }, [fetchAlerts, autoRefresh]);

  // Component implementation...
};
```

## Testing Guidelines

### Test Structure

We follow the **Arrange-Act-Assert** pattern:

```python
# tests/test_fraud_detection.py
import pytest
from unittest.mock import Mock, patch
from core_ml_service.fraud_detection_engine import FraudDetectionEngine

class TestFraudDetectionEngine:
    """Test suite for FraudDetectionEngine class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.engine = FraudDetectionEngine()
        self.sample_transaction = {
            "transaction_id": "test_123",
            "amount": 1500.0,
            "customer_id": "customer_456",
            "merchant_id": "merchant_789"
        }
    
    def test_predict_fraud_legitimate_transaction(self):
        """Test fraud prediction for legitimate transaction."""
        # Arrange
        legitimate_transaction = {**self.sample_transaction, "amount": 50.0}
        
        # Act
        result = self.engine.predict(legitimate_transaction)
        
        # Assert
        assert result["fraud_score"] < 0.5
        assert result["prediction"] == "LEGITIMATE"
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
    
    def test_predict_fraud_suspicious_transaction(self):
        """Test fraud prediction for suspicious transaction."""
        # Arrange
        suspicious_transaction = {**self.sample_transaction, "amount": 10000.0}
        
        # Act  
        result = self.engine.predict(suspicious_transaction)
        
        # Assert
        assert result["fraud_score"] > 0.7
        assert result["prediction"] in ["SUSPICIOUS", "FRAUD"]
    
    @patch('core_ml_service.fraud_detection_engine.GraphSAGEModel')
    def test_predict_with_model_error(self, mock_model):
        """Test error handling when ML model fails."""
        # Arrange
        mock_model.return_value.predict.side_effect = Exception("Model error")
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            self.engine.predict(self.sample_transaction)
        
        assert "Model error" in str(exc_info.value)
```

### Test Coverage Requirements

- **Minimum Coverage**: 80% for all services
- **Critical Components**: 95% coverage required
- **Integration Tests**: All API endpoints
- **Performance Tests**: Load testing for key endpoints

```bash
# Run tests with coverage
pytest --cov=core_ml_service --cov-report=html --cov-report=term-missing
coverage report --show-missing

# Java test coverage
mvn clean test jacoco:report
```

### Integration Testing

```python
# tests/integration/test_api_integration.py
import pytest
import httpx
from fastapi.testclient import TestClient
from api_gateway.main import app

@pytest.fixture
def client():
    """Create test client for API integration tests."""
    return TestClient(app)

@pytest.mark.integration
def test_fraud_analysis_end_to_end(client):
    """Test complete fraud analysis pipeline."""
    # Arrange
    transaction_data = {
        "transaction_id": "integration_test_001",
        "amount": 2500.0,
        "currency": "USD",
        "customer_id": "test_customer_123",
        "merchant_id": "test_merchant_456"
    }
    
    # Act
    response = client.post("/api/v1/fraud/analyze", json=transaction_data)
    
    # Assert
    assert response.status_code == 200
    
    result = response.json()
    assert "fraud_score" in result
    assert "risk_level" in result
    assert "prediction" in result
    assert "processing_time_ms" in result
    
    # Verify fraud score is valid
    assert 0 <= result["fraud_score"] <= 1
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert result["prediction"] in ["LEGITIMATE", "SUSPICIOUS", "FRAUD"]
```

## Documentation Standards

### Code Documentation

- **Python**: Use Google-style docstrings
- **Java**: Use Javadoc with complete @param and @return tags
- **TypeScript**: Use TSDoc comments for complex functions

### API Documentation

- All endpoints must have OpenAPI/Swagger documentation
- Include request/response examples
- Document error codes and responses

```python
# Example: Well-documented FastAPI endpoint
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional

class TransactionRequest(BaseModel):
    """Request model for fraud analysis."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    customer_id: str = Field(..., description="Customer identifier")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")

class FraudAnalysisResponse(BaseModel):
    """Response model for fraud analysis results."""
    transaction_id: str = Field(..., description="Transaction identifier")
    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    prediction: str = Field(..., description="Prediction: LEGITIMATE, SUSPICIOUS, FRAUD")

@app.post(
    "/api/v1/fraud/analyze",
    response_model=FraudAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze transaction for fraud",
    description="Analyzes a single transaction using ML models and returns fraud assessment",
    response_description="Fraud analysis results with score and recommendation",
    tags=["Fraud Detection"]
)
async def analyze_fraud(
    transaction: TransactionRequest
) -> FraudAnalysisResponse:
    """
    Analyze a transaction for potential fraud.
    
    This endpoint processes a transaction through multiple fraud detection
    algorithms including GraphSAGE neural networks and ensemble methods.
    
    Args:
        transaction: Transaction data to analyze
        
    Returns:
        Fraud analysis results including score and risk level
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
        
    Example:
        ```json
        {
            "transaction_id": "txn_123",
            "amount": 1500.0,
            "customer_id": "customer_456"
        }
        ```
    """
    try:
        # Implementation here...
        pass
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transaction data: {str(e)}"
        )
```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**: Run all tests locally
2. **Check code quality**: Run linting and formatting
3. **Update documentation**: If adding features
4. **Add tests**: For new functionality
5. **Update CHANGELOG**: Following Keep a Changelog format

```bash
# Pre-submission checklist
make test-all          # Run all tests
make lint-all          # Check code quality  
make docs-build        # Build documentation
git rebase develop     # Rebase on latest develop
```

### Pull Request Template

When creating a PR, use this template:

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality) 
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Related Issues
Closes #(issue number)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: At least one maintainer review required
3. **Testing**: All tests must pass
4. **Documentation**: Updated if needed
5. **Approval**: Approved before merging

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Docker version: [e.g. 24.0.5]
- Python version: [e.g. 3.11.2]
- Browser (if applicable): [e.g. Chrome 118]

**Logs**
```
Include relevant log output
```

**Additional Context**
Any other context about the problem.
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Clear description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
Describe how you envision this feature working.

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other context or screenshots about the feature request.
```

## Development Guidelines

### Performance Considerations

- **Database Queries**: Use indexes and avoid N+1 queries
- **API Responses**: Keep response sizes reasonable
- **Memory Usage**: Profile memory usage for ML models
- **Caching**: Implement appropriate caching strategies

### Security Best Practices

- **Input Validation**: Validate all inputs
- **Authentication**: Use JWT tokens properly
- **Authorization**: Implement proper RBAC
- **Secrets**: Never commit secrets to version control
- **Dependencies**: Keep dependencies updated

### Monitoring and Logging

- **Structured Logging**: Use JSON format for logs
- **Metrics**: Implement Prometheus metrics
- **Tracing**: Add distributed tracing for complex flows
- **Alerting**: Set up appropriate alerts

## Troubleshooting

### Common Development Issues

#### Docker Issues
```bash
# Container won't start
docker-compose down
docker system prune -f
docker-compose up -d

# Port conflicts
sudo netstat -tulpn | grep :8000
sudo lsof -i :8000
```

#### Python Environment Issues
```bash
# Dependencies conflicts
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# Import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Database Connection Issues
```bash
# Check database status
docker-compose ps postgres neo4j redis

# Reset databases
docker-compose down -v
docker-compose up -d postgres neo4j redis
```

### Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/Youss2f/fraudguard-360/issues)
- **GitHub Discussions**: [Ask questions and discuss](https://github.com/Youss2f/fraudguard-360/discussions)
- **Documentation**: [Read the full documentation](https://youss2f.github.io/fraudguard-360/)

## Recognition

Contributors are recognized in:
- **README.md**: Contributors section
- **CHANGELOG.md**: Release notes
- **Documentation**: Author attribution
- **GitHub**: Contributor statistics

---

Thank you for contributing to FraudGuard 360°! Your contributions help advance telecom fraud detection research and development.