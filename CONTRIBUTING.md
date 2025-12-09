# Contributing to FraudGuard-360

Thank you for considering contributing to FraudGuard-360! This document provides guidelines and instructions for contributing to this fraud detection microservices platform.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment. We welcome contributions from developers of all skill levels.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fraudguard-360.git
   cd fraudguard-360
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Youss2f/fraudguard-360.git
   ```

## Development Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Kubernetes (kubectl) - optional
- Terraform - optional

### Local Environment

1. **Copy environment template**:
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

2. **Install Python dependencies**:
   ```bash
   # For API Gateway
   cd src/api-gateway
   pip install -r requirements.txt
   
   # For ML Service
   cd ../ml-service
   pip install -r requirements.txt
   
   # For Risk Scoring Service
   cd ../risk-scoring-service
   pip install -r requirements.txt
   ```

3. **Start infrastructure services**:
   ```bash
   docker-compose up -d kafka zookeeper postgres redis neo4j prometheus
   ```

4. **Run tests**:
   ```bash
   pytest src/api-gateway/tests/
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new fraud detection rules, ML models, or service enhancements
- **Documentation**: Improve README, API docs, or code comments
- **Tests**: Add unit tests, integration tests, or improve test coverage
- **Performance**: Optimize algorithms or service performance
- **Infrastructure**: Enhance Docker, Kubernetes, or Terraform configurations

### Contribution Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes** following our coding standards

3. **Write/update tests** for your changes

4. **Run tests locally**:
   ```bash
   pytest src/api-gateway/tests/ -v --cov
   flake8 src/
   bandit -r src/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add clear, descriptive commit message"
   ```
   
   **Commit Message Guidelines**:
   - Use present tense ("Add feature" not "Added feature")
   - Use imperative mood ("Move cursor to..." not "Moves cursor to...")
   - Limit first line to 72 characters
   - Reference issues: "Fix transaction validation (fixes #123)"

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return values
- Maximum line length: **100 characters**
- Use **docstrings** for all public functions, classes, and modules

Example:
```python
def calculate_fraud_score(transaction: Dict[str, Any]) -> float:
    """
    Calculate fraud risk score for a transaction.
    
    Args:
        transaction: Dictionary containing transaction details
        
    Returns:
        Float between 0.0 and 1.0 indicating fraud probability
        
    Raises:
        ValueError: If transaction is missing required fields
    """
    pass
```

### Code Quality Tools

We use the following tools (configured in the repository):

- **flake8**: Linting and style checking
- **bandit**: Security vulnerability scanning
- **pytest**: Unit and integration testing
- **black** (optional): Code formatting

Run quality checks:
```bash
# Linting
flake8 src/ --max-line-length=100

# Security scan
bandit -r src/ -c .bandit

# Type checking (optional)
mypy src/
```

## Testing Guidelines

### Test Requirements

- **All new features must include tests**
- **Bug fixes should include regression tests**
- Aim for **minimum 80% code coverage**
- Tests should be **fast, isolated, and deterministic**

### Test Structure

```
src/
└── api-gateway/
    ├── app.py
    └── tests/
        ├── __init__.py
        ├── test_api.py
        ├── test_utils.py
        └── test_integration.py
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

def test_transaction_validation():
    """Test transaction validation logic"""
    # Arrange
    valid_transaction = {
        "amount": 100.0,
        "merchant_id": "M123",
        "user_id": "U456"
    }
    
    # Act
    result = validate_transaction(valid_transaction)
    
    # Assert
    assert result is True
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest src/api-gateway/tests/test_api.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v
```

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated (README, docstrings, API docs)
- [ ] No merge conflicts with main branch
- [ ] Commit messages are clear and descriptive

### PR Template

When opening a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- Describe tests added/modified
- Include test results/screenshots if applicable

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. At least **one maintainer approval** required
2. All **CI checks must pass** (tests, linting, security)
3. **Address review comments** promptly
4. Once approved, maintainers will merge your PR

## Reporting Issues

### Bug Reports

Use the GitHub issue tracker. Include:

- **Clear title** describing the issue
- **Steps to reproduce** the bug
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, Docker version)
- **Logs or error messages**
- **Screenshots** if applicable

### Feature Requests

Describe:

- **Problem** you're trying to solve
- **Proposed solution** or feature
- **Alternatives considered**
- **Additional context** (use cases, examples)

## Project Structure

```
fraudguard-360/
├── src/                        # Source code
│   ├── api-gateway/           # REST API service
│   ├── ml-service/            # Machine learning inference
│   └── risk-scoring-service/  # Rule-based scoring
├── infrastructure/            # Deployment configs
│   ├── kubernetes/           # K8s manifests
│   └── terraform/            # Infrastructure as Code
├── helm/                      # Helm charts
├── docs/                      # Documentation
├── tests/                     # Integration tests
└── scripts/                   # Utility scripts
```

## Questions or Need Help?

- **Open an issue** for questions about the codebase
- **Start a discussion** for general questions or ideas
- **Review existing issues/PRs** for similar questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to FraudGuard-360.
