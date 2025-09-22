# Contributing to FraudGuard 360°

Thank you for your interest in contributing to FraudGuard 360°! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## 🤝 Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

## 🚀 Getting Started

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Node.js 18+ and npm 8+
- Python 3.11+ with pip 22+
- Java 11+ and Maven 3.8+
- Git 2.30+

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/fraudguard-360.git
   cd fraudguard-360
   ```

2. **Set up Development Environment**
   ```bash
   # Install dependencies
   make install-deps
   
   # Start development services
   docker-compose -f docker-compose.dev.yml up -d
   
   # Verify setup
   make health-check
   ```

3. **Configure IDE**
   - Install recommended VS Code extensions (see `.vscode/extensions.json`)
   - Configure code formatting and linting

## 🔄 Development Process

### Branch Strategy

We use **GitFlow** with the following branch types:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features or enhancements
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes
- `release/*`: Release preparation

### Feature Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Development**
   - Write code following our [coding standards](#coding-standards)
   - Add/update tests with minimum 85% coverage
   - Update documentation if needed

3. **Testing**
   ```bash
   # Run all tests
   make test
   
   # Run specific service tests
   make test-frontend
   make test-api
   make test-ml
   make test-flink
   ```

4. **Commit Changes**
   - Follow [Conventional Commits](#commit-guidelines)
   - Make atomic commits with clear messages

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create PR through GitHub interface
   ```

## 📏 Coding Standards

### Python (API Gateway, ML Service)

```python
# Use Black formatter with line length 88
# Follow PEP 8 style guide
# Use type hints consistently

from typing import List, Optional, Dict, Any
import asyncio

async def process_cdr_data(
    cdrs: List[Dict[str, Any]], 
    threshold: float = 0.75
) -> Optional[Dict[str, Any]]:
    """Process CDR data for fraud detection.
    
    Args:
        cdrs: List of call detail records
        threshold: Fraud detection threshold
        
    Returns:
        Processed results or None if no fraud detected
    """
    # Implementation here
    pass
```

**Tools & Configuration:**
- **Formatter**: Black (line length: 88)
- **Linter**: pylint, flake8
- **Type Checker**: mypy
- **Import Sorter**: isort

### TypeScript/React (Frontend)

```typescript
// Use functional components with hooks
// Follow strict TypeScript configuration
// Implement proper error boundaries

interface FraudAlert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: Date;
  description: string;
}

const FraudAlertComponent: React.FC<{ alert: FraudAlert }> = ({ alert }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    <div className={`alert alert-${alert.severity}`}>
      {/* Component implementation */}
    </div>
  );
};
```

**Tools & Configuration:**
- **Formatter**: Prettier
- **Linter**: ESLint with TypeScript rules
- **Type Checker**: TypeScript strict mode
- **Testing**: Jest + React Testing Library

### Java (Flink Jobs)

```java
// Follow Google Java Style Guide
// Use meaningful variable and method names
// Implement proper error handling

public class FraudDetectionOperator extends ProcessFunction<CDR, FraudAlert> {
    
    private static final Logger LOG = LoggerFactory.getLogger(FraudDetectionOperator.class);
    private final double fraudThreshold;
    
    public FraudDetectionOperator(double fraudThreshold) {
        this.fraudThreshold = fraudThreshold;
    }
    
    @Override
    public void processElement(CDR cdr, Context context, Collector<FraudAlert> out) {
        // Implementation here
    }
}
```

**Tools & Configuration:**
- **Formatter**: Google Java Format
- **Linter**: Checkstyle
- **Static Analysis**: SpotBugs
- **Build Tool**: Maven with enforcer plugin

## 🧪 Testing Requirements

### Test Coverage Requirements

| Component | Minimum Coverage | Current Coverage |
|-----------|------------------|------------------|
| API Gateway | 85% | 92% ✅ |
| ML Service | 85% | 89% ✅ |
| Frontend | 80% | 87% ✅ |
| Flink Jobs | 85% | 91% ✅ |

### Testing Strategy

1. **Unit Tests**: Test individual functions/components
2. **Integration Tests**: Test service interactions
3. **E2E Tests**: Test complete user workflows
4. **Performance Tests**: Load and stress testing

### Running Tests

```bash
# All tests with coverage
make test

# Specific test suites
make test-unit
make test-integration
make test-e2e
make test-performance

# Watch mode for development
make test-watch
```

## 📝 Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `perf`: Performance improvements

### Examples

```bash
feat(api): add fraud detection endpoint with GraphSAGE integration

- Implement POST /api/v1/detect-fraud endpoint
- Add GraphSAGE model inference pipeline
- Include comprehensive input validation
- Add Prometheus metrics for monitoring

Closes #123
```

```bash
fix(frontend): resolve memory leak in graph visualization

The cytoscape.js instance wasn't being properly destroyed
when switching between dashboard views.

Fixes #456
```

## 📥 Pull Request Process

### Before Creating PR

1. **Rebase on latest develop**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout feature/your-feature
   git rebase develop
   ```

2. **Run full test suite**
   ```bash
   make test-all
   make lint-all
   ```

3. **Update documentation**
   - Update README if needed
   - Add/update API documentation
   - Update architecture diagrams

### PR Template

Use our PR template (automatically loaded from `.github/pull_request_template.md`):

```markdown
## 📝 Description
Brief description of changes and motivation.

## 🔗 Related Issues
Closes #123
Relates to #456

## 🧪 Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## 📋 Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### PR Review Process

1. **Automated Checks**: All CI/CD checks must pass
2. **Code Review**: At least 2 approvals required
3. **Testing**: QA approval for significant changes
4. **Documentation**: Technical writing review if needed

### Merge Strategy

- **Feature branches**: Squash and merge to develop
- **Release branches**: Merge commit to main
- **Hotfixes**: Cherry-pick to main and develop

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Monthly contributor spotlight

## 📞 Getting Help

- **General Questions**: GitHub Discussions
- **Bug Reports**: GitHub Issues
- **Security Issues**: security@fraudguard360.com
- **Chat**: Slack #fraudguard-dev channel

## 📚 Resources

- [Architecture Documentation](./docs/architecture.md)
- [API Reference](./docs/api.md)
- [Development Setup](./docs/development.md)
- [Deployment Guide](./docs/deployment.md)

---

Thank you for contributing to FraudGuard 360°! 🚀