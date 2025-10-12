# GitHub Security Configuration

This document outlines the security configuration and setup for FraudGuard 360°.

## Security Workflows

### 1. CodeQL Analysis (`codeql.yml`)
- **Purpose:** Static application security testing (SAST)
- **Languages:** Java, JavaScript/TypeScript, Python
- **Schedule:** Weekly on Mondays at 6:24 PM UTC
- **Triggers:** Push to main/develop, Pull requests
- **Query Suites:** security-extended, security-and-quality

### 2. Comprehensive Security Scanning (`security-comprehensive.yml`)
- **Purpose:** Multi-layered security analysis
- **Features:**
  - CodeQL SAST analysis
  - Dependency vulnerability scanning (Safety, npm audit, OWASP)
  - Secret scanning (TruffleHog)
  - Container security scanning (Trivy)
  - Infrastructure security (Checkov)
- **Schedule:** Weekly on Mondays at 2 AM UTC

### 3. Dependency Management (`dependabot.yml` + `dependabot-automerge.yml`)
- **Purpose:** Automated security updates
- **Ecosystems:** Python (pip), Node.js (npm), Java (Maven), Docker, GitHub Actions
- **Auto-merge:** Security patches and minor updates
- **Manual review:** Major version updates

## Security Policies

### 1. Branch Protection Rules
Recommended settings for `main` branch:
```yaml
protection_rules:
  required_status_checks:
    strict: true
    contexts:
      - "CodeQL Analysis"
      - "Security Scanning"
      - "CI/CD Pipeline"
  enforce_admins: true
  required_pull_request_reviews:
    required_approving_review_count: 1
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
  restrictions: null
```

### 2. Repository Security Settings
Enable these security features in repository settings:

#### Code Security and Analysis
- [x] **Dependency graph** - Track repository dependencies
- [x] **Dependabot alerts** - Security vulnerability alerts
- [x] **Dependabot security updates** - Automatic security fixes
- [x] **Dependabot version updates** - Keep dependencies up to date
- [x] **Code scanning** - CodeQL analysis for vulnerabilities
- [x] **Secret scanning** - Detect secrets in commits
- [x] **Secret scanning push protection** - Block commits with secrets

#### Advanced Security (GitHub Enterprise)
- [x] **Secret scanning for partner patterns**
- [x] **Push protection for secret scanning**
- [x] **Code scanning autofix**

### 3. Environment Protection Rules

#### Development Environment
- No restrictions
- Automatic deployments

#### Staging Environment  
- Required reviewers: 1
- Wait timer: 5 minutes
- Deployment protection rules

#### Production Environment
- Required reviewers: 2 (including code owner)
- Wait timer: 30 minutes
- Deployment window restrictions
- Branch restrictions: `main` only

## Security Scanning Results

### Expected Security Scan Outputs

1. **CodeQL Analysis Results**
   - Location: Security tab → Code scanning alerts
   - Format: SARIF files uploaded to GitHub
   - Categories: Security vulnerabilities, Quality issues

2. **Dependency Scanning Results**
   - Python: Safety JSON reports
   - Node.js: npm audit reports  
   - Java: OWASP Dependency Check reports
   - Location: Security tab → Dependabot alerts

3. **Secret Scanning Results**
   - TruffleHog scan results
   - GitHub native secret scanning
   - Location: Security tab → Secret scanning alerts

4. **Container Security Results**
   - Trivy vulnerability reports
   - Base image security analysis
   - Location: Actions artifacts, Security tab

## Monitoring and Alerting

### 1. Security Alert Notifications
Configure notifications for:
- High/Critical security vulnerabilities
- Failed security scans
- New Dependabot alerts
- Secret scanning detections

### 2. Metrics to Monitor
- **Security scan pass rate** (target: 100%)
- **Time to resolve critical vulnerabilities** (target: < 24 hours)
- **Dependency update frequency** (target: weekly)
- **Code coverage with security tests** (target: > 80%)

## Troubleshooting

### Common CodeQL Issues

1. **Build Failures in CodeQL**
   ```bash
   # Java build issues
   cd stream-processor-flink && mvn clean compile -DskipTests
   cd flink-jobs && mvn clean compile -DskipTests
   
   # Node.js build issues
   cd frontend && npm ci && npm run build
   ```

2. **Memory Issues**
   - Increase runner memory allocation
   - Split analysis across multiple jobs
   - Use matrix strategy for large codebases

3. **Language Detection Issues**
   - Explicitly specify languages in workflow
   - Check file extensions and patterns
   - Verify source code locations

### Dependency Scanning Issues

1. **False Positives**
   - Review and dismiss false positives
   - Use .safety-policy.yml for Python
   - Configure npm audit ignore patterns

2. **Network Issues**
   - Cache dependency downloads
   - Use offline vulnerability databases
   - Configure proxy settings if needed

## Security Best Practices

### 1. Code Development
- Never commit secrets or credentials
- Use parameterized queries to prevent SQL injection
- Validate all input data
- Implement proper error handling
- Use secure coding standards

### 2. Dependency Management
- Regularly update dependencies
- Review dependency licenses
- Pin dependency versions in production
- Use lockfiles (package-lock.json, requirements.txt)

### 3. Infrastructure Security
- Use minimal base images
- Regular container image updates
- Implement network segmentation
- Enable logging and monitoring
- Use encrypted communication

### 4. Access Control
- Implement principle of least privilege
- Use multi-factor authentication
- Regular access reviews
- Secure service accounts

## Compliance and Reporting

### 1. Security Reporting
- Weekly security scan summaries
- Monthly vulnerability trend reports
- Quarterly security posture assessments
- Annual penetration testing

### 2. Compliance Standards
- OWASP Top 10 compliance
- NIST Cybersecurity Framework alignment
- SOC 2 Type II controls
- GDPR data protection measures

---

**Last Updated:** October 2025
**Next Review:** January 2026
**Owner:** Security Team