# FedOps Audio - CI/CD Documentation

## Overview

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the FedOps Audio Federated Learning project. The pipeline is implemented using GitHub Actions and provides automated testing, building, security scanning, and deployment.

## Pipeline Architecture

```
┌─────────────┐
│  Git Push   │
│  /PR/Tag    │
└──────┬──────┘
       │
       ├──────────────────┬──────────────────┬────────────────┐
       ▼                  ▼                  ▼                ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐
│Code Quality │  │   Security   │  │    Tests     │  │ Build  │
│   Checks    │  │   Scanning   │  │   (Unit +    │  │ Docker │
│             │  │              │  │ Integration) │  │ Images │
└──────┬──────┘  └──────┬───────┘  └──────┬───────┘  └───┬────┘
       │                │                  │              │
       └────────────────┴──────────────────┴──────────────┘
                                   │
                                   ▼
                        ┌────────────────────┐
                        │  Container Scan    │
                        │   (Trivy/Grype)    │
                        └──────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            ┌──────────────┐           ┌──────────────────┐
            │   Staging    │           │   Production     │
            │  Deployment  │           │   Deployment     │
            │ (develop br.)│           │  (version tags)  │
            └──────────────┘           └──────────────────┘
```

## Workflows

### 1. Main CI/CD Pipeline (`ci-cd.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Version tags (e.g., `v1.0.0`)
- Manual workflow dispatch

**Jobs:**

#### a. Code Quality Checks
- **Black**: Code formatting verification
- **isort**: Import statement organization
- **Flake8**: PEP8 compliance and linting
- **Pylint**: Advanced static analysis
- **MyPy**: Type checking

**Configuration:** See `.flake8` and `pyproject.toml`

#### b. Security Scanning
- **Bandit**: Python security vulnerability scanner
- **Safety**: Dependency vulnerability checker
- Generates JSON reports uploaded as artifacts

#### c. Testing
- **Matrix testing** across Python 3.10 and 3.11
- **pytest** with coverage reporting
- Test types:
  - Unit tests (`tests/test_model.py`)
  - Integration tests (`tests/test_flower_integration.py`)
- **Coverage reports** uploaded to Codecov

#### d. Docker Image Building
- Multi-stage builds for:
  - Production image (`Dockerfile`)
  - CI image (`Dockerfile.ci`)
  - Development image (`Dockerfile.dev`)
- Images pushed to GitHub Container Registry (GHCR)
- **Tagging strategy:**
  - `latest` - latest from main branch
  - `<branch>-<sha>` - branch-specific builds
  - `v1.0.0` - semantic version tags
  - `ci-<sha>` - CI-specific images

#### e. Container Security Scanning
- **Trivy** vulnerability scanner
- Results uploaded to GitHub Security tab
- Scans for:
  - OS vulnerabilities
  - Python package vulnerabilities
  - Misconfigurations

#### f. Deployment
- **Staging**: Automatic deployment on `develop` branch push
- **Production**: Manual approval required for version tags
- Uses GitHub Environments for protection rules

### 2. Pull Request Checks (`pr-checks.yml`)

**Triggers:**
- Pull request opened, synchronized, or reopened

**Jobs:**

#### a. PR Validation
- Validates PR title follows conventional commits format
- Examples:
  - ✅ `feat: add new audio preprocessing`
  - ✅ `fix(client): resolve connection timeout`
  - ❌ `updated code`

#### b. Automated Code Review
- Quick formatting and linting checks
- Provides warnings for code quality issues

#### c. Size Check
- Warns if PR changes more than 500 lines
- Encourages smaller, focused PRs

#### d. Test Coverage Report
- Runs tests with coverage
- Comments coverage report on PR

### 3. Legacy CI Workflow (`ci.yml`)

**Status:** Deprecated (replaced by `ci-cd.yml`)

Basic workflow that builds CI Docker image and runs import tests.

## Configuration Files

### `.flake8`
```ini
[flake8]
max-line-length = 100
ignore = E203, E501, W503
```

### `pyproject.toml`
Contains configuration for:
- Black formatter
- isort import sorter
- pytest test runner
- coverage.py coverage tool
- mypy type checker
- bandit security scanner

### `.dockerignore`
Excludes unnecessary files from Docker builds:
- Tests and test artifacts
- Documentation
- Development tools
- Large datasets
- CI/CD configurations

## GitHub Secrets Required

Configure these secrets in your GitHub repository settings:

### Required Secrets

1. **`GITHUB_TOKEN`** (automatically provided)
   - Used for GHCR authentication and API access

### Optional Secrets

2. **`CODECOV_TOKEN`**
   - For uploading coverage reports to Codecov
   - Get from: https://codecov.io

3. **`SLACK_WEBHOOK_URL`**
   - For deployment notifications
   - Create in Slack app settings

4. **`DEPLOY_SSH_KEY`** (if using SSH deployment)
   - SSH private key for deployment server access

5. **`KUBE_CONFIG`** (if deploying to Kubernetes)
   - Kubernetes cluster configuration

## Environment Variables

Configure in GitHub repository settings under Environments:

### Staging Environment
```yaml
- DEPLOY_URL: https://staging.fedops.example.com
- DOCKER_REGISTRY: ghcr.io
```

### Production Environment
```yaml
- DEPLOY_URL: https://fedops.example.com
- DOCKER_REGISTRY: ghcr.io
- REQUIRE_APPROVALS: true (recommended)
```

## Docker Image Naming Convention

Images are pushed to GitHub Container Registry with the following naming:

```
ghcr.io/<username>/fedops-audio-flwr:<tag>
```

**Available tags:**
- `latest` - Latest stable build from main
- `develop` - Latest from develop branch
- `v1.2.3` - Semantic version release
- `main-abc1234` - Commit-specific build
- `ci-abc1234` - CI-specific build
- `dev-abc1234` - Development build

## Running Tests Locally

### Install test dependencies:
```bash
pip install -r requirements.txt
```

### Run all tests:
```bash
pytest tests/ -v
```

### Run with coverage:
```bash
pytest tests/ --cov=myapp --cov-report=html
```

### View coverage report:
```bash
open htmlcov/index.html
```

## Code Quality Checks Locally

### Format code:
```bash
black myapp/ app.py
isort myapp/ app.py
```

### Lint code:
```bash
flake8 myapp/ app.py
pylint myapp/
```

### Type checking:
```bash
mypy myapp/
```

### Security scanning:
```bash
bandit -r myapp/
safety check
```

## Deployment Process

### Staging Deployment (Automatic)

1. Merge PR to `develop` branch
2. CI/CD pipeline runs automatically
3. On success, deploys to staging environment
4. Staging URL: `https://staging.fedops.example.com`

### Production Deployment (Manual)

1. Create a release branch from `main`:
   ```bash
   git checkout main
   git pull
   git checkout -b release/v1.0.0
   ```

2. Update version in `pyproject.toml`

3. Create and push tag:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

4. GitHub Actions workflow triggers
5. Requires manual approval in GitHub Environments
6. On approval, deploys to production

## Monitoring CI/CD Pipeline

### View Workflow Runs
- Navigate to: `Actions` tab in GitHub repository
- View status, logs, and artifacts

### Check Build Status Badge
Add to README.md:
```markdown
![CI/CD Pipeline](https://github.com/oscerpk/fedops-audio-flwr/actions/workflows/ci-cd.yml/badge.svg)
```

### Download Artifacts
- Security reports (Bandit, Safety)
- Coverage reports
- Test results

## Troubleshooting

### Common Issues

#### 1. Docker build fails
- **Check**: Dockerfile syntax
- **Check**: Base image availability
- **Solution**: Run locally: `docker build -f Dockerfile .`

#### 2. Tests fail in CI but pass locally
- **Check**: Python version compatibility (CI uses 3.10 and 3.11)
- **Check**: Missing system dependencies
- **Solution**: Add dependencies to workflow's `apt-get install` step

#### 3. Coverage too low
- **Check**: Test completeness
- **Solution**: Add more unit tests
- Coverage target: 80%+

#### 4. Security vulnerabilities found
- **Check**: Bandit and Safety reports in artifacts
- **Solution**: Update dependencies or fix code issues

#### 5. Container scan fails
- **Check**: Trivy scan results
- **Solution**: Update base image or vulnerable packages

## Best Practices

### For Contributors

1. **Always create feature branches**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Follow conventional commits**
   - `feat:` New features
   - `fix:` Bug fixes
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring

3. **Write tests for new code**
   - Unit tests for functions/classes
   - Integration tests for workflows

4. **Run quality checks before pushing**
   ```bash
   black myapp/ app.py
   flake8 myapp/ app.py
   pytest tests/
   ```

5. **Keep PRs small and focused**
   - Target: < 500 lines changed
   - Single responsibility

### For Maintainers

1. **Review PR checks before merging**
   - All tests passing
   - Coverage maintained/improved
   - No security issues

2. **Use protected branches**
   - Require PR reviews
   - Require status checks to pass

3. **Regular dependency updates**
   - Use Dependabot
   - Review security advisories

4. **Monitor deployment health**
   - Check Prometheus/Grafana dashboards
   - Review application logs

## Advanced Configuration

### Customize Deployment

Edit `.github/workflows/ci-cd.yml` deployment jobs:

```yaml
- name: Deploy to production
  run: |
    # Add your deployment commands
    kubectl apply -f k8s/
    # Or use docker-compose
    docker-compose up -d
    # Or SSH deployment
    ssh user@server 'cd /app && docker-compose pull && docker-compose up -d'
```

### Add Kubernetes Deployment

```yaml
- name: Deploy to Kubernetes
  run: |
    kubectl config use-context production
    kubectl set image deployment/fedops \
      fedops=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
    kubectl rollout status deployment/fedops
```

### Add Notification Integrations

```yaml
- name: Notify Microsoft Teams
  uses: aliencube/microsoft-teams-actions@v0.8.0
  with:
    webhook_uri: ${{ secrets.TEAMS_WEBHOOK_URL }}
    title: Deployment Complete
    summary: Version ${{ github.ref }} deployed to production
```

## Performance Optimization

### Cache Strategies

**Python dependencies caching:**
```yaml
- uses: actions/setup-python@v5
  with:
    python-version: '3.11'
    cache: 'pip'
```

**Docker layer caching:**
```yaml
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Parallel Job Execution

Jobs run in parallel when possible:
- Code quality + Security + Tests (parallel)
- Build only after tests pass
- Deploy only after build succeeds

## Security Considerations

1. **Never commit secrets**
   - Use GitHub Secrets
   - Use environment variables

2. **Scan containers regularly**
   - Trivy runs on every build
   - Review security tab

3. **Keep dependencies updated**
   - Monitor Dependabot alerts
   - Review Safety reports

4. **Use signed commits** (recommended)
   ```bash
   git config --global commit.gpgsign true
   ```

5. **Implement branch protection**
   - Require reviews
   - Require signed commits
   - No force pushes

## Support and Resources

- **GitHub Actions Documentation**: https://docs.github.com/actions
- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/
- **Flower Framework**: https://flower.ai/docs/
- **Project Issues**: https://github.com/oscerpk/fedops-audio-flwr/issues

## Changelog

- **2024-11-30**: Initial CI/CD pipeline implementation
  - Added comprehensive testing
  - Added security scanning
  - Added multi-environment deployment
  - Added PR checks

---

For questions or issues with CI/CD, please open an issue or contact the DevOps team.
