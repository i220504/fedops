# GitHub Actions Setup Checklist

## ✅ Pre-Deployment Checklist

### 1. Repository Configuration

- [ ] GitHub Actions is enabled
  - Go to: Settings → Actions → General
  - Select: "Allow all actions and reusable workflows"

- [ ] Branch protection rules set up
  - Go to: Settings → Branches → Add rule
  - Branch name pattern: `main`
  - Check: "Require a pull request before merging"
  - Check: "Require status checks to pass"
  - Select status checks: `code-quality`, `test`, `security`

### 2. Secrets Configuration (Optional but Recommended)

Navigate to: Settings → Secrets and variables → Actions

- [ ] `CODECOV_TOKEN` (for coverage reports)
  - Get from: https://codecov.io
  - Used for: Uploading test coverage

- [ ] `SLACK_WEBHOOK_URL` (for notifications)
  - Get from: Slack App Settings
  - Used for: Deployment notifications

- [ ] `DEPLOY_SSH_KEY` (if using SSH deployment)
  - Generate: `ssh-keygen -t ed25519 -C "github-actions"`
  - Used for: SSH deployment access

### 3. Environment Setup

#### Create Staging Environment
- [ ] Go to: Settings → Environments → New environment
- [ ] Name: `staging`
- [ ] Add environment variables:
  - `DEPLOY_URL`: `https://staging.fedops.example.com`

#### Create Production Environment
- [ ] Go to: Settings → Environments → New environment
- [ ] Name: `production`
- [ ] Add environment variables:
  - `DEPLOY_URL`: `https://fedops.example.com`
- [ ] Enable: "Required reviewers" (add team members)
- [ ] Enable: "Wait timer" (optional, e.g., 5 minutes)

### 4. Workflow Customization

- [ ] Update deployment scripts in `.github/workflows/ci-cd.yml`
  - Lines 266-282 (staging deployment)
  - Lines 284-305 (production deployment)

- [ ] Update image registry if not using GHCR
  - Current: `ghcr.io`
  - Alternative: `docker.io`, `registry.gitlab.com`, etc.

- [ ] Update notification endpoints (if using)
  - Lines 313-333 in `ci-cd.yml`

### 5. Documentation Updates

- [ ] Update repository URLs in documentation
  - Replace `<username>/<repo>` with actual values
  - Files: `README.md`, `docs/CI_CD_GUIDE.md`, `docs/QUICK_START.md`

- [ ] Update deployment URLs
  - Replace example URLs with actual URLs
  - Files: `docs/CI_CD_GUIDE.md`

### 6. Local Testing

- [ ] Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

- [ ] Run tests locally
  ```bash
  pytest tests/ -v
  ```

- [ ] Run code quality checks
  ```bash
  black --check myapp/ app.py
  flake8 myapp/ app.py
  ```

- [ ] Build Docker images locally
  ```bash
  docker build -f Dockerfile -t fedops-audio:test .
  docker build -f Dockerfile.ci -t fedops-audio:ci .
  ```

### 7. First Push

- [ ] Stage all changes
  ```bash
  git add .
  ```

- [ ] Commit with conventional commit message
  ```bash
  git commit -m "feat: add CI/CD pipeline with GitHub Actions"
  ```

- [ ] Push to trigger pipeline
  ```bash
  git push origin main
  ```

- [ ] Verify workflow runs
  - Go to: Actions tab
  - Check: All jobs complete successfully

### 8. Post-Deployment

- [ ] Add status badges to README
  ```markdown
  ![CI/CD](https://github.com/oscerpk/fedops-audio-flwr/actions/workflows/ci-cd.yml/badge.svg)
  ```

- [ ] Configure Dependabot (optional)
  - Go to: Settings → Security → Dependabot
  - Enable: "Dependabot alerts"
  - Enable: "Dependabot security updates"

- [ ] Set up code scanning (optional)
  - Go to: Security → Code scanning
  - Enable: "CodeQL analysis"

- [ ] Configure notifications
  - Go to: Settings → Notifications
  - Set up email/webhook notifications

## 🔍 Verification Steps

### After First Push

- [ ] All workflow jobs completed successfully
- [ ] Docker images pushed to registry
- [ ] No security vulnerabilities found
- [ ] Test coverage reports generated
- [ ] Artifacts uploaded successfully

### Test Pull Request Flow

- [ ] Create test branch
  ```bash
  git checkout -b test/pr-flow
  echo "# Test" >> test.md
  git add test.md
  git commit -m "test: verify PR checks"
  git push origin test/pr-flow
  ```

- [ ] Create pull request
- [ ] Verify PR checks run automatically
- [ ] Verify coverage comment posted (if Codecov configured)
- [ ] Merge PR and verify main branch workflow

### Test Release Flow

- [ ] Create test tag
  ```bash
  git tag -a v0.1.0-test -m "Test release"
  git push origin v0.1.0-test
  ```

- [ ] Verify release workflow triggers
- [ ] Verify GitHub release created
- [ ] Verify production deployment (if configured)
- [ ] Delete test tag after verification
  ```bash
  git tag -d v0.1.0-test
  git push origin :refs/tags/v0.1.0-test
  ```

## 🚨 Troubleshooting

### Workflow Fails

- [ ] Check workflow logs in Actions tab
- [ ] Verify all secrets are configured correctly
- [ ] Check Docker Hub/GHCR access tokens
- [ ] Verify deployment endpoints are accessible

### Docker Build Fails

- [ ] Check Dockerfile syntax
- [ ] Verify base image is accessible
- [ ] Check for sufficient runner disk space
- [ ] Review build logs for specific errors

### Tests Fail

- [ ] Run tests locally with same Python version
- [ ] Check for missing system dependencies
- [ ] Verify test data/fixtures are present
- [ ] Review test logs for specific failures

### Deployment Fails

- [ ] Verify deployment credentials are valid
- [ ] Check network connectivity to deployment target
- [ ] Verify deployment scripts have correct permissions
- [ ] Review deployment logs

## 📚 Additional Resources

- [ ] Read [docs/CI_CD_GUIDE.md](docs/CI_CD_GUIDE.md)
- [ ] Read [docs/QUICK_START.md](docs/QUICK_START.md)
- [ ] Review [GitHub Actions Documentation](https://docs.github.com/actions)
- [ ] Join Flower community for support

## ✅ Completion

When all items are checked:

✨ **Your CI/CD pipeline is ready to use!** ✨

---

**Last Updated**: 2024-11-30
**Version**: 1.0.0
