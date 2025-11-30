# FedOps Audio - Quick Start Guide for CI/CD

## 🚀 Quick Setup (5 minutes)

### 1. Enable GitHub Actions

GitHub Actions should be enabled by default. If not:
1. Go to your repository settings
2. Navigate to `Actions` → `General`
3. Ensure "Allow all actions and reusable workflows" is selected

### 2. Configure Secrets (Optional but Recommended)

Navigate to: **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

#### Optional Secrets:
- `CODECOV_TOKEN`: For coverage reports (get from https://codecov.io)
- `SLACK_WEBHOOK_URL`: For deployment notifications

### 3. Push Your Code

```bash
git add .
git commit -m "feat: add CI/CD pipeline"
git push origin main
```

The CI/CD pipeline will automatically run! 🎉

---

## 📋 Pre-Push Checklist

Before pushing code, run these commands locally:

```bash
# 1. Format code
black myapp/ app.py

# 2. Sort imports
isort myapp/ app.py

# 3. Lint code
flake8 myapp/ app.py

# 4. Run tests
pytest tests/ -v

# 5. Check coverage
pytest tests/ --cov=myapp --cov-report=term
```

---

## 🔧 Common Tasks

### Run Tests Locally
```bash
pytest tests/ -v --cov=myapp
```

### Build Docker Image Locally
```bash
docker build -f Dockerfile -t fedops-audio:local .
```

### Format All Code
```bash
black myapp/ app.py
isort myapp/ app.py
```

### Security Check
```bash
bandit -r myapp/
safety check
```

---

## 🌿 Branch Strategy

```
main (production)
  ↑
develop (staging)
  ↑
feat/your-feature (feature branches)
```

### Working on a Feature:
```bash
# 1. Create feature branch from develop
git checkout develop
git pull
git checkout -b feat/audio-preprocessing

# 2. Make changes and commit
git add .
git commit -m "feat: add audio preprocessing"

# 3. Push and create PR
git push origin feat/audio-preprocessing
```

---

## 📦 Release Process

### Creating a Release:

```bash
# 1. Update version in pyproject.toml
# 2. Create and push tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# GitHub Actions will automatically:
# - Run all tests
# - Build Docker images
# - Create GitHub Release
# - Deploy to production (after approval)
```

---

## 🎯 CI/CD Pipeline Stages

```
┌──────────────┐
│ Pull Request │
└──────┬───────┘
       │
       ├─► Code Quality ✓
       ├─► Security Scan ✓
       └─► Unit Tests ✓
       
┌──────────────┐
│  Push/Merge  │
└──────┬───────┘
       │
       ├─► All PR Checks ✓
       ├─► Build Docker ✓
       ├─► Container Scan ✓
       └─► Deploy 🚀
```

---

## 🐛 Troubleshooting

### Pipeline Failing?

1. **Check the Actions tab** in GitHub
2. **View the logs** of failed jobs
3. **Common fixes:**
   - Format code: `black myapp/ app.py`
   - Fix imports: `isort myapp/ app.py`
   - Run tests: `pytest tests/`

### Docker Build Failing?

```bash
# Test locally
docker build -f Dockerfile .

# Check Docker daemon
docker ps

# Clean cache
docker builder prune
```

### Tests Failing in CI?

```bash
# Run tests with same Python version as CI
python --version  # Should be 3.10 or 3.11

# Install exact dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

---

## 📊 Monitoring

### View Pipeline Status
- Go to **Actions** tab in GitHub
- Check workflow runs
- View logs and artifacts

### Add Status Badge to README
```markdown
![CI/CD](https://github.com/oscerpk/fedops-audio-flwr/actions/workflows/ci-cd.yml/badge.svg)
```

---

## ✅ Best Practices

### Commit Messages
Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring

Examples:
```bash
git commit -m "feat: add voice activity detection"
git commit -m "fix(client): resolve timeout issue"
git commit -m "docs: update API documentation"
```

### Pull Requests
- Keep PRs small (< 500 lines)
- Write clear descriptions
- Link related issues
- Request reviews

---

## 🔒 Security

### Never Commit:
- ❌ API keys
- ❌ Passwords
- ❌ Private keys
- ❌ .env files

### Always Use:
- ✅ GitHub Secrets
- ✅ Environment variables
- ✅ .gitignore

---

## 📚 Resources

- **Full Documentation**: See [docs/CI_CD_GUIDE.md](CI_CD_GUIDE.md)
- **GitHub Actions**: https://docs.github.com/actions
- **Flower Docs**: https://flower.ai/docs/
- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/

---

## 🆘 Need Help?

1. Check [docs/CI_CD_GUIDE.md](CI_CD_GUIDE.md) for detailed documentation
2. Open an issue in GitHub
3. Contact the DevOps team

---

**Happy Coding! 🎉**
