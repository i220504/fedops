# myapp: A Flower / PyTorch app

[![CI/CD Pipeline](https://github.com/oscerpk/fedops-audio-flwr/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/oscerpk/fedops-audio-flwr/actions/workflows/ci-cd.yml)
[![Code Quality](https://github.com/oscerpk/fedops-audio-flwr/actions/workflows/pr-checks.yml/badge.svg)](https://github.com/oscerpk/fedops-audio-flwr/actions/workflows/pr-checks.yml)

A federated learning application for audio classification using Flower framework and PyTorch, with comprehensive CI/CD pipeline.

## 🚀 Quick Start

### Installation

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

> **Tip:** Your `pyproject.toml` file can define more than just the dependencies of your Flower app. You can also use it to specify hyperparameters for your runs and control which Flower Runtime is used. By default, it uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

## Run with the Simulation Engine

In the `myapp` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## 🔄 CI/CD Pipeline

This project includes a comprehensive CI/CD pipeline with:

- ✅ **Automated Testing**: Unit and integration tests
- ✅ **Code Quality**: Black, Flake8, isort, Pylint, MyPy
- ✅ **Security Scanning**: Bandit, Safety, Trivy
- ✅ **Docker Builds**: Multi-stage builds with caching
- ✅ **Deployments**: Staging and production environments

### Quick Setup

See [docs/QUICK_START.md](docs/QUICK_START.md) for 5-minute setup guide.

### Full Documentation

See [docs/CI_CD_GUIDE.md](docs/CI_CD_GUIDE.md) for comprehensive documentation.

### Testing Locally

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=myapp --cov-report=html

# Code quality
black myapp/ app.py
flake8 myapp/ app.py
```

## 📦 Docker

Build and run with Docker:

```bash
# Production
docker build -f Dockerfile -t fedops-audio:prod .
docker run -p 5002:5002 fedops-audio:prod

# Development
docker-compose up
```

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
