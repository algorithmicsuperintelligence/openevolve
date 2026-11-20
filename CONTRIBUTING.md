# Contributing to OpenEvolve

Thank you for your interest in contributing to OpenEvolve! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/codelion/openevolve.git`
3. Install the package in development mode: `pip install -e ".[dev]"`
4. Set up environment for testing:
   ```bash
   # Unit tests don't require a real API key, but the environment variable must be set
   export OPENAI_API_KEY=test-key-for-unit-tests
   ```
5. Run the tests to ensure everything is working: `python -m unittest discover tests`

**Note**: The unit tests do not make actual API calls to OpenAI or any LLM provider. However, the `OPENAI_API_KEY` environment variable must be set to any non-empty value for the tests to run. You can use a placeholder value like `test-key-for-unit-tests`.

## Development Environment

We recommend using a virtual environment for development:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -e ".[dev]"

# For running tests (no actual API calls are made)
export OPENAI_API_KEY=test-key-for-unit-tests

# For testing with real LLMs during development
# export OPENAI_API_KEY=your-actual-api-key
```

### LLM Configuration for Development

When developing features that interact with LLMs:

1. **Local Development**: Use a mock API key for unit tests
2. **Integration Testing**: Use your actual API key and configure `api_base` if using alternative providers
3. **Cost Management**: Consider using cheaper models or [optillm](https://github.com/codelion/optillm) for rate limiting during development

### Pre-commit Checks

Before opening a PR, install and run the configured pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

These hooks catch common issues early, including malformed YAML, accidental private key commits, trailing whitespace, and overly large files.

## Pull Request Process

1. Create a new branch for your feature or bugfix: `git checkout -b feat-your-feature-name`
2. Make your changes
3. Add tests for your changes
4. Run the tests to make sure everything passes:
   ```bash
   export OPENAI_API_KEY=test-key-for-unit-tests
   python -m unittest discover tests
   ```
5. Commit your changes: `git commit -m "Add your descriptive commit message"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Submit a pull request to the main repository

## Adding Examples

We encourage adding new examples to showcase OpenEvolve's capabilities. To add a new example:

1. Create a new directory in the `examples` folder
2. Include all necessary files (initial program, evaluation code, etc.)
3. Add a README.md explaining the example
4. Make sure the example can be run with minimal setup

## Reporting Issues

When reporting issues, please include:

1. A clear description of the issue
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, etc.)

## Feature Requests

Feature requests are welcome. Please provide:

1. A clear description of the feature
2. The motivation for adding this feature
3. Possible implementation ideas (if any)

## Code of Conduct

Please be respectful and considerate of others when contributing to the project. We aim to create a welcoming and inclusive environment for all contributors.
