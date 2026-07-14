# Integration Tests

This directory contains integration tests for OpenEvolve. Tests are organized into two categories:

## Fast Tests (CI)

**Smoke tests** that run in CI to validate basic functionality without requiring slow LLM inference:

```bash
# Run only fast tests (for CI)
pytest tests/integration/ -m "not slow"
```

These tests:
- Complete in <10 seconds total
- Test core API validation, configuration, and basic component initialization
- No real LLM calls required

## Slow Tests (Local Development)

**Full integration tests** with real LLM inference for comprehensive validation:

```bash
# Run all tests including slow ones (for local development)
pytest tests/integration/

# Run only slow tests
pytest tests/integration/ -m "slow"
```

These tests:
- Take ~30-60 minutes to complete
- Use a real optillm server with the codelion/dhara-250m model
- Test complete evolution pipelines, checkpointing, island migration, etc.
- Require optillm server running on localhost:8000

## Test Setup

For slow tests that require LLM inference:

1. **Install optillm**: `pip install optillm`
2. **Start server**: `OPTILLM_API_KEY=optillm optillm --model google/gemma-3-270m-it --port 8000`
3. **Set environment**: `export OPTILLM_API_KEY=optillm OPENAI_API_KEY=optillm`
4. **Run tests**: `pytest tests/integration/ -m "slow"`

## Configuration

All integration tests use:
- **0 retries** for fast failure
- **120 second timeout** per LLM call
- **In-memory database** for speed
- **Small iteration counts** (1-8 iterations) for CI compatibility

## CI Behavior

GitHub Actions will:
- Run **fast tests only** (`-m "not slow"`) 
- Complete in <30 seconds
- Validate core functionality without requiring model downloads
- Skip all tests marked with `@pytest.mark.slow`