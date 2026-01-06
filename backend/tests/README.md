# Backend Tests

This directory contains comprehensive tests for the ViT Scope backend.

## Running Tests

From the project root:
```bash
cd backend
PYTHONPATH=/data/models/users/hufe/vit-scope/backend uv run --no-project pytest tests/ -v
```

Or using pytest directly (if dependencies are installed):
```bash
cd backend
pytest tests/ -v
```

## Test Structure

### `test_engine.py`
Tests for the SiglipEngine:
- Engine initialization
- Configuration access
- Image normalization (PIL and tensor inputs)
- Forward pass correctness
- Output shapes and types

### `test_cache.py`
Tests for the ActivationCache:
- Cache initialization
- ID generation (uniqueness)
- Store and retrieve operations
- Existence checks
- Multiple image handling

### `test_plugins.py`
Tests for plugins (specifically CosineTokensPlugin):
- Plugin properties
- Event handling
- Cosine similarity computation
- Edge cases (invalid events, missing grid dimensions, numpy inputs)
- Self-similarity validation

### `test_api.py`
Tests for API endpoints:
- Root endpoint
- Plugin listing
- Image upload (valid and invalid)
- Plugin event handling
- Error cases (non-existent images, plugins)
- Multiple image uploads

## Test Coverage

The test suite covers:
- ✅ Engine functionality (normalize, forward, config)
- ✅ Cache operations (store, get, exists, generate_id)
- ✅ Plugin system (cosine similarity computation)
- ✅ API endpoints (upload, plugin events, error handling)
- ✅ Edge cases and error conditions

## Fixtures

Common fixtures are defined in `conftest.py`:
- `sample_image`: Sample PIL Image for testing
- `engine`: SiglipEngine instance (CPU mode)
- `cache`: Fresh ActivationCache instance
- `cosine_plugin`: CosineTokensPlugin instance
- `mock_token_embeddings`: Mock token embeddings tensor
- `mock_activations`: Mock activations dictionary
- `mock_engine_config`: Mock engine configuration

## Notes

- Tests use CPU mode to avoid GPU requirements
- Model downloads happen automatically on first test run
- Tests are designed to be fast and isolated
- All tests should pass before merging code

