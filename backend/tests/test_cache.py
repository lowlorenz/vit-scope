"""Tests for the ActivationCache."""
import pytest
import torch
from app.state.cache import ActivationCache
from app.engine.base import EngineConfig


def test_cache_initialization(cache):
    """Test that cache initializes empty."""
    assert cache is not None
    # Cache should be empty initially
    assert len(cache._cache) == 0


def test_generate_id(cache):
    """Test ID generation."""
    image_id = cache.generate_id()
    assert isinstance(image_id, str)
    assert len(image_id) > 0
    
    # Generate multiple IDs and ensure they're unique
    ids = [cache.generate_id() for _ in range(10)]
    assert len(set(ids)) == 10


def test_store_and_get(cache, mock_token_embeddings, mock_engine_config):
    """Test storing and retrieving activations."""
    image_id = cache.generate_id()
    
    cache.store(
        image_id=image_id,
        token_embeddings=mock_token_embeddings,
        engine_config=mock_engine_config,
        grid_h=27,
        grid_w=27,
        image_metadata={"test": "data"},
    )
    
    # Retrieve
    cached_data = cache.get(image_id)
    assert cached_data is not None
    assert cached_data["token_embeddings"].shape == mock_token_embeddings.shape
    assert cached_data["engine_config"] == mock_engine_config
    assert cached_data["grid_h"] == 27
    assert cached_data["grid_w"] == 27
    assert cached_data["image_metadata"]["test"] == "data"


def test_get_nonexistent(cache):
    """Test getting non-existent image returns None."""
    result = cache.get("nonexistent_id")
    assert result is None


def test_exists(cache, mock_token_embeddings, mock_engine_config):
    """Test exists check."""
    image_id = cache.generate_id()
    
    assert not cache.exists(image_id)
    
    cache.store(
        image_id=image_id,
        token_embeddings=mock_token_embeddings,
        engine_config=mock_engine_config,
        grid_h=27,
        grid_w=27,
    )
    
    assert cache.exists(image_id)


def test_multiple_images(cache, mock_token_embeddings, mock_engine_config):
    """Test storing multiple images."""
    image_ids = [cache.generate_id() for _ in range(5)]
    
    for i, image_id in enumerate(image_ids):
        cache.store(
            image_id=image_id,
            token_embeddings=mock_token_embeddings + i,  # Different embeddings
            engine_config=mock_engine_config,
            grid_h=27,
            grid_w=27,
        )
    
    # Verify all are stored
    for image_id in image_ids:
        assert cache.exists(image_id)
        cached_data = cache.get(image_id)
        assert cached_data is not None

