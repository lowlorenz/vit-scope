"""Pytest configuration and fixtures."""
import pytest
import torch
from PIL import Image
import numpy as np
from app.engine.siglip_engine import SiglipEngine
from app.state.cache import ActivationCache
from app.plugins.cosine_tokens import CosineTokensPlugin


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    # Create a simple 384x384 RGB image
    img_array = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def engine():
    """Create a SiglipEngine instance for testing."""
    # Use CPU for testing to avoid GPU requirements
    return SiglipEngine(device="cpu")


@pytest.fixture
def cache():
    """Create a fresh ActivationCache instance for testing."""
    return ActivationCache()


@pytest.fixture
def cosine_plugin():
    """Create a CosineTokensPlugin instance for testing."""
    return CosineTokensPlugin()


@pytest.fixture
def mock_token_embeddings():
    """Create mock token embeddings for testing."""
    # Create embeddings for a 27x27 grid (729 tokens) with 1152 dimensions
    num_tokens = 729
    hidden_size = 1152
    return torch.randn(num_tokens, hidden_size)


@pytest.fixture
def mock_activations(mock_token_embeddings):
    """Create mock activations dictionary."""
    from app.engine.base import EngineConfig
    
    return {
        "token_embeddings": mock_token_embeddings,
        "grid_h": 27,
        "grid_w": 27,
    }


@pytest.fixture
def mock_engine_config():
    """Create mock engine config."""
    from app.engine.base import EngineConfig
    return EngineConfig(patch_size=14, img_size=384)

