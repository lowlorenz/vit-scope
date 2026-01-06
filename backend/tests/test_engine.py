"""Tests for the SiglipEngine."""

import pytest
import torch
from PIL import Image
import numpy as np
from app.engine.siglip_engine import SiglipEngine
from app.engine.base import EngineConfig


def test_engine_initialization(engine):
    """Test that engine initializes correctly."""
    assert engine is not None
    assert engine.device in ["cpu", "cuda"]
    assert engine.processor is not None
    assert engine.model is not None


def test_engine_config(engine):
    """Test engine configuration."""
    config = engine.config
    assert isinstance(config, EngineConfig)
    assert config.patch_size == 14
    assert config.img_size == 384


def test_normalize_pil_image(engine, sample_image):
    """Test normalization with PIL Image."""
    normalized = engine.normalize(sample_image)

    assert isinstance(normalized, torch.Tensor)
    assert normalized.dim() == 4  # [batch, channels, height, width]
    assert normalized.shape[0] == 1  # batch size
    assert normalized.shape[1] == 3  # RGB channels
    assert normalized.shape[2] == 384  # height
    assert normalized.shape[3] == 384  # width


def test_normalize_tensor(engine):
    """Test normalization with tensor input."""
    # Create a tensor image [C, H, W]
    img_tensor = torch.randint(0, 255, (3, 384, 384), dtype=torch.uint8).float()
    normalized = engine.normalize(img_tensor)

    assert isinstance(normalized, torch.Tensor)
    assert normalized.dim() == 4
    assert normalized.shape[0] == 1


def test_forward_pass(engine, sample_image):
    """Test forward pass returns correct shapes."""
    normalized = engine.normalize(sample_image)
    logits, activations = engine.forward(normalized)

    # Check logits (for SigLIP, this is the pooled embedding)
    assert isinstance(logits, torch.Tensor)
    assert logits.dim() == 2  # [batch, hidden_size]
    assert logits.shape[0] == 1
    assert logits.shape[1] > 0  # Should have embedding dimension

    # Check activations
    assert isinstance(activations, dict)
    assert "token_embeddings" in activations
    assert "hidden_states" in activations

    token_embeddings = activations["token_embeddings"]
    assert isinstance(token_embeddings, torch.Tensor)
    assert token_embeddings.dim() == 2  # [num_tokens, hidden_size]

    # Check hidden_states
    hidden_states = activations["hidden_states"]
    assert isinstance(hidden_states, list)
    assert len(hidden_states) > 0  # Should have at least embeddings + layers

    # For SigLIP: 1 embedding + 27 layers = 28 total
    assert len(hidden_states) == 28

    # Check each hidden state shape
    for i, hidden in enumerate(hidden_states):
        assert isinstance(hidden, torch.Tensor)
        assert hidden.dim() == 2  # [num_tokens, hidden_size]
        assert hidden.shape == token_embeddings.shape  # Same shape for all layers

    # Final layer should match token_embeddings
    assert torch.allclose(token_embeddings, hidden_states[-1])

    # For 384x384 image with patch_size 14, expect ~27x27 = 729 tokens
    num_tokens = token_embeddings.shape[0]
    assert num_tokens > 0
    assert token_embeddings.shape[1] > 0  # hidden_size > 0


def test_forward_pass_deterministic(engine, sample_image):
    """Test that forward pass is deterministic (no grad)."""
    normalized = engine.normalize(sample_image)

    # Should not require gradients
    assert not normalized.requires_grad

    logits, activations = engine.forward(normalized)

    # Outputs should be on CPU
    assert logits.device.type == "cpu"
    assert activations["token_embeddings"].device.type == "cpu"


def test_logit_lens_3d_input(engine):
    """Test logit_lens with 3D input (batch, tokens, hidden_size)."""
    batch_size = 1
    num_tokens = 729
    hidden_size = 1152

    residual_stream = torch.randn(batch_size, num_tokens, hidden_size)
    result = engine.logit_lens(residual_stream)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 2  # [batch, hidden_size]
    assert result.shape[0] == batch_size
    assert result.shape[1] == hidden_size
    assert result.device.type == "cpu"


def test_logit_lens_2d_input(engine):
    """Test logit_lens with 2D input (tokens, hidden_size)."""
    num_tokens = 729
    hidden_size = 1152

    residual_stream = torch.randn(num_tokens, hidden_size)
    result = engine.logit_lens(residual_stream)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 1  # [hidden_size]
    assert result.shape[0] == hidden_size
    assert result.device.type == "cpu"


def test_logit_lens_different_layers(engine):
    """Test that logit_lens produces different outputs for different inputs."""
    num_tokens = 729
    hidden_size = 1152

    # Create two different residual streams
    residual_stream_1 = torch.randn(1, num_tokens, hidden_size)
    residual_stream_2 = (
        torch.randn(1, num_tokens, hidden_size) + 1.0
    )  # Different values

    result_1 = engine.logit_lens(residual_stream_1)
    result_2 = engine.logit_lens(residual_stream_2)

    # Results should be different
    assert not torch.allclose(result_1, result_2, atol=1e-5)


def test_logit_lens_device_handling(engine):
    """Test that logit_lens handles device correctly."""
    num_tokens = 729
    hidden_size = 1152

    # Create tensor on CPU
    residual_stream = torch.randn(1, num_tokens, hidden_size)

    result = engine.logit_lens(residual_stream)

    # Result should always be on CPU
    assert result.device.type == "cpu"
