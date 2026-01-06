"""Tests for text embedding functionality."""
import pytest
import torch
from app.engine.siglip_engine import SiglipEngine


def test_embed_text_single_string(engine):
    """Test embedding a single text string."""
    text = "a photo of a cat"
    embeddings = engine.embed_text(text)

    # Check output type and shape
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.dim() == 2  # [batch_size, hidden_size]
    assert embeddings.shape[0] == 1  # Single text
    assert embeddings.shape[1] > 0  # Hidden size > 0

    # Should be on CPU
    assert embeddings.device.type == "cpu"


def test_embed_text_multiple_strings(engine):
    """Test embedding multiple text strings."""
    texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
    embeddings = engine.embed_text(texts)

    # Check output type and shape
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.dim() == 2  # [batch_size, hidden_size]
    assert embeddings.shape[0] == len(texts)  # Batch size matches input
    assert embeddings.shape[1] > 0  # Hidden size > 0

    # Should be on CPU
    assert embeddings.device.type == "cpu"


def test_embed_text_different_texts_produce_different_embeddings(engine):
    """Test that different texts produce different embeddings."""
    text1 = "a photo of a cat"
    text2 = "a photo of a dog"

    emb1 = engine.embed_text(text1)
    emb2 = engine.embed_text(text2)

    # Embeddings should be different
    assert not torch.allclose(emb1, emb2, atol=1e-5)


def test_embed_text_same_text_produces_same_embedding(engine):
    """Test that the same text produces the same embedding."""
    text = "a photo of a cat"

    emb1 = engine.embed_text(text)
    emb2 = engine.embed_text(text)

    # Embeddings should be identical (deterministic)
    assert torch.allclose(emb1, emb2, atol=1e-6)


def test_embed_text_batch_vs_single(engine):
    """Test that batch embedding matches individual embeddings."""
    texts = ["a photo of a cat", "a photo of a dog"]

    # Get batch embeddings
    batch_embeddings = engine.embed_text(texts)

    # Get individual embeddings
    single_emb1 = engine.embed_text(texts[0])
    single_emb2 = engine.embed_text(texts[1])

    # Batch embeddings should match individual embeddings
    # Use slightly higher tolerance as batch processing might have minor differences
    assert torch.allclose(batch_embeddings[0], single_emb1[0], atol=1e-4)
    assert torch.allclose(batch_embeddings[1], single_emb2[0], atol=1e-4)

