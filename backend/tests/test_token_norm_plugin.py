"""Tests for token norm plugin."""
import pytest
import torch
import numpy as np
from app.plugins.token_norm import TokenNormPlugin
from app.engine.base import EngineConfig


def test_token_norm_plugin_properties():
    """Test plugin properties."""
    plugin = TokenNormPlugin()
    assert plugin.plugin_id == "token_norm"
    assert plugin.name == "Token Norm"


def test_token_norm_plugin_compute_norm():
    """Test token norm computation."""
    plugin = TokenNormPlugin()
    config = EngineConfig(patch_size=14, img_size=384)
    
    # Create mock activations
    num_tokens = 100
    hidden_size = 64
    hidden_states = [torch.randn(num_tokens, hidden_size) * (i + 1) for i in range(5)]
    
    activations = {
        "token_embeddings": hidden_states[-1],
        "hidden_states": hidden_states,
        "grid_h": 10,
        "grid_w": 10,
    }
    
    event = {
        "type": "compute_norm",
        "layer_index": 0,
    }
    
    result = plugin.handle_event(
        image_id="test",
        event=event,
        activations=activations,
        engine_config=config,
    )
    
    # Check result structure
    assert isinstance(result, dict)
    assert "heatmap" in result
    assert "grid_h" in result
    assert "grid_w" in result
    assert "layer_index" in result
    assert "min_value" in result
    assert "max_value" in result
    
    # Check heatmap
    heatmap = result["heatmap"]
    assert isinstance(heatmap, list)
    assert len(heatmap) == result["grid_h"]
    assert len(heatmap[0]) == result["grid_w"]
    
    # Check layer index
    assert result["layer_index"] == 0
    
    # Check min/max values
    assert result["min_value"] > 0  # Norms should be positive
    assert result["max_value"] > result["min_value"]
    
    # Check heatmap values are positive (norms)
    for row in heatmap:
        for val in row:
            assert val >= 0


def test_token_norm_plugin_different_layers():
    """Test that token norm produces different results for different layers."""
    plugin = TokenNormPlugin()
    config = EngineConfig(patch_size=14, img_size=384)
    
    # Create distinct hidden states - make them VERY different
    num_tokens = 100
    hidden_size = 64
    hidden_states = []
    for i in range(5):
        # Each layer has different magnitude
        layer_tensor = torch.randn(num_tokens, hidden_size) * (i + 1) * 10.0
        hidden_states.append(layer_tensor)
    
    activations = {
        "token_embeddings": hidden_states[-1],
        "hidden_states": hidden_states,
        "grid_h": 10,
        "grid_w": 10,
    }
    
    # Test with layer 0
    event = {"type": "compute_norm", "layer_index": 0}
    result0 = plugin.handle_event("test", event, activations, config)
    
    # Test with layer 2
    event["layer_index"] = 2
    result2 = plugin.handle_event("test", event, activations, config)
    
    # Different layers should produce different results
    heatmap0 = np.array(result0["heatmap"])
    heatmap2 = np.array(result2["heatmap"])
    
    max_diff = np.abs(heatmap0 - heatmap2).max()
    assert max_diff > 1e-5, (
        f"Token norm results are identical across layers. "
        f"Max difference: {max_diff}"
    )
    
    # Layer 2 should have larger norms (scaled by factor)
    assert result2["max_value"] > result0["max_value"]


def test_token_norm_plugin_without_layer_index():
    """Test plugin works without layer_index (uses final layer)."""
    plugin = TokenNormPlugin()
    config = EngineConfig(patch_size=14, img_size=384)
    
    num_tokens = 100
    hidden_size = 64
    hidden_states = [torch.randn(num_tokens, hidden_size) for _ in range(5)]
    
    activations = {
        "token_embeddings": hidden_states[-1],
        "hidden_states": hidden_states,
        "grid_h": 10,
        "grid_w": 10,
    }
    
    event = {"type": "compute_norm"}
    
    result = plugin.handle_event("test", event, activations, config)
    
    # Should default to final layer (index 4)
    assert result["layer_index"] == 4


def test_token_norm_plugin_invalid_event_type():
    """Test that invalid event type raises error."""
    plugin = TokenNormPlugin()
    config = EngineConfig(patch_size=14, img_size=384)
    
    activations = {
        "token_embeddings": torch.randn(100, 64),
        "grid_h": 10,
        "grid_w": 10,
    }
    
    event = {"type": "invalid_event"}
    
    with pytest.raises(ValueError, match="Unknown event type"):
        plugin.handle_event("test", event, activations, config)

