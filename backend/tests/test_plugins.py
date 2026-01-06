"""Tests for plugins."""
import pytest
import torch
from app.plugins.cosine_tokens import CosineTokensPlugin
from app.engine.base import EngineConfig


def test_cosine_plugin_properties(cosine_plugin):
    """Test plugin properties."""
    assert cosine_plugin.plugin_id == "cosine_tokens"
    assert cosine_plugin.name == "Cosine Similarity"


def test_cosine_plugin_handle_event(cosine_plugin, mock_activations, mock_engine_config):
    """Test cosine similarity computation."""
    image_id = "test_image_123"
    event = {
        "type": "patch_click",
        "patch_index": 100,  # Select patch at index 100
    }
    
    result = cosine_plugin.handle_event(
        image_id=image_id,
        event=event,
        activations=mock_activations,
        engine_config=mock_engine_config,
    )
    
    # Check result structure
    assert isinstance(result, dict)
    assert "heatmap" in result
    assert "grid_h" in result
    assert "grid_w" in result
    assert "selected_patch_index" in result
    assert "layer_index" in result
    
    # Check heatmap
    heatmap = result["heatmap"]
    assert isinstance(heatmap, list)
    assert len(heatmap) == result["grid_h"]
    assert len(heatmap[0]) == result["grid_w"]
    
    # Check selected patch index
    assert result["selected_patch_index"] == 100
    
    # Check heatmap values are in valid range (cosine similarity: -1 to 1)
    # Allow small numerical errors due to floating point precision
    for row in heatmap:
        for val in row:
            assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6, f"Cosine similarity value {val} out of range [-1, 1]"


def test_cosine_plugin_with_layer_selection(cosine_plugin, mock_engine_config):
    """Test cosine similarity with layer selection."""
    import torch
    
    # Create mock activations with multiple layers
    num_tokens = 100
    hidden_size = 64
    hidden_states = [torch.randn(num_tokens, hidden_size) for _ in range(5)]
    
    activations = {
        "token_embeddings": hidden_states[-1],
        "hidden_states": hidden_states,
        "grid_h": 10,
        "grid_w": 10,
    }
    
    # Test with layer 0
    event = {
        "type": "patch_click",
        "patch_index": 50,
        "layer_index": 0,
    }
    
    result = cosine_plugin.handle_event(
        image_id="test",
        event=event,
        activations=activations,
        engine_config=mock_engine_config,
    )
    
    assert result["layer_index"] == 0
    
    # Test with layer 3
    event["layer_index"] = 3
    result = cosine_plugin.handle_event(
        image_id="test",
        event=event,
        activations=activations,
        engine_config=mock_engine_config,
    )
    
    assert result["layer_index"] == 3
    
    # Test without layer_index (should use final layer)
    event_no_layer = {
        "type": "patch_click",
        "patch_index": 50,
    }
    result = cosine_plugin.handle_event(
        image_id="test",
        event=event_no_layer,
        activations=activations,
        engine_config=mock_engine_config,
    )
    
    # Should default to final layer (index 4 in this case)
    assert result["layer_index"] == 4


def test_cosine_plugin_selected_patch_high_similarity(cosine_plugin, mock_activations, mock_engine_config):
    """Test that selected patch has high similarity with itself."""
    image_id = "test_image_123"
    selected_patch = 200
    
    event = {
        "type": "patch_click",
        "patch_index": selected_patch,
    }
    
    result = cosine_plugin.handle_event(
        image_id=image_id,
        event=event,
        activations=mock_activations,
        engine_config=mock_engine_config,
    )
    
    heatmap = result["heatmap"]
    row = selected_patch // result["grid_w"]
    col = selected_patch % result["grid_w"]
    
    # Selected patch should have similarity ~1.0 with itself
    selected_similarity = heatmap[row][col]
    assert selected_similarity > 0.99  # Should be very close to 1.0


def test_cosine_plugin_invalid_event_type(cosine_plugin, mock_activations, mock_engine_config):
    """Test that invalid event type raises error."""
    event = {
        "type": "invalid_event",
        "patch_index": 100,
    }
    
    with pytest.raises(ValueError, match="Unknown event type"):
        cosine_plugin.handle_event(
            image_id="test",
            event=event,
            activations=mock_activations,
            engine_config=mock_engine_config,
        )


def test_cosine_plugin_without_grid_dimensions(cosine_plugin, mock_token_embeddings, mock_engine_config):
    """Test plugin works when grid dimensions are not provided."""
    activations = {
        "token_embeddings": mock_token_embeddings,
        # No grid_h or grid_w
    }
    
    event = {
        "type": "patch_click",
        "patch_index": 100,
    }
    
    result = cosine_plugin.handle_event(
        image_id="test",
        event=event,
        activations=activations,
        engine_config=mock_engine_config,
    )
    
    # Should infer grid dimensions
    assert "grid_h" in result
    assert "grid_w" in result
    assert result["grid_h"] * result["grid_w"] == mock_token_embeddings.shape[0]


def test_cosine_plugin_numpy_input(cosine_plugin, mock_engine_config):
    """Test plugin handles numpy array input."""
    import numpy as np
    
    # Create numpy array embeddings
    num_tokens = 100
    hidden_size = 64
    embeddings_np = np.random.randn(num_tokens, hidden_size).astype(np.float32)
    
    activations = {
        "token_embeddings": embeddings_np,
        "grid_h": 10,
        "grid_w": 10,
    }
    
    event = {
        "type": "patch_click",
        "patch_index": 50,
    }
    
    result = cosine_plugin.handle_event(
        image_id="test",
        event=event,
        activations=activations,
        engine_config=mock_engine_config,
    )
    
    assert "heatmap" in result
    assert len(result["heatmap"]) == 10


def test_cosine_similarity_different_layers(cosine_plugin, engine, sample_image, mock_engine_config):
    """Test that cosine similarity produces different results for different layers.
    
    This test uses real engine activations to verify that layer selection
    actually changes the cosine similarity computation.
    """
    import numpy as np
    
    # Run forward pass to get real hidden states
    normalized = engine.normalize(sample_image)
    logits, activations = engine.forward(normalized)
    
    # Extract hidden states and grid dimensions
    hidden_states = activations["hidden_states"]
    token_embeddings = activations["token_embeddings"]
    num_tokens = token_embeddings.shape[0]
    
    # Compute grid dimensions
    grid_h = grid_w = int(np.sqrt(num_tokens))
    if grid_h * grid_w != num_tokens:
        grid_h = grid_w = mock_engine_config.img_size // mock_engine_config.patch_size
    
    # Prepare activations dict with all layers
    activations_dict = {
        "token_embeddings": token_embeddings,
        "hidden_states": hidden_states,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }
    
    # Select a patch index (not at edges to avoid boundary issues)
    patch_index = num_tokens // 2
    
    # Compute cosine similarity for different layers
    layers_to_test = [0, 5, 10, 15, 20, 27]  # Test various layers including embeddings and final
    
    results = {}
    for layer_idx in layers_to_test:
        if layer_idx >= len(hidden_states):
            continue
            
        event = {
            "type": "patch_click",
            "patch_index": patch_index,
            "layer_index": layer_idx,
        }
        
        result = cosine_plugin.handle_event(
            image_id="test",
            event=event,
            activations=activations_dict,
            engine_config=mock_engine_config,
        )
        
        results[layer_idx] = np.array(result["heatmap"])
        assert result["layer_index"] == layer_idx
    
    # Verify that different layers produce different results
    # Compare each layer with all others
    layer_indices = list(results.keys())
    differences_found = False
    
    for i, layer_i in enumerate(layer_indices):
        for layer_j in layer_indices[i + 1:]:
            heatmap_i = results[layer_i]
            heatmap_j = results[layer_j]
            
            # Compute maximum absolute difference
            max_diff = np.abs(heatmap_i - heatmap_j).max()
            
            # Different layers should produce different results
            # (allowing for small numerical differences)
            if max_diff > 1e-5:
                differences_found = True
                break
        
        if differences_found:
            break
    
    assert differences_found, (
        f"Cosine similarity results are identical across layers {layer_indices}. "
        "This suggests layer selection is not working correctly."
    )
    
    # Additional check: verify that the selected patch has similarity ~1.0 with itself
    # for all layers (this should be consistent)
    row = patch_index // grid_w
    col = patch_index % grid_w
    
    for layer_idx, heatmap in results.items():
        selected_similarity = heatmap[row, col]
        assert selected_similarity > 0.99, (
            f"Selected patch should have similarity ~1.0 with itself "
            f"at layer {layer_idx}, got {selected_similarity}"
        )

