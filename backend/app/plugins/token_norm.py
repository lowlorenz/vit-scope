"""Token norm plugin for visualizing token embedding magnitudes."""

import torch
import numpy as np
from typing import Dict, Any
from .base import Plugin
from ..engine.base import EngineConfig


class TokenNormPlugin(Plugin):
    """Plugin for computing and visualizing token embedding norms."""

    @property
    def plugin_id(self) -> str:
        """Unique identifier for the plugin."""
        return "token_norm"

    @property
    def name(self) -> str:
        """Human-readable name for the plugin."""
        return "Token Norm"

    def handle_event(
        self,
        image_id: str,
        event: Dict[str, Any],
        activations: Dict[str, Any],
        engine_config: EngineConfig,
    ) -> Dict[str, Any]:
        """Handle event to compute token norms.

        Args:
            image_id: Unique identifier for the image
            event: Event data containing 'type' and optional 'layer_index'
            activations: Dictionary containing 'token_embeddings' and 'hidden_states'
            engine_config: Engine configuration used

        Returns:
            Dictionary with heatmap of token norms and grid dimensions
        """
        if event.get("type") != "compute_norm":
            raise ValueError(f"Unknown event type: {event.get('type')}")

        layer_index = event.get("layer_index", None)  # Default to None (use final layer)

        # Convert layer_index to int if it's provided (handles string conversion from JSON)
        if layer_index is not None:
            try:
                layer_index = int(layer_index)
            except (ValueError, TypeError):
                layer_index = None

        # Use specified layer or default to final layer (token_embeddings)
        if layer_index is not None and "hidden_states" in activations:
            hidden_states = activations["hidden_states"]
            if (
                hidden_states is not None
                and isinstance(hidden_states, list)
                and len(hidden_states) > 0
                and 0 <= layer_index < len(hidden_states)
            ):
                token_embeddings = hidden_states[layer_index]
            else:
                # Fallback to final layer if invalid layer_index
                token_embeddings = activations["token_embeddings"]
                layer_index = None  # Reset to None to indicate fallback
        else:
            # Use final layer (backward compatible)
            token_embeddings = activations["token_embeddings"]
            layer_index = None

        # Convert to tensor if numpy array
        if isinstance(token_embeddings, np.ndarray):
            token_embeddings = torch.from_numpy(token_embeddings)

        num_tokens = token_embeddings.shape[0]

        # Compute L2 norm for each token: [num_tokens, d] -> [num_tokens]
        norms = torch.norm(token_embeddings, p=2, dim=1)  # L2 norm along embedding dimension

        # Convert to numpy
        norms_np = norms.numpy()

        # Get grid dimensions from activations if available, otherwise infer
        grid_h = activations.get("grid_h")
        grid_w = activations.get("grid_w")

        if grid_h is None or grid_w is None:
            # Fallback: infer from token count and config
            grid_h = grid_w = int(np.sqrt(num_tokens))
            if grid_h * grid_w != num_tokens:
                grid_h = grid_w = engine_config.img_size // engine_config.patch_size

        # Reshape to grid
        heatmap = norms_np.reshape(grid_h, grid_w).tolist()

        # Determine final layer_index to return
        if layer_index is not None:
            final_layer_index = layer_index
        else:
            # Use final layer index
            hidden_states = activations.get("hidden_states")
            if hidden_states and isinstance(hidden_states, list) and len(hidden_states) > 0:
                final_layer_index = len(hidden_states) - 1
            else:
                final_layer_index = 0  # Fallback

        # Compute min/max for colorbar scaling
        min_norm = float(norms_np.min())
        max_norm = float(norms_np.max())

        return {
            "heatmap": heatmap,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "layer_index": final_layer_index,
            "min_value": min_norm,
            "max_value": max_norm,
        }

