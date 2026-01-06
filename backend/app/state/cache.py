"""In-memory activation cache."""

from typing import Dict, Optional, Any
import uuid
from ..engine.base import EngineConfig


class ActivationCache:
    """In-memory cache for storing image activations."""

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[str, Dict[str, Any]] = {}

    def store(
        self,
        image_id: str,
        token_embeddings: Any,
        engine_config: EngineConfig,
        grid_h: int,
        grid_w: int,
        hidden_states: Optional[Any] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store activations for an image.

        Args:
            image_id: Unique identifier for the image
            token_embeddings: Token embeddings tensor (final layer)
            engine_config: Engine configuration used
            grid_h: Grid height (number of patches vertically)
            grid_w: Grid width (number of patches horizontally)
            hidden_states: Optional list of hidden states from all layers
            image_metadata: Optional metadata about the image
        """
        self._cache[image_id] = {
            "token_embeddings": token_embeddings,
            "engine_config": engine_config,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "hidden_states": hidden_states,
            "image_metadata": image_metadata or {},
        }

    def get(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve activations for an image.

        Args:
            image_id: Unique identifier for the image

        Returns:
            Dictionary with activations and config, or None if not found
        """
        return self._cache.get(image_id)

    def exists(self, image_id: str) -> bool:
        """Check if image_id exists in cache.

        Args:
            image_id: Unique identifier for the image

        Returns:
            True if exists, False otherwise
        """
        return image_id in self._cache

    def generate_id(self) -> str:
        """Generate a new unique image ID.

        Returns:
            Unique image ID string
        """
        return str(uuid.uuid4())


# Global cache instance
cache = ActivationCache()
