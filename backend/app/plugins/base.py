"""Base plugin interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..engine.base import EngineConfig


class Plugin(ABC):
    """Base interface for plugins."""
    
    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique identifier for the plugin."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the plugin."""
        pass
    
    def on_image_loaded(
        self,
        image_id: str,
        activations: Dict[str, Any],
        engine_config: EngineConfig
    ) -> Optional[Dict[str, Any]]:
        """Called when an image is loaded and activations are cached.
        
        Args:
            image_id: Unique identifier for the image
            activations: Dictionary containing activations (e.g., token_embeddings)
            engine_config: Engine configuration used
            
        Returns:
            Optional plugin state to store, or None
        """
        return None
    
    @abstractmethod
    def handle_event(
        self,
        image_id: str,
        event: Dict[str, Any],
        activations: Dict[str, Any],
        engine_config: EngineConfig
    ) -> Dict[str, Any]:
        """Handle an event from the frontend.
        
        Args:
            image_id: Unique identifier for the image
            event: Event data from frontend
            activations: Dictionary containing activations (e.g., token_embeddings)
            engine_config: Engine configuration used
            
        Returns:
            Result dictionary to send back to frontend
        """
        pass


