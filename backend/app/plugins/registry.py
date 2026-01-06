"""Plugin registry for discovering and managing plugins."""
from typing import Dict, Optional
from .base import Plugin
from .cosine_tokens import CosineTokensPlugin
from .token_norm import TokenNormPlugin


class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        """Initialize registry with built-in plugins."""
        self._plugins: Dict[str, Plugin] = {}
        self._register_builtin_plugins()
    
    def _register_builtin_plugins(self):
        """Register built-in plugins."""
        self.register(CosineTokensPlugin())
        self.register(TokenNormPlugin())
    
    def register(self, plugin: Plugin) -> None:
        """Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        self._plugins[plugin.plugin_id] = plugin
    
    def get(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin by ID.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(plugin_id)
    
    def list_plugins(self) -> Dict[str, str]:
        """List all registered plugins.
        
        Returns:
            Dictionary mapping plugin_id to plugin name
        """
        return {pid: plugin.name for pid, plugin in self._plugins.items()}


# Global registry instance
registry = PluginRegistry()


