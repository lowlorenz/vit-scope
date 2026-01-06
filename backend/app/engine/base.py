"""Base engine interface for ViT models."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
from pydantic import BaseModel


class EngineConfig(BaseModel):
    """Engine configuration."""

    patch_size: int
    img_size: int


class Engine(ABC):
    """Base interface for ViT engines."""

    @property
    @abstractmethod
    def config(self) -> EngineConfig:
        """Return engine configuration."""
        pass

    @abstractmethod
    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize input image tensor.

        Args:
            image: Input image tensor of shape [C, H, W]

        Returns:
            Normalized tensor
        """
        pass

    @abstractmethod
    def forward(
        self, normalized_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run forward pass.

        Args:
            normalized_input: Normalized input tensor of shape [1, C, H, W]

        Returns:
            Tuple of (logits, activations dict)
            activations should contain:
            - 'token_embeddings': Final layer token embeddings [num_tokens, d]
            - 'hidden_states': List of hidden states from all layers
              Each element: [num_tokens, d] (after removing batch dimension)
              Index 0: embeddings, indices 1-N: layers 0 to N-1
        """
        pass

    @abstractmethod
    def logit_lens(self, residual_stream: torch.Tensor) -> torch.Tensor:
        """Apply classification head to residual stream activation (for LogitLens).

        This method takes a residual stream activation from any layer and applies
        only the modules necessary for classification (e.g., layer norm + pooling head).

        Args:
            residual_stream: Hidden state tensor from any layer
                Shape: [batch_size, num_tokens, hidden_size] or [num_tokens, hidden_size]

        Returns:
            Pooled embedding after applying classification head
            Shape: [batch_size, hidden_size] or [hidden_size]
        """
        pass

    @abstractmethod
    def embed_text(self, text: str | list[str]) -> torch.Tensor:
        """Embed text input into a vector representation.

        This method processes text input(s) and returns their embeddings.
        For multimodal models, this typically uses the text encoder.

        Args:
            text: Single text string or list of text strings to embed

        Returns:
            Text embeddings tensor
            Shape: [batch_size, hidden_size] for single text
            Shape: [batch_size, hidden_size] for list of texts
        """
        pass
