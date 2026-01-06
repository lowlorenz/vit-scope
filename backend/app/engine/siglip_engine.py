"""SigLIP engine implementation."""

import torch
from transformers import SiglipProcessor, SiglipModel
from typing import Dict, Tuple
from .base import Engine, EngineConfig


class SiglipEngine(Engine):
    """Engine for SigLIP models.

    Supports all SigLIP model variants from HuggingFace, including:
    - google/siglip-so400m-patch14-384 (default)
    - google/siglip-base-patch16-224
    - google/siglip-base-patch16-256
    - google/siglip-large-patch16-256
    - And other SigLIP variants

    Configuration (patch_size, img_size) is automatically extracted from the model.
    """

    def __init__(
        self, model_name: str = "google/siglip-so400m-patch14-384", device: str = "cuda"
    ):
        """Initialize SigLIP engine.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Extract config from model dynamically (supports all SigLIP variants)
        vision_config = self.model.config.vision_config
        self._config = EngineConfig(
            patch_size=vision_config.patch_size,
            img_size=vision_config.image_size,
        )

    @property
    def config(self) -> EngineConfig:
        """Return engine configuration."""
        return self._config

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize input image using SigLIP processor.

        Args:
            image: Input image tensor of shape [C, H, W] or PIL Image

        Returns:
            Normalized tensor of shape [1, C, H, W]
        """
        # If tensor, convert to PIL for processor
        from PIL import Image
        import numpy as np

        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL
            if image.dim() == 3:
                image = image.permute(1, 2, 0)  # CHW -> HWC
            image_np = image.cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            image = Image.fromarray(image_np)

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        return pixel_values

    def forward(
        self, normalized_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run forward pass.

        Args:
            normalized_input: Normalized input tensor of shape [1, C, H, W]

        Returns:
            Tuple of (logits, activations dict)
            - logits: Pooled image embedding after applying post_layernorm and head
              Shape: [batch_size, hidden_size] (for SigLIP: [1, 1152])
              This is the final output representation used for contrastive learning
            - activations: Dictionary containing:
              - 'token_embeddings': Final layer token embeddings [num_tokens, d]
              - 'hidden_states': List of hidden states from all layers
                Each element: [num_tokens, d] (after removing batch dimension)
                Index 0: embeddings, indices 1-27: layers 0 to 26
        """
        with torch.no_grad():
            # Collect hidden states from all layers using hooks
            all_hidden_states = []

            def hook_fn(module, input, output):
                # output[0] is hidden_states from the layer
                h = output[0].detach()
                # Ensure it has batch dimension
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                all_hidden_states.append(h)

            # Register hooks on all encoder layers
            hooks = []
            for layer in self.model.vision_model.encoder.layers:
                h = layer.register_forward_hook(hook_fn)
                hooks.append(h)

            try:
                # Get embeddings first
                embeddings = self.model.vision_model.embeddings(normalized_input)
                all_hidden_states.insert(0, embeddings.detach())

                # Run forward pass (hooks will capture layer outputs)
                outputs = self.model.vision_model(
                    pixel_values=normalized_input,
                    return_dict=True,
                )

                # Get final hidden state (before pooling)
                # Shape: [batch_size, num_tokens, hidden_size]
                final_hidden_state = outputs.last_hidden_state

                # Remove batch dimension from all hidden states: [num_tokens, hidden_size]
                all_hidden_states_cpu = [h[0].cpu() for h in all_hidden_states]
                token_embeddings = all_hidden_states_cpu[-1]  # Final layer

                # For SigLIP, apply the proper pooling head (post_layernorm + attention head)
                # This is the same as logit_lens but applied to the final hidden state
                normalized = self.model.vision_model.post_layernorm(final_hidden_state)
                pooled_output = self.model.vision_model.head(
                    normalized
                )  # [batch_size, hidden_size]

                # Return the pooled embedding as logits (this is SigLIP's final output representation)
                logits = pooled_output.cpu()

                activations = {
                    "token_embeddings": token_embeddings,
                    "hidden_states": all_hidden_states_cpu,
                }

                return logits, activations

            finally:
                # Remove hooks
                for h in hooks:
                    h.remove()

    def logit_lens(self, residual_stream: torch.Tensor) -> torch.Tensor:
        """Apply classification head to residual stream activation (for LogitLens).

        For SigLIP, this applies:
        1. post_layernorm: Final layer normalization
        2. head: Attention-based pooling head (with learned probe)

        Args:
            residual_stream: Hidden state tensor from any layer
                Shape: [batch_size, num_tokens, hidden_size] or [num_tokens, hidden_size]

        Returns:
            Pooled embedding after applying classification head
            Shape: [batch_size, hidden_size] or [hidden_size]
        """
        with torch.no_grad():
            # Ensure tensor is on correct device
            if residual_stream.device != self.device:
                residual_stream = residual_stream.to(self.device)

            # Handle 2D input (no batch dimension)
            was_2d = residual_stream.dim() == 2
            if was_2d:
                residual_stream = residual_stream.unsqueeze(
                    0
                )  # [1, num_tokens, hidden_size]

            # Apply post_layernorm
            normalized = self.model.vision_model.post_layernorm(residual_stream)

            # Apply attention pooling head
            pooled = self.model.vision_model.head(normalized)

            # Remove batch dimension if input was 2D
            if was_2d:
                pooled = pooled.squeeze(0)

            return pooled.cpu()

    def embed_text(self, text: str | list[str]) -> torch.Tensor:
        """Embed text input using SigLIP text encoder.

        Args:
            text: Single text string or list of text strings to embed

        Returns:
            Text embeddings tensor
            Shape: [batch_size, hidden_size]
        """
        with torch.no_grad():
            # Process text using the processor
            inputs = self.processor(text=text, return_tensors="pt", padding=True)

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Use the model's get_text_features method which handles
            # text encoding and projection automatically
            text_embeds = self.model.get_text_features(**inputs)

            return text_embeds.cpu()
