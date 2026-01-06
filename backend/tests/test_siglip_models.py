"""Tests for all supported SigLIP model variants."""
import pytest
import torch
from PIL import Image
import numpy as np
from app.engine.siglip_engine import SiglipEngine
from app.engine.base import EngineConfig


# All supported SigLIP models
SUPPORTED_SIGLIP_MODELS = [
    # SigLIP v1 models
    "google/siglip-base-patch16-224",
    "google/siglip-base-patch16-256",
    "google/siglip-base-patch16-384",
    "google/siglip-base-patch16-512",
    "google/siglip-large-patch16-256",
    "google/siglip-large-patch16-384",
    "google/siglip-so400m-patch14-384",
    # SigLIP 2 models
    "google/siglip2-base-patch16-224",
    "google/siglip2-base-patch16-256",
    "google/siglip2-large-patch16-256",
    "google/siglip2-so400m-patch14-384",
]


@pytest.fixture(params=SUPPORTED_SIGLIP_MODELS)
def siglip_model_name(request):
    """Fixture that provides each supported SigLIP model name."""
    return request.param


@pytest.fixture
def siglip_engine(siglip_model_name):
    """Create a SiglipEngine instance for each model variant."""
    return SiglipEngine(model_name=siglip_model_name, device="cpu")


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    # Create a simple 384x384 RGB image (will be resized by model)
    img_array = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


class TestSiglipModelVariants:
    """Test suite for all SigLIP model variants."""

    def test_model_initialization(self, siglip_engine, siglip_model_name):
        """Test that each model can be initialized."""
        assert siglip_engine is not None
        assert siglip_engine.model is not None
        assert siglip_engine.processor is not None
        assert siglip_engine.device == "cpu"

    def test_config_extraction(self, siglip_engine, siglip_model_name):
        """Test that config is correctly extracted from each model."""
        config = siglip_engine.config
        assert isinstance(config, EngineConfig)
        assert config.patch_size > 0
        assert config.img_size > 0

        # Verify config matches model's actual config
        vision_config = siglip_engine.model.config.vision_config
        assert config.patch_size == vision_config.patch_size
        assert config.img_size == vision_config.image_size

    def test_normalize(self, siglip_engine, sample_image):
        """Test normalization works for all models."""
        normalized = siglip_engine.normalize(sample_image)

        assert isinstance(normalized, torch.Tensor)
        assert normalized.dim() == 4  # [batch, channels, height, width]
        assert normalized.shape[0] == 1
        assert normalized.shape[1] == 3  # RGB

    def test_forward_pass(self, siglip_engine, sample_image):
        """Test forward pass works for all models."""
        normalized = siglip_engine.normalize(sample_image)
        logits, activations = siglip_engine.forward(normalized)

        # Check logits
        assert isinstance(logits, torch.Tensor)
        assert logits.dim() == 2  # [batch, hidden_size]
        assert logits.shape[0] == 1
        assert logits.shape[1] > 0  # Hidden size > 0

        # Check activations
        assert isinstance(activations, dict)
        assert "token_embeddings" in activations
        assert "hidden_states" in activations

        token_embeddings = activations["token_embeddings"]
        assert isinstance(token_embeddings, torch.Tensor)
        assert token_embeddings.dim() == 2  # [num_tokens, hidden_size]

        hidden_states = activations["hidden_states"]
        assert isinstance(hidden_states, list)
        assert len(hidden_states) > 0

    def test_logit_lens(self, siglip_engine):
        """Test logit_lens works for all models."""
        # Create dummy residual stream
        config = siglip_engine.config
        num_tokens = (config.img_size // config.patch_size) ** 2
        hidden_size = siglip_engine.model.config.vision_config.hidden_size

        residual_stream = torch.randn(1, num_tokens, hidden_size)
        result = siglip_engine.logit_lens(residual_stream)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 2  # [batch, hidden_size]
        assert result.shape[0] == 1
        assert result.shape[1] == hidden_size

    def test_embed_text(self, siglip_engine):
        """Test text embedding works for all models."""
        text = "a photo of a cat"
        embeddings = siglip_engine.embed_text(text)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.dim() == 2  # [batch, hidden_size]
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0

    def test_embed_text_multiple(self, siglip_engine):
        """Test text embedding with multiple texts for all models."""
        texts = ["a photo of a cat", "a photo of a dog"]
        embeddings = siglip_engine.embed_text(texts)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.dim() == 2  # [batch, hidden_size]
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0

    def test_hidden_states_consistency(self, siglip_engine, sample_image):
        """Test that hidden states are consistent across layers."""
        normalized = siglip_engine.normalize(sample_image)
        logits, activations = siglip_engine.forward(normalized)

        hidden_states = activations["hidden_states"]
        token_embeddings = activations["token_embeddings"]

        # All hidden states should have the same shape
        for i, hidden in enumerate(hidden_states):
            assert hidden.shape == token_embeddings.shape, (
                f"Layer {i} has shape {hidden.shape}, "
                f"expected {token_embeddings.shape}"
            )

        # Final layer should match token_embeddings
        assert torch.allclose(token_embeddings, hidden_states[-1])

    def test_deterministic_output(self, siglip_engine, sample_image):
        """Test that forward pass is deterministic."""
        normalized = siglip_engine.normalize(sample_image)

        logits1, activations1 = siglip_engine.forward(normalized)
        logits2, activations2 = siglip_engine.forward(normalized)

        # Logits should be identical
        assert torch.allclose(logits1, logits2, atol=1e-6)

        # Token embeddings should be identical
        assert torch.allclose(
            activations1["token_embeddings"],
            activations2["token_embeddings"],
            atol=1e-6,
        )


class TestModelSpecificConfigs:
    """Test model-specific configurations."""

    @pytest.mark.parametrize(
        "model_name,expected_patch_size,expected_img_size",
        [
            # SigLIP v1 models
            ("google/siglip-base-patch16-224", 16, 224),
            ("google/siglip-base-patch16-256", 16, 256),
            ("google/siglip-base-patch16-384", 16, 384),
            ("google/siglip-base-patch16-512", 16, 512),
            ("google/siglip-large-patch16-256", 16, 256),
            ("google/siglip-large-patch16-384", 16, 384),
            ("google/siglip-so400m-patch14-384", 14, 384),
            # SigLIP 2 models
            ("google/siglip2-base-patch16-224", 16, 224),
            ("google/siglip2-base-patch16-256", 16, 256),
            ("google/siglip2-large-patch16-256", 16, 256),
            ("google/siglip2-so400m-patch14-384", 14, 384),
        ],
    )
    def test_model_config_values(
        self, model_name, expected_patch_size, expected_img_size
    ):
        """Test that each model has the expected configuration."""
        engine = SiglipEngine(model_name=model_name, device="cpu")
        config = engine.config

        assert config.patch_size == expected_patch_size, (
            f"{model_name}: Expected patch_size {expected_patch_size}, "
            f"got {config.patch_size}"
        )
        assert config.img_size == expected_img_size, (
            f"{model_name}: Expected img_size {expected_img_size}, "
            f"got {config.img_size}"
        )

