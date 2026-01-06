"""API routes for the ViT Scope backend."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import io
from PIL import Image
import torch
from typing import Dict, Any, Optional

from ..types import ImageUploadResponse, PluginEventRequest, CosineSimilarityResult
from ..engine.siglip_engine import SiglipEngine
from ..state.cache import cache
from ..plugins.registry import registry

router = APIRouter()

# Global engine instance (loaded at startup)
engine: SiglipEngine = None

# Available SigLIP models
AVAILABLE_MODELS = [
    "google/siglip-so400m-patch14-384",  # Default
    "google/siglip-base-patch16-224",
    "google/siglip-base-patch16-256",
    "google/siglip-base-patch16-384",
    "google/siglip-base-patch16-512",
    "google/siglip-large-patch16-256",
    "google/siglip-large-patch16-384",
    "google/siglip2-base-patch16-224",
    "google/siglip2-base-patch16-256",
    "google/siglip2-large-patch16-256",
    "google/siglip2-so400m-patch14-384",  # SigLIP 2 SoViT-400M
]


def initialize_engine(
    model_name: str = "google/siglip-so400m-patch14-384", device: str = "cuda"
):
    """Initialize the global engine instance."""
    global engine
    engine = SiglipEngine(model_name=model_name, device=device)


def switch_engine_model(model_name: str, device: str = "cuda"):
    """Switch to a different model.

    Args:
        model_name: Name of the model to load
        device: Device to run on ('cuda' or 'cpu')
    """
    global engine
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' is not in the list of available models")
    engine = SiglipEngine(model_name=model_name, device=device)


@router.get("/api/models")
async def list_models():
    """List all available models.

    Returns:
        Dictionary mapping model names to display names
    """
    model_info = {}
    for model_name in AVAILABLE_MODELS:
        # Create display name from model name
        display_name = model_name.replace("google/siglip", "SigLIP").replace("-", " ")
        if "so400m" in model_name:
            display_name = display_name.replace("siglip", "SigLIP SoViT-400M")
        elif "siglip2" in model_name:
            display_name = display_name.replace("siglip2", "SigLIP 2")
        model_info[model_name] = display_name
    return model_info


@router.post("/api/models/select")
async def select_model(request: Dict[str, Any]):
    """Select a model to use for processing.

    Args:
        request: Request body containing 'model_name'

    Returns:
        Success message with model info
    """
    model_name = request.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not available. Available models: {AVAILABLE_MODELS}",
        )

    try:
        switch_engine_model(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        config = engine.config
        return {
            "model_name": model_name,
            "patch_size": config.patch_size,
            "img_size": config.img_size,
            "message": f"Switched to {model_name}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/api/images", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
):
    """Upload an image and process it through the model.

    Args:
        file: Uploaded image file
        model_name: Optional model name to use (if not provided, uses current engine)

    Returns:
        Image metadata including image_id and grid dimensions
    """
    # Switch model if requested
    if model_name is not None:
        if model_name not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' is not available",
            )
        try:
            switch_engine_model(
                model_name, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )

    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Resize to model input size
    config = engine.config
    image = image.resize((config.img_size, config.img_size))

    # Normalize and run forward pass
    normalized = engine.normalize(image)
    logits, activations = engine.forward(normalized)

    # Compute grid dimensions from token count
    token_embeddings = activations["token_embeddings"]
    hidden_states = activations.get("hidden_states")  # All layer activations
    num_tokens = token_embeddings.shape[0]
    grid_h = grid_w = int((num_tokens) ** 0.5)

    # If not perfect square, use config-based calculation
    if grid_h * grid_w != num_tokens:
        grid_h = grid_w = config.img_size // config.patch_size

    # Store in cache
    image_id = cache.generate_id()
    cache.store(
        image_id=image_id,
        token_embeddings=token_embeddings,
        engine_config=config,
        grid_h=grid_h,
        grid_w=grid_w,
        hidden_states=hidden_states,
        image_metadata={
            "original_filename": file.filename,
            "content_type": file.content_type,
        },
    )

    # Notify plugins about new image
    for plugin in registry._plugins.values():
        plugin.on_image_loaded(image_id, activations, config)

    return ImageUploadResponse(
        image_id=image_id,
        grid_h=grid_h,
        grid_w=grid_w,
        patch_size=config.patch_size,
        img_size=config.img_size,
    )


@router.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Get an uploaded image (placeholder - returns error for now).

    In a full implementation, this would return the cached image.
    For v0, we'll return the image data URL in the upload response instead.
    """
    raise HTTPException(status_code=501, detail="Image retrieval not implemented in v0")


@router.post("/api/plugins/{plugin_id}/event")
async def handle_plugin_event(plugin_id: str, request: PluginEventRequest):
    """Handle a plugin event.

    Args:
        plugin_id: Plugin identifier
        request: Event request containing image_id and event data

    Returns:
        Plugin result
    """
    plugin = registry.get(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")

    # Get cached activations
    cached_data = cache.get(request.image_id)
    if cached_data is None:
        raise HTTPException(
            status_code=404, detail=f"Image '{request.image_id}' not found"
        )

    # Handle event
    try:
        # Pass grid dimensions and hidden states via activations for plugins to use
        activations = {
            "token_embeddings": cached_data["token_embeddings"],
            "grid_h": cached_data.get("grid_h"),
            "grid_w": cached_data.get("grid_w"),
            "hidden_states": cached_data.get("hidden_states"),  # All layer activations
        }
        result = plugin.handle_event(
            image_id=request.image_id,
            event=request.event,
            activations=activations,
            engine_config=cached_data["engine_config"],
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plugin error: {str(e)}")


@router.get("/api/plugins")
async def list_plugins():
    """List all available plugins.

    Returns:
        Dictionary mapping plugin_id to plugin name
    """
    return registry.list_plugins()
