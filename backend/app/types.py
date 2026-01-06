"""Pydantic models for API contracts."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ImageUploadResponse(BaseModel):
    """Response after image upload."""

    image_id: str
    grid_h: int
    grid_w: int
    patch_size: int
    img_size: int


class PluginEventRequest(BaseModel):
    """Request for plugin event handling."""

    image_id: str
    event: Dict[str, Any]


class CosineSimilarityResult(BaseModel):
    """Result from cosine similarity plugin."""

    heatmap: List[List[float]]
    grid_h: int
    grid_w: int
    selected_patch_index: int

