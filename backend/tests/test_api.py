"""Tests for API routes."""
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np
from app.main import app
from app.state.cache import cache
from app.api.routes import initialize_engine


@pytest.fixture
def client():
    """Create a test client."""
    # Initialize engine before creating client
    initialize_engine(device="cpu")
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for upload."""
    # Create a simple image
    img = Image.new("RGB", (384, 384), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes.getvalue()


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "ViT Scope API"


def test_list_plugins(client):
    """Test listing available plugins."""
    response = client.get("/api/plugins")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "cosine_tokens" in data
    assert data["cosine_tokens"] == "Cosine Similarity"


def test_upload_image(client, sample_image_bytes):
    """Test image upload endpoint."""
    files = {"file": ("test_image.png", sample_image_bytes, "image/png")}
    response = client.post("/api/images", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "image_id" in data
    assert "grid_h" in data
    assert "grid_w" in data
    assert "patch_size" in data
    assert "img_size" in data
    
    # Check values
    assert isinstance(data["image_id"], str)
    assert data["patch_size"] == 14
    assert data["img_size"] == 384
    assert data["grid_h"] > 0
    assert data["grid_w"] > 0
    
    # Verify image was cached
    assert cache.exists(data["image_id"])


def test_upload_invalid_file_type(client):
    """Test uploading non-image file."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/api/images", files=files)
    
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_plugin_event_cosine_similarity(client, sample_image_bytes):
    """Test cosine similarity plugin event."""
    # First upload an image
    files = {"file": ("test_image.png", sample_image_bytes, "image/png")}
    upload_response = client.post("/api/images", files=files)
    assert upload_response.status_code == 200
    image_id = upload_response.json()["image_id"]
    
    # Then trigger plugin event
    event_data = {
        "image_id": image_id,
        "event": {
            "type": "patch_click",
            "patch_index": 100,
        },
    }
    
    response = client.post("/api/plugins/cosine_tokens/event", json=event_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "heatmap" in data
    assert "grid_h" in data
    assert "grid_w" in data
    assert "selected_patch_index" in data
    assert data["selected_patch_index"] == 100


def test_plugin_event_nonexistent_image(client):
    """Test plugin event with non-existent image."""
    event_data = {
        "image_id": "nonexistent_id",
        "event": {
            "type": "patch_click",
            "patch_index": 100,
        },
    }
    
    response = client.post("/api/plugins/cosine_tokens/event", json=event_data)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_plugin_event_nonexistent_plugin(client, sample_image_bytes):
    """Test plugin event with non-existent plugin."""
    files = {"file": ("test_image.png", sample_image_bytes, "image/png")}
    upload_response = client.post("/api/images", files=files)
    image_id = upload_response.json()["image_id"]
    
    event_data = {
        "image_id": image_id,
        "event": {"type": "patch_click", "patch_index": 100},
    }
    
    response = client.post("/api/plugins/nonexistent_plugin/event", json=event_data)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_image_endpoint_not_implemented(client):
    """Test that get image endpoint returns 501."""
    response = client.get("/api/images/test_id")
    assert response.status_code == 501


def test_upload_multiple_images(client, sample_image_bytes):
    """Test uploading multiple images."""
    image_ids = []
    
    for i in range(3):
        files = {"file": (f"test_image_{i}.png", sample_image_bytes, "image/png")}
        response = client.post("/api/images", files=files)
        assert response.status_code == 200
        image_ids.append(response.json()["image_id"])
    
    # All should be unique
    assert len(set(image_ids)) == 3
    
    # All should be cached
    for image_id in image_ids:
        assert cache.exists(image_id)

