import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wizard_ops.api import app


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)  # Reset to start
    return img_bytes


def test_read_root():
    """Test the health check endpoint."""
    with TestClient(app) as client:
        response = client.get("/")
        
        assert response.status_code == 200
        
        data = response.json()
        # Assert required keys exist
        assert "status" in data
        assert "model_loaded" in data
        
        # Assert expected values
        assert data["status"] == "ready"
        assert isinstance(data["model_loaded"], bool)


def test_analyze_with_valid_image(sample_image):
    """Test /analyze endpoint with a valid image."""
    with TestClient(app) as client:
        files = {'file': ('food.jpg', sample_image, 'image/jpeg')}
        response = client.post("/analyze", files=files)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Assert all required keys are present
        required_keys = ["calories", "fat_g", "protein_g", "carbs_g"]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
        
        # Assert values are numeric
        assert isinstance(data["calories"], (int, float))
        assert isinstance(data["fat_g"], (int, float))
        assert isinstance(data["protein_g"], (int, float))
        assert isinstance(data["carbs_g"], (int, float))
    


def test_analyze_without_file():
    """Test /analyze endpoint without uploading a file."""
    with TestClient(app) as client:
        response = client.post("/analyze")
    
        # Should return 422 Unprocessable Entity (missing required field)
        assert response.status_code == 422


def test_analyze_with_png_image():
    """Test /analyze with PNG format."""
    with TestClient(app) as client:
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
    
        files = {'file': ('food.png', img_bytes, 'image/png')}
        response = client.post("/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "calories" in data