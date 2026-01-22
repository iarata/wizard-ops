import importlib
import io
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Generate test image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)  # Reset to start
    return img_bytes

@pytest.fixture
def mock_model():
    "Create mock model. Returns (mock_model,mock_device)"
    return (MagicMock(), MagicMock())

@pytest.fixture(autouse=True)
def clean_app_module():
    """Automatically clean up app module after each test."""
    yield
    if 'wizard_ops.backend.api' in sys.modules:
        del sys.modules['wizard_ops.backend.api']

def test_health_check(mock_model):
    """Test the health check endpoint. Assumes the checkpoint is already in tmp."""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('wizard_ops.backend.api.evaluate.load_model_for_inference', return_value=mock_model):

         # Force fresh import
         from wizard_ops.backend.api import app
         
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

def test_checkpoint_loading(mock_model):
    """Test that model is loaded when pathlib.Path.exists == False"""
    with patch('pathlib.Path.exists', return_value=False), \
         patch('wizard_ops.backend.api.evaluate.load_model_for_inference', return_value=mock_model) as mock_load_model, \
         patch('wizard_ops.backend.api.storage.Client') as mock_storage_client:
        

         mock_bucket = MagicMock()
         mock_blob = MagicMock()
         mock_storage_client.return_value.bucket.return_value = mock_bucket
         mock_bucket.blob.return_value = mock_blob

         # Force fresh import
         from wizard_ops.backend.api import app
        
         with TestClient(app):
             # Verify download happened
             mock_storage_client.assert_called_once()
             mock_blob.download_to_filename.assert_called_once()
             mock_load_model.assert_called_once()

         

def test_analyze_with_valid_image(sample_image, mock_model):
    """Test /analyze endpoint with a valid image."""

    # Patch google storage client, model loading, model path and predict function
    with patch('pathlib.Path.exists', return_value=True), \
         patch('wizard_ops.backend.api.evaluate.load_model_for_inference', return_value=mock_model), \
         patch('wizard_ops.backend.api.evaluate.predict_nutrition') as mock_predict:
         
         # Mock prediction
         mock_predict.return_value = {
             "normalized":
             {
                "total_calories": 0.0,
                "total_fat": 0.0,
                "total_protein": 0.0,
                "total_carb": 0.0,
                }
            }
         
         # Force fresh import
         if 'wizard_ops.backend.api' in sys.modules:
             import wizard_ops.backend.api
             importlib.reload(wizard_ops.backend.api)
             app = wizard_ops.backend.api.app
         else:
             from wizard_ops.backend.api import app

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

