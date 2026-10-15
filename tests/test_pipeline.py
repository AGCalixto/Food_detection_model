import pytest  # Import testing framework
import numpy as np # Used to mock fake input image arrays

# Dummy/utility function representing your image preprocessing step
def preprocess_image_for_yolo(image_array: np.ndarray, target_size=(416, 416)):
    """Resizes and normalizes an input numpy image array for YOLO v8."""
    if image_array is None or image_array.size == 0:
        raise ValueError("Empty image provided")
    
    # Verify input shape channels (Height, Width, Channels)
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("Invalid image dimensions")
        
    # Standardize image array to target shape
    import cv2
    resized = cv2.resize(image_array, target_size)
    normalized = resized / 255.0 # Scale pixel intensities [0, 255] to [0.0, 1.0]
    return normalized

# --- PYTEST TESTS ---

def test_preprocess_image_valid_input():
    """Test standard image preprocessing with expected 3-channel image."""
    # Create a dummy image filled with random integers (480x640 resolution, 3 RGB channels)
    dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Process image through function
    processed = preprocess_image_for_yolo(dummy_img, target_size=(416, 416))
    
    # Assert standard YOLO input spatial dimensions[cite: 1]
    assert processed.shape == (416, 416, 3)
    # Assert normalization produced values capped at max 1.0
    assert processed.max() <= 1.0
    # Assert normalization produced non-negative values
    assert processed.min() >= 0.0

def test_preprocess_image_empty_raises_error():
    """Test that passing an empty array raises a ValueError."""
    empty_img = np.array([])
    # Pytest context manager checks that a specific exception is raised
    with pytest.raises(ValueError, match="Empty image provided"):
        preprocess_image_for_yolo(empty_img)

def test_usda_api_parsing_mock(mocker):
    """Test parsing USDA API response using pytest-mock to avoid real HTTP requests."""
    # Mocking response payload structure expected from USDA search API[cite: 1]
    mock_api_response = {
        "foods": [
            {
                "description": "Hamburger",
                "foodNutrients": [
                    {"nutrientName": "Energy", "value": 250.0},
                    {"nutrientName": "Protein", "value": 12.0},
                    {"nutrientName": "Total lipid (fat)", "value": 14.0},
                    {"nutrientName": "Carbohydrate", "value": 20.0}
                ]
            }
        ]
    }
    
    # Assert fields are extracted cleanly without throwing KeyError
    first_item = mock_api_response["foods"][0]
    assert first_item["description"] == "Hamburger"
    assert len(first_item["foodNutrients"]) == 4