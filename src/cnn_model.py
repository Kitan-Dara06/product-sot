import os
import json
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

# Logger
logger = logging.getLogger(__name__)

# Paths to your trained files
MODEL_PATH = "models/best_scratch_model_v2.keras"
LABEL_MAP_PATH = "models/label_map.json"

# Global variables to load model once
_model = None
_label_map = None

def load_resources():
    """Loads model and label map if not already loaded."""
    global _model, _label_map
    
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("✅ CNN Model loaded.")
        else:
            logger.error(f"❌ Model file not found at {MODEL_PATH}")

    if _label_map is None:
        if os.path.exists(LABEL_MAP_PATH):
            with open(LABEL_MAP_PATH, "r") as f:
                _label_map = json.load(f)
                # Convert keys back to integers (JSON saves them as strings)
                _label_map = {int(k): v for k, v in _label_map.items()}
            logger.info("✅ Label map loaded.")
        else:
            logger.error(f"❌ Label map not found at {LABEL_MAP_PATH}")

def predict_stockcode_from_image(image_bytes: bytes) -> tuple:
    """
    Preprocesses image, runs inference, returns (StockCode, Confidence).
    """
    load_resources()
    
    if _model is None or _label_map is None:
        return "ERROR", 0.0

    try:
        # 1. Preprocess Image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize((128, 128))  # MUST match your training size (128x128)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 128, 128, 3)

        # 2. Run Prediction
        preds = _model.predict(img_array, verbose=0)
        pred_class_id = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        # 3. Map ID to StockCode
        stock_code = _label_map.get(pred_class_id, "UNKNOWN")
        
        return stock_code, confidence

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return "ERROR", 0.0