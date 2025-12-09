import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
from google import genai  # UPDATED: New SDK import
import os
import logging

logger = logging.getLogger(__name__)

# Optional: Set Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# NOTE: Global configuration is not needed in the new SDK;
# we initialize the Client where needed.

def preprocess_image(image: Image.Image) -> Image.Image:
    """Enhance image for OCR."""
    image = image.convert("L")  # Grayscale
    image = image.resize((max(300, image.width * 2), max(300, image.height * 2)), Image.LANCZOS)
    image = image.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    return image

def ocr_with_gemini(image_bytes: bytes) -> dict:
    """Use Gemini Vision to extract text with confidence."""
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not set — skipping Gemini OCR")
        return {"text": "", "confidence": 0.0, "error": "No API key"}

    try:
        # UPDATED: Initialize Client
        client = genai.Client(api_key=GEMINI_API_KEY)

        image = Image.open(BytesIO(image_bytes))

        # UPDATED: Use client.models.generate_content
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Corrected typo (2.5 -> 1.5)
            contents=[
                "Extract ONLY the handwritten or printed text in this image. Return plain text with no extra commentary.",
                image
            ]
        )

        text = response.text.strip() if response.text else ""
        # Gemini doesn’t return per-word confidence → use heuristic
        confidence = 0.95 if len(text) > 5 else 0.7
        return {"text": text, "confidence": confidence, "error": None}
    except Exception as e:
        logger.error(f"Gemini OCR failed: {e}")
        return {"text": "", "confidence": 0.0, "error": str(e)}

def ocr_with_tesseract(image_bytes: bytes) -> dict:
    """Use Tesseract with per-word confidence."""
    try:
        image = Image.open(BytesIO(image_bytes))
        preprocessed = preprocess_image(image)
        data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
        words, confs = [], []
        for i in range(len(data["text"])):
            conf = int(data["conf"][i])
            word = data["text"][i].strip()
            if conf > 30 and word:
                words.append(word)
                confs.append(conf)
        text = " ".join(words)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return {"text": text, "confidence": avg_conf / 100.0, "error": None}
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return {"text": "", "confidence": 0.0, "error": str(e)}

def extract_text_with_fallback(image_bytes: bytes) -> dict:
    """Primary OCR logic: Gemini → Tesseract fallback."""
    result = {"extracted_text": "", "cleaned_text": "", "confidence": 0.0, "ocr_source": "none", "notes": ""}

    # Try Gemini first
    if GEMINI_API_KEY:
        gemini = ocr_with_gemini(image_bytes)
        if gemini["confidence"] >= 0.8:
            result["extracted_text"] = gemini["text"]
            result["ocr_source"] = "gemini"
            result["confidence"] = gemini["confidence"]
        elif gemini["text"]:
            # Compare with Tesseract
            tesseract = ocr_with_tesseract(image_bytes)
            if tesseract["confidence"] > gemini["confidence"]:
                result.update({
                    "extracted_text": tesseract["text"],
                    "ocr_source": "tesseract",
                    "confidence": tesseract["confidence"]
                })
            else:
                result.update({
                    "extracted_text": gemini["text"],
                    "ocr_source": "gemini",
                    "confidence": gemini["confidence"]
                })
        else:
            # Gemini failed → use Tesseract
            tesseract = ocr_with_tesseract(image_bytes)
            result.update({
                "extracted_text": tesseract["text"],
                "ocr_source": "tesseract",
                "confidence": tesseract["confidence"]
            })
    else:
        # No Gemini → Tesseract only
        tesseract = ocr_with_tesseract(image_bytes)
        result.update({
            "extracted_text": tesseract["text"],
            "ocr_source": "tesseract",
            "confidence": tesseract["confidence"]
        })

    # Postprocess
    cleaned = result["extracted_text"].lower().strip()
    # Simple OCR error correction (optional)
    cleaned = cleaned.replace("0", "o").replace("1", "l").replace("5", "s")
    result["cleaned_text"] = cleaned

    if result["confidence"] < 0.5:
        result["notes"] = "low_confidence: result may be inaccurate"

    return result