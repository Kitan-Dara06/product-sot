from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import os
import sys
import logging
import re
import pandas as pd

# Import your modules
from src.vector_db import PineconeProductDB
from src.ocr_engine import extract_text_with_fallback
from src.cnn_model import predict_stockcode_from_image
from dotenv import load_dotenv

load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Flask App ===
app = Flask(__name__)
CORS(app)

# --- Swagger UI Configuration ---
SWAGGER_URL = '/docs'
API_URL = '/static/swagger.json'  # We will serve the spec from here
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "E-Commerce AI API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


# --- Service Initialization ---
try:
    df_products = pd.read_csv(CSV_DATA_PATH)
    product_lookup = df_products.set_index("StockCode").to_dict(orient="index")
    logger.info("✅ Product metadata loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load CSV data: {e}")
    product_lookup = {}

try:
    vector_db = PineconeProductDB(
        api_key=PINECONE_API_KEY,
        cloud="aws",
        region="us-east-1",
        index_name="product-recommendations"
    )
    logger.info("✅ Pinecone Vector DB connected.")
except Exception as e:
    logger.error(f"❌ Failed to connect to Pinecone: {e}")
    vector_db = None

# --- Helper Logic ---
def get_product_details(stock_code):
    code_str = str(stock_code)
    return product_lookup.get(code_str) or product_lookup.get(int(code_str) if code_str.isdigit() else None)

def search_related_products(query_text, top_k=5):
    if not vector_db:
        return []
    results = vector_db.search(query_text, top_k=top_k)
    matches = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        matches.append({
            "stock_code": match["id"],
            "description": meta.get("description", "N/A"),
            "country": meta.get("country", "Unknown"),
            "unit_price": meta.get("unit_price", 0.0),
            "score": match["score"]
        })
    return matches


# ==========================================
# NEW: Root Endpoint
# ==========================================
@app.route("/", methods=["GET"])
def root():
    """Redirects users to the documentation page."""
    return {"message": "Welcome to E-Commerce AI Endpoints"}

# ==========================================
# NEW: Health Check Endpoint
# ==========================================
@app.route("/health", methods=["GET"])
def health_check():
    """Returns the status of the API and its dependencies."""
    status = {
        "status": "healthy",
        "services": {
            "flask": "running",
            "pinecone": "connected" if vector_db else "disconnected",
        }
    }
    status_code = 200 if vector_db else 503
    return jsonify(status), status_code

# ==========================================
# NEW: Swagger JSON Endpoint
# ==========================================
@app.route("/static/swagger.json")
def create_swagger_spec():
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "E-Commerce AI API",
            "version": "1.0.0",
            "description": "API for Product Recommendations, OCR Search, and Visual Detection"
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Check API health",
                    "responses": {"200": {"description": "API is healthy"}}
                }
            },
            "/recommend": {
                "post": {
                    "summary": "Get product recommendations from text",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "example": {"query": "red alarm clock"}
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "List of recommended products"}}
                }
            },
            "/recommend-ocr": {
                "post": {
                    "summary": "Search products using an image of text",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"image": {"type": "string", "format": "binary"}}
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "OCR text and recommendations"}}
                }
            },
            "/detect-product": {
                "post": {
                    "summary": "Identify product from a photo",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"image": {"type": "string", "format": "binary"}}
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Identified product and similar items"}}
                }
            }
        }
    })



# === Endpoint 1: Text Query ===
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "")
    if not query: return jsonify({"error": "Empty query"}), 400
    logger.info(f"🔎 Text query: {query}")
    products = search_related_products(query)
    return jsonify({"response": f"Found {len(products)} products.", "products": products})

# === Endpoint 2: OCR Image Query ===
@app.route("/recommend-ocr", methods=["POST"])
def recommend_ocr():
    if "image" not in request.files: return jsonify({"error": "No image uploaded"}), 400
    image_bytes = request.files["image"].read()
    ocr_result = extract_text_with_fallback(image_bytes)
    cleaned_text = ocr_result.get("cleaned_text", "")
    products = search_related_products(cleaned_text) if cleaned_text else []
    return jsonify({
        "extracted_text": ocr_result.get("extracted_text"),
        "confidence": ocr_result.get("confidence"),
        "response": f"Query: '{cleaned_text}'",
        "products": products
    })

#cnn endpoint


# ==========================================
# ENDPOINT 3: Product Image Upload Interface
# ==========================================
@app.route("/detect-product", methods=["POST"])
def detect_product():
    """
    Input: Image file (photo of a product).
    Process: 
      1. CNN predicts class (e.g., "lunch_bag_pink_polkadot").
      2. App looks up this class in the CSV to get metadata (Real StockCode, Price).
      3. App searches Pinecone for similar items using the description.
    Output: Identified Product + Related Recommendations.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    try:
        # Step 1: Run CNN Classification
        # This returns the folder name (e.g., "lunch_bag_pink_polkadot")
        predicted_class, confidence = predict_stockcode_from_image(image_bytes)
        logger.info(f"📸 CNN Predicted Class: {predicted_class} ({confidence:.2f})")

        if predicted_class == "UNKNOWN":
            return jsonify({
                "detected_class": "UNKNOWN",
                "products": [],
                "response": "Product could not be identified."
            }), 200

        # Step 2: Lookup Product Metadata in CSV
        # The CNN class has underscores ("_") but CSV descriptions have spaces (" ")
        # We try to find the row where the formatted description matches the CNN class
        
        # Helper to format CSV description to match CNN class format (lowercase, spaces -> underscores)
        # e.g., "Lunch Bag Pink Polkadot" -> "lunch_bag_pink_polkadot"
        product_row = df_products[
            df_products["Description"].astype(str).str.lower().str.strip().str.replace(' ', '_') == predicted_class.lower()
        ]

        if product_row.empty:
            # Fallback if no exact match found
            description = predicted_class.replace("_", " ") # Make it readable
            real_stock_code = "UNKNOWN"
            logger.warning(f"⚠️ Class '{predicted_class}' recognized by CNN but not found in CSV.")
        else:
            # Success! Get the real details
            row = product_row.iloc[0]
            description = row["Description"]
            real_stock_code = str(row["StockCode"]) # The real ID (e.g., 22384)

        # Step 3: Find Related Products using the Description
        # We search Pinecone using the clean description to find similar items
        related_products = search_related_products(description, top_k=5)

        # Step 4: Build Response
        return jsonify({
            "detected_class": real_stock_code,      # Return the actual StockCode (e.g., 22384)
            "detected_label": predicted_class,      # Return the CNN class name (e.g. lunch_bag...)
            "identified_description": description,
            "confidence": float(confidence),
            "response": f"Identified as: {description}",
            "products": related_products
        })

    except Exception as e:
        logger.error(f"Product detection failed: {e}")
        return jsonify({
            "detected_class": "ERROR",
            "products": [],
            "response": "Product detection failed due to an internal error."
        }), 500
 
# === Run ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)