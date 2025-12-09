import streamlit as st
import requests
import pandas as pd
from PIL import Image

# --- Configuration ---
# If running Flask locally, use localhost. If on a cloud VM, use that IP.
BACKEND_URL = "https://special-robot-p6rrx9xppxr3qrj-5000.app.github.dev/"

st.set_page_config(page_title="E-Commerce AI System", layout="wide")

# --- UI Layout ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module:", [
    "Page 1: Text Query Search",
    "Page 2: Handwritten Query (OCR)",
    "Page 3: Visual Product Search (CNN)"
])

st.sidebar.markdown("---")
st.sidebar.info(f"Connected to Backend: `{BACKEND_URL}`")

# --- Helper to Display Results ---
def show_results(response_json):
    if "response" in response_json:
        st.success(response_json["response"])
    
    # Extra info for OCR/CNN
    if "extracted_text" in response_json:
        st.info(f"**Extracted Text:** {response_json['extracted_text']}")
    if "detected_class" in response_json:
        st.info(f"**Detected StockCode:** {response_json['detected_class']} (Confidence: {response_json.get('confidence', 0):.2%})")

    # Display Product Table
    products = response_json.get("products", [])
    if products:
        st.subheader("🛒 Recommended Products")
        df = pd.DataFrame(products)
        # Clean up columns for display
        cols_to_show = ["description", "unit_price", "country", "stock_code", "score"]
        existing_cols = [c for c in cols_to_show if c in df.columns]
        st.dataframe(df[existing_cols], use_container_width=True)
    else:
        st.warning("No product recommendations found.")

# ==========================================
# PAGE 1: Text Query Interface
# ==========================================
if page == "Page 1: Text Query Search":
    st.header("🔍 Search Products by Text")
    st.markdown("Enter a natural language description (e.g., *'red retrospot tea set'*).")

    with st.form("text_search"):
        query = st.text_input("Search Query")
        submitted = st.form_submit_button("Search")
        
        if submitted and query:
            with st.spinner("Searching vector database..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/recommend", json={"query": query})
                    if res.status_code == 200:
                        show_results(res.json())
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

# ==========================================
# PAGE 2: Image Query Interface (OCR)
# ==========================================
elif page == "Page 2: Handwritten Query (OCR)":
    st.header("📝 Search by Handwritten Note")
    st.markdown("Upload an image of a handwritten note to find products.")

    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Your Note", width=300)
        
        if st.button("Process Handwriting"):
            with st.spinner("Reading text and searching..."):
                try:
                    files = {"image": uploaded_file.getvalue()}
                    res = requests.post(f"{BACKEND_URL}/recommend-ocr", files=files)
                    if res.status_code == 200:
                        show_results(res.json())
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

# ==========================================
# PAGE 3: Product Image Upload Interface
# ==========================================
elif page == "Page 3: Visual Product Search (CNN)":
    st.header("📸 Visual Product Detection")
    st.markdown("Upload a photo of a product to identify it and find similar items.")

    uploaded_file = st.file_uploader("Upload Product Photo", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Product Photo", width=300)
        
        if st.button("Identify Product"):
            with st.spinner("Running CNN Model..."):
                try:
                    files = {"image": uploaded_file.getvalue()}
                    res = requests.post(f"{BACKEND_URL}/detect-product", files=files)
                    if res.status_code == 200:
                        show_results(res.json())
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")