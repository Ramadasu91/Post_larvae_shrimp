import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Streamlit App Title
st.set_page_config(page_title="Skavch Post larvae Shrimp Count Engine", page_icon="📊", layout="wide")

# Add an image to the header
st.image("bg1.jpg", use_column_width=True)

st.title("Skavch Post Larvae Shrimp Counting in Image")

# File uploader for image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    # Load the image using PIL
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to segment shrimps
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shrimp_count = 0

    # Draw bounding boxes and count shrimps
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 5000:  # Filter contours by area (adjust based on shrimp size)
            shrimp_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Add shrimp count text to the image
    cv2.putText(image_np, f"Count: {shrimp_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display results
    st.image(image_np, caption=f"Shrimp Count: {shrimp_count}", use_column_width=True)
