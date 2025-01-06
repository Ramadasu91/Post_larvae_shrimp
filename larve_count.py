# File: app.py
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to detect shrimps in the image
def detect_shrimps(image):
    # Convert to numpy array
    img_np = np.array(image)
    
    # Resize for consistency
    resized = cv2.resize(img_np, (800, 600))
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to segment the image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours and filter by size
    shrimp_count = 0
    output = resized.copy()
    for contour in contours:
        # Filter by area size to eliminate small noise
        area = cv2.contourArea(contour)
        if 500 < area < 10000:  # Adjust these values as needed
            shrimp_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, "Shrimp", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    return output, shrimp_count

# Streamlit UI
def main():
    st.title("Shrimp Detection App")
    st.write("Upload an image to detect shrimps.")

    # File uploader
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image
        st.write("Processing the image...")
        detected_image, shrimp_count = detect_shrimps(image)
        
        # Display the processed image
        st.image(detected_image, caption="Detected Shrimps", use_column_width=True)
        
        # Display the shrimp count
        st.write(f"Number of shrimps detected: {shrimp_count}")

# Run the app
if __name__ == "__main__":
    main()
