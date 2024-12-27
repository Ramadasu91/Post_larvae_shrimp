import cv2
import numpy as np
import streamlit as st
import tempfile

# Streamlit App Title
st.set_page_config(page_title="Skavch Postlarvae shrimp Count Engine", page_icon="📊", layout="wide")

# Add an image to the header
st.image("bg1.jpg", use_column_width=True)

st.title("Skavch Post Larvae Shrimp Detection and Counting")

# File uploader for video
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Read the video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    out = None

    # Background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    stframe = st.empty()  # Placeholder for displaying frames

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess each frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Background subtraction
        mask = back_sub.apply(blurred)
        _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shrimp_count = 0

        for contour in contours:
            # Filter out small or large contours
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Adjust the range based on shrimp size
                shrimp_count += 1
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Shrimp", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the count at the top left
        cv2.putText(frame, f"Count: {shrimp_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Initialize video writer
        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        out.write(frame)

        # Update the frame in Streamlit
        stframe.image(frame, channels="BGR")

    # Release resources
    cap.release()
    out.release()

    # Display the processed video for download
    st.success("Processing complete!")
    with open(output_path, "rb") as file:
        st.download_button("Download Processed Video", data=file, file_name="processed_video.avi", mime="video/avi")
