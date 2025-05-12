import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load Haar cascade file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

st.title("🔍 Viola-Jones Face Detection App")
st.markdown("""
Welcome to the face detection demo using the **Viola-Jones algorithm** with Haar cascades. 
Upload an image, tweak detection parameters, and see the results live. You can also download the final image.
""")

# Image uploader
uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

# Parameters
scaleFactor = st.slider("🔍 Scale Factor", 1.05, 1.5, 1.1, 0.01,
                        help="How much the image size is reduced at each image scale.")
minNeighbors = st.slider("👥 Min Neighbors", 1, 10, 5,
                         help="How many neighbors each candidate rectangle should have to retain it.")
rect_color = st.color_picker("🎨 Choose Rectangle Color", "#FF0000")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors
    )

    # Convert hex color to BGR
    rect_rgb = tuple(int(rect_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    rect_bgr = rect_rgb[::-1]  # Convert to BGR

    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), rect_bgr, 2)

    st.image(image_np, caption="🖼️ Detected Faces", use_column_width=True)

    # Save button
    save = st.button("💾 Save Image")
    if save:
        result_image = Image.fromarray(image_np)
        result_image.save("detected_faces_output.png")
        st.success("Image saved as `detected_faces_output.png`!")
