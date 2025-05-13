import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load Haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError('Failed to load face cascade classifier!')

st.title("🔍 Viola-Jones Face Detection App")

st.markdown("""
Upload an image to detect faces using the **Viola-Jones algorithm**!
""")

uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

scaleFactor = st.slider("🔍 Scale Factor", 1.05, 1.5, 1.1, 0.01)
minNeighbors = st.slider("👥 Min Neighbors", 1, 10, 5)
rect_color = st.color_picker("🎨 Choose Rectangle Color", "#FF0000")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(60, 60)
    )

    # Convert hex to BGR
    rect_rgb = tuple(int(rect_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    rect_bgr = rect_rgb[::-1]

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), rect_bgr, 2)

    st.image(image_np, caption="🖼️ Detected Faces", use_container_width=True)

    if st.button("💾 Save Image"):
        result_image = Image.fromarray(image_np)
        result_image.save("detected_faces_output.png")
        st.success("Image saved as `detected_faces_output.png`!")
