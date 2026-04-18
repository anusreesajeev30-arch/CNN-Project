import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("rice_model.h5")

classes = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

st.title("🍚 Rice Type Classification")

uploaded_file = st.file_uploader("Upload a rice image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    index = pred.argmax()
    confidence = np.max(pred)

    st.success(f"Predicted Rice Type: {classes[index]} ({confidence*100:.2f}% confidence)")