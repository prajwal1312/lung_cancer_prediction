import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow.keras.backend as K

# Load the trained model
model = tf.keras.models.load_model("lung_cancer.keras")

# Class labels (modify according to dataset)
class_labels = ["adenocarcinoma", "benign", "squamous_carcinoma"]

# Function to preprocess image & predict
def predict_lung_cancer(img):
    img = img.resize((224, 224))  # Resize to model input size
    img = np.array(img)  # Normalize (0 to 1 scale)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Get predictions
    predictions = model.predict(img)[0]  # Extract first row of predictions
    # st.write("Raw Model Predictions:", predictions)  # Debugging raw output

    # Ensure softmax is applied (if model outputs logits instead of probabilities)
    if np.max(predictions) > 1:
        predictions = K.softmax(predictions).numpy()

    # Sorting predictions
    sorted_indices = np.argsort(predictions)[::-1]  # Descending order
    sorted_labels = [class_labels[i] for i in sorted_indices]
    sorted_confidences = [predictions[i] * 100 for i in sorted_indices]  # Convert to percentage

    return sorted_labels, sorted_confidences

# Streamlit UI
st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🩺", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton > button { background-color: #007BFF; color: white; font-size: 18px; border-radius: 8px; padding: 10px; }
        .stButton > button:hover { background-color: #0056b3; }
        .stFileUploader { border: 2px dashed #007BFF; padding: 15px; border-radius: 10px; }
        .stSuccess { color: green; font-size: 20px; font-weight: bold; }
        .stInfo { color: #ff9800; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# Title & Description
st.title("🩺 Lung Cancer Prediction")
st.subheader("Upload a lung scan image to classify it as **Benign, Squamous Carcinoma, or Adenocarcinoma**.")
st.write("🔹 The model analyzes the lung scan and provides a classification along with confidence percentage.")

# File uploader
uploaded_file = st.file_uploader("📂 Drag & drop or choose a lung scan image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    if st.button("🚀 Predict"):
        sorted_labels, sorted_confidences = predict_lung_cancer(image)

        # Display top prediction
        st.success(f"**Prediction:** {sorted_labels[0]}")
        st.info(f"**Confidence:** {sorted_confidences[0]:.2f}%")

        # Show all class probabilities
        st.write("### 📊 Prediction Probabilities:")
        for label, conf in zip(sorted_labels, sorted_confidences):
            st.write(f"🔹 {label}: {conf:.2f}%")

