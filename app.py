import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model('/content/drive/MyDrive/final_model_revised/final_model.keras')

# Constants
IMG_SIZE = 150
CLASS_NAMES = ['inclusive for men', 'inclusive for both genders', 'not inclusive for both genders']

# Title and subtitle
st.title(" Graph Assessment Tool for Gender Inclusivity")
st.subheader(" Upload your graph image here")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Dropdown for mode
mode = st.selectbox("Choose mode:", ["Model Prediction", "Model Evaluation"])

# Button to trigger result
if st.button("Press for Result"):
    if mode == "Model Evaluation":
        # Load test data (optional: you can precompute and hardcode this)
        test_dir = "/content/drive/MyDrive/dataset/test2"
        test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_data = test_gen.flow_from_directory(
            directory=test_dir,
            batch_size=16,
            target_size=(IMG_SIZE, IMG_SIZE),
            class_mode='categorical',
            shuffle=False
        )
        loss, acc = model.evaluate(test_data, verbose=0)
        st.text(f" Test Accuracy: {acc:.4f}")
        st.text(f" Test Loss: {loss:.4f}")

    elif mode == "Model Prediction":
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            image = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred_probs = model.predict(img_array)
            pred_class = np.argmax(pred_probs, axis=1)[0]
            confidence = np.max(pred_probs)

            st.text(f" Prediction: {CLASS_NAMES[pred_class]}")
            st.text(f" Confidence: {confidence:.2f}")
        else:
            st.warning("Please upload an image first.")
