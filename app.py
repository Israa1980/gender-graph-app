import os, zipfile, gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 150
BATCH_SIZE = 16
CLASS_NAMES = ['inclusive for both genders', 'inclusive for male', 'not inclusive for both genders']

# Mode selector
mode = st.selectbox("Choose mode:", ["Model Prediction (upload images)", "Model Evaluation (test folder)"])

# --- Download model if not already present ---
if not os.path.exists('small_cnn_1_1.keras'):
    model_file_id = "1nkmMunVmkRCgmPeF_vrsWY56HmPCS9lV"  # replace with your Google Drive file ID
    gdown.download(f"https://drive.google.com/uc?id={model_file_id}",
                   'small_cnn_1_1.keras', quiet=False)

model = tf.keras.models.load_model('small_cnn_1_1.keras')

# --- Download and unzip Split Test Set ---
if not os.path.exists("test_1_1"):
    split_zip_id = "1xklR2o42Xg5ZpAUenD5FE93mnjfY_4JP"  # replace with your Google Drive file ID
    gdown.download(f"https://drive.google.com/uc?id={split_zip_id}",
                   "test_1_1.zip", quiet=False)
    with zipfile.ZipFile("test_1_1.zip", "r") as zip_ref:
        zip_ref.extractall("test_1_1")

# --- Download and unzip Separate Test Set ---
if not os.path.exists("test_only_1"):
    separate_zip_id = "1qx8VoNZWvxqRmktbaDSdm9iDI0Ria9yO"  # replace with your Google Drive file ID
    gdown.download(f"https://drive.google.com/uc?id={separate_zip_id}",
                   "separate_test.zip", quiet=False)
    with zipfile.ZipFile("separate_test.zip", "r") as zip_ref:
        zip_ref.extractall("test_only_1")

# --- Prediction Mode ---
if mode == "Model Prediction (upload images)":
    uploaded_files = st.file_uploader("Choose one or more images...",
                                      type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True)

    CLASS_DESCRIPTIONS = {
        "inclusive for male": (
            "Inclusive for male includes:\n"
            "- Complementary colour strategy (two complementary colours, base colour is dominant)\n"
            "- Complementary colour strategy (two complementary colours, complementary colour is dominant)\n"
            "- Monochromatic colour strategy (one lighter shade of the base colour beside the base colour, base colour is dominant)\n"
            "- Analogous colour strategy (two analogous colours, analogous colour is dominant)"
        ),
        "inclusive for both genders": (
            "Inclusive for both genders includes:\n"
            "- Split complementary colour harmony strategies (three complementary colours)\n"
            "- Analogous strategy (using three analogous colours)\n"
            "- Analogous strategy (using two analogous colours, base colour is the dominant colour)"
        ),
        "not inclusive for both genders": (
            "Not inclusive for both genders includes:\n"
            "- No colour strategy applied (bars all with one colour)\n"
            "- Monochromatic strategy (using two lighter shades of the base colour beside the base colour)\n"
            "- Monochromatic strategy (using one lighter shade of the base colour beside the base colour, lighter shade is dominant)"
        )
    }

    if st.button("Run Predictions"):
        if uploaded_files:
            for file in uploaded_files:
                image = tf.keras.utils.load_img(file, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = tf.keras.utils.img_to_array(image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                pred_probs = model.predict(img_array, verbose=0)
                pred_class = np.argmax(pred_probs, axis=1)[0]
                confidence = np.max(pred_probs)

                class_name = CLASS_NAMES[pred_class]
                description = CLASS_DESCRIPTIONS.get(class_name, "No description available.")

                st.text(f"File: {file.name}")
                st.text(f"Prediction: {class_name}")
                st.text(f"Confidence: {confidence:.2f}")
                st.markdown(f"**Description:** {description}")
                st.markdown("---")
        else:
            st.warning("Please upload one or more images.")

# --- Evaluation Mode ---
elif mode == "Model Evaluation (test folder)":
    if st.button("Evaluate on Both Test Sets"):
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        # Split Test Set
        split_gen = test_datagen.flow_from_directory(
            "test_1_1",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False
        )
        y_probs_split = model.predict(split_gen, verbose=0)
        y_pred_split = np.argmax(y_probs_split, axis=1)
        y_true_split = split_gen.classes
        acc_split = np.mean(y_true_split == y_pred_split)

        # Separate Test Set
        separate_gen = test_datagen.flow_from_directory(
            "test_only_1",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False
        )
        y_probs_sep = model.predict(separate_gen, verbose=0)
        y_pred_sep = np.argmax(y_probs_sep, axis=1)
        y_true_sep = separate_gen.classes
        acc_sep = np.mean(y_true_sep == y_pred_sep)

        # Display Results
        st.subheader("Evaluation Results")
        st.text(f"Split Test Set Accuracy:    {acc_split:.4f}")
        st.text(f"Separate Test Set Accuracy: {acc_sep:.4f}")

        # Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        cm_split = confusion_matrix(y_true_split, y_pred_split)
        sns.heatmap(cm_split, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
        axes[0].set_title("Split Test Set")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        cm_sep = confusion_matrix(y_true_sep, y_pred_sep)
        sns.heatmap(cm_sep, annot=True, fmt="d", cmap="Greens",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
        axes[1].set_title("Separate Test Set")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")

        st.pyplot(fig)

        # Classification Reports
        st.subheader("Classification Reports")
        report_split = classification_report(y_true_split, y_pred_split, target_names=CLASS_NAMES)
        report_sep = classification_report(y_true_sep, y_pred_sep, target_names=CLASS_NAMES)

        st.text("Split Test Set Report:\n" + report_split)
        st.text("Separate Test Set Report:\n" + report_sep)

        

       
