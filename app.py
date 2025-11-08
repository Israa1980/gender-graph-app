import os, zipfile, gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Page Title
st.title("Data Visualisation Inclusivity Assessment Tool: Gender Focus")

# --- Strategy definitions and image IDs ---
strategy_definitions = {
    "Split Complementary (3 colours)": {
        "definition": """**Split Complementary Colour Strategy**  
A variation of complementary colours. You start with one base colour, then instead of using its direct opposite, 
you use the two colours adjacent to that complementary colour without choosing the complementary colour itself (see Figure 1).""",
        "file_id": "1szBjfSzXe9evKK9i0NfoFvvhsgoX21zZ",
        "caption": "Figure 1 Split Complementary Colour Strategy"
    },
    "Analogous (3 colours)": {
        "definition": """**Analogous Colour Strategy (Three Colours)**  
Uses three colours sitting side by side on the colour wheel. This creates a harmonious, natural appearance with smooth transitions (see Figure 2).""",
        "file_id": "1kABcT7fSdOQ77S3YbMjUcLUNKxqQ9Cvi",
        "caption": "Figure 2 Analogous Colour Strategy (Three Colours)"
    },
    "Analogous (2 colours, base colour is dominant)": {
        "definition": """**Analogous (2 Colours, Base Colour is Dominant)**  
Uses two neighbouring colours on the wheel, with the base colour as the main focus and the adjacent colour serving as a secondary accent.""",
        "file_id": "1v40ReTUNP7VcGqsKRRJJyQ4Mj7U9l0rT",
        "caption": "Figure 3 Analogous (2 Colours, Base Colour is Dominant)"
    },
    "Analogous (2 colours, Analogous colour is dominant)": {
        "definition": """**Analogous (2 Colours, Analogous Colour is Dominant)**  
Uses two neighbouring colours on the wheel, with the adjacent colour as the main focus and the base colour acting as a secondary accent.""",
        "file_id": "1A9876j8pMySQeOhkgT6tJueOGMuXLHGD",
        "caption": "Figure 4 Analogous (2 Colours, Analogous Colour is Dominant)"
    },
    "Complementary (base colour is dominant)": {
        "definition": """**Complementary (Base Colour is Dominant)**  
Uses two colours that are opposite each other on the colour wheel (e.g., blue and orange). 
The base colour serves as the main focus, covering most of the design, while the complementary colour is used in small amounts for accents.""",
        "file_id": "1Kw_IBLmNDzQXumk1eFlN0r7BAj_ZikMX",
        "caption": "Figure 5 Complementary (Base Colour is Dominant)"
    },
    "Complementary (complementary colour is dominant)": {
        "definition": """**Complementary (Complementary Colour is Dominant)**  
Uses two colours that are opposite each other on the colour wheel (e.g., blue and orange). 
The complementary colour serves as the main focus, covering most of the design, while the base colour is used in small amounts for accents.""",
        "file_id": "1dGtQSmx2f1jfMQZnDeejRX8aDiWVNCcn",
        "caption": "Figure 6 Complementary (Complementary Colour is Dominant)"
    },
    "Monochromatic (2 lighter shades)": {
        "definition": """**Monochromatic (2 Lighter Shades)**  
A colour scheme that uses a single hue along with two lighter tints of that same colour.""",
        "file_id": "1HRP_LbFDB7M8F2EtLpV5zCt17IF9gxcr",
        "caption": "Figure 7 Monochromatic (2 Lighter Shades)"
    },
    "Monochromatic (1 lighter shade, base colour is dominant)": {
        "definition": """**Monochromatic (1 Lighter Shade, Base Colour is Dominant)**  
A single hue complemented by a lighter tint, with the primary colour as the focal point and the lighter shade used sparingly.""",
        "file_id": "1u3vvu-DkRoS_ei8IJXf3yYBmZiTdjfR7",
        "caption": "Figure 8 Monochromatic (1 Lighter Shade, Base Colour is Dominant)"
    },
    "Monochromatic (1 lighter shade, lighter shade is dominant)": {
        "definition": """**Monochromatic (1 Lighter Shade, Lighter Shade is Dominant)**  
A single hue complemented by a lighter tint, with the lighter shade as the focal point and the primary colour used sparingly.""",
        "file_id": "1YDPePQz34N-fyCqdxeBzMNiuGJr2neFC",
        "caption": "Figure 9 Monochromatic (1 Lighter Shade, Lighter Shade is Dominant)"
    },
    "No colour harmony strategies": {
        "definition": """**No Colour Harmony Strategies**  
Use only a single colour without employing any colour harmony techniques.""",
        "file_id": "11evyJZ6gDzheZFyQoxGX90ImWT6FYHcU",
        "caption": "Figure 10 No Colour Harmony Strategies"
    }
}

# Ensure local folder exists
os.makedirs("color_strategies", exist_ok=True)

# Display each strategy with definition above and caption below
for title, info in strategy_definitions.items():
    local_path = f"color_strategies/{title.replace(' ', '_')}.jpg"
    if not os.path.exists(local_path):
        gdown.download(f"https://drive.google.com/uc?id={info['file_id']}", local_path, quiet=False)
    st.markdown(info["definition"])
    st.image(local_path, caption=info["caption"], width=300)
    st.markdown("---")  



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
               " Your visualisation is considered **inclusive for males** because it uses one or more of the following colour harmony techniques:\n"
               "-  Complementary colour strategy (base colour dominant)\n"
               "-  Complementary colour strategy (complementary colour dominant)\n"
               "-  Monochromatic strategy (base colour dominant)\n"
               "-  Analogous strategy (analogous colour dominant)\n"
               "- To make it inclusive for *both genders*, try strategies like split complementary or three-colour analogous schemes\n"
               "- Split complementary colour harmony strategies (three complementary colours)\n"
               "- Analogous strategy (using three analogous colours)\n"
               "- Analogous strategy (using two analogous colours, base colour is the dominant colour)"

        ),
       "inclusive for both genders": (
        " Well done! Your data visualisation is considered **inclusive for both genders** because it uses one or more of the following colour harmony techniques:\n"
        "- Split complementary (three colours)\n"
        "- Analogous (three colours)\n"
        "- Analogous (two colours, base colour dominant)"
        ),
        "not inclusive for both genders": (
        " Your visualisation is considered **not inclusive for both genders** because it uses one of the following approaches:\n"
        "- No colour strategy (all bars one colour)\n"
        "- Monochromatic (two lighter shades)\n"
        "- Monochromatic (lighter shade dominant)\n"
        " To improve inclusivity, try split complementary or three-colour analogous strategies\n"
            "-  Split complementary colour harmony strategies (three complementary colours)\n"
            "-  Analogous strategy (using three analogous colours)\n"
            "-  Analogous strategy (using two analogous colours, base colour is the dominant colour)"

        )
    }
    
    if st.button("See Verdict"):
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
                st.text(f"verdict: {class_name}")
                st.markdown(
                    f"""
                    **Confidence Score:** {confidence:.2f}  
                    The confidence score indicates how certain the AI model is about the choice it has made. 
                    This score ranges from 0 to 1, with values closer to 1 indicating that the model is more confident in its decision.  
                    For instance, a score of **{confidence:.2f}** means the model is {confidence*100:.0f}% sure that the visualisation it selected **is {class_name}**.
                    """
                )
                   


                

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
        st.text(f" Test Set Accuracy:    {acc_split:.4f}")
        st.text(f"similar data Set Accuracy: {acc_sep:.4f}")

        # Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        cm_split = confusion_matrix(y_true_split, y_pred_split)
        sns.heatmap(cm_split, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
        axes[0].set_title("Test Set")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        cm_sep = confusion_matrix(y_true_sep, y_pred_sep)
        sns.heatmap(cm_sep, annot=True, fmt="d", cmap="Greens",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
        axes[1].set_title("similar data Set")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        plt.tight_layout()
        st.pyplot(fig)

        # Classification Reports
        st.subheader("Classification Reports")
        report_split = classification_report(y_true_split, y_pred_split, target_names=CLASS_NAMES)
        report_sep = classification_report(y_true_sep, y_pred_sep, target_names=CLASS_NAMES)

        st.text("Test Set Report:\n" + report_split)
        st.text("Similar data Set Report:\n" + report_sep)

        

       





     
       
       

       
