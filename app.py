import os, zipfile, gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile

# ---------------------------------------------------
# PAGE TITLE + OVERVIEW
# ---------------------------------------------------

st.title("Data Visualisation Inclusivity Assessment Tool: Gender Focus")

with st.container():
    st.markdown("### 🟨 Getting Started: Step-by-Step Guide")

    st.markdown("""
1) **Learn Colour Harmony Strategies**  
   ○ Explore definitions and examples of different colour harmony strategies.  
   ○ Each strategy is explained with a chart and colour wheel illustration.  

2) **Upload Your Visualisation**  
   ○ Use the uploader to provide one or more images of your data visualisation.  
   ○ You can add as many images as needed and click **See the Verdict** to view all results at once.  
   ○ *Note*: For best results, upload images sized **150 × 150 pixels**.  

3) **See Verdict**  
   ○ The model analyses your visualisation and provides a verdict:  
      ▪ Inclusive for both genders  
      ▪ Inclusive for male  
      ▪ Not inclusive for both genders  
   ○ You’ll also see a confidence score explaining how certain the model is.  

4) **Improvement Suggestions**  
   ○ If your visualisation is not fully inclusive, you’ll get tailored improvement strategies.  
   ○ Example images are shown to help you apply these strategies.  

5) **Model Evaluation (Optional)**  
   ○ Expand the evaluation section to see accuracy, confusion matrices, and classification reports.  
   ○ This helps you understand how robust the model is.
""")

# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------

IMG_SIZE = 150
BATCH_SIZE = 16
CLASS_NAMES = [
    'inclusive for both genders',
    'inclusive for male',
    'not inclusive for both genders'
]

# ---------------------------------------------------
# CACHED FUNCTIONS
# ---------------------------------------------------
@st.cache_resource
def load_model_cached():
    """Load the TensorFlow model once."""
    if not os.path.exists('small_cnn_1_1.keras'):
        model_file_id = "1nkmMunVmkRCgmPeF_vrsWY56HmPCS9lV"
        gdown.download(
            f"https://drive.google.com/uc?id={model_file_id}",
            'small_cnn_1_1.keras',
            quiet=False
        )
    return tf.keras.models.load_model('small_cnn_1_1.keras')


@st.cache_data
def ensure_test_dataset():
    """Download and extract test dataset (safe to cache)."""
    if not os.path.exists("test_1_1"):
        split_zip_id = "1xklR2o42Xg5ZpAUenD5FE93mnjfY_4JP"
        gdown.download(
            f"https://drive.google.com/uc?id={split_zip_id}",
            "test_1_1.zip",
            quiet=False
        )
        with zipfile.ZipFile("test_1_1.zip", "r") as zip_ref:
            zip_ref.extractall("test_1_1")
    return "test_1_1"


def prepare_test_dataset():
    """Create DirectoryIterator (NOT cached — fixes crash)."""
    ensure_test_dataset()
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    return test_datagen.flow_from_directory(
        "test_1_1",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )


@st.cache_data
def download_and_cache_image(file_id, filename):
    """Download image only once."""
    os.makedirs("color_strategies", exist_ok=True)
    out_path = os.path.join("color_strategies", filename)

    if not os.path.exists(out_path):
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            out_path,
            quiet=True
        )
    return out_path


# Load model once
model = load_model_cached()
# ---------------------------------------------------
# COLOUR HARMONY STRATEGIES
# ---------------------------------------------------

st.subheader("Color Harmony Strategies")

st.markdown("""
Colour harmony strategies combine colours in ways that feel balanced and accessible.
They help ensure that data visualisations are welcoming and easy to interpret for a wider audience.
""")

st.subheader("Color Harmony Strategies")

strategy_definitions = {
    "Split Complementary (3 colours)": {
        "definition": """**Split Complementary Colour Strategy**  
A variation of complementary colours. You start with one base colour, then instead of using its direct opposite, 
you use the two colours adjacent to that complementary colour without choosing the complementary colour itself (see Figure 1).""",
        "file_id": "1DX36ehGdg36pCRvmMt8RFtpNaJp9Z3h_",   # bar chart image
        "wheel_file_id": "1821e8EDcova8JS39ScgjWHTGcIOD6_EJ", # colour wheel image        
        "caption": "Figure 1 Split Complementary Colour Strategy"
    },
    "Analogous (3 colours)": {
        "definition": """**Analogous Colour Strategy (Three Colours)**  
Uses three colours sitting side by side on the colour wheel. This creates a harmonious, natural appearance with smooth transitions (see Figure 2).""",
        "file_id": "1RXysLypvs9YDGTU1T5O3qz-A0MJrGDQv",
         "wheel_file_id": "1NB2DHBTFlKyodeUzxsLm_n5PM8VWeuCs",
        "caption": "Figure 2 Analogous Colour Strategy (Three Colours)"
    },
    "Analogous (2 colours, base colour is dominant)": {
        "definition": """**Analogous (2 Colours, Base Colour is Dominant)**  
Uses two neighbouring colours on the wheel, with the base colour as the main focus and the adjacent colour serving as a secondary accent (see Figure 3).""",
        "file_id": "15bmGnz14rfQr3easGvdMYw-NqoH49WIH",
        "wheel_file_id": "1hW1txjzQBJevfRZwNUdJaL4UvCpuy9eu",
        "caption": "Figure 3 Analogous (2 Colours, Base Colour is Dominant)"
    },
    "Analogous (2 colours, Analogous colour is dominant)": {
        "definition": """**Analogous (2 Colours, Analogous Colour is Dominant)**  
Uses two neighbouring colours on the wheel, with the adjacent colour as the main focus and the base colour acting as a secondary accent (see Figure 4).""",
        "file_id": "1GWIyke0WQQcIHNXFFcPLfU6dzU5wmI89",
        "wheel_file_id": "1nwYdRcZGbUGj3Mc84qs1dAJFSV5N6AV6", # colour wheel image
        "caption": "Figure 4 Analogous (2 Colours, Analogous Colour is Dominant)"
    },
    "Complementary (base colour is dominant)": {
        "definition": """**Complementary (Base Colour is Dominant)**  
Uses two colours that are opposite each other on the colour wheel (e.g., blue and orange). 
The base colour serves as the main focus, covering most of the design, while the complementary colour is used in small amounts for accents (see Figure 5).""",
        "file_id": "1ZNY50ltt2wfP9GalDbsYM1NXp23hTG98",
        "wheel_file_id": "1C1zDvBHs0IaWNPe5vfQiBqMHVAIGNMer", # colour wheel image
        "caption": "Figure 5 Complementary (Base Colour is Dominant)"
    },
    "Complementary (complementary colour is dominant)": {
        "definition": """**Complementary (Complementary Colour is Dominant)**  
Uses two colours that are opposite each other on the colour wheel (e.g., blue and orange). 
The complementary colour serves as the main focus, covering most of the design, while the base colour is used in small amounts for accents (see Figure 6).""",
        "file_id": "1KZ2lyZYyjTWeu_NrjcivTFPhrMpki2oV",
        "wheel_file_id": "1ygogkS82ihinvasUYi-RP_1AbMSb1RMs", # colour wheel image
        "caption": "Figure 6 Complementary (Complementary Colour is Dominant)"
    },
    "Monochromatic (2 lighter shades)": {
        "definition": """**Monochromatic (2 Lighter Shades)**  
A colour scheme that uses a single hue along with two lighter tints of that same colour (see Figure 7).""",
        "file_id": "1aleSAy2-SKhacIBfHn7CcIOCmB6Jayai",
         "wheel_file_id": "11PoB6w-0bfpunDQc_bkO0tEAM88rkpGa", # colour wheel image
        "caption": "Figure 7 Monochromatic (2 Lighter Shades)"
    },
    "Monochromatic (1 lighter shade, base colour is dominant)": {
        "definition": """**Monochromatic (1 Lighter Shade, Base Colour is Dominant)**  
A single hue complemented by a lighter tint, with the primary colour as the focal point and the lighter shade used sparingly (see Figure 8).""",
        "file_id": "1_2xQtJPcSQGVVaPb8pwK97oSyhX5NXUZ",
        "wheel_file_id": "1hBvAvvbqC103rlE05Jxi4WNZt18UZHWJ", # colour wheel image
        "caption": "Figure 8 Monochromatic (1 Lighter Shade, Base Colour is Dominant)"
    },
    "Monochromatic (1 lighter shade, lighter shade is dominant)": {
        "definition": """**Monochromatic (1 Lighter Shade, Lighter Shade is Dominant)**  
A single hue complemented by a lighter tint, with the lighter shade as the focal point and the primary colour used sparingly (see Figure 9).""",
        "file_id": "1cbjyhdAg9FkPyLFlCiJ9akezrS8qxwmE",
        "wheel_file_id": "1l4hDds7xA4ORojIJNjWwdc58VEjU-U2f", # colour wheel image
        "caption": "Figure 9 Monochromatic (1 Lighter Shade, Lighter Shade is Dominant)"
    },
    "No colour harmony strategies": {
        "definition": """**No Colour Harmony Strategies**  
Use only a single colour without employing any colour harmony techniques (see Figure 10).""",
        "file_id": "142kIRmnKDKScjSfHx9LMC7yHvO5901cK",
        "caption": "Figure 10 No Colour Harmony Strategies"
    }
}


for title, info in strategy_definitions.items():
    chart_path = download_and_cache_image(info["file_id"], f"{title}_chart.jpg")

    wheel_path = None
    if "wheel_file_id" in info:
        wheel_path = download_and_cache_image(info["wheel_file_id"], f"{title}_wheel.jpg")

    st.markdown(info["definition"])
    col1, col2 = st.columns(2)

    with col1:
        st.image(chart_path, width=300, caption=info["caption"])

    with col2:
        if wheel_path:
            st.image(wheel_path, width=300)

    st.markdown("---")

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------

uploaded_files = st.file_uploader(
    "Choose one or more images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if st.button("See Verdict"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            import io
            image_bytes = uploaded_file.read()
            image = tf.keras.utils.load_img(
                io.BytesIO(image_bytes),
                target_size=(IMG_SIZE, IMG_SIZE)
            )
            img_array = tf.keras.utils.img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred_probs = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred_probs, axis=1)[0]
            confidence = np.max(pred_probs)

            class_name = CLASS_NAMES[pred_class]

            st.text(f"File: {uploaded_file.name}")
            st.text(f"Verdict: {class_name}")

            st.markdown(f"**Confidence Score:** {confidence:.2f}")

            st.markdown("---")
    else:
        st.warning("Please upload one or more images.")

# ---------------------------------------------------
# EVALUATION SECTION (ONE EXPANDER)
# ---------------------------------------------------

with st.expander("Model Evaluation (Test Set Only)"):

    test_gen = prepare_test_dataset()

    y_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = test_gen.classes
    acc = np.mean(y_true == y_pred)

    st.markdown(f"**Test Set Accuracy:** {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )
    st.pyplot(fig)

    st.markdown("""
    **How to read this heatmap:**  
    - Rows = true classes  
    - Columns = predicted classes  
    - Diagonal = correct predictions  
    - Off-diagonal = misclassifications  
    """)

    # Classification Report
    st.subheader("Classification Report")

    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    df_report = pd.DataFrame(report_dict).transpose()
    df_report = df_report.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")

    st.dataframe(
        df_report.style.format({
            "precision": "{:.2f}",
            "recall": "{:.2f}",
            "f1-score": "{:.2f}",
            "support": "{:.0f}"
        })
    )

    st.markdown("""
    **Why this matters:**  
    - **Precision**: How often predictions for a class are correct  
    - **Recall**: How often the class is correctly identified  
    - **F1-score**: Balance between precision and recall  
    - **Support**: Number of samples per class 
    These metrics indicate not only *how often* the model is correct, but also *how effectively* it manages each class. 
    """)

