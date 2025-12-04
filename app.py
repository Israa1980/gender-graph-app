import os, zipfile, gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
# Page Title
st.title("Data Visualisation Inclusivity Assessment Tool: Gender Focus")
# --- App Overview / Mindmap ---
with st.container():
    st.markdown("### 🟨 Getting Started: Step-by-Step Guide")

    st.markdown("""
1) **Learn Colour Harmony Strategies**  
   ○ Explore definitions and examples of different colour harmony strategies.  
   ○ Each strategy is explained with a chart and colour wheel illustration.  

2) **Upload Your Visualisation**  
   ○ Use the uploader to provide one or more images of your data visualisation, as you can add as many images as needed, and click the 'See the Verdict' button to view all results at once.  
   ○ *Note*: For more accurate results, it is preferable to upload images sized **150 × 150 pixels** (width × height).  
   Larger or smaller images will still work, but may slightly affect prediction accuracy.  

3) **See Verdict**  
   ○ The model analyses your visualisation and provides a verdict:  
      ▪ Inclusive for both genders  
      ▪ Inclusive for male  
      ▪ Not inclusive for both genders  
   ○ You’ll also see a confidence score explaining how certain the model is.  

4) **Improvement Suggestions**  
   ○ If your visualisation is not fully inclusive, you’ll get tailored improvement strategies.  
   ○ Example images are shown to help you apply these strategies in practice.  

5) **Model Evaluation (Optional)**  
   ○ Curious about reliability? Expand the evaluation section to see accuracy, confusion matrices, and classification reports.  
   ○ This helps you understand how robust the model is across different test sets.  
""")


# Constants
IMG_SIZE = 150
BATCH_SIZE = 16
CLASS_NAMES = ['inclusive for both genders', 'inclusive for male', 'not inclusive for both genders']
# --- Download and display Colour Harmony Strategy Images ---
st.subheader("Color Harmony Strategies")
st. markdown(
    """
    Colour harmony strategies are design approaches that combine colours in ways that feel balanced,
    visually appealing, and accessible. In the context of inclusivity, these strategies help ensure
    that data visualisations are welcoming and easy to interpret for a wider audience.
    By applying harmonious colour schemes, we reduce bias, improve readability, and make charts
    more engaging for everyone, regardless of gender or background. Here are some well-known colour harmony strategies.
    """
)

# --- Strategy definitions and image IDs ---
strategy_definitions = {
    "Split Complementary (3 colours)": {
        "definition": """**Split Complementary Colour Strategy**  
A variation of complementary colours. You start with one base colour, then instead of using its direct opposite, 
you use the two colours adjacent to that complementary colour without choosing the complementary colour itself (see Figure 1).""",
        "file_id": "1DX36ehGdg36pCRvmMt8RFtpNaJp9Z3h_",   # bar chart image
        "wheel_file_id": "1mi7PwxDZ_fRPwB7uUD3vyebkJW6ohzXP", # colour wheel image        
        "caption": "Figure 1 Split Complementary Colour Strategy"
    },
    "Analogous (3 colours)": {
        "definition": """**Analogous Colour Strategy (Three Colours)**  
Uses three colours sitting side by side on the colour wheel. This creates a harmonious, natural appearance with smooth transitions (see Figure 2).""",
        "file_id": "1kABcT7fSdOQ77S3YbMjUcLUNKxqQ9Cvi",
         "wheel_file_id": "1VFqdudW9tr44I0nxrrDt2mlD5TreNKrP",
        "caption": "Figure 2 Analogous Colour Strategy (Three Colours)"
    },
    "Analogous (2 colours, base colour is dominant)": {
        "definition": """**Analogous (2 Colours, Base Colour is Dominant)**  
Uses two neighbouring colours on the wheel, with the base colour as the main focus and the adjacent colour serving as a secondary accent (see Figure 3).""",
        "file_id": "1v40ReTUNP7VcGqsKRRJJyQ4Mj7U9l0rT",
        "wheel_file_id": "1SiHSaSWj9cVbSefwiY13ar3RroCb6sLZ",
        "caption": "Figure 3 Analogous (2 Colours, Base Colour is Dominant)"
    },
    "Analogous (2 colours, Analogous colour is dominant)": {
        "definition": """**Analogous (2 Colours, Analogous Colour is Dominant)**  
Uses two neighbouring colours on the wheel, with the adjacent colour as the main focus and the base colour acting as a secondary accent (see Figure 4).""",
        "file_id": "1A9876j8pMySQeOhkgT6tJueOGMuXLHGD",
        "wheel_file_id": "16kF74Ig4FdVbsge3xFlGjm_XBm2l-TXC", # colour wheel image
        "caption": "Figure 4 Analogous (2 Colours, Analogous Colour is Dominant)"
    },
    "Complementary (base colour is dominant)": {
        "definition": """**Complementary (Base Colour is Dominant)**  
Uses two colours that are opposite each other on the colour wheel (e.g., blue and orange). 
The base colour serves as the main focus, covering most of the design, while the complementary colour is used in small amounts for accents (see Figure 5).""",
        "file_id": "1Kw_IBLmNDzQXumk1eFlN0r7BAj_ZikMX",
          "wheel_file_id": "1w7dBh0vpqeh-niIAr5dex3Ii_NMBa4XQ", # colour wheel image
        "caption": "Figure 5 Complementary (Base Colour is Dominant)"
    },
    "Complementary (complementary colour is dominant)": {
        "definition": """**Complementary (Complementary Colour is Dominant)**  
Uses two colours that are opposite each other on the colour wheel (e.g., blue and orange). 
The complementary colour serves as the main focus, covering most of the design, while the base colour is used in small amounts for accents (see Figure 6).""",
        "file_id": "1dGtQSmx2f1jfMQZnDeejRX8aDiWVNCcn",
        "wheel_file_id": "1n5_vfneBBB0DVRB3WnZTxx1hRH_6odUf", # colour wheel image
        "caption": "Figure 6 Complementary (Complementary Colour is Dominant)"
    },
    "Monochromatic (2 lighter shades)": {
        "definition": """**Monochromatic (2 Lighter Shades)**  
A colour scheme that uses a single hue along with two lighter tints of that same colour (see Figure 7).""",
        "file_id": "1HRP_LbFDB7M8F2EtLpV5zCt17IF9gxcr",
         "wheel_file_id": "1SkN1pIPiEsNSLz7ynO-F8Gfuyk4EDaL_", # colour wheel image
        "caption": "Figure 7 Monochromatic (2 Lighter Shades)"
    },
    "Monochromatic (1 lighter shade, base colour is dominant)": {
        "definition": """**Monochromatic (1 Lighter Shade, Base Colour is Dominant)**  
A single hue complemented by a lighter tint, with the primary colour as the focal point and the lighter shade used sparingly (see Figure 8).""",
        "file_id": "1u3vvu-DkRoS_ei8IJXf3yYBmZiTdjfR7",
        "wheel_file_id": "1InFhYABbtQPtSAc1VMkkvuitSAGXwQQu", # colour wheel image
        "caption": "Figure 8 Monochromatic (1 Lighter Shade, Base Colour is Dominant)"
    },
    "Monochromatic (1 lighter shade, lighter shade is dominant)": {
        "definition": """**Monochromatic (1 Lighter Shade, Lighter Shade is Dominant)**  
A single hue complemented by a lighter tint, with the lighter shade as the focal point and the primary colour used sparingly (see Figure 9).""",
        "file_id": "1YDPePQz34N-fyCqdxeBzMNiuGJr2neFC",
        "wheel_file_id": "1BSzicMqCyjJJJOF0OEVN9WvVaRobQC9o", # colour wheel image
        "caption": "Figure 9 Monochromatic (1 Lighter Shade, Lighter Shade is Dominant)"
    },
    "No colour harmony strategies": {
        "definition": """**No Colour Harmony Strategies**  
Use only a single colour without employing any colour harmony techniques (see Figure 10).""",
        "file_id": "11evyJZ6gDzheZFyQoxGX90ImWT6FYHcU",
        "caption": "Figure 10 No Colour Harmony Strategies"
    }
}

# Ensure local folder exists
os.makedirs("color_strategies", exist_ok=True)

# Display each strategy with definition above and bold caption below
for title, info in strategy_definitions.items():
    # Paths for bar chart and wheel images
    chart_path = f"color_strategies/{title.replace(' ', '_')}_chart.jpg"
    wheel_path = f"color_strategies/{title.replace(' ', '_')}_wheel.jpg"

    # Download bar chart image
    if not os.path.exists(chart_path):
        gdown.download(f"https://drive.google.com/uc?id={info['file_id']}", chart_path, quiet=False)

    # Download wheel image
    if "wheel_file_id" in info and not os.path.exists(wheel_path):
        gdown.download(f"https://drive.google.com/uc?id={info['wheel_file_id']}", wheel_path, quiet=False)

    # Show definition
    st.markdown(info["definition"])
    # Side-by-side display
    col1, col2 = st.columns(2)
    with col1:
        st.image(chart_path, width=300, caption=info["caption"])
    with col2:
        if "wheel_file_id" in info:
            st.image(wheel_path, width=300)

    st.markdown("---")


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
# --- Prediction Section ---
uploaded_files = st.file_uploader("Choose one or more images...",
                                  type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True)





# --- Class descriptions ---
CLASS_DESCRIPTIONS = {
    "inclusive for male": (
        "Your visualisation is considered **inclusive for males** as it uses one of the following colour harmony techniques: "
        "Complementary colour strategy (base colour dominant), complementary colour strategy (complementary colour dominant), "
        "monochromatic strategy (base colour dominant), or analogous strategy (analogous colour dominant), "
        "which our studies suggest these approaches are generally inclusive for male users.\n\n"
        "**To improve inclusivity**, you could adopt strategies such as:\n"
        "- Split complementary colour harmony (three complementary colours)\n"
        "- Analogous strategy using three analogous colours\n"
        "- Analogous strategy with two analogous colours where the base colour is dominant."
    ),
    "inclusive for both genders": (
        "Well done! Your data visualisation is considered **inclusive for both genders** as it uses one of the following colour harmony techniques:\n"
        "- Split complementary (three colours)\n"
        "- Analogous (three colours)\n"
        "- Analogous (two colours, base colour dominant), which our studies suggest is generally inclusive for both genders of users."
    ),
    "not inclusive for both genders": (
        "Your visualisation is considered **not inclusive for both genders** as it uses one of the following colour harmony techniques: "
        "no colour strategy (all bars one colour), monochromatic with two lighter shades, or monochromatic where the lighter shade is dominant. "
        "Our studies suggest these approaches are not generally inclusive for both genders of users.\n\n"
        "**To improve inclusivity**, you could adopt strategies such as:\n"
        "- Split complementary colour harmony (three complementary colours)\n"
        "- Analogous strategy using three analogous colours\n"
        "- Analogous strategy with two analogous colours where the base colour is dominant."
    )
}

# --- Improvement strategy image mapping ---
IMPROVEMENT_IMAGES = {
    "split_complementary": {
        "file_id": "11xPmNavyXrtnX-2s5M82WQAaupyy9UvB",
        "desc": "Split complementary colour harmony (three complementary colours)."
    },
    "analogous_three": {
        "file_id": "1wKXUv_NJiMsulV0_-J1FYWCG9pq-lPHJ",
        "desc": "Analogous strategy using three analogous colours."
    },
    "analogous_two_base": {
        "file_id": "18t8PXkyWvPT9K-pBz5Yrp9vVKRFa1SuU",
        "desc": "Analogous strategy with two analogous colours where the base colour is dominant."
    }
}

# --- Verdict-to-strategy mapping ---
VERDICT_IMPROVEMENTS = {
    "inclusive for male": ["split_complementary", "analogous_three", "analogous_two_base"],
    "not inclusive for both genders": ["split_complementary", "analogous_three", "analogous_two_base"]
}

# --- Helper to download image from Drive by file ID ---
def download_image(file_id, target_dir=tempfile.gettempdir()):
    out_path = os.path.join(target_dir, f"{file_id}.png")
    if not os.path.exists(out_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, out_path, quiet=True)
    return out_path

# --- Main Streamlit logic ---
if st.button("See Verdict"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            import io
            image_bytes = uploaded_file.read()
            image = tf.keras.utils.load_img(io.BytesIO(image_bytes), target_size=(IMG_SIZE, IMG_SIZE))
            img_array = tf.keras.utils.img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred_probs = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred_probs, axis=1)[0]
            confidence = np.max(pred_probs)

            class_name = CLASS_NAMES[pred_class]
            explanation = CLASS_DESCRIPTIONS.get(class_name, "No explanation available.")

            st.text(f"File: {uploaded_file.name}")
            st.text(f"Verdict: {class_name}")

            # --- Confidence explanation ---
            confidence_msg = f"""
            **Confidence Score:** {confidence:.2f}  
            The confidence score indicates how certain the AI model is about the choice it has made. 
            This score ranges from 0 to 1, with values closer to 1 indicating that the model is more confident in its decision.  
            For instance, a score of **{confidence:.2f}** means the model is {confidence*100:.0f}% sure that the visualisation it selected **is {class_name}**.
            """
            if confidence >= 0.75:
                confidence_msg += "\n The model demonstrates strong confidence in this result, making the verdict highly reliable."
            elif confidence >= 0.60:
                confidence_msg += "\n The model shows moderate confidence in this prediction. While the outcome may be accurate, there is still a significant chance of error."
            else:
                confidence_msg += "\n The model’s confidence in this prediction is low, so keep in mind it could be incorrect."

            st.markdown(confidence_msg)

            # --- Explanation ---
            st.markdown(explanation)
            st.markdown("---")

            # --- Show improvement examples if applicable ---
            if class_name in VERDICT_IMPROVEMENTS:
                st.markdown("**Visual examples of suggested strategies:**")
                for strategy_key in VERDICT_IMPROVEMENTS[class_name]:
                    strategy = IMPROVEMENT_IMAGES.get(strategy_key)
                    if strategy:
                        try:
                            img_path = download_image(strategy["file_id"])
                            st.image(img_path, width=400)
                            st.markdown(strategy["desc"])
                        except Exception as e:
                            st.warning(f"Could not load image for {strategy_key}.")
    else:
        st.warning("Please upload one or more images.")


    
    


# Initialize session state for expander
if "expander_open" not in st.session_state:
    st.session_state.expander_open = True
# --- Evaluation Section Trigger ---
with st.expander("Want to Know If a Model Really Works? Click Here"):
    
    st.markdown(
        """
        ### What Does “Evaluation” Mean?
        Evaluation is like a reality check for the model. 
        We test it on two kinds of data:\n
        **Test Set**: Data saved from training, used to check how well the model learned.  
        **Similar Data Set**: Completely new data that the model has never encountered is used to evaluate its ability to handle new situations.

        Why Test the Model’s Reliability?  
        Evaluating both sets helps you understand three key aspects:  
        **Accuracy** refers to how often the model predicts correctly.  
        **Generalisation** shows whether it performs well on new, unseen data.  
        **Error patterns** reveal which classes the model most often confuses. 

        This matters because a model that performs well only on training data may not be trustworthy in practice.  
        By comparing results across both test sets, you can see if the model is truly **robust and reliable**.

        Click below to **Test the Model’s Reliability** and view the full performance report.
        """
    )
    if st.button("Test the Model’s Reliability"):
        st.caption("Get accuracy scores, confusion matrix, and detailed classification reports")        
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        # Test Set
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

        # similar Data Set
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
        st. markdown(
            f"""
            - **Test Set Accuracy:** {acc_split:.4f}  
            - **Similar Data Set Accuracy:** {acc_sep:.4f}  

            Higher accuracy means the model is making more correct predictions. 
            If accuracy is much lower on a similar data set, it suggests the model may not generalise well.
            """
        )
        

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
        axes[1].set_title("Similar Data Set")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(
            """
            **How to read these heatmaps:**  
            - Each row shows the *true class*.  
            - Each column shows the *predicted class*.  
            - Diagonal values are correct predictions.  
            - Off-diagonal values show misclassifications.  

            This helps you see which classes the model confuses most often.
            """
        )

        # --- Classification Reports ---
        st.subheader("Classification Reports")

        # Generate reports as dicts
        report_split_dict = classification_report(y_true_split, y_pred_split, target_names=CLASS_NAMES, output_dict=True)

        report_sep_dict = classification_report(y_true_sep, y_pred_sep, target_names=CLASS_NAMES, output_dict=True)


       # Convert to DataFrames
        df_split = pd.DataFrame(report_split_dict).transpose()
        df_sep = pd.DataFrame(report_sep_dict).transpose()
        # Drop summary rows you don’t want
        rows_to_drop = ["accuracy", "macro avg", "weighted avg"]
        df_split = df_split.drop(rows_to_drop, errors="ignore")
        df_sep = df_sep.drop(rows_to_drop, errors="ignore")

       # Display nicely in tables
        st.markdown("**Test Set Report**")
        st.dataframe(df_split.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

        st.markdown("**Similar Data Set Report**")
        st.dataframe(df_sep.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))
       
        st. markdown(
            """
            **Why this matters:**  
            The tables show precision, recall, F1-score, and support for each class.  
            - **Precision**: How often predictions for a class are correct.  
            - **Recall**: How often the class is correctly identified.  
            - **F1-score**: Balance between precision and recall.  
            - **Support**: Number of samples for each class.  
       
          - These metrics indicate not only *how often* the model is correct, but also *how effectively* it manages each class.
          - For instance, low recall for "inclusive for both genders" would imply that the model frequently fails to recognise charts that are inclusive for both genders.

            """
        )































 
     















  







    


     
       
       

       
