# app.py — Streamlit app for Skin Grafting (Hybrid ANN + CNN)

import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from tensorflow.keras.models import load_model

from predict import preprocess_input
from cnn_feature_extractor import extract_features


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Based Skin Graft Survival Prediction",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------------------------
# Load Artifacts
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("best_model.h5")
    encoder = joblib.load("encoder.joblib")
    template_df = pd.read_pickle("template_df.pkl")
    return model, encoder, template_df

model, encoder, template_df = load_artifacts()

encoder_cols = list(encoder.get_feature_names_out())
categorical_columns = list({c.split("_")[0] for c in encoder_cols})
numerical_columns = [c for c in template_df.columns if c not in encoder_cols]


# -------------------------------------------------
# Header / Hero Section
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🩺 AI-Based Skin Graft Survival Predictor</h1>
    <p style='text-align:center; font-size:18px; color:gray;'>
    Multimodal prediction using clinical parameters + wound image analysis (CNN + ANN)
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Input Form
# -------------------------------------------------
with st.form("prediction_form"):

    # -------- Clinical Section --------
    st.markdown(
        """
        <div style='padding:18px; border-radius:12px; background-color:#111827;'>
        <h3>🧬 Clinical Parameters</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        A = st.text_input("A (comma-separated)", "3,9")
        B = st.text_input("B (comma-separated)", "7,40")
        C = st.text_input("C (comma-separated)", "w3,-")
        DR = st.text_input("DR (comma-separated)", "w2,-")

    with col2:
        A1 = st.text_input("A.1 (comma-separated)", "1,3")
        B1 = st.text_input("B.1 (comma-separated)", "7,8")
        C1 = st.text_input("C.1 (comma-separated)", "w1,w1")
        DR1 = st.text_input("DR.1 (comma-separated)", "w2,w3")

    MLC = st.number_input("MLC (numeric)", value=0, step=1)

    # -------- Image Section --------
    st.markdown(
        """
        <div style='padding:18px; border-radius:12px; background-color:#111827; margin-top:20px;'>
        <h3>📷 Wound Image Upload</h3>
        <p style='color:gray;'>Optional but improves prediction accuracy</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(
        "Upload wound image",
        type=["jpg", "jpeg", "png"]
    )

    submit_btn = st.form_submit_button("🔍 Predict Survival")


# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
if submit_btn:

    inp = {
        "A": A, "B": B, "C": C, "DR": DR,
        "A.1": A1, "B.1": B1, "C.1": C1, "DR.1": DR1,
        "MLC": MLC
    }

    df_inp = pd.DataFrame([inp])

    processed_tabular = preprocess_input(
        df_inp,
        encoder,
        numerical_columns,
        categorical_columns,
        template_df
    )

    # CNN features
    if uploaded_image is not None:
        temp_img_path = "temp_uploaded_image.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        cnn_features = extract_features(temp_img_path)
    else:
        cnn_features = np.zeros(128)

    final_features = np.concatenate(
        [processed_tabular.values, cnn_features.reshape(1, -1)],
        axis=1
    )

    prediction = model.predict(final_features)
    survival_val = float(prediction[0][0])

    # -------- Result Section --------
    st.markdown("### 🔍 Prediction Result")

    st.metric(
    label="Predicted Graft Survival Score",
    value=f"{survival_val:.4f}"
)

    # Relative survival score (UI only)
    relative_score = (survival_val - 6.16) / (21.0 - 6.16)
    st.progress(float(np.clip(relative_score, 0, 1)))
    
    # Convert to percentile
    percentile = int(relative_score * 100)

    st.caption(
        f"📊 This prediction lies around the **{percentile}th percentile** "
        f"of the training dataset."
)

    # Dataset-based interpretation thresholds
    Q1 = 10.070535
    Q2 = 11.503043

    if survival_val < Q1:
        st.error("⚠️ Low predicted graft survival")
    elif survival_val < Q2:
        st.warning("⚠️ Moderate graft survival")
    else:
        st.success("✅ High graft survival")


    # Save prediction
    try:
        out_row = inp.copy()
        out_row["predicted_SURVIVAL"] = survival_val
        pd.DataFrame([out_row]).to_csv(
            "predictions.csv",
            mode="a",
            header=not os.path.exists("predictions.csv"),
            index=False
        )
        st.success("Prediction saved to predictions.csv")
    except Exception as e:
        st.error(f"Could not save prediction: {e}")


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Built by Vanshika Shukla and Rajnandni Singh | Hybrid CNN + ANN | Academic Project"
    "</p>",
    unsafe_allow_html=True
)
