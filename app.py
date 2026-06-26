# ============================================================
# Streamlit web calculator:
# ICU-acquired pressure injury prediction using final SVM model
# ============================================================

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="ICU Pressure Injury Risk Calculator",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 注意：这里要和 GitHub 仓库里的文件名完全一致
MODEL_PATH = BASE_DIR / "final_full_data_SVM.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"


# ============================================================
# Load model and metadata
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Please make sure final_full_data_SVM.pkl is uploaded to the GitHub repository root."
        )

    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)

    return loaded_model


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}. "
            "Please make sure model_metadata.json is uploaded to the GitHub repository root."
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


model = load_model()
metadata = load_metadata()

features = metadata["features"]
feature_labels = metadata["feature_labels"]
encoding = metadata["categorical_encoding"]
risk_thresholds = metadata["risk_thresholds"]


# ============================================================
# Helper functions
# ============================================================

def predict_probability(input_df: pd.DataFrame) -> float:
    """
    Predict probability using the loaded final SVM model.
    The saved model should include the same preprocessing pipeline
    used during model development.
    """

    # 情况1：直接保存的是 sklearn 模型或 Pipeline
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[:, 1][0]

    # 情况2：保存的是字典，模型在 model["model"] 里面
    elif isinstance(model, dict) and "model" in model:
        fitted_model = model["model"]

        if hasattr(fitted_model, "predict_proba"):
            prob = fitted_model.predict_proba(input_df)[:, 1][0]
        else:
            raise ValueError(
                "The fitted model inside the saved dictionary does not support predict_proba."
            )

    else:
        raise ValueError(
            "The loaded object does not support predict_proba. "
            "Please make sure the uploaded .pkl file is the final SVM probability model or pipeline."
        )

    return float(prob)


def classify_risk(prob: float) -> str:
    low_cut = risk_thresholds.get("low", 0.10)
    moderate_cut = risk_thresholds.get("moderate", 0.20)

    if prob < low_cut:
        return "Low risk"
    elif prob < moderate_cut:
        return "Moderate risk"
    else:
        return "High risk"


def risk_message(prob: float) -> str:
    risk_group = classify_risk(prob)

    if risk_group == "Low risk":
        return "The predicted risk is low. Continue routine pressure injury prevention and monitoring."
    elif risk_group == "Moderate risk":
        return "The predicted risk is moderate. Consider enhanced preventive measures and closer skin assessment."
    else:
        return "The predicted risk is high. Consider intensive pressure injury prevention, frequent repositioning, and comprehensive clinical assessment."


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("About this tool")

st.sidebar.markdown(
    """
This web calculator estimates the predicted risk of **ICU-acquired pressure injury**
using a final SVM model.

**Important:** This tool is intended for research demonstration only.  
It should not be used as a standalone clinical decision-making tool before multicenter external validation.
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
**Predictors**

1. Age  
2. SOFA score  
3. SpO₂  
4. Hemoglobin  
5. Albumin  
6. Mechanical ventilation  
7. Ventilation mode  
8. Vasoactive drugs
"""
)


# ============================================================
# Main page
# ============================================================

st.title("ICU-Acquired Pressure Injury Risk Calculator")

st.markdown(
    """
This calculator provides an estimated probability of ICU-acquired pressure injury
based on the final support vector machine model.

Please enter the patient's baseline clinical information below.
"""
)

st.warning(
    "For research demonstration only. Not intended for direct clinical decision-making before external validation."
)


# ============================================================
# Input form
# ============================================================

with st.form("prediction_form"):

    st.subheader("Patient information")

    age_years = st.number_input(
        "Age, years",
        min_value=18,
        max_value=110,
        value=65,
        step=1
    )

    sofa_score = st.number_input(
        "SOFA score",
        min_value=0,
        max_value=24,
        value=6,
        step=1
    )

    spo2_pct = st.number_input(
        "SpO₂, %",
        min_value=50.0,
        max_value=100.0,
        value=95.0,
        step=0.1
    )

    hemoglobin_gl = st.number_input(
        "Hemoglobin, g/L",
        min_value=30.0,
        max_value=220.0,
        value=110.0,
        step=1.0
    )

    albumin_gl = st.number_input(
        "Albumin, g/L",
        min_value=10.0,
        max_value=60.0,
        value=35.0,
        step=0.1
    )

    mechanical_ventilation_label = st.selectbox(
        "Mechanical ventilation",
        options=["No", "Yes"],
        index=0
    )

    ventilation_mode_label = st.selectbox(
        "Ventilation mode",
        options=[
            "No ventilation",
            "Non-invasive ventilation",
            "Invasive ventilation"
        ],
        index=0
    )

    vasoactive_drugs_label = st.selectbox(
        "Vasoactive drugs",
        options=["No", "Yes"],
        index=0
    )

    submitted = st.form_submit_button("Calculate risk")


# ============================================================
# Prediction
# ============================================================

if submitted:

    input_data = {
        "age_years": age_years,
        "sofa_score": sofa_score,
        "spo2_pct": spo2_pct,
        "hemoglobin_gl": hemoglobin_gl,
        "albumin_gl": albumin_gl,
        "mechanical_ventilation": encoding["mechanical_ventilation"][mechanical_ventilation_label],
        "ventilation_mode": encoding["ventilation_mode"][ventilation_mode_label],
        "vasoactive_drugs": encoding["vasoactive_drugs"][vasoactive_drugs_label]
    }

    input_df = pd.DataFrame([input_data], columns=features)

    try:
        prob = predict_probability(input_df)
        risk_group = classify_risk(prob)

        st.markdown("---")
        st.subheader("Prediction result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted probability",
                value=f"{prob * 100:.1f}%"
            )

        with col2:
            st.metric(
                label="Risk category",
                value=risk_group
            )

        if risk_group == "Low risk":
            st.success(risk_message(prob))
        elif risk_group == "Moderate risk":
            st.warning(risk_message(prob))
        else:
            st.error(risk_message(prob))

        st.markdown("### Input summary")

        display_df = pd.DataFrame({
            "Predictor": [
                "Age, years",
                "SOFA score",
                "SpO₂, %",
                "Hemoglobin, g/L",
                "Albumin, g/L",
                "Mechanical ventilation",
                "Ventilation mode",
                "Vasoactive drugs"
            ],
            "Value": [
                age_years,
                sofa_score,
                spo2_pct,
                hemoglobin_gl,
                albumin_gl,
                mechanical_ventilation_label,
                ventilation_mode_label,
                vasoactive_drugs_label
            ]
        })

        st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(
            "Prediction failed. Please check whether the saved model is compatible with the app."
        )
        st.exception(e)


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "This model is intended for research demonstration only. "
    "Clinical use requires further external validation and local calibration."
)
