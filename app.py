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
# 1. Page configuration
# ============================================================

st.set_page_config(
    page_title="ICU Pressure Injury Risk Calculator",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# 2. Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 兼容两种文件结构：
# 结构1：模型文件放在根目录
#   app.py
#   final_svm_model.pkl
#   model_metadata.json
#
# 结构2：模型文件放在 model 文件夹
#   app.py
#   model/final_svm_model.pkl
#   model/model_metadata.json

MODEL_CANDIDATE_PATHS = [
    BASE_DIR / "final_svm_model.pkl",
    BASE_DIR / "model" / "final_svm_model.pkl"
]

METADATA_CANDIDATE_PATHS = [
    BASE_DIR / "model_metadata.json",
    BASE_DIR / "model" / "model_metadata.json"
]


def find_existing_file(candidate_paths, file_description):
    """
    Find the first existing file from candidate paths.
    """
    for path in candidate_paths:
        if path.exists():
            return path

    checked_paths = "\n".join([str(p) for p in candidate_paths])

    raise FileNotFoundError(
        f"{file_description} was not found. Checked paths:\n{checked_paths}"
    )


MODEL_PATH = find_existing_file(
    MODEL_CANDIDATE_PATHS,
    "Model file"
)

METADATA_PATH = find_existing_file(
    METADATA_CANDIDATE_PATHS,
    "Metadata file"
)


# ============================================================
# 3. Default metadata fallback
# ============================================================

DEFAULT_METADATA = {
    "model_name": "SVM model for ICU-acquired pressure injury prediction",
    "outcome": "ICU-acquired pressure injury",
    "features": [
        "age_years",
        "sofa_score",
        "spo2_pct",
        "hemoglobin_gl",
        "albumin_gl",
        "mechanical_ventilation",
        "ventilation_mode",
        "vasoactive_drugs"
    ],
    "feature_labels": {
        "age_years": "Age, years",
        "sofa_score": "SOFA score",
        "spo2_pct": "SpO₂, %",
        "hemoglobin_gl": "Hemoglobin, g/L",
        "albumin_gl": "Albumin, g/L",
        "mechanical_ventilation": "Mechanical ventilation",
        "ventilation_mode": "Ventilation mode",
        "vasoactive_drugs": "Vasoactive drugs"
    },
    "categorical_encoding": {
        "mechanical_ventilation": {
            "No": 0,
            "Yes": 1
        },
        "ventilation_mode": {
            "No ventilation": 0,
            "Non-invasive ventilation": 1,
            "Invasive ventilation": 2
        },
        "vasoactive_drugs": {
            "No": 0,
            "Yes": 1
        }
    },
    "risk_thresholds": {
        "low": 0.10,
        "moderate": 0.20
    },
    "disclaimer": (
        "This calculator is intended for research demonstration only and should not be used "
        "as a standalone clinical decision-making tool before external validation."
    )
}


# ============================================================
# 4. Load model and metadata
# ============================================================

@st.cache_resource
def load_model():
    """
    Load fitted SVM model or sklearn Pipeline.
    The model must support predict_proba().
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model


@st.cache_data
def load_metadata():
    """
    Load model metadata.
    """
    if not METADATA_PATH.exists():
        return DEFAULT_METADATA

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 如果 metadata 缺少某些字段，则用默认值补齐
    for key, value in DEFAULT_METADATA.items():
        if key not in metadata:
            metadata[key] = value

    return metadata


try:
    model = load_model()
    metadata = load_metadata()
except Exception as e:
    st.error("Failed to load model or metadata.")
    st.exception(e)
    st.stop()


features = metadata["features"]
feature_labels = metadata.get("feature_labels", DEFAULT_METADATA["feature_labels"])
encoding = metadata.get("categorical_encoding", DEFAULT_METADATA["categorical_encoding"])
risk_thresholds = metadata.get("risk_thresholds", DEFAULT_METADATA["risk_thresholds"])


# ============================================================
# 5. Helper functions
# ============================================================

def get_predictor(model_object):
    """
    Extract a fitted model or pipeline that supports predict_proba().
    Supports:
    1. sklearn Pipeline / fitted estimator with predict_proba()
    2. dict containing model / pipeline / fitted_model / best_model
    """
    if hasattr(model_object, "predict_proba"):
        return model_object

    if isinstance(model_object, dict):
        for key in ["model", "pipeline", "fitted_model", "best_model", "final_model"]:
            if key in model_object and hasattr(model_object[key], "predict_proba"):
                return model_object[key]

    raise ValueError(
        "No fitted model or pipeline with predict_proba() was found in the uploaded .pkl file. "
        f"The loaded object type is: {type(model_object)}. "
        "Please make sure final_svm_model.pkl contains a fitted sklearn Pipeline or SVC(probability=True)."
    )


predictor = get_predictor(model)


def predict_probability(input_df: pd.DataFrame) -> float:
    """
    Predict probability of ICU-acquired pressure injury.
    """
    input_df = input_df.copy()

    # 确保变量顺序和建模时一致
    input_df = input_df[features]

    prob = predictor.predict_proba(input_df)[:, 1][0]

    return float(prob)


def classify_risk(prob: float) -> str:
    """
    Classify predicted probability into risk categories.
    """
    low_cut = float(risk_thresholds.get("low", 0.10))
    moderate_cut = float(risk_thresholds.get("moderate", 0.20))

    if prob < low_cut:
        return "Low risk"
    elif prob < moderate_cut:
        return "Moderate risk"
    else:
        return "High risk"


def risk_message(prob: float) -> str:
    """
    Clinical-style explanation for risk category.
    """
    risk_group = classify_risk(prob)

    if risk_group == "Low risk":
        return (
            "The predicted risk is low. Continue routine pressure injury prevention "
            "and standard skin monitoring."
        )
    elif risk_group == "Moderate risk":
        return (
            "The predicted risk is moderate. Consider enhanced preventive measures, "
            "closer skin assessment, and individualized nursing care."
        )
    else:
        return (
            "The predicted risk is high. Consider intensive pressure injury prevention, "
            "frequent repositioning, pressure redistribution support surfaces, and "
            "comprehensive clinical assessment."
        )


def build_input_dataframe(
    age_years,
    sofa_score,
    spo2_pct,
    hemoglobin_gl,
    albumin_gl,
    mechanical_ventilation_label,
    ventilation_mode_label,
    vasoactive_drugs_label
):
    """
    Build input dataframe using the same feature names and order as model training.
    """
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

    return input_df


# ============================================================
# 6. Sidebar
# ============================================================

st.sidebar.title("About this tool")

st.sidebar.markdown(
    """
This web calculator estimates the predicted risk of **ICU-acquired pressure injury**
using the final support vector machine model.

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

with st.sidebar.expander("Technical information", expanded=False):
    st.write("Model path:")
    st.code(str(MODEL_PATH))

    st.write("Metadata path:")
    st.code(str(METADATA_PATH))

    st.write("Model file exists:")
    st.write(MODEL_PATH.exists())

    st.write("Metadata file exists:")
    st.write(METADATA_PATH.exists())

    st.write("Loaded model type:")
    st.code(str(type(model)))

    st.write("Predictor type:")
    st.code(str(type(predictor)))

    st.write("predict_proba available:")
    st.write(hasattr(predictor, "predict_proba"))


# ============================================================
# 7. Main page
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
# 8. Input form
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
# 9. Prediction output
# ============================================================

if submitted:

    input_df = build_input_dataframe(
        age_years=age_years,
        sofa_score=sofa_score,
        spo2_pct=spo2_pct,
        hemoglobin_gl=hemoglobin_gl,
        albumin_gl=albumin_gl,
        mechanical_ventilation_label=mechanical_ventilation_label,
        ventilation_mode_label=ventilation_mode_label,
        vasoactive_drugs_label=vasoactive_drugs_label
    )

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

        with st.expander("Encoded model input", expanded=False):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(
            "Prediction failed. Please check whether the saved model is compatible with the app."
        )
        st.exception(e)


# ============================================================
# 10. Footer
# ============================================================

st.markdown("---")

st.caption(
    "This model is intended for research demonstration only. "
    "Clinical use requires further external validation and local calibration."
)
