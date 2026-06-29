# ============================================================
# Compact one-screen Streamlit calculator:
# ICU-acquired pressure injury prediction using final SVM model
# Screenshot-friendly version for manuscript
# ============================================================

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 1. Page configuration
# ============================================================

st.set_page_config(
    page_title="ICU Pressure Injury Risk Calculator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. Compact CSS for one-screen screenshot
# ============================================================

st.markdown(
    """
    <style>
    /* Page width and spacing */
    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 0.8rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }

    /* Hide Streamlit menu/footer for cleaner screenshot */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Title */
    h1 {
        font-size: 2.05rem !important;
        line-height: 1.15 !important;
        margin-bottom: 0.25rem !important;
        color: #1f2937;
    }

    h2, h3 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.45rem !important;
        color: #1f2937;
    }

    /* Reduce vertical gaps */
    div[data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 1.0rem;
    }

    /* Input label */
    label {
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
        margin-bottom: 0.1rem !important;
    }

    /* Input boxes */
    div[data-baseweb="input"] {
        min-height: 2.25rem !important;
    }

    div[data-baseweb="select"] {
        min-height: 2.25rem !important;
    }

    input {
        font-size: 0.92rem !important;
    }

    /* Cards */
    .card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 0.85rem;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .result-card {
        background-color: #f8fafc;
        border: 1px solid #dbeafe;
        border-radius: 0.85rem;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .title-note {
        font-size: 0.98rem;
        color: #4b5563;
        line-height: 1.45;
        margin-bottom: 0.55rem;
    }

    .disclaimer {
        font-size: 0.82rem;
        color: #92400e;
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 0.55rem;
        padding: 0.55rem 0.75rem;
        line-height: 1.35;
        margin-top: 0.35rem;
        margin-bottom: 0.65rem;
    }

    .small-note {
        font-size: 0.84rem;
        color: #6b7280;
        line-height: 1.45;
    }

    .risk-low {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        border-radius: 0.65rem;
        padding: 0.65rem 0.8rem;
        font-size: 0.92rem;
        line-height: 1.4;
        font-weight: 600;
    }

    .risk-moderate {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        border-radius: 0.65rem;
        padding: 0.65rem 0.8rem;
        font-size: 0.92rem;
        line-height: 1.4;
        font-weight: 600;
    }

    .risk-high {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        border-radius: 0.65rem;
        padding: 0.65rem 0.8rem;
        font-size: 0.92rem;
        line-height: 1.4;
        font-weight: 600;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 0.75rem 0.85rem;
        border-radius: 0.7rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.88rem !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        color: #111827;
    }

    .footer-note {
        font-size: 0.76rem;
        color: #6b7280;
        line-height: 1.35;
        margin-top: 0.65rem;
        border-top: 1px solid #e5e7eb;
        padding-top: 0.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATE_PATHS = [
    BASE_DIR / "final_svm_model.pkl",
    BASE_DIR / "model" / "final_svm_model.pkl"
]

METADATA_CANDIDATE_PATHS = [
    BASE_DIR / "model_metadata.json",
    BASE_DIR / "model" / "model_metadata.json"
]


def find_existing_file(candidate_paths, file_description):
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
# 4. Default metadata
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
# 5. Load model and metadata
# ============================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return DEFAULT_METADATA

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

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
encoding = metadata.get("categorical_encoding", DEFAULT_METADATA["categorical_encoding"])
risk_thresholds = metadata.get("risk_thresholds", DEFAULT_METADATA["risk_thresholds"])


# ============================================================
# 6. Helper functions
# ============================================================

def get_predictor(model_object):
    """
    Extract a fitted model or pipeline that supports predict_proba().
    """
    if hasattr(model_object, "predict_proba"):
        return model_object

    if isinstance(model_object, dict):
        for key in ["model", "pipeline", "fitted_model", "best_model", "final_model"]:
            if key in model_object and hasattr(model_object[key], "predict_proba"):
                return model_object[key]

    raise ValueError(
        "No fitted model or pipeline with predict_proba() was found in the uploaded .pkl file. "
        f"The loaded object type is: {type(model_object)}."
    )


predictor = get_predictor(model)


def predict_probability(input_df: pd.DataFrame) -> float:
    input_df = input_df.copy()
    input_df = input_df[features]
    prob = predictor.predict_proba(input_df)[:, 1][0]
    return float(prob)


def classify_risk(prob: float) -> str:
    low_cut = float(risk_thresholds.get("low", 0.10))
    moderate_cut = float(risk_thresholds.get("moderate", 0.20))

    if prob < low_cut:
        return "Low risk"
    elif prob < moderate_cut:
        return "Moderate risk"
    else:
        return "High risk"


def risk_message(prob: float) -> str:
    risk_group = classify_risk(prob)

    if risk_group == "Low risk":
        return "Routine pressure injury prevention and standard skin monitoring are suggested."
    elif risk_group == "Moderate risk":
        return "Enhanced preventive measures and closer skin assessment should be considered."
    else:
        return "Intensive prevention, frequent repositioning, and pressure redistribution should be considered."


def risk_css_class(risk_group: str) -> str:
    if risk_group == "Low risk":
        return "risk-low"
    elif risk_group == "Moderate risk":
        return "risk-moderate"
    else:
        return "risk-high"


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

    return pd.DataFrame([input_data], columns=features)


# ============================================================
# 7. Header
# ============================================================

st.title("ICU-Acquired Pressure Injury Risk Calculator")

st.markdown(
    """
    <div class="title-note">
    Estimated risk of ICU-acquired pressure injury based on a final support vector machine model.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="disclaimer">
    Research demonstration only. This tool is not intended for direct clinical decision-making before external validation and local calibration.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 8. One-screen layout
# ============================================================

left_col, right_col = st.columns([1.65, 1.0], gap="large")


# ============================================================
# 9. Inputs
# ============================================================

with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Patient information")

    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)

    with row1_col1:
        age_years = st.number_input(
            "Age, years",
            min_value=18,
            max_value=110,
            value=65,
            step=1
        )

    with row1_col2:
        sofa_score = st.number_input(
            "SOFA score",
            min_value=0,
            max_value=24,
            value=6,
            step=1
        )

    with row1_col3:
        spo2_pct = st.number_input(
            "SpO₂, %",
            min_value=50.0,
            max_value=100.0,
            value=95.0,
            step=0.1,
            format="%.1f"
        )

    with row1_col4:
        hemoglobin_gl = st.number_input(
            "Hemoglobin, g/L",
            min_value=30.0,
            max_value=220.0,
            value=110.0,
            step=1.0,
            format="%.0f"
        )

    with row2_col1:
        albumin_gl = st.number_input(
            "Albumin, g/L",
            min_value=10.0,
            max_value=60.0,
            value=35.0,
            step=0.1,
            format="%.1f"
        )

    with row2_col2:
        mechanical_ventilation_label = st.selectbox(
            "Mechanical ventilation",
            options=["No", "Yes"],
            index=0
        )

    with row2_col3:
        ventilation_mode_label = st.selectbox(
            "Ventilation mode",
            options=[
                "No ventilation",
                "Non-invasive ventilation",
                "Invasive ventilation"
            ],
            index=0
        )

    with row2_col4:
        vasoactive_drugs_label = st.selectbox(
            "Vasoactive drugs",
            options=["No", "Yes"],
            index=0
        )

    st.markdown(
        """
        <div class="small-note">
        All predictors should be assessed at baseline or within the first 24 hours after ICU admission.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 10. Automatic prediction
# ============================================================

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
    risk_text = risk_message(prob)
    css_class = risk_css_class(risk_group)
except Exception as e:
    prob = None
    risk_group = "Prediction failed"
    risk_text = str(e)
    css_class = "risk-high"


# ============================================================
# 11. Result panel
# ============================================================

with right_col:

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.subheader("Prediction result")

    if prob is not None:
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                label="Predicted probability",
                value=f"{prob * 100:.1f}%"
            )

        with metric_col2:
            st.metric(
                label="Risk category",
                value=risk_group
            )

        st.markdown(
            f"""
            <div class="{css_class}">
            {risk_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="small-note">
            Risk thresholds: low &lt; {float(risk_thresholds.get("low", 0.10)):.2f}; 
            moderate {float(risk_thresholds.get("low", 0.10)):.2f}–{float(risk_thresholds.get("moderate", 0.20)):.2f}; 
            high ≥ {float(risk_thresholds.get("moderate", 0.20)):.2f}.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.metric(
            label="Predicted probability",
            value="—"
        )

        st.metric(
            label="Risk category",
            value="Prediction failed"
        )

        st.markdown(
            f"""
            <div class="{css_class}">
            {risk_text}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 12. Compact input summary for screenshot
# ============================================================

st.markdown(
    """
    <div class="footer-note">
    Model inputs: age, SOFA score, SpO₂, hemoglobin, albumin, mechanical ventilation, ventilation mode, and vasoactive drug use. 
    This web calculator is intended for research demonstration only; clinical use requires multicenter external validation and local calibration.
    </div>
    """,
    unsafe_allow_html=True
)
