import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Kaduna South House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ─────────────────────────────────────────
# LOAD SAVED MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/gbr_model.pkl", "rb") as f:
        gbr = pickle.load(f)
    with open("models/calibrator.pkl", "rb") as f:
        cal = pickle.load(f)
    return gbr, cal

gbr_model, calibrator = load_models()

# ─────────────────────────────────────────
# ENCODING MAPS
# Must match encoding used during training
# ─────────────────────────────────────────
property_type_map = {
    'Bungalow':     0,
    'Duplex':       1,
    'Flat':         2,
    'Self-contain': 3
}

condition_map = {
    'Fair': 0,
    'New':  1,
    'Old':  2
}

# Fixed exchange rate used during model training
# Kaduna data was prepared using ₦1,500 per $1 USD
USD_TO_NAIRA = 1500

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🏠 Kaduna South House Price Predictor")
st.markdown(
    "This tool uses a **Transfer Learning** model trained on "
    "Malaysian housing data and calibrated with **37 Kaduna South "
    "housing records** to estimate local property prices."
)
st.divider()

# ─────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.selectbox(
        "Number of Bedrooms",
        options=[1, 2, 3, 4, 5, 6, 7],
        index=1
    )

    bathrooms = st.selectbox(
        "Number of Bathrooms",
        options=[1, 2, 3, 4, 5, 6, 7],
        index=1
    )

    size_sqm = st.number_input(
        "Property Size (sqm)",
        min_value=20,
        max_value=1000,
        value=120,
        step=10
    )

with col2:
    property_type = st.selectbox(
        "Property Type",
        options=list(property_type_map.keys())
    )

    condition = st.selectbox(
        "Property Condition",
        options=list(condition_map.keys()),
        help="New = Newly built or excellent condition | "
             "Fair = Average condition | "
             "Old = Older or poor condition"
    )

st.divider()

# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
if st.button("Predict House Price", type="primary",
             use_container_width=True):

    # Encode inputs
    property_type_encoded = property_type_map[property_type]
    condition_encoded     = condition_map[condition]

    # Build feature DataFrame with correct column names
    # Order must match SHARED_FEATURES used during training
    features = pd.DataFrame([[
        bedrooms,
        bathrooms,
        size_sqm,
        property_type_encoded,
        condition_encoded
    ]], columns=[
        'Bedrooms',
        'Bathrooms',
        'Size_sqm',
        'Property_Type',
        'Condition'
    ])

    # Stage 1 — GBR raw prediction (transfer step)
    # GBR trained on Malaysia predicts on Kaduna features
    raw_prediction = gbr_model.predict(features)[0]

    # Stage 2 — Calibration (domain adaptation)
    # Linear Regression maps Malaysian prediction
    # to Kaduna South price scale
    calibrated_prediction = calibrator.predict(
        np.array([[raw_prediction]])
    )[0]

    # Convert USD to Naira using fixed training rate
    price_naira = calibrated_prediction * USD_TO_NAIRA

    # ── Display Results ──
    st.subheader("Prediction Result")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.metric(
            label="Estimated Price (USD)",
            value=f"${calibrated_prediction:,.2f}"
        )

    with col_r2:
        st.metric(
            label="Estimated Price (Naira)",
            value=f"₦{price_naira:,.0f}"
        )

    # ── Pipeline Breakdown ──
    with st.expander("See prediction pipeline breakdown"):

        st.write(
            "**Step 1 — GBR Raw Prediction "
            "(Malaysia source model):**"
        )
        st.write(f"${raw_prediction:,.2f}")

        st.write(
            "**Step 2 — Calibration Formula Applied:**"
        )
        st.write(
            f"Kaduna Price = 0.094719 × "
            f"{raw_prediction:,.2f} + (−5,160.56)"
        )
        st.write(
            f"**Final Calibrated Price = "
            f"${calibrated_prediction:,.2f}**"
        )

        st.write(
            "**Step 3 — Currency Conversion:**"
        )
        st.write(
            f"₦{price_naira:,.0f} = "
            f"${calibrated_prediction:,.2f} × "
            f"₦{USD_TO_NAIRA:,}"
        )

    # ── Disclaimer ──
    st.info(
        "⚠️ **Disclaimer:** This estimate is generated using a "
        "transfer learning model trained on Malaysian housing data "
        "and calibrated using 37 Kaduna South housing records. "
        "Predictions are indicative estimates intended to support "
        "decision-making and should be interpreted alongside local "
        "market knowledge. Model accuracy: R² = 0.9173."
    )

# ─────────────────────────────────────────
# MODEL INFORMATION FOOTER
# ─────────────────────────────────────────
st.divider()
st.subheader("Model Information")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.metric(
        label="Model R²",
        value="0.9173"
    )

with col_m2:
    st.metric(
        label="MAE",
        value="$11,067"
    )

with col_m3:
    st.metric(
        label="RMSE",
        value="$18,123"
    )

st.caption(
    "Source domain: Malaysia housing dataset "
    "(37,954 records) · "
    "Target domain: Kaduna South (37 records) · "
    "Base model: Gradient Boosting Regressor · "
    "Calibration: Linear Regression · "
    "Exchange rate: ₦1,500 per $1 USD"
)