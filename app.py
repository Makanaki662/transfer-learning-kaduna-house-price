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
# (must match what was used during training)
# ─────────────────────────────────────────
property_type_map = {
    'Bungalow':    0,
    'Duplex':      1,
    'Flat':        2,
    'Self-contain': 3
}

condition_map = {
    'Fair': 0,
    'New':  1,
    'Old':  2
}

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
    condition_encoded      = condition_map[condition]

    # Build feature array
    # Order must match SHARED_FEATURES used in training:
    # ['Bedrooms', 'Bathrooms', 'Size_sqm',
    #  'Property_Type', 'Condition']
    features = np.array([[
        bedrooms,
        bathrooms,
        size_sqm,
        property_type_encoded,
        condition_encoded
    ]])

    # Stage 1 — GBR raw prediction (transfer step)
    raw_prediction = gbr_model.predict(features)[0]

    # Stage 2 — Calibration (domain adaptation)
    calibrated_prediction = calibrator.predict(
        np.array([[raw_prediction]])
    )[0]

    # Convert to Naira (approximate exchange rate)
    usd_to_naira = 1600
    price_naira  = calibrated_prediction * usd_to_naira

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

    # ── Pipeline breakdown ──
    with st.expander("See prediction pipeline breakdown"):
        st.write("**Step 1 — GBR Raw Prediction (Malaysia model):**")
        st.write(f"${raw_prediction:,.2f}")
        st.write("**Step 2 — Calibration Formula Applied:**")
        st.write(
            f"Kaduna Price = 0.094719 × {raw_prediction:,.2f}"
            f" + (−5,160.56)"
        )
        st.write(f"**Final Calibrated Price = ${calibrated_prediction:,.2f}**")

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
# FOOTER — MODEL INFORMATION
# ─────────────────────────────────────────
st.divider()
st.subheader("Model Information")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.metric("Model R²", "0.9173")

with col_m2:
    st.metric("MAE", "$11,067")

with col_m3:
    st.metric("RMSE", "$18,123")

st.caption(
    "Source domain: Malaysia housing dataset (37,954 records) · "
    "Target domain: Kaduna South (37 records) · "
    "Calibration: Linear Regression · "
    "Base model: Gradient Boosting Regressor"
)