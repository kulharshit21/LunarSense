
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="LunarSense-3 Dashboard", layout="wide")

st.title("🌙 LunarSense-3: Lunar Anomaly Detection")
st.markdown("**Multimodal Sensor Fusion for Chandrayaan-3**")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select", ["Dashboard", "Model Info", "Make Prediction", "Event Catalog"])

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("03_models/fusion_baseline_xgb.pkl")
    scalers = joblib.load("03_models/feature_scalers.pkl")
    return model, scalers

model, scalers = load_models()

# Load event catalog
@st.cache_data
def load_catalog():
    return pd.read_csv("07_deployment/event_catalog.csv")

# Dashboard
if page == "Dashboard":
    st.header("📊 Model Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "67.53%")
    with col2:
        st.metric("Precision", "65.52%")
    with col3:
        st.metric("Recall", "55.88%")
    with col4:
        st.metric("ROC-AUC", "0.7476")

    # Performance plot
    metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'ChaSTE': [0.8182, 0.8125, 0.5417, 0.6500],
        'ILSA': [0.7134, 0.3014, 0.3284, 0.3143],
        'Fusion': [0.6753, 0.6552, 0.5588, 0.6032]
    })

    fig = px.bar(metrics, x='Metric', y=['ChaSTE', 'ILSA', 'Fusion'],
                 title="Model Comparison", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# Model Info
elif page == "Model Info":
    st.header("🤖 Model Information")

    with open("07_deployment/MODEL_CARD.json") as f:
        model_card = json.load(f)

    st.json(model_card)

# Make Prediction
elif page == "Make Prediction":
    st.header("🔮 Make Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Thermal (ChaSTE)")
        mean_temp = st.slider("Mean Temperature (K)", 100.0, 350.0, 250.0)
        std_temp = st.slider("Std Temperature (K)", 0.0, 50.0, 10.0)
        min_temp = st.slider("Min Temperature (K)", 100.0, 300.0, 200.0)
        max_temp = st.slider("Max Temperature (K)", 200.0, 350.0, 300.0)
        drift_rate = st.slider("Drift Rate (K/h)", -10.0, 10.0, 0.0)
        qc_flag = st.selectbox("QC Flag", [0, 1, 2, 3])

    with col2:
        st.subheader("Seismic (ILSA)")
        n_events = st.slider("N Events", 0, 100, 10)
        max_amplitude = st.slider("Max Amplitude (m/s)", 0.0, 1.0, 0.5)
        rms = st.slider("RMS (m/s)", 0.0, 1.0, 0.3)
        max_sta_lta = st.slider("Max STA/LTA", 0.0, 10.0, 2.0)
        qc_flag2 = st.selectbox("Seismic QC Flag", [0, 1, 2, 3])

    if st.button("Predict"):
        X_chaste = scalers['chaste'].transform([[mean_temp, std_temp, min_temp, max_temp, drift_rate, qc_flag]])
        X_ilsa = scalers['ilsa'].transform([[n_events, max_amplitude, rms, max_sta_lta, qc_flag2]])
        X_fusion = np.concatenate([X_chaste, X_ilsa], axis=1)

        pred = model.predict(X_fusion)[0]
        prob = model.predict_proba(X_fusion)[0, 1]

        st.success(f"**Prediction:** {'🔴 ANOMALY' if pred == 1 else '🟢 NORMAL'}")
        st.info(f"**Confidence:** {prob*100:.1f}%")

# Event Catalog
elif page == "Event Catalog":
    st.header("📋 Event Catalog")
    catalog = load_catalog()

    st.dataframe(catalog, use_container_width=True)

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Events", len(catalog[catalog['prediction'] == 1]))
    with col2:
        st.metric("Detection Rate", f"{(catalog['prediction'].sum()/len(catalog)*100):.1f}%")
    with col3:
        st.metric("Avg Confidence", f"{catalog['confidence'].mean():.2f}")
