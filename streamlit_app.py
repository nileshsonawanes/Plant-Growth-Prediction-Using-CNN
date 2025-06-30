import streamlit as st
import os
import cv2
import pickle
import shutil
import plant_disease
import predict as pc

from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# === Setup ===
st.set_page_config(page_title="Harvestify", layout="wide")
st.title("🌿 Harvestify - Smart Plant Analysis")

# === Paths ===
input_dir = "static/input"
os.makedirs(input_dir, exist_ok=True)

# === Load Fertilizer Model ===
classifier_path = os.path.join(os.path.dirname(__file__), "classifier.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "fertilizer.pkl")

model = pickle.load(open(classifier_path, "rb"))
ferti_encoder = pickle.load(open(encoder_path, "rb"))

# === Sidebar Navigation ===
option = st.sidebar.radio("Choose a Feature", [
    "🏡 Home",
    "🌱 Plant Growth Prediction",
    "🦠 Disease Detection",
    "🧪 Fertilizer Recommendation"
])

# === HOME ===
if option == "🏡 Home":
    st.header("Welcome to Harvestify!")
    st.markdown("""
        **Harvestify** is a smart agriculture tool that helps:
        - 🌱 Predict plant growth stage from images
        - 🦠 Detect plant diseases
        - 🧪 Recommend fertilizers based on soil and crop data
        
        Upload images or input parameters and get instant AI-based insights!
    """)

# === GROWTH PREDICTION ===
elif option == "🌱 Plant Growth Prediction":
    st.header("🌱 Plant Growth Stage Prediction")
    uploaded = st.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"], key="growth")

    if uploaded is not None:
        filepath = os.path.join(input_dir, uploaded.name)
        with open(filepath, "wb") as f:
            f.write(uploaded.getbuffer())

        st.image(uploaded, caption="Uploaded Image", width=300)
        st.write("Predicting...")

        try:
            closest_stage, stages_result, additional_info_dict = pc.main(f"input/{uploaded.name}")
            st.success(f"Predicted Growth Stage: Stage {closest_stage}")

            st.subheader("🌿 Stage-wise Result")
            for stage, img in stages_result.items():
                st.markdown(f"**{stage}** {img}")

            st.subheader("📋 Recommendations")
            for k, v in additional_info_dict.items():
                st.markdown(f"**Stage {k}:**")
                st.markdown(f"- {v['plant_info']}")
                st.markdown(f"- 💧 {v['soil_moisture']}")
                st.markdown(f"- 🌾 {v['fertilizer_recommendation']}")
        except Exception as e:
            st.error(f"Error: {e}")

# === DISEASE DETECTION ===
elif option == "🦠 Disease Detection":
    st.header("🦠 Plant Disease Detection")
    uploaded = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"], key="disease")

    if uploaded is not None:
        filepath = os.path.join(input_dir, uploaded.name)
        with open(filepath, "wb") as f:
            f.write(uploaded.getbuffer())

        st.image(uploaded, caption="Uploaded Image", width=300)
        st.write("Detecting...")

        try:
            label, info = plant_disease.predict_disease(filepath)
            st.success(f"Prediction: {label}")
            st.markdown(f"**Info:** {info}")
        except Exception as e:
            st.error(f"Error: {e}")

# === FERTILIZER RECOMMENDATION ===
elif option == "🧪 Fertilizer Recommendation":
    st.header("🧪 Fertilizer Recommendation")
    with st.form("fertilizer_form"):
        temp = st.text_input("Temperature")
        humi = st.text_input("Humidity")
        mois = st.text_input("Moisture")
        soil = st.text_input("Soil Type (numeric)")
        crop = st.text_input("Crop Type (numeric)")
        nitro = st.text_input("Nitrogen")
        pota = st.text_input("Potassium")
        phosp = st.text_input("Phosphorous")

        submitted = st.form_submit_button("Predict Fertilizer")

        if submitted:
            try:
                features = [temp, humi, mois, soil, crop, nitro, pota, phosp]
                if not all(f.isdigit() for f in features):
                    st.error("Please enter valid numeric values.")
                else:
                    features = [int(f) for f in features]
                    pred_class = model.predict([features])[0]
                    ferti = ferti_encoder.inverse_transform([pred_class])[0]
                    st.success(f"Recommended Fertilizer: {ferti}")
            except Exception as e:
                st.error(f"Error: {e}")
