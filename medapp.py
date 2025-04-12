import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 🧠 Load DL model
model = load_model("disease_model.h5")

# 🔤 Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 🩺 Load symptom list
with open("symptom_list.pkl", "rb") as f:
    symptom_list = pickle.load(f)

# 🚀 Streamlit UI
st.set_page_config(page_title="🧠 Medical Chatbot", layout="centered")
st.title("🤖 HealthPath: AI Disease Predictor")
st.write("Select symptoms you're experiencing and get an AI-based diagnosis!")

# 📝 User Input
selected_symptoms = st.multiselect("Select your symptoms:", symptom_list)

# 🔍 Predict Button
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Encode user input
        input_data = np.zeros((1, len(symptom_list)))
        for symptom in selected_symptoms:
            if symptom in symptom_list:
                idx = symptom_list.index(symptom)
                input_data[0][idx] = 1

        # Prediction
        prediction = model.predict(input_data)
        predicted_index = np.argmax(prediction)
        predicted_disease = label_encoder.inverse_transform([predicted_index])[0]

        st.success(f"🧬 Predicted Disease: **{predicted_disease}**")
