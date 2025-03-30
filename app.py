import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# App title and description
st.title("🏥 Health Insurance Cost Prediction")
st.markdown("📝 Enter your details below to predict insurance cost")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", min_value=18, max_value=100, value=25)
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=20.0)
    children = st.number_input("👶 Number of Children", min_value=0, max_value=10, value=0)

with col2:
    sex = st.selectbox("⚧️ Sex", ["male", "female"])
    smoker = st.selectbox("🚬 Smoker", ["yes", "no"])
    region = st.selectbox("🌎 Region", ["southeast", "southwest", "northeast", "northwest"])

# Convert categorical variables
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region_map = {"southeast": 0, "southwest": 1, "northeast": 2, "northwest": 3}
region = region_map[region]

# Prediction button
if st.button("💰 Predict Insurance Cost"):
    # Create input array for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"💫 Predicted Insurance Cost: ${prediction[0]:,.2f}")

# Add some additional information
with st.expander("ℹ️ About this app"):
    st.write("""
    This app predicts the annual medical insurance cost based on:
    * 👤 Age
    * ⚧️ Sex
    * ⚖️ BMI (Body Mass Index)
    * 👶 Number of Children
    * 🚬 Smoking Status
    * 🌎 Region
    
    The prediction is made using a machine learning model trained on historical insurance data.
    """)

# Add sidebar with developer info
st.sidebar.header("👨‍💻 Developer Info")
st.sidebar.info(
    """
    This app is developed using:
    - 🎈 Streamlit
    - 🐍 Python
    - 🤖 Scikit-learn
    """
)