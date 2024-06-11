import streamlit as st
import pandas as pd
import requests

# Define the prediction endpoint
prediction_endpoint = "http://127.0.0.1:5000/predict"

# Streamlit title and file uploader
st.title("Text Sentiment Predictor")
uploaded_file = st.file_uploader("Choose a CSV file for bulk prediction - Upload the file and click on Predict", type="csv")

# Text input for sentiment prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Function to handle API response and display prediction
def handle_response(response):
    if 'prediction' in response:
        st.write(f"Predicted sentiment: {response['prediction']}")
    else:
        st.error("Error: Prediction not available.")

# Prediction on single sentence or bulk prediction
if st.button("Predict"):
    if uploaded_file is not None:
        # Perform bulk prediction from uploaded CSV file
        file = {"file": uploaded_file}
        response = requests.post(prediction_endpoint, files=file)
        if response.status_code == 200:
            response_data = response.json()
            handle_response(response_data)
        else:
            st.error("Error: Failed to get predictions from the API.")
    else:
        # Perform prediction on single sentence input
        response = requests.post(prediction_endpoint, data={"text": user_input})
        if response.status_code == 200:
            response_data = response.json()
            handle_response(response_data)
        else:
            st.error("Error: Failed to get predictions from the API.")
