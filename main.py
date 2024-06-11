import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.title("Text Sentiment Predictor")

prediction_endpoint = "http://127.0.0.1:5000/"

uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction - Upload the file and click on Predict",
    type="csv",
)

user_input = st.text_input("Enter text and click on Predict", "")

if st.button("Predict"):
    if uploaded_file is not None:
        file = {"file": uploaded_file}
        response = requests.post(prediction_endpoint, files=file)
        st.write(response.content)  # Debug statement to inspect the response content
        response_bytes = BytesIO(response.content)
        response_df = pd.read_csv(response_bytes)

        st.download_button(
            label="Download Predictions",
            data=response_bytes,
            file_name="Predictions.csv",
            key="result_download_button",
        )
        # Add visualization of graph if needed
    else:
        response = requests.post(prediction_endpoint, data={"text": user_input})
        st.write(response.content)  # Debug statement to inspect the response content
        response = response.json()
        st.write(f"Predicted sentiment: {response['prediction']}")
