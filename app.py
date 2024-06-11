from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Select the predictor to be loaded from Models folder
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    
    try:
        if request.method == "POST":
            if request.is_json:
                # JSON data
                text_input = request.json.get("text", "")
            else:
                # Form data
                text_input = request.form.get("text", "")

            if text_input:
                predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
                return jsonify({"prediction": predicted_sentiment})
            else:
                return jsonify({"error": "No text input provided."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    print("Corpus:", corpus)  # Add print statement to check the corpus
    X_prediction = cv.transform(corpus).toarray()
    print("X_prediction:", X_prediction)  # Add print statement to check X_prediction
    X_prediction_scl = scaler.transform(X_prediction)
    print("X_prediction_scl:", X_prediction_scl)  # Add print statement to check X_prediction_scl
    y_predictions = predictor.predict_proba(X_prediction_scl)
    print("y_predictions:", y_predictions)  # Add print statement to check y_predictions
    y_predictions = y_predictions.argmax(axis=1)[0]
    print("Predicted class index:", y_predictions)  # Add print statement to check predicted class index

    return "Positive" if y_predictions == 1 else "Negative"



def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    return predictions_csv



def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
