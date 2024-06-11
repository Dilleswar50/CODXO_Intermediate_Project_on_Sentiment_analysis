from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

STOPWORDS = set(stopwords.words("english"))

# Load models
predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
scaler = pickle.load(open("Models/scaler.pkl", "rb"))
cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "text" in request.form:
            text_input = request.form["text"]
            predicted_sentiment = predict_text(text_input)
            return jsonify({"prediction": predicted_sentiment})  # Return prediction as JSON
        elif "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = predict_file(data)
            return send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

    return render_template("landing.html")


def predict_text(text_input):
    corpus = preprocess_text(text_input)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    print("Predicted probabilities:", y_predictions)
    sentiment = "Positive" if y_predictions[0][1] > 0.5 else "Negative"
    print("Predicted sentiment:", sentiment)
    return sentiment


def predict_file(data):
    corpus = []
    for sentence in data["Sentence"]:
        corpus.append(" ".join(preprocess_text(sentence)))
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    print("Predicted probabilities:", y_predictions)
    sentiments = ["Positive" if prob[1] > 0.5 else "Negative" for prob in y_predictions]
    print("Predicted sentiments:", sentiments)
    data["Predicted sentiment"] = sentiments
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    # Implement graph generation if needed
    return predictions_csv, None


def preprocess_text(text):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return [" ".join(review)]


if __name__ == "__main__":
    app.run(port=5000, debug=True)
