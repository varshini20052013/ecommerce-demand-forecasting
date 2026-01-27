from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load trained model
with open("../models/demand_forecast_model.pkl", "rb") as file:
    model = pickle.load(file)

# SAME encoding used during training
PRODUCT_ENCODING = {
    "P001": 0,
    "P002": 1,
    "P003": 2
}

CATEGORY_ENCODING = {
    "Electronics": 0,
    "Clothing": 1,
    "Groceries": 2
}

@app.route("/")
def home():
    return "E-commerce Demand Forecasting API (Product-aware) is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Encode product & category
    product_encoded = PRODUCT_ENCODING[data["product_id"]]
    category_encoded = CATEGORY_ENCODING[data["category"]]

    # Prepare input
    input_data = pd.DataFrame([{
        "price": data["price"],
        "day": data["day"],
        "month": data["month"],
        "day_of_week": data["day_of_week"],
        "product_id_encoded": product_encoded,
        "category_encoded": category_encoded
    }])

    prediction = model.predict(input_data)[0]

    return jsonify({
        "predicted_quantity_sold": round(float(prediction))
    })

if __name__ == "__main__":
    app.run(debug=True)
