from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load trained model (trained using joblib)
model = joblib.load("../models/demand_forecast_model.pkl")


@app.route("/")
def home():
    return "E-commerce Demand Forecasting API is running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    CATEGORY_ENCODING = {
        "Electronics": 0,
        "Clothing": 1,
        "Groceries": 2,
        "Home": 3,
        "Stationery": 4,
        "Sports": 5
    }

    category_encoded = CATEGORY_ENCODING[data["category"]]

    input_data = pd.DataFrame([{
        "category": category_encoded,
        "price": data["price"],
        "day": data["day"],
        "month": data["month"],
        "day_of_week": data["day_of_week"]
    }])

    prediction = model.predict(input_data)[0]
    demand = round(float(prediction))

    # category specific safety stock
    SAFETY_STOCK = {
        "Electronics": 10,
        "Clothing": 15,
        "Groceries": 30,
        "Home": 12,
        "Stationery": 8,
        "Sports": 10
    }

    recommended_stock = demand + SAFETY_STOCK[data["category"]]

    # demand level classification
    if demand >= 60:
        level = "High"
    elif demand >= 30:
        level = "Medium"
    else:
        level = "Low"

    return jsonify({
        "predicted_quantity_sold": demand,
        "recommended_stock": recommended_stock,
        "demand_level": level
    })

if __name__ == "__main__":
    app.run(debug=True)