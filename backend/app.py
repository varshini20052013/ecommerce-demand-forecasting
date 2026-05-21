from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("../models/demand_forecast_model.pkl")

# ✅ SAME mapping used during dataset creation
CATEGORY_MAP = {
    "Electronics": 0,
    "Clothing": 1,
    "Groceries": 2,
    "Home": 3,
    "Stationery": 4,
    "Sports": 5
}

PRODUCT_MAP = {
    "P001":0,"P002":1,"P003":2,
    "P004":3,"P005":4,"P006":5,
    "P007":6,"P008":7,"P009":8,
    "P010":9,"P011":10,"P012":11,
    "P013":12,"P014":13,"P015":14,
    "P016":15,"P017":16,"P018":17,
    "P019":18,"P020":19,"P021":20,
    "P022":21,"P023":22
}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("INPUT:", data)  # 🔍 debug

        # ✅ Convert properly (STRING → NUMBER)
        category = CATEGORY_MAP.get(data.get("category"), 0)
        product_id = PRODUCT_MAP.get(data.get("product_id"), 0)

        price = float(data.get("price", 0))
        day = int(data.get("day", 1))
        month = int(data.get("month", 1))
        day_of_week = int(data.get("day_of_week", 0))
        festival = int(data.get("festival", 0))

        input_data = pd.DataFrame([{
            "category": category,
            "product_id": product_id,
            "price": price,
            "day": day,
            "month": month,
            "day_of_week": day_of_week,
            "festival": festival
        }])

        prediction = model.predict(input_data)[0]
        demand = int(round(prediction))

        # Demand level
        if demand >= 60:
            level = "High"
        elif demand >= 30:
            level = "Medium"
        else:
            level = "Low"

        stock = demand + int(demand * 0.2)

        return jsonify({
            "predicted_quantity_sold": demand,
            "demand_level": level,
            "recommended_stock": stock,
            "festival_name": "Detected" if festival == 1 else "None"
        })

    except Exception as e:
        print("ERROR:", e)

        # ✅ Always safe response
        return jsonify({
            "predicted_quantity_sold": 0,
            "demand_level": "Error",
            "recommended_stock": 0,
            "festival_name": "Error"
        })


if __name__ == "__main__":
    app.run(debug=True)