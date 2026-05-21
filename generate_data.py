import pandas as pd
import random

# ✅ FIXED ENCODING (MUST match backend)
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

products = {
    "Electronics": ["P001","P002","P003"],
    "Clothing": ["P004","P005","P006","P019","P020","P021","P022","P023"],
    "Groceries": ["P007","P008","P009"],
    "Home": ["P010","P011","P012"],
    "Stationery": ["P013","P014","P015"],
    "Sports": ["P016","P017","P018"]
}

rows = []

for _ in range(3000):

    category = random.choice(list(products.keys()))
    product_id = random.choice(products[category])

    price = random.randint(100, 2000)
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    day_of_week = random.randint(0, 6)

    # 🎉 Festival detection
    festival = 0

    if month in [10, 11]: festival = 1   # Diwali
    elif month == 12: festival = 1       # Christmas
    elif month in [4, 5]: festival = 1   # Eid
    elif month == 1: festival = 1        # New Year
    elif month == 3: festival = 1        # Holi
    elif month in [6, 7]: festival = 1   # School reopening

    # 🔥 STRONG BASE DEMAND
    base_demand = {
        "Electronics": 80,
        "Clothing": 40,
        "Groceries": 100,
        "Home": 30,
        "Stationery": 20,
        "Sports": 50
    }

    demand = base_demand[category] + random.randint(-5, 5)

    # 🚀 STRONG FESTIVAL LOGIC
    if festival == 1:

        if month in [10, 11]:  # Diwali
            if category == "Clothing":
                demand += random.randint(50, 80)
            elif category == "Electronics":
                demand += random.randint(40, 70)
            elif category == "Home":
                demand += random.randint(30, 50)

        elif month == 12:  # Christmas
            if category == "Electronics":
                demand += random.randint(50, 80)
            elif category == "Clothing":
                demand += random.randint(30, 50)

        elif month == 1:  # New Year
            if category == "Clothing":
                demand += random.randint(40, 60)
            elif category == "Groceries":
                demand += random.randint(50, 80)

        elif month in [4, 5]:  # Eid
            if category == "Clothing":
                demand += random.randint(50, 80)
            elif category == "Groceries":
                demand += random.randint(40, 70)

        elif month == 3:  # Holi
            if category == "Clothing":
                demand += random.randint(30, 50)

        elif month in [6, 7]:  # School reopening
            if category == "Stationery":
                demand += random.randint(60, 100)

    # ✅ STORE ENCODED VALUES (IMPORTANT FIX)
    rows.append([
        PRODUCT_MAP[product_id],
        CATEGORY_MAP[category],
        price,
        day,
        month,
        day_of_week,
        festival,
        demand
    ])

# ✅ Create DataFrame
df = pd.DataFrame(rows, columns=[
    "product_id",
    "category",
    "price",
    "day",
    "month",
    "day_of_week",
    "festival",
    "quantity_sold"
])

df.to_csv("data/sales_data.csv", index=False)

print("✅ Dataset generated with", len(df), "rows (FINAL FIXED VERSION)")