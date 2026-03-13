import pandas as pd
import random

products = {
    "Electronics": ["P001","P002","P003"],
    "Clothing": ["P004","P005","P006"],
    "Groceries": ["P007","P008","P009"],
    "Home": ["P010","P011","P012"],
    "Stationery": ["P013","P014","P015"],
    "Sports": ["P016","P017","P018"]
}

rows = []

for _ in range(500):

    category = random.choice(list(products.keys()))
    product_id = random.choice(products[category])

    price = random.randint(100,2000)

    day = random.randint(1,28)
    month = random.randint(1,12)
    day_of_week = random.randint(0,6)

    base_demand = {
        "Electronics":50,
        "Clothing":30,
        "Groceries":60,
        "Home":25,
        "Stationery":20,
        "Sports":35
    }

    demand = base_demand[category] + random.randint(-10,10)

    rows.append([
        product_id,
        category,
        price,
        day,
        month,
        day_of_week,
        demand
    ])

df = pd.DataFrame(rows, columns=[
    "product_id",
    "category",
    "price",
    "day",
    "month",
    "day_of_week",
    "quantity_sold"
])

df.to_csv("data/sales_data.csv", index=False)

print("Dataset generated with", len(df), "rows")