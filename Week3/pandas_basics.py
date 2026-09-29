## Pandas 
## Setup & Load Data

import pandas as pd
df = pd.read_csv("sales_data.csv")
print(df.head())


## Explore Data

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)


## Clean Data
# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()
print("Duplicates removed.")


## Analyze Sales
# Calculate total revenue
total_revenue = df["Total_Sales"].sum()
print(f"Total Revenue: ₹{total_revenue:,.2f}")


# Find best product
product_sales = df.groupby("Product")["Total_Sales"].sum()
best_product = product_sales.idxmax()
print(f"Best Product: {best_product}")


##Create Report

print("\n--- Sales Report ---")
print(f"Total Revenue: ₹{total_revenue:,.2f}")
print(f"Best Product: {best_product}")
print("\nSales by Product:")
print(product_sales.sort_values(ascending=False))





