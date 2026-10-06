# E-commerce Sales Analysis

## 1. Project Overview
This project analyzes e-commerce sales transactions to identify sales trends, product performance, regional performance, and key business insights.

The dataset contains **100 transactions** covering **2024-01-01 to 2024-04-09**.

## 2. Objectives
- Analyze overall sales performance
- Identify the best-performing products
- Compare regional sales
- Analyze sales trends over time
- Calculate important business KPIs
- Generate visualizations and actionable insights

## 3. Dataset
Columns:
- `Date` - Transaction date
- `Product` - Product purchased
- `Quantity` - Units sold
- `Price` - Unit price
- `Customer_ID` - Customer identifier
- `Region` - Sales region
- `Total_Sales` - Transaction sales value

## 4. Tools & Technologies
- Python
- Pandas
- Matplotlib
- Google Collab Notebook
- Git & GitHub

## 5. Key Results
| KPI | Value |
|---|---:|
| Total Sales | 12,365,048 |
| Total Orders | 100 |
| Quantity Sold | 478 |
| Average Order Value | 123,650.48 |
| Top Product | Laptop |
| Top Region | North |

## 6. Key Insights
1. **Laptop** generated the highest revenue at **3,889,210**.
2. **North** was the strongest region with revenue of **3,983,635**.
3. Laptop sales contributed the largest share of product revenue in the dataset.
4. Sales should be monitored monthly to identify growth and weaker periods.
5. Product-level and regional performance can support inventory and marketing decisions.

## 7. Project Structure
```text
ecommerce-sales-analysis/
├── README.md
├── main.py
├── analysis.ipynb
├── requirements.txt
├── data/
│   └── sales.csv
├── visualizations/
│   ├── monthly_sales.png
│   ├── product_sales.png
│   ├── regional_sales.png
│   └── product_quantity.png
└── report/
    └── analysis_report.md
```

## 8. How to Run
```bash
pip install -r requirements.txt
python main.py
```

For the notebook:
```bash
jupyter notebook analysis.ipynb
```
