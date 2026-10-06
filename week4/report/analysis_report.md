# E-commerce Sales Analysis Report

## Executive Summary
This analysis examines 100 e-commerce transactions from 2024-01-01 to 2024-04-09. The dataset records product, quantity, price, customer, region, and total sales information.

The business generated **12,365,048** in recorded sales from **478 units**.

## Data Preparation
- Converted `Date` into datetime format.
- Checked missing values.
- Checked duplicate records.
- Verified that `Total_Sales` is consistent with `Quantity × Price`.
- No missing values were found in the supplied dataset.
- All 100 sales records passed the sales-value validation.

## KPI Analysis
- Total Sales: **12,365,048**
- Total Orders: **100**
- Total Quantity Sold: **478**
- Average Order Value: **123,650.48**

## Product Analysis
Product
Laptop        3889210
Tablet        2884340
Phone         2859394
Headphones    1384033
Monitor       1348071

The highest-revenue product was **Laptop**, generating **3,889,210**.

## Regional Analysis
Region
North    3983635
South    3737852
East     2519639
West     2123922

The strongest region was **North**, generating **3,983,635**.

## Monthly Analysis
Date
2024-01    4120524
2024-02    2656050
2024-03    4485006
2024-04    1103468

## Business Insights
1. Focus inventory and promotional planning on high-revenue products.
2. The top-performing region can be used as a benchmark for other regions.
3. Lower-performing products may need pricing, promotion, or demand analysis.
4. Monthly monitoring can help identify changes in demand.
5. Combining sales, quantity, and regional information provides a better view than looking at revenue alone.

## Conclusion
The analysis demonstrates how Python and Pandas can transform raw transaction data into useful business information. The generated visualizations make product, regional, and time-based patterns easier to understand and support data-driven decision making.
