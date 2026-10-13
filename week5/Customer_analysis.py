# Customer Analysis Script


from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "work" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=Path, default=Path("customer_churn.csv"))
    parser.add_argument("--sales", type=Path, default=Path("sales_data.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "analysis_output")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} CSV is missing required columns: {', '.join(sorted(missing))}")


def clean_data(customers: pd.DataFrame, sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    notes: list[str] = []
    customers = customers.copy()
    sales = sales.copy()

    customer_numeric = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]
    sales_numeric = ["Quantity", "Price", "Total_Sales"]
    for col in customer_numeric:
        customers[col] = pd.to_numeric(customers[col], errors="coerce")
    for col in sales_numeric:
        sales[col] = pd.to_numeric(sales[col], errors="coerce")
    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")

    for label, df in (("customers", customers), ("sales", sales)):
        before = len(df)
        df.drop_duplicates(inplace=True)
        if len(df) != before:
            notes.append(f"Removed {before - len(df)} duplicate {label} rows.")
        missing_before = int(df.isna().sum().sum())
        # Numeric missing values use the column median; categorical values use
        # the most frequent value. Rows with no usable date are excluded.
        for col in df.columns:
            if not df[col].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            else:
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])
        if label == "sales":
            invalid_dates = int(df["Date"].isna().sum())
            if invalid_dates:
                df.dropna(subset=["Date"], inplace=True)
                notes.append(f"Dropped {invalid_dates} sales rows with invalid dates.")
        remaining = int(df.isna().sum().sum())
        if missing_before:
            notes.append(f"Filled missing values in {label} data where possible; {remaining} missing values remain.")

    # Sales file contains both unit price and a provided line total. Reconcile
    # the total to quantity * price so downstream summaries are consistent.
    calculated_sales = sales["Quantity"] * sales["Price"]
    mismatch = ~((sales["Total_Sales"] - calculated_sales).abs() < 0.01)
    if mismatch.any():
        notes.append(f"Recalculated Total_Sales for {int(mismatch.sum())} rows where it differed from Quantity × Price.")
        sales.loc[mismatch, "Total_Sales"] = calculated_sales[mismatch]
    sales["Month"] = sales["Date"].dt.to_period("M").astype(str)
    customers["EstimatedLifetimeValue"] = customers["MonthlyCharges"] * customers["Tenure"]
    return customers, sales, notes


def write_table(df: pd.DataFrame, output: Path, name: str) -> None:
    df.to_csv(output / f"{name}.csv", index=False)


def save_charts(customers: pd.DataFrame, sales: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    monthly = sales.groupby("Month", as_index=False)["Total_Sales"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly["Month"], monthly["Total_Sales"], marker="o", color="#2563eb", linewidth=2)
    ax.set(title="Monthly Sales Revenue", xlabel="Month", ylabel="Revenue")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output / "monthly_sales_trend.png", dpi=160)
    plt.close(fig)

    product = sales.groupby("Product")["Total_Sales"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    product.plot(kind="barh", ax=ax, color="#10b981")
    ax.set(title="Revenue by Product", xlabel="Revenue", ylabel="Product")
    fig.tight_layout()
    fig.savefig(output / "product_revenue.png", dpi=160)
    plt.close(fig)

    region = sales.groupby("Region")["Total_Sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    region.plot(kind="bar", ax=ax, color="#f59e0b")
    ax.set(title="Revenue by Region", xlabel="Region", ylabel="Revenue")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output / "regional_revenue.png", dpi=160)
    plt.close(fig)

    churn = customers.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    churn.plot(kind="bar", ax=ax, color="#ef4444")
    ax.set(title="Churn Rate by Contract", xlabel="Contract", ylabel="Churn rate (%)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output / "churn_by_contract.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.customers.exists():
        raise FileNotFoundError(f"Customer CSV not found: {args.customers}")
    if not args.sales.exists():
        raise FileNotFoundError(f"Sales CSV not found: {args.sales}")
    args.output.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(args.customers)
    sales = pd.read_csv(args.sales)
    require_columns(customers, {"CustomerID", "Tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "PaperlessBilling", "SeniorCitizen", "Churn"}, "Customer")
    require_columns(sales, {"Date", "Product", "Quantity", "Price", "Customer_ID", "Region", "Total_Sales"}, "Sales")
    raw_customer_shape, raw_sales_shape = customers.shape, sales.shape
    missing_customers = int(customers.isna().sum().sum())
    missing_sales = int(sales.isna().sum().sum())
    customers, sales, notes = clean_data(customers, sales)

    customer_summary = customers.describe(include="all").transpose().reset_index(names="Column")
    sales_summary = sales.drop(columns=["Month"]).describe(include="all").transpose().reset_index(names="Column")
    top_customers = customers.sort_values("EstimatedLifetimeValue", ascending=False)[["CustomerID", "Tenure", "MonthlyCharges", "EstimatedLifetimeValue", "Churn"]]
    contract_retention = customers.groupby("Contract").agg(
        Customers=("CustomerID", "nunique"), Churn_Rate=("Churn", "mean"),
        Avg_Tenure_Months=("Tenure", "mean"), Avg_Lifetime_Value=("EstimatedLifetimeValue", "mean")
    ).reset_index()
    contract_retention["Retention_Rate"] = 1 - contract_retention["Churn_Rate"]
    contract_retention[["Churn_Rate", "Retention_Rate"]] *= 100

    monthly_sales = sales.groupby("Month", as_index=False).agg(Revenue=("Total_Sales", "sum"), Units=("Quantity", "sum"), Transactions=("Date", "count"))
    product_sales = sales.groupby("Product", as_index=False).agg(Revenue=("Total_Sales", "sum"), Units=("Quantity", "sum"), Transactions=("Date", "count")).sort_values("Revenue", ascending=False)
    regional_sales = sales.groupby("Region", as_index=False).agg(Revenue=("Total_Sales", "sum"), Units=("Quantity", "sum"), Transactions=("Date", "count")).sort_values("Revenue", ascending=False)
    customer_sales = sales.groupby("Customer_ID", as_index=False).agg(Revenue=("Total_Sales", "sum"), Units=("Quantity", "sum"), Transactions=("Date", "count")).sort_values("Revenue", ascending=False)
    weekday_sales = sales.assign(Weekday=sales["Date"].dt.day_name()).groupby("Weekday", as_index=False)["Total_Sales"].sum().sort_values("Total_Sales", ascending=False)

    
    if sales["Date"].duplicated().any():
        baskets = sales.groupby("Date")["Product"].apply(lambda values: sorted(set(values)))
        pair_counts: dict[tuple[str, str], int] = {}
        for basket in baskets:
            for pair in combinations(basket, 2):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        pairs = pd.DataFrame([{"Product_1": p[0], "Product_2": p[1], "Cooccurrences": n} for p, n in pair_counts.items()]).sort_values("Cooccurrences", ascending=False)
        pair_note = "Product pairs were counted within dates containing multiple products (date used as transaction proxy)."
    else:
        pairs = pd.DataFrame(columns=["Product_1", "Product_2", "Cooccurrences"])
        pair_note = "Cross-selling pairs cannot be identified: every sales date has exactly one product row, and no basket/order ID is provided."

    for name, df in {
        "customer_data_quality": pd.DataFrame({"Dataset": ["customers", "sales"], "Rows": [raw_customer_shape[0], raw_sales_shape[0]], "Columns": [raw_customer_shape[1], raw_sales_shape[1]], "Missing_cells_before_cleaning": [missing_customers, missing_sales], "Duplicate_rows_before_cleaning": [int(pd.read_csv(args.customers).duplicated().sum()), int(pd.read_csv(args.sales).duplicated().sum())]}),
        "customer_descriptive_statistics": customer_summary,
        "sales_descriptive_statistics": sales_summary,
        "top_customers_by_estimated_lifetime_value": top_customers,
        "retention_by_contract": contract_retention,
        "monthly_sales": monthly_sales,
        "sales_by_product": product_sales,
        "sales_by_region": regional_sales,
        "sales_by_customer": customer_sales,
        "sales_by_weekday": weekday_sales,
        "product_pairs": pairs,
    }.items():
        write_table(df, args.output, name)

    save_charts(customers, sales, args.output)
    total_revenue = sales["Total_Sales"].sum()
    avg_order = sales["Total_Sales"].mean()
    churn_rate = customers["Churn"].mean()
    top_customer = top_customers.iloc[0]
    top_product = product_sales.iloc[0]
    top_region = regional_sales.iloc[0]
    worst_contract = contract_retention.sort_values("Churn_Rate", ascending=False).iloc[0]
    start_date, end_date = sales["Date"].min().date(), sales["Date"].max().date()

    note_lines = "\n".join(f"- {note}" for note in notes) or "- No missing values or duplicate rows required correction."
    report = f"""# Customer and Sales Analysis Report

## Executive summary

- **Sales revenue:** ${total_revenue:,.2f} across {len(sales):,} transactions; average transaction value **${avg_order:,.2f}**.
- **Customers:** {customers['CustomerID'].nunique():,} unique customer records in the churn file; observed churn rate **{churn_rate:.1%}** (retention **{1-churn_rate:.1%}**).
- **Highest estimated customer lifetime value:** {top_customer['CustomerID']} at **${top_customer['EstimatedLifetimeValue']:,.2f}** (monthly charge × tenure estimate).
- **Best-selling product by revenue:** {top_product['Product']} at **${top_product['Revenue']:,.2f}**.
- **Highest-revenue region:** {top_region['Region']} at **${top_region['Revenue']:,.2f}**.
- **Highest-churn contract segment:** {worst_contract['Contract']} at **{worst_contract['Churn_Rate']:.1f}%**.

- Customer file: {raw_customer_shape[0]:,} rows and {raw_customer_shape[1]} columns. Sales file: {raw_sales_shape[0]:,} rows and {raw_sales_shape[1]} columns.
- Sales dates cover **{start_date} through {end_date}**. This is a short sample, so treat month-to-month movement as descriptive rather than a reliable seasonal pattern.
- Cleaning actions:
{note_lines}

- **Customer value:** `EstimatedLifetimeValue` is monthly charges multiplied by observed tenure. It is a simple proxy, not a forecast of future revenue. The provided `TotalCharges` is retained in the customer statistics.
- **Retention:** retention is calculated as 1 minus the observed churn share, with churn comparisons by contract in `retention_by_contract.csv`.
- **Cross-selling:** {pair_note}
- **Customer linkage:** customer IDs in the sales file (`CUST...`) and churn file (`C...`) use different formats; they cannot be joined reliably from the supplied files.

"""
    (args.output / "executive_summary.md").write_text(report, encoding="utf-8")
    print(f"Analysis complete. Outputs saved to: {args.output.resolve()}")
    print(f"Revenue: ${total_revenue:,.2f} | Churn: {churn_rate:.1%} | Top product: {top_product['Product']} | Top region: {top_region['Region']}")


if __name__ == "__main__":
    main()
