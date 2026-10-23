from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------- Page setup ----------
st.set_page_config(
    page_title="Sales Insights",
    page_icon="📊",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #101827 0%, #111827 55%, #172033 100%);
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.10);
            padding: 18px;
            border-radius: 14px;
        }
        [data-testid="stMetricLabel"] {
            color: #aab8ce;
        }
        h1, h2, h3 {
            color: #f4f7fb;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load and validate data ----------
data_path = Path(__file__).with_name("sales_data.csv")

if not data_path.exists():
    st.error("Put sales_data.csv in the same folder as dashboard.py.")
    st.stop()

sales = pd.read_csv(data_path)

required_columns = {
    "Date", "Product", "Quantity", "Price", "Customer_ID", "Region", "Total_Sales"
}
missing_columns = required_columns - set(sales.columns)

if missing_columns:
    st.error(f"CSV is missing these columns: {', '.join(sorted(missing_columns))}")
    st.stop()

sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
for column in ["Quantity", "Price", "Total_Sales"]:
    sales[column] = pd.to_numeric(sales[column], errors="coerce")

sales = sales.dropna(
    subset=["Date", "Product", "Region", "Quantity", "Price", "Total_Sales"]
)

if sales.empty:
    st.error("There are no valid sales rows to display.")
    st.stop()


# ---------- Sidebar filters ----------
st.sidebar.title("🎛️ Dashboard filters")

min_date = sales["Date"].min().date()
max_date = sales["Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

products = sorted(sales["Product"].unique())
regions = sorted(sales["Region"].unique())

selected_products = st.sidebar.multiselect(
    "Products",
    products,
    default=products,
)

selected_regions = st.sidebar.multiselect(
    "Regions",
    regions,
    default=regions,
)

filtered = sales[
    sales["Product"].isin(selected_products)
    & sales["Region"].isin(selected_regions)
].copy()

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered = filtered[
        filtered["Date"].dt.date.between(start_date, end_date)
    ]

if filtered.empty:
    st.warning("No sales match those filters. Try selecting a wider date range.")
    st.stop()


# ---------- Header and key metrics ----------
st.title("📊 Sales Insights")
st.caption("Explore sales trends, products, and regions. Use the filters to interact with the data.")

total_sales = filtered["Total_Sales"].sum()
units_sold = filtered["Quantity"].sum()
average_sale = filtered["Total_Sales"].mean()
number_of_orders = len(filtered)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total sales", f"{total_sales:,.0f}")
k2.metric("Units sold", f"{units_sold:,.0f}")
k3.metric("Average sale", f"{average_sale:,.0f}")
k4.metric("Sales records", f"{number_of_orders:,}")


# ---------- Prepare chart data ----------
daily_sales = (
    filtered.groupby("Date", as_index=False)["Total_Sales"]
    .sum()
    .sort_values("Date")
)

product_sales = (
    filtered.groupby("Product", as_index=False)["Total_Sales"]
    .sum()
    .sort_values("Total_Sales", ascending=True)
)

region_sales = (
    filtered.groupby("Region", as_index=False)["Total_Sales"]
    .sum()
    .sort_values("Total_Sales", ascending=False)
)

product_region = filtered.pivot_table(
    index="Product",
    columns="Region",
    values="Total_Sales",
    aggfunc="sum",
    fill_value=0,
)


# ---------- Interactive charts ----------
st.subheader("Sales over time")

daily_fig = px.area(
    daily_sales,
    x="Date",
    y="Total_Sales",
    markers=True,
    template="plotly_dark",
    color_discrete_sequence=["#55c2ff"],
    labels={"Date": "Date", "Total_Sales": "Sales"},
)
daily_fig.update_traces(
    line=dict(width=3),
    fillcolor="rgba(85, 194, 255, 0.16)",
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sales: %{y:,.0f}<extra></extra>",
)
daily_fig.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(daily_fig, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("Sales by product")
    product_fig = px.bar(
        product_sales,
        x="Total_Sales",
        y="Product",
        orientation="h",
        text="Total_Sales",
        template="plotly_dark",
        color="Total_Sales",
        color_continuous_scale=["#3767d5", "#35c6b0", "#ffd166"],
        labels={"Total_Sales": "Sales", "Product": ""},
    )
    product_fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Sales: %{x:,.0f}<extra></extra>",
    )
    product_fig.update_layout(
        height=350,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=30, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(product_fig, use_container_width=True)

with right:
    st.subheader("Sales by region")
    region_fig = px.pie(
        region_sales,
        names="Region",
        values="Total_Sales",
        hole=0.58,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    region_fig.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Sales: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
    )
    region_fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(region_fig, use_container_width=True)

st.subheader("Product and region comparison")
heatmap = go.Figure(
    data=go.Heatmap(
        z=product_region.values,
        x=product_region.columns.astype(str),
        y=product_region.index.astype(str),
        colorscale="Tealgrn",
        colorbar=dict(title="Sales"),
        hovertemplate="Product: %{y}<br>Region: %{x}<br>Sales: %{z:,.0f}<extra></extra>",
    )
)
heatmap.update_layout(
    template="plotly_dark",
    height=380,
    xaxis_title="Region",
    yaxis_title="Product",
    margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(heatmap, use_container_width=True)


# ---------- Data table and download ----------
with st.expander("View filtered sales records"):
    st.dataframe(
        filtered.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "Download filtered data as CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales.csv",
    mime="text/csv",
)