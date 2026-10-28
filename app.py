
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

FORECASTS_DIR = PROJECT_DIR / "forecasts"
REPORTS_DIR = PROJECT_DIR / "reports"
CONFIG_DIR = PROJECT_DIR / "config"

st.set_page_config(
    page_title="Demand Intelligence | Inventory AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# DARK THEME
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0b1020;
        color: #f1f5f9;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #273449;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #151f33, #101827);
        border: 1px solid #273449;
        padding: 20px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9caec5;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .hero {
        background: linear-gradient(120deg, #14213d, #172554, #164e63);
        border: 1px solid #334155;
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: 30px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 15px;
    }

    .section-title {
        color: #e2e8f0;
        font-size: 22px;
        font-weight: 700;
        margin-top: 16px;
        margin-bottom: 12px;
    }

    .notice {
        background-color: #332719;
        border: 1px solid #8b6b2d;
        color: #fde68a;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 16px;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 13px;
    }

    div.stButton > button {
        border-radius: 9px;
        border: 1px solid #334155;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #273449;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data(ttl=60)
def load_json(path_string):
    path = Path(path_string)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

forecast_path = FORECASTS_DIR / "demand_forecast_7_days.csv"
inventory_path = FORECASTS_DIR / "inventory_optimization_recommendations.csv"
store_path = FORECASTS_DIR / "store_inventory_summary.csv"
risk_path = REPORTS_DIR / "inventory_risk_summary.csv"
summary_path = REPORTS_DIR / "part4_inventory_summary.json"

forecast = load_csv(str(forecast_path))
inventory = load_csv(str(inventory_path))
stores = load_csv(str(store_path))
risk_summary = load_csv(str(risk_path))
summary = load_json(str(summary_path))

if forecast.empty or inventory.empty:
    st.error(
        "Dashboard data is missing. Please run Part 4 in Google Colab first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# NORMALIZE DATA
# ---------------------------------------------------------------------------

if "forecast_date" in forecast.columns:
    forecast["forecast_date"] = pd.to_datetime(
        forecast["forecast_date"], errors="coerce"
    )

numeric_columns_forecast = [
    "store_nbr", "predicted_sales"
]
for col in numeric_columns_forecast:
    if col in forecast.columns:
        forecast[col] = pd.to_numeric(
            forecast[col], errors="coerce"
        )

numeric_columns_inventory = [
    "store_nbr",
    "historical_avg_daily_sales_28d",
    "simulated_current_stock",
    "estimated_reorder_point",
    "estimated_safety_stock",
    "estimated_target_stock",
    "suggested_order_quantity",
    "forecast_demand_next_7_days",
    "forecast_daily_average"
]

for col in numeric_columns_inventory:
    if col in inventory.columns:
        inventory[col] = pd.to_numeric(
            inventory[col], errors="coerce"
        ).fillna(0)

for col in ["family", "inventory_risk"]:
    if col in inventory.columns:
        inventory[col] = inventory[col].astype(str)

if "reorder_recommended" in inventory.columns:
    inventory["reorder_recommended"] = (
        inventory["reorder_recommended"]
        .astype(str).str.lower().isin(["true", "1", "yes"])
    )

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 📦 Demand Intelligence")
st.sidebar.caption("AI Forecasting & Inventory Control")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Demand Forecasting",
        "Inventory Optimization",
        "Store Analysis",
        "Product Analysis",
        "Data Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Filters")

all_stores = sorted(
    inventory["store_nbr"].dropna().unique().tolist()
)
all_families = sorted(
    inventory["family"].dropna().unique().tolist()
)
all_risks = [
    r for r in ["Critical", "High", "Medium", "Low"]
    if r in inventory["inventory_risk"].unique()
]

selected_stores = st.sidebar.multiselect(
    "Select stores",
    options=all_stores,
    default=all_stores
)

selected_families = st.sidebar.multiselect(
    "Select product families",
    options=all_families,
    default=all_families
)

selected_risks = st.sidebar.multiselect(
    "Inventory risk",
    options=all_risks,
    default=all_risks
)

if st.sidebar.button("🔄 Refresh dashboard"):
    st.cache_data.clear()
    st.rerun()

# Apply filters
filtered_inventory = inventory[
    inventory["store_nbr"].isin(selected_stores)
    & inventory["family"].isin(selected_families)
    & inventory["inventory_risk"].isin(selected_risks)
].copy()

filtered_forecast = forecast[
    forecast["store_nbr"].isin(selected_stores)
    & forecast["family"].isin(selected_families)
].copy()

# ---------------------------------------------------------------------------
# COMMON HELPERS
# ---------------------------------------------------------------------------

def money_or_number(value):
    return f"{value:,.0f}"

def show_hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_inventory_warning():
    st.markdown(
        """
        <div class="notice">
            <b>Research prototype — simulated inventory:</b>
            The dataset contains historical sales, not actual stock-on-hand.
            Inventory, lead time, reorder point and order quantity are
            estimates based on assumptions. Future promotions are assumed
            to be zero.
        </div>
        """,
        unsafe_allow_html=True
    )

def dark_chart(fig, height=400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe"),
        margin=dict(l=20, r=20, t=55, b=20),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#263449")
    st.plotly_chart(fig, use_container_width=True)

def display_download(df, filename, label):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )

# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------

if page == "Overview":

    show_hero(
        "Demand Intelligence Dashboard",
        "Machine-learning forecasts, demand trends, and inventory decision support."
    )

    show_inventory_warning()

    total_demand = filtered_forecast["predicted_sales"].sum()

    total_stores = filtered_inventory["store_nbr"].nunique()
    total_families = filtered_inventory["family"].nunique()

    order_quantity = filtered_inventory[
        "suggested_order_quantity"
    ].sum()

    critical_count = (
        filtered_inventory["inventory_risk"] == "Critical"
    ).sum()

    high_count = (
        filtered_inventory["inventory_risk"] == "High"
    ).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast demand · 7 days", money_or_number(total_demand))
    col2.metric("Stores selected", f"{total_stores:,}")
    col3.metric("Product families selected", f"{total_families:,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Suggested order quantity", money_or_number(order_quantity))
    col5.metric("Critical items", f"{critical_count:,}")
    col6.metric("High-risk items", f"{high_count:,}")

    st.markdown(
        '<div class="section-title">Forecast demand trend</div>',
        unsafe_allow_html=True
    )

    if not filtered_forecast.empty:
        daily = (
            filtered_forecast.groupby("forecast_date", as_index=False)
            ["predicted_sales"].sum()
        )

        fig = px.area(
            daily,
            x="forecast_date",
            y="predicted_sales",
            markers=True,
            title="Total predicted sales by date"
        )
        dark_chart(fig, 390)

    st.markdown(
        '<div class="section-title">Inventory risk distribution</div>',
        unsafe_allow_html=True
    )

    if not filtered_inventory.empty:
        risk_counts = (
            filtered_inventory["inventory_risk"]
            .value_counts()
            .reindex(["Critical", "High", "Medium", "Low"], fill_value=0)
            .rename_axis("Risk")
            .reset_index(name="Items")
        )

        fig = px.bar(
            risk_counts,
            x="Risk",
            y="Items",
            color="Risk",
            title="Store-product combinations by simulated risk"
        )
        dark_chart(fig, 330)

    st.markdown("### Highest suggested order quantities")

    top = filtered_inventory.sort_values(
        "suggested_order_quantity", ascending=False
    ).head(10)

    if not top.empty:
        display_cols = [
            "store_nbr", "family", "forecast_demand_next_7_days",
            "simulated_current_stock", "suggested_order_quantity",
            "inventory_risk"
        ]
        st.dataframe(
            top[display_cols],
            use_container_width=True,
            hide_index=True
        )

# ---------------------------------------------------------------------------
# DEMAND FORECASTING
# ---------------------------------------------------------------------------

elif page == "Demand Forecasting":

    show_hero(
        "Demand Forecasting",
        "Explore the model's seven-day recursive sales forecasts."
    )

    st.info(
        "Predictions are model estimates. Future promotions are assumed to be zero."
    )

    if filtered_forecast.empty:
        st.warning("No forecast rows match the selected filters.")
    else:
        total_predicted = filtered_forecast["predicted_sales"].sum()
        mean_daily = (
            filtered_forecast.groupby("forecast_date")["predicted_sales"]
            .sum().mean()
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total forecast sales", money_or_number(total_predicted))
        c2.metric("Average daily forecast", money_or_number(mean_daily))
        c3.metric("Forecast records", f"{len(filtered_forecast):,}")

        daily = (
            filtered_forecast.groupby("forecast_date", as_index=False)
            ["predicted_sales"].sum()
        )

        fig = px.line(
            daily,
            x="forecast_date",
            y="predicted_sales",
            markers=True,
            title="Forecast demand over the next seven days"
        )
        dark_chart(fig, 430)

        st.markdown("### Product-family demand")

        family_demand = (
            filtered_forecast.groupby("family", as_index=False)
            ["predicted_sales"].sum()
            .sort_values("predicted_sales", ascending=False)
            .head(15)
        )

        fig = px.bar(
            family_demand.sort_values("predicted_sales"),
            x="predicted_sales",
            y="family",
            orientation="h",
            title="Top product families by predicted demand"
        )
        dark_chart(fig, 480)

        st.markdown("### Forecast records")
        st.dataframe(
            filtered_forecast.sort_values(
                ["forecast_date", "store_nbr", "family"]
            ),
            use_container_width=True,
            hide_index=True
        )

        display_download(
            filtered_forecast,
            "filtered_demand_forecast.csv",
            "⬇️ Download filtered forecast CSV"
        )

# ---------------------------------------------------------------------------
# INVENTORY OPTIMIZATION
# ---------------------------------------------------------------------------

elif page == "Inventory Optimization":

    show_hero(
        "Inventory Optimization",
        "Review simulated stock, reorder points, and suggested order quantities."
    )

    show_inventory_warning()

    if filtered_inventory.empty:
        st.warning("No inventory records match the selected filters.")
    else:
        total_order = filtered_inventory[
            "suggested_order_quantity"
        ].sum()

        critical = (
            filtered_inventory["inventory_risk"] == "Critical"
        ).sum()

        high = (
            filtered_inventory["inventory_risk"] == "High"
        ).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Suggested order quantity", money_or_number(total_order))
        c2.metric("Critical combinations", f"{critical:,}")
        c3.metric("High-risk combinations", f"{high:,}")

        st.markdown("### Inventory recommendations")

        display_cols = [
            "store_nbr",
            "family",
            "forecast_demand_next_7_days",
            "simulated_current_stock",
            "expected_lead_time_demand",
            "estimated_safety_stock",
            "estimated_reorder_point",
            "estimated_target_stock",
            "suggested_order_quantity",
            "inventory_risk",
            "reorder_recommended"
        ]

        display_cols = [
            c for c in display_cols if c in filtered_inventory.columns
        ]

        st.dataframe(
            filtered_inventory[display_cols],
            use_container_width=True,
            hide_index=True
        )

        display_download(
            filtered_inventory,
            "filtered_inventory_recommendations.csv",
            "⬇️ Download inventory recommendations"
        )

        st.markdown("### Order quantity by inventory risk")

        fig = px.box(
            filtered_inventory,
            x="inventory_risk",
            y="suggested_order_quantity",
            color="inventory_risk",
            title="Suggested order quantity distribution"
        )
        dark_chart(fig, 380)

        st.markdown("### Items requiring the largest orders")

        top = filtered_inventory.nlargest(
            20, "suggested_order_quantity"
        ).copy()

        top["item"] = (
            "Store " + top["store_nbr"].astype(str)
            + " · " + top["family"]
        )

        fig = px.bar(
            top.sort_values("suggested_order_quantity"),
            x="suggested_order_quantity",
            y="item",
            orientation="h",
            color="inventory_risk",
            title="Top 20 suggested order quantities"
        )
        dark_chart(fig, 600)

# ---------------------------------------------------------------------------
# STORE ANALYSIS
# ---------------------------------------------------------------------------

elif page == "Store Analysis":

    show_hero(
        "Store Performance Analysis",
        "Compare predicted demand and simulated inventory recommendations across stores."
    )

    if filtered_forecast.empty:
        st.warning("No forecast data matches the selected filters.")
    else:
        store_forecast = (
            filtered_forecast.groupby("store_nbr", as_index=False)
            ["predicted_sales"].sum()
            .rename(columns={"predicted_sales": "forecast_demand_7_days"})
        )

        store_inventory = (
            filtered_inventory.groupby("store_nbr", as_index=False)
            .agg(
                suggested_order_quantity=(
                    "suggested_order_quantity", "sum"
                ),
                simulated_current_stock=(
                    "simulated_current_stock", "sum"
                ),
                critical_items=(
                    "inventory_risk",
                    lambda x: int((x == "Critical").sum())
                )
            )
        )

        store_view = store_forecast.merge(
            store_inventory,
            on="store_nbr",
            how="left"
        ).fillna(0)

        fig = px.bar(
            store_view.sort_values("forecast_demand_7_days"),
            x="store_nbr",
            y="forecast_demand_7_days",
            title="Seven-day predicted demand by store"
        )
        dark_chart(fig, 430)

        fig = px.scatter(
            store_view,
            x="forecast_demand_7_days",
            y="suggested_order_quantity",
            size="critical_items",
            hover_name="store_nbr",
            title="Forecast demand versus suggested order quantity",
            labels={
                "forecast_demand_7_days": "7-day forecast demand",
                "suggested_order_quantity": "Suggested order quantity",
                "store_nbr": "Store"
            }
        )
        dark_chart(fig, 430)

        st.dataframe(
            store_view.sort_values(
                "forecast_demand_7_days", ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        display_download(
            store_view,
            "store_analysis.csv",
            "⬇️ Download store analysis"
        )

# ---------------------------------------------------------------------------
# PRODUCT ANALYSIS
# ---------------------------------------------------------------------------

elif page == "Product Analysis":

    show_hero(
        "Product Family Analysis",
        "Identify product families with high forecast demand and simulated inventory risk."
    )

    if filtered_forecast.empty:
        st.warning("No forecast data matches the selected filters.")
    else:
        family_forecast = (
            filtered_forecast.groupby("family", as_index=False)
            ["predicted_sales"].sum()
            .rename(columns={"predicted_sales": "forecast_demand_7_days"})
        )

        family_inventory = (
            filtered_inventory.groupby("family", as_index=False)
            .agg(
                suggested_order_quantity=(
                    "suggested_order_quantity", "sum"
                ),
                simulated_current_stock=(
                    "simulated_current_stock", "sum"
                ),
                critical_items=(
                    "inventory_risk",
                    lambda x: int((x == "Critical").sum())
                ),
                high_risk_items=(
                    "inventory_risk",
                    lambda x: int((x == "High").sum())
                )
            )
        )

        family_view = family_forecast.merge(
            family_inventory,
            on="family",
            how="left"
        ).fillna(0)

        fig = px.bar(
            family_view.sort_values("forecast_demand_7_days"),
            x="forecast_demand_7_days",
            y="family",
            orientation="h",
            title="Predicted demand by product family"
        )
        dark_chart(fig, 650)

        fig = px.scatter(
            family_view,
            x="forecast_demand_7_days",
            y="suggested_order_quantity",
            size="critical_items",
            hover_name="family",
            title="Product-family demand versus suggested orders"
        )
        dark_chart(fig, 450)

        st.dataframe(
            family_view.sort_values(
                "forecast_demand_7_days", ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        display_download(
            family_view,
            "product_family_analysis.csv",
            "⬇️ Download product family analysis"
        )

# ---------------------------------------------------------------------------
# DATA EXPLORER
# ---------------------------------------------------------------------------

elif page == "Data Explorer":

    show_hero(
        "Data Explorer",
        "Inspect and export the saved model forecasts and inventory outputs."
    )

    dataset_choice = st.selectbox(
        "Choose a dataset",
        [
            "Demand Forecasts",
            "Inventory Recommendations",
            "Store Inventory Summary",
            "Inventory Risk Summary"
        ]
    )

    dataset_map = {
        "Demand Forecasts": forecast,
        "Inventory Recommendations": inventory,
        "Store Inventory Summary": stores,
        "Inventory Risk Summary": risk_summary
    }

    chosen_df = dataset_map[dataset_choice]

    if chosen_df.empty:
        st.warning("The selected dataset is empty or unavailable.")
    else:
        st.caption(
            f"{len(chosen_df):,} rows · {len(chosen_df.columns)} columns"
        )

        search_text = st.text_input(
            "Search visible data",
            placeholder="Enter a store number, family, or risk level..."
        )

        display_df = chosen_df.copy()

        if search_text.strip():
            mask = display_df.astype(str).apply(
                lambda col: col.str.contains(
                    search_text, case=False, na=False
                )
            ).any(axis=1)

            display_df = display_df[mask]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        display_download(
            display_df,
            "dashboard_data_export.csv",
            "⬇️ Download selected dataset"
        )

    st.markdown("### Project summary")

    if summary:
        st.json(summary)

    st.markdown("### Methodology notes")
    st.write(
        "- Forecasts are generated from the previously trained model."
    )
    st.write(
        "- The forecast horizon is seven days, using recursive predictions."
    )
    st.write(
        "- Missing historical dates were filled with zero sales in Part 4."
    )
    st.write(
        "- Future promotions are assumed to be zero."
    )
    st.write(
        "- Inventory levels and order quantities are simulated."
    )
    st.write(
        "- Lead time, review period, and safety stock calculations are assumptions."
    )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <p class="small-muted">
    AI-Powered Demand Forecasting and Inventory Optimization System |
    Research Prototype | Machine Learning + Streamlit + Plotly
    </p>
    """,
    unsafe_allow_html=True
)
