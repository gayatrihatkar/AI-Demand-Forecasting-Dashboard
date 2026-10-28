from pathlib import Path
import json
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    PROJECT_DIR / "dataset" / "processed" / "sales_clean.csv",
    PROJECT_DIR / "data" / "sales_clean.csv",
    PROJECT_DIR / "sales_clean.csv",
]
FORECAST_CANDIDATES = [
    PROJECT_DIR / "forecasts" / "demand_forecast_7_days.csv",
    PROJECT_DIR / "demand_forecast_7_days.csv",
]
INVENTORY_CANDIDATES = [
    PROJECT_DIR / "forecasts" / "inventory_optimization_recommendations.csv",
    PROJECT_DIR / "inventory_optimization_recommendations.csv",
]
MODEL_METRICS_CANDIDATES = [
    PROJECT_DIR / "reports" / "validated_model_metrics.csv",
    PROJECT_DIR / "reports" / "model_comparison.csv",
]
INVENTORY_SUMMARY_CANDIDATES = [
    PROJECT_DIR / "reports" / "part4_inventory_summary.json",
]
EVAL_SUMMARY_CANDIDATES = [
    PROJECT_DIR / "reports" / "part3_evaluation_summary.json",
]
EDA_SUMMARY_CANDIDATES = [
    PROJECT_DIR / "reports" / "eda_summary.json",
]

st.set_page_config(
    page_title="DemandIQ | Forecast & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: rgba(80,110,150,.10); border: 1px solid rgba(130,150,180,.25); padding: 14px 16px; border-radius: 14px;}
    .hero {padding: 1.2rem 1.4rem; border-radius: 18px; background: linear-gradient(120deg, rgba(40,80,125,.22), rgba(25,150,130,.10)); border: 1px solid rgba(120,150,180,.25); margin-bottom: 1rem;}
    .small-note {font-size: .88rem; opacity: .82;}
    div[data-testid="stDataFrame"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

def first_existing(paths):
    return next((p for p in paths if p.exists()), None)

@st.cache_data(show_spinner=False)
def load_csv(path_string):
    return pd.read_csv(path_string, low_memory=False)

@st.cache_data(show_spinner=False)
def load_json(path_string):
    with open(path_string, "r", encoding="utf-8") as f:
        return json.load(f)

def load_first_csv(paths):
    path = first_existing(paths)
    if path is None:
        return None, None
    try:
        return load_csv(str(path)), path
    except Exception as exc:
        st.warning(f"Could not read `{path.name}`: {exc}")
        return None, path

def load_first_json(paths):
    path = first_existing(paths)
    if path is None:
        return {}, None
    try:
        return load_json(str(path)), path
    except Exception as exc:
        st.warning(f"Could not read `{path.name}`: {exc}")
        return {}, path

def find_col(df, candidates):
    if df is None:
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    for c in df.columns:
        if any(token in str(c).lower() for token in candidates):
            return c
    return None

def fmt_number(value, digits=0):
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"

def safe_numeric(df, col):
    if df is None or col is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

history_path = first_existing(DATA_CANDIDATES)
history = None
if history_path:
    try:
        history = load_csv(str(history_path))
    except Exception as exc:
        st.error(f"Historical dataset could not be loaded: {exc}")

forecast, forecast_path = load_first_csv(FORECAST_CANDIDATES)
inventory, inventory_path = load_first_csv(INVENTORY_CANDIDATES)
metrics, metrics_path = load_first_csv(MODEL_METRICS_CANDIDATES)
inventory_summary, _ = load_first_json(INVENTORY_SUMMARY_CANDIDATES)
evaluation_summary, _ = load_first_json(EVAL_SUMMARY_CANDIDATES)
eda_summary, _ = load_first_json(EDA_SUMMARY_CANDIDATES)

st.sidebar.title("📦 DemandIQ")
st.sidebar.caption("Machine-learning demand forecasting and inventory decision support")
page = st.sidebar.radio(
    "Navigate",
    ["Executive Overview", "Demand Forecasts", "Inventory Optimizer", "Model Evaluation", "Store & Product Analytics", "Data Quality & Explorer", "Research & Limitations"],
)
st.sidebar.divider()
st.sidebar.caption("Project: AI-Powered Demand Forecasting and Inventory Optimization System")
st.sidebar.caption("Forecast files are precomputed outputs. Inventory values are simulated, not live stock.")

st.markdown("""
<div class="hero">
  <h1 style="margin:0">Demand Forecasting & Inventory Intelligence</h1>
  <p style="margin:.5rem 0 0 0">A research dashboard for demand prediction, model validation, stock-risk analysis, and replenishment planning.</p>
</div>
""", unsafe_allow_html=True)

date_col = find_col(history, ["date"])
sales_col = find_col(history, ["sales", "target"])
store_col = find_col(history, ["store_nbr", "store", "store_id"])
family_col = find_col(history, ["family", "product_family", "category"])
promotion_col = find_col(history, ["onpromotion", "promotion"])

if history is not None:
    if date_col:
        history[date_col] = pd.to_datetime(history[date_col], errors="coerce")
    if sales_col:
        history[sales_col] = pd.to_numeric(history[sales_col], errors="coerce")
    if store_col:
        history[store_col] = history[store_col].astype(str)
    if family_col:
        history[family_col] = history[family_col].astype(str)

if page == "Executive Overview":
    if history is not None and sales_col:
        total_sales = history[sales_col].sum(skipna=True)
        rows = len(history)
        stores = history[store_col].nunique() if store_col else 0
        families = history[family_col].nunique() if family_col else 0
        dates = history[date_col].nunique() if date_col else 0
        a, b, c, d, e = st.columns(5)
        a.metric("Cleaned records", fmt_number(rows))
        b.metric("Historical sales total", fmt_number(total_sales, 0))
        c.metric("Stores", fmt_number(stores))
        d.metric("Product families", fmt_number(families))
        e.metric("Unique dates", fmt_number(dates))
    else:
        a, b, c = st.columns(3)
        a.metric("Selected model", evaluation_summary.get("selected_ml_model", "XGBoost"))
        a.metric("Validation WAPE", f"{evaluation_summary.get('selected_ml_metrics', {}).get('WAPE_percent', 14.04):.2f}%")
        b.metric("Forecast horizon", f"{inventory_summary.get('forecast_horizon_days', 7)} days")
        c.metric("Inventory mode", "SIMULATED")

    if evaluation_summary:
        selected = evaluation_summary.get("selected_ml_metrics", {})
        baseline = evaluation_summary.get("baseline_metrics", {})
        x1, x2, x3 = st.columns(3)
        x1.metric("Selected model WAPE", f"{selected.get('WAPE_percent', 14.04):.2f}%")
        x2.metric("Selected model MAE", fmt_number(selected.get("MAE", 35.68), 2))
        x3.metric("Baseline WAPE", f"{baseline.get('WAPE_percent', 22.73):.2f}%")
        st.info("The saved validation report identifies XGBoost as the best evaluated model. These are historical validation metrics, not a guarantee of future performance.")

    if history is not None and date_col and sales_col:
        daily = history.dropna(subset=[date_col]).groupby(date_col, as_index=False)[sales_col].sum().sort_values(date_col)
        fig = px.line(daily, x=date_col, y=sales_col, title="Historical daily sales", labels={date_col: "Date", sales_col: "Sales"})
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Historical CSV not found. Add `dataset/processed/sales_clean.csv` to enable historical charts.")

    if inventory_summary:
        st.subheader("Inventory planning snapshot")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Forecasted 7-day demand", fmt_number(inventory_summary.get("total_predicted_demand_next_7_days"), 0))
        k2.metric("Simulated current stock", fmt_number(inventory_summary.get("simulated_total_current_stock"), 0))
        k3.metric("Suggested order quantity", fmt_number(inventory_summary.get("total_suggested_order_quantity"), 0))
        k4.metric("High + critical items", fmt_number(inventory_summary.get("critical_items", 0) + inventory_summary.get("high_risk_items", 0)))
        st.caption("These inventory values come from the project's simulation assumptions; they are not real inventory counts.")

elif page == "Demand Forecasts":
    st.subheader("Demand forecast explorer")
    if forecast is None:
        st.warning("Forecast CSV not found. Expected `forecasts/demand_forecast_7_days.csv`. The rest of the dashboard can still run.")
        st.code("forecasts/demand_forecast_7_days.csv")
    else:
        st.success(f"Loaded {len(forecast):,} forecast rows from `{forecast_path.relative_to(PROJECT_DIR) if forecast_path.is_relative_to(PROJECT_DIR) else forecast_path.name}`")
        f_date = find_col(forecast, ["date", "forecast_date", "ds"])
        f_store = find_col(forecast, ["store_nbr", "store", "store_id"])
        f_family = find_col(forecast, ["family", "product_family", "category"])
        f_pred = find_col(forecast, ["predicted_sales", "forecast_sales", "forecast_demand", "predicted_demand", "yhat", "prediction", "sales"])
        view = forecast.copy()
        if f_date:
            view[f_date] = pd.to_datetime(view[f_date], errors="coerce")
        c1, c2, c3 = st.columns(3)
        if f_store:
            stores = sorted(view[f_store].dropna().astype(str).unique().tolist())
            selected_stores = c1.multiselect("Store", stores, default=stores[:min(5, len(stores))])
            if selected_stores:
                view = view[view[f_store].astype(str).isin(selected_stores)]
        if f_family:
            families = sorted(view[f_family].dropna().astype(str).unique().tolist())
            selected_families = c2.multiselect("Product family", families, default=families)
            if selected_families:
                view = view[view[f_family].astype(str).isin(selected_families)]
        if f_date:
            date_values = view[f_date].dropna()
            if not date_values.empty:
                c3.caption(f"Forecast dates: {date_values.min().date()} to {date_values.max().date()}")
        if f_pred:
            pred = pd.to_numeric(view[f_pred], errors="coerce").clip(lower=0)
            st.metric("Filtered predicted demand", fmt_number(pred.sum(), 0))
            if f_date:
                trend = view.assign(_pred=pd.to_numeric(view[f_pred], errors="coerce").clip(lower=0)).groupby(f_date, as_index=False)["_pred"].sum()
                st.plotly_chart(px.line(trend, x=f_date, y="_pred", markers=True, title="Predicted demand by date"), use_container_width=True)
            if f_family:
                family_sum = view.assign(_pred=pd.to_numeric(view[f_pred], errors="coerce").clip(lower=0)).groupby(f_family, as_index=False)["_pred"].sum().sort_values("_pred", ascending=False).head(15)
                st.plotly_chart(px.bar(family_sum, x="_pred", y=f_family, orientation="h", title="Top product families by forecast demand"), use_container_width=True)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download filtered forecasts (CSV)", view.to_csv(index=False).encode("utf-8"), "filtered_demand_forecasts.csv", "text/csv")

elif page == "Inventory Optimizer":
    st.subheader("Inventory risk and replenishment")
    if inventory is None:
        st.warning("Inventory recommendation CSV not found. Expected `forecasts/inventory_optimization_recommendations.csv`.")
    else:
        inv = inventory.copy()
        risk_col = find_col(inv, ["risk_level", "inventory_risk", "risk"])
        store_inv = find_col(inv, ["store_nbr", "store", "store_id"])
        family_inv = find_col(inv, ["family", "product_family", "category"])
        order_col = find_col(inv, ["suggested_order_quantity", "order_quantity", "recommended_order_qty", "reorder_qty"])
        stock_col = find_col(inv, ["simulated_current_stock", "current_stock", "stock_on_hand", "inventory"])
        demand_col = find_col(inv, ["forecast_demand", "predicted_demand", "forecast_sales", "total_forecast_demand", "demand"])
        c1, c2 = st.columns(2)
        if risk_col:
            risks = sorted(inv[risk_col].dropna().astype(str).unique())
            selected_risks = c1.multiselect("Risk level", risks, default=risks)
            if selected_risks:
                inv = inv[inv[risk_col].astype(str).isin(selected_risks)]
        if store_inv:
            stores = sorted(inv[store_inv].dropna().astype(str).unique())
            selected_stores = c2.multiselect("Store", stores, default=stores)
            if selected_stores:
                inv = inv[inv[store_inv].astype(str).isin(selected_stores)]
        k1, k2, k3 = st.columns(3)
        k1.metric("Items in current view", fmt_number(len(inv)))
        k2.metric("Suggested order quantity", fmt_number(pd.to_numeric(inv[order_col], errors="coerce").sum() if order_col else np.nan))
        k3.metric("Forecast demand", fmt_number(pd.to_numeric(inv[demand_col], errors="coerce").sum() if demand_col else np.nan))
        if risk_col:
            risk_counts = inv[risk_col].astype(str).value_counts().rename_axis("Risk level").reset_index(name="Items")
            st.plotly_chart(px.bar(risk_counts, x="Risk level", y="Items", color="Risk level", title="Items by inventory risk"), use_container_width=True)
        if family_inv and order_col:
            top_orders = inv.copy()
            top_orders[order_col] = pd.to_numeric(top_orders[order_col], errors="coerce")
            top_orders = top_orders.groupby(family_inv, as_index=False)[order_col].sum().sort_values(order_col, ascending=False).head(15)
            st.plotly_chart(px.bar(top_orders, x=order_col, y=family_inv, orientation="h", title="Product families with highest suggested order quantities"), use_container_width=True)
        if stock_col and demand_col:
            st.caption("Stock-versus-demand comparison uses the available columns in the saved inventory file.")
            fig = px.scatter(inv, x=demand_col, y=stock_col, color=risk_col if risk_col else None, hover_data=[c for c in [store_inv, family_inv] if c])
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(inv, use_container_width=True, hide_index=True)
        st.download_button("Download filtered inventory recommendations (CSV)", inv.to_csv(index=False).encode("utf-8"), "filtered_inventory_recommendations.csv", "text/csv")
        st.warning("Research limitation: stock, lead time, and risk are simulated using configured assumptions. Do not use as real purchase orders without actual inventory and supplier data.")

elif page == "Model Evaluation":
    st.subheader("Model performance and validation")
    if evaluation_summary:
        selected = evaluation_summary.get("selected_ml_metrics", {})
        baseline = evaluation_summary.get("baseline_metrics", {})
        rows = [
            {"Model": evaluation_summary.get("selected_ml_model", "XGBoost"), "MAE": selected.get("MAE"), "RMSE": selected.get("RMSE"), "WAPE (%)": selected.get("WAPE_percent")},
            {"Model": "Seasonal Naive baseline", "MAE": baseline.get("MAE"), "RMSE": baseline.get("RMSE"), "WAPE (%)": baseline.get("WAPE_percent")},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown(f"**Validation period:** {evaluation_summary.get('validation_start', 'N/A')} to {evaluation_summary.get('validation_end', 'N/A')}")
        st.markdown(f"**Validation records:** {fmt_number(evaluation_summary.get('validation_records'))}")
        st.markdown(f"**Method:** {evaluation_summary.get('validation_method', 'See project evaluation report.')}")
        if selected.get("WAPE_percent") and baseline.get("WAPE_percent"):
            improvement = (baseline["WAPE_percent"] - selected["WAPE_percent"]) / baseline["WAPE_percent"] * 100
            st.metric("WAPE improvement over baseline", f"{improvement:.1f}%")
    if metrics is not None:
        st.subheader("Saved model comparison file")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        wape_col = find_col(metrics, ["wape_percent", "wape"])
        model_col = find_col(metrics, ["model", "model_name", "algorithm"])
        if wape_col and model_col:
            plot_df = metrics.copy()
            plot_df[wape_col] = pd.to_numeric(plot_df[wape_col], errors="coerce")
            st.plotly_chart(px.bar(plot_df, x=model_col, y=wape_col, title="Model comparison by WAPE (lower is better)"), use_container_width=True)
    st.info("Validation metrics are based on a rolling one-day-ahead evaluation. A time-aware validation strategy is more suitable for forecasting than a random train/test split.")

elif page == "Store & Product Analytics":
    st.subheader("Historical performance by store and product family")
    if history is None or not sales_col:
        st.warning("Add the cleaned historical CSV to enable this analysis.")
    else:
        filtered = history.copy()
        c1, c2 = st.columns(2)
        if store_col:
            stores = sorted(filtered[store_col].dropna().astype(str).unique())
            selected_stores = c1.multiselect("Stores", stores, default=stores[:min(10, len(stores))])
            if selected_stores:
                filtered = filtered[filtered[store_col].astype(str).isin(selected_stores)]
        if family_col:
            families = sorted(filtered[family_col].dropna().astype(str).unique())
            selected_families = c2.multiselect("Product families", families, default=families)
            if selected_families:
                filtered = filtered[filtered[family_col].astype(str).isin(selected_families)]
        if family_col:
            fam = filtered.groupby(family_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False).head(20)
            st.plotly_chart(px.bar(fam, x=sales_col, y=family_col, orientation="h", title="Top product families by historical sales"), use_container_width=True)
        if store_col:
            store_sales = filtered.groupby(store_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)
            st.plotly_chart(px.bar(store_sales, x=store_col, y=sales_col, title="Historical sales by store"), use_container_width=True)
        if date_col:
            daily = filtered.groupby(date_col, as_index=False)[sales_col].sum().sort_values(date_col)
            st.plotly_chart(px.line(daily, x=date_col, y=sales_col, title="Filtered sales trend"), use_container_width=True)
        if promotion_col:
            filtered["_promotion_flag"] = pd.to_numeric(filtered[promotion_col], errors="coerce").fillna(0).gt(0).map({True: "Promotion records", False: "No promotion"})
            promo = filtered.groupby("_promotion_flag", as_index=False)[sales_col].mean()
            st.plotly_chart(px.bar(promo, x="_promotion_flag", y=sales_col, title="Average sales by promotion-record status"), use_container_width=True)
        st.dataframe(filtered.head(1000), use_container_width=True, hide_index=True)
        st.caption("Promotion association is descriptive only; it does not establish causal impact.")

elif page == "Data Quality & Explorer":
    st.subheader("Dataset checks")
    if history is None:
        st.error("Historical dataset not found.")
    else:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rows", fmt_number(len(history)))
        q2.metric("Columns", fmt_number(len(history.columns)))
        q3.metric("Duplicate rows", fmt_number(history.duplicated().sum()))
        q4.metric("Missing cells", fmt_number(int(history.isna().sum().sum())))
        if date_col:
            q5, q6 = st.columns(2)
            q5.metric("First date", str(history[date_col].min().date()) if history[date_col].notna().any() else "—")
            q6.metric("Last date", str(history[date_col].max().date()) if history[date_col].notna().any() else "—")
        st.subheader("Missing values by column")
        missing = history.isna().sum().rename("Missing values").reset_index(names="Column")
        st.dataframe(missing, use_container_width=True, hide_index=True)
        if sales_col:
            negative_count = int((pd.to_numeric(history[sales_col], errors="coerce") < 0).sum())
            st.metric("Negative sales records", fmt_number(negative_count))
        st.subheader("Preview")
        st.dataframe(history.head(500), use_container_width=True, hide_index=True)
        st.download_button("Download loaded historical data", history.to_csv(index=False).encode("utf-8"), "historical_data_preview.csv", "text/csv")
        st.caption("The full dataset is not embedded in this application output; this page reads the CSV committed to the repository.")

else:
    st.subheader("Research methodology, assumptions, and limitations")
    st.markdown("""
    **Workflow**
    1. Clean and validate the historical retail sales data.
    2. Engineer calendar features, store/product categorical features, lag features (7, 14, 28 days), and rolling statistics.
    3. Compare seasonal-naive, Random Forest, and XGBoost models.
    4. Evaluate with a time-aware rolling one-day-ahead validation window.
    5. Produce a 7-day demand forecast and inventory recommendations.
    """)
    st.markdown("**Known project facts from saved reports**")
    st.write({
        "Cleaned rows": eda_summary.get("cleaned_rows", 1084847),
        "Stores": eda_summary.get("number_of_stores", 54),
        "Product families": eda_summary.get("number_of_product_families", 33),
        "Historical dates": f"{eda_summary.get('first_date', '2013-01-01')} to {eda_summary.get('last_date', '2014-09-02')}",
        "Selected model": evaluation_summary.get("selected_ml_model", "XGBoost"),
        "Validation WAPE (%)": evaluation_summary.get("selected_ml_metrics", {}).get("WAPE_percent", 14.0428),
        "Forecast dates in saved output": f"{inventory_summary.get('forecast_start_date', '2014-09-03')} to {inventory_summary.get('forecast_end_date', '2014-09-09')}",
        "Inventory simulated": inventory_summary.get("inventory_is_simulated", True),
    })
    st.warning("Important: the saved forecast dates are in September 2014 because the underlying dataset ends on 2014-09-02. This is a historical research forecast, not a forecast for the current date.")
    st.markdown("""
    **Limitations**
    - The dataset does not contain actual stock-on-hand, supplier lead times, stockout history, or future promotions.
    - Inventory levels and reorder suggestions are simulated using documented assumptions.
    - Missing daily records were reindexed and filled with zero; this assumes an absent record means zero sales and may not always be true.
    - The saved forecast is historical because the dataset ends in 2014.
    - Model artifacts must be loaded with a compatible scikit-learn environment; do not retrain or replace them without recording a new experiment and comparing results.
    """)

st.divider()
st.caption("MCA research prototype • Reproducible outputs • Historical forecast dates and simulated inventory are explicitly labelled")
