from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import traceback

# ============================================================
# Demand Intelligence — MCA Research Dashboard
# Reads existing project outputs; does not retrain or fabricate
# model forecasts. Inventory fields are explicitly simulated.
# ============================================================
BASE = Path(__file__).resolve().parent
FORECASTS = BASE / "forecasts"
REPORTS = BASE / "reports"
CONFIG = BASE / "config"
DATASET = BASE / "dataset" / "processed"

st.set_page_config(
    page_title="Demand Intelligence | Retail AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed Teal & Beige theme (theme selector intentionally removed)
T = {"bg":"#F4F0E6", "panel":"#FFFCF5", "sidebar":"#153E3B", "text":"#18312F", "muted":"#526B66", "accent":"#0F766E", "accent2":"#C49A55", "border":"#D8D0BF", "plot":"plotly_white"}
st.markdown(f"""
<style>
.stApp {{background:{T['bg']}; color:{T['text']};}}
[data-testid="stSidebar"] {{background:linear-gradient(180deg,#123B37 0%,#174943 58%,#103631 100%); border-right:1px solid #2D625A;}}
[data-testid="stSidebar"] * {{color:#F8F5EC !important;}}
[data-testid="stSidebar"] section {{padding-top:1.2rem;}}
/* Branded sidebar header */
.sidebar-brand {{padding:17px 15px 19px; margin:4px 0 18px; border:1px solid rgba(235,225,201,.2); border-radius:18px; background:linear-gradient(135deg,rgba(246,239,220,.13),rgba(246,239,220,.035)); box-shadow:0 10px 28px rgba(0,0,0,.10);}}
.sidebar-brand-top {{display:flex; align-items:center; gap:11px;}}
.sidebar-brand-icon {{display:flex; align-items:center; justify-content:center; width:43px; height:43px; border-radius:13px; background:#E7DCC4; color:#174943 !important; font-size:22px; box-shadow:inset 0 0 0 1px rgba(21,62,59,.08);}}
.sidebar-brand-title {{font-size:17px; line-height:1.2; font-weight:750; letter-spacing:.01em; color:#FFFDF7 !important;}}
.sidebar-brand-sub {{font-size:11px; margin-top:5px; color:#D4E6DD !important; letter-spacing:.09em; text-transform:uppercase;}}
.sidebar-brand-rule {{height:3px; width:52px; margin-top:15px; border-radius:4px; background:linear-gradient(90deg,#D8B978,#F1E7D1);}}
[data-testid="stSidebar"] div.stButton > button {{background:#E7DCC4 !important; color:#153E3B !important; border:1px solid #F1E7D1 !important; border-radius:12px !important; font-weight:700 !important; min-height:43px; box-shadow:0 4px 12px rgba(0,0,0,.10);}}
[data-testid="stSidebar"] div.stButton > button:hover {{background:#F5EBD7 !important; color:#103631 !important; border-color:#F5EBD7 !important; transform:translateY(-1px);}}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {{font-size:11px !important; font-weight:800 !important; letter-spacing:.15em !important; color:#BBD2C9 !important; margin:5px 0 12px !important;}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{gap:6px;}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {{display:flex; align-items:center; padding:9px 11px; margin:0; border:1px solid rgba(241,231,209,.10); border-radius:11px; background:rgba(255,255,255,.035); transition:all .18s ease;}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover {{background:rgba(231,220,196,.13); border-color:rgba(231,220,196,.38);}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{background:linear-gradient(100deg,rgba(231,220,196,.22),rgba(231,220,196,.09)); border:1px solid rgba(231,220,196,.58); box-shadow:inset 3px 0 0 #E7DCC4;}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p {{font-size:13px !important; font-weight:600; color:#FFFDF7 !important;}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label [data-testid="stMarkdownContainer"] {{padding-left:3px;}}
[data-testid="stMetric"] {{background:linear-gradient(135deg,{T['panel']},{T['bg']}); border:1px solid {T['border']}; padding:17px 18px; border-radius:15px;}}
[data-testid="stMetricLabel"] {{color:{T['muted']} !important;}}
[data-testid="stMetricValue"] {{color:{T['text']} !important;}}
h1,h2,h3 {{color:{T['text']} !important; letter-spacing:-0.025em;}}
p, label, .stCaption {{color:{T['muted']};}}
div.stButton > button, div.stDownloadButton > button {{border-radius:10px; border:1px solid {T['border']}; font-weight:600;}}
div.stButton > button {{background:{T['accent']} !important; color:#FFFFFF !important; border:1px solid {T['accent']} !important;}}
div.stButton > button:hover {{background:#0B5F59 !important; color:#FFFFFF !important; border-color:#0B5F59 !important;}}
div.stDownloadButton > button {{background:{T['panel']} !important; color:{T['text']} !important;}}
div.stButton > button[kind="primary"] {{background:{T['accent']}; color:#FFFFFF; border:0;}}
[data-testid="stDataFrame"] {{border:1px solid {T['border']}; border-radius:12px;}}
.hero {{background:linear-gradient(115deg,#153E3B,#246B63); border:1px solid #2D776D; padding:26px 28px; border-radius:18px; margin:5px 0 20px;}}
.hero h1 {{font-size:30px; margin:0 0 8px; color:#FFFDF7 !important;}}
.hero p {{font-size:15px; margin:0; color:#E2F0E8;}}
.callout {{background:{T['panel']}; border:1px solid {T['border']}; border-left:4px solid {T['accent']}; padding:13px 16px; border-radius:10px; margin:8px 0 16px; color:{T['text']};}}
.small-muted {{color:{T['muted']}; font-size:12px;}}
@media (max-width: 700px) {{ .sidebar-brand {{padding:13px 11px;}} .sidebar-brand-title {{font-size:15px;}} }}
/* Sidebar polish: ensure nested Streamlit text respects contrast and nav reads as cards. */
[data-testid="stSidebar"] div.stButton > button,
[data-testid="stSidebar"] div.stButton > button p,
[data-testid="stSidebar"] div.stButton > button span {{ color:#153E3B !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {{ display:none; }}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{ gap:7px !important; padding-top:2px; }}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {{
  min-height:42px; padding:9px 12px !important; margin:0 !important;
  border:1px solid rgba(231,220,196,.14) !important; border-radius:12px !important;
  background:linear-gradient(105deg,rgba(255,255,255,.065),rgba(255,255,255,.025)) !important;
  box-shadow:0 2px 8px rgba(0,0,0,.045); transition:background .18s ease,border-color .18s ease,transform .18s ease;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover {{
  background:rgba(231,220,196,.13) !important; border-color:rgba(231,220,196,.48) !important; transform:translateX(2px);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{
  background:linear-gradient(100deg,#E7DCC4,#D7C6A4) !important; border-color:#F4E9D2 !important;
  box-shadow:inset 4px 0 0 #0F766E,0 5px 14px rgba(0,0,0,.12) !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p {{ color:#153E3B !important; font-weight:750 !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p {{ font-size:13.5px !important; line-height:1.25; }}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label [data-testid="stMarkdownContainer"] {{ padding-left:5px; }}
.sidebar-section-label {{font-size:10px; font-weight:800; letter-spacing:.18em; color:#BBD2C9 !important; margin:19px 0 9px; text-transform:uppercase;}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def read_csv(path):
    p = Path(path)
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=60)
def read_json(path):
    p = Path(path)
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}


@st.cache_data(ttl=60)
def read_table(path):
    """Read CSV or Parquet without requiring a specific stored format."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame()


def normalize_model_name(value):
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if "xgboost" in text or text in {"xgb", "xgb regressor"}:
        return "XGBoost"
    if "random forest" in text or "randomforest" in text:
        return "Random Forest"
    if "seasonal" in text or "naive" in text:
        return "Seasonal Naive"
    return str(value).strip()


def make_features(history, future_date, store_id, family, promotion=0):
    """Build inference features from one continuous daily sales series."""
    values = pd.Series(history, dtype="float64").fillna(0.0).tolist()
    def lag(n):
        return float(values[-n]) if len(values) >= n else 0.0
    last7 = values[-7:]
    last28 = values[-28:]
    rolling7 = float(np.mean(last7)) if last7 else 0.0
    rolling28 = float(np.mean(last28)) if last28 else 0.0
    std7 = float(np.std(last7, ddof=1)) if len(last7) > 1 else 0.0
    d = pd.Timestamp(future_date)
    return {
        "store_nbr": store_id, "family": family, "year": d.year,
        "month": d.month, "day": d.day, "day_of_week": d.dayofweek,
        "day_of_year": d.dayofyear, "week_of_year": int(d.isocalendar().week),
        "is_weekend": int(d.dayofweek >= 5), "onpromotion": promotion,
        "sales_lag_7": lag(7), "sales_lag_14": lag(14), "sales_lag_28": lag(28),
        "sales_rolling_mean_7": rolling7, "sales_rolling_mean_28": rolling28,
        "sales_rolling_std_7": std7,
    }


def generate_forecast(history_df, model_name, horizon, model_config):
    """Generate recursive forecasts in daily batches using a saved compatible model."""
    required = {"date", "store_nbr", "family", "sales"}
    missing = required - set(history_df.columns)
    if missing:
        raise ValueError("Historical data is missing required columns: " + ", ".join(sorted(missing)))
    data = history_df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["sales"] = pd.to_numeric(data["sales"], errors="coerce").fillna(0).clip(lower=0)
    data["store_nbr"] = pd.to_numeric(data["store_nbr"], errors="coerce")
    data = data.dropna(subset=["date", "store_nbr", "family"])
    if data.empty:
        raise ValueError("Historical sales data is empty after date and key validation.")
    last_date = data["date"].max().normalize()
    feature_columns = (model_config or {}).get("feature_columns", [
        "store_nbr", "family", "year", "month", "day", "day_of_week", "day_of_year", "week_of_year", "is_weekend", "onpromotion", "sales_lag_7", "sales_lag_14", "sales_lag_28", "sales_rolling_mean_7", "sales_rolling_mean_28", "sales_rolling_std_7"
    ])
    estimator = None
    if model_name != "Seasonal Naive":
        try:
            import joblib
        except Exception as exc:
            raise RuntimeError(f"joblib is unavailable: {exc}") from exc
        candidates = {
            "XGBoost": [BASE / "models" / "xgboost_model.joblib", BASE / "models" / "best_demand_model.joblib"],
            "Random Forest": [BASE / "models" / "random_forest_model.joblib"],
        }.get(model_name, [])
        model_path = next((x for x in candidates if x.exists()), None)
        if model_path is None:
            raise FileNotFoundError(f"No saved model file found for {model_name}. Checked: " + ", ".join(str(x.relative_to(BASE)) for x in candidates))
        try:
            estimator = joblib.load(model_path)
        except Exception as exc:
            raise RuntimeError(f"Could not load the saved {model_name} artifact ({model_path.name}). This may be a scikit-learn/package-version compatibility issue. The existing model was not changed or retrained. Details: {type(exc).__name__}: {exc}") from exc
        if not hasattr(estimator, "predict"):
            raise RuntimeError(f"The saved artifact {model_path.name} does not expose predict().")
    histories = {}
    for (store_id, family), group in data.groupby(["store_nbr", "family"], sort=False):
        series = group.groupby("date")["sales"].sum().sort_index()
        full_index = pd.date_range(series.index.min().normalize(), last_date, freq="D")
        histories[(store_id, family)] = series.reindex(full_index, fill_value=0.0).astype(float).tolist()
    output = []
    for step in range(1, int(horizon) + 1):
        target_date = last_date + pd.Timedelta(days=step)
        keys = list(histories.keys())
        if model_name == "Seasonal Naive":
            preds = [max(0.0, float(histories[k][-7] if len(histories[k]) >= 7 else histories[k][-1])) for k in keys]
        else:
            rows = [make_features(histories[k], target_date, k[0], k[1], promotion=0) for k in keys]
            X = pd.DataFrame([{c: row.get(c, 0) for c in feature_columns} for row in rows], columns=feature_columns)
            try:
                preds = np.asarray(estimator.predict(X), dtype=float).reshape(-1).tolist()
            except Exception as exc:
                raise RuntimeError(f"Saved {model_name} model loaded, but prediction failed for its expected feature schema. Check model preprocessing, package versions and feature columns. Details: {type(exc).__name__}: {exc}") from exc
            if len(preds) != len(keys):
                raise RuntimeError(f"Model returned {len(preds)} predictions for {len(keys)} store-product combinations.")
        for key, pred in zip(keys, preds):
            pred = max(0.0, float(pred)) if np.isfinite(pred) else 0.0
            output.append({"forecast_date": target_date, "store_nbr": key[0], "family": key[1], "predicted_sales": pred, "model_used": model_name, "forecast_generated_at": datetime.now().isoformat(timespec="seconds"), "onpromotion_assumption": 0})
            histories[key].append(pred)
    return pd.DataFrame(output)

def calculate_inventory_scenario(forecast_df, historical_df, lead_days=3, review_days=7, service_z=1.65, stock_cover_days=3.0, min_order=0, pack_size=1):
    """Transparent scenario calculator; stock is simulated unless user provides actual values."""
    if forecast_df.empty:
        return pd.DataFrame()
    hist = historical_df.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["sales"] = pd.to_numeric(hist["sales"], errors="coerce").fillna(0).clip(lower=0)
    grouped = hist.groupby(["store_nbr", "family"])["sales"]
    avg = grouped.mean().rename("historical_avg_daily_sales_28d")
    std = grouped.std().fillna(0).rename("historical_daily_sales_std")
    f = forecast_df.copy()
    f["forecast_date"] = pd.to_datetime(f["forecast_date"], errors="coerce")
    dem = f.groupby(["store_nbr", "family"]).agg(forecast_daily_average=("predicted_sales", "mean"), forecast_horizon_demand=("predicted_sales", "sum")).reset_index()
    dem = dem.join(avg, on=["store_nbr", "family"]).join(std, on=["store_nbr", "family"])
    dem[["historical_avg_daily_sales_28d", "historical_daily_sales_std"]] = dem[["historical_avg_daily_sales_28d", "historical_daily_sales_std"]].fillna(0)
    dem["assumed_current_stock"] = dem["historical_avg_daily_sales_28d"].clip(lower=0) * float(stock_cover_days)
    dem["expected_lead_time_demand"] = dem["forecast_daily_average"].clip(lower=0) * int(lead_days)
    dem["estimated_safety_stock"] = float(service_z) * dem["historical_daily_sales_std"] * np.sqrt(max(int(lead_days), 1))
    dem["estimated_reorder_point"] = dem["expected_lead_time_demand"] + dem["estimated_safety_stock"]
    dem["estimated_target_stock"] = dem["forecast_daily_average"].clip(lower=0) * (int(lead_days)+int(review_days)) + dem["estimated_safety_stock"]
    dem["suggested_order_quantity"] = (dem["estimated_target_stock"] - dem["assumed_current_stock"]).clip(lower=0)
    if int(pack_size) > 1:
        dem["suggested_order_quantity"] = np.ceil(dem["suggested_order_quantity"] / int(pack_size)) * int(pack_size)
    dem["suggested_order_quantity"] = dem["suggested_order_quantity"].clip(lower=int(min_order))
    dem["estimated_days_of_supply"] = np.where(dem["forecast_daily_average"] > 0, dem["assumed_current_stock"] / dem["forecast_daily_average"], np.inf)
    dem["inventory_risk"] = np.select([dem["assumed_current_stock"] < dem["expected_lead_time_demand"], dem["assumed_current_stock"] < dem["estimated_reorder_point"]], ["Critical", "High"], default="Low")
    dem["reorder_recommended"] = dem["suggested_order_quantity"] > 0
    dem["inventory_is_simulated"] = True
    return dem

forecast = read_csv(FORECASTS / "demand_forecast_7_days.csv")
inventory = read_csv(FORECASTS / "inventory_optimization_recommendations.csv")
stores = read_csv(FORECASTS / "store_inventory_summary.csv")
risk_summary = read_csv(REPORTS / "inventory_risk_summary.csv")
summary = read_json(REPORTS / "part4_inventory_summary.json")
metrics = read_csv(REPORTS / "model_comparison.csv")
if metrics.empty: metrics = read_csv(REPORTS / "validated_model_metrics.csv")
# Normalize common metric-header variants from saved evaluation files.
if not metrics.empty:
    metrics.columns = [str(c).strip() for c in metrics.columns]
    rename = {}
    for col in metrics.columns:
        key = col.strip().lower().replace(" ", "_").replace("%", "percent")
        if key in {"model", "model_name", "algorithm"}: rename[col] = "Model"
        elif key in {"mae", "mean_absolute_error"}: rename[col] = "MAE"
        elif key in {"rmse", "root_mean_squared_error"}: rename[col] = "RMSE"
        elif key in {"wape", "wape_percent", "wapepercent"}: rename[col] = "WAPE_percent"
        elif key in {"mape", "mape_percent", "mapepercent"}: rename[col] = "MAPE_percent"
    metrics = metrics.rename(columns=rename)
    if "Model" in metrics.columns: metrics["Model"] = metrics["Model"].map(normalize_model_name)
    if "Model" not in metrics.columns and len(metrics.columns):
        metrics = metrics.rename(columns={metrics.columns[0]: "Model"})
    for col in ["MAE", "RMSE", "WAPE_percent", "MAPE_percent"]:
        if col in metrics.columns: metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

# If evaluation CSVs were not committed, use the project's previously recorded
# validation results as a clearly labelled fallback (not newly calculated metrics).
# These values are from the project's existing evaluation summary.
if metrics.empty:
    metrics = pd.DataFrame([
        {"Model": "XGBoost", "MAE": 35.6759, "RMSE": 182.5178, "WAPE_percent": 14.0428},
        {"Model": "Random Forest", "MAE": 37.2572, "RMSE": 193.7198, "WAPE_percent": 14.6652},
        {"Model": "Seasonal Naive", "MAE": 57.7536, "RMSE": 303.7497, "WAPE_percent": 22.7330},
    ])
    metrics_source = "project_recorded_fallback"
else:
    metrics_source = "saved_report"

clean_data = read_table(DATASET / "sales_clean.parquet")
if clean_data.empty: clean_data = read_table(DATASET / "sales_clean.csv")
features_data = read_table(DATASET / "modeling_features.parquet")
if features_data.empty: features_data = read_table(DATASET / "modeling_features.csv")
config = read_json(CONFIG / "inventory_config.json")
model_config = read_json(CONFIG / "model_config.json")

if forecast.empty or inventory.empty:
    st.error("Required forecast or inventory output files are missing.")
    st.markdown("Expected files: `forecasts/demand_forecast_7_days.csv` and `forecasts/inventory_optimization_recommendations.csv`.")
    st.info("Keep the existing project folders beside app.py. Run the forecasting notebook's output-generation step and commit the CSV files to GitHub.")
    st.stop()

# Normalize fields
if "forecast_date" in forecast:
    forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"], errors="coerce")
for c in ["store_nbr","predicted_sales"]:
    if c in forecast: forecast[c] = pd.to_numeric(forecast[c], errors="coerce")
for c in ["store_nbr","historical_avg_daily_sales_28d","simulated_current_stock","estimated_reorder_point","estimated_safety_stock","estimated_target_stock","suggested_order_quantity","forecast_demand_next_7_days","forecast_daily_average","expected_lead_time_demand"]:
    if c in inventory: inventory[c] = pd.to_numeric(inventory[c], errors="coerce").fillna(0)
for c in ["family","inventory_risk"]:
    if c in inventory: inventory[c] = inventory[c].astype(str)
if "reorder_recommended" in inventory:
    inventory["reorder_recommended"] = inventory["reorder_recommended"].astype(str).str.lower().isin(["true","1","yes"])

def fmt(v): return f"{float(v):,.0f}" if pd.notna(v) else "—"
def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)
def note():
    st.markdown('<div class="callout"><b>Research prototype — simulated inventory.</b> Historical sales are available, but actual stock-on-hand and supplier lead times are not. Stock, reorder points, risk labels and order quantities are estimates based on assumptions; future promotions are assumed to be zero.</div>', unsafe_allow_html=True)
def chart(fig, height=400):
    fig.update_layout(template=T["plot"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=T["text"], height=height, margin=dict(l=18,r=18,t=55,b=18), legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
def download(df, name, label="⬇️ Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")

all_stores = sorted(inventory["store_nbr"].dropna().unique().tolist()) if "store_nbr" in inventory else []
all_families = sorted(inventory["family"].dropna().unique().tolist()) if "family" in inventory else []
risk_order = ["Critical","High","Medium","Low"]
all_risks = [r for r in risk_order if r in inventory.get("inventory_risk", pd.Series(dtype=str)).unique()]
with st.sidebar:
    # Visual-only sidebar branding; dashboard data and feature logic remain unchanged.
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-brand-top">
        <div class="sidebar-brand-icon">◈</div>
        <div>
          <div class="sidebar-brand-title">Demand Intelligence</div>
          <div class="sidebar-brand-sub">Retail AI · Forecasting</div>
        </div>
      </div>
      <div class="sidebar-brand-rule"></div>
      <div style="font-size:12px;line-height:1.5;color:#D4E6DD;margin-top:13px;">Demand planning, inventory signals<br>and decision support in one place.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("↻   Refresh data", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.markdown('<div class="sidebar-section-label">Workspace</div>', unsafe_allow_html=True)

pages = [
    "Executive Overview", "Demand Forecasting", "Inventory Optimizer",
    "Smart Alerts", "Sales Explorer", "What-if Simulator",
    "Model Evaluation", "Store & Product Analytics",
    "Business Intelligence", "Data Quality & Explorer"
]
with st.sidebar:
    page = st.radio("NAVIGATION", pages, index=0)

# Global filters are placed in the Executive Overview content area, not the sidebar.
# Their selections persist in session state and are applied to all dashboard sections.
if "stores_filter" not in st.session_state: st.session_state["stores_filter"] = all_stores.copy()
if "families_filter" not in st.session_state: st.session_state["families_filter"] = all_families.copy()
if "risk_filter" not in st.session_state: st.session_state["risk_filter"] = all_risks.copy()
if page == "Executive Overview":
    st.markdown("### 🔎 Global filters")
    fx1, fx2, fx3 = st.columns([1, 1.5, 1])
    with fx1:
        st.multiselect("Stores", all_stores, key="stores_filter")
    with fx2:
        st.multiselect("Product families", all_families, key="families_filter")
    with fx3:
        st.multiselect("Inventory risk", all_risks, key="risk_filter")
    st.caption("These filters are shared by the other dashboard sections. Return to Executive Overview to change them.")
selected_stores = [x for x in st.session_state["stores_filter"] if x in all_stores]
selected_families = [x for x in st.session_state["families_filter"] if x in all_families]
selected_risks = [x for x in st.session_state["risk_filter"] if x in all_risks]
fi = inventory[inventory["store_nbr"].isin(selected_stores) & inventory["family"].isin(selected_families) & inventory["inventory_risk"].isin(selected_risks)].copy()
ff = forecast[forecast["store_nbr"].isin(selected_stores) & forecast["family"].isin(selected_families)].copy()
if "forecast_date" in ff: ff = ff.dropna(subset=["forecast_date"])
total_demand = ff["predicted_sales"].sum() if "predicted_sales" in ff else 0
total_orders = fi["suggested_order_quantity"].sum() if "suggested_order_quantity" in fi else 0
critical_count = int((fi.get("inventory_risk", pd.Series(dtype=str)) == "Critical").sum())
high_count = int((fi.get("inventory_risk", pd.Series(dtype=str)) == "High").sum())

# -------------------- EXECUTIVE OVERVIEW --------------------
if page == "Executive Overview":
    hero("Demand Intelligence", "AI-powered demand forecasting, inventory risk intelligence, and retail decision support.")
    note()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("7-day forecast demand", fmt(total_demand))
    c2.metric("Stores in view", f"{fi['store_nbr'].nunique():,}")
    c3.metric("Suggested order quantity", fmt(total_orders))
    c4.metric("Critical / high risk", f"{critical_count:,} / {high_count:,}")
    st.markdown("### Executive decision console")
    st.caption("Select a model to update its validation scorecard and comparison view. This does not retrain the model or regenerate forecast CSVs.")
    if metrics_source == "project_recorded_fallback":
        st.info("Evaluation report CSV not found in this deployment; showing the project's previously recorded validation metrics.")
    if not metrics.empty and "Model" in metrics.columns:
        model_names = metrics["Model"].dropna().astype(str).drop_duplicates().tolist()
    else:
        model_names = ["XGBoost", "Random Forest", "Seasonal Naive"]
    if not model_names:
        model_names = ["No saved model metrics"]
    chosen_model = st.selectbox("Select model", model_names, key="overview_model")
    # The selected model is applied immediately, without a separate non-reactive button.
    st.session_state["active_model_view"] = chosen_model
    active = st.session_state["active_model_view"]
    if not metrics.empty and "Model" in metrics.columns and active in metrics["Model"].astype(str).values:
        row = metrics[metrics["Model"].astype(str) == active].iloc[0]
        a,b,c,d = st.columns(4)
        a.metric("Selected model", active)
        b.metric("MAE", f"{row['MAE']:,.3f}" if "MAE" in metrics.columns and pd.notna(row.get("MAE")) else "Not available")
        c.metric("RMSE", f"{row['RMSE']:,.3f}" if "RMSE" in metrics.columns and pd.notna(row.get("RMSE")) else "Not available")
        d.metric("WAPE", f"{row['WAPE_percent']:.2f}%" if "WAPE_percent" in metrics.columns and pd.notna(row.get("WAPE_percent")) else (f"{row['MAPE_percent']:.2f}% MAPE" if "MAPE_percent" in metrics.columns and pd.notna(row.get("MAPE_percent")) else "Not available"))
    elif metrics.empty:
        st.warning("No model evaluation metrics are available. Add reports/model_comparison.csv or reports/validated_model_metrics.csv.")
    else:
        st.info("The selected model has no matching row in the saved evaluation report.")
    if not ff.empty:
        daily=ff.groupby("forecast_date",as_index=False)["predicted_sales"].sum()
        chart(px.area(daily,x="forecast_date",y="predicted_sales",markers=True,title="Forecast demand timeline"),380)
    left,right=st.columns([1,1])
    with left:
        if not fi.empty:
            rc=fi["inventory_risk"].value_counts().reindex(risk_order,fill_value=0).rename_axis("Risk").reset_index(name="Items")
            chart(px.bar(rc,x="Risk",y="Items",color="Risk",title="Inventory risk snapshot"),320)
    with right:
        st.markdown("#### Highest-priority combinations")
        if not fi.empty:
            priority=fi.copy()
            priority["_risk_rank"]=priority["inventory_risk"].map({"Critical":0,"High":1,"Medium":2,"Low":3}).fillna(4)
            priority=priority.sort_values(["_risk_rank","suggested_order_quantity"],ascending=[True,False]).head(10)
            cols=[c for c in ["store_nbr","family","inventory_risk","simulated_current_stock","estimated_reorder_point","suggested_order_quantity"] if c in priority]
            st.dataframe(priority[cols],use_container_width=True,hide_index=True)
    st.markdown("### Quick actions")
    q1,q2,q3=st.columns(3)
    with q1: st.info("**Investigate risk**\n\nOpen Smart Alerts to review critical and high-risk combinations.")
    with q2: st.info("**Test assumptions**\n\nUse What-if Simulator to estimate changes from stock cover and lead-time assumptions.")
    with q3: st.info("**Explain performance**\n\nUse Model Evaluation to compare saved validation metrics.")

# -------------------- DEMAND FORECASTING --------------------
elif page == "Demand Forecasting":
    hero("Demand Forecasting", "Explore saved forecasts or generate a new horizon using a compatible saved model.")
    st.markdown("### Generate a forecast")
    with st.expander("Forecast controls", expanded=False):
        model_choice = st.selectbox("Forecast model", ["XGBoost", "Random Forest", "Seasonal Naive"], index=0, key="forecast_model_choice")
        horizon_choice = st.slider("Forecast horizon (days)", 1, 28, 7, key="forecast_horizon_choice")
        st.caption("Future promotions are assumed to be zero. A generated forecast is kept in this session until downloaded; it does not overwrite your committed forecast CSV.")
        if st.button("Generate forecast with selected model", type="primary", key="generate_forecast_btn"):
            if clean_data.empty:
                st.error("Historical dataset is unavailable. Add dataset/processed/sales_clean.parquet or sales_clean.csv.")
            else:
                with st.spinner("Preparing historical series and generating predictions…"):
                    try:
                        generated = generate_forecast(clean_data, model_choice, horizon_choice, model_config)
                        st.session_state["generated_forecast_df"] = generated
                        st.session_state["generated_forecast_model"] = model_choice
                        st.session_state["generated_forecast_horizon"] = horizon_choice
                        run_history = st.session_state.get("forecast_run_history", [])
                        run_history.insert(0, {"generated_at": datetime.now().isoformat(timespec="seconds"), "model": model_choice, "horizon_days": int(horizon_choice), "rows": int(len(generated)), "total_predicted_demand": float(generated["predicted_sales"].sum())})
                        st.session_state["forecast_run_history"] = run_history[:25]
                        st.success(f"Generated {len(generated):,} forecast rows using {model_choice}.")
                    except Exception as exc:
                        st.error(f"Forecast generation could not complete: {exc}")
                        st.caption("Your saved models and existing forecast files were not overwritten. If this is a model compatibility error, use the original training environment to export a compatible inference pipeline.")
    generated = st.session_state.get("generated_forecast_df", pd.DataFrame())
    if not generated.empty:
        st.markdown(f"#### Current generated run · {st.session_state.get('generated_forecast_model', 'Unknown')} · {st.session_state.get('generated_forecast_horizon', 0)} days")
        gf = generated.copy()
        gf["forecast_date"] = pd.to_datetime(gf["forecast_date"], errors="coerce")
        gf = gf[gf["store_nbr"].isin(selected_stores) & gf["family"].isin(selected_families)]
        ga, gb, gc = st.columns(3)
        ga.metric("Generated demand", fmt(gf["predicted_sales"].sum()) if not gf.empty else "0")
        gb.metric("Generated rows", f"{len(gf):,}")
        gc.metric("Forecast dates", f"{gf['forecast_date'].nunique():,}" if not gf.empty else "0")
        if not gf.empty:
            gdaily = gf.groupby("forecast_date", as_index=False)["predicted_sales"].sum()
            chart(px.area(gdaily, x="forecast_date", y="predicted_sales", markers=True, title="Newly generated demand forecast"), 360)
            st.dataframe(gf.head(1000), use_container_width=True, hide_index=True)
            download(gf, "generated_demand_forecast.csv", "⬇️ Download generated forecast")
            st.markdown("#### Inventory recommendations for this generated run")
            ic1, ic2, ic3, ic4 = st.columns(4)
            inv_lead = ic1.number_input("Lead time (days)", 1, 30, 3, key="generated_inv_lead")
            inv_review = ic2.number_input("Review period (days)", 1, 28, 7, key="generated_inv_review")
            inv_cover = ic3.number_input("Assumed stock cover (days)", 0.0, 30.0, 3.0, 0.5, key="generated_inv_cover")
            inv_service = ic4.selectbox("Service level", [90, 95, 97, 99], index=1, key="generated_inv_service")
            z_values = {90: 1.282, 95: 1.645, 97: 1.881, 99: 2.326}
            if st.button("Calculate inventory recommendations", key="generated_inventory_btn"):
                try:
                    generated_inventory = calculate_inventory_scenario(generated, clean_data, int(inv_lead), int(inv_review), z_values[int(inv_service)], float(inv_cover))
                    st.session_state["generated_inventory_df"] = generated_inventory
                    st.success("Inventory recommendations calculated from the generated demand and clearly labelled stock assumptions.")
                except Exception as exc:
                    st.error(f"Inventory calculation could not complete: {type(exc).__name__}: {exc}")
            gi = st.session_state.get("generated_inventory_df", pd.DataFrame())
            if not gi.empty:
                ga, gb, gc = st.columns(3)
                ga.metric("Scenario order quantity", fmt(gi["suggested_order_quantity"].sum()))
                gb.metric("Critical combinations", int((gi["inventory_risk"] == "Critical").sum()))
                gc.metric("Simulated inventory rows", f"{len(gi):,}")
                st.dataframe(gi.sort_values("suggested_order_quantity", ascending=False).head(100), use_container_width=True, hide_index=True)
                download(gi, "generated_run_inventory_recommendations.csv", "⬇️ Download generated-run inventory recommendations")
        st.markdown("---")
    history_runs = st.session_state.get("forecast_run_history", [])
    if history_runs:
        with st.expander("Forecast run history (this browser session)", expanded=False):
            history_df = pd.DataFrame(history_runs)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            download(history_df, "forecast_run_history.csv", "⬇️ Download run history")
            st.caption("This history is session-based. Download it to keep a durable record; Streamlit Cloud's local filesystem is not treated as permanent storage.")
    st.info("The saved forecast is shown below. A generated run appears above only after you request one and the saved model is compatible.")
    if not ff.empty and "forecast_date" in ff and ff["forecast_date"].notna().any():
        st.warning(f"Saved forecast date range: {ff['forecast_date'].min().date()} to {ff['forecast_date'].max().date()}. These are the dates stored in the current project outputs; they are not automatically rolling real-world forecasts.")
    if ff.empty: st.warning("No forecast rows match the current filters.")
    else:
        a,b,c=st.columns(3); a.metric("Forecast total",fmt(total_demand)); b.metric("Forecast records",f"{len(ff):,}"); c.metric("Forecast dates",str(ff["forecast_date"].nunique()))
        daily=ff.groupby("forecast_date",as_index=False)["predicted_sales"].sum()
        chart(px.line(daily,x="forecast_date",y="predicted_sales",markers=True,title="Daily predicted sales"),420)
        fam=ff.groupby("family",as_index=False)["predicted_sales"].sum().sort_values("predicted_sales",ascending=False).head(15)
        chart(px.bar(fam.sort_values("predicted_sales"),x="predicted_sales",y="family",orientation="h",title="Top 15 product families by forecast demand"),500)
        st.dataframe(ff.sort_values(["forecast_date","store_nbr","family"]),use_container_width=True,hide_index=True)
        download(ff,"filtered_demand_forecast.csv")

# -------------------- INVENTORY OPTIMIZER --------------------
elif page == "Inventory Optimizer":
    hero("Inventory Optimizer", "Prioritize replenishment using simulated stock, lead-time demand, safety stock and reorder-point assumptions.")
    note()
    if fi.empty: st.warning("No inventory rows match the filters.")
    else:
        a,b,c=st.columns(3); a.metric("Suggested order quantity",fmt(total_orders)); b.metric("Critical items",critical_count); c.metric("High-risk items",high_count)
        cols=[c for c in ["store_nbr","family","forecast_demand_next_7_days","simulated_current_stock","expected_lead_time_demand","estimated_safety_stock","estimated_reorder_point","estimated_target_stock","suggested_order_quantity","inventory_risk","reorder_recommended"] if c in fi]
        st.dataframe(fi[cols].sort_values("suggested_order_quantity",ascending=False),use_container_width=True,hide_index=True)
        chart(px.box(fi,x="inventory_risk",y="suggested_order_quantity",color="inventory_risk",title="Order quantity distribution by risk level"),380)
        download(fi,"inventory_recommendations_filtered.csv")

# -------------------- SMART ALERTS --------------------
elif page == "Smart Alerts":
    hero("Smart Alerts", "Action-oriented exception monitoring for store–product combinations that may require attention.")
    note()
    if fi.empty: st.warning("No matching items.")
    else:
        critical=fi[fi["inventory_risk"]=="Critical"].copy()
        high=fi[fi["inventory_risk"]=="High"].copy()
        a,b,c=st.columns(3); a.metric("Critical alerts",len(critical)); b.metric("High-risk alerts",len(high)); c.metric("Reorder flagged",int(fi["reorder_recommended"].sum()) if "reorder_recommended" in fi else int((fi["suggested_order_quantity"]>0).sum()))
        alerts=fi[fi["inventory_risk"].isin(["Critical","High"])].copy()
        alerts["Alert reason"] = np.where(alerts["inventory_risk"].eq("Critical"), "Simulated stock is below estimated lead-time demand", "Simulated stock is below estimated reorder point")
        alerts["Suggested action"]=np.where(alerts["inventory_risk"].eq("Critical"),"Urgent review: validate stock assumption and prioritize replenishment","Review soon: check reorder point and planned replenishment")
        if "reorder_recommended" in alerts:
            alerts["Suggested action"]=np.where(alerts["reorder_recommended"],alerts["Suggested action"],"Monitor; no reorder flag in saved recommendation")
        st.markdown("### High-risk product queue")
        st.dataframe(alerts.sort_values(["inventory_risk","suggested_order_quantity"],ascending=[True,False]),use_container_width=True,hide_index=True)
        if alerts.empty: st.success("No critical or high-risk items in the current filter selection.")
        else: download(alerts,"smart_inventory_alerts.csv","⬇️ Export smart alerts")

# -------------------- SALES EXPLORER --------------------
elif page == "Sales Explorer":
    hero("Sales Explorer", "Search, slice, compare and export the saved forecast data.")
    if ff.empty: st.warning("No forecast rows to explore.")
    else:
        x1,x2,x3=st.columns(3)
        date_options=sorted(ff["forecast_date"].dropna().dt.date.unique().tolist())
        chosen_dates=x1.multiselect("Forecast dates",date_options,default=date_options)
        sort_by=x2.selectbox("Rank by",["predicted_sales","store_nbr","family"])
        top_n=x3.slider("Rows to display",10,500,50,step=10)
        search=st.text_input("Search store or product family",placeholder="e.g. 12 or GROCERY")
        view=ff[ff["forecast_date"].dt.date.isin(chosen_dates)].copy()
        if search.strip():
            mask=view.astype(str).apply(lambda col:col.str.contains(search,case=False,na=False)).any(axis=1)
            view=view[mask]
        view=view.sort_values(sort_by,ascending=(sort_by!="predicted_sales")).head(top_n)
        a,b=st.columns(2); a.metric("Rows matching filters",f"{len(view):,}"); b.metric("Displayed forecast sales",fmt(view["predicted_sales"].sum()))
        st.dataframe(view,use_container_width=True,hide_index=True)
        download(view,"sales_explorer_export.csv")

# -------------------- WHAT-IF SIMULATOR --------------------
elif page == "What-if Simulator":
    hero("What-if Simulator", "Explore how inventory recommendations change under alternative operating assumptions.")
    note()
    st.warning("This is a transparent scenario estimate, not a new trained-model forecast. It recalculates order recommendations from saved demand and user-entered assumptions.")
    col1,col2,col3,col4,col5=st.columns(5)
    stock_cover=col1.slider("Assumed stock cover (days)",0.0,14.0,3.0,0.5)
    lead_days=col2.slider("Lead time (days)",1,30,3)
    review_days=col3.slider("Review period (days)",1,28,7)
    service_level=col4.selectbox("Safety-stock service level", [90, 95, 97, 99], index=1)
    pack_size=col5.number_input("Order pack size", min_value=1, max_value=1000, value=1, step=1)
    z_map={90:1.282,95:1.645,97:1.881,99:2.326}
    safety_z=z_map[int(service_level)]
    st.markdown("#### Optional cost-aware scenario")
    cost1, cost2, cost3 = st.columns(3)
    unit_holding_cost = cost1.number_input("Estimated holding cost per unit / period", min_value=0.0, value=0.10, step=0.05, key="whatif_holding_cost")
    fixed_order_cost = cost2.number_input("Estimated fixed cost per order", min_value=0.0, value=25.0, step=5.0, key="whatif_order_cost")
    stockout_penalty = cost3.number_input("Estimated stockout penalty per unit", min_value=0.0, value=5.0, step=0.5, key="whatif_stockout_penalty")
    st.caption("Costs are user-entered scenario assumptions, not measured costs from the historical dataset.")
    sim=fi.copy()
    if not sim.empty:
        daily_col="historical_avg_daily_sales_28d"
        if daily_col in sim:
            sim["scenario_stock"]=sim[daily_col].clip(lower=0)*stock_cover
            sim["scenario_lead_demand"]=sim[daily_col].clip(lower=0)*lead_days
            sim["scenario_safety_stock"]=sim["scenario_lead_demand"]*safety_pct/100
            sim["scenario_reorder_point"]=sim["scenario_lead_demand"]+sim["scenario_safety_stock"]
            sim["scenario_target_stock"]=sim[daily_col].clip(lower=0)*(lead_days+7)+sim["scenario_safety_stock"]
            sim["scenario_order_qty"]=(sim["scenario_target_stock"]-sim["scenario_stock"]).clip(lower=0)
            sim["scenario_risk"]=np.select([sim["scenario_stock"]<sim["scenario_lead_demand"],sim["scenario_stock"]<sim["scenario_reorder_point"]],["Critical","High"],default="Low")
            a,b,c,d,e=st.columns(5)
            a.metric("Scenario order quantity",fmt(sim["scenario_order_qty"].sum()))
            b.metric("Change vs saved orders",fmt(sim["scenario_order_qty"].sum()-sim.get("suggested_order_quantity",pd.Series(0,index=sim.index)).sum()))
            c.metric("Critical scenarios",int((sim["scenario_risk"]=="Critical").sum()))
            d.metric("High scenarios",int((sim["scenario_risk"]=="High").sum()))
            e.metric("Compared combinations",f"{len(sim):,}")
            sim["estimated_holding_cost"] = sim["scenario_order_qty"] * float(unit_holding_cost)
            sim["estimated_ordering_cost"] = np.where(sim["scenario_order_qty"] > 0, float(fixed_order_cost), 0.0)
            sim["estimated_stockout_penalty"] = (sim["scenario_lead_demand"] - sim["scenario_stock"]).clip(lower=0) * float(stockout_penalty)
            sim["estimated_scenario_cost"] = sim[["estimated_holding_cost", "estimated_ordering_cost", "estimated_stockout_penalty"]].sum(axis=1)
            st.metric("Estimated scenario cost (assumption-based)", f"{sim['estimated_scenario_cost'].sum():,.2f}")
            compare=sim.groupby("scenario_risk",as_index=False).size().rename(columns={"scenario_risk":"Risk","size":"Items"})
            chart(px.bar(compare,x="Risk",y="Items",color="Risk",title="Scenario risk distribution"),320)
            showcols=[c for c in ["store_nbr","family","historical_avg_daily_sales_28d","scenario_stock","scenario_reorder_point","scenario_order_qty","scenario_risk","estimated_holding_cost","estimated_ordering_cost","estimated_stockout_penalty","estimated_scenario_cost"] if c in sim]
            st.dataframe(sim[showcols].sort_values("scenario_order_qty",ascending=False),use_container_width=True,hide_index=True)
            download(sim[showcols],"what_if_scenario.csv","⬇️ Export scenario")
        else:
            st.warning("The inventory file does not include historical_avg_daily_sales_28d, so this scenario cannot be calculated.")

# -------------------- MODEL EVALUATION --------------------
elif page == "Model Evaluation":
    hero("Model Evaluation Lab", "Compare validation metrics and inspect evidence behind model selection.")
    if metrics_source == "project_recorded_fallback":
        st.info("Using the project's previously recorded validation results because the evaluation CSV is not present in this deployment. Commit the original evaluation report to use the file directly.")
    if metrics.empty or "Model" not in metrics.columns:
        st.error("No readable saved model metrics were found. Expected reports/model_comparison.csv or reports/validated_model_metrics.csv.")
        st.info("Check the file's headers and confirm that it is committed to GitHub under the reports/ folder. The dashboard will not invent evaluation results.")
    else:
        metric_cols = [c for c in ["MAE", "RMSE", "WAPE_percent", "MAPE_percent"] if c in metrics.columns and metrics[c].notna().any()]
        if not metric_cols:
            st.warning("The metrics file loaded, but its columns were not recognized. Expected model, MAE, RMSE, WAPE_percent and/or MAPE_percent.")
            st.dataframe(metrics, use_container_width=True, hide_index=True)
        else:
            score = metrics[["Model"] + metric_cols].copy()
            for col in metric_cols: score[col] = pd.to_numeric(score[col], errors="coerce")
            st.markdown("### Validation scorecard")
            st.dataframe(score, use_container_width=True, hide_index=True)
            preferred = "WAPE_percent" if "WAPE_percent" in metric_cols else metric_cols[0]
            valid = score.dropna(subset=[preferred]).sort_values(preferred)
            if not valid.empty:
                best = valid.iloc[0]
                a,b,c = st.columns(3)
                a.metric("Best saved model", str(best["Model"]))
                b.metric(f"Lowest {preferred.replace('_percent','').replace('_',' ').upper()}", f"{best[preferred]:.3f}" + ("%" if "percent" in preferred else ""))
                c.metric("Models evaluated", f"{score['Model'].nunique():,}")
                chart(px.bar(valid, x="Model", y=preferred, color="Model", title=f"{preferred.replace('_percent','').replace('_',' ').upper()} comparison — lower is better"), 380)
            error_cols = [c for c in ["MAE", "RMSE"] if c in metric_cols]
            if error_cols:
                err = score.melt(id_vars="Model", value_vars=error_cols, var_name="Metric", value_name="Error")
                chart(px.bar(err, x="Model", y="Error", color="Metric", barmode="group", title="Absolute error comparison — lower is better"), 390)
            st.markdown("### Metric guide")
            st.write("- **MAE:** average absolute prediction error in the target's units.")
            st.write("- **RMSE:** gives larger errors more weight.")
            st.write("- **WAPE:** total absolute error divided by total actual demand; lower is better.")
            st.write("- **MAPE:** average percentage error; interpretation can be unstable when actual values are zero or very small.")
            st.caption("These are saved validation results. Model selection changes the displayed comparison only; it does not retrain models or regenerate forecasts.")
            st.markdown("### Performance by store and product family")
            store_errors = read_csv(REPORTS / "store_error_analysis.csv")
            family_errors = read_csv(REPORTS / "product_family_error_analysis.csv")
            se, fe = st.columns(2)
            with se:
                st.markdown("#### Store-level WAPE")
                if not store_errors.empty and {"store_nbr", "WAPE_percent"}.issubset(store_errors.columns):
                    store_errors["WAPE_percent"] = pd.to_numeric(store_errors["WAPE_percent"], errors="coerce")
                    store_errors = store_errors.dropna(subset=["WAPE_percent"])
                    st.dataframe(store_errors.sort_values("WAPE_percent").head(10), use_container_width=True, hide_index=True)
                    chart(px.bar(store_errors.sort_values("WAPE_percent").head(15), x="store_nbr", y="WAPE_percent", title="Lowest WAPE stores (top 15 by rank)"), 320)
                else:
                    st.info("Store error analysis report is not available in this deployment.")
            with fe:
                st.markdown("#### Product-family WAPE")
                if not family_errors.empty and {"family", "WAPE_percent"}.issubset(family_errors.columns):
                    family_errors["WAPE_percent"] = pd.to_numeric(family_errors["WAPE_percent"], errors="coerce")
                    family_errors = family_errors.dropna(subset=["WAPE_percent"])
                    st.dataframe(family_errors.sort_values("WAPE_percent", ascending=False).head(10), use_container_width=True, hide_index=True)
                    chart(px.bar(family_errors.sort_values("WAPE_percent", ascending=False).head(15), x="WAPE_percent", y="family", orientation="h", title="Highest WAPE product families (top 15)"), 420)
                else:
                    st.info("Product-family error analysis report is not available in this deployment.")

# -------------------- STORE & PRODUCT ANALYTICS --------------------
elif page == "Store & Product Analytics":
    hero("Store & Product Analytics", "Compare demand concentration, replenishment needs, and risk across retail segments.")
    if ff.empty or fi.empty: st.warning("Forecast or inventory data is empty for these filters.")
    else:
        store_d=ff.groupby("store_nbr",as_index=False)["predicted_sales"].sum().rename(columns={"predicted_sales":"forecast_demand_7_days"})
        store_i=fi.groupby("store_nbr",as_index=False).agg(order_quantity=("suggested_order_quantity","sum"),critical_items=("inventory_risk",lambda s:(s=="Critical").sum()),high_risk_items=("inventory_risk",lambda s:(s=="High").sum()))
        sv=store_d.merge(store_i,on="store_nbr",how="left").fillna(0)
        fam_d=ff.groupby("family",as_index=False)["predicted_sales"].sum().rename(columns={"predicted_sales":"forecast_demand_7_days"})
        fam_i=fi.groupby("family",as_index=False).agg(order_quantity=("suggested_order_quantity","sum"),critical_items=("inventory_risk",lambda s:(s=="Critical").sum()),high_risk_items=("inventory_risk",lambda s:(s=="High").sum()))
        fv=fam_d.merge(fam_i,on="family",how="left").fillna(0)
        t1,t2=st.tabs(["🏬 Store performance","🛍️ Product-family performance"])
        with t1:
            chart(px.bar(sv.sort_values("forecast_demand_7_days"),x="store_nbr",y="forecast_demand_7_days",title="7-day demand by store"),420)
            chart(px.scatter(sv,x="forecast_demand_7_days",y="order_quantity",size="critical_items",hover_name="store_nbr",title="Demand vs suggested orders by store"),400)
            st.dataframe(sv.sort_values("forecast_demand_7_days",ascending=False),use_container_width=True,hide_index=True)
            download(sv,"store_performance_analytics.csv")
        with t2:
            chart(px.bar(fv.sort_values("forecast_demand_7_days"),x="forecast_demand_7_days",y="family",orientation="h",color="critical_items",title="Demand and critical risk by product family"),600)
            chart(px.treemap(fv,path=["family"],values="forecast_demand_7_days",color="order_quantity",title="Product-family demand treemap coloured by suggested orders"),500)
            st.dataframe(fv.sort_values("forecast_demand_7_days",ascending=False),use_container_width=True,hide_index=True)
            download(fv,"product_family_analytics.csv")

# -------------------- BUSINESS INTELLIGENCE --------------------
elif page == "Business Intelligence":
    hero("Business Intelligence", "Translate model outputs into a concise operational picture for retail decision-makers.")
    note()
    a,b,c,d=st.columns(4)
    a.metric("Forecast demand",fmt(total_demand)); b.metric("Proposed orders",fmt(total_orders)); c.metric("Critical combinations",critical_count); d.metric("High-risk combinations",high_count)
    if not ff.empty:
        daily=ff.groupby("forecast_date",as_index=False)["predicted_sales"].sum()
        daily["share_pct"]=100*daily["predicted_sales"]/max(daily["predicted_sales"].sum(),1)
        chart(px.line(daily,x="forecast_date",y="predicted_sales",markers=True,title="Daily demand outlook"),360)
        st.markdown("### Forecast contribution by product family")
        fam=ff.groupby("family",as_index=False)["predicted_sales"].sum().sort_values("predicted_sales",ascending=False)
        fam["demand_share_pct"]=100*fam["predicted_sales"]/max(fam["predicted_sales"].sum(),1)
        chart(px.bar(fam.head(12).sort_values("predicted_sales"),x="predicted_sales",y="family",orientation="h",title="Top demand contributors"),430)
        st.dataframe(fam,use_container_width=True,hide_index=True)
    # Explainability: global feature importance is shown only if the optional library and saved model support it.
    st.markdown("### Forecast explainability")
    model_path = BASE / "models" / "xgboost_model.joblib"
    if model_path.exists():
        try:
            import joblib
            explain_model = joblib.load(model_path)
            importance = getattr(explain_model, "feature_importances_", None)
            feature_names = model_config.get("feature_columns", [])
            if importance is not None and len(importance) == len(feature_names):
                imp = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values("Importance", ascending=False).head(12)
                chart(px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Saved XGBoost feature importance"), 420)
                st.caption("This is model feature importance, not a causal explanation. SHAP can be added when the saved artifact loads compatibly in the deployment environment.")
            else:
                st.info("The saved model does not expose compatible feature_importances_. No explanation was fabricated.")
        except Exception as exc:
            st.info(f"Feature-importance view is unavailable because the saved model could not be loaded in this runtime ({type(exc).__name__}). Existing forecasts and metrics remain available.")
    else:
        st.info("Add models/xgboost_model.joblib to enable model feature-importance inspection.")
    st.markdown("### Decision cues")
    st.write(f"- **Inventory attention:** {critical_count} critical and {high_count} high-risk combinations in the current filtered view.")
    st.write(f"- **Replenishment workload:** suggested order quantity sums to {fmt(total_orders)} in the current filtered view.")
    st.write("- Treat these as decision-support signals, not purchase orders: inventory inputs are simulated.")
    briefing = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "forecast_demand_selected_view": float(total_demand),
        "suggested_order_quantity_selected_view": float(total_orders),
        "critical_items_selected_view": int(critical_count),
        "high_risk_items_selected_view": int(high_count),
        "stores_in_view": int(fi["store_nbr"].nunique()) if "store_nbr" in fi else 0,
        "product_families_in_view": int(fi["family"].nunique()) if "family" in fi else 0,
        "inventory_values_are_simulated": True,
        "model_metrics_source": metrics_source,
    }
    st.download_button("⬇️ Download executive briefing (JSON)", json.dumps(briefing, indent=2).encode("utf-8"), file_name="executive_business_briefing.json", mime="application/json")

# -------------------- DATA QUALITY & EXPLORER --------------------
elif page == "Data Quality & Explorer":
    hero("Data Quality & Explorer", "Inspect available datasets, completeness, distributions, and exportable records.")
    dataset_map={"Demand forecasts":forecast,"Inventory recommendations":inventory,"Store inventory summary":stores,"Inventory risk summary":risk_summary,"Model evaluation metrics":metrics,"Cleaned historical sales":clean_data,"Modeling features":features_data}
    label=st.selectbox("Dataset to inspect",list(dataset_map))
    data=dataset_map[label]
    if data.empty:
        st.warning(f"{label} is not present in this deployment. Add the relevant file to the repository to enable this view.")
    else:
        a,b,c=st.columns(3); a.metric("Rows",f"{len(data):,}"); b.metric("Columns",len(data.columns)); c.metric("Duplicate rows",f"{int(data.duplicated().sum()):,}")
        missing=data.isna().sum().rename("Missing values").reset_index().rename(columns={"index":"Column"})
        missing["Missing %"]=100*missing["Missing values"]/max(len(data),1)
        t1,t2,t3=st.tabs(["Preview & search","Column quality","Numeric profile"])
        with t1:
            search=st.text_input("Search all columns",placeholder="Enter a store, product family, date, or risk")
            view=data.copy()
            if search.strip():
                mask=view.astype(str).apply(lambda col:col.str.contains(search,case=False,na=False)).any(axis=1)
                view=view[mask]
            st.caption(f"Showing {min(len(view),1000):,} of {len(view):,} matching rows")
            st.dataframe(view.head(1000),use_container_width=True,hide_index=True)
            download(view,f"{label.lower().replace(' ','_')}.csv")
        with t2:
            st.dataframe(missing,use_container_width=True,hide_index=True)
            if missing["Missing values"].sum()==0: st.success("No missing values detected in this loaded table.")
            else: st.warning("Missing values are present; inspect before modeling.")
        with t3:
            numeric=data.select_dtypes(include=np.number)
            if numeric.empty: st.info("No numeric columns in this dataset.")
            else: st.dataframe(numeric.describe().T,use_container_width=True)
            if not numeric.empty:
                col=st.selectbox("Numeric distribution",numeric.columns)
                chart(px.histogram(data,x=col,nbins=40,title=f"Distribution of {col}"),350)
    st.markdown("### Available research artifacts")
    artifact_names=["model_comparison.csv","validated_model_metrics.csv","part3_evaluation_summary.json","product_family_error_analysis.csv","validation_error_analysis.csv","eda_summary.json"]
    found=[f"{n}: {'available' if (REPORTS/n).exists() else 'not included in this deployment'}" for n in artifact_names]
    st.write("\n".join(f"- {x}" for x in found))


st.markdown("---")
st.markdown('<div class="small-muted">AI-Powered Demand Forecasting & Inventory Optimization • Retail decision support • Inventory estimates are assumption-based and labelled as simulated.</div>',unsafe_allow_html=True)
