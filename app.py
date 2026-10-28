from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

THEMES = {
    "Midnight Blue": {"bg":"#0B1220","panel":"#111C30","sidebar":"#0D1728","text":"#F1F5F9","muted":"#A8B7CC","accent":"#38BDF8","accent2":"#818CF8","border":"#26364D","plot":"plotly_dark"},
    "Emerald Executive": {"bg":"#071A17","panel":"#102B25","sidebar":"#09231E","text":"#ECFDF5","muted":"#A7C8BC","accent":"#34D399","accent2":"#A3E635","border":"#245247","plot":"plotly_dark"},
    "Royal Purple": {"bg":"#140F24","panel":"#211936","sidebar":"#19132D","text":"#F5F3FF","muted":"#C4B5FD","accent":"#A78BFA","accent2":"#F0ABFC","border":"#44345F","plot":"plotly_dark"},
    "Light Professional": {"bg":"#F3F6FB","panel":"#FFFFFF","sidebar":"#E8EEF7","text":"#172033","muted":"#536179","accent":"#2563EB","accent2":"#0D9488","border":"#D3DDEB","plot":"plotly_white"},
    "Graphite & Gold": {"bg":"#141414","panel":"#202020","sidebar":"#191919","text":"#FAF7EF","muted":"#C7C0AE","accent":"#F5C451","accent2":"#E9A23B","border":"#454039","plot":"plotly_dark"},
}
with st.sidebar:
    st.markdown("## 📦 Demand Intelligence")
    st.caption("Retail forecasting • inventory decision support")
    theme_name = st.selectbox("🎨 Dashboard theme", list(THEMES), index=0)
T = THEMES[theme_name]
st.markdown(f"""
<style>
:root {{ color-scheme: {'light' if theme_name == 'Light Professional' else 'dark'}; }}
.stApp {{background:{T['bg']}; color:{T['text']};}}
[data-testid="stSidebar"] {{background:{T['sidebar']}; border-right:1px solid {T['border']};}}
[data-testid="stSidebar"] * {{color:{T['text']} !important;}}
[data-testid="stMetric"] {{background:linear-gradient(135deg,{T['panel']},{T['bg']}); border:1px solid {T['border']}; padding:17px 18px; border-radius:15px;}}
[data-testid="stMetricLabel"] {{color:{T['muted']} !important;}}
[data-testid="stMetricValue"] {{color:{T['text']} !important;}}
h1,h2,h3 {{color:{T['text']} !important; letter-spacing:-0.025em;}}
p, label, .stCaption {{color:{T['muted']};}}
div.stButton > button, div.stDownloadButton > button {{border-radius:10px; border:1px solid {T['border']}; font-weight:600;}}
div.stButton > button[kind="primary"] {{background:{T['accent']}; color:{T['bg']}; border:0;}}
[data-testid="stDataFrame"] {{border:1px solid {T['border']}; border-radius:12px;}}
.hero {{background:linear-gradient(115deg,{T['panel']},{T['bg']}); border:1px solid {T['border']}; padding:26px 28px; border-radius:18px; margin:5px 0 20px;}}
.hero h1 {{font-size:30px; margin:0 0 8px;}}
.hero p {{font-size:15px; margin:0; color:{T['muted']};}}
.callout {{background:{T['panel']}; border:1px solid {T['border']}; border-left:4px solid {T['accent']}; padding:13px 16px; border-radius:10px; margin:8px 0 16px; color:{T['text']};}}
.small-muted {{color:{T['muted']}; font-size:12px;}}
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

forecast = read_csv(FORECASTS / "demand_forecast_7_days.csv")
inventory = read_csv(FORECASTS / "inventory_optimization_recommendations.csv")
stores = read_csv(FORECASTS / "store_inventory_summary.csv")
risk_summary = read_csv(REPORTS / "inventory_risk_summary.csv")
summary = read_json(REPORTS / "part4_inventory_summary.json")
metrics = read_csv(REPORTS / "model_comparison.csv")
if metrics.empty: metrics = read_csv(REPORTS / "validated_model_metrics.csv")
clean_data = read_csv(DATASET / "sales_clean.csv")
features_data = read_csv(DATASET / "modeling_features.csv")
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
    st.markdown("---")
    st.markdown("### 🔎 Global filters")
    selected_stores = st.multiselect("Stores", all_stores, default=all_stores, key="stores_filter")
    selected_families = st.multiselect("Product families", all_families, default=all_families, key="families_filter")
    selected_risks = st.multiselect("Inventory risk", all_risks, default=all_risks, key="risk_filter")
    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.markdown("---")
    st.caption("MCA research prototype • v2 dashboard")

pages = [
    "Executive Overview", "Demand Forecasting", "Inventory Optimizer",
    "Smart Alerts", "Sales Explorer", "What-if Simulator",
    "Model Evaluation", "Store & Product Analytics",
    "Business Intelligence", "Data Quality & Explorer", "Research & Limitations"
]
with st.sidebar:
    page = st.radio("NAVIGATION", pages, index=0)

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
    st.caption("Select a saved model to inspect its validation score. This selector changes the evaluation view; it does not silently swap/retrain the model behind the saved forecast CSV.")
    model_names = metrics["Model"].astype(str).tolist() if not metrics.empty and "Model" in metrics else ["XGBoost","Random Forest","Seasonal Naive"]
    chosen_model = st.selectbox("Select model", model_names, key="overview_model")
    if st.button("▶ Apply model selection", type="primary", key="apply_overview_model"):
        st.session_state["active_model_view"] = chosen_model
    active = st.session_state.get("active_model_view", chosen_model)
    if not metrics.empty and "Model" in metrics:
        row = metrics[metrics["Model"].astype(str) == active]
        if not row.empty:
            r=row.iloc[0]
            a,b,c=st.columns(3)
            a.metric("Selected model", active)
            if "MAE" in row: b.metric("MAE", f"{float(r['MAE']):.2f}")
            if "RMSE" in row: c.metric("RMSE", f"{float(r['RMSE']):.2f}")
            if "WAPE_percent" in row: st.caption(f"WAPE: {float(r['WAPE_percent']):.2f}%")
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
    hero("Demand Forecasting", "Explore seven-day forecast outputs across selected stores and product families.")
    st.info("The forecast shown is the saved output produced by the project's forecasting pipeline.")
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
    col1,col2,col3=st.columns(3)
    stock_cover=col1.slider("Assumed current stock cover (days)",0.0,14.0,3.0,0.5)
    lead_days=col2.slider("Assumed supplier lead time (days)",1,14,3)
    safety_pct=col3.slider("Safety-stock buffer (%)",0,100,20,5)
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
            a,b,c,d=st.columns(4)
            a.metric("Scenario order quantity",fmt(sim["scenario_order_qty"].sum()))
            b.metric("Critical scenarios",int((sim["scenario_risk"]=="Critical").sum()))
            c.metric("High scenarios",int((sim["scenario_risk"]=="High").sum()))
            d.metric("Compared combinations",f"{len(sim):,}")
            compare=sim.groupby("scenario_risk",as_index=False).size().rename(columns={"scenario_risk":"Risk","size":"Items"})
            chart(px.bar(compare,x="Risk",y="Items",color="Risk",title="Scenario risk distribution"),320)
            showcols=[c for c in ["store_nbr","family","historical_avg_daily_sales_28d","scenario_stock","scenario_reorder_point","scenario_order_qty","scenario_risk"] if c in sim]
            st.dataframe(sim[showcols].sort_values("scenario_order_qty",ascending=False),use_container_width=True,hide_index=True)
            download(sim[showcols],"what_if_scenario.csv","⬇️ Export scenario")
        else:
            st.warning("The inventory file does not include historical_avg_daily_sales_28d, so this scenario cannot be calculated.")

# -------------------- MODEL EVALUATION --------------------
elif page == "Model Evaluation":
    hero("Model Evaluation Lab", "Compare saved validation metrics and inspect model-selection evidence.")
    if metrics.empty:
        st.error("Model metrics file is missing. Expected reports/model_comparison.csv or reports/validated_model_metrics.csv.")
        st.caption("No evaluation metrics are fabricated when the report file is absent.")
    else:
        st.markdown("### Validation scorecard")
        st.dataframe(metrics,use_container_width=True,hide_index=True)
        if {"Model","WAPE_percent"}.issubset(metrics.columns):
            best=metrics.loc[pd.to_numeric(metrics["WAPE_percent"],errors="coerce").idxmin()]
            a,b,c=st.columns(3)
            a.metric("Lowest-WAPE model",str(best["Model"]))
            b.metric("Best WAPE",f"{float(best['WAPE_percent']):.2f}%")
            c.metric("Models compared",len(metrics))
            chart(px.bar(metrics,x="Model",y="WAPE_percent",color="Model",title="WAPE comparison — lower is better"),370)
        if {"Model","MAE","RMSE"}.issubset(metrics.columns):
            melted=metrics.melt(id_vars="Model",value_vars=["MAE","RMSE"],var_name="Metric",value_name="Error")
            chart(px.bar(melted,x="Model",y="Error",color="Metric",barmode="group",title="MAE and RMSE comparison — lower is better"),390)
        st.markdown("### What the metrics mean")
        st.write("- **MAE:** average absolute prediction error in the target's units.")
        st.write("- **RMSE:** penalizes large errors more strongly than MAE.")
        st.write("- **WAPE:** total absolute error divided by total actual demand; lower is better.")
        st.markdown("### Validation notes")
        st.write("Saved project evaluation uses a 28-day validation window (2014-08-06 to 2014-09-02) and rolling one-day-ahead evaluation with actual previous-day lags, as documented in the project.")
        st.caption("These are saved evaluation results. Selecting a model here does not retrain it or regenerate forecast files.")

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
    st.markdown("### Decision cues")
    st.write(f"- **Inventory attention:** {critical_count} critical and {high_count} high-risk combinations in the current filtered view.")
    st.write(f"- **Replenishment workload:** suggested order quantity sums to {fmt(total_orders)} in the current filtered view.")
    st.write("- Treat these as decision-support signals, not purchase orders: inventory inputs are simulated.")

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

# -------------------- RESEARCH & LIMITATIONS --------------------
elif page == "Research & Limitations":
    hero("Research Methodology & Limitations", "An evidence-based summary of the current prototype, its evaluation, and the boundaries of its conclusions.")
    st.markdown("### Research workflow")
    steps=[("1. Data preparation","Clean historical store-family sales, parse dates, and prepare lag/rolling/calendar features."),("2. Model training","Compare tree-based models with a seasonal-naive baseline using the saved training pipeline."),("3. Validation","Evaluate on a held-out 28-day period using rolling one-day-ahead predictions."),("4. Forecasting","Generate a seven-day store–product-family forecast from the saved pipeline outputs."),("5. Inventory decision support","Estimate stock cover, reorder point, safety stock and suggested order quantities using stated assumptions.")]
    for title,desc in steps: st.markdown(f"**{title}**  \n{desc}")
    st.markdown("### Saved model evidence")
    if not metrics.empty: st.dataframe(metrics,use_container_width=True,hide_index=True)
    else: st.info("Model evaluation file not present.")
    st.markdown("### Important limitations")
    limitations=[
        "The source dataset is historical store sales; it does not provide verified on-hand inventory, stockout events, or supplier lead times.",
        "Inventory, stock cover, safety stock, reorder points, and suggested orders are simulated estimates, not real operational inventory records.",
        "Future promotions are assumed to be zero; actual promotional activity could materially change demand.",
        "Missing dates filled with zero sales may conflate a true zero-sales day with missing observations.",
        "The forecast horizon and evaluation period are limited; performance can vary by store, product family, season, and external events.",
        "The dashboard reads saved forecast outputs. Changing the model selector does not automatically retrain a model or regenerate predictions."
    ]
    for item in limitations: st.markdown(f"- {item}")
    st.markdown("### Configuration used for inventory assumptions")
    if config: st.json(config)
    else: st.caption("Inventory configuration JSON is not included in this deployment.")
    if summary: 
        with st.expander("View saved forecast summary JSON"): st.json(summary)
    st.markdown("### Viva-ready interpretation")
    st.write("This prototype demonstrates a retail demand forecasting workflow and a transparent inventory-planning simulation. It supports exploratory decisions and model comparison, but it is not a production replenishment system until real inventory, supplier, promotion, and stockout data are integrated.")

st.markdown("---")
st.markdown('<div class="small-muted">AI-Powered Demand Forecasting & Inventory Optimization • MCA research prototype • Values are derived from saved project artifacts; simulated inventory is labelled explicitly.</div>',unsafe_allow_html=True)
