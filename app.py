import re
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Hoth Intelligence Hub", layout="wide")
st.title("🚀 Hoth Industries: Supplier Intelligence Hub")
st.markdown("---")

TOP_N = 10

# =========================================================
# ENTITY RESOLUTION
# =========================================================
LEGAL_SUFFIXES = {
    "inc","incorporated","llc","ltd","limited",
    "corp","corporation","co","company"
}

def normalize_supplier_key(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split(" ")
    while parts and parts[-1] in LEGAL_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts).strip()

def apply_entity_resolution(df, col, manual_map=None):
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    out["_k"] = out[col].apply(normalize_supplier_key)

    if manual_map:
        out["_k"] = out["_k"].replace(manual_map)

    def pick_longest(vals):
        vals = list(set(vals))
        vals.sort(key=lambda x: (-len(x), x))
        return vals[0]

    canonical = out.groupby("_k")[col].agg(pick_longest).to_dict()
    out[col] = out["_k"].map(canonical).fillna(out[col])
    return out.drop(columns="_k")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load():
    orders = pd.read_csv("Copy of supplier_orders.csv")
    quality = pd.read_csv("Copy of quality_inspections.csv")
    rfq = pd.read_csv("Copy of rfq_responses.csv")

    manual_map = {"apex mfg":"apex manufacturing"}
    orders = apply_entity_resolution(orders,"supplier_name",manual_map)
    rfq = apply_entity_resolution(rfq,"supplier_name",manual_map)
    quality = apply_entity_resolution(quality,"supplier_name",manual_map)

    return orders,quality,rfq

orders, quality, rfq = load()
st.success("Data loaded & normalized")

# =========================================================
# KPIs
# =========================================================
spend = orders.groupby("supplier_name")["po_amount"].sum().reset_index(name="total_spend")

orders["on_time"] = (pd.to_datetime(orders["actual_delivery_date"]) <= 
                     pd.to_datetime(orders["promised_date"])).astype(int)
on_time = orders.groupby("supplier_name")["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"]*100).round(1)
on_time = on_time.drop(columns="on_time")

q = quality.merge(orders[["order_id","supplier_name"]],on="order_id",how="left")
q["defect_rate"] = q["parts_rejected"]/q["parts_inspected"]
defects = q.groupby("supplier_name")["defect_rate"].mean().reset_index()
defects["defect_rate"] = (defects["defect_rate"]*100).round(1)

avg_price = rfq.groupby("supplier_name")["quoted_price"].mean().reset_index(name="avg_price")

supplier = spend.merge(on_time,on="supplier_name",how="left")\
                .merge(defects,on="supplier_name",how="left")\
                .merge(avg_price,on="supplier_name",how="left")\
                .fillna(0)

# =========================================================
# SCORING
# =========================================================
max_price = supplier["avg_price"].replace(0,pd.NA).max()
supplier["price_score"] = 100*(1-(supplier["avg_price"]/max_price))
supplier["price_score"] = supplier["price_score"].fillna(0).clip(0,100)

supplier["performance_score"] = (
    supplier["on_time_rate"]*.45 +
    (100-supplier["defect_rate"])*.35 +
    supplier["price_score"]*.20
).round(1)

def flag(r):
    if r.defect_rate>=8: return "🔴 Quality Risk"
    if r.on_time_rate<=85: return "🟠 Delivery Risk"
    if r.price_score<=40: return "🟡 Cost Risk"
    return "🟢 Strategic"

supplier["risk_flag"] = supplier.apply(flag,axis=1)
supplier = supplier.sort_values("performance_score",ascending=False)

# =========================================================
# SEARCH
# =========================================================
st.subheader("🔎 Search Suppliers")
qtxt = st.text_input("Search supplier")

def filt(df):
    if not qtxt: return df
    return df[df.supplier_name.str.lower().str.contains(qtxt.lower())]

# =========================================================
# MAIN TABLES
# =========================================================
st.subheader("🏭 Unified Supplier View")
st.dataframe(filt(supplier).head(TOP_N),use_container_width=True)

st.subheader("🔴 Highest Risk Suppliers")
sev = {"🔴 Quality Risk":0,"🟠 Delivery Risk":1,"🟡 Cost Risk":2,"🟢 Strategic":3}
risk = supplier.copy()
risk["_s"] = risk.risk_flag.map(sev)
risk = risk.sort_values(["_s","performance_score"])
st.dataframe(filt(risk).head(TOP_N),use_container_width=True)

st.subheader("🟢 Top Performing Suppliers")
st.dataframe(filt(supplier.sort_values("performance_score",ascending=False)).head(TOP_N),
             use_container_width=True)

# =========================================================
# 💰 CONSOLIDATION ENGINE (NEW)
# =========================================================
st.markdown("---")
st.header("💰 Supplier Consolidation Opportunities")

best_price = supplier[supplier.avg_price>0].avg_price.min()
supplier["price_delta"] = supplier.avg_price - best_price
supplier["est_units"] = supplier.total_spend / supplier.avg_price.replace(0,pd.NA)
supplier["savings"] = (supplier.price_delta * supplier.est_units).fillna(0)

total_savings = supplier["savings"].sum()
st.metric("Estimated Annual Savings via Consolidation", f"${total_savings:,.0f}")

st.dataframe(
    supplier.sort_values("savings",ascending=False)[
        ["supplier_name","total_spend","avg_price","savings","risk_flag"]
    ],
    use_container_width=True
)

# =========================================================
# 📊 EXECUTIVE BUBBLE CHART (NEW)
# =========================================================
st.markdown("---")
st.header("📊 Supplier Risk vs Spend")

chart = alt.Chart(supplier).mark_circle(size=200).encode(
    x="performance_score",
    y="total_spend",
    size="total_spend",
    color="risk_flag",
    tooltip=["supplier_name","total_spend","performance_score","risk_flag"]
).interactive()

st.altair_chart(chart,use_container_width=True)

st.caption("Top-right = high spend + high risk → immediate executive focus")
