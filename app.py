import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hoth Intelligence Hub", layout="wide")
st.title("🚀 Hoth Industries: Supplier Intelligence Hub")
st.markdown("---")

# -------------------------------
# Helpers
# -------------------------------
def read_csv_flexible(candidates):
    """Try multiple filenames (useful for Streamlit Cloud vs local copies)."""
    last_err = None
    for f in candidates:
        try:
            return pd.read_csv(f)
        except Exception as e:
            last_err = e
    raise last_err

def safe_to_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# -------------------------------
# Load + clean data
# -------------------------------
@st.cache_data
def load_data():
    orders = read_csv_flexible([
        "Copy of supplier_orders.csv",
        "Copy of supplier_orders (1).csv",
        "supplier_orders.csv"
    ])

    quality = read_csv_flexible([
        "Copy of quality_inspections.csv",
        "Copy of quality_inspections (1).csv",
        "quality_inspections.csv"
    ])

    rfqs = read_csv_flexible([
        "Copy of rfq_responses.csv",
        "Copy of rfq_responses (1).csv",
        "rfq_responses.csv"
    ])

    # Normalize supplier names (extend as needed)
    name_map = {
        "APEX MFG": "Apex Manufacturing Inc",
        "Apex Mfg": "Apex Manufacturing Inc",
        "Apex Manufacturing": "Apex Manufacturing Inc",
    }

    if "supplier_name" in orders.columns:
        orders["supplier_name"] = orders["supplier_name"].replace(name_map)

    if "supplier_name" in rfqs.columns:
        rfqs["supplier_name"] = rfqs["supplier_name"].replace(name_map)

    # Dates
    orders = safe_to_datetime(orders, "order_date")
    orders = safe_to_datetime(orders, "promised_date")
    orders = safe_to_datetime(orders, "actual_delivery_date")
    quality = safe_to_datetime(quality, "inspection_date")
    rfqs = safe_to_datetime(rfqs, "quote_date")

    return orders, quality, rfqs

try:
    orders, quality, rfqs = load_data()
    st.success("✅ Data loaded & supplier names normalized")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -------------------------------
# Build KPIs
# -------------------------------

# 1) Spend (orders -> po_amount)
if "po_amount" not in orders.columns:
    st.error("Expected 'po_amount' column in orders but didn't find it.")
    st.stop()

spend = orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index()
spend.columns = ["supplier_name", "total_spend"]

# 2) On-time rate (actual_delivery_date <= promised_date)
required_cols = {"promised_date", "actual_delivery_date"}
if not required_cols.issubset(set(orders.columns)):
    st.error("Expected 'promised_date' and 'actual_delivery_date' columns in orders.")
    st.stop()

orders_kpi = orders.copy()
orders_kpi["on_time"] = (orders_kpi["actual_delivery_date"] <= orders_kpi["promised_date"]).astype(float)
on_time = orders_kpi.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"] * 100).round(1)
on_time = on_time.drop(columns=["on_time"])

# 3) Defect rate (quality -> join to orders by order_id to get supplier_name)
if "order_id" not in quality.columns or "order_id" not in orders.columns:
    st.error("Expected 'order_id' in both orders and quality datasets.")
    st.stop()

q = quality.merge(
    orders[["order_id", "supplier_name"]],
    on="order_id",
    how="left"
)

# defect rate = parts_rejected / parts_inspected
if {"parts_rejected", "parts_inspected"}.issubset(set(q.columns)):
    q["defect_rate"] = (q["parts_rejected"] / q["parts_inspected"]).replace([pd.NA, float("inf")], 0)
else:
    # fallback if columns differ
    q["defect_rate"] = 0.0

defects = q.groupby("supplier_name", dropna=False)["defect_rate"].mean().reset_index()
defects["defect_rate"] = (defects["defect_rate"] * 100).round(1)

# 4) Avg RFQ price
if "quoted_price" not in rfqs.columns:
    st.error("Expected 'quoted_price' column in RFQs but didn't find it.")
    st.stop()

avg_price = rfqs.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index()
avg_price.columns = ["supplier_name", "avg_price"]
avg_price["avg_price"] = avg_price["avg_price"].round(2)

# -------------------------------
# Unified supplier table
# -------------------------------
supplier_master = (
    spend.merge(on_time, on="supplier_name", how="left")
         .merge(defects, on="supplier_name", how="left")
         .merge(avg_price, on="supplier_name", how="left")
)

supplier_master = supplier_master.fillna({
    "on_time_rate": 0.0,
    "defect_rate": 0.0,
    "avg_price": 0.0
})

# -------------------------------
# Simple performance score
# (higher is better)
# -------------------------------
# Normalize avg_price to a 0-100 scale (lower price = higher score)
if supplier_master["avg_price"].max() > 0:
    supplier_master["price_score"] = 100 * (1 - (supplier_master["avg_price"] / supplier_master["avg_price"].max()))
else:
    supplier_master["price_score"] = 0.0

supplier_master["performance_score"] = (
    (supplier_master["on_time_rate"] * 0.45) +
    ((100 - supplier_master["defect_rate"]) * 0.35) +
    (supplier_master["price_score"] * 0.20)
).round(1)

# Risk flags
def risk_flag(row):
    if row["defect_rate"] >= 8:
        return "🔴 Quality Risk"
    if row["on_time_rate"] <= 85:
        return "🟠 Delivery Risk"
    if row["price_score"] <= 40:
        return "🟡 Cost Risk"
    return "🟢 Strategic"

supplier_master["risk_flag"] = supplier_master.apply(risk_flag, axis=1)

supplier_master = supplier_master.sort_values("performance_score", ascending=False)

# -------------------------------
# Display
# -------------------------------
st.subheader("🏭 Unified Supplier Intelligence View")
st.dataframe(supplier_master, use_container_width=True)

st.markdown("### 🔴 Highest Risk Suppliers")
risk_tbl = supplier_master[supplier_master["risk_flag"].str.contains("🔴|🟠")]
st.dataframe(risk_tbl, use_container_width=True)

st.markdown("### 🟢 Top Performing Suppliers")
st.dataframe(supplier_master.head(5), use_container_width=True)

with st.expander("Debug: show column names"):
    st.write("Orders columns:", list(orders.columns))
    st.write("Quality columns:", list(quality.columns))
    st.write("RFQ columns:", list(rfqs.columns))
