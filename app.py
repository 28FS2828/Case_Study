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
        "supplier_orders.csv",
    ])

    quality = read_csv_flexible([
        "Copy of quality_inspections.csv",
        "Copy of quality_inspections (1).csv",
        "quality_inspections.csv",
    ])

    rfqs = read_csv_flexible([
        "Copy of rfq_responses.csv",
        "Copy of rfq_responses (1).csv",
        "rfq_responses.csv",
    ])

    # --- Normalize supplier names (extend as needed) ---
    canonical_apex = "Apex Manufacturing Inc"
    name_map = {
        "APEX MFG": canonical_apex,
        "Apex Mfg": canonical_apex,
        "Apex Manufacturing": canonical_apex,
        "APEX Manufacturing": canonical_apex,
        "APEX Manufacturing Inc": canonical_apex,
        "Apex Manufacturing Inc": canonical_apex,
    }

    if "supplier_name" in orders.columns:
        orders["supplier_name"] = orders["supplier_name"].replace(name_map).astype(str).str.strip()

    if "supplier_name" in rfqs.columns:
        rfqs["supplier_name"] = rfqs["supplier_name"].replace(name_map).astype(str).str.strip()

    if "supplier_name" in quality.columns:
        quality["supplier_name"] = quality["supplier_name"].replace(name_map).astype(str).str.strip()

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
# KPI CALCULATIONS
# -------------------------------

# Spend
if "po_amount" not in orders.columns:
    st.error("Expected 'po_amount' column in orders but didn't find it.")
    st.stop()

if "supplier_name" not in orders.columns:
    st.error("Expected 'supplier_name' column in orders but didn't find it.")
    st.stop()

spend = orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index()
spend.columns = ["supplier_name", "total_spend"]

# On-time rate from dates
required_cols = {"promised_date", "actual_delivery_date"}
if not required_cols.issubset(set(orders.columns)):
    st.error("Expected 'promised_date' and 'actual_delivery_date' columns in orders.")
    st.stop()

orders_kpi = orders.copy()
orders_kpi["on_time"] = (orders_kpi["actual_delivery_date"] <= orders_kpi["promised_date"]).astype(float)
on_time = orders_kpi.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"] * 100).round(1)
on_time = on_time.drop(columns=["on_time"])

# Defect rate: join quality -> orders by order_id (quality may not have supplier_name)
if "order_id" not in quality.columns or "order_id" not in orders.columns:
    st.error("Expected 'order_id' in both orders and quality datasets.")
    st.stop()

q = quality.merge(
    orders[["order_id", "supplier_name"]],
    on="order_id",
    how="left",
)

# defect rate = parts_rejected / parts_inspected (fallback to 0 if not present)
if {"parts_rejected", "parts_inspected"}.issubset(set(q.columns)):
    q["defect_rate"] = (q["parts_rejected"] / q["parts_inspected"]).replace([pd.NA, float("inf")], 0)
else:
    q["defect_rate"] = 0.0

defects = q.groupby("supplier_name", dropna=False)["defect_rate"].mean().reset_index()
defects["defect_rate"] = (defects["defect_rate"] * 100).round(1)

# Avg RFQ price
if "supplier_name" not in rfqs.columns:
    st.error("Expected 'supplier_name' column in RFQs but didn't find it.")
    st.stop()

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
    "avg_price": 0.0,
})

# -------------------------------
# Simple performance score
# -------------------------------
# price_score: lower avg_price -> higher score (0-100)
max_price = supplier_master["avg_price"].replace(0, pd.NA).max()
if pd.notna(max_price) and max_price > 0:
    supplier_master["price_score"] = 100 * (1 - (supplier_master["avg_price"] / max_price))
    supplier_master["price_score"] = supplier_master["price_score"].fillna(0).clip(0, 100)
else:
    supplier_master["price_score"] = 0.0

supplier_master["performance_score"] = (
    (supplier_master["on_time_rate"] * 0.45) +
    ((100 - supplier_master["defect_rate"]) * 0.35) +
    (supplier_master["price_score"] * 0.20)
).round(1)

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
# Display: Unified view + slices
# -------------------------------
st.subheader("🏭 Unified Supplier Intelligence View")
st.dataframe(supplier_master, use_container_width=True)

st.markdown("### 🔴 Highest Risk Suppliers")
risk_tbl = supplier_master[supplier_master["risk_flag"].str.contains("🔴|🟠")]
st.dataframe(risk_tbl, use_container_width=True)

st.markdown("### 🟢 Top Performing Suppliers")
st.dataframe(supplier_master.head(5), use_container_width=True)

# -------------------------------
# 💰 Financial Impact Estimation (prototype-level)
# -------------------------------
st.markdown("---")
st.header("💰 Estimated Financial Impact (Prototype)")

# Use lowest non-zero avg RFQ as benchmark
nonzero_prices = supplier_master["avg_price"].replace(0, pd.NA).dropna()
lowest_price = nonzero_prices.min() if len(nonzero_prices) else pd.NA

# Estimate volume from spend/avg_price (only where avg_price > 0)
supplier_master["est_units"] = 0.0
mask_price = supplier_master["avg_price"] > 0
supplier_master.loc[mask_price, "est_units"] = (
    supplier_master.loc[mask_price, "total_spend"] / supplier_master.loc[mask_price, "avg_price"]
)

# Overpay vs lowest benchmark (only where benchmark exists)
supplier_master["estimated_overpay"] = 0.0
if pd.notna(lowest_price):
    supplier_master["price_delta_vs_best"] = (supplier_master["avg_price"] - lowest_price).clip(lower=0)
    supplier_master["estimated_overpay"] = (supplier_master["price_delta_vs_best"] * supplier_master["est_units"]).fillna(0)
else:
    supplier_master["price_delta_vs_best"] = 0.0

total_overpay = float(supplier_master["estimated_overpay"].sum())

# Defect cost model: assume 0.5x of defective spend as rework/expedite impact (simple placeholder)
supplier_master["defect_cost"] = supplier_master["total_spend"] * (supplier_master["defect_rate"] / 100.0) * 0.5
total_defect_cost = float(supplier_master["defect_cost"].sum())

# Delivery risk exposure: spend for suppliers below 85% on-time
late_spend = float(supplier_master.loc[supplier_master["on_time_rate"] < 85, "total_spend"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Potential Cost Leakage (vs best RFQ)", f"${total_overpay:,.0f}")
c2.metric("Estimated Cost of Quality Issues", f"${total_defect_cost:,.0f}")
c3.metric("Spend Exposed to Delivery Risk", f"${late_spend:,.0f}")

st.caption("Modeled estimates for prototype demonstration (assumptions intentionally simple).")

# Optional: show the impact drivers table
with st.expander("Show impact drivers by supplier"):
    impact_cols = [
        "supplier_name",
        "total_spend",
        "avg_price",
        "price_delta_vs_best",
        "estimated_overpay",
        "defect_rate",
        "defect_cost",
        "on_time_rate",
        "risk_flag",
    ]
    st.dataframe(supplier_master[impact_cols].sort_values("estimated_overpay", ascending=False), use_container_width=True)

# Debug
with st.expander("Debug: show column names"):
    st.write("Orders columns:", list(orders.columns))
    st.write("Quality columns:", list(quality.columns))
    st.write("RFQ columns:", list(rfqs.columns))
