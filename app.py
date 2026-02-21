import re
import pandas as pd
import streamlit as st

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Hoth Intelligence Hub", layout="wide")
st.title("🚀 Hoth Industries: Supplier Intelligence Hub")
st.markdown("---")

TOP_N = 10  # Standardize row counts across tables

# =========================================================
# ENTITY RESOLUTION HELPERS
# =========================================================
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "l.l.c", "ltd", "limited",
    "corp", "corporation", "co", "company", "gmbh", "s.a", "sa"
}

def normalize_supplier_key(name: str) -> str:
    """Stable normalization key for entity resolution (no extra libs)."""
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)      # punctuation -> spaces
    s = re.sub(r"\s+", " ", s).strip()     # collapse whitespace

    parts = s.split(" ")
    while parts and parts[-1] in LEGAL_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts).strip()

def apply_entity_resolution(df: pd.DataFrame, col: str, manual_key_map: dict | None = None) -> pd.DataFrame:
    """
    Resolve near-duplicate supplier entities using normalized keys + optional manual overrides.
    Canonical naming rule: ALWAYS choose the longest (most complete) company name.
    """
    if col not in df.columns:
        return df

    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    out["_supplier_key"] = out[col].apply(normalize_supplier_key)

    if manual_key_map:
        out["_supplier_key"] = out["_supplier_key"].replace(manual_key_map)

    def pick_longest_name(values):
        vals = [v for v in values if isinstance(v, str) and v.strip()]
        vals = list(set([v.strip() for v in vals]))
        if not vals:
            return ""
        vals.sort(key=lambda x: (-len(x), x))  # longest first, then alphabetical
        return vals[0]

    canonical = (
        out.groupby("_supplier_key")[col]
           .agg(pick_longest_name)
           .to_dict()
    )

    out[col] = out["_supplier_key"].map(canonical).fillna(out[col])
    out = out.drop(columns=["_supplier_key"])
    return out

# =========================================================
# GENERAL HELPERS
# =========================================================
def read_csv_flexible(candidates):
    """Try multiple filenames (Streamlit Cloud vs local copies)."""
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

def with_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out

def apply_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    q = query.strip().lower()
    return df[df["supplier_name"].astype(str).str.lower().str.contains(q, na=False)]

# =========================================================
# LOAD + CLEAN DATA
# =========================================================
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

    # Manual overrides (normalized_key -> canonical_key)
    # Keep minimal; only add when you see stubborn edge cases in QA
    manual_key_map = {
        "apex mfg": "apex manufacturing",
    }

    orders  = apply_entity_resolution(orders,  "supplier_name", manual_key_map)
    rfqs    = apply_entity_resolution(rfqs,    "supplier_name", manual_key_map)
    quality = apply_entity_resolution(quality, "supplier_name", manual_key_map)  # only if present

    # Dates
    orders  = safe_to_datetime(orders,  "order_date")
    orders  = safe_to_datetime(orders,  "promised_date")
    orders  = safe_to_datetime(orders,  "actual_delivery_date")
    quality = safe_to_datetime(quality, "inspection_date")
    rfqs    = safe_to_datetime(rfqs,    "quote_date")

    return orders, quality, rfqs

try:
    orders, quality, rfqs = load_data()
    st.success("✅ Data loaded & supplier names normalized (entity resolution applied)")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# =========================================================
# ENTITY RESOLUTION QA
# =========================================================
with st.expander("🧼 Entity Resolution QA (show potential duplicates)"):
    if "supplier_name" not in orders.columns:
        st.write("Orders file does not contain supplier_name.")
    else:
        raw = orders["supplier_name"].astype(str).str.strip()
        keys = raw.apply(normalize_supplier_key)

        tmp = pd.DataFrame({"supplier_name_raw": raw, "normalized_key": keys})
        counts = tmp.groupby("normalized_key")["supplier_name_raw"].nunique().reset_index(name="raw_name_variants")
        multi = counts[counts["raw_name_variants"] > 1].sort_values("raw_name_variants", ascending=False)

        if multi.empty:
            st.write("✅ No potential duplicates detected via normalization key.")
        else:
            st.write("Potential duplicates (multiple raw names collapsing to same normalized key):")
            show_keys = multi["normalized_key"].head(20).tolist()
            st.dataframe(
                tmp[tmp["normalized_key"].isin(show_keys)]
                  .sort_values(["normalized_key", "supplier_name_raw"]),
                use_container_width=True,
                hide_index=True
            )
            st.caption("If any are true duplicates that didn't merge cleanly, add a manual_key_map entry in load_data().")

# =========================================================
# KPI CALCULATIONS
# =========================================================
if "po_amount" not in orders.columns:
    st.error("Expected 'po_amount' column in orders but didn't find it.")
    st.stop()

if "supplier_name" not in orders.columns:
    st.error("Expected 'supplier_name' column in orders but didn't find it.")
    st.stop()

# Spend
spend = orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index()
spend.columns = ["supplier_name", "total_spend"]

# On-time rate
required_cols = {"promised_date", "actual_delivery_date"}
if not required_cols.issubset(set(orders.columns)):
    st.error("Expected 'promised_date' and 'actual_delivery_date' columns in orders.")
    st.stop()

orders_kpi = orders.copy()
orders_kpi["on_time"] = (orders_kpi["actual_delivery_date"] <= orders_kpi["promised_date"]).astype(float)
on_time = orders_kpi.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"] * 100).round(1)
on_time = on_time.drop(columns=["on_time"])

# Defect rate (quality -> orders by order_id)
if "order_id" not in quality.columns or "order_id" not in orders.columns:
    st.error("Expected 'order_id' in both orders and quality datasets.")
    st.stop()

q = quality.merge(
    orders[["order_id", "supplier_name"]],
    on="order_id",
    how="left",
)

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

# =========================================================
# UNIFIED SUPPLIER TABLE
# =========================================================
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

# =========================================================
# PERFORMANCE SCORING + FLAGS
# =========================================================
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

# Sort best-to-worst by score
supplier_master = supplier_master.sort_values("performance_score", ascending=False)

# =========================================================
# SEARCH UI
# =========================================================
st.subheader("🔎 Search Suppliers")
search_query = st.text_input(
    "Search by supplier name",
    placeholder="e.g., Apex, Stellar, TitanForge..."
)

# Apply search filter for display tables
filtered_master = apply_search(supplier_master, search_query)

# =========================================================
# DISPLAY TABLES (STANDARDIZED TOP_N)
# =========================================================
st.subheader("🏭 Unified Supplier Intelligence View (Top 10)")
st.dataframe(with_rank(filtered_master.head(TOP_N)), use_container_width=True, hide_index=True)

st.markdown(f"### 🔴 Highest Risk Suppliers (Top {TOP_N})")
risk_tbl = supplier_master[supplier_master["risk_flag"].str.contains("🔴|🟠|🟡")]
# show worst first: quality risk -> delivery -> cost, then by low score
severity_rank = {"🔴 Quality Risk": 0, "🟠 Delivery Risk": 1, "🟡 Cost Risk": 2, "🟢 Strategic": 3}
risk_tbl = risk_tbl.copy()
risk_tbl["_sev"] = risk_tbl["risk_flag"].map(severity_rank).fillna(9)
risk_tbl = risk_tbl.sort_values(["_sev", "performance_score"], ascending=[True, True]).drop(columns=["_sev"])
risk_tbl = apply_search(risk_tbl, search_query)
st.dataframe(with_rank(risk_tbl.head(TOP_N)), use_container_width=True, hide_index=True)

st.markdown(f"### 🟢 Top Performing Suppliers (Top {TOP_N})")
top_tbl = apply_search(supplier_master.sort_values("performance_score", ascending=False), search_query)
st.dataframe(with_rank(top_tbl.head(TOP_N)), use_container_width=True, hide_index=True)

# =========================================================
# 💰 FINANCIAL IMPACT (PROTOTYPE ESTIMATES)
# =========================================================
st.markdown("---")
st.header("💰 Estimated Financial Impact (Prototype)")

st.caption(
    "These estimates will change as supplier identity resolution improves (merging duplicates changes spend, pricing benchmarks, and unit estimates)."
)

# Use lowest non-zero avg RFQ as benchmark
nonzero_prices = supplier_master["avg_price"].replace(0, pd.NA).dropna()
lowest_price = nonzero_prices.min() if len(nonzero_prices) else pd.NA

# Estimate units from spend/avg_price (only where avg_price > 0)
supplier_master["est_units"] = 0.0
mask_price = supplier_master["avg_price"] > 0
supplier_master.loc[mask_price, "est_units"] = (
    supplier_master.loc[mask_price, "total_spend"] / supplier_master.loc[mask_price, "avg_price"]
)

# Overpay vs lowest benchmark
supplier_master["estimated_overpay"] = 0.0
if pd.notna(lowest_price):
    supplier_master["price_delta_vs_best"] = (supplier_master["avg_price"] - lowest_price).clip(lower=0)
    supplier_master["estimated_overpay"] = (
        supplier_master["price_delta_vs_best"] * supplier_master["est_units"]
    ).fillna(0)
else:
    supplier_master["price_delta_vs_best"] = 0.0

total_overpay = float(supplier_master["estimated_overpay"].sum())

# Defect cost model: placeholder (simple)
supplier_master["defect_cost"] = supplier_master["total_spend"] * (supplier_master["defect_rate"] / 100.0) * 0.5
total_defect_cost = float(supplier_master["defect_cost"].sum())

# Delivery risk exposure: spend for suppliers below 85% on-time
late_spend = float(supplier_master.loc[supplier_master["on_time_rate"] < 85, "total_spend"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Potential Cost Leakage (vs best RFQ)", f"${total_overpay:,.0f}")
c2.metric("Estimated Cost of Quality Issues", f"${total_defect_cost:,.0f}")
c3.metric("Spend Exposed to Delivery Risk", f"${late_spend:,.0f}")

st.caption("Modeled estimates for prototype demonstration (assumptions intentionally simple).")

with st.expander("Show impact drivers by supplier"):
    impact_cols = [
        "supplier_name",
        "total_spend",
        "avg_price",
        "price_score",
        "price_delta_vs_best",
        "estimated_overpay",
        "defect_rate",
        "defect_cost",
        "on_time_rate",
        "performance_score",
        "risk_flag",
    ]
    st.dataframe(
        with_rank(supplier_master[impact_cols].sort_values("estimated_overpay", ascending=False)),
        use_container_width=True,
        hide_index=True
    )

with st.expander("Debug: show column names"):
    st.write("Orders columns:", list(orders.columns))
    st.write("Quality columns:", list(quality.columns))
    st.write("RFQ columns:", list(rfqs.columns))
