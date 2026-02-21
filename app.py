import re
import pandas as pd
import streamlit as st
import altair as alt

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Hoth Intelligence Hub", layout="wide")
st.title("🚀 Hoth Industries: Supplier Intelligence Hub")
st.markdown("---")

TOP_N = 10  # Standardize row counts across tables

# =========================================================
# CONSISTENT RISK COLOR MAP (charts match table semantics)
# =========================================================
RISK_ORDER = ["🔴 Quality Risk", "🟠 Delivery Risk", "🟡 Cost Risk", "🟢 Strategic"]
RISK_COLORS = ["#D62728", "#FF7F0E", "#F2C12E", "#2CA02C"]  # red, orange, gold, green

risk_color_scale = alt.Scale(domain=RISK_ORDER, range=RISK_COLORS)

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

def cap_top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.head(n) if len(df) > n else df

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

    manual_key_map = {
        "apex mfg": "apex manufacturing",
    }

    orders  = apply_entity_resolution(orders,  "supplier_name", manual_key_map)
    rfqs    = apply_entity_resolution(rfqs,    "supplier_name", manual_key_map)
    quality = apply_entity_resolution(quality, "supplier_name", manual_key_map)  # only if present

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
# DEFINITIONS / SCORING (EXPLAINABILITY PANEL)
# =========================================================
with st.expander("ℹ️ Definitions & Scoring (how to interpret the dashboard)", expanded=False):
    st.markdown(
        """
**Core KPIs**
- **total_spend**: Total purchase order spend per supplier (sum of `po_amount`).
- **on_time_rate**: % of orders delivered on or before the promised date.
- **defect_rate**: % of inspected parts rejected (avg of `parts_rejected / parts_inspected`).
- **avg_price**: Average RFQ quoted price per supplier.

**Scores**
- **price_score (0–100)**: Lower `avg_price` → higher score.  
  `100 * (1 - avg_price / max_avg_price)` (clipped to 0–100). If `avg_price = 0`, score is 0 (missing RFQ pricing).
- **performance_score**: Weighted composite (higher = better).  
  `0.45 * on_time_rate + 0.35 * (100 - defect_rate) + 0.20 * price_score`

**Risk Flags**
- 🔴 **Quality Risk**: `defect_rate >= 8%`
- 🟠 **Delivery Risk**: `on_time_rate <= 85%`
- 🟡 **Cost Risk**: `price_score <= 40`
- 🟢 **Strategic**: none triggered
        """
    )

# =========================================================
# KPI CALCULATIONS
# =========================================================
if "po_amount" not in orders.columns or "supplier_name" not in orders.columns:
    st.error("Orders must contain 'supplier_name' and 'po_amount'.")
    st.stop()

spend = orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index()
spend.columns = ["supplier_name", "total_spend"]

required_cols = {"promised_date", "actual_delivery_date"}
if not required_cols.issubset(set(orders.columns)):
    st.error("Orders must contain 'promised_date' and 'actual_delivery_date'.")
    st.stop()

orders_kpi = orders.copy()
orders_kpi["on_time"] = (orders_kpi["actual_delivery_date"] <= orders_kpi["promised_date"]).astype(float)
on_time = orders_kpi.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"] * 100).round(1)
on_time = on_time.drop(columns=["on_time"])

if "order_id" not in quality.columns or "order_id" not in orders.columns:
    st.error("Both quality and orders must contain 'order_id'.")
    st.stop()

q = quality.merge(orders[["order_id", "supplier_name"]], on="order_id", how="left")
if {"parts_rejected", "parts_inspected"}.issubset(set(q.columns)):
    q["defect_rate"] = (q["parts_rejected"] / q["parts_inspected"]).replace([pd.NA, float("inf")], 0)
else:
    q["defect_rate"] = 0.0

defects = q.groupby("supplier_name", dropna=False)["defect_rate"].mean().reset_index()
defects["defect_rate"] = (defects["defect_rate"] * 100).round(1)

if "supplier_name" not in rfqs.columns or "quoted_price" not in rfqs.columns:
    st.error("RFQs must contain 'supplier_name' and 'quoted_price'.")
    st.stop()

avg_price = rfqs.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index()
avg_price.columns = ["supplier_name", "avg_price"]
avg_price["avg_price"] = avg_price["avg_price"].round(2)

supplier_master = (
    spend.merge(on_time, on="supplier_name", how="left")
         .merge(defects, on="supplier_name", how="left")
         .merge(avg_price, on="supplier_name", how="left")
).fillna({"on_time_rate": 0.0, "defect_rate": 0.0, "avg_price": 0.0})

# =========================================================
# UNIQUE SUPPLIER COUNTS (LEVEL-SET METRICS)
# =========================================================
st.subheader("📌 Data Coverage (Supplier Counts)")
cA, cB = st.columns(2)
cA.metric("Unique suppliers (orders raw)", int(orders["supplier_name"].nunique()))
cB.metric("Unique suppliers (master table)", int(supplier_master["supplier_name"].nunique()))
st.caption("If the master table count is lower, entity resolution merged duplicates into a single supplier record.")

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
supplier_master = supplier_master.sort_values("performance_score", ascending=False)

# =========================================================
# SEARCH UI
# =========================================================
st.subheader("🔎 Search Suppliers")
search_query = st.text_input("Search by supplier name", placeholder="e.g., Apex, Stellar, TitanForge...")
filtered_master = apply_search(supplier_master, search_query)

# =========================================================
# DISPLAY TABLES (STANDARDIZED TOP_N)
# =========================================================
st.subheader(f"🏭 Unified Supplier Intelligence View (Top {TOP_N})")
st.dataframe(with_rank(cap_top_n(filtered_master, TOP_N)), use_container_width=True, hide_index=True)

st.markdown(f"### 🔴 Highest Risk Suppliers (Top {TOP_N})")
severity_rank = {"🔴 Quality Risk": 0, "🟠 Delivery Risk": 1, "🟡 Cost Risk": 2, "🟢 Strategic": 3}
risk_tbl = supplier_master.copy()
risk_tbl["_sev"] = risk_tbl["risk_flag"].map(severity_rank).fillna(9)
risk_tbl = risk_tbl.sort_values(["_sev", "performance_score"], ascending=[True, True]).drop(columns=["_sev"])
risk_tbl = apply_search(risk_tbl, search_query)
st.dataframe(with_rank(cap_top_n(risk_tbl, TOP_N)), use_container_width=True, hide_index=True)

st.markdown(f"### 🟢 Top Performing Suppliers (Top {TOP_N})")
top_tbl = apply_search(supplier_master.sort_values("performance_score", ascending=False), search_query)
st.dataframe(with_rank(cap_top_n(top_tbl, TOP_N)), use_container_width=True, hide_index=True)

# =========================================================
# 💰 CONSOLIDATION OPPORTUNITIES
# =========================================================
st.markdown("---")
st.header("💰 Supplier Consolidation Opportunities")

pricing_pool = supplier_master[supplier_master["avg_price"] > 0].copy()
if pricing_pool.empty:
    st.warning("No RFQ pricing available (avg_price > 0) — cannot estimate consolidation savings.")
else:
    best_price = float(pricing_pool["avg_price"].min())
    tmp = supplier_master.copy()
    tmp["price_delta_vs_best"] = (tmp["avg_price"] - best_price).clip(lower=0)
    tmp["est_units"] = 0.0
    mask_price = tmp["avg_price"] > 0
    tmp.loc[mask_price, "est_units"] = tmp.loc[mask_price, "total_spend"] / tmp.loc[mask_price, "avg_price"]
    tmp["estimated_savings"] = (tmp["price_delta_vs_best"] * tmp["est_units"]).fillna(0)

    total_savings = float(tmp["estimated_savings"].sum())
    st.metric("Estimated Annual Savings via Consolidation (Model)", f"${total_savings:,.0f}")

    show_cols = ["supplier_name", "total_spend", "avg_price", "estimated_savings", "risk_flag"]
    st.dataframe(
        with_rank(tmp.sort_values("estimated_savings", ascending=False)[show_cols]),
        use_container_width=True,
        hide_index=True
    )
    st.caption("Model: benchmark against lowest non-zero avg RFQ price; units approximated as spend / avg_price.")

# =========================================================
# ✅ EXECUTIVE VISUALS (REPLACES SCATTER)
# =========================================================
st.markdown("---")
st.header("📊 Executive Supplier Snapshot")

viz_df = supplier_master.copy()
viz_df["spend_m"] = (viz_df["total_spend"] / 1_000_000).round(2)
viz_df["supplier_label"] = viz_df["supplier_name"]

# ---- Bar 1: Spend ($M)
st.subheader("Spend by Supplier ($M)")
spend_chart = (
    alt.Chart(viz_df)
    .mark_bar()
    .encode(
        y=alt.Y("supplier_label:N", sort="-x", title=None),
        x=alt.X("spend_m:Q", title="Total Spend ($M)"),
        color=alt.Color("risk_flag:N", scale=risk_color_scale, legend=alt.Legend(title="Risk Flag")),
        tooltip=[
            alt.Tooltip("supplier_name:N", title="Supplier"),
            alt.Tooltip("total_spend:Q", title="Total Spend ($)", format=",.0f"),
            alt.Tooltip("spend_m:Q", title="Total Spend ($M)", format=",.2f"),
            alt.Tooltip("risk_flag:N", title="Risk"),
        ],
    )
)
st.altair_chart(spend_chart, use_container_width=True)

# ---- Bar 2: Performance Score (0–100)
st.subheader("Performance Score by Supplier (0–100)")
perf_chart = (
    alt.Chart(viz_df)
    .mark_bar()
    .encode(
        y=alt.Y("supplier_label:N", sort="-x", title=None),
        x=alt.X("performance_score:Q", title="Performance Score", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("risk_flag:N", scale=risk_color_scale, legend=None),
        tooltip=[
            alt.Tooltip("supplier_name:N", title="Supplier"),
            alt.Tooltip("performance_score:Q", title="Performance Score", format=",.1f"),
            alt.Tooltip("on_time_rate:Q", title="On-Time %", format=",.1f"),
            alt.Tooltip("defect_rate:Q", title="Defect %", format=",.1f"),
            alt.Tooltip("avg_price:Q", title="Avg RFQ Price", format=",.2f"),
            alt.Tooltip("risk_flag:N", title="Risk"),
        ],
    )
)
st.altair_chart(perf_chart, use_container_width=True)

# ---- Priority Quadrant Table
st.subheader("Priority Quadrant (Simple + Executive-Friendly)")
x_cut = float(viz_df["performance_score"].median()) if len(viz_df) else 50.0
y_cut = float(viz_df["total_spend"].median()) if len(viz_df) else 0.0

quad = viz_df.copy()
quad["spend_bucket"] = quad["total_spend"].apply(lambda v: "High Spend" if v >= y_cut else "Low Spend")
quad["perf_bucket"] = quad["performance_score"].apply(lambda v: "Low Performance" if v < x_cut else "High Performance")
quad["quadrant"] = quad["spend_bucket"] + " / " + quad["perf_bucket"]

quad_order = [
    "High Spend / Low Performance",
    "High Spend / High Performance",
    "Low Spend / Low Performance",
    "Low Spend / High Performance",
]

quad = quad.sort_values(["total_spend"], ascending=False)
quad["quadrant"] = pd.Categorical(quad["quadrant"], categories=quad_order, ordered=True)
quad = quad.sort_values(["quadrant", "total_spend"], ascending=[True, False])

st.caption(f"Median cutoffs: Performance = {x_cut:.1f}, Spend = ${y_cut:,.0f}. Primary focus = **High Spend / Low Performance**.")
st.dataframe(
    quad[["quadrant", "supplier_name", "total_spend", "performance_score", "risk_flag", "on_time_rate", "defect_rate", "avg_price"]],
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 💰 FINANCIAL IMPACT (PROTOTYPE ESTIMATES)
# =========================================================
st.markdown("---")
st.header("💰 Estimated Financial Impact (Prototype)")
st.caption(
    "These estimates update as supplier identity resolution improves (merging duplicates changes spend, pricing benchmarks, and unit estimates)."
)

nonzero_prices = supplier_master["avg_price"].replace(0, pd.NA).dropna()
lowest_price = nonzero_prices.min() if len(nonzero_prices) else pd.NA

supplier_master["est_units"] = 0.0
mask_price = supplier_master["avg_price"] > 0
supplier_master.loc[mask_price, "est_units"] = (
    supplier_master.loc[mask_price, "total_spend"] / supplier_master.loc[mask_price, "avg_price"]
)

supplier_master["estimated_overpay"] = 0.0
if pd.notna(lowest_price):
    supplier_master["price_delta_vs_best"] = (supplier_master["avg_price"] - lowest_price).clip(lower=0)
    supplier_master["estimated_overpay"] = (
        supplier_master["price_delta_vs_best"] * supplier_master["est_units"]
    ).fillna(0)
else:
    supplier_master["price_delta_vs_best"] = 0.0

total_overpay = float(supplier_master["estimated_overpay"].sum())
supplier_master["defect_cost"] = supplier_master["total_spend"] * (supplier_master["defect_rate"] / 100.0) * 0.5
total_defect_cost = float(supplier_master["defect_cost"].sum())
late_spend = float(supplier_master.loc[supplier_master["on_time_rate"] < 85, "total_spend"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Potential Cost Leakage (vs best RFQ)", f"${total_overpay:,.0f}")
c2.metric("Estimated Cost of Quality Issues", f"${total_defect_cost:,.0f}")
c3.metric("Spend Exposed to Delivery Risk", f"${late_spend:,.0f}")

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
