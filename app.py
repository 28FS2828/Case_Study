import re
from datetime import date, timedelta
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

def read_text_flexible(candidates):
    """Try multiple filenames for supplier notes."""
    last_err = None
    for f in candidates:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                return fp.read()
        except Exception as e:
            last_err = e
    return ""  # notes are optional; don't hard fail

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

def find_best_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Find a column whose name contains one of the keywords."""
    cols = list(df.columns)
    low = {c: c.lower() for c in cols}
    for kw in keywords:
        for c in cols:
            if kw in low[c]:
                return c
    return None

# =========================================================
# SUPPLIER NOTES PARSER (lightweight)
# =========================================================
def parse_supplier_notes(notes_text: str) -> dict:
    """
    Parse supplier_notes.txt into {canonical_supplier_key: {title, risk_hint, bullets}}.
    This is intentionally simple/robust for a prototype.
    """
    notes = {}
    if not notes_text:
        return notes

    blocks = re.split(r"\n=+\n", notes_text)
    for b in blocks:
        b = b.strip()
        if not b:
            continue

        # Try to detect a "HEADER - ..." line
        header_match = re.search(r"^([A-Z0-9 &/]+)\s*-\s*(.+)$", b.splitlines()[0].strip(), flags=re.IGNORECASE)
        if header_match:
            supplier_raw = header_match.group(1).strip()
            descriptor = header_match.group(2).strip()
            k = normalize_supplier_key(supplier_raw)

            # pick a "risk hint" from descriptor
            desc_low = descriptor.lower()
            if "caution" in desc_low or "high risk" in desc_low:
                risk_hint = "🔴 Quality Risk"
            elif "gold" in desc_low or "standard" in desc_low:
                risk_hint = "🟢 Strategic"
            else:
                risk_hint = ""

            # Pull a few short bullet snippets (first ~4 non-empty lines after header)
            lines = [ln.strip() for ln in b.splitlines()[1:] if ln.strip()]
            bullets = []
            for ln in lines:
                # keep it short
                bullets.append(ln[:160] + ("…" if len(ln) > 160 else ""))
                if len(bullets) >= 4:
                    break

            notes[k] = {
                "title": supplier_raw.title(),
                "descriptor": descriptor,
                "risk_hint": risk_hint,
                "bullets": bullets
            }

    return notes

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

    notes_text = read_text_flexible([
        "supplier_notes.txt",
        "Supplier_Notes.txt",
        "SUPPLIER_NOTES.txt",
        "supplier_notes (1).txt",
    ])

    # Manual overrides (normalized_key -> canonical_key)
    manual_key_map = {
        "apex mfg": "apex manufacturing",
        "apex manufacturing inc": "apex manufacturing",
        "apex mfg inc": "apex manufacturing",
    }

    orders  = apply_entity_resolution(orders,  "supplier_name", manual_key_map)
    rfqs    = apply_entity_resolution(rfqs,    "supplier_name", manual_key_map)
    quality = apply_entity_resolution(quality, "supplier_name", manual_key_map)

    # Dates
    orders  = safe_to_datetime(orders,  "order_date")
    orders  = safe_to_datetime(orders,  "promised_date")
    orders  = safe_to_datetime(orders,  "actual_delivery_date")
    quality = safe_to_datetime(quality, "inspection_date")
    rfqs    = safe_to_datetime(rfqs,    "quote_date")

    return orders, quality, rfqs, notes_text

try:
    orders, quality, rfqs, supplier_notes_text = load_data()
    st.success("✅ Data loaded & supplier names normalized (entity resolution applied)")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

supplier_notes = parse_supplier_notes(supplier_notes_text)

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

# Spend
spend = orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index()
spend.columns = ["supplier_name", "total_spend"]

# On-time rate
required_cols = {"promised_date", "actual_delivery_date"}
if not required_cols.issubset(set(orders.columns)):
    st.error("Orders must contain 'promised_date' and 'actual_delivery_date'.")
    st.stop()

orders_kpi = orders.copy()
orders_kpi["on_time"] = (orders_kpi["actual_delivery_date"] <= orders_kpi["promised_date"]).astype(float)
on_time = orders_kpi.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"] * 100).round(1)
on_time = on_time.drop(columns=["on_time"])

# Defect rate
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

# Avg RFQ price
if "supplier_name" not in rfqs.columns or "quoted_price" not in rfqs.columns:
    st.error("RFQs must contain 'supplier_name' and 'quoted_price'.")
    st.stop()

avg_price = rfqs.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index()
avg_price.columns = ["supplier_name", "avg_price"]
avg_price["avg_price"] = avg_price["avg_price"].round(2)

# Unified table
supplier_master = (
    spend.merge(on_time, on="supplier_name", how="left")
         .merge(defects, on="supplier_name", how="left")
         .merge(avg_price, on="supplier_name", how="left")
).fillna({"on_time_rate": 0.0, "defect_rate": 0.0, "avg_price": 0.0})

# =========================================================
# SCORING + FLAGS
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
# DATA COVERAGE
# =========================================================
st.subheader("📌 Data Coverage (Supplier Counts)")
cA, cB = st.columns(2)
cA.metric("Unique suppliers (orders raw)", int(orders["supplier_name"].nunique()))
cB.metric("Unique suppliers (master table)", int(supplier_master["supplier_name"].nunique()))
st.caption("If the master table count is lower, entity resolution merged duplicates into a single supplier record.")

# =========================================================
# SEARCH UI
# =========================================================
st.subheader("🔎 Search Suppliers")
search_query = st.text_input("Search by supplier name", placeholder="e.g., Apex, Stellar, TitanForge...")
filtered_master = apply_search(supplier_master, search_query)

# =========================================================
# DISPLAY TABLES
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
# 💰 SUPPLIER CONSOLIDATION OPPORTUNITIES
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
# ✅ REAL-TIME DECISION SUPPORT (NEW)
# =========================================================
st.markdown("---")
st.header("⚡ Real-Time Sourcing Decision Support (3-week decision)")

# Simple “countdown” assumption (prototype)
decision_due = date.today() + timedelta(days=21)
days_left = (decision_due - date.today()).days
st.info(f"📅 Decision window: **{days_left} days remaining** (prototype assumes a 3-week deadline).")

# Try to discover a part/commodity/category column (if present)
orders_part_col = find_best_col(orders, ["commodity", "category", "part", "component", "item", "material", "product"])
rfq_part_col = find_best_col(rfqs, ["commodity", "category", "part", "component", "item", "material", "product"])

decision_cols = st.columns([1.2, 1.2, 1.6])
with decision_cols[0]:
    req_on_time = st.slider("Minimum On-Time %", 0, 100, 90)
with decision_cols[1]:
    max_defects = st.slider("Maximum Defect %", 0, 20, 5)
with decision_cols[2]:
    st.write("Weights (what matters most right now)")
    w_delivery = st.slider("Delivery weight", 0.0, 1.0, 0.45, 0.05)
    w_quality  = st.slider("Quality weight",  0.0, 1.0, 0.35, 0.05)
    w_cost     = st.slider("Cost weight",     0.0, 1.0, 0.20, 0.05)
    w_sum = w_delivery + w_quality + w_cost
    if w_sum == 0:
        w_delivery, w_quality, w_cost = 0.45, 0.35, 0.20
        w_sum = 1.0
    w_delivery, w_quality, w_cost = w_delivery / w_sum, w_quality / w_sum, w_cost / w_sum

commodity_choice = None
if orders_part_col or rfq_part_col:
    # Build a merged list of commodity values from whichever column(s) exist
    vals = set()
    if orders_part_col:
        vals |= set(orders[orders_part_col].dropna().astype(str).unique().tolist())
    if rfq_part_col:
        vals |= set(rfqs[rfq_part_col].dropna().astype(str).unique().tolist())
    vals = sorted([v for v in vals if v.strip()])

    if vals:
        commodity_choice = st.selectbox(
            "Filter decision to a commodity/category (optional)",
            ["(All)"] + vals
        )
else:
    st.caption("No part/commodity column detected in the sample data — decision support will rank suppliers overall.")

# Build a decision-scoped view (optional)
decision_orders = orders.copy()
decision_rfqs = rfqs.copy()

if commodity_choice and commodity_choice != "(All)":
    if orders_part_col and orders_part_col in decision_orders.columns:
        decision_orders = decision_orders[decision_orders[orders_part_col].astype(str) == commodity_choice]
    if rfq_part_col and rfq_part_col in decision_rfqs.columns:
        decision_rfqs = decision_rfqs[decision_rfqs[rfq_part_col].astype(str) == commodity_choice]

# Recompute scoped KPIs for decision (so it feels “real-time” and specific)
if len(decision_orders) == 0:
    st.warning("No orders match the selected commodity filter. Showing overall supplier ranking instead.")
    decision_kpi = supplier_master.copy()
else:
    # Spend
    spend_d = decision_orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index()
    spend_d.columns = ["supplier_name", "total_spend"]

    # On-time
    decision_orders_kpi = decision_orders.copy()
    decision_orders_kpi["on_time"] = (decision_orders_kpi["actual_delivery_date"] <= decision_orders_kpi["promised_date"]).astype(float)
    on_time_d = decision_orders_kpi.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
    on_time_d["on_time_rate"] = (on_time_d["on_time"] * 100).round(1)
    on_time_d = on_time_d.drop(columns=["on_time"])

    # Defects: join via order_id if possible
    qd = quality.merge(decision_orders[["order_id", "supplier_name"]], on="order_id", how="inner")
    if {"parts_rejected", "parts_inspected"}.issubset(set(qd.columns)) and len(qd) > 0:
        qd["defect_rate"] = (qd["parts_rejected"] / qd["parts_inspected"]).replace([pd.NA, float("inf")], 0)
        defects_d = qd.groupby("supplier_name", dropna=False)["defect_rate"].mean().reset_index()
        defects_d["defect_rate"] = (defects_d["defect_rate"] * 100).round(1)
    else:
        defects_d = pd.DataFrame({"supplier_name": spend_d["supplier_name"], "defect_rate": 0.0})

    # RFQ avg price (scoped)
    if len(decision_rfqs) > 0:
        avg_price_d = decision_rfqs.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index()
        avg_price_d.columns = ["supplier_name", "avg_price"]
        avg_price_d["avg_price"] = avg_price_d["avg_price"].round(2)
    else:
        avg_price_d = pd.DataFrame({"supplier_name": spend_d["supplier_name"], "avg_price": 0.0})

    decision_kpi = (
        spend_d.merge(on_time_d, on="supplier_name", how="left")
               .merge(defects_d, on="supplier_name", how="left")
               .merge(avg_price_d, on="supplier_name", how="left")
    ).fillna({"on_time_rate": 0.0, "defect_rate": 0.0, "avg_price": 0.0})

    # Cost normalization inside decision scope
    max_price_d = decision_kpi["avg_price"].replace(0, pd.NA).max()
    if pd.notna(max_price_d) and max_price_d > 0:
        decision_kpi["price_score"] = 100 * (1 - (decision_kpi["avg_price"] / max_price_d))
        decision_kpi["price_score"] = decision_kpi["price_score"].fillna(0).clip(0, 100)
    else:
        decision_kpi["price_score"] = 0.0

    decision_kpi["performance_score"] = (
        (decision_kpi["on_time_rate"] * w_delivery) +
        ((100 - decision_kpi["defect_rate"]) * w_quality) +
        (decision_kpi["price_score"] * w_cost)
    ).round(1)

    decision_kpi["risk_flag"] = decision_kpi.apply(risk_flag, axis=1)

# Apply fit thresholds
decision_kpi["meets_on_time"] = decision_kpi["on_time_rate"] >= req_on_time
decision_kpi["meets_quality"] = decision_kpi["defect_rate"] <= max_defects
decision_kpi["fit"] = decision_kpi["meets_on_time"] & decision_kpi["meets_quality"]
decision_kpi["fit_status"] = decision_kpi["fit"].map({True: "✅ Fit", False: "❌ Not fit"})

# Overlay supplier notes
def note_snippet(supplier_name: str) -> str:
    k = normalize_supplier_key(supplier_name)
    n = supplier_notes.get(k)
    if not n:
        return ""
    # one-line summary
    desc = n.get("descriptor", "")
    bullets = n.get("bullets", [])
    line = desc
    if bullets:
        line = f"{desc} | {bullets[0]}"
    return line[:200] + ("…" if len(line) > 200 else "")

decision_kpi["notes_hint"] = decision_kpi["supplier_name"].apply(note_snippet)

st.subheader("✅ Recommendation (Decision-Time Ranking)")
st.caption("This view is designed for **the sourcing decision meeting**: pick thresholds, set weights, and instantly see recommended suppliers + tribal knowledge.")

show_cols = [
    "supplier_name", "fit_status", "performance_score", "risk_flag",
    "on_time_rate", "defect_rate", "avg_price", "total_spend", "notes_hint"
]
decision_view = decision_kpi.sort_values(["fit", "performance_score"], ascending=[False, False])
st.dataframe(with_rank(cap_top_n(decision_view[show_cols], TOP_N)), use_container_width=True, hide_index=True)

# “At-a-glance” callouts
fit_count = int(decision_kpi["fit"].sum())
st.metric("Suppliers meeting thresholds", f"{fit_count} / {len(decision_kpi)}")

# Quick supplier brief (mobile-friendly pattern)
st.subheader("📱 Quick Supplier Brief (Search + Notes)")
supplier_pick = st.selectbox("Select a supplier to view details", sorted(decision_kpi["supplier_name"].astype(str).unique().tolist()))
row = decision_kpi[decision_kpi["supplier_name"] == supplier_pick].iloc[0].to_dict()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Performance Score", f"{row.get('performance_score', 0)}")
c2.metric("On-Time %", f"{row.get('on_time_rate', 0)}")
c3.metric("Defect %", f"{row.get('defect_rate', 0)}")
c4.metric("Risk Flag", f"{row.get('risk_flag', '')}")

st.write("**Tribal knowledge / supplier notes:**")
k = normalize_supplier_key(supplier_pick)
n = supplier_notes.get(k)
if n:
    st.write(f"- **{n.get('descriptor','')}**")
    for b in n.get("bullets", []):
        st.write(f"- {b}")
else:
    st.write("- (No notes found in supplier_notes.txt for this supplier.)")

# =========================================================
# ✅ EXECUTIVE VISUALS (BARS + QUADRANT TABLE)
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
        y=alt.Y(
            "supplier_label:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labelAlign="left", labelPadding=12, labelLimit=400)
        ),
        x=alt.X("spend_m:Q", title="Total Spend ($M)"),
        color=alt.Color("risk_flag:N", scale=risk_color_scale, legend=alt.Legend(title="Risk Flag")),
        tooltip=[
            alt.Tooltip("supplier_name:N", title="Supplier"),
            alt.Tooltip("total_spend:Q", title="Total Spend ($)", format=",.0f"),
            alt.Tooltip("spend_m:Q", title="Total Spend ($M)", format=",.2f"),
            alt.Tooltip("risk_flag:N", title="Risk"),
        ],
    )
    .properties(height=260)
    .configure_view(strokeOpacity=0)
    .configure_axisY(labelAngle=0)
)
st.altair_chart(spend_chart, use_container_width=True)

# ---- Bar 2: Performance Score (0–100)
st.subheader("Performance Score by Supplier (0–100)")
perf_chart = (
    alt.Chart(viz_df)
    .mark_bar()
    .encode(
        y=alt.Y(
            "supplier_label:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labelAlign="left", labelPadding=12, labelLimit=400)
        ),
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
    .properties(height=260)
    .configure_view(strokeOpacity=0)
    .configure_axisY(labelAngle=0)
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
    quad[[
        "quadrant",
        "supplier_name",
        "total_spend",
        "performance_score",
        "risk_flag",
        "on_time_rate",
        "defect_rate",
        "avg_price"
    ]],
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 💰 FINANCIAL IMPACT (PROTOTYPE ESTIMATES)
# =========================================================
st.markdown("---")
st.header("💰 Estimated Financial Impact (Prototype)")
st.caption("These are prototype estimates to make decision-making tangible in the meeting.")

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
