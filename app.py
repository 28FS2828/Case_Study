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

TOP_N = 10  # standard row cap for tables

# =========================================================
# RESET FILTERS (SESSION STATE)
# =========================================================
DEFAULTS = {
    "search_query": "",
    "decision_in_days": 21,
    "req_on_time": 90,
    "max_defects": 5,
    "w_delivery": 0.45,
    "w_quality": 0.35,
    "w_cost": 0.20,
    "capability_source": "RFQs only",
    "min_lines": 2,
    "show_coverage": True,
    "category_choice": "(All Categories)",
}

def init_defaults():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

init_defaults()

# Global reset button (top of app)
top_left, top_right = st.columns([1, 3])
with top_left:
    if st.button("🔄 Reset filters", key="reset_filters_top", use_container_width=True):
        reset_filters()
        st.rerun()
with top_right:
    st.caption("Use **Reset filters** to restore default thresholds, weights, and category scope.")

# =========================================================
# CONSISTENT RISK COLOR MAP (charts match table semantics)
# =========================================================
RISK_ORDER = ["🔴 Quality Risk", "🟠 Delivery Risk", "🟡 Cost Risk", "🟢 Strategic"]
RISK_COLORS = ["#D62728", "#FF7F0E", "#F2C12E", "#2CA02C"]  # red, orange, gold, green
risk_color_scale = alt.Scale(domain=RISK_ORDER, range=RISK_COLORS)

# =========================================================
# DISPLAY LABELS (UNITS IN HEADERS)
# =========================================================
DISPLAY_COLS = {
    "supplier_name": "Supplier",
    "risk_flag": "Risk Flag",
    "fit_status": "Fit Status",
    "notes_hint": "Supplier Notes (Tribal Knowledge)",

    "total_spend": "Total Spend ($)",
    "spend_m": "Total Spend ($M)",
    "avg_price": "Avg RFQ Price ($/unit)",
    "price_delta_vs_best": "Price Delta vs Best ($/unit)",

    "on_time_rate": "On-Time Rate (%)",
    "defect_rate": "Defect Rate (%)",

    "price_score": "Price Score (0–100)",
    "performance_score": "Performance Score (0–100)",

    "estimated_savings": "Est. Savings ($)",
    "estimated_overpay": "Est. Overpay ($)",
    "defect_cost": "Est. Defect Cost ($)",

    "quadrant": "Quadrant",
    "part_category": "Part Category",
}

def format_for_display(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    keep = [c for c in cols if c in out.columns]
    out = out[keep]
    out = out.rename(columns={c: DISPLAY_COLS.get(c, c) for c in keep})
    return out

def with_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out

# =========================================================
# EXEC-FRIENDLY FORMATTING ($ commas, %, etc.)
# =========================================================
def _fmt_money(x):
    if pd.isna(x):
        return ""
    return f"${x:,.0f}"

def _fmt_money_2(x):
    if pd.isna(x):
        return ""
    return f"${x:,.2f}"

def _fmt_pct(x):
    if pd.isna(x):
        return ""
    return f"{x:.1f}%"

def _fmt_score(x):
    if pd.isna(x):
        return ""
    return f"{x:.1f}"

def style_exec_table(df: pd.DataFrame):
    money_cols = [c for c in df.columns if "($)" in c and "($/unit)" not in c]
    money_unit_cols = [c for c in df.columns if "($/unit)" in c]
    pct_cols = [c for c in df.columns if "(%)" in c]
    score_cols = [c for c in df.columns if "(0–100)" in c or "(0-100)" in c]

    styler = df.style
    if money_cols:
        styler = styler.format({c: _fmt_money for c in money_cols})
    if money_unit_cols:
        styler = styler.format({c: _fmt_money_2 for c in money_unit_cols})
    if pct_cols:
        styler = styler.format({c: _fmt_pct for c in pct_cols})
    if score_cols:
        styler = styler.format({c: _fmt_score for c in score_cols})
    return styler

def show_table(df: pd.DataFrame, max_rows: int = TOP_N):
    df_show = df.head(max_rows) if len(df) > max_rows else df
    st.dataframe(style_exec_table(df_show), use_container_width=True, hide_index=True)

# =========================================================
# ENTITY RESOLUTION
# =========================================================
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "l.l.c", "ltd", "limited",
    "corp", "corporation", "co", "company", "gmbh", "s.a", "sa"
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

def apply_entity_resolution(df: pd.DataFrame, col: str, manual_key_map: dict | None = None) -> pd.DataFrame:
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

    canonical = out.groupby("_supplier_key")[col].agg(pick_longest_name).to_dict()
    out[col] = out["_supplier_key"].map(canonical).fillna(out[col])
    out = out.drop(columns=["_supplier_key"])
    return out

# =========================================================
# HELPERS
# =========================================================
def read_csv_flexible(candidates):
    last_err = None
    for f in candidates:
        try:
            return pd.read_csv(f)
        except Exception as e:
            last_err = e
    raise last_err

def read_text_flexible(candidates):
    for f in candidates:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                return fp.read()
        except Exception:
            pass
    return ""

def safe_to_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def apply_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    q = query.strip().lower()
    return df[df["supplier_name"].astype(str).str.lower().str.contains(q, na=False)]

def find_best_col(df: pd.DataFrame, preferred: list[str]) -> str | None:
    cols = df.columns.tolist()
    low = {c: c.lower() for c in cols}
    for kw in preferred:
        for c in cols:
            if kw in low[c]:
                return c
    return None

# =========================================================
# PART CATEGORY
# =========================================================
def categorize_part(text: str) -> str:
    if pd.isna(text):
        return "Other / Unknown"
    t = str(text).lower()

    rules = [
        ("Motors / Actuation", ["motor", "actuator", "servo", "stepper", "gearbox"]),
        ("Controls / Electronics", ["controller", "pcb", "board", "sensor", "wire", "wiring", "harness", "electronic", "firmware", "chip", "connector"]),
        ("Fins / Aero Surfaces", ["fin", "aero", "wing", "stabilizer", "airfoil"]),
        ("Brackets / Machined & Fabricated", ["bracket", "sheet", "fabricat", "weld", "machin", "cnc", "milling", "turning", "laser", "cut", "bend"]),
        ("Fasteners / Hardware", ["bolt", "screw", "nut", "washer", "fastener", "thread", "rivet", "stud"]),
        ("Plastics / Polymer", ["plastic", "poly", "abs", "nylon", "resin", "injection", "mold", "polymer"]),
        ("Metals / Raw Material", ["aluminum", "steel", "stainless", "titanium", "alloy", "bar", "rod", "plate", "sheet metal"]),
        ("Packaging", ["pack", "crate", "box", "foam", "label", "pallet"]),
    ]
    for cat, kws in rules:
        if any(kw in t for kw in kws):
            return cat
    return "Other / Unknown"

# =========================================================
# SUPPLIER NOTES
# =========================================================
def parse_supplier_notes(notes_text: str) -> dict:
    notes = {}
    if not notes_text:
        return notes

    blocks = re.split(r"\n=+\n", notes_text)
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        first = b.splitlines()[0].strip()

        m = re.match(r"^([A-Z0-9 &/]+)\s*-\s*(.+)$", first, flags=re.IGNORECASE)
        if not m:
            continue

        supplier_raw = m.group(1).strip()
        descriptor = m.group(2).strip()
        k = normalize_supplier_key(supplier_raw)

        lines = [ln.strip() for ln in b.splitlines()[1:] if ln.strip()]
        bullets = []
        for ln in lines:
            bullets.append(ln[:160] + ("…" if len(ln) > 160 else ""))
            if len(bullets) >= 4:
                break

        notes[k] = {"descriptor": descriptor, "bullets": bullets}
    return notes

def note_snippet(supplier_notes: dict, supplier_name: str) -> str:
    k = normalize_supplier_key(supplier_name)
    n = supplier_notes.get(k)
    if not n:
        return ""
    desc = n.get("descriptor", "")
    bullets = n.get("bullets", [])
    line = desc
    if bullets:
        line = f"{desc} | {bullets[0]}"
    return line[:200] + ("…" if len(line) > 200 else "")

# =========================================================
# LOAD DATA
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
        "supplier_notes (1).txt",
        "SUPPLIER_NOTES.txt",
    ])

    manual_key_map = {
        "apex mfg": "apex manufacturing",
        "apex manufacturing inc": "apex manufacturing",
        "apex mfg inc": "apex manufacturing",
    }

    orders = apply_entity_resolution(orders, "supplier_name", manual_key_map)
    rfqs = apply_entity_resolution(rfqs, "supplier_name", manual_key_map)

    orders = safe_to_datetime(orders, "order_date")
    orders = safe_to_datetime(orders, "promised_date")
    orders = safe_to_datetime(orders, "actual_delivery_date")
    quality = safe_to_datetime(quality, "inspection_date")
    rfqs = safe_to_datetime(rfqs, "quote_date")

    return orders, quality, rfqs, notes_text

try:
    orders, quality, rfqs, supplier_notes_text = load_data()
    supplier_notes = parse_supplier_notes(supplier_notes_text)
    st.success("✅ Data loaded & supplier names normalized (entity resolution applied)")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# =========================================================
# DEFINITIONS
# =========================================================
with st.expander("ℹ️ Definitions & Scoring (how to interpret the dashboard)", expanded=False):
    st.markdown(
        """
**Core KPIs**
- **Total Spend ($)**: Sum of purchase order spend per supplier.
- **On-Time Rate (%)**: % of orders delivered on/before promised date.
- **Defect Rate (%)**: Avg rejected rate across inspections.
- **Avg RFQ Price ($/unit)**: Avg quoted unit price (unit varies by part).

**Scores**
- **Price Score (0–100)**: Lower avg price → higher score.
- **Performance Score (0–100)**: Composite of delivery, quality, and cost.

**Risk Flags**
- 🔴 **Quality Risk**: Defect Rate ≥ 8%
- 🟠 **Delivery Risk**: On-Time Rate ≤ 85%
- 🟡 **Cost Risk**: Price Score ≤ 40
- 🟢 **Strategic**: none triggered
        """
    )

# =========================================================
# MASTER KPIs
# =========================================================
required_orders_cols = {"order_id", "supplier_name", "po_amount", "promised_date", "actual_delivery_date"}
missing = required_orders_cols - set(orders.columns)
if missing:
    st.error(f"Orders file missing required columns: {sorted(list(missing))}")
    st.stop()

spend = orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index()
spend.columns = ["supplier_name", "total_spend"]

orders_kpi = orders.copy()
orders_kpi["on_time"] = (orders_kpi["actual_delivery_date"] <= orders_kpi["promised_date"]).astype(float)
on_time = orders_kpi.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"] * 100).round(1)
on_time = on_time.drop(columns=["on_time"])

if "order_id" not in quality.columns:
    st.error("Quality inspections file must contain 'order_id'.")
    st.stop()

q = quality.merge(orders[["order_id", "supplier_name"]], on="order_id", how="left")
if {"parts_rejected", "parts_inspected"}.issubset(set(q.columns)):
    q["defect_rate"] = (q["parts_rejected"] / q["parts_inspected"]).replace([pd.NA, float("inf")], 0)
else:
    q["defect_rate"] = 0.0

defects = q.groupby("supplier_name", dropna=False)["defect_rate"].mean().reset_index()
defects["defect_rate"] = (defects["defect_rate"] * 100).round(1)

if not {"supplier_name", "quoted_price"}.issubset(set(rfqs.columns)):
    st.error("RFQ responses file must contain 'supplier_name' and 'quoted_price'.")
    st.stop()

avg_price = rfqs.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index()
avg_price.columns = ["supplier_name", "avg_price"]
avg_price["avg_price"] = avg_price["avg_price"].round(2)

supplier_master = (
    spend.merge(on_time, on="supplier_name", how="left")
         .merge(defects, on="supplier_name", how="left")
         .merge(avg_price, on="supplier_name", how="left")
).fillna({"on_time_rate": 0.0, "defect_rate": 0.0, "avg_price": 0.0})

max_price = supplier_master["avg_price"].replace(0, pd.NA).max()
if pd.notna(max_price) and max_price > 0:
    supplier_master["price_score"] = (100 * (1 - (supplier_master["avg_price"] / max_price))).fillna(0).clip(0, 100)
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
# SEARCH
# =========================================================
st.subheader("🔎 Search Suppliers")
search_query = st.text_input(
    "Search by supplier name",
    placeholder="e.g., Apex, Stellar, TitanForge...",
    key="search_query",
)
filtered_master = apply_search(supplier_master, st.session_state["search_query"])

# =========================================================
# TABLE
# =========================================================
st.subheader(f"🏭 Unified Supplier Intelligence View (Top {TOP_N})")
tbl = with_rank(format_for_display(filtered_master, ["supplier_name", "total_spend", "on_time_rate", "defect_rate", "avg_price", "price_score", "performance_score", "risk_flag"]))
show_table(tbl, TOP_N)

# =========================================================
# CONSOLIDATION OPPORTUNITIES
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
    st.metric("Estimated Annual Savings via Consolidation (Model)", _fmt_money(float(tmp["estimated_savings"].sum())))
    tbl = with_rank(format_for_display(tmp.sort_values("estimated_savings", ascending=False), ["supplier_name", "total_spend", "avg_price", "price_delta_vs_best", "estimated_savings", "risk_flag"]))
    show_table(tbl, TOP_N)

# =========================================================
# ⚡ REAL-TIME DECISION SUPPORT
# =========================================================
st.markdown("---")
st.header("⚡ Real-Time Sourcing Decision Support")

# In-section reset button (what you were looking for)
rt_left, rt_right = st.columns([1, 3])
with rt_left:
    if st.button("🔄 Reset decision filters", key="reset_decision_filters", use_container_width=True):
        reset_filters()
        st.rerun()
with rt_right:
    st.caption("Resets decision due date, thresholds, weights, capability evidence, min-lines, and category scope.")

d1, d2 = st.columns([1, 2])
with d1:
    st.number_input("Decision due in (days)", min_value=1, max_value=120, step=1, key="decision_in_days")
with d2:
    decision_due = date.today() + timedelta(days=int(st.session_state["decision_in_days"]))
    st.info(f"📅 Decision due: **{decision_due.strftime('%b %d, %Y')}** (in **{int(st.session_state['decision_in_days'])}** days)")

c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
with c1:
    st.slider("Minimum On-Time Rate (%)", 0, 100, key="req_on_time")
with c2:
    st.slider("Maximum Defect Rate (%)", 0, 20, key="max_defects")
with c3:
    st.write("Weights (Delivery / Quality / Cost)")
    st.slider("Delivery weight", 0.0, 1.0, 0.45, 0.05, key="w_delivery")
    st.slider("Quality weight",  0.0, 1.0, 0.35, 0.05, key="w_quality")
    st.slider("Cost weight",     0.0, 1.0, 0.20, 0.05, key="w_cost")

w_sum = st.session_state["w_delivery"] + st.session_state["w_quality"] + st.session_state["w_cost"]
if w_sum == 0:
    st.session_state["w_delivery"], st.session_state["w_quality"], st.session_state["w_cost"] = 0.45, 0.35, 0.20
    w_sum = 1.0
w_delivery = st.session_state["w_delivery"] / w_sum
w_quality  = st.session_state["w_quality"] / w_sum
w_cost     = st.session_state["w_cost"] / w_sum

st.subheader("Scope by Part Category (capability filter)")
cap1, cap2, cap3 = st.columns([1.2, 1.2, 1.6])
with cap1:
    st.radio("Capability evidence", ["RFQs only", "Orders only", "Orders + RFQs"], horizontal=True, key="capability_source")
with cap2:
    st.slider("Minimum lines in category", 1, 10, step=1, key="min_lines")
with cap3:
    st.checkbox("Show category coverage counts", key="show_coverage")

orders_part_col = "part_description" if "part_description" in orders.columns else find_best_col(
    orders, ["part_description", "commodity", "category", "part", "component", "item", "material", "product", "description", "item_description"]
)
rfq_part_col = "part_description" if "part_description" in rfqs.columns else find_best_col(
    rfqs, ["part_description", "commodity", "category", "part", "component", "item", "material", "product", "description", "item_description"]
)

decision_orders = orders.copy()
decision_rfqs = rfqs.copy()
decision_orders["part_category"] = decision_orders[orders_part_col].apply(categorize_part) if orders_part_col else "Other / Unknown"
decision_rfqs["part_category"] = decision_rfqs[rfq_part_col].apply(categorize_part) if rfq_part_col else "Other / Unknown"

observed_categories = sorted(
    set(decision_orders["part_category"].dropna().astype(str).unique().tolist())
    | set(decision_rfqs["part_category"].dropna().astype(str).unique().tolist())
)
st.selectbox("Select part category", ["(All Categories)"] + observed_categories, key="category_choice")

def capability_counts(source_choice: str) -> pd.DataFrame:
    parts = []
    if source_choice in ("Orders only", "Orders + RFQs"):
        parts.append(decision_orders.groupby(["supplier_name", "part_category"]).size().reset_index(name="lines"))
    if source_choice in ("RFQs only", "Orders + RFQs"):
        parts.append(decision_rfqs.groupby(["supplier_name", "part_category"]).size().reset_index(name="lines"))
    if not parts:
        return pd.DataFrame(columns=["supplier_name", "part_category", "lines"])
    cc = pd.concat(parts, ignore_index=True)
    return cc.groupby(["supplier_name", "part_category"], as_index=False)["lines"].sum()

cap_counts = capability_counts(st.session_state["capability_source"])

if st.session_state["show_coverage"] and not cap_counts.empty:
    cov = cap_counts.copy()
    if st.session_state["category_choice"] != "(All Categories)":
        cov = cov[cov["part_category"] == st.session_state["category_choice"]]
    cov = cov.sort_values(["lines", "supplier_name"], ascending=[False, True])
    st.caption("Coverage = number of matching lines (based on selected capability evidence).")
    cov_disp = with_rank(format_for_display(cov, ["supplier_name", "part_category", "lines"]))
    show_table(cov_disp, max_rows=50)

if st.session_state["category_choice"] == "(All Categories)" or cap_counts.empty:
    eligible_suppliers = set(supplier_master["supplier_name"].astype(str).unique().tolist())
else:
    eligible = cap_counts[
        (cap_counts["part_category"] == st.session_state["category_choice"]) &
        (cap_counts["lines"] >= st.session_state["min_lines"])
    ]["supplier_name"].astype(str).unique().tolist()
    eligible_suppliers = set(eligible)

scoped_orders = decision_orders.copy()
scoped_rfqs = decision_rfqs.copy()

if st.session_state["category_choice"] != "(All Categories)":
    scoped_orders = scoped_orders[scoped_orders["part_category"] == st.session_state["category_choice"]]
    scoped_rfqs = scoped_rfqs[scoped_rfqs["part_category"] == st.session_state["category_choice"]]

scoped_orders = scoped_orders[scoped_orders["supplier_name"].astype(str).isin(eligible_suppliers)]
scoped_rfqs = scoped_rfqs[scoped_rfqs["supplier_name"].astype(str).isin(eligible_suppliers)]

scoped = len(scoped_orders) > 0 and len(eligible_suppliers) > 0

if not scoped:
    if st.session_state["category_choice"] != "(All Categories)":
        st.warning("No suppliers met the category filter (given evidence source + min line threshold). Showing overall ranking.")
    decision_kpi = supplier_master.copy()
else:
    spend_d = scoped_orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index()
    spend_d.columns = ["supplier_name", "total_spend"]

    scoped_orders_kpi = scoped_orders.copy()
    scoped_orders_kpi["on_time"] = (scoped_orders_kpi["actual_delivery_date"] <= scoped_orders_kpi["promised_date"]).astype(float)
    on_time_d = scoped_orders_kpi.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
    on_time_d["on_time_rate"] = (on_time_d["on_time"] * 100).round(1)
    on_time_d = on_time_d.drop(columns=["on_time"])

    qd = quality.merge(scoped_orders[["order_id", "supplier_name"]], on="order_id", how="inner")
    if {"parts_rejected", "parts_inspected"}.issubset(set(qd.columns)) and len(qd) > 0:
        qd["defect_rate"] = (qd["parts_rejected"] / qd["parts_inspected"]).replace([pd.NA, float("inf")], 0)
        defects_d = qd.groupby("supplier_name", dropna=False)["defect_rate"].mean().reset_index()
        defects_d["defect_rate"] = (defects_d["defect_rate"] * 100).round(1)
    else:
        defects_d = pd.DataFrame({"supplier_name": spend_d["supplier_name"], "defect_rate": 0.0})

    if len(scoped_rfqs) > 0:
        avg_price_d = scoped_rfqs.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index()
        avg_price_d.columns = ["supplier_name", "avg_price"]
        avg_price_d["avg_price"] = avg_price_d["avg_price"].round(2)
    else:
        avg_price_d = pd.DataFrame({"supplier_name": spend_d["supplier_name"], "avg_price": 0.0})

    decision_kpi = (
        spend_d.merge(on_time_d, on="supplier_name", how="left")
              .merge(defects_d, on="supplier_name", how="left")
              .merge(avg_price_d, on="supplier_name", how="left")
    ).fillna({"on_time_rate": 0.0, "defect_rate": 0.0, "avg_price": 0.0})

    max_price_d = decision_kpi["avg_price"].replace(0, pd.NA).max()
    decision_kpi["price_score"] = (100 * (1 - (decision_kpi["avg_price"] / max_price_d))).fillna(0).clip(0, 100) if pd.notna(max_price_d) and max_price_d > 0 else 0.0

    decision_kpi["performance_score"] = (
        (decision_kpi["on_time_rate"] * w_delivery) +
        ((100 - decision_kpi["defect_rate"]) * w_quality) +
        (decision_kpi["price_score"] * w_cost)
    ).round(1)

    decision_kpi["risk_flag"] = decision_kpi.apply(risk_flag, axis=1)

decision_kpi["fit"] = (decision_kpi["on_time_rate"] >= st.session_state["req_on_time"]) & (decision_kpi["defect_rate"] <= st.session_state["max_defects"])
decision_kpi["fit_status"] = decision_kpi["fit"].map({True: "✅ Fit", False: "❌ Not fit"})
decision_kpi["notes_hint"] = decision_kpi["supplier_name"].apply(lambda s: note_snippet(supplier_notes, s))

st.subheader("✅ Decision-Time Shortlist (ranked)")
decision_view = decision_kpi.sort_values(["fit", "performance_score"], ascending=[False, False])
tbl = with_rank(format_for_display(decision_view, ["supplier_name", "fit_status", "performance_score", "risk_flag", "on_time_rate", "defect_rate", "avg_price", "total_spend", "notes_hint"]))
show_table(tbl, TOP_N)
