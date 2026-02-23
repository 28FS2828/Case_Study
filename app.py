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
# RESET FILTERS (SESSION STATE) — SAFE PATTERN
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

RESET_ALL_FLAG = "__reset_all__"
RESET_DECISION_FLAG = "__reset_decision__"


def init_defaults():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_defaults():
    # Must run BEFORE widgets with these keys are created
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


init_defaults()

# Apply reset flags BEFORE any widgets are created
if st.session_state.get(RESET_ALL_FLAG) or st.session_state.get(RESET_DECISION_FLAG):
    apply_defaults()
    st.session_state.pop(RESET_ALL_FLAG, None)
    st.session_state.pop(RESET_DECISION_FLAG, None)

# Global reset button (top of app)
top_left, top_right = st.columns([1, 3])
with top_left:
    if st.button("🔄 Reset filters", key="reset_filters_top", use_container_width=True):
        st.session_state[RESET_ALL_FLAG] = True
        st.rerun()
with top_right:
    st.caption("Resets search + decision filters + capability scope back to defaults.")

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

    # New (Fix #2): apples-to-apples delta within same RFQ line (rfq_id)
    "avg_delta_vs_best": "Avg Delta vs Best (same part) ($/unit)",

    "on_time_rate": "On-Time Rate (%)",
    "defect_rate": "Defect Rate (%)",

    "price_score": "Price Score (0–100)",
    "performance_score": "Performance Score (0–100)",

    "estimated_savings": "Est. Savings ($)",
    "estimated_overpay": "Est. Overpay ($)",
    "defect_cost": "Est. Defect Cost ($)",

    "part_category": "Part Category",
    "lines": "# Lines",
    "rfqs": "# RFQs",
    "pct_not_best": "% Quotes Not Best",
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
# EXEC-FRIENDLY FORMATTING (ROBUST)
# We format values as STRINGS before st.dataframe(),
# so commas + no trailing zeros ALWAYS show.
# =========================================================
def _fmt_money(x):
    if pd.isna(x):
        return ""
    try:
        return "${:,.0f}".format(float(x))
    except Exception:
        return str(x)


def _fmt_money_2(x):
    if pd.isna(x):
        return ""
    try:
        return "${:,.2f}".format(float(x))
    except Exception:
        return str(x)


def _fmt_pct(x):
    if pd.isna(x):
        return ""
    try:
        return "{:.1f}%".format(float(x))
    except Exception:
        return str(x)


def _fmt_score(x):
    if pd.isna(x):
        return ""
    try:
        return "{:.1f}".format(float(x))
    except Exception:
        return str(x)


def dataframe_pretty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert selected columns into pretty strings so Streamlit always displays:
    - $ with commas (no trailing zeros)
    - $/unit with cents
    - % with 1 decimal
    - scores with 1 decimal
    """
    out = df.copy()

    for c in out.columns:
        # Money columns: "($)" but not per-unit
        if "($)" in c and "($/unit)" not in c:
            out[c] = out[c].apply(_fmt_money)
        # Per-unit money (includes our delta column too)
        elif "($/unit)" in c:
            out[c] = out[c].apply(_fmt_money_2)
        # Percent columns
        elif "(%)" in c:
            out[c] = out[c].apply(_fmt_pct)
        # Score columns
        elif "(0–100)" in c or "(0-100)" in c:
            out[c] = out[c].apply(_fmt_score)

    return out


def show_table(df: pd.DataFrame, max_rows: int = TOP_N):
    df_show = df.head(max_rows) if len(df) > max_rows else df
    st.dataframe(dataframe_pretty(df_show), use_container_width=True, hide_index=True)


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


def _add_part_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    part_col = "part_description" if "part_description" in out.columns else find_best_col(
        out, ["part_description", "commodity", "category", "part", "component", "item", "material", "product", "description", "item_description"]
    )
    if part_col:
        out["part_category"] = out[part_col].apply(categorize_part)
    else:
        out["part_category"] = "Other / Unknown"
    return out


# =========================================================
# FIX #2: APPLES-TO-APPLES PRICE DELTA (same part / same RFQ line)
# We compute delta vs best within the same rfq_id (preferred), else fallback to normalized part_description.
# =========================================================
def _pick_rfq_line_key(rfqs_df: pd.DataFrame) -> str | None:
    # Best: rfq_id indicates one RFQ line with multiple supplier quotes
    if "rfq_id" in rfqs_df.columns:
        return "rfq_id"
    # Next best: explicit line id / part number
    for c in ["rfq_line_id", "line_id", "part_number", "item_id", "part_id"]:
        if c in rfqs_df.columns:
            return c
    # Fallback: part_description (normalized)
    if "part_description" in rfqs_df.columns:
        return "part_description"
    return None


def _normalize_part_text(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def rfq_competitiveness_by_supplier(rfqs_df: pd.DataFrame, category_choice: str) -> pd.DataFrame:
    """
    Returns supplier-level RFQ pricing competitiveness in the selected part_category scope.

    Metrics:
      - avg_price_scope: mean quoted_price within scope
      - avg_delta_vs_best: mean(quoted_price - best_price_for_same_rfq_line)
      - pct_not_best: % of quotes that are NOT the best for that rfq line
      - lines: number of quotes (rows)
      - rfqs: number of distinct rfq lines (rfq_id or equivalent)
    """
    if rfqs_df is None or rfqs_df.empty:
        return pd.DataFrame(columns=["supplier_name", "avg_price_scope", "avg_delta_vs_best", "pct_not_best", "lines", "rfqs"])

    if not {"supplier_name", "quoted_price"}.issubset(set(rfqs_df.columns)):
        return pd.DataFrame(columns=["supplier_name", "avg_price_scope", "avg_delta_vs_best", "pct_not_best", "lines", "rfqs"])

    r = _add_part_category(rfqs_df)

    # Scope filter
    if category_choice != "(All Categories)":
        r = r[r["part_category"] == category_choice]

    if r.empty:
        return pd.DataFrame(columns=["supplier_name", "avg_price_scope", "avg_delta_vs_best", "pct_not_best", "lines", "rfqs"])

    # Clean price
    r = r.copy()
    r["quoted_price"] = pd.to_numeric(r["quoted_price"], errors="coerce")
    r = r[(r["quoted_price"].notna()) & (r["quoted_price"] > 0)]

    if r.empty:
        return pd.DataFrame(columns=["supplier_name", "avg_price_scope", "avg_delta_vs_best", "pct_not_best", "lines", "rfqs"])

    # Choose rfq line key
    line_key = _pick_rfq_line_key(r)
    if line_key is None:
        return pd.DataFrame(columns=["supplier_name", "avg_price_scope", "avg_delta_vs_best", "pct_not_best", "lines", "rfqs"])

    r["_rfq_line_key"] = r[line_key].astype(str)
    if line_key == "part_description":
        r["_rfq_line_key"] = r["_rfq_line_key"].apply(_normalize_part_text)

    # Best price per rfq line
    best = r.groupby("_rfq_line_key", dropna=False)["quoted_price"].min().reset_index().rename(columns={"quoted_price": "best_price_line"})
    r = r.merge(best, on="_rfq_line_key", how="left")

    r["delta_vs_best_line"] = (r["quoted_price"] - r["best_price_line"]).clip(lower=0)
    r["is_best"] = (r["delta_vs_best_line"] <= 1e-9).astype(int)

    # Aggregate to supplier
    g = r.groupby("supplier_name", dropna=False).agg(
        avg_price_scope=("quoted_price", "mean"),
        avg_delta_vs_best=("delta_vs_best_line", "mean"),
        lines=("quoted_price", "size"),
        rfqs=("_rfq_line_key", "nunique"),
        pct_not_best=("is_best", lambda s: 100.0 * (1.0 - (float(s.sum()) / float(len(s))) if len(s) else 0.0)),
    ).reset_index()

    g["avg_price_scope"] = g["avg_price_scope"].round(2)
    g["avg_delta_vs_best"] = g["avg_delta_vs_best"].round(2)
    g["pct_not_best"] = g["pct_not_best"].round(1)

    return g


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

**Fix #2: Apples-to-apples pricing competitiveness**
- **Avg Delta vs Best (same part) ($/unit)** is computed **within the same RFQ line** (uses `rfq_id` when available).
- This avoids comparing different products across suppliers.
- If the Part Category scope is **(All Categories)**, the delta is still apples-to-apples **within each RFQ line**, but spans multiple categories overall.
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

avg_price = rfqs.copy()
avg_price["quoted_price"] = pd.to_numeric(avg_price["quoted_price"], errors="coerce")
avg_price = avg_price[(avg_price["quoted_price"].notna()) & (avg_price["quoted_price"] > 0)]
avg_price = avg_price.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index()
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
st.text_input(
    "Search by supplier name",
    placeholder="e.g., Apex, Stellar, TitanForge...",
    key="search_query",
)
filtered_master = apply_search(supplier_master, st.session_state["search_query"])

# =========================================================
# TABLES
# =========================================================
st.subheader(f"🏭 Unified Supplier Intelligence View (Top {TOP_N})")
cols_master = ["supplier_name", "total_spend", "on_time_rate", "defect_rate", "avg_price", "price_score", "performance_score", "risk_flag"]
tbl = with_rank(format_for_display(filtered_master, cols_master))
show_table(tbl, TOP_N)

st.markdown(f"### 🔴 Highest Risk Suppliers (Top {TOP_N})")
severity_rank = {"🔴 Quality Risk": 0, "🟠 Delivery Risk": 1, "🟡 Cost Risk": 2, "🟢 Strategic": 3}
risk_tbl = supplier_master.copy()
risk_tbl["_sev"] = risk_tbl["risk_flag"].map(severity_rank).fillna(9)
risk_tbl = risk_tbl.sort_values(["_sev", "performance_score"], ascending=[True, True]).drop(columns=["_sev"])
risk_tbl = apply_search(risk_tbl, st.session_state["search_query"])
show_table(with_rank(format_for_display(risk_tbl, cols_master)), TOP_N)

st.markdown(f"### 🟢 Top Performing Suppliers (Top {TOP_N})")
top_tbl = apply_search(supplier_master.sort_values("performance_score", ascending=False), st.session_state["search_query"])
show_table(with_rank(format_for_display(top_tbl, cols_master)), TOP_N)

# =========================================================
# FIX #3 (refactor): Central function for pricing deltas + overpay model
# =========================================================
def build_pricing_impact(
    supplier_master_df: pd.DataFrame,
    rfqs_df: pd.DataFrame,
    category_choice: str
) -> pd.DataFrame:
    """
    Returns supplier-level impact model DF merged onto supplier_master.

    - Uses Fix #2: avg_delta_vs_best computed within same RFQ line (rfq_id preferred)
    - Uses est_units ~ spend / avg_price_scope (fallback to overall avg_price if no scoped rfqs)
    - estimated_overpay ~ avg_delta_vs_best * est_units
    """
    comp = rfq_competitiveness_by_supplier(rfqs_df, category_choice)

    out = supplier_master_df.copy()
    out = out.merge(comp, on="supplier_name", how="left")

    # Fill missing (suppliers with no RFQs in scope)
    out["avg_price_scope"] = out["avg_price_scope"].fillna(out["avg_price"])
    out["avg_delta_vs_best"] = out["avg_delta_vs_best"].fillna(0.0)
    out["lines"] = out["lines"].fillna(0).astype(int)
    out["rfqs"] = out["rfqs"].fillna(0).astype(int)
    out["pct_not_best"] = out["pct_not_best"].fillna(0.0)

    # Estimate units using spend and scoped avg price
    out["est_units"] = 0.0
    mask_price = out["avg_price_scope"] > 0
    out.loc[mask_price, "est_units"] = out.loc[mask_price, "total_spend"] / out.loc[mask_price, "avg_price_scope"]

    out["estimated_overpay"] = (out["avg_delta_vs_best"] * out["est_units"]).fillna(0.0)

    # For display consistency (keep "avg_price" as the used value)
    out["avg_price"] = out["avg_price_scope"].round(2)
    out["avg_delta_vs_best"] = out["avg_delta_vs_best"].round(2)

    return out


# =========================================================
# CONSOLIDATION OPPORTUNITIES (Fix #2 + #3)
# =========================================================
st.markdown("---")
st.header("💰 Supplier Consolidation Opportunities")

selected_cat = st.session_state.get("category_choice", "(All Categories)")
impact = build_pricing_impact(supplier_master, rfqs, selected_cat)

st.metric("Estimated Annual Savings via Consolidation (Model)", _fmt_money(float(impact["estimated_overpay"].sum())))

cols_cons = ["supplier_name", "total_spend", "avg_price", "avg_delta_vs_best", "estimated_overpay", "risk_flag", "rfqs", "lines", "pct_not_best"]
show_table(with_rank(format_for_display(impact.sort_values("estimated_overpay", ascending=False), cols_cons)), TOP_N)

scope_label = "All Categories" if selected_cat == "(All Categories)" else selected_cat
st.caption(
    f"Model: **Avg Delta vs Best** is computed within the same RFQ line (rfq_id when available) within scope = **{scope_label}**; "
    "estimated units approximated as spend / scoped avg RFQ price; overpay ~= avg delta * units."
)

# =========================================================
# ⚡ REAL-TIME DECISION SUPPORT
# =========================================================
st.markdown("---")
st.header("⚡ Real-Time Sourcing Decision Support")

# In-section reset button (safe: sets flag + reruns)
rt_left, rt_right = st.columns([1, 3])
with rt_left:
    if st.button("🔄 Reset decision filters", key="reset_decision_filters", use_container_width=True):
        st.session_state[RESET_DECISION_FLAG] = True
        st.rerun()
with rt_right:
    st.caption("Resets decision due date, thresholds, weights, capability evidence, min-lines, and category scope.")

# Decision timing inputs
d1, d2 = st.columns([1, 2])
with d1:
    st.number_input("Decision due in (days)", min_value=1, max_value=120, step=1, key="decision_in_days")
with d2:
    decision_due = date.today() + timedelta(days=int(st.session_state["decision_in_days"]))
    st.info(f"📅 Decision due: **{decision_due.strftime('%b %d, %Y')}** (in **{int(st.session_state['decision_in_days'])}** days)")

# Thresholds + weights
c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
with c1:
    st.slider("Minimum On-Time Rate (%)", 0, 100, key="req_on_time")
with c2:
    st.slider("Maximum Defect Rate (%)", 0, 20, key="max_defects")
with c3:
    st.write("Weights (Delivery / Quality / Cost)")
    st.slider("Delivery weight", 0.0, 1.0, 0.45, 0.05, key="w_delivery")
    st.slider("Quality weight", 0.0, 1.0, 0.35, 0.05, key="w_quality")
    st.slider("Cost weight", 0.0, 1.0, 0.20, 0.05, key="w_cost")

w_sum = st.session_state["w_delivery"] + st.session_state["w_quality"] + st.session_state["w_cost"]
if w_sum == 0:
    w_delivery, w_quality, w_cost = 0.45, 0.35, 0.20
else:
    w_delivery = st.session_state["w_delivery"] / w_sum
    w_quality = st.session_state["w_quality"] / w_sum
    w_cost = st.session_state["w_cost"] / w_sum

# Capability filter controls
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
    show_table(with_rank(format_for_display(cov, ["supplier_name", "part_category", "lines"])), max_rows=50)

# Eligible suppliers from capability evidence + min threshold
if st.session_state["category_choice"] == "(All Categories)" or cap_counts.empty:
    eligible_suppliers = set(supplier_master["supplier_name"].astype(str).unique().tolist())
else:
    eligible = cap_counts[
        (cap_counts["part_category"] == st.session_state["category_choice"]) &
        (cap_counts["lines"] >= st.session_state["min_lines"])
    ]["supplier_name"].astype(str).unique().tolist()
    eligible_suppliers = set(eligible)

# Scope orders/rfqs used for decision view
scoped_orders = decision_orders.copy()
scoped_rfqs = decision_rfqs.copy()

if st.session_state["category_choice"] != "(All Categories)":
    scoped_orders = scoped_orders[scoped_orders["part_category"] == st.session_state["category_choice"]]
    scoped_rfqs = scoped_rfqs[scoped_rfqs["part_category"] == st.session_state["category_choice"]]

scoped_orders = scoped_orders[scoped_orders["supplier_name"].astype(str).isin(eligible_suppliers)]
scoped_rfqs = scoped_rfqs[scoped_rfqs["supplier_name"].astype(str).isin(eligible_suppliers)]

scoped = len(scoped_orders) > 0 and len(eligible_suppliers) > 0

# Recompute decision KPIs within scope (or fallback)
if not scoped:
    if st.session_state["category_choice"] != "(All Categories)":
        st.warning("No suppliers met the category filter (given evidence source + min-line threshold). Showing overall ranking.")
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

    # Avg RFQ price within scope
    if len(scoped_rfqs) > 0:
        scoped_rfqs = scoped_rfqs.copy()
        scoped_rfqs["quoted_price"] = pd.to_numeric(scoped_rfqs["quoted_price"], errors="coerce")
        scoped_rfqs = scoped_rfqs[(scoped_rfqs["quoted_price"].notna()) & (scoped_rfqs["quoted_price"] > 0)]
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
    if pd.notna(max_price_d) and max_price_d > 0:
        decision_kpi["price_score"] = (100 * (1 - (decision_kpi["avg_price"] / max_price_d))).fillna(0).clip(0, 100)
    else:
        decision_kpi["price_score"] = 0.0

    decision_kpi["performance_score"] = (
        (decision_kpi["on_time_rate"] * w_delivery) +
        ((100 - decision_kpi["defect_rate"]) * w_quality) +
        (decision_kpi["price_score"] * w_cost)
    ).round(1)

    decision_kpi["risk_flag"] = decision_kpi.apply(risk_flag, axis=1)

# Fit logic + notes overlay
decision_kpi["fit"] = (decision_kpi["on_time_rate"] >= st.session_state["req_on_time"]) & (decision_kpi["defect_rate"] <= st.session_state["max_defects"])
decision_kpi["fit_status"] = decision_kpi["fit"].map({True: "✅ Fit", False: "❌ Not fit"})
decision_kpi["notes_hint"] = decision_kpi["supplier_name"].apply(lambda s: note_snippet(supplier_notes, s))

st.subheader("✅ Decision-Time Shortlist (ranked)")
decision_view = decision_kpi.sort_values(["fit", "performance_score"], ascending=[False, False])
cols_decision = [
    "supplier_name", "fit_status", "performance_score", "risk_flag",
    "on_time_rate", "defect_rate", "avg_price", "total_spend", "notes_hint"
]
show_table(with_rank(format_for_display(decision_view, cols_decision)), TOP_N)
st.metric("Suppliers meeting thresholds", f"{int(decision_kpi['fit'].sum())} / {len(decision_kpi)}")

# =========================================================
# FINANCIAL IMPACT (Fix #2 + #3)
# =========================================================
st.markdown("---")
st.header("💰 Estimated Financial Impact (Prototype)")

selected_cat_impact = st.session_state.get("category_choice", "(All Categories)")
impact_df = build_pricing_impact(supplier_master, rfqs, selected_cat_impact)

impact_df["defect_cost"] = impact_df["total_spend"] * (impact_df["defect_rate"] / 100.0) * 0.5
late_spend = float(impact_df.loc[impact_df["on_time_rate"] < 85, "total_spend"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Potential Cost Leakage ($)", _fmt_money(float(impact_df["estimated_overpay"].sum())))
c2.metric("Estimated Cost of Quality Issues ($)", _fmt_money(float(impact_df["defect_cost"].sum())))
c3.metric("Spend Exposed to Delivery Risk ($)", _fmt_money(late_spend))

with st.expander("Show impact drivers by supplier"):
    impact_cols = [
        "supplier_name",
        "total_spend",
        "avg_price",
        "avg_delta_vs_best",
        "estimated_overpay",
        "rfqs",
        "lines",
        "pct_not_best",
        "price_score",
        "defect_rate",
        "defect_cost",
        "on_time_rate",
        "performance_score",
        "risk_flag",
    ]
    show_table(with_rank(format_for_display(impact_df.sort_values("estimated_overpay", ascending=False), impact_cols)), TOP_N)

scope_label_imp = "All Categories" if selected_cat_impact == "(All Categories)" else selected_cat_impact
st.caption(f"Pricing delta benchmark is computed **within the same RFQ line** within scope: **{scope_label_imp}** (RFQs only).")

with st.expander("Debug: show column names"):
    st.write("Orders columns:", list(orders.columns))
    st.write("Quality columns:", list(quality.columns))
    st.write("RFQ columns:", list(rfqs.columns))
