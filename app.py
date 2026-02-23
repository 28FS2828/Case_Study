import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import altair as alt

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Hoth Procurement Command Center", layout="wide")
st.title("🏭 Hoth Industries: Strategic Procurement Command Center")
st.markdown("---")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
h1, h2, h3 {letter-spacing: -0.02em;}
div[data-testid="stMetricValue"] {font-size: 1.8rem;}
div[data-testid="stMetricLabel"] {font-size: 0.95rem;}
div[data-testid="stExpander"] {border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

TOP_N = 10

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

RESET_ALL_FLAG = "__reset_all__"
RESET_DECISION_FLAG = "__reset_decision__"


def init_defaults():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


init_defaults()

if st.session_state.get(RESET_ALL_FLAG) or st.session_state.get(RESET_DECISION_FLAG):
    apply_defaults()
    st.session_state.pop(RESET_ALL_FLAG, None)
    st.session_state.pop(RESET_DECISION_FLAG, None)

top_left, top_right = st.columns([1, 3])
with top_left:
    if st.button("🔄 Reset filters", key="reset_filters_top", use_container_width=True):
        st.session_state[RESET_ALL_FLAG] = True
        st.rerun()
with top_right:
    st.caption("Resets search + decision filters + capability scope back to defaults.")

# =========================================================
# RISK COLOR MAP
# =========================================================
RISK_ORDER = ["🔴 Quality Risk", "🟠 Delivery Risk", "🟡 Cost Risk", "🟢 Strategic"]
RISK_COLORS = ["#D62728", "#FF7F0E", "#F2C12E", "#2CA02C"]
risk_color_scale = alt.Scale(domain=RISK_ORDER, range=RISK_COLORS)

# =========================================================
# DISPLAY LABELS
# =========================================================
DISPLAY_COLS = {
    "supplier_name": "Supplier",
    "risk_flag": "Risk Flag",
    "fit_status": "Fit Status",
    "notes_hint": "Supplier Notes (Tribal Knowledge)",
    "total_spend": "Total Spend ($)",
    "spend_m": "Total Spend ($M)",
    "avg_price": "Avg RFQ Price ($/unit)",
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
    "orders": "# Orders",
}


def format_for_display(df: pd.DataFrame, cols: list) -> pd.DataFrame:
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
# FORMATTING HELPERS
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
    out = df.copy()
    for c in out.columns:
        if "($)" in c and "($/unit)" not in c:
            out[c] = out[c].apply(_fmt_money)
        elif "($/unit)" in c:
            out[c] = out[c].apply(_fmt_money_2)
        elif "(%)" in c:
            out[c] = out[c].apply(_fmt_pct)
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


def apply_entity_resolution(df: pd.DataFrame, col: str, manual_key_map: dict = None) -> pd.DataFrame:
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    out["_supplier_key"] = out[col].apply(normalize_supplier_key)
    if manual_key_map:
        out["_supplier_key"] = out["_supplier_key"].replace(manual_key_map)

    def pick_longest_name(values):
        vals = list(set([v.strip() for v in values if isinstance(v, str) and v.strip()]))
        if not vals:
            return ""
        vals.sort(key=lambda x: (-len(x), x))
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


def find_best_col(df: pd.DataFrame, preferred: list) -> str:
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
        ("Heat Exchangers", ["heat exchanger", "exchanger", "thermal", "coil", "radiator"]),
        ("Air Handling / Dampers", ["damper", "louver", "filter", "hepa", "diffuser", "grille", "duct", "fan", "blower", "air handling"]),
        ("Bearings / Seals", ["bearing", "seal", "bushing", "race"]),
        ("Shafts / Mechanical", ["shaft", "gear", "coupling", "hub", "sprocket", "pulley"]),
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
# APPLES-TO-APPLES PRICE DELTA (same part / same RFQ line)
# =========================================================
def _pick_rfq_line_key(rfqs_df: pd.DataFrame) -> str:
    if "rfq_id" in rfqs_df.columns:
        return "rfq_id"
    for c in ["rfq_line_id", "line_id", "part_number", "item_id", "part_id"]:
        if c in rfqs_df.columns:
            return c
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
    empty = pd.DataFrame(columns=["supplier_name", "avg_price_scope", "avg_delta_vs_best", "pct_not_best", "lines", "rfqs"])
    if rfqs_df is None or rfqs_df.empty:
        return empty
    if not {"supplier_name", "quoted_price"}.issubset(set(rfqs_df.columns)):
        return empty

    r = _add_part_category(rfqs_df)
    if category_choice != "(All Categories)":
        r = r[r["part_category"] == category_choice]
    if r.empty:
        return empty

    r = r.copy()
    r["quoted_price"] = pd.to_numeric(r["quoted_price"], errors="coerce")
    r = r[(r["quoted_price"].notna()) & (r["quoted_price"] > 0)]
    if r.empty:
        return empty

    line_key = _pick_rfq_line_key(r)
    if line_key is None:
        return empty

    r["_rfq_line_key"] = r[line_key].astype(str)
    if line_key == "part_description":
        r["_rfq_line_key"] = r["_rfq_line_key"].apply(_normalize_part_text)

    best = r.groupby("_rfq_line_key", dropna=False)["quoted_price"].min().reset_index().rename(columns={"quoted_price": "best_price_line"})
    r = r.merge(best, on="_rfq_line_key", how="left")
    r["delta_vs_best_line"] = (r["quoted_price"] - r["best_price_line"]).clip(lower=0)
    r["is_best"] = (r["delta_vs_best_line"] <= 1e-9).astype(int)

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
# PRICING IMPACT MODEL
# =========================================================
def build_pricing_impact(supplier_master_df: pd.DataFrame, rfqs_df: pd.DataFrame, category_choice: str) -> pd.DataFrame:
    comp = rfq_competitiveness_by_supplier(rfqs_df, category_choice)
    out = supplier_master_df.copy()
    out = out.merge(comp, on="supplier_name", how="left")
    out["avg_price_scope"] = out["avg_price_scope"].fillna(out["avg_price"])
    out["avg_delta_vs_best"] = out["avg_delta_vs_best"].fillna(0.0)
    out["lines"] = out["lines"].fillna(0).astype(int)
    out["rfqs"] = out["rfqs"].fillna(0).astype(int)
    out["pct_not_best"] = out["pct_not_best"].fillna(0.0)
    out["est_units"] = 0.0
    mask_price = out["avg_price_scope"] > 0
    out.loc[mask_price, "est_units"] = out.loc[mask_price, "total_spend"] / out.loc[mask_price, "avg_price_scope"]
    out["estimated_overpay"] = (out["avg_delta_vs_best"] * out["est_units"]).fillna(0.0)
    out["avg_price"] = out["avg_price_scope"].round(2)
    out["avg_delta_vs_best"] = out["avg_delta_vs_best"].round(2)
    return out


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
            bullets.append(ln[:200] + ("…" if len(ln) > 200 else ""))
            if len(bullets) >= 6:
                break
        notes[k] = {"descriptor": descriptor, "bullets": bullets, "raw": b}
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


def get_full_note(supplier_notes: dict, supplier_name: str) -> dict:
    k = normalize_supplier_key(supplier_name)
    return supplier_notes.get(k, {})


# =========================================================
# LOAD DATA — with file uploader fallback
# =========================================================
@st.cache_data
def load_data_from_paths(orders_path, quality_path, rfqs_path, notes_path):
    """Load from file paths (local dev mode)."""
    orders = pd.read_csv(orders_path)
    quality = pd.read_csv(quality_path)
    rfqs = pd.read_csv(rfqs_path)
    notes_text = read_text_flexible([notes_path])
    return orders, quality, rfqs, notes_text


@st.cache_data
def load_data_from_uploads(orders_bytes, quality_bytes, rfqs_bytes, notes_bytes):
    """Load from uploaded file bytes."""
    import io
    orders = pd.read_csv(io.BytesIO(orders_bytes))
    quality = pd.read_csv(io.BytesIO(quality_bytes))
    rfqs = pd.read_csv(io.BytesIO(rfqs_bytes))
    notes_text = notes_bytes.decode("utf-8", errors="ignore") if notes_bytes else ""
    return orders, quality, rfqs, notes_text


def try_load_local():
    """Attempt to load from known local/repo paths."""
    path_sets = [
        (
            "Copy_of_supplier_orders.csv",
            "Copy_of_quality_inspections.csv",
            "Copy_of_rfq_responses.csv",
            "supplier_notes.txt",
        ),
        (
            "Copy of supplier_orders.csv",
            "Copy of quality_inspections.csv",
            "Copy of rfq_responses.csv",
            "supplier_notes.txt",
        ),
        (
            "supplier_orders.csv",
            "quality_inspections.csv",
            "rfq_responses.csv",
            "supplier_notes.txt",
        ),
    ]
    for op, qp, rp, np in path_sets:
        try:
            orders = pd.read_csv(op)
            quality = pd.read_csv(qp)
            rfqs = pd.read_csv(rp)
            notes_text = read_text_flexible([np])
            return orders, quality, rfqs, notes_text
        except Exception:
            continue
    return None


def process_raw_data(orders, quality, rfqs, notes_text):
    """Apply entity resolution + type coercions."""
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


# ---- Try local load first; fall back to file uploader ----
local_result = try_load_local()

if local_result is not None:
    orders_raw, quality_raw, rfqs_raw, notes_text_raw = local_result
    orders, quality, rfqs, supplier_notes_text = process_raw_data(orders_raw, quality_raw, rfqs_raw, notes_text_raw)
    supplier_notes = parse_supplier_notes(supplier_notes_text)
    st.success("✅ Data loaded from local files. Entity resolution applied.")
else:
    st.warning("⚠️ Local data files not found. Please upload your files below.")
    with st.expander("📂 Upload Data Files", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            uf_orders = st.file_uploader("supplier_orders.csv", type=["csv"], key="uf_orders")
            uf_quality = st.file_uploader("quality_inspections.csv", type=["csv"], key="uf_quality")
        with col2:
            uf_rfqs = st.file_uploader("rfq_responses.csv", type=["csv"], key="uf_rfqs")
            uf_notes = st.file_uploader("supplier_notes.txt", type=["txt"], key="uf_notes")

    if all([uf_orders, uf_quality, uf_rfqs]):
        orders_raw, quality_raw, rfqs_raw, notes_text_raw = load_data_from_uploads(
            uf_orders.read(), uf_quality.read(), uf_rfqs.read(),
            uf_notes.read() if uf_notes else b""
        )
        orders, quality, rfqs, supplier_notes_text = process_raw_data(orders_raw, quality_raw, rfqs_raw, notes_text_raw)
        supplier_notes = parse_supplier_notes(supplier_notes_text)
        st.success("✅ Data loaded from uploads. Entity resolution applied.")
    else:
        st.info("Upload the three required CSV files to continue.")
        st.stop()

# Reload button
reload_col, _ = st.columns([1, 4])
with reload_col:
    if st.button("🔁 Reload data", key="reload_data"):
        st.cache_data.clear()
        st.rerun()

# =========================================================
# DEFINITIONS
# =========================================================
with st.expander("ℹ️ Definitions & Scoring", expanded=False):
    st.markdown("""
**Core KPIs**
- **Total Spend ($)**: Sum of purchase order spend per supplier.
- **On-Time Rate (%)**: % of orders with a valid delivery date, delivered on/before promised date. Orders with missing delivery dates are excluded (not counted as late).
- **Defect Rate (%)**: Avg rejected rate across inspections.
- **Avg RFQ Price ($/unit)**: Avg quoted unit price.

**Scores**
- **Price Score (0–100)**: Lower avg price → higher score.
- **Performance Score (0–100)**: Composite of delivery (45%), quality (35%), and cost (20%).

**Risk Flags**
- 🔴 **Quality Risk**: Defect Rate ≥ 8%
- 🟠 **Delivery Risk**: On-Time Rate ≤ 85%
- 🟡 **Cost Risk**: Price Score ≤ 40
- 🟢 **Strategic**: None triggered

**Pricing Competitiveness**
- **Avg Delta vs Best (same part)** is computed within the same RFQ line (uses `rfq_id` when available). Apples-to-apples comparison only.
""")

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

# FIX: Exclude NaT delivery dates from on-time calculation (don't count as late)
orders_kpi = orders.copy()
orders_kpi_valid = orders_kpi[orders_kpi["actual_delivery_date"].notna() & orders_kpi["promised_date"].notna()].copy()
orders_kpi_valid["on_time"] = (orders_kpi_valid["actual_delivery_date"] <= orders_kpi_valid["promised_date"]).astype(float)
on_time = orders_kpi_valid.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"] * 100).round(1)
on_time = on_time.drop(columns=["on_time"])

# Order count per supplier
order_counts = orders.groupby("supplier_name", dropna=False)["order_id"].nunique().reset_index()
order_counts.columns = ["supplier_name", "orders"]

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
         .merge(order_counts, on="supplier_name", how="left")
).fillna({"on_time_rate": 0.0, "defect_rate": 0.0, "avg_price": 0.0, "orders": 0})

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
# TABS
# =========================================================
tab_overview, tab_rfq_lookup, tab_decision, tab_trends, tab_financial = st.tabs([
    "📊 Executive Overview",
    "🔎 Before You Send That RFQ",
    "⚡ Decision Support",
    "📈 Trends",
    "💰 Financial Impact",
])

# =========================================================
# TAB 1: EXECUTIVE OVERVIEW
# =========================================================
with tab_overview:
    st.header("🧠 Executive Action Brief")

    _exec_scope = st.session_state.get("category_choice", "(All Categories)")
    impact_exec = build_pricing_impact(supplier_master, rfqs, _exec_scope)

    _total_spend_all = float(supplier_master["total_spend"].sum())
    supplier_master_exec = supplier_master.copy()
    supplier_master_exec["spend_share_pct"] = 0.0
    if _total_spend_all > 0:
        supplier_master_exec["spend_share_pct"] = (supplier_master_exec["total_spend"] / _total_spend_all * 100.0).round(1)

    def _dep_flag(pct: float) -> str:
        try:
            p = float(pct)
        except Exception:
            return ""
        if p >= 35:
            return "🔴 Critical"
        if p >= 20:
            return "🟠 High"
        if p >= 10:
            return "🟡 Moderate"
        return "🟢 Low"

    supplier_master_exec["dependence_risk"] = supplier_master_exec["spend_share_pct"].apply(_dep_flag)

    def switchability_by_supplier(rfqs_df: pd.DataFrame, category_choice: str) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["supplier_name", "avg_alternatives", "switchability"])
        if rfqs_df is None or rfqs_df.empty:
            return empty
        r = _add_part_category(rfqs_df)
        if category_choice != "(All Categories)":
            r = r[r["part_category"] == category_choice]
        if r.empty:
            return empty
        r = r.copy()
        r["quoted_price"] = pd.to_numeric(r["quoted_price"], errors="coerce")
        r = r[(r["quoted_price"].notna()) & (r["quoted_price"] > 0)]
        if r.empty:
            return empty
        line_key = _pick_rfq_line_key(r)
        if line_key is None:
            return empty
        r["_rfq_line_key"] = r[line_key].astype(str)
        if line_key == "part_description":
            r["_rfq_line_key"] = r["_rfq_line_key"].apply(_normalize_part_text)
        alt_counts = r.groupby("_rfq_line_key")["supplier_name"].nunique().reset_index(name="suppliers_for_line")
        r = r.merge(alt_counts, on="_rfq_line_key", how="left")
        r["alternatives"] = (r["suppliers_for_line"] - 1).clip(lower=0)
        g = r.groupby("supplier_name", dropna=False)["alternatives"].mean().reset_index()
        g = g.rename(columns={"alternatives": "avg_alternatives"})
        g["avg_alternatives"] = g["avg_alternatives"].round(1)

        def _sw(a):
            try:
                x = float(a)
            except Exception:
                return "LOW"
            if x >= 2.0:
                return "HIGH"
            if x >= 1.0:
                return "MED"
            return "LOW"

        g["switchability"] = g["avg_alternatives"].apply(_sw)
        return g

    sw = switchability_by_supplier(rfqs, _exec_scope)
    impact_exec = impact_exec.merge(sw, on="supplier_name", how="left")
    impact_exec["avg_alternatives"] = impact_exec["avg_alternatives"].fillna(0.0)
    impact_exec["switchability"] = impact_exec["switchability"].fillna("LOW")

    # Executive KPI strip
    k1, k2, k3, k4 = st.columns(4)
    late_spend_exec = float(supplier_master.loc[supplier_master["on_time_rate"] < 85, "total_spend"].sum())
    defect_cost_exec = float((supplier_master["total_spend"] * (supplier_master["defect_rate"] / 100.0) * 0.5).sum())
    pricing_leak_exec = float(impact_exec["estimated_overpay"].sum())
    critical_deps = int((supplier_master_exec["spend_share_pct"] >= 35).sum())

    k1.metric("Potential Pricing Leakage", _fmt_money(pricing_leak_exec))
    k2.metric("Spend Exposed to Delivery Risk", _fmt_money(late_spend_exec))
    k3.metric("Est. Cost of Quality Issues", _fmt_money(defect_cost_exec))
    k4.metric("Critical Supplier Dependencies", f"{critical_deps}")

    # Top 3 Actions
    st.subheader("Top 3 Actions (next 30–60 days)")
    actions = []
    cand_price = impact_exec.sort_values("estimated_overpay", ascending=False).head(15).copy()
    cand_price = cand_price[cand_price["avg_alternatives"] >= 1.0]
    if len(cand_price) > 0:
        r0 = cand_price.iloc[0]
        actions.append(f"**Negotiate or consolidate away from {r0['supplier_name']}** — est. leakage **{_fmt_money(float(r0['estimated_overpay']))}**; switchability **{r0['switchability']}** (avg alternatives ≈ {r0['avg_alternatives']}).")
    if {"total_spend", "defect_rate"}.issubset(supplier_master.columns):
        tmp_dc = supplier_master.copy()
        tmp_dc["defect_cost"] = tmp_dc["total_spend"] * (tmp_dc["defect_rate"] / 100.0) * 0.5
        r1 = tmp_dc.sort_values("defect_cost", ascending=False).head(1).iloc[0]
        actions.append(f"**Contain quality at {r1['supplier_name']}** — est. quality cost **{_fmt_money(float(r1['defect_cost']))}**; defect rate **{_fmt_pct(float(r1['defect_rate']))}**.")
    tmp_dep = supplier_master_exec.sort_values("spend_share_pct", ascending=False).head(1)
    if len(tmp_dep) > 0:
        r2 = tmp_dep.iloc[0]
        actions.append(f"**De-risk dependency on {r2['supplier_name']}** — **{_fmt_pct(float(r2['spend_share_pct']))}** of total spend; dependence risk **{r2['dependence_risk']}**. Establish 1–2 qualified backups.")
    for a in actions[:3]:
        st.markdown(f"- {a}")

    # Supplier Positioning Matrix
    st.subheader("Supplier Positioning Matrix")
    st.caption("Bubble size = spend. Hover for details. Top-left = high performance + competitive pricing (ideal).")

    pos = impact_exec.merge(
        supplier_master_exec[["supplier_name", "spend_share_pct", "dependence_risk"]],
        on="supplier_name", how="left"
    )
    pos["spend_m"] = (pos["total_spend"] / 1_000_000.0).round(2)

    chart = (
        alt.Chart(pos)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X("avg_delta_vs_best:Q", title="Avg Delta vs Best ($/unit) — lower is better"),
            y=alt.Y("performance_score:Q", title="Performance Score (0–100)", scale=alt.Scale(domain=[0, 100])),
            size=alt.Size("total_spend:Q", title="Spend ($)", legend=None),
            color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk Flag"),
            tooltip=[
                alt.Tooltip("supplier_name:N", title="Supplier"),
                alt.Tooltip("risk_flag:N", title="Risk"),
                alt.Tooltip("performance_score:Q", title="Performance", format=".1f"),
                alt.Tooltip("avg_delta_vs_best:Q", title="Δ vs best ($/unit)", format=".2f"),
                alt.Tooltip("avg_alternatives:Q", title="Avg alternatives", format=".1f"),
                alt.Tooltip("switchability:N", title="Switchability"),
                alt.Tooltip("spend_m:Q", title="Spend ($M)", format=".2f"),
                alt.Tooltip("spend_share_pct:Q", title="Spend share (%)", format=".1f"),
                alt.Tooltip("dependence_risk:N", title="Dependence risk"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

    # Supplier Tables
    st.subheader("🔎 Search Suppliers")
    st.text_input("Search by supplier name", placeholder="e.g., Apex, Stellar, QuickFab...", key="search_query")
    filtered_master = apply_search(supplier_master, st.session_state["search_query"])

    st.subheader(f"🏭 Unified Supplier Intelligence (Top {TOP_N})")
    cols_master = ["supplier_name", "orders", "total_spend", "on_time_rate", "defect_rate", "avg_price", "price_score", "performance_score", "risk_flag"]
    show_table(with_rank(format_for_display(filtered_master, cols_master)), TOP_N)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"#### 🔴 Highest Risk (Top {TOP_N})")
        severity_rank = {"🔴 Quality Risk": 0, "🟠 Delivery Risk": 1, "🟡 Cost Risk": 2, "🟢 Strategic": 3}
        risk_tbl = supplier_master.copy()
        risk_tbl["_sev"] = risk_tbl["risk_flag"].map(severity_rank).fillna(9)
        risk_tbl = risk_tbl.sort_values(["_sev", "performance_score"], ascending=[True, True]).drop(columns=["_sev"])
        risk_tbl = apply_search(risk_tbl, st.session_state["search_query"])
        show_table(with_rank(format_for_display(risk_tbl, cols_master)), TOP_N)
    with col_b:
        st.markdown(f"#### 🟢 Top Performing (Top {TOP_N})")
        top_tbl = apply_search(supplier_master.sort_values("performance_score", ascending=False), st.session_state["search_query"])
        show_table(with_rank(format_for_display(top_tbl, cols_master)), TOP_N)

    # 30-day plan
    st.subheader("🧭 30-Day Action Plan")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
**Week 1 — Triage**
- Review top 5 leakage suppliers and validate on 3–5 RFQ lines each
- Align thresholds with ops (OTD + quality) and define "do-not-touch" suppliers

**Week 2 — Commercial**
- Run price fact-base discussions with suppliers showing high deltas vs best
- Launch should-cost / target pricing for high-volume lines
        """)
    with p2:
        st.markdown("""
**Week 3 — Resilience**
- For any supplier with **≥35%** spend share: qualify at least **1 backup**
- For delivery-risk spend: confirm lead times, capacity, and expedite levers

**Week 4 — Lock in**
- Consolidate where switchability is HIGH/MED and performance is acceptable
- Deploy monthly supplier scorecard review cadence (OTD, defects, pricing index)
        """)


# =========================================================
# TAB 2: "BEFORE YOU SEND THAT RFQ" — SUPPLIER INTEL CARD
# NEW: This is the killer demo feature that directly solves the core pain point
# =========================================================
with tab_rfq_lookup:
    st.header("🔎 Before You Send That RFQ")
    st.markdown("""
> *"Last month Cody sent an RFQ to QuickFab Industries. Three weeks late, 30% reject rate.  
> Later we remembered we'd used them before in 2021 with the same issues!"*  
> — Graham, Engineering Director

**This tool gives your team instant supplier intelligence at the moment of decision.**
""")

    all_suppliers = sorted(supplier_master["supplier_name"].dropna().unique().tolist())
    selected_supplier = st.selectbox("Select or search a supplier:", ["— Select a supplier —"] + all_suppliers, key="rfq_lookup_supplier")

    if selected_supplier and selected_supplier != "— Select a supplier —":
        row = supplier_master[supplier_master["supplier_name"] == selected_supplier]
        if row.empty:
            st.warning("No data found for this supplier.")
        else:
            row = row.iloc[0]
            note = get_full_note(supplier_notes, selected_supplier)

            # Risk banner
            risk = row["risk_flag"]
            if "Quality" in risk:
                st.error(f"⛔ **{risk}** — Review quality history carefully before placing order.")
            elif "Delivery" in risk:
                st.warning(f"⚠️ **{risk}** — Factor in potential delays and buffer your timeline.")
            elif "Cost" in risk:
                st.info(f"💛 **{risk}** — This supplier is not price-competitive. Solicit alternatives.")
            else:
                st.success(f"✅ **{risk}** — This supplier is a strong performer. Proceed with confidence.")

            # KPI Cards
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("On-Time Rate", _fmt_pct(row["on_time_rate"]),
                      delta="✅ Above threshold" if row["on_time_rate"] >= 90 else "❌ Below 90%",
                      delta_color="normal" if row["on_time_rate"] >= 90 else "inverse")
            c2.metric("Defect Rate", _fmt_pct(row["defect_rate"]),
                      delta="✅ Acceptable" if row["defect_rate"] <= 5 else "❌ High",
                      delta_color="normal" if row["defect_rate"] <= 5 else "inverse")
            c3.metric("Performance Score", _fmt_score(row["performance_score"]) + " / 100")
            c4.metric("Total Spend", _fmt_money(row["total_spend"]))
            c5.metric("Avg RFQ Price", _fmt_money_2(row["avg_price"]) + " / unit")

            st.markdown("---")

            # Tribal knowledge / notes
            if note:
                st.subheader("📋 Tribal Knowledge (from supplier_notes.txt)")
                descriptor = note.get("descriptor", "")
                if descriptor:
                    st.markdown(f"**Internal Assessment: {descriptor}**")
                bullets = note.get("bullets", [])
                if bullets:
                    for b in bullets:
                        st.markdown(f"- {b}")
            else:
                st.info("No tribal knowledge notes found for this supplier.")

            st.markdown("---")

            # Order history for this supplier
            st.subheader("📦 Recent Order History")
            sup_orders = orders[orders["supplier_name"] == selected_supplier].copy()
            if not sup_orders.empty:
                sup_orders_valid = sup_orders[sup_orders["actual_delivery_date"].notna() & sup_orders["promised_date"].notna()].copy()
                sup_orders_valid["on_time"] = sup_orders_valid["actual_delivery_date"] <= sup_orders_valid["promised_date"]
                sup_orders_valid["days_late"] = (sup_orders_valid["actual_delivery_date"] - sup_orders_valid["promised_date"]).dt.days

                display_cols = [c for c in ["order_id", "part_description", "order_date", "promised_date", "actual_delivery_date", "po_amount"] if c in sup_orders.columns]
                if "on_time" in sup_orders_valid.columns:
                    sup_orders_valid["On Time?"] = sup_orders_valid["on_time"].map({True: "✅ Yes", False: "❌ No"})
                    sup_orders_valid["Days Late"] = sup_orders_valid["days_late"].apply(lambda x: str(int(x)) if x > 0 else "—")
                    display_cols_ext = display_cols + ["On Time?", "Days Late"]
                    show_table(sup_orders_valid[display_cols_ext].sort_values("order_date", ascending=False) if "order_date" in display_cols_ext else sup_orders_valid[display_cols_ext], max_rows=10)
                else:
                    show_table(sup_orders[display_cols].sort_values("order_date", ascending=False) if "order_date" in display_cols else sup_orders[display_cols], max_rows=10)
            else:
                st.info("No orders found for this supplier.")

            # Quality inspections for this supplier
            st.subheader("🔬 Quality Inspection History")
            sup_quality = q[q["supplier_name"] == selected_supplier].copy() if "supplier_name" in q.columns else pd.DataFrame()
            if not sup_quality.empty:
                q_cols = [c for c in ["order_id", "inspection_date", "parts_inspected", "parts_rejected", "rejection_reason", "rework_required"] if c in sup_quality.columns]
                show_table(sup_quality[q_cols].sort_values("inspection_date", ascending=False) if "inspection_date" in q_cols else sup_quality[q_cols], max_rows=10)
            else:
                st.info("No quality inspections found for this supplier.")

            # RFQ history
            st.subheader("💲 RFQ / Pricing History")
            sup_rfqs = rfqs[rfqs["supplier_name"] == selected_supplier].copy()
            if not sup_rfqs.empty:
                rfq_cols = [c for c in ["rfq_id", "part_description", "quote_date", "quoted_price", "lead_time_weeks", "notes"] if c in sup_rfqs.columns]
                show_table(sup_rfqs[rfq_cols].sort_values("quote_date", ascending=False) if "quote_date" in rfq_cols else sup_rfqs[rfq_cols], max_rows=10)

                # Compare to market (best price for same RFQ lines)
                if "rfq_id" in rfqs.columns and "quoted_price" in rfqs.columns:
                    sup_rfq_ids = set(sup_rfqs["rfq_id"].astype(str).unique())
                    market = rfqs[rfqs["rfq_id"].astype(str).isin(sup_rfq_ids)].copy()
                    market["quoted_price"] = pd.to_numeric(market["quoted_price"], errors="coerce")
                    market = market[market["quoted_price"].notna() & (market["quoted_price"] > 0)]
                    if not market.empty:
                        best_by_rfq = market.groupby("rfq_id")["quoted_price"].min().reset_index().rename(columns={"quoted_price": "best_market_price"})
                        sup_rfqs_comp = sup_rfqs.merge(best_by_rfq, on="rfq_id", how="left")
                        sup_rfqs_comp["quoted_price"] = pd.to_numeric(sup_rfqs_comp["quoted_price"], errors="coerce")
                        sup_rfqs_comp["vs_best"] = sup_rfqs_comp["quoted_price"] - sup_rfqs_comp["best_market_price"]
                        avg_premium = sup_rfqs_comp["vs_best"].mean()
                        if avg_premium > 0:
                            st.warning(f"📊 This supplier prices on average **{_fmt_money_2(avg_premium)}/unit above the best market quote** across comparable RFQ lines.")
                        elif avg_premium <= 0:
                            st.success(f"📊 This supplier prices on average **{_fmt_money_2(abs(avg_premium))}/unit below or equal to the best market quote** — competitive!")
            else:
                st.info("No RFQ data found for this supplier.")

            # Recommendation
            st.markdown("---")
            st.subheader("🤖 System Recommendation")
            rec_parts = []
            if row["on_time_rate"] < 85:
                rec_parts.append(f"**Delivery risk** (on-time rate: {_fmt_pct(row['on_time_rate'])}). Add buffer weeks to your timeline.")
            if row["defect_rate"] > 5:
                rec_parts.append(f"**Quality risk** (defect rate: {_fmt_pct(row['defect_rate'])}). Plan for incoming inspection and potential rework.")
            if row["price_score"] <= 40:
                rec_parts.append(f"**Pricing not competitive** (price score: {_fmt_score(row['price_score'])}/100). Solicit at least 2 alternative quotes.")
            if rec_parts:
                st.warning("**Proceed with caution:**\n\n" + "\n\n".join([f"- {r}" for r in rec_parts]))
            else:
                st.success(f"**Recommended supplier.** {selected_supplier} has strong performance history. Proceed.")

    else:
        st.info("👆 Select a supplier above to see their full intelligence card.")
        st.markdown("**Example: Try searching for 'QuickFab' to see a high-risk supplier, or 'Stellar' to see a top performer.**")


# =========================================================
# TAB 3: DECISION SUPPORT
# =========================================================
with tab_decision:
    st.header("⚡ Real-Time Sourcing Decision Support")

    rt_left, rt_right = st.columns([1, 3])
    with rt_left:
        if st.button("🔄 Reset decision filters", key="reset_decision_filters", use_container_width=True):
            st.session_state[RESET_DECISION_FLAG] = True
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
        st.write("**Scoring Weights**")
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

    # FIX: Show effective weights so user understands normalization
    st.caption(f"**Effective weights (normalized):** Delivery {w_delivery:.0%} / Quality {w_quality:.0%} / Cost {w_cost:.0%}")

    st.subheader("Scope by Part Category")
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
            st.warning("No suppliers met the category filter. Showing overall ranking.")
        decision_kpi = supplier_master.copy()
    else:
        spend_d = scoped_orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index()
        spend_d.columns = ["supplier_name", "total_spend"]

        scoped_orders_valid = scoped_orders[scoped_orders["actual_delivery_date"].notna() & scoped_orders["promised_date"].notna()].copy()
        scoped_orders_valid["on_time"] = (scoped_orders_valid["actual_delivery_date"] <= scoped_orders_valid["promised_date"]).astype(float)
        on_time_d = scoped_orders_valid.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
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

    decision_kpi["fit"] = (decision_kpi["on_time_rate"] >= st.session_state["req_on_time"]) & (decision_kpi["defect_rate"] <= st.session_state["max_defects"])
    decision_kpi["fit_status"] = decision_kpi["fit"].map({True: "✅ Fit", False: "❌ Not fit"})
    decision_kpi["notes_hint"] = decision_kpi["supplier_name"].apply(lambda s: note_snippet(supplier_notes, s))

    st.subheader("✅ Decision-Time Shortlist (ranked)")
    decision_view = decision_kpi.sort_values(["fit", "performance_score"], ascending=[False, False])
    cols_decision = ["supplier_name", "fit_status", "performance_score", "risk_flag", "on_time_rate", "defect_rate", "avg_price", "total_spend", "notes_hint"]
    show_table(with_rank(format_for_display(decision_view, cols_decision)), TOP_N)
    st.metric("Suppliers meeting thresholds", f"{int(decision_kpi['fit'].sum())} / {len(decision_kpi)}")

    # Consolidation Opportunities
    st.markdown("---")
    st.subheader("💰 Supplier Consolidation Opportunities")
    selected_cat = st.session_state.get("category_choice", "(All Categories)")
    impact = build_pricing_impact(supplier_master, rfqs, selected_cat)
    st.metric("Estimated Annual Savings via Consolidation", _fmt_money(float(impact["estimated_overpay"].sum())))
    cols_cons = ["supplier_name", "total_spend", "avg_price", "avg_delta_vs_best", "estimated_overpay", "risk_flag", "rfqs", "lines", "pct_not_best"]
    show_table(with_rank(format_for_display(impact.sort_values("estimated_overpay", ascending=False), cols_cons)), TOP_N)
    scope_label = "All Categories" if selected_cat == "(All Categories)" else selected_cat
    st.caption(f"Model: Avg Delta vs Best computed within same RFQ line, scope = **{scope_label}**.")


# =========================================================
# TAB 4: TRENDS
# NEW: On-time delivery trends over time (Mike's Monday morning meeting ask)
# =========================================================
with tab_trends:
    st.header("📈 Performance Trends Over Time")
    st.markdown("Monthly on-time delivery and spend trends — *the charts Mike wants for Monday morning.*")

    if "order_date" not in orders.columns:
        st.warning("No `order_date` column found. Cannot generate trends.")
    else:
        orders_trend = orders_kpi_valid.copy()
        orders_trend["month"] = orders_trend["order_date"].dt.to_period("M").dt.to_timestamp()

        # Top suppliers by spend for filtering
        top_suppliers_by_spend = supplier_master.sort_values("total_spend", ascending=False)["supplier_name"].head(8).tolist()
        trend_filter = st.multiselect(
            "Filter suppliers (default: top 6 by spend)",
            options=all_suppliers,
            default=top_suppliers_by_spend[:6],
            key="trend_supplier_filter"
        )
        if not trend_filter:
            trend_filter = top_suppliers_by_spend[:6]

        orders_trend_filtered = orders_trend[orders_trend["supplier_name"].isin(trend_filter)]

        if orders_trend_filtered.empty:
            st.info("No data for selected suppliers.")
        else:
            # Monthly on-time rate by supplier
            monthly_otr = orders_trend_filtered.groupby(["month", "supplier_name"])["on_time"].agg(["mean", "count"]).reset_index()
            monthly_otr.columns = ["month", "supplier_name", "on_time_rate", "order_count"]
            monthly_otr["on_time_rate"] = (monthly_otr["on_time_rate"] * 100).round(1)
            # Only show months with at least 1 order
            monthly_otr = monthly_otr[monthly_otr["order_count"] >= 1]

            st.subheader("On-Time Delivery Rate by Supplier (Monthly)")
            otr_chart = (
                alt.Chart(monthly_otr)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("on_time_rate:Q", title="On-Time Rate (%)", scale=alt.Scale(domain=[0, 105])),
                    color=alt.Color("supplier_name:N", title="Supplier"),
                    tooltip=[
                        alt.Tooltip("supplier_name:N", title="Supplier"),
                        alt.Tooltip("month:T", title="Month", format="%b %Y"),
                        alt.Tooltip("on_time_rate:Q", title="On-Time Rate (%)", format=".1f"),
                        alt.Tooltip("order_count:Q", title="Orders"),
                    ]
                )
                .properties(height=350)
            )
            # 90% threshold line
            threshold_line = alt.Chart(pd.DataFrame({"y": [90]})).mark_rule(
                color="red", strokeDash=[6, 4], opacity=0.7
            ).encode(y="y:Q")
            st.altair_chart((otr_chart + threshold_line).resolve_scale(color="independent"), use_container_width=True)
            st.caption("Red dashed line = 90% on-time threshold.")

            # Monthly spend by supplier
            st.subheader("Monthly Spend by Supplier")
            orders_with_cat = orders.copy()
            orders_with_cat["month"] = orders_with_cat["order_date"].dt.to_period("M").dt.to_timestamp()
            orders_with_cat_filtered = orders_with_cat[orders_with_cat["supplier_name"].isin(trend_filter)]
            monthly_spend = orders_with_cat_filtered.groupby(["month", "supplier_name"])["po_amount"].sum().reset_index()
            monthly_spend["spend_k"] = (monthly_spend["po_amount"] / 1000).round(1)

            spend_chart = (
                alt.Chart(monthly_spend)
                .mark_bar()
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("spend_k:Q", title="Spend ($K)"),
                    color=alt.Color("supplier_name:N", title="Supplier"),
                    tooltip=[
                        alt.Tooltip("supplier_name:N", title="Supplier"),
                        alt.Tooltip("month:T", title="Month", format="%b %Y"),
                        alt.Tooltip("spend_k:Q", title="Spend ($K)", format=".1f"),
                    ]
                )
                .properties(height=300)
            )
            st.altair_chart(spend_chart, use_container_width=True)

            # Defect rate over time
            st.subheader("Defect Rate Over Time (by Inspection Month)")
            if "inspection_date" in quality.columns:
                q_trend = q.copy()
                q_trend["month"] = q_trend["inspection_date"].dt.to_period("M").dt.to_timestamp()
                q_trend_filtered = q_trend[q_trend["supplier_name"].isin(trend_filter)]
                if not q_trend_filtered.empty:
                    monthly_defect = q_trend_filtered.groupby(["month", "supplier_name"])["defect_rate"].mean().reset_index()
                    monthly_defect["defect_rate_pct"] = (monthly_defect["defect_rate"] * 100).round(2)
                    defect_chart = (
                        alt.Chart(monthly_defect)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("month:T", title="Month"),
                            y=alt.Y("defect_rate_pct:Q", title="Avg Defect Rate (%)"),
                            color=alt.Color("supplier_name:N", title="Supplier"),
                            tooltip=[
                                alt.Tooltip("supplier_name:N", title="Supplier"),
                                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                                alt.Tooltip("defect_rate_pct:Q", title="Defect Rate (%)", format=".2f"),
                            ]
                        )
                        .properties(height=300)
                    )
                    defect_threshold = alt.Chart(pd.DataFrame({"y": [5]})).mark_rule(
                        color="orange", strokeDash=[6, 4], opacity=0.7
                    ).encode(y="y:Q")
                    st.altair_chart((defect_chart + defect_threshold).resolve_scale(color="independent"), use_container_width=True)
                    st.caption("Orange dashed line = 5% defect rate threshold.")
            else:
                st.info("No `inspection_date` column in quality data — trend not available.")

    # Cost of poor quality breakdown
    st.markdown("---")
    st.subheader("📊 Cost of Poor Quality by Supplier")
    copq = supplier_master.copy()
    copq["defect_cost"] = (copq["total_spend"] * (copq["defect_rate"] / 100.0) * 0.5).round(0)
    copq = copq[copq["defect_cost"] > 0].sort_values("defect_cost", ascending=False).head(10)

    if not copq.empty:
        copq_chart = (
            alt.Chart(copq)
            .mark_bar()
            .encode(
                x=alt.X("defect_cost:Q", title="Est. Cost of Quality Issues ($)"),
                y=alt.Y("supplier_name:N", sort="-x", title="Supplier"),
                color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk Flag"),
                tooltip=[
                    alt.Tooltip("supplier_name:N", title="Supplier"),
                    alt.Tooltip("defect_cost:Q", title="Est. Quality Cost ($)", format="$,.0f"),
                    alt.Tooltip("defect_rate:Q", title="Defect Rate (%)", format=".1f"),
                    alt.Tooltip("total_spend:Q", title="Total Spend ($)", format="$,.0f"),
                ]
            )
            .properties(height=320)
        )
        st.altair_chart(copq_chart, use_container_width=True)
        st.caption("Est. quality cost = 50% of spend × defect rate (conservative rework cost model).")
    else:
        st.info("No quality cost data to display.")


# =========================================================
# TAB 5: FINANCIAL IMPACT
# =========================================================
with tab_financial:
    st.header("💰 Estimated Financial Impact")

    selected_cat_impact = st.session_state.get("category_choice", "(All Categories)")
    impact_df = build_pricing_impact(supplier_master, rfqs, selected_cat_impact)
    impact_df["defect_cost"] = impact_df["total_spend"] * (impact_df["defect_rate"] / 100.0) * 0.5
    late_spend = float(impact_df.loc[impact_df["on_time_rate"] < 85, "total_spend"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Potential Pricing Leakage (Model)", _fmt_money(float(impact_df["estimated_overpay"].sum())))
    c2.metric("Est. Cost of Quality Issues", _fmt_money(float(impact_df["defect_cost"].sum())))
    c3.metric("Spend Exposed to Delivery Risk", _fmt_money(late_spend))

    st.markdown("---")

    # Pricing leakage waterfall-style bar chart
    st.subheader("Pricing Leakage by Supplier")
    leak_df = impact_df[impact_df["estimated_overpay"] > 0].sort_values("estimated_overpay", ascending=False).head(10)
    if not leak_df.empty:
        leak_chart = (
            alt.Chart(leak_df)
            .mark_bar()
            .encode(
                x=alt.X("estimated_overpay:Q", title="Estimated Overpay ($)"),
                y=alt.Y("supplier_name:N", sort="-x", title="Supplier"),
                color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk"),
                tooltip=[
                    alt.Tooltip("supplier_name:N", title="Supplier"),
                    alt.Tooltip("estimated_overpay:Q", title="Est. Overpay ($)", format="$,.0f"),
                    alt.Tooltip("avg_delta_vs_best:Q", title="Δ vs best ($/unit)", format=".2f"),
                    alt.Tooltip("pct_not_best:Q", title="% Quotes Not Best", format=".1f"),
                ]
            )
            .properties(height=320)
        )
        st.altair_chart(leak_chart, use_container_width=True)

    with st.expander("📋 Full impact driver table (all suppliers)"):
        impact_cols = [
            "supplier_name", "total_spend", "avg_price", "avg_delta_vs_best",
            "estimated_overpay", "rfqs", "lines", "pct_not_best",
            "price_score", "defect_rate", "defect_cost", "on_time_rate",
            "performance_score", "risk_flag",
        ]
        show_table(with_rank(format_for_display(impact_df.sort_values("estimated_overpay", ascending=False), impact_cols)), max_rows=50)

    scope_label_imp = "All Categories" if selected_cat_impact == "(All Categories)" else selected_cat_impact
    st.caption(f"Pricing delta computed **within the same RFQ line** (rfq_id), scope: **{scope_label_imp}**. Overpay model: avg_delta × estimated_units.")

    with st.expander("⚙️ Data limitations & production roadmap"):
        st.markdown("""
**This prototype is intentionally scoped to 3 hours of build time.** For production:
- Connect directly to SAP/ERP for accurate PO quantities and goods receipts
- Add commodity index benchmarks (e.g., aluminum LME) for should-cost modeling
- Weight pricing deltas by actual order quantities, not spend-derived unit estimates
- Track lead-time variability and expedite frequency as delivery-risk signals
- Predictive alerts: flag suppliers with declining trend in OTD or defect rate

**What this already does well:**
- Apples-to-apples pricing within same RFQ line (not cross-product comparisons)
- Entity resolution for messy supplier names (Apex Mfg / Apex Manufacturing Inc / APEX MFG)
- Integrates tribal knowledge (supplier_notes.txt) into every decision view
- On-time rate excludes missing delivery dates (doesn't count NaT as "late")
""")

    with st.expander("🔧 Debug: column names"):
        st.write("Orders:", list(orders.columns))
        st.write("Quality:", list(quality.columns))
        st.write("RFQs:", list(rfqs.columns))
