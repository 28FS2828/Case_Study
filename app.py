"""
Hoth Industries — Supplier Intelligence Platform
(Executive-polished, safe baseline: preserves core logic + calculations)
"""

import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import altair as alt

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Hoth Industries · Supplier Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
/* ---- Layout polish ---- */
.block-container { padding-top: 1.25rem; padding-bottom: 2.0rem; max-width: 1350px; }
hr { margin: 1.1rem 0; }

/* ---- Typography ---- */
h1 { font-size: 1.65rem; font-weight: 750; letter-spacing: -0.03em; margin-bottom: 0.25rem; }
h2 { font-size: 1.25rem; font-weight: 650; letter-spacing: -0.02em; margin-top: 0.25rem; }
h3 { font-size: 1.05rem; font-weight: 650; }

/* ---- Metrics ---- */
div[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 750; }
div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #6b7280; }

/* ---- Controls ---- */
div[data-testid="stTabs"] button { font-size: 0.92rem; font-weight: 550; }
div[data-testid="stExpander"] { border-radius: 12px; border: 1px solid #e5e7eb; }
div[data-testid="stDataFrame"] { border-radius: 10px; }

/* ---- Subtle card look for containers ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 12px !important;
  border: 1px solid #e5e7eb !important;
}

/* ---- Cleaner captions ---- */
small, .stCaption { color: #6b7280 !important; }

/* ---- Reduce extra whitespace between elements ---- */
div[data-testid="stVerticalBlock"] > div:has(> div.stMarkdown) { margin-bottom: 0.15rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CONSTANTS & DEFAULTS
# ─────────────────────────────────────────────
TOP_N = 10

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
    "show_coverage": False,
    "category_choice": "(All Categories)",
    "trend_supplier_filter": [],
}

RESET_FLAG = "__reset__"

RISK_ORDER = ["🔴 Quality Risk", "🟠 Delivery Risk", "🟡 Cost Risk", "🟢 Strategic"]
RISK_COLORS = ["#D62728", "#FF7F0E", "#F2C12E", "#2CA02C"]
risk_color_scale = alt.Scale(domain=RISK_ORDER, range=RISK_COLORS)

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "llc",
    "l.l.c",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "gmbh",
    "s.a",
    "sa",
}

DISPLAY_COLS = {
    "supplier_name": "Supplier",
    "risk_flag": "Risk",
    "fit_status": "Fit",
    "notes_hint": "Tribal Knowledge",
    "total_spend": "Total Spend ($)",
    "avg_price": "Avg Quote Price ($/unit)",
    "avg_delta_vs_best": "Avg Premium vs Best ($/unit)",
    "on_time_rate": "On-Time Rate (%)",
    "defect_rate": "Defect Rate (%)",
    "price_score": "Price Score (0-100)",
    "performance_score": "Performance Score (0-100)",
    "estimated_overpay": "Est. Pricing Leakage ($)",
    "defect_cost": "Est. Quality Cost ($)",
    "part_category": "Part Category",
    "lines": "# Lines",
    "rfqs": "# RFQs",
    "pct_not_best": "% Quotes Above Best (%)",
    "orders": "# Orders",
    "spend_share_pct": "Spend Share (%)",
    "avg_alternatives": "Avg Alternatives",
    "switchability": "Switchability",
}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_defaults():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


init_defaults()
if st.session_state.get(RESET_FLAG):
    apply_defaults()
    st.session_state.pop(RESET_FLAG, None)

# ─────────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────────
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
        elif "(0-100)" in c or "(0\u20131" in c:
            out[c] = out[c].apply(_fmt_score)
    return out


def format_for_display(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df[[c for c in cols if c in df.columns]].copy()
    return out.rename(columns={c: DISPLAY_COLS.get(c, c) for c in out.columns})


def with_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out.insert(0, "#", range(1, len(out) + 1))
    return out


def show_table(df: pd.DataFrame, max_rows: int = TOP_N):
    df_show = df.head(max_rows) if len(df) > max_rows else df
    st.dataframe(dataframe_pretty(df_show), use_container_width=True, hide_index=True)


def divider():
    st.markdown("---")


# ─────────────────────────────────────────────
# ENTITY RESOLUTION
# ─────────────────────────────────────────────
def normalize_supplier_key(name: str) -> str:
    if pd.isna(name):
        return ""
    s = re.sub(r"[^a-z0-9\s]", " ", str(name).lower().strip())
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split()
    while parts and parts[-1] in LEGAL_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def apply_entity_resolution(df: pd.DataFrame, col: str, manual_map: dict = None) -> pd.DataFrame:
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    out["_key"] = out[col].apply(normalize_supplier_key)
    if manual_map:
        out["_key"] = out["_key"].replace(manual_map)

    def pick_longest(vals):
        v = list(set([s.strip() for s in vals if isinstance(s, str) and s.strip()]))
        return sorted(v, key=lambda x: (-len(x), x))[0] if v else ""

    canonical = out.groupby("_key")[col].agg(pick_longest).to_dict()
    out[col] = out["_key"].map(canonical).fillna(out[col])
    return out.drop(columns=["_key"])


# ─────────────────────────────────────────────
# PART CATEGORIZATION
# ─────────────────────────────────────────────
PART_RULES = [
    ("Motors / Actuation", ["motor", "actuator", "servo", "stepper", "gearbox"]),
    (
        "Controls / Electronics",
        ["controller", "pcb", "board", "sensor", "wire", "harness", "electronic", "connector", "vfd", "plc"],
    ),
    ("Heat Exchangers", ["heat exchanger", "exchanger", "coil", "radiator"]),
    ("Air Handling / Dampers", ["damper", "louver", "filter", "hepa", "diffuser", "grille", "duct", "fan", "blower"]),
    ("Fins / Aero Surfaces", ["fin", "aero", "wing", "stabilizer", "airfoil"]),
    ("Brackets / Fabricated Parts", ["bracket", "fabricat", "weld", "machin", "cnc", "laser", "cut", "bend", "sheet"]),
    ("Shafts / Mechanical", ["shaft", "gear", "coupling", "hub", "pulley"]),
    ("Bearings / Seals", ["bearing", "seal", "bushing"]),
    ("Fasteners / Hardware", ["bolt", "screw", "nut", "washer", "fastener", "rivet"]),
    ("Metals / Raw Material", ["aluminum", "steel", "stainless", "titanium", "alloy", "bar", "rod", "plate"]),
    ("Plastics / Polymer", ["plastic", "nylon", "resin", "injection", "mold", "polymer"]),
    ("Packaging", ["pack", "crate", "box", "foam", "pallet"]),
]


def categorize_part(text: str) -> str:
    if pd.isna(text):
        return "Other / Unknown"
    t = str(text).lower()
    for cat, kws in PART_RULES:
        if any(kw in t for kw in kws):
            return cat
    return "Other / Unknown"


def _add_part_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col = next(
        (c for c in ["part_description", "commodity", "category", "component", "item", "description"] if c in out.columns),
        None,
    )
    out["part_category"] = out[col].apply(categorize_part) if col else "Other / Unknown"
    return out


# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────
def _pick_rfq_line_key(df: pd.DataFrame):
    for c in ["rfq_id", "rfq_line_id", "line_id", "part_number", "item_id", "part_description"]:
        if c in df.columns:
            return c
    return None


def _norm_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(x).lower().strip())).strip()


def rfq_competitiveness(rfqs_df: pd.DataFrame, cat: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["supplier_name", "avg_price_scope", "avg_delta_vs_best", "pct_not_best", "lines", "rfqs"])
    if rfqs_df is None or rfqs_df.empty:
        return empty
    if not {"supplier_name", "quoted_price"}.issubset(rfqs_df.columns):
        return empty

    r = _add_part_category(rfqs_df)
    if cat != "(All Categories)":
        r = r[r["part_category"] == cat]
    if r.empty:
        return empty

    r = r.copy()
    r["quoted_price"] = pd.to_numeric(r["quoted_price"], errors="coerce")
    r = r[r["quoted_price"].gt(0) & r["quoted_price"].notna()]
    if r.empty:
        return empty

    lk = _pick_rfq_line_key(r)
    if lk is None:
        return empty

    # Apples-to-apples: compare within the same RFQ line (or normalized description when needed)
    r["_lk"] = r[lk].astype(str).apply(_norm_text if lk == "part_description" else (lambda x: x))
    best = r.groupby("_lk")["quoted_price"].min().rename("best_price")
    r = r.join(best, on="_lk")
    r["delta"] = (r["quoted_price"] - r["best_price"]).clip(lower=0)
    r["is_best"] = (r["delta"] <= 1e-9).astype(int)

    g = (
        r.groupby("supplier_name", dropna=False)
        .agg(
            avg_price_scope=("quoted_price", "mean"),
            avg_delta_vs_best=("delta", "mean"),
            lines=("quoted_price", "size"),
            rfqs=("_lk", "nunique"),
            pct_not_best=("is_best", lambda s: 100 * (1 - (s.sum() / len(s))) if len(s) else 0),
        )
        .reset_index()
    )
    return g.round({"avg_price_scope": 2, "avg_delta_vs_best": 2, "pct_not_best": 1})


def build_pricing_impact(master: pd.DataFrame, rfqs_df: pd.DataFrame, cat: str) -> pd.DataFrame:
    comp = rfq_competitiveness(rfqs_df, cat)
    out = master.merge(comp, on="supplier_name", how="left")
    out["avg_price_scope"] = out["avg_price_scope"].fillna(out["avg_price"])
    out["avg_delta_vs_best"] = out["avg_delta_vs_best"].fillna(0.0)
    out["lines"] = out["lines"].fillna(0).astype(int)
    out["rfqs"] = out["rfqs"].fillna(0).astype(int)
    out["pct_not_best"] = out["pct_not_best"].fillna(0.0)

    mask = out["avg_price_scope"] > 0
    out["est_units"] = 0.0
    out.loc[mask, "est_units"] = out.loc[mask, "total_spend"] / out.loc[mask, "avg_price_scope"]

    out["estimated_overpay"] = (out["avg_delta_vs_best"] * out["est_units"]).fillna(0.0)
    out["avg_price"] = out["avg_price_scope"].round(2)
    out["avg_delta_vs_best"] = out["avg_delta_vs_best"].round(2)
    return out


def switchability(rfqs_df: pd.DataFrame, cat: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["supplier_name", "avg_alternatives", "switchability"])
    if rfqs_df is None or rfqs_df.empty:
        return empty

    r = _add_part_category(rfqs_df)
    if cat != "(All Categories)":
        r = r[r["part_category"] == cat]
    if r.empty:
        return empty

    r = r.copy()
    r["quoted_price"] = pd.to_numeric(r["quoted_price"], errors="coerce")
    r = r[r["quoted_price"].gt(0) & r["quoted_price"].notna()]
    if r.empty:
        return empty

    lk = _pick_rfq_line_key(r)
    if lk is None:
        return empty

    # Keep keying consistent with competitiveness logic for stability
    r["_lk"] = r[lk].astype(str).apply(_norm_text if lk == "part_description" else (lambda x: x))
    n = r.groupby("_lk")["supplier_name"].nunique().rename("n")
    r = r.join(n, on="_lk")
    r["alts"] = (r["n"] - 1).clip(lower=0)

    g = r.groupby("supplier_name", dropna=False)["alts"].mean().reset_index(name="avg_alternatives")
    g["avg_alternatives"] = g["avg_alternatives"].round(1)
    g["switchability"] = g["avg_alternatives"].apply(lambda a: "HIGH" if a >= 2 else ("MED" if a >= 1 else "LOW"))
    return g


# ─────────────────────────────────────────────
# SUPPLIER NOTES
# ─────────────────────────────────────────────
def parse_supplier_notes(text: str) -> dict:
    notes = {}
    if not text:
        return notes
    for b in re.split(r"\n=+\n", text):
        b = b.strip()
        if not b:
            continue
        first = b.splitlines()[0].strip()
        m = re.match(r"^([A-Z0-9 &/]+)\s*-\s*(.+)$", first, re.IGNORECASE)
        if not m:
            continue
        k = normalize_supplier_key(m.group(1).strip())
        bullets = [ln.strip()[:200] for ln in b.splitlines()[1:] if ln.strip()][:6]
        notes[k] = {"descriptor": m.group(2).strip(), "bullets": bullets}
    return notes


def note_snippet(notes: dict, name: str) -> str:
    n = notes.get(normalize_supplier_key(name), {})
    if not n:
        return ""
    parts = [n.get("descriptor", "")]
    if n.get("bullets"):
        parts.append(n["bullets"][0])
    line = " | ".join(p for p in parts if p)
    return line[:200] + ("..." if len(line) > 200 else "")


def get_full_note(notes: dict, name: str) -> dict:
    return notes.get(normalize_supplier_key(name), {})


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def read_text_flexible(paths):
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            pass
    return ""


def safe_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


MANUAL_KEY_MAP = {
    "apex mfg": "apex manufacturing",
    "apex manufacturing inc": "apex manufacturing",
    "apex mfg inc": "apex manufacturing",
}


def process_raw(orders, quality, rfqs, notes_text):
    orders = apply_entity_resolution(orders, "supplier_name", MANUAL_KEY_MAP)
    rfqs = apply_entity_resolution(rfqs, "supplier_name", MANUAL_KEY_MAP)
    for df, cols in [
        (orders, ["order_date", "promised_date", "actual_delivery_date"]),
        (quality, ["inspection_date"]),
        (rfqs, ["quote_date"]),
    ]:
        for c in cols:
            safe_dt(df, c)
    return orders, quality, rfqs, notes_text


def try_load_local():
    for op, qp, rp, np in [
        ("Copy_of_supplier_orders.csv", "Copy_of_quality_inspections.csv", "Copy_of_rfq_responses.csv", "supplier_notes.txt"),
        ("Copy of supplier_orders.csv", "Copy of quality_inspections.csv", "Copy of rfq_responses.csv", "supplier_notes.txt"),
        ("supplier_orders.csv", "quality_inspections.csv", "rfq_responses.csv", "supplier_notes.txt"),
    ]:
        try:
            o = pd.read_csv(op)
            q = pd.read_csv(qp)
            r = pd.read_csv(rp)
            n = read_text_flexible([np])
            return o, q, r, n
        except Exception:
            continue
    return None


# ---- Attempt load ----
local = try_load_local()
if local:
    orders, quality, rfqs, supplier_notes_text = process_raw(*local)
    supplier_notes = parse_supplier_notes(supplier_notes_text)
else:
    st.warning("Local data files not found. Please upload below.")
    with st.expander("Upload Data Files", expanded=True):
        c1, c2 = st.columns(2)
        uf_o = c1.file_uploader("supplier_orders.csv", type=["csv"])
        uf_q = c1.file_uploader("quality_inspections.csv", type=["csv"])
        uf_r = c2.file_uploader("rfq_responses.csv", type=["csv"])
        uf_n = c2.file_uploader("supplier_notes.txt", type=["txt"])
    if all([uf_o, uf_q, uf_r]):
        import io

        o = pd.read_csv(io.BytesIO(uf_o.read()))
        q0 = pd.read_csv(io.BytesIO(uf_q.read()))
        r = pd.read_csv(io.BytesIO(uf_r.read()))
        n = uf_n.read().decode("utf-8", "ignore") if uf_n else ""
        orders, quality, rfqs, supplier_notes_text = process_raw(o, q0, r, n)
        supplier_notes = parse_supplier_notes(supplier_notes_text)
    else:
        st.info("Upload all three CSV files to continue.")
        st.stop()

# ─────────────────────────────────────────────
# BUILD SUPPLIER MASTER
# ─────────────────────────────────────────────
required = {"order_id", "supplier_name", "po_amount", "promised_date", "actual_delivery_date"}
if missing := required - set(orders.columns):
    st.error(f"Orders file missing columns: {sorted(missing)}")
    st.stop()

spend = orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index(name="total_spend")

# On-time: exclude rows with missing dates
ot_valid = orders[orders["actual_delivery_date"].notna() & orders["promised_date"].notna()].copy()
ot_valid["on_time"] = (ot_valid["actual_delivery_date"] <= ot_valid["promised_date"]).astype(float)
on_time = ot_valid.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"] * 100).round(1)
on_time = on_time.drop(columns=["on_time"])

order_counts = orders.groupby("supplier_name", dropna=False)["order_id"].nunique().reset_index(name="orders")

if "order_id" not in quality.columns:
    st.error("quality_inspections.csv must contain 'order_id'.")
    st.stop()

q = quality.merge(orders[["order_id", "supplier_name"]], on="order_id", how="left")
if {"parts_rejected", "parts_inspected"}.issubset(q.columns):
    q["defect_rate"] = (q["parts_rejected"] / q["parts_inspected"]).replace([pd.NA, float("inf")], 0)
else:
    q["defect_rate"] = 0.0
defects = q.groupby("supplier_name", dropna=False)["defect_rate"].mean().reset_index()
defects["defect_rate"] = (defects["defect_rate"] * 100).round(1)

if not {"supplier_name", "quoted_price"}.issubset(rfqs.columns):
    st.error("rfq_responses.csv must contain 'supplier_name' and 'quoted_price'.")
    st.stop()

ap = rfqs.copy()
ap["quoted_price"] = pd.to_numeric(ap["quoted_price"], errors="coerce")
ap = ap[ap["quoted_price"].gt(0) & ap["quoted_price"].notna()]
ap = ap.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index(name="avg_price")
ap["avg_price"] = ap["avg_price"].round(2)

supplier_master = (
    spend.merge(on_time, on="supplier_name", how="left")
    .merge(defects, on="supplier_name", how="left")
    .merge(ap, on="supplier_name", how="left")
    .merge(order_counts, on="supplier_name", how="left")
).fillna({"on_time_rate": 0.0, "defect_rate": 0.0, "avg_price": 0.0, "orders": 0})

mp = supplier_master["avg_price"].replace(0, pd.NA).max()
supplier_master["price_score"] = (
    (100 * (1 - supplier_master["avg_price"] / mp)).clip(0, 100).fillna(0) if pd.notna(mp) and mp > 0 else 0.0
)
supplier_master["performance_score"] = (
    supplier_master["on_time_rate"] * 0.45 + (100 - supplier_master["defect_rate"]) * 0.35 + supplier_master["price_score"] * 0.20
).round(1)

total_spend_all = float(supplier_master["total_spend"].sum())
supplier_master["spend_share_pct"] = (
    (supplier_master["total_spend"] / total_spend_all * 100).round(1) if total_spend_all > 0 else 0.0
)


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
all_suppliers = sorted(supplier_master["supplier_name"].dropna().unique().tolist())

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
h1.title("Hoth Industries · Supplier Intelligence Platform")
h1.caption("Unified supplier performance, pricing competitiveness, and sourcing decision support.")
with h2:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        if st.button("Reset Filters", use_container_width=True, key="reset_top"):
            st.session_state[RESET_FLAG] = True
            st.rerun()
    with rc2:
        if st.button("Reload Data", use_container_width=True, key="reload_top"):
            # Safe even if you aren't using cache_data elsewhere
            st.cache_data.clear()
            st.rerun()

divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_overview, tab_intel, tab_decision, tab_trends, tab_financial = st.tabs(
    [
        "📊 Executive Overview",
        "🔍 Supplier Intel",
        "⚡ Sourcing Decision",
        "📈 Performance Trends",
        "💰 Financial Impact",
    ]
)

# ═══════════════════════════════════════════════════════
# TAB 1 · EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════
with tab_overview:
    cat_exec = st.session_state.get("category_choice", "(All Categories)")
    impact_exec = build_pricing_impact(supplier_master, rfqs, cat_exec)
    sw_exec = switchability(rfqs, cat_exec)
    impact_exec = impact_exec.merge(sw_exec, on="supplier_name", how="left")
    impact_exec["avg_alternatives"] = impact_exec["avg_alternatives"].fillna(0.0)
    impact_exec["switchability"] = impact_exec["switchability"].fillna("LOW")
    impact_exec["defect_cost"] = impact_exec["total_spend"] * (impact_exec["defect_rate"] / 100) * 0.5

    pricing_leak = float(impact_exec["estimated_overpay"].sum())
    late_spend = float(supplier_master.loc[supplier_master["on_time_rate"] < 85, "total_spend"].sum())
    defect_cost = float(impact_exec["defect_cost"].sum())
    n_quality_risk = int((supplier_master["risk_flag"] == "🔴 Quality Risk").sum())
    n_delivery_risk = int((supplier_master["risk_flag"] == "🟠 Delivery Risk").sum())

    # KPI Strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Est. Pricing Leakage",
        _fmt_money(pricing_leak),
        help="Avg price premium above best comparable RFQ quote (apples-to-apples), scaled across spend-derived units",
    )
    k2.metric(
        "Spend at Delivery Risk",
        _fmt_money(late_spend),
        help="Total spend with suppliers whose on-time rate is below 85%",
    )
    k3.metric(
        "Est. Quality / Rework Cost",
        _fmt_money(defect_cost),
        help="Defect rate × spend × 0.5 rework cost multiplier (conservative)",
    )
    k4.metric(
        "Suppliers with Active Risk",
        f"{n_quality_risk + n_delivery_risk}",
        help=f"Quality Risk: {n_quality_risk}  |  Delivery Risk: {n_delivery_risk}",
    )

    divider()

    # Priority Actions
    st.subheader("Priority Actions")
    st.caption("Auto-generated from your data: highest-leverage interventions first.")

    actions = []

    cand = impact_exec[impact_exec["avg_alternatives"] >= 1].sort_values("estimated_overpay", ascending=False)
    if len(cand):
        r = cand.iloc[0]
        actions.append(
            (
                "Pricing",
                f"**Renegotiate or re-source {r['supplier_name']}**",
                f"Est. {_fmt_money(float(r['estimated_overpay']))} annual leakage · "
                f"{r['switchability']} switchability · {int(r['avg_alternatives'])} qualified alternative(s)",
            )
        )

    tmp = impact_exec.sort_values("defect_cost", ascending=False)
    if len(tmp) and float(tmp.iloc[0]["defect_rate"]) > 0:
        r = tmp.iloc[0]
        actions.append(
            (
                "Quality",
                f"**Place {r['supplier_name']} on corrective action**",
                f"Defect rate {_fmt_pct(float(r['defect_rate']))} · Est. {_fmt_money(float(r['defect_cost']))} rework cost",
            )
        )

    tmp2 = supplier_master.sort_values("spend_share_pct", ascending=False)
    if len(tmp2) and float(tmp2.iloc[0]["spend_share_pct"]) >= 20:
        r = tmp2.iloc[0]
        actions.append(
            (
                "Concentration",
                f"**Qualify backup supplier(s) for {r['supplier_name']}**",
                f"{_fmt_pct(float(r['spend_share_pct']))} of total spend concentrated — single-point-of-failure exposure",
            )
        )

    badge_colors = {"Pricing": "#FF7F0E", "Quality": "#D62728", "Concentration": "#6366f1"}

    for label, title, detail in actions[:3]:
        with st.container(border=True):
            c1, c2 = st.columns([1, 9])
            c1.markdown(
                f"<div style='background:{badge_colors.get(label,'#6b7280')};color:white;"
                f"border-radius:8px;padding:6px 8px;font-size:0.75rem;font-weight:800;"
                f"text-align:center;margin-top:2px'>{label.upper()}</div>",
                unsafe_allow_html=True,
            )
            c2.markdown(title)
            c2.caption(detail)

    divider()

    # Positioning Matrix
    st.subheader("Supplier Positioning Matrix")
    st.caption(
        "Bubble size = total spend. Hover for details. "
        "**Ideal suppliers sit top-left**: high performance score, low pricing premium."
    )

    pos = impact_exec.merge(supplier_master[["supplier_name", "spend_share_pct"]], on="supplier_name", how="left")
    pos["spend_m"] = (pos["total_spend"] / 1e6).round(3)

    matrix = (
        alt.Chart(pos)
        .mark_circle(opacity=0.84, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X(
                "avg_delta_vs_best:Q",
                title="Avg Pricing Premium vs Best Quote ($/unit) — lower is better",
                scale=alt.Scale(zero=True),
            ),
            y=alt.Y(
                "performance_score:Q",
                title="Performance Score (0–100) — higher is better",
                scale=alt.Scale(domain=[0, 100]),
            ),
            size=alt.Size("total_spend:Q", legend=None, scale=alt.Scale(range=[90, 2300])),
            color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk"),
            tooltip=[
                alt.Tooltip("supplier_name:N", title="Supplier"),
                alt.Tooltip("risk_flag:N", title="Risk"),
                alt.Tooltip("performance_score:Q", title="Performance", format=".1f"),
                alt.Tooltip("on_time_rate:Q", title="On-Time Rate (%)", format=".1f"),
                alt.Tooltip("defect_rate:Q", title="Defect Rate (%)", format=".1f"),
                alt.Tooltip("avg_delta_vs_best:Q", title="Price Premium ($/unit)", format=".2f"),
                alt.Tooltip("switchability:N", title="Switchability"),
                alt.Tooltip("spend_m:Q", title="Spend ($M)", format=".3f"),
                alt.Tooltip("spend_share_pct:Q", title="Spend Share (%)", format=".1f"),
            ],
        )
        .properties(height=410)
    )

    ref_line = alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(color="#d1d5db", strokeDash=[5, 5], strokeWidth=1.5).encode(
        y="y:Q"
    )
    st.altair_chart(matrix + ref_line, use_container_width=True)

    divider()

    # Supplier table
    col_s, _ = st.columns([2, 2])
    search_q = col_s.text_input("Search suppliers", placeholder="Type supplier name...", key="search_query")

    filtered = supplier_master.copy()
    if search_q:
        filtered = filtered[filtered["supplier_name"].str.lower().str.contains(search_q.lower(), na=False)]

    cols_tbl = ["supplier_name", "orders", "total_spend", "on_time_rate", "defect_rate", "avg_price", "performance_score", "risk_flag"]
    show_table(with_rank(format_for_display(filtered.sort_values("performance_score", ascending=False), cols_tbl)), TOP_N)

    with st.expander("How scores & risk flags are calculated"):
        st.markdown(
            """
| Metric | Definition |
|---|---|
| **On-Time Rate** | % of orders delivered on or before promised date (missing delivery dates excluded, not penalized) |
| **Defect Rate** | Average rejection rate across quality inspections |
| **Performance Score** | Delivery 45% + Quality 35% + Price Competitiveness 20% |
| **Price Score** | 100 × (1 − avg_price / max_avg_price) — higher score means cheaper relative to peer group |
| **🔴 Quality Risk** | Defect Rate ≥ 8% |
| **🟠 Delivery Risk** | On-Time Rate ≤ 85% |
| **🟡 Cost Risk** | Price Score ≤ 40/100 |
| **🟢 Strategic** | No risk flags triggered |
"""
        )

# ═══════════════════════════════════════════════════════
# TAB 2 · SUPPLIER INTEL
# ═══════════════════════════════════════════════════════
with tab_intel:
    st.subheader("Supplier Intelligence Card")
    st.caption(
        "Look up any supplier before placing an order. "
        "Surfaces performance, internal notes, pricing competitiveness, and a clear recommendation — in one view."
    )

    selected = st.selectbox("Select a supplier:", ["— Choose a supplier —"] + all_suppliers, key="intel_supplier")

    if selected and selected != "— Choose a supplier —":
        row = supplier_master[supplier_master["supplier_name"] == selected]
        if row.empty:
            st.warning("No data found for this supplier.")
        else:
            row = row.iloc[0]
            note = get_full_note(supplier_notes, selected)
            risk = row["risk_flag"]

            # Executive alert banner
            if "Quality" in risk:
                st.error(f"STOP — {risk}: Review quality history before awarding new work.")
            elif "Delivery" in risk:
                st.warning(f"CAUTION — {risk}: Add schedule buffer / expedite plan.")
            elif "Cost" in risk:
                st.info(f"NOTE — {risk}: Not price-competitive. Solicit at least 2 alternatives.")
            else:
                st.success(f"CLEAR — {risk}: Consistently strong performer.")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(
                "On-Time Rate",
                _fmt_pct(row["on_time_rate"]),
                delta=("Above 90% target" if row["on_time_rate"] >= 90 else "Below 90% target"),
                delta_color=("normal" if row["on_time_rate"] >= 90 else "inverse"),
            )
            m2.metric(
                "Defect Rate",
                _fmt_pct(row["defect_rate"]),
                delta=("Within 5% threshold" if row["defect_rate"] <= 5 else "Exceeds 5% threshold"),
                delta_color=("normal" if row["defect_rate"] <= 5 else "inverse"),
            )
            m3.metric("Performance Score", _fmt_score(row["performance_score"]) + " / 100")
            m4.metric("Total Spend", _fmt_money(row["total_spend"]))
            m5.metric("Avg Quoted Price", _fmt_money_2(row["avg_price"]) + " /unit")

            divider()
            left, right = st.columns([1.05, 1])

            with left:
                # Tribal knowledge
                if note:
                    st.markdown("**Internal Notes (Tribal Knowledge)**")
                    desc = note.get("descriptor", "")
                    if desc:
                        st.markdown(f"*Team assessment: **{desc}***")
                    for b in note.get("bullets", []):
                        st.markdown(f"- {b}")
                else:
                    st.info("No internal notes on file for this supplier.")

                divider()
                st.markdown("**Recommendation**")
                rec = []
                if row["on_time_rate"] < 85:
                    rec.append(
                        f"Add **{max(1, round((90 - row['on_time_rate']) / 10))} week(s)** schedule buffer — on-time rate is {_fmt_pct(row['on_time_rate'])}."
                    )
                if row["defect_rate"] > 5:
                    rec.append(f"Require **100% incoming inspection** — defect rate is {_fmt_pct(row['defect_rate'])}.")
                if row["price_score"] <= 40:
                    rec.append(f"**Solicit competing quotes** — price score is {_fmt_score(row['price_score'])}/100.")

                if rec:
                    for r_ in rec:
                        st.warning(r_)
                else:
                    st.success(f"No concerns flagged. **{selected}** is a reliable sourcing choice.")

                # Pricing vs market
                if "rfq_id" in rfqs.columns:
                    sup_rfqs_raw = rfqs[rfqs["supplier_name"] == selected].copy()
                    sup_rfqs_raw["quoted_price"] = pd.to_numeric(sup_rfqs_raw["quoted_price"], errors="coerce")
                    sup_rfqs_raw = sup_rfqs_raw[sup_rfqs_raw["quoted_price"].notna()]
                    if not sup_rfqs_raw.empty:
                        sup_ids = set(sup_rfqs_raw["rfq_id"].astype(str))
                        mkt = rfqs[rfqs["rfq_id"].astype(str).isin(sup_ids)].copy()
                        mkt["quoted_price"] = pd.to_numeric(mkt["quoted_price"], errors="coerce")
                        mkt = mkt[mkt["quoted_price"].gt(0) & mkt["quoted_price"].notna()]
                        if not mkt.empty:
                            best_mkt = mkt.groupby("rfq_id")["quoted_price"].min().rename("best")
                            comp_df = sup_rfqs_raw.join(best_mkt, on="rfq_id")
                            premium = (comp_df["quoted_price"] - comp_df["best"]).mean()
                            divider()
                            st.markdown("**Pricing vs. Market**")
                            if premium > 0:
                                st.warning(
                                    f"Avg {_fmt_money_2(premium)}/unit **above** best market quote across comparable RFQ lines."
                                )
                            else:
                                st.success(
                                    f"Avg {_fmt_money_2(abs(premium))}/unit **at or below** best market quote — competitive."
                                )

            with right:
                st.markdown("**Recent Order History**")
                sup_orders = orders[orders["supplier_name"] == selected].copy()
                if not sup_orders.empty:
                    sov = sup_orders[sup_orders["actual_delivery_date"].notna() & sup_orders["promised_date"].notna()].copy()
                    sov["on_time"] = sov["actual_delivery_date"] <= sov["promised_date"]
                    sov["days_diff"] = (sov["actual_delivery_date"] - sov["promised_date"]).dt.days
                    sov["Status"] = sov["on_time"].map({True: "✅ On Time", False: "❌ Late"})
                    sov["Variance"] = sov["days_diff"].apply(
                        lambda x: f"-{abs(int(x))}d early" if x < 0 else (f"+{int(x)}d late" if x > 0 else "On schedule")
                    )
                    disp = [c for c in ["order_id", "part_description", "order_date", "Status", "Variance", "po_amount"] if c in sov.columns or c in ["Status", "Variance"]]
                    show_table(
                        sov[disp].sort_values("order_date", ascending=False) if "order_date" in disp else sov[disp],
                        max_rows=8,
                    )
                else:
                    st.info("No order history found.")

                st.markdown("**Quality Inspections**")
                sup_q = q[q["supplier_name"] == selected].copy() if "supplier_name" in q.columns else pd.DataFrame()
                if not sup_q.empty:
                    q_disp = [c for c in ["inspection_date", "parts_inspected", "parts_rejected", "rejection_reason", "rework_required"] if c in sup_q.columns]
                    sup_q_show = sup_q[q_disp].copy()
                    if {"parts_inspected", "parts_rejected"}.issubset(sup_q_show.columns):
                        sup_q_show["Defect %"] = (sup_q_show["parts_rejected"] / sup_q_show["parts_inspected"] * 100).round(1).astype(str) + "%"
                    show_table(
                        sup_q_show.sort_values("inspection_date", ascending=False) if "inspection_date" in sup_q_show.columns else sup_q_show,
                        max_rows=8,
                    )
                else:
                    st.info("No inspection records found.")
    else:
        st.info("Select a supplier above to view their full intelligence card.")
        st.markdown(
            """
**Try these:**
- **QuickFab Industries** — High-risk, documented repeat failures  
- **Stellar Metalworks** — Gold standard performer  
- **Apex Manufacturing** — Names consolidated from messy variants in the source data
"""
        )

# ═══════════════════════════════════════════════════════
# TAB 3 · SOURCING DECISION
# ═══════════════════════════════════════════════════════
with tab_decision:
    st.subheader("Real-Time Sourcing Decision Support")
    st.caption("Set requirements and generate a ranked shortlist of qualified suppliers for a part category.")

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        st.number_input("Decision needed in (days)", min_value=1, max_value=120, step=1, key="decision_in_days")
        deadline = date.today() + timedelta(days=int(st.session_state["decision_in_days"]))
        st.caption(f"Deadline: **{deadline.strftime('%b %d, %Y')}**")
    with ctrl2:
        st.slider("Min On-Time Rate (%)", 0, 100, key="req_on_time")
        st.slider("Max Defect Rate (%)", 0, 20, key="max_defects")
    with ctrl3:
        st.markdown("**Scoring Weights**")
        wc1, wc2, wc3 = st.columns(3)
        wc1.slider("Delivery", 0.0, 1.0, step=0.05, key="w_delivery")
        wc2.slider("Quality", 0.0, 1.0, step=0.05, key="w_quality")
        wc3.slider("Cost", 0.0, 1.0, step=0.05, key="w_cost")
        ws = st.session_state["w_delivery"] + st.session_state["w_quality"] + st.session_state["w_cost"]
        if ws == 0:
            wd, wq, wc = 0.45, 0.35, 0.20
        else:
            wd, wq, wc = st.session_state["w_delivery"] / ws, st.session_state["w_quality"] / ws, st.session_state["w_cost"] / ws
        st.caption(f"Effective: Delivery {wd:.0%} · Quality {wq:.0%} · Cost {wc:.0%}")

    divider()

    col_cat1, col_cat2, col_cat3 = st.columns([2, 1, 1])
    with col_cat1:
        orders_cat = _add_part_category(orders)
        rfqs_cat = _add_part_category(rfqs)
        all_cats = sorted(set(orders_cat["part_category"].unique()) | set(rfqs_cat["part_category"].unique()))
        st.selectbox("Part Category", ["(All Categories)"] + all_cats, key="category_choice")
    with col_cat2:
        st.radio("Evidence from", ["RFQs only", "Orders only", "Orders + RFQs"], key="capability_source")
    with col_cat3:
        st.slider("Min qualifying lines", 1, 10, step=1, key="min_lines")
        st.checkbox("Show coverage detail", key="show_coverage")

    chosen_cat = st.session_state["category_choice"]

    cap_parts = []
    if st.session_state["capability_source"] in ("Orders only", "Orders + RFQs"):
        cap_parts.append(orders_cat.groupby(["supplier_name", "part_category"]).size().reset_index(name="lines"))
    if st.session_state["capability_source"] in ("RFQs only", "Orders + RFQs"):
        cap_parts.append(rfqs_cat.groupby(["supplier_name", "part_category"]).size().reset_index(name="lines"))

    cap_counts = (
        pd.concat(cap_parts, ignore_index=True).groupby(["supplier_name", "part_category"], as_index=False)["lines"].sum()
        if cap_parts
        else pd.DataFrame()
    )

    if st.session_state["show_coverage"] and not cap_counts.empty:
        cov = cap_counts if chosen_cat == "(All Categories)" else cap_counts[cap_counts["part_category"] == chosen_cat]
        st.caption("Lines in category per supplier (based on selected evidence source):")
        show_table(with_rank(format_for_display(cov.sort_values("lines", ascending=False), ["supplier_name", "part_category", "lines"])), max_rows=20)

    if chosen_cat == "(All Categories)" or cap_counts.empty:
        eligible = set(supplier_master["supplier_name"].astype(str))
    else:
        eligible = set(
            cap_counts[(cap_counts["part_category"] == chosen_cat) & (cap_counts["lines"] >= st.session_state["min_lines"])][
                "supplier_name"
            ].astype(str)
        )

    sc_orders = orders_cat[orders_cat["supplier_name"].astype(str).isin(eligible)]
    sc_rfqs = rfqs_cat[rfqs_cat["supplier_name"].astype(str).isin(eligible)]
    if chosen_cat != "(All Categories)":
        sc_orders = sc_orders[sc_orders["part_category"] == chosen_cat]
        sc_rfqs = sc_rfqs[sc_rfqs["part_category"] == chosen_cat]

    if sc_orders.empty:
        if chosen_cat != "(All Categories)":
            st.warning("No suppliers found for this category. Showing overall rankings.")
        decision_kpi = supplier_master.copy()
    else:
        sc_spend = sc_orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index(name="total_spend")

        sc_ot = sc_orders[sc_orders["actual_delivery_date"].notna() & sc_orders["promised_date"].notna()].copy()
        sc_ot["on_time"] = (sc_ot["actual_delivery_date"] <= sc_ot["promised_date"]).astype(float)
        sc_otr = sc_ot.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
        sc_otr["on_time_rate"] = (sc_otr["on_time"] * 100).round(1)
        sc_otr = sc_otr.drop(columns=["on_time"])

        sc_q = quality.merge(sc_orders[["order_id", "supplier_name"]], on="order_id", how="inner")
        if {"parts_rejected", "parts_inspected"}.issubset(sc_q.columns) and len(sc_q):
            sc_q["defect_rate"] = (sc_q["parts_rejected"] / sc_q["parts_inspected"]).replace([pd.NA, float("inf")], 0)
            sc_def = sc_q.groupby("supplier_name", dropna=False)["defect_rate"].mean().reset_index()
            sc_def["defect_rate"] = (sc_def["defect_rate"] * 100).round(1)
        else:
            sc_def = pd.DataFrame({"supplier_name": sc_spend["supplier_name"], "defect_rate": 0.0})

        sc_rp = sc_rfqs.copy()
        sc_rp["quoted_price"] = pd.to_numeric(sc_rp["quoted_price"], errors="coerce")
        sc_rp = sc_rp[sc_rp["quoted_price"].gt(0) & sc_rp["quoted_price"].notna()]
        sc_ap = (
            sc_rp.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index(name="avg_price")
            if not sc_rp.empty
            else pd.DataFrame({"supplier_name": sc_spend["supplier_name"], "avg_price": 0.0})
        )
        sc_ap["avg_price"] = sc_ap["avg_price"].round(2)

        decision_kpi = (
            sc_spend.merge(sc_otr, on="supplier_name", how="left").merge(sc_def, on="supplier_name", how="left").merge(sc_ap, on="supplier_name", how="left")
        ).fillna({"on_time_rate": 0.0, "defect_rate": 0.0, "avg_price": 0.0})

        mp2 = decision_kpi["avg_price"].replace(0, pd.NA).max()
        decision_kpi["price_score"] = (100 * (1 - decision_kpi["avg_price"] / mp2)).clip(0, 100).fillna(0) if pd.notna(mp2) and mp2 > 0 else 0.0
        decision_kpi["performance_score"] = (decision_kpi["on_time_rate"] * wd + (100 - decision_kpi["defect_rate"]) * wq + decision_kpi["price_score"] * wc).round(1)
        decision_kpi["risk_flag"] = decision_kpi.apply(risk_flag, axis=1)

    decision_kpi["fit"] = (decision_kpi["on_time_rate"] >= st.session_state["req_on_time"]) & (decision_kpi["defect_rate"] <= st.session_state["max_defects"])
    decision_kpi["fit_status"] = decision_kpi["fit"].map({True: "✅ Meets criteria", False: "❌ Below threshold"})
    decision_kpi["notes_hint"] = decision_kpi["supplier_name"].apply(lambda s: note_snippet(supplier_notes, s))

    n_fit = int(decision_kpi["fit"].sum())
    n_total = len(decision_kpi)

    if n_fit == 0:
        st.warning(
            f"No suppliers meet the current thresholds (≥{st.session_state['req_on_time']}% on-time, ≤{st.session_state['max_defects']}% defects). Consider relaxing criteria."
        )
    else:
        st.success(f"**{n_fit} of {n_total} suppliers** meet your thresholds — ranked below by performance score.")

    ranked = decision_kpi.sort_values(["fit", "performance_score"], ascending=[False, False])
    show_table(
        with_rank(
            format_for_display(
                ranked,
                [
                    "supplier_name",
                    "fit_status",
                    "performance_score",
                    "risk_flag",
                    "on_time_rate",
                    "defect_rate",
                    "avg_price",
                    "total_spend",
                    "notes_hint",
                ],
            )
        ),
        TOP_N,
    )

    divider()
    st.subheader("Consolidation Opportunities")
    st.caption(f"Pricing analysis within scope: **{chosen_cat}**")
    impact_dec = build_pricing_impact(supplier_master, rfqs, chosen_cat)
    st.metric(
        "Est. Annual Savings Available",
        _fmt_money(float(impact_dec["estimated_overpay"].sum())),
        help="Based on avg price premium above best comparable quote × estimated units",
    )
    show_table(
        with_rank(
            format_for_display(
                impact_dec.sort_values("estimated_overpay", ascending=False),
                ["supplier_name", "total_spend", "avg_price", "avg_delta_vs_best", "estimated_overpay", "risk_flag", "rfqs", "pct_not_best"],
            )
        ),
        TOP_N,
    )

# ═══════════════════════════════════════════════════════
# TAB 4 · PERFORMANCE TRENDS
# ═══════════════════════════════════════════════════════
with tab_trends:
    st.subheader("Performance Trends")
    st.caption("Delivery reliability, quality, and spend allocation over time — designed for weekly ops reviews and executive reporting.")

    if "order_date" not in orders.columns:
        st.warning("No `order_date` column found — trends unavailable.")
    else:
        top6 = supplier_master.sort_values("total_spend", ascending=False)["supplier_name"].head(6).tolist()
        if not st.session_state.get("trend_supplier_filter"):
            st.session_state["trend_supplier_filter"] = top6

        tf = st.multiselect(
            "Suppliers to display:",
            options=all_suppliers,
            default=st.session_state.get("trend_supplier_filter", top6),
            key="trend_supplier_filter",
        )
        if not tf:
            tf = top6

        ot_trend = ot_valid[ot_valid["supplier_name"].isin(tf)].copy()
        ot_trend["month"] = ot_trend["order_date"].dt.to_period("M").dt.to_timestamp()

        # ── Chart 1: Delivery Heatmap ──
        st.markdown("#### Delivery Reliability — Monthly On-Time Rate Heatmap")
        st.caption("Each cell shows on-time %. Blank = no orders. Use for supplier accountability in weekly reviews.")

        if ot_trend.empty:
            st.info("No delivery data for selected suppliers.")
        else:
            heat = ot_trend.groupby(["month", "supplier_name"])["on_time"].agg(["mean", "count"]).reset_index()
            heat.columns = ["month", "supplier_name", "on_time_rate", "n_orders"]
            heat = heat[heat["n_orders"] >= 1]
            heat["otr_pct"] = (heat["on_time_rate"] * 100).round(1)

            heatmap = (
                alt.Chart(heat)
                .mark_rect(cornerRadius=3)
                .encode(
                    x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y", labelAngle=-30)),
                    y=alt.Y("supplier_name:N", title=None, sort=alt.SortField("otr_pct", order="descending")),
                    color=alt.Color(
                        "otr_pct:Q",
                        title="On-Time %",
                        scale=alt.Scale(domain=[50, 100], range=["#D62728", "#FACC15", "#16A34A"]),
                        legend=alt.Legend(orient="bottom", gradientLength=220),
                    ),
                    tooltip=[
                        alt.Tooltip("supplier_name:N", title="Supplier"),
                        alt.Tooltip("month:T", title="Month", format="%B %Y"),
                        alt.Tooltip("otr_pct:Q", title="On-Time %", format=".1f"),
                        alt.Tooltip("n_orders:Q", title="# Orders"),
                    ],
                )
                .properties(height=max(140, len(tf) * 48))
            )
            text_ov = (
                alt.Chart(heat)
                .mark_text(fontSize=10, fontWeight="bold")
                .encode(
                    x=alt.X("month:T"),
                    y=alt.Y("supplier_name:N", sort=alt.SortField("otr_pct", order="descending")),
                    text=alt.Text("otr_pct:Q", format=".0f"),
                    color=alt.value("white"),
                )
            )
            st.altair_chart(heatmap + text_ov, use_container_width=True)

        divider()

        # ── Chart 2: Late Delivery Scatter ──
        st.markdown("#### Late Deliveries — Days Late per Order")
        st.caption("Each dot = one late order. Size = PO value. Persistent dots indicate systemic reliability issues.")

        late_orders = ot_valid[ot_valid["supplier_name"].isin(tf)].copy()
        late_orders["days_late"] = (late_orders["actual_delivery_date"] - late_orders["promised_date"]).dt.days
        late_orders = late_orders[late_orders["days_late"] > 0].copy()
        late_orders["month"] = late_orders["order_date"].dt.to_period("M").dt.to_timestamp()
        if "po_amount" not in late_orders.columns:
            late_orders["po_amount"] = 1000.0

        if late_orders.empty:
            st.success("No late deliveries recorded for the selected suppliers.")
        else:
            sc = (
                alt.Chart(late_orders)
                .mark_circle(opacity=0.78)
                .encode(
                    x=alt.X("month:T", title="Month", axis=alt.Axis(format="%b %y")),
                    y=alt.Y("days_late:Q", title="Days Late", scale=alt.Scale(zero=True)),
                    size=alt.Size("po_amount:Q", legend=None, scale=alt.Scale(range=[30, 350])),
                    color=alt.Color("supplier_name:N", title="Supplier"),
                    tooltip=[
                        alt.Tooltip("supplier_name:N", title="Supplier"),
                        alt.Tooltip("order_id:N", title="Order ID"),
                        alt.Tooltip("month:T", title="Month", format="%B %Y"),
                        alt.Tooltip("days_late:Q", title="Days Late"),
                        alt.Tooltip("po_amount:Q", title="PO Value ($)", format="$,.0f"),
                    ]
                    + ([alt.Tooltip("part_description:N", title="Part")] if "part_description" in late_orders.columns else []),
                )
                .properties(height=310)
            )
            st.altair_chart(sc, use_container_width=True)
            st.caption(f"{len(late_orders)} late orders shown across {late_orders['supplier_name'].nunique()} supplier(s).")

        divider()

        # ── Chart 3: Quality — defect rate bar ──
        st.markdown("#### Quality — Avg Defect Rate by Supplier")
        st.caption("Overall rejection rate across inspections. Dashed line marks the 5% corrective-action threshold.")

        defect_bar_data = defects[defects["supplier_name"].isin(tf)].sort_values("defect_rate", ascending=False).copy()
        defect_bar_data["above"] = defect_bar_data["defect_rate"].apply(lambda x: "Above threshold" if x > 5 else "Within threshold")

        if not defect_bar_data.empty:
            dbar = (
                alt.Chart(defect_bar_data)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("supplier_name:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-20)),
                    y=alt.Y("defect_rate:Q", title="Avg Defect Rate (%)", scale=alt.Scale(zero=True)),
                    color=alt.Color(
                        "above:N",
                        scale=alt.Scale(domain=["Above threshold", "Within threshold"], range=["#D62728", "#2CA02C"]),
                        legend=alt.Legend(title="Status", orient="bottom"),
                    ),
                    tooltip=[
                        alt.Tooltip("supplier_name:N", title="Supplier"),
                        alt.Tooltip("defect_rate:Q", title="Defect Rate (%)", format=".2f"),
                    ],
                )
                .properties(height=270)
            )
            tline = alt.Chart(pd.DataFrame({"y": [5]})).mark_rule(color="#FF7F0E", strokeDash=[6, 3], strokeWidth=2).encode(y="y:Q")
            st.altair_chart(dbar + tline, use_container_width=True)

        divider()

        # ── Chart 4: Spend share stacked area ──
        st.markdown("#### Spend Allocation — Monthly Share by Supplier")
        st.caption("Tracks sourcing concentration over time. Rising share from one supplier can signal growing dependency risk.")

        spend_trend = orders[orders["supplier_name"].isin(tf)].copy()
        spend_trend["month"] = spend_trend["order_date"].dt.to_period("M").dt.to_timestamp()
        ms = spend_trend.groupby(["month", "supplier_name"])["po_amount"].sum().reset_index()
        ms["spend_k"] = (ms["po_amount"] / 1000).round(1)

        if not ms.empty:
            area = (
                alt.Chart(ms)
                .mark_area(opacity=0.72, interpolate="monotone")
                .encode(
                    x=alt.X("month:T", title="Month", axis=alt.Axis(format="%b %y")),
                    y=alt.Y("spend_k:Q", title="Spend ($K)", stack="normalize", axis=alt.Axis(format=".0%")),
                    color=alt.Color("supplier_name:N", title="Supplier"),
                    tooltip=[
                        alt.Tooltip("supplier_name:N", title="Supplier"),
                        alt.Tooltip("month:T", title="Month", format="%B %Y"),
                        alt.Tooltip("spend_k:Q", title="Spend ($K)", format=".1f"),
                    ],
                )
                .properties(height=290)
            )
            st.altair_chart(area, use_container_width=True)
            st.caption("Y-axis shows proportional share of spend that month.")

# ═══════════════════════════════════════════════════════
# TAB 5 · FINANCIAL IMPACT
# ═══════════════════════════════════════════════════════
with tab_financial:
    st.subheader("Financial Impact Analysis")
    st.caption("Quantifies supplier underperformance across pricing leakage, quality/rework cost, and delivery risk exposure.")

    cat_fin = st.session_state.get("category_choice", "(All Categories)")
    impact_df = build_pricing_impact(supplier_master, rfqs, cat_fin)
    impact_df["defect_cost"] = (impact_df["total_spend"] * (impact_df["defect_rate"] / 100) * 0.5).round(0)
    impact_df["total_risk"] = impact_df["estimated_overpay"] + impact_df["defect_cost"]

    late_spend_fin = float(impact_df.loc[impact_df["on_time_rate"] < 85, "total_spend"].sum())
    total_pricing = float(impact_df["estimated_overpay"].sum())
    total_quality = float(impact_df["defect_cost"].sum())
    total_risk = total_pricing + total_quality

    # KPIs
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Quantified Risk", _fmt_money(total_risk), help="Pricing leakage + estimated quality cost combined")
    f2.metric("Pricing Leakage", _fmt_money(total_pricing), help="Avg premium above best comparable RFQ quote × estimated units")
    f3.metric("Est. Quality / Rework Cost", _fmt_money(total_quality), help="Defect rate × total spend × 0.5 rework cost factor")
    f4.metric("Spend at Delivery Risk", _fmt_money(late_spend_fin), help="Total spend with suppliers below 85% on-time rate")

    divider()

    # ── Chart 1: Stacked risk bar ──
    st.markdown("#### Financial Exposure by Supplier — Pricing + Quality Cost")
    st.caption("Suppliers with the highest combined exposure. Stacked bars show leakage vs rework cost breakdown.")

    risk_plot = impact_df[impact_df["total_risk"] > 0].sort_values("total_risk", ascending=False).head(12).copy()
    if not risk_plot.empty:
        melt = pd.melt(
            risk_plot[["supplier_name", "estimated_overpay", "defect_cost"]],
            id_vars="supplier_name",
            var_name="type",
            value_name="cost",
        )
        melt["type"] = melt["type"].map(
            {
                "estimated_overpay": "Pricing Leakage",
                "defect_cost": "Quality / Rework Cost",
            }
        )
        stacked = (
            alt.Chart(melt)
            .mark_bar()
            .encode(
                x=alt.X("cost:Q", title="Estimated Cost ($)", axis=alt.Axis(format="$,.0f")),
                y=alt.Y("supplier_name:N", sort="-x", title=None),
                color=alt.Color(
                    "type:N",
                    scale=alt.Scale(domain=["Pricing Leakage", "Quality / Rework Cost"], range=["#FF7F0E", "#D62728"]),
                    title="Cost Type",
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("supplier_name:N", title="Supplier"),
                    alt.Tooltip("type:N", title="Cost Type"),
                    alt.Tooltip("cost:Q", title="Est. Cost ($)", format="$,.0f"),
                ],
            )
            .properties(height=max(250, len(risk_plot) * 36))
        )
        st.altair_chart(stacked, use_container_width=True)

    divider()

    # ── Chart 2: Spend vs Performance ──
    st.markdown("#### Spend Concentration vs. Performance Score")
    st.caption("**High spend + low performance = highest priority.** Bottom-right quadrant = immediate intervention targets.")

    sp_vs_perf = impact_df.copy()
    sp_vs_perf["spend_m"] = (sp_vs_perf["total_spend"] / 1e6).round(3)

    scatter2 = (
        alt.Chart(sp_vs_perf)
        .mark_circle(size=160, opacity=0.84, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("spend_m:Q", title="Total Spend ($M)", scale=alt.Scale(zero=True)),
            y=alt.Y("performance_score:Q", title="Performance Score (0–100)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk"),
            tooltip=[
                alt.Tooltip("supplier_name:N", title="Supplier"),
                alt.Tooltip("spend_m:Q", title="Spend ($M)", format=".3f"),
                alt.Tooltip("performance_score:Q", title="Performance Score", format=".1f"),
                alt.Tooltip("estimated_overpay:Q", title="Pricing Leakage ($)", format="$,.0f"),
                alt.Tooltip("defect_cost:Q", title="Quality Cost ($)", format="$,.0f"),
                alt.Tooltip("on_time_rate:Q", title="On-Time Rate (%)", format=".1f"),
                alt.Tooltip("defect_rate:Q", title="Defect Rate (%)", format=".1f"),
                alt.Tooltip("risk_flag:N", title="Risk Flag"),
            ],
        )
        .properties(height=370)
    )
    ref = alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(color="#d1d5db", strokeDash=[5, 5], strokeWidth=1.5).encode(y="y:Q")
    st.altair_chart(scatter2 + ref, use_container_width=True)
    st.caption("Dashed line = 75 performance score. Below line with large spend = urgent focus.")

    divider()

    # ── Chart 3: Pricing Premium Distribution ──
    st.markdown("#### Pricing Competitiveness — Premium vs. Best Market Quote")
    st.caption("Avg premium for apples-to-apples RFQ lines. Zero is ideal.")

    price_plot = impact_df[impact_df["avg_delta_vs_best"] > 0].sort_values("avg_delta_vs_best", ascending=False).head(12).copy()
    if not price_plot.empty:
        price_bar = (
            alt.Chart(price_plot)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("avg_delta_vs_best:Q", title="Avg Price Premium vs Best Quote ($/unit)", scale=alt.Scale(zero=True)),
                y=alt.Y("supplier_name:N", sort="-x", title=None),
                color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk", legend=None),
                tooltip=[
                    alt.Tooltip("supplier_name:N", title="Supplier"),
                    alt.Tooltip("avg_delta_vs_best:Q", title="Premium ($/unit)", format=".2f"),
                    alt.Tooltip("pct_not_best:Q", title="% Quotes Not Best", format=".1f"),
                    alt.Tooltip("rfqs:Q", title="# RFQs"),
                ],
            )
            .properties(height=max(230, len(price_plot) * 34))
        )
        st.altair_chart(price_bar, use_container_width=True)

    divider()

    with st.expander("Full impact table — all suppliers"):
        show_table(
            with_rank(
                format_for_display(
                    impact_df.sort_values("total_risk", ascending=False),
                    [
                        "supplier_name",
                        "total_spend",
                        "on_time_rate",
                        "defect_rate",
                        "avg_price",
                        "avg_delta_vs_best",
                        "pct_not_best",
                        "rfqs",
                        "estimated_overpay",
                        "defect_cost",
                        "performance_score",
                        "risk_flag",
                    ],
                )
            ),
            max_rows=50,
        )

    with st.expander("Methodology & data notes"):
        st.markdown(
            """
**Pricing Leakage**  
Avg Price Premium is computed *within the same RFQ line* (apples-to-apples — never compares prices across different parts).  
Estimated Units = total spend ÷ avg quoted price. Leakage = Premium × Estimated Units.

**Quality Cost**  
Est. Rework Cost = total spend × defect rate × 0.5 (conservative rework cost factor).  
Actual costs vary by part complexity, labor rates, and re-inspection time.

**Delivery Risk**  
Spend flagged as "at risk" = suppliers with on-time rate < 85%.  
Excludes orders with missing delivery dates (not penalized).

**Roadmap to Higher Precision**  
- Connect ERP goods receipts for actual received quantities (removes spend-derived unit estimate)  
- Add commodity price benchmarks (should-cost validation)  
- Track trend direction (improving vs declining) for predictive alerts  
- Incorporate freight and expedite costs into total cost of ownership  
"""
        )

    with st.expander("Debug — column names"):
        st.write("Orders:", list(orders.columns))
        st.write("Quality:", list(quality.columns))
        st.write("RFQs:", list(rfqs.columns))
