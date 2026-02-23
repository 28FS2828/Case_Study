"""
Hoth Industries — Supplier Intelligence Platform
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

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.03em; }
h2 { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.02em; }
h3 { font-size: 1.05rem; font-weight: 600; }
div[data-testid="stMetricValue"] { font-size: 1.65rem; font-weight: 700; }
div[data-testid="stMetricLabel"] { font-size: 0.82rem; color: #6b7280; }
div[data-testid="stExpander"] { border-radius: 10px; border: 1px solid #e5e7eb; }
div[data-testid="stTabs"] button { font-size: 0.9rem; font-weight: 500; }
hr { margin: 1.2rem 0; }
div[data-testid="stDataFrame"] { border-radius: 8px; }
/* Make action cards equal height */
div[data-testid="stHorizontalBlock"] > div { flex: 1; }
</style>
""", unsafe_allow_html=True)

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

RISK_ORDER  = ["🔴 Quality Risk", "🟠 Delivery Risk", "🟡 Cost Risk", "🟢 Strategic"]
RISK_COLORS = ["#D62728", "#FF7F0E", "#F2C12E", "#2CA02C"]
risk_color_scale = alt.Scale(domain=RISK_ORDER, range=RISK_COLORS)

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "l.l.c", "ltd", "limited",
    "corp", "corporation", "co", "company", "gmbh", "s.a", "sa",
}

DISPLAY_COLS = {
    "supplier_name":     "Supplier",
    "risk_flag":         "Risk",
    "fit_status":        "Fit",
    "notes_hint":        "Tribal Knowledge",
    "total_spend":       "Total Spend ($)",
    "avg_price":         "Avg Quote ($/unit)",
    "avg_delta_vs_best": "Avg Premium vs Best ($/unit)",
    "on_time_rate":      "On-Time Rate (%)",
    "defect_rate":       "Defect Rate (%)",
    "price_score":       "Price Score (0-100)",
    "performance_score": "Performance Score (0-100)",
    "estimated_overpay": "Est. Pricing Leakage ($)",
    "defect_cost":       "Est. Quality Cost ($)",
    "part_category":     "Part Category",
    "lines":             "# Lines",
    "rfqs":              "# RFQs",
    "pct_not_best":      "% Above Best (%)",
    "orders":            "# Orders",
    "spend_share_pct":   "Spend Share (%)",
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
    if pd.isna(x): return ""
    try:    return "${:,.0f}".format(float(x))
    except: return str(x)

def _fmt_money_2(x):
    if pd.isna(x): return ""
    try:    return "${:,.2f}".format(float(x))
    except: return str(x)

def _fmt_pct(x):
    if pd.isna(x): return ""
    try:    return "{:.1f}%".format(float(x))
    except: return str(x)

def _fmt_score(x):
    if pd.isna(x): return ""
    try:    return "{:.1f}".format(float(x))
    except: return str(x)

def dataframe_pretty(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if "($)" in c and "($/unit)" not in c:
            out[c] = out[c].apply(_fmt_money)
        elif "($/unit)" in c:
            out[c] = out[c].apply(_fmt_money_2)
        elif "(%)" in c:
            out[c] = out[c].apply(_fmt_pct)
        elif "(0-100)" in c:
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

# ─────────────────────────────────────────────
# ENTITY RESOLUTION
# ─────────────────────────────────────────────
def normalize_supplier_key(name: str) -> str:
    if pd.isna(name): return ""
    s = re.sub(r"[^a-z0-9\s]", " ", str(name).lower().strip())
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split()
    while parts and parts[-1] in LEGAL_SUFFIXES:
        parts.pop()
    return " ".join(parts)

def apply_entity_resolution(df: pd.DataFrame, col: str, manual_map: dict = None) -> pd.DataFrame:
    if col not in df.columns: return df
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
    ("Motors / Actuation",          ["motor", "actuator", "servo", "stepper", "gearbox"]),
    ("Controls / Electronics",      ["controller", "pcb", "board", "sensor", "wire", "harness", "electronic", "connector", "vfd", "plc"]),
    ("Heat Exchangers",             ["heat exchanger", "exchanger", "coil", "radiator"]),
    ("Air Handling / Dampers",      ["damper", "louver", "filter", "hepa", "diffuser", "grille", "duct", "fan", "blower"]),
    ("Fins / Aero Surfaces",        ["fin", "aero", "wing", "stabilizer", "airfoil"]),
    ("Brackets / Fabricated Parts", ["bracket", "fabricat", "weld", "machin", "cnc", "laser", "cut", "bend", "sheet"]),
    ("Shafts / Mechanical",         ["shaft", "gear", "coupling", "hub", "pulley"]),
    ("Bearings / Seals",            ["bearing", "seal", "bushing"]),
    ("Fasteners / Hardware",        ["bolt", "screw", "nut", "washer", "fastener", "rivet"]),
    ("Metals / Raw Material",       ["aluminum", "steel", "stainless", "titanium", "alloy", "bar", "rod", "plate"]),
    ("Plastics / Polymer",          ["plastic", "nylon", "resin", "injection", "mold", "polymer"]),
    ("Packaging",                   ["pack", "crate", "box", "foam", "pallet"]),
]

def categorize_part(text: str) -> str:
    if pd.isna(text): return "Other / Unknown"
    t = str(text).lower()
    for cat, kws in PART_RULES:
        if any(kw in t for kw in kws):
            return cat
    return "Other / Unknown"

def _add_part_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col = next((c for c in ["part_description","commodity","category","component","item","description"]
                if c in out.columns), None)
    out["part_category"] = out[col].apply(categorize_part) if col else "Other / Unknown"
    return out

# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────
def _pick_rfq_line_key(df: pd.DataFrame):
    for c in ["rfq_id","rfq_line_id","line_id","part_number","item_id","part_description"]:
        if c in df.columns: return c
    return None

def _norm_text(x):
    if pd.isna(x): return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(x).lower().strip())).strip()

def rfq_competitiveness(rfqs_df: pd.DataFrame, cat: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["supplier_name","avg_price_scope","avg_delta_vs_best","pct_not_best","lines","rfqs"])
    if rfqs_df is None or rfqs_df.empty: return empty
    if not {"supplier_name","quoted_price"}.issubset(rfqs_df.columns): return empty
    r = _add_part_category(rfqs_df)
    if cat != "(All Categories)": r = r[r["part_category"] == cat]
    if r.empty: return empty
    r = r.copy()
    r["quoted_price"] = pd.to_numeric(r["quoted_price"], errors="coerce")
    r = r[r["quoted_price"].gt(0) & r["quoted_price"].notna()]
    if r.empty: return empty
    lk = _pick_rfq_line_key(r)
    if lk is None: return empty
    r["_lk"] = r[lk].astype(str).apply(_norm_text if lk == "part_description" else lambda x: x)
    best = r.groupby("_lk")["quoted_price"].min().rename("best_price")
    r = r.join(best, on="_lk")
    r["delta"]   = (r["quoted_price"] - r["best_price"]).clip(lower=0)
    r["is_best"] = (r["delta"] <= 1e-9).astype(int)
    g = r.groupby("supplier_name", dropna=False).agg(
        avg_price_scope  =("quoted_price","mean"),
        avg_delta_vs_best=("delta","mean"),
        lines            =("quoted_price","size"),
        rfqs             =("_lk","nunique"),
        pct_not_best     =("is_best", lambda s: 100*(1-(s.sum()/len(s))) if len(s) else 0),
    ).reset_index()
    return g.round({"avg_price_scope":2,"avg_delta_vs_best":2,"pct_not_best":1})

def build_pricing_impact(master: pd.DataFrame, rfqs_df: pd.DataFrame, cat: str) -> pd.DataFrame:
    comp = rfq_competitiveness(rfqs_df, cat)
    out  = master.merge(comp, on="supplier_name", how="left")
    out["avg_price_scope"]   = out["avg_price_scope"].fillna(out["avg_price"])
    out["avg_delta_vs_best"] = out["avg_delta_vs_best"].fillna(0.0)
    out["lines"]             = out["lines"].fillna(0).astype(int)
    out["rfqs"]              = out["rfqs"].fillna(0).astype(int)
    out["pct_not_best"]      = out["pct_not_best"].fillna(0.0)
    mask = out["avg_price_scope"] > 0
    out["est_units"] = 0.0
    out.loc[mask, "est_units"] = out.loc[mask,"total_spend"] / out.loc[mask,"avg_price_scope"]
    out["estimated_overpay"] = (out["avg_delta_vs_best"] * out["est_units"]).fillna(0.0)
    out["avg_price"]         = out["avg_price_scope"].round(2)
    out["avg_delta_vs_best"] = out["avg_delta_vs_best"].round(2)
    return out

def switchability(rfqs_df: pd.DataFrame, cat: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["supplier_name","avg_alternatives","switchability"])
    if rfqs_df is None or rfqs_df.empty: return empty
    r = _add_part_category(rfqs_df)
    if cat != "(All Categories)": r = r[r["part_category"] == cat]
    if r.empty: return empty
    r = r.copy()
    r["quoted_price"] = pd.to_numeric(r["quoted_price"], errors="coerce")
    r = r[r["quoted_price"].gt(0) & r["quoted_price"].notna()]
    if r.empty: return empty
    lk = _pick_rfq_line_key(r)
    if lk is None: return empty
    r["_lk"] = r[lk].astype(str)
    n = r.groupby("_lk")["supplier_name"].nunique().rename("n")
    r = r.join(n, on="_lk")
    r["alts"] = (r["n"] - 1).clip(lower=0)
    g = r.groupby("supplier_name", dropna=False)["alts"].mean().reset_index(name="avg_alternatives")
    g["avg_alternatives"] = g["avg_alternatives"].round(1)
    g["switchability"] = g["avg_alternatives"].apply(
        lambda a: "HIGH" if a >= 2 else ("MED" if a >= 1 else "LOW")
    )
    return g

# ─────────────────────────────────────────────
# SUPPLIER NOTES — FIXED PARSER
# Correctly handles the ===\nHEADER\n=== structure
# where content blocks are between section headers
# ─────────────────────────────────────────────
def parse_supplier_notes(text: str) -> dict:
    """
    Parses supplier_notes.txt which uses:
        ================================================================================
        SUPPLIER NAME - DESCRIPTOR
        ================================================================================
        content...
    """
    notes = {}
    if not text:
        return notes

    # Find all section boundaries: lines of ===...=== (40+ chars)
    # Then grab the header line(s) and content between sections
    section_pattern = re.compile(r"={40,}\n(.*?)\n={40,}", re.DOTALL)
    sections = list(section_pattern.finditer(text))

    for i, sec in enumerate(sections):
        header_raw = sec.group(1).strip()
        # Take first line of header only
        first_line = header_raw.splitlines()[0].strip()

        # Skip non-supplier headers
        if first_line.upper() in ("END OF NOTES", "GENERAL PROCUREMENT ISSUES"):
            continue

        # Parse supplier name and optional descriptor
        # Handle "NAME - DESCRIPTOR" pattern
        dash_m = re.match(r"^(.+?)\s+-\s+(.+)$", first_line)
        if dash_m:
            supplier_raw = dash_m.group(1).strip()
            descriptor   = dash_m.group(2).strip()
        else:
            supplier_raw = first_line.strip()
            descriptor   = ""

        # Handle multi-name lines like "APEX MFG / APEX MFG INC / ..."
        # Use the first name as the canonical key for lookup
        if "/" in supplier_raw:
            supplier_raw = supplier_raw.split("/")[0].strip()

        key = normalize_supplier_key(supplier_raw)
        if not key or key in ("supplier performance notes", "end of notes", "general procurement issues"):
            continue

        # Extract content between this section header and the next
        content_start = sec.end()
        content_end   = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        content       = text[content_start:content_end].strip()

        # Collect meaningful lines as bullet points (skip NOTE lines and very short lines)
        bullets = []
        for line in content.splitlines():
            line = line.strip()
            if not line: continue
            if line.startswith("**") and "NOTE" in line.upper(): continue
            if len(line) < 15: continue
            bullets.append(line[:220] + ("..." if len(line) > 220 else ""))
            if len(bullets) >= 8:
                break

        notes[key] = {
            "descriptor":    descriptor,
            "bullets":       bullets,
            "supplier_raw":  supplier_raw,
        }

    return notes


def note_snippet(notes: dict, name: str) -> str:
    n = notes.get(normalize_supplier_key(name), {})
    if not n: return ""
    parts = [n.get("descriptor", "")]
    if n.get("bullets"): parts.append(n["bullets"][0])
    line = " | ".join(p for p in parts if p)
    return (line[:200] + "...") if len(line) > 200 else line

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
        except: pass
    return ""

def safe_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

MANUAL_KEY_MAP = {
    "apex mfg":             "apex manufacturing",
    "apex manufacturing inc":"apex manufacturing",
    "apex mfg inc":         "apex manufacturing",
}

def process_raw(orders, quality, rfqs, notes_text):
    orders = apply_entity_resolution(orders, "supplier_name", MANUAL_KEY_MAP)
    rfqs   = apply_entity_resolution(rfqs,   "supplier_name", MANUAL_KEY_MAP)
    for df, cols in [
        (orders,  ["order_date","promised_date","actual_delivery_date"]),
        (quality, ["inspection_date"]),
        (rfqs,    ["quote_date"]),
    ]:
        for c in cols: safe_dt(df, c)
    return orders, quality, rfqs, notes_text

def try_load_local():
    for op, qp, rp, np in [
        ("Case_Study/supplier_orders.csv",     "Case_Study/quality_inspections.csv",     "Case_Study/rfq_responses.csv",     "Case_Study/supplier_notes.txt"),
        ("supplier_notes.txt",                 "",                                         "",                                 "supplier_notes.txt"),
        ("Copy_of_supplier_orders.csv",        "Copy_of_quality_inspections.csv",         "Copy_of_rfq_responses.csv",        "supplier_notes.txt"),
        ("Copy of supplier_orders.csv",        "Copy of quality_inspections.csv",         "Copy of rfq_responses.csv",        "supplier_notes.txt"),
        ("supplier_orders.csv",                "quality_inspections.csv",                 "rfq_responses.csv",                "supplier_notes.txt"),
    ]:
        try:
            o = pd.read_csv(op)
            q = pd.read_csv(qp)
            r = pd.read_csv(rp)
            n = read_text_flexible([np])
            return o, q, r, n
        except: continue
    return None

local = try_load_local()
if local:
    orders, quality, rfqs, supplier_notes_text = process_raw(*local)
    supplier_notes = parse_supplier_notes(supplier_notes_text)
else:
    st.warning("Local data files not found. Please upload below.")
    with st.expander("Upload Data Files", expanded=True):
        c1, c2 = st.columns(2)
        uf_o = c1.file_uploader("supplier_orders.csv",     type=["csv"])
        uf_q = c1.file_uploader("quality_inspections.csv", type=["csv"])
        uf_r = c2.file_uploader("rfq_responses.csv",       type=["csv"])
        uf_n = c2.file_uploader("supplier_notes.txt",      type=["txt"])
    if all([uf_o, uf_q, uf_r]):
        import io
        o = pd.read_csv(io.BytesIO(uf_o.read()))
        q = pd.read_csv(io.BytesIO(uf_q.read()))
        r = pd.read_csv(io.BytesIO(uf_r.read()))
        n = uf_n.read().decode("utf-8","ignore") if uf_n else ""
        orders, quality, rfqs, supplier_notes_text = process_raw(o, q, r, n)
        supplier_notes = parse_supplier_notes(supplier_notes_text)
    else:
        st.info("Upload all three CSV files above to continue.")
        st.stop()

# ─────────────────────────────────────────────
# BUILD SUPPLIER MASTER
# ─────────────────────────────────────────────
required = {"order_id","supplier_name","po_amount","promised_date","actual_delivery_date"}
if missing := required - set(orders.columns):
    st.error(f"Orders file missing columns: {sorted(missing)}")
    st.stop()

spend = orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index(name="total_spend")

ot_valid = orders[orders["actual_delivery_date"].notna() & orders["promised_date"].notna()].copy()
ot_valid["on_time"] = (ot_valid["actual_delivery_date"] <= ot_valid["promised_date"]).astype(float)
on_time = ot_valid.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"] * 100).round(1)
on_time = on_time.drop(columns=["on_time"])

order_counts = orders.groupby("supplier_name", dropna=False)["order_id"].nunique().reset_index(name="orders")

if "order_id" not in quality.columns:
    st.error("quality_inspections.csv must contain 'order_id'.")
    st.stop()

q = quality.merge(orders[["order_id","supplier_name"]], on="order_id", how="left")
if {"parts_rejected","parts_inspected"}.issubset(q.columns):
    q["defect_rate"] = (q["parts_rejected"] / q["parts_inspected"]).replace([pd.NA, float("inf")], 0)
else:
    q["defect_rate"] = 0.0
defects = q.groupby("supplier_name", dropna=False)["defect_rate"].mean().reset_index()
defects["defect_rate"] = (defects["defect_rate"] * 100).round(1)

if not {"supplier_name","quoted_price"}.issubset(rfqs.columns):
    st.error("rfq_responses.csv must contain 'supplier_name' and 'quoted_price'.")
    st.stop()
ap = rfqs.copy()
ap["quoted_price"] = pd.to_numeric(ap["quoted_price"], errors="coerce")
ap = ap[ap["quoted_price"].gt(0) & ap["quoted_price"].notna()]
ap = ap.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index(name="avg_price")
ap["avg_price"] = ap["avg_price"].round(2)

supplier_master = (
    spend.merge(on_time,      on="supplier_name", how="left")
         .merge(defects,      on="supplier_name", how="left")
         .merge(ap,           on="supplier_name", how="left")
         .merge(order_counts, on="supplier_name", how="left")
).fillna({"on_time_rate":0.0,"defect_rate":0.0,"avg_price":0.0,"orders":0})

mp = supplier_master["avg_price"].replace(0, pd.NA).max()
supplier_master["price_score"] = (
    (100*(1-supplier_master["avg_price"]/mp)).clip(0,100).fillna(0)
    if pd.notna(mp) and mp > 0 else 0.0
)
supplier_master["performance_score"] = (
    supplier_master["on_time_rate"]        * 0.45 +
    (100 - supplier_master["defect_rate"]) * 0.35 +
    supplier_master["price_score"]         * 0.20
).round(1)

total_spend_all = float(supplier_master["total_spend"].sum())
supplier_master["spend_share_pct"] = (
    (supplier_master["total_spend"] / total_spend_all * 100).round(1)
    if total_spend_all > 0 else 0.0
)

def risk_flag(row):
    if row["defect_rate"] >= 8:   return "🔴 Quality Risk"
    if row["on_time_rate"] <= 85: return "🟠 Delivery Risk"
    if row["price_score"] <= 40:  return "🟡 Cost Risk"
    return "🟢 Strategic"

supplier_master["risk_flag"] = supplier_master.apply(risk_flag, axis=1)
supplier_master = supplier_master.sort_values("performance_score", ascending=False)
all_suppliers   = sorted(supplier_master["supplier_name"].dropna().unique().tolist())

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
h1.title("Hoth Industries · Supplier Intelligence Platform")
h1.caption("Unified supplier performance, pricing competitiveness, and sourcing decision support.")
with h2:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        if st.button("Reset Filters", use_container_width=True, key="reset_top"):
            st.session_state[RESET_FLAG] = True
            st.rerun()
    with rc2:
        if st.button("Reload Data", use_container_width=True, key="reload_top"):
            st.cache_data.clear()
            st.rerun()

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_overview, tab_intel, tab_decision, tab_trends, tab_financial = st.tabs([
    "📊 Executive Overview",
    "🔍 Supplier Intelligence",
    "⚡ Sourcing Decision",
    "📈 Performance Trends",
    "💰 Financial Impact",
])

# ═══════════════════════════════════════════════════════
# TAB 1 · EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════
with tab_overview:
    cat_exec    = st.session_state.get("category_choice","(All Categories)")
    impact_exec = build_pricing_impact(supplier_master, rfqs, cat_exec)
    sw_exec     = switchability(rfqs, cat_exec)
    impact_exec = impact_exec.merge(sw_exec, on="supplier_name", how="left")
    impact_exec["avg_alternatives"] = impact_exec["avg_alternatives"].fillna(0.0)
    impact_exec["switchability"]    = impact_exec["switchability"].fillna("LOW")
    impact_exec["defect_cost"]      = impact_exec["total_spend"] * (impact_exec["defect_rate"]/100) * 0.5

    pricing_leak    = float(impact_exec["estimated_overpay"].sum())
    late_spend      = float(supplier_master.loc[supplier_master["on_time_rate"] < 85,"total_spend"].sum())
    defect_cost     = float(impact_exec["defect_cost"].sum())
    n_quality_risk  = int((supplier_master["risk_flag"] == "🔴 Quality Risk").sum())
    n_delivery_risk = int((supplier_master["risk_flag"] == "🟠 Delivery Risk").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Est. Pricing Leakage",       _fmt_money(pricing_leak),
              help="Avg premium paid above best comparable market quote, scaled by spend")
    k2.metric("Spend at Delivery Risk",     _fmt_money(late_spend),
              help="Total spend with suppliers whose on-time rate is below 85%")
    k3.metric("Est. Quality / Rework Cost", _fmt_money(defect_cost),
              help="Defect rate x spend x 0.5 rework cost multiplier")
    k4.metric("Suppliers with Active Risk", f"{n_quality_risk + n_delivery_risk}",
              help=f"Quality Risk: {n_quality_risk}  |  Delivery Risk: {n_delivery_risk}")

    st.markdown("---")

    # ── Priority Actions (equal-width columns, concise text) ──
    st.subheader("Priority Actions")

    actions = []

    cand = impact_exec[impact_exec["avg_alternatives"] >= 1].sort_values("estimated_overpay", ascending=False)
    if len(cand):
        r = cand.iloc[0]
        overpay_pct = (float(r["avg_delta_vs_best"]) / float(r["avg_price"]) * 100) if float(r["avg_price"]) > 0 else 0
        actions.append({
            "type":    "Pricing",
            "color":   "#FF7F0E",
            "title":   f"Renegotiate {r['supplier_name']}",
            "stat":    _fmt_money(float(r["estimated_overpay"])),
            "stat_lbl":"Est. annual leakage",
            "detail":  f"Avg quote is ~{overpay_pct:.0f}% above best market price for the same parts. {int(r['avg_alternatives'])} alternative supplier(s) already quote these RFQ lines — leverage them.",
        })

    tmp = impact_exec.sort_values("defect_cost", ascending=False)
    if len(tmp) and float(tmp.iloc[0]["defect_rate"]) > 0:
        r = tmp.iloc[0]
        actions.append({
            "type":    "Quality",
            "color":   "#D62728",
            "title":   f"Address {r['supplier_name']}",
            "stat":    _fmt_pct(float(r["defect_rate"])),
            "stat_lbl":"Defect rate",
            "detail":  f"Est. {_fmt_money(float(r['defect_cost']))} in rework costs annually. Issue a corrective action request and require first-article inspection on the next order.",
        })

    tmp2 = supplier_master.sort_values("spend_share_pct", ascending=False)
    if len(tmp2) and float(tmp2.iloc[0]["spend_share_pct"]) >= 20:
        r = tmp2.iloc[0]
        actions.append({
            "type":    "Concentration",
            "color":   "#6366f1",
            "title":   f"Diversify from {r['supplier_name']}",
            "stat":    _fmt_pct(float(r["spend_share_pct"])),
            "stat_lbl":"of total spend",
            "detail":  f"Over-reliance on one supplier creates single-point-of-failure risk. Qualify 1–2 backup suppliers and target a max of 30% spend concentration.",
        })

    if actions:
        cols = st.columns(len(actions))
        for col, a in zip(cols, actions[:3]):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"<span style='background:{a['color']};color:white;border-radius:4px;"
                        f"padding:2px 8px;font-size:0.72rem;font-weight:700'>{a['type'].upper()}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**{a['title']}**")
                    st.metric(a["stat_lbl"], a["stat"])
                    st.caption(a["detail"])

    st.markdown("---")

    # ── Supplier Risk Heatmap (replaces scatter) ──────────────
    st.subheader("Supplier Risk Overview")
    st.caption("Each row is a supplier. Color = risk level. Width of bar = relative spend. Hover for details.")

    heat_data = impact_exec.merge(
        supplier_master[["supplier_name","spend_share_pct"]], on="supplier_name", how="left"
    ).sort_values("performance_score", ascending=True).copy()
    heat_data["spend_m"] = (heat_data["total_spend"] / 1e6).round(3)

    # Color encoding based on risk
    risk_sev = {"🔴 Quality Risk":0,"🟠 Delivery Risk":1,"🟡 Cost Risk":2,"🟢 Strategic":3}
    heat_data["risk_sev"] = heat_data["risk_flag"].map(risk_sev).fillna(3)

    bar_chart = (
        alt.Chart(heat_data)
        .mark_bar(height=22, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("supplier_name:N",
                    sort=alt.EncodingSortField(field="performance_score", order="ascending"),
                    title=None,
                    axis=alt.Axis(labelLimit=160, labelFontSize=12)),
            x=alt.X("performance_score:Q",
                    title="Performance Score (0–100)",
                    scale=alt.Scale(domain=[0,100])),
            color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk",
                            legend=alt.Legend(orient="bottom", direction="horizontal")),
            tooltip=[
                alt.Tooltip("supplier_name:N",    title="Supplier"),
                alt.Tooltip("risk_flag:N",        title="Risk"),
                alt.Tooltip("performance_score:Q",title="Performance Score",  format=".1f"),
                alt.Tooltip("on_time_rate:Q",     title="On-Time Rate (%)",   format=".1f"),
                alt.Tooltip("defect_rate:Q",      title="Defect Rate (%)",    format=".1f"),
                alt.Tooltip("spend_m:Q",          title="Spend ($M)",         format=".3f"),
                alt.Tooltip("spend_share_pct:Q",  title="Spend Share (%)",    format=".1f"),
                alt.Tooltip("avg_delta_vs_best:Q",title="Price Premium ($/unit)", format=".2f"),
            ],
        )
        .properties(height=max(280, len(heat_data) * 32))
    )

    # Threshold reference line at 75
    ref_line = alt.Chart(pd.DataFrame({"x":[75]})).mark_rule(
        color="#9ca3af", strokeDash=[5,4], strokeWidth=1.5
    ).encode(x="x:Q")

    ref_label = alt.Chart(pd.DataFrame({"x":[75],"label":["Target ≥75"]})).mark_text(
        align="left", dx=4, dy=-6, fontSize=10, color="#9ca3af"
    ).encode(x="x:Q", text="label:N")

    st.altair_chart(bar_chart + ref_line + ref_label, use_container_width=True)
    st.caption("Sorted by performance score (lowest at top = highest priority). Score = Delivery 45% + Quality 35% + Price 20%.")

    st.markdown("---")

    # ── Supplier Table ──────────────────────────────────────────
    col_s, _ = st.columns([2, 2])
    search_q = col_s.text_input("Search suppliers", placeholder="Type supplier name...", key="search_query")

    filtered = supplier_master.copy()
    if search_q:
        filtered = filtered[filtered["supplier_name"].str.lower().str.contains(search_q.lower(), na=False)]

    cols_tbl = ["supplier_name","orders","total_spend","on_time_rate","defect_rate","avg_price","performance_score","risk_flag"]
    show_table(with_rank(format_for_display(filtered.sort_values("performance_score", ascending=False), cols_tbl)), TOP_N)

    with st.expander("How scores and risk flags are calculated"):
        st.markdown("""
| Metric | Definition |
|---|---|
| **On-Time Rate** | % of orders delivered on or before promised date (missing delivery dates are excluded, not penalized) |
| **Defect Rate** | Average rejection rate across all quality inspections linked to this supplier |
| **Performance Score** | Composite: Delivery 45% + Quality Quality 35% + Price Competitiveness 20% |
| **Price Score** | 100 × (1 − avg_price / max_avg_price in peer group) — higher = more competitive |
| **🔴 Quality Risk** | Defect Rate ≥ 8% |
| **🟠 Delivery Risk** | On-Time Rate ≤ 85% |
| **🟡 Cost Risk** | Price Score ≤ 40 / 100 |
| **🟢 Strategic** | No risk flags triggered |
""")


# ═══════════════════════════════════════════════════════
# TAB 2 · SUPPLIER INTELLIGENCE
# ═══════════════════════════════════════════════════════
with tab_intel:
    st.subheader("Supplier Intelligence Card")
    st.caption(
        "Look up any supplier before placing an order. "
        "Surfaces performance data, internal team knowledge, pricing vs. market, and a plain-English recommendation."
    )

    selected = st.selectbox("Select a supplier:", ["— Choose a supplier —"] + all_suppliers, key="intel_supplier")

    if selected and selected != "— Choose a supplier —":
        row  = supplier_master[supplier_master["supplier_name"] == selected]
        if row.empty:
            st.warning("No data found for this supplier.")
        else:
            row  = row.iloc[0]
            note = get_full_note(supplier_notes, selected)
            risk = row["risk_flag"]

            if "Quality"  in risk: st.error(  f"STOP — {risk}: Review quality history carefully before proceeding.")
            elif "Delivery" in risk: st.warning(f"CAUTION — {risk}: Build additional schedule buffer into this order.")
            elif "Cost"   in risk: st.info(   f"NOTE — {risk}: Not price-competitive. Solicit at least 2 alternative quotes.")
            else:                   st.success( f"CLEAR — {risk}: Consistently strong performer.")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("On-Time Rate",     _fmt_pct(row["on_time_rate"]),
                      delta=("Above 90% target" if row["on_time_rate"] >= 90 else "Below 90% target"),
                      delta_color=("normal" if row["on_time_rate"] >= 90 else "inverse"))
            m2.metric("Defect Rate",      _fmt_pct(row["defect_rate"]),
                      delta=("Within 5% threshold" if row["defect_rate"] <= 5 else "Exceeds 5% threshold"),
                      delta_color=("normal" if row["defect_rate"] <= 5 else "inverse"))
            m3.metric("Performance Score",_fmt_score(row["performance_score"]) + " / 100")
            m4.metric("Total Spend",      _fmt_money(row["total_spend"]))
            m5.metric("Avg Quoted Price", _fmt_money_2(row["avg_price"]) + " /unit")

            st.markdown("---")
            left, right = st.columns([1.05, 1])

            with left:
                if note:
                    st.markdown("**Internal Notes (Tribal Knowledge)**")
                    desc = note.get("descriptor","")
                    if desc:
                        st.markdown(f"*Team assessment: **{desc}***")
                    for b in note.get("bullets",[]):
                        st.markdown(f"- {b}")
                else:
                    st.info("No internal notes found for this supplier.")

                st.markdown("---")
                st.markdown("**Recommendation**")
                rec = []
                if row["on_time_rate"] < 85:
                    rec.append(f"Add **{max(1, round((90 - row['on_time_rate'])/10))} week(s)** of schedule buffer — on-time rate is {_fmt_pct(row['on_time_rate'])}.")
                if row["defect_rate"] > 5:
                    rec.append(f"Require **100% incoming inspection** — defect rate is {_fmt_pct(row['defect_rate'])}.")
                if row["price_score"] <= 40:
                    rec.append(f"**Solicit competing quotes** — price score is {_fmt_score(row['price_score'])}/100.")
                if rec:
                    for r_ in rec: st.warning(r_)
                else:
                    st.success(f"No concerns flagged. {selected} is a reliable sourcing choice.")

                if "rfq_id" in rfqs.columns:
                    sup_rfqs_raw = rfqs[rfqs["supplier_name"] == selected].copy()
                    sup_rfqs_raw["quoted_price"] = pd.to_numeric(sup_rfqs_raw["quoted_price"], errors="coerce")
                    sup_rfqs_raw = sup_rfqs_raw[sup_rfqs_raw["quoted_price"].notna()]
                    if not sup_rfqs_raw.empty:
                        sup_ids = set(sup_rfqs_raw["rfq_id"].astype(str))
                        mkt     = rfqs[rfqs["rfq_id"].astype(str).isin(sup_ids)].copy()
                        mkt["quoted_price"] = pd.to_numeric(mkt["quoted_price"], errors="coerce")
                        mkt = mkt[mkt["quoted_price"].gt(0) & mkt["quoted_price"].notna()]
                        if not mkt.empty:
                            best_mkt = mkt.groupby("rfq_id")["quoted_price"].min().rename("best")
                            comp_df  = sup_rfqs_raw.join(best_mkt, on="rfq_id")
                            premium  = (comp_df["quoted_price"] - comp_df["best"]).mean()
                            st.markdown("---")
                            st.markdown("**Pricing vs. Market**")
                            if premium > 0:
                                st.warning(f"Avg {_fmt_money_2(premium)}/unit **above** best market quote across comparable RFQ lines.")
                            else:
                                st.success(f"Avg {_fmt_money_2(abs(premium))}/unit **at or below** best market quote — competitive.")

            with right:
                st.markdown("**Recent Order History**")
                sup_orders = orders[orders["supplier_name"] == selected].copy()
                if not sup_orders.empty:
                    sov = sup_orders[sup_orders["actual_delivery_date"].notna() & sup_orders["promised_date"].notna()].copy()
                    sov["on_time"]  = sov["actual_delivery_date"] <= sov["promised_date"]
                    sov["days_diff"]= (sov["actual_delivery_date"] - sov["promised_date"]).dt.days
                    sov["Status"]   = sov["on_time"].map({True:"✅ On Time", False:"❌ Late"})
                    sov["Variance"] = sov["days_diff"].apply(
                        lambda x: f"-{abs(int(x))}d early" if x < 0 else (f"+{int(x)}d late" if x > 0 else "On schedule")
                    )
                    disp = [c for c in ["order_id","part_description","order_date","Status","Variance","po_amount"]
                            if c in sov.columns or c in ["Status","Variance"]]
                    show_table(sov[disp].sort_values("order_date", ascending=False) if "order_date" in disp else sov[disp], max_rows=8)
                else:
                    st.info("No order history found.")

                st.markdown("**Quality Inspections**")
                sup_q = q[q["supplier_name"] == selected].copy() if "supplier_name" in q.columns else pd.DataFrame()
                if not sup_q.empty:
                    q_disp = [c for c in ["inspection_date","parts_inspected","parts_rejected","rejection_reason","rework_required"] if c in sup_q.columns]
                    sup_q_show = sup_q[q_disp].copy()
                    if {"parts_inspected","parts_rejected"}.issubset(sup_q_show.columns):
                        sup_q_show["Defect %"] = (
                            sup_q_show["parts_rejected"] / sup_q_show["parts_inspected"] * 100
                        ).round(1).astype(str) + "%"
                    show_table(sup_q_show.sort_values("inspection_date", ascending=False)
                               if "inspection_date" in sup_q_show.columns else sup_q_show, max_rows=8)
                else:
                    st.info("No inspection records found.")
    else:
        st.info("Select a supplier above to view their full intelligence card.")
        st.markdown("""
**Try these examples:**
- **QuickFab Industries** — High-risk supplier with documented repeat failures
- **Stellar Metalworks** — Gold standard performer
- **Apex Manufacturing** — Names were consolidated from 4 different variants found in the source data
        """)


# ═══════════════════════════════════════════════════════
# TAB 3 · SOURCING DECISION
# ═══════════════════════════════════════════════════════
with tab_decision:
    st.subheader("Sourcing Decision Support")
    st.caption("Filter and rank qualified suppliers for a specific part category and timeline. Adjust thresholds to reflect your order's requirements.")

    # Controls in a clean 3-column layout
    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 1.6])

    with ctrl1:
        st.markdown("**Timeline**")
        st.number_input("Decision needed in (days)", min_value=1, max_value=120, step=1, key="decision_in_days")
        deadline = date.today() + timedelta(days=int(st.session_state["decision_in_days"]))
        st.info(f"Deadline: **{deadline.strftime('%b %d, %Y')}**")

    with ctrl2:
        st.markdown("**Quality & Delivery Thresholds**")
        st.slider("Minimum On-Time Rate (%)", 0, 100, key="req_on_time")
        st.slider("Maximum Defect Rate (%)", 0, 20, key="max_defects")

    with ctrl3:
        st.markdown("**Scoring Weights**")
        st.caption("Adjust to reflect what matters most for this order.")
        wc1, wc2, wc3 = st.columns(3)
        wc1.slider("Delivery", 0.0, 1.0, step=0.05, key="w_delivery")
        wc2.slider("Quality",  0.0, 1.0, step=0.05, key="w_quality")
        wc3.slider("Cost",     0.0, 1.0, step=0.05, key="w_cost")
        ws = st.session_state["w_delivery"] + st.session_state["w_quality"] + st.session_state["w_cost"]
        if ws == 0: wd, wq, wc = 0.45, 0.35, 0.20
        else:       wd, wq, wc = st.session_state["w_delivery"]/ws, st.session_state["w_quality"]/ws, st.session_state["w_cost"]/ws
        st.caption(f"Effective weights: Delivery **{wd:.0%}** · Quality **{wq:.0%}** · Cost **{wc:.0%}**")

    st.markdown("---")
    st.markdown("**Part Category & Capability Scope**")

    col_cat1, col_cat2, col_cat3 = st.columns([2, 1.2, 1])
    with col_cat1:
        orders_cat = _add_part_category(orders)
        rfqs_cat   = _add_part_category(rfqs)
        all_cats   = sorted(set(orders_cat["part_category"].unique()) | set(rfqs_cat["part_category"].unique()))
        st.selectbox("Part Category", ["(All Categories)"] + all_cats, key="category_choice",
                     help="Filter to suppliers with demonstrated capability in this category")
    with col_cat2:
        st.radio("Evidence from", ["RFQs only","Orders only","Orders + RFQs"], key="capability_source",
                 help="Which data source to use when determining supplier capability in the chosen category")
    with col_cat3:
        st.slider("Min lines to qualify", 1, 10, step=1, key="min_lines",
                  help="Minimum number of orders or RFQ lines required in this category to be considered capable")
        st.checkbox("Show coverage detail", key="show_coverage")

    chosen_cat = st.session_state["category_choice"]

    # Build capability counts
    cap_parts = []
    if st.session_state["capability_source"] in ("Orders only","Orders + RFQs"):
        cap_parts.append(orders_cat.groupby(["supplier_name","part_category"]).size().reset_index(name="lines"))
    if st.session_state["capability_source"] in ("RFQs only","Orders + RFQs"):
        cap_parts.append(rfqs_cat.groupby(["supplier_name","part_category"]).size().reset_index(name="lines"))
    cap_counts = (
        pd.concat(cap_parts, ignore_index=True).groupby(["supplier_name","part_category"], as_index=False)["lines"].sum()
        if cap_parts else pd.DataFrame()
    )

    if st.session_state["show_coverage"] and not cap_counts.empty:
        cov = cap_counts if chosen_cat == "(All Categories)" else cap_counts[cap_counts["part_category"] == chosen_cat]
        st.caption("Supplier lines in selected category:")
        show_table(with_rank(format_for_display(cov.sort_values("lines", ascending=False),
                                                ["supplier_name","part_category","lines"])), max_rows=20)

    # Eligible suppliers
    if chosen_cat == "(All Categories)" or cap_counts.empty:
        eligible = set(supplier_master["supplier_name"].astype(str))
    else:
        eligible = set(cap_counts[
            (cap_counts["part_category"] == chosen_cat) &
            (cap_counts["lines"] >= st.session_state["min_lines"])
        ]["supplier_name"].astype(str))

    sc_orders = orders_cat[orders_cat["supplier_name"].astype(str).isin(eligible)]
    sc_rfqs   = rfqs_cat[rfqs_cat["supplier_name"].astype(str).isin(eligible)]
    if chosen_cat != "(All Categories)":
        sc_orders = sc_orders[sc_orders["part_category"] == chosen_cat]
        sc_rfqs   = sc_rfqs[sc_rfqs["part_category"] == chosen_cat]

    if sc_orders.empty:
        if chosen_cat != "(All Categories)":
            st.warning("No suppliers found for this category / evidence combination. Showing overall rankings.")
        decision_kpi = supplier_master.copy()
    else:
        sc_spend = sc_orders.groupby("supplier_name", dropna=False)["po_amount"].sum().reset_index(name="total_spend")
        sc_ot = sc_orders[sc_orders["actual_delivery_date"].notna() & sc_orders["promised_date"].notna()].copy()
        sc_ot["on_time"] = (sc_ot["actual_delivery_date"] <= sc_ot["promised_date"]).astype(float)
        sc_otr = sc_ot.groupby("supplier_name", dropna=False)["on_time"].mean().reset_index()
        sc_otr["on_time_rate"] = (sc_otr["on_time"] * 100).round(1)
        sc_otr = sc_otr.drop(columns=["on_time"])

        sc_q = quality.merge(sc_orders[["order_id","supplier_name"]], on="order_id", how="inner")
        if {"parts_rejected","parts_inspected"}.issubset(sc_q.columns) and len(sc_q):
            sc_q["defect_rate"] = (sc_q["parts_rejected"]/sc_q["parts_inspected"]).replace([pd.NA, float("inf")], 0)
            sc_def = sc_q.groupby("supplier_name", dropna=False)["defect_rate"].mean().reset_index()
            sc_def["defect_rate"] = (sc_def["defect_rate"]*100).round(1)
        else:
            sc_def = pd.DataFrame({"supplier_name":sc_spend["supplier_name"],"defect_rate":0.0})

        sc_rp = sc_rfqs.copy()
        sc_rp["quoted_price"] = pd.to_numeric(sc_rp["quoted_price"], errors="coerce")
        sc_rp = sc_rp[sc_rp["quoted_price"].gt(0) & sc_rp["quoted_price"].notna()]
        sc_ap = (
            sc_rp.groupby("supplier_name", dropna=False)["quoted_price"].mean().reset_index(name="avg_price")
            if not sc_rp.empty
            else pd.DataFrame({"supplier_name":sc_spend["supplier_name"],"avg_price":0.0})
        )
        sc_ap["avg_price"] = sc_ap["avg_price"].round(2)

        decision_kpi = (
            sc_spend.merge(sc_otr, on="supplier_name", how="left")
                    .merge(sc_def, on="supplier_name", how="left")
                    .merge(sc_ap,  on="supplier_name", how="left")
        ).fillna({"on_time_rate":0.0,"defect_rate":0.0,"avg_price":0.0})

        mp2 = decision_kpi["avg_price"].replace(0, pd.NA).max()
        decision_kpi["price_score"] = (
            (100*(1-decision_kpi["avg_price"]/mp2)).clip(0,100).fillna(0)
            if pd.notna(mp2) and mp2 > 0 else 0.0
        )
        decision_kpi["performance_score"] = (
            decision_kpi["on_time_rate"]*wd +
            (100 - decision_kpi["defect_rate"])*wq +
            decision_kpi["price_score"]*wc
        ).round(1)
        decision_kpi["risk_flag"] = decision_kpi.apply(risk_flag, axis=1)

    decision_kpi["fit"] = (
        (decision_kpi["on_time_rate"] >= st.session_state["req_on_time"]) &
        (decision_kpi["defect_rate"]  <= st.session_state["max_defects"])
    )
    decision_kpi["fit_status"] = decision_kpi["fit"].map({True:"✅ Meets criteria", False:"❌ Below threshold"})
    decision_kpi["notes_hint"] = decision_kpi["supplier_name"].apply(lambda s: note_snippet(supplier_notes, s))

    n_fit   = int(decision_kpi["fit"].sum())
    n_total = len(decision_kpi)

    st.markdown("---")

    if n_fit == 0:
        st.warning(
            f"No suppliers meet the current thresholds "
            f"(≥{st.session_state['req_on_time']}% on-time, ≤{st.session_state['max_defects']}% defects). "
            "Consider relaxing criteria or selecting a broader category."
        )
    else:
        st.success(f"**{n_fit} of {n_total} suppliers** meet your thresholds — ranked below by performance score.")

    ranked = decision_kpi.sort_values(["fit","performance_score"], ascending=[False,False])
    show_table(with_rank(format_for_display(ranked, [
        "supplier_name","fit_status","performance_score","risk_flag",
        "on_time_rate","defect_rate","avg_price","total_spend","notes_hint"
    ])), TOP_N)

    st.markdown("---")
    st.subheader("Pricing Consolidation Opportunities")
    st.caption(f"Scope: **{chosen_cat}** — suppliers paying more than the best available market price for the same RFQ line.")
    impact_dec = build_pricing_impact(supplier_master, rfqs, chosen_cat)
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Est. Annual Savings Available", _fmt_money(float(impact_dec["estimated_overpay"].sum())),
                  help="Based on avg price premium × estimated units across all suppliers in scope")
    col_m2.metric("Suppliers with Pricing Premium", str(int((impact_dec["estimated_overpay"] > 0).sum())))
    show_table(with_rank(format_for_display(
        impact_dec.sort_values("estimated_overpay", ascending=False),
        ["supplier_name","total_spend","avg_price","avg_delta_vs_best","estimated_overpay","risk_flag","rfqs","pct_not_best"]
    )), TOP_N)


# ═══════════════════════════════════════════════════════
# TAB 4 · PERFORMANCE TRENDS
# ═══════════════════════════════════════════════════════
with tab_trends:
    st.subheader("Performance Trends")
    st.caption("Track supplier reliability over time. Quickly spot who is improving, who is declining, and who has chronic issues.")

    if "order_date" not in orders.columns:
        st.warning("No `order_date` column found — trends unavailable.")
    else:
        top6 = supplier_master.sort_values("total_spend", ascending=False)["supplier_name"].head(6).tolist()
        if not st.session_state.get("trend_supplier_filter"):
            st.session_state["trend_supplier_filter"] = top6

        tf = st.multiselect(
            "Select suppliers to compare:",
            options=all_suppliers,
            default=st.session_state.get("trend_supplier_filter", top6),
            key="trend_supplier_filter",
        )
        if not tf: tf = top6

        ot_trend = ot_valid[ot_valid["supplier_name"].isin(tf)].copy()
        ot_trend["month"] = ot_trend["order_date"].dt.to_period("M").dt.to_timestamp()

        # ── Chart 1: On-Time Rate Line Chart (cleaner than heatmap) ──
        st.markdown("#### On-Time Delivery Rate — Monthly Trend")
        st.caption("Track reliability over time. The red zone below 85% is where delivery risk is flagged. A falling line signals a supplier in decline.")

        if ot_trend.empty:
            st.info("No delivery data for selected suppliers.")
        else:
            monthly_otr = ot_trend.groupby(["month","supplier_name"])["on_time"].agg(["mean","count"]).reset_index()
            monthly_otr.columns = ["month","supplier_name","on_time_rate","n_orders"]
            monthly_otr = monthly_otr[monthly_otr["n_orders"] >= 1].copy()
            monthly_otr["otr_pct"] = (monthly_otr["on_time_rate"] * 100).round(1)

            # Red zone background
            red_zone = alt.Chart(pd.DataFrame({"y1":[0],"y2":[85]})).mark_rect(
                color="#FEE2E2", opacity=0.4
            ).encode(y="y1:Q", y2="y2:Q")

            line = (
                alt.Chart(monthly_otr)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y", labelAngle=-30, tickCount="month")),
                    y=alt.Y("otr_pct:Q", title="On-Time Rate (%)", scale=alt.Scale(domain=[0,105])),
                    color=alt.Color("supplier_name:N", title="Supplier"),
                    tooltip=[
                        alt.Tooltip("supplier_name:N", title="Supplier"),
                        alt.Tooltip("month:T",         title="Month",     format="%B %Y"),
                        alt.Tooltip("otr_pct:Q",       title="On-Time %", format=".1f"),
                        alt.Tooltip("n_orders:Q",      title="# Orders"),
                    ]
                )
            )
            threshold = alt.Chart(pd.DataFrame({"y":[85]})).mark_rule(
                color="#EF4444", strokeDash=[6,3], strokeWidth=1.8, opacity=0.8
            ).encode(y="y:Q")
            thresh_label = alt.Chart(pd.DataFrame({"y":[85],"label":["Risk threshold: 85%"]})).mark_text(
                align="left", dx=6, dy=-7, fontSize=10, color="#EF4444"
            ).encode(y="y:Q", text="label:N")

            st.altair_chart(
                (red_zone + line + threshold + thresh_label).properties(height=320).resolve_scale(color="independent"),
                use_container_width=True
            )

        st.markdown("---")

        # ── Chart 2: Defect Rate Over Time ──
        st.markdown("#### Defect Rate — Quarterly Rolling Average")
        st.caption("Smoothed to reduce noise from single-order anomalies. A rising line signals a systemic quality problem emerging.")

        if "inspection_date" in quality.columns:
            q_trend = q[q["supplier_name"].isin(tf)].copy()
            q_trend["quarter"] = q_trend["inspection_date"].dt.to_period("Q").dt.to_timestamp()
            if not q_trend.empty:
                q_quarterly = q_trend.groupby(["quarter","supplier_name"])["defect_rate"].agg(["mean","count"]).reset_index()
                q_quarterly.columns = ["quarter","supplier_name","defect_rate","n_insp"]
                q_quarterly = q_quarterly[q_quarterly["n_insp"] >= 1].copy()
                q_quarterly["defect_pct"] = (q_quarterly["defect_rate"] * 100).round(2)

                red_zone_q = alt.Chart(pd.DataFrame({"y1":[5],"y2":[100]})).mark_rect(
                    color="#FEE2E2", opacity=0.35
                ).encode(y="y1:Q", y2="y2:Q")

                defect_line = (
                    alt.Chart(q_quarterly)
                    .mark_line(point=True, strokeWidth=2.5)
                    .encode(
                        x=alt.X("quarter:T", title=None, axis=alt.Axis(format="Q%q %Y", labelAngle=-20)),
                        y=alt.Y("defect_pct:Q", title="Defect Rate (%)", scale=alt.Scale(zero=True)),
                        color=alt.Color("supplier_name:N", title="Supplier"),
                        tooltip=[
                            alt.Tooltip("supplier_name:N", title="Supplier"),
                            alt.Tooltip("quarter:T",       title="Quarter",    format="%Y Q%q"),
                            alt.Tooltip("defect_pct:Q",    title="Defect Rate (%)", format=".2f"),
                            alt.Tooltip("n_insp:Q",        title="# Inspections"),
                        ]
                    )
                )
                defect_threshold = alt.Chart(pd.DataFrame({"y":[5]})).mark_rule(
                    color="#EF4444", strokeDash=[6,3], strokeWidth=1.8, opacity=0.8
                ).encode(y="y:Q")
                defect_thresh_label = alt.Chart(pd.DataFrame({"y":[5],"label":["Action threshold: 5%"]})).mark_text(
                    align="left", dx=6, dy=-7, fontSize=10, color="#EF4444"
                ).encode(y="y:Q", text="label:N")

                st.altair_chart(
                    (red_zone_q + defect_line + defect_threshold + defect_thresh_label)
                    .properties(height=280).resolve_scale(color="independent"),
                    use_container_width=True
                )
        else:
            st.info("No inspection date column available for quality trends.")

        st.markdown("---")

        # ── Chart 3: Days Late Distribution ──
        st.markdown("#### Late Deliveries — How Late? (Days Late per Order)")
        st.caption("Each bar = average days late per month for that supplier. Taller bars = bigger scheduling impact. Only late orders shown.")

        late_df = ot_trend[ot_valid["supplier_name"].isin(tf)].copy() if not ot_trend.empty else pd.DataFrame()
        if not late_df.empty:
            late_df["days_late"] = (late_df["actual_delivery_date"] - late_df["promised_date"]).dt.days
            late_df = late_df[late_df["days_late"] > 0].copy()

        if not late_df.empty and len(late_df):
            late_monthly = late_df.groupby(["month","supplier_name"]).agg(
                avg_days_late=("days_late","mean"),
                n_late=("days_late","count"),
            ).reset_index()
            late_monthly["avg_days_late"] = late_monthly["avg_days_late"].round(1)

            late_bars = (
                alt.Chart(late_monthly)
                .mark_bar(opacity=0.85, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y", labelAngle=-30, tickCount="month")),
                    y=alt.Y("avg_days_late:Q", title="Avg Days Late"),
                    color=alt.Color("supplier_name:N", title="Supplier"),
                    xOffset=alt.XOffset("supplier_name:N"),
                    tooltip=[
                        alt.Tooltip("supplier_name:N",  title="Supplier"),
                        alt.Tooltip("month:T",          title="Month",        format="%B %Y"),
                        alt.Tooltip("avg_days_late:Q",  title="Avg Days Late",format=".1f"),
                        alt.Tooltip("n_late:Q",         title="# Late Orders"),
                    ]
                )
                .properties(height=260)
            )
            st.altair_chart(late_bars, use_container_width=True)
            st.caption(f"Showing {len(late_df)} late orders across selected suppliers. Grouped bars = same month, different suppliers.")
        else:
            st.success("No late deliveries recorded for the selected suppliers in the available data.")

        st.markdown("---")

        # ── Chart 4: Spend Concentration Over Time ──
        st.markdown("#### Spend Share — Monthly Allocation")
        st.caption("Each band's height = share of monthly spend. A growing band from a single supplier signals concentration risk building over time.")

        spend_trend = orders[orders["supplier_name"].isin(tf)].copy()
        spend_trend["month"] = spend_trend["order_date"].dt.to_period("M").dt.to_timestamp()
        ms = spend_trend.groupby(["month","supplier_name"])["po_amount"].sum().reset_index()
        ms["spend_k"] = (ms["po_amount"] / 1000).round(1)

        if not ms.empty:
            area = (
                alt.Chart(ms)
                .mark_area(opacity=0.75, interpolate="monotone")
                .encode(
                    x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y", labelAngle=-30)),
                    y=alt.Y("spend_k:Q", title="Share of Monthly Spend",
                             stack="normalize", axis=alt.Axis(format=".0%")),
                    color=alt.Color("supplier_name:N", title="Supplier"),
                    tooltip=[
                        alt.Tooltip("supplier_name:N",title="Supplier"),
                        alt.Tooltip("month:T",        title="Month",    format="%B %Y"),
                        alt.Tooltip("spend_k:Q",      title="Spend ($K)", format=".1f"),
                    ]
                )
                .properties(height=260)
            )
            st.altair_chart(area, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 5 · FINANCIAL IMPACT
# ═══════════════════════════════════════════════════════
with tab_financial:
    st.subheader("Financial Impact Analysis")
    st.caption("Quantifies the cost of supplier underperformance: pricing leakage, quality rework cost, and delivery risk exposure.")

    cat_fin       = st.session_state.get("category_choice","(All Categories)")
    impact_df     = build_pricing_impact(supplier_master, rfqs, cat_fin)
    impact_df["defect_cost"] = (impact_df["total_spend"] * (impact_df["defect_rate"]/100) * 0.5).round(0)
    impact_df["total_risk"]  = impact_df["estimated_overpay"] + impact_df["defect_cost"]
    late_spend_fin = float(impact_df.loc[impact_df["on_time_rate"] < 85,"total_spend"].sum())
    total_pricing  = float(impact_df["estimated_overpay"].sum())
    total_quality  = float(impact_df["defect_cost"].sum())
    total_risk     = total_pricing + total_quality

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Quantified Risk",      _fmt_money(total_risk),
              help="Pricing leakage + est. quality cost combined")
    f2.metric("Pricing Leakage",            _fmt_money(total_pricing),
              help="Avg premium above best comparable RFQ quote × estimated units")
    f3.metric("Est. Quality / Rework Cost", _fmt_money(total_quality),
              help="Defect rate × total spend × 0.5 rework cost factor")
    f4.metric("Spend at Delivery Risk",     _fmt_money(late_spend_fin),
              help="Total spend with suppliers below 85% on-time rate")

    st.markdown("---")

    # ── Chart 1: Combined risk bar — ALL suppliers, full names, left-aligned ──
    st.markdown("#### Total Financial Exposure by Supplier")
    st.caption("Pricing leakage + est. quality cost. All suppliers shown, sorted by total exposure. Supplier names are fully displayed.")

    # Include ALL suppliers (not just top N), full name display
    risk_plot = impact_df[impact_df["total_risk"] > 0].sort_values("total_risk", ascending=True).copy()

    if not risk_plot.empty:
        melt = pd.melt(
            risk_plot[["supplier_name","estimated_overpay","defect_cost"]],
            id_vars="supplier_name", var_name="type", value_name="cost"
        )
        melt["type"] = melt["type"].map({
            "estimated_overpay": "Pricing Leakage",
            "defect_cost":       "Quality / Rework Cost",
        })
        melt = melt[melt["cost"] > 0]

        stacked = (
            alt.Chart(melt)
            .mark_bar()
            .encode(
                x=alt.X("cost:Q",
                         title="Estimated Cost ($)",
                         axis=alt.Axis(format="$,.0f", labelAngle=0)),
                y=alt.Y("supplier_name:N",
                         sort=alt.EncodingSortField(field="cost", op="sum", order="descending"),
                         title=None,
                         axis=alt.Axis(
                             labelLimit=220,   # wide enough for full names
                             labelFontSize=12,
                             labelAlign="right",
                         )),
                color=alt.Color("type:N",
                                scale=alt.Scale(
                                    domain=["Pricing Leakage","Quality / Rework Cost"],
                                    range=["#FF7F0E","#D62728"]
                                ),
                                title="Cost Type",
                                legend=alt.Legend(orient="bottom")),
                tooltip=[
                    alt.Tooltip("supplier_name:N", title="Supplier"),
                    alt.Tooltip("type:N",          title="Cost Type"),
                    alt.Tooltip("cost:Q",          title="Est. Cost ($)", format="$,.0f"),
                ]
            )
            .properties(height=max(280, len(risk_plot) * 38))
        )
        st.altair_chart(stacked, use_container_width=True)

    st.markdown("---")

    # ── Chart 2: Pricing premium bar — all suppliers ──
    st.markdown("#### Pricing Premium vs. Best Market Quote")
    st.caption("How much more does each supplier charge on average vs. the lowest competing quote for the exact same RFQ line? Zero = fully competitive.")

    price_plot = impact_df[impact_df["avg_delta_vs_best"] > 0].sort_values("avg_delta_vs_best", ascending=True).copy()
    if not price_plot.empty:
        price_bar = (
            alt.Chart(price_plot)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("avg_delta_vs_best:Q",
                         title="Avg Price Premium vs Best Quote ($/unit)",
                         scale=alt.Scale(zero=True)),
                y=alt.Y("supplier_name:N",
                         sort=alt.EncodingSortField(field="avg_delta_vs_best", order="descending"),
                         title=None,
                         axis=alt.Axis(
                             labelLimit=220,
                             labelFontSize=12,
                             labelAlign="right",
                         )),
                color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk", legend=None),
                tooltip=[
                    alt.Tooltip("supplier_name:N",     title="Supplier"),
                    alt.Tooltip("avg_delta_vs_best:Q", title="Premium ($/unit)",   format=".2f"),
                    alt.Tooltip("pct_not_best:Q",      title="% Quotes Not Best",  format=".1f"),
                    alt.Tooltip("rfqs:Q",              title="# RFQs"),
                    alt.Tooltip("estimated_overpay:Q", title="Est. Leakage ($)",   format="$,.0f"),
                ]
            )
            .properties(height=max(220, len(price_plot) * 36))
        )
        st.altair_chart(price_bar, use_container_width=True)

    st.markdown("---")

    # ── Chart 3: Performance vs Spend ──
    st.markdown("#### Spend Concentration vs. Performance")
    st.caption("**Bottom-right = highest risk**: large spend on a poorly performing supplier. These are your most urgent intervention targets.")

    sp_vs_perf = impact_df.copy()
    sp_vs_perf["spend_m"] = (sp_vs_perf["total_spend"] / 1e6).round(3)

    scatter2 = (
        alt.Chart(sp_vs_perf)
        .mark_circle(size=150, opacity=0.82, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("spend_m:Q",          title="Total Spend ($M)",           scale=alt.Scale(zero=True)),
            y=alt.Y("performance_score:Q", title="Performance Score (0–100)",  scale=alt.Scale(domain=[0,100])),
            color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk"),
            tooltip=[
                alt.Tooltip("supplier_name:N",     title="Supplier"),
                alt.Tooltip("spend_m:Q",           title="Spend ($M)",            format=".3f"),
                alt.Tooltip("performance_score:Q", title="Performance Score",      format=".1f"),
                alt.Tooltip("estimated_overpay:Q", title="Pricing Leakage ($)",    format="$,.0f"),
                alt.Tooltip("defect_cost:Q",       title="Quality Cost ($)",       format="$,.0f"),
                alt.Tooltip("on_time_rate:Q",      title="On-Time Rate (%)",       format=".1f"),
                alt.Tooltip("defect_rate:Q",       title="Defect Rate (%)",        format=".1f"),
                alt.Tooltip("risk_flag:N",         title="Risk Flag"),
            ]
        )
        .properties(height=340)
    )
    ref = alt.Chart(pd.DataFrame({"y":[75]})).mark_rule(
        color="#d1d5db", strokeDash=[5,5], strokeWidth=1.5
    ).encode(y="y:Q")

    # Add supplier name labels
    labels = (
        alt.Chart(sp_vs_perf)
        .mark_text(align="left", dx=8, dy=0, fontSize=10, color="#374151")
        .encode(
            x=alt.X("spend_m:Q"),
            y=alt.Y("performance_score:Q"),
            text=alt.Text("supplier_name:N"),
        )
    )
    st.altair_chart(scatter2 + ref + labels, use_container_width=True)
    st.caption("Dashed line = 75 performance score reference. Bottom-right quadrant = large spend, weak performance.")

    st.markdown("---")

    with st.expander("Full impact table — all suppliers"):
        show_table(with_rank(format_for_display(
            impact_df.sort_values("total_risk", ascending=False),
            ["supplier_name","total_spend","on_time_rate","defect_rate",
             "avg_price","avg_delta_vs_best","pct_not_best","rfqs",
             "estimated_overpay","defect_cost","performance_score","risk_flag"]
        )), max_rows=50)

    with st.expander("Methodology & data notes"):
        st.markdown("""
**Pricing Leakage**
The avg price premium is computed *within the same RFQ line* — never comparing prices across different parts.
Estimated Units = total spend ÷ avg quoted price. Leakage = Premium × Estimated Units.

**Quality Cost**
Est. Rework Cost = total spend × defect rate × 0.5 (conservative rework factor).
Actual costs vary by part complexity, labor rates, and re-inspection requirements.

**Delivery Risk**
Spend flagged "at risk" = suppliers with on-time rate < 85%. Orders with missing delivery dates are excluded, not penalized.

**Roadmap to Higher Precision**
- Connect SAP/ERP goods receipts for actual received quantities (removes the spend-derived unit estimate)
- Add commodity price benchmarks for should-cost validation
- Track supplier trend direction (improving vs. declining) to enable predictive alerts
- Incorporate freight and expedite costs into the total cost of ownership model
""")

    with st.expander("Debug — column names"):
        st.write("Orders:", list(orders.columns))
        st.write("Quality:", list(quality.columns))
        st.write("RFQs:", list(rfqs.columns))
