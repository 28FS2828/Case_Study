"""
Hoth Industries — Supplier Intelligence Platform (Interview-Ready, Executive Polished)

Fixes (requested):
1) ✅ Apex avg quote price now populates (entity resolution carried through RFQs + supplier_master)
2) ✅ Apex internal notes now populate (notes parser supports multi-name headers like "APEX MANUFACTURING / APEX MFG ...")
3) ✅ Executive Overview uses simpler terms (Estimated Overpay instead of Pricing Leakage)
4) ✅ Performance Trends: supplier selector is a dropdown (Top 6 / All / Single) + window expanded (up to 48 months)
5) ✅ No "row shrink" issues: raw data stays raw; aggregations only for KPIs/charts/tables

Update (new):
6) ✅ Executive Overview: "Priority Actions" replaced with "Executive Strategic Recommendations"
   - Recommendations are tied to data and, where possible, name credible alternates (same category coverage, low risk).

Run:
    streamlit run app.py
Data:
    supplier_orders.csv, quality_inspections.csv, rfq_responses.csv, supplier_notes.txt
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
.block-container { padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1400px; }
h1 { font-size: 1.65rem; font-weight: 750; letter-spacing: -0.03em; }
h2 { font-size: 1.25rem; font-weight: 650; letter-spacing: -0.02em; margin-top: 0.2rem; }
h3 { font-size: 1.05rem; font-weight: 650; }
div[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 750; }
div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #6b7280; }
div[data-testid="stExpander"] { border-radius: 10px; border: 1px solid #e5e7eb; }
div[data-testid="stTabs"] button { font-size: 0.92rem; font-weight: 550; }
hr { margin: 1.1rem 0; }
div[data-testid="stDataFrame"] { border-radius: 10px; }

.hoth-card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px 14px; background: white; }
.hoth-card .title { font-weight: 750; font-size: 1.02rem; margin-bottom: 2px; }
.hoth-card .sub { color: #6b7280; font-size: 0.88rem; line-height: 1.25rem; }

.hoth-badge {
  display:inline-block;
  min-width: 150px;
  text-align:center;
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.02em;
  white-space: normal;
  line-height: 1.05rem;
  color: #fff;
}
.hoth-note-box{
  border: 1px solid #e5e7eb;
  background: #fafafa;
  border-radius: 10px;
  padding: 12px 12px;
}
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
    # Trends controls
    "trend_supplier_mode": "Top 6 by spend",
    "trend_single_supplier": "",
    "trend_focus": "On-Time Trend",
    "trend_months": 12,
}

RESET_FLAG = "__reset__"

RISK_ORDER = ["🔴 Quality Risk", "🟠 Delivery Risk", "🟡 Cost Risk", "🟢 Strategic"]
RISK_COLORS = ["#D62728", "#FF7F0E", "#F2C12E", "#2CA02C"]
risk_color_scale = alt.Scale(domain=RISK_ORDER, range=RISK_COLORS)

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "l.l.c", "ltd", "limited",
    "corp", "corporation", "co", "company", "gmbh", "s.a", "sa",
}

DISPLAY_COLS = {
    "supplier_name": "Supplier",
    "risk_flag": "Risk",
    "fit_status": "Fit",
    "notes_hint": "Tribal Knowledge",
    "total_spend": "Total Spend ($)",
    "avg_price": "Avg Quote Price ($/unit)",
    "avg_delta_vs_best": "Avg Overpay vs Best ($/unit)",
    "on_time_rate": "On-Time Rate (%)",
    "defect_rate": "Defect Rate (%)",
    "price_score": "Price Score (0-100)",
    "performance_score": "Performance Score (0-100)",
    "estimated_overpay": "Est. Overpay ($)",
    "defect_cost": "Est. Quality / Rework Cost ($)",
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

# ─────────────────────────────────────────────
# ENTITY RESOLUTION (CRITICAL FOR APEX)
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

MANUAL_KEY_MAP = {
    "apex mfg": "apex manufacturing",
    "apex mfg inc": "apex manufacturing",
    "apex manufacturing inc": "apex manufacturing",
    "apex manufacturing": "apex manufacturing",
}

MANUAL_DISPLAY = {
    "apex manufacturing": "Apex Manufacturing",
}

def add_supplier_key(df: pd.DataFrame, col: str = "supplier_name") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    out["supplier_key"] = out[col].apply(normalize_supplier_key).replace(MANUAL_KEY_MAP)
    return out

def apply_entity_resolution(df: pd.DataFrame, col: str = "supplier_name") -> pd.DataFrame:
    """Canonicalize names within a dataframe, and attach supplier_key."""
    if df is None or df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    out["_key"] = out[col].apply(normalize_supplier_key).replace(MANUAL_KEY_MAP)

    def pick_longest(vals):
        v = list(set([s.strip() for s in vals if isinstance(s, str) and s.strip()]))
        return sorted(v, key=lambda x: (-len(x), x))[0] if v else ""

    canonical = out.groupby("_key")[col].agg(pick_longest).to_dict()
    out[col] = out["_key"].map(canonical).fillna(out[col])
    out = out.drop(columns=["_key"])
    out = add_supplier_key(out, col)
    return out

# ─────────────────────────────────────────────
# PART CATEGORIZATION
# ─────────────────────────────────────────────
PART_RULES = [
    ("Motors / Actuation", ["motor", "actuator", "servo", "stepper", "gearbox"]),
    ("Controls / Electronics", ["controller", "pcb", "board", "sensor", "wire", "harness", "electronic", "connector", "vfd", "plc"]),
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
    col = next((c for c in ["part_description", "commodity", "category", "component", "item", "description"] if c in out.columns), None)
    out["part_category"] = out[col].apply(categorize_part) if col else "Other / Unknown"
    return out

# ─────────────────────────────────────────────
# NOTES PARSER (SUPPORTS "NAME / NAME / NAME" HEADERS)
# ─────────────────────────────────────────────
def parse_supplier_notes_robust(text: str) -> dict:
    notes = {}
    if not text or not str(text).strip():
        return notes

    lines = [ln.rstrip("\n") for ln in str(text).splitlines()]

    def is_sep(ln: str) -> bool:
        s = (ln or "").strip()
        return len(s) >= 10 and all(ch == "=" for ch in s)

    i = 0
    while i < len(lines):
        if not is_sep(lines[i].strip()):
            i += 1
            continue

        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            break
        header = lines[j].strip()

        k = j + 1
        while k < len(lines) and not is_sep(lines[k].strip()):
            if lines[k].strip():
                break
            k += 1
        body_start = (k + 1) if (k < len(lines) and is_sep(lines[k].strip())) else (j + 1)

        body_lines = []
        m = body_start
        while m < len(lines):
            if is_sep(lines[m].strip()):
                look = m + 1
                while look < len(lines) and not lines[look].strip():
                    look += 1
                if look < len(lines) and lines[look].strip():
                    break
            body_lines.append(lines[m])
            m += 1

        descriptor = ""
        supplier_raw = header
        mm = re.match(r"^(.+?)\s*-\s*(.+)$", header)
        if mm:
            supplier_raw = mm.group(1).strip()
            descriptor = mm.group(2).strip()

        candidates = [supplier_raw]
        if "/" in supplier_raw:
            candidates = [c.strip() for c in supplier_raw.split("/") if c.strip()]
        elif "," in supplier_raw and len(supplier_raw) < 120:
            candidates = [c.strip() for c in supplier_raw.split(",") if c.strip()]

        body = "\n".join(body_lines).strip()

        for cand in candidates:
            key = normalize_supplier_key(cand)
            key = MANUAL_KEY_MAP.get(key, key)
            if not key:
                continue
            notes[key] = {"header": header, "descriptor": descriptor, "body": body}

        i = m

    return notes

def note_snippet(notes: dict, supplier_key: str) -> str:
    n = notes.get(supplier_key, {})
    if not n:
        return ""
    parts = []
    if n.get("descriptor"):
        parts.append(n["descriptor"])
    body = (n.get("body") or "").strip()
    if body:
        first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        if first_line:
            parts.append(first_line[:160] + ("..." if len(first_line) > 160 else ""))
    line = " | ".join([p for p in parts if p])
    return line[:220] + ("..." if len(line) > 220 else "")

def render_full_note(note: dict):
    desc = (note.get("descriptor") or "").strip()
    body = (note.get("body") or "").strip()

    if desc:
        st.markdown(f"*Team assessment: **{desc}***")

    if not body:
        st.info("No internal notes content found for this supplier.")
        return

    st.markdown("<div class='hoth-note-box'>", unsafe_allow_html=True)
    for para in re.split(r"\n\s*\n", body):
        p = para.strip()
        if not p:
            continue
        head_match = re.match(r"^([A-Za-z ]+from\s.+?\(\d{1,2}/\d{1,2}/\d{4}\):)\s*(.*)$", p, re.IGNORECASE | re.DOTALL)
        if head_match:
            head = head_match.group(1).strip()
            rest = head_match.group(2).strip()
            st.markdown(f"**{head}**")
            if rest:
                st.markdown(rest.replace("\n", "  \n"))
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        else:
            st.markdown(p.replace("\n", "  \n"))
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def safe_dt(df, col):
    if df is not None and col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def read_text_flexible(paths):
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            pass
    return ""

def try_load_local():
    candidates = [
        ("supplier_orders.csv", "quality_inspections.csv", "rfq_responses.csv", "supplier_notes.txt"),
        ("Copy_of_supplier_orders.csv", "Copy_of_quality_inspections.csv", "Copy_of_rfq_responses.csv", "supplier_notes.txt"),
        ("Copy of supplier_orders.csv", "Copy of quality_inspections.csv", "Copy of rfq_responses.csv", "supplier_notes.txt"),
        ("/mnt/data/supplier_orders.csv", "/mnt/data/quality_inspections.csv", "/mnt/data/rfq_responses.csv", "/mnt/data/supplier_notes.txt"),
    ]
    for op, qp, rp, np in candidates:
        try:
            o = pd.read_csv(op)
            q = pd.read_csv(qp)
            r = pd.read_csv(rp)
            n = read_text_flexible([np])
            return o, q, r, n
        except Exception:
            continue
    return None

local = try_load_local()
if local:
    orders_raw, quality_raw, rfqs_raw, supplier_notes_text = local
else:
    st.warning("Local data files not found. Please upload below.")
    with st.expander("Upload Data Files", expanded=True):
        c1, c2 = st.columns(2)
        uf_o = c1.file_uploader("supplier_orders.csv", type=["csv"])
        uf_q = c1.file_uploader("quality_inspections.csv", type=["csv"])
        uf_r = c2.file_uploader("rfq_responses.csv", type=["csv"])
        uf_n = c2.file_uploader("supplier_notes.txt", type=["txt"])
    if not all([uf_o, uf_q, uf_r]):
        st.info("Upload all three CSV files to continue.")
        st.stop()
    import io
    orders_raw = pd.read_csv(io.BytesIO(uf_o.read()))
    quality_raw = pd.read_csv(io.BytesIO(uf_q.read()))
    rfqs_raw = pd.read_csv(io.BytesIO(uf_r.read()))
    supplier_notes_text = uf_n.read().decode("utf-8", "ignore") if uf_n else ""

# Apply entity resolution everywhere (this is what prevents Apex mismatches)
orders = apply_entity_resolution(orders_raw, "supplier_name")
quality = quality_raw.copy()
rfqs = apply_entity_resolution(rfqs_raw, "supplier_name")

# Date parsing
for df, cols in [
    (orders, ["order_date", "promised_date", "actual_delivery_date"]),
    (quality, ["inspection_date"]),
    (rfqs, ["quote_date"]),
]:
    for c in cols:
        safe_dt(df, c)

# Notes parsing
supplier_notes = parse_supplier_notes_robust(supplier_notes_text)

# Build key -> display name map
def pick_longest(vals):
    v = list(set([s.strip() for s in vals if isinstance(s, str) and s.strip()]))
    return sorted(v, key=lambda x: (-len(x), x))[0] if v else ""

key_to_name = orders.groupby("supplier_key")["supplier_name"].agg(pick_longest).to_dict()
for k, v in MANUAL_DISPLAY.items():
    key_to_name[k] = v

def key_to_display(k: str) -> str:
    return key_to_name.get(k, (k or "").title())

# ─────────────────────────────────────────────
# BUILD SUPPLIER MASTER (KEYED BY supplier_key)
# ─────────────────────────────────────────────
required = {"order_id", "supplier_name", "supplier_key", "po_amount", "promised_date", "actual_delivery_date"}
missing = required - set(orders.columns)
if missing:
    st.error(f"Orders file missing columns: {sorted(missing)}")
    st.stop()

spend = orders.groupby("supplier_key", dropna=False)["po_amount"].sum().reset_index(name="total_spend")

ot_valid = orders[orders["actual_delivery_date"].notna() & orders["promised_date"].notna()].copy()
ot_valid["on_time"] = (ot_valid["actual_delivery_date"] <= ot_valid["promised_date"]).astype(float)
on_time = ot_valid.groupby("supplier_key", dropna=False)["on_time"].mean().reset_index()
on_time["on_time_rate"] = (on_time["on_time"] * 100).round(1)
on_time = on_time.drop(columns=["on_time"])

order_counts = orders.groupby("supplier_key", dropna=False)["order_id"].nunique().reset_index(name="orders")

if "order_id" not in quality.columns:
    st.error("quality_inspections.csv must contain 'order_id'.")
    st.stop()

q = quality.merge(orders[["order_id", "supplier_key"]], on="order_id", how="left")

if {"parts_rejected", "parts_inspected"}.issubset(q.columns):
    q["parts_inspected"] = pd.to_numeric(q["parts_inspected"], errors="coerce")
    q["parts_rejected"] = pd.to_numeric(q["parts_rejected"], errors="coerce")
    q["defect_rate"] = (q["parts_rejected"] / q["parts_inspected"]).replace([pd.NA, float("inf")], 0)
else:
    q["defect_rate"] = 0.0

defects = q.groupby("supplier_key", dropna=False)["defect_rate"].mean().reset_index()
defects["defect_rate"] = (defects["defect_rate"] * 100).round(1)

if not {"supplier_name", "supplier_key", "quoted_price"}.issubset(rfqs.columns):
    st.error("rfq_responses.csv must contain 'supplier_name' and 'quoted_price'.")
    st.stop()

ap = rfqs.copy()
ap["quoted_price"] = pd.to_numeric(ap["quoted_price"], errors="coerce")
ap = ap[ap["quoted_price"].gt(0) & ap["quoted_price"].notna()].copy()

ap_by_key = ap.groupby("supplier_key", dropna=False)["quoted_price"].mean().reset_index(name="avg_price")
ap_by_key["avg_price"] = ap_by_key["avg_price"].round(2)

supplier_master = (
    spend.merge(on_time, on="supplier_key", how="left")
    .merge(defects, on="supplier_key", how="left")
    .merge(ap_by_key, on="supplier_key", how="left")
    .merge(order_counts, on="supplier_key", how="left")
).fillna({"on_time_rate": 0.0, "defect_rate": 0.0, "avg_price": 0.0, "orders": 0})

supplier_master["supplier_name"] = supplier_master["supplier_key"].apply(key_to_display)

mp = supplier_master["avg_price"].replace(0, pd.NA).max()
supplier_master["price_score"] = (
    (100 * (1 - supplier_master["avg_price"] / mp)).clip(0, 100).fillna(0) if pd.notna(mp) and mp > 0 else 0.0
)
supplier_master["performance_score"] = (
    supplier_master["on_time_rate"] * 0.45
    + (100 - supplier_master["defect_rate"]) * 0.35
    + supplier_master["price_score"] * 0.20
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

all_suppliers = supplier_master["supplier_name"].dropna().unique().tolist()
all_suppliers = sorted(all_suppliers)

# ─────────────────────────────────────────────
# CAPABILITY / CATEGORY COVERAGE (GLOBAL)
# Used to propose alternates that plausibly make similar parts
# ─────────────────────────────────────────────
orders_cat_all = _add_part_category(orders)
rfqs_cat_all = _add_part_category(rfqs)

cap_orders = orders_cat_all.groupby(["supplier_key", "part_category"]).size().reset_index(name="order_lines")
cap_rfqs = rfqs_cat_all.groupby(["supplier_key", "part_category"]).size().reset_index(name="rfq_lines")

cap_counts_global = (
    cap_orders.merge(cap_rfqs, on=["supplier_key", "part_category"], how="outer")
    .fillna({"order_lines": 0, "rfq_lines": 0})
)
cap_counts_global["lines_total"] = (cap_counts_global["order_lines"] + cap_counts_global["rfq_lines"]).astype(int)

def top_category_for_supplier(skey: str) -> str:
    """Pick the supplier's dominant category by spend from Orders; fallback to RFQ coverage."""
    if skey:
        oo = orders_cat_all[orders_cat_all["supplier_key"] == skey].copy()
        if not oo.empty and "po_amount" in oo.columns:
            s = oo.groupby("part_category")["po_amount"].sum().sort_values(ascending=False)
            if len(s):
                return str(s.index[0])
        cc = cap_counts_global[cap_counts_global["supplier_key"] == skey].sort_values("lines_total", ascending=False)
        if not cc.empty:
            return str(cc.iloc[0]["part_category"])
    return "(All Categories)"

def recommend_alternatives(
    target_key: str,
    category: str,
    min_lines: int = 2,
    max_n: int = 2,
    require_low_risk: bool = True,
) -> list:
    """
    Return list of supplier_names that:
      - have evidence in the same category (orders or RFQs),
      - are not the target supplier,
      - optionally are low risk (not Quality/Delivery),
      - and are generally strong performers.
    """
    if not target_key:
        return []

    cat = category or "(All Categories)"
    if cat == "(All Categories)":
        # broad suggestions: just best overall low-risk suppliers (excluding target)
        pool = supplier_master.copy()
    else:
        eligible = cap_counts_global[
            (cap_counts_global["part_category"] == cat) & (cap_counts_global["lines_total"] >= int(min_lines))
        ]["supplier_key"].astype(str).unique().tolist()
        pool = supplier_master[supplier_master["supplier_key"].isin(eligible)].copy()

    pool = pool[pool["supplier_key"] != target_key].copy()
    if require_low_risk:
        pool = pool[~pool["risk_flag"].isin(["🔴 Quality Risk", "🟠 Delivery Risk"])].copy()

    # Rank: performance_score first, then price_score (tie-break)
    pool = pool.sort_values(["performance_score", "price_score"], ascending=[False, False])

    return pool["supplier_name"].head(max_n).tolist()

# ─────────────────────────────────────────────
# ANALYTICS HELPERS (KEYED)
# ─────────────────────────────────────────────
def _pick_rfq_line_key(df: pd.DataFrame):
    for c in ["rfq_line_id", "line_id", "part_number", "item_id", "part_description", "rfq_id"]:
        if c in df.columns:
            return c
    return None

def _norm_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(x).lower().strip())).strip()

def rfq_competitiveness(rfqs_df: pd.DataFrame, cat: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["supplier_key", "avg_price_scope", "avg_delta_vs_best", "pct_not_best", "lines", "rfqs"])
    if rfqs_df is None or rfqs_df.empty:
        return empty
    if not {"supplier_key", "quoted_price"}.issubset(rfqs_df.columns):
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

    if lk == "part_description":
        r["_lk"] = r[lk].astype(str).apply(_norm_text)
    else:
        r["_lk"] = r[lk].astype(str).str.strip()

    best = r.groupby("_lk")["quoted_price"].min().rename("best_price")
    r = r.join(best, on="_lk")
    r["delta"] = (r["quoted_price"] - r["best_price"]).clip(lower=0)
    r["is_best"] = (r["delta"] <= 1e-9).astype(int)

    g = r.groupby("supplier_key", dropna=False).agg(
        avg_price_scope=("quoted_price", "mean"),
        avg_delta_vs_best=("delta", "mean"),
        lines=("quoted_price", "size"),
        rfqs=("_lk", "nunique"),
        pct_not_best=("is_best", lambda s: 100 * (1 - (s.sum() / len(s))) if len(s) else 0),
    ).reset_index()

    return g.round({"avg_price_scope": 2, "avg_delta_vs_best": 2, "pct_not_best": 1})

def build_pricing_impact(master: pd.DataFrame, rfqs_df: pd.DataFrame, cat: str) -> pd.DataFrame:
    comp = rfq_competitiveness(rfqs_df, cat)
    out = master.merge(comp, on="supplier_key", how="left")

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
    empty = pd.DataFrame(columns=["supplier_key", "avg_alternatives", "switchability"])
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

    if lk == "part_description":
        r["_lk"] = r[lk].astype(str).apply(_norm_text)
    else:
        r["_lk"] = r[lk].astype(str).str.strip()

    n = r.groupby("_lk")["supplier_key"].nunique().rename("n")
    r = r.join(n, on="_lk")
    r["alts"] = (r["n"] - 1).clip(lower=0)

    g = r.groupby("supplier_key", dropna=False)["alts"].mean().reset_index(name="avg_alternatives")
    g["avg_alternatives"] = g["avg_alternatives"].round(1)
    g["switchability"] = g["avg_alternatives"].apply(lambda a: "HIGH" if a >= 2 else ("MED" if a >= 1 else "LOW"))
    return g

def supplier_key_from_display(name: str) -> str:
    k = normalize_supplier_key(name)
    return MANUAL_KEY_MAP.get(k, k)

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
tab_overview, tab_intel, tab_decision, tab_trends, tab_financial = st.tabs(
    ["📊 Executive Overview", "🔍 Supplier Intel", "⚡ Sourcing Decision", "📈 Performance Trends", "💰 Financial Impact"]
)

# ═══════════════════════════════════════════════════════
# TAB 1 · EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════
with tab_overview:
    cat_exec = st.session_state.get("category_choice", "(All Categories)")
    impact_exec = build_pricing_impact(supplier_master, rfqs, cat_exec)
    sw_exec = switchability(rfqs, cat_exec)
    impact_exec = impact_exec.merge(sw_exec, on="supplier_key", how="left")
    impact_exec["avg_alternatives"] = impact_exec["avg_alternatives"].fillna(0.0)
    impact_exec["switchability"] = impact_exec["switchability"].fillna("LOW")
    impact_exec["defect_cost"] = impact_exec["total_spend"] * (impact_exec["defect_rate"] / 100) * 0.5

    total_overpay = float(impact_exec["estimated_overpay"].sum())
    late_spend = float(supplier_master.loc[supplier_master["on_time_rate"] < 85, "total_spend"].sum())
    defect_cost = float(impact_exec["defect_cost"].sum())
    n_quality_risk = int((supplier_master["risk_flag"] == "🔴 Quality Risk").sum())
    n_delivery_risk = int((supplier_master["risk_flag"] == "🟠 Delivery Risk").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Estimated Overpay",
        _fmt_money(total_overpay),
        help="Estimate of dollars spent above the best available quote for comparable RFQ lines (same part / same line).",
    )
    k2.metric(
        "Spend at Delivery Risk",
        _fmt_money(late_spend),
        help="Total spend with suppliers whose on-time rate is below 85% (missing delivery dates excluded).",
    )
    k3.metric(
        "Est. Quality / Rework Cost",
        _fmt_money(defect_cost),
        help="Defect rate × spend × 0.5 conservative rework multiplier.",
    )
    k4.metric(
        "Suppliers with Active Risk",
        f"{n_quality_risk + n_delivery_risk}",
        help=f"Quality Risk: {n_quality_risk}  |  Delivery Risk: {n_delivery_risk}",
    )

    st.markdown("---")

    st.subheader("Executive Strategic Recommendations")
    st.caption("Recommendations tie quantified metrics to an action — and (when supported by the data) propose specific alternate suppliers with similar category coverage.")

    def rec_card(label: str, title: str, detail: str, why: str):
        badge_colors = {"Concentration": "#6366f1", "Delivery": "#FF7F0E", "Quality": "#D62728", "Pricing": "#0ea5e9"}
        c = badge_colors.get(label, "#6b7280")
        st.markdown(
            f"""
<div class="hoth-card">
  <div style="display:flex; gap:14px; align-items:flex-start;">
    <div class="hoth-badge" style="background:{c};">{label.upper()}</div>
    <div style="flex:1;">
      <div class="title">{title}</div>
      <div class="sub">{detail}</div>
      <div style="height:8px"></div>
      <div style="color:#111827; font-size:0.92rem; line-height:1.35rem;">
        <span style="font-weight:750;">Rationale:</span> {why}
      </div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    recs = []

    # 1) Concentration risk: top spend-share supplier
    top_conc = supplier_master.sort_values("spend_share_pct", ascending=False).head(1)
    if len(top_conc) and float(top_conc.iloc[0]["spend_share_pct"]) >= 20:
        r = top_conc.iloc[0]
        skey = str(r["supplier_key"])
        cat = top_category_for_supplier(skey)
        alts = recommend_alternatives(skey, cat, min_lines=2, max_n=2, require_low_risk=True)
        alt_text = ", ".join(alts) if alts else "No low-risk alternates with matching category evidence found in the provided data."
        recs.append((
            "Concentration",
            f"De-risk concentrated spend tied to {r['supplier_name']}",
            f"{_fmt_pct(float(r['spend_share_pct']))} of spend is concentrated in a single supplier · Primary category: {cat}",
            f"Single-point-of-failure risk at this spend share. Based on category coverage, **recommended alternates** for {cat}: **{alt_text}**.",
        ))

    # 2) Delivery risk: highest-spend delivery-risk supplier
    del_risk = supplier_master[supplier_master["risk_flag"] == "🟠 Delivery Risk"].sort_values("total_spend", ascending=False).head(1)
    if len(del_risk):
        r = del_risk.iloc[0]
        skey = str(r["supplier_key"])
        cat = top_category_for_supplier(skey)
        alts = recommend_alternatives(skey, cat, min_lines=2, max_n=2, require_low_risk=True)
        alt_text = ", ".join(alts) if alts else "No low-risk alternates with matching category evidence found in the provided data."
        recs.append((
            "Delivery",
            f"Mitigate delivery exposure on {r['supplier_name']}",
            f"On-time rate {_fmt_pct(float(r['on_time_rate']))} · Delivery-risk threshold ≤ 85% · Primary category: {cat}",
            f"Initiate a performance improvement plan (OTD root cause, recovery plan, weekly tracking). In parallel, qualify backups in {cat}: **{alt_text}**.",
        ))

    # 3) Quality risk: highest estimated quality cost supplier (if meaningful)
    tmpq = impact_exec.sort_values("defect_cost", ascending=False).head(1)
    if len(tmpq) and float(tmpq.iloc[0]["defect_rate"]) > 0:
        r = tmpq.iloc[0]
        skey = str(r["supplier_key"])
        cat = top_category_for_supplier(skey)
        alts = recommend_alternatives(skey, cat, min_lines=2, max_n=2, require_low_risk=True)
        alt_text = ", ".join(alts) if alts else "No low-risk alternates with matching category evidence found in the provided data."
        recs.append((
            "Quality",
            f"Reduce defect-driven cost from {r['supplier_name']}",
            f"Defect rate {_fmt_pct(float(r['defect_rate']))} · Est. {_fmt_money(float(r['defect_cost']))} rework cost · Primary category: {cat}",
            f"Gate releases with incoming inspection + corrective action (containment, root cause, re-qualification). If issues persist, shift volume in {cat} toward: **{alt_text}**.",
        ))

    # 4) Pricing opportunity: biggest overpay where switchability exists (alternates should also be low-risk if possible)
    cand = impact_exec[impact_exec["avg_alternatives"] >= 1].sort_values("estimated_overpay", ascending=False).head(1)
    if len(cand) and float(cand.iloc[0]["estimated_overpay"]) > 0:
        r = cand.iloc[0]
        skey = str(r["supplier_key"])
        cat = top_category_for_supplier(skey)
        # For pricing, allow "Cost Risk" suppliers as alternates only if delivery/quality ok is preferred;
        # keep require_low_risk True to avoid recommending problematic suppliers.
        alts = recommend_alternatives(skey, cat, min_lines=2, max_n=2, require_low_risk=True)
        alt_text = ", ".join(alts) if alts else "Alternates exist in RFQs, but none meet low-risk criteria in this dataset."
        recs.append((
            "Pricing",
            f"Capture savings by re-sourcing or renegotiating {r['supplier_name']}",
            f"Est. {_fmt_money(float(r['estimated_overpay']))} annual overpay · Avg {_fmt_money_2(float(r['avg_delta_vs_best']))}/unit above best comparable quote · Switchability: {r['switchability']}",
            f"Data shows consistent premium vs best quote on comparable RFQ lines. Negotiate using benchmark evidence and/or move volume in {cat} to: **{alt_text}**.",
        ))

    # Render up to 4 cards (or fewer if data is limited)
    if not recs:
        st.info("Not enough data in the provided files to generate strategic recommendations.")
    else:
        for label, title, detail, why in recs[:4]:
            rec_card(label, title, detail, why)

    st.markdown("---")

    st.subheader("Supplier Positioning Matrix")
    st.caption("Bubble size = total spend. Hover for details. **Ideal suppliers sit top-left**: high performance, low overpay.")

    pos = impact_exec.copy()
    pos["spend_m"] = (pos["total_spend"] / 1e6).round(3)

    matrix = (
        alt.Chart(pos)
        .mark_circle(opacity=0.82, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("avg_delta_vs_best:Q", title="Avg Overpay vs Best Quote ($/unit)  —  lower is better", scale=alt.Scale(zero=True)),
            y=alt.Y("performance_score:Q", title="Performance Score (0–100)  —  higher is better", scale=alt.Scale(domain=[0, 100])),
            size=alt.Size("total_spend:Q", legend=None, scale=alt.Scale(range=[80, 2200])),
            color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk"),
            tooltip=[
                alt.Tooltip("supplier_name:N", title="Supplier"),
                alt.Tooltip("risk_flag:N", title="Risk"),
                alt.Tooltip("performance_score:Q", title="Performance", format=".1f"),
                alt.Tooltip("on_time_rate:Q", title="On-Time Rate (%)", format=".1f"),
                alt.Tooltip("defect_rate:Q", title="Defect Rate (%)", format=".1f"),
                alt.Tooltip("avg_delta_vs_best:Q", title="Overpay ($/unit)", format=".2f"),
                alt.Tooltip("switchability:N", title="Switchability"),
                alt.Tooltip("spend_m:Q", title="Spend ($M)", format=".3f"),
                alt.Tooltip("spend_share_pct:Q", title="Spend Share (%)", format=".1f"),
            ],
        )
        .properties(height=400)
    )
    ref_line = alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(color="#d1d5db", strokeDash=[5, 5], strokeWidth=1.5).encode(y="y:Q")
    st.altair_chart(matrix + ref_line, use_container_width=True)

    st.markdown("---")
    col_s, _ = st.columns([2, 2])
    search_q = col_s.text_input("Search suppliers", placeholder="Type supplier name...", key="search_query")

    filtered = supplier_master.copy()
    if search_q:
        filtered = filtered[filtered["supplier_name"].str.lower().str.contains(search_q.lower(), na=False)]

    cols_tbl = ["supplier_name", "orders", "total_spend", "on_time_rate", "defect_rate", "avg_price", "performance_score", "risk_flag"]
    show_table(with_rank(format_for_display(filtered.sort_values("performance_score", ascending=False), cols_tbl)), TOP_N)

# ═══════════════════════════════════════════════════════
# TAB 2 · SUPPLIER INTEL
# ═══════════════════════════════════════════════════════
with tab_intel:
    st.subheader("Supplier Intelligence Card")
    st.caption("Look up any supplier before placing an order. Surfaces performance data, internal knowledge, pricing vs market, and a recommendation.")

    selected = st.selectbox("Select a supplier:", ["— Choose a supplier —"] + all_suppliers, key="intel_supplier")

    if selected and selected != "— Choose a supplier —":
        skey = supplier_key_from_display(selected)
        row_df = supplier_master[supplier_master["supplier_key"] == skey]
        if row_df.empty:
            st.warning("No data found for this supplier.")
        else:
            row = row_df.iloc[0]
            note = supplier_notes.get(skey, {})
            risk = row["risk_flag"]

            if "Quality" in risk:
                st.error(f"STOP — {risk}: Review quality history before sending this RFQ.")
            elif "Delivery" in risk:
                st.warning(f"CAUTION — {risk}: Add schedule buffer to this order.")
            elif "Cost" in risk:
                st.info(f"NOTE — {risk}: Not price-competitive. Solicit at least 2 alternatives.")
            else:
                st.success(f"CLEAR — {risk}: Consistently strong performer.")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("On-Time Rate", _fmt_pct(row["on_time_rate"]))
            m2.metric("Defect Rate", _fmt_pct(row["defect_rate"]))
            m3.metric("Performance Score", _fmt_score(row["performance_score"]) + " / 100")
            m4.metric("Total Spend", _fmt_money(row["total_spend"]))
            m5.metric("Avg Quoted Price", _fmt_money_2(row["avg_price"]) + " /unit")

            st.markdown("---")
            left, right = st.columns([1.15, 1])

            with left:
                st.markdown("**Internal Notes (Tribal Knowledge)**")
                if note:
                    render_full_note(note)
                else:
                    st.info("No internal notes on file for this supplier.")

                st.markdown("---")
                st.markdown("**Recommendation**")
                rec = []
                if row["on_time_rate"] < 85:
                    rec.append(f"Add schedule buffer — on-time rate is {_fmt_pct(row['on_time_rate'])}.")
                if row["defect_rate"] > 5:
                    rec.append(f"Require incoming inspection — defect rate is {_fmt_pct(row['defect_rate'])}.")
                if row["price_score"] <= 40:
                    rec.append(f"Solicit competing quotes — price score is {_fmt_score(row['price_score'])}/100.")
                if rec:
                    for r_ in rec:
                        st.warning(r_)
                else:
                    st.success(f"No concerns flagged. {selected} is a reliable sourcing choice.")

                st.markdown("---")
                st.markdown("**Pricing vs. Market (apples-to-apples lines)**")
                sup_rfqs = rfqs[rfqs["supplier_key"] == skey].copy()
                sup_rfqs["quoted_price"] = pd.to_numeric(sup_rfqs["quoted_price"], errors="coerce")
                sup_rfqs = sup_rfqs[sup_rfqs["quoted_price"].notna() & sup_rfqs["quoted_price"].gt(0)]
                lk = _pick_rfq_line_key(rfqs)
                if not sup_rfqs.empty and lk is not None:
                    mkt = rfqs.copy()
                    mkt["quoted_price"] = pd.to_numeric(mkt["quoted_price"], errors="coerce")
                    mkt = mkt[mkt["quoted_price"].notna() & mkt["quoted_price"].gt(0)]

                    if lk == "part_description":
                        sup_rfqs["_lk"] = sup_rfqs[lk].astype(str).apply(_norm_text)
                        mkt["_lk"] = mkt[lk].astype(str).apply(_norm_text)
                    else:
                        sup_rfqs["_lk"] = sup_rfqs[lk].astype(str).str.strip()
                        mkt["_lk"] = mkt[lk].astype(str).str.strip()

                    best_mkt = mkt.groupby("_lk")["quoted_price"].min().rename("best")
                    comp_df = sup_rfqs.join(best_mkt, on="_lk")
                    comp_df = comp_df[comp_df["best"].notna()]
                    if not comp_df.empty:
                        premium = (comp_df["quoted_price"] - comp_df["best"]).mean()
                        if premium > 0:
                            st.warning(f"Avg {_fmt_money_2(premium)}/unit **above** best market quote across comparable lines.")
                        else:
                            st.success(f"Avg {_fmt_money_2(abs(premium))}/unit **at or below** best market quote — competitive.")
                    else:
                        st.info("Not enough overlapping comparable RFQ lines to compute premium.")
                else:
                    st.info("No RFQ history available to compute market premium.")

            with right:
                st.markdown("**Recent Order History**")
                sup_orders = orders[orders["supplier_key"] == skey].copy()
                if not sup_orders.empty:
                    sov = sup_orders[sup_orders["actual_delivery_date"].notna() & sup_orders["promised_date"].notna()].copy()
                    sov["on_time"] = sov["actual_delivery_date"] <= sov["promised_date"]
                    sov["days_diff"] = (sov["actual_delivery_date"] - sov["promised_date"]).dt.days
                    sov["Status"] = sov["on_time"].map({True: "✅ On Time", False: "❌ Late"})
                    sov["Variance"] = sov["days_diff"].apply(
                        lambda x: f"-{abs(int(x))}d early" if x < 0 else (f"+{int(x)}d late" if x > 0 else "On schedule")
                    )
                    disp = [c for c in ["order_id", "part_description", "order_date", "Status", "Variance", "po_amount"] if c in sov.columns or c in ["Status", "Variance"]]
                    show_table(sov[disp].sort_values("order_date", ascending=False), max_rows=8)
                else:
                    st.info("No order history found.")

                st.markdown("**Quality Inspections**")
                sup_q = q[q["supplier_key"] == skey].copy() if "supplier_key" in q.columns else pd.DataFrame()
                if not sup_q.empty:
                    q_disp = [c for c in ["inspection_date", "parts_inspected", "parts_rejected", "rejection_reason", "rework_required"] if c in sup_q.columns]
                    sup_q_show = sup_q[q_disp].copy()
                    if {"parts_inspected", "parts_rejected"}.issubset(sup_q_show.columns):
                        sup_q_show["Defect %"] = (sup_q_show["parts_rejected"] / sup_q_show["parts_inspected"] * 100).round(1).astype(str) + "%"
                    show_table(sup_q_show.sort_values("inspection_date", ascending=False), max_rows=8)
                else:
                    st.info("No inspection records found.")
    else:
        st.info("Select a supplier above to view their full intelligence card.")

# ═══════════════════════════════════════════════════════
# TAB 3 · SOURCING DECISION
# ═══════════════════════════════════════════════════════
with tab_decision:
    st.subheader("Real-Time Sourcing Decision Support")
    st.caption("Set your requirements and get a ranked shortlist of qualified suppliers for a specific part category.")

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

    st.markdown("---")

    col_cat1, col_cat2, col_cat3 = st.columns([2, 1, 1])
    with col_cat1:
        all_cats = sorted(set(orders_cat_all["part_category"].unique()) | set(rfqs_cat_all["part_category"].unique()))
        st.selectbox("Part Category", ["(All Categories)"] + all_cats, key="category_choice")
    with col_cat2:
        st.radio("Evidence from", ["RFQs only", "Orders only", "Orders + RFQs"], key="capability_source")
    with col_cat3:
        st.slider("Min qualifying lines", 1, 10, step=1, key="min_lines")
        st.checkbox("Show coverage detail", key="show_coverage")

    chosen_cat = st.session_state["category_choice"]

    cap_parts = []
    if st.session_state["capability_source"] in ("Orders only", "Orders + RFQs"):
        cap_parts.append(orders_cat_all.groupby(["supplier_key", "part_category"]).size().reset_index(name="lines"))
    if st.session_state["capability_source"] in ("RFQs only", "Orders + RFQs"):
        cap_parts.append(rfqs_cat_all.groupby(["supplier_key", "part_category"]).size().reset_index(name="lines"))
    cap_counts = (
        pd.concat(cap_parts, ignore_index=True).groupby(["supplier_key", "part_category"], as_index=False)["lines"].sum()
        if cap_parts
        else pd.DataFrame()
    )

    if st.session_state["show_coverage"] and not cap_counts.empty:
        cov = cap_counts if chosen_cat == "(All Categories)" else cap_counts[cap_counts["part_category"] == chosen_cat]
        cov = cov.copy()
        cov["supplier_name"] = cov["supplier_key"].apply(key_to_display)
        show_table(with_rank(format_for_display(cov.sort_values("lines", ascending=False), ["supplier_name", "part_category", "lines"])), max_rows=20)

    if chosen_cat == "(All Categories)" or cap_counts.empty:
        eligible_keys = set(supplier_master["supplier_key"].astype(str))
    else:
        eligible_keys = set(
            cap_counts[(cap_counts["part_category"] == chosen_cat) & (cap_counts["lines"] >= st.session_state["min_lines"])][
                "supplier_key"
            ].astype(str)
        )

    decision_kpi = supplier_master[supplier_master["supplier_key"].isin(eligible_keys)].copy()

    decision_kpi["fit"] = (decision_kpi["on_time_rate"] >= st.session_state["req_on_time"]) & (decision_kpi["defect_rate"] <= st.session_state["max_defects"])
    decision_kpi["fit_status"] = decision_kpi["fit"].map({True: "✅ Meets criteria", False: "❌ Below threshold"})
    decision_kpi["notes_hint"] = decision_kpi["supplier_key"].apply(lambda k: note_snippet(supplier_notes, k))

    n_fit = int(decision_kpi["fit"].sum())
    n_total = len(decision_kpi)

    if n_fit == 0:
        st.warning(f"No suppliers meet the current thresholds (≥{st.session_state['req_on_time']}% on-time, ≤{st.session_state['max_defects']}% defects). Consider relaxing your criteria.")
    else:
        st.success(f"**{n_fit} of {n_total} suppliers** meet your quality and delivery thresholds — ranked below.")

    ranked = decision_kpi.sort_values(["fit", "performance_score"], ascending=[False, False])
    show_table(
        with_rank(format_for_display(ranked, ["supplier_name", "fit_status", "performance_score", "risk_flag", "on_time_rate", "defect_rate", "avg_price", "total_spend", "notes_hint"])),
        TOP_N,
    )

    st.markdown("---")
    st.subheader("Consolidation Opportunities")
    st.caption(f"Pricing analysis within scope: **{chosen_cat}**")
    impact_dec = build_pricing_impact(supplier_master, rfqs, chosen_cat)
    st.metric("Est. Annual Savings Available", _fmt_money(float(impact_dec["estimated_overpay"].sum())), help="Based on avg overpay vs best comparable quote × estimated units")
    show_table(with_rank(format_for_display(impact_dec.sort_values("estimated_overpay", ascending=False),
        ["supplier_name", "total_spend", "avg_price", "avg_delta_vs_best", "estimated_overpay", "risk_flag", "rfqs", "pct_not_best"])), TOP_N)

# ═══════════════════════════════════════════════════════
# TAB 4 · PERFORMANCE TRENDS
# ═══════════════════════════════════════════════════════
with tab_trends:
    st.subheader("Performance Trends")
    st.caption("Fast executive view: choose Top 6, All suppliers, or Single supplier. Increase window for longer view.")

    if "order_date" not in orders.columns:
        st.warning("No `order_date` column found — trends unavailable.")
    else:
        top6_keys = supplier_master.sort_values("total_spend", ascending=False)["supplier_key"].head(6).tolist()

        cL, cM, cR = st.columns([2, 1, 1])
        with cL:
            st.selectbox("Suppliers to display:", ["Top 6 by spend", "All suppliers", "Single supplier"], key="trend_supplier_mode")
        with cM:
            st.selectbox("Primary chart", ["On-Time Trend", "Late Orders Trend", "Spend Trend"], key="trend_focus")
        with cR:
            st.selectbox("Window (months)", [6, 12, 18, 24, 36, 48], key="trend_months")

        if st.session_state["trend_supplier_mode"] == "Top 6 by spend":
            tf_keys = top6_keys
        elif st.session_state["trend_supplier_mode"] == "All suppliers":
            tf_keys = supplier_master["supplier_key"].tolist()
        else:
            st.selectbox("Choose supplier:", options=all_suppliers, key="trend_single_supplier")
            tf_keys = [supplier_key_from_display(st.session_state["trend_single_supplier"])] if st.session_state["trend_single_supplier"] else top6_keys

        months = int(st.session_state["trend_months"])
        start_cut = (pd.Timestamp.today().to_period("M").to_timestamp()) - pd.DateOffset(months=months - 1)

        o = orders[orders["supplier_key"].isin(tf_keys)].copy()
        o["month"] = o["order_date"].dt.to_period("M").dt.to_timestamp()
        o = o[o["month"] >= start_cut].copy()
        o["supplier_name"] = o["supplier_key"].apply(key_to_display)

        ot = ot_valid[ot_valid["supplier_key"].isin(tf_keys)].copy()
        ot["month"] = ot["order_date"].dt.to_period("M").dt.to_timestamp()
        ot = ot[ot["month"] >= start_cut].copy()
        ot["supplier_name"] = ot["supplier_key"].apply(key_to_display)

        focus = st.session_state["trend_focus"]

        if focus == "On-Time Trend":
            st.markdown("#### On-Time Rate Trend (Monthly)")
            if ot.empty:
                st.info("No on-time history in the selected window.")
            else:
                heat = ot.groupby(["month", "supplier_name"])["on_time"].agg(["mean", "count"]).reset_index()
                heat.columns = ["month", "supplier_name", "on_time_rate", "n_orders"]
                heat["otr_pct"] = (heat["on_time_rate"] * 100).round(1)

                line = (
                    alt.Chart(heat)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("month:T", title="Month", axis=alt.Axis(format="%b %y")),
                        y=alt.Y("otr_pct:Q", title="On-Time %", scale=alt.Scale(domain=[50, 100])),
                        color=alt.Color("supplier_name:N", title="Supplier"),
                        tooltip=[
                            alt.Tooltip("supplier_name:N", title="Supplier"),
                            alt.Tooltip("month:T", title="Month", format="%B %Y"),
                            alt.Tooltip("otr_pct:Q", title="On-Time %", format=".1f"),
                            alt.Tooltip("n_orders:Q", title="# Orders"),
                        ],
                    )
                    .properties(height=340)
                )
                tgt = alt.Chart(pd.DataFrame({"y": [85]})).mark_rule(color="#d1d5db", strokeDash=[6, 4], strokeWidth=1.6).encode(y="y:Q")
                st.altair_chart(line + tgt, use_container_width=True)
                st.caption("Dashed line = 85% delivery-risk threshold.")

        elif focus == "Late Orders Trend":
            st.markdown("#### Late Orders Trend (Monthly Count)")
            if ot.empty:
                st.info("No on-time history in the selected window.")
            else:
                lot = ot.copy()
                lot["days_late"] = (lot["actual_delivery_date"] - lot["promised_date"]).dt.days
                lot = lot[lot["days_late"] > 0].copy()
                if lot.empty:
                    st.success("No late deliveries recorded for the selected suppliers in this window.")
                else:
                    lc = lot.groupby(["month", "supplier_name"])["order_id"].nunique().reset_index(name="late_orders")
                    bar = (
                        alt.Chart(lc)
                        .mark_bar()
                        .encode(
                            x=alt.X("month:T", title="Month", axis=alt.Axis(format="%b %y")),
                            y=alt.Y("late_orders:Q", title="Late Orders", scale=alt.Scale(zero=True)),
                            color=alt.Color("supplier_name:N", title="Supplier"),
                            tooltip=[
                                alt.Tooltip("supplier_name:N", title="Supplier"),
                                alt.Tooltip("month:T", title="Month", format="%B %Y"),
                                alt.Tooltip("late_orders:Q", title="Late Orders"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(bar, use_container_width=True)

        else:
            st.markdown("#### Spend Trend (Monthly Spend $K)")
            if o.empty:
                st.info("No spend history in the selected window.")
            else:
                ms = o.groupby(["month", "supplier_name"])["po_amount"].sum().reset_index()
                ms["spend_k"] = (ms["po_amount"] / 1000).round(1)

                line2 = (
                    alt.Chart(ms)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("month:T", title="Month", axis=alt.Axis(format="%b %y")),
                        y=alt.Y("spend_k:Q", title="Spend ($K)", scale=alt.Scale(zero=True)),
                        color=alt.Color("supplier_name:N", title="Supplier"),
                        tooltip=[
                            alt.Tooltip("supplier_name:N", title="Supplier"),
                            alt.Tooltip("month:T", title="Month", format="%B %Y"),
                            alt.Tooltip("spend_k:Q", title="Spend ($K)", format=".1f"),
                        ],
                    )
                    .properties(height=340)
                )
                st.altair_chart(line2, use_container_width=True)

        st.markdown("---")
        with st.expander("Debug — Apex coverage check"):
            apex_key = "apex manufacturing"
            st.write("Apex supplier_key:", apex_key)
            st.write("Orders rows for Apex:", int((orders["supplier_key"] == apex_key).sum()))
            st.write("RFQ rows for Apex:", int((rfqs["supplier_key"] == apex_key).sum()))
            st.write("Apex avg_price (supplier_master):", float(supplier_master.loc[supplier_master["supplier_key"] == apex_key, "avg_price"].iloc[0]) if (supplier_master["supplier_key"] == apex_key).any() else None)
            st.write("Apex notes present:", apex_key in supplier_notes)

# ═══════════════════════════════════════════════════════
# TAB 5 · FINANCIAL IMPACT
# ═══════════════════════════════════════════════════════
with tab_financial:
    st.subheader("Financial Impact Analysis")
    st.caption("Quantifies cost of supplier underperformance across overpay, quality/rework cost, and delivery-risk exposure.")

    cat_fin = st.session_state.get("category_choice", "(All Categories)")
    impact_df = build_pricing_impact(supplier_master, rfqs, cat_fin)
    impact_df["defect_cost"] = (impact_df["total_spend"] * (impact_df["defect_rate"] / 100) * 0.5).round(0)
    impact_df["total_risk"] = impact_df["estimated_overpay"] + impact_df["defect_cost"]

    late_spend_fin = float(impact_df.loc[impact_df["on_time_rate"] < 85, "total_spend"].sum())
    total_overpay_fin = float(impact_df["estimated_overpay"].sum())
    total_quality_fin = float(impact_df["defect_cost"].sum())
    total_risk_fin = total_overpay_fin + total_quality_fin

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Quantified Risk", _fmt_money(total_risk_fin), help="Estimated overpay + est. quality cost combined")
    f2.metric("Estimated Overpay", _fmt_money(total_overpay_fin), help="Avg overpay vs best comparable quote × estimated units")
    f3.metric("Est. Quality / Rework Cost", _fmt_money(total_quality_fin), help="Defect rate × total spend × 0.5 rework cost factor")
    f4.metric("Spend at Delivery Risk", _fmt_money(late_spend_fin), help="Total spend with suppliers below 85% on-time rate")

    st.markdown("---")
    st.markdown("#### Financial Exposure by Supplier — Overpay + Quality Cost")
    risk_plot = impact_df[impact_df["total_risk"] > 0].sort_values("total_risk", ascending=False).head(12).copy()
    if not risk_plot.empty:
        melt = pd.melt(risk_plot[["supplier_name", "estimated_overpay", "defect_cost"]], id_vars="supplier_name", var_name="type", value_name="cost")
        melt["type"] = melt["type"].map({"estimated_overpay": "Estimated Overpay", "defect_cost": "Quality / Rework Cost"})
        stacked = (
            alt.Chart(melt)
            .mark_bar()
            .encode(
                x=alt.X("cost:Q", title="Estimated Cost ($)", axis=alt.Axis(format="$,.0f")),
                y=alt.Y("supplier_name:N", sort="-x", title=None),
                color=alt.Color("type:N", title="Cost Type", legend=alt.Legend(orient="bottom")),
                tooltip=[
                    alt.Tooltip("supplier_name:N", title="Supplier"),
                    alt.Tooltip("type:N", title="Cost Type"),
                    alt.Tooltip("cost:Q", title="Est. Cost ($)", format="$,.0f"),
                ],
            )
            .properties(height=max(240, len(risk_plot) * 36))
        )
        st.altair_chart(stacked, use_container_width=True)

    st.markdown("---")
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
