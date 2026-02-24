"""
Hoth Industries — Supplier Intelligence Platform (Executive Polished)
- Fixes supplier notes parsing + displays FULL notes (not just a snippet)
- Improves Priority Actions cards (badge width + richer "why" context)
- Simplifies Performance Trends so insights are obvious in ~2 seconds
- Adds Apex notes + matching robustness
- Performance Trends: dropdown mode (Top 6 / All / Single) + larger window options
- Executive Overview: simpler terminology (Estimated Overpay instead of Pricing Leakage)
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
  min-width: 140px;         /* wider so text fits */
  text-align:center;
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.02em;
  white-space: normal;      /* allow wrapping */
  line-height: 1.05rem;
  color: #fff;
}
.hoth-note-box{
  border: 1px solid #e5e7eb;
  background: #fafafa;
  border-radius: 10px;
  padding: 12px 12px;
}
.hoth-note-meta{
  color:#6b7280;
  font-size:0.85rem;
  margin-top: 2px;
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

    # Trends controls (updated to dropdown mode)
    "trend_mode": "Top 6 by spend",
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

# NOTE: label simplified per your ask
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
    "estimated_overpay": "Estimated Overpay ($)",  # renamed from Pricing Leakage
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


# Manual key normalization mapping: normalized -> normalized
MANUAL_KEY_MAP = {
    "apex mfg": "apex manufacturing",
    "apex manufacturing inc": "apex manufacturing",
    "apex mfg inc": "apex manufacturing",
    "apex manufacturing": "apex manufacturing",
}


def apply_entity_resolution(df: pd.DataFrame, col: str, manual_map: dict = None) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    out["_key"] = out[col].apply(normalize_supplier_key)

    if manual_map:
        # manual_map maps normalized keys -> normalized keys
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
# ANALYTICS
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

    # Apples-to-apples comparison key
    if lk == "part_description":
        r["_lk"] = r[lk].astype(str).apply(_norm_text)
    else:
        r["_lk"] = r[lk].astype(str).str.strip()

    best = r.groupby("_lk")["quoted_price"].min().rename("best_price")
    r = r.join(best, on="_lk")
    r["delta"] = (r["quoted_price"] - r["best_price"]).clip(lower=0)
    r["is_best"] = (r["delta"] <= 1e-9).astype(int)

    g = r.groupby("supplier_name", dropna=False).agg(
        avg_price_scope=("quoted_price", "mean"),
        avg_delta_vs_best=("delta", "mean"),
        lines=("quoted_price", "size"),
        rfqs=("_lk", "nunique"),
        pct_not_best=("is_best", lambda s: 100 * (1 - (s.sum() / len(s))) if len(s) else 0),
    ).reset_index()

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

    # simpler name: Estimated Overpay (same math)
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

    r["_lk"] = r[lk].astype(str).str.strip()
    n = r.groupby("_lk")["supplier_name"].nunique().rename("n")
    r = r.join(n, on="_lk")
    r["alts"] = (r["n"] - 1).clip(lower=0)

    g = r.groupby("supplier_name", dropna=False)["alts"].mean().reset_index(name="avg_alternatives")
    g["avg_alternatives"] = g["avg_alternatives"].round(1)
    g["switchability"] = g["avg_alternatives"].apply(lambda a: "HIGH" if a >= 2 else ("MED" if a >= 1 else "LOW"))
    return g


# ─────────────────────────────────────────────
# SUPPLIER NOTES (ROBUST PARSER + FULL DISPLAY)
# ─────────────────────────────────────────────
def _note_key(name: str) -> str:
    """
    Centralized note key logic (FIX for Apex missing notes):
    normalize -> apply MANUAL_KEY_MAP.
    """
    k = normalize_supplier_key(name)
    return MANUAL_KEY_MAP.get(k, k)


def parse_supplier_notes_robust(text: str) -> dict:
    """
    Robustly parses notes in either of these common formats:
      A) Old:  SUPPLIER - Descriptor\n(bullets...) separated by "\n====\n"
      B) New:  ======\nSUPPLIER - Descriptor\n======\n(content...) blocks
    Returns:
      { normalized_supplier_key: {"header_supplier":..., "descriptor":..., "body":..., "raw_supplier_header":...} }
    """
    notes = {}
    if not text or not str(text).strip():
        return notes

    lines = [ln.rstrip("\n") for ln in str(text).splitlines()]

    def is_sep(ln: str) -> bool:
        s = (ln or "").strip()
        return len(s) >= 10 and all(ch == "=" for ch in s)

    i = 0
    while i < len(lines):
        ln = lines[i].strip()

        if is_sep(ln):
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

            if k < len(lines) and is_sep(lines[k].strip()):
                body_start = k + 1
            else:
                body_start = j + 1

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

            supplier_raw = header
            descriptor = ""
            mm = re.match(r"^(.+?)\s*-\s*(.+)$", header)
            if mm:
                supplier_raw = mm.group(1).strip()
                descriptor = mm.group(2).strip()

            # FIX: apply manual map after normalization
            supplier_key = _note_key(supplier_raw)

            if supplier_key:
                body = "\n".join(body_lines).strip()
                if "end of notes" not in header.lower():
                    notes[supplier_key] = {
                        "header_supplier": supplier_raw,
                        "descriptor": descriptor,
                        "body": body,
                        "raw_supplier_header": header,
                    }

            i = m
            continue

        i += 1

    if not notes:
        blocks = re.split(r"\n=+\n", str(text))
        for b in blocks:
            b = b.strip()
            if not b:
                continue
            first = b.splitlines()[0].strip()
            mm = re.match(r"^(.+?)\s*-\s*(.+)$", first)
            if not mm:
                continue
            supplier_raw = mm.group(1).strip()
            descriptor = mm.group(2).strip()
            supplier_key = _note_key(supplier_raw)
            body = "\n".join(b.splitlines()[1:]).strip()
            notes[supplier_key] = {
                "header_supplier": supplier_raw,
                "descriptor": descriptor,
                "body": body,
                "raw_supplier_header": first,
            }

    return notes


def note_snippet(notes: dict, name: str) -> str:
    n = notes.get(_note_key(name), {})
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


def get_full_note(notes: dict, name: str) -> dict:
    return notes.get(_note_key(name), {})


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

        head_match = re.match(
            r"^([A-Za-z ]+from\s.+?\(\d{1,2}/\d{1,2}/\d{4}\):)\s*(.*)$",
            p,
            re.IGNORECASE | re.DOTALL,
        )
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
def read_text_flexible(paths):
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            pass
    return ""


def safe_dt(df, col):
    if df is not None and col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def process_raw(orders, quality, rfqs, notes_text):
    orders = apply_entity_resolution(orders, "supplier_name", MANUAL_KEY_MAP)
    rfqs = apply_entity_resolution(rfqs, "supplier_name", MANUAL_KEY_MAP)
    quality = apply_entity_resolution(quality, "supplier_name", MANUAL_KEY_MAP)

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
        ("supplier_orders.csv", "quality_inspections.csv", "rfq_responses.csv", "/mnt/data/supplier_notes.txt"),
        ("Copy_of_supplier_orders.csv", "Copy_of_quality_inspections.csv", "Copy_of_rfq_responses.csv", "/mnt/data/supplier_notes.txt"),
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


local = try_load_local()
if local:
    orders, quality, rfqs, supplier_notes_text = process_raw(*local)
    supplier_notes = parse_supplier_notes_robust(supplier_notes_text)
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
        q = pd.read_csv(io.BytesIO(uf_q.read()))
        r = pd.read_csv(io.BytesIO(uf_r.read()))
        n = uf_n.read().decode("utf-8", "ignore") if uf_n else ""
        orders, quality, rfqs, supplier_notes_text = process_raw(o, q, r, n)
        supplier_notes = parse_supplier_notes_robust(supplier_notes_text)
    else:
        st.info("Upload all three CSV files to continue.")
        st.stop()

# ─────────────────────────────────────────────
# BUILD SUPPLIER MASTER
# ─────────────────────────────────────────────
required = {"order_id", "supplier_name", "po_amount", "promised_date", "actual_delivery_date"}
missing = required - set(orders.columns)
if missing:
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

ap["supplier_name"] = ap["supplier_name"].astype(str).str.strip()
ap = apply_entity_resolution(ap, "supplier_name", MANUAL_KEY_MAP)

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
all_suppliers = sorted(supplier_master["supplier_name"].dropna().unique().tolist())

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
    impact_exec = impact_exec.merge(sw_exec, on="supplier_name", how="left")
    impact_exec["avg_alternatives"] = impact_exec["avg_alternatives"].fillna(0.0)
    impact_exec["switchability"] = impact_exec["switchability"].fillna("LOW")
    impact_exec["defect_cost"] = impact_exec["total_spend"] * (impact_exec["defect_rate"] / 100) * 0.5

    # simpler term
    total_overpay = float(impact_exec["estimated_overpay"].sum())
    late_spend = float(supplier_master.loc[supplier_master["on_time_rate"] < 85, "total_spend"].sum())
    defect_cost = float(impact_exec["defect_cost"].sum())
    n_quality_risk = int((supplier_master["risk_flag"] == "🔴 Quality Risk").sum())
    n_delivery_risk = int((supplier_master["risk_flag"] == "🟠 Delivery Risk").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Estimated Overpay",
        _fmt_money(total_overpay),
        help="Estimate of dollars spent above the best available quote on comparable RFQ lines (same part / same RFQ line).",
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

    st.subheader("Priority Actions")
    st.caption("Clear, explainable recommendations tied to quantified impact (overpay, quality cost, and concentration risk).")

    def action_card(label: str, title: str, detail: str, why: str):
        badge_colors = {"Pricing": "#FF7F0E", "Quality": "#D62728", "Concentration": "#6366f1"}
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
        <span style="font-weight:750;">Why:</span> {why}
      </div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    actions = []

    cand = impact_exec[impact_exec["avg_alternatives"] >= 1].sort_values("estimated_overpay", ascending=False)
    if len(cand):
        r = cand.iloc[0]
        actions.append(
            (
                "Pricing",
                f"Renegotiate or re-source {r['supplier_name']}",
                f"Est. {_fmt_money(float(r['estimated_overpay']))} annual overpay · {r['switchability']} switchability · {int(r['avg_alternatives'])} qualified alternative(s) on file",
                f"{r['supplier_name']} is priced **{_fmt_money_2(float(r['avg_delta_vs_best']))}/unit above the best** comparable quote on average (same RFQ line). With {int(r['avg_alternatives'])} alternative(s), savings are achievable without stalling schedules.",
            )
        )

    tmp = impact_exec.sort_values("defect_cost", ascending=False)
    if len(tmp) and float(tmp.iloc[0]["defect_rate"]) > 0:
        r = tmp.iloc[0]
        actions.append(
            (
                "Quality",
                f"Place {r['supplier_name']} on corrective action",
                f"Defect rate {_fmt_pct(float(r['defect_rate']))} · Est. {_fmt_money(float(r['defect_cost']))} rework cost",
                f"Quality-driven waste is material here: defect rate **{_fmt_pct(float(r['defect_rate']))}** implies ~{_fmt_money(float(r['defect_cost']))} in estimated rework cost. Require containment actions (inspection plan, root cause, and re-qualification gates).",
            )
        )

    tmp2 = supplier_master.sort_values("spend_share_pct", ascending=False)
    if len(tmp2) and float(tmp2.iloc[0]["spend_share_pct"]) >= 20:
        r = tmp2.iloc[0]
        actions.append(
            (
                "Concentration",
                f"Qualify backup supplier(s) for {r['supplier_name']}",
                f"{_fmt_pct(float(r['spend_share_pct']))} of total spend concentrated in one supplier — single-point-of-failure risk",
                f"When a single supplier holds **{_fmt_pct(float(r['spend_share_pct']))}** of spend, any slip (capacity, disruption, quality) becomes a business-wide outage. Qualify at least one backup for critical categories tied to this supplier.",
            )
        )

    for label, title, detail, why in actions[:3]:
        action_card(label, title, detail, why)

    st.markdown("---")

    st.subheader("Supplier Positioning Matrix")
    st.caption(
        "Bubble size = total spend. Hover for details. **Ideal suppliers sit top-left**: high performance score, low pricing premium."
    )

    pos = impact_exec.merge(supplier_master[["supplier_name", "spend_share_pct"]], on="supplier_name", how="left")
    pos["spend_m"] = (pos["total_spend"] / 1e6).round(3)

    matrix = (
        alt.Chart(pos)
        .mark_circle(opacity=0.82, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X(
                "avg_delta_vs_best:Q",
                title="Avg Pricing Premium vs Best Quote ($/unit)  —  lower is better",
                scale=alt.Scale(zero=True),
            ),
            y=alt.Y(
                "performance_score:Q",
                title="Performance Score (0–100)  —  higher is better",
                scale=alt.Scale(domain=[0, 100]),
            ),
            size=alt.Size("total_spend:Q", legend=None, scale=alt.Scale(range=[80, 2200])),
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
        "Surfaces performance data, internal team knowledge, pricing vs. market, and a clear recommendation — all in one view."
    )

    selected = st.selectbox("Select a supplier:", ["— Choose a supplier —"] + all_suppliers, key="intel_supplier")

    if selected and selected != "— Choose a supplier —":
        row_df = supplier_master[supplier_master["supplier_name"] == selected]
        if row_df.empty:
            st.warning("No data found for this supplier.")
        else:
            row = row_df.iloc[0]
            note = get_full_note(supplier_notes, selected)
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
                    rec.append(
                        f"Add **{max(1, round((90 - row['on_time_rate']) / 10))} week(s)** of schedule buffer — on-time rate is {_fmt_pct(row['on_time_rate'])}."
                    )
                if row["defect_rate"] > 5:
                    rec.append(f"Require **100% incoming inspection** — defect rate is {_fmt_pct(row['defect_rate'])}.")
                if row["price_score"] <= 40:
                    rec.append(f"**Solicit competing quotes** — price score is {_fmt_score(row['price_score'])}/100.")
                if rec:
                    for r_ in rec:
                        st.warning(r_)
                else:
                    st.success(f"No concerns flagged. {selected} is a reliable sourcing choice.")

                st.markdown("---")
                st.markdown("**Pricing vs. Market**")
                sup_rfqs_raw = rfqs[rfqs["supplier_name"] == selected].copy()
                sup_rfqs_raw["quoted_price"] = pd.to_numeric(sup_rfqs_raw["quoted_price"], errors="coerce")
                sup_rfqs_raw = sup_rfqs_raw[sup_rfqs_raw["quoted_price"].notna() & sup_rfqs_raw["quoted_price"].gt(0)]

                lk = _pick_rfq_line_key(rfqs) if rfqs is not None and not rfqs.empty else None
                if not sup_rfqs_raw.empty and lk is not None and lk in rfqs.columns:
                    mkt = rfqs.copy()
                    mkt["quoted_price"] = pd.to_numeric(mkt["quoted_price"], errors="coerce")
                    mkt = mkt[mkt["quoted_price"].notna() & mkt["quoted_price"].gt(0)]

                    if lk == "part_description":
                        sup_rfqs_raw["_lk"] = sup_rfqs_raw[lk].astype(str).apply(_norm_text)
                        mkt["_lk"] = mkt[lk].astype(str).apply(_norm_text)
                    else:
                        sup_rfqs_raw["_lk"] = sup_rfqs_raw[lk].astype(str).str.strip()
                        mkt["_lk"] = mkt[lk].astype(str).str.strip()

                    best_mkt = mkt.groupby("_lk")["quoted_price"].min().rename("best")
                    comp_df = sup_rfqs_raw.join(best_mkt, on="_lk")
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
                    show_table(sov[disp].sort_values("order_date", ascending=False) if "order_date" in disp else sov[disp], max_rows=8)
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

        decision_kpi = sc_spend.merge(sc_otr, on="supplier_name", how="left").merge(sc_def, on="supplier_name", how="left").merge(
            sc_ap, on="supplier_name", how="left"
        ).fillna({"on_time_rate": 0.0, "defect_rate": 0.0, "avg_price": 0.0})

        mp2 = decision_kpi["avg_price"].replace(0, pd.NA).max()
        decision_kpi["price_score"] = (
            (100 * (1 - decision_kpi["avg_price"] / mp2)).clip(0, 100).fillna(0) if pd.notna(mp2) and mp2 > 0 else 0.0
        )
        decision_kpi["performance_score"] = (decision_kpi["on_time_rate"] * wd + (100 - decision_kpi["defect_rate"]) * wq + decision_kpi["price_score"] * wc).round(1)
        decision_kpi["risk_flag"] = decision_kpi.apply(risk_flag, axis=1)

    decision_kpi["fit"] = (decision_kpi["on_time_rate"] >= st.session_state["req_on_time"]) & (decision_kpi["defect_rate"] <= st.session_state["max_defects"])
    decision_kpi["fit_status"] = decision_kpi["fit"].map({True: "✅ Meets criteria", False: "❌ Below threshold"})
    decision_kpi["notes_hint"] = decision_kpi["supplier_name"].apply(lambda s: note_snippet(supplier_notes, s))

    n_fit = int(decision_kpi["fit"].sum())
    n_total = len(decision_kpi)

    if n_fit == 0:
        st.warning(
            f"No suppliers meet the current thresholds (≥{st.session_state['req_on_time']}% on-time, ≤{st.session_state['max_defects']}% defects). Consider relaxing your criteria."
        )
    else:
        st.success(f"**{n_fit} of {n_total} suppliers** meet your quality and delivery thresholds — ranked below by performance score.")

    ranked = decision_kpi.sort_values(["fit", "performance_score"], ascending=[False, False])
    show_table(
        with_rank(
            format_for_display(
                ranked,
                ["supplier_name", "fit_status", "performance_score", "risk_flag", "on_time_rate", "defect_rate", "avg_price", "total_spend", "notes_hint"],
            )
        ),
        TOP_N,
    )

    st.markdown("---")
    st.subheader("Consolidation Opportunities")
    st.caption(f"Pricing analysis within scope: **{chosen_cat}**")
    impact_dec = build_pricing_impact(supplier_master, rfqs, chosen_cat)
    st.metric("Est. Annual Savings Available", _fmt_money(float(impact_dec["estimated_overpay"].sum())), help="Based on avg premium above best comparable quote × estimated units")
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
# TAB 4 · PERFORMANCE TRENDS (UPDATED: DROPDOWN MODE + BIGGER WINDOWS)
# ═══════════════════════════════════════════════════════
with tab_trends:
    st.subheader("Performance Trends (Simplified)")
    st.caption("Designed for fast comprehension: clear KPIs + one primary trend chart + one supporting chart.")

    if "order_date" not in orders.columns:
        st.warning("No `order_date` column found — trends unavailable.")
    else:
        top6 = supplier_master.sort_values("total_spend", ascending=False)["supplier_name"].head(6).tolist()

        cL, cM, cR = st.columns([2, 1, 1])

        with cL:
            st.selectbox(
                "Suppliers to display:",
                options=["Top 6 by spend", "All suppliers", "Single supplier"],
                key="trend_mode",
            )
            mode = st.session_state["trend_mode"]

            if mode == "Top 6 by spend":
                tf = top6
            elif mode == "All suppliers":
                tf = all_suppliers
            else:
                st.selectbox("Choose supplier:", options=all_suppliers, key="trend_single_supplier")
                tf = [st.session_state["trend_single_supplier"]] if st.session_state["trend_single_supplier"] else top6

        with cM:
            st.selectbox("Primary chart", ["On-Time Trend", "Late Orders Trend", "Spend Trend"], key="trend_focus")

        with cR:
            st.selectbox("Window", [6, 12, 18, 24, 36, 48, 60], key="trend_months")

        months = int(st.session_state["trend_months"])
        start_cut = (pd.Timestamp.today().to_period("M").to_timestamp()) - pd.DateOffset(months=months - 1)

        o = orders.copy()
        o = o[o["supplier_name"].isin(tf)].copy()
        o["month"] = o["order_date"].dt.to_period("M").dt.to_timestamp()
        o = o[o["month"] >= start_cut].copy()

        ot = ot_valid.copy()
        ot = ot[ot["supplier_name"].isin(tf)].copy()
        ot["month"] = ot["order_date"].dt.to_period("M").dt.to_timestamp()
        ot = ot[ot["month"] >= start_cut].copy()

        spend_recent = o.groupby("supplier_name")["po_amount"].sum().rename("spend").reset_index()

        late_recent = ot.copy()
        late_recent["days_late"] = (late_recent["actual_delivery_date"] - late_recent["promised_date"]).dt.days
        late_recent = late_recent[late_recent["days_late"] > 0]
        late_cnt = late_recent.groupby("supplier_name")["order_id"].nunique().rename("late_orders").reset_index()

        otr_m = ot.groupby("supplier_name")["on_time"].mean().rename("otr").reset_index()
        otr_m["otr_pct"] = (otr_m["otr"] * 100).round(1)
        otr_m = otr_m.drop(columns=["otr"])

        def_kpi = defects[defects["supplier_name"].isin(tf)].copy()

        k = spend_recent.merge(otr_m, on="supplier_name", how="left").merge(def_kpi, on="supplier_name", how="left").merge(late_cnt, on="supplier_name", how="left")
        k = k.fillna({"otr_pct": 0.0, "defect_rate": 0.0, "late_orders": 0.0})

        if not k.empty:
            worst_otr = k.sort_values("otr_pct").iloc[0]
            most_late = k.sort_values("late_orders", ascending=False).iloc[0]
            most_spend = k.sort_values("spend", ascending=False).iloc[0]
            st.markdown("#### Key Insights (at a glance)")
            i1, i2, i3 = st.columns(3)
            i1.info(f"**Lowest on-time**: {worst_otr['supplier_name']} ({worst_otr['otr_pct']:.1f}%)")
            i2.warning(f"**Most late orders**: {most_late['supplier_name']} ({int(most_late['late_orders'])} late)")
            i3.success(f"**Highest spend**: {most_spend['supplier_name']} ({_fmt_money(float(most_spend['spend']))})")

            st.markdown("##### Snapshot (selected suppliers)")
            snap_cols = ["supplier_name", "spend", "otr_pct", "defect_rate", "late_orders"]
            snap = k[snap_cols].rename(columns={"supplier_name": "Supplier", "spend": "Spend ($)", "otr_pct": "On-Time %", "defect_rate": "Defect %", "late_orders": "Late Orders"})
            st.dataframe(snap, use_container_width=True, hide_index=True)

        st.markdown("---")

        focus = st.session_state["trend_focus"]

        if focus == "On-Time Trend":
            st.markdown("#### On-Time Rate Trend (Monthly)")
            st.caption("Simple line chart. Higher is better. Each line is a supplier.")

            if ot.empty:
                st.info("No on-time history in the selected window.")
            else:
                heat = ot.groupby(["month", "supplier_name"])["on_time"].agg(["mean", "count"]).reset_index()
                heat.columns = ["month", "supplier_name", "on_time_rate", "n_orders"]
                heat["otr_pct"] = (heat["on_time_rate"] * 100).round(1)
                heat = heat[heat["n_orders"] >= 1]

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
            st.caption("Count of late orders per month.")

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
            st.caption("How spend is shifting over time.")

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
        with st.expander("Debug: why might Apex prices or notes be missing?"):
            st.write("Suppliers in supplier_master:", supplier_master["supplier_name"].tolist())
            st.write("Suppliers with avg_price == 0:", supplier_master.loc[supplier_master["avg_price"] <= 0, "supplier_name"].tolist())

            rfq_sup = rfqs.copy()
            rfq_sup["quoted_price"] = pd.to_numeric(rfq_sup.get("quoted_price"), errors="coerce")
            rfq_sup = rfq_sup[rfq_sup["quoted_price"].notna() & rfq_sup["quoted_price"].gt(0)]
            rfq_sup["supplier_key"] = rfq_sup["supplier_name"].apply(normalize_supplier_key)
            apex_like = rfq_sup[rfq_sup["supplier_key"].str.contains("apex", na=False)]
            st.write("RFQ rows containing 'apex' after normalization:", len(apex_like))
            if not apex_like.empty:
                st.dataframe(apex_like[["supplier_name", "quoted_price"]].head(20), use_container_width=True, hide_index=True)

            st.write("Parsed note keys (sample):", sorted(list(supplier_notes.keys()))[:50])
            st.write("Apex lookup key:", _note_key("Apex Manufacturing"))
            st.write("Apex note present:", _note_key("Apex Manufacturing") in supplier_notes)

# ═══════════════════════════════════════════════════════
# TAB 5 · FINANCIAL IMPACT
# ═══════════════════════════════════════════════════════
with tab_financial:
    st.subheader("Financial Impact Analysis")
    st.caption("Quantifies cost across pricing (overpay), quality/rework, and delivery-risk exposure.")

    cat_fin = st.session_state.get("category_choice", "(All Categories)")
    impact_df = build_pricing_impact(supplier_master, rfqs, cat_fin)
    impact_df["defect_cost"] = (impact_df["total_spend"] * (impact_df["defect_rate"] / 100) * 0.5).round(0)
    impact_df["total_risk"] = impact_df["estimated_overpay"] + impact_df["defect_cost"]

    late_spend_fin = float(impact_df.loc[impact_df["on_time_rate"] < 85, "total_spend"].sum())
    total_overpay_fin = float(impact_df["estimated_overpay"].sum())
    total_quality = float(impact_df["defect_cost"].sum())
    total_risk = total_overpay_fin + total_quality

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Quantified Risk", _fmt_money(total_risk), help="Estimated overpay + est. quality cost combined")
    f2.metric("Estimated Overpay", _fmt_money(total_overpay_fin), help="Avg premium above best comparable quote × estimated units")
    f3.metric("Est. Quality / Rework Cost", _fmt_money(total_quality), help="Defect rate × total spend × 0.5 rework cost factor")
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

    st.markdown("#### Spend Concentration vs. Performance Score")
    sp_vs_perf = impact_df.copy()
    sp_vs_perf["spend_m"] = (sp_vs_perf["total_spend"] / 1e6).round(3)
    scatter2 = (
        alt.Chart(sp_vs_perf)
        .mark_circle(size=160, opacity=0.82, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("spend_m:Q", title="Total Spend ($M)", scale=alt.Scale(zero=True)),
            y=alt.Y("performance_score:Q", title="Performance Score (0–100)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("risk_flag:N", scale=risk_color_scale, title="Risk"),
            tooltip=[
                alt.Tooltip("supplier_name:N", title="Supplier"),
                alt.Tooltip("spend_m:Q", title="Spend ($M)", format=".3f"),
                alt.Tooltip("performance_score:Q", title="Performance Score", format=".1f"),
                alt.Tooltip("estimated_overpay:Q", title="Estimated Overpay ($)", format="$,.0f"),
                alt.Tooltip("defect_cost:Q", title="Quality Cost ($)", format="$,.0f"),
                alt.Tooltip("on_time_rate:Q", title="On-Time Rate (%)", format=".1f"),
                alt.Tooltip("defect_rate:Q", title="Defect Rate (%)", format=".1f"),
                alt.Tooltip("risk_flag:N", title="Risk Flag"),
            ],
        )
        .properties(height=360)
    )
    ref = alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(color="#d1d5db", strokeDash=[5, 5], strokeWidth=1.5).encode(y="y:Q")
    st.altair_chart(scatter2 + ref, use_container_width=True)

    st.markdown("---")

    st.markdown("#### Pricing Competitiveness — Premium vs. Best Market Quote")
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
            .properties(height=max(220, len(price_plot) * 34))
        )
        st.altair_chart(price_bar, use_container_width=True)

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

    with st.expander("Methodology & data notes"):
        st.markdown(
            """
**Estimated Overpay**
Premium is computed *within the same RFQ line* (apples-to-apples — never compares prices across different parts).
Estimated Units = total spend ÷ avg quoted price. Overpay = Premium × Estimated Units.

**Quality Cost**
Est. Rework Cost = total spend × defect rate × 0.5 (conservative rework cost factor).

**Delivery Risk**
Spend flagged as "at risk" = suppliers with on-time rate < 85%. Excludes orders with missing delivery dates (not penalized).
"""
        )

    with st.expander("Debug — column names"):
        st.write("Orders:", list(orders.columns))
        st.write("Quality:", list(quality.columns))
        st.write("RFQs:", list(rfqs.columns))

    with st.expander("Debug — internal notes coverage"):
        st.write("Parsed note keys:", sorted(list(supplier_notes.keys()))[:50])
        st.write("Suppliers missing notes (by normalized key):")
        missing_notes = []
        for s in all_suppliers:
            if _note_key(s) not in supplier_notes:
                missing_notes.append(s)
        st.write(missing_notes)
