import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hoth Intelligence Hub", layout="wide")

st.title("🚀 Hoth Industries: Supplier Intelligence Hub")
st.markdown("---")

# ---------------------------------------------------
# LOAD + CLEAN DATA
# ---------------------------------------------------
@st.cache_data
def load_and_clean_data():
    orders = pd.read_csv('Copy of supplier_orders.csv')
    quality = pd.read_csv('Copy of quality_inspections.csv')
    rfqs = pd.read_csv('Copy of rfq_responses.csv')

    # --- Normalize supplier names ---
    name_map = {
        'APEX MFG': 'Apex Manufacturing Inc',
        'Apex Mfg': 'Apex Manufacturing Inc',
        'Apex Manufacturing': 'Apex Manufacturing Inc'
    }

    orders['supplier_name'] = orders['supplier_name'].replace(name_map)
    rfqs['supplier_name'] = rfqs['supplier_name'].replace(name_map)
    quality['supplier_name'] = quality['supplier_name'].replace(name_map)

    return orders, quality, rfqs

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
try:
    orders, quality, rfqs = load_and_clean_data()
    st.success("✅ Data loaded and supplier names normalized")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---------------------------------------------------
# KPI CALCULATIONS
# ---------------------------------------------------

# Total spend per supplier
spend = orders.groupby('supplier_name')['order_value'].sum().reset_index()
spend.columns = ['supplier_name', 'total_spend']

# On-time delivery rate
on_time = orders.groupby('supplier_name')['on_time'].mean().reset_index()
on_time.columns = ['supplier_name', 'on_time_rate']
on_time['on_time_rate'] = on_time['on_time_rate'] * 100

# Defect rate
defects = quality.groupby('supplier_name')['defect'].mean().reset_index()
defects.columns = ['supplier_name', 'defect_rate']
defects['defect_rate'] = defects['defect_rate'] * 100

# Avg RFQ price
avg_price = rfqs.groupby('supplier_name')['quoted_price'].mean().reset_index()
avg_price.columns = ['supplier_name', 'avg_price']

# ---------------------------------------------------
# MERGE INTO MASTER SUPPLIER TABLE
# ---------------------------------------------------
supplier_master = spend.merge(on_time, on='supplier_name', how='left') \
                       .merge(defects, on='supplier_name', how='left') \
                       .merge(avg_price, on='supplier_name', how='left')

supplier_master = supplier_master.fillna(0)

# ---------------------------------------------------
# SIMPLE PERFORMANCE SCORE
# ---------------------------------------------------
# Higher score = better supplier
supplier_master['performance_score'] = (
    (supplier_master['on_time_rate'] * 0.4) +
    ((100 - supplier_master['defect_rate']) * 0.3) +
    ((1 / (supplier_master['avg_price'] + 1)) * 100 * 0.3)
)

supplier_master = supplier_master.sort_values(by='performance_score', ascending=False)

# ---------------------------------------------------
# RISK FLAGS
# ---------------------------------------------------
def risk_flag(row):
    if row['defect_rate'] > 8:
        return "🔴 High Quality Risk"
    if row['on_time_rate'] < 85:
        return "🟠 Delivery Risk"
    if row['avg_price'] > supplier_master['avg_price'].mean():
        return "🟡 Cost Risk"
    return "🟢 Strategic Supplier"

supplier_master['risk_flag'] = supplier_master.apply(risk_flag, axis=1)

# ---------------------------------------------------
# DISPLAY DASHBOARD
# ---------------------------------------------------
st.subheader("🏭 Supplier Intelligence Overview")
st.dataframe(supplier_master, use_container_width=True)

# Top risks
st.markdown("### 🔴 Highest Risk Suppliers")
high_risk = supplier_master[supplier_master['risk_flag'].str.contains("🔴|🟠")]
st.dataframe(high_risk, use_container_width=True)

# Opportunities
st.markdown("### 🟢 Strategic / High Performing Suppliers")
top_suppliers = supplier_master.head(5)
st.dataframe(top_suppliers, use_container_width=True)
