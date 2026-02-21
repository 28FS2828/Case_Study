import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hoth Intelligence Hub", layout="wide")

st.title("🚀 Hoth Industries: Supplier Intelligence Hub")
st.markdown("---")

# 1. Load Data (Exact filenames from your repo)
def load_and_clean_data():
    # Reading the files exactly as they are named in your folder
    orders = pd.read_csv('Copy of supplier_orders.csv')
    quality = pd.read_csv('Copy of quality_inspections.csv')
    rfqs = pd.read_csv('Copy of rfq_responses.csv')
    
    # 2. The "Secret Sauce": Entity Resolution
    # This fixes the 'Apex' fragmentation issue
    name_map = {
        'APEX MFG': 'Apex Manufacturing Inc',
        'Apex Mfg': 'Apex Manufacturing Inc',
        'Apex Manufacturing': 'Apex Manufacturing Inc'
    }
    orders['supplier_name'] = orders['supplier_name'].replace(name_map)
    rfqs['supplier_name'] = rfqs['supplier_name'].replace(name_map)
    
    return orders, quality, rfqs

# Execute the loading
try:
    orders, quality, rfqs = load_and_clean_data()
    st.success("✅ Data Loaded & 'Apex' Entities Resolved!")
    
    # Show the cleaned list to verify
    st.subheader("Verified Supplier List")
    st.write(orders['supplier_name'].unique())
    
except Exception as e:
    st.error(f"Error loading data: {e}")
