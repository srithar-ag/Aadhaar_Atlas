import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="UIDAI Aadhaar Enrolment Analytics",
    layout="wide"
)

st.title("📊 Aadhaar Atlas – Enrolment Analytics Dashboard")
st.markdown("**UIDAI Hackathon | Aggregated & Anonymised Data**")

# ---------------------------------
# Load Data
# ---------------------------------
@st.cache_data
def load_data():
    files = [
        "data/api_data_aadhar_enrolment_0_500000.csv",
        "data/api_data_aadhar_enrolment_500000_1000000.csv",
        "data/api_data_aadhar_enrolment_1000000_1006029.csv"
    ]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
    return df

df = load_data()

# ---------------------------------
# Sidebar Filters
# ---------------------------------
st.sidebar.header("Filters")

state = st.sidebar.selectbox(
    "Select State",
    options=["All"] + sorted(df['state'].unique().tolist())
)

if state != "All":
    df = df[df['state'] == state]

district = st.sidebar.selectbox(
    "Select District",
    options=["All"] + sorted(df['district'].unique().tolist())
)

if district != "All":
    df = df[df['district'] == district]

# ---------------------------------
# Feature Engineering
# ---------------------------------
df['total_enrolments'] = (
    df['age_0_5'] +
    df['age_5_17'] +
    df['age_18_greater']
)

# ---------------------------------
# KPI Metrics
# ---------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Enrolments", f"{int(df['total_enrolments'].sum()):,}")
col2.metric("Children (0–17)", f"{int((df['age_0_5'] + df['age_5_17']).sum()):,}")
col3.metric("Adults (18+)", f"{int(df['age_18_greater'].sum()):,}")

# ---------------------------------
# Time Series Trend
# ---------------------------------
st.subheader("📈 Enrolment Trend Over Time")

trend = df.groupby('date')['total_enrolments'].sum().reset_index()

fig, ax = plt.subplots()
ax.plot(trend['date'], trend['total_enrolments'])
ax.set_xlabel("Date")
ax.set_ylabel("Total Enrolments")
st.pyplot(fig)

# ---------------------------------
# Age-wise Distribution
# ---------------------------------
st.subheader("👶🧑 Age-wise Enrolment Distribution")

age_data = pd.DataFrame({
    "Age Group": ["0–5 Years", "5–17 Years", "18+ Years"],
    "Count": [
        df['age_0_5'].sum(),
        df['age_5_17'].sum(),
        df['age_18_greater'].sum()
    ]
})

fig2, ax2 = plt.subplots()
ax2.bar(age_data['Age Group'], age_data['Count'])
ax2.set_ylabel("Enrolment Count")
st.pyplot(fig2)

# ---------------------------------
# District-Level Volatility (Anomaly Proxy)
# ---------------------------------
st.subheader("🚨 District-Level Enrolment Volatility")

district_stats = df.groupby('district')['total_enrolments'].std().reset_index()
district_stats.columns = ['district', 'volatility']
district_stats['z_score'] = zscore(district_stats['volatility'].fillna(0))

anomalies = district_stats[district_stats['z_score'].abs() > 3]

if anomalies.empty:
    st.success("No extreme volatility detected.")
else:
    st.warning("High-volatility districts detected:")
    st.dataframe(anomalies.sort_values('z_score', ascending=False))

# ---------------------------------
# Insights
# ---------------------------------
st.subheader("🧠 Key Insights")

st.markdown("""
- Enrolment trends show **temporal variability**, indicating periodic demand surges  
- Age-wise distribution highlights **strong child enrolment dependency**, especially in urban districts  
- High district volatility may indicate **migration, outreach drives, or administrative stress**
""")

st.markdown("---")
st.caption("UIDAI Hackathon Dashboard | Built in VS Code using Streamlit")
