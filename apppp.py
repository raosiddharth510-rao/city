# City Mobility & Traffic Insights Platform

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ============================
# Helper Functions
# ============================

def extract_numeric(value):
    try:
        if isinstance(value, str):
            cleaned = re.sub(r"[^0-9.-]", "", value)
            return float(cleaned) if cleaned else np.nan
        return float(value)
    except:
        return np.nan

# ============================
# Streamlit App
# ============================
st.title("🚦 City Mobility Insights Platform (Traffic + Road Classification)")

st.markdown("""
This dashboard performs:

- *Vehicle Speed Trends*
- *Hourly Traffic Insight*
- *Vehicle Type Distribution*
- *Road-Type Impact Analysis*
- *Worst-Time Analysis by Weather*
- *Worst-Area Analysis by Incident Type*
""")

# ----------------------------
# Upload Section
# ----------------------------
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    return df

if file:
    df = load_data(file)
else:
    st.warning("Please upload the dataset to continue.")
    st.stop()

st.success("Dataset Loaded Successfully!")

# ============================
# Data Cleaning
# ============================
st.header("🔧 Data Cleaning & Processing")

numeric_like_cols = [c for c in df.columns if any(x in c.lower() for x in ["speed", "kmph", "level", "count"])]
for col in numeric_like_cols:
    df[col] = df[col].apply(extract_numeric)

# Detect datetime
datetime_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
for col in datetime_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

if datetime_cols:
    dtcol = datetime_cols[0]
    if pd.api.types.is_datetime64_any_dtype(df[dtcol]):
        df["hour"] = df[dtcol].dt.hour
        df["day"] = df[dtcol].dt.date
    else:
        df["hour"] = pd.NA
else:
    df["hour"] = pd.NA

# Detect columns
area_col, road_type_col, weather_col, incident_col = None, None, None, None

for col in df.columns:
    if "area" in col.lower() or "location" in col.lower():
        area_col = col
    if "road" in col.lower():
        road_type_col = col
    if "weather" in col.lower():
        weather_col = col
    if "incident" in col.lower() or "accident" in col.lower() or "event" in col.lower():
        incident_col = col

# Speed Column
speed_col = None
for col in df.columns:
    if "speed" in col.lower() or "kmph" in col.lower():
        speed_col = col
        break

hour_col = "hour"

st.write(df.head())

# ============================
# Section 1 — Speed vs Hour
# ============================
st.header("📊 Vehicle Speed vs Hour")

if speed_col and hour_col:
    fig, ax = plt.subplots()
    ax.scatter(df[hour_col], df[speed_col], alpha=0.5)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Speed (kmph)")
    ax.set_title("Hourly Vehicle Speed Trend")
    st.pyplot(fig)

# ============================
# Section 2 — Vehicle Type
# ============================
st.header("🚗 Vehicle Type Insights")

vehicle_type_col = None
for col in df.columns:
    if "vehicle" in col.lower() or "type" in col.lower():
        vehicle_type_col = col

if vehicle_type_col:
    st.bar_chart(df[vehicle_type_col].value_counts())

# ============================
# Section 3 — Road-Type Impact
# ============================
st.header("🛣 Road Type Impact Analysis")

if road_type_col and speed_col:
    fig, ax = plt.subplots()
    sns.boxplot(x=df[road_type_col], y=df[speed_col], ax=ax)
    ax.set_title("Speed Variation by Road Type")
    st.pyplot(fig)

# ============================
# Section 4 — Worst Time & Area (Speed Heatmap)
# ============================
st.header("🔥 Worst Time & Area (Speed Heatmap)")

if speed_col and area_col:
    pivot = df.pivot_table(values=speed_col, index="hour", columns=area_col, aggfunc="mean").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, ax=ax)
    ax.set_title("Speed Drop Heatmap (Hour × Area)")
    st.pyplot(fig)

# ============================
# NEW — Worst Time by Weather
# ============================
st.header("🌧 Worst Time Analysis by Weather Condition")

if weather_col and speed_col:
    pivot_w = df.pivot_table(values=speed_col, index="hour", columns=weather_col, aggfunc="mean").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_w, ax=ax)
    ax.set_title(" Weather (Speed Drop)")
    st.pyplot(fig)

# ============================
# NEW — Worst Area by Incident Type
# ============================
st.header("🚨 Worst Area Analysis by Incident Type")

if incident_col and area_col and speed_col:
    pivot_i = df.pivot_table(
        values=speed_col,
        index=area_col,
        columns=incident_col,
        aggfunc="mean"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_i, ax=ax)
    ax.set_title(" Incident Type (Speed Drop)")
    st.pyplot(fig)

# ============================
# Data Explorer
# ============================
st.header("📁 Data Explorer")
st.dataframe(df)