import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="Traffic-Pollution Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Overview",
        "Transport Mode Insights",
        "Weather Impact Analysis",
        "Data Explorer"
    ]
)

# ----------------------------------------------------------
# LOAD DATA FUNCTION
# ----------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load and clean dataset. Replace this with your dataset path.
    """
    try:
        df = pd.read_csv("traffic_pollution.csv")

        # --- CLEANING ---
        df.columns = df.columns.str.strip()

        # Remove unwanted chars using regex
        df["location"] = df["location"].astype(str).apply(lambda x: re.sub(r"[^A-Za-z0-9 ]", "", x))

        # Convert datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Feature Engineering
        df["hour"] = df["timestamp"].dt.hour
        df["date"] = df["timestamp"].dt.date
        df["day"] = df["timestamp"].dt.day_name()
        df["is_rain"] = np.where(df["rain"] > 0, 1, 0)

        return df

    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
df = load_data()

if df.empty:
    st.warning("Dataset not found. Upload using Data Explorer.")
else:
    st.success("Dataset Loaded Successfully!")

# ----------------------------------------------------------
# PAGE 1: OVERVIEW
# ----------------------------------------------------------
if page == "Overview":
    st.title("🚦 Traffic & Pollution Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Traffic Count", round(df["traffic_count"].mean(), 2))
    col2.metric("Avg PM2.5", round(df["pm25"].mean(), 2))
    col3.metric("Avg NO₂", round(df["no2"].mean(), 2))
    col4.metric("Avg Temperature (°C)", round(df["temperature"].mean(), 2))

    st.subheader("📈 Traffic vs Pollution (Correlation)")
    fig_scatter = px.scatter(
        df,
        x="traffic_count",
        y="pm25",
        color="temperature",
        trendline="ols",
        title="Traffic Count vs PM2.5"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("⏳ Hourly Traffic Pattern")
    hourly_df = df.groupby("hour")["traffic_count"].mean().reset_index()
    fig_hourly = px.line(hourly_df, x="hour", y="traffic_count", markers=True)
    st.plotly_chart(fig_hourly, use_container_width=True)

    st.subheader("🌍 Area-wise Pollution")
    area_df = df.groupby("location")[["pm25", "pm10", "no2", "co"]].mean().reset_index()
    fig_area = px.bar(area_df, x="location", y="pm25", title="Area-wise PM2.5")
    st.plotly_chart(fig_area, use_container_width=True)


# ----------------------------------------------------------
# PAGE 2: TRANSPORT MODE INSIGHTS
# ----------------------------------------------------------
elif page == "Transport Mode Insights":
    st.title("🚗 Transport Mode Insights")

    mode_df = df.groupby("mode")["traffic_count"].sum().reset_index()
    fig_mode = px.pie(mode_df, names="mode", values="traffic_count",
                      title="Transport Mode Contribution")
    st.plotly_chart(fig_mode, use_container_width=True)

    st.subheader("Mode-wise Pollution Impact")
    mode_poll = df.groupby("mode")[["pm25", "pm10", "no2", "co"]].mean().reset_index()
    fig_mode_poll = px.bar(
        mode_poll,
        x="mode",
        y=["pm25", "pm10", "no2", "co"],
        barmode="group",
        title="Pollution Contribution by Transport Mode"
    )
    st.plotly_chart(fig_mode_poll, use_container_width=True)


# ----------------------------------------------------------
# PAGE 3: WEATHER IMPACT ANALYSIS
# ----------------------------------------------------------
elif page == "Weather Impact Analysis":
    st.title("⛈ Weather Impact on Traffic & Pollution")

    st.subheader("Does Rain Reduce Traffic?")
    rain_traffic = df.groupby("is_rain")["traffic_count"].mean().reset_index()
    rain_traffic["Rain"] = rain_traffic["is_rain"].map({0: "No Rain", 1: "Rain"})

    fig_rain = px.bar(
        rain_traffic,
        x="Rain",
        y="traffic_count",
        title="Traffic During Rain vs No Rain"
    )
    st.plotly_chart(fig_rain, use_container_width=True)

    st.subheader("Rain vs Pollution Levels")
    rain_poll = df.groupby("is_rain")[["pm25", "pm10"]].mean().reset_index()
    rain_poll["Rain"] = rain_poll["is_rain"].map({0: "No Rain", 1: "Rain"})

    fig_rain_poll = px.bar(
        rain_poll,
        x="Rain",
        y=["pm25", "pm10"],
        barmode="group",
        title="Pollution During Rain vs No Rain"
    )
    st.plotly_chart(fig_rain_poll, use_container_width=True)


# ----------------------------------------------------------
# PAGE 4: DATA EXPLORER
# ----------------------------------------------------------
elif page == "Data Explorer":
    st.title("📂 Data Explorer")

    st.write("Upload CSV to analyze")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        user_df = pd.read_csv(uploaded)
        st.success("File uploaded successfully!")
        st.dataframe(user_df.head())

        st.subheader("Columns Summary")
        st.write(user_df.describe())

        st.subheader("Quick Chart")
        cols = user_df.columns.tolist()
        x = st.selectbox("X-axis", cols)
        y = st.selectbox("Y-axis", cols)

        fig_temp = px.scatter(user_df, x=x, y=y, title=f"{x} vs {y}")
        st.plotly_chart(fig_temp, use_container_width=True)
