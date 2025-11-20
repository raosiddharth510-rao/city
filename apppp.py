# app.py
"""
Traffic - Pollution - Weather Insight Dashboard (single-file Streamlit app)
Save as app.py and run:
    pip install streamlit pandas numpy plotly altair matplotlib pytz openpyxl
    streamlit run app.py

Features:
- Upload traffic, pollution, weather CSV/Excel files (or use sample synthetic data)
- Cleaning, time parsing, regex-normalization, merging, feature engineering
- Pages: Overview, Transport Mode Insights, Weather Impact Analysis, Data Explorer
- Visualizations: KPI cards, time-series, heatmaps, scatter, boxplots, mode contribution
- Export filtered cleaned dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, time
import pytz
import plotly.express as px
import altair as alt

# ---------------------------
# Configuration
# ---------------------------
st.set_page_config(page_title="Traffic ⇄ Pollution ⇄ Weather Dashboard", layout="wide")
TZ = pytz.timezone("Asia/Kolkata")  # default timezone for timestamps

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data(ttl=60*30)
def read_file(uploaded_file):
    """Read CSV / Excel / Parquet into DataFrame."""
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        elif name.endswith(".parquet"):
            return pd.read_parquet(uploaded_file)
        else:
            # fallback
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read {uploaded_file.name}: {e}")
        return pd.DataFrame()

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: re.sub(r'\s+', '_', str(c).strip().lower()))

def safe_to_datetime(s):
    """Convert series to timezone-aware datetimes in TZ."""
    try:
        s = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
    except Exception:
        s = pd.to_datetime(s.astype(str), errors='coerce', infer_datetime_format=True)
    # localize naive to UTC then convert, or localize to TZ if already naive
    s = s.dt.tz_localize(None)
    try:
        s = s.dt.tz_localize('UTC').dt.tz_convert(TZ)
    except Exception:
        try:
            s = s.dt.tz_convert(TZ)
        except Exception:
            s = s.dt.tz_localize(TZ)
    return s

def extract_area(location):
    """Heuristic to extract area name from a messy location string."""
    if pd.isna(location):
        return "unknown"
    s = str(location)
    parts = re.split(r'[,\-;/|@]', s)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return s.strip()
    # prefer last token if short & meaningful
    for t in reversed(parts):
        if re.search(r'[A-Za-z0-9]', t) and len(t) <= 30:
            return t
    return parts[-1]

def numeric_coerce(df, cols, fillna=None):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            if fillna is not None:
                df[c] = df[c].fillna(fillna)
    return df

# ---------------------------
# Synthetic sample data generator
# ---------------------------
def generate_sample(n_hours=168):
    rng = pd.date_range(end=datetime.now(TZ), periods=n_hours, freq='H')
    areas = ['Central', 'North', 'South', 'East', 'West']
    traffic_rows, poll_rows, weather_rows = [], [], []
    for ts in rng:
        for area in areas:
            base = int(abs(np.random.normal(200, 120)))
            hour = ts.hour
            rush = 1.9 if (7 <= hour <= 9 or 17 <= hour <= 19) else (0.7 + np.random.rand())
            tc = int(base * rush)
            cars = int(tc * np.random.uniform(0.4, 0.7))
            bikes = int(tc * np.random.uniform(0.05, 0.25))
            buses = int(tc * np.random.uniform(0.01, 0.06))
            trucks = int(tc * np.random.uniform(0.01, 0.08))
            autos = int(tc * np.random.uniform(0.02, 0.08))
            traffic_rows.append({
                'timestamp': ts, 'location': f"{area} Ward", 'area': area,
                'traffic_count': tc, 'cars': cars, 'bikes': bikes, 'buses': buses, 'trucks': trucks, 'autos': autos
            })
            # pollution correlated to traffic with noise
            pm25 = max(5, np.random.normal(20 + tc/60, 10))
            pm10 = max(10, np.random.normal(50 + tc/30, 15))
            no2 = max(3, np.random.normal(15 + tc/120, 5))
            co = max(0.1, np.random.normal(0.6 + tc/600, 0.2))
            poll_rows.append({'timestamp': ts + pd.Timedelta(minutes=np.random.randint(-10, 11)),
                              'location': f"{area} Ward", 'area': area,
                              'pm2_5': round(pm25,2), 'pm10': round(pm10,2), 'no2': round(no2,2), 'co': round(co,3)})
            # weather
            raining = np.random.choice([0,1], p=[0.82,0.18])
            rain_mm = round(np.random.exponential(1.2) * raining, 2)
            humidity = round(40 + tc/10 * (0.02 + 0.01*np.random.randn()),1)
            temp = round(25 + np.random.normal(0,4),1)
            weather_rows.append({'timestamp': ts + pd.Timedelta(minutes=np.random.randint(-15,16)),
                                 'location': f"{area} Ward", 'area': area,
                                 'rain': rain_mm, 'humidity': humidity, 'temperature': temp,
                                 'condition': 'Rain' if rain_mm > 0 else 'Clear'})
    return pd.DataFrame(traffic_rows), pd.DataFrame(poll_rows), pd.DataFrame(weather_rows)

# ---------------------------
# Data pipeline
# ---------------------------
def prepare_data(traffic_df, pollution_df, weather_df):
    # if none provided -> generate sample
    if (traffic_df is None or traffic_df.empty) and (pollution_df is None or pollution_df.empty) and (weather_df is None or weather_df.empty):
        traffic_df, pollution_df, weather_df = generate_sample(n_hours=168)

    # clean colnames
    traffic_df = clean_colnames(traffic_df) if traffic_df is not None else pd.DataFrame()
    pollution_df = clean_colnames(pollution_df) if pollution_df is not None else pd.DataFrame()
    weather_df = clean_colnames(weather_df) if weather_df is not None else pd.DataFrame()

    # ensure timestamp columns
    for df, name in [(traffic_df, 'traffic'), (pollution_df, 'pollution'), (weather_df, 'weather')]:
        if 'timestamp' not in df.columns:
            st.warning(f"{name} data missing 'timestamp' — filling with NaT")
            df['timestamp'] = pd.NaT

    # parse timestamps
    traffic_df['timestamp'] = safe_to_datetime(traffic_df['timestamp']) if not traffic_df.empty else pd.Series(dtype='datetime64[ns]')
    pollution_df['timestamp'] = safe_to_datetime(pollution_df['timestamp']) if not pollution_df.empty else pd.Series(dtype='datetime64[ns]')
    weather_df['timestamp'] = safe_to_datetime(weather_df['timestamp']) if not weather_df.empty else pd.Series(dtype='datetime64[ns]')

    # derive area if missing
    for df in [traffic_df, pollution_df, weather_df]:
        if 'area' not in df.columns:
            if 'location' in df.columns:
                df['area'] = df['location'].apply(extract_area)
            else:
                df['area'] = 'unknown'

    # coerce numerics
    pollution_df = numeric_coerce(pollution_df, ['pm2_5','pm10','no2','co'])
    traffic_df = numeric_coerce(traffic_df, ['traffic_count','cars','bikes','buses','trucks','autos'], fillna=0)
    weather_df = numeric_coerce(weather_df, ['rain','humidity','temperature'], fillna=0)

    # floor to hourly bins for merging
    for df in [traffic_df, pollution_df, weather_df]:
        if not df.empty:
            df['ts_hour'] = df['timestamp'].dt.floor('H')

    # aggregate
    if not traffic_df.empty:
        traffic_agg = traffic_df.groupby(['ts_hour','area'], as_index=False).agg({
            'traffic_count':'sum', 'cars':'sum','bikes':'sum','buses':'sum','trucks':'sum','autos':'sum'
        })
    else:
        traffic_agg = pd.DataFrame(columns=['ts_hour','area','traffic_count'])

    poll_cols = [c for c in ['pm2_5','pm10','no2','co'] if c in pollution_df.columns]
    if not pollution_df.empty and poll_cols:
        pollution_agg = pollution_df.groupby(['ts_hour','area'], as_index=False)[poll_cols].mean()
    else:
        pollution_agg = pd.DataFrame(columns=['ts_hour','area']+poll_cols)

    if not weather_df.empty:
        weather_agg = weather_df.groupby(['ts_hour','area'], as_index=False).agg({'rain':'sum','humidity':'mean','temperature':'mean'})
    else:
        weather_agg = pd.DataFrame(columns=['ts_hour','area','rain','humidity','temperature'])

    # merge
    master = traffic_agg.merge(pollution_agg, on=['ts_hour','area'], how='left')
    master = master.merge(weather_agg, on=['ts_hour','area'], how='left')

    # forward/backfill pollution per area
    if not pollution_agg.empty:
        master = master.sort_values(['area','ts_hour'])
        for c in poll_cols:
            master[c] = master.groupby('area')[c].apply(lambda g: g.fillna(method='ffill').fillna(method='bfill'))

    # rename pm2_5 -> pm2.5 for display
    if 'pm2_5' in master.columns:
        master = master.rename(columns={'pm2_5':'pm2.5'})

    # features
    def pollution_index(row):
        w = {'pm2.5':0.4,'pm10':0.25,'no2':0.15,'co':0.2}
        s = 0.0
        for k,wt in w.items():
            s += (row.get(k,0) or 0) * wt
        return s

    def traffic_index(row):
        if 'traffic_count' in row:
            return row['traffic_count'] or 0
        # fallback to sum of mode counts
        return sum([row.get(c,0) or 0 for c in ['cars','bikes','buses','trucks','autos']])

    master['pollution_index'] = master.apply(pollution_index, axis=1)
    master['traffic_index'] = master.apply(traffic_index, axis=1)
    master['weather_severity'] = master.apply(lambda r: min(r.get('rain',0)/10,5) + (r.get('humidity',0)/30 if r.get('humidity',0) else 0), axis=1)

    master['timestamp'] = master['ts_hour']
    master['hour'] = master['timestamp'].dt.hour
    master['weekday'] = master['timestamp'].dt.day_name()
    master['is_rush'] = master['hour'].apply(lambda h: 1 if (7 <= h <= 9 or 17 <= h <= 19) else 0)

    master = master.sort_values(['area','timestamp'])
    master['prev_traffic'] = master.groupby('area')['traffic_index'].shift(1)
    master['traffic_change_pct'] = ((master['traffic_index'] - master['prev_traffic']) / master['prev_traffic'].replace(0,np.nan)).fillna(0) * 100
    master['rain_flag'] = master['rain'].apply(lambda x: 1 if (pd.notna(x) and x>0) else 0 if 'rain' in master.columns else 0)

    master['congestion_pollution_kpi'] = master['traffic_index']*0.6 + master['pollution_index']*0.4

    # select useful columns
    cols_keep = ['timestamp','area','hour','weekday','is_rush','traffic_index','traffic_count','cars','bikes','buses','trucks','autos',
                 'pollution_index','pm2.5','pm10','no2','co','rain','humidity','temperature','weather_severity','traffic_change_pct','rain_flag','congestion_pollution_kpi']
    master = master[[c for c in cols_keep if c in master.columns]]

    # ensure timezone-aware
    try:
        master['timestamp'] = master['timestamp'].dt.tz_convert(TZ)
    except Exception:
        try:
            master['timestamp'] = master['timestamp'].dt.tz_localize(TZ)
        except Exception:
            pass

    return master

# ---------------------------
# Plot helpers
# ---------------------------
def kpi_cards(df):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Avg Traffic (index)", f"{int(df['traffic_index'].mean()):,}" if not df.empty else "—")
    with c2:
        st.metric("Avg Pollution Index", f"{df['pollution_index'].mean():.1f}" if 'pollution_index' in df.columns and not df.empty else "—")
    with c3:
        if not df.empty:
            worst = df.sort_values('traffic_index', ascending=False).iloc[0]
            st.metric("Worst Traffic (sample)", worst['timestamp'].strftime("%Y-%m-%d %H:%M"), f"{int(worst['traffic_index']):,}")
        else:
            st.metric("Worst Traffic (sample)", "—", "—")
    with c4:
        if not df.empty:
            worstp = df.sort_values('pollution_index', ascending=False).iloc[0]
            st.metric("Worst Pollution (sample)", worstp['timestamp'].strftime("%Y-%m-%d %H:%M"), f"{worstp['pollution_index']:.1f}")
        else:
            st.metric("Worst Pollution (sample)", "—", "—")

def timeseries_plot(df, area=None):
    if area:
        d = df[df['area']==area]
    else:
        d = df.groupby('timestamp', as_index=False).agg({'traffic_index':'sum','pollution_index':'mean','rain':'sum'})
    if d.empty:
        st.info("No data for the selected filters.")
        return
    fig = px.line(d, x='timestamp', y=['traffic_index','pollution_index'], labels={'value':'Index','timestamp':'Timestamp'},
                  title=f"Traffic vs Pollution {'- '+area if area else ''}")
    st.plotly_chart(fig, use_container_width=True)

def heatmap_hour_area(df, metric='traffic_index'):
    heat = df.groupby(['hour','area'], as_index=False)[metric].mean()
    if heat.empty:
        st.info("No data for heatmap.")
        return
    chart = alt.Chart(heat).mark_rect().encode(
        x=alt.X('hour:O', title='Hour of day'),
        y=alt.Y('area:N', title='Area'),
        color=alt.Color(f'{metric}:Q', title=metric.replace('_',' ').title())
    ).properties(width='100%', height=350, title=f"{metric.replace('_',' ').title()} by Hour & Area")
    st.altair_chart(chart, use_container_width=True)

def mode_contribution(df, area=None, hour=None):
    cols = [c for c in ['cars','bikes','buses','trucks','autos'] if c in df.columns]
    if not cols:
        st.info("No mode columns found in dataset.")
        return
    d = df.copy()
    if area:
        d = d[d['area']==area]
    if hour is not None:
        d = d[d['hour']==hour]
    if d.empty:
        st.info("No data for selected filters.")
        return
    sums = d[cols].sum().reset_index()
    sums.columns = ['mode','count']
    fig = px.bar(sums, x='mode', y='count', title=f"Mode contribution {'- '+area if area else ''} {'@'+str(hour) if hour is not None else ''}")
    st.plotly_chart(fig, use_container_width=True)

def scatter_rain_vs_traffic(df, area=None):
    d = df.copy()
    if area:
        d = d[d['area']==area]
    d = d[d['prev_traffic'].notnull()] if 'prev_traffic' in d.columns else d
    if d.empty:
        st.info("Not enough data for rain vs traffic analysis.")
        return
    fig = px.scatter(d, x='rain', y='traffic_change_pct', color='area', trendline='ols', hover_data=['timestamp'], title='Rain (mm) vs Traffic Change % (vs prev hour)')
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# App UI
# ---------------------------
def main():
    st.title("Traffic ⇄ Pollution ⇄ Weather — Insight Dashboard")
    st.markdown("Analyze relationships between traffic patterns, air pollution, and weather (rain).")

    st.sidebar.header("Data Upload")
    st.sidebar.write("Upload CSV/Excel files for traffic, pollution, weather. Or leave empty to use sample data.")
    up_tr = st.sidebar.file_uploader("Traffic file (optional)", type=['csv','xlsx','xls','parquet'])
    up_po = st.sidebar.file_uploader("Pollution file (optional)", type=['csv','xlsx','xls','parquet'])
    up_we = st.sidebar.file_uploader("Weather file (optional)", type=['csv','xlsx','xls','parquet'])

    # Read provided files
    traffic_df = read_file(up_tr) if up_tr else pd.DataFrame()
    pollution_df = read_file(up_po) if up_po else pd.DataFrame()
    weather_df = read_file(up_we) if up_we else pd.DataFrame()

    with st.spinner("Preparing data..."):
        master = prepare_data(traffic_df, pollution_df, weather_df)

    # Global filters
    st.sidebar.header("Filters")
    if not master.empty:
        min_date = master['timestamp'].min().date()
        max_date = master['timestamp'].max().date()
    else:
        min_date = datetime.now(TZ).date()
        max_date = datetime.now(TZ).date()

    date_range = st.sidebar.date_input("Date range (start, end)", value=(min_date, max_date))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
    start_dt = TZ.localize(datetime.combine(start_date, time.min))
    end_dt = TZ.localize(datetime.combine(end_date, time.max))

    areas = sorted(master['area'].dropna().unique().tolist()) if not master.empty else []
    selected_area = st.sidebar.selectbox("Area", options=["All"] + areas, index=0)
    hour_range = st.sidebar.slider("Hour range", 0, 23, (0, 23))

    st.sidebar.header("Pages")
    page = st.sidebar.selectbox("Choose page", options=["Overview","Transport Mode Insights","Weather Impact Analysis","Data Explorer"])

    # Apply filters
    df = master.copy()
    if not df.empty:
        df = df[(df['timestamp']>=start_dt) & (df['timestamp']<=end_dt)]
        if selected_area != "All":
            df = df[df['area']==selected_area]
        df = df[(df['hour']>=hour_range[0]) & (df['hour']<=hour_range[1])]

    # Pages
    if page == "Overview":
        st.header("Overview")
        if df.empty:
            st.info("No data available for selected filters yet.")
        kpi_cards(df if not df.empty else master)
        st.subheader("Traffic vs Pollution Time Series")
        timeseries_area = st.selectbox("Timeseries area (All aggregates)", options=["All"] + areas, index=0)
        timeseries_plot(df if timeseries_area=="All" else master, None if timeseries_area=="All" else timeseries_area)
        st.subheader("Heatmap (Hour × Area)")
        metric = st.selectbox("Choose metric", options=['traffic_index','pollution_index','congestion_pollution_kpi'], index=0)
        heatmap_hour_area(master if not master.empty else df, metric=metric)
        st.subheader("Top Areas by Congestion–Pollution KPI")
        if not master.empty:
            top_n = st.slider("Top N areas", 3, 10, 5)
            top = master.groupby('area', as_index=False)['congestion_pollution_kpi'].mean().nlargest(top_n, 'congestion_pollution_kpi')
            st.bar_chart(top.set_index('area')['congestion_pollution_kpi'])

    elif page == "Transport Mode Insights":
        st.header("Transport Mode Insights")
        mode_area = st.selectbox("Area for mode insights", options=["All"]+areas, index=0)
        mode_hour = st.slider("Hour (choose -1 for all)", -1, 23, -1)
        mode_contribution(master if mode_area=="All" else master[master['area']==mode_area],
                          None if mode_area=="All" else mode_area,
                          None if mode_hour==-1 else mode_hour)
        st.markdown("**Estimated pollution contribution by mode (heuristic)**")
        st.info("This is an estimation based on count × emission factor heuristics.")
        default_factors = {'cars':1.0,'bikes':0.2,'buses':2.0,'trucks':2.5,'autos':0.8}
        st.write("Emission factors (change in code to tune):", default_factors)
        cols = [c for c in ['cars','bikes','buses','trucks','autos'] if c in master.columns]
        if cols:
            tmp = master.copy()
            tmp['estimated_emission'] = 0
            for c in cols:
                tmp['estimated_emission'] += tmp[c] * default_factors.get(c,1.0)
            me = {c: (tmp[c]*default_factors.get(c,1.0)).sum() for c in cols}
            medf = pd.DataFrame(list(me.items()), columns=['mode','est_emission'])
            fig = px.pie(medf, values='est_emission', names='mode', title='Estimated share by mode')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No mode columns present.")

    elif page == "Weather Impact Analysis":
        st.header("Weather Impact Analysis")
        st.markdown("Study how rain (and other weather) impacts traffic & pollution.")
        rain_area = st.selectbox("Area for rain analysis", options=["All"]+areas, index=0)
        df_rain = master if rain_area=="All" else master[master['area']==rain_area]
        st.subheader("Rain vs Traffic Change %")
        scatter_rain_vs_traffic(df_rain, None if rain_area=="All" else rain_area)
        with st.expander("Aggregated: Rain vs No Rain"):
            if df_rain.empty:
                st.info("No data")
            else:
                df_rain['rain_flag'] = df_rain['rain'].apply(lambda r: 1 if (pd.notna(r) and r>0) else 0)
                stats = df_rain.groupby('rain_flag', as_index=False).agg({'traffic_index':'mean','pollution_index':'mean','traffic_change_pct':'mean'})
                stats['rain_flag'] = stats['rain_flag'].map({0:'No Rain',1:'Rain'})
                st.dataframe(stats)
        st.subheader("Before - During - After Rain Comparison (boxplots)")
        r = master.copy()
        if not r.empty and 'rain' in r.columns:
            r['rain_flag'] = (r['rain']>0).astype(int)
            r['rain_prev'] = r.groupby('area')['rain_flag'].shift(-1).fillna(0).astype(int)
            r['rain_next'] = r.groupby('area')['rain_flag'].shift(1).fillna(0).astype(int)
            labels = []
            for _, row in r.iterrows():
                if row['rain_flag']==1:
                    labels.append('During Rain')
                elif row['rain_next']==1:
                    labels.append('Before Rain')
                elif row['rain_prev']==1:
                    labels.append('After Rain')
                else:
                    labels.append('Normal')
            r['rain_window'] = labels
            if 'traffic_index' in r.columns:
                fig = px.box(r, x='rain_window', y='traffic_index', title='Traffic Index: Before/During/After Rain')
                st.plotly_chart(fig, use_container_width=True)
            if 'pollution_index' in r.columns:
                fig2 = px.box(r, x='rain_window', y='pollution_index', title='Pollution Index: Before/During/After Rain')
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Insufficient data for before/during/after analysis.")

    elif page == "Data Explorer":
        st.header("Data Explorer")
        st.write(f"Rows (filtered): {len(df)}")
        st.dataframe(df.head(500))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV of filtered data", csv, file_name="cleaned_filtered_data.csv", mime="text/csv")
        if up_tr:
            with st.expander("Uploaded Traffic file preview"):
                st.dataframe(read_file(up_tr).head(200))
        if up_po:
            with st.expander("Uploaded Pollution file preview"):
                st.dataframe(read_file(up_po).head(200))
        if up_we:
            with st.expander("Uploaded Weather file preview"):
                st.dataframe(read_file(up_we).head(200))

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit. Adjust heuristics in code for production-calibration.")

if __name__ == "__main__":
    main()
