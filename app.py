# app.py

"""Robust Business Intelligence Streamlit app (patched).
- Fixes datetime resample TypeError by selecting numeric columns before aggregation.
- Adds defensive checks for required columns and parsing.
"""
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Sales BI — Dashboard (patched)', layout='wide', initial_sidebar_state='expanded')

@st.cache_data
def generate_synthetic_sales(n_customers=1000, start_date='2023-01-01', end_date='2024-12-31'):
    rng = pd.date_range(start=start_date, end=end_date, freq='H')
    np.random.seed(42)
    customer_ids = [f"C{10000 + i}" for i in range(n_customers)]
    avg_orders_per_customer = 8
    rows = []
    product_categories = ['Electronics', 'Home', 'Clothing', 'Grocery', 'Sports', 'Beauty']
    cities = [
        ('Mumbai', 19.07, 72.87),
        ('Delhi', 28.61, 77.23),
        ('Bengaluru', 12.97, 77.59),
        ('Hyderabad', 17.38, 78.48),
        ('Chennai', 13.08, 80.27),
        ('Kolkata', 22.57, 88.36),
        ('Pune', 18.52, 73.85),
    ]
    for cust in customer_ids:
        n_orders = max(1, np.random.poisson(avg_orders_per_customer))
        first_order = pd.to_datetime(start_date) + pd.to_timedelta(np.random.randint(0, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days), unit='D')
        for i in range(n_orders):
            order_time = first_order + pd.to_timedelta(np.random.exponential(scale=30), unit='D') + pd.to_timedelta(np.random.randint(0,24), unit='h')
            if order_time > pd.to_datetime(end_date):
                continue
            order_id = f"O{np.random.randint(10**7, 10**8)}"
            cat = np.random.choice(product_categories, p=[0.18,0.16,0.18,0.26,0.12,0.10])
            price = {
                'Electronics': np.random.normal(120, 60),
                'Home': np.random.normal(60, 30),
                'Clothing': np.random.normal(35, 20),
                'Grocery': np.random.normal(12, 6),
                'Sports': np.random.normal(55, 25),
                'Beauty': np.random.normal(22, 10)
            }[cat]
            price = max(2, round(price,2))
            qty = np.random.choice([1,1,1,2], p=[0.6,0.2,0.1,0.1])
            revenue = round(price * qty * (1 + np.random.normal(0, 0.03)), 2)
            city, lat, lon = cities[np.random.randint(0, len(cities))]
            rows.append({
                'order_id': order_id,
                'customer_id': cust,
                'order_datetime': order_time,
                'category': cat,
                'price': price,
                'quantity': qty,
                'revenue': revenue,
                'city': city,
                'lat': lat + np.random.normal(0, 0.03),
                'lon': lon + np.random.normal(0, 0.03),
            })
    df = pd.DataFrame(rows)
    df = df.sort_values('order_datetime').reset_index(drop=True)
    df['order_date'] = df['order_datetime'].dt.date
    return df

st.sidebar.title("Data & Filters (patched)")
uploaded = st.sidebar.file_uploader("Upload transactions CSV (optional). Required: order_datetime, customer_id, revenue", type=['csv'])
use_sample = st.sidebar.button("Load synthetic sample data")

if uploaded:
    try:
        df = pd.read_csv(uploaded, parse_dates=['order_datetime'])
        st.sidebar.success("Uploaded file loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    df = generate_synthetic_sales()
    if use_sample:
        st.sidebar.success("Synthetic sample loaded")

# Defensive checks for required columns
required_cols = {'order_datetime', 'customer_id', 'revenue', 'order_id'}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Dataset missing required columns: {missing}.\nHint: ensure CSV has columns named exactly as above.")
    st.stop()

# Ensure datetime parsing
df['order_datetime'] = pd.to_datetime(df['order_datetime'], errors='coerce')
if df['order_datetime'].isna().any():
    n_bad = df['order_datetime'].isna().sum()
    st.warning(f"Found {n_bad} rows with unparsable 'order_datetime' and they will be dropped.")
    df = df.dropna(subset=['order_datetime']).copy()

# derived fields
df['order_date'] = df['order_datetime'].dt.date
df['order_month'] = df['order_datetime'].dt.to_period('M').dt.to_timestamp()

# Sidebar filters
min_date = df['order_datetime'].min().date()
max_date = df['order_datetime'].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
if len(date_range) != 2:
    st.sidebar.error("Select start and end date")
    st.stop()
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

categories = ['All'] + sorted(df['category'].dropna().unique().tolist())
category = st.sidebar.selectbox("Category", categories)

cities = sorted(df['city'].dropna().unique().tolist())
city_select = st.sidebar.multiselect("Cities (optional)", options=cities, default=cities)

# Apply filters
mask = (df['order_datetime'] >= start_dt) & (df['order_datetime'] <= end_dt)
if category != 'All':
    mask &= (df['category'] == category)
if city_select:
    mask &= df['city'].isin(city_select)

df_filt = df.loc[mask].copy()
if df_filt.empty:
    st.warning("No data in selected filters. Try expanding the date range or removing other filters.")
    st.stop()

# KPIs
def compute_kpis(df_in):
    total_revenue = float(df_in['revenue'].sum())
    total_orders = int(df_in['order_id'].nunique())
    total_customers = int(df_in['customer_id'].nunique())
    aov = total_revenue / total_orders if total_orders else 0
    avg_order_frequency = df_in.groupby('customer_id')['order_id'].nunique().mean()
    return {
        'Total Revenue': total_revenue,
        'Total Orders': total_orders,
        'Total Customers': total_customers,
        'Avg Order Value (AOV)': aov,
        'Avg Orders/Customer': avg_order_frequency
    }

kpis = compute_kpis(df_filt)
k1, k2, k3, k4, k5 = st.columns([1.8,1,1,1,1])
k1.metric("Total Revenue", f"₹ {kpis['Total Revenue']:,.0f}")
k2.metric("Total Orders", f"{kpis['Total Orders']:,}")
k3.metric("Total Customers", f"{kpis['Total Customers']:,}")
k4.metric("Avg Order Value (AOV)", f"₹ {kpis['Avg Order Value (AOV)']:.2f}")
k5.metric("Avg Orders / Customer", f"{kpis['Avg Orders/Customer']:.2f}")

st.markdown("---")

# Time series: revenue by time (fixed: select numeric column before sum)
st.header("Revenue Over Time")
ts_agg = st.selectbox("Aggregate by", ['D', 'W', 'M'], index=1)
# Ensure revenue is numeric
df_filt['revenue'] = pd.to_numeric(df_filt['revenue'], errors='coerce').fillna(0.0)

if ts_agg == 'D':
    df_ts = df_filt.set_index('order_datetime')['revenue'].resample('D').sum().reset_index()
elif ts_agg == 'W':
    df_ts = df_filt.set_index('order_datetime')['revenue'].resample('W').sum().reset_index()
else:
    df_ts = df_filt.set_index('order_datetime')['revenue'].resample('M').sum().reset_index()

fig_ts = px.line(df_ts, x='order_datetime', y='revenue', title='Revenue time series', labels={'order_datetime':'Date','revenue':'Revenue (₹)'})
fig_ts.add_trace(go.Bar(x=df_ts['order_datetime'], y=df_ts['revenue'], name='Revenue (bar)', opacity=0.18))
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown('---')

# Category breakdown
st.header('Category Breakdown & Top Items')
left, right = st.columns([2,1])
with left:
    cat_df = df_filt.groupby('category').agg(revenue=('revenue','sum'), orders=('order_id','nunique')).reset_index().sort_values('revenue', ascending=False)
    fig_cat = px.pie(cat_df, values='revenue', names='category', title='Revenue share by category', hole=0.35)
    st.plotly_chart(fig_cat, use_container_width=True)
    st.dataframe(cat_df.style.format({'revenue':'₹{:,.2f}','orders':'{:,}'}), height=240)
with right:
    st.subheader('Top 10 Customers by Revenue')
    top_cust = df_filt.groupby('customer_id').agg(revenue=('revenue','sum'), orders=('order_id','nunique')).reset_index().sort_values('revenue', ascending=False).head(10)
    st.table(top_cust.style.format({'revenue':'₹{:,.2f}'}))

st.markdown('---')

# Cohort retention (monthly) - limited to cohorts present in df
st.header('Cohort Retention (monthly)')
cohort = df[['customer_id','order_datetime']].copy()
cohort['order_month'] = cohort['order_datetime'].dt.to_period('M').dt.to_timestamp()
first_purchase = cohort.groupby('customer_id')['order_month'].min().reset_index().rename(columns={'order_month':'cohort_month'})
cohort = cohort.merge(first_purchase, on='customer_id')
cohort['period_number'] = ((cohort['order_month'].dt.year - cohort['cohort_month'].dt.year) * 12 + (cohort['order_month'].dt.month - cohort['cohort_month'].dt.month)).astype(int)
cohort_counts = cohort.groupby(['cohort_month','period_number'])['customer_id'].nunique().reset_index()
cohort_pivot = cohort_counts.pivot_table(index='cohort_month', columns='period_number', values='customer_id')
if cohort_pivot.empty:
    st.info('Not enough data to build cohort heatmap.')
else:
    cohort_sizes = cohort_pivot.iloc[:,0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0).fillna(0)
    retain_display = retention.copy().iloc[-12:]
    fig_cohort = px.imshow(retain_display, text_auto='.0%', labels=dict(x='Months since cohort', y='Cohort month', color='Retention'), aspect='auto', title='Customer retention by cohort (last 12 cohorts)')
    st.plotly_chart(fig_cohort, use_container_width=True)

st.markdown('---')

# -------------------------
# RFM segmentation
# -------------------------
st.header('RFM Segmentation (customers)')

# snapshot date for recency
snapshot_date = df['order_datetime'].max() + pd.Timedelta(days=1)

# compute raw RFM metrics
rfm = df.groupby('customer_id').agg(
    recency_days=('order_datetime', lambda x: (snapshot_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('revenue', 'sum')
).reset_index()

# safe qcut helper (fallback if too few unique values)
def safe_qcut(series, q):
    try:
        return pd.qcut(series, q, labels=list(range(1, q+1))).astype(int)
    except Exception:
        # fallback: use percentile bins on ranks
        ranks = series.rank(method='first', pct=True)
        return pd.cut(ranks, bins=q, labels=list(range(1, q+1))).astype(int)

# score mapping: r = recency (recent -> higher score), f = frequency, m = monetary
rfm['r_score'] = safe_qcut(rfm['recency_days'], 5).rsub(6)  # invert so recent -> 5
rfm['f_score'] = safe_qcut(rfm['frequency'], 5)
rfm['m_score'] = safe_qcut(rfm['monetary'], 5)
rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

# define segments from scores, then create column BEFORE aggregating
def rfm_segment(row):
    if row['r_score'] >= 4 and row['f_score'] >= 4 and row['m_score'] >= 4:
        return 'Champions'
    if row['r_score'] >= 3 and row['f_score'] >= 3:
        return 'Loyal'
    if row['r_score'] <= 2 and row['f_score'] >= 4:
        return 'At Risk (freq)'
    if row['r_score'] <= 2:
        return 'Requires Winback'
    return 'Others'

rfm['segment'] = rfm.apply(rfm_segment, axis=1)

# now safe to compute counts: use rename_axis + reset_index(name=...) to avoid duplicate names
seg_counts = rfm['segment'].value_counts().rename_axis('segment').reset_index(name='count')
# ensure columns are unique and ordered
seg_counts.columns = ['segment', 'count']

# plot and display
fig_rfm = px.bar(seg_counts, x='segment', y='count', title='RFM customer segments', text_auto=True)
st.plotly_chart(fig_rfm, use_container_width=True)

st.dataframe(rfm.sort_values('monetary', ascending=False).head(15).style.format({'monetary':'₹{:,.2f}', 'recency_days':'{:,}', 'frequency':'{:,}'}))

# ensure unique column names
seg_counts.columns = ['segment', 'count']

def rfm_segment(row):
    if row['r_score'] >= 4 and row['f_score'] >= 4 and row['m_score'] >= 4:
        return 'Champions'
    if row['r_score'] >= 3 and row['f_score'] >= 3:
        return 'Loyal'
    if row['r_score'] <= 2 and row['f_score'] >= 4:
        return 'At Risk (freq)'
    if row['r_score'] <= 2:
        return 'Requires Winback'
    return 'Others'

rfm['segment'] = rfm.apply(rfm_segment, axis=1)
seg_counts = rfm['segment'].value_counts().rename_axis('segment').reset_index(name='count')
seg_counts.columns = ['segment', 'count']

fig_rfm = px.bar(seg_counts, x='segment', y='count', title='RFM customer segments', text_auto=True)
st.plotly_chart(fig_rfm, use_container_width=True)
st.dataframe(rfm.sort_values('monetary', ascending=False).head(15).style.format({'monetary':'₹{:,.2f}','recency_days':'{:,}','frequency':'{:,}'}))

# Geo
st.header('Geographic Sales (by city)')
city_geo = df_filt.groupby(['city','lat','lon']).agg(revenue=('revenue','sum'), orders=('order_id','nunique')).reset_index()
if city_geo.empty:
    st.info('No geographic data to display.')
else:
    fig_map = px.scatter_mapbox(city_geo, lat='lat', lon='lon', size='revenue', hover_name='city', hover_data={'revenue':':.0f','orders':':.0f'}, zoom=3, height=400)
    fig_map.update_layout(mapbox_style='open-street-map')
    fig_map.update_layout(margin={'r':0,'t':0,'l':0,'b':0})
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown('---')
st.header('Export & Storytelling')
col1, col2 = st.columns(2)
with col1:
    csv_bytes = df_filt.to_csv(index=False).encode('utf-8')
    st.download_button('Download filtered transactions (CSV)', data=csv_bytes, file_name='transactions_filtered.csv', mime='text/csv')
with col2:
    st.write('Suggested short story for stakeholders:')
    st.info(f"During the selected period, we generated **₹ {kpis['Total Revenue']:,.0f}** in revenue from **{kpis['Total Customers']:,}** customers. Focus: champions and loyal segments—top customers deliver significant revenue; consider targeted winback campaigns for 'Requires Winback'.")
