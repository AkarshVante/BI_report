# streamlit_app.py
"""
Business Intelligence Dashboard — Sales Analytics
Single-file Streamlit app that:
 - generates synthetic transactional sales data (or accepts upload)
 - shows KPIs, time series, product/category breakdowns
 - cohort retention heatmap
 - RFM segmentation with customer table
 - geographic sales scatter map
 - interactive filters (date range, category, region)
Designed for resume/demo / Streamlit Cloud deployment.
"""

from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import quantile_transform

st.set_page_config(page_title="Sales BI — Dashboard", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Utility: generate synthetic dataset
# -------------------------
@st.cache_data
def generate_synthetic_sales(n_customers=1000, start_date='2023-01-01', end_date='2024-12-31'):
    rng = pd.date_range(start=start_date, end=end_date, freq='H')
    np.random.seed(42)
    # create customers
    customer_ids = [f"C{10000 + i}" for i in range(n_customers)]
    avg_orders_per_customer = 8
    # Generate transactions
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
        # decide number of orders for this customer
        n_orders = max(1, np.random.poisson(avg_orders_per_customer))
        first_order = pd.to_datetime(start_date) + pd.to_timedelta(np.random.randint(0, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days), unit='D')
        for i in range(n_orders):
            # spread purchases over entire period after first order
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

# -------------------------
# Load / Upload data
# -------------------------
st.sidebar.title("Data & Filters")
uploaded = st.sidebar.file_uploader("Upload transactions CSV (optional). Required columns: order_datetime, customer_id, revenue, category, city", type=['csv'])
use_sample = st.sidebar.button("Load synthetic sample data")

if uploaded:
    df = pd.read_csv(uploaded, parse_dates=['order_datetime'])
    st.sidebar.success("Uploaded file loaded")
elif use_sample or not uploaded:
    df = generate_synthetic_sales()
    if use_sample:
        st.sidebar.success("Synthetic sample loaded")

# minimal required columns check
required_cols = {'order_datetime', 'customer_id', 'revenue'}
if not required_cols.issubset(set(df.columns)):
    st.error(f"Dataset missing required columns: {required_cols - set(df.columns)}")
    st.stop()

# convert datetimes
df['order_datetime'] = pd.to_datetime(df['order_datetime'])
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

categories = ['All'] + sorted(df['category'].unique().tolist())
category = st.sidebar.selectbox("Category", categories)

cities = ['All'] + sorted(df['city'].unique().tolist())
city_select = st.sidebar.multiselect("Cities (multi)", ['All'], options=cities) if False else st.sidebar.multiselect("Cities (optional)", options=sorted(df['city'].unique()), default=sorted(df['city'].unique()))

# filter
mask = (df['order_datetime'] >= start_dt) & (df['order_datetime'] <= end_dt)
if category != 'All':
    mask &= (df['category'] == category)
if city_select and len(city_select) > 0:
    mask &= df['city'].isin(city_select)

df_filt = df.loc[mask].copy()
if df_filt.empty:
    st.warning("No data in selected filters. Try expanding the date range or removing other filters.")
    st.stop()

# -------------------------
# KPIs
# -------------------------
def compute_kpis(df):
    total_revenue = df['revenue'].sum()
    total_orders = df['order_id'].nunique()
    total_customers = df['customer_id'].nunique()
    aov = total_revenue / total_orders if total_orders else 0
    avg_order_frequency = df.groupby('customer_id')['order_id'].nunique().mean()
    return {
        'Total Revenue': total_revenue,
        'Total Orders': total_orders,
        'Total Customers': total_customers,
        'Avg Order Value (AOV)': aov,
        'Avg Orders/Customer': avg_order_frequency
    }

kpis = compute_kpis(df_filt)
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns([1.8,1,1,1,1])
kpi1.metric("Total Revenue", f"₹ {kpis['Total Revenue']:,.0f}")
kpi2.metric("Total Orders", f"{kpis['Total Orders']:,}")
kpi3.metric("Total Customers", f"{kpis['Total Customers']:,}")
kpi4.metric("Avg Order Value (AOV)", f"₹ {kpis['Avg Order Value (AOV)']:.2f}")
kpi5.metric("Avg Orders / Customer", f"{kpis['Avg Orders/Customer']:.2f}")

st.markdown("---")

# -------------------------
# Time series: revenue by time (daily / weekly)
# -------------------------
st.header("Revenue Over Time")
ts_agg = st.selectbox("Aggregate by", ['D', 'W', 'M'], index=1)
if ts_agg == 'D':
    df_ts = df_filt.set_index('order_datetime').resample('D').sum()['revenue'].reset_index()
elif ts_agg == 'W':
    df_ts = df_filt.set_index('order_datetime').resample('W').sum()['revenue'].reset_index()
else:
    df_ts = df_filt.set_index('order_datetime').resample('M').sum()['revenue'].reset_index()

fig_ts = px.line(df_ts, x='order_datetime', y='revenue', title="Revenue time series", labels={'order_datetime':'Date', 'revenue':'Revenue (₹)'})
fig_ts.add_trace(go.Bar(x=df_ts['order_datetime'], y=df_ts['revenue'], name='Revenue (bar)', opacity=0.2, marker={'color':'lightgray'}))
st.plotly_chart(fig_ts, use_container_width=True)

# -------------------------
# Category & product breakdown
# -------------------------
st.header("Category Breakdown & Top Items")
left, right = st.columns([2,1])

with left:
    cat_df = df_filt.groupby('category').agg(revenue=('revenue','sum'), orders=('order_id','nunique')).reset_index().sort_values('revenue', ascending=False)
    fig_cat = px.pie(cat_df, values='revenue', names='category', title='Revenue share by category', hole=0.35)
    st.plotly_chart(fig_cat, use_container_width=True)

    top_products = df_filt.groupby(['category']).agg(total_revenue=('revenue','sum')).reset_index().sort_values('total_revenue', ascending=False)
    st.dataframe(cat_df.style.format({"revenue":"₹{:,.2f}", "orders":"{:,}"}), height=240)

with right:
    st.subheader("Top 10 Customers by Revenue")
    top_cust = df_filt.groupby('customer_id').agg(revenue=('revenue','sum'), orders=('order_id','nunique')).reset_index().sort_values('revenue', ascending=False).head(10)
    st.table(top_cust.style.format({"revenue":"₹{:,.2f}"}))

st.markdown("---")

# -------------------------
# Cohort analysis (monthly cohorts, retention)
# -------------------------
st.header("Cohort Retention (monthly)")
# prepare
cohort = df[['customer_id','order_datetime']].copy()
cohort['order_month'] = cohort['order_datetime'].dt.to_period('M').dt.to_timestamp()
first_purchase = cohort.groupby('customer_id')['order_month'].min().reset_index().rename(columns={'order_month':'cohort_month'})
cohort = cohort.merge(first_purchase, on='customer_id')
cohort['period_number'] = ((cohort['order_month'].dt.year - cohort['cohort_month'].dt.year) * 12 + (cohort['order_month'].dt.month - cohort['cohort_month'].dt.month)).astype(int)

cohort_counts = cohort.groupby(['cohort_month','period_number'])['customer_id'].nunique().reset_index()
cohort_pivot = cohort_counts.pivot_table(index='cohort_month', columns='period_number', values='customer_id')
cohort_sizes = cohort_pivot.iloc[:,0]
retention = cohort_pivot.divide(cohort_sizes, axis=0).fillna(0)

# Optionally restrict displayed cohorts to the selected date window (so it reflects filtered timeframe)
retain_display = retention.copy().iloc[-12:]  # last 12 cohorts
fig_cohort = px.imshow(retain_display, text_auto='.0%', labels=dict(x="Months since cohort", y="Cohort month", color="Retention"), aspect="auto", title="Customer retention by cohort (last 12 cohorts)")
st.plotly_chart(fig_cohort, use_container_width=True)

st.markdown("---")

# -------------------------
# RFM segmentation
# -------------------------
st.header("RFM Segmentation (customers)")
snapshot_date = df['order_datetime'].max() + pd.Timedelta(days=1)
rfm = df.groupby('customer_id').agg(
    recency_days=('order_datetime', lambda x: (snapshot_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('revenue', 'sum')
).reset_index()

# convert to scores (1-5)
rfm['r_score'] = pd.qcut(rfm['recency_days'], 5, labels=[5,4,3,2,1]).astype(int)  # recent -> higher score
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5]).astype(int)
rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
# quick segment labels
def rfm_segment(row):
    if row['r_score'] >=4 and row['f_score']>=4 and row['m_score']>=4:
        return 'Champions'
    if row['r_score']>=3 and row['f_score']>=3:
        return 'Loyal'
    if row['r_score']<=2 and row['f_score']>=4:
        return 'At Risk (freq)'
    if row['r_score']<=2:
        return 'Requires Winback'
    return 'Others'

rfm['segment'] = rfm.apply(rfm_segment, axis=1)
seg_counts = rfm['segment'].value_counts().reset_index().rename(columns={'index':'segment','segment':'count'})
fig_rfm = px.bar(seg_counts, x='segment', y='count', title='RFM customer segments', text_auto=True)
st.plotly_chart(fig_rfm, use_container_width=True)

st.dataframe(rfm.sort_values('monetary', ascending=False).head(15).style.format({"monetary":"₹{:,.2f}", "recency_days":"{:,}", "frequency":"{:,}"}))

st.markdown("---")

# -------------------------
# Geo: Sales by city (scatter on map)
# -------------------------
st.header("Geographic Sales (by city)")
city_geo = df_filt.groupby(['city','lat','lon']).agg(revenue=('revenue','sum'), orders=('order_id','nunique')).reset_index()
fig_map = px.scatter_mapbox(city_geo, lat="lat", lon="lon", size="revenue", hover_name="city", hover_data={'revenue':':.0f','orders':':.0f'}, zoom=3, height=400)
fig_map.update_layout(mapbox_style="open-street-map")
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# -------------------------
# Export / notes / storytelling
# -------------------------
st.header("Export & Storytelling")
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download filtered transactions (CSV)", data=df_filt.to_csv(index=False), file_name="transactions_filtered.csv")
with col2:
    st.write("Suggested short story for stakeholders:")
    st.info("""During the selected period, we generated **₹ {:,.0f}** in revenue from **{:,}** customers.
Focus: champions and loyal segments—top 20% of customers deliver > X% revenue. Recommend targeted winback campaigns for 'Requires Winback' and A/B test a loyalty discount for 'Loyal' segment to increase frequency.""".format(kpis['Total Revenue'], kpis['Total Customers']))

st.markdown("**Pro tips:** Use real product SKU hierarchy, integrate with shipping/returns, add attribution channels (UTM) and microsurveys for guaranteed lift in insights.")
