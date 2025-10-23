# uber_case_dashboard.py
# Streamlit Uber (NCR) Case Study Dashboard – Polished & Enhanced Version
# Shows clean EDA insights + metrics + visuals + downloadable dataset.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.colors as mcolors

st.set_page_config(page_title="Uber NCR Case Dashboard", layout="wide")
sns.set_style("whitegrid")

# -------------------------
# Helpers: reading & preprocessing
# -------------------------
@st.cache_data
def read_csv_bytes(uploaded_bytes):
    """Safely read uploaded CSV file."""
    try:
        return pd.read_csv(io.BytesIO(uploaded_bytes), low_memory=False)
    except Exception:
        return pd.read_csv(io.BytesIO(uploaded_bytes), encoding="latin-1", low_memory=False)

def preprocess(df):
    """Clean and create minimal features used across tabs."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Normalize column names
    renames = {"date": "Date", "time": "Time", "booking value": "Booking Value"}
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Parse datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Time" in df.columns:
        t = pd.to_datetime(df["Time"], errors="coerce")
        if not t.isna().all():
            df["Hour"] = t.dt.hour
        else:
            df["Hour"] = df["Time"].astype(str).str.extract(r'(\d{1,2})(?::\d{2})?')[0].astype(float)

    # Convert numeric
    num_cols = [
        "Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating",
        "Cancelled Rides by Customer", "Cancelled Rides by Driver", "Incomplete Rides"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    if "Date" in df.columns:
        df["Day"] = df["Date"].dt.date
        df["DayOfWeek"] = df["Date"].dt.day_name()
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
    if {"Pickup Location", "Drop Location"} <= set(df.columns):
        df["Route"] = df["Pickup Location"].fillna("Unknown") + " → " + df["Drop Location"].fillna("Unknown")

    # Normalize Booking Status
    if "Booking Status" in df.columns:
        df["Booking Status"] = df["Booking Status"].astype(str).str.title().replace({"Canceled": "Cancelled"})
        df["Is_Completed"] = df["Booking Status"] == "Completed"
        df["Is_Cancelled"] = df["Booking Status"] == "Cancelled"
        df["Is_Incomplete"] = df["Booking Status"] == "Incomplete"

    return df

# -------------------------
# Plotting helpers
# -------------------------
def hex_colors_from_cmap(cmap_name, n):
    cmap = plt.cm.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(i / max(1, n-1))) for i in range(n)]

def plot_countbar(series, title, rotation=0):
    series = series.dropna()
    if series.empty:
        st.info("No data to plot.")
        return
    top = series.value_counts().head(20)
    colors = hex_colors_from_cmap("Paired", len(top))
    fig, ax = plt.subplots(figsize=(6,3))
    top.plot(kind="bar", ax=ax, color=colors)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", rotation=rotation, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)

def plot_line_dates(x, y, title, xlabel="", ylabel=""):
    if len(x)==0 or len(y)==0:
        st.info("No data to plot.")
        return
    fig, ax = plt.subplots(figsize=(9,3))
    ax.plot(x, y, linewidth=1.8, marker="o", markersize=3, color="#1f77b4")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    st.pyplot(fig)

def plot_hist(series, title, bins=20):
    series = series.dropna()
    if series.empty:
        st.info("No data to plot.")
        return
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(series, bins=bins, color="skyblue", edgecolor="white", alpha=0.8)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)

# -------------------------
# App UI
# -------------------------
st.title("🚕 Uber (NCR) – Case Study Dashboard")
st.markdown("Upload the `ncr_uber_ridebooking.csv` file (or similar Uber-format CSV).")

uploaded = st.file_uploader("Upload Uber CSV", type=["csv"])
if uploaded is None:
    st.info("Upload your CSV to start. Ensure columns like Date, Time, Booking Status, Booking Value, Ride Distance, etc.")
    st.stop()

# read & preprocess
try:
    raw = read_csv_bytes(uploaded.read())
    df = preprocess(raw)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# -------------------------
# Tabs
# -------------------------
tab_overview, tab_rides, tab_cancel, tab_revenue, tab_people = st.tabs(
    ["Overview", "Ride Demand & Trends", "Cancellations", "Revenue & Payments", "Engagement Analysis"]
)

# -------------------------
# Overview Tab
# -------------------------
with tab_overview:
    st.header("Overview Summary")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.metric("Rows", f"{df.shape[0]:,}")
        st.metric("Columns", f"{df.shape[1]}")
    with c2:
        st.metric("Unique Customers", f"{df['Customer ID'].nunique():,}" if "Customer ID" in df else "N/A")
        st.metric("Unique Bookings", f"{df['Booking ID'].nunique():,}" if "Booking ID" in df else "N/A")
    with c3:
        if "Booking Value" in df.columns:
            st.metric("Total Revenue", f"₹{df['Booking Value'].sum():,.0f}")
            st.metric("Avg Booking Value", f"₹{df['Booking Value'].mean():.2f}")
        else:
            st.metric("Total Revenue", "N/A")
            st.metric("Avg Booking Value", "N/A")

    # Key Insights Section
    st.subheader("📊 Key Insights")
    insights = []
    if "Booking ID" in df.columns:
        insights.append(f"**{df['Booking ID'].nunique():,} total bookings** analyzed.")
    if "Booking Value" in df.columns:
        insights.append(f"**₹{df['Booking Value'].sum():,.0f} total revenue** generated.")
    if "Is_Cancelled" in df.columns:
        insights.append(f"**{df['Is_Cancelled'].mean()*100:.2f}% overall cancellation rate.**")
    if "Hour" in df.columns:
        hourly = df.groupby("Hour").size().reindex(range(24), fill_value=0)
        top_hours = hourly.sort_values(ascending=False).head(3).index.tolist()
        insights.append(f"Peak booking hours: **{', '.join(map(str, top_hours))} hrs**.")
    st.markdown("- " + "\n- ".join(insights))

    st.markdown("### Dataset Snapshot")
    if "Date" in df.columns:
        st.write(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    st.dataframe(df.head(6))

    st.subheader("Missing Values (Top 10 Columns)")
    mv = df.isna().sum().sort_values(ascending=False).head(10)
    st.table(mv.rename_axis("column").reset_index(name="missing_count"))

# -------------------------
# Ride Demand & Trends
# -------------------------
with tab_rides:
    st.header("Ride Demand & Time Trends")
    if "Booking Status" in df.columns:
        st.subheader("Booking Status Distribution")
        plot_countbar(df["Booking Status"], "Booking Status", rotation=20)

    if "Date" in df.columns:
        daily = df.groupby(df["Date"].dt.date).size()
        st.subheader("Daily Rides Trend")
        plot_line_dates(daily.index, daily.values, "Daily Rides", "Date", "Rides")

    if "Hour" in df.columns:
        st.subheader("Hourly Demand Pattern")
        hourly = df.groupby("Hour").size().reindex(range(24), fill_value=0)
        fig, ax = plt.subplots(figsize=(9,2.5))
        ax.plot(hourly.index, hourly.values, marker='o', linewidth=1.6, color="#2ca02c")
        ax.set_title("Bookings by Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Bookings")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)
        top3 = hourly.sort_values(ascending=False).head(3)
        st.markdown(f"**Peak Hours:** {', '.join(map(str, top3.index))}")

# -------------------------
# Cancellations
# -------------------------
with tab_cancel:
    st.header("Cancellations Analysis")
    if "Is_Cancelled" in df.columns:
        st.metric("Overall Cancellation Rate", f"{df['Is_Cancelled'].mean()*100:.2f}%")

    if "Cancelled Rides by Customer" in df.columns:
        st.subheader("Customer-Initiated Cancellations")
        col = df["Cancelled Rides by Customer"].astype(str).str.strip().str.lower()
        cancelled_cust = (~col.isin(["0", "0.0", "nan", "none", "", "false"])) & col.notna()
        st.write(f"Customer cancellations: **{cancelled_cust.mean()*100:.2f}%**")

    if "Cancelled Rides by Driver" in df.columns:
        st.subheader("Driver-Initiated Cancellations")
        col2 = df["Cancelled Rides by Driver"].astype(str).str.strip().str.lower()
        cancelled_drv = (~col2.isin(["0", "0.0", "nan", "none", "", "false"])) & col2.notna()
        st.write(f"Driver cancellations: **{cancelled_drv.mean()*100:.2f}%**")

    if "Reason for cancelling by Customer" in df.columns:
        st.subheader("Top Customer Cancellation Reasons")
        plot_countbar(df["Reason for cancelling by Customer"], "Customer cancellation reasons", rotation=25)

    if "Driver Cancellation Reason" in df.columns:
        st.subheader("Top Driver Cancellation Reasons")
        plot_countbar(df["Driver Cancellation Reason"], "Driver cancellation reasons", rotation=25)

# -------------------------
# Revenue & Payments
# -------------------------
with tab_revenue:
    st.header("Revenue & Payment Insights")
    if "Booking Value" in df.columns:
        st.subheader("Booking Value Distribution")
        plot_hist(df["Booking Value"], "Booking Value distribution", bins=30)

        st.subheader("Daily Revenue Trend")
        if "Date" in df.columns:
            rev_daily = df.groupby(df["Date"].dt.date)["Booking Value"].sum()
            plot_line_dates(rev_daily.index, rev_daily.values, "Daily Revenue", "Date", "Revenue (₹)")

    if "Payment Method" in df.columns:
        st.subheader("Payment Method Mix")
        plot_countbar(df["Payment Method"], "Payment Method", rotation=25)

# -------------------------
# Engagement Analysis (Drivers & Customers)
# -------------------------
with tab_people:
    st.header("Engagement Analysis – Drivers & Customers")

    if "Customer ID" in df.columns:
        st.subheader("Top 10 Customers by Ride Count")
        top_cust = df["Customer ID"].value_counts().head(10)
        st.table(top_cust.rename_axis("Customer ID").reset_index(name="ride_count"))

    if "Driver ID" in df.columns:
        st.subheader("Top 10 Drivers by Ride Count")
        top_drv = df["Driver ID"].value_counts().head(10)
        st.table(top_drv.rename_axis("Driver ID").reset_index(name="ride_count"))

    if "Ride Distance" in df.columns:
        st.subheader("Average Ride Distance")
        st.write(f"Mean: {df['Ride Distance'].mean():.2f} km | Median: {df['Ride Distance'].median():.2f} km")

    if st.checkbox("Show Correlation Heatmap (numerical columns)"):
        num_cols = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(num_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# -------------------------
# Export cleaned dataset
# -------------------------
st.markdown("---")
buf = io.BytesIO()
df.to_csv(buf, index=False)
buf.seek(0)
st.download_button("⬇️ Download Cleaned Dataset", data=buf, file_name="cleaned_ncr_uber.csv", mime="text/csv")
