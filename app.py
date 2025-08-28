# uber_case_dashboard.py
# Simple Uber (NCR) case-study dashboard
# Minimal, readable, Uber-specific tabs (Overview, Rides, Cancellations, Revenue & Payments, Drivers/Customers)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.colors as mcolors

st.set_page_config(page_title="Uber (NCR) Case Dashboard", layout="wide")
sns.set_style("whitegrid")

# -------------------------
# Helpers: reading & preprocessing (kept minimal)
# -------------------------
@st.cache_data #caches the returned data stores and when on purpose gives the result if the alr existed called 
def read_csv_bytes(uploaded_bytes):
    # robust reading that accepts bytes from uploader
    try:
        return pd.read_csv(io.BytesIO(uploaded_bytes), low_memory=False)#low_memory=false meaning that the pandas read entire file into memory at once so it can determine the datatypes for each col more accurate
    except Exception:
        return pd.read_csv(io.BytesIO(uploaded_bytes), encoding="latin-1", low_memory=False)#latin-1 means the char encoding used in the writing or reading in the context of file operations

def preprocess(df):
    """Clean and create minimal features used across tabs."""
    df = df.copy()

    # trim column names
    df.columns = [c.strip() for c in df.columns]

    # Normalise expected column names (if lower-case present)
    # We'll handle both 'Date' and 'date' etc.
    if "date" in df.columns and "Date" not in df.columns:
        df.rename(columns={"date": "Date"}, inplace=True)
    if "time" in df.columns and "Time" not in df.columns:
        df.rename(columns={"time": "Time"}, inplace=True)
    if "booking value" in df.columns and "Booking Value" not in df.columns:
        df.rename(columns={"booking value": "Booking Value"}, inplace=True)

    # Parse date/time (safe)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Time" in df.columns:
        t = pd.to_datetime(df["Time"], errors="coerce")
        if not t.isna().all():
            df["Hour"] = t.dt.hour
        else:
            # Try extract hour from text like "14:30"
            try:
                df["Hour"] = df["Time"].astype(str).str.extract(r'(\d{1,2})(?::\d{2})?')[0].astype(float).astype("Int64")
            except Exception:
                df["Hour"] = pd.NA

    # Numeric casts
    for col in ["Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating",
                "Cancelled Rides by Customer", "Cancelled Rides by Driver", "Incomplete Rides"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill simple medians for numeric columns where appropriate (keeps label simple)
    for col in ["Driver Ratings", "Customer Rating", "Booking Value", "Ride Distance"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    if "Date" in df.columns:
        df["Day"] = df["Date"].dt.date
        df["DayOfWeek"] = df["Date"].dt.day_name()
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
    if ("Pickup Location" in df.columns) and ("Drop Location" in df.columns):
        df["Route"] = df["Pickup Location"].fillna("Unknown") + " → " + df["Drop Location"].fillna("Unknown")

    # Normalize Booking Status
    if "Booking Status" in df.columns:
        df["Booking Status"] = df["Booking Status"].astype(str).str.title().replace({"Canceled": "Cancelled"})
        df["Is_Completed"] = (df["Booking Status"] == "Completed")
        df["Is_Cancelled"] = (df["Booking Status"] == "Cancelled")
        df["Is_Incomplete"] = (df["Booking Status"] == "Incomplete")
    return df

# -------------------------
# Plotting helpers (simple & consistent)
# -------------------------
def hex_colors_from_cmap(cmap_name, n):
    cmap = plt.cm.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(i / max(1, n-1))) for i in range(n)]

def plot_countbar(series, title, rotation=0, ax=None):
    series = series.dropna()
    if series.empty:
        st.info("No data to plot.")
        return
    top = series.value_counts().head(20)
    colors = hex_colors_from_cmap("Paired", len(top))
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,3))
    top.plot(kind="bar", ax=ax, color=colors)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", rotation=rotation, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    if fig:
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
st.title("🚕 Uber (NCR) — Case Study Dashboard")
st.markdown("Upload the `ncr_uber_ridebooking.csv` file (or similar Uber-format CSV). Dashboard is Uber-focused — good for interviews and portfolio.")

uploaded = st.file_uploader("Upload Uber CSV", type=["csv"])
if uploaded is None:
    st.info("Upload your CSV to start. Use the same column names as the dataset (e.g., Date, Time, Booking ID, Booking Status, Booking Value, Ride Distance, Vehicle Type, Pickup Location, Drop Location).")
    st.stop()

# read and preprocess
try:
    raw = read_csv_bytes(uploaded.read())
except Exception as e:
    st.error("Could not read uploaded file: " + str(e))
    st.stop()

df = preprocess(raw)

# -------------------------
# Tabs: Overview, Rides, Cancellations, Revenue & Payments, Drivers & Customers
# -------------------------
tab_overview, tab_rides, tab_cancel, tab_revenue, tab_people = st.tabs(
    ["Overview", "Rides Overview", "Cancellations", "Revenue & Payments", "Drivers & Customers"]
)

# -------------------------
# Overview Tab
# -------------------------
with tab_overview:
    st.header("Overview")
    st.write("Quick dataset info and top-level KPIs.")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.metric("Rows", f"{df.shape[0]:,}")
        st.metric("Columns", f"{df.shape[1]}")
    with c2:
        if "Customer ID" in df.columns:
            st.metric("Unique Customers", f"{df['Customer ID'].nunique():,}")
        else:
            st.metric("Unique Customers", "N/A")
        if "Booking ID" in df.columns:
            st.metric("Unique Bookings", f"{df['Booking ID'].nunique():,}")
        else:
            st.metric("Unique Bookings", "N/A")
    with c3:
        if "Booking Value" in df.columns:
            st.metric("Total Revenue", f"₹{df['Booking Value'].sum():,.0f}")
            st.metric("Avg Booking Value", f"₹{df['Booking Value'].mean():.2f}")
        else:
            st.metric("Total Revenue", "N/A")
            st.metric("Avg Booking Value", "N/A")

    st.markdown("**Date range & sample preview**")
    if "Date" in df.columns:
        st.write(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    st.dataframe(df.head(6))

    st.subheader("Missing values (top columns)")
    mv = (df.isna().sum()).sort_values(ascending=False).head(10)
    st.table(mv.rename_axis("column").reset_index(name="missing_count"))

# -------------------------
# Rides Overview Tab
# -------------------------
with tab_rides:
    st.header("Rides Overview")
    # Booking status distribution
    if "Booking Status" in df.columns:
        st.subheader("Booking Status Distribution")
        plot_countbar(df["Booking Status"], "Booking Status", rotation=20)
    else:
        st.info("No Booking Status column found.")

    # Daily rides trend
    if "Date" in df.columns:
        daily = df.groupby(df["Date"].dt.date).size().rename("count")
        st.subheader("Daily Rides")
        plot_line_dates(daily.index, daily.values, "Daily Rides", xlabel="Date", ylabel="Rides")
    else:
        st.info("No Date column found — cannot show time trends.")

    # Hourly demand if Hour exists
    if "Hour" in df.columns:
        st.subheader("Hourly demand (0–23)")
        hourly = df.groupby("Hour").size().reindex(range(24), fill_value=0)
        fig, ax = plt.subplots(figsize=(9,2.5))
        ax.plot(hourly.index, hourly.values, marker='o', linewidth=1.6, color="#2ca02c")
        ax.set_title("Bookings by Hour")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Bookings")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

# -------------------------
# Cancellations Tab
# -------------------------
with tab_cancel:
    st.header("Cancellations Analysis")

    if "Is_Cancelled" in df.columns:
        rate = df["Is_Cancelled"].mean() * 100
        st.metric("Overall cancellation rate", f"{rate:.2f}%")
    else:
        st.info("No Booking Status -> cannot compute cancellation rate.")

    # Customer-initiated cancellations column might be counts or flags; handle strings too
    if "Cancelled Rides by Customer" in df.columns:
        col = df["Cancelled Rides by Customer"].astype(str).str.strip().str.lower()
        # treat as positive cancellation if not "0" or not "nan"
        is_cancelled_cust = (~col.isin(["0", "0.0", "nan", "none", "", "false"])) & col.notna()
        st.write("Customer-initiated cancellation (% of rows):", f"{is_cancelled_cust.mean()*100:.2f}%")
    if "Cancelled Rides by Driver" in df.columns:
        col2 = df["Cancelled Rides by Driver"].astype(str).str.strip().str.lower()
        is_cancelled_drv = (~col2.isin(["0", "0.0", "nan", "none", "", "false"])) & col2.notna()
        st.write("Driver-initiated cancellation (% of rows):", f"{is_cancelled_drv.mean()*100:.2f}%")

    # cancellation reasons (top)
    if "Reason for cancelling by Customer" in df.columns:
        st.subheader("Top customer cancellation reasons")
        plot_countbar(df["Reason for cancelling by Customer"], "Customer cancellation reasons", rotation=25)
    if "Driver Cancellation Reason" in df.columns:
        st.subheader("Top driver cancellation reasons")
        plot_countbar(df["Driver Cancellation Reason"], "Driver cancellation reasons", rotation=25)

# -------------------------
# Revenue & Payments Tab
# -------------------------
with tab_revenue:
    st.header("Revenue & Payment Insights")
    if "Booking Value" in df.columns:
        st.subheader("Booking value distribution")
        plot_hist(df["Booking Value"], "Booking Value distribution", bins=30)

        st.subheader("Revenue trend (daily)")
        if "Date" in df.columns:
            rev_daily = df.groupby(df["Date"].dt.date)["Booking Value"].sum()
            plot_line_dates(rev_daily.index, rev_daily.values, "Daily Revenue", xlabel="Date", ylabel="Revenue (₹)")
    else:
        st.info("No Booking Value column present.")

    if "Payment Method" in df.columns:
        st.subheader("Payment Method mix")
        plot_countbar(df["Payment Method"], "Payment Method", rotation=25)

# -------------------------
# Drivers & Customers Tab
# -------------------------
with tab_people:
    st.header("Top Drivers & Customers (Activity)")
    # top drivers
    if "Driver Ratings" in df.columns and "Driver Ratings" in df.columns:
        pass  # keep simple; show top counts instead

    if "Customer ID" in df.columns:
        st.subheader("Top 10 Customers by number of rides")
        top_cust = df["Customer ID"].value_counts().head(10)
        st.table(top_cust.rename_axis("Customer ID").reset_index(name="ride_count"))

    if "Driver ID" in df.columns:
        st.subheader("Top 10 Drivers by number of rides")
        top_drv = df["Driver ID"].value_counts().head(10)
        st.table(top_drv.rename_axis("Driver ID").reset_index(name="ride_count"))

    # distances and value per km
    if "Ride Distance" in df.columns:
        st.subheader("Avg ride distance")
        st.write(f"{df['Ride Distance'].mean():.2f} km (mean) — {df['Ride Distance'].median():.2f} km (median)")
    if "Value_per_km" in df.columns:
        st.subheader("Value per km (median by vehicle)")
        if "Vehicle Type" in df.columns:
            vpkm = df.groupby("Vehicle Type")["Value_per_km"].median().sort_values(ascending=False)
            st.table(vpkm.rename_axis("Vehicle Type").reset_index(name="median_value_per_km"))

# -------------------------
# Export cleaned dataset (download)
# -------------------------
st.markdown("---")
buf = io.BytesIO()
df.to_csv(buf, index=False)
buf.seek(0)
st.download_button("Download cleaned / filtered CSV", data=buf, file_name="cleaned_ncr_uber.csv", mime="text/csv")
