import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Advanced Sales Analytics", layout="wide")

st.title("📊 Advanced Data-Driven Insight & Prediction System")

# Upload file
file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if file:
    # ---------------------------
    # LOAD DATA (optimized)
    # ---------------------------
    data = pd.read_csv(file, low_memory=False)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(data.head())

    # ---------------------------
    # AUTOMATED DATA CLEANING
    # ---------------------------
    st.subheader("🧹 Data Cleaning Process")

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Strip column names
    data.columns = data.columns.str.strip()

    # Fill missing values
    for col in data.select_dtypes(include=np.number).columns:
        data[col].fillna(data[col].median(), inplace=True)

    for col in data.select_dtypes(include='object').columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    st.success("Data cleaned successfully ✅")

    # Convert date
    if "Order Date" in data.columns:
        data["Order Date"] = pd.to_datetime(data["Order Date"], errors='coerce')
        data.dropna(subset=["Order Date"], inplace=True)

    # ---------------------------
    # KPI DASHBOARD
    # ---------------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{data['Sales'].sum():,.2f}")
    col2.metric("Total Profit", f"{data['Profit'].sum():,.2f}")
    col3.metric("Total Orders", len(data))

    # ---------------------------
    # FILTERS (Interactive)
    # ---------------------------
    st.sidebar.header("🔍 Filters")

    if "Category" in data.columns:
        category = st.sidebar.multiselect("Select Category", data["Category"].unique(), default=data["Category"].unique())
        data = data[data["Category"].isin(category)]

    # ---------------------------
    # ADVANCED VISUALS
    # ---------------------------

    st.subheader("📈 Monthly Sales Trend")

    if "Order Date" in data.columns:
        monthly = data.groupby(data["Order Date"].dt.to_period("M"))["Sales"].sum()

        fig, ax = plt.subplots()
        monthly.plot(ax=ax, marker="o")
        st.pyplot(fig)

    st.subheader("🏆 Top Products")

    if "Product Name" in data.columns:
        top_products = data.groupby("Product Name")["Sales"].sum().nlargest(10)

        fig, ax = plt.subplots()
        top_products.plot(kind="barh", ax=ax)
        st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots()
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Sales vs Profit")

    fig, ax = plt.subplots()
    sns.scatterplot(x="Sales", y="Profit", data=data, ax=ax)
    st.pyplot(fig)

    st.subheader("📉 Sales Distribution")

    fig, ax = plt.subplots()
    sns.histplot(data["Sales"], bins=50, ax=ax)
    st.pyplot(fig)

    # ---------------------------
    # INSIGHTS (AUTO)
    # ---------------------------
    st.subheader("🧠 Automated Insights")

    best_product = data.groupby("Product Name")["Sales"].sum().idxmax()
    best_month = data.groupby(data["Order Date"].dt.month)["Sales"].sum().idxmax()

    st.write(f"✔ Best selling product: **{best_product}**")
    st.write(f"✔ Highest sales month: **Month {best_month}**")

    # ---------------------------
    # PREDICTION SYSTEM
    # ---------------------------
    st.subheader("🔮 Predict Next Month Sales")

    if "Product Name" in data.columns and "Order Date" in data.columns:
        product = st.selectbox("Select Product", data["Product Name"].unique())

        product_data = data[data["Product Name"] == product]

        monthly_sales = product_data.groupby(product_data["Order Date"].dt.to_period("M"))["Sales"].sum()
        monthly_sales = monthly_sales.reset_index()

        monthly_sales["MonthIndex"] = np.arange(len(monthly_sales))

        if len(monthly_sales) > 2:
            X = monthly_sales[["MonthIndex"]]
            y = monthly_sales["Sales"]

            model = LinearRegression()
            model.fit(X, y)

            next_month = np.array([[len(monthly_sales)]])
            prediction = model.predict(next_month)

            st.success(f"📊 Predicted Next Month Sales: {prediction[0]:.2f}")

            # Plot prediction
            fig, ax = plt.subplots()
            ax.plot(monthly_sales["MonthIndex"], y, marker='o', label="Actual")
            ax.scatter(len(monthly_sales), prediction, label="Prediction")
            ax.legend()
            st.pyplot(fig)

        else:
            st.warning("Not enough data for prediction")
