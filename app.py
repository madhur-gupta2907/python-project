import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Premium Data Dashboard", layout="wide")

st.title("🚀 Advanced Data Analytics Dashboard")

# ---------------------------
# FILE UPLOAD
# ---------------------------
file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if file:
    data = pd.read_csv(file, low_memory=False)

    # ---------------------------
    # CLEANING
    # ---------------------------
    data.drop_duplicates(inplace=True)
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

    for col in data.select_dtypes(include=np.number):
        data[col].fillna(data[col].median(), inplace=True)

    for col in data.select_dtypes(include='object'):
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Convert numeric safely
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except:
            pass

    # Detect date column
    date_col = None
    for col in data.columns:
        if "date" in col:
            date_col = col
            break

    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        data.dropna(subset=[date_col], inplace=True)

    # Column types
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include='object').columns.tolist()

    st.success("✅ Data cleaned & processed")

    # ---------------------------
    # KPI DASHBOARD
    # ---------------------------
    st.subheader("📊 Key Metrics")

    c1, c2, c3 = st.columns(3)

    if "sales" in data.columns:
        c1.metric("Total Sales", f"{data['sales'].sum():,.0f}")
    else:
        c1.metric("Rows", len(data))

    if "profit" in data.columns:
        c2.metric("Total Profit", f"{data['profit'].sum():,.0f}")
    else:
        c2.metric("Columns", len(data.columns))

    c3.metric("Records", len(data))

    # ---------------------------
    # SIDEBAR FILTERS
    # ---------------------------
    st.sidebar.header("🔍 Smart Filters")

    if categorical_cols:
        filter_col = st.sidebar.selectbox("Select Filter Column", categorical_cols)
        filter_val = st.sidebar.multiselect(
            "Select Values",
            data[filter_col].unique(),
            default=data[filter_col].unique()
        )
        data = data[data[filter_col].isin(filter_val)]

    # ---------------------------
    # AUTO TIME ANALYSIS
    # ---------------------------
    if date_col:
        st.subheader("📈 Time Series Analysis")

        monthly = data.groupby(data[date_col].dt.to_period("M"))["sales"].sum()

        fig, ax = plt.subplots()
        monthly.plot(marker="o", ax=ax)
        st.pyplot(fig)

    # ---------------------------
    # TOP PRODUCTS
    # ---------------------------
    if "product" in data.columns:
        st.subheader("🏆 Top Products")

        top_products = data.groupby("product")["sales"].sum().nlargest(10)

        fig, ax = plt.subplots()
        top_products.plot(kind="barh", ax=ax)
        st.pyplot(fig)

    # ---------------------------
    # SMART VISUALIZATION
    # ---------------------------
    st.subheader("📊 Smart Visualization")

    x_axis = st.selectbox("Select X-axis", data.columns)
    y_axis = st.selectbox("Select Y-axis", numeric_cols)

    fig, ax = plt.subplots()

    try:
        if x_axis in categorical_cols:
            st.info("📊 Using BAR chart (best for categories)")
            data.groupby(x_axis)[y_axis].mean().head(20).plot(kind="bar", ax=ax)

        elif x_axis in numeric_cols:
            st.info("🔍 Using SCATTER chart (numeric comparison)")
            ax.scatter(data[x_axis], data[y_axis])

        elif "date" in x_axis:
            st.info("📈 Using LINE chart (time-based)")
            temp = data.groupby(data[x_axis])[y_axis].sum()
            temp.plot(ax=ax)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

    # ---------------------------
    # HEATMAP
    # ---------------------------
    st.subheader("🔥 Correlation Heatmap")

    if len(numeric_cols) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ---------------------------
    # AUTO INSIGHTS
    # ---------------------------
    st.subheader("🧠 Key Insights")

    if "sales" in data.columns and "product" in data.columns:
        best_product = data.groupby("product")["sales"].sum().idxmax()
        st.write(f"✔ Best selling product: **{best_product}**")

    if date_col:
        best_month = data.groupby(data[date_col].dt.month)["sales"].sum().idxmax()
        st.write(f"✔ Highest sales month: **Month {best_month}**")

    st.write("✔ Business performance trends identified")

    # ---------------------------
    # PREDICTION
    # ---------------------------
    st.subheader("🔮 Prediction System")

    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Feature", numeric_cols)
        y_col = st.selectbox("Target", numeric_cols, index=1)

        df_model = data[[x_col, y_col]].dropna()

        X = df_model[[x_col]]
        y = df_model[y_col]

        model = LinearRegression()
        model.fit(X, y)

        val = st.number_input(f"Enter {x_col}")

        if st.button("Predict"):
            pred = model.predict([[val]])
            st.success(f"Predicted {y_col}: {pred[0]:.2f}")

    # ---------------------------
    # FOOTER
    # ---------------------------
    st.markdown("---")
    st.markdown("✨ Developed for Data Analytics Project | Streamlit Dashboard")
