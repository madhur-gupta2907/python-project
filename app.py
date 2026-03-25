import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="Advanced Data Analytics", layout="wide")

st.title("📊 Data-Driven Insight Generation & Prediction System")

# Upload dataset
file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if file:
    # -------------------------
    # LOAD DATA
    # -------------------------
    data = pd.read_csv(file, low_memory=False)

    st.subheader("📄 Data Preview")
    st.dataframe(data.head())

    # -------------------------
    # DATA CLEANING (AUTO)
    # -------------------------
    data.drop_duplicates(inplace=True)
    data.columns = data.columns.str.strip()

    # Fill missing values
    for col in data.select_dtypes(include=np.number).columns:
        data[col].fillna(data[col].median(), inplace=True)

    for col in data.select_dtypes(include='object').columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    st.success("✅ Data cleaned successfully")

    # -------------------------
    # COLUMN DETECTION
    # -------------------------
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include='object').columns.tolist()

    # -------------------------
    # KPI METRICS
    # -------------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    if "Sales" in data.columns:
        col1.metric("Total Sales", f"{data['Sales'].sum():,.2f}")
    else:
        col1.metric("Rows", len(data))

    if "Profit" in data.columns:
        col2.metric("Total Profit", f"{data['Profit'].sum():,.2f}")
    else:
        col2.metric("Columns", len(data.columns))

    col3.metric("Total Records", len(data))

    # -------------------------
    # SIDEBAR FILTERS
    # -------------------------
    st.sidebar.header("🔍 Filters")

    if len(categorical_cols) > 0:
        selected_col = st.sidebar.selectbox("Filter Column", categorical_cols)
        selected_val = st.sidebar.multiselect(
            "Select Values",
            data[selected_col].unique(),
            default=data[selected_col].unique()
        )
        data = data[data[selected_col].isin(selected_val)]

    # -------------------------
    # VISUALIZATION SECTION
    # -------------------------
    st.subheader("📊 Visualization")

    if len(numeric_cols) > 0:
        x_axis = st.selectbox("Select X-axis", data.columns)
        y_axis = st.selectbox("Select Y-axis", numeric_cols)

        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])

        fig, ax = plt.subplots()

        if chart_type == "Bar":
            sns.barplot(x=x_axis, y=y_axis, data=data, ax=ax)

        elif chart_type == "Line":
            sns.lineplot(x=x_axis, y=y_axis, data=data, ax=ax)

        elif chart_type == "Scatter":
            sns.scatterplot(x=x_axis, y=y_axis, data=data, ax=ax)

        plt.xticks(rotation=45)
        st.pyplot(fig)

    # -------------------------
    # DISTRIBUTION
    # -------------------------
    st.subheader("📉 Distribution")

    if len(numeric_cols) > 0:
        num_col = st.selectbox("Select Column for Distribution", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(data[num_col], bins=30, ax=ax)
        st.pyplot(fig)

    # -------------------------
    # CORRELATION HEATMAP
    # -------------------------
    st.subheader("🔥 Correlation Heatmap")

    if len(numeric_cols) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # -------------------------
    # AUTO INSIGHTS
    # -------------------------
    st.subheader("🧠 Automated Insights")

    if len(numeric_cols) > 0:
        max_col = numeric_cols[0]
        max_value = data[max_col].max()
        st.write(f"✔ Highest {max_col}: {max_value}")

    st.write("✔ Data cleaned and processed automatically")
    st.write("✔ Visual patterns identified using charts")

    # -------------------------
    # PREDICTION SYSTEM
    # -------------------------
    st.subheader("🔮 Prediction (Linear Regression)")

    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Select Feature (X)", numeric_cols)
        y_col = st.selectbox("Select Target (Y)", numeric_cols, index=1)

        X = data[[x_col]]
        y = data[y_col]

        model = LinearRegression()
        model.fit(X, y)

        val = st.number_input(f"Enter value for {x_col}")

        if st.button("Predict"):
            pred = model.predict([[val]])
            st.success(f"Predicted {y_col}: {pred[0]:.2f}")

    else:
        st.warning("Not enough numeric columns for prediction")
