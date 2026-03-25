import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Data Analytics System", layout="wide")

st.title("📊 Smart Data Analytics & Prediction Dashboard")

# -------------------------
# FILE UPLOAD
# -------------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:
    # -------------------------
    # LOAD DATA
    # -------------------------
    data = pd.read_csv(file, low_memory=False)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(data.head())

    # -------------------------
    # CLEANING
    # -------------------------
    data.drop_duplicates(inplace=True)
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

    # Fill missing values
    for col in data.select_dtypes(include=np.number).columns:
        data[col].fillna(data[col].median(), inplace=True)

    for col in data.select_dtypes(include='object').columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Convert numeric safely
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except:
            pass

    st.success("✅ Data cleaned automatically")

    # -------------------------
    # COLUMN DETECTION
    # -------------------------
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include='object').columns.tolist()

    # Detect date column
    date_col = None
    for col in data.columns:
        if "date" in col:
            date_col = col
            break

    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        data.dropna(subset=[date_col], inplace=True)

    # -------------------------
    # KPI DASHBOARD
    # -------------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    if "sales" in data.columns:
        col1.metric("Total Sales", f"{data['sales'].sum():,.2f}")
    else:
        col1.metric("Rows", len(data))

    if "profit" in data.columns:
        col2.metric("Total Profit", f"{data['profit'].sum():,.2f}")
    else:
        col2.metric("Columns", len(data.columns))

    col3.metric("Total Records", len(data))

    # -------------------------
    # SIDEBAR FILTER
    # -------------------------
    st.sidebar.header("🔍 Filter")

    if len(categorical_cols) > 0:
        filter_col = st.sidebar.selectbox("Select Column", categorical_cols)
        filter_val = st.sidebar.multiselect(
            "Select Values",
            data[filter_col].unique(),
            default=data[filter_col].unique()
        )
        data = data[data[filter_col].isin(filter_val)]

    # -------------------------
    # VISUALIZATION
    # -------------------------
    st.subheader("📊 Visualization")

    if len(numeric_cols) > 0:
        x_axis = st.selectbox("X-axis", data.columns)
        y_axis = st.selectbox("Y-axis (numeric)", numeric_cols)

        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])

        plot_data = data[[x_axis, y_axis]].dropna()

        # Reduce large categories
        if plot_data[x_axis].nunique() > 20:
            plot_data = plot_data.groupby(x_axis)[y_axis].mean().reset_index().head(20)

        fig, ax = plt.subplots()

        try:
            if chart_type == "Bar":
                sns.barplot(x=x_axis, y=y_axis, data=plot_data, ax=ax)

            elif chart_type == "Line":
                sns.lineplot(x=x_axis, y=y_axis, data=plot_data, ax=ax)

            elif chart_type == "Scatter":
                sns.scatterplot(x=x_axis, y=y_axis, data=plot_data, ax=ax)

            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Graph error: {e}")

    # -------------------------
    # DISTRIBUTION
    # -------------------------
    st.subheader("📉 Distribution")

    if len(numeric_cols) > 0:
        num_col = st.selectbox("Select Column", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(data[num_col].dropna(), bins=30, ax=ax)
        st.pyplot(fig)

    # -------------------------
    # HEATMAP
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
        st.write(f"✔ Highest value in {max_col}: {data[max_col].max()}")

    st.write("✔ Data cleaned and analyzed automatically")
    st.write("✔ Trends visualized using charts")

    # -------------------------
    # PREDICTION
    # -------------------------
    st.subheader("🔮 Prediction System")

    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Feature (X)", numeric_cols)
        y_col = st.selectbox("Target (Y)", numeric_cols, index=1)

        df_model = data[[x_col, y_col]].dropna()

        X = df_model[[x_col]]
        y = df_model[y_col]

        model = LinearRegression()
        model.fit(X, y)

        val = st.number_input(f"Enter value for {x_col}")

        if st.button("Predict"):
            pred = model.predict([[val]])
            st.success(f"Predicted {y_col}: {pred[0]:.2f}")

    else:
        st.warning("Not enough numeric columns for prediction")
