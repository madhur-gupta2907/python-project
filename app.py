import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("📊 Superstore Project Website")

file = st.file_uploader("Upload CSV file")

if file:
    data = pd.read_csv(file)

    st.write(data.head())

    data.columns = data.columns.str.strip()

    st.subheader("Top Products")
    top_products = data.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots()
    top_products.plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("Prediction")

    X = data[["Sales"]]
    y = data["Profit"]

    model = LinearRegression()
    model.fit(X, y)

    val = st.number_input("Enter Sales Value")

    if st.button("Predict"):
        pred = model.predict([[val]])
        st.success(pred[0])