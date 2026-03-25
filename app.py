import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

plt.style.use("ggplot")

data = pd.read_csv("superstore.csv")
data.head()

print(data.columns)
data.info()

data.columns = data.columns.str.strip()

print(data.isnull().sum())

data = data.drop_duplicates()

data.describe()

top_products = data.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
top_products.plot(kind="bar")

plt.title("Top 10 Selling Products")
plt.xlabel("Product Name")
plt.ylabel("Total Sales")

plt.show()

plt.figure(figsize=(8,5))

sns.barplot(x="Category", y="Sales", data=data)

plt.title("Sales by Category")

plt.show()

plt.figure(figsize=(8,5))

sns.barplot(x="Region", y="Profit", data=data)

plt.title("Profit by Region")

plt.show()

data["Order Date"] = pd.to_datetime(data["Order Date"])

monthly_sales = data.groupby(data["Order Date"].dt.month)["Sales"].sum()

plt.figure(figsize=(8,5))

monthly_sales.plot(marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="Sales", y="Profit", data=data)
plt.title("Sales vs Profit")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(data["Sales"], bins=30)
plt.title("Sales Distribution")
plt.show()

top_regions = data.groupby("Region")["Profit"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
top_regions.plot(kind="bar")
plt.title("Profit by Region")
plt.ylabel("Profit")
plt.show()

X = data[["Sales"]]
y = data["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", error)

plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.show()

results = pd.DataFrame({
    "Actual Profit": y_test.values,
    "Predicted Profit": predictions
})
print("Actual vs Predicted Profit")
print(results.head(10))
