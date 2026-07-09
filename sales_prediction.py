import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*60)
print("        SALES PREDICTION USING PYTHON")
print("="*60)

# Load dataset
data = pd.read_csv("sales.csv")

print("\nDataset Loaded Successfully")
print("Dataset Shape:", data.shape)

print("\nFirst 5 Rows")
print(data.head())

print("\nMissing Values")
print(data.isnull().sum())

# Features and Target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Model...")

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Trained Successfully")

# Prediction
y_pred = model.predict(X_test)

print("\nModel Performance")
print("R2 Score :", round(r2_score(y_test, y_pred), 3))
print("MAE :", round(mean_absolute_error(y_test, y_pred), 3))
print("MSE :", round(mean_squared_error(y_test, y_pred), 3))

# Save Model
joblib.dump(model, "sales_model.pkl")
print("\nModel Saved Successfully as sales_model.pkl")

# User Prediction
print("\nPredict New Sales")

tv = float(input("Enter TV Advertisement Budget: "))
radio = float(input("Enter Radio Advertisement Budget: "))
newspaper = float(input("Enter Newspaper Advertisement Budget: "))

prediction = model.predict([[tv, radio, newspaper]])

print("\nPredicted Sales =", round(prediction[0], 2))

# Graph
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

print("\nProject Completed Successfully")