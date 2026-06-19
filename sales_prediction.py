import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
import joblib

# Load dataset
data = pd.read_csv("sales.csv")

print("First 5 rows")
print(data.head())

print("\nDataset Information")
print(data.info())

print("\nChecking null values")
print(data.isnull().sum())

# Features and target
X = data[['TV','Radio','Newspaper']]
y = data['Sales']

# Split dataset
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model training
model = LinearRegression()

model.fit(X_train,y_train)

# Prediction
prediction=model.predict(X_test)

# Accuracy
print("\nR2 Score:")
print(r2_score(y_test,prediction))

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test,prediction))

# Save model
joblib.dump(model,"model.pkl")

print("\nModel saved successfully")

# Predict custom input
tv=float(input("TV Advertising Budget: "))
radio=float(input("Radio Advertising Budget: "))
news=float(input("Newspaper Advertising Budget: "))

new_prediction=model.predict([[tv,radio,news]])

print("Predicted Sales:",new_prediction[0])

# Graph
plt.scatter(y_test,prediction)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()
