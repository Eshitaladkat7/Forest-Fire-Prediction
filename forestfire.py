import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('forestfires.csv')

#Data Preprocessing
# Convert categorical data into numeric
data['month'] = data['month'].astype('category').cat.codes
data['day'] = data['day'].astype('category').cat.codes

# Features and target variable
X = data.drop(columns=['area'])  # Features
y = data['area']  # Target (fire area)

# Log-transform the target variable to handle skewness
y = np.log1p(y)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Performance evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualizing predictions vs actual
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual values (log-transformed)")
plt.ylabel("Predicted values (log-transformed)")
plt.title("Actual vs Predicted")
plt.show()

# Feature importance
importance = model.feature_importances_
feature_names = X.columns
plt.barh(feature_names, importance)
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Predicting Fire Area")
plt.show()
