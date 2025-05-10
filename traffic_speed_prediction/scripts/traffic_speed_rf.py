import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Ensure output directory exists
os.makedirs('../outputs', exist_ok=True)

# Step 1: Load and preprocess data
data = pd.read_csv('../data/traffic_data.csv')

# Handle missing values by filling with median
data.fillna(data.median(), inplace=True)

# Cap outliers (traffic_speed > 99th percentile)
speed_threshold = data['traffic_speed'].quantile(0.99)
data['traffic_speed'] = data['traffic_speed'].clip(upper=speed_threshold)

# Step 2: Prepare features and target
X = data[['hour', 'day', 'weather', 'road_type']].values
y = data['traffic_speed'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Step 3: Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(rf, '../outputs/rf_model.pkl')
joblib.dump(scaler_X, '../outputs/scaler_X.pkl')

# Step 4: Make predictions
y_pred = rf.predict(X_test_scaled)

# Save predictions to CSV
predictions = pd.DataFrame({
    'actual_speed': y_test,
    'predicted_speed': y_pred
})
predictions.to_csv('../outputs/predictions.csv', index=False)

# Step 5: Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Step 6: Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test[:50])), y_test[:50], color='blue', label='Actual Traffic Speed')
plt.plot(range(len(y_pred[:50])), y_pred[:50], color='red', label='Predicted Traffic Speed')
plt.xlabel('Test Sample Index')
plt.ylabel('Traffic Speed (km/h)')
plt.title('Random Forest Prediction of Traffic Speed')
plt.legend()
plt.grid(True)
plt.savefig('../outputs/traffic_speed_prediction.png')
plt.close()
