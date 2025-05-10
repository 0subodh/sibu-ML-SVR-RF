import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Ensure output directory exists
os.makedirs('../outputs', exist_ok=True)

# Step 1: Load and preprocess data
# Load traffic data from CSV (simulating real-world traffic sensor data)
data = pd.read_csv('../data/traffic_data.csv')

# Handle missing values by filling with median (common in real datasets)
data.fillna(data.median(), inplace=True)

# Detect and cap outliers (e.g., traffic volume > 99th percentile)
volume_threshold = data['traffic_volume'].quantile(0.99)
data['traffic_volume'] = data['traffic_volume'].clip(upper=volume_threshold)

# Step 2: Prepare features and target
# Features: hour, day, weather (encoded as numeric); Target: traffic_volume
X = data[['hour', 'day', 'weather']].values
y = data['traffic_volume'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features and target
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Step 3: Train SVR model
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train_scaled, y_train_scaled)

# Save the model and scalers for future use
joblib.dump(svr, '../outputs/svr_model.pkl')
joblib.dump(scaler_X, '../outputs/scaler_X.pkl')
joblib.dump(scaler_y, '../outputs/scaler_y.pkl')

# Step 4: Make predictions
y_pred_scaled = svr.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Save predictions to CSV
predictions = pd.DataFrame({
    'actual_volume': y_test,
    'predicted_volume': y_pred
})
predictions.to_csv('../outputs/predictions.csv', index=False)

# Step 5: Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Step 6: Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test[:50])), y_test[:50], color='blue', label='Actual Traffic Volume')
plt.plot(range(len(y_pred[:50])), y_pred[:50], color='red', label='Predicted Traffic Volume')
plt.xlabel('Test Sample Index')
plt.ylabel('Traffic Volume (vehicles/hour)')
plt.title('SVR Prediction of Traffic Volume')
plt.legend()
plt.grid(True)
plt.savefig('../outputs/traffic_flow_prediction.png')
plt.close()
