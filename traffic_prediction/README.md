Traffic Flow Prediction with Support Vector Regression
This project uses Support Vector Regression (SVR) to predict traffic volume (vehicles per hour) based on the hour of the day, day of the week, and weather conditions. Tailored for transportation research, it provides a practical example for urban planning, traffic management, and congestion analysis. The project includes a synthetic traffic dataset, preprocessing, model training, and visualization, making it beginner-friendly and adaptable for real-world traffic data.
Table of Contents

Overview
Project Structure
Prerequisites
Running the Project
Outputs
Data Description
Extending the Project
Troubleshooting
License

Overview
The project employs SVR, a machine learning algorithm, to model non-linear relationships in traffic data. It uses synthetic data to simulate real-world traffic patterns, including rush hours, weekend effects, and weather impacts. The workflow includes:

Loading and preprocessing traffic data (handling missing values and outliers).
Training an SVR model with a radial basis function (RBF) kernel.
Generating predictions and evaluating performance with metrics like Mean Squared Error (MSE) and R-squared.
Visualizing results and saving the model for future use.

This setup is ideal for researchers studying traffic flow dynamics or practitioners developing traffic management solutions.
Project Structure

data/:
traffic_data.csv: Synthetic dataset with traffic volume data.

scripts/:
traffic_flow_svr.py: Python script for SVR model training and analysis.

outputs/:
predictions.csv: Actual vs. predicted traffic volumes.
traffic_flow_prediction.png: Plot of actual vs. predicted traffic volumes.
svr_model.pkl, scaler_X.pkl, scaler_y.pkl: Saved SVR model and scalers.

requirements.txt: Lists required Python dependencies.

Prerequisites

Operating System: Linux, macOS, or Windows.
Software:
python3 (version 3.8 or higher): To run the Python script.
pip: To verify installed dependencies.

Dependencies:
Ensure the following Python packages are installed (listed in requirements.txt):
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
joblib>=1.2.0

Install them manually if needed:pip install -r traffic_prediction/requirements.txt

Running the Project

Navigate to the Project Directory:

Open a terminal and change to the project root directory:cd traffic_prediction

Run the Python Script:

Execute the SVR analysis script:python3 scripts/traffic_flow_svr.py

This will:
Load and preprocess data/traffic_data.csv.
Train the SVR model.
Generate predictions and save them to outputs/predictions.csv.
Create a visualization in outputs/traffic_flow_prediction.png.
Save the model and scalers to outputs/ (svr_model.pkl, scaler_X.pkl, scaler_y.pkl).
Print model performance metrics (MSE and R-squared) to the console.

Outputs
The outputs/ directory contains:

Predictions:
predictions.csv: CSV file with columns:
actual_volume: True traffic volumes from the test set.
predicted_volume: SVR-predicted traffic volumes.

Visualization:
traffic_flow_prediction.png: Plot comparing actual vs. predicted traffic volumes for the first 50 test samples.

Model Files:
svr_model.pkl: Trained SVR model.
scaler_X.pkl: Scaler for features (hour, day, weather).
scaler_y.pkl: Scaler for the target (traffic_volume).

Console Output:
Displays MSE and R-squared score, indicating model accuracy and fit.

Data Description

File: data/traffic_data.csv
Columns:
hour: Hour of the day (0–23).
day: Day of the week (0=Monday, 6=Sunday).
weather: Weather condition (0=clear, 1=rain, 2=fog).
traffic_volume: Traffic volume (vehicles per hour).

Details:
Synthetic data simulates real-world traffic patterns:
Peak traffic during rush hours (7–9 AM, 4–6 PM).
Lower traffic on weekends (days 5–6).
Weather effects (e.g., rain or fog reduces volume).

Preprocessing in traffic_flow_svr.py:
Missing values filled with the median.
Outliers capped at the 99th percentile.

Extending the Project
To enhance the project for research or practical applications:

Use Real Data:
Replace data/traffic_data.csv with real traffic sensor data, ensuring the same column structure (hour, day, weather, traffic_volume).

Add Features:
Modify traffic_flow_svr.py to include features like road type, holidays, traffic incidents, or temperature.

Tune Hyperparameters:
Adjust SVR parameters in traffic_flow_svr.py (e.g., C, epsilon, kernel) to improve performance.

Reuse the Model:
Load the saved model and scalers for predictions on new data:import joblib
import pandas as pd

# Load model and scalers

svr = joblib.load('outputs/svr_model.pkl')
scaler_X = joblib.load('outputs/scaler_X.pkl')
scaler_y = joblib.load('outputs/scaler_y.pkl')

# Example new data

new_data = pd.DataFrame({'hour': [8], 'day': [0], 'weather': [0]})
new_data_scaled = scaler_X.transform(new_data)
pred_scaled = svr.predict(new_data_scaled)
pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
print(f"Predicted traffic volume: {pred[0]:.2f} vehicles/hour")

Troubleshooting

Module Not Found Error:
Ensure all dependencies are installed:pip install -r traffic_prediction/requirements.txt

Verify Python and pip versions:python3 --version
pip --version

File Not Found Error:
Confirm you're in the traffic_prediction/ directory:cd traffic_prediction

Check that data/traffic_data.csv and scripts/traffic_flow_svr.py exist.

Unexpected Output:
Ensure traffic_data.csv has the correct format (columns: hour, day, weather, traffic_volume).
Review console output for errors during script execution.

License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code for research or commercial purposes.

For questions or contributions, feel free to modify the scripts or share feedback!
