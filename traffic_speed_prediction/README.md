# Traffic Speed Prediction with Random Forest

This project uses **Random Forest Regression** to predict traffic speed (km/h) based on the hour of the day, day of the week, weather conditions, and road type. Designed for transportation research, it supports applications like route planning, traffic management, and congestion analysis. The project includes a synthetic traffic dataset, preprocessing, model training, and visualization, making it beginner-friendly and adaptable for real-world data.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Running the Project](#running-the-project)
- [Outputs](#outputs)
- [Data Description](#data-description)
- [Extending the Project](#extending-the-project)
- [Troubleshooting](#troubleshooting)
- [Version Control](#version-control)
- [License](#license)

## Overview

The project employs a Random Forest Regressor to model complex relationships in traffic data, using synthetic data to simulate real-world patterns like rush hours, weekend effects, weather impacts, and road type variations. The workflow includes:
- Loading and preprocessing traffic data (handling missing values and outliers).
- Training a Random Forest model with 100 trees.
- Generating predictions and evaluating performance with Mean Squared Error (MSE) and R-squared metrics.
- Visualizing results and saving the model for reuse.

This setup is ideal for researchers studying traffic dynamics or practitioners optimizing transportation systems.

## Project Structure

- `data/`:
  - `traffic_data.csv`: Synthetic dataset with traffic speed data.
- `scripts/`:
  - `traffic_speed_rf.py`: Python script for Random Forest model training and analysis.
- `outputs/`:
  - `predictions.csv`: Actual vs. predicted traffic speeds.
  - `traffic_speed_prediction.png`: Plot of actual vs. predicted traffic speeds.
  - `rf_model.pkl`, `scaler_X.pkl`: Saved Random Forest model and feature scaler.
- `requirements.txt`: Lists required Python dependencies.
- `.gitignore`: Specifies files to exclude from version control (e.g., output files, Python cache).

## Prerequisites

- **Operating System**: Linux, macOS, or Windows.
- **Software**:
  - `python3` (version 3.8 or higher): To run the Python script.
  - `pip`: To verify installed dependencies.
- **Dependencies**:
  - Ensure the following Python packages are installed (listed in `requirements.txt`):
    - `numpy>=1.24.0`
    - `pandas>=2.0.0`
    - `scikit-learn>=1.3.0`
    - `matplotlib>=3.7.0`
    - `joblib>=1.2.0`
  - Install them if needed:
    ```bash
    pip install -r traffic_speed_prediction/requirements.txt
    ```

## Running the Project

1. **Navigate to the Project Directory**:
   - Open a terminal and change to the project root:
     ```bash
     cd traffic_speed_prediction
     ```

2. **Execute the Python Script**:
   - Run the Random Forest analysis script:
     ```bash
     python3 scripts/traffic_speed_rf.py
     ```
   - This will:
     - Load and preprocess `data/traffic_data.csv`.
     - Train the Random Forest model.
     - Save predictions to `outputs/predictions.csv`.
     - Generate a plot in `outputs/traffic_speed_prediction.png`.
     - Save the model and scaler to `outputs/` (`rf_model.pkl`, `scaler_X.pkl`.
     - Print MSE and R-squared metrics to the console.

## Outputs

The `outputs/` directory contains:
- **Predictions**:
  - `predictions.csv`: CSV file with:
    - `actual_speed`: True traffic speeds from the test set.
    - `predicted_speed`: Random Forest-predicted speeds.
- **Visualization**:
  - `traffic_speed_prediction.png`: Plot comparing actual vs. predicted traffic speeds for the first 50 test samples.
- **Model Files**:
  - `rf_model.pkl`: Trained Random Forest model.
  - `scaler_X.pkl`: Scaler for features (`hour`, `day`, `weather`, `road_type`).
- **Console Output**:
  - Displays MSE and R-squared score, indicating model performance.

## Data Description

- **File**: `data/traffic_data.csv`
- **Columns**:
  - `hour`: Hour of the day (0–23).
  - `day`: Day of the week (0=Monday, 6=Sunday).
  - `weather`: Weather condition (0=clear, 1=rain, 2=fog).
  - `road_type`: Road category (0=highway, 1=arterial, 2=local).
  - `traffic_speed`: Average traffic speed (km/h).
- **Details**:
  - Synthetic data simulates real-world patterns:
    - Higher speeds during off-peak hours, lower during rush hours (7–9 AM, 4–6 PM).
    - Slightly lower speeds on weekends.
    - Weather impacts (e.g., rain or fog reduces speed).
    - Road type effects (e.g., highways have higher speeds, local roads lower).
  - Preprocessing in `traffic_speed_rf.py`:
    - Missing values filled with the median.
    - Outliers capped at the 99th percentile.

## Extending the Project

To enhance the project:
- **Use Real Data**:
  - Replace `data/traffic_data.csv` with real traffic sensor data, ensuring the same column structure (`hour`, `day`, `weather`, `road_type`, `traffic_speed`).
- **Add Features**:
  - Edit `traffic_speed_rf.py` to include features like traffic volume, holidays, or incidents.
- **Tune Hyperparameters**:
  - Modify Random Forest parameters in `traffic_speed_rf.py` (e.g., `n_estimators`, `max_depth`) to optimize performance.
- **Reuse the Model**:
  - Load the saved model and scaler for new predictions:
    ```python
    import joblib
    import pandas as pd

    # Load model and scaler
    rf = joblib.load('outputs/rf_model.pkl')
    scaler_X = joblib.load('outputs/scaler_X.pkl')

    # Example new data
    new_data = pd.DataFrame({'hour': [8], 'day': [0], 'weather': [0], 'road_type': [0]})
    new_data_scaled = scaler_X.transform(new_data)
    pred = rf.predict(new_data_scaled)
    print(f"Predicted traffic speed: {pred[0]:.2f} km/h")
    ```

## Troubleshooting

- **Module Not Found**:
  - Ensure dependencies are installed:
    ```bash
    pip install -r traffic_speed_prediction/requirements.txt
    ```
  - Check Python and pip versions:
    ```bash
    python3 --version
    pip --version
    ```
- **File Not Found**:
  - Verify you're in the `traffic_speed_prediction/` directory:
    ```bash
    cd traffic_speed_prediction
    ```
  - Confirm `data/traffic_data.csv` and `scripts/traffic_speed_rf.py` exist.
- **Incorrect Output**:
  - Ensure `traffic_data.csv` has the correct format (columns: `hour`, `day`, `weather`, `road_type`, `traffic_speed`).
  - Check console output for errors.

## Version Control

- The `.gitignore` file excludes unnecessary files from version control, such as:
  - Output files in `outputs/` (e.g., `predictions.csv`, `traffic_speed_prediction.png`).
  - Python cache (`__pycache__`, `.pyc`).
  - Virtual environments (`venv/`, `.env/`).
  - System files (e.g., `.DS_Store`).
- Track `data/traffic_data.csv`, `scripts/traffic_speed_rf.py`, `requirements.txt`, and `.gitignore` in your repository.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code for research or commercial purposes.

---

For questions or contributions, modify the scripts or share feedback!
