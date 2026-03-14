# Predictive Alerting for Cloud Resource Anomalies

## Dataset
The dataset used is the "Cloud Resource Usage Dataset for Anomaly Detection" from Kaggle (https://www.kaggle.com/datasets/programmer3/cloud-resource-usage-dataset-for-anomaly-detection).

The original dataset contains the following features:
- `Timestamp`: Time of the measurement
- `CPU_Usage`: CPU utilization percentage
- `Memory_Usage`: Memory utilization percentage
- `Disk_IO`: Disk I/O operations
- `Network_IO`: Network I/O operations
- `Workload_Type`: Type of workload (e.g., Crypto_Mining)
- `User_ID`: Identifier for the user
- `Anomaly_Label`: Binary label indicating anomaly (1) or normal (0)

## Feature Engineering
Analyzing the features, I decided not to use `User_ID` because in the future there may be more users, and `Workload_Type` because I discovered that all anomalies occur with the Crypto_Mining workload. This could cause the model to rely heavily on this categorical feature rather than learning patterns in the system metrics.

From the remaining features, I engineered additional features including:
- **Time-related features**: Cyclical encoding of hour and day of week (sin/cos transformations)
- **Rolling statistics**: Mean, standard deviation, max, min, median, skewness, kurtosis over a window of 30 time steps
- **Lag features**: Values from 1, 3, and 5 steps back
- **Change features**: Differences and percentage changes
- **Ratio features**: Ratios between CPU/Memory, CPU/Disk, CPU/Network, Disk/Network

## Model Training
I experimented with various scikit-learn models including Random Forest, XGBoost, and LightGBM. Ultimately, the Random Forest classifier provided the best recall score for anomaly detection.

The training process includes:
- Temporal split: 60% train, 20% validation, 20% test
- Feature scaling using StandardScaler
- Hyperparameter tuning via GridSearchCV
- Threshold selection based on incident-level recall on validation set

## Results
- **ROC-AUC**: 0.5112
- **PR-AUC**: 0.5913
- **Incident Recall** (at threshold 0.10): 0.990
- **False Positives per unit time**: 0.399
- Best threshold selected: 0.10 (based on F1 score: 0.7472)

## Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook `predict.ipynb`

## Dependencies
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- tensorflow
- matplotlib

## Usage
Open `predict.ipynb` in Jupyter and run the cells sequentially. The notebook includes data loading, preprocessing, model training, and evaluation.

## Future Work
- Implement LSTM models for sequence prediction
- Add more advanced feature engineering
- Deploy as a real-time alerting system
- Experiment with additional datasets