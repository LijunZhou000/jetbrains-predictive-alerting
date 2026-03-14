import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, make_scorer, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_time_features(df, timestamp_col="Timestamp"):
    """
    Creates cyclical time-based features from a timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    timestamp_col : str
        Name of the timestamp column

    Returns
    -------
    df : pd.DataFrame
        Dataframe with new time features
    new_features : list
        List of created feature names
    """

    df = df.copy()
    new_features = []

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Hour features
    df["hour"] = df[timestamp_col].dt.hour
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    # Day of week features
    df["dayofweek"] = df[timestamp_col].dt.dayofweek
    df["dow_sin"] = np.sin(2*np.pi*df["dayofweek"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dayofweek"]/7)

    new_features.extend([
        "hour",
        "hour_sin",
        "hour_cos",
        "dayofweek",
        "dow_sin",
        "dow_cos"
    ])

    return df, new_features

def extract_features(df, columns, window, group_col="User_ID"):
    """
    Extract rolling, lag, and change-based features for time-series columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to compute features for
    window : int
        Rolling window size
    group_col : str
        Column defining independent time-series (e.g. User_ID)

    Returns
    -------
    df : pd.DataFrame
        Dataframe with new features
    new_features : list
        List of created feature names
    """

    df = df.copy()
    new_features = []

    grouped = df.groupby(group_col)

    for col in columns:

        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe.")

        # rolling object
        rolling = grouped[col].rolling(window=window, min_periods=1)

        # Rolling statistics
        # df[f"{col}_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        # df[f"{col}_std_{window}"] = rolling.std().reset_index(level=0, drop=True)
        # df[f"{col}_max_{window}"] = rolling.max().reset_index(level=0, drop=True)
        # df[f"{col}_min_{window}"] = rolling.min().reset_index(level=0, drop=True)
        # df[f"{col}_median_{window}"] = rolling.median().reset_index(level=0, drop=True)
        # df[f"{col}_skew_{window}"] = rolling.skew().reset_index(level=0, drop=True)
        # df[f"{col}_kurt_{window}"] = rolling.kurt().reset_index(level=0, drop=True)
        df[f"{col}_mean_{window}"] = rolling.mean().shift(1).reset_index(level=0, drop=True)
        df[f"{col}_std_{window}"] = rolling.std().shift(1).reset_index(level=0, drop=True)
        df[f"{col}_max_{window}"] = rolling.max().shift(1).reset_index(level=0, drop=True)
        df[f"{col}_min_{window}"] = rolling.min().shift(1).reset_index(level=0, drop=True)
        df[f"{col}_median_{window}"] = rolling.median().shift(1).reset_index(level=0, drop=True)
        df[f"{col}_skew_{window}"] = rolling.skew().shift(1).reset_index(level=0, drop=True)
        df[f"{col}_kurt_{window}"] = rolling.kurt().shift(1).reset_index(level=0, drop=True)

        # Range
        df[f"{col}_range_{window}"] = (
            df[f"{col}_max_{window}"] - df[f"{col}_min_{window}"]
        )

        # Lag features
        df[f"{col}_lag_1"] = grouped[col].shift(1)
        df[f"{col}_lag_3"] = grouped[col].shift(3)
        df[f"{col}_lag_5"] = grouped[col].shift(5)

        # Change features
        # df[f"{col}_diff_1"] = grouped[col].diff(1)
        # df[f"{col}_pct_change_1"] = grouped[col].pct_change()
        df[f"{col}_diff_1"] = grouped[col].diff(1).shift(1)
        df[f"{col}_pct_change_1"] = grouped[col].pct_change().shift(1)

        # Extend feature list
        new_features.extend([
            f"{col}_mean_{window}",
            f"{col}_std_{window}",
            f"{col}_max_{window}",
            f"{col}_min_{window}",
            f"{col}_median_{window}",
            f"{col}_skew_{window}",
            f"{col}_kurt_{window}",
            f"{col}_range_{window}",
            f"{col}_lag_1",
            f"{col}_lag_3",
            f"{col}_lag_5",
            f"{col}_diff_1",
            f"{col}_pct_change_1"
        ])

    return df, new_features

# base_features = ["CPU_Usage", "Memory_Usage", "Disk_IO", "Network_IO"]
# dummy_features = [c for c in df.columns if c.startswith("WT_") or c.startswith("UID_")]
# feature_cols = base_features + dummy_features
# df = pd.get_dummies(
#     df,
#     columns=["Workload_Type", "User_ID"],
#     prefix=["WT", "UID"],
#     dtype=int
# )
# df = pd.get_dummies(
#     df,
#     columns=["Workload_Type"],
#     prefix=["WT"],
#     dtype=int
# )

# base_features = ["Anomaly_Label", "CPU_Usage", "Memory_Usage", "Disk_IO", "Network_IO"]
# feature_cols += new_features

def create_ratio_features(df):
    """
    Creates ratio-based features between key system metrics.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with new features added
    new_features : list
        List of created feature names
    """
    
    df = df.copy()
    new_features = []

    ratios = {
        "CPU_mem_ratio": ("CPU_Usage", "Memory_Usage"),
        "CPU_disk_ratio": ("CPU_Usage", "Disk_IO"),
        "CPU_net_ratio": ("CPU_Usage", "Network_IO"),
        "disk_net_ratio": ("Disk_IO", "Network_IO"),
    }

    for name, (num, den) in ratios.items():
        df[name] = df[num] / (df[den] + 1e-6)
        new_features.append(name)

    return df, new_features

def perform_grid_search(X_train, y_train, model=None, param_grid=None, cv=None, scoring='roc_auc', n_jobs=-1, random_state=42):
    """
    Performs grid search for hyperparameter tuning on a given model.
    
    Parameters:
    - X_train (array-like): Training features.
    - y_train (array-like): Training labels.
    - model: The estimator to tune. Defaults to RandomForestClassifier().
    - param_grid (dict): Dictionary of parameters to try. Defaults to a basic grid for RandomForest.
    - cv: Cross-validation strategy. Defaults to TimeSeriesSplit for time series.
    - scoring (str): Scoring metric. Default 'roc_auc'.
    - n_jobs (int): Number of jobs for parallel processing. Default -1.
    - random_state (int): Random state for reproducibility. Default 42.
    
    Returns:
    - dict: {'best_model': fitted best estimator, 'best_params': dict of best params, 'cv_results': dict of cv results}
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    if model is None:
        model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100,300,500],  # Reducido para evitar overfitting
            'max_depth': [5, 10, None],  # Profundidad limitada
            'min_samples_split': [10, 20],  # Más restricción
            'min_samples_leaf': [5, 10],  # Más restricción
            'max_features': ['sqrt']
        }
    
    if cv is None:
        cv = TimeSeriesSplit(n_splits=5)  # CV temporal para series
    
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2
    )
    
    search.fit(X_train, y_train)
    
    return {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'cv_results': search.cv_results_
    }
    
    
def get_event_intervals(labels_array):
    """
    Convierte una secuencia binaria (0/1) en lista de intervalos [start_idx, end_idx] de incidentes.
    """
    labels = np.asarray(labels_array)
    intervals = []
    in_event = False
    start = None

    for i, val in enumerate(labels):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            intervals.append((start, i - 1))
    if in_event:
        intervals.append((start, len(labels) - 1))
    return intervals

def evaluate_alerts(y_true, y_proba, threshold, H, step_duration=1.0):
    """
    Evalúa el sistema de alertas a nivel de intervalo.
    y_true: secuencia 0/1 (por paso).
    y_proba: probabilidades modelo (misma longitud).
    threshold: umbral para alerta.
    H: horizonte (en pasos).
    step_duration: duración de un paso (por ejemplo, minutos u horas) para normalizar FP/time.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    # 1) Intervalos de incidente en y_true
    incident_intervals = get_event_intervals(y_true)

    # 2) Para cada intervalo, ver si hay una alerta en la ventana [t_start - H, t_start)
    detected = 0
    lead_times = []

    for (start, end) in incident_intervals:
        # ventana de look-back para detección anticipada
        lookback_start = max(0, start - H)
        alert_indices = np.where(y_pred[lookback_start:start] == 1)[0]
        if len(alert_indices) > 0:
            detected += 1
            # tomamos la primera alerta
            first_alert_idx = alert_indices[0] + lookback_start
            lead = start - first_alert_idx
            lead_times.append(lead)

    incident_recall = detected / len(incident_intervals) if incident_intervals else 0.0
    avg_lead_steps = np.mean(lead_times) if lead_times else 0.0

    # 3) Falsos positivos: alertas fuera de cualquier intervalo de incidente
    fp_count = 0
    total_time = len(y_true) * step_duration

    # máscara de puntos dentro de incidentes
    in_event_mask = np.zeros_like(y_true, dtype=bool)
    for (start, end) in incident_intervals:
        in_event_mask[start:end+1] = True

    # FP son alertas en índices donde no hay incidente
    fp_indices = np.where((y_pred == 1) & (~in_event_mask))[0]
    fp_count = len(fp_indices)
    fp_per_unit_time = fp_count / total_time if total_time > 0 else 0.0

    results = {
        "incident_recall": incident_recall,
        "avg_lead_steps": avg_lead_steps,
        "fp_per_unit_time": fp_per_unit_time,
        "n_incidents": len(incident_intervals),
        "n_alerts": int(y_pred.sum()),
    }
    return results

# # === LSTM MEJORADO con split 60/20/20 + early stopping ===

# # 1) Reshape 3D para LSTM (samples, timesteps=W, features)
# num_features_per_step = X_train.shape[1] // W
# X_train_3d = X_train_scaled.reshape(X_train_scaled.shape[0], W, num_features_per_step)
# X_val_3d   = X_val_scaled.reshape(X_val_scaled.shape[0], W, num_features_per_step)
# X_test_3d  = X_test_scaled.reshape(X_test_scaled.shape[0], W, num_features_per_step)

# print(f"LSTM shapes: train={X_train_3d.shape}, val={X_val_3d.shape}")

# # 2) Modelo LSTM BIDIRECCIONAL + capas más profundas
# lstm_model = Sequential([
#     # Bidirectional LSTM captura dependencias forward/backward
#     tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True), 
#                                   input_shape=(W, num_features_per_step)),
#     BatchNormalization(),
#     Dropout(0.3),
    
#     tf.keras.layers.Bidirectional(LSTM(32, return_sequences=False)),
#     BatchNormalization(),
#     Dropout(0.4),
    
#     Dense(16, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='sigmoid')
# ])

# # 3) Compilar con class_weight para desbalance
# class_weight = {0: 1.0, 1: 10.0}  # Peso alto para anomalías

# lstm_model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss='binary_crossentropy',
#     metrics=[tf.keras.metrics.AUC(name='auc', curve='PR')]  # PR-AUC directamente
# )

# # 4) Callbacks avanzados
# callbacks = [
#     EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True, verbose=1),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
# ]

# # 5) Entrenar con validación explícita (train+20% val interno)
# history = lstm_model.fit(
#     X_train_3d, y_train,
#     validation_data=(X_val_3d, y_val),
#     epochs=50,  # Más epochs con early stopping
#     batch_size=64,  # Batch más grande
#     class_weight=class_weight,
#     callbacks=callbacks,
#     verbose=1
# )

# # 6) Predicciones y métricas
# y_test_proba_lstm = lstm_model.predict(X_test_3d, verbose=0).flatten()

# print("\n=== LSTM MEJORADO ===")
# print("Classification Report (thr=0.5):")
# print(classification_report(y_test, (y_test_proba_lstm >= 0.5).astype(int), digits=4))

# roc_auc_lstm = roc_auc_score(y_test, y_test_proba_lstm)
# precision, recall, _ = precision_recall_curve(y_test, y_test_proba_lstm)
# pr_auc_lstm = auc(recall, precision)

# print(f"LSTM ROC-AUC: {roc_auc_lstm:.4f}")
# print(f"LSTM PR-AUC:  {pr_auc_lstm:.4f}")

# # 7) Incident-level metrics (¡IMPORTANTE!)
# lstm_incident_results = evaluate_alerts(y_test, y_test_proba_lstm, threshold=0.30, H=H)
# print(f"LSTM Incident Recall@0.30: {lstm_incident_results['incident_recall']:.1%}")
# print(f"LSTM FP/unit@0.30:         {lstm_incident_results['fp_per_unit_time']:.3f}")
