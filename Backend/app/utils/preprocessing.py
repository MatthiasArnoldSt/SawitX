import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ===============================
# Data Preprocessing Utilities
# ===============================
class DataPreprocessor:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def fit_transform(self, df, target_col, feature_cols):
        X = df[feature_cols].values
        y = df[[target_col]].values

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        return X_scaled, y_scaled

    def transform(self, df, feature_cols):
        return self.scaler_X.transform(df[feature_cols].values)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)

# ===============================
# Sequence Creation (LSTM input)
# ===============================
def create_sequences(X, y, time_steps=14, forecast_horizon=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps - forecast_horizon + 1):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps:i + time_steps + forecast_horizon].flatten())
    return np.array(X_seq), np.array(y_seq)
