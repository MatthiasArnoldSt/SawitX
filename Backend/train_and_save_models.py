import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
import joblib
import os
import random

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

def load_and_prepare_data():
    """Load and prepare data for training"""
    print("Loading data...")
    df = pd.read_parquet("../data/cleaned_data_2.parquet")
    
    # Define features and target
    raw_features = ["import", "export", "production", "end_stock",
                    "cpo_futures", "usd_myr_rate", "brent_oil_futures",
                    "soybean_futures", "precipitation", "avg_temperature", "avg_humidity"]
    
    engineered_features = ["lag_1","lag_3","lag_7","rolling_mean_7",
                          "rolling_mean_30","rolling_std_7","rolling_std_30",
                          "pct_change_1","pct_change_7"]
    
    all_features = raw_features + engineered_features
    target_col = "ffb_1%_oer"
    
    X = df[all_features].values
    y = df[target_col].values.reshape(-1, 1)
    
    # Train/Val/Test split
    N = len(df)
    train_size = int(N * 0.7)
    val_size = int(N * 0.2)
    
    X_train_raw = X[:train_size]
    X_val_raw = X[train_size:train_size + val_size]
    X_test_raw = X[train_size + val_size:]
    
    y_train_raw = y[:train_size]
    y_val_raw = y[train_size:train_size + val_size]
    y_test_raw = y[train_size + val_size:]
    
    # Scale data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train = scaler_x.fit_transform(X_train_raw)
    X_val = scaler_x.transform(X_val_raw)
    X_test = scaler_x.transform(X_test_raw)
    
    y_train = scaler_y.fit_transform(y_train_raw)
    y_val = scaler_y.transform(y_val_raw)
    y_test = scaler_y.transform(y_test_raw)
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            scaler_x, scaler_y, all_features)

def create_multi_step_sequences(X, y, lookback, horizon):
    """Create sequences for LSTM training"""
    Xs, ys = [], []
    for i in range(lookback, len(X) - horizon + 1):
        Xs.append(X[i - lookback:i])
        ys.append(y[i:i + horizon].ravel())
    return np.array(Xs), np.array(ys)

def train_lstm_model(X_train, y_train, X_val, y_val, lookback=90, horizon=14):
    """Train LSTM model"""
    print("Training LSTM model...")
    
    # Create sequences
    X_train_seq, y_train_seq = create_multi_step_sequences(X_train, y_train, lookback, horizon)
    X_val_seq, y_val_seq = create_multi_step_sequences(X_val, y_val, lookback, horizon)
    
    print(f"Training sequences: {X_train_seq.shape}, {y_train_seq.shape}")
    
    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, X_train_seq.shape[2])),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(horizon)
    ])
    
    model.compile(optimizer="adam", loss=Huber())
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model

def train_xgboost_models(X_train, y_train, X_val, y_val, lstm_model, scaler_y, lookback=90, horizon=14):
    """Train XGBoost models for residual correction"""
    print("Training XGBoost models for residual correction...")
    
    # Create sequences for prediction
    X_train_seq, y_train_seq = create_multi_step_sequences(X_train, y_train, lookback, horizon)
    X_val_seq, y_val_seq = create_multi_step_sequences(X_val, y_val, lookback, horizon)
    
    # LSTM predictions
    y_pred_train_lstm = lstm_model.predict(X_train_seq)
    y_pred_val_lstm = lstm_model.predict(X_val_seq)
    
    # Inverse transform predictions
    def inv_y(mat_scaled, scaler):
        flat = mat_scaled.reshape(-1, 1)
        inv_flat = scaler.inverse_transform(flat)
        return inv_flat.reshape(mat_scaled.shape)
    
    y_pred_train_lstm_inv = inv_y(y_pred_train_lstm, scaler_y)
    y_pred_val_lstm_inv = inv_y(y_pred_val_lstm, scaler_y)
    y_train_lstm_inv = inv_y(y_train_seq, scaler_y)
    y_val_lstm_inv = inv_y(y_val_seq, scaler_y)
    
    # Prepare features for XGBoost
    n_train_seq = y_pred_train_lstm_inv.shape[0]
    n_val_seq = y_pred_val_lstm_inv.shape[0]
    
    X_train_tail = X_train[lookback: lookback + n_train_seq]
    X_val_tail = X_val[lookback: lookback + n_val_seq]
    
    # Augment features with LSTM predictions
    X_train_xgb_aug = np.hstack([y_pred_train_lstm_inv, X_train_tail])
    X_val_xgb_aug = np.hstack([y_pred_val_lstm_inv, X_val_tail])
    
    # Calculate residuals
    train_residuals = y_train_lstm_inv - y_pred_train_lstm_inv
    val_residuals = y_val_lstm_inv - y_pred_val_lstm_inv
    
    # Train XGBoost models for each step
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.3,
        "seed": SEED,
    }
    
    xgb_models = []
    
    for step in range(horizon):
        print(f"Training XGBoost model for step {step + 1}/{horizon}")
        
        y_train_step = train_residuals[:, step]
        y_val_step = val_residuals[:, step]
        
        dtrain = xgb.DMatrix(X_train_xgb_aug, label=y_train_step)
        dval = xgb.DMatrix(X_val_xgb_aug, label=y_val_step)
        
        evallist = [(dtrain, 'train'), (dval, 'val')]
        
        model_step = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        xgb_models.append(model_step)
    
    return xgb_models

def save_models(lstm_model, xgb_models, scaler_x, scaler_y):
    """Save all trained models and scalers"""
    os.makedirs('models', exist_ok=True)
    
    # Save LSTM model
    save_model(lstm_model, 'models/lstm_model.h5')
    print("✓ LSTM model saved")
    
    # Save XGBoost models
    for i, model in enumerate(xgb_models):
        model.save_model(f'models/xgb_model_step_{i}.model')
    print("✓ XGBoost models saved")
    
    # Save scalers
    joblib.dump(scaler_x, 'models/scaler_x.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    print("✓ Scalers saved")
    
    print("✓ All models and scalers saved successfully!")

def main():
    """Main training function"""
    print("Starting model training...")
    
    # Load and prepare data
    (X_train, X_val, X_test, y_train, y_val, y_test, 
     scaler_x, scaler_y, feature_names) = load_and_prepare_data()
    
    # Train LSTM model
    lstm_model = train_lstm_model(X_train, y_train, X_val, y_val)
    
    # Train XGBoost models
    xgb_models = train_xgboost_models(X_train, y_train, X_val, y_val, lstm_model, scaler_y)
    
    # Save all models
    save_models(lstm_model, xgb_models, scaler_x, scaler_y)
    
    print("🎉 Model training completed successfully!")

if __name__ == '__main__':
    main()