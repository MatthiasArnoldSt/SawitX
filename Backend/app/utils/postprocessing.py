import pandas as pd
import numpy as np

def fuse_hybrid_forecast(y_pred_lstm, y_pred_xgb, scaler_y):
    """Combine hybrid forecast results"""
    lstm_inv = scaler_y.inverse_transform(y_pred_lstm)
    hybrid_pred = lstm_inv + y_pred_xgb.reshape(lstm_inv.shape)
    return hybrid_pred

def make_forecast_output(df, forecast_values, scaler_y, days=28):
    """Format forecast results into a readable DataFrame"""
    last_date = pd.to_datetime(df["date"].iloc[-1])
    forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]

    forecast_values = np.array(forecast_values).flatten()
    forecast_inv = scaler_y.inverse_transform(forecast_values.reshape(-1, 1)).flatten()

    pct_changes = np.concatenate([[0], np.diff(forecast_inv) / forecast_inv[:-1] * 100])
    result_df = pd.DataFrame({
        "date": forecast_dates,
        "forecast_price": forecast_inv[:days],
        "pct_change": np.round(pct_changes[:days], 2)
    })
    return result_df
