from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

app = Flask(__name__)
CORS(app)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Global variables for models and scalers
lstm_model = None
xgb_models = None
scaler_x = None
scaler_y = None
feature_names = None

def load_models():
    """Load trained models and scalers"""
    global lstm_model, xgb_models, scaler_x, scaler_y, feature_names
    
    try:
        # Load LSTM model
        lstm_model = load_model('models/lstm_model.h5')
        print("✓ LSTM model loaded")
        
        # Load XGBoost models
        xgb_models = []
        for i in range(14):  # 14-day forecast horizon
            model = xgb.Booster()
            model.load_model(f'models/xgb_model_step_{i}.model')
            xgb_models.append(model)
        print("✓ XGBoost models loaded")
        
        # Load scalers
        scaler_x = joblib.load('models/scaler_x.pkl')
        scaler_y = joblib.load('models/scaler_y.pkl')
        print("✓ Scalers loaded")
        
        # Define feature names (from your code)
        raw_features = ["import", "export", "production", "end_stock",
                       "cpo_futures", "usd_myr_rate", "brent_oil_futures",
                       "soybean_futures", "precipitation", "avg_temperature", "avg_humidity"]
        
        engineered_features = ["lag_1","lag_3","lag_7","rolling_mean_7",
                              "rolling_mean_30","rolling_std_7","rolling_std_30",
                              "pct_change_1","pct_change_7"]
        
        feature_names = raw_features + engineered_features
        
    except Exception as e:
        print(f"Error loading models: {e}")
        # Fallback to mock data if models aren't available
        lstm_model = None

def create_multi_step_sequences(X, lookback, horizon):
    """Create sequences for LSTM prediction"""
    Xs = []
    for i in range(lookback, len(X) - horizon + 1):
        Xs.append(X[i - lookback:i])
    return np.array(Xs)

def get_latest_data():
    """Get the most recent data for prediction"""
    try:
        df = pd.read_parquet("../data/cleaned_data_2.parquet")
        
        # Get the last LOOKBACK days for prediction
        lookback = 90
        recent_data = df.tail(lookback).copy()
        
        return recent_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def generate_real_forecast():
    """Generate forecast using your actual LSTM-XGBoost model"""
    global lstm_model, xgb_models, scaler_x, scaler_y, feature_names
    
    # If models aren't loaded, use mock data as fallback
    if lstm_model is None:
        return generate_mock_forecast()
    
    try:
        # Get latest data
        df = get_latest_data()
        if df is None:
            return generate_mock_forecast()
        
        # Prepare features
        X_recent = df[feature_names].values
        y_recent = df["ffb_1%_oer"].values.reshape(-1, 1)
        
        # Scale features
        X_scaled = scaler_x.transform(X_recent)
        y_scaled = scaler_y.transform(y_recent)
        
        # Create sequences for LSTM
        lookback = 90
        horizon = 14
        
        X_seq = create_multi_step_sequences(X_scaled, lookback, horizon)
        
        # Get the most recent sequence for prediction
        X_pred = X_seq[-1:].reshape(1, lookback, len(feature_names))
        
        # LSTM prediction (scaled)
        y_pred_lstm_scaled = lstm_model.predict(X_pred)
        
        # Inverse transform LSTM prediction
        y_pred_lstm = y_pred_lstm_scaled.reshape(-1, 1)
        y_pred_lstm_inv = scaler_y.inverse_transform(y_pred_lstm)
        y_pred_lstm_inv = y_pred_lstm_inv.reshape(1, -1)[0]
        
        # Prepare features for XGBoost residual correction
        X_tail = X_scaled[lookback:lookback + 1]  # Current features
        
        # Augment with LSTM predictions
        X_xgb_aug = np.hstack([y_pred_lstm_scaled.reshape(1, -1), X_tail])
        
        # XGBoost residual prediction
        y_pred_residuals = []
        for step in range(horizon):
            dtest = xgb.DMatrix(X_xgb_aug)
            pred_res = xgb_models[step].predict(dtest)
            y_pred_residuals.append(pred_res[0])
        
        # Final prediction = LSTM + Residuals
        y_pred_final = y_pred_lstm_inv + np.array(y_pred_residuals)
        
        # Generate dates
        last_date = df.index[-1] if hasattr(df.index, 'dtype') else pd.Timestamp.now()
        future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
        
        # Calculate confidence based on historical performance and volatility
        historical_volatility = np.std(y_recent[-30:]) / np.mean(y_recent[-30:]) * 100
        base_confidence = max(85 - historical_volatility * 2, 60)
        
        forecast_data = []
        for i, date in enumerate(future_dates):
            # Confidence decreases for longer horizons
            confidence = base_confidence * (1 - (i * 0.03))
            confidence = max(min(confidence, 95), 65)
            
            forecast_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(float(y_pred_final[i]), 2),
                "confidence": round(confidence, 1)
            })
        
        # Past data (last 90 days)
        past_dates = [last_date - timedelta(days=i) for i in range(89, -1, -1)]
        past_prices = y_recent[-90:].flatten() if len(y_recent) >= 90 else y_recent.flatten()
        
        past_data = []
        for i, (date, price) in enumerate(zip(past_dates, past_prices)):
            past_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(float(price), 2)
            })
        
        return {
            "past_data": past_data,
            "forecast": forecast_data
        }
        
    except Exception as e:
        print(f"Error in real forecast generation: {e}")
        # Fallback to mock data
        return generate_mock_forecast()

def generate_mock_forecast():
    """Fallback mock data generator"""
    end_date = datetime.now()
    start_date_past = end_date - timedelta(days=90)
    start_date_future = end_date + timedelta(days=1)
    
    # More realistic mock data based on FFB price characteristics
    past_dates = [start_date_past + timedelta(days=x) for x in range(90)]
    base_price = 850  # Typical FFB price range
    past_prices = [base_price + np.random.normal(0, 25) for _ in range(90)]
    
    # Future forecast (next 14 days)
    future_dates = [start_date_future + timedelta(days=x) for x in range(14)]
    
    # More realistic trend
    trend = np.random.normal(0.05, 0.1, 14).cumsum()
    future_prices = past_prices[-1] + trend * 15
    confidence = np.clip(90 - np.abs(trend * 50), 70, 95)
    
    forecast_data = []
    for i, date in enumerate(future_dates):
        forecast_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "price": round(future_prices[i], 2),
            "confidence": round(confidence[i], 1)
        })
    
    return {
        "past_data": [
            {"date": d.strftime("%Y-%m-%d"), "price": round(p, 2)} 
            for d, p in zip(past_dates, past_prices)
        ],
        "forecast": forecast_data
    }

def generate_recommendation():
    """Generate recommendation using your actual model forecast"""
    forecast_data = generate_real_forecast()
    current_price = forecast_data["past_data"][-1]["price"]
    forecast_prices = [f["price"] for f in forecast_data["forecast"]]
    
    # Use your actual recommendation engine logic
    return recommendation_engine2(
        current_price=current_price,
        forecast=forecast_prices,
        volatility=np.std(forecast_prices) / np.mean(forecast_prices) * 100,
        returns=np.diff(forecast_prices) / forecast_prices[:-1] * 100,
        model_rmse=2.0  # You can calculate this from your validation
    )

def recommendation_engine2(current_price, forecast, volatility, returns, model_rmse=None, k=0.8):
    """Your actual recommendation engine"""
    # Your original recommendation engine code
    short_term = np.mean(forecast[:3])
    long_term = np.mean(forecast[7:14])
    avg_forecast = np.mean(forecast)

    short_change = (short_term - current_price) / current_price * 100
    long_change = (long_term - current_price) / current_price * 100
    overall_change = (avg_forecast - current_price) / current_price * 100

    # Trend strength
    slope = np.polyfit(range(len(forecast)), forecast, 1)[0]
    trend_strength = slope / current_price * 100

    # Model uncertainty
    if model_rmse is None:
        model_rmse = 2.0
    uncertainty = model_rmse * (volatility / 10)
    confidence = np.clip(100 - uncertainty, 0, 100)
    
    # Market regime
    avg_daily_change = returns.mean() if len(returns) > 0 else 0
    base_vol = returns.std() if len(returns) > 0 else 10

    if volatility > base_vol * 1.3:
        regime = "High Volatility"
        k *= 1.2
    elif avg_daily_change > 0.3:
        regime = "Uptrend"
    elif avg_daily_change < -0.3:
        regime = "Downtrend"
    else:
        regime = "Stable"

    threshold = k * volatility

    # Decision logic
    if trend_strength > threshold and long_change > threshold:
        action = "BUY"
        color = "green"
    elif trend_strength < -threshold and long_change < -threshold:
        action = "SELL"
        color = "red"
    elif short_change < -threshold and long_change > threshold:
        action = "HOLD"
        color = "yellow"
    elif abs(overall_change) < threshold:
        action = "NEUTRAL"
        color = "blue"
    else:
        action = "WAIT"
        color = "orange"

    return {
        "current_price": current_price,
        "price_change": round(short_change, 2),
        "recommendation": action,
        "color": color,
        "explanation": f"{action} — Trend: {trend_strength:.1f}%, Volatility: {volatility:.1f}%",
        "volatility": round(volatility, 2),
        "confidence": round(confidence, 2),
        "market_regime": regime
    }

def generate_daily_recommendations():
    """Generate daily recommendations using actual forecast"""
    forecast_data = generate_real_forecast()
    recommendations = []
    
    for i, day in enumerate(forecast_data["forecast"]):
        price = day["price"]
        confidence = day["confidence"]
        
        # Use simplified recommendation logic for daily breakdown
        if i < len(forecast_data["forecast"]) - 1:
            next_day_price = forecast_data["forecast"][i + 1]["price"]
            daily_change = (next_day_price - price) / price * 100
        else:
            daily_change = 0
        
        if daily_change > 1.5:
            action = "BUY"
            color = "green"
            explanation = "Expected price increase tomorrow"
        elif daily_change < -1.5:
            action = "SELL"
            color = "red"
            explanation = "Expected price decrease tomorrow"
        else:
            action = "HOLD"
            color = "yellow"
            explanation = "Stable price expected"
        
        recommendations.append({
            "date": day["date"],
            "forecast_price": price,
            "recommendation": action,
            "color": color,
            "explanation": explanation,
            "confidence": confidence
        })
    
    return recommendations

# Load models when the app starts
@app.before_request
def initialize():
    load_models()

# API Routes (same as before)
@app.route('/api/forecast')
def get_forecast():
    days = request.args.get('days', default=14, type=int)
    forecast_data = generate_real_forecast()
    
    # Limit forecast to requested days
    limited_forecast = forecast_data["forecast"][:days]
    
    return jsonify({
        "past_data": forecast_data["past_data"],
        "forecast": limited_forecast
    })

@app.route('/api/recommendation')
def get_recommendation():
    return jsonify(generate_recommendation())

@app.route('/api/daily-recommendations')
def get_daily_recommendations():
    return jsonify(generate_daily_recommendations())

@app.route('/api/historical-trends')
def get_historical_trends():
    return jsonify(generate_real_forecast())

@app.route('/api/export-csv')
def export_csv():
    forecast_data = generate_real_forecast()
    
    # Combine past and forecast data
    all_data = []
    for item in forecast_data["past_data"]:
        all_data.append({
            "date": item["date"],
            "price": item["price"],
            "type": "historical"
        })
    
    for item in forecast_data["forecast"]:
        all_data.append({
            "date": item["date"],
            "price": item["price"],
            "type": "forecast",
            "confidence": item["confidence"]
        })
    
    df = pd.DataFrame(all_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'ffb_forecast_{datetime.now().strftime("%Y%m%d")}.csv'
    )

# Add this route at the end of your app.py, before if __name__ == '__main__'

@app.route('/')
def home():
    return jsonify({
        "message": "FFB Forecast API is running!",
        "endpoints": {
            "/api/forecast": "Get price forecasts",
            "/api/recommendation": "Get trading recommendation", 
            "/api/daily-recommendations": "Get daily recommendations",
            "/api/historical-trends": "Get historical data",
            "/api/export-csv": "Export data as CSV"
        },
        "status": "operational"
    })

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, port=5000)