import numpy as np
import pandas as pd

df = pd.read_parquet("../data_backend/cleaned_data_2.parquet")

ffb = df['ffb_1%_oer']
returns = ffb.pct_change().dropna() * 100
#volatility = (np.std(latest_forecast) / np.mean(latest_forecast)) * 100


def recommendation_engine2(
    current_price, 
    forecast, 
    volatility, 
    returns,
    model_rmse=None, 
    k=0.8
):
    """
    Enhanced recommendation engine combining volatility, forecast trend,
    and model uncertainty to give adaptive buy/sell/hold advice.
    """

    # --- STEP 1: Basic stats ---
    short_term = np.mean(forecast[:3])
    long_term  = np.mean(forecast[7:14])
    avg_forecast = np.mean(forecast)

    short_change = (short_term - current_price) / current_price * 100
    long_change  = (long_term - current_price) / current_price * 100
    overall_change = (avg_forecast - current_price) / current_price * 100

    # --- STEP 2: Forecast trend strength (slope of 14-day forecast) ---
    slope = np.polyfit(range(len(forecast)), forecast, 1)[0]
    trend_strength = slope / current_price * 100  # normalize to percentage

    # --- STEP 3: Estimate model uncertainty & confidence ---
    if model_rmse is None:
        model_rmse = 2.0  # default fallback
    uncertainty = model_rmse * (volatility / 10)
    confidence = np.clip(100 - uncertainty, 0, 100)
    
    # --- STEP 4: Detect market regime ---
    avg_daily_change = returns.mean()
    #recent_vol = volatility
    base_vol = returns.std()

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

    # --- STEP 5: Decision logic ---
    if trend_strength > threshold and long_change > threshold:
        action = "BUY — Uptrend forming"
    elif trend_strength < -threshold and long_change < -threshold:
        action = "SELL — Downtrend forming"
    elif short_change < -threshold and long_change > threshold:
        action = "HOLD harvest — temporary dip"
    elif abs(overall_change) < threshold:
        action = "NEUTRAL — Market stable"
    else:
        action = "WAIT — Unclear direction"

    # --- STEP 6: Construct readable output ---
    recommendation = {
        "short_term_change(%)": round(short_change, 2),
        "long_term_change(%)": round(long_change, 2),
        "overall_change(%)": round(overall_change, 2),
        "trend_strength(%)": round(trend_strength, 2),
        "volatility(%)": round(volatility, 2),
        "uncertainty(%)": round(uncertainty, 2),
        "confidence(%)": round(confidence, 2),
        "market_regime": regime,
        "threshold(%)": round(threshold, 2),
        "recommendation": action
    }

    return recommendation

