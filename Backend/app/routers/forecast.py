from fastapi import APIRouter
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from app.utils.preprocessing import prepare_input_data
from app.utils.postprocessing import make_forecast_output

router = APIRouter()

# Load models once
lstm_model = tf.keras.models.load_model("app/models/lstm_model.h5")
xgb_model = xgb.Booster()
xgb_model.load_model("app/models/xgb_model.json")

@router.get("/")
def get_forecast(days: int = 28):
    # Load recent data
    #df = pd.read_csv("app/data/latest_data.csv")
    df = pd.read_parquet("../data_backend/cleaned_data_2.parquet")


    # Preprocess input
    X_input, scaler_y = prepare_input_data(df)

    # Predict with LSTM
    y_pred_lstm = lstm_model.predict(X_input)

    # Predict residuals with XGBoost
    dmatrix = xgb.DMatrix(X_input)
    residuals_pred = xgb_model.predict(dmatrix)

    # Combine hybrid output
    final_forecast = y_pred_lstm.flatten() + residuals_pred

    # Format results
    forecast_df = make_forecast_output(df, final_forecast, scaler_y, days)
    return forecast_df.to_dict(orient="records")
