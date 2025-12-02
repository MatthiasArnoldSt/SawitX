from fastapi import APIRouter
import pandas as pd
from app.utils.recommendation import generate_recommendation

router = APIRouter()

@router.get("/")
def get_recommendation():
    forecast = pd.read_json("app/data/forecast_result.json")
    recommendation = generate_recommendation(forecast)
    return recommendation
