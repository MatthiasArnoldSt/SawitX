from fastapi import APIRouter
import pandas as pd

router = APIRouter()

@router.get("/")
def get_history():
    df = pd.read_csv("app/data/historical_data.csv")
    return df.to_dict(orient="records")

@router.get("/export")
def export_history():
    df = pd.read_csv("app/data/historical_data.csv")
    file_path = "app/data/exported_history.csv"
    df.to_csv(file_path, index=False)
    return {"status": "success", "file": file_path}
