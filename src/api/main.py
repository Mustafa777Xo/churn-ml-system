from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd


from src.api.schemas import PredictRequest
from src.data.load import clean_data
from src.inference.predict import load_model_and_metadata, predict


MODEL = None
METADATA = None
VERSION = None


@asynccontextmanager
async def load_model(app: FastAPI):
    global MODEL, METADATA, VERSION
    MODEL, METADATA, VERSION = load_model_and_metadata("latest")
    yield


app = FastAPI(lifespan=load_model)


@app.post("/predict")
def predict_endpoint(payload: PredictRequest):
    global MODEL, METADATA, VERSION

    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Normalize input to list
    if payload.record is not None:
        records = [payload.record]
        single = True
    else:
        records = payload.records
        single = False

    # Build DataFrame
    df = pd.DataFrame([r.model_dump() for r in records])

    # Apply same cleaning logic as training (corece TotalCharges, drop ID)
    try:
        df_clean = clean_data(df, training=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    preds = predict(df_clean, MODEL, METADATA, VERSION)
    results = preds.to_dict(orient="records")

    return results[0] if single else results
