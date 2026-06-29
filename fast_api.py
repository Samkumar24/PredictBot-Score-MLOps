from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor_bot_score.utils.common import yaml_load
from src.predictor_bot_score.constants import CONFIG_PATH
import mlflow
import pandas as pd


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
REGISTRY_NAME  = "predict-bot-challenge"
ALIAS = "production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
Model_url = f"models:/{REGISTRY_NAME}@{ALIAS}"
model = mlflow.pyfunc.load_model(Model_url)


app = FastAPI(title="PredictBot Score Serving")

class PredictionInput(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    lag_96: float
    rolling_std_4: float


@app.get("/")
def main_page():
    return {"message": "Welcome to the main page!"}

@app.get("/health")
def model_health():
    return{
        "status": "ok",
          "model_uri": Model_url,
          "model "   : list(model.metadata.flavors.keys())
    }
@app.post('/predict')
def predict(input_data: PredictionInput):
    df = pd.DataFrame([input_data.model_dump()])
    prediction = model.predict(df)
    return {'prediction': float(prediction[0])}