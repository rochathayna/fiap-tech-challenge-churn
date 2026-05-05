import logging
import time
import joblib
import torch
import pandas as pd
from fastapi import FastAPI, Request
from src.api.schemas import CustomerFeatures, PredictionResponse
from src.models.mlp import ChurnMLP
from src.data.preprocessing import FEATURES_NUM, FEATURES_CAT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.0.0")

MODEL_PATH = "models/mlp_churn.pt"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
INPUT_DIM = 44

preprocessor = joblib.load(PREPROCESSOR_PATH)
model = ChurnMLP(input_dim=INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()
logger.info("Modelo carregado com sucesso!")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    response.headers["X-Process-Time"] = str(duration)
    return response

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    data = pd.DataFrame([customer.model_dump()])
    X = preprocessor.transform(data[FEATURES_NUM + FEATURES_CAT])
    tensor = torch.FloatTensor(X)
    prob = model.predict_proba(tensor).item()
    logger.info(f"Predição realizada: prob={prob:.4f}")
    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_label=prob >= 0.5
    )