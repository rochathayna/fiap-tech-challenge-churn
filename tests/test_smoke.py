import joblib
import torch

from src.models.mlp import ChurnMLP


def test_modelo_carrega():
    model = ChurnMLP(input_dim=44)
    state = torch.load("models/mlp_churn.pt", weights_only=True)
    model.load_state_dict(state)
    assert model is not None

def test_preprocessor_carrega():
    preprocessor = joblib.load("models/preprocessor.pkl")
    assert preprocessor is not None

def test_modelo_inferencia():
    model = ChurnMLP(input_dim=44)
    state = torch.load("models/mlp_churn.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    x = torch.randn(1, 44)
    prob = model.predict_proba(x).item()
    assert 0.0 <= prob <= 1.0