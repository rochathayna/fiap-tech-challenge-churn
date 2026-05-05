from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    payload = {
        "tenure": 12,
        "MonthlyCharges": 65.0,
        "TotalCharges": 780.0,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0

def test_predict_schema_invalido():
    response = client.post("/predict", json={"tenure": -1})
    assert response.status_code == 422