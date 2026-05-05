from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, le=100)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_label: bool
    model_version: str = "1.0.0"