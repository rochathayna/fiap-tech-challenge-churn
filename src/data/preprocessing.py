import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

FEATURES_NUM = ['tenure', 'MonthlyCharges', 'TotalCharges']
FEATURES_CAT = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ('num', StandardScaler(), FEATURES_NUM),
        ('cat', OneHotEncoder(
            handle_unknown='ignore', sparse_output=False
        ), FEATURES_CAT)
    ])

def load_and_clean(path: str) -> pd.DataFrame:
    logger.info(f"Carregando dataset de {path}")
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna().copy()
    logger.info(f"Dataset carregado: {df.shape}")
    return df