import pandas as pd
import pandera as pa

DATA_PATH = (
    r'C:\Users\thayn\fiap-tech-challenge-churn'
    r'\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv'
)

schema = pa.DataFrameSchema({
    "tenure": pa.Column(int, pa.Check.ge(0)),
    "MonthlyCharges": pa.Column(float, pa.Check.ge(0)),
    "TotalCharges": pa.Column(float, pa.Check.ge(0)),
    "Churn": pa.Column(str, pa.Check.isin(["Yes", "No"])),
})


def test_dataset_schema():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )
    df = df.dropna()
    schema.validate(df)