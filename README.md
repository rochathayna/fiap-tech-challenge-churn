<h1 align="center">🔮 Churn Prediction — FIAP Tech Challenge</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/MLflow-2.10-blue?logo=mlflow" />
  <img src="https://img.shields.io/badge/testes-6%20passando-brightgreen?logo=pytest" />
</p>

<p align="center">
  Pipeline end-to-end de previsão de churn para telecomunicações.<br/>
  EDA → Baselines → MLP PyTorch → API FastAPI → MLflow Tracking
</p>

---

## 📋 Sobre o Projeto

Uma operadora de telecomunicações enfrenta alta taxa de cancelamento de clientes.
Este projeto constrói um modelo preditivo de churn do zero — desde a exploração dos dados
até uma API de inferência em produção — aplicando boas práticas de engenharia de ML.

**Dataset:** [Telco Customer Churn — IBM/Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7.032 clientes | 18 features | 26.5% de churn

---

## 📊 Resultados

| Modelo | AUC-ROC | F1 | PR-AUC |
|---|---|---|---|
| DummyClassifier | 0.492 | 0.255 | 0.264 |
| Logistic Regression | 0.845 | 0.600 | 0.659 |
| **MLP PyTorch** | **0.820** | **0.519** | **0.583** |

**Principais fatores de risco identificados:**
- Contrato mês a mês (~42% de churn)
- Serviço Fiber optic sem suporte técnico
- Clientes com menos de 12 meses de contrato

---

## 🏗️ Arquitetura

Input (44 features)
↓
Linear(64) + ReLU + Dropout(0.3)
↓
Linear(32) + ReLU + Dropout(0.3)
↓
Linear(1) + Sigmoid
↓
Probabilidade de Churn

---

## 🚀 Setup

### Pré-requisitos
- Python 3.10+
- Git

### 1. Clonar o repositório
```bash
git clone https://github.com/rochathayna/fiap-tech-challenge-churn.git
cd fiap-tech-challenge-churn
```

### 2. Instalar dependências
```bash
pip install -e ".[dev]"
```

### 3. Baixar o dataset
Baixar o [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e salvar em `data/raw/`.

### 4. Rodar os notebooks
```bash
jupyter notebook
```
Execute `notebooks/01_eda_baselines.ipynb` e depois `notebooks/02_mlp_pytorch.ipynb`.

### 5. Rodar a API
```bash
uvicorn src.api.main:app --reload
```
Acesse a documentação em: http://localhost:8000/docs

### 6. Rodar os testes
```bash
pytest tests/ -v
```

---

## 📁 Estrutura do Projeto

fiap-tech-challenge-churn/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app
│   │   └── schemas.py       # Pydantic schemas
│   ├── data/
│   │   └── preprocessing.py # Pipeline sklearn
│   ├── models/
│   │   └── mlp.py           # MLP PyTorch
│   └── utils/
├── data/
│   ├── raw/                 # Dataset original
│   └── processed/           # Dataset limpo
├── models/                  # Artefatos (.pt, .pkl)
├── notebooks/
│   ├── 01_eda_baselines.ipynb
│   └── 02_mlp_pytorch.ipynb
├── tests/
│   ├── test_smoke.py
│   └── test_api.py
├── docs/
│   └── model_card.md
├── pyproject.toml
├── Makefile
└── README.md

---

## 🔌 API

### `GET /health`
```json
{"status": "ok", "version": "1.0.0"}
```

### `POST /predict`
```json
{
  "tenure": 12,
  "MonthlyCharges": 65.0,
  "TotalCharges": 780.0,
  "Contract": "Month-to-month",
  ...
}
```
**Resposta:**
```json
{
  "churn_probability": 0.7821,
  "churn_label": true,
  "model_version": "1.0.0"
}
```

---

## 📄 Documentação

- [Model Card](docs/model_card.md) — limitações, vieses e plano de monitoramento

---

## 👩‍💻 Autora

**Thayná Rocha** — FIAP Pós Tech Machine Learning Engineering