\# Churn Prediction — FIAP Tech Challenge



Rede neural MLP para previsão de churn em telecomunicações.

Pipeline completo: EDA → Baselines → MLP PyTorch → API FastAPI.



\## Tecnologias

\- PyTorch — rede neural MLP

\- Scikit-Learn — pré-processamento e baselines

\- MLflow — tracking de experimentos

\- FastAPI — API de inferência

\- Pytest — testes automatizados



\## Resultados

| Modelo | AUC-ROC | F1 | PR-AUC |

|--------|---------|-----|--------|

| Dummy | 0.492 | 0.255 | 0.264 |

| Logistic Regression | 0.845 | 0.600 | 0.659 |

| MLP (PyTorch) | 0.820 | 0.519 | 0.583 |



\## Setup



\### 1. Clonar o repositório

```bash

git clone https://github.com/rochathayna/fiap-tech-challenge-churn.git

cd fiap-tech-challenge-churn

```



\### 2. Instalar dependências

```bash

pip install -e ".\[dev]"

```



\### 3. Baixar dataset

Baixar o dataset Telco Customer Churn do Kaggle e salvar em `data/raw/`.



\### 4. Treinar o modelo

```bash

python notebooks/02\_mlp\_pytorch.ipynb

```



\### 5. Rodar a API

```bash

uvicorn src.api.main:app --reload

```



\### 6. Rodar os testes

```bash

pytest tests/ -v

```



\## Estrutura

├── src/

│   ├── api/          # FastAPI

│   ├── data/         # Pré-processamento

│   ├── models/       # MLP PyTorch

│   └── utils/        # Logging

├── data/

│   ├── raw/          # Dataset original

│   └── processed/    # Dataset limpo

├── models/           # Artefatos salvos

├── notebooks/        # EDA e treinamento

├── tests/            # Testes automatizados

└── docs/             # Model Card


## Documentação

\- \[Model Card](docs/model\_card.md)

\- API Docs: http://localhost:8000/docs



\## Autora

Thayná Rocha — FIAP Pós Tech

