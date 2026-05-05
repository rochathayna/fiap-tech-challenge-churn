\# Model Card — Churn Prediction MLP



\## Descrição

Rede neural MLP treinada para prever cancelamento de clientes (churn)

em uma operadora de telecomunicações.



\## Dados

\- \*\*Dataset:\*\* Telco Customer Churn (IBM/Kaggle)

\- \*\*Período:\*\* 2017

\- \*\*Registros:\*\* 7.032 clientes após limpeza

\- \*\*Features:\*\* 18 (3 numéricas, 15 categóricas)



\## Performance

| Métrica | MLP | LogReg | Dummy |

|---------|-----|--------|-------|

| AUC-ROC | 0.82 | 0.845 | 0.492 |

| F1 | 0.52 | 0.600 | 0.255 |

| PR-AUC | 0.58 | 0.659 | 0.264 |



\## Arquitetura

\- Input: 44 features (após encoding)

\- Camadas: 44 → 64 → 32 → 1

\- Ativação: ReLU + Dropout(0.3)

\- Saída: Sigmoid (probabilidade)

\- Threshold: 0.5



\## Limitações

\- Dataset de 2017 — pode não refletir comportamentos atuais

\- Dados de uma única operadora — viés geográfico/cultural

\- Desbalanceamento: 73.5% não-churn vs 26.5% churn

\- Não inclui dados de uso (ligações, dados consumidos)



\## Cenários de Falha

\- Clientes novos com comportamento atípico

\- Mudanças de mercado não refletidas nos dados históricos

\- Features fora do range de treinamento



\## Uso Correto

\- Identificar clientes em risco para ações de retenção

\- NÃO usar para decisões automáticas sem revisão humana



\## Monitoramento

\- Monitorar drift de features mensalmente

\- Retreinar se AUC-ROC cair abaixo de 0.75

\- Alertar se taxa de churn previsto divergir >10% do real

