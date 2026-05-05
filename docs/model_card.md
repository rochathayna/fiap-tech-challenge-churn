\# 🗂️ Model Card — Churn Prediction MLP



\## Descrição

Rede neural MLP treinada para prever cancelamento de clientes (churn)

em uma operadora de telecomunicações. O modelo classifica clientes com

risco de cancelamento para que a empresa possa tomar ações de retenção.



\---



\## 📦 Dados



| Campo | Valor |

|---|---|

| Dataset | Telco Customer Churn (IBM/Kaggle) |

| Período | 2017 |

| Registros | 7.032 clientes após limpeza |

| Features numéricas | tenure, MonthlyCharges, TotalCharges |

| Features categóricas | 15 (contrato, internet, serviços, etc.) |

| Balanceamento | 73.5% não-churn / 26.5% churn |



\---



\## 📊 Performance



| Métrica | MLP | LogReg | Dummy |

|---|---|---|---|

| AUC-ROC | 0.82 | 0.845 | 0.492 |

| F1-score | 0.52 | 0.600 | 0.255 |

| PR-AUC | 0.58 | 0.659 | 0.264 |



\---



\## 🏗️ Arquitetura



| Componente | Detalhe |

|---|---|

| Tipo | MLP (Multilayer Perceptron) |

| Input | 44 features após encoding |

| Camadas | 44 → 64 → 32 → 1 |

| Ativação | ReLU + Dropout(0.3) |

| Saída | Sigmoid (probabilidade 0-1) |

| Threshold | 0.5 |

| Framework | PyTorch 2.0 |



\---



\## ⚠️ Limitações



\- Dataset de 2017 — pode não refletir comportamentos atuais

\- Dados de uma única operadora — viés geográfico e cultural

\- Não inclui dados de uso real (ligações, dados consumidos)

\- Desempenho inferior da MLP vs Regressão Logística neste dataset



\---



\## 🚨 Cenários de Falha



\- Clientes novos com comportamento muito diferente do histórico

\- Mudanças bruscas de mercado não refletidas nos dados

\- Features fora do range visto no treinamento



\---



\## ✅ Uso Correto



\- Identificar clientes em risco para ações proativas de retenção

\- Priorizar contatos de equipe comercial

\- \*\*NÃO usar\*\* para decisões automáticas sem revisão humana



\---



\## 📡 Plano de Monitoramento



| Métrica | Frequência | Alerta |

|---|---|---|

| AUC-ROC em produção | Mensal | < 0.75 |

| Taxa de churn previsto vs real | Semanal | Divergência > 10% |

| Drift de features | Mensal | PSI > 0.2 |

| Latência da API | Contínuo | p95 > 500ms |



\*\*Playbook de resposta:\*\* Se AUC cair abaixo de 0.75, retreinar com dados mais recentes e comparar com modelo atual antes de substituir.

