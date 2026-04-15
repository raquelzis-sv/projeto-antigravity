# Standardized ML Training Pipeline (SMTP)

## Visão Geral
Este projeto implementa um framework padronizado para o desenvolvimento, treinamento e avaliação de modelos de IA supervisionados, focado em Engenharia de Machine Learning (MLE).

## Estrutura do Projeto
- `data/`: Dados brutos e processados.
- `models/`: Local de salvamento de modelos.
- `notebooks/`: Exploração inicial e EDAs.
- `src/`: Código fonte modular.
  - `train.py`: Script de treinamento.
  - `preprocess.py`: Limpeza de dados.
  - `utils.py`: Funções auxiliares.
- `tests/`: Testes unitários.

## Como Executar
O projeto utiliza o MLflow para gerenciamento de experimentos.
```bash
mlflow run .
```
