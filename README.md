# ML-regression-analysis-from-zero-to-deploy
Projeto de Machine Learning com uma tarefa de regressão e posteriormente deploy em cloud.

Possíveis projetos:

# Insurance
source: https://www.kaggle.com/datasets/mirichoi0218/insurance/data

## Como treinar o modelo

O serviço FastAPI verifica se o arquivo `model_charges.joblib` existe ao iniciar. Caso não exista, ele tenta treinar automaticamente usando `dataset/insurance.csv`. Você também pode treinar manualmente com o comando abaixo:

```bash
python - <<'PY'
from pathlib import Path
from source.app_treino_deploy import train_and_save_model, DATA_PATH

train_and_save_model(DATA_PATH)
print(f"Modelo salvo em {Path('model_charges.joblib').resolve()}")
PY
```

## Como executar a API

```bash
uvicorn source.app_treino_deploy:app --reload
```

## Como executar a interface Streamlit

```bash
streamlit run source/ui_app.py
```

A interface coleta idade, sexo, indicador de fumo, peso (kg), altura (m), número de filhos e região. Ela calcula o IMC localmente para exibição e envia os dados para a rota `/predict` da API FastAPI, exibindo o custo previsto na tela.
