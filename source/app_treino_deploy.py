# ============================================================
# Regressão Linear com Pipeline + FastAPI
# Pronto para produção
# ============================================================

# -------------------------
# 1. Imports
# -------------------------
import joblib
import pandas as pd
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# 2. Configurações
# -------------------------
MODEL_PATH = "model_charges.joblib"

NUM_FEATURES = ["age", "bmi", "children"]
CAT_FEATURES = ["sex", "smoker", "region"]
TARGET = "charges"

# -------------------------
# 3. Treinamento do modelo
# -------------------------

def train_and_save_model(file_path: str):
    data = pd.read_csv(file_path)

    X = data[NUM_FEATURES + CAT_FEATURES]
    y = data[TARGET]

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUM_FEATURES),
        ("cat", categorical_pipeline, CAT_FEATURES)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R2:", r2_score(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print("Modelo salvo em", MODEL_PATH)


# -------------------------
# 4. FastAPI
# -------------------------
app = FastAPI(title="Medical Charges Prediction API")

model = joblib.load(MODEL_PATH)


class PatientInput(BaseModel):
    age: int
    bmi: float
    children: int
    sex: str        # male / female
    smoker: str     # yes / no
    region: str     # southwest, southeast, northwest, northeast


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: PatientInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)[0]

    return {
        "predicted_charges": round(float(prediction), 2)
    }


# -------------------------
# 5. Execução local
# -------------------------
# uvicorn regressao_linear_api_fastapi:app --reload

# Para treinar o modelo:
# train_and_save_model("insurance.csv")
