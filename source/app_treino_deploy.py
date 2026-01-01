# ============================================================
# Regressão Linear com Pipeline + FastAPI
# Pronto para produção
# ============================================================

# -------------------------
# 1. Imports
# -------------------------
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------
# 2. Configurações
# -------------------------
MODEL_PATH = Path("models/model_charges.joblib")
DATA_PATH = Path("dataset/insurance.csv")
DEFAULT_HEIGHT_M = 1.70

NUM_FEATURES = ["age", "bmi", "children"]
CAT_FEATURES = ["sex", "smoker", "region"]
TARGET = "charges"

# -------------------------
# 3. Treinamento do modelo
# -------------------------

def compute_bmi(weight_kg: pd.Series, height_m: pd.Series) -> pd.Series:
    return weight_kg / (height_m ** 2)


def train_and_save_model(file_path: Path):
    data = pd.read_csv(file_path)

    data["height_m"] = DEFAULT_HEIGHT_M
    data["weight_kg"] = data["bmi"] * (data["height_m"] ** 2)
    data["bmi"] = compute_bmi(data["weight_kg"], data["height_m"])

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

model = None


class PatientInput(BaseModel):
    age: int
    weight_kg: float
    height_m: float
    children: int
    sex: str        # male / female
    smoker: str     # yes / no
    region: str     # southwest, southeast, northwest, northeast


def _ensure_sklearn_backward_compatibility() -> None:
    """
    Some serialized models created with older scikit-learn versions expect the
    private class `_RemainderColsList`, which was removed in newer releases.
    If the attribute is missing, we register a tiny shim so `joblib.load`
    can unpickle the pipeline without errors.
    """
    try:
        import sklearn.compose._column_transformer as ct
    except Exception:
        return

    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList


def load_model() -> None:
    global model

    if model is not None:
        return

    if not MODEL_PATH.exists():
        if DATA_PATH.exists():
            train_and_save_model(DATA_PATH)
        else:
            raise RuntimeError(
                f"Nenhum modelo encontrado em {MODEL_PATH} e arquivo de treino ausente em {DATA_PATH}."
            )

    try:
        _ensure_sklearn_backward_compatibility()
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Arquivo de modelo não localizado em {MODEL_PATH}. Verifique a etapa de treinamento."
        ) from exc
    except Exception as exc:  # pragma: no cover - fallback defensivo
        raise RuntimeError(f"Falha ao carregar o modelo: {exc}") from exc


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: PatientInput):
    load_model()

    input_dict = input_data.dict()
    input_dict["bmi"] = compute_bmi(
        pd.Series([input_dict["weight_kg"]]), pd.Series([input_dict["height_m"]])
    )[0]
    df = pd.DataFrame([{
        "age": input_dict["age"],
        "bmi": input_dict["bmi"],
        "children": input_dict["children"],
        "sex": input_dict["sex"],
        "smoker": input_dict["smoker"],
        "region": input_dict["region"],
    }])
    prediction = model.predict(df)[0]

    return {
        "predicted_charges": round(float(prediction), 2)
    }


# -------------------------
# 5. Execução local
# -------------------------
# uvicorn source.app_treino_deploy:app --reload

# Para treinar o modelo:
# from source.app_treino_deploy import train_and_save_model, DATA_PATH
# train_and_save_model(DATA_PATH)
