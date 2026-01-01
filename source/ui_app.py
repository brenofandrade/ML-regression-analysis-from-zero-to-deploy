import requests
import streamlit as st

API_URL_DEFAULT = "http://localhost:8000"


def compute_bmi(weight_kg: float, height_m: float) -> float:
    if height_m <= 0:
        raise ValueError("Height must be greater than zero to compute BMI.")
    return weight_kg / (height_m ** 2)


def call_prediction_api(api_url: str, payload: dict) -> float:
    response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data.get("predicted_charges")


def main():
    st.set_page_config(page_title="Medical Charges Predictor", page_icon="💡")
    st.title("Medical Charges Predictor")
    st.write(
        "Forneça os dados do paciente para estimar o custo de seguro. O app envia os "
        "dados para a API FastAPI e mostra o resultado retornado."
    )

    api_url = st.text_input("URL da API", value=API_URL_DEFAULT)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Idade", min_value=0, max_value=120, value=30, step=1)
        weight_kg = st.number_input(
            "Peso (kg)", min_value=0.0, max_value=400.0, value=75.0, step=0.1
        )
        children = st.number_input(
            "Número de filhos", min_value=0, max_value=10, value=0, step=1
        )
    with col2:
        height_m = st.number_input(
            "Altura (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01
        )
        sex = st.selectbox("Sexo", options=["male", "female"])
        smoker = st.selectbox("Fumante?", options=["yes", "no"])
        region = st.selectbox(
            "Região",
            options=["southwest", "southeast", "northwest", "northeast"],
        )

    bmi = None
    bmi_error = None
    try:
        bmi = compute_bmi(weight_kg, height_m)
    except ValueError as exc:
        bmi_error = str(exc)

    bmi_container = st.empty()
    if bmi_error:
        bmi_container.error(bmi_error)
    elif bmi is not None:
        bmi_container.info(f"IMC calculado: {bmi:.2f}")

    if st.button("Calcular custo"):
        try:
            payload = {
                "age": int(age),
                "weight_kg": float(weight_kg),
                "height_m": float(height_m),
                "children": int(children),
                "sex": sex,
                "smoker": smoker,
                "region": region,
            }
            predicted = call_prediction_api(api_url, payload)
            st.success(f"Custo previsto: ${predicted:.2f}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Erro ao chamar a API: {exc}")
        except Exception as exc:  # pragma: no cover - feedback para UI
            st.error(f"Falha ao processar a previsão: {exc}")


if __name__ == "__main__":
    main()
