# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

def predict_via_api(payload):
    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        # Si el status code no es 2xx, lanza excepción para entrar en except
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.HTTPError as http_err:
        # Errores 4xx / 5xx
        return None, f"HTTP error: {resp.status_code} – {resp.text}"
    except requests.exceptions.Timeout:
        return None, "La petición tardó demasiado tiempo y se agotó el tiempo de espera."
    except requests.exceptions.ConnectionError:
        return None, "No se pudo conectar con el servidor de la API."
    except Exception as err:
        return None, f"Error inesperado: {err}"
    
def main():
    st.title("Salary Prediction Demo")

    # 1) Age
    age = st.selectbox("Age", list(range(18,71)), index=22)

    # 2) Gender
    gender = st.selectbox("Gender", ["Male","Female"])

    # 3) Education Level
    education = st.selectbox("Education Level", ["Bachelor's","Master's","PhD"])

    # 4) Job Title
    common = [
      "Data Analyst","Business Analyst","Software Engineer",
      "Sales Associate","Project Manager","Other..."
    ]
    jt_choice = st.selectbox("Job Title", common)
    if jt_choice=="Other...":
        job_title = st.text_input("Enter your Job Title", max_chars=256)
    else:
        job_title = jt_choice

    # 5) Years of Experience
    years_exp = st.slider("Years of Experience", 0, 50, 1)

    # 6) Description
    placeholder = (
      "I am a 36-year-old female Sales Associate with a Bachelor's degree "
      "and seven years of experience in the field…"
    )
    description = st.text_area(
        "Brief Professional Description",
        placeholder=placeholder,
        max_chars=1000,
        height=200
    )
    st.caption(f"{len(description)}/1000 chars")

    if st.button("🔍 Predict via API"):
        payload = {
            "age": age,
            "gender": gender,
            "education_level": education,
            "job_title": job_title,
            "years_of_experience": years_exp,
            "description": description
        }
        result, error = predict_via_api(payload)
        if error:
            st.error(error)
        else:
            salary = result["predicted_salary"]
            ci_low, ci_high = result["confidence_interval"]
            st.success(f"🎯 Predicted salary: **${salary:,.0f}**")
            st.info(f"💬 Confidence interval: [${ci_low:,.0f}, ${ci_high:,.0f}]")

if __name__ == "__main__":
    main()
