import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

def fmt(x):
    # formatea con separador de miles y luego convierte espacios en no-quiebres
    return f"{x:,.0f}".replace(" ", "\u00A0")

def predict_via_api(payload):
    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        # If the status code is not 2xx, raise an exception to trigger the except block.
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.HTTPError as http_err:
        # 4xx/5xx errors
        return None, f"HTTP error: {resp.status_code} – {resp.text}"
    except requests.exceptions.Timeout:
        return None, "The request took too long and timed out. Please try again later."
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the API server. Please check if the server is running."
    except Exception as err:
        return None, f"Unexpected error: {err}"

def main():
    st.title("🧮 Salary Prediction Demo")

    with st.form("predict_form"):
        # 1) Age
        age = st.selectbox("Age *", list(range(18,71)), index=22)
        # 2) Gender
        gender = st.selectbox("Gender *", ["Male","Female"])
        # 3) Education Level
        education = st.selectbox("Education Level *", ["Bachelor's","Master's","PhD"])
        # 4) Job Title
        common = ["Data Analyst","Business Analyst","Software Engineer",
                  "Sales Associate","Project Manager","Other..."]
        jt_choice = st.selectbox("Job Title *", common)
        if jt_choice == "Other...":
            job_title = st.text_input("Enter your Job Title *", max_chars=256)
        else:
            job_title = jt_choice
        # 5) Years of Experience
        years_exp = st.slider("Years of Experience *", 0, 50, 1)
        # 6) Professional Description
        placeholder = (
          "I am a 36-year-old female Sales Associate with a Bachelor's degree "
          "and seven years of experience in the field…"
        )
        description = st.text_area(
            "Brief Professional Description *",
            value="", 
            placeholder=placeholder,
            max_chars=1000,
            height=200
        )
        st.caption(f"{len(description)}/1000 chars")

        # This button only returns True once per form submission:
        submitted = st.form_submit_button("🔍 Predict via API")

    # only run when user hits the button
    if submitted:
        errors = []
        if not description.strip():
            errors.append("🚫 **Description** is required.")

        if errors:
            for msg in errors:
                st.error(msg)
        else:
            payload = {
                "age": age,
                "gender": gender,
                "education_level": education,
                "job_title": job_title,
                "years_of_experience": years_exp,
                "description": description,
            }
            result, error = predict_via_api(payload)
            if error:
                st.error(error)
            else:
                salary = result["predicted_salary"]
                ci_low, ci_high = result["confidence_interval"]
                st.success(f"🎯 Predicted salary: **${salary:,.0f}**")
                st.info(f"💬 Confidence interval: [{fmt(ci_low)} – {fmt(ci_high)}]")
                st.balloons()

if __name__ == "__main__":
    main()
