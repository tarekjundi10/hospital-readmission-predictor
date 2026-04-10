import streamlit as st
import requests

API_URL = "https://mental-health-predictor-t0og.onrender.com/predict"

st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="app/brain-favicon.png",
    layout="centered"
)

st.title("Mental Health Risk Predictor")
st.markdown("Enter lifestyle information below to assess mental health risk based on a machine learning model trained on 3,000 records.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=65, value=30)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    exercise_level = st.selectbox("Exercise Level", options=["Low", "Moderate", "High"])
    diet_type = st.selectbox("Diet Type", options=["Balanced", "Vegan", "Vegetarian"])
    sleep_hours = st.slider("Sleep Hours per Night", min_value=2.0, max_value=12.0, value=7.0, step=0.5)

with col2:
    work_hours = st.slider("Work Hours per Week", min_value=10, max_value=80, value=40)
    screen_time = st.slider("Screen Time per Day (Hours)", min_value=1.0, max_value=12.0, value=4.0, step=0.5)
    social_score = st.slider("Social Interaction Score", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
    happiness_score = st.slider("Happiness Score", min_value=1.0, max_value=10.0, value=6.0, step=0.1)

st.divider()

gender_map = {"Male": 1, "Female": 0}
exercise_map = {"Low": 1, "Moderate": 2, "High": 0}
diet_map = {"Balanced": 0, "Vegan": 1, "Vegetarian": 2}

if st.button("Run Prediction", use_container_width=True, type="primary"):
    payload = {
        "age": age,
        "gender": gender_map[gender],
        "exercise_level": exercise_map[exercise_level],
        "diet_type": diet_map[diet_type],
        "sleep_hours": sleep_hours,
        "work_hours_per_week": work_hours,
        "screen_time_per_day": screen_time,
        "social_interaction_score": social_score,
        "happiness_score": happiness_score
    }

    with st.spinner("Analyzing..."):
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            risk = result["risk_level"]
            confidence = result["confidence"]
            recommendation = result["recommendation"]

            st.divider()

            if risk == "High Risk":
                st.error(f"Result: {risk}")
            else:
                st.success(f"Result: {risk}")

            st.metric(label="Model Confidence", value=f"{round(confidence * 100, 1)}%")
            st.info(f"Recommendation: {recommendation}")

            st.divider()
            st.markdown("#### Risk Factors Summary")
            factors = {
                "Sleep Hours": sleep_hours,
                "Work Hours per Week": work_hours,
                "Screen Time (hrs/day)": screen_time,
                "Social Interaction Score": social_score,
                "Happiness Score": happiness_score
            }
            for factor, value in factors.items():
                st.write(f"- **{factor}:** {value}")

        except Exception as e:
            st.error(f"Could not connect to the API. Make sure it is running. Error: {e}")

st.divider()
st.caption("Model: Random Forest | F1: 0.740 | AUC: 0.807 | Dataset: Mental Health & Lifestyle Habits 2019-2024")