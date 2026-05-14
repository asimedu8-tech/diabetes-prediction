# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import pickle
import os

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tuned_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# =========================
# HELPER FUNCTIONS
# =========================
def age_to_category(age):
    if 18 <= age <= 24:
        return 1
    elif 25 <= age <= 29:
        return 2
    elif 30 <= age <= 34:
        return 3
    elif 35 <= age <= 39:
        return 4
    elif 40 <= age <= 44:
        return 5
    elif 45 <= age <= 49:
        return 6
    elif 50 <= age <= 54:
        return 7
    elif 55 <= age <= 59:
        return 8
    elif 60 <= age <= 64:
        return 9
    elif 65 <= age <= 69:
        return 10
    elif 70 <= age <= 74:
        return 11
    elif 75 <= age <= 79:
        return 12
    else:
        return 13


def education_to_category(e):
    return {
        "Never / kindergarten": 1,
        "Elementary school": 2,
        "Junior school": 3,
        "High school Graduate": 4,
        "College": 5,
        "College graduate": 6
    }.get(e, 1)


def income_to_category(i):
    return {
        "Less than $10,000": 1,
        "$10,000 to less than $15,000": 2,
        "$15,000 to less than $20,000": 3,
        "$20,000 to less than $25,000": 4,
        "$25,000 to less than $35,000": 5,
        "$35,000 to less than $50,000": 6,
        "$50,000 to less than $75,000": 7,
        "$75,000 or more": 8
    }.get(i, 1)


# =========================
# MAIN APP
# =========================
def run():

    st.title("🩺 Diabetes Risk Prediction App")

    st.write("Fill your health details below:")

    # -------------------------
    # INPUTS
    # -------------------------
    HighBP = 1 if st.radio("High BP?", ["No", "Yes"]) == "Yes" else 0
    HighChol = 1 if st.radio("High Cholesterol?", ["No", "Yes"]) == "Yes" else 0
    CholCheck = 1 if st.radio("Cholesterol Check?", ["No", "Yes"]) == "Yes" else 0
    BMI = st.slider("BMI", 10.0, 60.0, 25.0)

    Smoker = 1 if st.radio("Smoker?", ["No", "Yes"]) == "Yes" else 0
    Stroke = 1 if st.radio("Stroke?", ["No", "Yes"]) == "Yes" else 0
    HeartDisease = 1 if st.radio("Heart Disease?", ["No", "Yes"]) == "Yes" else 0

    PhysActivity = 1 if st.radio("Physical Activity?", ["No", "Yes"]) == "Yes" else 0
    Fruits = 1 if st.radio("Fruits daily?", ["No", "Yes"]) == "Yes" else 0
    Veggies = 1 if st.radio("Veggies daily?", ["No", "Yes"]) == "Yes" else 0

    Alcohol = 1 if st.radio("Heavy Alcohol?", ["No", "Yes"]) == "Yes" else 0
    Healthcare = 1 if st.radio("Healthcare?", ["No", "Yes"]) == "Yes" else 0
    NoDoc = 1 if st.radio("No doctor due cost?", ["No", "Yes"]) == "Yes" else 0

    GenHlth = st.selectbox("General Health", [1, 2, 3, 4, 5])
    MentHlth = st.slider("Mental Health Days", 0, 30, 0)
    PhysHlth = st.slider("Physical Health Days", 0, 30, 0)

    DiffWalk = 1 if st.radio("Difficulty walking?", ["No", "Yes"]) == "Yes" else 0
    Sex = 1 if st.radio("Gender", ["Female", "Male"]) == "Male" else 0

    Age = st.number_input("Age", 1, 120, 30)
    Age_cat = age_to_category(Age)

    Education = education_to_category(
        st.selectbox("Education", [
            "Never / kindergarten",
            "Elementary school",
            "Junior school",
            "High school Graduate",
            "College",
            "College graduate"
        ])
    )

    Income = income_to_category(
        st.selectbox("Income", [
            "Less than $10,000",
            "$10,000 to less than $15,000",
            "$15,000 to less than $20,000",
            "$20,000 to less than $25,000",
            "$25,000 to less than $35,000",
            "$35,000 to less than $50,000",
            "$50,000 to less than $75,000",
            "$75,000 or more"
        ])
    )

    # -------------------------
    # PREDICTION
    # -------------------------
    if st.button("Predict"):

        input_data = pd.DataFrame({
            'HighBP': [HighBP],
            'HighChol': [HighChol],
            'CholCheck': [CholCheck],
            'BMI': [BMI],
            'Smoker': [Smoker],
            'Stroke': [Stroke],
            'HeartDiseaseorAttack': [HeartDisease],
            'PhysActivity': [PhysActivity],
            'Fruits': [Fruits],
            'Veggies': [Veggies],
            'HvyAlcoholConsump': [Alcohol],
            'AnyHealthcare': [Healthcare],
            'NoDocbcCost': [NoDoc],
            'GenHlth': [GenHlth],
            'MentHlth': [MentHlth],
            'PhysHlth': [PhysHlth],
            'DiffWalk': [DiffWalk],
            'Sex': [Sex],
            'Age': [Age_cat],
            'Education': [Education],
            'Income': [Income]
        })

        result = model.predict(input_data)

        st.subheader("Result:")

        if result[0] == 0:
            st.success("Low risk of diabetes 👍")
        else:
            st.error("High risk of diabetes ⚠️")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    run()