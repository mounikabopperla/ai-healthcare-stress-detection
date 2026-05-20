import streamlit as st
import pandas as pd
import numpy as np
import joblib
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Healthcare Stress Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Powered Healthcare Stress Level Detection")
st.write("Machine Learning + RAG + LLM powered stress prediction system")

# ============================================================
# GROQ API KEY
# ============================================================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# LOAD SAVED MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("best_stress_model_pipeline.joblib")
    return model

best_pipeline = load_model()

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_feature_engineering(df):
    df = df.copy()

    if "Work_Hours" in df.columns and "Sleep_Duration" in df.columns:
        df["WorkLifeRatio"] = (
            df["Work_Hours"] /
            df["Sleep_Duration"].replace(0, np.nan)
        )

    if "Screen_Time" in df.columns and "Work_Hours" in df.columns:
        df["ScreenStressScore"] = (
            df["Screen_Time"] *
            df["Work_Hours"]
        )

    health_cols = []

    for col in ["Blood_Pressure", "Caffeine_Intake"]:
        if col in df.columns:
            health_cols.append(col)

    if len(health_cols) > 0:
        df["HealthRiskIndex"] = (
            df[health_cols].mean(axis=1)
        )

    return df

# ============================================================
# RAG KNOWLEDGE BASE
# ============================================================

health_knowledge = [
    "Poor sleep quality and short sleep duration can increase stress and reduce mental recovery.",
    "Long work hours are linked with burnout, fatigue, and higher stress levels.",
    "High screen time can increase mental fatigue and disturb sleep patterns.",
    "Regular physical activity can reduce stress and improve mood regulation.",
    "Meditation and breathing exercises can help reduce stress and improve emotional control.",
    "High caffeine intake may increase anxiety, disturb sleep, and worsen stress symptoms.",
    "High blood pressure can be associated with chronic stress and lifestyle imbalance."
]

# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

@st.cache_resource
def build_rag_collection():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name="stress_health_knowledge"
    )

    existing = collection.count()

    if existing == 0:
        embeddings = embedding_model.encode(health_knowledge).tolist()

        collection.add(
            documents=health_knowledge,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(health_knowledge))]
        )

    return collection, embedding_model

rag_collection, embedding_model = build_rag_collection()

# ============================================================
# BUILD RAG QUERY
# ============================================================

def build_rag_query(user_data, predicted_stress):
    query_parts = [f"{predicted_stress} stress"]

    if user_data.get("Sleep_Duration", 8) < 7:
        query_parts.append("poor sleep sleep deprivation")

    if user_data.get("Work_Hours", 8) > 9:
        query_parts.append("long work hours burnout")

    if user_data.get("Screen_Time", 4) > 6:
        query_parts.append("high screen time mental fatigue")

    if user_data.get("Caffeine_Intake", 1) >= 3:
        query_parts.append("high caffeine anxiety sleep disturbance")

    if user_data.get("Physical_Activity", 3) < 3:
        query_parts.append("low exercise physical inactivity")

    if user_data.get("Blood_Pressure", 120) >= 130:
        query_parts.append("high blood pressure chronic stress")

    if str(user_data.get("Meditation_Practice", "No")).lower() in ["no", "0"]:
        query_parts.append("no meditation stress management")

    return " ".join(query_parts)

# ============================================================
# RETRIEVE RAG CONTEXT
# ============================================================

def retrieve_rag_context(user_data, predicted_stress, top_k=3):
    query = build_rag_query(user_data, predicted_stress)

    query_embedding = embedding_model.encode([query]).tolist()

    results = rag_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]

    return retrieved_docs, query

# ============================================================
# HELPER FUNCTIONS FOR CLEAN OUTPUT
# ============================================================

def format_value(value):
    """
    Rounds long decimal numbers so the AI report looks cleaner.
    """
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return round(float(value), 2)
    except:
        pass

    return value


def parse_time_to_minutes(time_value):
    try:
        dt = pd.to_datetime(time_value)
        return dt.hour * 60 + dt.minute
    except:
        return np.nan

# ============================================================
# LLM REPORT GENERATION
# ============================================================

def generate_llm_report(
    user_name,
    user_data,
    prediction,
    probabilities,
    rag_docs
):
    risk_features = []

    important_features = [
        "WorkLifeRatio",
        "ScreenStressScore",
        "HealthRiskIndex",
        "Sleep_Duration",
        "Work_Hours",
        "Screen_Time",
        "Blood_Pressure"
    ]

    for feature in important_features:
        if feature in user_data:
            risk_features.append(
                f"{feature}: {format_value(user_data[feature])}"
            )

    risk_text = "\n".join(risk_features[:5])

    prob_text = "\n".join([
        f"{k}: {v * 100:.1f}%"
        for k, v in probabilities.items()
    ])

    rag_text = "\n\n".join(rag_docs)

    prompt = f"""
You are an AI healthcare stress analyst.

User Name:
{user_name}

Predicted Stress Level:
{prediction}

Model Confidence:
{prob_text}

Top Risk Features:
{risk_text}

Medical Knowledge Retrieved by RAG:
{rag_text}

Task:
Write a personalized stress report.

Requirements:
1. Clearly mention the predicted stress level.
2. Explain why stress was predicted.
3. Mention actual risk factors using clean rounded numbers.
4. Give 3 practical suggestions.
5. Avoid generic advice.
6. Keep the report simple and readable.
7. Keep under 150 words.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=350
    )

    return response.choices[0].message.content

# ============================================================
# USER INPUT UI
# ============================================================

st.subheader("Enter Healthcare Details")

col1, col2, col3 = st.columns(3)

with col1:
    user_name = st.text_input("Name", "Test User")
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    occupation = st.selectbox(
        "Occupation",
        ["Student", "Software Engineer", "Healthcare", "Teacher", "Other"]
    )
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Other"])
    sleep_duration = st.number_input(
        "Sleep Duration",
        min_value=0.0,
        max_value=15.0,
        value=6.0
    )
    sleep_quality = st.number_input(
        "Sleep Quality",
        min_value=1.0,
        max_value=5.0,
        value=3.0
    )

with col2:
    wake_up_time = st.text_input("Wake Up Time", "7:00 AM")
    bed_time = st.text_input("Bed Time", "10:00 PM")
    physical_activity = st.number_input(
        "Physical Activity",
        min_value=0.0,
        max_value=30.0,
        value=3.0
    )
    screen_time = st.number_input(
        "Screen Time",
        min_value=0.0,
        max_value=24.0,
        value=6.0
    )
    caffeine = st.number_input("Caffeine Intake", min_value=0, max_value=10, value=2)
    alcohol = st.number_input("Alcohol Intake", min_value=0, max_value=10, value=0)
    smoking = st.selectbox("Smoking Habit", ["No", "Yes"])

with col3:
    work_hours = st.number_input(
        "Work Hours",
        min_value=0.0,
        max_value=24.0,
        value=8.0
    )
    travel_time = st.number_input(
        "Travel Time",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )
    social_interactions = st.number_input(
        "Social Interactions",
        min_value=0,
        max_value=20,
        value=5
    )
    meditation = st.selectbox("Meditation Practice", ["Yes", "No"])
    exercise_type = st.selectbox(
        "Exercise Type",
        ["Cardio", "Yoga", "Strength", "None", "Other"]
    )
    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=80,
        max_value=200,
        value=120
    )
    cholesterol = st.number_input(
        "Cholesterol Level",
        min_value=100,
        max_value=300,
        value=180
    )
    blood_sugar = st.number_input(
        "Blood Sugar Level",
        min_value=60,
        max_value=250,
        value=90
    )

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict Stress Level"):

    input_data = {
        "Age": age,
        "Gender": gender,
        "Occupation": occupation,
        "Marital_Status": marital_status,
        "Sleep_Duration": sleep_duration,
        "Sleep_Quality": sleep_quality,
        "Wake_Up_Time": wake_up_time,
        "Bed_Time": bed_time,
        "Physical_Activity": physical_activity,
        "Screen_Time": screen_time,
        "Caffeine_Intake": caffeine,
        "Alcohol_Intake": alcohol,
        "Smoking_Habit": smoking,
        "Work_Hours": work_hours,
        "Travel_Time": travel_time,
        "Social_Interactions": social_interactions,
        "Meditation_Practice": meditation,
        "Exercise_Type": exercise_type,
        "Blood_Pressure": blood_pressure,
        "Cholesterol_Level": cholesterol,
        "Blood_Sugar_Level": blood_sugar,
    }

    input_df = pd.DataFrame([input_data])

    input_df = add_feature_engineering(input_df)

    input_df["BedTime_Minutes"] = input_df["Bed_Time"].apply(parse_time_to_minutes)
    input_df["WakeTime_Minutes"] = input_df["Wake_Up_Time"].apply(parse_time_to_minutes)

    prediction = best_pipeline.predict(input_df)[0]

    if hasattr(best_pipeline, "predict_proba"):
        prob_array = best_pipeline.predict_proba(input_df)[0]
        model_classes = best_pipeline.classes_
        probabilities = dict(zip(model_classes, prob_array))
        confidence = round(max(probabilities.values()) * 100, 2)
    else:
        probabilities = {prediction: 1.0}
        confidence = 100.0

    user_data = input_df.iloc[0].to_dict()

    rag_docs, rag_query = retrieve_rag_context(
        user_data=user_data,
        predicted_stress=prediction,
        top_k=3
    )

    ai_report = generate_llm_report(
        user_name=user_name,
        user_data=user_data,
        prediction=prediction,
        probabilities=probabilities,
        rag_docs=rag_docs
    )

    # Show stress level with color
    if prediction == "Low":
        st.success(f"Predicted Stress Level: {prediction}")
    elif prediction == "Medium":
        st.warning(f"Predicted Stress Level: {prediction}")
    else:
        st.error(f"Predicted Stress Level: {prediction}")

    # Show confidence label
    if confidence >= 80:
        confidence_label = "High Confidence"
    elif confidence >= 50:
        confidence_label = "Moderate Confidence"
    else:
        confidence_label = "Low Confidence"

    st.metric(
        "Model Confidence",
        f"{confidence_label} ({confidence}%)"
    )

    # Show probabilities as percentages
    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame([probabilities]) * 100
    prob_df = prob_df.round(2).astype(str) + "%"

    st.dataframe(prob_df)

    st.subheader("Retrieved Healthcare Knowledge")
    for i, doc in enumerate(rag_docs, start=1):
        st.write(f"{i}. {doc}")

    st.subheader("AI Healthcare Report")
    st.write(ai_report)

    st.markdown("---")
    st.caption(
        "AI Powered Healthcare Stress Detection System | Built using ML + RAG + LLM + Streamlit"
    )
