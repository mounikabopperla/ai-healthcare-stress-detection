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

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name="stress_health_knowledge"
    )

    existing = collection.count()

    if existing == 0:

        embeddings = embedding_model.encode(
            health_knowledge
        ).tolist()

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
        query_parts.append(
            "poor sleep sleep deprivation"
        )

    if user_data.get("Work_Hours", 8) > 9:
        query_parts.append(
            "long work hours burnout"
        )

    if user_data.get("Screen_Time", 4) > 6:
        query_parts.append(
            "high screen time mental fatigue"
        )

    if user_data.get("Caffeine_Intake", 1) >= 3:
        query_parts.append(
            "high caffeine anxiety sleep disturbance"
        )

    if user_data.get("Physical_Activity", 3) < 3:
        query_parts.append(
            "low exercise physical inactivity"
        )

    if user_data.get("Blood_Pressure", 120) >= 130:
        query_parts.append(
            "high blood pressure chronic stress"
        )

    if user_data.get("Meditation_Practice", 0) == 0:
        query_parts.append(
            "no meditation stress management"
        )

    return " ".join(query_parts)

# ============================================================
# RETRIEVE RAG CONTEXT
# ============================================================

def retrieve_rag_context(
    user_data,
    predicted_stress,
    top_k=3
):

    query = build_rag_query(
        user_data,
        predicted_stress
    )

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = rag_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]

    return retrieved_docs, query

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

    if client is None:
        return "Please enter Groq API Key."

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
                f"{feature}: {user_data[feature]}"
            )

    risk_text = "\n".join(
        risk_features[:5]
    )

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
1. Explain why stress was predicted.
2. Mention actual risk factors.
3. Give 3 suggestions.
4. Avoid generic advice.
5. Keep under 150 words.
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

    user_name = st.text_input(
        "Name",
        "Test User"
    )

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=25
    )

    sleep_duration = st.number_input(
        "Sleep Duration",
        min_value=0.0,
        max_value=15.0,
        value=6.0
    )

with col2:

    work_hours = st.number_input(
        "Work Hours",
        min_value=0.0,
        max_value=24.0,
        value=8.0
    )

    screen_time = st.number_input(
        "Screen Time",
        min_value=0.0,
        max_value=24.0,
        value=6.0
    )

    caffeine = st.number_input(
        "Caffeine Intake",
        min_value=0,
        max_value=10,
        value=2
    )

with col3:

    physical_activity = st.number_input(
        "Physical Activity",
        min_value=0.0,
        max_value=30.0,
        value=3.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=80,
        max_value=200,
        value=120
    )

    meditation = st.selectbox(
        "Meditation Practice",
        ["Yes", "No"]
    )

# ============================================================
# CONVERT VALUES
# ============================================================

meditation_value = (
    1 if meditation == "Yes"
    else 0
)

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict Stress Level"):

    input_data = {

        "Age": age,

        "Sleep_Duration": sleep_duration,

        "Work_Hours": work_hours,

        "Screen_Time": screen_time,

        "Caffeine_Intake": caffeine,

        "Physical_Activity": physical_activity,

        "Blood_Pressure": blood_pressure,

        "Meditation_Practice": meditation_value
    }

    input_df = pd.DataFrame([input_data])

    input_df = add_feature_engineering(
        input_df
    )

    prediction = best_pipeline.predict(
        input_df
    )[0]

    if hasattr(best_pipeline, "predict_proba"):

        prob_array = best_pipeline.predict_proba(
            input_df
        )[0]

        model_classes = best_pipeline.classes_

        probabilities = dict(
            zip(model_classes, prob_array)
        )

        confidence = round(
            max(probabilities.values()) * 100,
            2
        )

    else:

        probabilities = {
            prediction: 1.0
        }

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

    st.success(
        f"Predicted Stress Level: {prediction}"
    )

    st.metric(
        "Model Confidence",
        f"{confidence}%"
    )

    st.subheader("Prediction Probabilities")

    st.dataframe(
        pd.DataFrame([probabilities])
    )

    st.subheader("Retrieved Healthcare Knowledge")

    for i, doc in enumerate(rag_docs, start=1):

        st.write(f"{i}. {doc}")

    st.subheader("AI Healthcare Report")

    st.write(ai_report)