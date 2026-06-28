
from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import pickle
import os
import json
import re

from catboost import CatBoostClassifier
from sklearn.metrics.pairwise import cosine_similarity

from nlp.openai_client import call_openai
from nlp.emergency_detection import detect_emergency
from nlp.recommendations_engine import generate_recommendations
from nlp.question_engine import (
    search_symptoms_by_text,
    get_next_symptom_to_ask,
    format_question_with_llm,
    should_predict,
    get_current_phase,
    MAX_QUESTIONS
)

from download_models import download_all_models

# =============================================================================
# APP INIT
# =============================================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# DOWNLOAD MODELS (SAFE FOR RENDER)
# =============================================================================

if os.environ.get("RENDER"):
    download_all_models()

# =============================================================================
# LOAD ML MODEL
# =============================================================================

model = CatBoostClassifier()
model.load_model(os.path.join(MODEL_DIR, "disease_prediction_model.cbm"))

symptom_columns = pickle.load(
    open(os.path.join(MODEL_DIR, "symptom_columns.pkl"), "rb")
)

disease_classes = pickle.load(
    open(os.path.join(MODEL_DIR, "disease_classes.pkl"), "rb")
)

# =============================================================================
# LAZY NLP LOADING (IMPORTANT FOR RENDER MEMORY)
# =============================================================================

symptom_list = None
symptom_embeddings = None
embedding_model = None


def load_nlp_assets():
    global symptom_list, symptom_embeddings

    if symptom_list is None:
        with open(os.path.join(BASE_DIR, "data/symptom_list.json")) as f:
            symptom_list = json.load(f)

    if symptom_embeddings is None:
        with open(os.path.join(BASE_DIR, "data/symptom_embeddings.pkl"), "rb") as f:
            symptom_embeddings = pickle.load(f)


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model


# =============================================================================
# SYMPTOM NLP
# =============================================================================

def extract_symptoms_llm(text):
    prompt = f"""Extract ONLY symptoms from text.
Return comma separated list only.

Text: {text}
Output:"""
    result = call_openai(prompt)
    return [s.strip() for s in result.split(",")]


def clean_llm_output(symptoms):
    cleaned = []
    for s in symptoms:
        s = re.sub(r'[^a-z\s]', '', s.lower()).strip()
        if len(s) > 2:
            cleaned.append(s)
    return cleaned


def map_symptom(symptom):
    load_nlp_assets()

    emb = get_embedding_model().encode([symptom])
    sim = cosine_similarity(emb, symptom_embeddings)[0]

    top_idx = np.argsort(sim)[-5:][::-1]
    candidates = [symptom_list[i] for i in top_idx]

    prompt = f"""
Map symptom to closest match.

Symptom: {symptom}
Options: {candidates}

Return ONLY one option.
"""

    mapped = call_openai(prompt).strip()

    return mapped if mapped in candidates else candidates[0]


def normalize_symptoms(raw):
    return list(dict.fromkeys([map_symptom(s) for s in raw]))


# =============================================================================
# PREDICTION
# =============================================================================

def predict_disease(symptoms):
    vec = np.zeros(len(symptom_columns))

    for s in symptoms:
        if s in symptom_columns:
            vec[symptom_columns.index(s)] = 1

    probs = model.predict_proba(vec.reshape(1, -1))[0]

    results = [
        {"disease": disease_classes[i], "confidence": float(p)}
        for i, p in enumerate(probs)
    ]

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:5]


# =============================================================================
# SYMPTOM SEARCH
# =============================================================================

@app.route("/symptoms/search")
def symptoms_search():
    try:
        q = request.args.get("q", "")
        if len(q) < 2:
            return jsonify({"results": []})

        return jsonify({"results": search_symptoms_by_text(q, 10)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# START ASSESSMENT
# =============================================================================

@app.route("/assess/start", methods=["POST"])
def assess_start():
    try:
        data = request.get_json()
        selected = data.get("selected_symptoms", [])

        if not selected:
            return jsonify({"error": "No symptoms"}), 400

        text = " ".join(selected)

        emergency = detect_emergency(text, selected, selected)

        if emergency["is_emergency"]:
            return jsonify({"status": "emergency", **emergency})

        next_symptom = get_next_symptom_to_ask(
            selected, [], [], None
        )

        if not next_symptom:
            return jsonify({
                "status": "predicted",
                "predictions": predict_disease(selected)
            })

        q = format_question_with_llm(next_symptom, selected, "collect")

        return jsonify({
            "status": "question",
            "symptom": next_symptom,
            "question": q["question"],
            "options": q["options"],
            "confirmed_symptoms": selected,
            "absent_symptoms": [],
            "asked_symptoms": [next_symptom],
            "questions_asked": 1
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ANSWER FLOW (COLLECT + DISCRIMINATE)
# =============================================================================

@app.route("/assess/answer", methods=["POST"])
def assess_answer():
    try:
        d = request.get_json()

        symptom = d.get("symptom")
        answer = d.get("answer")
        confirmed = d.get("confirmed_symptoms", [])
        absent = d.get("absent_symptoms", [])
        asked = d.get("asked_symptoms", [])
        q_count = d.get("questions_asked", 0)

        if answer == "Yes" and symptom not in confirmed:
            confirmed.append(symptom)
        elif answer == "No":
            absent.append(symptom)

        if symptom not in asked:
            asked.append(symptom)

        phase = get_current_phase(confirmed, q_count)

        if phase == "collect":
            next_symptom = get_next_symptom_to_ask(
                confirmed, absent, asked, None
            )

            if not next_symptom:
                preds = predict_disease(confirmed)
                return jsonify({
                    "status": "predicted",
                    "predictions": preds
                })

            q = format_question_with_llm(next_symptom, confirmed, "collect")

            return jsonify({
                "status": "question",
                "symptom": next_symptom,
                "question": q["question"],
                "options": q["options"],
                "questions_asked": q_count + 1
            })

        # DISCRIMINATE PHASE
        preds = predict_disease(confirmed)

        ready, reason = should_predict(confirmed, q_count, preds)

        if ready:
            return jsonify({
                "status": "predicted",
                "predictions": preds,
                "reason": reason
            })

        next_symptom = get_next_symptom_to_ask(
            confirmed, absent, asked, preds
        )

        if not next_symptom:
            return jsonify({
                "status": "predicted",
                "predictions": preds
            })

        q = format_question_with_llm(next_symptom, confirmed, "discriminate")

        return jsonify({
            "status": "question",
            "symptom": next_symptom,
            "question": q["question"],
            "options": q["options"],
            "questions_asked": q_count + 1
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# EXPLAIN
# =============================================================================

@app.route("/assess/explain", methods=["POST"])
def explain():
    try:
        d = request.get_json()

        prompt = f"""
Explain disease {d.get('disease')} simply.
Symptoms: {d.get('symptoms')}
Confidence: {d.get('confidence')}
"""

        return jsonify({
            "explanation": call_openai(prompt)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# FEEDBACK
# =============================================================================

@app.route("/assess/feedback", methods=["POST"])
def feedback():
    try:
        d = request.get_json()

        prompt = f"""
Evaluate patient feedback:
Disease: {d.get('predicted_disease')}
Comment: {d.get('user_comment')}
"""

        raw = call_openai(prompt)

        return jsonify({"result": raw})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# RECOMMENDATIONS
# =============================================================================

@app.route("/assess/recommendations", methods=["POST"])
def recommendations():
    try:
        d = request.get_json()

        return jsonify(generate_recommendations(
            d.get("disease"),
            d.get("symptoms"),
            d.get("confidence"),
            d.get("user", {}).get("age"),
            d.get("user", {}).get("gender"),
            d.get("section", "diet")
        ))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# PREDICT LEGACY
# =============================================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        symptoms = request.get_json().get("symptoms", [])
        preds = predict_disease(symptoms)
        return jsonify({"predictions": preds})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# HEALTH
# =============================================================================

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)