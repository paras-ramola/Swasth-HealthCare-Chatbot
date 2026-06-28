# application.py
# =============================================================================
# FLASK ML SERVICE
# =============================================================================


from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import pickle
import os
import json
import re

from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
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

#  IMPORTANT: import your downloader
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
# DOWNLOAD MODELS (ONLY ONCE PER STARTUP)
# =============================================================================

download_all_models()

# =============================================================================
# LOAD MODEL ARTIFACTS
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
# LOAD NLP ARTIFACTS
# =============================================================================

with open(os.path.join(BASE_DIR, "data/symptom_list.json")) as f:
    symptom_list = json.load(f)

with open(os.path.join(BASE_DIR, "data/symptom_embeddings.pkl"), "rb") as f:
    symptom_embeddings = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# =============================================================================
# SYMPTOM EXTRACTION — for legacy /predict endpoint
# =============================================================================

def extract_symptoms_llm(text: str) -> list:
    prompt = f"""Extract ONLY symptoms from this sentence.
Return ONLY comma separated symptoms. No explanations.
Sentence: {text}
Output:"""
    result   = call_openai(prompt)
    symptoms = [s.strip() for s in result.split(",")]
    return symptoms


def clean_llm_output(symptoms: list) -> list:
    cleaned = []
    for s in symptoms:
        s = s.lower()
        s = re.sub(r'[^a-z\s]', '', s)
        s = s.strip()
        if len(s) > 2:
            cleaned.append(s)
    return cleaned


def map_symptom(user_symptom: str) -> str:
    user_embedding = embedding_model.encode([user_symptom])
    similarity     = cosine_similarity(user_embedding, symptom_embeddings)[0]
    top_k_idx      = np.argsort(similarity)[-5:][::-1]
    candidates     = [symptom_list[i] for i in top_k_idx]

    prompt = f"""Map this symptom to the closest one from the list.
Symptom: {user_symptom}
List: {candidates}
Return ONLY one item from the list."""

    mapped = call_openai(prompt)
    if mapped not in candidates:
        mapped = candidates[0]
    return mapped


def extract_and_clean(text: str) -> list:
    return clean_llm_output(extract_symptoms_llm(text))


def normalize_symptoms(raw_symptoms: list) -> list:
    normalized = []
    for symptom in raw_symptoms:
        mapped = map_symptom(symptom)
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized


# =============================================================================
# DISEASE PREDICTION
# =============================================================================

def predict_disease(
    confirmed_symptoms: list,
    absent_symptoms:    list = None
) -> list:
    """
    Runs CatBoost.

    confirmed = 1  (user said YES)
    absent    = 0  (user said NO  — meaningful zero)
    unasked   = 0  (unknown       — minimized by Phase 1 questions)

    Returns top 5 diseases above 3% threshold.
    """

    input_vector = np.zeros(len(symptom_columns))

    for symptom in confirmed_symptoms:
        if symptom in symptom_columns:
            index = symptom_columns.index(symptom)
            input_vector[index] = 1

    input_vector = input_vector.reshape(1, -1)
    probs        = model.predict_proba(input_vector)[0]

    # return the top 5 diseases regardless of a minimum threshold
    predictions   = []

    for i, prob in enumerate(probs):
        predictions.append({
            "disease":    disease_classes[i],
            "confidence": float(prob)
        })

    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return predictions[:5]


# =============================================================================
# ENDPOINT — GET /symptoms/search
# =============================================================================

@app.route("/symptoms/search", methods=["GET"])
def symptoms_search():
    try:
        query = request.args.get("q", "").strip()
        if not query or len(query) < 2:
            return jsonify({"results": []}), 200

        results = search_symptoms_by_text(query, top_k=10)
        return jsonify({"results": results}), 200

    except Exception as e:
        print(f"ERROR in /symptoms/search: {str(e)}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ENDPOINT — POST /assess/start
# =============================================================================

@app.route("/assess/start", methods=["POST"])
def assess_start():
    """
    Called when user clicks Start Assessment.

    Always starts in Phase 1 — collect.
    Never runs CatBoost here.
    Just asks the first high-IG question.
    """
    try:
        data              = request.get_json()
        selected_symptoms = data.get("selected_symptoms", [])

        if not selected_symptoms:
            return jsonify({"error": "No symptoms selected"}), 400

        print("\n" + "="*55)
        print(f"ASSESSMENT START — Phase 1 begins")
        print(f"Selected: {selected_symptoms}")
        print("="*55)

        # ── emergency check ───────────────────────────────────────────────────
        symptom_text = " ".join(s.replace("_", " ") for s in selected_symptoms)
        emergency    = detect_emergency(
            user_text           = symptom_text,
            raw_symptoms        = [s.replace("_", " ") for s in selected_symptoms],
            normalized_symptoms = selected_symptoms
        )

        if emergency["is_emergency"]:
            return jsonify({
                "status":              "emergency",
                "is_emergency":        True,
                "esi_level":           emergency.get("esi_level"),
                "suspected_condition": emergency.get("suspected_condition"),
                "message":             emergency.get("message"),
                "immediate_actions":   emergency.get("immediate_actions", [])
            }), 200

        # ── Phase 1 — get first high-IG question ─────────────────────────────
        # do NOT run CatBoost here
        # selected_symptoms are confirmed — do not ask about them again
        next_symptom = get_next_symptom_to_ask(
            confirmed_symptoms  = selected_symptoms,
            absent_symptoms     = [],
            asked_symptoms      = [],
            current_predictions = None    # Phase 1 — no predictions
        )

        if not next_symptom:
            # edge case — predict with just selected symptoms
            predictions = predict_disease(selected_symptoms)
            return jsonify({
                "status":             "predicted",
                "source":             "model",
                "predictions":        predictions,
                "confirmed_symptoms": selected_symptoms,
                "absent_symptoms":    [],
                "questions_asked":    0,
                "phase":              "collect"
            }), 200

        question_data = format_question_with_llm(
            symptom            = next_symptom,
            confirmed_symptoms = selected_symptoms,
            phase              = "collect"
        )

        print(f"Phase 1 first question: {question_data['question']} [{next_symptom}]")

        return jsonify({
            "status":             "question",
            "phase":              "collect",
            "is_emergency":       False,
            "symptom":            question_data["symptom"],
            "question":           question_data["question"],
            "options":            question_data["options"],
            "confirmed_symptoms": selected_symptoms,
            "absent_symptoms":    [],
            "asked_symptoms":     [next_symptom],
            "questions_asked":    1,
            "progress": {
                "asked":    1,
                "max":      MAX_QUESTIONS,
                "phase":    "collect",
                "phase_info": "Collecting your symptoms..."
            }
        }), 200

    except Exception as e:
        print(f"ERROR in /assess/start: {str(e)}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ENDPOINT — POST /assess/answer
# =============================================================================

@app.route("/assess/answer", methods=["POST"])
def assess_answer():
    """
    Called every time user answers a question.

    Phase 1 — COLLECT:
      Do not run CatBoost
      Ask next high-IG symptom
      Keep collecting until 4+ confirmed OR 10+ questions

    Phase 2 — DISCRIMINATE:
      Run CatBoost on confirmed symptoms
      Check if confident enough to predict
      If not → ask most discriminating question between top 4 diseases
      If yes → return prediction
    """
    try:
        data               = request.get_json()
        symptom            = data.get("symptom")
        answer             = data.get("answer")
        confirmed_symptoms = data.get("confirmed_symptoms", [])
        absent_symptoms    = data.get("absent_symptoms",    [])
        asked_symptoms     = data.get("asked_symptoms",     [])
        questions_asked    = data.get("questions_asked",    0)

        if not symptom or not answer:
            return jsonify({"error": "Missing symptom or answer"}), 400

        # ── update symptom lists ──────────────────────────────────────────────
        if answer == "Yes" and symptom not in confirmed_symptoms:
            confirmed_symptoms.append(symptom)
        elif answer == "No" and symptom not in absent_symptoms:
            absent_symptoms.append(symptom)
        # "Not sure" → skip both lists

        if symptom not in asked_symptoms:
            asked_symptoms.append(symptom)

        print(f"\nAnswer: {symptom} → {answer}")
        print(f"Confirmed ({len(confirmed_symptoms)}): {confirmed_symptoms}")
        print(f"Absent    ({len(absent_symptoms)}):    {absent_symptoms}")

        # ── decide current phase ──────────────────────────────────────────────
        phase = get_current_phase(confirmed_symptoms, questions_asked)
        print(f"Phase: {phase.upper()}")

        # ── emergency check — pass the newly answered symptom so the
        #    fast-path can skip LLM stages for non-risky symptoms ─────────────
        if confirmed_symptoms:
            symptom_text = " ".join(s.replace("_", " ") for s in confirmed_symptoms)
            emergency    = detect_emergency(
                user_text           = symptom_text,
                raw_symptoms        = [s.replace("_", " ") for s in confirmed_symptoms],
                normalized_symptoms = confirmed_symptoms,
                new_symptom         = symptom   # ← enables fast path
            )
            if emergency["is_emergency"]:
                return jsonify({
                    "status":              "emergency",
                    "is_emergency":        True,
                    "esi_level":           emergency.get("esi_level"),
                    "suspected_condition": emergency.get("suspected_condition"),
                    "message":             emergency.get("message"),
                    "immediate_actions":   emergency.get("immediate_actions", [])
                }), 200

        # ── PHASE 1 — COLLECT ─────────────────────────────────────────────────
        # Do NOT run CatBoost in Phase 1
        # Just ask the next high-IG question

        if phase == "collect":
            print("Phase 1 — asking high-IG question, not running model yet")

            next_symptom = get_next_symptom_to_ask(
                confirmed_symptoms  = confirmed_symptoms,
                absent_symptoms     = absent_symptoms,
                asked_symptoms      = asked_symptoms,
                current_predictions = None    # no predictions in Phase 1
            )

            if not next_symptom:
                # no more questions — force into Phase 2
                print("No more Phase 1 questions — forcing Phase 2")
                current_predictions = predict_disease(confirmed_symptoms, absent_symptoms)
                return _build_prediction_response(
                    confirmed_symptoms, absent_symptoms,
                    questions_asked, current_predictions, "forced_phase2"
                )

            # ── format question (cache hit = instant, miss = ~0.8 s) ─────────
            question_data = format_question_with_llm(
                symptom            = next_symptom,
                confirmed_symptoms = confirmed_symptoms,
                phase              = "collect"
            )

            asked_symptoms.append(next_symptom)

            print(f"Phase 1 question: {question_data['question']} [{next_symptom}]")

            return jsonify({
                "status":             "question",
                "phase":              "collect",
                "is_emergency":       False,
                "symptom":            question_data["symptom"],
                "question":           question_data["question"],
                "options":            question_data["options"],
                "confirmed_symptoms": confirmed_symptoms,
                "absent_symptoms":    absent_symptoms,
                "asked_symptoms":     asked_symptoms,
                "questions_asked":    questions_asked + 1,
                "progress": {
                    "asked":      questions_asked + 1,
                    "max":        MAX_QUESTIONS,
                    "phase":      "collect",
                    "phase_info": f"Collecting symptoms ({len(confirmed_symptoms)} confirmed so far)..."
                }
            }), 200

        # ── PHASE 2 — DISCRIMINATE ────────────────────────────────────────────
        # NOW run CatBoost for the first time
        # Check if confident enough to predict
        # If not — ask discriminating question between top 4 diseases

        print("Phase 2 — running CatBoost for first time (or update)")

        current_predictions = predict_disease(confirmed_symptoms, absent_symptoms)

        print(f"Top predictions:")
        for p in current_predictions[:4]:
            print(f"  {p['disease']:40} {p['confidence']:.2%}")

        # ── check if ready to predict ─────────────────────────────────────────
        ready, reason = should_predict(
            confirmed_symptoms  = confirmed_symptoms,
            questions_asked     = questions_asked,
            current_predictions = current_predictions
        )

        print(f"Should predict: {ready} — {reason}")

        if ready:
            return _build_prediction_response(
                confirmed_symptoms, absent_symptoms,
                questions_asked, current_predictions, reason
            )

        # ── not ready — pick next discriminating symptom ─────────────────────
        next_symptom = get_next_symptom_to_ask(
            confirmed_symptoms  = confirmed_symptoms,
            absent_symptoms     = absent_symptoms,
            asked_symptoms      = asked_symptoms,
            current_predictions = current_predictions   # Phase 2 passes predictions
        )

        if not next_symptom:
            print("No more discriminating questions — predicting now")
            return _build_prediction_response(
                confirmed_symptoms, absent_symptoms,
                questions_asked, current_predictions, "no more questions"
            )

        # ── format question in parallel with nothing (already fast from cache)
        #    If this is a cache miss, run it in parallel with any pending work ─
        question_data = format_question_with_llm(
            symptom            = next_symptom,
            confirmed_symptoms = confirmed_symptoms,
            phase              = "discriminate"
        )

        asked_symptoms.append(next_symptom)

        print(f"Phase 2 question: {question_data['question']} [{next_symptom}]")

        return jsonify({
            "status":             "question",
            "phase":              "discriminate",
            "is_emergency":       False,
            "symptom":            question_data["symptom"],
            "question":           question_data["question"],
            "options":            question_data["options"],
            "confirmed_symptoms": confirmed_symptoms,
            "absent_symptoms":    absent_symptoms,
            "asked_symptoms":     asked_symptoms,
            "questions_asked":    questions_asked + 1,
            "progress": {
                "asked":      questions_asked + 1,
                "max":        MAX_QUESTIONS,
                "phase":      "discriminate",
                "phase_info": f"Narrowing down between {len(current_predictions[:4])} conditions..."
            }
        }), 200

    except Exception as e:
        print(f"ERROR in /assess/answer: {str(e)}")
        return jsonify({"error": str(e)}), 500


def _build_prediction_response(
    confirmed_symptoms:  list,
    absent_symptoms:     list,
    questions_asked:     int,
    predictions:         list,
    reason:              str
) -> object:
    """
    Builds the final prediction response.
    Called when ready to show results.
    """
    print(f"\nFINAL PREDICTION — {reason}")
    print(f"Confirmed symptoms: {confirmed_symptoms}")
    print(f"Top disease: {predictions[0]['disease']} ({predictions[0]['confidence']:.0%})")

    return jsonify({
        "status":              "predicted",
        "source":              "model",
        "is_emergency":        False,
        "predictions":         predictions,
        "confirmed_symptoms":  confirmed_symptoms,
        "absent_symptoms":     absent_symptoms,
        "questions_asked":     questions_asked,
        "reason":              reason
    }), 200


# =============================================================================
# ENDPOINT — POST /assess/explain
# =============================================================================

@app.route("/assess/explain", methods=["POST"])
def assess_explain():
    try:
        data       = request.get_json()
        disease    = data.get("disease")
        symptoms   = data.get("symptoms", [])
        confidence = data.get("confidence", 0)

        if not disease:
            return jsonify({"error": "No disease provided"}), 400

        symptoms_str = ", ".join(symptoms).replace("_", " ")

        prompt = f"""You are a medical assistant explaining a health assessment to a patient.

Assessment result:
  Symptoms  : {symptoms_str}
  Condition : {disease}
  Confidence: {confidence:.0%}

Write a warm, clear explanation (under 120 words) covering:
1. Why these symptoms suggest this condition
2. What this condition is in simple terms
3. What the patient should do next (always recommend a real doctor)
4. Reminder that this is not a diagnosis

No bullet points. No headers. Just caring natural paragraphs.
"""

        explanation = call_openai(prompt)

        return jsonify({
            "disease":     disease,
            "confidence":  confidence,
            "explanation": explanation
        }), 200

    except Exception as e:
        print(f"ERROR in /assess/explain: {str(e)}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ENDPOINT — POST /assess/feedback
# =============================================================================

@app.route("/assess/feedback", methods=["POST"])
def assess_feedback():
    try:
        data              = request.get_json()
        predicted_disease = data.get("predicted_disease")
        user_comment      = data.get("user_comment")
        symptoms          = data.get("symptoms", [])
        confidence        = data.get("confidence", 0)

        if not predicted_disease or not user_comment:
            return jsonify({"error": "Missing disease or comment"}), 400

        symptoms_str = ", ".join(symptoms).replace("_", " ")

        prompt = f"""You are reviewing patient feedback on a health assessment.

Assessment:
  Symptoms : {symptoms_str}
  Predicted: {predicted_disease} ({confidence:.0%} confidence)

Patient says: "{user_comment}"

Evaluate if patient raises valid medical points.
DECISION: Valid points → user_correct: true. Confused → user_correct: false.
Response: empathetic, under 80 words.

Respond ONLY with valid JSON:
{{
  "user_correct": true or false,
  "reasoning": "your evaluation",
  "response_to_patient": "what to say",
  "action": "continue_questions" or "explain_prediction"
}}
"""

        raw   = call_openai(prompt)
        raw   = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)

        if not match:
            return jsonify({"error": "Could not parse response"}), 500

        result = json.loads(match.group())

        return jsonify({
            "user_correct":        bool(result.get("user_correct", False)),
            "reasoning":           result.get("reasoning", ""),
            "response_to_patient": result.get("response_to_patient", ""),
            "action":              result.get("action", "explain_prediction")
        }), 200

    except Exception as e:
        print(f"ERROR in /assess/feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500



# =============================================================================
# ENDPOINT — POST /assess/recommendations
# =============================================================================

@app.route("/assess/recommendations", methods=["POST"])
def assess_recommendations():
    try:
        data       = request.get_json()
        disease    = data.get("disease")
        confidence = data.get("confidence", 0)
        symptoms   = data.get("symptoms", [])
        section    = data.get("section", "diet")   # diet | workout | precautions
        user       = data.get("user", {})

        if not disease:
            return jsonify({"error": "No disease provided"}), 400

        age    = user.get("age")
        gender = user.get("gender")

        recommendations = generate_recommendations(
            disease    = disease,
            symptoms   = symptoms,
            confidence = confidence,
            age        = age,
            gender     = gender,
            section    = section
        )

        return jsonify(recommendations), 200

    except Exception as e:
        print(f"ERROR in /assess/recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# LEGACY — POST /predict
# =============================================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json()
        symptoms = data.get("symptoms")
        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        user_text           = " ".join(symptoms)
        raw_symptoms        = extract_and_clean(user_text)
        normalized_symptoms = normalize_symptoms(raw_symptoms)

        emergency = detect_emergency(
            user_text           = user_text,
            raw_symptoms        = raw_symptoms,
            normalized_symptoms = normalized_symptoms
        )

        if emergency["is_emergency"]:
            return jsonify({
                "is_emergency":        True,
                "esi_level":           emergency.get("esi_level"),
                "suspected_condition": emergency.get("suspected_condition"),
                "message":             emergency.get("message"),
                "immediate_actions":   emergency.get("immediate_actions", []),
                "predictions":         []
            }), 200

        predictions = predict_disease(normalized_symptoms)
        return jsonify({
            "is_emergency":        False,
            "normalized_symptoms": normalized_symptoms,
            "predictions":         predictions
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ML service", "port": 5001}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
