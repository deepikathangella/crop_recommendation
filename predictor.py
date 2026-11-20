# predictor.py
# This file loads the saved model and makes a recommendation.
# It is kept separate from app.py so it is easy to test without the web server.

import joblib
import pandas as pd

# Allowed values for each input. Anything outside these is rejected as invalid.
# (These are wide, realistic limits. The "unusual input" check below is stricter,
# it compares with what the model saw during training.)
VALID_LIMITS = {
    "N": (0, 300),
    "P": (0, 200),
    "K": (0, 400),
    "temperature": (-10, 60),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 3000),
}

# load everything that train_model.py saved
bundle = joblib.load("crop_model.pkl")
model = bundle["model"]
feature_names = bundle["feature_names"]
ood_scaler = bundle["ood_scaler"]
ood_finder = bundle["ood_finder"]
distance_limit = bundle["distance_limit"]
confidence_limit = bundle["confidence_limit"]
train_min = bundle["train_min"]
train_max = bundle["train_max"]


def check_input(values):
    """Check that every value is a number inside the valid limits.
    Returns a dictionary of errors (empty if everything is fine)."""
    errors = {}
    for name in feature_names:
        if name not in values:
            errors[name] = "Missing value"
            continue
        value = values[name]
        # bool is a number type in Python, but True/False is not a valid input
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors[name] = "Invalid input"
            continue
        low, high = VALID_LIMITS[name]
        if value != value or value < low or value > high:   # value != value is True only for NaN
            errors[name] = "Invalid input (allowed range " + str(low) + " to " + str(high) + ")"
    return errors


def make_recommendation(values):
    """values is a dictionary like {"N": 90, "P": 42, ...}. Returns a result dictionary."""
    row = pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)

    # 1. model prediction and confidence
    probabilities = model.predict_proba(row)[0]
    classes = model.classes_
    order = probabilities.argsort()[::-1]                 # highest probability first
    top3 = []
    for i in order[:3]:
        top3.append({"crop": str(classes[i]), "probability": round(float(probabilities[i]), 4)})
    best_crop = top3[0]["crop"]
    confidence = top3[0]["probability"]

    # 2. out-of-range checks
    # (a) is any single value outside what the model saw in training?
    outside = []
    for name in feature_names:
        if values[name] < train_min[name] or values[name] > train_max[name]:
            outside.append(name)

    # (b) is the COMBINATION of values unusual? (far from the nearest training rows)
    scaled = ood_scaler.transform(row)
    distances, _ = ood_finder.kneighbors(scaled)
    unusual_score = float(distances.mean())
    combination_unusual = unusual_score > distance_limit

    # 3. decide what to tell the user
    if len(outside) > 0 or combination_unusual:
        status = "unusual"
        message = "These values are unusual compared with the training data, so no crop is recommended."
        recommended = None
    elif confidence < confidence_limit:
        status = "low_confidence"
        message = "The model is not sure. Treat the crops below as possible options, not a firm answer."
        recommended = best_crop
    else:
        status = "confident"
        message = "The model is confident about this recommendation."
        recommended = best_crop

    return {
        "status": status,
        "message": message,
        "recommended_crop": recommended,
        "confidence": confidence,
        "top3": top3,
        "unusual_score": round(unusual_score, 3),
        "unusual_limit": round(distance_limit, 3),
        "features_outside_training_range": outside,
    }
