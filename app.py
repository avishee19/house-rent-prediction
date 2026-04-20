from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json, os
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# ── Load saved artifacts ───────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

try:
    model    = joblib.load(os.path.join(BASE, "model.pkl"))
    scaler   = joblib.load(os.path.join(BASE, "scaler.pkl"))
    features = json.load(open(os.path.join(BASE, "features.json")))
    print("✅ Model, scaler and features loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Missing file: {e}")
    print("   Copy model.pkl, scaler.pkl, features.json into the backend/ folder")

# ── Encoding maps (must match what LabelEncoder did during training) ───────────
CITY_MAP     = {"Bangalore":0, "Chennai":1, "Delhi":2,
                "Hyderabad":3, "Kolkata":4, "Mumbai":5}
FURNISH_MAP  = {"Furnished":0, "Semi-Furnished":1, "Unfurnished":2}
TENANT_MAP   = {"Bachelors":0, "Bachelors/Family":1, "Family":2}
CONTACT_MAP  = {"Contact Agent":0, "Contact Builder":1, "Contact Owner":2}
AREA_MAP     = {"Carpet Area":0, "Built Area":1, "Super Area":2}
FLOOR_MAP    = {
    "Ground out of 2":0, "Ground out of 3":1, "Ground out of 4":2,
    "1 out of 2":3,  "1 out of 3":4,
    "2 out of 3":5,  "2 out of 4":6,
    "3 out of 4":7,  "Lower Basement":8,
    "Upper Basement":9, "Other":10,
}

LABELS = {0: "Low",    1: "Medium",  2: "High"}
RANGES = {
    0: "Below ₹10,000/month",
    1: "₹10,000 – ₹30,000/month",
    2: "Above ₹30,000/month",
}

prediction_log = []   # in-memory store (resets on restart)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "RentIQ API is running 🏠"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
      "bhk": 2, "size": 1000, "city": "Mumbai",
      "furnishing": "Furnished", "tenant": "Bachelors/Family",
      "bathroom": 2, "area_type": "Super Area", "floor": "1 out of 3"
    }
    """
    try:
        body = request.get_json(force=True)

        # Build a row with the exact feature order used during training
        row = {
            "BHK":               int(body.get("bhk", 2)),
            "Size":              float(body.get("size", 1000)),
            "Floor":             FLOOR_MAP.get(body.get("floor", "Other"), 10),
            "Area Type":         AREA_MAP.get(body.get("area_type", "Super Area"), 2),
            "Area Locality":     0,          # default for unknown localities
            "City":              CITY_MAP.get(body.get("city", "Mumbai"), 5),
            "Furnishing Status": FURNISH_MAP.get(body.get("furnishing", "Unfurnished"), 2),
            "Tenant Preferred":  TENANT_MAP.get(body.get("tenant", "Bachelors/Family"), 1),
            "Bathroom":          int(body.get("bathroom", 2)),
            "Point of Contact":  2,          # default: Contact Owner
        }

        df     = pd.DataFrame([row])[features]
        scaled = scaler.transform(df)
        pred   = int(model.predict(scaled)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(scaled)[0]
            confidence = round(float(max(proba)) * 100, 1)

        result = {
            "category":   LABELS[pred],
            "range":      RANGES[pred],
            "code":       pred,
            "confidence": confidence,
        }

        # Save to log
        prediction_log.append({**body, **result})
        if len(prediction_log) > 100:
            prediction_log.pop(0)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/history", methods=["GET"])
def get_history():
    """Returns last 10 predictions (newest first)."""
    return jsonify(prediction_log[-10:][::-1])


@app.route("/stats", methods=["GET"])
def get_stats():
    """Summary of all predictions so far."""
    if not prediction_log:
        return jsonify({"total": 0, "low": 0, "medium": 0, "high": 0})
    codes = [p.get("code", -1) for p in prediction_log]
    return jsonify({
        "total":  len(prediction_log),
        "low":    codes.count(0),
        "medium": codes.count(1),
        "high":   codes.count(2),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
