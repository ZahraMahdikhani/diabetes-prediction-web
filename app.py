import io
import os
import json
import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import (
    Flask, request, render_template, flash,
    redirect, url_for, jsonify, send_file
)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


import traceback

@app.errorhandler(Exception)
def handle_exception(e):
    print("ERROR OCCURRED:")
    traceback.print_exc()
    return "Internal Server Error", 500

# =========================
# Configuration
# =========================

# مسیر مطلق پروژه (مهم برای اجرا در Docker / Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "diabetes_model.pkl")
)

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(BASE_DIR, "predictions.db")
)

THRESHOLD = float(os.environ.get("THRESHOLD", 0.502))

# Only the 10 features the model was trained on
SELECTED_FEATURES = [
    "HighBP",
    "HighChol",
    "GenHlth",
    "PhysHlth",
    "DiffWalk",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Gender",
    "Age",
    "BMI",
]

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_random_key_2025")

model = None


# =========================
# Model loading
# =========================
def load_model():
    global model

    if model is not None:
        return

    print("Loading model from:", MODEL_PATH)
    print("File exists:", os.path.exists(MODEL_PATH))

    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")


@app.before_request
def ensure_model_loaded():
    load_model()


# =========================
# Database helpers
# =========================
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_json TEXT NOT NULL,
                prob REAL NOT NULL,
                result INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def save_record(input_dict: dict, prob: float, result: int) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(input_dict, ensure_ascii=False)

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
            (created_at, input_json, prob, result)
            VALUES (?, ?, ?, ?)
            """,
            (created_at, payload, float(prob), int(result)),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_record(rec_id: int):
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, created_at, input_json, prob, result
            FROM predictions
            WHERE id = ?
            """,
            (rec_id,),
        ).fetchone()

    if not row:
        return None

    return {
        "id": int(row["id"]),
        "created_at": row["created_at"],
        "input": json.loads(row["input_json"]),
        "prob": float(row["prob"]),
        "result": int(row["result"]),
    }


init_db()


# =========================
# Input validation
# =========================
def _get_value(src, key):
    if src is None:
        return None
    return src.get(key)


def parse_and_validate(form_or_json):

    data = {}
    errors = []

    required_fields = ["height_cm", "weight_kg"] + [
        f for f in SELECTED_FEATURES if f != "BMI"
    ]

    for field in required_fields:
        raw = _get_value(form_or_json, field)

        if raw is None or str(raw).strip() == "":
            errors.append(f"فیلد اجباری وارد نشده است: {field}")
            continue

        try:
            if field in ("height_cm", "weight_kg"):
                val = float(raw)
            else:
                val = int(float(raw))

            if field == "height_cm" and not (90 <= val <= 230):
                errors.append("قد باید بین ۹۰ تا ۲۳۰ باشد")

            if field == "weight_kg" and not (25 <= val <= 220):
                errors.append("وزن باید بین ۲۵ تا ۲۲۰ باشد")

            data[field] = val

        except:
            errors.append(f"مقدار نامعتبر برای {field}")

    if errors:
        return False, {"errors": errors}

    # BMI
    height_m = data["height_cm"] / 100
    bmi = data["weight_kg"] / (height_m ** 2)
    bmi = round(bmi, 1)
    data["BMI"] = bmi

    row_dict = {f: data[f] for f in SELECTED_FEATURES}
    row = pd.DataFrame([row_dict], columns=SELECTED_FEATURES)

    return True, {"data": data, "row": row}


# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        valid, payload = parse_and_validate(request.form)

        if not valid:
            for err in payload["errors"]:
                flash(err, "danger")
            return redirect(url_for("index"))

        row = payload["row"]
        user_data = payload["data"]

        prob = float(model.predict_proba(row)[0][1])
        result = int(prob > THRESHOLD)

        rec_id = save_record(user_data, prob, result)

        return render_template(
            "result.html",
            prediction={
                "result": result,
                "prob": f"{prob:.1%}",
                "id": rec_id,
                "date": datetime.now().strftime("%Y-%m-%d"),
            },
        )

    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():

    if not request.is_json:
        return jsonify({"error": "JSON required"}), 400

    valid, payload = parse_and_validate(request.get_json())

    if not valid:
        return jsonify(payload), 400

    row = payload["row"]
    user_data = payload["data"]

    prob = float(model.predict_proba(row)[0][1])
    result = int(prob > THRESHOLD)

    rec_id = save_record(user_data, prob, result)

    return jsonify(
        {
            "probability": round(prob, 4),
            "result": result,
            "record_id": rec_id,
        }
    )


@app.route("/download/<int:rec_id>")
def download_pdf(rec_id):

    rec = get_record(rec_id)
    if not rec:
        return "Record not found", 404

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    p.setFont("Helvetica", 14)
    p.drawString(50, height - 50, "Diabetes Risk Report")

    p.drawString(50, height - 90, f"Record ID: {rec['id']}")
    p.drawString(50, height - 110, f"Probability: {rec['prob']:.2f}")

    p.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="report.pdf",
        mimetype="application/pdf",
    )


@app.route("/record/<int:rec_id>")
def view_record(rec_id):
    rec = get_record(rec_id)
    if not rec:
        return jsonify({"error": "Not found"}), 404
    return jsonify(rec)


# =========================
# Run
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
