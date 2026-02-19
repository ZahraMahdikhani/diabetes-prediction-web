import io
import os
import json
import sqlite3
from datetime import datetime
import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    send_file, url_for, redirect, flash
)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ----- Configuration -----
DB_PATH = "predictions.db"
MODEL_PATH = "diabetes_model.pkl"

# ----- Load model and threshold -----
THRESHOLD = float(os.environ.get("THRESHOLD", 0.502))
model = joblib.load(MODEL_PATH)

# فقط ۱۰ ویژگی که مدل با آنها آموزش دیده
SELECTED_FEATURES = [
    'HighBP',
    'HighChol',
    'GenHlth',
    'PhysHlth',
    'DiffWalk',
    'HeartDiseaseorAttack',
    'PhysActivity',
    'Gender',
    'Age',
    'BMI'
]

# ----- Flask app -----
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_random_key_2025")

# ----- Database helpers -----
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            input_json TEXT,
            prob REAL,
            result INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_record(input_dict, prob, result):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (created_at, input_json, prob, result) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), json.dumps(input_dict), float(prob), int(result))
    )
    conn.commit()
    rec_id = c.lastrowid
    conn.close()
    return rec_id

def get_record(rec_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, created_at, input_json, prob, result FROM predictions WHERE id = ?", (rec_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "created_at": row[1],
        "input": json.loads(row[2]),
        "prob": row[3],
        "result": row[4]
    }

init_db()

# ----- Input validation and parsing -----
def parse_and_validate(form_or_json, source="form"):
    data = {}
    errors = []

    # فیلدهای مورد نیاز کاربر
    required_fields = ["height_cm", "weight_kg"] + [f for f in SELECTED_FEATURES if f != "BMI"]

    for field in required_fields:
        raw = form_or_json.get(field)
        if raw is None or str(raw).strip() == "":
            errors.append(f"فیلد اجباری وارد نشده است: {field}")
            continue

        try:
            if field in ["height_cm", "weight_kg"]:
                val = float(raw)
            else:
                val = int(float(raw))  # برای اطمینان از تبدیل

            # اعتبارسنجی محدوده‌ها
            if field == "height_cm" and not (90 <= val <= 230):
                errors.append("قد باید بین ۹۰ تا ۲۳۰ سانتی‌متر باشد")
            elif field == "weight_kg" and not (25 <= val <= 220):
                errors.append("وزن باید بین ۲۵ تا ۲۲۰ کیلوگرم باشد")
            elif field == "Age" and not (1 <= val <= 13):
                errors.append("گروه سنی باید بین ۱ تا ۱۳ باشد (طبق دسته‌بندی BRFSS)")
            elif field == "GenHlth" and not (1 <= val <= 5):
                errors.append("سلامت عمومی باید بین ۱ (عالی) تا ۵ (ضعیف) باشد")
            elif field == "PhysHlth" and not (0 <= val <= 30):
                errors.append("PhysHlth باید بین ۰ تا ۳۰ باشد")

            data[field] = val

        except (ValueError, TypeError):
            errors.append(f"مقدار نامعتبر برای {field}: {raw}")

    if errors:
        return False, {"errors": errors}

    # محاسبه BMI
    try:
        height_m = data["height_cm"] / 100
        if height_m <= 0:
            raise ValueError("قد نمی‌تواند صفر یا منفی باشد")
        bmi = data["weight_kg"] / (height_m ** 2)
        bmi = round(bmi, 1)
        if bmi < 10 or bmi > 80:
            errors.append(f"BMI محاسبه شده غیرمنطقی است: {bmi:.1f}")
            return False, {"errors": errors}
        data["BMI"] = bmi
    except Exception as e:
        return False, {"errors": [f"خطا در محاسبه BMI: {str(e)}"]}

    # آماده‌سازی داده برای مدل (دقیقاً به همان ترتیب و نام ستون‌ها)
    try:
        row_dict = {f: data[f] for f in SELECTED_FEATURES}
        row = pd.DataFrame([row_dict])
    except Exception as e:
        return False, {"errors": [f"خطا در آماده‌سازی داده برای مدل: {str(e)}"]}

    return True, {"data": data, "row": row}

# ----- Routes -----
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        valid, payload = parse_and_validate(request.form)
        if not valid:
            for error in payload["errors"]:
                flash(error, "danger")
            return redirect(url_for("index"))

        row = payload["row"]
        user_data = payload["data"]

        # پیش‌بینی
        prob = model.predict_proba(row)[0][1]
        result = int(prob > THRESHOLD)

        # ذخیره
        rec_id = save_record(user_data, prob, result)

        prob_percent = f"{prob:.1%}"
        return render_template("result.html", prediction={
            "result": result,
            "prob": prob_percent,
            "id": rec_id,
            "date": datetime.now().strftime("%Y-%m-%d")
        })

    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "JSON request required"}), 400

    valid, payload = parse_and_validate(request.get_json(), source="api")
    if not valid:
        return jsonify({"errors": payload["errors"]}), 400

    row = payload["row"]
    user_data = payload["data"]

    prob = model.predict_proba(row)[0][1]
    result = int(prob > THRESHOLD)
    rec_id = save_record(user_data, prob, result)

    return jsonify({
        "probability": round(float(prob), 4),
        "risk_level": "high" if result else "low",
        "result": result,
        "record_id": rec_id
    })


@app.route("/download/<int:rec_id>")
def download_pdf(rec_id):
    rec = get_record(rec_id)
    if not rec:
        return "Record not found", 404

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    p.setFillColorRGB(0.1, 0.5, 0.7)
    p.setFont("Helvetica-Bold", 20)
    p.drawCentredString(width / 2, height - 70, "گزارش ارزیابی خطر دیابت")
    p.setFont("Helvetica", 12)
    p.drawCentredString(width / 2, height - 100, "پیش‌بینی خطر دیابت نوع ۲")

    # Report details
    p.setFillColorRGB(0, 0, 0)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, height - 140, "جزئیات گزارش")
    p.setFont("Helvetica", 11)
    p.drawString(50, height - 165, f"شماره گزارش: {rec['id']}")
    p.drawString(50, height - 185, f"تاریخ: {datetime.fromisoformat(rec['created_at']).strftime('%Y-%m-%d %H:%M')}")
    p.drawString(50, height - 205, f"احتمال خطر: {rec['prob']:.1%}")
    p.drawString(50, height - 225, f"نتیجه: {'خطر بالا' if rec['result'] == 1 else 'خطر پایین'}")

    # Risk box
    if rec['result'] == 1:
        p.setFillColorRGB(0.9, 0.3, 0.3)
    else:
        p.setFillColorRGB(0.3, 0.8, 0.4)
    p.rect(380, height - 240, 160, 50, fill=1)
    p.setFillColorRGB(1, 1, 1)
    p.setFont("Helvetica-Bold", 16)
    p.drawCentredString(460, height - 220, "خطر بالا" if rec['result'] == 1 else "خطر پایین")

    # User inputs
    p.setFillColorRGB(0, 0, 0)
    p.setFont("Helvetica-Bold", 13)
    p.drawString(50, height - 290, "پاسخ‌های شما")
    p.setFont("Helvetica", 11)

    y = height - 320

    readable_names = {
        "height_cm": "قد (سانتی‌متر)",
        "weight_kg": "وزن (کیلوگرم)",
        "HighBP": "فشار خون بالا",
        "HighChol": "کلسترول بالا",
        "GenHlth": "سلامت عمومی (۱=عالی ... ۵=ضعیف)",
        "PhysHlth": "روزهای سلامت جسمی ضعیف (۳۰ روز اخیر)",
        "DiffWalk": "مشکل در راه رفتن",
        "HeartDiseaseorAttack": "سابقه بیماری قلبی یا حمله قلبی",
        "PhysActivity": "فعالیت بدنی منظم",
        "Gender": "جنسیت (۰=زن، ۱=مرد)",
        "Age": "گروه سنی",
        "BMI": "شاخص توده بدنی (محاسبه شده)"
    }

    input_data = rec["input"]

    for key in ["height_cm", "weight_kg"] + SELECTED_FEATURES:
        if key not in input_data:
            continue
        label = readable_names.get(key, key)
        value = input_data[key]

        if isinstance(value, float) and key in ["BMI", "prob"]:
            value_str = f"{value:.1f}"
        else:
            value_str = str(value)

        p.drawString(60, y, f"• {label}")
        p.drawString(320, y, value_str)
        y -= 22

        if y < 60:
            p.showPage()
            y = height - 50

    # Footer
    p.setFont("Helvetica-Oblique", 9)
    p.setFillColorRGB(0.5, 0.5, 0.5)
    p.drawCentredString(width / 2, 30, "این ابزار فقط برای غربالگری است • تشخیص پزشکی نیست • با پزشک مشورت کنید")

    p.save()
    buffer.seek(0)

    filename = f"diabetes_risk_report_{rec['id']}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf"
    )


@app.route("/record/<int:rec_id>")
def view_record(rec_id):
    rec = get_record(rec_id)
    if not rec:
        return "Not found", 404
    return jsonify(rec)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)