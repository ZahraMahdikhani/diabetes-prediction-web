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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _persian_available = True
except ImportError:
    _persian_available = False

# =========================
# Configuration
# =========================
def _path_in_app_dir(rel_path: str) -> str:
    """Resolve path relative to this app's directory (works regardless of CWD)."""
    if os.path.isabs(rel_path):
        return rel_path
    app_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(app_dir, rel_path)


DB_PATH = "predictions.db"
MODEL_PATH = "./diabetes_model.pkl"
THRESHOLD = 0.502

# Resolve relative paths from app directory so CWD doesn't matter
DB_PATH = _path_in_app_dir(DB_PATH)
MODEL_PATH = _path_in_app_dir(MODEL_PATH)

# Only the 10 features the model was trained on (exact names/order)
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

model = None  # global singleton
_model_load_error = None  # last error message when load failed (for 503 message)


# =========================
# Model loading
# =========================
def load_model() -> None:
    """Load ML model once (idempotent). Does not raise if file is missing."""
    global model, _model_load_error
    if model is not None:
        return
    _model_load_error = None
    if not os.path.isfile(MODEL_PATH):
        _model_load_error = f"Model file not found at {MODEL_PATH}"
        print(_model_load_error)
        return
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully from", MODEL_PATH)
    except FileNotFoundError:
        _model_load_error = f"Model file not found: {MODEL_PATH}"
        print(_model_load_error)
    except Exception as e:
        err_msg = str(e).lower()
        _model_load_error = str(e)
        print(f"Error loading model from {MODEL_PATH}: {e}")
        if "libomp" in err_msg or "lib_lightgbm" in err_msg:
            _model_load_error += " — On macOS run: brew install libomp"
            print("  -> On macOS, run: brew install libomp")


@app.before_request
def ensure_model_loaded():
    load_model()


def _model_unavailable_response():
    """Return 503 response when model is not loaded (avoids 502 from worker crash)."""
    msg = "Model not available."
    if _model_load_error:
        msg += " " + _model_load_error
    else:
        msg += f" Ensure diabetes_model.pkl exists (or set MODEL_PATH). Path used: {MODEL_PATH}"
    return (
        msg,
        503,
        {"Content-Type": "text/plain; charset=utf-8"},
    )


# =========================
# Database helpers
# =========================
def get_conn():
    # check_same_thread=False helps when using dev server/threading
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
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
            "INSERT INTO predictions (created_at, input_json, prob, result) VALUES (?, ?, ?, ?)",
            (created_at, payload, float(prob), int(result)),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_record(rec_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, created_at, input_json, prob, result FROM predictions WHERE id = ?",
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


# Initialize DB at import time
init_db()


# =========================
# Input validation & parsing
# =========================
def _get_value(src, key: str):
    # Works for request.form (MultiDict) and JSON dict
    if src is None:
        return None
    return src.get(key)


def parse_and_validate(form_or_json, source: str = "form"):
    """
    Validates user inputs, computes BMI, and returns:
      (True, {"data": <dict>, "row": <pd.DataFrame>})
      (False, {"errors": [..]})
    """
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
                # allow "1.0" but store as int
                val = int(float(raw))

            # Range checks
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

    # Compute BMI
    try:
        height_m = data["height_cm"] / 100.0
        if height_m <= 0:
            raise ValueError("قد نمی‌تواند صفر یا منفی باشد")

        bmi = data["weight_kg"] / (height_m ** 2)
        bmi = round(float(bmi), 1)

        if bmi < 10 or bmi > 80:
            return False, {"errors": [f"BMI محاسبه شده غیرمنطقی است: {bmi:.1f}"]}

        data["BMI"] = bmi
    except Exception as e:
        return False, {"errors": [f"خطا در محاسبه BMI: {str(e)}"]}

    # Prepare row for model (exact columns)
    try:
        row_dict = {f: data[f] for f in SELECTED_FEATURES}
        row = pd.DataFrame([row_dict], columns=SELECTED_FEATURES)
    except Exception as e:
        return False, {"errors": [f"خطا در آماده‌سازی داده برای مدل: {str(e)}"]}

    return True, {"data": data, "row": row}


# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if model is None:
            return _model_unavailable_response()
        valid, payload = parse_and_validate(request.form, source="form")
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
    if model is None:
        err = _model_load_error or f"Ensure diabetes_model.pkl exists or set MODEL_PATH. Path used: {MODEL_PATH}"
        return jsonify({"error": "Model not available. " + err}), 503
    if not request.is_json:
        return jsonify({"error": "JSON request required"}), 400

    valid, payload = parse_and_validate(request.get_json(), source="api")
    if not valid:
        return jsonify({"errors": payload["errors"]}), 400

    row = payload["row"]
    user_data = payload["data"]

    prob = float(model.predict_proba(row)[0][1])
    result = int(prob > THRESHOLD)
    rec_id = save_record(user_data, prob, result)

    return jsonify(
        {
            "probability": round(prob, 4),
            "risk_level": "high" if result else "low",
            "result": result,
            "record_id": rec_id,
            "threshold": THRESHOLD,
        }
    )


# =========================
# PDF helpers (Persian / RTL)
# =========================
_PDF_FONT_NAME = None
# Fallback download URL for Vazirmatn (single TTF, no zip)
VAZIRMATN_REGULAR_URL = (
    "https://github.com/rastikerdar/vazirmatn/raw/master/fonts/ttf/Vazirmatn-Regular.ttf"
)


def _register_pdf_font():
    """Register a Persian-capable font. Use Vazirmatn (local or auto-download) or Windows Tahoma/Arial."""
    global _PDF_FONT_NAME
    if _PDF_FONT_NAME:
        return _PDF_FONT_NAME
    app_dir = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(app_dir, "static", "fonts")
    vazir_path = os.path.join(fonts_dir, "Vazirmatn-Regular.ttf")

    def try_register(path: str, name: str) -> bool:
        if not path or not path.endswith(".ttf") or not os.path.isfile(path):
            return False
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            return True
        except Exception:
            return False

    # 1) Prefer local Vazirmatn
    if try_register(vazir_path, "Vazirmatn"):
        _PDF_FONT_NAME = "Vazirmatn"
        return _PDF_FONT_NAME

    # 2) On Windows, scan Fonts folder for Tahoma then Arial (both have Arabic/Persian)
    if os.name == "nt":
        win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
        if os.path.isdir(win_fonts):
            for preferred in ("tahoma", "arial"):
                for f in os.listdir(win_fonts):
                    if f.lower().endswith(".ttf") and preferred in f.lower():
                        path = os.path.join(win_fonts, f)
                        name = "Tahoma" if preferred == "tahoma" else "Arial"
                        if try_register(path, name):
                            _PDF_FONT_NAME = name
                            return _PDF_FONT_NAME

    # 3) Auto-download Vazirmatn to static/fonts and use it
    import urllib.request
    for url in (
        VAZIRMATN_REGULAR_URL,
        "https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@master/fonts/ttf/Vazirmatn-Regular.ttf",
    ):
        try:
            os.makedirs(fonts_dir, exist_ok=True)
            urllib.request.urlretrieve(url, vazir_path)
            if os.path.isfile(vazir_path) and try_register(vazir_path, "Vazirmatn"):
                _PDF_FONT_NAME = "Vazirmatn"
                return _PDF_FONT_NAME
        except Exception as e:
            print(f"Could not download Persian font from {url[:50]}...: {e}")
            continue

    _PDF_FONT_NAME = "Helvetica"
    return _PDF_FONT_NAME


def _persian_pdf(text: str) -> str:
    """Reshape and bidi for Persian/Arabic so it displays correctly in PDF."""
    if not text or not _persian_available:
        return text
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def _pdf_draw_rtl(c, x_right, y, text, font_name, size):
    c.setFont(font_name, size)
    c.drawRightString(x_right, y, _persian_pdf(text))


def _pdf_draw_centred(c, x_center, y, text, font_name, size):
    c.setFont(font_name, size)
    s = _persian_pdf(text)
    w = c.stringWidth(s, font_name, size)
    c.drawString(x_center - w / 2, y, s)


@app.route("/download/<int:rec_id>")
def download_pdf(rec_id: int):
    rec = get_record(rec_id)
    if not rec:
        return "Record not found", 404

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    right = width - 50
    font_name = _register_pdf_font()

    # Header
    p.setFillColorRGB(0.1, 0.5, 0.7)
    _pdf_draw_centred(p, width / 2, height - 70, "گزارش ارزیابی خطر دیابت", font_name, 20)
    _pdf_draw_centred(p, width / 2, height - 100, "پیش‌بینی خطر دیابت نوع ۲", font_name, 12)

    # Report details (RTL)
    p.setFillColorRGB(0, 0, 0)
    _pdf_draw_rtl(p, right, height - 140, "جزئیات گزارش", font_name, 14)
    _pdf_draw_rtl(p, right, height - 165, f"شماره گزارش: {rec['id']}", font_name, 11)

    try:
        created_dt = datetime.fromisoformat(rec["created_at"])
        created_str = created_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        created_str = str(rec["created_at"])

    _pdf_draw_rtl(p, right, height - 185, f"تاریخ: {created_str}", font_name, 11)
    _pdf_draw_rtl(p, right, height - 205, f"احتمال خطر: {rec['prob']:.1%}", font_name, 11)
    _pdf_draw_rtl(
        p, right, height - 225,
        f"نتیجه: {'خطر بالا' if rec['result'] == 1 else 'خطر پایین'}",
        font_name, 11,
    )

    # Risk box
    if rec["result"] == 1:
        p.setFillColorRGB(0.9, 0.3, 0.3)
    else:
        p.setFillColorRGB(0.3, 0.8, 0.4)
    p.rect(380, height - 240, 160, 50, fill=1, stroke=0)
    p.setFillColorRGB(1, 1, 1)
    _pdf_draw_centred(p, 460, height - 220, "خطر بالا" if rec["result"] == 1 else "خطر پایین", font_name, 16)

    # User inputs
    p.setFillColorRGB(0, 0, 0)
    _pdf_draw_rtl(p, right, height - 290, "پاسخ‌های شما", font_name, 13)

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
        "BMI": "شاخص توده بدنی (محاسبه شده)",
    }
    input_data = rec["input"]

    for key in ["height_cm", "weight_kg"] + SELECTED_FEATURES:
        if key not in input_data:
            continue
        label = readable_names.get(key, key)
        value = input_data[key]
        if isinstance(value, float) and key in ("BMI",):
            value_str = f"{value:.1f}"
        else:
            value_str = str(value)
        _pdf_draw_rtl(p, right, y, f"• {label}", font_name, 11)
        p.setFont(font_name, 11)
        p.drawString(320, y, value_str)
        y -= 22
        if y < 60:
            p.showPage()
            y = height - 50

    # Footer
    p.setFillColorRGB(0.5, 0.5, 0.5)
    _pdf_draw_centred(
        p, width / 2, 30,
        "این ابزار فقط برای غربالگری است • تشخیص پزشکی نیست • با پزشک مشورت کنید",
        font_name, 9,
    )

    p.save()
    buffer.seek(0)

    filename = f"diabetes_risk_report_{rec['id']}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


@app.route("/record/<int:rec_id>")
def view_record(rec_id: int):
    rec = get_record(rec_id)
    if not rec:
        return jsonify({"error": "Not found"}), 404
    return jsonify(rec)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
