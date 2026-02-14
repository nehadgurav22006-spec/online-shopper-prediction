from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Initialize DB
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  input_data TEXT,
                  prediction TEXT,
                  probability REAL,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Home / Login
@app.route("/")
def login():
    return render_template("login.html")

# Prediction Page
@app.route("/home")
def home():
    return render_template("index.html")

# Predict Route
@app.route("/predict", methods=["POST"])
def predict():

    features = []

    for i in range(1, 11):
        value = request.form.get(f"feature{i}")

        if value is None or value.strip() == "":
            features.append(0.0)
        else:
            features.append(float(value))

    print("Received:", features)
    print("Length:", len(features))

    data_array = np.array(features).reshape(1, -1)

    scaled_data = scaler.transform(data_array)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1] * 100

    result = "Will Purchase" if prediction == 1 else "Will Not Purchase"

    return render_template(
        "index.html",
        prediction_text=result,
        probability=round(probability, 2)
    )

# Admin Dashboard
@app.route("/admin")
def admin():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("admin.html", predictions=rows)

import csv
from flask import Response

@app.route("/export")
def export_csv():

    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions")
    rows = c.fetchall()
    conn.close()

    def generate():
        yield "ID,Input Data,Prediction,Probability,Timestamp\n"
        for row in rows:
            yield f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n"

    return Response(generate(),
                    mimetype="text/csv",
                    headers={"Content-Disposition":
                             "attachment;filename=predictions.csv"})

@app.route('/dashboard')
def dashboard():
    buy_count = len([p for p in predictions if p == 1])
    not_buy_count = len([p for p in predictions if p == 0])
    return render_template('dashboard.html',
                           buy_count=buy_count,
                           not_buy_count=not_buy_count)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
