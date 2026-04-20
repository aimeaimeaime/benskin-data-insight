import os
from flask import Flask, render_template, request, redirect, jsonify
from supabase import create_client

app = Flask(__name__)

# ENV VARIABLES (Render / local compatible)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    data = {
        "age": int(request.form["age"]),
        "experience": int(request.form["experience"]),
        "heures_travail": float(request.form["heures_travail"]),
        "clients": int(request.form["clients"]),
        "revenu": float(request.form["revenu"]),
        "fatigue": int(request.form["fatigue"]),
        "accident": True if request.form["accident"] == "1" else False
    }

    supabase.table("conducteurs").insert(data).execute()

    return redirect("/dashboard")


@app.route("/dashboard")
def dashboard():
    response = supabase.table("conducteurs").select("*").execute()
    data = response.data

    if not data:
        return render_template("dashboard.html", stats={}, data=[])

    import pandas as pd
    df = pd.DataFrame(data)

    stats = {
        "total": len(df),
        "revenu_moyen": round(df["revenu"].mean(), 2),
        "fatigue_moyenne": round(df["fatigue"].mean(), 2),
        "accident_rate": round(df["accident"].mean() * 100, 2)
    }

    return render_template("dashboard.html", stats=stats, data=data)


# REQUIRED FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)