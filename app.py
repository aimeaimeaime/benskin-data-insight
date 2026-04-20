from flask import Flask, render_template, request, redirect
from supabase import create_client, Client
import os

app = Flask(__name__)

# ================= CONFIG =================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Variables d'environnement Supabase manquantes")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        # récupération + validation
        age = int(request.form.get("age", 0))
        experience = int(request.form.get("experience", 0))
        heures_travail = float(request.form.get("heures_travail", 0))
        clients = int(request.form.get("clients", 0))
        revenu = float(request.form.get("revenu", 0))
        fatigue = int(request.form.get("fatigue", 0))
        accident = True if request.form.get("accident") == "1" else False

        # vérification simple
        if age <= 0 or experience < 0:
            return "Données invalides"

        data = {
            "age": age,
            "experience": experience,
            "heures_travail": heures_travail,
            "clients": clients,
            "revenu": revenu,
            "fatigue": fatigue,
            "accident": accident
        }

        # insertion
        response = supabase.table("conducteurs").insert(data).execute()

        # debug utile
        print("INSERT:", response)

        return redirect("/dashboard")

    except Exception as e:
        print("ERREUR SUBMIT:", str(e))
        return f"Erreur serveur: {str(e)}"


@app.route("/dashboard")
def dashboard():
    try:
        response = supabase.table("conducteurs").select("*").execute()
        data = response.data

        return render_template("result.html", data=data)

    except Exception as e:
        print("ERREUR DASHBOARD:", str(e))
        return f"Erreur dashboard: {str(e)}"


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)