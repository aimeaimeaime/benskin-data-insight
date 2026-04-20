# from flask import Flask, render_template, request, redirect
# from supabase import create_client, Client
# import os

# app = Flask(__name__)

# # ================= CONFIG =================
# SUPABASE_URL = os.environ.get("SUPABASE_URL")
# SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# if not SUPABASE_URL or not SUPABASE_KEY:
#     raise Exception("Variables d'environnement Supabase manquantes")

# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# # ================= ROUTES =================

# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/submit", methods=["POST"])
# def submit():
#     try:
#         # récupération + validation
#         age = int(request.form.get("age", 0))
#         experience = int(request.form.get("experience", 0))
#         heures_travail = float(request.form.get("heures_travail", 0))
#         clients = int(request.form.get("clients", 0))
#         revenu = float(request.form.get("revenu", 0))
#         fatigue = int(request.form.get("fatigue", 0))
#         accident = True if request.form.get("accident") == "1" else False

#         # vérification simple
#         if age <= 0 or experience < 0:
#             return "Données invalides"

#         data = {
#             "age": age,
#             "experience": experience,
#             "heures_travail": heures_travail,
#             "clients": clients,
#             "revenu": revenu,
#             "fatigue": fatigue,
#             "accident": accident
#         }

#         # insertion
#         response = supabase.table("conducteurs").insert(data).execute()

#         # debug utile
#         print("INSERT:", response)

#         return redirect("/dashboard")

#     except Exception as e:
#         print("ERREUR SUBMIT:", str(e))
#         return f"Erreur serveur: {str(e)}"


# @app.route("/dashboard")
# def dashboard():
#     try:
#         response = supabase.table("conducteurs").select("*").execute()
#         data = response.data

#         return render_template("result.html", data=data)

#     except Exception as e:
#         print("ERREUR DASHBOARD:", str(e))
#         return f"Erreur dashboard: {str(e)}"


# # ================= RUN =================
# if __name__ == "__main__":
#     app.run(debug=True)





from flask import Flask, render_template, request, redirect
from supabase import create_client
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import json

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        data = {
            "age": int(request.form.get("age")),
            "experience": int(request.form.get("experience")),
            "heures_travail": float(request.form.get("heures_travail")),
            "clients": int(request.form.get("clients")),
            "revenu": float(request.form.get("revenu")),
            "fatigue": int(request.form.get("fatigue")),
            "accident": True if request.form.get("accident") == "1" else False
        }

        supabase.table("conducteurs").insert(data).execute()

        return redirect("/dashboard")

    except Exception as e:
        return f"Erreur: {str(e)}"


@app.route("/dashboard")
def dashboard():
    try:
        response = supabase.table("conducteurs").select("*").execute()
        data = response.data

        df = pd.DataFrame(data)

        if df.empty:
            return "Aucune donnée"

        # ===== IA : Régression =====
        X = df[["heures_travail", "clients"]]
        y = df["revenu"]

        model = LinearRegression()
        model.fit(X, y)

        df["revenu_predit"] = model.predict(X)

        # ===== IA : Clustering =====
        kmeans = KMeans(n_clusters=3, n_init=10)
        df["cluster"] = kmeans.fit_predict(df[["revenu", "fatigue"]])

        # ===== Graphes =====
        graph_data = {
            "revenu": df["revenu"].tolist(),
            "heures": df["heures_travail"].tolist(),
            "fatigue": df["fatigue"].tolist(),
            "clusters": df["cluster"].tolist()
        }

        return render_template("dashboard.html",
                               graph_data=json.dumps(graph_data))

    except Exception as e:
        return f"Erreur dashboard: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)