# from flask import Flask, render_template, request, redirect
# from supabase import create_client
# import os
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler # Pour un clustering fiable
# import json

# app = Flask(__name__)

# # Assure-toi que ces variables sont bien configurées sur ton serveur (Render/Heroku)
# SUPABASE_URL = os.environ.get("SUPABASE_URL")
# SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
# supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/submit", methods=["POST"])
# def submit():
#     try:
#         data = {
#             "age": int(request.form.get("age")),
#             "experience": int(request.form.get("experience")),
#             "heures_travail": float(request.form.get("heures_travail")),
#             "clients": int(request.form.get("clients")),
#             "revenu": float(request.form.get("revenu")),
#             "fatigue": int(request.form.get("fatigue")),
#             "accident": request.form.get("accident") == "1"
#         }
#         supabase.table("conducteurs").insert(data).execute()
#         return redirect("/dashboard")
#     except Exception as e:
#         return f"Erreur lors de l'enregistrement : {str(e)}"

# @app.route("/dashboard")
# def dashboard():
#     try:
#         response = supabase.table("conducteurs").select("*").execute()
#         data = response.data
#         df = pd.DataFrame(data)

#         if df.empty or len(df) < 3: # Sécurité : besoin d'au moins 3 points pour le clustering
#             return "<h1>Pas assez de données pour l'analyse (minimum 3 conducteurs requis).</h1><a href='/'>Ajouter des données</a>"

#         # IA : Régression Linéaire (Analyse prédictive)
#         X = df[["heures_travail", "clients"]]
#         y = df["revenu"]
#         model = LinearRegression().fit(X, y)
#         df["revenu_predit"] = model.predict(X)

#         # IA : Clustering (Normalisation obligatoire pour la fiabilité)
#         scaler = StandardScaler()
#         features = df[["revenu", "fatigue"]]
#         scaled_features = scaler.fit_transform(features)
        
#         kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
#         df["cluster"] = kmeans.fit_predict(scaled_features)

#         # Préparation des données pour Plotly (On envoie un dictionnaire, pas un JSON string)
#         graph_data = {
#             "revenu": df["revenu"].tolist(),
#             "heures": df["heures_travail"].tolist(),
#             "fatigue": df["fatigue"].tolist(),
#             "clusters": df["cluster"].tolist(),
#             "revenu_predit": df["revenu_predit"].tolist()
#         }

#         return render_template("dashboard.html", graph_data=graph_data)

#     except Exception as e:
#         return f"Erreur dashboard: {str(e)}"

# if __name__ == "__main__":
#     app.run(debug=True)






from flask import Flask, render_template, request, redirect
from supabase import create_client
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
            "accident": request.form.get("accident") == "1"
        }
        supabase.table("conducteurs").insert(data).execute()
        return redirect("/dashboard")
    except Exception as e:
        return f"Erreur d'enregistrement : {str(e)}"

@app.route("/dashboard")
def dashboard():
    try:
        response = supabase.table("conducteurs").select("*").execute()
        df = pd.DataFrame(response.data)

        if len(df) < 3:
            return "<h1>Données insuffisantes pour l'IA (Besoin de 3 entrées minimum).</h1><a href='/'>Ajouter des données</a>"

        # --- 1. RÉGRESSION LINÉAIRE MULTIPLE (Revenu) ---
        X_reg = df[["heures_travail", "clients"]]
        y_reg = df["revenu"]
        reg_model = LinearRegression().fit(X_reg, y_reg)
        df["revenu_predit"] = reg_model.predict(X_reg)
        r2_score = reg_model.score(X_reg, y_reg)

        # --- 2. CLASSIFICATION NON-SUPERVISÉE (K-Means) ---
        scaler = StandardScaler()
        features_km = df[["revenu", "fatigue"]]
        scaled_km = scaler.fit_transform(features_km)
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        df["cluster"] = kmeans.fit_predict(scaled_km)

        # --- 3. CLASSIFICATION SUPERVISÉE (Random Forest - Risque Accident) ---
        X_clf = df[["fatigue", "experience", "age"]]
        y_clf = df["accident"].astype(int)
        clf_model = RandomForestClassifier(n_estimators=10).fit(X_clf, y_clf)
        # Probabilité de risque d'accident
        df["risque_score"] = clf_model.predict_proba(X_clf)[:, 1] * 100

        # Tri pour un graphique de régression propre
        df_plot = df.sort_values(by="heures_travail")

        graph_data = {
            "heures": df_plot["heures_travail"].tolist(),
            "revenu": df_plot["revenu"].tolist(),
            "revenu_predit": df_plot["revenu_predit"].tolist(),
            "fatigue": df["fatigue"].tolist(),
            "clusters": df["cluster"].tolist(),
            "risque": df["risque_score"].tolist(),
            "r2": round(r2_score, 2)
        }

        # Conversion des données pour le tableau HTML
        table_data = df.to_dict(orient="records")

        return render_template("dashboard.html", graph_data=graph_data, table_data=table_data)

    except Exception as e:
        return f"Erreur Dashboard: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
