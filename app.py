from flask import Flask, render_template, request, redirect
from supabase import create_client
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Pour un clustering fiable
import json

app = Flask(__name__)

# Assure-toi que ces variables sont bien configurées sur ton serveur (Render/Heroku)
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
        return f"Erreur lors de l'enregistrement : {str(e)}"

@app.route("/dashboard")
def dashboard():
    try:
        response = supabase.table("conducteurs").select("*").execute()
        data = response.data
        df = pd.DataFrame(data)

        if df.empty or len(df) < 3: # Sécurité : besoin d'au moins 3 points pour le clustering
            return "<h1>Pas assez de données pour l'analyse (minimum 3 conducteurs requis).</h1><a href='/'>Ajouter des données</a>"

        # IA : Régression Linéaire (Analyse prédictive)
        X = df[["heures_travail", "clients"]]
        y = df["revenu"]
        model = LinearRegression().fit(X, y)
        df["revenu_predit"] = model.predict(X)

        # IA : Clustering (Normalisation obligatoire pour la fiabilité)
        scaler = StandardScaler()
        features = df[["revenu", "fatigue"]]
        scaled_features = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        df["cluster"] = kmeans.fit_predict(scaled_features)

        # Préparation des données pour Plotly (On envoie un dictionnaire, pas un JSON string)
        graph_data = {
            "revenu": df["revenu"].tolist(),
            "heures": df["heures_travail"].tolist(),
            "fatigue": df["fatigue"].tolist(),
            "clusters": df["cluster"].tolist(),
            "revenu_predit": df["revenu_predit"].tolist()
        }

        return render_template("dashboard.html", graph_data=graph_data)

    except Exception as e:
        return f"Erreur dashboard: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
