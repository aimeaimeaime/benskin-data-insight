from flask import Flask, render_template, request, redirect
from supabase import create_client
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fatigue_color(val):
    """Couleur CSS selon le niveau de fatigue."""
    if val > 7:
        return "#ef4444"
    elif val > 4:
        return "#f59e0b"
    return "#22c55e"


def fatigue_width(val):
    """Largeur CSS (%) de la barre de fatigue."""
    return f"{val * 10}%"


def risk_class(score):
    """Classe CSS du badge risque selon le score."""
    if score > 60:
        return "risk-high"
    elif score > 30:
        return "risk-mid"
    return "risk-low"


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
            "distance_km": float(request.form.get("distance_km")),
            "carburant_litre": float(request.form.get("carburant_litre")),
            "note_client": float(request.form.get("note_client")),
            "zone": request.form.get("zone"),
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

        if len(df) < 5:
            return render_template("insufficient.html", count=len(df))

        # ── Sécurisation : colonnes ajoutées récemment peuvent être NULL ─────
        df["distance_km"]     = pd.to_numeric(df.get("distance_km"),     errors="coerce").fillna(0.0)
        df["carburant_litre"] = pd.to_numeric(df.get("carburant_litre"), errors="coerce").fillna(0.0)
        df["note_client"]     = pd.to_numeric(df.get("note_client"),     errors="coerce").fillna(3.0)
        df["zone"]            = df.get("zone", pd.Series(["Inconnue"] * len(df))).fillna("Inconnue")

        # ── 1. RÉGRESSION LINÉAIRE SIMPLE (Revenu ~ Heures) ──────────────────
        X_simple = df[["heures_travail"]]
        y_rev = df["revenu"]
        reg_simple = LinearRegression().fit(X_simple, y_rev)
        df["revenu_predit_simple"] = reg_simple.predict(X_simple)
        r2_simple = round(reg_simple.score(X_simple, y_rev), 3)
        coef_simple = round(float(reg_simple.coef_[0]), 2)
        intercept_simple = round(float(reg_simple.intercept_), 2)

        # ── 2. RÉGRESSION LINÉAIRE MULTIPLE (Revenu ~ Heures + Clients + Distance)
        X_multi = df[["heures_travail", "clients", "distance_km"]]
        reg_multi = LinearRegression().fit(X_multi, y_rev)
        df["revenu_predit_multi"] = reg_multi.predict(X_multi)
        r2_multi = round(reg_multi.score(X_multi, y_rev), 3)
        coefs_multi = {
            "heures_travail": round(float(reg_multi.coef_[0]), 2),
            "clients":        round(float(reg_multi.coef_[1]), 2),
            "distance_km":    round(float(reg_multi.coef_[2]), 2),
        }

        # ── 3. RÉDUCTION DE DIMENSIONNALITÉ (PCA) ────────────────────────────
        scaler = StandardScaler()
        features_pca = df[["age", "experience", "heures_travail", "clients",
                            "revenu", "fatigue", "distance_km", "carburant_litre", "note_client"]]
        scaled_pca = scaler.fit_transform(features_pca)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_pca)
        df["pca1"] = pca_result[:, 0]
        df["pca2"] = pca_result[:, 1]
        variance_ratio = [round(v * 100, 1) for v in pca.explained_variance_ratio_]

        # ── 4. CLASSIFICATION SUPERVISÉE (Random Forest — Risque Accident) ───
        X_clf = df[["fatigue", "experience", "age", "heures_travail", "distance_km"]]
        y_clf = df["accident"].astype(int)
        clf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_model.fit(X_clf, y_clf)
        df["risque_score"] = clf_model.predict_proba(X_clf)[:, 1] * 100
        feature_importance = dict(zip(
            X_clf.columns,
            [round(v * 100, 1) for v in clf_model.feature_importances_]
        ))

        # ── 5. CLASSIFICATION NON-SUPERVISÉE (K-Means) ───────────────────────
        features_km = df[["revenu", "fatigue", "clients", "note_client"]]
        scaled_km = scaler.fit_transform(features_km)
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        df["cluster"] = kmeans.fit_predict(scaled_km)
        cluster_labels = {0: "Performant", 1: "Modéré", 2: "À risque"}
        df["cluster_label"] = df["cluster"].map(cluster_labels)

        # ── Statistiques descriptives ─────────────────────────────────────────
        stats = {
            "total":      len(df),
            "revenu_moy": round(df["revenu"].mean(), 0),
            "heures_moy": round(df["heures_travail"].mean(), 1),
            "risque_moy": round(df["risque_score"].mean(), 1),
            "accidents":  int(df["accident"].sum()),
            "note_moy":   round(df["note_client"].mean(), 2),
        }

        # ── Données graphiques ────────────────────────────────────────────────
        df_plot = df.sort_values(by="heures_travail")

        graph_data = {
            "heures":               df_plot["heures_travail"].tolist(),
            "revenu":               df_plot["revenu"].tolist(),
            "revenu_predit_simple": df_plot["revenu_predit_simple"].tolist(),
            "r2_simple":            r2_simple,
            "coef_simple":          coef_simple,
            "intercept_simple":     intercept_simple,
            "revenu_predit_multi":  df_plot["revenu_predit_multi"].tolist(),
            "r2_multi":             r2_multi,
            "coefs_multi":          coefs_multi,
            "pca1":                 df["pca1"].tolist(),
            "pca2":                 df["pca2"].tolist(),
            "variance_ratio":       variance_ratio,
            "risque":               df["risque_score"].tolist(),
            "risque_labels":        [f"C{i+1}" for i in range(len(df))],
            "feature_importance":   feature_importance,
            "clusters":             df["cluster"].tolist(),
            "cluster_labels_list":  df["cluster_label"].tolist(),
            "fatigue":              df["fatigue"].tolist(),
            "stats":                stats,
        }

        # ── Table : toute la logique de style calculée ici en Python ─────────
        rows = df[["age", "experience", "heures_travail", "clients",
                   "revenu", "fatigue", "distance_km", "note_client",
                   "zone", "risque_score", "cluster_label"]].copy()

        rows["fatigue_color"] = rows["fatigue"].apply(fatigue_color)
        rows["fatigue_width"] = rows["fatigue"].apply(fatigue_width)
        rows["risk_class"]    = rows["risque_score"].apply(risk_class)

        table_data = rows.to_dict(orient="records")

        return render_template("dashboard.html",
                               graph_data=json.dumps(graph_data),
                               table_data=table_data,
                               stats=stats)

    except Exception as e:
        import traceback
        return f"<pre>Erreur Dashboard:\n{traceback.format_exc()}</pre>"


if __name__ == "__main__":
    app.run(debug=True)