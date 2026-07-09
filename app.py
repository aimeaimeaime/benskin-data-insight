

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

# ══════════════════════════════════════════════════════════════
#  RÈGLES DE VALIDATION BACKEND — source de vérité unique
# ══════════════════════════════════════════════════════════════
FIELD_RULES = {
    "age":             {"type": int,   "min": 18,  "max": 70,     "label": "Âge"},
    "experience":      {"type": int,   "min": 0,   "max": 40,     "label": "Années d'expérience"},
    "heures_travail":  {"type": float, "min": 1.0, "max": 24.0,   "label": "Heures de travail"},
    "clients":         {"type": int,   "min": 1,   "max": 100,    "label": "Nombre de clients"},
    "revenu":          {"type": float, "min": 500, "max": 100000, "label": "Revenu journalier"},
    "fatigue":         {"type": int,   "min": 1,   "max": 10,     "label": "Niveau de fatigue"},
    "distance_km":     {"type": float, "min": 1.0, "max": 500.0,  "label": "Distance parcourue"},
    "carburant_litre": {"type": float, "min": 0.5, "max": 50.0,   "label": "Carburant consommé"},
    "note_client":     {"type": float, "min": 0.0, "max": 5.0,    "label": "Note client"},
}


def parse_and_validate(form_data):
    """
    Parse, convertit et valide tous les champs du formulaire.
    - Les entiers saisis dans un champ float sont automatiquement convertis.
    - Les valeurs hors bornes lèvent une ValueError avec un message clair.
    Retourne un dict propre ou lève ValueError avec le message d'erreur.
    """
    result = {}
    errors = []

    for field, rules in FIELD_RULES.items():
        raw = form_data.get(field, "").strip()

        # Champ manquant ou vide
        if not raw:
            errors.append(f"Le champ « {rules['label']} » est obligatoire.")
            continue

        # Conversion numérique
        try:
            numeric = float(raw)          # toujours parser en float d'abord
        except ValueError:
            errors.append(
                f"« {rules['label']} » : la valeur « {raw} » n'est pas un nombre valide."
            )
            continue

        # Vérification des bornes
        if numeric < rules["min"] or numeric > rules["max"]:
            errors.append(
                f"« {rules['label']} » doit être compris entre "
                f"{rules['min']} et {rules['max']} "
                f"(valeur reçue : {numeric})."
            )
            continue

        # Conversion vers le type cible
        # float(entier) → float avec décimale : 8 → 8.0
        if rules["type"] == float:
            result[field] = round(float(numeric), 2)
        else:
            # int : on arrondit au cas où le client envoie "8.0"
            result[field] = int(round(numeric))

    # Champ zone (string)
    zone = form_data.get("zone", "").strip()
    valid_zones = {
        "Centre-ville", "Mvog-Ada", "Mfoundi",
        "Bastos", "Ngousso", "Ntougou", "Olembe", "Nkol-Afeme",
        "Biyem-Assi", "Melen", "Essos", "Nkomo", "Mvog-Betsi", "Ekounou",
        "Nkolbisson", "Mimboman", "Nkol-Messeng",
        "Mendong", "Nkoldongo", "Kondengui",
        "Autre"
    }
    if not zone:
        errors.append("Le champ « Zone de travail » est obligatoire.")
    elif zone not in valid_zones:
        errors.append(f"Zone « {zone} » non reconnue.")
    else:
        result["zone"] = zone

    # Champ accident (booléen)
    result["accident"] = form_data.get("accident") == "1"

    if errors:
        raise ValueError(" | ".join(errors))

    return result


# ══════════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES TABLEAU
# ══════════════════════════════════════════════════════════════
def fatigue_color(val):
    if val > 7:   return "#ef4444"
    if val > 4:   return "#f59e0b"
    return "#22c55e"

def fatigue_width(val):
    return f"{int(val) * 10}%"

def risk_class(score):
    if score > 60: return "risk-high"
    if score > 30: return "risk-mid"
    return "risk-low"


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        data = parse_and_validate(request.form)
        supabase.table("conducteurs").insert(data).execute()
        return redirect("/dashboard")

    except ValueError as ve:
        # Erreur de validation : message lisible renvoyé à l'utilisateur
        return render_template("index.html",
                               error=str(ve)), 400

    except Exception as e:
        return render_template("index.html",
                               error=f"Erreur serveur inattendue : {str(e)}"), 500


@app.route("/dashboard")
def dashboard():
    try:
        response = supabase.table("conducteurs").select("*").execute()
        df = pd.DataFrame(response.data)

        if len(df) < 5:
            return render_template("insufficient.html", count=len(df))

        # ── Sécurisation colonnes potentiellement NULL ────────────────────────
        df["distance_km"]     = pd.to_numeric(df.get("distance_km"),     errors="coerce").fillna(0.0)
        df["carburant_litre"] = pd.to_numeric(df.get("carburant_litre"), errors="coerce").fillna(0.0)
        df["note_client"]     = pd.to_numeric(df.get("note_client"),     errors="coerce").fillna(3.0)
        df["zone"]            = df.get("zone", pd.Series(["Inconnue"] * len(df))).fillna("Inconnue")

        # ── 1. RÉGRESSION LINÉAIRE SIMPLE (Revenu ~ Heures) ──────────────────
        X_simple = df[["heures_travail"]]
        y_rev    = df["revenu"]
        reg_simple = LinearRegression().fit(X_simple, y_rev)
        df["revenu_predit_simple"] = reg_simple.predict(X_simple)
        r2_simple        = round(reg_simple.score(X_simple, y_rev), 3)
        coef_simple      = round(float(reg_simple.coef_[0]), 2)
        intercept_simple = round(float(reg_simple.intercept_), 2)

        # ── 2. RÉGRESSION LINÉAIRE MULTIPLE (Revenu ~ Heures + Clients + Distance)
        X_multi  = df[["heures_travail", "clients", "distance_km"]]
        reg_multi = LinearRegression().fit(X_multi, y_rev)
        df["revenu_predit_multi"] = reg_multi.predict(X_multi)
        r2_multi    = round(reg_multi.score(X_multi, y_rev), 3)
        coefs_multi = {
            "heures_travail": round(float(reg_multi.coef_[0]), 2),
            "clients":        round(float(reg_multi.coef_[1]), 2),
            "distance_km":    round(float(reg_multi.coef_[2]), 2),
        }

        # ── 3. RÉDUCTION DE DIMENSIONNALITÉ (PCA) ────────────────────────────
        scaler       = StandardScaler()
        features_pca = df[["age", "experience", "heures_travail", "clients",
                            "revenu", "fatigue", "distance_km",
                            "carburant_litre", "note_client"]]
        scaled_pca   = scaler.fit_transform(features_pca)
        pca          = PCA(n_components=2)
        pca_result   = pca.fit_transform(scaled_pca)
        df["pca1"]   = pca_result[:, 0]
        df["pca2"]   = pca_result[:, 1]
        variance_ratio = [round(v * 100, 1) for v in pca.explained_variance_ratio_]

        # ── 4. CLASSIFICATION SUPERVISÉE (Random Forest) ─────────────────────
        X_clf     = df[["fatigue", "experience", "age", "heures_travail", "distance_km"]]
        y_clf     = df["accident"].astype(int)
        clf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_model.fit(X_clf, y_clf)
        df["risque_score"]   = clf_model.predict_proba(X_clf)[:, 1] * 100
        feature_importance   = dict(zip(
            X_clf.columns,
            [round(v * 100, 1) for v in clf_model.feature_importances_]
        ))

        # ── 5. CLASSIFICATION NON-SUPERVISÉE (K-Means) ───────────────────────
        features_km      = df[["revenu", "fatigue", "clients", "note_client"]]
        scaled_km        = scaler.fit_transform(features_km)
        kmeans           = KMeans(n_clusters=3, n_init=10, random_state=42)
        df["cluster"]    = kmeans.fit_predict(scaled_km)
        cluster_map      = {0: "Performant", 1: "Modéré", 2: "À risque"}
        df["cluster_label"] = df["cluster"].map(cluster_map)

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
        df_plot    = df.sort_values(by="heures_travail")
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

        # ── Table avec style calculé en Python ───────────────────────────────
        rows = df[["age", "experience", "heures_travail", "clients",
                   "revenu", "fatigue", "distance_km", "note_client",
                   "zone", "risque_score", "cluster_label"]].copy()
        rows["fatigue_color"] = rows["fatigue"].apply(fatigue_color)
        rows["fatigue_width"] = rows["fatigue"].apply(fatigue_width)
        rows["risk_class"]    = rows["risque_score"].apply(risk_class)
        table_data            = rows.to_dict(orient="records")

        return render_template("dashboard.html",
                               graph_data=json.dumps(graph_data),
                               table_data=table_data,
                               stats=stats)

    except Exception as e:
        import traceback
        return f"<pre>Erreur Dashboard:\n{traceback.format_exc()}</pre>"


if __name__ == "__main__":
    app.run(debug=True)