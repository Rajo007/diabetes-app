import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import base64
from io import BytesIO

# -----------------------------
# 🎯 CONFIGURATION DE LA PAGE
# -----------------------------
st.set_page_config(
    page_title="🩺 Dépistage du Diabète",
    layout="wide",
    page_icon="🩺"
)

# -----------------------------
# 📦 CHARGEMENT DU MODÈLE
# -----------------------------
pipeline = joblib.load("modele_diabete.pkl")
best_threshold = joblib.load("seuil_f2.pkl")

# -----------------------------
# 🔧 AIDES
# -----------------------------
FEATURES = [
    "HighBP", "HighChol", "BMI", "Stroke", "HeartDiseaseorAttack",
    "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "DiffWalk", "Sex", "Age"
]

def make_df(values_dict):
    return pd.DataFrame([[values_dict[k] for k in FEATURES]], columns=FEATURES)

def build_background(n=120, seed=42):
    rng = np.random.default_rng(seed)
    bg = pd.DataFrame(columns=FEATURES)
    bg["HighBP"] = rng.integers(0, 2, n)
    bg["HighChol"] = rng.integers(0, 2, n)
    bg["BMI"] = rng.uniform(18, 40, n)
    bg["Stroke"] = rng.integers(0, 2, n)
    bg["HeartDiseaseorAttack"] = rng.integers(0, 2, n)
    bg["Smoker"] = rng.integers(0, 2, n)
    bg["PhysActivity"] = rng.integers(0, 2, n)
    bg["Fruits"] = rng.integers(0, 2, n)
    bg["Veggies"] = rng.integers(0, 2, n)
    bg["HvyAlcoholConsump"] = rng.integers(0, 2, n)
    bg["DiffWalk"] = rng.integers(0, 2, n)
    bg["Sex"] = rng.integers(0, 2, n)
    bg["Age"] = rng.integers(1, 14, n)
    return bg

def model_proba_fn(X_like):
    if isinstance(X_like, np.ndarray):
        X_df = pd.DataFrame(X_like, columns=FEATURES)
    else:
        X_df = pd.DataFrame(X_like, columns=FEATURES)
    return pipeline.predict_proba(X_df)[:, 1]

def render_health_gauge(BMI, HighBP, Smoker, PhysActivity, Fruits, Veggies):
    health_score = 100
    if BMI > 30: health_score -= 15
    if HighBP: health_score -= 10
    if Smoker: health_score -= 10
    if PhysActivity == 0: health_score -= 10
    if Fruits == 0: health_score -= 5
    if Veggies == 0: health_score -= 5

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "Score Santé Global"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if health_score > 70 else "orange" if health_score > 40 else "red"}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    return health_score

def compute_findrisc(BMI, Age, PhysActivity, Fruits, Veggies, HighBP, HighChol, FamilyHistory=0):
    score = 0
    if Age >= 9: score += 4  # ≥65 ans
    elif Age >= 7: score += 3  # 55–64
    elif Age >= 5: score += 2  # 45–54
    if BMI >= 30: score += 3
    elif BMI >= 25: score += 1
    if PhysActivity == 0: score += 2
    if Fruits == 0 or Veggies == 0: score += 1
    if HighBP == 1: score += 2
    if HighChol == 1: score += 2
    score += FamilyHistory
    return score

def interpret_findrisc(score):
    if score < 7: return "Faible (1/100 développeront un diabète)", "green"
    elif score < 12: return "Légèrement élevé (1/25)", "orange"
    elif score < 15: return "Modéré (1/6)", "orange"
    elif score < 20: return "Élevé (1/3)", "red"
    else: return "Très élevé (1/2)", "red"

def download_text_button(filename: str, text: str, label: str):
    buffer = BytesIO()
    buffer.write(text.encode("utf-8"))
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# -----------------------------
# 🎨 HEADER
# -----------------------------
st.title("🩺 Application de Dépistage du Diabète")
st.markdown(
    "Outil de **dépistage préliminaire** basé sur un modèle ML. "
    "⚠️ Objectif : **minimiser les faux négatifs** (rappel élevé). "
    "En cas de résultat positif, consultez un médecin."
)
st.markdown("---")

# -----------------------------
# 📝 FORMULAIRE
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    HighBP = st.radio("Hypertension ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    HighChol = st.radio("Cholestérol élevé ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    BMI = st.slider("IMC (Indice de Masse Corporelle)", 10.0, 50.0, 25.0)
    Stroke = st.radio("Antécédent d'AVC ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    HeartDiseaseorAttack = st.radio("Maladie cardiaque ou infarctus ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

with col2:
    Smoker = st.radio("Fumeur (≥100 cigarettes vie entière) ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    PhysActivity = st.radio("Activité physique régulière ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    Fruits = st.radio("Fruits ≥ 1 fois/jour ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    Veggies = st.radio("Légumes ≥ 1 fois/jour ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    HvyAlcoholConsump = st.radio("Consommation excessive d'alcool ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

with col3:
    DiffWalk = st.radio("Difficulté à marcher ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    Sex = st.radio("Sexe", [0, 1], format_func=lambda x: "Homme" if x == 1 else "Femme")
    Age = st.slider("Tranche d'âge (1=18-24, 13=80+)", 1, 13, 5)
    st.markdown("#### 📌 Cliquez pour prédire")
    predict_button = st.button("📊 Lancer la prédiction", use_container_width=True)

# -----------------------------
# 🔮 PRÉDICTION
# -----------------------------
if predict_button:
    inputs = {
        "HighBP": HighBP, "HighChol": HighChol, "BMI": BMI, "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack, "Smoker": Smoker,
        "PhysActivity": PhysActivity, "Fruits": Fruits, "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump, "DiffWalk": DiffWalk,
        "Sex": Sex, "Age": Age
    }
    X = make_df(inputs)

    # Résultat modèle
    proba = pipeline.predict_proba(X)[0][1]
    y_hat = 1 if proba >= best_threshold else 0

    # Résultat jauge santé
    health_score = render_health_gauge(BMI, HighBP, Smoker, PhysActivity, Fruits, Veggies)

    # Résultat FINDRISC
    fr_score = compute_findrisc(BMI, Age, PhysActivity, Fruits, Veggies, HighBP, HighChol)
    fr_text, fr_color = interpret_findrisc(fr_score)

    fig_fr = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fr_score,
        title={'text': "Score FINDRISC"},
        gauge={
            'axis': {'range': [0, 15]},  # max 15 pour la jauge
            'bar': {'color': fr_color},
            'steps': [
                {'range': [0, 4], 'color': "green"},
                {'range': [5, 8], 'color': "yellow"},
                {'range': [9, 12], 'color': "orange"},
                {'range': [13, 15], 'color': "red"}
            ]
        }
    ))
    st.plotly_chart(fig_fr, use_container_width=True)

    # -----------------------------
    # AFFICHAGE DES TROIS BLOCS
    # -----------------------------
    st.markdown("## 🧾 Résultats comparés")
    colA, colB, colC = st.columns(3)

    with colA:
        st.subheader("🔮 Modèle ML")
        if y_hat == 1:
            st.error(f"⚠️ Risque détecté ({proba*100:.1f} %)")
        else:
            st.success(f"✅ Aucun signe détecté ({proba*100:.1f} %)")

    with colB:
        st.subheader("📊 Santé globale (jauge)")
        st.metric("Score santé", f"{health_score}/100")

    with colC:
        st.subheader("📌 FINDRISC")
        st.write(f"Score : **{fr_score}**")
        st.markdown(f"<span style='color:{fr_color}; font-weight:bold'>{fr_text}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------
    # 📋 FICHE PATIENT
    # -----------------------------
    st.subheader("📋 Profil patient (entrées)")
    st.dataframe(X.T.rename(columns={0: "Valeur"}))

   




    # -----------------------------
    # 🧠 EXPLICABILITÉ DU MODÈLE (SHAP)
    # -----------------------------
    st.subheader("🔍 Pourquoi ce résultat ? (Explicabilité locale)")
    st.caption("Explication locale basée sur SHAP Kernel (résultats stables pour les mêmes entrées).")

    try:
        # 🔹 Fixer le seed pour que SHAP soit déterministe
        rng = np.random.default_rng(123)  # seed fixe
        bg = build_background(n=120, seed=123)  # même seed pour cohérence

        # 🔹 Initialisation de l’explainer Kernel SHAP (déterministe)
        explainer = shap.KernelExplainer(model_proba_fn, bg, link="logit")

        # 🔹 Calcul des valeurs SHAP pour ce patient (stable)
        shap_values = explainer.shap_values(X.values, nsamples=200, l1_reg="aic")
        phi = shap_values[0]  # KernelExplainer retourne une liste

        # 🔹 Organisation des contributions par importance
        contrib = pd.Series(phi, index=FEATURES).sort_values(key=lambda s: s.abs(), ascending=False)

        # 🔹 Affichage graphique (toutes les variables, pour plus de cohérence)
        st.markdown("**Contributions locales (toutes les variables, pour ce patient)**")
        fig, ax = plt.subplots(figsize=(6, 6))
        contrib.iloc[::-1].plot(kind="barh", ax=ax)
        ax.set_xlabel("Contribution SHAP (log-odds)")
        ax.set_ylabel("Variables")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        st.pyplot(fig)

        # 🔹 Interprétation texte des top 5 variables
        st.markdown("**Interprétation rapide (Top 6)**")
        for name, val in contrib.head(6).items():
            direction = "↑ augmente" if val > 0 else "↓ diminue"
            st.write(f"- **{name}** : contribution {direction} le risque "
                     f"(valeur SHAP {val:+.3f}) — valeur saisie = {X.iloc[0][name]}")

        st.markdown("---")

        # 🔹 Importance globale approximative (fond de référence, stable)
        st.subheader("🌐 Importance globale approximative (sur fond de référence)")
        bg_sample = bg.sample(n=min(100, len(bg)), random_state=123).values
        shap_bg = explainer.shap_values(bg_sample, nsamples=200, l1_reg="aic")
        imp_global = pd.Series(np.mean(np.abs(shap_bg), axis=0), index=FEATURES).sort_values(ascending=True)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        imp_global.plot(kind="barh", ax=ax2)
        ax2.set_xlabel("Importance SHAP moyenne |φ|")
        ax2.set_ylabel("Variables")
        ax2.grid(axis="x", linestyle="--", alpha=0.4)
        st.pyplot(fig2)

    except Exception as e:
        st.warning("Explicabilité SHAP non disponible. Vérifiez `shap` dans requirements.txt. "
                   f"Détails: {e}")









    # -----------------------------
    # 💡 CONSEILS PERSONNALISÉS
    # -----------------------------
    st.subheader("💡 Conseils personnalisés (prévention)")
    tips = []
    if BMI > 30:
        tips.append("Réduire l'IMC diminue le risque cardiométabolique.")
    if HighBP:
        tips.append("Surveillez votre tension artérielle régulièrement.")
    if Smoker:
        tips.append("Le sevrage tabagique réduit fortement les risques.")
    if PhysActivity == 0:
        tips.append("Visez ≥150 min/semaine d’activité physique modérée.")
    if Fruits == 0 or Veggies == 0:
        tips.append("Augmentez fruits et légumes pour améliorer le profil métabolique.")
    if len(tips) == 0:
        st.success("👍 Profil d'hygiène de vie globalement favorable.")
    else:
        for t in tips:
            st.write(f"• {t}")

    # -----------------------------
    # 📑 EXPORT RAPPORT
    # -----------------------------
    st.subheader("📑 Exporter un rapport (texte)")
    report = [
        f"Résultat dépistage : {'POSITIF' if y_hat==1 else 'NEGATIF'}",
        f"Probabilité estimée (modèle) : {proba*100:.1f} %",
        f"Seuil décisionnel utilisé : {best_threshold:.4f}",
        f"Score Santé (indicatif) : {health_score}/100",
        f"Score FINDRISC : {fr_score} ({fr_text})",
        "",
        "Entrées principales :"
    ]
    for k in FEATURES:
        report.append(f"- {k}: {X.iloc[0][k]}")
    if 'contrib' in locals():
        report.append("")
        report.append("Top contributeurs (SHAP local) :")
        for name, val in contrib.head(5).items():
            report.append(f"- {name}: {val:+.3f}")

    download_text_button("rapport_diabete.txt", "\n".join(report), "📥 Télécharger le rapport")
