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
# 📦 CHARGEMENT DU MODÈLE (une seule fois)
# -----------------------------
@st.cache_resource
def load_models():
    pipeline = joblib.load("modele_diabete.pkl")
    best_threshold = joblib.load("seuil_f2.pkl")
    return pipeline, best_threshold

pipeline, best_threshold = load_models()

# -----------------------------
# 🔧 INITIALISATION SESSION STATE
# -----------------------------
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'proba' not in st.session_state:
    st.session_state.proba = None
if 'health_score' not in st.session_state:
    st.session_state.health_score = None
if 'fr_score' not in st.session_state:
    st.session_state.fr_score = None
if 'fr_text' not in st.session_state:
    st.session_state.fr_text = None
if 'fr_color' not in st.session_state:
    st.session_state.fr_color = None
if 'shap_contrib' not in st.session_state:
    st.session_state.shap_contrib = None

# -----------------------------
# 🔧 AIDES / FONCTIONS
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
    HighBP = st.radio("Hypertension ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="HighBP")
    HighChol = st.radio("Cholestérol élevé ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="HighChol")
    BMI = st.slider("IMC (Indice de Masse Corporelle)", 10.0, 50.0, 25.0, key="BMI")
    Stroke = st.radio("Antécédent d'AVC ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="Stroke")
    HeartDiseaseorAttack = st.radio("Maladie cardiaque ou infarctus ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="HeartDiseaseorAttack")

with col2:
    Smoker = st.radio("Fumeur (≥100 cigarettes vie entière) ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="Smoker")
    PhysActivity = st.radio("Activité physique régulière ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="PhysActivity")
    Fruits = st.radio("Fruits ≥ 1 fois/jour ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="Fruits")
    Veggies = st.radio("Légumes ≥ 1 fois/jour ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="Veggies")
    HvyAlcoholConsump = st.radio("Consommation excessive d'alcool ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="HvyAlcoholConsump")

with col3:
    DiffWalk = st.radio("Difficulté à marcher ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", key="DiffWalk")
    Sex = st.radio("Sexe", [0, 1], format_func=lambda x: "Homme" if x == 1 else "Femme", key="Sex")
    Age = st.slider("Tranche d'âge (1=18-24, 13=80+)", 1, 13, 5, key="Age")
    st.markdown("#### 📌 Cliquez pour prédire")
    predict_button = st.button("📊 Lancer la prédiction")

# -----------------------------
# 🔮 PRÉDICTION
# -----------------------------
if predict_button:
    # ✅ Stocker les entrées
    inputs = {
        "HighBP": HighBP, "HighChol": HighChol, "BMI": BMI, "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack, "Smoker": Smoker,
        "PhysActivity": PhysActivity, "Fruits": Fruits, "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump, "DiffWalk": DiffWalk,
        "Sex": Sex, "Age": Age
    }
    st.session_state.inputs = inputs
    X = make_df(inputs)

    # 🔹 Résultat modèle
    proba = pipeline.predict_proba(X)[0][1]
    y_hat = 1 if proba >= best_threshold else 0
    st.session_state.prediction = y_hat
    st.session_state.proba = proba

    # 🔹 Résultat jauge santé
    health_score = render_health_gauge(BMI, HighBP, Smoker, PhysActivity, Fruits, Veggies)
    st.session_state.health_score = health_score

    # 🔹 Résultat FINDRISC
    fr_score = compute_findrisc(BMI, Age, PhysActivity, Fruits, Veggies, HighBP, HighChol)
    fr_text, fr_color = interpret_findrisc(fr_score)
    st.session_state.fr_score = fr_score
    st.session_state.fr_text = fr_text
    st.session_state.fr_color = fr_color

    # 🔹 Explicabilité SHAP
    try:
        rng = np.random.default_rng(123)
        bg = build_background(n=120, seed=123)
        explainer = shap.KernelExplainer(model_proba_fn, bg, link="logit")
        shap_values = explainer.shap_values(X.values, nsamples=200, l1_reg="aic")
        contrib = pd.Series(shap_values[0], index=FEATURES).sort_values(key=lambda s: s.abs(), ascending=False)
        st.session_state.shap_contrib = contrib
    except Exception:
        st.session_state.shap_contrib = None

# -----------------------------
# 🔹 AFFICHAGE SI PRÉDICTION EXISTE
# -----------------------------
if st.session_state.prediction is not None:
    inputs = st.session_state.inputs
    X = make_df(inputs)
    y_hat = st.session_state.prediction
    proba = st.session_state.proba
    health_score = st.session_state.health_score
    fr_score = st.session_state.fr_score
    fr_text = st.session_state.fr_text
    fr_color = st.session_state.fr_color
    contrib = st.session_state.shap_contrib

    # --- Affichage Modèle / Santé / FINDRISC
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

    # --- Fiche patient
    st.subheader("📋 Profil patient (entrées)")
    st.dataframe(X.T.rename(columns={0: "Valeur"}))

    # --- SHAP
    if contrib is not None:
        st.subheader("🔍 Pourquoi ce résultat ? (Explicabilité locale)")
        st.caption("Explication locale basée sur SHAP Kernel (résultats stables pour les mêmes entrées).")
        fig, ax = plt.subplots(figsize=(6, 6))
        contrib.iloc[::-1].plot(kind="barh", ax=ax)
        ax.set_xlabel("Contribution SHAP (log-odds)")
        ax.set_ylabel("Variables")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        st.pyplot(fig)

    # --- Conseils
    st.subheader("💡 Conseils personnalisés (prévention)")
    tips = []
    if BMI > 30: tips.append("Réduire l'IMC diminue le risque cardiométabolique.")
    if HighBP: tips.append("Surveillez votre tension artérielle régulièrement.")
    if Smoker: tips.append("Le sevrage tabagique réduit fortement les risques.")
    if PhysActivity == 0: tips.append("Visez ≥150 min/semaine d’activité physique modérée.")
    if Fruits == 0 or Veggies == 0: tips.append("Augmentez fruits et légumes pour améliorer le profil métabolique.")
    if len(tips) == 0:
        st.success("👍 Profil d'hygiène de vie globalement favorable.")
    else:
        for t in tips: st.write(f"• {t}")

    # --- Export rapport
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
    if contrib is not None:
        report.append("")
        report.append("Top contributeurs (SHAP local) :")
        for name, val in contrib.head(5).items():
            report.append(f"- {name}: {val:+.3f}")
    download_text_button("rapport_diabete.txt", "\n".join(report), "📥 Télécharger le rapport")
