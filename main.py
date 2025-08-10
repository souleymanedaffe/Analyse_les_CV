import pandas as pd
import streamlit as st
from utils import (
    extract_text_from_pdf_bytes,
    load_job_titles,
    score_texts,
    extract_skills_from_offer_text
)

# --- CONFIG DE LA PAGE ---
st.set_page_config(
    page_title="Classement CV IA",
    page_icon="",
    layout="wide"
)

# --- CSS POUR GROS TEXTES ---
st.markdown("""
    <style>
    /* Fond dégradé */
    .stApp {
        background: linear-gradient(to bottom right, #f5f7fa, #c3cfe2);
        font-family: 'Segoe UI', sans-serif;
    }
    /* Titres XXL */
    .big-title {
        font-size: 50px !important;
        font-weight: 900;
        color: #2b2d42;
        text-align: center;
        padding-bottom: 20px;
    }
    /* Sous-titre */
    .sub-title {
        font-size: 22px;
        font-weight: 500;
        text-align: center;
        color: #555;
        padding-bottom: 30px;
    }
    /* Boutons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 0.8em 1.5em;
        font-size: 20px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    /* Blocs info */
    .info-box {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# --- TITRE ---
st.markdown("<div class='big-title'>🔍 IA de Classement Automatique des Candidats</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>📌 Analyse les CV, détecte les compétences et classe les candidats en quelques secondes</div>", unsafe_allow_html=True)

# Bloc d'explication
st.markdown("""
<div class='info-box'>
    <b>📜 Étapes :</b>  
    1️⃣ Choisir le métier ciblé  
    2️⃣ Fournir l’offre d’emploi (PDF ou texte)  
    3️⃣ L’IA détecte automatiquement les compétences clés  
    4️⃣ Uploader plusieurs CV  
    5️⃣ Analyse et obtention d’une shortlist classée  
</div>
""", unsafe_allow_html=True)

# --- Métier cible ---
jobs = load_job_titles("job_titles.csv")
col1, col2 = st.columns([2, 1])
with col1:
    selected_job = st.selectbox("🧭 **Métier cible :**", jobs, index=0 if jobs else None)
with col2:
    top_k = st.number_input("**NOMBRE DE CV SOUHAITÉ**", 1, 100, 10)

# --- Source des compétences ---
mode = st.radio(
    "📌 **Source des compétences :**",
    ["Téléverser l'offre (PDF/TXT)", "Coller le texte de l'offre"],
    horizontal=True
)

offer_text = ""
detected_skills = []

def show_detected_readonly(skills):
    if skills:
        st.success("✅ Compétences détectées automatiquement :")
        st.markdown("\n".join([f"- {s}" for s in skills]))
    else:
        st.warning("⚠ Aucune compétence détectée automatiquement dans ce texte.")

if mode == "Téléverser l'offre (PDF/TXT)":
    offre_file = st.file_uploader("📄 **Téléverser l'offre**", type=["pdf", "txt"], key="offre")
    if offre_file:
        if offre_file.name.lower().endswith(".pdf"):
            offer_text = extract_text_from_pdf_bytes(offre_file.read())
        else:
            offer_text = offre_file.read().decode("utf-8", errors="ignore")

        st.text_area("📝 **Offre détectée (modifiable)**", value=offer_text, height=220)
        detected_skills = extract_skills_from_offer_text(offer_text, top_k=20)
        show_detected_readonly(detected_skills)

elif mode == "Coller le texte de l'offre":
    offer_text = st.text_area("📋 **Colle ici l'offre :**", height=220, placeholder="Colle l'annonce…")
    if offer_text.strip():
        detected_skills = extract_skills_from_offer_text(offer_text, top_k=20)
        show_detected_readonly(detected_skills)

# --- Construire la requête ---
query_text = selected_job
if detected_skills:
    query_text += " avec compétences en " + ", ".join(detected_skills)

# --- Upload multi CV ---
uploaded_files = st.file_uploader("📄 **Téléverser plusieurs CV (PDF)**", type=["pdf"], accept_multiple_files=True)

if st.button("🚀 Lancer l'analyse", disabled=not (uploaded_files and selected_job)):
    blobs = {f.name: f.getvalue() for f in uploaded_files}

    rows = []
    for name, blob in blobs.items():
        try:
            cv_text = extract_text_from_pdf_bytes(blob)
            if not cv_text:
                rows.append({"fichier": name, "score": 0.0, "pertinence_%": 0.0, "note": "❌ Aucun texte extrait"})
                continue
            s = score_texts(query_text, cv_text)
            rows.append({"fichier": name, "score": s, "pertinence_%": round(s*100, 2), "note": ""})
        except Exception as e:
            rows.append({"fichier": name, "score": 0.0, "pertinence_%": 0.0, "note": f"Erreur: {e}"})

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    st.markdown("<h2>📊 Résultats du Classement</h2>", unsafe_allow_html=True)
    st.dataframe(df[["fichier", "pertinence_%", "note"]].head(top_k), use_container_width=True)

    st.markdown("<h2>✅ Shortlist</h2>", unsafe_allow_html=True)
    selected_files = st.multiselect(
        "Choisis les meilleurs profils :",
        options=df["fichier"].tolist(),
        default=df["fichier"].head(min(3, len(df))).tolist()
    )

    if selected_files:
        shortlist = df[df["fichier"].isin(selected_files)][["fichier", "pertinence_%"]]
        csv = shortlist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger la shortlist (CSV)",
            data=csv,
            file_name=f"shortlist_{selected_job.replace(' ', '_')}.csv",
            mime="text/csv"
        )

    st.markdown("<h2>📂 Télécharger chaque CV lisible</h2>", unsafe_allow_html=True)
    # On ne garde que ceux où note != "❌ Aucun texte extrait"
    df_text_ok = df[df["note"] != "❌ Aucun texte extrait"]

    if df_text_ok.empty:
        st.warning("⚠ Aucun CV lisible par l'IA n'a été trouvé.")
    else:
        for _, row in df_text_ok.head(top_k).iterrows():
            st.download_button(
                label=f"📥 {row['fichier']} ({row['pertinence_%']}%)",
                data=blobs[row["fichier"]],
                file_name=row["fichier"],
                mime="application/pdf",
                key=f"dl_{row['fichier']}"
            )



# Signature
st.markdown("""
<div class="footer">
    Réalisé par <strong>SOULEYMANE DAFFE - DATA SCIENTIST/ANALYST/DEV</strong>
</div>
""", unsafe_allow_html=True)