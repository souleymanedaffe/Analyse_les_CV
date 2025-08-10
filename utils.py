import re
from io import BytesIO
from typing import List

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# ⚙️ Modèle SBERT
# - Si tu as le modèle en local : mets le chemin du dossier
# - Sinon, mets le nom HF "all-MiniLM-L6-v2" (téléchargement auto)
# ------------------------------------------------------------
MODEL_PATH = "./models/all-MiniLM-L6-v2"   # ex: dossier local
# MODEL_PATH = "all-MiniLM-L6-v2"         # ou nom Hugging Face

_model = None
_kw_model = None


def load_model() -> SentenceTransformer:
    """Charge le modèle SBERT (une seule fois)."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_PATH)
        except Exception:
            # Fallback : tente le nom public si le chemin local n'existe pas
            _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def kw_model() -> KeyBERT:
    """KeyBERT basé sur le modèle SBERT chargé."""
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT(load_model())
    return _kw_model


# -------------------- Extraction texte PDF --------------------

def _extract_with_pymupdf(data: bytes) -> str:
    parts: List[str] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text("text") or "")
    return "\n".join(parts).strip()


def _extract_with_pdfplumber(data: bytes) -> str:
    parts: List[str] = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            parts.append(t)
    return "\n".join(parts).strip()


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extrait le texte d'un PDF (PyMuPDF -> fallback pdfplumber)."""
    txt = _extract_with_pymupdf(data)
    if txt:
        return txt
    return _extract_with_pdfplumber(data)


# -------------------- Métier / Similarité --------------------

def load_job_titles(path: str = "job_titles.csv") -> List[str]:
    df = pd.read_csv(path)
    return df["job_title"].dropna().astype(str).tolist()


def score_texts(query_text: str, cv_text: str) -> float:
    """Similarité cosinus entre (métier + compétences) et texte de CV."""
    model = load_model()
    emb = model.encode([query_text, cv_text])
    return float(cosine_similarity([emb[0]], [emb[1]])[0, 0])


# -------------------- Nettoyage & extraction compétences --------------------

def clean_offer_text(text: str) -> str:
    """Nettoyage léger (puces, emojis, espaces)."""
    if not text:
        return ""
    t = re.sub(r"[•\u2022▪️➡️✅👉🔎🚀🧠🤝🎁📌🔧💼📝⭐]+", " ", text)
    t = t.replace("\r", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def extract_skills_from_offer_text(text: str, top_k: int = 20) -> list[str]:
    """
    Extrait automatiquement des compétences/mots-clés depuis le texte de l'offre,
    sans liste pré-définie. Utilise KeyBERT (SBERT) avec un post-traitement simple.
    """
    if not text or not text.strip():
        return []

    # Nettoyage de base
    t = re.sub(r"[•\u2022▪️➡️✅👉🔎🚀🧠🤝🎁📌🔧💼📝⭐]+", " ", text)
    t = t.replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # Extraction via KeyBERT (modèle SBERT)
    keywords = kw_model().extract_keywords(
        t,
        keyphrase_ngram_range=(1, 3),   # 1 à 3 mots
        stop_words=None,                # Pas de stop words pour éviter de retirer des techs
        use_mmr=True,                   # Maximal Marginal Relevance pour la diversité
        diversity=0.7,
        nr_candidates=80,
        top_n=top_k
    )

    # Post-traitement : découpe, nettoyage et dédoublonnage
    found = set()
    for phrase, _ in keywords:
        for chunk in re.split(r"[•\-\u2022,;/\n]+", phrase):
            c = re.sub(r"\s+", " ", chunk).strip(" .;,-·\t")
            c = re.sub(r"^\d+\.?\s*", "", c)
            if c:
                found.add(c)

    return list(found)[:top_k]
