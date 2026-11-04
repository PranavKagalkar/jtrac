"""J-TRAC v0.1: transparent, non-decisional petition review prototype."""
import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="J-TRAC | Judicial Triage Research", page_icon="⚖️", layout="wide")
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 150_000

@dataclass
class Petition:
    name: str
    text: str
    notes: str


def extract(upload):
    if upload.size > MAX_FILE_BYTES:
        return None, "File exceeds 8 MB limit."
    raw = upload.getvalue()
    ext = Path(upload.name).suffix.lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                return None, "Encrypted PDF is not supported."
            pages = [page.extract_text() or "" for page in reader.pages[:100]]
            text = "\n".join(pages)
        elif ext == ".docx":
            doc = Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs)
            for table in doc.tables:
                for row in table.rows:
                    text += "\n" + " | ".join(cell.text for cell in row.cells)
        elif ext == ".txt":
            text = raw.decode("utf-8", errors="replace")
        else:
            return None, "Unsupported file format."
    except Exception as exc:
        return None, f"Could not extract document text ({type(exc).__name__})."
    text = text[:MAX_TEXT_CHARS].strip()
    if not text:
        return None, "No selectable text found. Scanned PDFs require OCR, which v0.1 does not support."
    return text, None


def find_snippet(text, pattern, width=120):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    start, end = max(0, match.start()-width//2), min(len(text), match.end()+width//2)
    return " ".join(text[start:end].split())

INDICATORS = [
    ("Personal liberty / detention", r"\b(?:habeas corpus|unlawful detention|illegal detention|personal liberty|preventive detention|custody|arrest)\b"),
    ("Bail", r"\b(?:anticipatory bail|regular bail|cancellation of bail|bail application|grant of bail)\b"),
    ("Demolition", r"\b(?:demolition|bulldozer|demolish(?:ed|ing)?)\b"),
    ("Eviction / dispossession", r"\b(?:eviction|dispossession|forcible possession|evict(?:ed|ion)?)\b"),
    ("Death penalty", r"\b(?:death penalty|capital punishment|death sentence)\b"),
    ("Urgent interim relief language", r"\b(?:urgent interim relief|urgent interim order|interim stay|ad interim|interim protection|stay of operation)\b"),
    ("Urgency wording (not proof)", r"\b(?:extremely urgent|grave urgency|urgent hearing|immediate judicial intervention|urgently)\b"),
]
SUBJECT_HINTS = [
    ("Criminal matters — inspect current official roster", r"\b(?:criminal|fir|accused|bail|arrest|custody|conviction)\b"),
    ("Habeas corpus / preventive detention — inspect current official roster", r"\b(?:habeas corpus|preventive detention)\b"),
    ("Demolition — inspect current official roster", r"\b(?:demolition|demolish|bulldozer)\b"),
]


def checklist(text):
    return [(label, find_snippet(text, pattern)) for label, pattern in INDICATORS]


def relationship(a, b):
    legal = bool(re.search(r"\b(?:article\s+(?:14|19|21|32|136)|constitutional validity|fundamental rights)\b", a, re.I) and re.search(r"\b(?:article\s+(?:14|19|21|32|136)|constitutional validity|fundamental rights)\b", b, re.I))
    # These are deliberately weak hints, not legal determinations.
    return "Shared broad constitutional-language indicator" if legal else "No rule-based relationship confirmed"

st.title("⚖️ J-TRAC")
st.caption("Judicial Triage, Roster and Associated Cases · Research prototype v0.1")
st.warning("**Research demonstration only.** Not affiliated with the Supreme Court of India. No binding urgency decisions, formal case tagging, bench allocation, or legal advice. Use only synthetic or lawfully public documents. Files are processed during the current app session; do not upload confidential or personal case records.")

with st.expander("What this version does — and does not do", expanded=False):
    st.write("Uses transparent keyword indicators and TF-IDF document similarity. It does **not** use an LLM, verify court filing defects, check live rosters/circulars, query the Court's case database, or determine legal eligibility for listing. Any flags require independent human review.")

uploads = st.file_uploader("Upload 1–10 sample petitions (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True, help="Use synthetic or public, non-confidential documents only")
if not uploads:
    st.info("Start by uploading a sample petition. You can test with the three synthetic sample files included in the project download.")
    st.stop()
if len(uploads) > 10:
    st.error("Please upload no more than 10 documents per session.")
    st.stop()

petitions = []
for upload in uploads:
    content, error = extract(upload)
    if error:
        st.error(f"{upload.name}: {error}")
    else:
        petitions.append(Petition(upload.name, content, "Text extracted; not independently validated"))
if not petitions:
    st.stop()

st.subheader("1. Petition review")
selected = st.selectbox("Choose a document", range(len(petitions)), format_func=lambda i: petitions[i].name)
p = petitions[selected]
st.caption(f"Extracted {len(p.text):,} characters from {p.name}; text may be incomplete, especially in complex PDFs.")

left, right = st.columns(2)
with left:
    st.markdown("#### Potential urgency indicators")
    hits = checklist(p.text)
    found = [(name, snippet) for name, snippet in hits if snippet]
    if found:
        for name, snippet in found:
            st.markdown(f"**{name}**")
            st.caption(f"Document excerpt: …{snippet}…")
    else:
        st.write("No configured indicators found. This does **not** establish non-urgency.")
    st.info("**Status: Human assessment required.** Filing verification, cured defects, prayers for interim relief, case category, service requirements, mentioning procedure, and applicable current directions are not established by this tool.")
with right:
    st.markdown("#### Possible subject-category cues")
    subject_hits = [(name, find_snippet(p.text, pat)) for name, pat in SUBJECT_HINTS]
    subject_hits = [(name, excerpt) for name, excerpt in subject_hits if excerpt]
    if subject_hits:
        for name, excerpt in subject_hits:
            st.markdown(f"**{name}**")
            st.caption(f"Document excerpt: …{excerpt}…")
    else:
        st.write("No subject-category cues detected.")
    st.caption("No subject code or judge is assigned. Consult the latest official roster and authorised Registry personnel.")

with st.expander("View extracted text"):
    st.text_area("Extracted text", p.text, height=240, disabled=True)

st.subheader("2. Potentially related documents")
if len(petitions) < 2:
    st.info("Upload at least two documents to compare possible similarities.")
else:
    texts = [x.text for x in petitions]
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=15000).fit_transform(texts)
        similarities = cosine_similarity(matrix)
        rows = []
        for j, other in enumerate(petitions):
            if j == selected:
                continue
            rows.append({"Other document": other.name, "Text similarity (0–1)": round(float(similarities[selected,j]), 3), "Relationship cue": relationship(p.text, other.text), "Formal judicial connection": "Not established"})
        st.dataframe(pd.DataFrame(rows).sort_values("Text similarity (0–1)", ascending=False), hide_index=True, use_container_width=True)
        st.caption("TF-IDF similarity is lexical, not a legal finding. Shared terms can produce false matches; different wording can hide genuine connections. A formal connected-case status requires authoritative court records/orders.")
    except ValueError:
        st.warning("Not enough usable text to compute similarity.")

st.subheader("3. Human-review checklist")
for label in [
    "Confirm applicable Supreme Court Rules and latest listing circulars",
    "Check filing status, Registry defects, and whether defects are cured",
    "Inspect actual prayer for urgent interim relief and supporting facts",
    "Check any required service, mentioning request, or judicial direction",
    "Confirm current roster category with authoritative documents",
    "Check court orders and Registry records before treating matters as formally connected",
    "Record reviewer, date, evidence, and unresolved uncertainties",
]:
    st.checkbox(label, key=f"review_{label}")
st.caption("Checkboxes are session-local review aids; they do not authorise any court action or persist as an audit record.")
