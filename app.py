import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import json
import re
import os
import requests
import pandas as pd
import sqlite3
import bcrypt
from datetime import datetime
from bs4 import BeautifulSoup
from pymongo import MongoClient
from openai import OpenAI
from functools import partial
from typing import TypedDict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ── URIs — defined first so all functions can use them ───────────────────────────────
MONGO_URI      = os.getenv("MONGO_URI")
AUTH_MONGO_URI = os.getenv("AUTH_MONGO_URI")


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Islamic Knowledge Assistant",
    page_icon="☪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Cinzel:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

:root {
    --gold:       #C9A84C;
    --gold-light: #E8D5A3;
    --dark:       #0D0D0D;
    --surface:    #141414;
    --surface2:   #1C1C1C;
    --border:     #2A2A2A;
    --text:       #E8E0D0;
    --text-muted: #8A8070;
}

/* Light mode override */
@media (prefers-color-scheme: light) {
    :root {
        --dark:       #FAFAFA;
        --surface:    #F0F0F0;
        --surface2:   #E8E8E8;
        --border:     #D0D0D0;
        --text:       #1A1A1A;
        --text-muted: #666666;
    }
}

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: var(--dark) !important;
    color: var(--text) !important;
}

/* Force dark background on Streamlit app container */
.stApp {
    background-color: var(--dark) !important;
    color: var(--text) !important;
}

/* Fix white flash on load */
.main .block-container {
    background-color: var(--dark) !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem; max-width: 1200px; }

/* Sidebar — always visible fix */
[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
    min-width: 260px !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 260px !important;
    width: 260px !important;
    transform: none !important;
    visibility: visible !important;
}
/* Collapse button override — keep sidebar always open */
button[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Cinzel', serif;
    color: var(--gold);
}

/* Page title */
.page-title {
    font-family: 'Cinzel', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-family: 'Amiri', serif;
    font-size: 1.3rem;
    color: var(--text-muted);
    margin-bottom: 2rem;
}

/* Divider */
.gold-divider {
    border: none;
    border-top: 1px solid var(--gold);
    opacity: 0.3;
    margin: 1.5rem 0;
}

/* Cards */
.result-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    font-family: 'Source Sans 3', sans-serif;
    white-space: pre-wrap;
    line-height: 1.8;
}
.arabic-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-right: 3px solid var(--gold);
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    font-family: 'Amiri', serif;
    font-size: 1.25rem;
    direction: rtl;
    text-align: right;
    line-height: 2.2;
}
.score-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-top: 3px solid var(--gold);
    border-radius: 4px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.9rem;
}
.section-label {
    font-family: 'Cinzel', serif;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.badge {
    display: inline-block;
    background: var(--gold);
    color: var(--dark);
    font-family: 'Cinzel', serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    margin-bottom: 1rem;
}
.warning-card {
    background: #1a1200;
    border: 1px solid #5a4000;
    border-left: 3px solid #C9A84C;
    border-radius: 4px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    font-family: 'Amiri', serif;
    font-size: 1.1rem;
    direction: rtl;
    text-align: right;
    line-height: 2;
}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    border-radius: 4px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 1px var(--gold) !important;
}

/* Buttons */
.stButton > button {
    background: var(--gold) !important;
    color: var(--dark) !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Radio */
.stRadio label { color: var(--text) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--gold) !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
}
.streamlit-expanderContent {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Cinzel', serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-muted) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .page-title { font-size: 1.4rem !important; }
    .page-subtitle { font-size: 1rem !important; }
    .arabic-card { font-size: 1rem !important; padding: 1rem !important; }
    .result-card { font-size: 0.9rem !important; padding: 1rem !important; }
    .stButton > button { width: 100% !important; }
}

</style>
""", unsafe_allow_html=True)


# ── Database: init, log, fetch ─────────────────────────────────────────────────
def init_db():
    pass  # Logs now stored in MongoDB gp_auth database

@st.cache_resource
def get_logs_collection():
    client = MongoClient(AUTH_MONGO_URI)
    db     = client["gp_auth"]
    return db["login_logs"], db["session_logs"]

def log_login(username, name):
    login_col, _ = get_logs_collection()
    login_col.insert_one({
        "username":  username,
        "name":      name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def log_session(username, final_state: dict):
    _, session_col = get_logs_collection()
    session_col.insert_one({
        "username":            username,
        "question":            final_state.get("Question", ""),
        "request_type":        final_state.get("User_Request_type", ""),
        "retrieved_hadith":    final_state.get("Retrieved_Hadith", ""),
        "retrieved_quran":     final_state.get("Retrieved_Quran", ""),
        "generated_khutbah":   final_state.get("generated_khutbah", ""),
        "verification_result": final_state.get("verification_result", ""),
        "generated_answer":    final_state.get("generated_answer", ""),
        "best_khutbah":        final_state.get("best_khutbah", ""),
        "source_hadiths":      final_state.get("source_hadiths", []),
        "source_quran":        final_state.get("source_quran", []),
        "all_attempts":        final_state.get("all_attempts", []),
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def get_login_logs():
    login_col, _ = get_logs_collection()
    docs = list(login_col.find({}, {"_id": 0}).sort("timestamp", -1))
    return pd.DataFrame(docs) if docs else pd.DataFrame(columns=["username","name","timestamp"])

def get_session_logs():
    _, session_col = get_logs_collection()
    docs = list(session_col.find({}, {"_id": 0}).sort("timestamp", -1))
    return pd.DataFrame(docs) if docs else pd.DataFrame(columns=["username","question","request_type","timestamp"])

init_db()

# ── MongoDB Auth ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_auth_collection():
    client = MongoClient(AUTH_MONGO_URI)
    db     = client["gp_auth"]
    col    = db["users"]
    if col.count_documents({}) == 0:
        col.insert_one({
            "username":   "admin",
            "name":       "Admin",
            "email":      "Jalaltoubeh@gmail.com",
            "password":   bcrypt.hashpw("xxxx".encode(), bcrypt.gensalt()).decode(),
            "role":       "admin",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    return col

def auth_find_user(username: str):
    return get_auth_collection().find_one({"username": username}, {"_id": 0})

def auth_verify_password(username: str, password: str) -> bool:
    user = auth_find_user(username)
    if not user:
        return False
    return bcrypt.checkpw(password.encode(), user["password"].encode())

def auth_register_user(username, name, email, password):
    col = get_auth_collection()
    if col.find_one({"username": username}):
        return False, "Username already exists."
    if col.find_one({"email": email}):
        return False, "Email already registered."
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    col.insert_one({
        "username":   username,
        "name":       name,
        "email":      email,
        "password":   hashed,
        "role":       "user",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    return True, "Account created successfully!"

def auth_get_all_users():
    return list(get_auth_collection().find({}, {"_id": 0, "password": 0}))

# ── Initialize auth collection at startup ──────────────────────────────────────
get_auth_collection()

# ── Login / Register UI ─────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>☪ Islamic Knowledge Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>مساعد المعرفة الإسلامية</div>", unsafe_allow_html=True)
st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

if not st.session_state.get("authenticated"):
    auth_tab1, auth_tab2 = st.tabs(["Login", "Create Account"])

    with auth_tab1:
        st.markdown("#### Login")
        with st.form("login_form"):
            login_user = st.text_input("Username")
            login_pass = st.text_input("Password", type="password")
            submitted  = st.form_submit_button("Login")
        if submitted:
            login_user = login_user.strip()
            login_pass = login_pass.strip()
            if not login_user or not login_pass:
                st.error("Please enter username and password.")
            elif auth_verify_password(login_user, login_pass):
                user_doc = auth_find_user(login_user)
                st.session_state["authenticated"] = True
                st.session_state["username"]       = login_user
                st.session_state["name"]           = user_doc["name"]
                st.session_state["role"]           = user_doc.get("role", "user")
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with auth_tab2:
        st.markdown("#### Create a new account")
        with st.form("register_form"):
            reg_name     = st.text_input("Full name")
            reg_email    = st.text_input("Email")
            reg_username = st.text_input("Username")
            reg_pass     = st.text_input("Password",         type="password")
            reg_pass2    = st.text_input("Confirm password", type="password")
            reg_submitted = st.form_submit_button("Register")
        if reg_submitted:
            if not all([reg_name, reg_email, reg_username, reg_pass, reg_pass2]):
                st.error("Please fill in all fields.")
            elif reg_pass != reg_pass2:
                st.error("Passwords do not match.")
            elif len(reg_pass) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                ok, msg = auth_register_user(reg_username, reg_name, reg_email, reg_pass)
                if ok:
                    st.success(msg + " Please go to the Login tab.")
                else:
                    st.error(msg)
    st.stop()

# ── Logged in ───────────────────────────────────────────────────────────────────
username = st.session_state.get("username", "")
name     = st.session_state.get("name", "")

if "logged_this_session" not in st.session_state:
    log_login(username, name)
    st.session_state.logged_this_session = True

# ── Config ─────────────────────────────────────────────────────────────────────
DB_NAME                    = "gp"
COLLECTION_NAME            = "ahadeeth"
VECTOR_INDEX_NAME          = "ahadeeth_index"
VECTOR_PATH                = "embedding"
QURAN_COLLECTION_NAME      = "ayat"
QURAN_VECTOR_INDEX_NAME    = "vector_index"
QURAN_VECTOR_PATH          = "embedding"
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY")
EMBED_MODEL           = "text-embedding-3-small"
GEN_MODEL             = "gpt-4.1-mini"
TOP_K                 = 5

SCORE_THRESHOLDS = {
    "bleu":    0.10,
    "rouge":   0.25,
    "bert_f1": 0.75
}

# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    mongo_client     = MongoClient(MONGO_URI)
    collection       = mongo_client[DB_NAME][COLLECTION_NAME]
    quran_collection = mongo_client[DB_NAME][QURAN_COLLECTION_NAME]
    ai               = OpenAI(api_key=OPENAI_API_KEY)
    llm              = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=OPENAI_API_KEY)

    muslim  = json.load(open("Sahih Muslim.json",     "r", encoding="utf-8"))
    bukhari = json.load(open("Sahih al-Bukhari.json", "r", encoding="utf-8"))

    df1 = pd.DataFrame(muslim)
    df2 = pd.DataFrame(bukhari)
    for df in [df1, df2]:
        df["Chapter_Title_Arabic"] = (
            df["Chapter_Title_Arabic"].astype(str).astype(object)
            .str.replace(r'\d+', '', regex=True)
            .str.replace(r'[^\w\s\u0600-\u06FF]', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
    Book_muslim  = df1["Chapter_Title_Arabic"].unique()
    Book_Bukhari = df2["Chapter_Title_Arabic"].unique()

    return collection, quran_collection, ai, llm, Book_muslim, Book_Bukhari

collection, quran_collection, ai, llm, Book_muslim, Book_Bukhari = load_resources()

# ── Helper functions ───────────────────────────────────────────────────────────
def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def embed_query(text):
    return ai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def translate_ar_to_en(query):
    r = ai.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translator for an Islamic search system. "
                    "Translate the Arabic search query to a short, clear English search query "
                    "suitable for retrieving Quran verses or Hadith. "
                    "Return ONLY the translated query. No explanation, no punctuation around it."
                )
            },
            {"role": "user", "content": query}
        ],
        max_tokens=60, temperature=0
    )
    return r.choices[0].message.content.strip()

def translate_to_arabic(text):
    r = ai.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Translate the following to Arabic. Return ONLY the Arabic translation, nothing else."
            },
            {"role": "user", "content": text}
        ],
        max_tokens=60, temperature=0
    )
    return r.choices[0].message.content.strip()

def embed_dual_query(query):
    pairs = [(query, embed_query(query))]
    if is_arabic(query):
        eng = translate_ar_to_en(query)
        pairs.append((eng, embed_query(eng)))
    return pairs

def merge_results(results_list, limit=10):
    seen = {}
    for results in results_list:
        for doc in results:
            key = doc.get("reference") or str(doc)
            if key not in seen or doc.get("score", 0) > seen[key].get("score", 0):
                seen[key] = doc
    return sorted(seen.values(), key=lambda d: d.get("score", 0), reverse=True)[:limit]

def is_relevant_hadeeth(question, hadiths):
    if not hadiths: return []
    lines = "\n\n".join(f"[{i}] {d.get('text','')[:300]}" for i, d in enumerate(hadiths))
    prompt = (
        "You are a relevance filter for an Islamic hadith search system.\n\n"
        f"User question: {question}\n\n"
        "Below are candidate hadiths (by index). Reply with ONLY a JSON array of the indices\n"
        "that are genuinely relevant to the question topic — even if they use different words.\n"
        "If NONE are relevant, return an empty array: []\n\n"
        f"Hadiths:\n{lines}\n\n"
        "Reply with ONLY a JSON array like [0, 2, 4] or []. No explanation."
    )
    r = ai.chat.completions.create(model=GEN_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=50, temperature=0)
    try:
        raw = r.choices[0].message.content.strip()
        indices = json.loads(re.search(r"\[.*?\]", raw).group())
        return [hadiths[i] for i in indices if i < len(hadiths)]
    except: return []

def is_relevant_quran(question, ayat):
    if not ayat: return []
    lines = "\n\n".join(f"[{i}] {d.get('text','')[:300]}" for i, d in enumerate(ayat))
    prompt = (
        "You are a relevance filter for a Quranic ayah search system.\n\n"
        f"User question: {question}\n\n"
        "Below is a list of candidate ayat with indices. Return ONLY a JSON array\n"
        "of the indices that are genuinely relevant to the question.\n\n"
        "Relevant ayat may use different words but express the same concept.\n"
        "If NONE are relevant, return an empty array: []\n\n"
        f"Ayat:\n{lines}\n\n"
        "Reply with ONLY a JSON array like [0, 2, 4] or []. No explanation."
    )
    r = ai.chat.completions.create(model=GEN_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=50, temperature=0)
    try:
        raw = r.choices[0].message.content.strip()
        indices = json.loads(re.search(r"\[.*?\]", raw).group())
        return [ayat[i] for i in indices if i < len(ayat)]
    except: return []

def retrieve_context(query, question, k=TOP_K):
    all_results = []
    for _, emb in embed_dual_query(query):
        results = list(collection.aggregate([
            {"$vectorSearch": {"index": VECTOR_INDEX_NAME, "path": VECTOR_PATH, "queryVector": emb, "numCandidates": 2000, "limit": 20}},
            {"$project": {"_id":0, "arabic":1, "text":1, "reference":1, "book":1, "score":{"$meta":"vectorSearchScore"}}}
        ]))
        all_results.append(is_relevant_hadeeth(question, results))
    return merge_results(all_results, limit=k)

def retrieve_quran_context(query, question, k=TOP_K):
    all_results = []
    for _, emb in embed_dual_query(query):
        results = list(quran_collection.aggregate([
            {"$vectorSearch": {"index": QURAN_VECTOR_INDEX_NAME, "path": QURAN_VECTOR_PATH, "queryVector": emb, "numCandidates": 2000, "limit": 20}},
            {"$project": {"_id":0, "surah_id":1, "surah_name":1, "ayah_id":1, "text":1, "score":{"$meta":"vectorSearchScore"}}}
        ]))
        formatted = [{"reference": f"{r['surah_id']}:{r['ayah_id']}", "surah_name": r["surah_name"], "text": r["text"], "score": r["score"]} for r in results]
        all_results.append(is_relevant_quran(question, formatted))
    return merge_results(all_results, limit=k)

def build_context(docs):
    return "\n\n".join(
        f"Hadith {i}\nArabic:\n{d.get('arabic','')}\n\nEnglish:\n{d.get('text','')}\n\nReference:\n{d.get('reference','')}"
        for i, d in enumerate(docs, 1)
    )

def build_quran_context(docs):
    return "\n\n".join(
        f"Ayah {i}\nReference:\n{d.get('reference','')} ({d.get('surah_name','')})\n\nText:\n{d.get('text','')}"
        for i, d in enumerate(docs, 1)
    )

# ── Scoring functions ──────────────────────────────────────────────────────────
def compute_scores(reference: str, generated: str) -> dict:
    ref_tokens = reference.split()
    hyp_tokens = generated.split()
    smoothie   = SmoothingFunction().method4
    bleu       = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

    scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge   = scorer.score(reference, generated)["rougeL"].fmeasure

    P, R, F1 = bert_score_fn([generated], [reference], lang="en", verbose=False)
    bert_f1 = round(F1.mean().item(), 4)

    return {
        "bleu":    round(bleu, 4),
        "rouge":   round(rouge, 4),
        "bert_f1": bert_f1
    }

def is_khutbah_valid(scores: dict) -> bool:
    return (
        scores["bleu"]    >= SCORE_THRESHOLDS["bleu"]    and
        scores["rouge"]   >= SCORE_THRESHOLDS["rouge"]   and
        scores["bert_f1"] >= SCORE_THRESHOLDS["bert_f1"]
    )

def pick_best_attempt(attempts: list) -> dict:
    return max(attempts, key=lambda a: a["scores"]["bert_f1"])

# ── Alukah functions ───────────────────────────────────────────────────────────
ALUKAH_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "ar,en;q=0.9",
    "Accept":          "text/html,application/xhtml+xml"
}

def scrape_alukah_khutbah(url: str) -> str:
    try:
        session = requests.Session()
        session.headers.update(ALUKAH_HEADERS)
        response = session.get(url, timeout=10)
        soup     = BeautifulSoup(response.text, "html.parser")
        body = (
            soup.find("div", class_="article-content") or
            soup.find("div", class_="post-content")    or
            soup.find("div", class_="content")         or
            soup.find("article")
        )
        return body.get_text(separator="\n", strip=True) if body else ""
    except Exception:
        return ""
def search_alukah_khutbah(topic: str) -> dict:
    try:
        url_response = ai.chat.completions.create(
            model="gpt-4o-search-preview",
            max_completion_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Search alukah.net for a Friday khutbah about the given topic. "
                        "Return ONLY the direct URL to the article. "
                        "Nothing else. Just the URL."
                    )
                },
                {
                    "role": "user",
                    "content": f"Find a khutbah on alukah.net about: {topic}"
                }
            ]
        )
        url = url_response.choices[0].message.content.strip()

        if "alukah.net" not in url:
            return {"title": None, "url": None, "text": None}

        text_response = ai.chat.completions.create(
            model="gpt-4o-search-preview",
            max_completion_tokens=4000,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Islamic research assistant. "
                        "Retrieve and return the FULL Arabic khutbah text from the given URL. "
                        "Return ONLY the khutbah text in Arabic. "
                        "No explanation, no translation, no JSON. Just the raw Arabic text."
                    )
                },
                {
                    "role": "user",
                    "content": f"Get the full khutbah text from: {url}"
                }
            ]
        )

        text = text_response.choices[0].message.content.strip()

        if "الخطبة الأولى" in text:
            text = text.split("الخطبة الأولى")[1].strip()
        if "الخطبة الثانية" in text:
            text = text.split("الخطبة الثانية")[0].strip()

        if len(text) < 100:
            return {"title": None, "url": None, "text": None}

        return {"title": topic, "url": url, "text": text}

    except Exception:
        return {"title": None, "url": None, "text": None}

def fetch_reference_khutbah(llm_topic: str) -> str:
    result = search_alukah_khutbah(llm_topic)
    if result["text"]:
        return result["text"]
    return ""

# ── State ──────────────────────────────────────────────────────────────────────
class State(TypedDict):
    Question:            str
    User_Request_type:   str
    Retrieved_Hadith:    str
    Retrieved_Quran:     str
    Topic:               str
    generated_khutbah:   str
    generated_answer:    str
    User_Hadith:         str
    verification_result: str
    round:               int
    next:                str
    reference_khutbah:   str
    khutbah_scores:      dict
    validation_passed:   bool
    regen_count:         int
    all_attempts:        list
    best_khutbah:        str
    topic_not_found:     bool
    llm_topic:           str
    # New: Hadith from khotbah
    khotbah_input:       str
    source_hadiths:      list
    source_quran:        list

# ── Agents ─────────────────────────────────────────────────────────────────────
def Planner(State, persona, name):
    # If user manually selected a mode from sidebar, skip LLM classification
    if State.get("User_Request_type", "").strip():
        return State
    question = State['Question']
    system_prompt = f"""
  You are a {persona}.

  Task:
  Select exactly one of four request types based on the user's question.

  Hard rules:
  - You MUST output exactly one of: Hadith verification OR Khotbah generation OR Hadith Return OR Hadith from khotbah.
  - Do NOT rephrase or shorten.
  - Do NOT explain.
  - If nothing fits clearly, output: Couldn't detect.

  Examples:
  User: "I want a khutbah about wudu"
  Output: Khotbah generation

  User: "Talk about Ramadan virtues"
  Output: Khotbah generation

  User: "رسول الله صلى الله عليه وسلم قال: (من حمل علينا السلاح، فليس منا، ومن غشنا، فليس منا)"
  Output: Hadith verification

  User: "hadeeth about niyya"
  Output: Hadith Return

  User: "الحمد لله رب العالمين... [long khotbah text] ... what hadiths were used in this khotbah?"
  Output: Hadith from khotbah

  User: "ما هي الأحاديث المستخدمة في هذه الخطبة: [khotbah text]"
  Output: Hadith from khotbah
  """
    r = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    State['User_Request_type'] = r.content.strip()
    return State

def Topic_extractor(State, persona, name):
    question = State["Question"]

    # 1. Try to find a structural chapter heading (useful for Khutbah formatting)
    system_prompt_db = f"""
    You are a {persona}.
    Task: Select relevant chapter(s) heading from the CHAPTER LIST.
    CHAPTER LIST:
    BUKHARI: {Book_Bukhari}
    MUSLIM: {Book_muslim}
    Output ONLY the heading or 'Couldn't detect'.
    """
    response = llm.invoke([SystemMessage(content=system_prompt_db), HumanMessage(content=question)])
    content = response.content.strip()

    # 2. Extract a general semantic topic for web fallbacks (Alukah)
    system_prompt_llm = """
    Extract the core Islamic topic as a short Arabic phrase (2-5 words).
    Output ONLY the phrase or 'EMPTY'.
    """
    response2 = llm.invoke([SystemMessage(content=system_prompt_llm), HumanMessage(content=question)])
    llm_topic = response2.content.strip()
    State["llm_topic"] = "" if llm_topic == "EMPTY" else llm_topic

    # Logic Fix: Decouple 'Topic' from retrieval blocking
    if "Couldn't detect" not in content and content != "":
        State["Topic"] = content
    else:
        # If no structural chapter is found, use the semantic topic as the 'Topic'
        State["Topic"] = State["llm_topic"]

    # Only set topic_not_found to True if BOTH the chapter list match 
    # AND the semantic extraction failed.
    State["topic_not_found"] = (State["Topic"] == "" and State["llm_topic"] == "")
    
    return State

def retrieve_agent(State, persona, name):
    # Use the Question directly for vector search to ensure semantic relevance
    # even if a specific chapter heading (Topic) wasn't detected.
    raw = retrieve_context(State["Question"], State["Question"], k=5)
    State["Retrieved_Hadith"] = build_context(raw) if raw else "NO_RELEVANT_HADITH_FOUND"
    return State

def Quran_retrieve_agent(State, persona, name):
    raw = retrieve_quran_context(State["Topic"], State["Question"], k=5)
    State["Retrieved_Quran"] = build_quran_context(raw) if raw else "NO_RELEVANT_QURAN_FOUND"
    return State

def Hadith_Return(State, persona, name):
    raw = State.get("Retrieved_Hadith", "")
    State["generated_answer"] = raw.strip() if raw.strip() and raw != "NO_RELEVANT_HADITH_FOUND" else "No supporting hadith found."
    return State

def Khotbah_generation(State, persona, name):
    question = State['Question']
    hadith   = State['Retrieved_Hadith']
    quran    = State['Retrieved_Quran']

    system_prompt = f"""
  You are {persona}.

  Task:
  Generate a complete Friday Khutbah in ARABIC based ONLY on the provided HADITH_CONTEXT and QURAN_CONTEXT below.

  Strict Requirements (VERY IMPORTANT):
  1) The entire khutbah MUST be written in Arabic, regardless of the user's language.
  2) You MUST use only the hadiths and ayat provided in the contexts.
  3) Do NOT introduce any external hadith or Islamic text.
  4) When referencing a hadith inside the khutbah, place a citation number in this format: [1], [2], etc.
  5) When referencing a Quran inside the khutbah, place a citation number in this format: [1], [2], etc.
  6) When referencing an ayah, use this format: (Surah:Ayah).
  7) Do NOT write the source details inside the khutbah body — only use citation numbers.
  8) Khotbah length must be between 1100-1500 words.
  9) After finishing the khutbah, add a section titled exactly: المراجع:
  10) Under 'المراجع:', list references using the same numbers used in the khutbah.
  11) Each reference must include:
      - Source (e.g., Sahih al-Bukhari / Sahih Muslim / Quran)
      - Full Arabic text (exactly as retrieved)
  12) If the provided contexts are insufficient, respond exactly with:
      لا توجد أحاديث أو آيات كافية في السياق المسترجع.
  13) Do NOT repeat any sentence or supplication under any circumstance.
  14) The khutbah must focus on explanation, not supplication.
  15) Supplications must appear ONLY at the end of the khutbah.
  16) Do NOT generate long repeated dua sequences.
  17) ALWAYS include at least 1 ayah related to the topic of the khutbah

  Structure:
  1) Opening praise (Hamd and Salawat) - short
  2) Main topic explanation (majority of the khutbah)
  3) Practical advice and reminders
  4) Conclusion
  5) Final supplications (دعاء)

  Rules for supplications (VERY STRICT):
  - Include MINIMUM 5 and MAXIMUM 10 supplications only
  - Each supplication must be written once only
  - Do NOT repeat any supplication
  - Do NOT loop phrases like 'اللهم اجعلنا'
  - Each supplication must be different and meaningful
  - Total supplication section must not exceed 10 lines

  HADITH_CONTEXT:
  {hadith}

  QURAN_CONTEXT:
  {quran}
  """
    r = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    State["generated_khutbah"] = r.content.strip()

    if not State.get("reference_khutbah", "").strip():
        topic     = State.get("llm_topic") or State.get("Topic") or question
        reference = fetch_reference_khutbah(topic)
        State["reference_khutbah"] = reference

    return State

def Khutbah_validator(State, persona, name):
    generated = State.get("generated_khutbah", "").strip()
    reference = State.get("reference_khutbah", "").strip()
    attempts  = State.get("all_attempts", [])

    if not generated:
        State["validation_passed"] = False
        return State

    if not reference:
        State["validation_passed"] = True
        State["best_khutbah"]      = generated
        return State

    scores = compute_scores(reference=reference, generated=generated)
    State["khutbah_scores"] = scores

    attempt_num = State.get("regen_count", 0) + 1
    attempts.append({"khutbah": generated, "scores": scores, "attempt": attempt_num})
    State["all_attempts"]      = attempts
    State["validation_passed"] = is_khutbah_valid(scores)
    return State

def Alukah_fallback(State, persona, name):
    question  = State["Question"]
    llm_topic = State.get("llm_topic", "").strip()

    search_term = llm_topic or (question if is_arabic(question) else translate_to_arabic(question))
    result = search_alukah_khutbah(search_term)

    if result["text"]:
        State["generated_answer"] = (
            f"⚠️ تنبيه: هذه الخطبة لم يتم توليدها بواسطة النظام.\n"
            f"الموضوع المطلوب غير موجود في قاعدة البيانات المحلية،\n"
            f"لذلك تم جلب هذه الخطبة من موقع الألوكة الشرعية.\n\n"
            f"{'='*50}\n\n"
            f"{result['text']}\n\n"
            f"{'='*50}\n\n"
            f"📌 المرجع:\n"
            f"العنوان: {result['title']}\n"
            f"الرابط: {result['url']}"
        )
    else:
        State["generated_answer"] = (
            f"عذراً، لم يتم العثور على موضوع مطابق في قاعدة البيانات أو موقع الألوكة.\n"
            f"الموضوع المطلوب: {search_term}"
        )
    return State

def Extract_user_hadith(State, persona, name):
    system_prompt = f"""
   You are {persona}.

   Task:
   Extract the hadith text from the user's message.

   Rules:
   - Output ONLY the extracted hadith text (no labels, no quotes, no parentheses, no explanation).
   - Remove any surrounding wrappers like: (), "", '', «».
   - Remove leading phrases like:
     'قال رسول الله صلى الله عليه وسلم', 'عن', 'رضي الله عنه', 'is this hadith right', 'حديث', etc.
   - If there are multiple candidate snippets, output the LONGEST snippet that looks like the hadith statement.
   - If no hadith text is present, output exactly: EMPTY

   Examples:
   User: is this hadith right? رسول الله صلى الله عليه وسلم قال: (من حمل علينا السلاح، فليس منا، ومن غشنا، فليس منا)
   Output: من حمل علينا السلاح، فليس منا، ومن غشنا، فليس منا

   User: حديث (إنما الأعمال بالنيات) صحيح؟
   Output: إنما الأعمال بالنيات

   User: I have a question about prayer
   Output: EMPTY
   """.strip()
    r = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=State['Question'])])
    State["User_Hadith"] = r.content.strip()
    return State

def Hadith_verification(State, persona, name):
    user_hadith    = State["User_Hadith"]
    hadith_context = State["Retrieved_Hadith"]

    if not hadith_context or hadith_context == "NO_RELEVANT_HADITH_FOUND":
        State["verification_result"] = "NO Could be in another Book"
        return State

    system_prompt = f"""
    You are {persona}.

    Task:
    Verify whether the USER_HADITH matches EXACTLY (word-for-word) any hadith in HADITH_CONTEXT.

    Strict matching rules:
    - Matching is EXACT Arabic text match (word-for-word).
    - If ANY word is missing, added, changed, or misspelled => NOT A MATCH.
    - Ignore ONLY: extra spaces, new lines.
    - Do NOT use semantic similarity.

    Output rules:
    CASE 1 (match found):
    YES
    MATCH:
    <print the full matched hadith object exactly as stored in HADITH_CONTEXT>

    CASE 2 (no match but close):
    NO
    DID_YOU_MEAN:
    <print ALL hadith objects from HADITH_CONTEXT exactly as stored>

    CASE 3 (no match and not close):
    NO
    Could be in another Book

    USER_HADITH:
    {user_hadith}

    HADITH_CONTEXT:
    {hadith_context}
    """
    r = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=State["User_Hadith"])])
    State["verification_result"] = r.content.strip()
    return State

# ── Probability scoring helpers ────────────────────────────────────────────────
W_VEC, W_LLM, W_EXACT = 0.40, 0.45, 0.15

def check_exact_match(hadith_text: str, khotbah_text: str, min_words: int = 5) -> int:
    """Returns 100 if a 5+ word phrase matches verbatim, 50 if 3-4 words, 0 if none."""
    words = hadith_text.split()
    for n in [5, 4, 3]:
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            if phrase in khotbah_text:
                return 100 if n >= 5 else 50
    return 0

def compute_match_probability(vec_score: float, llm_score: int, exact_score: int) -> int:
    """Combine three signals into a final 0-100 probability."""
    score = (vec_score * 100 * W_VEC) + (llm_score * W_LLM) + (exact_score * W_EXACT)
    return round(score)

def score_hadith_match_llm(khotbah_text: str, hadith_arabic: str, hadith_english: str) -> int:
    """Ask GPT-4 to judge 0-100 how likely this hadith was used as source in the khotbah."""
    prompt = f"""You are an Islamic scholarly assistant.

Given a KHOTBAH (Friday sermon) and a HADITH, score from 0 to 100 how likely it is that
this specific hadith was a SOURCE used by the imam when writing this khotbah.

Scoring guide:
- 90-100: Hadith topic is central to the khotbah AND wording is very close or quoted
- 70-89 : Hadith topic aligns clearly with the khotbah main theme
- 50-69 : Hadith is loosely related to the khotbah topic
- 20-49 : Weak connection, could be coincidental
- 0-19  : No meaningful connection

KHOTBAH (first 800 chars):
{khotbah_text[:800]}

HADITH ARABIC:
{hadith_arabic[:300]}

HADITH ENGLISH:
{hadith_english[:300]}

Reply with ONLY a single integer between 0 and 100. No explanation."""
    try:
        r = ai.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5, temperature=0
        )
        return min(100, max(0, int(r.choices[0].message.content.strip())))
    except:
        return 50

def score_quran_match_llm(khotbah_text: str, ayah_text: str, reference: str) -> int:
    """Ask GPT-4 to judge 0-100 how likely this ayah was used as source in the khotbah."""
    prompt = f"""You are an Islamic scholarly assistant.

Given a KHOTBAH (Friday sermon) and a QURAN AYAH, score from 0 to 100 how likely it is that
this specific ayah was referenced or quoted by the imam in this khotbah.

Scoring guide:
- 90-100: Ayah is directly quoted or its reference appears in the khotbah
- 70-89 : Ayah theme is central to the khotbah topic
- 50-69 : Ayah is loosely relevant to the khotbah
- 20-49 : Weak connection
- 0-19  : No meaningful connection

KHOTBAH (first 800 chars):
{khotbah_text[:800]}

AYAH ({reference}):
{ayah_text[:300]}

Reply with ONLY a single integer between 0 and 100. No explanation."""
    try:
        r = ai.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5, temperature=0
        )
        return min(100, max(0, int(r.choices[0].message.content.strip())))
    except:
        return 50

# ── New agents: Hadith from khotbah subgraph ───────────────────────────────────
def Extract_khotbah_input(state: State, persona: str, name: str) -> State:
    """Extracts and cleans the khotbah text from the user question."""
    question = state["Question"]
    prompt = """You are an assistant that extracts khotbah text from user messages.

The user has pasted a khotbah (Friday sermon) and wants to find its hadith/quran sources.
Extract ONLY the khotbah body text — remove any surrounding question like
"what hadiths are in this" or "ما هي الاحاديث في".

If the entire message IS the khotbah, return it as-is.
Return ONLY the khotbah text, nothing else."""
    r = ai.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": question}
        ],
        max_tokens=2000, temperature=0
    )
    state["khotbah_input"] = r.choices[0].message.content.strip()
    return state

def Retrieve_hadiths_for_khotbah(state: State, persona: str, name: str) -> State:
    """Retrieves candidate hadiths from MongoDB, scores each with the 3-signal algorithm."""
    khotbah = state.get("khotbah_input", "").strip()
    if not khotbah:
        state["source_hadiths"] = []
        return state

    # Vector search using khotbah text as query
    emb = embed_query(khotbah[:500])  # embed first 500 chars for efficiency
    raw_results = list(collection.aggregate([
        {"": {
            "index": VECTOR_INDEX_NAME, "path": VECTOR_PATH,
            "queryVector": emb, "numCandidates": 2000, "limit": 15
        }},
        {"": {"_id":0, "arabic":1, "text":1, "reference":1, "book":1,
                       "score":{"":"vectorSearchScore"}}}
    ]))

    scored = []
    for doc in raw_results:
        vec_score   = doc.get("score", 0.0)
        arabic      = doc.get("arabic", "")
        english     = doc.get("text", "")
        reference   = doc.get("reference", "")

        llm_score   = score_hadith_match_llm(khotbah, arabic, english)
        exact_score = check_exact_match(arabic, khotbah)
        final_pct   = compute_match_probability(vec_score, llm_score, exact_score)

        scored.append({
            "arabic":      arabic,
            "text":        english,
            "reference":   reference,
            "vec_score":   round(vec_score, 4),
            "llm_score":   llm_score,
            "exact_score": exact_score,
            "final_pct":   final_pct
        })

    # Sort by final probability, keep top 8, filter out weak matches
    scored.sort(key=lambda x: x["final_pct"], reverse=True)
    state["source_hadiths"] = [s for s in scored[:8] if s["final_pct"] >= 20]
    return state

def Retrieve_quran_for_khotbah(state: State, persona: str, name: str) -> State:
    """Retrieves candidate ayat from MongoDB, scores each with the 3-signal algorithm."""
    khotbah = state.get("khotbah_input", "").strip()
    if not khotbah:
        state["source_quran"] = []
        return state

    emb = embed_query(khotbah[:500])
    raw_results = list(quran_collection.aggregate([
        {"": {
            "index": QURAN_VECTOR_INDEX_NAME, "path": QURAN_VECTOR_PATH,
            "queryVector": emb, "numCandidates": 2000, "limit": 15
        }},
        {"": {"_id":0, "surah_id":1, "surah_name":1, "ayah_id":1,
                       "text":1, "score":{"":"vectorSearchScore"}}}
    ]))

    scored = []
    for doc in raw_results:
        vec_score  = doc.get("score", 0.0)
        ayah_text  = doc.get("text", "")
        reference  = f"{doc.get('surah_id','')}:{doc.get('ayah_id','')}"
        surah_name = doc.get("surah_name", "")

        llm_score   = score_quran_match_llm(khotbah, ayah_text, reference)
        exact_score = check_exact_match(ayah_text, khotbah)
        final_pct   = compute_match_probability(vec_score, llm_score, exact_score)

        scored.append({
            "reference":   reference,
            "surah_name":  surah_name,
            "text":        ayah_text,
            "vec_score":   round(vec_score, 4),
            "llm_score":   llm_score,
            "exact_score": exact_score,
            "final_pct":   final_pct
        })

    scored.sort(key=lambda x: x["final_pct"], reverse=True)
    state["source_quran"] = [s for s in scored[:8] if s["final_pct"] >= 20]
    return state

def build_khotbah_sources_subgraph():
    """
    Flow: node_extract_khotbah → node_hadith_retrieval → node_quran_retrieval → END
    """
    builder = StateGraph(State)
    builder.add_node("node_extract_khotbah",
        partial(Extract_khotbah_input,          persona="Khotbah Extractor",  name="Khotbah Extractor"))
    builder.add_node("node_source_hadith",
        partial(Retrieve_hadiths_for_khotbah,   persona="Hadith Source Finder", name="Hadith Source Finder"))
    builder.add_node("node_source_quran",
        partial(Retrieve_quran_for_khotbah,     persona="Quran Source Finder",  name="Quran Source Finder"))
    builder.set_entry_point("node_extract_khotbah")
    builder.add_edge("node_extract_khotbah", "node_source_hadith")
    builder.add_edge("node_source_hadith",   "node_source_quran")
    builder.add_edge("node_source_quran",     END)
    return builder.compile()

def run_supervisor_agent(state: State, persona: str, name: str):
    current_round = state.get("round", 0) + 1

    if current_round >= 50:
        attempts = state.get("all_attempts", [])
        if attempts:
            best = pick_best_attempt(attempts)
            return {"next": "FINISH", "round": current_round, "best_khutbah": best["khutbah"]}
        return {"next": "FINISH", "round": current_round}

    user_request_type   = (state.get("User_Request_type")   or "").strip().lower()
    topic               = (state.get("Topic")               or "").strip()
    retrieved_hadith    = (state.get("Retrieved_Hadith")    or "").strip()
    retrieved_quran     = (state.get("Retrieved_Quran")     or "").strip()
    generated_khutbah   = (state.get("generated_khutbah")   or "").strip()
    generated_answer    = (state.get("generated_answer")    or "").strip()
    user_hadith         = (state.get("User_Hadith")         or "").strip()
    verification_result = (state.get("verification_result") or "").strip()
    validation_passed   = state.get("validation_passed", None)
    regen_count         = state.get("regen_count", 0)
    all_attempts        = state.get("all_attempts", [])
    topic_not_found     = state.get("topic_not_found", False)

    has_request_type      = len(user_request_type) > 0
    has_topic             = len(topic) > 0
    has_retrieved_hadith  = len(retrieved_hadith) > 0
    has_retrieved_quran   = len(retrieved_quran) > 0
    has_generated_khutbah = len(generated_khutbah) > 0
    has_generated_answer  = len(generated_answer) > 0
    has_user_hadith       = len(user_hadith) > 0
    has_verification      = len(verification_result) > 0

    if not has_request_type:
        nxt = "planning_subgraph"

    elif user_request_type == "hadith return":
        if not has_topic or not has_retrieved_hadith:
            nxt = "planning_subgraph"
        elif not has_generated_answer:
            nxt = "hadith_return_subgraph"
        else:
            nxt = "FINISH"

    elif user_request_type == "khotbah generation":
        if not has_topic and not topic_not_found:
            nxt = "planning_subgraph"
        elif topic_not_found and not has_generated_answer:
            nxt = "generation_subgraph"
        elif topic_not_found and has_generated_answer:
            nxt = "FINISH"
        elif not has_retrieved_hadith or not has_retrieved_quran:
            nxt = "planning_subgraph"
        elif regen_count >= 3:
            if all_attempts:
                best = pick_best_attempt(all_attempts)
                return {"next": "FINISH", "round": current_round, "best_khutbah": best["khutbah"]}
            return {"next": "FINISH", "round": current_round}
        elif not has_generated_khutbah:
            nxt = "generation_subgraph"
        elif validation_passed is None:
            nxt = "generation_subgraph"
        elif validation_passed is True:
            return {"next": "FINISH", "round": current_round, "best_khutbah": generated_khutbah}
        elif regen_count < 3:
            new_regen = regen_count + 1
            return {
                "next":              "generation_subgraph",
                "round":             current_round,
                "generated_khutbah": "",
                "validation_passed": None,
                "regen_count":       new_regen,
                "reference_khutbah": state.get("reference_khutbah", ""),
                "all_attempts":      all_attempts
            }
        else:
            nxt = "FINISH"

    elif user_request_type == "hadith verification":
        if not has_topic or not has_retrieved_hadith:
            nxt = "planning_subgraph"
        elif not has_user_hadith or not has_verification:
            nxt = "verification_subgraph"
        else:
            nxt = "FINISH"

    elif user_request_type == "hadith from khotbah":
        source_hadiths = state.get("source_hadiths", None)
        source_quran   = state.get("source_quran",   None)
        if source_hadiths is None or source_quran is None:
            nxt = "khotbah_sources_subgraph"
        else:
            nxt = "FINISH"

    else:
        nxt = "planning_subgraph"

    return {"next": nxt, "round": current_round}

# ── Build Graph ────────────────────────────────────────────────────────────────
# ── Subgraph builders ─────────────────────────────────────────────────────────
def build_planning_subgraph():
    """
    Flow: node_planner → node_topic_extractor → node_hadith_retrieval → node_quran_retrieval → END
    """
    builder = StateGraph(State)
    builder.add_node("node_planner",          partial(Planner,              persona="Planner",          name="Planner"))
    builder.add_node("node_topic_extractor",  partial(Topic_extractor,      persona="Topic Extractor",  name="Topic Extractor"))
    builder.add_node("node_hadith_retrieval", partial(retrieve_agent,       persona="Hadith Retrieval", name="Hadith Retrieval"))
    builder.add_node("node_quran_retrieval",  partial(Quran_retrieve_agent, persona="Quran Retrieval",  name="Quran Retrieval"))
    builder.set_entry_point("node_planner")
    builder.add_edge("node_planner",          "node_topic_extractor")
    builder.add_edge("node_topic_extractor",  "node_hadith_retrieval")
    builder.add_edge("node_hadith_retrieval", "node_quran_retrieval")
    builder.add_edge("node_quran_retrieval",   END)
    return builder.compile()

def build_generation_subgraph():
    """
    Flow: START → route_entry → node_khotbah → node_validator → END
                             ↘→ node_alukah_fallback          → END
    """
    node_khotbah         = partial(Khotbah_generation, persona="Khotbah Generation", name="Khotbah Generation")
    node_validator       = partial(Khutbah_validator,  persona="Quality Checker",    name="Khutbah Validator")
    node_alukah_fallback = partial(Alukah_fallback,    persona="Web Researcher",     name="Alukah Fallback")

    def route_generation_entry(state: State):
        return "node_alukah_fallback" if state.get("topic_not_found") else "node_khotbah"

    builder = StateGraph(State)
    builder.add_node("node_khotbah",         node_khotbah)
    builder.add_node("node_validator",       node_validator)
    builder.add_node("node_alukah_fallback", node_alukah_fallback)
    builder.set_conditional_entry_point(
        route_generation_entry,
        {"node_khotbah": "node_khotbah", "node_alukah_fallback": "node_alukah_fallback"}
    )
    builder.add_edge("node_khotbah",        "node_validator")
    builder.add_edge("node_validator",       END)
    builder.add_edge("node_alukah_fallback", END)
    return builder.compile()

def build_verification_subgraph():
    """
    Flow: node_extract_hadith → node_hadith_verify → END
    """
    builder = StateGraph(State)
    builder.add_node("node_extract_hadith", partial(Extract_user_hadith, persona="Hadith Extraction",   name="Hadith Extractor"))
    builder.add_node("node_hadith_verify",  partial(Hadith_verification, persona="Hadith Verification", name="Hadith Verification"))
    builder.set_entry_point("node_extract_hadith")
    builder.add_edge("node_extract_hadith", "node_hadith_verify")
    builder.add_edge("node_hadith_verify",   END)
    return builder.compile()

def build_hadith_return_subgraph():
    """
    Flow: node_hadith_return → END
    """
    builder = StateGraph(State)
    builder.add_node("node_hadith_return", partial(Hadith_Return, persona="Hadith Return", name="Hadith Return"))
    builder.set_entry_point("node_hadith_return")
    builder.add_edge("node_hadith_return", END)
    return builder.compile()

@st.cache_resource
def build_graph():
    """
    Parent graph:
    START → node_supervisor ──┬→ planning_subgraph     → node_supervisor
                               ├→ generation_subgraph   → node_supervisor
                               ├→ verification_subgraph → node_supervisor
                               ├→ hadith_return_subgraph → node_supervisor
                               └→ END
    """
    node_supervisor = partial(run_supervisor_agent, persona="Project Manager", name="Supervisor")

    builder = StateGraph(State)
    builder.add_node("node_supervisor",           node_supervisor)
    builder.add_node("planning_subgraph",         build_planning_subgraph())
    builder.add_node("generation_subgraph",       build_generation_subgraph())
    builder.add_node("verification_subgraph",     build_verification_subgraph())
    builder.add_node("hadith_return_subgraph",    build_hadith_return_subgraph())
    builder.add_node("khotbah_sources_subgraph",  build_khotbah_sources_subgraph())

    builder.set_entry_point("node_supervisor")

    def route(state: State):
        return END if state.get("next") == "FINISH" else state.get("next")

    builder.add_conditional_edges(
        "node_supervisor",
        route,
        {
            "planning_subgraph":        "planning_subgraph",
            "generation_subgraph":      "generation_subgraph",
            "verification_subgraph":    "verification_subgraph",
            "hadith_return_subgraph":   "hadith_return_subgraph",
            "khotbah_sources_subgraph": "khotbah_sources_subgraph",
            END: END,
        }
    )
    builder.add_edge("planning_subgraph",         "node_supervisor")
    builder.add_edge("generation_subgraph",       "node_supervisor")
    builder.add_edge("verification_subgraph",     "node_supervisor")
    builder.add_edge("hadith_return_subgraph",    "node_supervisor")
    builder.add_edge("khotbah_sources_subgraph",  "node_supervisor")
    return builder.compile()

graph = build_graph()

# ── Theme state ────────────────────────────────────────────────────────────────
if "light_mode" not in st.session_state:
    st.session_state.light_mode = False

if st.session_state.light_mode:
    st.markdown("""
    <style>
    /* ── Full light mode override ── */
    html, body, [class*="css"] {
        background-color: #F5F5F0 !important;
        color: #1A1A1A !important;
    }
    .stApp, .main, .main .block-container {
        background-color: #F5F5F0 !important;
        color: #1A1A1A !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #EAEAE5 !important;
        border-right: 1px solid #D0CFC8 !important;
    }
    [data-testid="stSidebar"] * {
        color: #1A1A1A !important;
    }
    /* Radio buttons */
    .stRadio label, .stRadio div, .stRadio p {
        color: #1A1A1A !important;
    }
    /* Cards */
    .arabic-card {
        background: #EAEAE5 !important;
        border-color: #D0CFC8 !important;
        color: #1A1A1A !important;
    }
    .result-card {
        background: #EAEAE5 !important;
        border-color: #D0CFC8 !important;
        color: #1A1A1A !important;
    }
    .score-card {
        background: #EAEAE5 !important;
        border-color: #D0CFC8 !important;
        color: #1A1A1A !important;
    }
    .warning-card {
        background: #FFF8E1 !important;
        border-color: #C9A84C !important;
        color: #2A1A00 !important;
    }
    /* Text input / textarea */
    .stTextArea textarea, .stTextInput input {
        background-color: #EAEAE5 !important;
        color: #1A1A1A !important;
        border-color: #C0BFB8 !important;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background: #EAEAE5 !important;
        color: #C9A84C !important;
        border-color: #D0CFC8 !important;
    }
    .streamlit-expanderContent {
        background: #EAEAE5 !important;
        border-color: #D0CFC8 !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #EAEAE5 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #666 !important;
    }
    /* General text elements */
    p, span, div, label, small {
        color: #1A1A1A !important;
    }
    .text-muted, [style*="8A8070"] {
        color: #666666 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ── Theme toggle — top right ───────────────────────────────────────────────────
col_spacer, col_toggle = st.columns([11, 1])
with col_toggle:
    label = "☀️" if not st.session_state.light_mode else "🌙"
    if st.button(label, key="theme_btn", help="Toggle light / dark mode"):
        st.session_state.light_mode = not st.session_state.light_mode
        st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ☪ Islamic Knowledge")
    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

    # ── User info + logout ──────────────────────────────────────────────────────
    st.markdown(f"<small style='color:#8A8070'>Logged in as <b>{name}</b></small>", unsafe_allow_html=True)
    if st.button("Logout", key="logout_btn"):
        for k in ["authenticated", "username", "name", "role", "logged_this_session"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

    mode = st.radio(
        "Request Type",
        ["Auto Detect", "Khotbah Generation", "Hadith Return", "Hadith Verification", "Hadith from Khotbah"],
        index=0
    )

    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#8A8070'>Sources: Sahih al-Bukhari · Sahih Muslim · Quran</small>",
        unsafe_allow_html=True
    )

    # ── Admin panel (only visible to admin role) ──────────────────────────────
    if st.session_state.get("role") == "admin":
        st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
        st.markdown("### 🔐 Admin Panel")
        admin_view = st.radio(
            "View",
            ["None", "Login History", "Session History", "All Users"],
            key="admin_view"
        )

        if admin_view == "Login History":
            st.markdown("---")
            df_logins = get_login_logs()
            st.markdown(f"**Total logins:** {len(df_logins)}")
            st.dataframe(df_logins[["username", "name", "timestamp"]], use_container_width=True)

        elif admin_view == "Session History":
            st.markdown("---")
            df_sessions = get_session_logs()
            if df_sessions.empty:
                st.info("No sessions recorded yet.")
            else:
                # Filter by user
                all_users = ["All"] + sorted(df_sessions["username"].unique().tolist())
                filter_user = st.selectbox("Filter by user", all_users)
                if filter_user != "All":
                    df_sessions = df_sessions[df_sessions["username"] == filter_user]

                st.markdown(f"**Total sessions:** {len(df_sessions)}")
                st.dataframe(
                    df_sessions[["id", "username", "question", "request_type", "timestamp"]],
                    use_container_width=True
                )

                st.markdown("**View full session details:**")
                session_id = st.number_input("Session ID", min_value=1, step=1)
                if st.button("Show Session"):
                    row = df_sessions[df_sessions["id"] == session_id]
                    if not row.empty:
                        r = row.iloc[0]
                        st.markdown(f"**User:** {r['username']}  \n**Time:** {r['timestamp']}")
                        st.markdown(f"**Request type:** {r['request_type']}")
                        st.markdown(f"**Question:** {r['question']}")

                        agents = [
                            ("📚 Retrieved Hadith",    "retrieved_hadith"),
                            ("📖 Retrieved Quran",     "retrieved_quran"),
                            ("✍️ Generated Khutbah",   "generated_khutbah"),
                            ("✅ Verification Result", "verification_result"),
                            ("💬 Generated Answer",    "generated_answer"),
                            ("🕌 Best Khutbah",        "best_khutbah"),
                        ]
                        for label, col in agents:
                            content = r.get(col, "").strip()
                            if content:
                                with st.expander(label):
                                    st.write(content)

                        # Show scoring attempts if any
                        attempts_raw = r.get("all_attempts", "[]")
                        try:
                            attempts = json.loads(attempts_raw)
                            if attempts:
                                with st.expander("📊 Validation Attempts"):
                                    for a in attempts:
                                        s = a["scores"]
                                        st.markdown(
                                            f"**Attempt #{a['attempt']}** — "
                                            f"BLEU: {s['bleu']} | ROUGE-L: {s['rouge']} | BERT-F1: {s['bert_f1']}"
                                        )
                        except Exception:
                            pass
                    else:
                        st.warning("Session ID not found.")

        elif admin_view == "All Users":
            st.markdown("---")
            users = auth_get_all_users()
            if users:
                df_users = pd.DataFrame(users)
                st.markdown(f"**Total registered users:** {len(df_users)}")
                st.dataframe(df_users, use_container_width=True)
            else:
                st.info("No users found.")

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>Islamic Knowledge Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>مساعد المعرفة الإسلامية</div>", unsafe_allow_html=True)
st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

question = st.text_area(
    "Enter your question",
    placeholder="e.g.  اريد خطبة عن الصبر  /  hadith about patience  /  is this hadith correct: ...",
    height=100,
    label_visibility="collapsed"
)

run = st.button("Submit  ›")

if run and question.strip():
    mode_map = {
        "Khotbah Generation":    "Khotbah generation",
        "Hadith Return":         "Hadith Return",
        "Hadith Verification":   "Hadith verification",
        "Hadith from Khotbah":   "Hadith from khotbah",
    }
    preset_type = mode_map.get(mode, "")

    initial_state: State = {
        "Question":           question.strip(),
        "User_Request_type":  preset_type,
        "Retrieved_Hadith":   "",
        "Retrieved_Quran":    "",
        "Topic":              "",
        "generated_khutbah":  "",
        "generated_answer":   "",
        "User_Hadith":        "",
        "verification_result": "",
        "round":              0,
        "next":               "",
        "reference_khutbah":  "",
        "khutbah_scores":     {},
        "validation_passed":  None,
        "regen_count":        0,
        "all_attempts":       [],
        "best_khutbah":       "",
        "topic_not_found":    False,
        "llm_topic":          "",
        "khotbah_input":      "",
        "source_hadiths":     None,
        "source_quran":       None,
    }

    with st.spinner("Processing..."):
        final = graph.invoke(initial_state, {"recursion_limit": 100})
        log_session(username, final)  # ✅ Save full pipeline output to DB

    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

    req_type = final.get("User_Request_type", "")

    # ── Khutbah result ────────────────────────────────────
    best_khutbah = final.get("best_khutbah", "").strip()
    if best_khutbah:
        st.markdown("<div class='section-label'>Generated Khutbah</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='arabic-card'>{best_khutbah}</div>", unsafe_allow_html=True)

        # Show scores if available
        attempts = final.get("all_attempts", [])
        if attempts:
            st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Validation Scores</div>", unsafe_allow_html=True)
            best = pick_best_attempt(attempts)
            for a in attempts:
                s = a["scores"]
                is_best = a["attempt"] == best["attempt"]
                badge = " ⭐ Best" if is_best else ""
                st.markdown(
                    f"<div class='score-card'>"
                    f"<b>Attempt #{a['attempt']}{badge}</b><br>"
                    f"BLEU: {s['bleu']} &nbsp;|&nbsp; ROUGE-L: {s['rouge']} &nbsp;|&nbsp; BERT-F1: {s['bert_f1']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # ── Alukah fallback / Hadith answer ──────────────────
    elif final.get("generated_answer", "").strip():
        answer = final["generated_answer"].strip()
        st.markdown("<div class='section-label'>Result</div>", unsafe_allow_html=True)
        if "⚠️" in answer:
            parts          = answer.split("=" * 50)
            warning_header = parts[0].strip().replace("\n", "<br>") if parts else ""
            khutbah_body   = parts[1].strip() if len(parts) > 1 else ""
            reference_foot = parts[2].strip().replace("\n", "<br>") if len(parts) > 2 else ""
            st.markdown(f"<div class='warning-card'>{warning_header}</div>", unsafe_allow_html=True)
            if khutbah_body:
                st.markdown(f"<div class='arabic-card'>{khutbah_body}</div>", unsafe_allow_html=True)
            if reference_foot:
                st.markdown(f"<div class='score-card' style='direction:rtl;text-align:right'>{reference_foot}</div>", unsafe_allow_html=True)
        else:
            answer_html = answer.replace("\n", "<br>")
            st.markdown(f"<div class='arabic-card'>{answer_html}</div>", unsafe_allow_html=True)
    # ── Verification result ───────────────────────────────
    elif final.get("verification_result", "").strip():
        st.markdown("<div class='section-label'>Verification</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-card'>{final['verification_result']}</div>", unsafe_allow_html=True)

    # ── Hadith from khotbah result ───────────────────────
    source_hadiths = final.get("source_hadiths") or []
    source_quran   = final.get("source_quran")   or []
    if source_hadiths or source_quran:
        st.markdown("<div class='section-label'>Source Analysis</div>", unsafe_allow_html=True)

        def pct_color(p):
            if p >= 75: return "#1D9E75"
            if p >= 50: return "#C9A84C"
            return "#8A8070"

        def pct_label(p):
            if p >= 75: return "Strong match"
            if p >= 50: return "Possible match"
            return "Weak match"

        if source_hadiths:
            st.markdown("<div class='section-label' style='font-size:0.65rem'>Hadith sources</div>", unsafe_allow_html=True)
            for i, h in enumerate(source_hadiths, 1):
                pct = h['final_pct']
                col = pct_color(pct)
                lbl = pct_label(pct)
                st.markdown(
                    f"""<div class='score-card' style='border-left:3px solid {col}'>
                    <span style='color:{col};font-weight:700;font-size:1.1rem'>{pct}%</span>
                    &nbsp;<span style='color:{col};font-size:0.75rem'>{lbl}</span>
                    &nbsp;&nbsp;<b>#{i}</b> — {h.get('reference','')}<br>
                    <span style='font-size:0.85rem;color:var(--text-muted)'>
                    Vector: {h['vec_score']} &nbsp;|&nbsp;
                    LLM: {h['llm_score']} &nbsp;|&nbsp;
                    Exact: {h['exact_score']}
                    </span>
                    </div>""",
                    unsafe_allow_html=True
                )
                with st.expander(f"View hadith #{i} — {h.get('reference','')}"):
                    st.markdown(f"<div class='arabic-card'>{h.get('arabic','')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='result-card'>{h.get('text','')}</div>", unsafe_allow_html=True)

        if source_quran:
            st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label' style='font-size:0.65rem'>Quran sources</div>", unsafe_allow_html=True)
            for i, q in enumerate(source_quran, 1):
                pct = q['final_pct']
                col = pct_color(pct)
                lbl = pct_label(pct)
                st.markdown(
                    f"""<div class='score-card' style='border-left:3px solid {col}'>
                    <span style='color:{col};font-weight:700;font-size:1.1rem'>{pct}%</span>
                    &nbsp;<span style='color:{col};font-size:0.75rem'>{lbl}</span>
                    &nbsp;&nbsp;<b>#{i}</b> — Surah {q.get('surah_name','')} ({q.get('reference','')})<br>
                    <span style='font-size:0.85rem;color:var(--text-muted)'>
                    Vector: {q['vec_score']} &nbsp;|&nbsp;
                    LLM: {q['llm_score']} &nbsp;|&nbsp;
                    Exact: {q['exact_score']}
                    </span>
                    </div>""",
                    unsafe_allow_html=True
                )
                with st.expander(f"View ayah #{i} — {q.get('reference','')}"):
                    st.markdown(f"<div class='arabic-card'>{q.get('text','')}</div>", unsafe_allow_html=True)

    # ── Sources ───────────────────────────────────────────
    hadith_ctx = final.get("Retrieved_Hadith", "").strip()
    quran_ctx  = final.get("Retrieved_Quran",  "").strip()

    if hadith_ctx and hadith_ctx != "NO_RELEVANT_HADITH_FOUND" or quran_ctx and quran_ctx != "NO_RELEVANT_QURAN_FOUND":
        st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Retrieved Sources</div>", unsafe_allow_html=True)

        tab_labels = []
        if hadith_ctx and hadith_ctx != "NO_RELEVANT_HADITH_FOUND": tab_labels.append("Hadith")
        if quran_ctx  and quran_ctx  != "NO_RELEVANT_QURAN_FOUND":  tab_labels.append("Quran")

        if tab_labels:
            tabs = st.tabs(tab_labels)
            idx  = 0
            if hadith_ctx and hadith_ctx != "NO_RELEVANT_HADITH_FOUND":
                with tabs[idx]:
                    for block in hadith_ctx.split("\n\n"):
                        if block.strip():
                            st.markdown(f"<div class='result-card'>{block.strip()}</div>", unsafe_allow_html=True)
                idx += 1
            if quran_ctx and quran_ctx != "NO_RELEVANT_QURAN_FOUND":
                with tabs[idx]:
                    for block in quran_ctx.split("\n\n"):
                        if block.strip():
                            st.markdown(f"<div class='arabic-card'>{block.strip()}</div>", unsafe_allow_html=True)

    with st.expander("Raw State"):
        st.json(final)

elif run and not question.strip():
    st.warning("Please enter a question.")

