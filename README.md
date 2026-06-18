# khutbah-ai

A multi-agent LangGraph pipeline that generates, validates, and verifies Islamic khutbahs using RAG over Sahih al-Bukhari and Sahih Muslim.

---

## Overview

khutbah-ai is a graduation project built at Princess Sumaya University for Technology (PSUT). It uses a supervisor-based multi-agent architecture to handle four types of Islamic knowledge requests:

- **Khutbah Generation** — generates a full Friday sermon in Arabic grounded in retrieved Hadith and Quran ayat
- **Hadith Verification** — verifies whether a user-provided hadith exists word-for-word in the database
- **Hadith Return** — retrieves relevant hadiths for a given topic
- **Hadith from Khutbah** — analyzes a khutbah text and identifies which hadiths and ayat were used as sources

---

## Architecture

The system is built on **LangGraph** with a parent supervisor graph and five subgraphs:

```
START → Supervisor
           ├── Planning Subgraph       (Planner → Topic Extractor → Hadith Retrieval → Quran Retrieval)
           ├── Generation Subgraph     (Khutbah Generation → Validator → Alukah Fallback)
           ├── Verification Subgraph   (Hadith Extractor → Hadith Verification)
           ├── Hadith Return Subgraph  (Hadith Return)
           └── Khotbah Sources Subgraph (Extract Khotbah → Hadith Source Finder → Quran Source Finder)
```

The supervisor routes between subgraphs based on state and loops until validation passes or max attempts are reached.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Agent Framework | LangGraph + LangChain |
| LLM | GPT-4.1-mini (OpenAI) |
| Embeddings | text-embedding-3-small |
| Vector Database | MongoDB Atlas Vector Search |
| Hadith Database | Sahih al-Bukhari + Sahih Muslim (JSON) |
| Validation | BLEU, ROUGE-L, BERTScore |
| Auth | MongoDB + bcrypt |
| Web Fallback | Alukah.net via GPT-4o search |

---

## Features

- Arabic and English input support with automatic translation
- Dual-query embedding (original + translated) for better retrieval
- LLM-based relevance filtering on retrieved hadiths and ayat
- Khutbah quality validation using BLEU / ROUGE-L / BERTScore against a reference from Alukah
- Auto-regeneration loop (up to 3 attempts) with best attempt selection
- Admin panel with login history and session logs
- Light/dark mode toggle

---

## Setup

1. Clone the repo
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file based on `.env.example`:
```
MONGO_URI=your-mongodb-uri
AUTH_MONGO_URI=your-auth-mongodb-uri
OPENAI_API_KEY=your-openai-key
```
4. Run the app:
```bash
streamlit run app.py
```

---

## Project Structure

```
khutbah-ai/
├── app.py                  # Main application and agent pipeline
├── Sahih Muslim.json       # Hadith dataset
├── Sahih al-Bukhari.json   # Hadith dataset
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## By

Built by Jalal Toubeh in Princess Sumaya University for Technology (PSUT), 2025/2026
