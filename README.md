# Learnix Research

🎉 Official Public Release: Learnix Research is now officially released!

An AI-powered Adversarial Retrieval-Augmented Generation (RAG) platform designed for researchers, academics, and peer-reviewers. It stress-tests research queries and thesis claims by retrieving real-time academic literature, identifying methodology flaws, evaluation data leakage, and baseline omissions, and generating dynamic, evidence-backed risk reports and mitigations.

## Component Overview

| Component | What it is |
| :--- | :--- |
| **Backend Server** (`uvicorn app:app --reload`) | A FastAPI server that handles LLM synthesis, literature retrieval, and secure DB connections. |
| **Interactive Laboratory** | A Single Page Application (SPA) where users input thesis claims, explore literature, and view dynamic risk reports. |

## How it works

```text
User Query ──► Adversarial RAG Pipeline
                    │
                    ▼
             LLM (Gemini 2.0 Flash)  ◄──────► Search APIs (ArXiv, Zenodo, PubPeer, etc.)
                    │                              ├─ methodology_flaws
                    ▼                              ├─ evaluation_data_leakage
             Dynamic Risk Report                   └─ baseline_omissions
                    │
                    ▼
             Interactive Laboratory SPA
```

The frontend connects to the backend API which handles authentication, literature retrieval, and Gemini synthesis via Supabase for secure data isolation.

## Project structure
```text
Learnix-RAG/
├── app.py              # Starts the FastAPI server and handles routing
├── requirements.txt
├── backend/            # Backend API and RAG logic
│   ├── gemini_agent.py # Gemini 2.0 Flash integration and synthesis
│   ├── database.py     # Supabase DB operations and RLS
│   └── adversarial_rag.py # Main logic for literature retrieval and risk reporting
│
└── static/             # Frontend Single Page Application
    ├── index.html      # Main UI entrypoint
    ├── style.css       # UI styling (Vanilla CSS + Tailwind)
    └── favicon.svg     # Brand logo
```

## Quick start (For Developers)

### 1. Prerequisites
- Python ≥ 3.10
- A Supabase Cloud project
- A Google Gemini API Key

### 2. Clone & install
```bash
git clone https://github.com/Abinav-max/Learnix-RAG.git
cd Learnix-RAG
pip install -r requirements.txt
```

### 3. Set up environment
Create a `.env` file in the root directory:
```bash
touch .env
```
(Open the newly created `.env` file and fill in your API keys using the reference below)

### 4. Database Setup
Execute the DDL script in `backend/supabase_schema.sql` inside your Supabase SQL Editor to create all 7 required tables and Row Level Security (RLS) policies.

### 5. Run — Local Server

```bash
uvicorn app:app --reload --port 8000
```
Starts the FastAPI backend server on `http://127.0.0.1:8000`. Navigate to this URL in your browser to interact with the Learnix Research SPA.

## Environment variables
Create a `.env` file and fill in the values below.

| Variable | Required | Where to get it |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | ✅ | [aistudio.google.com](https://aistudio.google.com/) |
| `SUPABASE_URL` | ✅ | [supabase.com](https://supabase.com/) → API settings |
| `SUPABASE_KEY` | ✅ | [supabase.com](https://supabase.com/) → API settings |
| `SMTP_SERVER` | Optional | e.g., `smtp.gmail.com` |
| `SMTP_PORT` | Optional | e.g., `587` |
| `SENDER_EMAIL` | Optional | Your sending email address |
| `SENDER_APP_PASSWORD`| Optional | Your email provider app password |

## Tech stack
- **FastAPI** — Backend framework
- **Google Gemini 2.0 Flash** — AI Engine for synthesis and mitigations
- **Supabase Cloud PostgreSQL** — Database with RLS
- **Live Search APIs** — ArXiv, Zenodo, PubPeer, CrossRef, OpenAlex, Semantic Scholar
- **HTML5/JS/TailwindCSS** — Frontend SPA UI

## License
MIT