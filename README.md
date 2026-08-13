# Learnix Research

An AI-powered Adversarial Retrieval-Augmented Generation (RAG) platform designed for researchers, academics, and peer-reviewers. It stress-tests research queries and thesis claims by retrieving real-time academic literature, identifying methodology flaws, evaluation data leakage, and baseline omissions, and generating dynamic, evidence-backed risk reports and mitigations.

## Key Features

- **Adversarial RAG & Risk Analysis**: Stress-tests thesis claims by scanning academic literature for methodology flaws, evaluation data leakage, and baseline omissions.
- **BGE Semantic Embedding & Cross-Encoder Reranking**:
  - **Primary Embedding**: `BAAI/bge-small-en-v1.5` (384 dimensions) via PyTorch `sentence_transformers` or ONNX Runtime `fastembed` (Render free tier optimized).
  - **Reranker Engine**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for high-precision pairwise relevance ranking.
  - **Hybrid Pipeline**: 30% TF-IDF + 70% BGE Cosine Similarity with automatic fallback triggers.
- **Multi-Publisher Real-Time Search**: Live API integration with ArXiv, Zenodo, PubPeer, CrossRef, OpenAlex, Semantic Scholar, and bioRxiv.
- **Gemini 2.0 Flash Synthesis**: Multi-source paper synthesis, ambiguity resolution, and query-bound evidence-backed mitigations.
- **100% Supabase Cloud PostgreSQL**: Secure data storage with Row Level Security (RLS) across 7 tables (`users`, `sessions`, `otps`, `chat_history`, `saved_papers`, `critiques`, `hotspots_cache`).
- **SMTP Email OTP Verification**: 6-digit verification for account registration, email updates, and password recovery.
- **Interactive Laboratory SPA**: Single Page Application with glassmorphism UI, paper critique modals, saved paper management, and trending academic hotspots.

## Component Overview

| Component | What it is |
| :--- | :--- |
| **Backend Server** (`uvicorn app:app --reload`) | A FastAPI server handling LLM synthesis, BGE/Cross-Encoder retrieval, and secure Supabase DB operations. |
| **BGE Embedding Service** (`backend/embedding_service.py`) | Dual-engine BGE v1.5 vector embedding service (`sentence_transformers` PyTorch / `fastembed` ONNX) with SHA-256 caching. |
| **Cross-Encoder Service** (`backend/embedding_service.py`) | Pairwise relevance reranker using `ms-marco-MiniLM-L-6-v2` to filter out irrelevant research results. |
| **Interactive Laboratory** | A Single Page Application (SPA) where users input thesis claims, explore literature, and view dynamic risk reports. |

## How it works

```text
User Query ──► Adversarial RAG Pipeline
                    │
                    ▼
  BGE & Cross-Encoder Reranking (384-dim BGE-small + MS-MARCO)
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

The frontend connects to the backend API which handles authentication, literature retrieval, hybrid semantic reranking, and Gemini synthesis via Supabase for secure data isolation.

## Project structure

```text
Learnix-RAG/
├── app.py                     # Starts the FastAPI server and handles API routing
├── requirements.txt           # Base requirements (FastAPI, Supabase, fastembed)
├── requirements-ml.txt        # Optional PyTorch/sentence-transformers dependencies
├── render.yaml                # Render cloud deployment configuration
├── backend/                   # Backend API and RAG logic
│   ├── embedding_service.py   # BGE v1.5 embedding & Cross-Encoder reranking pipeline
│   ├── gemini_agent.py        # Gemini 2.0 Flash integration & synthesis
│   ├── database.py            # Real-time search dispatcher & critique aggregation pipeline
│   ├── db.py                  # 100% Supabase Cloud PostgreSQL DB client & RLS queries
│   ├── auth.py                # OTP email verification & password security
│   ├── live_agent.py          # Real-time multi-publisher academic REST APIs
│   ├── adversarial_rag.py     # Risk scoring, flaw taxonomy, and mitigation engine
│   └── supabase_schema.sql    # Supabase DDL schema definition for 7 core tables
│
└── static/                    # Frontend Single Page Application
    ├── index.html             # Main UI entrypoint with glassmorphism design
    └── favicon.svg            # Brand logo
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

*(Optional: For full PyTorch GPU/CPU acceleration, install `requirements-ml.txt`)*:
```bash
pip install -r requirements-ml.txt
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
| `GOOGLE_API_KEY` | ✅ | [aistudio.google.com](https://aistudio.google.com/) (Same as `GEMINI_API_KEY`) |
| `SUPABASE_URL` | ✅ | [supabase.com](https://supabase.com/) → API settings |
| `SUPABASE_KEY` | ✅ | [supabase.com](https://supabase.com/) → API settings (`anon` public key) |
| `SUPABASE_ANON_KEY` | Optional | [supabase.com](https://supabase.com/) → API settings |
| `SMTP_SERVER` | Optional | e.g., `smtp.gmail.com` |
| `SMTP_PORT` | Optional | e.g., `587` |
| `SENDER_EMAIL` | Optional | Your sending email address |
| `SENDER_APP_PASSWORD` | Optional | Your email provider app password |

## Tech stack
- **FastAPI** — Backend REST framework
- **BGE v1.5 & Cross-Encoder** — `BAAI/bge-small-en-v1.5` & `cross-encoder/ms-marco-MiniLM-L-6-v2` semantic reranking
- **Google Gemini 2.0 Flash** — AI Engine for synthesis and mitigations
- **Supabase Cloud PostgreSQL** — Database with RLS policies
- **Live Search APIs** — ArXiv, Zenodo, PubPeer, CrossRef, OpenAlex, Semantic Scholar, bioRxiv
- **HTML5/JS/TailwindCSS** — Frontend SPA UI with glassmorphism design

## License
MIT
