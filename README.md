# Learnix Research — Peer-Review Devil's Advocate RAG Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)
[![Gemini 2.0](https://img.shields.io/badge/Google_Gemini_2.0_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**Learnix Research** is an AI-powered Adversarial Retrieval-Augmented Generation (RAG) platform designed for researchers, academics, and peer-reviewers. It stress-tests research queries and thesis claims by retrieving real-time academic literature, identifying methodology flaws, evaluation data leakage, and baseline omissions, and generating dynamic, evidence-backed risk reports and mitigations.

---

## 🌟 Key Features

- **Adversarial RAG Pipeline**: Evaluates user thesis claims against live academic literature fetched from ArXiv, Zenodo, PubPeer, CrossRef, OpenAlex, and Semantic Scholar.
- **Gemini 2.0 Flash Synthesis**: Multi-source research synthesis, relevance gate filtering, and topic-aware hypothesis analysis using Google Gemini.
- **Real-Time Dynamic Mitigations**: Generates 3 query-bound, literature-linked recommendations per search, identifying unaddressed edge cases and missing evaluation baselines.
- **100% Pure Supabase Cloud DB**: High-performance PostgreSQL database with Row Level Security (RLS) ensuring strict per-user data isolation.
- **Multi-User Security & Isolation**: Separate chat history, saved papers, recent searches, and profile settings for every user account.
- **Gmail SMTP OTP Verification**: 6-digit email verification for account registration, email address updates, and password recovery.
- **Interactive Laboratory SPA**: Single Page Application built with dark/light themes, grain overlays, saved paper management, and deep-dive paper critique modals.

---

## 🏗️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend Framework** | Python 3.10+, FastAPI, Uvicorn |
| **Database** | Supabase Cloud PostgreSQL (7 Tables) |
| **AI Engine** | Google Gemini 2.0 Flash (`google-generativeai`) |
| **Search APIs** | Live REST integration with ArXiv, Zenodo, PubPeer, CrossRef, OpenAlex, Semantic Scholar |
| **Authentication** | Bearer Token Sessions, Gmail SMTP TLS OTP, PBKDF2 HMAC SHA-256 Hashing |
| **Frontend UI** | HTML5, Vanilla JavaScript, Vanilla CSS, TailwindCSS (CDN), Material Symbols, Google Fonts |

---

## 📊 Database Schema

The system operates on 7 PostgreSQL tables in Supabase:

- **`users`**: Manages user accounts, names, emails, hashed passwords, salts, and avatar images.
- **`sessions`**: Bearer token session authentication with 7-day inactivity expiration.
- **`otps`**: One-Time Passwords for registration, email changes, and password resets.
- **`chat_history`**: Isolated per-user query history and message logs.
- **`saved_papers`**: Per-user bookmarked academic papers.
- **`critiques`**: Cached paper abstracts, flaw classifications, and attack vectors.
- **`hotspots_cache`**: Real-time research trends and topic cache.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Abinav-max/Learnix-RAG.git
cd Learnix-RAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
# Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Supabase Cloud PostgreSQL
SUPABASE_URL=https://your_project.supabase.co
SUPABASE_KEY=your_supabase_anon_or_service_key_here

# Gmail SMTP OTP Delivery
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_APP_PASSWORD=your_gmail_app_password
```

### 4. Database Setup
Execute the DDL script in `backend/supabase_schema.sql` inside your Supabase SQL Editor to create all 7 required tables and RLS policies.

### 5. Run Local Development Server
```bash
uvicorn app:app --reload --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 🧪 Testing

Run the automated backend test suite:
```bash
python -m unittest test_backend.py test_interrogation_engine.py test_interrogation_fast.py test_tag_blindness.py
```

---

## 🌐 Cloud Deployment

Pre-configured for cloud hosting platforms:
- **Render**: `render.yaml` included for one-click deployment.
- **Vercel / Railway / Heroku**: Standard WSGI/ASGI entrypoint via `app:app`.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.