"""
FastAPI Server for Peer-Review Devil's Advocate (Learnix Research)
Serves the unified Single-Page Application (SPA) and REST API with Gemini 2.0 Flash Agent.
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import random
import time
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()
# Models
class SearchRequest(BaseModel):
    query: str
    source: Optional[str] = "All"
    attack_vector: Optional[str] = "All"

class ReportRequest(BaseModel):
    query: str
    critique_ids: Optional[List[str]] = None
    exclude_ids: Optional[List[str]] = None

class ChatMessageRequest(BaseModel):
    role: str
    message: str

class DeleteChatMessageRequest(BaseModel):
    message: str

class SavePaperRequest(BaseModel):
    paper_id: str
    title: Optional[str] = ""
    authors: Optional[str] = ""
    abstract: Optional[str] = ""
    source: Optional[str] = "ArXiv"
    source_id: Optional[str] = ""
    attack_vector: Optional[str] = "Peer Review"
    url: Optional[str] = ""

class DeletePaperRequest(BaseModel):
    paper_id: str

class IngestRequest(BaseModel):
    title: str
    authors: Optional[str] = "Anonymous"
    year: Optional[int] = 2024
    source: Optional[str] = "ArXiv"
    url: Optional[str] = "https://arxiv.org/abs/2401.00001"
    raw_text: str

# Auth Models
class SendOTPRequest(BaseModel):
    email: str
    purpose: Optional[str] = "registration"

class VerifyOTPRequest(BaseModel):
    email: str
    otp: str
    purpose: Optional[str] = "registration"

class RegisterRequest(BaseModel):
    name: str
    email: str
    otp: str
    password: str
    confirm_password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ResetPasswordRequest(BaseModel):
    email: str
    otp: str
    new_password: str
    confirm_password: str

class UpdateProfileRequest(BaseModel):
    name: str
    email: str
    avatar: Optional[str] = None
    otp: Optional[str] = None


from backend.database import CRITIQUE_DATABASE
from backend.live_agent import fetch_arxiv_realtime, fetch_pubpeer_realtime, fetch_arxiv_by_id
from backend.gemini_agent import run_gemini_devils_advocate, tag_severity_gemini
from backend.adversarial_rag import (
    run_adversarial_search, 
    generate_academic_risk_report, 
    classify_paragraph
)
from backend.auth import (
    hash_password, verify_password,
    validate_password_complexity, create_and_send_otp, verify_otp,
    consume_otp, create_session, get_session, destroy_session
)
from backend.db import (
    get_user_by_email, create_user_db, update_user_password_db, update_user_profile_db,
    save_chat_message, get_chat_history, delete_chat_message_db, clear_chat_history_db,
    save_paper_db, get_saved_papers_db, delete_saved_paper_db,
    get_critique_by_id_db, save_critique_chunk_db, get_all_critiques_db,
    get_hotspots_cache_db, save_hotspots_cache_db
)

app = FastAPI(title="Learnix RAG Peer-Review API")

@app.get("/api/status")
def status_endpoint():
    return {
        "status": "online",
        "system": "Learnix Research Realtime Agent",
        "classifier": "DistilBERT Adversarial Gate Active"
    }

@app.get("/api/deep-dive/{critique_id}")
def deep_dive_endpoint(critique_id: str):
    chunk = get_critique_by_id_db(critique_id)
    if not chunk:
        prefix = critique_id.split("-")[0].lower() if "-" in critique_id else "arxiv"
        source_name_map = {
            "zenodo": ("Zenodo", f"https://zenodo.org/records/{critique_id.replace('zenodo-', '')}"),
            "pubpeer": ("PubPeer", f"https://doi.org/{critique_id.replace('pubpeer-', '').replace('-', '/')}"),
            "biorxiv": ("bioRxiv", f"https://www.biorxiv.org/content/{critique_id.replace('biorxiv-', '').replace('-', '/')}"),
            "medrxiv": ("medRxiv", f"https://www.medrxiv.org/content/{critique_id.replace('medrxiv-', '').replace('-', '/')}"),
            "openalex": ("OpenAlex", f"https://openalex.org/{critique_id.replace('openalex-', '')}"),
            "s2": ("Semantic Scholar", f"https://www.semanticscholar.org/paper/{critique_id.replace('s2-', '')}"),
            "pmc": ("PubMed Central", f"https://pmc.ncbi.nlm.nih.gov/articles/{critique_id.replace('pmc-', '')}/"),
            "doaj": ("DOAJ", "https://doaj.org"),
            "openaire": ("OpenAIRE", "https://www.openaire.eu/")
        }
        src, src_url = source_name_map.get(prefix, ("ArXiv", f"https://arxiv.org/abs/{critique_id.replace('arxiv-', '')}"))

        clean_id = critique_id.replace("arxiv-", "")
        live_paper = fetch_arxiv_by_id(clean_id)
        if live_paper and live_paper.get("status") != "UNAVAILABLE":
            raw_text = live_paper.get("raw_text", live_paper.get("text", ""))
            chunk = {
                "id": critique_id,
                "title": live_paper.get("title", "ArXiv Research Paper"),
                "authors": ", ".join(live_paper.get("authors", [])) if isinstance(live_paper.get("authors"), list) else str(live_paper.get("authors", "")),
                "year": live_paper.get("year", datetime.now().year),
                "source": src,
                "source_id": clean_id,
                "url": src_url,
                "section": f"{src} Realtime Audit Record",
                "attack_vector": live_paper.get("attack_vector", "Benchmark Contamination"),
                "target": live_paper.get("title", "Research Paper"),
                "risk_level": live_paper.get("risk_level", "Major"),
                "skepticism_score": float(live_paper.get("skepticism_score", 88.0)),
                "replication_prob": float(live_paper.get("replication_prob", 20.0)),
                "distilbert_tag": live_paper.get("distilbert_tag", f"Methodological Limitation — {src} Audit"),
                "text": raw_text[:500] if raw_text else "No text abstract available."
            }
            save_critique_chunk_db(chunk)
        else:
            raise HTTPException(status_code=404, detail=f"Critique item '{critique_id}' not found.")
        
    skep = float(chunk.get("skepticism_score", 85.0))
    repl = float(chunk.get("replication_prob", round(100.0 - skep, 1)))

    return {
        "id": chunk["id"],
        "title": chunk["title"],
        "authors": chunk["authors"],
        "year": chunk["year"],
        "source": chunk["source"],
        "source_id": chunk.get("source_id", ""),
        "url": chunk.get("url", ""),
        "section": chunk.get("section", "Peer Review Forum"),
        "attack_vector": chunk.get("attack_vector", "Methodological Limitation"),
        "target": chunk.get("target", chunk["title"]),
        "risk_level": chunk.get("risk_level", "Major"),
        "skepticism_score": skep,
        "replication_prob": repl,
        "distilbert_tag": chunk.get("distilbert_tag", "Methodological Limitation"),
        "text": chunk.get("text", chunk.get("raw_text", "No text abstract available.")),
        "rebuttal_odds": round(max(5.0, 100.0 - skep), 1)
    }

@app.post("/api/ingest")
def ingest_endpoint(req: IngestRequest):
    paragraphs = [p.strip() for p in req.raw_text.split("\n\n") if p.strip()]
    ingested_chunks = []
    dropped_count = 0
    
    for idx, p in enumerate(paragraphs):
        p_type, conf, tag = classify_paragraph(p)
        if p_type == "Limitation/Critique":
            chunk_id = f"crit-custom-{int(time.time())}-{idx+1}"
            new_item = {
                "id": chunk_id,
                "title": req.title,
                "authors": req.authors,
                "year": req.year,
                "source": "User Ingest",
                "source_id": f"Custom-{random.randint(100,999)}",
                "url": req.url,
                "section": "Uploaded Manuscript - Scrutinized Paragraph",
                "attack_vector": "Custom Critique / Limitation",
                "target": req.title,
                "risk_level": "Major",
                "skepticism_score": 85.0,
                "replication_prob": 25.0,
                "paragraph_type": p_type,
                "distilbert_tag": tag,
                "text": p,
                "query_keywords": [w.lower() for w in p.split() if len(w) > 4][:10],
                "severity": "Major",
                "mitigation_suggestion": "Review data preprocessing and baseline comparisons for potential biases or leakage."
            }
            save_critique_chunk_db(new_item)
            ingested_chunks.append(new_item)
        else:
            dropped_count += 1
            
    return {
        "status": "success",
        "total_paragraphs_evaluated": len(paragraphs),
        "limitation_chunks_stored": len(ingested_chunks),
        "marketing_chunks_dropped": dropped_count,
        "stored_chunks": ingested_chunks
    }

@app.get("/api/hotspots")
def hotspots_endpoint(refresh: bool = False):
    today_str = datetime.now().strftime("%Y-%m-%d")
    date_display = datetime.now().strftime('%b %d, %Y')

    if not refresh:
        cached = get_hotspots_cache_db(today_str)
        if cached:
            return cached

    try:
        hotspots = []
        
        # 1. Fetch requested high-impact ArXiv papers dynamically
        priority_ids = ["2504.16021", "1907.02664", "2005.07866"]
        for p_id in priority_ids:
            try:
                paper = fetch_arxiv_by_id(p_id)
                if paper and paper.get("status") != "UNAVAILABLE":
                    hotspots.append({
                        "id": paper["id"],
                        "source_tag": f"ARXIV:{paper.get('source_id', p_id)}",
                        "source_type": "arxiv",
                        "title": paper["title"],
                        "summary": paper.get("raw_text", paper.get("text", ""))[:200] + "...",
                        "skepticism_index": round(paper.get("skepticism_score", 88.0), 1),
                        "metric_label": "Skepticism Index:",
                        "updated": f"Live ArXiv Sync ({date_display})",
                        "badge_icon": "analytics"
                    })
                    save_critique_chunk_db(paper)
            except Exception as ex_id:
                print(f"[hotspots id fetch warning]: {ex_id}")

        # 2. Fetch live real-time papers based on daily topic rotation
        day_of_year = datetime.now().timetuple().tm_yday
        daily_queries = [
            "cognitive flow reasoning support interventions",
            "byzantine resilient distributed optimization",
            "reasoning benchmarks data contamination leakage",
            "agentic reasoning self correction",
            "transformer architectural redundancy linear baselines"
        ]
        chosen_query = daily_queries[day_of_year % len(daily_queries)]

        live_arxiv = fetch_arxiv_realtime(chosen_query, max_results=3, sort_by_latest=True)
        live_pubpeer = fetch_pubpeer_realtime("deep learning model critique flaw", max_results=3, sort_by_latest=True)
        
        for idx, paper in enumerate((live_arxiv or []) + (live_pubpeer or [])):
            if not any(h["id"] == paper["id"] for h in hotspots):
                source_prefix = paper.get("source", "ArXiv").upper()
                source_tag = f"{source_prefix}:{paper.get('source_id', paper['id'])}"
                hotspots.append({
                    "id": paper["id"],
                    "source_tag": source_tag,
                    "source_type": paper.get("source", "arxiv").lower(),
                    "title": paper["title"],
                    "summary": paper.get("raw_text", paper.get("text", ""))[:200] + "...",
                    "skepticism_index": round(86.0 + (idx * 2.3) % 12, 1),
                    "metric_label": "Skepticism Index:",
                    "updated": f"Live Daily Sync ({date_display})",
                    "badge_icon": "analytics"
                })
                save_critique_chunk_db({
                    "id": paper["id"],
                    "title": paper["title"],
                    "authors": paper.get("authors", ["Academic Audit Team"]),
                    "year": paper.get("year", datetime.now().year),
                    "source": paper.get("source", "ArXiv"),
                    "source_id": paper.get("source_id", paper["id"]),
                    "url": paper.get("url", "https://arxiv.org"),
                    "section": paper.get("section", "Peer-Review Audit"),
                    "attack_vector": paper.get("attack_vector", "Methodological Flaw"),
                    "target": paper["title"],
                    "risk_level": paper.get("risk_level", "Major"),
                    "skepticism_score": paper.get("skepticism_score", 88.0),
                    "replication_prob": paper.get("replication_prob", 12.0),
                    "paragraph_type": "Limitation/Critique",
                    "distilbert_tag": paper.get("distilbert_tag", "Methodological Limitation"),
                    "text": paper.get("raw_text", paper.get("text", "")),
                    "severity": paper.get("risk_level", "Major"),
                    "mitigation_suggestion": "Conduct independent post-pub replication audit."
                })

        if hotspots:
            save_hotspots_cache_db(hotspots, today_str)

    except Exception as e:
        print(f"[hotspots warning]: {e}")

    return get_hotspots_cache_db(today_str) or []

from fastapi import Header



# Authentication Endpoints
@app.post("/api/auth/send-otp")
def send_otp_endpoint(req: SendOTPRequest):
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email address.")

    success, msg = create_and_send_otp(req.email, req.purpose)
    if not success:
        raise HTTPException(status_code=500, detail=msg)
    return {"status": "success", "message": msg}

@app.post("/api/auth/verify-otp")
def verify_otp_endpoint(req: VerifyOTPRequest):
    success, msg = verify_otp(req.email, req.otp, req.purpose)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}

@app.post("/api/auth/register")
def register_endpoint(req: RegisterRequest):
    email = req.email.strip().lower()
    name = req.name.strip()

    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    valid_pass, pass_msg = validate_password_complexity(req.password)
    if not valid_pass:
        raise HTTPException(status_code=400, detail=pass_msg)

    valid_otp, otp_msg = verify_otp(email, req.otp, "registration")
    if not valid_otp:
        raise HTTPException(status_code=400, detail=otp_msg)

    if get_user_by_email(email):
        raise HTTPException(status_code=400, detail="Email is already registered.")

    hashed_pw, salt = hash_password(req.password)
    created = create_user_db(name, email, hashed_pw, salt)
    if not created:
        raise HTTPException(status_code=400, detail="Email is already registered.")

    consume_otp(email)

    return {
        "status": "success",
        "message": "Account created successfully! Please sign in with your email and password."
    }

@app.post("/api/auth/login")
def login_endpoint(req: LoginRequest):
    email = req.email.strip().lower()
    user = get_user_by_email(email)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not verify_password(req.password, user["password_hash"], user["salt"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_session(email, user["name"])
    return {
        "status": "success",
        "message": "Logged in successfully!",
        "token": token,
        "user": {
            "name": user["name"], 
            "email": email,
            "avatar": user.get("avatar")
        }
    }

@app.post("/api/auth/update-profile")
def update_profile_endpoint(req: UpdateProfileRequest, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    old_email = session["email"]
    new_email = req.email.strip().lower()
    new_name = req.name.strip()

    if not new_email or "@" not in new_email:
        raise HTTPException(status_code=400, detail="Invalid email address.")
    if not new_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    if new_email != old_email:
        existing = get_user_by_email(new_email)
        if existing:
            raise HTTPException(status_code=400, detail="Email is already in use by another account.")
        if not req.otp:
            raise HTTPException(status_code=400, detail="Verification code required to update email address. Please click 'Send Verification Code'.")
        valid_otp, otp_msg = verify_otp(new_email, req.otp, "email_change")
        if not valid_otp:
            valid_otp, otp_msg = verify_otp(new_email, req.otp, "reset")
            if not valid_otp:
                raise HTTPException(status_code=400, detail=otp_msg)
        consume_otp(new_email)

    success = update_user_profile_db(old_email, new_name, new_email, req.avatar)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update profile in database.")

    user_info = get_user_by_email(new_email)
    return {
        "status": "success",
        "message": "Profile updated successfully!",
        "user": {
            "name": new_name,
            "email": new_email,
            "avatar": user_info.get("avatar") if user_info else req.avatar
        }
    }

@app.post("/api/auth/reset-password")
def reset_password_endpoint(req: ResetPasswordRequest):
    email = req.email.strip().lower()
    user = get_user_by_email(email)

    if not user:
        raise HTTPException(status_code=404, detail="No account registered with this email.")

    if req.new_password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    valid_pass, pass_msg = validate_password_complexity(req.new_password)
    if not valid_pass:
        raise HTTPException(status_code=400, detail=pass_msg)

    valid_otp, otp_msg = verify_otp(email, req.otp, "reset")
    if not valid_otp:
        raise HTTPException(status_code=400, detail=otp_msg)

    hashed_pw, salt = hash_password(req.new_password)
    update_user_password_db(email, hashed_pw, salt)
    consume_otp(email)

    return {"status": "success", "message": "Password reset successfully. You can now log in with your new password."}

@app.get("/api/auth/me")
def me_endpoint(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    user = get_user_by_email(session["email"])
    avatar_val = user.get("avatar") if user else None
    return {
        "status": "success", 
        "user": {
            "name": user["name"] if user else session["name"], 
            "email": session["email"],
            "avatar": avatar_val
        }
    }

@app.post("/api/auth/logout")
def logout_endpoint(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        destroy_session(token)
    return {"status": "success", "message": "Logged out."}

# Chat History Endpoints
@app.get("/api/chat/history")
def get_chat_history_endpoint(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    history = get_chat_history(session["email"])
    return {"status": "success", "history": history}

@app.post("/api/chat/message")
def save_chat_message_endpoint(req: ChatMessageRequest, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    save_chat_message(session["email"], req.role, req.message)
    return {"status": "success", "message": "Message saved."}

@app.post("/api/chat/delete-item")
def delete_chat_item_endpoint(req: DeleteChatMessageRequest, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    delete_chat_message_db(session["email"], req.message)
    return {"status": "success", "message": "Item deleted."}

@app.post("/api/chat/clear")
def clear_chat_history_endpoint(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    clear_chat_history_db(session["email"])
    return {"status": "success", "message": "History cleared."}

# Saved Papers Endpoints
@app.get("/api/papers/saved")
def get_saved_papers_endpoint(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    papers = get_saved_papers_db(session["email"])
    return {"status": "success", "papers": papers}

@app.post("/api/papers/save")
def save_paper_endpoint(req: SavePaperRequest, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    save_paper_db(
        email=session["email"],
        paper_id=req.paper_id,
        title=req.title or "",
        authors=req.authors or "",
        abstract=req.abstract or "",
        source=req.source or "ArXiv",
        source_id=req.source_id or "",
        attack_vector=req.attack_vector or "Peer Review",
        url=req.url or ""
    )
    return {"status": "success", "message": "Paper saved."}

@app.post("/api/papers/remove")
def remove_saved_paper_endpoint(req: DeletePaperRequest, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    delete_saved_paper_db(session["email"], req.paper_id)
    return {"status": "success", "message": "Paper removed."}

# API Endpoints
@app.post("/api/search")
def search_endpoint(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    results = run_adversarial_search(
        user_query=req.query,
        source_filter=req.source,
        attack_vector_filter=req.attack_vector
    )
    return results

@app.post("/api/preview-search")
def preview_search_endpoint(req: SearchRequest):
    """
    Live Interrogation Mode — Phase 1: Preview.
    Returns ONLY candidate titles, sources, and relevance scores.
    No full report synthesis. User must approve before full search.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    
    from backend.live_agent import (
        detect_query_domain, has_explicit_ids, extract_explicit_ids,
        extract_query_keywords
    )
    
    results = run_adversarial_search(
        user_query=req.query,
        source_filter=req.source,
        attack_vector_filter=req.attack_vector
    )
    
    query_keywords = extract_query_keywords(req.query)
    domain = detect_query_domain(req.query)
    has_ids = has_explicit_ids(req.query)
    extracted_ids = extract_explicit_ids(req.query) if has_ids else {}
    
    candidates = []
    for item in results.get("results", []):
        candidates.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "source": item.get("source"),
            "relevance_score": item.get("relevance_score", 0),
            "keyword_overlap": item.get("keyword_overlap", 0),
            "attack_vector": item.get("attack_vector"),
            "exact_id_fetch": item.get("exact_id_fetch", False)
        })
    
    return {
        "query": req.query,
        "domain": domain,
        "query_keywords": query_keywords,
        "has_explicit_ids": has_ids,
        "extracted_ids": extracted_ids,
        "total_candidates": len(candidates),
        "candidates": candidates
    }

@app.post("/api/generate-report")
def generate_report_endpoint(req: ReportRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty for report generation.")
    report = generate_academic_risk_report(req.query, req.critique_ids, req.exclude_ids)
    return report

@app.get("/favicon.ico")
@app.get("/favicon.svg")
def get_favicon():
    favicon_path = os.path.join(os.path.dirname(__file__), "static", "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/")
def get_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Peer-Review Devil's Advocate Backend Running</h1>")

if os.path.exists(os.path.join(os.path.dirname(__file__), "static")):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{full_path:path}")
def catch_all_spa_route(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found.")
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Peer-Review Devil's Advocate Backend Running</h1>")

