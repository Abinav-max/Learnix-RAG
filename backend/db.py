"""
Database Module for Learnix Research (100% Supabase Cloud Architecture)
Uses the official Supabase Python client to connect to PostgreSQL.
All persistent storage (users, sessions, OTPs, chat history, saved papers, critiques, hotspots cache)
relies exclusively on Supabase cloud database.
"""

import os
import time
from typing import Dict, Any, Optional, List
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv(override=True)

_supabase_client = None
_current_key = None

def get_supabase() -> Client:
    global _supabase_client, _current_key
    load_dotenv(override=True)
    
    url = (os.environ.get("SUPABASE_URL", "") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "")).strip()
    key = (
        os.environ.get("SUPABASE_KEY", "") or 
        os.environ.get("SUPABASE_ANON_KEY", "") or 
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "") or 
        os.environ.get("SUPABASE_SERVICE_KEY", "")
    ).strip()
    
    if not url or not key:
        raise ValueError("[CRITICAL] SUPABASE_URL or SUPABASE_KEY is missing in environment variables! Please configure them.")
    
    if _supabase_client is None or _current_key != key:
        _supabase_client = create_client(url, key)
        _current_key = key
        
    return _supabase_client

# --- User Operations ---

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    email_clean = email.strip().lower()
    try:
        response = get_supabase().table("users").select("*").eq("email", email_clean).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        print(f"[Supabase Error] get_user_by_email: {e}")
        return None

def create_user_db(name: str, email: str, password_hash: str, salt: str) -> bool:
    email_clean = email.strip().lower()
    name_clean = name.strip()
    created_at = time.time()
    try:
        get_supabase().table("users").insert({
            "email": email_clean,
            "name": name_clean,
            "password_hash": password_hash,
            "salt": salt,
            "created_at": created_at
        }).execute()
        return True
    except Exception as e:
        print(f"[Supabase Error] create_user_db: {e}")
        return False

def update_user_password_db(email: str, password_hash: str, salt: str) -> bool:
    email_clean = email.strip().lower()
    try:
        response = get_supabase().table("users").update({
            "password_hash": password_hash,
            "salt": salt
        }).eq("email", email_clean).execute()
        return bool(response.data and len(response.data) > 0)
    except Exception as e:
        print(f"[Supabase Error] update_user_password_db: {e}")
        return False

def update_user_profile_db(old_email: str, name: str, new_email: str, avatar: Optional[str] = None) -> bool:
    old_clean = old_email.strip().lower()
    new_clean = new_email.strip().lower()
    name_clean = name.strip()

    try:
        user = get_user_by_email(old_clean)
        if not user:
            return False

        if old_clean == new_clean:
            update_data = {"name": name_clean}
            if avatar is not None:
                update_data["avatar"] = avatar
            
            get_supabase().table("users").update(update_data).eq("email", old_clean).execute()
            get_supabase().table("sessions").update({"name": name_clean}).eq("email", old_clean).execute()
        else:
            avatar_val = avatar if avatar is not None else user.get("avatar")
            
            get_supabase().table("users").insert({
                "email": new_clean,
                "name": name_clean,
                "password_hash": user["password_hash"],
                "salt": user["salt"],
                "avatar": avatar_val,
                "created_at": user.get("created_at", time.time())
            }).execute()
            
            get_supabase().table("sessions").update({
                "email": new_clean,
                "name": name_clean
            }).eq("email", old_clean).execute()
            
            get_supabase().table("users").delete().eq("email", old_clean).execute()

        return True
    except Exception as e:
        print(f"[Supabase Error] update_user_profile_db: {e}")
        return False

# --- Session Operations ---

def create_session_db(token: str, email: str, name: str, expires_at: float) -> None:
    email_clean = email.strip().lower()
    created_at = time.time()
    try:
        get_supabase().table("sessions").upsert({
            "token": token,
            "email": email_clean,
            "name": name,
            "created_at": created_at,
            "expires_at": expires_at
        }).execute()
    except Exception as e:
        print(f"[Supabase Error] create_session_db: {e}")

def get_session_db(token: str) -> Optional[Dict[str, Any]]:
    try:
        response = get_supabase().table("sessions").select("*").eq("token", token).execute()
        if not response.data or len(response.data) == 0:
            return None
            
        session = response.data[0]
        if time.time() > session["expires_at"]:
            destroy_session_db(token)
            return None
            
        new_expires = time.time() + 604800
        try:
            get_supabase().table("sessions").update({"expires_at": new_expires}).eq("token", token).execute()
        except Exception:
            pass

        return session
    except Exception as e:
        print(f"[Supabase Error] get_session_db: {e}")
        return None

def destroy_session_db(token: str) -> bool:
    try:
        response = get_supabase().table("sessions").delete().eq("token", token).execute()
        return bool(response.data and len(response.data) > 0)
    except Exception as e:
        print(f"[Supabase Error] destroy_session_db: {e}")
        return False

# --- OTP Operations ---

def set_otp_db(email: str, otp: str, expires_at: float, purpose: str) -> None:
    email_clean = email.strip().lower()
    try:
        get_supabase().table("otps").upsert({
            "email": email_clean,
            "otp": otp,
            "expires": expires_at,
            "purpose": purpose
        }).execute()
    except Exception as e:
        print(f"[Supabase Error] set_otp_db: {e}")

def get_otp_db(email: str) -> Optional[Dict[str, Any]]:
    email_clean = email.strip().lower()
    try:
        response = get_supabase().table("otps").select("*").eq("email", email_clean).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        print(f"[Supabase Error] get_otp_db: {e}")
        return None

def delete_otp_db(email: str) -> None:
    email_clean = email.strip().lower()
    try:
        get_supabase().table("otps").delete().eq("email", email_clean).execute()
    except Exception as e:
        print(f"[Supabase Error] delete_otp_db: {e}")

# --- Chat History Operations ---

def save_chat_message(email: str, role: str, message: str) -> bool:
    email_clean = email.strip().lower()
    created_at = time.time()
    try:
        get_supabase().table("chat_history").insert({
            "email": email_clean,
            "role": role,
            "message": message,
            "created_at": created_at
        }).execute()
        return True
    except Exception as e:
        print(f"[Supabase Error] save_chat_message: {e}")
        return False

def get_chat_history(email: str) -> list:
    email_clean = email.strip().lower()
    try:
        response = get_supabase().table("chat_history").select("*").eq("email", email_clean).order("created_at").execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"[Supabase Error] get_chat_history: {e}")
        return []

def delete_chat_message_db(email: str, message: str) -> bool:
    email_clean = email.strip().lower()
    try:
        response = get_supabase().table("chat_history").delete().match({"email": email_clean, "message": message}).execute()
        return bool(response.data and len(response.data) > 0)
    except Exception as e:
        print(f"[Supabase Error] delete_chat_message_db: {e}")
        return False

def clear_chat_history_db(email: str) -> bool:
    email_clean = email.strip().lower()
    try:
        get_supabase().table("chat_history").delete().eq("email", email_clean).execute()
        return True
    except Exception as e:
        print(f"[Supabase Error] clear_chat_history_db: {e}")
        return False

# --- Saved Papers Operations ---

def save_paper_db(
    email: str, 
    paper_id: str, 
    title: str = "", 
    authors: str = "", 
    abstract: str = "",
    source: str = "ArXiv",
    source_id: str = "",
    attack_vector: str = "Peer Review",
    url: str = ""
) -> bool:
    email_clean = email.strip().lower()
    saved_at = time.time()
    try:
        get_supabase().table("saved_papers").upsert({
            "email": email_clean,
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "source": source,
            "source_id": source_id,
            "attack_vector": attack_vector,
            "url": url,
            "saved_at": saved_at
        }).execute()
        return True
    except Exception as e:
        print(f"[Supabase Error] save_paper_db: {e}")
        return False

def get_saved_papers_db(email: str) -> list:
    email_clean = email.strip().lower()
    try:
        response = get_supabase().table("saved_papers").select("*").eq("email", email_clean).order("saved_at", desc=True).execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"[Supabase Error] get_saved_papers_db: {e}")
        return []

def delete_saved_paper_db(email: str, paper_id: str) -> bool:
    email_clean = email.strip().lower()
    try:
        response = get_supabase().table("saved_papers").delete().match({"email": email_clean, "paper_id": paper_id}).execute()
        return bool(response.data and len(response.data) > 0)
    except Exception as e:
        print(f"[Supabase Error] delete_saved_paper_db: {e}")
        return False

# --- Critiques & Hotspots Operations ---

def save_critique_chunk_db(chunk: Dict[str, Any]) -> bool:
    if not chunk or "id" not in chunk:
        return False
    try:
        data = {
            "id": str(chunk["id"]),
            "title": chunk.get("title", ""),
            "authors": chunk.get("authors", ""),
            "year": int(chunk.get("year", 2024)) if isinstance(chunk.get("year"), (int, float, str)) and str(chunk.get("year")).isdigit() else 2024,
            "source": chunk.get("source", ""),
            "source_id": chunk.get("source_id", ""),
            "url": chunk.get("url", ""),
            "section": chunk.get("section", ""),
            "attack_vector": chunk.get("attack_vector", ""),
            "target": chunk.get("target", ""),
            "risk_level": chunk.get("risk_level", ""),
            "skepticism_score": float(chunk.get("skepticism_score", 0.0)),
            "replication_prob": float(chunk.get("replication_prob", 0.0)),
            "paragraph_type": chunk.get("paragraph_type", ""),
            "adversarial_tag": chunk.get("adversarial_tag", chunk.get("distilbert_tag", "")),
            "text": chunk.get("text", ""),
            "raw_text": chunk.get("raw_text", chunk.get("text", "")),
            "severity": chunk.get("severity", ""),
            "mitigation_suggestion": chunk.get("mitigation_suggestion", ""),
            "updated_at": time.time()
        }
        try:
            get_supabase().table("critiques").upsert(data).execute()
        except Exception as err:
            if "adversarial_tag" in str(err) or "PGRST204" in str(err):
                # Fallback for Supabase databases that have not executed the schema migration yet
                data["distilbert_tag"] = data.pop("adversarial_tag", "")
                get_supabase().table("critiques").upsert(data).execute()
            else:
                raise err
        return True
    except Exception as e:
        print(f"[Supabase Warning] save_critique_chunk_db: {e}")
        return False

def get_critique_by_id_db(critique_id: str) -> Optional[Dict[str, Any]]:
    try:
        response = get_supabase().table("critiques").select("*").eq("id", critique_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        print(f"[Supabase Warning] get_critique_by_id_db: {e}")
        return None

def get_all_critiques_db() -> list:
    try:
        response = get_supabase().table("critiques").select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"[Supabase Warning] get_all_critiques_db: {e}")
        return []

def save_hotspots_cache_db(hotspots: list, cache_date: str) -> bool:
    try:
        get_supabase().table("hotspots_cache").upsert({
            "cache_key": f"hotspots_{cache_date}",
            "cache_date": cache_date,
            "data": hotspots,
            "updated_at": time.time()
        }).execute()
        return True
    except Exception as e:
        print(f"[Supabase Warning] save_hotspots_cache_db: {e}")
        return False

def get_hotspots_cache_db(cache_date: str) -> Optional[list]:
    try:
        response = get_supabase().table("hotspots_cache").select("*").eq("cache_key", f"hotspots_{cache_date}").execute()
        if response.data and len(response.data) > 0:
            row = response.data[0]
            if time.time() - row.get("updated_at", 0) < 1800:
                raw_data = row.get("data")
                return raw_data
        return None
    except Exception as e:
        print(f"[Supabase Warning] get_hotspots_cache_db: {e}")
        return None
