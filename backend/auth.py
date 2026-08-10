"""
Authentication Module for Peer-Review Devil's Advocate
Handles User Accounts, Password Validation, Email OTP delivery via Gmail SMTP, and Sessions.
"""

import os
import re
import json
import secrets
import hashlib
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Note: JSON based state management has been completely migrated to db.py SQLite implementation

# SMTP Configuration
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "learnix.research@gmail.com")
SENDER_APP_PASSWORD = os.environ.get("SENDER_APP_PASSWORD", "")

# --- Password Hashing & Verification ---

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hashes password using PBKDF2 HMAC SHA256 with a salt."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return hashed, salt

def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies case-sensitive password against stored hash."""
    hashed, _ = hash_password(password, salt)
    return secrets.compare_digest(hashed, stored_hash)

# --- Password Complexity Validator ---

def validate_password_complexity(password: str) -> Tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter (A-Z)."
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter (a-z)."
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one numeric digit (0-9)."
    if not re.search(r'[^A-Za-z0-9]', password):
        return False, "Password must contain at least one special character (!@#$%^&*...)."
    return True, "Password meets all security criteria."

# --- OTP Generation & SMTP Email Delivery ---

def generate_otp() -> str:
    return str(secrets.randbelow(900000) + 100000)

def send_otp_email(target_email: str, otp_code: str, purpose: str = "Account Registration") -> Tuple[bool, str]:
    """Sends OTP via Gmail SMTP with Learnix Web App Theme."""
    subject = f"Learnix Research - {purpose} Verification Code: {otp_code}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <style>
        body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; background-color: #fcf9f8; color: #1c1b1b; margin: 0; padding: 30px 15px; }}
        .card {{ max-width: 520px; margin: 0 auto; background: #ffffff; border: 1px solid #c3c8c1; border-radius: 20px; padding: 36px; box-shadow: 0 10px 30px rgba(6, 27, 14, 0.05); }}
        .logo-badge {{ display: inline-block; background: #1b3022; color: #d0e9d4; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; padding: 6px 14px; border-radius: 50px; margin-bottom: 20px; }}
        .header {{ font-family: Georgia, serif; font-size: 26px; font-weight: 700; color: #061b0e; margin: 0 0 16px 0; tracking-tight: -0.5px; }}
        .otp-box {{ background: #f6f3f2; border: 2px dashed #061b0e; border-radius: 14px; font-size: 40px; font-weight: 800; letter-spacing: 12px; color: #061b0e; text-align: center; padding: 22px; margin: 28px 0; text-indent: 12px; }}
        .text {{ font-size: 14px; color: #5e5f5c; line-height: 1.6; margin-bottom: 16px; }}
        .footer {{ font-size: 12px; color: #737973; text-align: center; margin-top: 32px; border-top: 1px solid #e0e0dc; padding-top: 20px; }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="logo-badge">Learnix Research Portal</div>
        <div class="header">Peer-Review Verification</div>
        <div class="text">Hello,</div>
        <div class="text">Your one-time verification code for <strong>{purpose}</strong> is:</div>
        <div class="otp-box">{otp_code}</div>
        <div class="text">This code will expire in <strong>10 minutes</strong>. If you did not request this code, please ignore this email.</div>
        <div class="footer">&copy; Learnix Research Platform &bull; Peer-Review Devil's Advocate &amp; RAG Engine</div>
      </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Learnix Research <{SENDER_EMAIL}>"
    msg["To"] = target_email
    msg.attach(MIMEText(html_content, "html"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, target_email, msg.as_string())
        server.quit()
        return True, "Verification email sent successfully."
    except Exception as e:
        print(f"[SMTP Error] Failed to send email to {target_email}: {e}")
        return False, f"Failed to send email: {str(e)}"

from backend.db import (
    get_user_by_email, create_user_db, update_user_password_db,
    create_session_db, get_session_db, destroy_session_db,
    set_otp_db, get_otp_db, delete_otp_db
)

# --- Helper Store Operations ---

def create_and_send_otp(email: str, purpose: str = "registration") -> Tuple[bool, str]:
    email_clean = email.strip().lower()
    otp = generate_otp()
    expires_at = time.time() + 600
    
    set_otp_db(email_clean, otp, expires_at, purpose)

    display_purpose = "Account Registration" if purpose == "registration" else ("Email Change Verification" if purpose == "email_change" else "Password Reset")
    success, msg = send_otp_email(email_clean, otp, display_purpose)
    return success, msg

def verify_otp(email: str, entered_otp: str, purpose: str = "registration") -> Tuple[bool, str]:
    email_clean = email.strip().lower()
    data = get_otp_db(email_clean)
    if not data:
        return False, "No OTP request found for this email. Please request a new code."

    if time.time() > data["expires"]:
        delete_otp_db(email_clean)
        return False, "OTP code has expired. Please request a new code."

    if data["purpose"] != purpose:
        return False, "OTP code purpose mismatch."

    if data["otp"] != entered_otp.strip():
        return False, "Invalid OTP code. Please check your email and try again."

    return True, "OTP verified successfully."

def consume_otp(email: str) -> None:
    delete_otp_db(email.strip().lower())

# --- Session Management (7 Days Expiry in Database) ---

def create_session(email: str, name: str) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + 604800 # 7 days
    create_session_db(token, email, name, expires_at)
    return token

def get_session(token: str) -> Optional[Dict[str, Any]]:
    return get_session_db(token)

def destroy_session(token: str) -> bool:
    return destroy_session_db(token)


