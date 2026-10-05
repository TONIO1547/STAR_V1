#!/usr/bin/env python3
"""Authentication layer for the S.T.A.R Flask server (Python 3.6+)."""

import base64
import getpass
import hashlib
import hmac
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
from datetime import timedelta
from functools import wraps
from urllib.parse import urlsplit

from flask import jsonify, redirect, render_template_string, request, session, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 600000
MAX_FAILURES = 5
FAILURE_WINDOW_SECONDS = 15 * 60
BLOCK_SECONDS = 15 * 60

LOGIN_HTML = r'''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Connexion S.T.A.R</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center;
           background: #090d0b; color: #eef8f1; font-family: Arial, sans-serif; }
    main { width: min(92vw, 390px); padding: 30px; border: 1px solid #24552f;
           border-radius: 14px; background: #101713; box-shadow: 0 18px 60px #0009; }
    h1 { margin: 0 0 6px; color: #39e463; font-size: 27px; }
    p { margin: 0 0 24px; color: #9daf9f; }
    label { display: block; margin: 14px 0 6px; font-size: 14px; }
    input { width: 100%; padding: 12px; border: 1px solid #35483a; border-radius: 8px;
            background: #090d0b; color: white; font-size: 16px; }
    input:focus { outline: 2px solid #39e463; border-color: transparent; }
    button { width: 100%; margin-top: 22px; padding: 12px; border: 0; border-radius: 8px;
             background: #28bd4b; color: #041006; font-weight: bold; font-size: 16px; cursor: pointer; }
    .error { padding: 10px; border: 1px solid #b83f47; border-radius: 7px;
             background: #3a1619; color: #ffd9dc; font-size: 14px; }
  </style>
</head>
<body>
  <main>
    <h1>S.T.A.R</h1>
    <p>Console de contrôle sécurisée</p>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="post" action="{{ url_for('star_login') }}" autocomplete="on">
      <input type="hidden" name="next" value="{{ next_url }}">
      <label for="username">Identifiant</label>
      <input id="username" name="username" required autofocus autocomplete="username">
      <label for="password">Mot de passe</label>
      <input id="password" name="password" type="password" required autocomplete="current-password">
      <button type="submit">Se connecter</button>
    </form>
  </main>
</body>
</html>'''


def _b64encode(raw):
    return base64.b64encode(raw).decode("ascii")


def _b64decode(value):
    return base64.b64decode(value.encode("ascii"), validate=True)


def hash_password(password, iterations=PASSWORD_ITERATIONS):
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return "{}${}${}${}".format(
        PASSWORD_SCHEME, iterations, _b64encode(salt), _b64encode(digest)
    )


def verify_password(password, encoded):
    try:
        scheme, iterations_text, salt_text, digest_text = encoded.split("$", 3)
        if scheme != PASSWORD_SCHEME:
            return False
        iterations = int(iterations_text)
        if iterations < 100000 or iterations > 2000000:
            return False
        salt = _b64decode(salt_text)
        expected = _b64decode(digest_text)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError, UnicodeError):
        return False


def _safe_next(value):
    if not value:
        return "/"
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
        return "/"
    return value


def _client_key():
    peer = request.remote_addr or "unknown"
    if peer in ("127.0.0.1", "::1"):
        cloudflare_ip = request.headers.get("CF-Connecting-IP", "").strip()
        if cloudflare_ip and len(cloudflare_ip) <= 64:
            return cloudflare_ip
    return peer


def _same_origin_request():
    origin = request.headers.get("Origin", "")
    if not origin:
        return False
    try:
        return urlsplit(origin).netloc.lower() == request.host.lower()
    except (TypeError, ValueError):
        return False


def install_star_auth(app):
    """Install login, lockout, session and request protections on ``app``."""
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1)
    username = os.environ.get("STAR_USERNAME", "")
    password_hash = os.environ.get("STAR_PASSWORD_HASH", "")
    secret_key = os.environ.get("STAR_SECRET_KEY", "")

    if not username or not password_hash or len(secret_key) < 32:
        raise RuntimeError(
            "S.T.A.R authentication is not configured. "
            "Run: sudo python3 star_auth.py --configure"
        )
    if not verify_password("configuration-test-password", password_hash) and not password_hash.startswith(
        PASSWORD_SCHEME + "$"
    ):
        raise RuntimeError("STAR_PASSWORD_HASH has an unsupported format")

    app.secret_key = secret_key
    app.config.update(
        SESSION_COOKIE_NAME="star_session",
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        MAX_CONTENT_LENGTH=16 * 1024,
    )

    failures = {}
    failures_lock = threading.Lock()

    def authenticated():
        current = session.get("star_user", "")
        try:
            return hmac.compare_digest(current.encode("utf-8"), username.encode("utf-8"))
        except (AttributeError, UnicodeError):
            return False

    def current_block(key, now):
        with failures_lock:
            record = failures.get(key)
            if not record:
                return 0
            if record["blocked_until"] > now:
                return int(record["blocked_until"] - now) + 1
            recent = [
                stamp for stamp in record["failures"]
                if now - stamp < FAILURE_WINDOW_SECONDS
            ]
            if recent:
                record["failures"] = recent
            else:
                failures.pop(key, None)
            return 0

    def register_failure(key, now):
        with failures_lock:
            record = failures.setdefault(key, {"failures": [], "blocked_until": 0})
            record["failures"] = [
                stamp for stamp in record["failures"]
                if now - stamp < FAILURE_WINDOW_SECONDS
            ]
            record["failures"].append(now)
            if len(record["failures"]) >= MAX_FAILURES:
                record["failures"] = []
                record["blocked_until"] = now + BLOCK_SECONDS
                return BLOCK_SECONDS
            return 0

    @app.before_request
    def star_require_login():
        if request.endpoint == "star_login":
            return None
        if not authenticated():
            session.clear()
            if request.path == "/motor":
                return jsonify({"status": "error", "message": "authentication required"}), 401
            return redirect(url_for("star_login", next=request.full_path.rstrip("?")))
        if request.path == "/motor" and request.method == "POST" and not _same_origin_request():
            return jsonify({"status": "error", "message": "invalid request origin"}), 403
        return None

    @app.after_request
    def star_security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; frame-ancestors https://star-ai.fr https://www.star-ai.fr; base-uri 'none'; form-action 'self'"
        )
        return response

    @app.route("/login", methods=["GET", "POST"])
    def star_login():
        next_url = _safe_next(request.values.get("next", "/"))
        if authenticated():
            return redirect(next_url)

        key = _client_key()
        now = time.time()
        remaining = current_block(key, now)
        error = None
        status = 200

        if remaining:
            error = "Trop de tentatives. Réessaie dans {} minute(s).".format(
                max(1, (remaining + 59) // 60)
            )
            status = 429
        elif request.method == "POST":
            supplied_username = request.form.get("username", "")
            supplied_password = request.form.get("password", "")
            try:
                username_ok = hmac.compare_digest(
                    supplied_username.encode("utf-8"), username.encode("utf-8")
                )
            except (AttributeError, UnicodeError):
                username_ok = False
            password_ok = verify_password(supplied_password, password_hash)

            if username_ok and password_ok:
                with failures_lock:
                    failures.pop(key, None)
                session.clear()
                session["star_user"] = username
                session["authenticated_at"] = int(now)
                session.permanent = True
                return redirect(next_url)

            blocked_for = register_failure(key, now)
            if blocked_for:
                error = "Trop de tentatives. Accès bloqué pendant 15 minutes."
                status = 429
            else:
                error = "Identifiant ou mot de passe incorrect."
                status = 401

        response = render_template_string(
            LOGIN_HTML, error=error, next_url=next_url
        )
        return response, status

    @app.route("/logout", methods=["GET", "POST"])
    def star_logout():
        session.clear()
        return redirect(url_for("star_login"))

    return app


def _configure_env(env_path="/etc/star.env"):
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        sys.exit("Relance cette commande avec sudo.")

    username = input("Identifiant S.T.A.R : ").strip()
    if not re.match(r"^[A-Za-z0-9_.@-]{3,64}$", username):
        sys.exit("Identifiant invalide (3 à 64 caractères : lettres, chiffres, . _ @ -).")

    password = getpass.getpass("Mot de passe S.T.A.R (12 caractères minimum) : ")
    confirmation = getpass.getpass("Confirme le mot de passe : ")
    if password != confirmation:
        sys.exit("Les deux mots de passe ne correspondent pas.")
    if len(password) < 12:
        sys.exit("Le mot de passe doit contenir au moins 12 caractères.")

    existing = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as handle:
            existing = handle.readlines()
        backup_path = env_path + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(env_path, backup_path)
        os.chmod(backup_path, 0o600)

    managed = ("STAR_USERNAME=", "STAR_PASSWORD_HASH=", "STAR_SECRET_KEY=")
    preserved = [line for line in existing if not line.startswith(managed)]
    if preserved and not preserved[-1].endswith("\n"):
        preserved[-1] += "\n"
    preserved.extend([
        "STAR_USERNAME={}\n".format(username),
        "STAR_PASSWORD_HASH={}\n".format(hash_password(password)),
        "STAR_SECRET_KEY={}\n".format(secrets.token_hex(32)),
    ])

    directory = os.path.dirname(env_path) or "."
    fd, temp_path = tempfile.mkstemp(prefix=".star-env-", dir=directory, text=True)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.writelines(preserved)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, env_path)
        os.chmod(env_path, 0o600)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    finally:
        password = None
        confirmation = None

    print("Configuration enregistrée dans {} (mot de passe non stocké).".format(env_path))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--configure":
        _configure_env()
    else:
        print("Usage: sudo python3 star_auth.py --configure")
        sys.exit(2)
