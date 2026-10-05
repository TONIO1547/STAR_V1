"""Server-to-device authentication for the S.T.A.R. Flask agent."""

import hmac
import logging
import os

from flask import jsonify, request


LOGGER = logging.getLogger("star.device_auth")


def install_device_auth(app):
    """Require the site-held bearer token on every Flask route."""

    expected_token = os.environ.get("STAR_DEVICE_TOKEN", "").strip()
    if len(expected_token) < 32:
        raise RuntimeError(
            "STAR_DEVICE_TOKEN must be configured with at least 32 characters"
        )
    expected_header = "Bearer " + expected_token

    @app.before_request
    def require_device_token():
        provided = request.headers.get("Authorization", "")
        if hmac.compare_digest(provided, expected_header):
            return None
        LOGGER.warning(
            "device_auth_denied remote=%s path=%s",
            request.remote_addr or "unknown",
            request.path,
        )
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    @app.after_request
    def device_security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

