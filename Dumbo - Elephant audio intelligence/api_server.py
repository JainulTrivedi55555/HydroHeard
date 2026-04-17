"""
=============================================================================
🔒 Auth0-Secured API for ElephantVoices Denoiser
=============================================================================
Flask API that wraps the denoiser pipeline with Auth0 authentication.
Deploy on DigitalOcean Droplet.

Setup:
  1. Create Auth0 account → create API → get domain & audience
  2. Set environment variables (see below)
  3. pip install -r requirements.txt
  4. python api_server.py
=============================================================================
"""

import os
import json
import tempfile
from functools import wraps
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import jwt
from jwt import PyJWKClient

# Import our denoiser
from elephant_denoiser import process_recording, Config

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload

# ── Auth0 Configuration ──────────────────────────────────────────
AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN", "your-tenant.auth0.com")
AUTH0_AUDIENCE = os.environ.get("AUTH0_AUDIENCE", "https://elephant-denoiser-api")
ALGORITHMS = ["RS256"]


class AuthError(Exception):
    def __init__(self, error, status_code):
        self.error = error
        self.status_code = status_code


def get_token_from_header():
    auth = request.headers.get("Authorization", None)
    if not auth:
        raise AuthError({"message": "Authorization header missing"}, 401)
    parts = auth.split()
    if parts[0].lower() != "bearer" or len(parts) != 2:
        raise AuthError({"message": "Invalid authorization header"}, 401)
    return parts[1]


def requires_auth(f):
    """Auth0 JWT verification decorator."""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            token = get_token_from_header()
            jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
            jwks_client = PyJWKClient(jwks_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token, signing_key.key,
                algorithms=ALGORITHMS,
                audience=AUTH0_AUDIENCE,
                issuer=f"https://{AUTH0_DOMAIN}/"
            )
            request.user = payload
        except AuthError as e:
            return jsonify(e.error), e.status_code
        except Exception as e:
            return jsonify({"message": f"Auth failed: {str(e)}"}), 401
        return f(*args, **kwargs)
    return decorated


# ── API Routes ───────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "service": "ElephantVoices Rumble Denoiser API",
        "version": "1.0.0",
        "hackathon": "HackSMU VII",
        "status": "healthy"
    })


@app.route("/denoise", methods=["POST"])
@requires_auth
def denoise():
    """
    POST /denoise
    Body: multipart/form-data with:
      - file: WAV audio file
      - noise_type (optional): airplane | car | generator
      - call_start (optional): float seconds
      - call_end (optional): float seconds
    Returns: cleaned WAV file
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only WAV files supported"}), 400

    noise_type = request.form.get("noise_type")
    call_start = request.form.get("call_start")
    call_end = request.form.get("call_end")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, secure_filename(file.filename))
        output_path = os.path.join(tmpdir, "cleaned.wav")
        file.save(input_path)

        try:
            result = process_recording(
                input_path, output_path,
                call_start=float(call_start) if call_start else None,
                call_end=float(call_end) if call_end else None,
                noise_type_override=noise_type
            )
            return send_file(output_path, mimetype="audio/wav",
                           as_attachment=True,
                           download_name=f"cleaned_{file.filename}")
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
@requires_auth
def analyze():
    """
    POST /analyze — classify noise type only (no denoising).
    Returns JSON with Gemini analysis.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, secure_filename(file.filename))
        file.save(input_path)

        try:
            from elephant_denoiser import load_audio, compute_stft, classify_noise_with_gemini
            y, sr = load_audio(input_path)
            magnitude, phase, S = compute_stft(y, sr)
            analysis = classify_noise_with_gemini(magnitude, sr)
            return jsonify(analysis)
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.errorhandler(AuthError)
def handle_auth_error(ex):
    return jsonify(ex.error), ex.status_code


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting ElephantVoices API on port {port}")
    print(f"🔒 Auth0 domain: {AUTH0_DOMAIN}")
    app.run(host="0.0.0.0", port=port, debug=True)
