"""Flask API for the online Horus BGE Transformer."""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.explainability_service import (  # noqa: E402
    ContractError,
    OnlineExplainabilityService,
)


_SERVICE: OnlineExplainabilityService | None = None
_SERVICE_LOCK = threading.Lock()


def get_service() -> OnlineExplainabilityService:
    """Load BGE and the seed-42 Transformer once per API process."""
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = OnlineExplainabilityService.from_environment()
    return _SERVICE


def create_app(service: Any | None = None) -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    if service is not None:
        app.config["HORUS_SERVICE"] = service

    def active_service():
        return app.config.get("HORUS_SERVICE") or get_service()

    @app.get("/api/health")
    def health():
        current = app.config.get("HORUS_SERVICE") or _SERVICE
        return jsonify({
            "status": "ok",
            "modelLoaded": current is not None,
            "device": getattr(getattr(current, "runtime", None), "device", None),
        })

    @app.post("/api/analyse_message")
    def analyse_message():
        return jsonify({
            "error": "deprecated_endpoint",
            "message": "A pré-análise lexical agora é executada localmente pela extensão.",
        }), 410

    @app.post("/api/analyse_history")
    def analyse_history():
        try:
            payload = request.get_json(silent=False)
            result = active_service().analyse(payload)
            return jsonify(result)
        except ContractError as exc:
            return jsonify({"error": "invalid_request", "message": str(exc)}), 400
        except (FileNotFoundError, RuntimeError) as exc:
            app.logger.exception("Falha no serviço BGE")
            return jsonify({
                "error": "model_unavailable",
                "message": str(exc),
            }), 503

    @app.get("/api/explanations/<job_id>")
    def explanation_status(job_id: str):
        result = active_service().explanation_status(job_id)
        if result is None:
            return jsonify({
                "error": "job_not_found",
                "message": "Job inexistente ou expirado.",
            }), 404
        status = result.get("status")
        status_code = (
            202 if status in {"queued", "running"}
            else 500 if status == "failed"
            else 200
        )
        return jsonify(result), status_code

    return app


app = create_app()


if __name__ == "__main__":
    service = get_service()
    print(
        "Horus BGE API iniciada na porta 5000 "
        f"(device={getattr(service.runtime, 'device', 'unknown')})"
    )
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
