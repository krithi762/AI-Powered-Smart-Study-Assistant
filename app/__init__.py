import os

from flask import Flask, flash, jsonify, redirect, request, url_for

from config import config_by_name
from database.db import init_db


def create_app(config_name: str = "default") -> Flask:
    """
    Application Factory Pattern for Flask.
    Initializes configurations, sets up directories, prepares database, and registers blueprints.
    """
    app = Flask(__name__)

    # Load configuration
    config_class = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_class)

    # Ensure vital operational directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["DATABASE_DIR"], exist_ok=True)
    os.makedirs(app.config["MODELS_DIR"], exist_ok=True)

    # Initialize SQLite database schema
    init_db(app.config["DATABASE_PATH"])

    # Register application blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # Global Error Handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handles uploads exceeding the 16 MB maximum file size limit."""
        if (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or request.is_json
            or "application/json" in request.headers.get("Accept", "")
        ):
            return jsonify({
                "success": False,
                "error": "The uploaded file exceeds the 16 MB maximum size limit."
            }), 413
        flash("The uploaded file exceeds the 16 MB maximum size limit.", "error")
        return redirect(url_for("main.index")), 413

    @app.errorhandler(404)
    def page_not_found(error):
        if (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or request.is_json
            or "application/json" in request.headers.get("Accept", "")
        ):
            return jsonify({"success": False, "error": "Requested resource not found."}), 404
        return redirect(url_for("main.index")), 404

    return app
