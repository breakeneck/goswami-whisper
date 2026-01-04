from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()


def create_app(config_class=None):
    """Application factory for Flask app."""
    app = Flask(__name__)

    # Load configuration
    if config_class is None:
        from config import Config
        app.config.from_object(Config)
    else:
        app.config.from_object(config_class)

    # Ensure upload folder exists
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)

    # Ensure ChromaDB folder exists
    os.makedirs(app.config.get('CHROMA_PERSIST_DIR', 'chroma_db'), exist_ok=True)

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    from app.routes import main_bp, upload_bp, search_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(search_bp, url_prefix='/search')

    # Create database tables
    with app.app_context():
        db.create_all()

    return app

