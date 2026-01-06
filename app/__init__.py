from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

db = SQLAlchemy()
migrate = Migrate()


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
    migrate.init_app(app, db)

    # Register blueprints
    from app.routes import main_bp, upload_bp, search_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(search_bp, url_prefix='/search')
    app.register_blueprint(api_bp, url_prefix='/api')

    # Import models to register them
    from app.models import Transcription, Upload, Transcribe, Content

    # Create database tables
    with app.app_context():
        db.create_all()

    return app

