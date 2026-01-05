import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

    # Database
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_PORT = os.environ.get('MYSQL_PORT', '3306')
    MYSQL_USER = os.environ.get('MYSQL_USER', 'goswami')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'goswamipassword')
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'goswami_whisper')

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@"
        f"{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File uploads
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(os.path.dirname(__file__), 'uploads'))
    ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'webm', 'm4a', 'ogg', 'flac', 'mpeg', 'mpga'}
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max file size

    # OpenAI
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

    # Anthropic (Claude)
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

    # LLM Provider for text formatting: 'anthropic' (Claude) or 'openai' (GPT)
    # Claude has 200K context window, better for long transcriptions
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'anthropic')

    # Whisper
    WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base')
    WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

    # ChromaDB
    CHROMA_PERSIST_DIR = os.environ.get('CHROMA_PERSIST_DIR', os.path.join(os.path.dirname(__file__), 'chroma_db'))

