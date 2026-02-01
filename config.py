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

    # Google Gemini
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

    # xAI (Grok)
    XAI_API_KEY = os.environ.get('XAI_API_KEY', '')

    # Zhipu AI (智谱AI/BigModel)
    ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY', '')

    # LM Studio
    LMSTUDIO_BASE_URL = os.environ.get('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')

    # Transcription providers and models
    TRANSCRIBE_PROVIDERS = {
        'whisper': {
            'name': 'OpenAI Whisper',
            'models': ['medium', 'large', 'large-v3']
        },
        'faster-whisper': {
            'name': 'Faster Whisper',
            'models': ['medium', 'large-v2', 'large-v3']
        },
        'qwen3-asr': {
            'name': 'Qwen3-ASR',
            'models': ['Qwen/Qwen3-ASR']
        }
    }

    # Formatting providers and models
    FORMAT_PROVIDERS = {
        'openai': {
            'name': 'OpenAI',
            'models': ['gpt-5.2', 'gpt-5.1', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o-mini']
        },
        'anthropic': {
            'name': 'Anthropic (Claude)',
            'models': ['claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307']
        },
        'gemini': {
            'name': 'Google Gemini',
            'models': ['gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash']
        },
        'xai': {
            'name': 'xAI (Grok)',
            'models': ['grok-3', 'grok-3-fast', 'grok-3-mini', 'grok-3-mini-fast', 'grok-2-1212', 'grok-2-vision-1212', 'grok-vision-beta', 'grok-beta']
        },
        'zhipu': {
            'name': 'Zhipu AI (智谱)',
            'models': ['glm-4.7', 'glm-4.7-flash', 'glm-4.7-flashx', 'glm-4.6', 'glm-4.6v-flashx', 'glm-4.5-air', 'glm-4-plus', 'glm-4-long', 'glm-4-air', 'glm-4-airx', 'glm-4-flash', 'glm-4-flashx', 'glm-z1-air', 'glm-z1-airx', 'glm-z1-flash']
        },
        'lmstudio': {
            'name': 'LM Studio (Local)',
            'models': []  # Will be loaded dynamically
        }
    }

    # ChromaDB
    CHROMA_PERSIST_DIR = os.environ.get('CHROMA_PERSIST_DIR', os.path.join(os.path.dirname(__file__), 'chroma_db'))
